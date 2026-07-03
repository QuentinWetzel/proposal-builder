"""
Orchestrator — stage 2. Closes the graph:

    START -> planner -> orchestrator --Send()--> section_worker (xN, parallel)
                                                     |
                                              (reducer-merged)
                                                     v
                                                 assemble -> END

- planner: stage 1 (planner.py) — brief -> SectionPlan.
- orchestrator: reads plan.active_items(), fans out ONE worker per active
  section via LangGraph's Send API. Sections run in parallel; each worker is
  the full agentic loop (worker.run_section_loop: retrieve -> draft ->
  critique -> decide, with stop conditions).
- template-mode sections (cover, next steps) don't need the loop: they're
  rendered deterministically in assemble, not dispatched.
- assemble: merges SectionOutcomes into the draft-for-review payload —
  accepted sections in canonical order + the human-review queue (escalated
  sections with their flags reports). This payload is what the review UI
  renders; nothing is exported until a human clears the queue.

Node functions are thin wrappers over framework-agnostic logic (planner_node,
run_section_loop), so an ADK port re-wires edges, not behavior.
"""

from __future__ import annotations

from functools import partial
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from critique import ClaimExtractor
from progress import log
from planner import (
    CANONICAL_ORDER,
    CanonicalSection,
    LLMClient,
    RAGClient,
    SectionPlan,
    SectionPlanItem,
    StructuredBrief,
    StubLLMClient,
    planner_node,
)
from worker import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_TAU,
    Drafter,
    SectionOutcome,
    StopReason,
    StubDrafter,
    run_section_loop,
)

# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


def _merge_outcomes(left: dict, right: dict) -> dict:
    out = dict(left)
    out.update(right)
    return out


class GraphState(TypedDict, total=False):
    raw_brief: str
    brief: StructuredBrief
    plan: SectionPlan
    # section name -> SectionOutcome (as dict), reducer-merged as workers finish
    outcomes: Annotated[dict[str, dict], _merge_outcomes]
    # assemble() output: the review payload
    assembled: dict[str, Any]


class WorkerState(TypedDict):
    """Payload each Send() carries to one section worker."""
    item: SectionPlanItem
    brief: StructuredBrief


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def dispatch_sections(state: GraphState) -> list[Send]:
    """Orchestrator: one Send per active, loop-eligible section."""
    plan: SectionPlan = state["plan"]
    brief: StructuredBrief = state["brief"]
    sends = [
        Send("section_worker", WorkerState(item=item, brief=brief))
        for item in plan.active_items()
        if item.mode != "template"  # cover/next-steps render in assemble
    ]
    log(f"orchestrator: fanning out {len(sends)} section workers in parallel")
    return sends


async def section_worker_node(
    state: WorkerState,
    *,
    rag: RAGClient,
    drafter: Drafter,
    tau: float,
    max_iterations: int,
    extractor: ClaimExtractor | None = None,
) -> dict:
    outcome = await run_section_loop(
        state["item"], brief=state["brief"], rag=rag, drafter=drafter,
        tau=tau, max_iterations=max_iterations, extractor=extractor,
    )
    return {"outcomes": {outcome.section: outcome.model_dump()}}


def _render_template(section: CanonicalSection, item: SectionPlanItem,
                     brief: StructuredBrief) -> str:
    if section is CanonicalSection.COVER:
        return (f"{item.title}\n{brief.client or ''} — "
                f"{brief.service_line or ''}\n[date]")
    if section is CanonicalSection.NEXT_STEPS:
        return (f"{item.title}\nProposed next steps: alignment call, "
                f"scope confirmation, kick-off planning.\n[contacts]")
    return item.title


def assemble_node(state: GraphState) -> dict:
    """Merge worker outcomes into the review payload, canonical order."""
    plan: SectionPlan = state["plan"]
    brief: StructuredBrief = state["brief"]
    outcomes = {k: SectionOutcome(**v) for k, v in state.get("outcomes", {}).items()}

    sections: list[dict[str, Any]] = []
    review_queue: list[dict[str, Any]] = []

    for item in plan.active_items():
        if item.mode == "template":
            sections.append({
                "section": item.section.value, "title": item.title,
                "status": "templated",
                "content": _render_template(item.section, item, brief),
            })
            continue

        oc = outcomes.get(item.section.value)
        if oc is None:  # defensive: worker never reported
            review_queue.append({"section": item.section.value,
                                 "reason": "no_outcome", "flags": ""})
            continue

        if oc.accepted:
            refs = {}  # object_id -> CredentialRef, deduped across iterations
            for it in oc.iterations:
                for p in it.critique.claim_proofs:
                    if p.supported:
                        for r in p.credential_refs:
                            refs.setdefault(r.object_id, r)
            sections.append({
                "section": oc.section, "title": item.title,
                "status": "accepted", "mode": item.mode,
                "content": oc.final_draft,
                "confidence": oc.final_critique.confidence
                if oc.final_critique else None,
                "iterations": len(oc.iterations),
                "credential_ids": sorted({
                    cid for it in oc.iterations
                    for p in it.critique.claim_proofs if p.supported
                    for cid in p.credential_ids
                }),
                "credential_refs": [refs[k].model_dump()
                                    for k in sorted(refs)],
            })
        else:
            sections.append({
                "section": oc.section, "title": item.title,
                "status": f"escalated:{oc.stop_reason.value}",
                "content": oc.final_draft,  # shown, clearly flagged, not exportable
            })
            review_queue.append({
                "section": oc.section,
                "reason": oc.stop_reason.value,
                "flags": oc.flags_report(),
            })

    log(f"assemble: {sum(s['status'] == 'accepted' for s in sections)} accepted, "
        f"{sum(s['status'] == 'templated' for s in sections)} templated, "
        f"{len(review_queue)} in review queue -> "
        f"exportable={not review_queue}")
    return {"assembled": {
        "client": brief.client,
        "sections": sections,
        "review_queue": review_queue,
        "exportable": not review_queue,  # nothing exports past unresolved flags
    }}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph(
    *,
    rag: RAGClient,
    drafter: Drafter | None = None,
    llm: LLMClient | None = None,
    tau: float = DEFAULT_TAU,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    extractor: ClaimExtractor | None = None,
):
    llm = llm or StubLLMClient()
    drafter = drafter or StubDrafter()

    g = StateGraph(GraphState)
    g.add_node("planner", partial(planner_node, llm=llm))
    g.add_node("section_worker",
               partial(section_worker_node, rag=rag, drafter=drafter,
                       tau=tau, max_iterations=max_iterations,
                       extractor=extractor))
    g.add_node("assemble", assemble_node)

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", dispatch_sections, ["section_worker"])
    g.add_edge("section_worker", "assemble")
    g.add_edge("assemble", END)
    return g.compile()


# ---------------------------------------------------------------------------
# End-to-end demo: brief in -> draft-with-flags out
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    import json

    from planner import RetrievalSpec, RetrievedChunk

    class DemoRAG:
        """Section retrievals return corpus-ish chunks; one drafter output leaks,
        proof retrieval only proves 'savings'/'purchasing' claims."""

        async def retrieve(self, spec: RetrievalSpec):
            if spec.sources == ["OneShelf > Credentials Library"]:
                if any(k in spec.query.lower() for k in ("savings", "purchasing")):
                    return [RetrievedChunk(text="credential: 23M€ committed savings",
                                           score=0.0,
                                           metadata={"object_id": "cred..s7"})]
                return []
            return [RetrievedChunk(
                text=f"slide material for {spec.query[:40]}", score=0.7,
                metadata={"object_id": "prop..s3"})]

        async def enrich(self, spec: RetrievalSpec):
            return await self.retrieve(spec)

    class DemoDrafter:
        """Leaks 'Thales' in the work_plan draft; everything else is clean+provable."""

        async def draft(self, *, item, brief, chunks, feedback) -> str:
            client = brief.client or "the client"
            if item.section.value == "work_plan" and not feedback:
                return (f"{item.title} for {client}: phased plan as delivered "
                        f"for Thales, with weekly steering.")
            return (f"{item.title} for {client}: we have delivered purchasing "
                    f"savings programs; adapted from: {chunks[0].text}")

    async def main():
        graph = build_graph(rag=DemoRAG(), drafter=DemoDrafter())
        result = await graph.ainvoke(
            {"raw_brief": "RFP: ACME Aerospace, purchasing transformation."})
        a = result["assembled"]
        print(f"client={a['client']}  exportable={a['exportable']}\n")
        for s in a["sections"]:
            line = f"[{s['status']:26s}] {s['section']}"
            if "confidence" in s and s.get("confidence") is not None:
                line += f"  conf={s['confidence']}  iters={s['iterations']}"
                if s.get("credential_ids"):
                    line += f"  creds={s['credential_ids']}"
            print(line)
        print("\n--- REVIEW QUEUE ---")
        for r in a["review_queue"]:
            print(r["flags"] or f"{r['section']}: {r['reason']}")

    asyncio.run(main())
