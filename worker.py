"""
Worker — the per-section agentic loop (the centerpiece).

    retrieve (planner spec on iter 0, critique's proposed spec after)
       -> draft (adapt retrieved material to the new prospect)
       -> critique (structured schema: proofs, leaks, confidence, next query)
       -> decide:
            LEAK detected            -> HARD BLOCK: stop, escalate to human.
                                        Never auto-regenerate around a leak.
            confidence >= tau, clean -> ACCEPT.
            budget exhausted         -> ESCALATE with flags.
            else                     -> LOOP: re-retrieve with the critique's
                                        proposed RetrievalSpec and redraft.

Stop conditions are first-class: SectionOutcome records WHY the loop stopped,
every iteration is kept in the trace (drafts, critiques, specs used), and the
constants (tau, budget) are explicit parameters, not buried defaults.

Framework-agnostic: plain async functions. The LangGraph node (stage 2 wiring)
wraps run_section_loop(); an ADK port wraps the same function.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from critique import (
    ClaimExtractor,
    ClaimProof,
    SectionCritique,
    credential_refs_from,
    critique_section,
)
from progress import log
from planner import (
    RAGClient,
    RetrievalSpec,
    RetrievedChunk,
    SectionPlanItem,
    StructuredBrief,
)

# ---------------------------------------------------------------------------
# Stop-condition constants — explicit, configurable, defended in the README
# ---------------------------------------------------------------------------

# Accept threshold on critique confidence. 0.6 admits clean claim-free
# sections (the _confidence no-claims floor): no leaks + no unsupported
# claims + nothing to prove = acceptable. Real claims still score 0..1.
DEFAULT_TAU: float = 0.6
DEFAULT_MAX_ITERATIONS: int = 3   # draft attempts before human escalation


class StopReason(str, Enum):
    ACCEPTED = "accepted"                    # confidence >= tau, no leak
    LEAK_HARD_BLOCK = "leak_hard_block"      # leak detected -> human, immediately
    BUDGET_EXHAUSTED = "budget_exhausted"    # max iterations hit -> human
    NO_MATERIAL = "no_material"              # retrieval empty -> human


class IterationTrace(BaseModel):
    iteration: int
    spec_used: RetrievalSpec
    n_retrieved: int
    draft: str
    critique: SectionCritique


class SectionOutcome(BaseModel):
    section: str
    stop_reason: StopReason
    accepted: bool
    needs_human: bool
    final_draft: str = ""
    final_critique: SectionCritique | None = None
    iterations: list[IterationTrace] = Field(default_factory=list)

    def flags_report(self) -> str:
        """Human-facing summary for escalated sections."""
        if not self.needs_human or self.final_critique is None:
            return ""
        c = self.final_critique
        lines = [f"Section '{self.section}' escalated ({self.stop_reason.value}):"]
        for s in c.leak_spans:
            lines.append(f"  LEAK [{s.kind}] '{s.span}' — …{s.context}…")
        for p in c.claim_proofs:
            if not p.supported:
                lines.append(f"  UNSUPPORTED CLAIM: {p.claim[:100]}")
        for f in c.stale_or_unverified_figures:
            lines.append(f"  UNVERIFIED FIGURE: {f}")
        lines.append(f"  confidence={c.confidence} after {len(self.iterations)} iteration(s)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Drafting seam — Gemini/Vertex in production
# ---------------------------------------------------------------------------


@runtime_checkable
class Drafter(Protocol):
    async def draft(
        self,
        *,
        item: SectionPlanItem,
        brief: StructuredBrief,
        chunks: list[RetrievedChunk],
        feedback: str,
    ) -> str:
        """Adapt retrieved material to the new prospect. `feedback` is the
        actionable critique summary on loop iterations ('' on the first)."""
        ...


class StubDrafter:
    """Deterministic placeholder: stitches retrieved texts, addressed to the
    current client. Real impl = Gemini with the retrieved chunks as context
    and the critique feedback as revision instructions."""

    async def draft(self, *, item, brief, chunks, feedback) -> str:
        material = " ".join(c.text for c in chunks[:3])
        client = brief.client or "the client"
        base = f"{item.title} for {client}: {material}"
        if feedback:
            base += " [revised per critique]"
        return base


def _verbatim_content(chunks: list[RetrievedChunk]) -> str:
    """Stitch retrieved assets unmodified, each headed by a provenance link
    (markdown) so every entry points back to its source slide."""
    parts = []
    for c in chunks:
        label = c.metadata.get("file_name") or c.metadata.get("object_id")
        if label and c.metadata.get("slide_number") is not None:
            label = f"{label} · slide {c.metadata['slide_number']}"
        url = c.metadata.get("web_url")
        if label and url:
            parts.append(f"**[{label}]({url})**\n\n{c.text}")
        elif label:
            parts.append(f"**{label}**\n\n{c.text}")
        else:
            parts.append(c.text)
    return "\n\n---\n\n".join(parts)


def _feedback_from(critique: SectionCritique) -> str:
    """Actionable revision instructions from a critique (leaks never reach here —
    they hard-block before any redraft)."""
    parts: list[str] = []
    for p in critique.claim_proofs:
        if not p.supported:
            parts.append(f"Remove or substantiate: '{p.claim[:80]}' (no credential found).")
    for f in critique.stale_or_unverified_figures:
        parts.append(f"Drop or source the figure {f}.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


async def run_section_loop(
    item: SectionPlanItem,
    *,
    brief: StructuredBrief,
    rag: RAGClient,
    drafter: Drafter,
    tau: float = DEFAULT_TAU,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    extractor: ClaimExtractor | None = None,
) -> SectionOutcome:
    assert item.retrieval is not None, "worker requires a retrieval spec"
    spec: RetrievalSpec = item.retrieval
    traces: list[IterationTrace] = []
    feedback = ""

    name = item.section.value

    # retrieve_only sections (CVs, credentials) are near-verbatim assets: the
    # library content IS the section. No drafting, no claim proofs — generated
    # prose about people/credentials is exactly what can't be substantiated.
    if item.mode == "retrieve_only":
        log(f"{name}: retrieving assets verbatim (retrieve_only, top_k={spec.top_k})")
        chunks = await rag.retrieve(spec)
        if not chunks:
            log(f"{name}: STOP no_material (empty retrieval)")
            return SectionOutcome(
                section=name, stop_reason=StopReason.NO_MATERIAL,
                accepted=False, needs_human=True,
            )
        content = _verbatim_content(chunks)
        critique = SectionCritique(
            claims_supported_by_credential=True,
            claim_proofs=[ClaimProof(
                claim=f"verbatim library assets ({len(chunks)} chunks)",
                supported=True,
                credential_ids=[str(c.metadata.get("object_id"))
                                for c in chunks if c.metadata.get("object_id")],
                credential_refs=credential_refs_from(chunks),
            )],
            confidential_leak_detected=False,
            confidence=1.0,
        )
        log(f"{name}: STOP accepted ({len(chunks)} assets passed through verbatim)")
        return SectionOutcome(
            section=name, stop_reason=StopReason.ACCEPTED,
            accepted=True, needs_human=False,
            final_draft=content, final_critique=critique,
            iterations=[IterationTrace(iteration=0, spec_used=spec,
                                       n_retrieved=len(chunks), draft=content,
                                       critique=critique)],
        )
    for i in range(max_iterations):
        # RETRIEVE — planner's spec first, critique's proposed spec after.
        log(f"{name}: iter {i} retrieving ({item.mode}, top_k={spec.top_k})")
        chunks = (
            await rag.enrich(spec)
            if item.mode == "generate_with_rag"
            else await rag.retrieve(spec)
        )
        if not chunks:
            log(f"{name}: STOP no_material (empty retrieval)")
            return SectionOutcome(
                section=item.section.value, stop_reason=StopReason.NO_MATERIAL,
                accepted=False, needs_human=True, iterations=traces,
            )

        # DRAFT — adapt to the prospect (with revision feedback on loops).
        log(f"{name}: iter {i} drafting from {len(chunks)} chunks"
            + (" (with critique feedback)" if feedback else ""))
        draft = await drafter.draft(item=item, brief=brief, chunks=chunks,
                                    feedback=feedback)

        # CRITIQUE — structured self-reflection.
        log(f"{name}: iter {i} critiquing draft ({len(draft)} chars)")
        critique = await critique_section(
            draft, section=item.section, brief=brief, rag=rag,
            prior_spec=spec, iteration=i, extractor=extractor,
            chunks=chunks,  # provenance for attributed leak detection
        )
        traces.append(IterationTrace(
            iteration=i, spec_used=spec, n_retrieved=len(chunks),
            draft=draft, critique=critique,
        ))

        # DECIDE — stop conditions, in priority order.
        n_unsupported = sum(not p.supported for p in critique.claim_proofs)
        log(f"{name}: iter {i} critique done — confidence={critique.confidence} "
            f"leak={critique.confidential_leak_detected} "
            f"unsupported_claims={n_unsupported}")
        if critique.confidential_leak_detected:
            log(f"{name}: STOP leak_hard_block -> human review")
            # Hard block: a leak means retrieval surfaced another client's
            # material. A human must see WHAT surfaced — never silently redraft.
            return SectionOutcome(
                section=item.section.value, stop_reason=StopReason.LEAK_HARD_BLOCK,
                accepted=False, needs_human=True,
                final_draft=draft, final_critique=critique, iterations=traces,
            )

        if critique.confidence >= tau:
            log(f"{name}: STOP accepted (confidence {critique.confidence} >= tau {tau})")
            return SectionOutcome(
                section=item.section.value, stop_reason=StopReason.ACCEPTED,
                accepted=True, needs_human=False,
                final_draft=draft, final_critique=critique, iterations=traces,
            )

        # LOOP: aim the next retrieval at the gap; carry revision feedback.
        feedback = _feedback_from(critique)
        if critique.proposed_next_query is not None:
            spec = critique.proposed_next_query
        # else: same spec, redraft on feedback alone.

    log(f"{name}: STOP budget_exhausted after {max_iterations} iterations -> human review")
    last = traces[-1]
    return SectionOutcome(
        section=item.section.value, stop_reason=StopReason.BUDGET_EXHAUSTED,
        accepted=False, needs_human=True,
        final_draft=last.draft, final_critique=last.critique, iterations=traces,
    )


# ---------------------------------------------------------------------------
# Demo: three scenarios through the real loop with scripted RAG/drafter stubs
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    from planner import CanonicalSection

    class ScriptedRAG:
        """Returns configurable chunks; proof retrievals hit or miss by keyword."""

        def __init__(self, section_chunks, proof_hits: set[str]):
            self._chunks = section_chunks
            self._proof_hits = proof_hits  # keywords whose claims find credentials

        async def retrieve(self, spec: RetrievalSpec):
            if spec.sources == ["OneShelf > Credentials Library"]:
                # per-claim proof retrieval
                if any(k in spec.query.lower() for k in self._proof_hits):
                    return [RetrievedChunk(text="credential slide", score=0.0,
                                           metadata={"object_id": "cred..s7"})]
                return []
            return self._chunks

        async def enrich(self, spec: RetrievalSpec):
            return await self.retrieve(spec)

    class ScriptedDrafter:
        def __init__(self, drafts: list[str]):
            self._drafts, self._i = drafts, 0

        async def draft(self, *, item, brief, chunks, feedback) -> str:
            d = self._drafts[min(self._i, len(self._drafts) - 1)]
            self._i += 1
            return d

    async def main():
        brief = StructuredBrief(client="ACME Aerospace",
                                industry_sector="Aerospace & Defense",
                                service_line="Supply Chain & Procurement")
        item = SectionPlanItem(
            section=CanonicalSection.APPROACH, mode="generate_with_rag",
            title="Approach & Methodology",
            retrieval=RetrievalSpec(query="approach purchasing transformation",
                                    sources=["OneShelf > Proposals Library"]),
        )
        chunks = [RetrievedChunk(text="three-phase purchasing approach", score=0.8)]

        # 1) LEAK -> hard block on iteration 0, no redraft.
        rag = ScriptedRAG(chunks, proof_hits={"savings"})
        out = await run_section_loop(
            item, brief=brief, rag=rag,
            drafter=ScriptedDrafter(
                ["Our approach mirrors the roadmap we delivered for Airbus."]),
        )
        print(f"[1] {out.stop_reason.value:18s} iters={len(out.iterations)} "
              f"human={out.needs_human}")
        print(out.flags_report(), "\n")

        # 2) Unsupported claim -> loop with proposed_next_query -> revised draft accepted.
        out = await run_section_loop(
            item, brief=brief, rag=ScriptedRAG(chunks, proof_hits={"savings"}),
            drafter=ScriptedDrafter([
                "We have implemented quantum procurement in twelve galaxies.",  # unprovable
                "We have delivered savings programs in aerospace procurement.",  # provable
            ]),
        )
        print(f"[2] {out.stop_reason.value:18s} iters={len(out.iterations)} "
              f"accepted={out.accepted} confidence={out.final_critique.confidence}")
        print("    iter0 next_query:",
              out.iterations[0].critique.proposed_next_query.query[:70], "\n")

        # 3) Never provable -> budget exhausted -> escalate with flags.
        out = await run_section_loop(
            item, brief=brief, rag=ScriptedRAG(chunks, proof_hits=set()),
            drafter=ScriptedDrafter(
                ["We have delivered warp-drive retrofits for major OEMs."]),
        )
        print(f"[3] {out.stop_reason.value:18s} iters={len(out.iterations)} "
              f"human={out.needs_human}")
        print(out.flags_report())

    asyncio.run(main())
