"""
Proposal Builder — Planner stage.

Architecture (this file implements stage 1 of 3):

    PLANNER (this file)
        parse RFP brief -> StructuredBrief
        LLM proposes active sections -> rules validate/repair
        planner derives per-section retrieval specs (facet filters)
        emits SectionPlan
              |
              v
    ORCHESTRATOR  (stage 2 — stubbed edge here)
        fan out one WORKER per active section via Send()
              |
              v
    WORKER = generator/evaluator loop  (stage 3 — protocol boundary here)
        narrative sections -> RAGClient.enrich() then generate
        creds/CV sections  -> RAGClient.retrieve() only
        evaluate -> accept or loop with feedback (max N)

The planner owns filter derivation: it has the whole-RFP view, so it maps the
brief onto per-section facet specs. Workers receive a ready RetrievalSpec and
stay dumb. This keeps the plan fully inspectable in the review-before-export UI.

Downstream stages bolt onto GraphState without reshaping it. The RAGClient
Protocol is the single seam where the real LlamaIndex endpoint gets adapted in.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Annotated, Any, Literal, Protocol, TypedDict, runtime_checkable

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Canonical structure (derived from the Proposals Library analysis)
# ---------------------------------------------------------------------------


class CanonicalSection(str, Enum):
    """The canonical proposal sections. Order here is the default deck order."""

    COVER = "cover"
    CONTEXT_OBJECTIVES = "context_objectives"
    SCOPE_DELIVERABLES = "scope_deliverables"
    APPROACH = "approach"
    WORK_PLAN = "work_plan"
    TEAM = "team"
    CREDENTIALS = "credentials"
    COMMERCIAL = "commercial"
    NEXT_STEPS = "next_steps"


# Sections that must always be present regardless of what the LLM proposes.
MANDATORY_SECTIONS: set[CanonicalSection] = {
    CanonicalSection.COVER,
    CanonicalSection.COMMERCIAL,
    CanonicalSection.NEXT_STEPS,
}

# Canonical ordering used to sort/repair any proposed plan.
CANONICAL_ORDER: list[CanonicalSection] = list(CanonicalSection)

# Routing: which sections are retrieval-only (near-verbatim assets) vs.
# generated narrative enriched by RAG. Cover/next_steps are templated (neither).
RETRIEVE_ONLY: set[CanonicalSection] = {
    CanonicalSection.CREDENTIALS,
    CanonicalSection.TEAM,  # CVs are pulled, not written
}
GENERATE_WITH_RAG: set[CanonicalSection] = {
    CanonicalSection.CONTEXT_OBJECTIVES,
    CanonicalSection.SCOPE_DELIVERABLES,
    CanonicalSection.APPROACH,
    CanonicalSection.WORK_PLAN,
    CanonicalSection.COMMERCIAL,
}


def default_mode(section: CanonicalSection) -> "GenerationMode":
    if section in RETRIEVE_ONLY:
        return "retrieve_only"
    if section in GENERATE_WITH_RAG:
        return "generate_with_rag"
    return "template"


GenerationMode = Literal["retrieve_only", "generate_with_rag", "template"]


# ---------------------------------------------------------------------------
# Structured brief (output of the RFP-extraction call)
# ---------------------------------------------------------------------------


class StructuredBrief(BaseModel):
    """Structured fields extracted from the raw RFP brief.

    Fields intentionally mirror the Algolia facets so the planner can translate
    them straight into per-section retrieval filters.
    """

    client: str | None = Field(None, description="Client / company name")
    industry_sector: str | None = Field(
        None, description="Maps to IndustrySector facet, e.g. 'Aerospace & Defense'"
    )
    service_line: str | None = Field(
        None, description="Maps to ServiceLine facet, e.g. 'Supply Chain & Procurement'"
    )
    functions: list[str] = Field(
        default_factory=list, description="Maps to Function facet values"
    )
    scope_items: list[str] = Field(
        default_factory=list, description="Discrete scope / requirement items in the RFP"
    )
    stated_deliverables: list[str] = Field(default_factory=list)
    timeline: str | None = None
    language: str | None = Field(
        None, description="Proposal language: French / English / German"
    )
    notes: str | None = None


# ---------------------------------------------------------------------------
# Retrieval spec — planner-derived, worker-consumed
# ---------------------------------------------------------------------------


class RetrievalSpec(BaseModel):
    """A ready-to-execute retrieval instruction for one section.

    `facet_filters` uses the same attribute names the Algolia index exposes.
    `query` is the natural-language query for the hybrid/RAG retrieval.
    `sources` narrows to library (e.g. Credentials Library, CV Library).
    """

    query: str
    facet_filters: dict[str, list[str]] = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list)
    top_k: int = 6


class SectionPlanItem(BaseModel):
    section: CanonicalSection
    active: bool = True
    mode: GenerationMode
    title: str = Field(..., description="Human-facing section title for the deck")
    rationale: str = Field("", description="Why this section is in/out — shown in UI")
    retrieval: RetrievalSpec | None = None

    @field_validator("mode")
    @classmethod
    def _mode_matches_section(cls, v: GenerationMode, info: Any) -> GenerationMode:
        return v


class SectionPlan(BaseModel):
    items: list[SectionPlanItem]

    def active_items(self) -> list[SectionPlanItem]:
        return [i for i in self.items if i.active]


# ---------------------------------------------------------------------------
# LLM proposal model (what the section-proposal call returns, pre-repair)
# ---------------------------------------------------------------------------


class ProposedSection(BaseModel):
    section: CanonicalSection
    include: bool
    reason: str = ""


class SectionProposal(BaseModel):
    """Raw LLM output: which canonical sections to include and why."""

    sections: list[ProposedSection]


# ---------------------------------------------------------------------------
# RAG client seam — the single boundary to the LlamaIndex endpoint
# ---------------------------------------------------------------------------


class RetrievedChunk(BaseModel):
    text: str
    # Sigmoid-normalized rerank score in (0,1); None = pipeline didn't surface it.
    # None is 'unknown', NOT zero — 0.02 is a real score meaning 'confidently
    # irrelevant', which is different information.
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class RAGClient(Protocol):
    """Protocol the worker uses. Adapt your LlamaIndex repo to this.

    Paste your endpoint's real signature and we write a thin adapter that
    conforms — nothing else in the graph changes.
    """

    async def retrieve(self, spec: RetrievalSpec) -> list[RetrievedChunk]:
        """Pure retrieval — used by retrieve_only sections (creds, CVs)."""
        ...

    async def enrich(self, spec: RetrievalSpec) -> list[RetrievedChunk]:
        """Retrieval for grounding a generated narrative section."""
        ...


class StubRAGClient:
    """Runnable placeholder so this file works end-to-end today."""

    async def retrieve(self, spec: RetrievalSpec) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                text=f"[stub retrieval] query={spec.query!r} "
                f"filters={spec.facet_filters} sources={spec.sources}",
                score=1.0,
                metadata={"stub": True},
            )
        ]

    async def enrich(self, spec: RetrievalSpec) -> list[RetrievedChunk]:
        return await self.retrieve(spec)


# ---------------------------------------------------------------------------
# LLM seam — swap for your model client (structured output)
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMClient(Protocol):
    async def extract_brief(self, raw_brief: str) -> StructuredBrief:
        ...

    async def propose_sections(self, brief: StructuredBrief) -> SectionProposal:
        ...


class StubLLMClient:
    """Deterministic placeholder. Replace with a structured-output LLM call."""

    async def extract_brief(self, raw_brief: str) -> StructuredBrief:
        # A real impl uses structured output (Pydantic) against the raw brief.
        return StructuredBrief(
            client="ACME Aerospace",
            industry_sector="Aerospace & Defense",
            service_line="Supply Chain & Procurement",
            functions=["Purchasing"],
            scope_items=["Assess current purchasing org", "Identify savings"],
            stated_deliverables=["As-is assessment", "Savings roadmap"],
            timeline="12 weeks",
            language="English",
            notes=f"(stub) parsed from {len(raw_brief)} chars",
        )

    async def propose_sections(self, brief: StructuredBrief) -> SectionProposal:
        # A real impl asks the LLM which canonical sections this RFP warrants.
        return SectionProposal(
            sections=[
                ProposedSection(section=s, include=True, reason="relevant")
                for s in CanonicalSection
                if s
                not in {CanonicalSection.NEXT_STEPS}  # pretend LLM forgot one
            ]
        )


# ---------------------------------------------------------------------------
# Filter derivation — planner owns this
# ---------------------------------------------------------------------------


def _facet_filters_for(
    section: CanonicalSection, brief: StructuredBrief
) -> tuple[dict[str, list[str]], list[str], str]:
    """Return (facet_filters, sources, query) for a section given the brief."""
    filters: dict[str, list[str]] = {}
    sources: list[str] = []

    # Common facets from the brief, applied where they help relevance.
    if brief.industry_sector:
        filters["IndustrySector"] = [brief.industry_sector]
    if brief.service_line:
        filters["ServiceLine"] = [brief.service_line]

    if section is CanonicalSection.CREDENTIALS:
        sources = ["OneShelf > Credentials Library"]
        query = (
            f"credentials {brief.service_line or ''} "
            f"{brief.industry_sector or ''}".strip()
        )
    elif section is CanonicalSection.TEAM:
        sources = ["OneShelf > CV Library"]
        # CVs: query-only matching. Facet filters (sector or Function) starve
        # retrieval — the live CV Library doesn't facet cleanly by Function.
        filters = {}
        query = f"consultant CV {' '.join(brief.functions) or brief.service_line or ''}".strip()
    else:
        sources = ["OneShelf > Proposals Library"]
        topic = ", ".join(brief.scope_items) or (brief.service_line or "")
        query = f"{section.value.replace('_', ' ')} {topic}".strip()

    return filters, sources, query


def _build_retrieval(
    section: CanonicalSection, mode: GenerationMode, brief: StructuredBrief
) -> RetrievalSpec | None:
    if mode == "template":
        return None
    filters, sources, query = _facet_filters_for(section, brief)
    return RetrievalSpec(query=query, facet_filters=filters, sources=sources)


def _title_for(section: CanonicalSection, brief: StructuredBrief) -> str:
    titles = {
        CanonicalSection.COVER: f"{brief.client or 'Client'} — Technical & Commercial Proposal",
        CanonicalSection.CONTEXT_OBJECTIVES: "Context & Objectives",
        CanonicalSection.SCOPE_DELIVERABLES: "Scope & Deliverables",
        CanonicalSection.APPROACH: "Approach & Methodology",
        CanonicalSection.WORK_PLAN: "Work Plan & Planning",
        CanonicalSection.TEAM: "Team & Organization",
        CanonicalSection.CREDENTIALS: "Credentials & References",
        CanonicalSection.COMMERCIAL: "Commercial Proposal",
        CanonicalSection.NEXT_STEPS: "Next Steps",
    }
    return titles[section]


# ---------------------------------------------------------------------------
# Validate / repair — deterministic rules over the LLM proposal
# ---------------------------------------------------------------------------


def validate_and_repair(
    proposal: SectionProposal, brief: StructuredBrief
) -> SectionPlan:
    """Turn a raw LLM proposal into a well-formed, ordered SectionPlan.

    Rules:
      1. Force mandatory sections present (cover, commercial, next_steps).
      2. Drop anything not in the canonical enum (enum already guarantees this).
      3. Enforce canonical ordering.
      4. Assign routing mode + planner-derived retrieval spec.
    """
    included: dict[CanonicalSection, str] = {
        p.section: p.reason for p in proposal.sections if p.include
    }

    # Rule 1: mandatory sections always in.
    for m in MANDATORY_SECTIONS:
        included.setdefault(m, "mandatory section — always included")

    # Rules 3 + 4: order canonically, assign mode + retrieval.
    items: list[SectionPlanItem] = []
    for section in CANONICAL_ORDER:
        active = section in included
        mode = default_mode(section)
        items.append(
            SectionPlanItem(
                section=section,
                active=active,
                mode=mode,
                title=_title_for(section, brief),
                rationale=included.get(section, "not warranted by this RFP"),
                retrieval=_build_retrieval(section, mode, brief) if active else None,
            )
        )
    return SectionPlan(items=items)


# ---------------------------------------------------------------------------
# LangGraph state + planner node
# ---------------------------------------------------------------------------


def _merge_sections(left: dict, right: dict) -> dict:
    """Reducer for the sections map that workers will populate in stage 2."""
    out = dict(left)
    out.update(right)
    return out


class GraphState(TypedDict, total=False):
    # Inputs
    raw_brief: str
    # Planner outputs
    brief: StructuredBrief
    plan: SectionPlan
    # Orchestrator/worker outputs (stage 2/3) — reducer-merged as workers finish
    sections: Annotated[dict[str, dict], _merge_sections]


async def planner_node(state: GraphState, *, llm: LLMClient) -> dict:
    """Stage 1: parse brief, propose sections, validate/repair into a plan."""
    from progress import log

    raw = state["raw_brief"]
    log(f"planner: extracting brief ({len(raw)} chars)")
    brief = await llm.extract_brief(raw)
    log(f"planner: brief extracted — client={brief.client!r} "
        f"sector={brief.industry_sector!r} language={brief.language!r}")
    proposal = await llm.propose_sections(brief)
    plan = validate_and_repair(proposal, brief)
    log(f"planner: plan ready — {len(plan.active_items())}/{len(plan.items)} "
        f"sections active")
    return {"brief": brief, "plan": plan, "sections": {}}


# ---------------------------------------------------------------------------
# Graph wiring — planner as entry point, orchestrator edge stubbed
# ---------------------------------------------------------------------------


def build_graph(llm: LLMClient | None = None):
    from functools import partial

    from langgraph.graph import END, START, StateGraph

    llm = llm or StubLLMClient()

    def _orchestrator_placeholder(state: GraphState) -> dict:
        # Stage 2 lives here: read state['plan'].active_items() and fan out
        # one worker per section via Send(). Stubbed so the graph runs today.
        return {}

    g = StateGraph(GraphState)
    g.add_node("planner", partial(planner_node, llm=llm))
    g.add_node("orchestrator", _orchestrator_placeholder)
    g.add_edge(START, "planner")
    g.add_edge("planner", "orchestrator")
    g.add_edge("orchestrator", END)
    return g.compile()


# ---------------------------------------------------------------------------
# Local run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        graph = build_graph()
        result = await graph.ainvoke(
            {"raw_brief": "RFP: ACME Aerospace seeks procurement transformation..."}
        )
        brief: StructuredBrief = result["brief"]
        plan: SectionPlan = result["plan"]

        print("=== STRUCTURED BRIEF ===")
        print(json.dumps(brief.model_dump(), indent=2))
        print("\n=== SECTION PLAN ===")
        for item in plan.items:
            flag = "ON " if item.active else "off"
            print(f"[{flag}] {item.section.value:20s} mode={item.mode:17s} {item.title}")
            if item.retrieval:
                print(
                    f"        query={item.retrieval.query!r}\n"
                    f"        filters={item.retrieval.facet_filters} "
                    f"sources={item.retrieval.sources}"
                )

    asyncio.run(main())
