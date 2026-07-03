"""
Critique agent — the self-reflection half of the draft -> critique -> refine loop.

Implements the spec's structured critique schema (Section 4) exactly:

    claims_supported_by_credential : bool + credential IDs
    confidential_leak_detected     : bool + spans (named people, figures, client names)
    stale_or_unverified_figures    : list of suspect figures
    confidence                     : float in [0,1] — drives the stop condition
    proposed_next_query            : structured RetrievalSpec (query + filter
                                     adjustments), NOT a bare string — so the
                                     re-retrieve can relax/tighten filters and
                                     escape a bad initial guess.

Proof retrieval is BOTH-level (your call):
  - the section is drafted from one section-level retrieval (planner spec), then
  - each factual claim runs its OWN proof retrieval against the Credentials
    Library — the claim text IS the query. `claims_supported_by_credential`
    is checkable because every claim either has a credential object_id or it
    doesn't.

Reuses evaluator.py's detection logic (leak scan, boilerplate strip); this module
is the output contract + the claim-proof step, not new detection machinery.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from evaluator import (
    KNOWN_CLIENTS,
    GroundednessJudge,
    StubGroundednessJudge,
    strip_boilerplate,
)
from planner import (
    CanonicalSection,
    RAGClient,
    RetrievalSpec,
    RetrievedChunk,
    StructuredBrief,
)

# ---------------------------------------------------------------------------
# Critique schema — the spec's structured JSON, as Pydantic
# ---------------------------------------------------------------------------


class CredentialRef(BaseModel):
    """Where a supporting credential lives — enough to render a link to it."""
    object_id: str
    file_name: str | None = None
    slide_number: int | None = None
    web_url: str | None = None


def credential_refs_from(chunks: list[RetrievedChunk]) -> list[CredentialRef]:
    return [CredentialRef(
        object_id=str(c.metadata["object_id"]),
        file_name=c.metadata.get("file_name"),
        slide_number=c.metadata.get("slide_number"),
        web_url=c.metadata.get("web_url"),
    ) for c in chunks if c.metadata.get("object_id")]


class ClaimProof(BaseModel):
    claim: str
    supported: bool
    credential_ids: list[str] = Field(
        default_factory=list, description="object_ids of supporting credential slides"
    )
    credential_refs: list[CredentialRef] = Field(
        default_factory=list, description="file/link refs for the supporting slides"
    )
    proof_query: str = Field("", description="The per-claim query that was run")


class LeakSpan(BaseModel):
    kind: str  # "client_name" | "person_name" | "figure" | "program_name"
    span: str
    context: str = ""


class SectionCritique(BaseModel):
    """One critique pass over one drafted section. The loop's control signals."""

    claims_supported_by_credential: bool
    claim_proofs: list[ClaimProof] = Field(default_factory=list)

    confidential_leak_detected: bool
    leak_spans: list[LeakSpan] = Field(default_factory=list)

    stale_or_unverified_figures: list[str] = Field(default_factory=list)

    confidence: float = Field(..., ge=0.0, le=1.0)

    proposed_next_query: RetrievalSpec | None = Field(
        None,
        description="Structured re-retrieval aimed at the gap this critique found. "
        "None when there's nothing more retrieval could fix.",
    )


# ---------------------------------------------------------------------------
# Claim extraction seam (LLM in production, heuristic stub here)
# ---------------------------------------------------------------------------


@runtime_checkable
class ClaimExtractor(Protocol):
    async def extract(self, draft: str) -> list[str]:
        """Return the factual, provable claims in the draft (experience,
        figures, outcomes) — not boilerplate or forward-looking language."""
        ...


class StubClaimExtractor:
    """Heuristic: sentences containing experience/outcome markers are claims."""

    _MARKERS = re.compile(
        r"\b(delivered|achieved|reduced|saved|implemented|deployed|built|"
        r"track record|experience|we have|savings|ROI|committed)\b",
        re.IGNORECASE,
    )

    async def extract(self, draft: str) -> list[str]:
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", draft) if s.strip()]
        return [s for s in sents if self._MARKERS.search(s)]


# ---------------------------------------------------------------------------
# Leak detection — reuse evaluator logic, emit spans per the spec
# ---------------------------------------------------------------------------

_MONEY = re.compile(r"[€$]\s?\d[\d.,]*\s?(?:[MKmk]€?|million|Mio)?")
# Named-person heuristic: honorific/title + capitalized name, or First Last near a role word.
_PERSON = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|CIO|CEO|CFO|VP|Director|Sponsor)\.?\s+[A-Z][a-zà-ü]+(?:\s+[A-Z][a-zà-ü]+)?"
)


def detect_leak_spans(
    draft: str,
    *,
    section: CanonicalSection,
    current_client: str | None,
    known_clients: set[str] = KNOWN_CLIENTS,
    chunks: list[RetrievedChunk] | None = None,
) -> list[LeakSpan]:
    """Section-aware span detection: third-party client names (sanctioned in
    credentials), named people, and hard figures outside commercial.

    Provenance-attributed: the roster of client names to scan for is the union
    of the static KNOWN_CLIENTS list and the `client` metadata of the retrieved
    chunks feeding this draft. If a chunk came from a Client X deck and X != the
    current client, X in the draft is a leak BY PROVENANCE — no name list needed.
    That closes the open-world gap: a client absent from the roster still gets
    caught the moment their deck is retrieved."""
    cleaned = strip_boilerplate(draft)
    spans: list[LeakSpan] = []
    cur = (current_client or "").strip().lower()

    provenance_clients = {
        str(c.metadata.get("client")).strip()
        for c in (chunks or [])
        if c.metadata.get("client")
    }
    scan_roster = known_clients | provenance_clients

    if section is not CanonicalSection.CREDENTIALS:
        for client in scan_roster:
            if client.lower() == cur:
                continue
            m = re.search(rf"\b{re.escape(client)}\b", cleaned, re.IGNORECASE)
            if m:
                spans.append(LeakSpan(kind="client_name", span=m.group(0),
                                      context=cleaned[max(0, m.start() - 40): m.end() + 40]))

    for m in _PERSON.finditer(cleaned):
        spans.append(LeakSpan(kind="person_name", span=m.group(0),
                              context=cleaned[max(0, m.start() - 40): m.end() + 40]))

    if section not in {CanonicalSection.COMMERCIAL, CanonicalSection.CREDENTIALS}:
        for m in _MONEY.finditer(cleaned):
            spans.append(LeakSpan(kind="figure", span=m.group(0),
                                  context=cleaned[max(0, m.start() - 40): m.end() + 40]))
    return spans


# ---------------------------------------------------------------------------
# Per-claim proof retrieval — the claim IS the query
# ---------------------------------------------------------------------------


def _proof_spec(claim: str, brief: StructuredBrief) -> RetrievalSpec:
    """Build the claim-level proof query against the Credentials Library.

    Starts scoped to the brief's sector/service line; the critique may propose
    a relaxed spec later if proof isn't found inside that scope.
    """
    filters: dict[str, list[str]] = {}
    if brief.industry_sector:
        filters["IndustrySector"] = [brief.industry_sector]
    if brief.service_line:
        filters["ServiceLine"] = [brief.service_line]
    return RetrievalSpec(
        query=claim[:200],
        facet_filters=filters,
        sources=["OneShelf > Credentials Library"],
        top_k=3,
    )


async def prove_claims(
    claims: list[str],
    *,
    brief: StructuredBrief,
    rag: RAGClient,
    support_threshold: float = 0.3,
) -> list[ClaimProof]:
    """Run one proof retrieval per claim. A claim is supported when at least one
    credential chunk clears the threshold. Chunks with score=None (pipeline not
    yet surfacing sigmoid-normalized rerank scores) count by presence — fail-open
    on 'unknown', threshold on known. Tune support_threshold on the golden set,
    like tau: sigmoid outputs are a stable scale, not calibrated probabilities."""
    proofs: list[ClaimProof] = []
    for claim in claims:
        spec = _proof_spec(claim, brief)
        chunks = await rag.retrieve(spec)
        hits = [c for c in chunks
                if c.score is None or c.score >= support_threshold]
        proofs.append(ClaimProof(
            claim=claim,
            supported=bool(hits),
            credential_ids=[str(c.metadata.get("object_id")) for c in hits
                            if c.metadata.get("object_id")],
            credential_refs=credential_refs_from(hits),
            proof_query=spec.query,
        ))
    return proofs


# ---------------------------------------------------------------------------
# proposed_next_query — structured, gap-aimed, filter-adjusting
# ---------------------------------------------------------------------------


def propose_next_query(
    proofs: list[ClaimProof],
    *,
    brief: StructuredBrief,
    prior_spec: RetrievalSpec,
    iteration: int,
) -> RetrievalSpec | None:
    """Aim the next retrieval at the biggest gap the critique found.

    Strategy: target the first unsupported claim. On later iterations, RELAX
    the sector filter — the proof may live in an adjacent sector's credential,
    and staying inside the initial filters is how loops get stuck.
    """
    unsupported = [p for p in proofs if not p.supported]
    if not unsupported:
        return None
    gap = unsupported[0].claim
    filters: dict[str, list[str]] = {}
    if brief.service_line:
        filters["ServiceLine"] = [brief.service_line]
    if iteration < 1 and brief.industry_sector:
        # first retry stays in-sector; later retries relax it
        filters["IndustrySector"] = [brief.industry_sector]
    return RetrievalSpec(
        query=f"credential evidence: {gap[:160]}",
        facet_filters=filters,
        sources=["OneShelf > Credentials Library", "OneShelf > Proposals Library"],
        top_k=6,
    )


# ---------------------------------------------------------------------------
# The critique pass
# ---------------------------------------------------------------------------


def _confidence(
    proofs: list[ClaimProof], leaks: list[LeakSpan], stale: list[str]
) -> float:
    """Deterministic confidence: claim-support ratio, floored by leaks.

    A leak zeroes confidence (hard block regardless, but the number should agree
    with the action). Stale figures shave. No claims at all -> mid confidence,
    not high: an unfalsifiable section isn't a trustworthy one.
    """
    if leaks:
        return 0.0
    if not proofs:
        base = 0.6
    else:
        base = sum(p.supported for p in proofs) / len(proofs)
    return max(0.0, round(base - 0.1 * len(stale), 3))


async def critique_section(
    draft: str,
    *,
    section: CanonicalSection,
    brief: StructuredBrief,
    rag: RAGClient,
    prior_spec: RetrievalSpec,
    iteration: int,
    extractor: ClaimExtractor | None = None,
    chunks: list[RetrievedChunk] | None = None,
) -> SectionCritique:
    extractor = extractor or StubClaimExtractor()

    claims = await extractor.extract(strip_boilerplate(draft))
    proofs = await prove_claims(claims, brief=brief, rag=rag)
    leaks = detect_leak_spans(draft, section=section,
                              current_client=brief.client,
                              chunks=chunks)

    # Stale/unverified figures: money spans in the draft that no supporting
    # credential chunk contains — a figure with no source is a suspect figure.
    stale: list[str] = []
    supported_text = " ".join(
        p.claim for p in proofs if p.supported
    )
    for m in _MONEY.finditer(strip_boilerplate(draft)):
        fig = m.group(0)
        holder = next((p for p in proofs if fig in p.claim), None)
        if holder is None or not holder.supported:
            stale.append(fig)

    all_supported = all(p.supported for p in proofs) if proofs else True
    leak_detected = bool(leaks)

    return SectionCritique(
        claims_supported_by_credential=all_supported,
        claim_proofs=proofs,
        confidential_leak_detected=leak_detected,
        leak_spans=leaks,
        stale_or_unverified_figures=list(dict.fromkeys(stale)),
        confidence=_confidence(proofs, leaks, stale),
        proposed_next_query=propose_next_query(
            proofs, brief=brief, prior_spec=prior_spec, iteration=iteration
        ),
    )
