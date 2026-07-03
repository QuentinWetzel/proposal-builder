"""
Evaluator — the check half of the generator/evaluator worker loop (stage 3).

Two axes run on EVERY section, per your instruction:

  1. PROOF (retrieval-backed)
     - generate_with_rag: every factual claim must trace to a retrieved
       citation. Uncovered claims fail and loop back, named.
     - retrieve_only (creds/CVs): "proof" = sufficiency. Did we retrieve enough
       on-target evidence, and does the top asset actually fit the brief?

  2. CONFIDENTIALITY LEAK (section-aware)
     The corpus is the failure mode: the Proposals Library is full of OTHER
     clients (Airbus, Safran, Thales, Pierre Fabre...). Risk = drafting for
     client A and leaking client B's name / pricing / engagement specifics that
     came from a retrieved slide.
       - Third-party client names are SANCTIONED in `credentials` (a credential
         is "we did X for <NamedClient>", named on purpose) and FLAGGED elsewhere.
       - The "© CYLAD Consulting ... Confidential and proprietary" slide-background
         boilerplate is stripped so it isn't misread as a leak.

The LLM-scored parts (claim extraction, coverage judgement) are behind a
protocol seam so the real Gemini/structured-output call drops in. The
deterministic parts (client-name leak scan, boilerplate strip) run as rules.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from planner import CanonicalSection, GenerationMode, RetrievedChunk


# ---------------------------------------------------------------------------
# Known-client roster — for the leak scan.
# In production, source this from the Algolia `Client` facet (the taxonomy has
# the full list) rather than hardcoding. Kept small here for the standalone run.
# ---------------------------------------------------------------------------

KNOWN_CLIENTS: set[str] = {
    "Airbus", "Airbus Helicopters", "Airbus Defence and Space", "Safran",
    "Thales", "Pierre Fabre", "Alstom", "Zodiac Aerospace", "Sanofi",
    "GE Healthcare", "Schneider Electric", "Engie", "Areva", "Valeo",
    "Bureau Veritas", "Stelia Aerospace", "Sonaca", "Liebherr Aerospace",
    "Essilor", "ATR", "Latecoere", "Arkopharma", "Colas", "Mecalac",
}

# The confidential boilerplate that rides on slide backgrounds in this corpus.
BOILERPLATE_PATTERNS: list[re.Pattern] = [
    re.compile(r"©\s*CYLAD\s+Consulting.*?written\s+agreement\s+of\s+CYLAD\s+Consulting\.?",
               re.IGNORECASE | re.DOTALL),
    re.compile(r"Confidential\s+and\s+proprietary\s+document", re.IGNORECASE),
]


def strip_boilerplate(text: str) -> str:
    for pat in BOILERPLATE_PATTERNS:
        text = pat.sub("", text)
    return text


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class Finding(BaseModel):
    axis: str  # "proof" | "confidentiality"
    severity: Severity
    message: str
    evidence: str = ""


class EvaluationResult(BaseModel):
    section: CanonicalSection
    accepted: bool
    findings: list[Finding] = Field(default_factory=list)

    def feedback(self) -> str:
        """Actionable feedback string fed back to the generator on a loop."""
        bad = [f for f in self.findings if f.severity is not Severity.PASS]
        if not bad:
            return ""
        lines = [f"- [{f.axis}/{f.severity.value}] {f.message}" for f in bad]
        return "Revise to address:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM seam for the judgement calls (claim extraction + coverage)
# ---------------------------------------------------------------------------


class ClaimCoverage(BaseModel):
    claim: str
    supported: bool
    supporting_evidence: str = ""


@runtime_checkable
class GroundednessJudge(Protocol):
    async def coverage(
        self, draft: str, evidence: list[str]
    ) -> list[ClaimCoverage]:
        """Extract factual claims from `draft`, mark each supported/unsupported
        against `evidence` (the retrieved chunk texts)."""
        ...


class StubGroundednessJudge:
    """Deterministic placeholder: treats a sentence as supported if any evidence
    chunk shares >=3 significant tokens with it. Replace with a Gemini call."""

    async def coverage(
        self, draft: str, evidence: list[str]
    ) -> list[ClaimCoverage]:
        ev_tokens = [set(_sig_tokens(e)) for e in evidence]
        out: list[ClaimCoverage] = []
        for sent in _sentences(draft):
            st = set(_sig_tokens(sent))
            hit = ""
            for e_text, e_tok in zip(evidence, ev_tokens):
                if len(st & e_tok) >= 3:
                    hit = e_text[:80]
                    break
            out.append(
                ClaimCoverage(
                    claim=sent, supported=bool(hit), supporting_evidence=hit
                )
            )
        return out


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


_STOP = {"the", "and", "for", "with", "our", "your", "this", "that", "will",
         "are", "was", "has", "have", "from", "into", "over", "per", "via"}


def _sig_tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text.lower())
            if w not in _STOP]


# ---------------------------------------------------------------------------
# Confidentiality leak scan (deterministic, section-aware)
# ---------------------------------------------------------------------------


def scan_leaks(
    draft: str,
    *,
    section: CanonicalSection,
    current_client: str | None,
    known_clients: set[str] = KNOWN_CLIENTS,
) -> list[Finding]:
    """Flag third-party client names appearing in the draft.

    Sanctioned in `credentials`; flagged elsewhere. The current client's own
    name is never a leak.
    """
    cleaned = strip_boilerplate(draft)
    findings: list[Finding] = []

    cur = (current_client or "").strip().lower()
    for client in known_clients:
        if client.lower() == cur:
            continue
        # word-boundary, case-insensitive
        if re.search(rf"\b{re.escape(client)}\b", cleaned, re.IGNORECASE):
            if section is CanonicalSection.CREDENTIALS:
                findings.append(Finding(
                    axis="confidentiality", severity=Severity.PASS,
                    message=f"Third-party client '{client}' present in credentials "
                            f"(sanctioned).", evidence=client))
            else:
                findings.append(Finding(
                    axis="confidentiality", severity=Severity.FAIL,
                    message=f"Leak: third-party client '{client}' appears in "
                            f"'{section.value}' — remove or genericize.",
                    evidence=client))

    # Heuristic: figures that look like another engagement's pricing/volumes,
    # outside the commercial section. A real impl ties numbers to their source
    # chunk's client metadata; here we just warn on €/$ + big numbers in
    # narrative sections. (Non-blocking.)
    if section not in {CanonicalSection.COMMERCIAL, CanonicalSection.CREDENTIALS}:
        if re.search(r"[€$]\s?\d[\d.,]{3,}", cleaned):
            findings.append(Finding(
                axis="confidentiality", severity=Severity.WARN,
                message="Monetary figure in a non-commercial section — verify it "
                        "isn't carried over from a retrieved past engagement."))
    return findings


# ---------------------------------------------------------------------------
# Proof check
# ---------------------------------------------------------------------------


async def check_proof(
    draft: str,
    *,
    mode: GenerationMode,
    chunks: list[RetrievedChunk],
    judge: GroundednessJudge,
    min_evidence: int = 3,
) -> list[Finding]:
    if mode == "retrieve_only":
        # Proof = sufficiency of retrieved evidence.
        n = len(chunks)
        if n == 0:
            return [Finding(axis="proof", severity=Severity.FAIL,
                            message="No assets retrieved for this section.")]
        if n < min_evidence:
            return [Finding(axis="proof", severity=Severity.WARN,
                            message=f"Only {n} asset(s) retrieved "
                                    f"(< {min_evidence}); coverage may be thin.")]
        scored = [c for c in chunks if c.score is not None]
        top = max(scored, key=lambda c: c.score) if scored else None
        # score is None when the pipeline doesn't surface rerank scores yet —
        # unknown, not poor fit. Warn only on a genuinely low surfaced score
        # (sigmoid-normalized scale; threshold tuned on the golden set).
        if top is not None and top.score < 0.3:
            return [Finding(axis="proof", severity=Severity.WARN,
                            message=f"Top asset score {top.score:.2f} is low — "
                                    f"the match may not fit the brief.")]
        return [Finding(axis="proof", severity=Severity.PASS,
                        message=f"{n} on-target assets retrieved.")]

    if mode == "template":
        return [Finding(axis="proof", severity=Severity.PASS,
                        message="Templated section — no claims to ground.")]

    # generate_with_rag: claim-to-citation coverage.
    evidence = [strip_boilerplate(c.text) for c in chunks]
    coverage = await judge.coverage(strip_boilerplate(draft), evidence)
    unsupported = [c for c in coverage if not c.supported]
    if unsupported:
        preview = "; ".join(c.claim[:60] for c in unsupported[:3])
        return [Finding(axis="proof", severity=Severity.FAIL,
                        message=f"{len(unsupported)} unsupported claim(s): {preview}")]
    return [Finding(axis="proof", severity=Severity.PASS,
                    message=f"All {len(coverage)} claims grounded in citations.")]


# ---------------------------------------------------------------------------
# Top-level evaluate() — called by the worker after each generate attempt
# ---------------------------------------------------------------------------


async def evaluate_section(
    draft: str,
    *,
    section: CanonicalSection,
    mode: GenerationMode,
    chunks: list[RetrievedChunk],
    current_client: str | None,
    judge: GroundednessJudge | None = None,
) -> EvaluationResult:
    judge = judge or StubGroundednessJudge()
    findings: list[Finding] = []
    findings += await check_proof(draft, mode=mode, chunks=chunks, judge=judge)
    findings += scan_leaks(draft, section=section, current_client=current_client)

    accepted = not any(f.severity is Severity.FAIL for f in findings)
    return EvaluationResult(section=section, accepted=accepted, findings=findings)


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        # A narrative draft for ACME that accidentally leaks "Airbus" from a
        # retrieved past-proposal slide.
        leaky = (
            "We propose a three-phase purchasing transformation for ACME. "
            "Our approach mirrors the savings roadmap we built for Airbus, "
            "targeting €4,200,000 in addressable spend."
        )
        chunks = [
            RetrievedChunk(text="Purchasing transformation three-phase approach "
                                "savings roadmap addressable spend.", score=0.8),
            RetrievedChunk(text="© CYLAD Consulting GmbH. All rights reserved. "
                                "Confidential and proprietary document.", score=0.2),
        ]
        res = await evaluate_section(
            leaky,
            section=CanonicalSection.APPROACH,
            mode="generate_with_rag",
            chunks=chunks,
            current_client="ACME Aerospace",
        )
        print("APPROACH section — accepted:", res.accepted)
        for f in res.findings:
            print(f"  [{f.axis}/{f.severity.value}] {f.message}")
        print("\nFeedback to generator:\n" + (res.feedback() or "(none)"))

        # Same client name inside a credentials section -> sanctioned, not a leak.
        cred = "Selected credential: purchasing savings program delivered for Airbus."
        res2 = await evaluate_section(
            cred,
            section=CanonicalSection.CREDENTIALS,
            mode="retrieve_only",
            chunks=chunks + [RetrievedChunk(text="cred", score=0.6)],
            current_client="ACME Aerospace",
        )
        print("\nCREDENTIALS section — accepted:", res2.accepted)
        for f in res2.findings:
            print(f"  [{f.axis}/{f.severity.value}] {f.message}")

    asyncio.run(main())
