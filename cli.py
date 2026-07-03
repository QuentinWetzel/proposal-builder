"""
Proposal Builder — CLI.

Live mode (real pipeline + Gemini):
    export GEMINI_API_KEY=...
    python cli.py --brief rfp.txt --pipeline-url http://localhost:8000

Offline mode (no services, stubbed LLM + built-in mini corpus — demos the full
graph: parallel sections, claim-proof loop, leak hard-block, review queue):
    python cli.py --offline

Output: draft_proposal.json (full payload incl. iteration traces) and
draft_proposal.md (human-readable draft + review queue) in --out (default .).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys

from orchestrator import build_graph
from planner import RetrievalSpec, RetrievedChunk
from worker import DEFAULT_MAX_ITERATIONS, DEFAULT_TAU

# ---------------------------------------------------------------------------
# Offline demo corpus — shaped like the real one (slides, clients, credentials)
# ---------------------------------------------------------------------------

_DEMO_SLIDES = [
    # (text, source, client, object_id, score)
    ("Three-phase purchasing transformation: diagnose, design, deploy. "
     "Workstreams per commodity family with weekly steering.",
     "OneShelf > Proposals Library", "Thales", "prop-thales..s12", 0.82),
    ("Phased work-packages, each with objectives, activities, deliverables "
     "and roles & responsibilities; Gantt roadmap over 12 weeks.",
     "OneShelf > Proposals Library", "Safran", "prop-safran..s31", 0.74),
    ("Context: rising input costs and fragmented supplier base drive the need "
     "for a structured procurement performance program.",
     "OneShelf > Proposals Library", "Alstom", "prop-alstom..s04", 0.71),
    ("Credential: purchasing savings program — 23M€ committed savings over "
     "7 years, 11M€ investment, ROI after 4 years.",
     "OneShelf > Credentials Library", "Airbus", "cred-airbus..s07", 0.88),
    ("Credential: supplier consolidation across A&D commodity families, "
     "double-digit unit-cost reduction.",
     "OneShelf > Credentials Library", "Liebherr Aerospace", "cred-lieb..s02", 0.79),
]


class OfflineRAG:
    """Keyword-scored retrieval over the demo corpus, honoring source filters.
    Proof queries (Credentials Library only) match on procurement vocabulary."""

    async def _search(self, spec: RetrievalSpec) -> list[RetrievedChunk]:
        rows = [r for r in _DEMO_SLIDES
                if not spec.sources or r[1] in spec.sources]
        q = set(spec.query.lower().split())
        out = []
        for text, source, client, oid, score in rows:
            overlap = len(q & set(text.lower().replace(":", " ").split()))
            # Mirror the real pipeline: when a source filter applies but the
            # query text doesn't hit, fall back to filters-only browse
            # (discounted score) rather than starving the section.
            if overlap:
                out.append(RetrievedChunk(
                    text=text, score=score,
                    metadata={"object_id": oid, "source": source,
                              "client": client, "file_name": oid.split("..")[0]}))
            elif spec.sources:
                out.append(RetrievedChunk(
                    text=text, score=round(score * 0.55, 2),
                    metadata={"object_id": oid, "source": source,
                              "client": client, "file_name": oid.split("..")[0]}))
        out.sort(key=lambda c: c.score or 0, reverse=True)
        return out[: spec.top_k]

    async def retrieve(self, spec):  # proof + creds/CV path
        hits = await self._search(spec)
        if spec.sources == ["OneShelf > Credentials Library"]:
            words = spec.query.lower()
            if not any(k in words for k in
                       ("purchasing", "savings", "procurement", "supplier",
                        "cost", "sourcing")):
                return []
        return hits

    async def enrich(self, spec):
        return await self._search(spec)


class OfflineDrafter:
    """First draft of work_plan leaks 'Thales' (it's in the retrieved material —
    exactly the real failure mode); everything else drafts clean + provable."""

    async def draft(self, *, item, brief, chunks, feedback) -> str:
        client = brief.client or "the client"
        material = chunks[0].text if chunks else ""
        if item.section.value == "work_plan" and not feedback:
            return (f"{item.title} for {client}: phased plan as delivered for "
                    f"Thales — {material}")
        return (f"{item.title} for {client}: we have delivered purchasing "
                f"savings programs in aerospace. Adapted approach: {material}")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _credential_links(section: dict) -> str:
    """Markdown links to the referenced credential slides; falls back to bare
    object_ids when the source has no web_url (offline corpus, legacy runs)."""
    refs = section.get("credential_refs")
    if not refs:
        return ", ".join(section.get("credential_ids") or []) or "—"
    parts = []
    for r in refs:
        label = r.get("file_name") or r["object_id"]
        if r.get("slide_number") is not None:
            label += f" · slide {r['slide_number']}"
        parts.append(f"[{label}]({r['web_url']})" if r.get("web_url") else label)
    return ", ".join(parts)


def render_markdown(assembled: dict) -> str:
    lines = [f"# Draft proposal — {assembled.get('client') or 'client'}",
             f"_exportable: {assembled['exportable']}_", ""]
    for s in assembled["sections"]:
        badge = s["status"]
        lines.append(f"## {s['title']}  `[{badge}]`")
        if s.get("confidence") is not None:
            meta = f"_confidence {s['confidence']} · {s['iterations']} iteration(s)"
            # retrieve_only entries carry their own source link inline
            if s.get("mode") != "retrieve_only":
                meta += f" · credentials: {_credential_links(s)}"
            lines.append(meta + "_")
        lines.append("")
        lines.append(s.get("content", ""))
        lines.append("")
    if assembled["review_queue"]:
        lines.append("---\n# HUMAN REVIEW REQUIRED")
        for r in assembled["review_queue"]:
            lines.append("```")
            lines.append(r["flags"] or f"{r['section']}: {r['reason']}")
            lines.append("```")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def amain() -> int:
    ap = argparse.ArgumentParser(description="Agentic proposal builder")
    ap.add_argument("--brief", type=pathlib.Path,
                    help="Path to the RFP brief (text). Required unless --offline.")
    ap.add_argument("--pipeline-url", default="http://localhost:8000",
                    help="chat-with-your-docs base URL (live mode)")
    ap.add_argument("--offline", action="store_true",
                    help="No services: stub LLM + built-in demo corpus")
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU)
    ap.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("."))
    args = ap.parse_args()

    if args.offline:
        rag, drafter, llm = OfflineRAG(), OfflineDrafter(), None  # stub LLM
        extractor = None  # heuristic StubClaimExtractor
        raw_brief = ("RFP: ACME Aerospace seeks a purchasing transformation "
                     "program: assess the purchasing organisation, identify "
                     "savings, 12 weeks.")
    else:
        if not args.brief:
            ap.error("--brief is required in live mode")
        raw_brief = args.brief.read_text()
        from llm_gemini import GeminiClaimExtractor, GeminiDrafter, GeminiLLMClient
        from rag_adapter import DocChatRAGClient
        # 7 sections fan out in parallel; generate=True calls queue on the
        # pipeline, so per-request latency stacks well past the 60s default.
        rag = DocChatRAGClient(args.pipeline_url, timeout=300.0)
        drafter = GeminiDrafter()
        llm = GeminiLLMClient()
        # LLM claim extraction: the regex stub over-extracts forward-looking
        # scope text as "claims", which can never prove and exhausts budgets.
        extractor = GeminiClaimExtractor()

    from progress import log
    mode = "offline" if args.offline else f"live ({args.pipeline_url})"
    log(f"run starting — {mode}, tau={args.tau}, "
        f"max_iterations={args.max_iterations}")
    graph = build_graph(rag=rag, drafter=drafter, llm=llm,
                        tau=args.tau, max_iterations=args.max_iterations,
                        extractor=extractor)
    result = await graph.ainvoke({"raw_brief": raw_brief})
    log("run complete — writing outputs")
    assembled = result["assembled"]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "draft_proposal.json").write_text(
        json.dumps({"assembled": assembled,
                    "plan": result["plan"].model_dump(),
                    "brief": result["brief"].model_dump(),
                    "outcomes": result.get("outcomes", {})},
                   indent=2, default=str))
    md = render_markdown(assembled)
    (args.out / "draft_proposal.md").write_text(md)

    print(md)
    print(f"\nWrote {args.out/'draft_proposal.json'} and {args.out/'draft_proposal.md'}")
    if not args.offline and hasattr(rag, "aclose"):
        await rag.aclose()
    return 0 if assembled["exportable"] else 2  # nonzero = human review pending


if __name__ == "__main__":
    sys.exit(asyncio.run(amain()))
