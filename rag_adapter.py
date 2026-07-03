"""
RAG adapter — conforms the Chat-with-your-docs pipeline to the RAGClient protocol.

Pinned against the real source (pipeline.py / schema.py / server.py):

  Request  : ChatRequest{query, filters: MetadataFilterSpec, explicit_filters_only}
  Response : ChatResponse{answer, citations: list[Citation], ...stats, timings}
  Endpoint : POST /chat  (also /chat/stream SSE — not needed here)

Two source facts drive the design:

  1. `explicit_filters_only=True` makes the pipeline SKIP understand_query and use
     ONLY our filters. This is exactly the planner-owns-filters contract: we don't
     fight per-section query understanding, we switch it off. So the planner's
     whole-RFP filters constrain both legs verbatim.

  2. MetadataFilterSpec fields are snake_case (industry_sector, service_line,
     function, source, ...). The planner emits Algolia-style names
     (IndustrySector, ServiceLine, Function). The adapter maps between them.
     Encoded-facet resolution (";#Label|GUID") is handled INSIDE the pipeline
     (_resolve_spec -> taxonomy). We pass plain human labels — do NOT encode here.

Contract now includes (pipeline-side edits landed):
  - ChatRequest.generate: False -> retrieval-only, no Gemini synthesis.
  - ChatRequest.top_k: per-request rerank depth (proof=3, section=6...).
  - Citation.rerank_score: sigmoid-normalized (0,1) via FlagEmbedding
    normalize=True. None only on legacy servers.
  - Citation.client: decoded Client label -> provenance for leak attribution.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from planner import RAGClient, RetrievalSpec, RetrievedChunk


# ---------------------------------------------------------------------------
# Facet-name mapping: planner (Algolia-style) -> MetadataFilterSpec (snake_case)
# ---------------------------------------------------------------------------

# Only these facets exist on MetadataFilterSpec. Anything else the planner emits
# is dropped (with the drop being observable via _unmapped()).
_FACET_MAP: dict[str, str] = {
    "Client": "client",
    "IndustrySector": "industry_sector",
    "ServiceLine": "service_line",
    "Function": "function",
    "DocumentPurpose": "document_purpose",
    "language": "language",
    "drive_name": "drive_name",
    "site_display_name": "site_display_name",
    # already snake_case / pass-through
    "source": "source",
}


def _map_filters(planner_filters: dict[str, list[str]]) -> tuple[dict[str, list[str]], list[str]]:
    """Return (metadata_filter_kwargs, unmapped_keys)."""
    out: dict[str, list[str]] = {}
    unmapped: list[str] = []
    for k, v in planner_filters.items():
        target = _FACET_MAP.get(k)
        if target is None:
            unmapped.append(k)
            continue
        out.setdefault(target, [])
        out[target].extend(v)
    return out, unmapped


def _to_filter_spec_dict(spec: RetrievalSpec) -> tuple[dict[str, Any], list[str]]:
    """Build the MetadataFilterSpec JSON from a planner RetrievalSpec."""
    filters, unmapped = _map_filters(spec.facet_filters)
    if spec.sources:
        filters.setdefault("source", [])
        filters["source"].extend(spec.sources)
    # de-dupe while preserving order
    for k in list(filters):
        filters[k] = list(dict.fromkeys(filters[k]))
    return filters, unmapped


# Algolia rejects query params of 512 bytes or more with a 500 from the
# pipeline. Clamp every outgoing query here — the adapter is the contract
# boundary — at a word boundary, with headroom.
_MAX_QUERY_BYTES = 480


def _clamp_query(q: str) -> str:
    if len(q.encode("utf-8")) < _MAX_QUERY_BYTES:
        return q
    cut = q.encode("utf-8")[:_MAX_QUERY_BYTES].decode("utf-8", errors="ignore")
    return cut.rsplit(" ", 1)[0] if " " in cut else cut


def _chat_request(spec: RetrievalSpec, *, generate: bool) -> dict[str, Any]:
    """Assemble the real ChatRequest body.

    explicit_filters_only=True -> planner filters are authoritative; the pipeline
    does not re-infer them from the query text. top_k rides through from the
    planner/critique spec (proof retrievals use 3, sections 6).
    """
    filter_spec, _unmapped = _to_filter_spec_dict(spec)
    return {
        "query": _clamp_query(spec.query),
        "filters": filter_spec,
        "explicit_filters_only": True,
        "generate": generate,
        "top_k": spec.top_k,
    }


# ---------------------------------------------------------------------------
# Response mapping: ChatResponse.citations -> list[RetrievedChunk]
# ---------------------------------------------------------------------------


def _citation_to_chunk(c: dict[str, Any]) -> RetrievedChunk:
    """Map one Citation. `snippet` is the slide text; metadata carries the rest.

    `score` is None until the pipeline surfaces rerank_score on Citation
    (see notes). None means 'unknown' — distinct from a low real score like
    0.02, which means 'confidently irrelevant'. Expected scale once surfaced:
    sigmoid-normalized (0,1) — normalize in the pipeline (FlagEmbedding
    compute_score(..., normalize=True)) so thresholds here are stable.
    """
    raw = c.get("rerank_score")
    return RetrievedChunk(
        text=c.get("snippet", ""),
        score=float(raw) if raw is not None else None,
        metadata={
            "object_id": c.get("object_id"),
            "file_name": c.get("file_name"),
            "slide_number": c.get("slide_number"),
            "source": c.get("source"),
            "web_url": c.get("web_url"),
            # decoded Client label — provenance for attributed leak detection:
            # "this chunk came from a Client X deck" beats name-list matching.
            "client": c.get("client"),
        },
    )


def _parse_response(payload: dict[str, Any]) -> tuple[list[RetrievedChunk], str | None]:
    chunks = [_citation_to_chunk(c) for c in payload.get("citations", [])]
    return chunks, payload.get("answer")


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


class DocChatRAGClient:
    """Conforms the FastAPI pipeline to planner.RAGClient.

    Both methods hit POST /chat with explicit_filters_only=True. They differ in
    how the response is used:
      retrieve() -> citations only (answer dropped) for retrieve_only sections.
      enrich()   -> answer surfaced as a grounding chunk for narrative sections.
    """

    def __init__(self, base_url: str, *, timeout: float = 60.0) -> None:
        self._base = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)

    async def _post_chat(self, spec: RetrievalSpec, *, generate: bool) -> dict[str, Any]:
        # One transient gateway error must not kill a whole graph run: a worker
        # exception panics LangGraph and discards every other section's work.
        attempts = 4
        for attempt in range(attempts):
            try:
                resp = await self._client.post(
                    f"{self._base}/chat", json=_chat_request(spec, generate=generate))
                if resp.status_code in (502, 503, 504) and attempt < attempts - 1:
                    raise httpx.TransportError(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                return resp.json()
            except httpx.TransportError:
                if attempt == attempts - 1:
                    raise
                await asyncio.sleep(2 ** attempt * 2)  # 2s, 4s, 8s
        raise AssertionError("unreachable")

    async def retrieve(self, spec: RetrievalSpec) -> list[RetrievedChunk]:
        payload = await self._post_chat(spec, generate=False)  # no Gemini cost
        chunks, _answer = _parse_response(payload)
        return chunks

    async def enrich(self, spec: RetrievalSpec) -> list[RetrievedChunk]:
        payload = await self._post_chat(spec, generate=True)
        chunks, answer = _parse_response(payload)
        if answer:
            chunks.insert(
                0,
                RetrievedChunk(
                    text=answer, score=1.0,
                    metadata={"kind": "synthesized_answer"},
                ),
            )
        return chunks

    async def aclose(self) -> None:
        await self._client.aclose()


# Structural conformance to the RAGClient protocol.
_: type[RAGClient] = DocChatRAGClient  # type: ignore[assignment]
