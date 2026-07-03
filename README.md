# Agentic Proposal Builder

Takes an RFP brief, plans the canonical proposal structure, and builds each
section through a **retrieve → draft → critique → re-retrieve loop** against a
metadata-filtered hybrid RAG pipeline ([chat-with-your-docs](../docchat)).
Sections are accepted only when every factual claim traces to a real credential
and no confidential material from other clients leaks in; anything else is
escalated to a human with precise flags.

## Run it

Offline demo (no services — stub LLM + built-in mini corpus; exercises the
full graph: parallel sections, claim-proof loop, leak hard-block, review queue):

    pip install langgraph langchain-core pydantic httpx
    python cli.py --offline

Live (pipeline + Gemini):

    export GEMINI_API_KEY=...
    python cli.py --brief rfp.txt --pipeline-url http://localhost:8000

UI (Gradio — paste/drop a brief, live per-worker progress + durations,
rendered draft with downloads; loads `.env` for live mode):

    pip install gradio
    python app.py

PPTX export (CYLAD template layouts; Gemini slide-ification of generated
sections, verbatim slide copy for team/credentials when the source deck is in
--slide-library; watermarked unless the review queue is clear — also a button
in the UI):

    pip install python-pptx
    python export_pptx.py draft_proposal.json --template <cylad_deck.pptx> \
        --slide-library ~/Downloads [--force-draft]

Outputs `draft_proposal.md` (draft + human-review queue) and
`draft_proposal.json` (full payload incl. per-iteration traces).
Exit code 2 = sections pending human review; nothing is exportable until the
queue is clear.

## Architecture

    planner (planner.py)        RFP -> StructuredBrief -> SectionPlan
      LLM proposes sections, deterministic rules validate/repair;
      planner owns per-section retrieval specs (facet filters).
    orchestrator (orchestrator.py)   LangGraph Send() fan-out, one worker
      per active section, parallel; template sections render directly.
    worker (worker.py)          the agentic loop, stop conditions first-class:
      leak -> HARD BLOCK to human. confidence >= tau -> accept.
      budget exhausted / no material -> escalate. else re-retrieve
      with the critique's proposed RetrievalSpec and redraft.
    critique (critique.py)      structured schema: claims_supported_by_credential
      (per-claim proof retrieval — the claim IS the query),
      confidential_leak_detected (provenance-attributed via Citation.client
      + roster + named-people/figure patterns), stale_or_unverified_figures,
      confidence (computed aggregate, not LLM self-score),
      proposed_next_query (full RetrievalSpec: can relax filters).
    rag_adapter (rag_adapter.py) conforms the pipeline's /chat contract:
      explicit_filters_only=True (planner filters authoritative),
      generate=False on proof/creds retrievals (no Gemini cost),
      per-request top_k, sigmoid-normalized rerank_score (None = unknown).
    llm_gemini (llm_gemini.py)  Gemini seams (google-genai, structured output,
      thinking disabled), cheap-model routing for extraction/claims.

## Tuning

tau (default 0.7), max iterations (default 3) — `cli.py --tau --max-iterations`.
`support_threshold` (0.3) in critique.py gates claim proof on the sigmoid-
normalized rerank score. All three should be calibrated on a golden set of
briefs, not reasoned about as probabilities.
