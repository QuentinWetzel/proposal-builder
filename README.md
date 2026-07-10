# Agentic Proposal Builder

Takes an RFP brief, plans the canonical proposal structure, and builds each
section through a **retrieve → draft → critique → re-retrieve loop** against a
metadata-filtered hybrid RAG pipeline ([chat-with-your-docs](https://github.com/QuentinWetzel/docchat)).
Sections are accepted only when every factual claim traces to a real credential
and no confidential material from other clients leaks in; anything else is
escalated to a human with precise flags.

## Run it

Offline demo (no services — stub LLM + built-in mini corpus; exercises the
full graph: parallel sections, claim-proof loop, leak hard-block, review queue):

    pip install -r requirements.txt   # offline needs only the first four
    python cli.py --offline

Live (pipeline + Gemini — copy `.env.example` to `.env` and pick an auth mode:
Vertex ADC or `GEMINI_API_KEY`; cli.py does not auto-load it):

    set -a; source .env; set +a
    python cli.py --brief rfp.txt --pipeline-url "$PB_PIPELINE_URL"

UI (Gradio — brief box prefilled with the example, live per-worker progress
+ durations, rendered draft with downloads; auto-loads `.env` for live mode):

    python app.py

PPTX export (CYLAD template layouts; Gemini slide-ification of generated
sections, verbatim slide copy for team/credentials when the source deck is in
the slide library; watermarked unless the review queue is clear — also a
button in the UI). The CYLAD template deck is bundled in `assets/`; the UI
uses it by default (`PB_PPTX_TEMPLATE` overrides, `PB_SLIDE_LIBRARY` sets
library folders, default `~/Downloads`). The CLI takes both explicitly:

    python export_pptx.py draft_proposal.json \
        --template "assets/202006 - PF - DAIS - REX Harmonie - Proposition CYLAD V3.pptx" \
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
    evaluator (evaluator.py)    the judges critique.py builds its verdict from —
      the check half of the generator/evaluator loop: groundedness judge for
      per-claim proof, known-client roster for leak attribution, CYLAD
      slide-background boilerplate stripping.
    rag_adapter (rag_adapter.py) conforms the pipeline's /chat contract:
      explicit_filters_only=True (planner filters authoritative),
      generate=False on proof/creds retrievals (no Gemini cost),
      per-request top_k, sigmoid-normalized rerank_score (None = unknown).
    llm_gemini (llm_gemini.py)  Gemini seams (google-genai, structured output,
      thinking disabled), cheap-model routing for extraction/claims.

## Tuning

tau (default 0.6), max iterations (default 3) — `cli.py --tau --max-iterations`.
`support_threshold` (0.3) in critique.py gates claim proof on the sigmoid-
normalized rerank score. All three should be calibrated on a golden set of
briefs, not reasoned about as probabilities.

## Deploy

Railway: `railway.json` runs `python app.py` (binds `0.0.0.0:$PORT` when
`PORT` is set). Configure the `.env.example` variables on the service;
`GOOGLE_APPLICATION_CREDENTIALS_JSON` carries Vertex ADC credentials on a
keyless host.
