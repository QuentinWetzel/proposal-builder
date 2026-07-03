"""
Proposal Builder — Gradio UI.

    .venv/bin/python app.py          # http://127.0.0.1:7860

Paste the RFP brief or drop a file, pick offline/live, hit Run. The status
board shows the planner, every section worker (they run in parallel), and
synthesis, each with live stage + duration; the finished draft renders below
with download links for draft_proposal.md / .json.

Progress comes from progress.add_listener: the same log() calls the CLI
prints to stderr are parsed here into per-section state. No pipeline code
runs differently under the UI.
"""

from __future__ import annotations

import asyncio
import html
import json
import os
import pathlib
import re
import tempfile
import time
from dataclasses import dataclass, field

import gradio as gr

import progress
from cli import OfflineDrafter, OfflineRAG, render_markdown
from orchestrator import build_graph
from worker import DEFAULT_MAX_ITERATIONS, DEFAULT_TAU

# ---------------------------------------------------------------------------
# Environment (live mode wants .env: Vertex ADC vars, PB_PIPELINE_URL)
# ---------------------------------------------------------------------------


def _load_env() -> None:
    p = pathlib.Path(__file__).parent / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            v = v.strip()
            if len(v) >= 2 and v[0] in "'\"" and v.endswith(v[0]):
                v = v[1:-1]
            else:  # unquoted: shell-style inline comments end the value
                v = re.split(r"\s+#", v, maxsplit=1)[0].strip()
            os.environ.setdefault(k.strip(), v)


def _materialize_service_account() -> None:
    """Vertex auth on PaaS hosts (Railway): no gcloud ADC there, so accept the
    service-account key as JSON in GOOGLE_APPLICATION_CREDENTIALS_JSON and
    point GOOGLE_APPLICATION_CREDENTIALS at a file holding it."""
    raw = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not raw or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return
    p = pathlib.Path(tempfile.gettempdir()) / "gcp-sa.json"
    p.write_text(raw)
    p.chmod(0o600)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(p)


_load_env()
_materialize_service_account()

DEFAULT_PIPELINE_URL = os.environ.get("PB_PIPELINE_URL", "http://localhost:8000")

DEMO_BRIEF = ("RFP: ACME Aerospace seeks a purchasing transformation "
              "program: assess the purchasing organisation, identify "
              "savings, 12 weeks.")

# ---------------------------------------------------------------------------
# Run state — one Row per pipeline stage, fed by parsed log events
# ---------------------------------------------------------------------------

_STATUS_STYLE = {
    "pending":   ("waiting", "#6b7280", "#f3f4f6"),
    "running":   ("running", "#1d4ed8", "#dbeafe"),
    "accepted":  ("accepted", "#15803d", "#dcfce7"),
    "templated": ("done", "#15803d", "#dcfce7"),
    "done":      ("done", "#15803d", "#dcfce7"),
    "escalated": ("needs human", "#b45309", "#fef3c7"),
    "leak":      ("LEAK — blocked", "#b91c1c", "#fee2e2"),
    "error":     ("error", "#b91c1c", "#fee2e2"),
}


@dataclass
class Row:
    label: str
    status: str = "pending"
    detail: str = ""
    start: float | None = None
    end: float | None = None

    def duration(self, now: float) -> str:
        if self.start is None:
            return "—"
        return f"{(self.end if self.end is not None else now) - self.start:.1f}s"


@dataclass
class RunState:
    planner: Row = field(default_factory=lambda: Row("Planner"))
    synthesis: Row = field(default_factory=lambda: Row("Synthesis"))
    workers: dict[str, Row] = field(default_factory=dict)  # insertion-ordered
    expected_workers: int | None = None
    stopped_workers: int = 0

    def worker(self, name: str, t: float) -> Row:
        if name not in self.workers:
            self.workers[name] = Row(
                label=name.replace("_", " ").title(), status="running", start=t)
        return self.workers[name]


_STOP_STATUS = {"accepted": "accepted", "leak_hard_block": "leak",
                "budget_exhausted": "escalated", "no_material": "escalated"}

_STOP_DETAIL = {"accepted": "accepted", "leak_hard_block": "leak → hard block",
                "budget_exhausted": "budget exhausted → human review",
                "no_material": "no material → human review"}


def apply_event(state: RunState, t: float, msg: str) -> None:
    m = re.match(r"^([a-z_]+): (.*)$", msg)
    if not m:
        return
    name, rest = m.group(1), m.group(2)

    if name == "planner":
        state.planner.status = "running"
        state.planner.detail = rest.split(" — ")[0]
        return

    if name == "orchestrator":
        state.planner.status, state.planner.end = "done", t
        fan = re.search(r"fanning out (\d+)", rest)
        if fan:
            state.expected_workers = int(fan.group(1))
            state.synthesis.detail = f"waiting for {fan.group(1)} workers"
        return

    if name == "assemble":
        state.synthesis.status, state.synthesis.end = "done", t
        if state.synthesis.start is None:
            state.synthesis.start = t
        state.synthesis.detail = rest.split(" -> ")[0]
        return

    row = state.worker(name, t)
    stop = re.match(r"STOP (\w+)", rest)
    if stop:
        reason = stop.group(1)
        row.status = _STOP_STATUS.get(reason, "escalated")
        row.detail = _STOP_DETAIL.get(reason, reason)
        conf = re.search(r"confidence ([\d.]+)", rest)
        if conf:
            row.detail += f" · confidence {conf.group(1)}"
        row.end = t
        state.stopped_workers += 1
        if state.stopped_workers == state.expected_workers:
            state.synthesis.status, state.synthesis.start = "running", t
            state.synthesis.detail = "assembling sections"
        return

    it = re.match(r"iter (\d+) (\w+)", rest)
    if it:
        i, stage = int(it.group(1)) + 1, it.group(2)
        if stage == "critique":  # "critique done — confidence=…"
            conf = re.search(r"confidence=([\d.]+)", rest)
            row.detail = f"iter {i} · confidence {conf.group(1)}" if conf \
                else f"iter {i} · critiqued"
        else:
            row.detail = f"iter {i} · {stage}"
    elif rest.startswith("retrieving assets verbatim"):
        row.detail = "retrieving (verbatim)"


def render_status(state: RunState, now: float) -> str:
    def tr(row: Row) -> str:
        label_txt, fg, bg = _STATUS_STYLE[row.status]
        badge = (f"<span style='background:{bg};color:{fg};padding:2px 10px;"
                 f"border-radius:10px;font-size:0.82em;font-weight:600;"
                 f"white-space:nowrap'>{label_txt}</span>")
        return (f"<tr><td style='padding:6px 14px 6px 0;font-weight:600;"
                f"white-space:nowrap'>{html.escape(row.label)}</td>"
                f"<td style='padding:6px 14px 6px 0'>{badge}</td>"
                f"<td style='padding:6px 14px 6px 0;color:#4b5563'>"
                f"{html.escape(row.detail) or '&nbsp;'}</td>"
                f"<td style='padding:6px 0;text-align:right;font-variant-numeric:"
                f"tabular-nums;white-space:nowrap'>{row.duration(now)}</td></tr>")

    rows = [tr(state.planner)]
    rows += [tr(r) for r in state.workers.values()]
    rows.append(tr(state.synthesis))
    return ("<table style='width:100%;border-collapse:collapse;font-size:0.95em'>"
            "<thead><tr style='text-align:left;color:#6b7280;font-size:0.85em'>"
            "<th style='padding:0 14px 4px 0'>Stage</th><th>Status</th>"
            "<th>Progress</th><th style='text-align:right'>Duration</th></tr>"
            "</thead><tbody>" + "".join(rows) + "</tbody></table>")


# ---------------------------------------------------------------------------
# The run handler — async generator streaming UI updates
# ---------------------------------------------------------------------------


async def run_pipeline(brief_text: str, mode: str, pipeline_url: str,
                       tau: float, max_iterations: int):
    offline = mode.startswith("Offline")
    raw_brief = (brief_text or "").strip()
    if not raw_brief:
        if not offline:
            raise gr.Error("Paste a brief or attach a file (live mode needs one).")
        raw_brief = DEMO_BRIEF

    rag = None
    if offline:
        rag, drafter, llm, extractor = OfflineRAG(), OfflineDrafter(), None, None
    else:
        from llm_gemini import GeminiClaimExtractor, GeminiDrafter, GeminiLLMClient
        from rag_adapter import DocChatRAGClient
        rag = DocChatRAGClient(pipeline_url.strip(), timeout=300.0)
        drafter, llm = GeminiDrafter(), GeminiLLMClient()
        extractor = GeminiClaimExtractor()

    graph = build_graph(rag=rag, drafter=drafter, llm=llm, tau=tau,
                        max_iterations=int(max_iterations), extractor=extractor)

    state = RunState()
    state.planner.start = time.monotonic()
    state.planner.status = "running"
    log_lines: list[str] = []
    q: asyncio.Queue[tuple[float, str]] = asyncio.Queue()

    def listener(t: float, msg: str) -> None:
        q.put_nowait((t, msg))

    progress.add_listener(listener)
    task = asyncio.create_task(graph.ainvoke({"raw_brief": raw_brief}))
    try:
        while True:
            drained = False
            try:
                t, msg = await asyncio.wait_for(q.get(), timeout=0.5)
                drained = True
                log_lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
                apply_event(state, t, msg)
                while not q.empty():
                    t, msg = q.get_nowait()
                    log_lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
                    apply_event(state, t, msg)
            except asyncio.TimeoutError:
                pass
            yield (render_status(state, time.monotonic()),
                   "\n".join(log_lines), gr.skip(), gr.skip(), gr.skip())
            if task.done() and not drained and q.empty():
                break

        result = task.result()  # re-raises pipeline errors
    except Exception as e:
        for row in [state.planner, state.synthesis, *state.workers.values()]:
            if row.status == "running":
                row.status, row.end = "error", time.monotonic()
        yield (render_status(state, time.monotonic()),
               "\n".join(log_lines), gr.skip(), gr.skip(), gr.skip())
        raise gr.Error(f"Run failed: {e}") from e
    finally:
        progress.remove_listener(listener)
        task.cancel()
        if rag is not None and hasattr(rag, "aclose"):
            await rag.aclose()

    assembled = result["assembled"]
    out = pathlib.Path(".")
    payload = {"assembled": assembled,
               "plan": result["plan"].model_dump(),
               "brief": result["brief"].model_dump(),
               "outcomes": result.get("outcomes", {})}
    (out / "draft_proposal.json").write_text(
        json.dumps(payload, indent=2, default=str))
    md = render_markdown(assembled)
    (out / "draft_proposal.md").write_text(md)

    yield (render_status(state, time.monotonic()), "\n".join(log_lines), md,
           [str((out / "draft_proposal.md").resolve()),
            str((out / "draft_proposal.json").resolve())],
           {"payload": payload, "live": not offline})


def load_brief_file(path: str | None):
    if not path:
        return gr.skip()
    return pathlib.Path(path).read_text()


# ---------------------------------------------------------------------------
# PPTX export
# ---------------------------------------------------------------------------

# bundled CYLAD deck ships with the app; PB_PPTX_TEMPLATE overrides it
_TEMPLATE_CANDIDATE = pathlib.Path(os.environ.get(
    "PB_PPTX_TEMPLATE",
    str(pathlib.Path(__file__).parent / "assets" /
        "202006 - PF - DAIS - REX Harmonie - Proposition CYLAD V3.pptx"))).expanduser()
DEFAULT_TEMPLATE = str(_TEMPLATE_CANDIDATE) if _TEMPLATE_CANDIDATE.exists() else None

_DOWNLOADS = pathlib.Path.home() / "Downloads"
DEFAULT_LIBRARY_PATHS = str(_DOWNLOADS) if _DOWNLOADS.is_dir() else ""


async def export_deck(run_data, template_path: str | None, library_paths: str):
    if not run_data:
        raise gr.Error("Run the pipeline first — nothing to export.")
    from export_pptx import export_pptx
    payload = run_data["payload"]
    if not template_path:
        raise gr.Error("Upload a template deck (.pptx) first.")
    template = pathlib.Path(template_path).expanduser()
    if not template.exists():
        raise gr.Error(f"Template deck not found: {template}")

    # Gemini slide-ification for generated sections (live runs only);
    # anything that fails falls back to the deterministic markdown split.
    slides = None
    if run_data.get("live"):
        try:
            from llm_gemini import GeminiSlidifier
            slidifier = GeminiSlidifier()
            secs = [s for s in payload["assembled"]["sections"]
                    if s["status"] == "accepted"
                    and s.get("mode") not in ("retrieve_only", "template")]
            results = await asyncio.gather(
                *(slidifier.slidify(s["title"], s["content"]) for s in secs),
                return_exceptions=True)
            slides = {s["section"]: r for s, r in zip(secs, results)
                      if not isinstance(r, BaseException)}
            failed = [s["section"] for s, r in zip(secs, results)
                      if isinstance(r, BaseException)]
            if failed:
                gr.Warning(f"Slidify failed for {', '.join(failed)} — "
                           "using markdown fallback there.")
        except Exception as e:
            gr.Warning(f"Gemini slidify unavailable ({e}) — markdown fallback.")

    if not payload["assembled"].get("exportable"):
        gr.Warning("Review queue is not clear — exporting a WATERMARKED draft.")
    library = [pathlib.Path(p.strip()).expanduser()
               for p in (library_paths or "").split(",") if p.strip()]
    out = export_pptx(payload, template=template,
                      out=pathlib.Path("draft_proposal.pptx"),
                      slide_library=library or None,
                      slides_by_section=slides, force_draft=True)
    return str(out.resolve())


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="Proposal Builder") as demo:
    gr.Markdown("# Agentic Proposal Builder\n"
                "RFP brief → planner → parallel section workers "
                "(retrieve · draft · critique loop) → assembled draft "
                "with a human-review queue.")

    with gr.Row():
        with gr.Column(scale=3):
            brief_box = gr.Textbox(label="RFP brief", lines=8,
                                   placeholder="Paste the brief here, or attach "
                                               "a file on the right…")
        with gr.Column(scale=1):
            brief_file = gr.File(label="…or drop a brief file",
                                 file_types=[".txt", ".md"], type="filepath")
            mode = gr.Radio(["Live (pipeline + Gemini)", "Offline demo"],
                            value="Live (pipeline + Gemini)", label="Mode")

    with gr.Accordion("Settings", open=False):
        pipeline_url = gr.Textbox(label="Pipeline URL (live mode)",
                                  value=DEFAULT_PIPELINE_URL)
        with gr.Row():
            tau = gr.Slider(0.0, 1.0, value=DEFAULT_TAU, step=0.05,
                            label="tau (accept threshold)")
            max_iters = gr.Slider(1, 6, value=DEFAULT_MAX_ITERATIONS, step=1,
                                  label="Max iterations per section")

    run_btn = gr.Button("Build proposal", variant="primary")
    status_html = gr.HTML(label="Progress")
    with gr.Accordion("Run log", open=False):
        log_box = gr.Textbox(label="", lines=14, max_lines=14, interactive=False)
    files_out = gr.Files(label="Outputs", visible=True)

    run_data = gr.State(None)
    with gr.Accordion("PPTX export (CYLAD format)", open=False):
        template_file = gr.File(label="Template deck (.pptx) — defaults to the "
                                      "bundled CYLAD deck",
                                file_types=[".pptx"], type="filepath",
                                value=DEFAULT_TEMPLATE)
        library_box = gr.Textbox(
            label="Slide library folder(s) on the server, comma-separated — "
                  "source decks for verbatim team/credential slides",
            value=DEFAULT_LIBRARY_PATHS)
        export_btn = gr.Button("Export PPTX")
        pptx_out = gr.File(label="Deck")

    draft_md = gr.Markdown(label="Draft proposal")

    brief_file.change(load_brief_file, brief_file, brief_box)
    run_btn.click(run_pipeline,
                  inputs=[brief_box, mode, pipeline_url, tau, max_iters],
                  outputs=[status_html, log_box, draft_md, files_out, run_data])
    export_btn.click(export_deck,
                     inputs=[run_data, template_file, library_box],
                     outputs=[pptx_out])

if __name__ == "__main__":
    # PORT is set by Railway (and most PaaS hosts); absent locally.
    on_paas = "PORT" in os.environ
    auth = None
    if os.environ.get("PB_UI_USER") and os.environ.get("PB_UI_PASS"):
        auth = (os.environ["PB_UI_USER"], os.environ["PB_UI_PASS"])
    demo.launch(
        server_name="0.0.0.0" if on_paas else "127.0.0.1",
        server_port=int(os.environ.get("PORT", 7860)),
        auth=auth,
    )
