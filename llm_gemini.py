"""
Gemini implementations of the proposal builder's LLM seams, matching the
pipeline's stack (google-genai SDK, Gemini Developer API).

Model routing (the cost story): cheap/structured steps — brief extraction,
section proposal, claim extraction — run on the flash model; drafting
(synthesis quality matters) can run on a larger model. Both default to
gemini-2.5-flash to match the pipeline; override via env.

    GOOGLE_GENAI_USE_VERTEXAI  'true' -> Vertex AI via ADC (no API key needed)
    GOOGLE_CLOUD_PROJECT       Vertex mode: GCP project (falls back to gcloud config)
    GOOGLE_CLOUD_LOCATION      Vertex mode: region, default 'global'
    GEMINI_API_KEY             required only in Developer-API mode
    PB_MODEL_CHEAP             default gemini-2.5-flash   (extraction, critique aids)
    PB_MODEL_DRAFT             default gemini-2.5-flash   (section drafting)

All structured calls use response_schema=<Pydantic model> so outputs parse
into the exact types the graph runs on. Thinking is disabled for the same
reason as the pipeline: these are grounded-extraction tasks, and the dynamic
thinking budget can eat max_output_tokens (see pipeline.py's note).
"""

from __future__ import annotations

import os

from google import genai
from google.genai import types as genai_types

from critique import ClaimExtractor
from planner import (
    LLMClient,
    ProposedSection,
    RetrievedChunk,
    SectionPlanItem,
    SectionProposal,
    StructuredBrief,
)
from pydantic import BaseModel
from worker import Drafter

_CHEAP = os.environ.get("PB_MODEL_CHEAP", "gemini-2.5-flash")
_DRAFT = os.environ.get("PB_MODEL_DRAFT", "gemini-2.5-flash")


def _client() -> genai.Client:
    if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in {"1", "true", "yes"}:
        return genai.Client(
            vertexai=True,
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
        )
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "Set GOOGLE_GENAI_USE_VERTEXAI=true (Vertex/ADC) or GEMINI_API_KEY. "
            "Run with --offline for the stubbed demo.")
    return genai.Client(api_key=key)


def _structured_config(schema) -> genai_types.GenerateContentConfig:
    return genai_types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=2048,
        response_mime_type="application/json",
        response_schema=schema,
        thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
    )


# ---------------------------------------------------------------------------
# LLMClient: brief extraction + section proposal (cheap model)
# ---------------------------------------------------------------------------

_EXTRACT_PROMPT = """Extract structured fields from this RFP brief for a consulting
proposal builder. The corpus facets are: IndustrySector (e.g. 'Aerospace & Defense'),
ServiceLine (e.g. 'Supply Chain & Procurement'), Function (e.g. 'Purchasing').
Map to those vocabularies where possible. Brief may be French, English or German;
set `language` accordingly and keep extracted values in the brief's language.

RFP brief:
{brief}"""

_PROPOSE_PROMPT = """Given this structured RFP brief, decide which canonical proposal
sections this RFP warrants. Include a section only if the brief gives it substance;
give a one-line reason each. Mandatory sections are enforced downstream — focus on
judgment calls (does this RFP warrant a separate work_plan? credentials?).

Brief:
{brief}"""


class GeminiLLMClient(LLMClient):
    def __init__(self) -> None:
        self._c = _client()

    async def extract_brief(self, raw_brief: str) -> StructuredBrief:
        resp = await self._c.aio.models.generate_content(
            model=_CHEAP,
            contents=_EXTRACT_PROMPT.format(brief=raw_brief),
            config=_structured_config(StructuredBrief),
        )
        return StructuredBrief.model_validate_json(resp.text)

    async def propose_sections(self, brief: StructuredBrief) -> SectionProposal:
        resp = await self._c.aio.models.generate_content(
            model=_CHEAP,
            contents=_PROPOSE_PROMPT.format(brief=brief.model_dump_json(indent=2)),
            config=_structured_config(SectionProposal),
        )
        return SectionProposal.model_validate_json(resp.text)


# ---------------------------------------------------------------------------
# Drafter: adapt retrieved material to the new prospect (draft model)
# ---------------------------------------------------------------------------

_DRAFT_SYSTEM = """You draft one section of a consulting proposal by adapting the
firm's retrieved past material to a NEW prospect. Hard rules:
- Address the new prospect only. NEVER name other clients, their people, or their
  programs — retrieved material may mention them; genericize ("a major aerospace
  OEM"), never copy names. (Exception: none — even in credential-flavored text,
  leave naming to the credentials section, which is assembled separately.)
- Do not invent experience or figures. Only state outcomes present in the
  retrieved material, and prefer stating them generically unless this section is
  'commercial'.
- Match the brief's language (French/English/German).
- Write tight consulting prose for a slide: short paragraphs, no filler."""

_DRAFT_USER = """Section to draft: {title} ({section})
Prospect brief:
{brief}

Retrieved material (adapt, don't copy):
{material}

{feedback_block}
Draft the section now (200-350 words)."""


class GeminiDrafter(Drafter):
    def __init__(self) -> None:
        self._c = _client()

    async def draft(
        self,
        *,
        item: SectionPlanItem,
        brief: StructuredBrief,
        chunks: list[RetrievedChunk],
        feedback: str,
    ) -> str:
        material = "\n---\n".join(c.text for c in chunks[:6])
        fb = f"Revision instructions from critique:\n{feedback}\n" if feedback else ""
        resp = await self._c.aio.models.generate_content(
            model=_DRAFT,
            contents=_DRAFT_USER.format(
                title=item.title, section=item.section.value,
                brief=brief.model_dump_json(indent=2),
                material=material, feedback_block=fb),
            config=genai_types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=1024,
                system_instruction=_DRAFT_SYSTEM,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return resp.text or ""


# ---------------------------------------------------------------------------
# ClaimExtractor: provable claims from a draft (cheap model)
# ---------------------------------------------------------------------------


class _Claims(BaseModel):
    claims: list[str]


_CLAIMS_PROMPT = """List the factual, PROVABLE claims in this proposal-section draft:
statements of past experience, delivered outcomes, quantified results, or capability
assertions that a credential slide could substantiate. Exclude forward-looking
commitments, methodology descriptions, and boilerplate. Return each claim verbatim.

Draft:
{draft}"""


class GeminiClaimExtractor(ClaimExtractor):
    def __init__(self) -> None:
        self._c = _client()

    async def extract(self, draft: str) -> list[str]:
        resp = await self._c.aio.models.generate_content(
            model=_CHEAP,
            contents=_CLAIMS_PROMPT.format(draft=draft),
            config=_structured_config(_Claims),
        )
        return _Claims.model_validate_json(resp.text).claims


# ---------------------------------------------------------------------------
# Slidifier: prose section -> slide-shaped specs for the PPTX export
# ---------------------------------------------------------------------------


_SLIDIFY_PROMPT = """Convert this proposal-section draft into consulting slides.
Rules:
- action_title: a full-sentence assertion (the slide's message), not a label.
- kicker: one short line framing the slide; may be empty.
- bullets: crisp fragments (max 12 words each), max {max_bullets} per slide,
  level 0 for main points, 1 for supporting detail. No markdown syntax.
- comment: one optional footnote line; usually empty.
- Prefer ONE slide; split only if the draft clearly covers 2+ distinct messages.
- Keep the draft's language. Do not add content that is not in the draft.

Section: {title}
Draft:
{draft}"""


class GeminiSlidifier:
    """Slide-ifies accepted sections at export time (draft model — layout
    judgment matters). Falls back on export_pptx.fallback_slidify upstream."""

    def __init__(self) -> None:
        self._c = _client()

    async def slidify(self, title: str, draft: str) -> "list":
        from export_pptx import SlideSpec, _MAX_BULLETS

        class _Slides(BaseModel):
            slides: list[SlideSpec]

        resp = await self._c.aio.models.generate_content(
            model=_DRAFT,
            contents=_SLIDIFY_PROMPT.format(
                title=title, draft=draft, max_bullets=_MAX_BULLETS),
            config=_structured_config(_Slides),
        )
        return _Slides.model_validate_json(resp.text).slides
