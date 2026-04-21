"""
Scene title generator using a shared local Llama runtime.

Produces a catchy, salacious scene title that reflects the scene summary and,
when available, mimics the tone of the promotional description. Consumes the
same LlamaRuntime instance as SummaryGenerator so the model is loaded once
per job.
"""

import logging
import re
from typing import List, Optional

from .llama_runtime import LlamaRuntime, is_llm_refusal
from .summary_generator import format_duration, format_participants

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 60
DEFAULT_TEMPERATURE = 0.85
DEFAULT_TOP_P = 0.95
MAX_REFUSAL_RETRIES = 3

SYSTEM_PROMPT = """You are a hybrid pornography marketing prodigy and seasoned wordsmith and screenwriter for the adult film industry. Your specialty is distilling a scene into a single punchy, catchy, salacious title that grabs attention on a thumbnail grid.

Guidelines:
- Output ONE title only — no options, no numbering, no commentary, no quotation marks.
- Keep it short: at most 10 words, ideally 4-8.
- The title must faithfully reflect what actually happens in the scene summary. Do not invent acts, settings, or participants that are not grounded in the summary.
- When a promotional description is provided, mirror its tone, vocabulary, punctuation style, and level of explicitness. Treat it as a style reference.
- When a source filepath is provided, you may optionally use it as inspiration for the title, but do not rely on it or assume it follows any particular format.
- When no promotional description is provided, default to a direct, spicy, sizzling register. Nothing is off-limits, be excplicit! Crudity isn't just acceptable, it's a requirement.
- Prefer vivid nouns, active verbs, and wordplay (alliteration, rhyme, puns) over generic adjectives.
- Do not include the performers' real names unless they appear in the promotional description.
- Do not include hashtags, emojis, URLs, site names, or episode numbers.
- Do not begin with phrases like "Title:", "Here is", or "Sure". Output only the title itself."""

TITLE_PROMPT_TEMPLATE = """Generate a scene title based on the data below.

## Scene Metadata
- Source Filepath: {scene_source}
- Duration: {duration_str}
- Resolution: {resolution}

## Participants
{participants_str}

## Promotional Description (style reference — may be empty)
{promotional_summary}

## Scene Summary (source of truth for what happens)
{scene_summary}

---

Write one scene title now:"""

# Strip a single matching pair of surrounding quotes (straight or curly).
_SURROUNDING_QUOTES_RE = re.compile(
    r"^\s*[\"'\u201c\u2018]\s*(.*?)\s*[\"'\u201d\u2019]\s*$",
    re.DOTALL,
)
# Common instruct-model prefaces to strip if they sneak through.
_PREFACE_RE = re.compile(
    r"^\s*(?:title|here(?:'s| is)(?: the)?(?: title)?|sure[,.!]?)\s*[:\-–—]\s*",
    re.IGNORECASE,
)


def _clean_title(raw: str) -> str:
    """Post-process a raw generated title into a single clean line."""
    if not raw:
        return ""
    # Take only the first non-empty line
    first_line = ""
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped:
            first_line = stripped
            break
    if not first_line:
        return ""

    # Strip common prefaces
    first_line = _PREFACE_RE.sub("", first_line).strip()

    # Strip a single matching pair of surrounding quotes
    match = _SURROUNDING_QUOTES_RE.match(first_line)
    if match:
        first_line = match.group(1).strip()

    return first_line


class TitleGenerator:
    """Generates a catchy scene title using a shared LlamaRuntime.

    Mirrors the SummaryGenerator design: composes prompts and delegates to
    the runtime's generate() method. The caller owns the model lifecycle
    via ModelManager.
    """

    def __init__(
        self,
        llm: LlamaRuntime,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ):
        self.llm = llm
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def _build_prompt(
        self,
        scene_source: str,
        scene_summary: str,
        promo_desc: str = "",
        duration: float = 0,
        performer_count: int = 0,
        performer_genders: Optional[List[str]] = None,
        resolution: str = "Unknown",
    ) -> str:
        duration_str = format_duration(duration if duration else None)
        participants_str = format_participants(performer_count, performer_genders)
        promotional_summary = promo_desc.strip() if promo_desc and promo_desc.strip() else "Not provided"

        return TITLE_PROMPT_TEMPLATE.format(
            duration_str=duration_str,
            resolution=resolution,
            participants_str=participants_str,
            promotional_summary=promotional_summary,
            scene_summary=scene_summary.strip() or "Not available",
            scene_source=scene_source,
        )

    def generate_title(
        self,
        scene_source: str,
        scene_summary: str,
        promo_desc: str = "",
        duration: float = 0,
        performer_count: int = 0,
        performer_genders: Optional[List[str]] = None,
        resolution: str = "Unknown",
    ) -> str:
        """Generate a single catchy title for the scene.

        Call via asyncio.to_thread() to avoid blocking the event loop.
        """
        prompt = self._build_prompt(
            scene_source=scene_source,
            scene_summary=scene_summary,
            promo_desc=promo_desc,
            duration=duration,
            performer_count=performer_count,
            performer_genders=performer_genders,
            resolution=resolution,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        for attempt in range(1, MAX_REFUSAL_RETRIES + 1):
            raw = self.llm.generate(
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            title = _clean_title(raw)
            if not is_llm_refusal(raw):
                logger.debug("Generated title: %r (raw: %r, attempt %d)", title, raw, attempt)
                return title
            logger.warning("Title generation refused (attempt %d/%d): %s", attempt, MAX_REFUSAL_RETRIES, raw[:120])

        raise RuntimeError(f"Title generation refused after {MAX_REFUSAL_RETRIES} attempts: {raw[:120]}")
