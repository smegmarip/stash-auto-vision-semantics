"""
Scene summary generator using a shared local Llama runtime.

Synthesizes frame-level captions and scene metadata into coherent narrative
summaries. Uses the same prompt structure as the training pipeline's
generate_scene_summaries.py to ensure consistency.

Does not own the model — takes a LlamaRuntime dependency so the same
loaded weights can be shared with other consumers (e.g. TitleGenerator).
"""

import logging
import re
from typing import Any, Callable, Dict, List, Optional

from .llama_runtime import LlamaRuntime, is_llm_refusal

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.3
MAX_REFUSAL_RETRIES = 3
# A real 2-4 paragraph summary should comfortably exceed this.
MIN_SUMMARY_LENGTH = 200

SYSTEM_PROMPT = """You are a scene summarizer. Given structured data about a video scene (metadata, participants, and frame-by-frame descriptions), generate a coherent narrative summary.

Guidelines:
- Synthesize frame descriptions into a flowing narrative, not a list
- Describe temporal progression: what happens first, then, throughout, finally
- Focus on actions, interactions, settings, and visual elements
- Ignore mentions of: watermarks, compression artifacts, image quality, "photograph" framing
- Use present tense for describing the scene
- Be specific about: clothing, positions, settings, camera angles, lighting
- Do not speculate about mood, emotions, or intent unless clearly evident
- Length: 2-4 paragraphs covering the entire scene"""

SUMMARY_PROMPT_TEMPLATE = """Summarize this video scene based on the structured data below.

## Scene Metadata
- Duration: {duration_str}
- Frame count: {frame_count} frames sampled
- Resolution: {resolution}

## Participants
{participants_str}

## Promotional Description
{promotional_summary}

## Frame-by-Frame Descriptions
{frame_descriptions}

---

Generate a coherent narrative summary of this scene:"""

# Patterns for cleaning frame captions before summarization
_CAPTION_MEDIUM_RE = re.compile(
    r"^(A|This|The) (photograph|image|picture|photo)\b",
    re.IGNORECASE,
)
_WATERMARK_RE = re.compile(r"[^.]*watermark[^.]*\.", re.IGNORECASE)
_ARTIFACT_RE = re.compile(r"[^.]*compression artifacts?[^.]*\.", re.IGNORECASE)
_RESOLUTION_RE = re.compile(r"[^.]*resolution[^.]*\.", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def clean_frame_caption(caption: str) -> str:
    """Clean a frame caption for scene summary context."""
    caption = _CAPTION_MEDIUM_RE.sub("The frame", caption)
    caption = _WATERMARK_RE.sub("", caption)
    caption = _ARTIFACT_RE.sub("", caption)
    caption = _RESOLUTION_RE.sub("", caption)
    caption = _WHITESPACE_RE.sub(" ", caption).strip()
    return caption


def format_duration(seconds: Optional[float]) -> str:
    """Format duration in seconds as a human-readable string."""
    if not seconds:
        return "Unknown"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def format_participants(count: int, genders: Optional[List[str]]) -> str:
    """Format participant info as a descriptive string."""
    if count == 0:
        return "No identified participants"
    gender_str = ", ".join(genders) if genders else "genders unknown"
    return f"{count} participant(s): {gender_str}"


class SummaryGenerator:
    """Generates narrative scene summaries using a shared LlamaRuntime.

    The model lifecycle is owned by the runtime — callers must ensure the
    runtime is loaded (via ModelManager) before invoking generate_summary.
    """

    def __init__(
        self,
        llm: LlamaRuntime,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        self.llm = llm
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _build_prompt(
        self,
        frame_captions: List[Dict[str, Any]],
        promo_desc: str = "",
        duration: float = 0,
        performer_count: int = 0,
        performer_genders: Optional[List[str]] = None,
        resolution: str = "Unknown",
    ) -> str:
        """Build the summary prompt from frame captions and metadata."""
        duration_str = format_duration(duration if duration else None)
        participants_str = format_participants(performer_count, performer_genders)
        promotional_summary = promo_desc or "Not available"

        frame_lines: List[str] = []
        for frame in frame_captions:
            timestamp = frame.get("timestamp", 0)
            caption = clean_frame_caption(frame.get("caption", ""))
            mins = int(timestamp // 60)
            secs = int(timestamp % 60)
            frame_lines.append(f"[{mins:02d}:{secs:02d}] {caption}")

        return SUMMARY_PROMPT_TEMPLATE.format(
            duration_str=duration_str,
            frame_count=len(frame_captions),
            resolution=resolution,
            participants_str=participants_str,
            promotional_summary=promotional_summary,
            frame_descriptions="\n".join(frame_lines),
        )

    def generate_summary(
        self,
        frame_captions: List[Dict[str, Any]],
        promo_desc: str = "",
        duration: float = 0,
        performer_count: int = 0,
        performer_genders: Optional[List[str]] = None,
        resolution: str = "Unknown",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        """Generate a narrative summary from frame captions and metadata.

        Call via asyncio.to_thread() to avoid blocking the event loop.

        Args:
            progress_callback: Optional callable(tokens_generated, max_tokens)
                invoked as tokens stream out.

        Returns:
            Narrative summary text (2-4 paragraphs).
        """
        prompt = self._build_prompt(
            frame_captions=frame_captions,
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
            summary = self.llm.generate(
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.9,
                progress_callback=progress_callback,
            )
            if not is_llm_refusal(summary, min_length=MIN_SUMMARY_LENGTH):
                logger.debug("Generated summary (%d chars, attempt %d)", len(summary), attempt)
                return summary
            logger.warning("Summary generation refused (attempt %d/%d): %s", attempt, MAX_REFUSAL_RETRIES, summary[:120])

        raise RuntimeError(f"Summary generation refused after {MAX_REFUSAL_RETRIES} attempts: {summary[:120]}")
