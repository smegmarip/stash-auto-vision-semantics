"""
Caption generator using JoyCaption beta-one with taxonomy-optimized prompt.

Generates detailed frame descriptions matching the classifier training pipeline.
Uses the same model (beta-one) and prompt (taxonomy) that produced training data,
ensuring consistency between caption style and downstream tag classification.

Standalone edition: bfloat16 only, no quantization. Model loaded once at startup.
"""

import gc
import logging
import re
import time
from typing import Dict, Any, List, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# JoyCaption beta-one (matches classifier training pipeline)
MODEL_NAME = "fancyfeast/llama-joycaption-beta-one-hf-llava"

# Taxonomy-optimized prompt (matches classifier training pipeline)
TAXONOMY_PROMPT = """Write a very long detailed description for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with "This image is…" or similar phrasing.

For people, describe: apparent age range, ethnicity, clothing and accessories, physical appearance, pose, and actions including any sexual positions and activity.
For the scene, describe: location or setting, objects present, and any interactions or activities.
Include information about camera angle, shot type (close-up, medium shot, wide shot), and lighting conditions."""

# Caption prefix fix patterns (from fix_caption_prefix.py)
_PREAMBLE_MEDIUM_RE = re.compile(
    r"^((?:A|The|This|In\s+this)\s+)(?:photograph|image|photo|picture)(\b)",
    re.IGNORECASE,
)
_MEDIUM_RE = re.compile(
    r"\b(photograph|image|photo(?!graphy)|picture)\b",
    re.IGNORECASE,
)


class CaptionGenerator:
    """Generates frame captions using JoyCaption beta-one with taxonomy prompt.

    Designed for the semantics service pipeline: load once, caption many frames,
    then unload to free GPU memory for other models.
    """

    def __init__(
        self,
        device: str = "cuda",
        max_new_tokens: int = 512,
        cache_dir: Optional[str] = None,
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.cache_dir = cache_dir
        self.model = None
        self.processor = None
        self._loaded = False
        self._vram_mb: Optional[float] = None

    def load(self) -> Dict[str, Any]:
        """Load JoyCaption model and processor in bfloat16.

        Call this before generating captions. Returns model info dict.

        Raises:
            RuntimeError: If model loading fails.
        """
        if self._loaded:
            return self.get_info()

        logger.info("Loading JoyCaption model: %s", MODEL_NAME)
        start = time.time()

        try:
            from transformers import (
                AutoProcessor,
                LlavaForConditionalGeneration,
            )

            # Load processor
            processor_kwargs: Dict[str, Any] = {"trust_remote_code": True}
            if self.cache_dir:
                processor_kwargs["cache_dir"] = self.cache_dir
            self.processor = AutoProcessor.from_pretrained(
                MODEL_NAME, **processor_kwargs
            )

            # Load model — bfloat16, no quantization
            model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
            if self.cache_dir:
                model_kwargs["cache_dir"] = self.cache_dir

            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            self.model = LlavaForConditionalGeneration.from_pretrained(
                MODEL_NAME,
                torch_dtype=dtype,
                device_map="auto",
                **model_kwargs,
            )

            self.model.eval()
            self._loaded = True

            if self.device == "cuda":
                torch.cuda.synchronize()
                self._vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)

            elapsed = time.time() - start
            logger.info(
                "Model loaded in %.1fs (VRAM: %.0fMB)",
                elapsed,
                self._vram_mb or 0,
            )
            return self.get_info()

        except Exception:
            logger.exception("Failed to load JoyCaption model")
            raise

    def unload(self) -> None:
        """Free GPU memory by unloading model and processor."""
        if not self._loaded:
            return

        logger.info("Unloading JoyCaption model")
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        self._loaded = False
        self._vram_mb = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("Model unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            "model": MODEL_NAME,
            "device": self.device,
            "quantization": "none",
            "loaded": self._loaded,
            "vram_mb": self._vram_mb,
            "max_new_tokens": self.max_new_tokens,
        }

    def generate_caption(self, image: Image.Image) -> str:
        """Generate a caption for a single frame image.

        Args:
            image: PIL Image (any mode; converted to RGB internally).

        Returns:
            Raw caption string (without frame prefix).

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Build conversation matching the training pipeline
        convo = [
            {"role": "system", "content": "You are a helpful image captioner."},
            {"role": "user", "content": TAXONOMY_PROMPT},
        ]

        convo_string = self.processor.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=True
        )
        if isinstance(convo_string, list):
            convo_string = convo_string[0] if convo_string else ""

        inputs = self.processor(
            text=[convo_string], images=[image], return_tensors="pt"
        ).to(self.device)

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,
            )

        # Decode only the generated tokens (skip input)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_len:]

        if len(generated_ids) == 0:
            logger.warning("No tokens generated for image")
            return ""

        caption = self.processor.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()

        logger.debug("Caption (%d tokens): %.100s...", len(generated_ids), caption)
        return caption

    def generate_captions(self, images: List[Image.Image]) -> List[str]:
        """Generate captions for multiple frames sequentially.

        JoyCaption works best processing one image at a time. For batched
        throughput, consider running multiple service instances.

        Args:
            images: List of PIL Images.

        Returns:
            List of caption strings (empty string on per-image failure).
        """
        captions: List[str] = []
        total = len(images)

        for i, image in enumerate(images):
            try:
                caption = self.generate_caption(image)
                captions.append(caption)
            except Exception:
                logger.exception("Error captioning image %d/%d", i + 1, total)
                captions.append("")

            # Periodic CUDA cache cleanup for long sequences
            if (i + 1) % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        return captions

    @staticmethod
    def fix_caption(caption: str, frame_index: int) -> str:
        """Apply caption prefix fixes matching classifier training format.

        Transforms raw VLM captions to match the format used during classifier
        training:
          1. Replaces "photograph/image/photo/picture" with "video frame"
          2. Prepends "Frame {index}: " prefix

        Args:
            caption: Raw caption text from generate_caption().
            frame_index: Zero-based frame index for temporal identity.

        Returns:
            Fixed caption, e.g. "Frame 3: A video frame of a classroom..."
        """
        if not caption:
            return f"Frame {frame_index}: "

        # Try structured preamble first (most common case):
        #   "A photograph of..." -> "A video frame of..."
        #   "This image shows..." -> "This video frame shows..."
        result, n = _PREAMBLE_MEDIUM_RE.subn(r"\1video frame\2", caption, count=1)
        if n > 0:
            return f"Frame {frame_index}: {result}"

        # Fallback: replace first bare medium word anywhere in caption
        result, n = _MEDIUM_RE.subn("video frame", caption, count=1)
        if n > 0:
            return f"Frame {frame_index}: {result}"

        # No medium word found; just prepend frame index
        return f"Frame {frame_index}: {caption}"
