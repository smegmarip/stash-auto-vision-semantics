"""
Tag classifier service wrapper.

Wraps the trained MultiViewClassifier for use as a service component.
Downloads model weights from HuggingFace on first use and caches locally.

Model source: https://huggingface.co/smegmarip/tag-classifier
"""
from __future__ import annotations

import gc
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# The train package is bundled at semantics-service/train/
# Add the service root so ``from train.model import ...`` resolves.
_SERVICE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SERVICE_ROOT))

from train.config import TrainConfig
from train.model import MultiViewClassifier
from train.tag_families import build_families

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HuggingFace download constants
# ---------------------------------------------------------------------------
HF_REPO = os.getenv("SEMANTICS_HF_REPO", "smegmarip/tag-classifier")
MODEL_VARIANTS: Dict[str, Dict[str, str]] = {
    "vision": {
        "model": os.getenv("SEMANTICS_HF_VISION_MODEL", "vision/best_model.pt"),
        "mapping": os.getenv("SEMANTICS_HF_VISION_TAG_MAPPING", "vision/tag_mapping.json"),
    },
    "text-only": {
        "model": os.getenv("SEMANTICS_HF_TEXT_MODEL", "text-only/best_model.pt"),
        "mapping": os.getenv("SEMANTICS_HF_TEXT_TAG_MAPPING", "text-only/tag_mapping.json"),
    },
}
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "models")

# ---------------------------------------------------------------------------
# Hierarchical decoder thresholds
# ---------------------------------------------------------------------------
T_PARENT = 0.60
T_LEAF = 0.78
CHILD_MARGIN = 0.08
MAX_CHILDREN = 2


# ---------------------------------------------------------------------------
# DecodedTag data class
# ---------------------------------------------------------------------------
@dataclass
class DecodedTag:
    """A tag emitted by the hierarchical decoder."""

    tag_id: str
    tag_name: str
    score: float
    is_leaf: bool
    path: List[str]
    decode_type: str  # "direct", "competition", or "parent_only"


# ---------------------------------------------------------------------------
# HierarchicalDecoder (inlined / simplified)
# ---------------------------------------------------------------------------
class HierarchicalDecoder:
    """Apply taxonomy-aware post-processing to raw sigmoid scores.

    Constraints enforced:
      * Parent-child: a child is only considered if its parent passes its
        own threshold.
      * Child competition: at most ``max_children`` siblings are accepted
        per parent, chosen by highest score within ``child_margin`` of the
        top sibling.
      * Parent activation: when a leaf is accepted its ancestor chain is
        activated automatically (tagged ``parent_only``).
    """

    def __init__(
        self,
        taxonomy: Dict[str, Any],
        t_parent: float = T_PARENT,
        t_leaf: float = T_LEAF,
        child_margin: float = CHILD_MARGIN,
        max_children: int = MAX_CHILDREN,
    ) -> None:
        self.t_parent = t_parent
        self.t_leaf = t_leaf
        self.child_margin = child_margin
        self.max_children = max_children

        self.tags: List[Dict[str, Any]] = taxonomy.get("tags", [])
        self.by_id: Dict[str, Dict[str, Any]] = taxonomy.get("by_id", {})

        # Build hierarchy maps
        self.children: Dict[str, List[str]] = defaultdict(list)
        self.parent: Dict[str, str] = {}
        self.depth: Dict[str, int] = {}
        self.is_leaf: Dict[str, bool] = {}

        for tag in self.tags:
            tag_id = tag["id"]
            parent_id = tag.get("parent_id")
            self.depth[tag_id] = tag.get("depth", 1)
            self.is_leaf[tag_id] = tag.get("is_leaf", True)
            if parent_id:
                self.parent[tag_id] = parent_id
                self.children[parent_id].append(tag_id)

        self.tags_by_depth = sorted(self.tags, key=lambda t: t.get("depth", 1))

    # ------------------------------------------------------------------ #
    # Core decode
    # ------------------------------------------------------------------ #

    def decode(
        self,
        scores: Dict[str, float],
        return_all: bool = False,
    ) -> List[DecodedTag]:
        """Decode raw per-tag scores into accepted tag predictions.

        Args:
            scores: Mapping of ``tag_id -> sigmoid score``.
            return_all: If *True*, include ancestor-only activations in the
                output.  Defaults to *False* (leaf / direct hits only).

        Returns:
            List of :class:`DecodedTag` sorted by score descending.
        """
        accepted: set[str] = set()
        decode_types: Dict[str, str] = {}

        # Step 1 -- top-down threshold evaluation
        for tag in self.tags_by_depth:
            tag_id = tag["id"]
            score = scores.get(tag_id, 0.0)

            parent_id = self.parent.get(tag_id)
            if parent_id and parent_id not in accepted:
                if scores.get(parent_id, 0.0) < self.t_parent:
                    continue

            threshold = self.t_leaf if self.is_leaf.get(tag_id, True) else self.t_parent
            if score >= threshold:
                accepted.add(tag_id)
                decode_types[tag_id] = "direct"

        # Step 2 -- child competition
        for parent_id, child_ids in self.children.items():
            if not child_ids:
                continue
            child_scores = sorted(
                [(cid, scores.get(cid, 0.0)) for cid in child_ids],
                key=lambda x: x[1],
                reverse=True,
            )
            if not child_scores or child_scores[0][1] < self.t_leaf:
                continue
            top_score = child_scores[0][1]
            selected = 0
            for cid, cscore in child_scores:
                if selected >= self.max_children:
                    break
                if top_score - cscore <= self.child_margin:
                    selected += 1
                    if cid not in accepted:
                        accepted.add(cid)
                        decode_types[cid] = "competition"

        # Step 3 -- ensure ancestor activation
        for tag_id in list(accepted):
            pid = self.parent.get(tag_id)
            while pid:
                if pid not in accepted:
                    accepted.add(pid)
                    decode_types[pid] = "parent_only"
                pid = self.parent.get(pid)

        # Build results
        results: List[DecodedTag] = []
        for tag_id in accepted:
            tag_info = self.by_id.get(tag_id, {})
            if not tag_info:
                continue
            if not return_all and decode_types.get(tag_id) == "parent_only":
                continue
            results.append(
                DecodedTag(
                    tag_id=tag_id,
                    tag_name=tag_info.get("name", ""),
                    score=scores.get(tag_id, 0.0),
                    is_leaf=self.is_leaf.get(tag_id, True),
                    path=tag_info.get("path", []),
                    decode_type=decode_types.get(tag_id, "direct"),
                )
            )

        results.sort(key=lambda t: t.score, reverse=True)
        return results


# ---------------------------------------------------------------------------
# Checkpoint resolution (mirrors train/inference.py)
# ---------------------------------------------------------------------------

def resolve_checkpoint(model_name: str, cache_dir: str = MODEL_CACHE_DIR) -> str:
    """Resolve a model variant name to a local checkpoint path.

    If the checkpoint is not cached locally it is downloaded from HuggingFace.
    A plain file path is returned as-is if it already exists on disk.
    """
    path = Path(model_name)
    if path.exists():
        return str(path)

    if model_name not in MODEL_VARIANTS:
        raise FileNotFoundError(
            f"Model '{model_name}' not found.  Use a file path or one of: "
            f"{', '.join(MODEL_VARIANTS)}"
        )

    variant = MODEL_VARIANTS[model_name]
    local_path = Path(cache_dir) / variant["model"]
    if local_path.exists():
        return str(local_path)

    # Download from HuggingFace Hub
    from huggingface_hub import hf_hub_download

    logger.info("Downloading %s model from %s ...", model_name, HF_REPO)
    hf_hub_download(
        repo_id=HF_REPO,
        filename=variant["model"],
        local_dir=cache_dir,
        local_dir_use_symlinks=False,
    )
    try:
        hf_hub_download(
            repo_id=HF_REPO,
            filename=variant["mapping"],
            local_dir=cache_dir,
            local_dir_use_symlinks=False,
        )
    except Exception:
        logger.warning("tag_mapping.json not found in HF repo; skipping")

    return str(local_path)


def _load_model_from_checkpoint(
    checkpoint_path: str,
    config: TrainConfig,
    num_tags: int,
    num_families: int,
    device: torch.device,
) -> MultiViewClassifier:
    """Instantiate a :class:`MultiViewClassifier` and load weights."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "config" in ckpt:
        saved = ckpt["config"]
        for key in (
            "backbone_name",
            "model_dim",
            "num_heads",
            "temporal_layers",
            "path_gru_layers",
            "num_frames",
            "max_seq_length",
            "focal_gamma",
            "vision_model",
            "gate_init_bias",
        ):
            if key in saved:
                setattr(config, key, saved[key])

    # Family bias is inert (zero-initialized, never trained — see RESEARCH.md
    # §27.6) but its nn.Embedding(154, 512) must be present for state_dict
    # compatibility. Hardcode to match the published checkpoint.
    # Strip _tag_cache buffers — they are rebuilt by build_tag_cache() and
    # their size depends on the runtime taxonomy, not the training set.
    state_dict = ckpt["model_state_dict"]
    state_dict.pop("_tag_cache", None)
    state_dict.pop("_tag_cache_valid", None)
    model = MultiViewClassifier(config, num_tags, 154)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    stage = ckpt.get("stage", "?")
    epoch = ckpt.get("epoch", "?")
    recall = ckpt.get("val_metrics", {}).get("recall@8", "?")
    logger.info("Loaded checkpoint: %s epoch %s (R@8=%s)", stage, epoch, recall)

    return model


# ---------------------------------------------------------------------------
# TagClassifier -- public service wrapper
# ---------------------------------------------------------------------------

class TagClassifier:
    """Wraps the trained MultiViewClassifier for service inference.

    Typical lifecycle::

        clf = TagClassifier(model_variant="text-only", device="cuda")
        clf.load_model()
        clf.load_taxonomy(taxonomy_dict)

        result = clf.predict(
            frame_captions=captions,
            summary=summary,
            promo_desc=promo,
            has_promo=True,
        )

    The ``taxonomy_dict`` must have the same shape as the exported
    ``taxonomy.json`` with keys ``tags``, ``by_id``, ``by_name``, and
    ``metadata``.
    """

    def __init__(
        self,
        model_variant: str = "text-only",
        device: str = "cuda",
    ) -> None:
        self.model: Optional[MultiViewClassifier] = None
        self.model_variant = model_variant
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.taxonomy: Optional[Dict[str, Any]] = None
        self.tag_id_to_idx: Optional[Dict[str, int]] = None
        self.tag_idx_to_id: Optional[Dict[int, str]] = None
        self.families: Optional[List[Dict[str, Any]]] = None
        self.tag_to_families: Optional[Dict[str, List[int]]] = None
        self.decoder: Optional[HierarchicalDecoder] = None

        self._config: Optional[TrainConfig] = None
        self._image_processor = None
        self._num_tags: int = 0

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #

    def load_model(self) -> None:
        """Download (if needed) and prepare the classifier checkpoint.

        The model is not fully instantiated until :meth:`load_taxonomy`
        provides the tag set (which determines num_tags for the tag cache).
        """
        checkpoint_path = resolve_checkpoint(self.model_variant)

        # Peek at the checkpoint to discover vision mode
        ckpt_meta = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        uses_vision = ckpt_meta.get("config", {}).get("use_vision", False) or any(k.startswith("vision_encoder.") for k in ckpt_meta.get("model_state_dict", {}))

        config = TrainConfig()
        config.gradient_checkpointing = False
        config.use_vision = uses_vision

        self._config = config
        self._checkpoint_path = checkpoint_path

        # Initialize vision image processor if this is a vision checkpoint
        if uses_vision:
            from transformers import AutoImageProcessor
            self._image_processor = AutoImageProcessor.from_pretrained(config.vision_model)
            logger.info("Vision image processor loaded: %s", config.vision_model)

        logger.info("Checkpoint ready: variant=%s  vision=%s  device=%s", self.model_variant, uses_vision, self.device)

    # ------------------------------------------------------------------ #
    # Taxonomy loading
    # ------------------------------------------------------------------ #

    def load_taxonomy(self, taxonomy: Dict[str, Any]) -> None:
        """Load the tag taxonomy and instantiate the model.

        The model can score any tag — the bi-encoder computes tag embeddings
        from text at runtime.  We build a tag_id_to_idx mapping covering
        the full taxonomy so every tag gets a score.  The checkpoint's
        family_bias size is read from the saved weights directly, and
        ``strict=False`` is used so the _tag_cache buffer (rebuilt here)
        can differ in size from the training set.

        Must be called *after* :meth:`load_model`.
        """
        if self._config is None:
            raise RuntimeError("Call load_model() before load_taxonomy()")

        self.taxonomy = taxonomy
        tags = taxonomy.get("tags", [])

        # Build tag <-> index mappings from the full taxonomy
        all_tag_ids = sorted(str(t["id"]) for t in tags)
        self.tag_id_to_idx = {tid: i for i, tid in enumerate(all_tag_ids)}
        self.tag_idx_to_id = {i: tid for tid, i in self.tag_id_to_idx.items()}
        self._num_tags = len(all_tag_ids)

        # Family bias is inert (never trained) but nn.Embedding(154, 512)
        # must match the checkpoint. Hardcode 154, pass empty families so
        # build_tag_cache skips the family bias lookup entirely.
        self.families, self.tag_to_families = [], {}

        self.model = _load_model_from_checkpoint(
            self._checkpoint_path,
            self._config,
            self._num_tags,
            154,
            self.device,
        )

        # Rebuild tag cache with the full taxonomy
        self.model.build_tag_cache(
            tags, self.tag_id_to_idx, self.tag_to_families, self.device
        )

        # Initialise hierarchical decoder
        self.decoder = HierarchicalDecoder(taxonomy)

        logger.info(
            "Taxonomy loaded: %d tags, decoder ready",
            self._num_tags,
        )

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def predict(
        self,
        frame_captions: List[str],
        summary: str,
        promo_desc: str = "",
        has_promo: bool = False,
        frame_images: Optional[List["Image.Image"]] = None,
        frame_timestamps: Optional[List[float]] = None,
        top_k: int = 30,
        min_score: float = 0.75,
        use_hierarchical_decoding: bool = True,
    ) -> Dict[str, Any]:
        """Run inference on a single scene.

        Args:
            frame_captions: List of 16 frame caption strings.
            summary: Narrative scene summary.
            promo_desc: Promotional / editorial description (optional).
            has_promo: Whether *promo_desc* is meaningful.
            frame_images: Optional list of PIL Images for vision model.
            frame_timestamps: Optional list of frame timestamps in seconds.
            top_k: Maximum tags to return.
            min_score: Minimum sigmoid confidence threshold.
            use_hierarchical_decoding: Apply hierarchy post-processing.

        Returns:
            Dict with keys:

            * ``tags`` -- ``List[dict]`` with ``tag_id``, ``tag_name``,
              ``score``, ``path``, ``decode_type``.
            * ``scene_embedding`` -- ``Optional[List[float]]`` (512-D).
              Always ``None`` from this method; use
              :meth:`get_scene_embedding` explicitly.
        """
        self._check_ready()

        batch = self._build_batch(
            frame_captions, summary, promo_desc, has_promo,
            frame_images=frame_images, frame_timestamps=frame_timestamps,
        )

        # Forward pass (returns sigmoid scores of shape [1, num_tags])
        scores_tensor = self.model.predict(batch, self.device).squeeze(0).cpu()

        if use_hierarchical_decoding and self.decoder is not None:
            return self._decode_hierarchical(scores_tensor, top_k, min_score)

        return self._decode_flat(scores_tensor, top_k, min_score)

    def get_scene_embedding(
        self,
        frame_captions: List[str],
        summary: str,
        promo_desc: str = "",
        has_promo: bool = False,
    ) -> List[float]:
        """Get the 512-D scene embedding vector.

        This runs :meth:`encode_scene` and returns the fused scene
        representation as a plain Python list of floats.
        """
        self._check_ready()

        batch = self._build_batch(frame_captions, summary, promo_desc, has_promo)

        with torch.no_grad():
            scene_repr, _, _, _ = self.model.encode_scene(batch, self.device)

        return scene_repr.squeeze(0).cpu().tolist()

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def is_loaded(self) -> bool:
        """True when the model and taxonomy are both ready for inference."""
        return self.model is not None

    @property
    def is_checkpoint_ready(self) -> bool:
        """True when load_model() has been called (checkpoint downloaded, config peeked)."""
        return self._config is not None

    @property
    def num_tags(self) -> int:
        return self._num_tags

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #

    def unload(self) -> None:
        """Free GPU memory and reset state."""
        if self.model is not None:
            del self.model
            self.model = None
        self.taxonomy = None
        self.tag_id_to_idx = None
        self.tag_idx_to_id = None
        self.families = None
        self.tag_to_families = None
        self.decoder = None
        self._num_tags = 0

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("TagClassifier unloaded")

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _check_ready(self) -> None:
        if self.model is None:
            raise RuntimeError(
                "Model not loaded.  Call load_model() then load_taxonomy() first."
            )
        if self.tag_id_to_idx is None:
            raise RuntimeError(
                "Taxonomy not loaded.  Call load_taxonomy() first."
            )

    def _build_batch(
        self,
        frame_captions: List[str],
        summary: str,
        promo_desc: str,
        has_promo: bool,
        frame_images: Optional[List] = None,
        frame_timestamps: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Build a single-scene batch dict matching ``collate_scenes`` format.

        When ``frame_images`` are provided and the model is vision-enabled,
        images are upscaled 3x (matching training) and processed through
        the SiGLIP image processor.
        """
        num_frames = self._config.num_frames if self._config else 16

        # Pad or truncate captions to expected frame count
        captions = list(frame_captions[:num_frames])
        while len(captions) < num_frames:
            captions.append("")

        # Use real timestamps if available, otherwise synthetic
        if frame_timestamps:
            timestamps = list(frame_timestamps[:num_frames])
            while len(timestamps) < num_frames:
                timestamps.append(timestamps[-1] if timestamps else 0.0)
        else:
            timestamps = [float(i) for i in range(num_frames)]

        batch: Dict[str, Any] = {
            "frame_captions": [captions],
            "frame_timestamps": [timestamps],
            "summaries": [summary],
            "promo_descs": [promo_desc if promo_desc else ""],
            "has_promo": torch.tensor([has_promo], dtype=torch.bool),
            "labels": torch.zeros(1, self._num_tags, dtype=torch.float32),
            "positive_mask": torch.zeros(1, self._num_tags, dtype=torch.bool),
        }

        # Vision path: process frame images through SiGLIP
        if (
            frame_images
            and self._image_processor is not None
            and self._config is not None
            and self._config.use_vision
        ):
            from PIL import Image as PILImage

            upscale = 3  # Matches training pipeline (inference.py)
            upscaled = []
            for img in frame_images[:num_frames]:
                if upscale > 1:
                    img = img.resize(
                        (img.width * upscale, img.height * upscale),
                        PILImage.LANCZOS,
                    )
                upscaled.append(img)
            # Pad with black images if needed
            while len(upscaled) < num_frames:
                size = (384, 384) if not upscaled else (upscaled[0].width, upscaled[0].height)
                upscaled.append(PILImage.new("RGB", size, (0, 0, 0)))

            pixel_values = self._image_processor(
                images=upscaled, return_tensors="pt"
            )["pixel_values"]
            batch["frame_images"] = pixel_values.unsqueeze(0)  # 1 x F x C x H x W
            batch["has_images"] = torch.tensor([True], dtype=torch.bool)
        else:
            batch["has_images"] = torch.tensor([False], dtype=torch.bool)

        return batch

    def _decode_flat(
        self,
        scores_tensor: torch.Tensor,
        top_k: int,
        min_score: float,
    ) -> Dict[str, Any]:
        """Simple top-k / threshold decoding without hierarchy."""
        sorted_scores, sorted_indices = scores_tensor.sort(descending=True)
        tags: List[Dict[str, Any]] = []

        for score_t, idx_t in zip(sorted_scores, sorted_indices):
            if len(tags) >= top_k:
                break
            score_val = score_t.item()
            if score_val < min_score:
                break

            idx = idx_t.item()
            tag_id = self.tag_idx_to_id[idx]
            tag_info = self.taxonomy["by_id"].get(tag_id, {})

            tags.append({
                "tag_id": tag_id,
                "tag_name": tag_info.get("name", ""),
                "score": round(score_val, 4),
                "path": tag_info.get("path_string", ""),
                "decode_type": "direct",
            })

        return {"tags": tags, "scene_embedding": None}

    def _decode_hierarchical(
        self,
        scores_tensor: torch.Tensor,
        top_k: int,
        min_score: float,
    ) -> Dict[str, Any]:
        """Hierarchical decoding with parent/child constraints."""
        # Build tag_id -> score mapping for the decoder
        score_dict: Dict[str, float] = {}
        for idx in range(self._num_tags):
            tag_id = self.tag_idx_to_id[idx]
            score_dict[tag_id] = scores_tensor[idx].item()

        decoded = self.decoder.decode(score_dict, return_all=False)

        # Apply min_score filter and top_k cap
        tags: List[Dict[str, Any]] = []
        for dt in decoded:
            if dt.score < min_score:
                continue
            if len(tags) >= top_k:
                break
            tag_info = self.taxonomy["by_id"].get(dt.tag_id, {})
            tags.append({
                "tag_id": dt.tag_id,
                "tag_name": dt.tag_name,
                "score": round(dt.score, 4),
                "path": tag_info.get("path_string", " > ".join(dt.path)),
                "decode_type": dt.decode_type,
            })

        return {"tags": tags, "scene_embedding": None}
