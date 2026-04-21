"""
Multi-View Classifier model with optional visual grounding.

Text-only mode (use_vision=False):
  frame captions → text backbone → temporal transformer → scene representation
  summary → cross-attention → scene representation
  promo_desc → provenance fusion → scene representation

Vision mode (use_vision=True) adds:
  sprite frames → vision encoder → gated fusion with text frames
  visual-text alignment loss
"""
import logging
import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        return self.norm(self.linear(x))


class GatedFusion(nn.Module):
    """Per-frame gated fusion of text and visual embeddings.
    Initialized text-favoring so the model starts from the working
    text-only representation."""

    def __init__(self, dim: int, text_bias: float = 0.85):
        super().__init__()
        self.gate_proj = nn.Linear(dim * 2, 1)
        nn.init.constant_(self.gate_proj.bias, math.log(text_bias / (1 - text_bias)))
        nn.init.zeros_(self.gate_proj.weight)

    def forward(self, text, vision):
        alpha = torch.sigmoid(self.gate_proj(torch.cat([text, vision], dim=-1)))
        return alpha * text + (1 - alpha) * vision


class MultiViewClassifier(nn.Module):

    def __init__(self, config, num_tags: int, num_families: int = 0):
        super().__init__()
        self.config = config
        self.num_tags = num_tags

        # ---- Text backbone ----
        logger.info(f"Loading backbone: {config.backbone_name}")
        self.backbone = AutoModel.from_pretrained(config.backbone_name, trust_remote_code=config.trust_remote_code)
        self.tokenizer = AutoTokenizer.from_pretrained(config.backbone_name, trust_remote_code=config.trust_remote_code)
        backbone_dim = self.backbone.config.hidden_size
        config.backbone_dim = backbone_dim
        logger.info(f"Backbone dim: {backbone_dim}")

        if config.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        # ---- Projection heads ----
        d = config.model_dim
        self.frame_proj = ProjectionHead(backbone_dim, d)
        self.summary_proj = ProjectionHead(backbone_dim, d)
        self.promo_proj = ProjectionHead(backbone_dim, d)
        self.tag_proj = ProjectionHead(backbone_dim, d)

        # ---- Temporal transformer ----
        self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.frame_positions = nn.Embedding(config.num_frames + 1, d)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=config.num_heads, dim_feedforward=d * 4,
            dropout=config.temporal_dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.temporal_layers)

        # ---- Summary cross-attention ----
        self.summary_cross_attn = nn.MultiheadAttention(d, config.num_heads, dropout=0.1, batch_first=True)
        self.summary_cross_norm = nn.LayerNorm(d)

        # ---- Promo handling ----
        self.missing_promo_embed = nn.Parameter(torch.zeros(d))

        # ---- Vision encoder (optional) ----
        self.vision_encoder = None
        self.vision_proj = None
        self.gated_fusion = None
        if config.use_vision:
            from transformers import SiglipVisionModel
            logger.info(f"Loading vision encoder: {config.vision_model}")
            self.vision_encoder = SiglipVisionModel.from_pretrained(config.vision_model)
            vision_dim = self.vision_encoder.config.hidden_size
            self.vision_proj = ProjectionHead(vision_dim, d)
            self.gated_fusion = GatedFusion(d, text_bias=config.gate_init_bias)
            self.vision_encoder.requires_grad_(False)
            logger.info(f"Vision dim: {vision_dim}, gate text_bias={config.gate_init_bias}")

        # ---- Scene fusion ----
        self.scene_fusion = nn.Sequential(nn.Linear(d * 3, d), nn.LayerNorm(d), nn.GELU())

        # ---- Path encoder ----
        self.root_marker = nn.Parameter(torch.randn(d) * 0.02)
        self.parent_marker = nn.Parameter(torch.randn(d) * 0.02)
        self.path_gru = nn.GRU(input_size=d, hidden_size=d, num_layers=config.path_gru_layers, batch_first=True)

        # ---- Tag combination ----
        self.tag_combine = nn.Sequential(nn.Linear(d * 2, d), nn.LayerNorm(d))

        # ---- Tag family modulation ----
        if config.family_modulation and num_families > 0:
            self.family_bias = nn.Embedding(num_families, d)
            nn.init.zeros_(self.family_bias.weight)
        else:
            self.family_bias = None

        # ---- Scoring ----
        self.log_temperature = nn.Parameter(torch.tensor(math.log(1.0 / config.init_temperature)))
        self.score_gate = nn.Parameter(torch.tensor(0.0))

        # ---- Consistency ----
        self.consistency_proj = nn.Linear(d, d, bias=False)

        # ---- Caches ----
        self.register_buffer("_tag_cache", torch.zeros(num_tags, d))
        self.register_buffer("_tag_cache_valid", torch.tensor(False))

        # ---- Internal state for loss computation ----
        self._last_visual_embeds = None
        self._last_text_frame_embeds = None

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    def _encode_backbone(self, texts, device):
        tokens = self.tokenizer(texts, padding=True, truncation=True, max_length=self.config.max_seq_length, return_tensors="pt").to(device)
        return self.backbone(**tokens).last_hidden_state[:, 0]

    def _encode_backbone_batched(self, texts, device, chunk_size=64):
        if len(texts) <= chunk_size:
            return self._encode_backbone(texts, device)
        parts = []
        for i in range(0, len(texts), chunk_size):
            parts.append(self._encode_backbone(texts[i:i + chunk_size], device))
        return torch.cat(parts, dim=0)

    # ------------------------------------------------------------------
    # Scene encoder
    # ------------------------------------------------------------------

    def encode_scene(self, batch, device):
        B = len(batch["summaries"])
        d = self.config.model_dim

        # Text encoding
        all_texts = []
        for captions in batch["frame_captions"]:
            all_texts.extend(captions[:self.config.num_frames])
        n_frames_total = B * self.config.num_frames
        all_texts.extend(batch["summaries"])
        all_texts.extend(batch["promo_descs"])

        all_embeds = self._encode_backbone_batched(all_texts, device)
        frame_embeds = self.frame_proj(all_embeds[:n_frames_total]).view(B, self.config.num_frames, d)
        summary_embed = self.summary_proj(all_embeds[n_frames_total:n_frames_total + B])
        promo_embed = self.promo_proj(all_embeds[n_frames_total + B:])

        # Visual grounding (optional)
        self._last_visual_embeds = None
        self._last_text_frame_embeds = frame_embeds.detach()

        if self.vision_encoder is not None and "frame_images" in batch:
            images = batch["frame_images"].to(device)
            B_img, F_img = images.shape[:2]
            flat_images = images.view(B_img * F_img, *images.shape[2:])
            ctx = torch.no_grad() if not any(p.requires_grad for p in self.vision_encoder.parameters()) else torch.enable_grad()
            with ctx:
                vision_out = self.vision_encoder(pixel_values=flat_images)
            visual_embeds = self.vision_proj(vision_out.pooler_output).view(B, F_img, d)
            self._last_visual_embeds = visual_embeds.detach()

            has_images = batch["has_images"].to(device).float().view(B, 1, 1)
            frame_embeds = self.gated_fusion(frame_embeds, visual_embeds) * has_images + frame_embeds * (1 - has_images)

        # Temporal transformer
        cls = self.cls_token.expand(B, -1, -1)
        sequence = torch.cat([cls, frame_embeds], dim=1)
        pos_ids = torch.arange(self.config.num_frames + 1, device=device)
        sequence = sequence + self.frame_positions(pos_ids).unsqueeze(0)
        temporal_out = self.temporal_encoder(sequence)
        frame_cls = temporal_out[:, 0]
        frame_states = temporal_out[:, 1:]

        # Summary cross-attention
        summary_q = summary_embed.unsqueeze(1)
        attn_out, _ = self.summary_cross_attn(summary_q, frame_states, frame_states)
        summary_attended = self.summary_cross_norm(attn_out.squeeze(1) + summary_embed)

        # Promo handling
        has_promo = batch["has_promo"].to(device).float().unsqueeze(1)
        promo_final = promo_embed * has_promo + self.missing_promo_embed.unsqueeze(0) * (1 - has_promo)

        # Scene fusion
        scene_repr = self.scene_fusion(torch.cat([frame_cls, summary_attended, promo_final], dim=-1))

        return scene_repr, frame_states, frame_cls, summary_embed

    # ------------------------------------------------------------------
    # Tag encoder
    # ------------------------------------------------------------------

    def encode_tag_text(self, tag_texts, device):
        return self.tag_proj(self._encode_backbone_batched(tag_texts, device))

    def encode_paths(self, paths, device):
        if not paths:
            return torch.zeros(0, self.config.model_dim, device=device)
        unique_nodes = list({node for path in paths for node in path})
        node_to_idx = {n: i for i, n in enumerate(unique_nodes)}
        node_embeds = self.encode_tag_text(unique_nodes, device)

        max_len = max(len(p) for p in paths)
        d = self.config.model_dim
        padded = torch.zeros(len(paths), max_len, d, device=device)
        lengths = []
        for i, path in enumerate(paths):
            lengths.append(len(path))
            for j, node in enumerate(path):
                padded[i, j] = node_embeds[node_to_idx[node]] + (self.root_marker if j == 0 else self.parent_marker)

        packed = nn.utils.rnn.pack_padded_sequence(padded, torch.tensor(lengths, device="cpu"), batch_first=True, enforce_sorted=False)
        _, hidden = self.path_gru(packed)
        return hidden[-1]

    def build_tag_cache(self, tags, tag_id_to_idx, tag_to_families, device):
        d = self.config.model_dim
        n = len(tag_id_to_idx)

        tag_texts = [""] * n
        tag_paths = [[] for _ in range(n)]
        for tag in tags:
            tid = str(tag["id"])
            if tid not in tag_id_to_idx:
                continue
            idx = tag_id_to_idx[tid]
            name, desc = tag.get("name", ""), tag.get("description", "")
            tag_texts[idx] = f"{name}: {desc}" if desc else name
            tag_paths[idx] = tag.get("all_paths", [])

        with torch.no_grad():
            text_embeds = self.encode_tag_text(tag_texts, device)

        all_paths, path_to_tag = [], []
        for tag_idx, paths in enumerate(tag_paths):
            for path in paths:
                if path:
                    path_to_tag.append(tag_idx)
                    all_paths.append(path)

        if all_paths:
            with torch.no_grad():
                all_path_embeds = self.encode_paths(all_paths, device)
            path_embed_per_tag = torch.zeros(n, d, device=device)
            for pi, tag_idx in enumerate(path_to_tag):
                if all_path_embeds[pi].norm() > path_embed_per_tag[tag_idx].norm():
                    path_embed_per_tag[tag_idx] = all_path_embeds[pi]
        else:
            path_embed_per_tag = torch.zeros(n, d, device=device)

        with torch.no_grad():
            combined = self.tag_combine(torch.cat([text_embeds, path_embed_per_tag], dim=-1))

        if self.family_bias is not None and tag_to_families:
            with torch.no_grad():
                for tid, fam_ids in tag_to_families.items():
                    if tid in tag_id_to_idx:
                        idx = tag_id_to_idx[tid]
                        combined[idx] += self.family_bias(torch.tensor(fam_ids, device=device)).mean(0)

        self._tag_cache = F.normalize(combined, dim=-1)
        self._tag_cache_valid = torch.tensor(True, device=device)
        logger.info(f"Tag cache built: {n} tags, {len(all_paths)} paths")

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, scene_repr, frame_states):
        tag_embeds = self._tag_cache
        temperature = self.log_temperature.clamp(max=2.65).exp()
        scene_norm = F.normalize(scene_repr, dim=-1)
        scene_scores = torch.mm(scene_norm, tag_embeds.T) * temperature
        frame_norm = F.normalize(frame_states, dim=-1)
        frame_tag_sim = torch.einsum("bfd,td->bft", frame_norm, tag_embeds)
        max_frame_scores = frame_tag_sim.max(dim=1).values * temperature
        gate = torch.sigmoid(self.score_gate)
        return gate * scene_scores + (1 - gate) * max_frame_scores

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(self, scores, labels, positive_mask, summary_embed, frame_cls):
        probs = torch.sigmoid(scores)
        bce = F.binary_cross_entropy_with_logits(scores, labels, reduction="none")
        focal_gamma = getattr(self.config, "focal_gamma", 1.0)
        pt = probs * labels + (1 - probs) * (1 - labels)
        focal_bce = bce * (1 - pt).pow(focal_gamma)

        pos_term = (focal_bce * positive_mask.float()).sum() / positive_mask.float().sum().clamp(min=1)
        neg_mask = ~positive_mask
        neg_term = (focal_bce * neg_mask.float()).sum() / neg_mask.float().sum().clamp(min=1)
        classification_loss = pos_term + self.config.pcw_lambda * neg_term

        frame_proj = self.consistency_proj(frame_cls)
        consistency = 1.0 - F.cosine_similarity(F.normalize(frame_proj, dim=-1), F.normalize(summary_embed.detach(), dim=-1), dim=-1).mean()

        total = classification_loss + self.config.consistency_weight * consistency

        # Visual-text alignment loss
        vision_alignment = torch.tensor(0.0, device=scores.device)
        if self._last_visual_embeds is not None:
            text_f = F.normalize(self._last_text_frame_embeds, dim=-1)
            vis_f = F.normalize(self._last_visual_embeds, dim=-1)
            vision_alignment = 1.0 - F.cosine_similarity(text_f, vis_f, dim=-1).mean()
            total = total + self.config.vision_alignment_weight * vision_alignment

        return {
            "total": total, "positive": pos_term.detach(), "negative": neg_term.detach(),
            "consistency": consistency.detach(), "vision_alignment": vision_alignment.detach(),
        }

    # ------------------------------------------------------------------
    # Forward / predict
    # ------------------------------------------------------------------

    def forward(self, batch, device):
        assert self._tag_cache_valid.item(), "Call build_tag_cache() before forward()"
        scene_repr, frame_states, frame_cls, summary_embed = self.encode_scene(batch, device)
        scores = self.score(scene_repr, frame_states)
        labels = batch["labels"].to(device)
        positive_mask = batch["positive_mask"].to(device)
        losses = self.compute_loss(scores, labels, positive_mask, summary_embed, frame_cls)
        return {"scores": scores, **losses}

    @torch.no_grad()
    def predict(self, batch, device):
        scene_repr, frame_states, _, _ = self.encode_scene(batch, device)
        return torch.sigmoid(self.score(scene_repr, frame_states))

    # ------------------------------------------------------------------
    # Freeze / unfreeze
    # ------------------------------------------------------------------

    def freeze_backbone(self):
        self.backbone.requires_grad_(False)
        self.backbone.gradient_checkpointing_disable()
        logger.info("Backbone frozen")

    def unfreeze_backbone(self):
        self.backbone.requires_grad_(True)
        if self.config.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
        logger.info("Backbone unfrozen")

    def freeze_vision(self):
        if self.vision_encoder:
            self.vision_encoder.requires_grad_(False)
            logger.info("Vision encoder frozen")

    def unfreeze_vision(self):
        if self.vision_encoder:
            self.vision_encoder.requires_grad_(True)
            logger.info("Vision encoder unfrozen")
