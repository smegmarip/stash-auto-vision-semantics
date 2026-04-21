"""
Training configuration: paths, hyperparameters, and training settings.
Supports both text-only and text+vision modes via use_vision flag.
"""
import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
LOADER_DATA_DIR = Path(os.getenv("LOADER_DATA_DIR", str(PROJECT_ROOT / "data")))


@dataclass
class TrainConfig:
    # --- Paths ---
    captions_path: str = str(LOADER_DATA_DIR / "captions.json")
    scenes_path: str = str(LOADER_DATA_DIR / "scenes.json")
    summaries_path: str = str(LOADER_DATA_DIR / "summaries.json")
    taxonomy_path: str = str(LOADER_DATA_DIR / "taxonomy.json")
    output_dir: str = str(PROJECT_ROOT / "models" / "default")

    # --- Text backbone ---
    backbone_name: str = "BAAI/bge-large-en-v1.5"
    backbone_dim: int = 1024
    trust_remote_code: bool = False

    # --- Vision (enabled via --use-vision) ---
    use_vision: bool = False
    vision_model: str = "google/siglip-base-patch16-384"
    frames_dir: str = str(LOADER_DATA_DIR / "frames")
    gate_init_bias: float = 0.85
    vision_alignment_weight: float = 0.05

    # --- Architecture ---
    model_dim: int = 512
    num_heads: int = 8
    temporal_layers: int = 2
    temporal_dropout: float = 0.1
    path_gru_layers: int = 1
    num_frames: int = 16
    max_seq_length: int = 256

    # --- Tag families ---
    family_depth: int = 2
    family_modulation: bool = True

    # --- Training stage 1: frozen backbones ---
    epochs_frozen: int = 5
    lr_frozen: float = 1e-3
    warmup_ratio_frozen: float = 0.1

    # --- Training stage 2: fine-tune text backbone ---
    epochs_finetune: int = 15
    lr_heads: float = 5e-4
    lr_backbone: float = 2e-5
    warmup_ratio_finetune: float = 0.05

    # --- Training stage 3: fine-tune vision backbone (optional) ---
    epochs_vision_finetune: int = 0
    lr_vision: float = 5e-6

    # --- Shared training ---
    batch_size: int = 4
    gradient_accumulation: int = 8
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    use_amp: bool = True
    gradient_checkpointing: bool = True

    # --- Loss ---
    pcw_lambda: float = 0.2
    consistency_weight: float = 0.1
    focal_gamma: float = 1.0
    init_temperature: float = 0.07

    # --- Data split ---
    val_ratio: float = 0.10
    test_ratio: float = 0.05

    # --- Tag cache ---
    tag_cache_refresh_epochs: int = 3

    # --- Checkpointing ---
    save_every_n_epochs: int = 1
    seed: int = 42

    def resolve_paths(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
