"""
Semantics Service - Data Models
Merged request/response models for tag classification pipeline.

Replaces both the old SigLIP semantics service and JoyCaption captioning service
with a unified interface backed by the trained multi-view tag classifier.
"""

import os
from enum import Enum
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    """Job processing status"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    NOT_IMPLEMENTED = "not_implemented"
    WAITING_FOR_GPU = "waiting_for_gpu"


class FrameSelectionMethod(str, Enum):
    """Frame selection strategies"""
    SCENE_BASED = "scene_based"
    INTERVAL = "interval"
    SPRITE_SHEET = "sprite_sheet"


class SemanticsOperation(str, Enum):
    """Selectable pipeline operations"""
    TITLE = "title"
    SUMMARY = "summary"
    TAGS = "tags"
    ALL = "all"


# ---------------------------------------------------------------------------
# Tag taxonomy models (from Stash GraphQL schema)
# ---------------------------------------------------------------------------

class TagRef(BaseModel):
    """Reference to a tag (used in parents/children arrays in Stash API)"""
    id: str
    name: str


class TagTaxonomyNode(BaseModel):
    """
    A node in the Stash tag taxonomy.
    Matches Stash GraphQL Tag entity for custom_taxonomy compatibility.
    """
    id: str
    name: str
    description: Optional[str] = Field(default=None)
    aliases: List[str] = Field(default_factory=list)
    ignore_auto_tag: bool = Field(default=False)
    parents: List[TagRef] = Field(default_factory=list)
    children: List[TagRef] = Field(default_factory=list)

    @property
    def parent_id(self) -> Optional[str]:
        return self.parents[0].id if self.parents else None

    @property
    def child_ids(self) -> List[str]:
        return [c.id for c in self.children]


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class SemanticsParameters(BaseModel):
    """
    Unified parameters for the tag classification pipeline.

    Covers frame extraction, captioning, summarization, and classification.
    """
    model_config = {"protected_namespaces": ()}

    # --- Classifier parameters ---
    model_variant: str = Field(
        default=os.getenv("CLASSIFIER_MODEL", "text-only"),
        description="Classifier model variant: 'vision', 'text-only', or path to checkpoint"
    )
    min_confidence: float = Field(
        default=float(os.getenv("SEMANTICS_MIN_CONFIDENCE", "0.75")),
        ge=0.0, le=1.0,
        description="Minimum confidence threshold for tag predictions"
    )
    top_k_tags: int = Field(
        default=30,
        ge=1, le=100,
        description="Maximum number of tags to return"
    )
    generate_embeddings: bool = Field(
        default=False,
        description="Return 512-D scene embeddings from the classifier"
    )
    use_hierarchical_decoding: bool = Field(
        default=True,
        description="Apply taxonomy-consistent post-processing (parent thresholds, child competition)"
    )

    # --- Frame extraction parameters ---
    frame_selection: FrameSelectionMethod = Field(
        default=FrameSelectionMethod.SPRITE_SHEET,
        description="How to select frames for captioning"
    )
    frames_per_scene: int = Field(
        default=16,
        ge=1, le=32,
        description="Frames to extract per scene (classifier trained on 16)"
    )
    sampling_interval: float = Field(
        default=2.0,
        ge=0.1, le=60.0,
        description="Frame sampling interval in seconds (interval mode)"
    )
    select_sharpest: bool = Field(
        default=True,
        description="Select sharpest frames per scene using Laplacian variance"
    )
    sharpness_candidate_multiplier: int = Field(
        default=3,
        ge=1, le=10,
        description="Extract N * frames_per_scene candidates for sharpness selection"
    )
    min_frame_quality: float = Field(
        default=0.05,
        ge=0.0, le=1.0,
        description="Minimum quality score to accept a frame (filters black/blank frames)"
    )

    # --- Operation selection ---
    operations: Optional[List[SemanticsOperation]] = Field(
        default=None,
        description="Operations to perform: 'title', 'summary', 'tags', or 'all'. "
                    "Default (null) performs all operations. Accepts a list of values."
    )

    # --- Captioning parameters ---
    use_quantization: bool = Field(
        default=True,
        description="Use 4-bit quantization for JoyCaption VLM (reduces VRAM ~17GB to ~8GB)"
    )

    # --- Scene metadata overrides (optional, fetched from Stash if not provided) ---
    details: Optional[str] = Field(
        default=None,
        description="Promotional/editorial scene description (promo_desc)"
    )
    sprite_vtt_url: Optional[str] = Field(
        default=None,
        description="URL to sprite VTT file for sprite_sheet frame selection"
    )
    sprite_image_url: Optional[str] = Field(
        default=None,
        description="URL to sprite grid image for sprite_sheet frame selection"
    )
    scene_boundaries: Optional[List[Dict[str, float]]] = Field(
        default=None,
        description="Pre-computed scene boundaries [{start_timestamp, end_timestamp}, ...]"
    )


class SceneContext(BaseModel):
    """Resolved scene data for the classification pipeline.

    Built from findScene GraphQL response and optionally overridden by
    request parameters.  This is the single source of truth for all
    pipeline steps (frame extraction, captioning, summarisation,
    classification).
    """
    scene_id: str
    source: str = ""
    title: Optional[str] = None
    # Sprite sheet
    sprite_image_url: Optional[str] = None
    sprite_vtt_url: Optional[str] = None
    # Promo / editorial description
    details: Optional[str] = None
    # Video metadata
    duration: float = 0
    frame_rate: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    # Performers
    performer_count: int = 0
    performer_genders: List[str] = Field(default_factory=list)

    @property
    def has_promo(self) -> bool:
        return bool(self.details and self.details.strip())

    @property
    def promo_desc(self) -> str:
        return self.details or ""

    @property
    def resolution(self) -> str:
        if self.width and self.height:
            return f"{self.width}x{self.height}"
        return "Unknown"


class AnalyzeSemanticsRequest(BaseModel):
    """Request to analyze scene semantics via tag classification pipeline."""

    source: str = Field(default="", description="Video path or URL (resolved from Stash if empty)")
    source_id: str = Field(..., description="Scene identifier (required)")
    job_id: Optional[str] = Field(default=None, description="Parent job ID for tracking")
    frame_extraction_job_id: Optional[str] = Field(
        default=None,
        description="Job ID from Frame Server (reuse extracted frames)"
    )
    scenes_job_id: Optional[str] = Field(
        default=None,
        description="Job ID from Scenes Service (fetch pre-computed scene boundaries)"
    )
    custom_taxonomy: Optional[Union[str, List[Dict[str, Any]]]] = Field(
        default=None,
        description="Custom taxonomy override: URL to fetch JSON, or inline array of Stash Tag objects "
                    "(from findTags query). Overrides the taxonomy loaded at startup."
    )
    parameters: SemanticsParameters = Field(default_factory=SemanticsParameters)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class AnalyzeSemanticsResponse(BaseModel):
    """Response for semantics analysis job submission"""
    job_id: str
    status: JobStatus
    message: Optional[str] = None
    created_at: str
    cache_hit: bool = Field(default=False)


class JobStatusResponse(BaseModel):
    """Job status response"""
    job_id: str
    status: JobStatus
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    stage: Optional[str] = None
    message: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    gpu_wait_position: Optional[int] = None


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class ClassifierTag(BaseModel):
    """A tag prediction from the trained classifier"""
    tag_id: str = Field(..., description="Stash tag ID")
    tag_name: str = Field(..., description="Tag name")
    score: float = Field(..., ge=0.0, le=1.0, description="Classifier confidence score")
    path: str = Field(default="", description="Taxonomy path (e.g., 'Semantics > Apparel > Accessories')")
    decode_type: str = Field(
        default="direct",
        description="How this tag was selected: 'direct' (above threshold) or 'parent_only' (activated by child)"
    )


class FrameCaptionResult(BaseModel):
    """Caption for a single frame"""
    frame_index: int
    timestamp: float
    caption: str = Field(..., description="JoyCaption taxonomy-prompt output")


class SemanticsOutcome(BaseModel):
    """Complete tag classification results for a scene"""
    tags: List[ClassifierTag] = Field(default_factory=list, description="Predicted tags sorted by score (descending)")
    frame_captions: List[FrameCaptionResult] = Field(description="Per-frame captions (16 frames)")
    scene_summary: Optional[str] = Field(default=None, description="LLM narrative summary of the scene (null when summary not in operations)")
    suggested_title: Optional[str] = Field(
        default=None,
        description="LLM-generated catchy scene title derived from summary and promotional description"
    )
    scene_embedding: Optional[List[float]] = Field(
        default=None,
        description="512-D scene embedding from classifier (if generate_embeddings=True)"
    )


class SemanticsMetadata(BaseModel):
    """Processing metadata"""
    source: str
    source_id: str
    total_frames_extracted: int
    frames_captioned: int
    classifier_model: str
    caption_model: str = "fancyfeast/llama-joycaption-beta-one-hf-llava"
    summary_model: str = "RedHatAI/Llama-3.1-8B-Instruct"
    processing_time_seconds: float
    device: str
    taxonomy_size: int = Field(description="Number of active tags in taxonomy")
    has_promo: bool = Field(description="Whether promotional description was available")
    sprite_image_url: Optional[str] = Field(default=None, description="Sprite sheet URL for frame thumbnails")
    sprite_vtt_url: Optional[str] = Field(default=None, description="Sprite VTT URL for frame coordinates")
    # Taxonomy tag name → ID lookup (for ancestor tag images in the UI)
    tag_name_to_id: Dict[str, str] = Field(default_factory=dict, description="Tag name to ID mapping from taxonomy")
    # Scene metadata from Stash
    scene: Optional["SceneMetadata"] = Field(default=None, description="Scene metadata from Stash")


class SceneMetadata(BaseModel):
    """Scene metadata resolved from Stash via findScene."""
    title: Optional[str] = None
    duration: Optional[float] = Field(default=None, description="Duration in seconds")
    resolution: Optional[str] = Field(default=None, description="Video resolution (e.g. 1920x1080)")
    frame_rate: Optional[float] = None
    performer_count: int = 0
    performer_genders: List[str] = Field(default_factory=list)


class SemanticsResults(BaseModel):
    """Complete semantics analysis results"""
    job_id: str
    source_id: str
    status: JobStatus
    semantics: SemanticsOutcome
    metadata: SemanticsMetadata


# ---------------------------------------------------------------------------
# Health / utility models
# ---------------------------------------------------------------------------

class TaxonomyStatus(BaseModel):
    """Taxonomy loading status"""
    loaded: bool
    tag_count: int = 0
    source: str = "none"
    last_loaded: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str = "semantics-service"
    version: str = "2.0.0"
    implemented: bool = True
    phase: int = 3
    message: Optional[str] = None
    classifier_model: Optional[str] = None
    classifier_loaded: bool = False
    device: Optional[str] = None
    taxonomy: TaxonomyStatus = Field(default_factory=TaxonomyStatus)
    default_min_confidence: float = 0.75
    queue: Optional[Dict[str, Any]] = Field(default=None, description="Queue stats: pending count, active job, worker_id")


class Frame(BaseModel):
    """Extracted frame metadata"""
    index: int
    timestamp: float
    url: str
    width: int
    height: int


class FrameMetadata(BaseModel):
    """Metadata for a single extracted frame"""
    video_path: str
    extraction_method: str
    total_frames: int
    video_duration_seconds: float
    video_fps: float
    processing_time_seconds: float


class FramesExtractionResult(BaseModel):
    """Frame extraction result metadata"""
    job_id: str
    status: JobStatus
    cache_key: str
    frames: List[Frame]
    metadata: FrameMetadata
