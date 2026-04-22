"""
Semantics Service — Standalone Edition

FastAPI server for tag classification using trained multi-view classifier.
Extracted from stash-auto-vision for batch processing on rented cloud GPUs.

Pipeline: sprite extraction → JoyCaption → LLM summary → tag classifier (with vision)

All models loaded at startup and stay resident (80GB VRAM).
No quantization, no GPU lease management, no model swapping.
"""

import gc
import io
import os
import time
import uuid
import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import httpx

from .models import (
    AnalyzeSemanticsRequest,
    AnalyzeSemanticsResponse,
    JobStatusResponse,
    SceneContext,
    SemanticsOutcome,
    SemanticsResults,
    SemanticsMetadata,
    SceneMetadata,
    ClassifierTag,
    FrameCaptionResult,
    HealthResponse,
    TaxonomyStatus,
    JobStatus,
    FrameSelectionMethod,
    SemanticsOperation,
)
from .cache_manager import CacheManager
from .classifier import TagClassifier
from .caption_generator import CaptionGenerator
from .llama_runtime import LlamaRuntime
from .summary_generator import SummaryGenerator
from .title_generator import TitleGenerator
from .taxonomy_builder import TaxonomyBuilder
from .sprite_parser import SpriteParser
from .job_queue import JobQueue
from .worker import SemanticsWorker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
STASH_URL = os.getenv("STASH_URL", "")
STASH_API_KEY = os.getenv("STASH_API_KEY", "")
SEMANTICS_TAG_ID = os.getenv("SEMANTICS_TAG_ID", "")
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "vision")
CLASSIFIER_DEVICE = os.getenv("CLASSIFIER_DEVICE", "cuda")
CACHE_TTL = int(os.getenv("CACHE_TTL", "31536000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global instances
# ---------------------------------------------------------------------------
cache_manager: Optional[CacheManager] = None
tag_classifier: Optional[TagClassifier] = None
sprite_parser: Optional[SpriteParser] = None
llama_runtime: Optional[LlamaRuntime] = None
summary_generator: Optional[SummaryGenerator] = None
title_generator: Optional[TitleGenerator] = None
caption_generator: Optional[CaptionGenerator] = None
job_queue: Optional[JobQueue] = None
worker: Optional[SemanticsWorker] = None
taxonomy_data: Optional[dict] = None
taxonomy_status: TaxonomyStatus = TaxonomyStatus(loaded=False)

JOB_LOCK_TTL = int(os.getenv("SEMANTICS_JOB_LOCK_TTL", "3600"))


async def _load_taxonomy_background():
    """Background task: fetch taxonomy from Stash and initialize classifier tag cache."""
    global taxonomy_data, taxonomy_status, tag_classifier

    if not STASH_URL:
        logger.warning("STASH_URL not set — taxonomy must be provided via custom_taxonomy parameter")
        return

    try:
        logger.info(f"Loading taxonomy from {STASH_URL} (root_tag_id={SEMANTICS_TAG_ID or 'all'})")
        builder = TaxonomyBuilder()
        taxonomy_data = await builder.build_from_stash(
            stash_url=STASH_URL,
            stash_api_key=STASH_API_KEY,
            root_tag_id=SEMANTICS_TAG_ID or None,
        )
        tag_count = len(taxonomy_data.get("tags", []))
        logger.info(f"Taxonomy loaded: {tag_count} tags")

        if tag_classifier and tag_classifier.is_checkpoint_ready:
            tag_classifier.load_taxonomy(taxonomy_data)
            logger.info("Classifier tag cache rebuilt from taxonomy")

        taxonomy_status.loaded = True
        taxonomy_status.tag_count = tag_count
        taxonomy_status.source = "stash"
        taxonomy_status.last_loaded = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())

    except Exception as e:
        logger.error(f"Failed to load taxonomy from Stash: {e}", exc_info=True)
        taxonomy_status.loaded = False
        taxonomy_status.source = f"error: {str(e)[:100]}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load all models at startup, cleanup on shutdown."""
    global cache_manager, tag_classifier, sprite_parser
    global llama_runtime, summary_generator, title_generator, caption_generator
    global job_queue, worker

    logger.info("Starting Standalone Semantics Service")

    # 1. Redis
    cache_manager = CacheManager(REDIS_URL, module="semantics", ttl=CACHE_TTL)
    await cache_manager.connect()
    logger.info("Cache manager initialized")

    # 2. Job queue
    job_queue = JobQueue(redis=cache_manager.redis, module="semantics", lock_ttl_seconds=JOB_LOCK_TTL)
    logger.info(f"Job queue initialized (worker_id={job_queue.worker_id}, lock_ttl={JOB_LOCK_TTL}s)")

    # 3. Sprite parser
    sprite_parser = SpriteParser(stash_api_key=STASH_API_KEY)
    logger.info("Sprite parser initialized")

    # 4. Load JoyCaption (bfloat16, ~16GB) — blocking
    caption_generator = CaptionGenerator(device=CLASSIFIER_DEVICE)
    logger.info("Loading JoyCaption (bfloat16)...")
    await asyncio.to_thread(caption_generator.load)
    logger.info("JoyCaption loaded")

    # 5. Load Llama runtime (bfloat16, ~16GB) — blocking
    llama_runtime = LlamaRuntime(device=CLASSIFIER_DEVICE)
    logger.info("Loading Llama runtime (bfloat16)...")
    await asyncio.to_thread(llama_runtime.load)
    summary_generator = SummaryGenerator(llm=llama_runtime)
    title_generator = TitleGenerator(llm=llama_runtime)
    logger.info("Llama runtime loaded")

    # 6. Load Tag Classifier (vision variant, ~3.8GB) — blocking
    tag_classifier = TagClassifier(model_variant=CLASSIFIER_MODEL, device=CLASSIFIER_DEVICE)
    try:
        tag_classifier.load_model()
        logger.info(f"Tag classifier loaded: {CLASSIFIER_MODEL}")
    except Exception as e:
        logger.error(f"Failed to load classifier model: {e}", exc_info=True)
        logger.warning("Classifier will be unavailable until model is loaded")

    # 7. Background taxonomy load from Stash (non-blocking)
    asyncio.create_task(_load_taxonomy_background())

    # 8. Start worker
    worker = SemanticsWorker(queue=job_queue, process_fn=_run_pipeline)
    await worker.start()

    logger.info("Standalone Semantics Service ready")

    yield

    # Cleanup
    logger.info("Shutting down Semantics Service...")
    if worker:
        await worker.stop()
    if cache_manager:
        await cache_manager.disconnect()
    logger.info("Semantics Service stopped")


app = FastAPI(
    title="Semantics Service (Standalone)",
    description="Tag classification using trained multi-view classifier — standalone cloud GPU edition",
    version="3.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Background processing
# ---------------------------------------------------------------------------

async def _load_custom_taxonomy(custom_taxonomy) -> dict:
    """Load taxonomy from custom_taxonomy parameter (URL or inline tags array)."""
    builder = TaxonomyBuilder()

    if isinstance(custom_taxonomy, str):
        # URL — fetch and parse
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(custom_taxonomy)
            resp.raise_for_status()
            data = resp.json()
        # Accept either raw tags array or findTags wrapper
        if isinstance(data, dict) and "data" in data:
            tags = data["data"]["findTags"]["tags"]
        elif isinstance(data, dict) and "tags" in data:
            tags = data["tags"]
        elif isinstance(data, list):
            tags = data
        else:
            raise ValueError("Unrecognized taxonomy JSON format")
    elif isinstance(custom_taxonomy, list):
        tags = custom_taxonomy
    else:
        raise ValueError(f"custom_taxonomy must be a URL string or list of tag dicts, got {type(custom_taxonomy)}")

    return builder.build_from_tags(tags, root_tag_id=SEMANTICS_TAG_ID or None)


async def _fetch_scene_from_stash(scene_id: str) -> Dict[str, Any]:
    """Fetch scene data from Stash via findScene GraphQL query."""
    if not STASH_URL:
        raise RuntimeError("STASH_URL not set — cannot fetch scene data")

    query = """
    query FindScene($id: ID!) {
        findScene(id: $id) {
            id
            title
            details
            paths { sprite vtt }
            performers { id name gender }
            files { path duration width height frame_rate }
        }
    }
    """
    graphql_url = f"{STASH_URL.rstrip('/')}/graphql"
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if STASH_API_KEY:
        headers["ApiKey"] = STASH_API_KEY

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(graphql_url, json={"query": query, "variables": {"id": scene_id}}, headers=headers)
        resp.raise_for_status()
        # Stash may return file paths with non-UTF-8 bytes (e.g., Latin-1
        # accented characters). Decode tolerantly to avoid crashing on them.
        import json as _json
        body = _json.loads(resp.content.decode("utf-8", errors="replace"))

    if "errors" in body and body["errors"]:
        raise RuntimeError(f"Stash GraphQL errors: {body['errors']}")

    scene = body.get("data", {}).get("findScene")
    if not scene:
        raise RuntimeError(f"Scene {scene_id} not found in Stash")

    return scene


def _build_scene_context(scene_id: str, stash_scene: Optional[Dict[str, Any]], request: AnalyzeSemanticsRequest) -> SceneContext:
    """Build SceneContext from Stash data, with request parameters as overrides."""
    params = request.parameters

    # Start from Stash data (or empty)
    if stash_scene:
        paths = stash_scene.get("paths") or {}
        primary_file = (stash_scene.get("files") or [{}])[0]
        performers = stash_scene.get("performers") or []
        ctx = SceneContext(
            scene_id=scene_id,
            source=primary_file.get("path") or request.source,
            title=stash_scene.get("title"),
            sprite_image_url=paths.get("sprite"),
            sprite_vtt_url=paths.get("vtt"),
            details=stash_scene.get("details"),
            duration=primary_file.get("duration") or 0,
            frame_rate=primary_file.get("frame_rate"),
            width=primary_file.get("width"),
            height=primary_file.get("height"),
            performer_count=len(performers),
            performer_genders=[p.get("gender") for p in performers if p.get("gender")],
        )
    else:
        ctx = SceneContext(scene_id=scene_id, source=request.source)

    # Request parameters override Stash data
    if params.sprite_image_url is not None:
        ctx.sprite_image_url = params.sprite_image_url
    if params.sprite_vtt_url is not None:
        ctx.sprite_vtt_url = params.sprite_vtt_url
    if params.details is not None:
        ctx.details = params.details
    if request.source:
        ctx.source = request.source

    return ctx


async def _extract_frame_images(
    ctx: SceneContext,
    params,
) -> Tuple[List[Image.Image], List[float]]:
    """Extract frames from sprite sheets via internalized SpriteParser.

    Returns:
        Tuple of (images, timestamps).
    """
    if not ctx.sprite_vtt_url or not ctx.sprite_image_url:
        raise RuntimeError(
            "Sprite VTT/image URLs required. Ensure the scene has generated "
            "sprites in Stash (no frame-server fallback in standalone mode)."
        )

    job_id = f"sprites_{uuid.uuid4().hex[:8]}"
    try:
        frames_data = await sprite_parser.process_sprites(
            sprite_vtt_url=ctx.sprite_vtt_url,
            sprite_image_url=ctx.sprite_image_url,
            job_id=job_id,
        )

        if not frames_data:
            raise RuntimeError("No frames extracted from sprite sheet")

        images = []
        timestamps = []
        for idx, timestamp, file_path, w, h in frames_data:
            img = Image.open(file_path).convert("RGB")
            images.append(img)
            timestamps.append(timestamp)

        return images, timestamps
    finally:
        sprite_parser.cleanup_job(job_id)


def _resolve_operations(params) -> set:
    """Return the set of active operation names from SemanticsParameters."""
    ops = params.operations
    if not ops or SemanticsOperation.ALL in ops:
        return {"title", "summary", "tags"}
    return {op.value for op in ops}


async def _run_pipeline(job_id: str, request_payload: dict):
    """Worker callback: full tag classification pipeline for a single job.

    Called by SemanticsWorker after acquiring the job from the Redis queue.
    All models are pre-loaded and resident — no load/unload cycles.
    """
    global taxonomy_data, taxonomy_status

    # Deserialize the stored request
    request = AnalyzeSemanticsRequest.model_validate(request_payload)

    start_time = time.time()

    try:
        logger.info(f"Starting semantics analysis job {job_id} for scene {request.source_id}")
        params = request.parameters
        active_ops = _resolve_operations(params)

        # Initialize job metadata
        now = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        cache_params = {
            "model": params.model_variant,
            "min_confidence": params.min_confidence,
            "top_k": params.top_k_tags,
            "frames_per_scene": params.frames_per_scene,
            "hierarchical": params.use_hierarchical_decoding,
            "operations": sorted(active_ops),
        }
        cache_key = cache_manager.generate_cache_key(request.source, cache_params)

        metadata = {
            "job_id": job_id,
            "status": JobStatus.PROCESSING.value,
            "progress": 0.0,
            "stage": "initializing",
            "message": "Initializing pipeline",
            "created_at": now,
            "started_at": now,
            "source_id": request.source_id,
            "source": request.source,
            "cache_key": cache_key,
        }
        await cache_manager.cache_job_metadata(job_id, cache_key, metadata)

        # --- Step 0: Taxonomy ---
        active_taxonomy = taxonomy_data
        if request.custom_taxonomy:
            await cache_manager.update_job_status(job_id, JobStatus.PROCESSING.value, progress=0.02, stage="loading_taxonomy", message="Loading custom taxonomy")
            active_taxonomy = await _load_custom_taxonomy(request.custom_taxonomy)
            logger.info(f"Custom taxonomy loaded: {len(active_taxonomy.get('tags', []))} tags")

        if not active_taxonomy or not active_taxonomy.get("tags"):
            raise RuntimeError("No taxonomy available. Set STASH_URL or provide custom_taxonomy.")

        # Ensure classifier has this taxonomy loaded
        if tag_classifier and tag_classifier.is_loaded:
            existing = tag_classifier.taxonomy
            if existing is not active_taxonomy:
                existing_ids = {str(t["id"]) for t in (existing or {}).get("tags", [])}
                new_ids = {str(t["id"]) for t in active_taxonomy.get("tags", [])}
                if existing_ids != new_ids:
                    logger.info(f"Taxonomy changed ({len(existing_ids)} → {len(new_ids)} tags), reloading classifier")
                    tag_classifier.load_taxonomy(active_taxonomy)
        elif tag_classifier and tag_classifier.is_checkpoint_ready:
            tag_classifier.load_taxonomy(active_taxonomy)
        elif not tag_classifier or not tag_classifier.is_checkpoint_ready:
            raise RuntimeError("Tag classifier checkpoint not ready")

        # --- Step 1: Resolve scene context from Stash ---
        await cache_manager.update_job_status(job_id, JobStatus.PROCESSING.value, progress=0.03, stage="fetching_scene", message="Fetching scene data from Stash")
        stash_scene = None
        try:
            stash_scene = await _fetch_scene_from_stash(request.source_id)
            logger.info(f"Fetched scene {request.source_id} from Stash")
        except Exception as e:
            logger.warning(f"Could not fetch scene from Stash: {e}")

        ctx = _build_scene_context(request.source_id, stash_scene, request)
        logger.info(f"Scene context: source={ctx.source}, sprites={'yes' if ctx.sprite_vtt_url else 'no'}, promo={'yes' if ctx.has_promo else 'no'}, duration={ctx.duration:.0f}s")

        # --- Step 1b: Scene boundaries (stubbed in standalone mode) ---
        scene_boundaries = params.scene_boundaries
        if request.scenes_job_id:
            logger.warning("scenes_job_id provided but scene detection is not available in standalone mode")

        # --- Step 2: Extract frames via internalized sprite parser ---
        await cache_manager.update_job_status(job_id, JobStatus.PROCESSING.value, progress=0.05, stage="extracting_frames", message="Extracting frames from sprites")
        frame_images, frame_timestamps = await _extract_frame_images(ctx, params)
        num_frames = len(frame_images)
        logger.info(f"Extracted {num_frames} frames from sprites")

        # Pad or truncate to exactly 16 frames (classifier requirement)
        target_frames = 16
        if num_frames < target_frames:
            logger.warning(f"Only {num_frames} frames extracted, padding to {target_frames}")
            while len(frame_images) < target_frames:
                frame_images.append(frame_images[-1])
                frame_timestamps.append(frame_timestamps[-1] if frame_timestamps else 0.0)
        elif num_frames > target_frames:
            import numpy as np
            indices = np.linspace(0, num_frames - 1, target_frames, dtype=int)
            frame_images = [frame_images[i] for i in indices]
            frame_timestamps = [frame_timestamps[i] for i in indices]

        # --- Step 3: Generate captions (models pre-loaded, no lease needed) ---
        await cache_manager.update_job_status(job_id, JobStatus.PROCESSING.value, progress=0.15, stage="captioning", message="Generating frame captions with JoyCaption")

        raw_captions = await asyncio.to_thread(caption_generator.generate_captions, frame_images)
        fixed_captions = [CaptionGenerator.fix_caption(c, i) for i, c in enumerate(raw_captions)]
        logger.info(f"Generated {len(fixed_captions)} captions")

        # --- Step 4: Generate scene summary ---
        frame_caption_dicts = [{"frame_index": i, "timestamp": frame_timestamps[i], "caption": c} for i, c in enumerate(fixed_captions)]

        scene_summary = ""
        if "summary" in active_ops:
            await cache_manager.update_job_status(job_id, JobStatus.PROCESSING.value, progress=0.60, stage="summarizing", message="Generating scene summary")

            scene_summary = await asyncio.to_thread(
                summary_generator.generate_summary,
                frame_caption_dicts, ctx.promo_desc, ctx.duration,
                ctx.performer_count, ctx.performer_genders, ctx.resolution,
            )
            logger.info(f"Summary generated ({len(scene_summary)} chars)")
        else:
            logger.info("Skipping summary generation (not in operations)")

        # --- Step 5: Generate suggested title (reuses loaded Llama) ---
        suggested_title: Optional[str] = None
        if "title" in active_ops:
            await cache_manager.update_job_status(job_id, JobStatus.PROCESSING.value, progress=0.80, stage="titling", message="Generating scene title")

            try:
                suggested_title = await asyncio.to_thread(
                    title_generator.generate_title,
                    scene_source=request.source, scene_summary=scene_summary,
                    promo_desc=ctx.promo_desc, duration=ctx.duration,
                    performer_count=ctx.performer_count,
                    performer_genders=ctx.performer_genders,
                    resolution=ctx.resolution,
                )
            except Exception as e:
                logger.warning(f"Title generation failed: {e}. Continuing without suggested_title.", exc_info=True)

            if suggested_title:
                logger.info(f"Suggested title: {suggested_title!r}")
        else:
            logger.info("Skipping title generation (not in operations)")

        # --- Step 6: Run tag classifier (with vision if available) ---
        tags = []
        scene_embedding = None
        if "tags" in active_ops:
            await cache_manager.update_job_status(job_id, JobStatus.PROCESSING.value, progress=0.85, stage="classifying", message="Running tag classifier")

            prediction = await asyncio.to_thread(
                tag_classifier.predict,
                frame_captions=fixed_captions,
                summary=scene_summary,
                promo_desc=ctx.promo_desc,
                has_promo=ctx.has_promo,
                frame_images=frame_images,
                frame_timestamps=frame_timestamps,
                top_k=params.top_k_tags,
                min_score=params.min_confidence,
                use_hierarchical_decoding=params.use_hierarchical_decoding,
            )

            tags = prediction["tags"]
            logger.info(f"Classifier returned {len(tags)} tags")

            # Optional: scene embedding
            if params.generate_embeddings:
                try:
                    scene_embedding = tag_classifier.get_scene_embedding(fixed_captions, scene_summary, ctx.promo_desc, ctx.has_promo)
                except Exception as e:
                    logger.warning(f"Embedding generation failed: {e}")
        else:
            logger.info("Skipping tag classification (not in operations)")

        # --- Build results ---
        processing_time = time.time() - start_time

        classifier_tags = [
            ClassifierTag(
                tag_id=t["tag_id"],
                tag_name=t["tag_name"],
                score=round(t["score"], 4),
                path=t.get("path", ""),
                decode_type=t.get("decode_type", "direct"),
            )
            for t in tags
        ]

        frame_caption_results = [
            FrameCaptionResult(frame_index=i, timestamp=frame_timestamps[i], caption=c)
            for i, c in enumerate(fixed_captions)
        ]

        outcome = SemanticsOutcome(
            tags=classifier_tags if "tags" in active_ops else [],
            frame_captions=frame_caption_results,
            scene_summary=scene_summary if "summary" in active_ops else None,
            suggested_title=suggested_title if "title" in active_ops else None,
            scene_embedding=scene_embedding if "tags" in active_ops else None,
        )

        tag_name_to_id = {str(t.get("name", "")): str(t["id"]) for t in active_taxonomy.get("tags", []) if t.get("name")}

        result_metadata = SemanticsMetadata(
            source=ctx.source,
            source_id=request.source_id,
            total_frames_extracted=num_frames,
            frames_captioned=len(fixed_captions),
            classifier_model=params.model_variant,
            processing_time_seconds=round(processing_time, 2),
            device=CLASSIFIER_DEVICE,
            taxonomy_size=len(active_taxonomy.get("tags", [])),
            has_promo=ctx.has_promo,
            sprite_image_url=ctx.sprite_image_url,
            sprite_vtt_url=ctx.sprite_vtt_url,
            tag_name_to_id=tag_name_to_id,
            scene=SceneMetadata(
                title=ctx.title,
                duration=ctx.duration if ctx.duration else None,
                resolution=ctx.resolution,
                frame_rate=ctx.frame_rate,
                performer_count=ctx.performer_count,
                performer_genders=ctx.performer_genders,
            ),
        )

        results = {
            "job_id": job_id,
            "source_id": request.source_id,
            "status": JobStatus.COMPLETED.value,
            "semantics": outcome.model_dump(),
            "metadata": result_metadata.model_dump(),
        }

        await cache_manager.cache_job_results(job_id, cache_key, results)
        await cache_manager.update_job_status(job_id, JobStatus.COMPLETED.value, progress=1.0, stage="completed", message=f"Analysis complete in {processing_time:.1f}s ({len(tags)} tags)")

        logger.info(f"Job {job_id} completed in {processing_time:.1f}s — {len(tags)} tags predicted")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        try:
            await cache_manager.update_job_status(job_id, JobStatus.FAILED.value, progress=0.0, stage="failed", message=f"Job failed: {str(e)[:200]}", error=str(e))
        except Exception:
            pass

    finally:
        gc.collect()


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.post("/semantics/analyze", response_model=AnalyzeSemanticsResponse, status_code=202)
async def analyze_semantics(request: AnalyzeSemanticsRequest):
    """Submit tag classification analysis job to the Redis-backed queue."""
    try:
        if not job_queue:
            raise HTTPException(status_code=503, detail="Job queue not initialized")

        # Check cache
        cache_params = {
            "model": request.parameters.model_variant,
            "min_confidence": request.parameters.min_confidence,
            "top_k": request.parameters.top_k_tags,
            "frames_per_scene": request.parameters.frames_per_scene,
            "hierarchical": request.parameters.use_hierarchical_decoding,
            "operations": sorted(_resolve_operations(request.parameters)),
        }
        cache_key = cache_manager.generate_cache_key(request.source, cache_params)
        cached_job_id = await cache_manager.get_cached_job_id(cache_key)

        if cached_job_id:
            logger.info(f"Cache hit for {request.source}: {cached_job_id}")
            if request.job_id and request.job_id != cached_job_id:
                await cache_manager.create_job_alias(request.job_id, cached_job_id)
            return AnalyzeSemanticsResponse(
                job_id=request.job_id or cached_job_id,
                status=JobStatus.COMPLETED,
                message="Results retrieved from cache",
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
                cache_hit=True,
            )

        job_id = request.job_id or str(uuid.uuid4())

        # Store the full request payload in Redis
        await job_queue.store_request(job_id, request.model_dump())

        # Push to pending queue
        pending_count = await job_queue.enqueue(job_id)

        logger.info(f"Job {job_id} enqueued for scene {request.source_id} (pending={pending_count})")
        return AnalyzeSemanticsResponse(
            job_id=job_id,
            status=JobStatus.QUEUED,
            message=f"Tag classification job queued ({pending_count} ahead in queue)",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            cache_hit=False,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/semantics/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get job status."""
    try:
        metadata = await cache_manager.get_job_metadata(job_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        return JobStatusResponse(
            job_id=job_id,
            status=JobStatus(metadata["status"]),
            progress=metadata.get("progress", 0.0),
            stage=metadata.get("stage"),
            message=metadata.get("message"),
            created_at=metadata["created_at"],
            started_at=metadata.get("started_at"),
            completed_at=metadata.get("completed_at"),
            error=metadata.get("error"),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/semantics/jobs/{job_id}/results", response_model=SemanticsResults)
async def get_job_results(job_id: str):
    """Get job results."""
    try:
        metadata = await cache_manager.get_job_metadata(job_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        if metadata["status"] != JobStatus.COMPLETED.value:
            raise HTTPException(status_code=409, detail=f"Job not completed (status: {metadata['status']})")

        results = await cache_manager.get_job_results(job_id)
        if not results:
            raise HTTPException(status_code=404, detail=f"Results not found for job: {job_id}")

        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job results: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/semantics/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    classifier_loaded = tag_classifier is not None and tag_classifier.is_loaded
    queue_stats = None
    if job_queue:
        try:
            queue_stats = await job_queue.queue_stats()
        except Exception as e:
            logger.warning(f"Failed to fetch queue stats: {e}")
    return HealthResponse(
        status="healthy" if classifier_loaded and taxonomy_status.loaded else "degraded",
        classifier_model=CLASSIFIER_MODEL if classifier_loaded else None,
        classifier_loaded=classifier_loaded,
        device=CLASSIFIER_DEVICE,
        taxonomy=taxonomy_status,
        default_min_confidence=float(os.getenv("SEMANTICS_MIN_CONFIDENCE", "0.75")),
        queue=queue_stats,
    )


# ---------------------------------------------------------------------------
# Stubbed endpoints (standalone mode)
# ---------------------------------------------------------------------------

@app.post("/frames/extract", status_code=501)
async def extract_frames_stub():
    """Stub: frame extraction not available in standalone mode."""
    return {"status": "failed", "message": "Frame extraction not available in standalone mode. Use /semantics/analyze — frames are extracted internally from sprites."}


@app.post("/scenes/detect", status_code=501)
async def detect_scenes_stub():
    """Stub: scene detection not available in standalone mode."""
    return {"status": "failed", "message": "Scene detection not available in standalone mode."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5004)
