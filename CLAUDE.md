# Stash Auto Vision Semantics — Standalone Cloud GPU Edition

**A standalone, single-container semantic analysis service extracted from [stash-auto-vision](https://github.com/smegmarip/stash-auto-vision) for batch processing on rented cloud GPUs (Vast.ai H100 80GB).**

---

## Purpose

The parent project `stash-auto-vision` runs a 10+ service Docker Compose stack on a server with an NVIDIA RTX A4000 (16GB VRAM). The semantics pipeline — tag classification, frame captioning, scene summarization — is the slowest service because 16GB VRAM forces sequential model loading/unloading (JoyCaption ~8GB, Llama 3.1 8B ~10GB, classifier ~1.4GB). Each job spends more time shuffling models in and out of VRAM than doing actual inference.

This project solves that by extracting the semantics pipeline into a standalone container that runs on a rented GPU with 80GB VRAM (e.g., H100 or A100 80GB) via Vast.ai. With 80GB, all three models stay resident simultaneously — no load/unload cycles, no quantization, no GPU lease management. Designed for batch processing large scene backlogs.

### What This Is

- A single Docker container that runs the semantics analysis pipeline
- Designed for Vast.ai's dockerized VM model (one container, one GPU)
- Connects back to a Stash instance via a tunnel (e.g., ngrok or VS Code port forward)
- Processes jobs submitted via its `/semantics/analyze` endpoint
- Intended for batch processing, not permanent deployment

### What This Is NOT

- Not a replacement for the full `stash-auto-vision` stack
- Not designed for multi-service orchestration
- Not intended for the Vision API rollup endpoint
- Not concerned with faces, scenes, or objects services

---

## Parent Project Reference

**Source repo:** [stash-auto-vision](https://github.com/smegmarip/stash-auto-vision)

The semantics service code lives at `stash-auto-vision/semantics-service/`. This project forks and simplifies that code. Refer to the parent project's documentation for the original architecture:

- `CLAUDE.md` — Full project specification
- `docs/SEMANTICS_SERVICE.md` — Semantics pipeline documentation
- `docs/RESOURCE_MANAGER.md` — GPU lease system (being removed)
- `docs/FRAME_SERVER.md` — Frame extraction (being internalized)

---

## Architecture: Parent vs. Standalone

### Parent (stash-auto-vision) — 16GB A4000

```
Request → Vision API → Semantics Service → Frame Server (sprites)
                                         → Resource Manager (GPU lease)
                                         → Redis (job queue + cache)
                                         → Stash (metadata + taxonomy)
                                         → JoyCaption (load → caption → unload)
                                         → Llama 3.1 8B (load → summary → unload)
                                         → Tag Classifier (resident)
```

- 10+ Docker Compose services
- Sequential model loading/unloading (VRAM constraint)
- 4-bit NF4 quantization on JoyCaption (bitsandbytes, leaks ~5GB on unload)
- int8 quantization on Llama (TorchAO)
- GPU lease acquisition/heartbeat/release on every job
- ModelManager enforces mutual exclusion between JoyCaption and Llama
- Container restart protocol to recover leaked VRAM
- Single-job mutex via Redis Lua scripts

### Standalone (this project) — 80GB H100

```
Request → Semantics Service → Stash (metadata + taxonomy, via tunnel)
                             → Sprite parsing (internalized)
                             → JoyCaption (resident, full precision)
                             → Llama 3.1 8B (resident, full precision)
                             → Tag Classifier (resident)
                             → Redis (embedded, in-container)
```

- Single Docker container
- All models loaded at startup, stay resident
- Full precision (bfloat16) — no quantization needed
- No GPU lease management
- No model load/unload cycles
- No container restart protocol
- Redis runs as an embedded process in the same container (or replaced with in-process state)

---

## Dependency Audit

Every external touchpoint of the parent semantics service, classified for this standalone deployment.

### Essential — Must Have

| Dependency                | Why                                                                | Notes                                                                                                                              |
| ------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Stash GraphQL API**     | Taxonomy loading (startup), scene metadata + sprite URLs (per job) | Accessed via tunnel (ngrok or VS Code port forward). Requires `STASH_URL`, `STASH_API_KEY`, `SEMANTICS_TAG_ID`.                    |
| **JoyCaption beta-one**   | Per-frame captioning VLM                                           | HuggingFace: `fancyfeast/llama-joycaption-beta-one-hf-llava`. On H100: load once at startup, bfloat16, ~16GB VRAM, stays resident. |
| **Llama 3.1 8B Instruct** | Scene summary + suggested title generation                         | HuggingFace: `RedHatAI/Llama-3.1-8B-Instruct`. On H100: load once at startup, bfloat16, ~16GB VRAM, stays resident.                |
| **Tag Classifier**        | Trained bi-encoder for tag prediction                              | HuggingFace: `smegmarip/tag-classifier`. ~1.4-3.8GB VRAM depending on variant. Loaded at startup, stays resident.                  |
| **Sprite sheet parsing**  | Extract frames from Stash sprite grids                             | Currently handled by frame-server. Must be internalized (see below).                                                               |
| **CUDA 12.x runtime**     | GPU inference                                                      | Base image: `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`                                                                         |

### Beneficial — Keep If Easy

| Dependency                | Why                                            | Recommendation                                                                                                                                                                                                                                                                                        |
| ------------------------- | ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Redis**                 | Job queue, cache, status tracking              | **Embed as a process in the container** (install `redis-server`, start in entrypoint). The job queue uses Lua scripts for atomic operations; replacing Redis with in-process state is possible but nontrivial. Embedding Redis is simpler and preserves restart resilience for a multi-day batch run. |
| **Content-based caching** | Skip already-processed scenes on batch restart | Essential for a multi-day run. If the container restarts mid-batch, cache hits let you resume without reprocessing. Keep the existing `CacheManager` and cache key logic intact.                                                                                                                      |
| **Job status tracking**   | Progress monitoring during batch run           | Keep the `/semantics/jobs/{job_id}/status` endpoint so the tagging plugin (or a script) can poll progress.                                                                                                                                                                                            |
| **Health endpoint**       | Verify service is up                           | Keep `/semantics/health` as-is.                                                                                                                                                                                                                                                                       |

### Superfluous — Remove Entirely

| Dependency                                     | Why It Exists                                               | Why It's Unnecessary                                                                                                      |
| ---------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **Resource Manager** (`gpu_client.py`)         | GPU lease brokering between competing services on 16GB card | Sole owner of 80GB GPU. Zero contention. No lease/heartbeat/eviction needed.                                              |
| **ModelManager exclusive model logic**         | Prevents JoyCaption + Llama coexisting in 16GB VRAM         | 80GB holds both simultaneously. No mutual exclusion needed.                                                               |
| **bitsandbytes NF4 quantization** (JoyCaption) | Compresses JoyCaption to ~7GB to fit 16GB card              | 80GB VRAM. Run bfloat16 for better caption quality and no VRAM leak.                                                      |
| **TorchAO int8 quantization** (Llama)          | Compresses Llama to ~10GB to fit 16GB card                  | 80GB VRAM. Run bfloat16 for better summary quality.                                                                       |
| **Container restart protocol**                 | Recovers ~5GB VRAM leaked by bitsandbytes NF4 after unload  | No quantization = no leak = no restart needed.                                                                            |
| **Model idle timeout / cleanup loop**          | Unloads idle models to free VRAM for other services         | All models stay resident. No unload needed.                                                                               |
| **Frame Server** (external service)            | Sprite parsing, video frame extraction, enhancement         | Internalize sprite parsing (~100 lines). No video frame extraction needed (sprites are the default and preferred source). |
| **Scenes Service**                             | Scene boundary detection                                    | Not needed. Sprite-sheet mode (default) doesn't use scene boundaries.                                                     |
| **Vision API orchestrator**                    | Multi-service coordination                                  | Direct `/semantics/analyze` calls only.                                                                                   |
| **Faces Service**                              | Face recognition                                            | Not involved in semantics.                                                                                                |
| **Objects Service**                            | Stub                                                        | Not implemented.                                                                                                          |
| **Schema Service**                             | Swagger UI aggregation                                      | Single service, use FastAPI's built-in `/docs`.                                                                           |
| **Jobs Viewer**                                | React monitoring UI                                         | Not needed for batch processing.                                                                                          |
| **Dependency Checker**                         | Startup orchestration for multi-service stack               | Single container, no dependencies to check.                                                                               |
| **GPU lease eviction callbacks**               | Resource-manager evicts services to free VRAM               | No resource-manager, no eviction.                                                                                         |

---

## Refactoring Specification

### 1. Internalize Sprite Sheet Parsing

The frame-server's sprite parser is ~150 lines in `frame-server/app/sprite_parser.py`. It:

1. Downloads the VTT file from a Stash URL via HTTP
2. Parses timestamp → coordinate mappings with regex (`#xywh=x,y,w,h`)
3. Downloads the sprite grid JPEG from a Stash URL
4. Slices tiles from the grid using OpenCV (`sprite_img[y:y+h, x:x+w]`)
5. Saves tiles as individual JPEGs to disk

**What to copy:** The `SpriteParser` class from `frame-server/app/sprite_parser.py` (lines 19-262). It depends only on `cv2`, `httpx`, `re`, and standard library — all already available in the semantics service.

**What to change:** Replace the `FrameServerClient` calls in the pipeline (lines 387-457 of `semantics-service/app/main.py`) with direct calls to the internalized sprite parser. The pipeline expects PIL Image objects; after slicing tiles with OpenCV, convert BGR→RGB and wrap in `PIL.Image`.

**What to drop:** The entire `frame_client.py` module, the `FRAME_SERVER_URL` config, and the async job polling loop for frame extraction.

### 2. Remove GPU Lease Management

**Files to remove/gut:**

- `gpu_client.py` — Delete entirely. All references to `gpu_client.lease()`, `mark_busy()`, `mark_idle()`, `on_evict()`, heartbeat loops.
- Remove GPU lease acquisition from `_run_pipeline()` (Stage 3 in parent). Models are pre-loaded; no lease needed.
- Remove `_evict_gpu_lease()` callback and `_check_restart_needed()` logic.
- Remove the perpetual lease request for the classifier at startup.
- Remove `RESOURCE_MANAGER_URL` from config.

### 3. Simplify Model Lifecycle

**Goal:** Load all three models once at startup. Never unload.

**JoyCaption (`caption_generator.py`):**

- Remove `BitsAndBytesConfig` quantization. Load with `torch_dtype=torch.bfloat16`.
- Remove the `load()` / `unload()` per-job cycle. Load in `__init__` or at startup.
- Remove `gc.collect()` + `torch.cuda.empty_cache()` cleanup (no unload = no cleanup needed).
- Keep the `generate_captions()` method intact.

**Llama 3.1 8B (`llama_runtime.py`):**

- Remove TorchAO int8 quantization. Load with `torch_dtype=torch.bfloat16`.
- Remove the `load()` / `unload()` cycle. Load at startup, keep resident.
- Keep `SummaryGenerator` and `TitleGenerator` wrappers intact — they share the runtime.

**Tag Classifier (`classifier.py`):**

- Remove perpetual GPU lease logic. Just load the model at startup.
- Keep `load_model()` and `load_taxonomy()` as-is.
- Keep `predict()` method intact.

**ModelManager (`model_manager.py`):**

- Simplify dramatically. Remove:
  - Exclusive model unload-before-load logic
  - Idle timeout cleanup loop (`_cleanup_loop`)
  - `busy_count` tracking (no contention)
- Keep the `using()` context manager as a lightweight wrapper (thread safety for `asyncio.to_thread` calls).
- Or replace with a simple class that just tracks which models are loaded.

### 4. Remove Scenes Service Client

- Delete `ScenesServerClient` usage and `SCENES_SERVER_URL` config.
- Remove Stage 1b (scene boundary fetching from scenes-service) from `_run_pipeline()`.
- Keep `scene_boundaries` as an optional request parameter (can be passed inline).

### 5. Keep Redis (Embedded)

- Install `redis-server` in the Docker image.
- Start it in the container entrypoint before the FastAPI app.
- Keep `CacheManager` and `JobQueue` as-is — they're well-tested and the Lua scripts provide correctness guarantees even in edge cases (container restart mid-batch).
- Keep `REDIS_URL` defaulting to `redis://localhost:6379/0`.
- **Critical for batch resilience:** If the container restarts during a multi-day run, the Redis AOF or RDB snapshot lets you resume without reprocessing completed scenes.

### 6. Keep the Job Queue (Single Worker)

- Keep `JobQueue` with its Lua scripts — it handles edge cases around job recovery after crashes.
- Keep `SemanticsWorker` single-job processing — even on H100, one job at a time is fine because inference is fast and the bottleneck shifts to sprite download latency.
- Future optimization: parallel jobs could be added later, but not needed for initial deployment.

### 7. Simplify Startup Sequence

**Current startup (parent):**

1. Connect Redis
2. Init JobQueue
3. Init FrameServerClient ← remove
4. Init GPUClient, announce startup ← remove
5. Init CaptionGenerator (lazy load)
6. Init LlamaRuntime (lazy load)
7. Init SummaryGenerator, TitleGenerator
8. Request classifier perpetual GPU lease ← remove
9. Load classifier model
10. Init ModelManager, register exclusives ← simplify
11. Background taxonomy load from Stash
12. Start worker

**Standalone startup:**

1. Connect Redis (localhost)
2. Init JobQueue
3. Init SpriteParser (internalized)
4. Load JoyCaption (bfloat16, ~16GB) — **blocking, at startup**
5. Load Llama 3.1 8B (bfloat16, ~16GB) — **blocking, at startup**
6. Load Tag Classifier (~1.4-3.8GB) — **blocking, at startup**
7. Background taxonomy load from Stash (via tunnel)
8. Start worker
9. Ready to accept jobs (~48GB VRAM used, ~32GB headroom)

### 8. Batch Submission Interface

The existing `/semantics/analyze` endpoint works as-is for the `stash-auto-vision-tagging` plugin. The plugin submits one job at a time and polls for completion. The plugin's batch mode can drive submissions for large backlogs.

Optionally, add a `/semantics/batch` endpoint that accepts an array of `source_id` values and enqueues them all at once — saves HTTP round-trip overhead.

---

## Container Specification

### Base Image

```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
```

### Key Packages

- Python 3.11+
- PyTorch 2.x with CUDA 12.x
- transformers, sentence-transformers, huggingface_hub
- httpx (async HTTP for Stash API)
- redis-server + redis-py (embedded)
- opencv-python-headless (sprite tile extraction)
- Pillow (image handling)
- FastAPI + uvicorn
- **No bitsandbytes** (no quantization needed)
- **No torchao** (no quantization needed)

### Volumes

| Mount                                          | Purpose                                              |
| ---------------------------------------------- | ---------------------------------------------------- |
| HuggingFace cache (`/root/.cache/huggingface`) | Model weights (persistent across container restarts) |
| Redis data (`/data/redis`)                     | Job state persistence for batch restart resilience   |
| Sprite temp (`/tmp/sprites`)                   | Temporary sprite tile storage                        |

### Environment Variables

| Variable                   | Required    | Default                          | Description                                                      |
| -------------------------- | ----------- | -------------------------------- | ---------------------------------------------------------------- |
| `STASH_URL`                | **Yes**     | —                                | Stash instance URL (via tunnel, e.g., `https://abc123.ngrok.io`) |
| `STASH_API_KEY`            | **Yes**     | —                                | Stash API key for authentication                                 |
| `SEMANTICS_TAG_ID`         | Recommended | —                                | Root tag ID for taxonomy subtree                                 |
| `CLASSIFIER_MODEL`         | No          | `vision`                         | Classifier variant (`text-only` or `vision`)                     |
| `CLASSIFIER_DEVICE`        | No          | `cuda`                           | Should always be `cuda` on H100                                  |
| `SEMANTICS_LLM_MODEL`      | No          | `RedHatAI/Llama-3.1-8B-Instruct` | Llama model for summary/title                                    |
| `SEMANTICS_LLM_DEVICE`     | No          | `cuda`                           | Should always be `cuda` on H100                                  |
| `SEMANTICS_MIN_CONFIDENCE` | No          | `0.75`                           | Tag score threshold                                              |
| `REDIS_URL`                | No          | `redis://localhost:6379/0`       | Embedded Redis                                                   |
| `HF_TOKEN`                 | If needed   | —                                | HuggingFace token for gated models                               |
| `LOG_LEVEL`                | No          | `INFO`                           | Logging level                                                    |
| `CACHE_TTL`                | No          | `31536000`                       | Cache TTL (1 year — keep results across restarts)                |

**Removed variables** (vs. parent): `FRAME_SERVER_URL`, `SCENES_SERVER_URL`, `RESOURCE_MANAGER_URL`, `SEMANTICS_MODEL_IDLE_TIMEOUT`, `SEMANTICS_JOB_LOCK_TTL` (can keep with sensible defaults).

### Entrypoint

```bash
#!/bin/bash
# Start embedded Redis
redis-server --daemonize yes --dir /data/redis --appendonly yes

# Start the semantics service
exec uvicorn app.main:app --host 0.0.0.0 --port 5004 --workers 1
```

### Vast.ai Configuration

- **Image:** Custom image pushed to Docker Hub or built from this repo
- **GPU:** H100 80GB (or A100 80GB as cheaper alternative)
- **Disk:** 100GB+ (model weights ~50GB for all three models at full precision)
- **Expose port:** 5004 (Vast.ai maps this automatically)
- **On-start command:** The entrypoint above

---

## VRAM Budget (H100 80GB, bfloat16)

| Model                            | Estimated VRAM | Lifecycle                   |
| -------------------------------- | -------------- | --------------------------- |
| JoyCaption beta-one (bfloat16)   | ~16GB          | Loaded at startup, resident |
| Llama 3.1 8B Instruct (bfloat16) | ~16GB          | Loaded at startup, resident |
| Tag Classifier (text-only)       | ~1.4GB         | Loaded at startup, resident |
| Tag Classifier (vision)          | ~3.8GB         | Loaded at startup, resident |
| PyTorch CUDA context             | ~1-2GB         | Runtime overhead            |
| KV cache (inference)             | ~2-4GB         | Transient during generation |
| **Total (text-only classifier)** | **~36-40GB**   | **50% of 80GB**             |
| **Total (vision classifier)**    | **~39-42GB**   | **52% of 80GB**             |

~40GB headroom for batch sizes, KV cache growth, or future model upgrades.

---

## Observed Performance

### Per-Job Breakdown (H100 80GB, measured)

| Stage                  | A4000 (16GB, quantized) | H100 (80GB, bfloat16) | Notes                                             |
| ---------------------- | ----------------------- | --------------------- | ------------------------------------------------- |
| Model loading          | 30-60s                  | 0s                    | Models pre-loaded, resident                       |
| Sprite download        | 1-3s                    | ~1s                   | Network-bound (tunnel latency)                    |
| JoyCaption (16 frames) | N/A (included in total) | **~73s**              | Sequential autoregressive decode, ~4.5s per frame |
| Llama summary          | N/A (included in total) | ~9s                   | bfloat16, no quantization overhead                |
| Llama title            | N/A (included in total) | ~0.2s                 | Short generation                                  |
| Tag classification     | <1s                     | ~0.1s                 | Includes SiGLIP vision encoding                   |
| **Total per job**      | **~6 min**              | **~90s**              | **~4x improvement**                               |

JoyCaption accounts for ~80% of total job time. This is inherent to sequential single-frame autoregressive generation and cannot be improved without changing the captioning approach (fewer frames, shorter captions, or a different model).

### Batch Cost Estimate

| Batch size    | Time       | Cost (H100 @ $1.69/hr) |
| ------------- | ---------- | ---------------------- |
| 1,000 scenes  | ~25 hours  | ~$42                   |
| 5,000 scenes  | ~125 hours | ~$211                  |
| 10,000 scenes | ~250 hours | ~$423                  |

---

## Implementation Plan

### Phase 1: Fork and Strip (Day 1)

1. Copy `semantics-service/` from parent project
2. Copy `SpriteParser` from `frame-server/app/sprite_parser.py`
3. Remove `gpu_client.py`, `frame_client.py`, scenes client references
4. Remove quantization configs (bitsandbytes, TorchAO)
5. Remove ModelManager exclusive model logic
6. Remove container restart protocol
7. Simplify startup to load all models eagerly

### Phase 2: Internalize Sprites (Day 1)

1. Integrate `SpriteParser` into the service
2. Replace `FrameServerClient` calls with direct sprite parsing
3. Add async HTTP download for VTT + sprite grid from Stash
4. Test sprite extraction with sample VTT/grid URLs

### Phase 3: Containerize (Day 1-2)

1. Write Dockerfile (CUDA base + Python deps + embedded Redis)
2. Write entrypoint script (start Redis, start uvicorn)
3. Test locally with `docker run --gpus all`
4. Push image to Docker Hub

### Phase 4: Deploy and Test (Day 2)

1. Provision GPU instance on Vast.ai (H100 or A100 80GB)
2. Start tunnel to Stash instance
3. Configure `STASH_URL` to tunnel endpoint
4. Submit test job, verify end-to-end
5. Begin batch processing

### Phase 5: Batch Execution

1. Use `stash-auto-vision-tagging` plugin's batch mode, or
2. Write a simple script to submit source_ids sequentially
3. Monitor progress via `/semantics/jobs/{job_id}/status`
4. Verify results are written back to Stash via the plugin

---

## Files to Copy from Parent

```
From: stash-auto-vision/semantics-service/
  app/
    main.py                    # Entry point, pipeline, endpoints (heavy refactoring)
    models.py                  # Pydantic models (keep as-is)
    cache_manager.py           # Redis cache (keep as-is)
    job_queue.py               # Redis job queue with Lua scripts (keep as-is)
    worker.py                  # Job worker loop (keep as-is)
    caption_generator.py       # JoyCaption VLM (remove quantization)
    llama_runtime.py           # Llama 3.1 8B (remove quantization)
    summary_generator.py       # Summary prompt/generation (keep as-is)
    title_generator.py         # Title prompt/generation (keep as-is)
    classifier.py              # Tag classifier (remove GPU lease logic)
    model_manager.py           # Model lifecycle (simplify dramatically)
    taxonomy_builder.py        # Stash taxonomy loading (keep as-is)
    hierarchical_decoder.py    # Taxonomy-aware post-processing (keep as-is)
  train/
    model.py                   # MultiViewClassifier architecture (keep as-is)
    dataset.py                 # Dataset class for classifier (keep as-is)
  Dockerfile                   # Rewrite for standalone + embedded Redis
  requirements.txt             # Trim unused deps (bitsandbytes, torchao)

From: stash-auto-vision/frame-server/
  app/sprite_parser.py         # SpriteParser class (copy, integrate)

NOT copied:
  app/gpu_client.py            # GPU lease management (removed)
  app/frame_client.py          # Frame server HTTP client (replaced by internalized sprites)
```

---

## Key Design Decisions

### Why Embed Redis Instead of Replacing It?

The `JobQueue` uses three Lua scripts for atomic job acquisition, release, and orphan recovery. These are battle-tested and handle edge cases (container restart mid-job, lock expiry, worker crash recovery). Reimplementing this with `asyncio.Queue` + `dict` saves ~6MB of RAM but introduces subtle concurrency bugs. Embedding Redis in the container is trivial (`apt install redis-server`) and preserves correctness.

Additionally, Redis AOF persistence means a container restart during a multi-day batch run doesn't lose job state. Completed jobs are cached and skipped on re-submission.

### Why Not Run Multiple Jobs in Parallel?

Even with 80GB VRAM, JoyCaption and Llama share the GPU's compute cores. Running two jobs in parallel would roughly halve throughput per job (GPU compute contention) while doubling VRAM usage. Sequential processing with all models resident is the sweet spot. If sprite download latency becomes the bottleneck, a prefetch queue could overlap network I/O with GPU inference.

### Why bfloat16 Instead of float32?

H100 natively supports bfloat16 with dedicated tensor cores — bfloat16 is actually faster than float32 on H100 while providing equivalent dynamic range. float16 risks overflow on some operations; bfloat16 is the standard choice for H100 inference.

### Why Not Use an A100 Instead?

A100 80GB would also work. On Vast.ai the A100 80GB is priced similarly to the H100 (~$1.54/hr vs ~$1.69/hr), so the H100 is preferable — it finishes sooner at nearly the same hourly rate. The A100 40GB ($0.79/hr) cannot keep all models resident without reintroducing model swapping.

---

## Compatibility

### stash-auto-vision-tagging Plugin

The plugin submits jobs to `/semantics/analyze` and polls `/semantics/jobs/{job_id}/status`. This standalone service preserves both endpoints with identical request/response schemas. The plugin works unchanged — just point it at the Vast.ai instance URL instead of the local semantics service.

### Results Format

Identical to the parent project's semantics output: `tags[]`, `frame_captions[]`, `scene_summary`, `suggested_title`, `scene_embedding`. The plugin writes these back to Stash via GraphQL mutations.

---

**Status:** Implemented and Deployed
**Parent Project:** [stash-auto-vision](https://github.com/smegmarip/stash-auto-vision)
**Last Updated:** 2026-04-21
