# Stash Auto Vision Semantics

A standalone semantic analysis service extracted from [stash-auto-vision](https://github.com/smegmarip/stash-auto-vision) for batch processing on rented cloud GPUs. Runs the full semantics pipeline — frame captioning, scene summarization, tag classification — in a single Docker container with all models resident in VRAM.

Designed for [Vast.ai](https://vast.ai/) deployment on 80GB GPUs (H100 or A100 80GB). Connects to a [Stash](https://github.com/stashapp/stash) instance via tunnel for scene metadata, sprites, and taxonomy.

## Features

- **All models resident** -- JoyCaption, Llama 3.1 8B, and a trained multi-view bi-encoder tag classifier stay loaded in VRAM. No model swapping, no quantization, no load/unload cycles.
- **Vision-enhanced classification** -- the tag classifier uses SiGLIP visual grounding (sprite frame images processed alongside text) for improved tag predictions. Enabled by default.
- **Internalized sprite parsing** -- downloads and parses Stash sprite sheets directly. No external frame-server dependency.
- **Embedded Redis** -- job queue, result cache, and status tracking run in-container. Redis AOF persistence enables resume-on-restart for long batch runs.
- **API-compatible** -- preserves the same `/semantics/analyze`, `/semantics/jobs/{id}/status`, and `/semantics/jobs/{id}/results` endpoints as the parent project. The [stash-auto-vision-tagging](https://github.com/smegmarip/stash-auto-vision-tagging) plugin works unchanged.
- **Content-based caching** -- completed scenes are cached by content hash. Container restarts skip already-processed scenes.

## Architecture

```
Stash (via tunnel) ──[GraphQL + sprite URLs]──> Semantics Service
                                                  ├── SpriteParser (internalized)
                                                  │     └── Download VTT + sprite grid, extract tiles
                                                  ├── JoyCaption beta-one (bfloat16, ~16GB VRAM)
                                                  │     └── 16 frames → 16 captions
                                                  ├── Llama 3.1 8B Instruct (bfloat16, ~16GB VRAM)
                                                  │     └── Captions → scene summary + title
                                                  ├── Tag Classifier + SiGLIP (vision, ~3.8GB VRAM)
                                                  │     └── Captions + summary + frame images → tags
                                                  └── Redis (embedded)
                                                        └── Job queue + result cache
```

### Parent vs. Standalone

The parent project runs 10+ Docker Compose services on a 16GB GPU, requiring sequential model loading/unloading with quantization. This standalone edition runs everything in one container on an 80GB GPU:

| Aspect         | Parent (A4000 16GB)            | Standalone (H100 80GB) |
| -------------- | ------------------------------ | ---------------------- |
| Models         | Sequential load/unload         | All resident           |
| Quantization   | NF4 (JoyCaption), int8 (Llama) | None (bfloat16)        |
| Sprite parsing | External frame-server          | Internalized           |
| GPU management | Resource manager + leases      | None needed            |
| Per-job time   | ~6 minutes                     | ~90 seconds            |
| Container      | Docker Compose (10+ services)  | Single container       |

## Prerequisites

1. A **Stash** instance accessible via a public tunnel (ngrok, VS Code port forward, or similar).
2. A **Vast.ai** account with a GPU instance (H100 80GB or A100 80GB recommended).
3. **Docker** for building the container image.
4. A **Docker Hub** account (or other container registry) to host the image.

## Installation

### 1. Build and push the Docker image

```sh
git clone https://github.com/smegmarip/stash-auto-vision-semantics.git
cd stash-auto-vision-semantics
docker buildx build --platform linux/amd64 -t yourusername/stash-semantics:latest --push .
```

### 2. Create a Vast.ai template

Go to [cloud.vast.ai/templates](https://cloud.vast.ai/templates/) and create a new template:

| Field              | Value                                 |
| ------------------ | ------------------------------------- |
| **Template Name**  | Stash Semantics Batch                 |
| **Image Path:Tag** | `yourusername/stash-semantics:latest` |
| **Launch Mode**    | Docker ENTRYPOINT                     |
| **Ports**          | `5004` TCP                            |
| **Disk Space**     | 100 GB                                |
| **Visibility**     | Private                               |

Add environment variables:

| Key                | Value                                                  |
| ------------------ | ------------------------------------------------------ |
| `STASH_URL`        | Your tunnel URL (e.g., `https://your-tunnel.ngrok.io`) |
| `STASH_API_KEY`    | Your Stash API key                                     |
| `SEMANTICS_TAG_ID` | Root tag ID for taxonomy subtree                       |
| `HF_TOKEN`         | HuggingFace token (if needed for gated models)         |

Set extra filters to target 80GB GPUs:

```
cuda_vers >= 12.4 gpu_ram >= 80000 verified=true
```

### 3. Start the instance

Rent an H100 or A100 80GB instance using your template. The container will:

1. Start embedded Redis
2. Load JoyCaption (~16GB, ~30s)
3. Load Llama 3.1 8B (~16GB, ~30s)
4. Load Tag Classifier with SiGLIP (~3.8GB, ~10s)
5. Fetch taxonomy from Stash
6. Begin accepting jobs on port 5004

First startup downloads model weights from HuggingFace (~50GB total). Subsequent starts use the cached weights from the persistent volume.

### 4. Submit jobs

Point the [stash-auto-vision-tagging](https://github.com/smegmarip/stash-auto-vision-tagging) plugin at the Vast.ai instance URL, or submit jobs directly:

```sh
curl -X POST https://<vast-instance-url>/semantics/analyze \
  -H "Content-Type: application/json" \
  -d '{"source_id": "123"}'
```

Poll for results:

```sh
curl https://<vast-instance-url>/semantics/jobs/<job_id>/status
curl https://<vast-instance-url>/semantics/jobs/<job_id>/results
```

## Configuration

All configuration is via environment variables. See `.env.example` for the full list.

| Variable                   | Description                                                 | Default                          |
| -------------------------- | ----------------------------------------------------------- | -------------------------------- |
| `STASH_URL`                | Stash instance URL, accessible via tunnel (required)        | --                               |
| `STASH_API_KEY`            | Stash API key for authentication (required)                 | --                               |
| `SEMANTICS_TAG_ID`         | Root tag ID for taxonomy subtree filtering                  | --                               |
| `CLASSIFIER_MODEL`         | Classifier variant: `vision` or `text-only`                 | `vision`                         |
| `CLASSIFIER_DEVICE`        | Inference device for the classifier                         | `cuda`                           |
| `SEMANTICS_LLM_MODEL`      | HuggingFace model ID for summary/title generation           | `RedHatAI/Llama-3.1-8B-Instruct` |
| `SEMANTICS_LLM_DEVICE`     | Inference device for the LLM                                | `cuda`                           |
| `SEMANTICS_HF_REPO`        | HuggingFace repo for classifier weights                     | `smegmarip/tag-classifier`       |
| `HF_TOKEN`                 | HuggingFace token for gated model access                    | --                               |
| `SEMANTICS_MIN_CONFIDENCE` | Minimum tag confidence threshold (0.0--1.0)                 | `0.75`                           |
| `SEMANTICS_JOB_LOCK_TTL`   | Max seconds a job can hold the active lock                  | `3600`                           |
| `REDIS_URL`                | Redis connection URL (embedded in container)                | `redis://localhost:6379/0`       |
| `CACHE_TTL`                | Cache TTL in seconds (1 year keeps results across restarts) | `31536000`                       |
| `LOG_LEVEL`                | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`          | `INFO`                           |

## Performance

Observed on an H100 80GB (Vast.ai, bfloat16, all models resident):

| Stage                        | Time     | Notes                                         |
| ---------------------------- | -------- | --------------------------------------------- |
| Sprite download + extraction | ~1s      | Network-bound (tunnel latency)                |
| JoyCaption (16 frames)       | ~73s     | Sequential autoregressive decode, ~4.5s/frame |
| Llama summary                | ~9s      | ~500 tokens generated                         |
| Llama title                  | ~0.2s    | Short generation                              |
| Tag classifier + SiGLIP      | ~0.1s    | Single forward pass                           |
| **Total per job**            | **~90s** |                                               |

JoyCaption accounts for ~80% of job time. This is inherent to sequential single-frame autoregressive generation with a VLM.

For comparison, the parent project on an A4000 (16GB) with quantization and model swapping takes ~6 minutes per job.

## VRAM Budget

| Model                            | VRAM          | Lifecycle       |
| -------------------------------- | ------------- | --------------- |
| JoyCaption beta-one (bfloat16)   | ~16 GB        | Resident        |
| Llama 3.1 8B Instruct (bfloat16) | ~16 GB        | Resident        |
| Tag Classifier + SiGLIP (vision) | ~3.8 GB       | Resident        |
| CUDA context + KV cache          | ~3-4 GB       | Runtime         |
| **Total**                        | **~39-42 GB** | **52% of 80GB** |

## API Endpoints

| Method | Path                               | Description                                           |
| ------ | ---------------------------------- | ----------------------------------------------------- |
| `POST` | `/semantics/analyze`               | Submit analysis job (returns immediately with job_id) |
| `GET`  | `/semantics/jobs/{job_id}/status`  | Poll job progress                                     |
| `GET`  | `/semantics/jobs/{job_id}/results` | Fetch completed results                               |
| `GET`  | `/semantics/health`                | Health check (model status, queue stats)              |
| `GET`  | `/docs`                            | Interactive API documentation (Swagger UI)            |

Request/response schemas are identical to the parent project. See the [OpenAPI spec](https://github.com/smegmarip/stash-auto-vision) for details.

## Repository Layout

```
stash-auto-vision-semantics/
├── README.md                        <- this file
├── CLAUDE.md                        <- design specification
├── Dockerfile                       <- CUDA 12.4 + Redis + Python 3.11
├── entrypoint.sh                    <- starts Redis, then uvicorn
├── requirements.txt                 <- Python dependencies (torch installed separately)
├── .env.example                     <- environment variable template
├── app/
│   ├── main.py                      <- FastAPI app, pipeline, endpoints
│   ├── models.py                    <- Pydantic request/response models
│   ├── cache_manager.py             <- Redis-backed result caching
│   ├── job_queue.py                 <- Redis job queue with Lua scripts
│   ├── worker.py                    <- single-job async worker loop
│   ├── caption_generator.py         <- JoyCaption beta-one wrapper (bfloat16)
│   ├── llama_runtime.py             <- Llama 3.1 8B wrapper (bfloat16)
│   ├── summary_generator.py         <- scene summary prompt/generation
│   ├── title_generator.py           <- scene title prompt/generation
│   ├── classifier.py                <- tag classifier with SiGLIP vision
│   ├── taxonomy_builder.py          <- Stash taxonomy → classifier format
│   └── sprite_parser.py             <- VTT parsing + sprite tile extraction
└── train/
    ├── model.py                     <- MultiViewClassifier architecture
    ├── config.py                    <- TrainConfig dataclass
    └── tag_families.py              <- tag family grouping
```

## Credits

- Extracted from [stash-auto-vision](https://github.com/smegmarip/stash-auto-vision).
- Tag classifier trained on the [tag-classification](https://huggingface.co/smegmarip/tag-classifier) dataset.
- Captioning via [JoyCaption beta-one](https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava).
- Summaries via [Llama 3.1 8B Instruct](https://huggingface.co/RedHatAI/Llama-3.1-8B-Instruct).
- Vision grounding via [SiGLIP](https://huggingface.co/google/siglip-base-patch16-384).
