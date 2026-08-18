# FLUX.2 klein 4B fp8 — Image Edit API

FastAPI service that runs [FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) with the
[official fp8 checkpoint](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8) in **native fp8 precision**
(transformer ~4 GB on GPU instead of ~7.75 GB bf16, no dequantization), packaged as a Docker image.

The image contains **dependencies only**. Project code and models are bind-mounted, so code changes never require
an image rebuild — just restart the container.

---

## 1. Download models

Three sources are needed:

### a) Pipeline components (tokenizer, scheduler, configs only)

Download only the code/config files, excluding large weights (~8 GB saved):

```bash
pip install -U "huggingface[hf_transfer]"

hf download black-forest-labs/FLUX.2-klein-4B \
    --exclude "*.bin" --exclude "*.safetensors" \
    --local-dir model
```

### b) Decoder weights

Download the dedicated small decoder separately:

```bash
hf download black-forest-labs/FLUX.2-small-decoder \
    --local-dir model/vae
```

### c) Text encoder (Qwen3 FP8)

Download the community FP8 Qwen3 text encoder:

```bash
hf download q10/Qwen3-8B-Base-FP8 \
    --local-dir model/text_encoder
```

### d) Diffusers-compatible fp8 transformer (Photoroom conversion)

The BFL fp8 repo only ships a ComfyUI-format single file that `diffusers` cannot load.
The community-standard diffusers conversion is used instead (recommended in the BFL repo's own discussion thread):

```bash
hf download Photoroom/FLUX.2-klein-4b-fp8-diffusers \
    --include "transformer_bf16/*" "transformer_fp8_static/*" "load_torchao.py" \
    --local-dir model_photoroom
```

### Expected layout

```
flux/
├── model/                          # black-forest-labs/FLUX.2-klein-4B (code/configs only)
│   ├── model_index.json
│   ├── scheduler/
│   ├── tokenizer/
│   ├── text_encoder/               # q10/Qwen3-8B-Base-FP8 weights
│   └── vae/                        # black-forest-labs/FLUX.2-small-decoder weights
├── model_photoroom/
│   ├── load_torchao.py
│   ├── transformer_bf16/           # base transformer (bf16)
│   └── transformer_fp8_static/     # torchao fp8 static checkpoint
├── service.py
├── Dockerfile
├── docker-compose.yml
└── .env
```

---

## 2. Run

> Internet access is only needed to **build** the image (pip). The proxy is **optional**:
> if your machine has normal internet, leave `WSL_PROXY` empty (or remove it from `.env`)
> and the build runs without any proxy. If you must use a proxy, set it in `.env` —
> it must be reachable from Docker build containers: with Docker Desktop on WSL2 use
> the WSL host IP (see `hostname -I`), not `127.0.0.1`.

```bash
docker compose up -d --build
```

Model loads on startup (~1–2 min). Watch progress:

```bash
docker logs -f flux-fp8-service
```

Restart after code changes (no rebuild):

```bash
docker compose restart flux-fp8
```

---

## 3. API

| Endpoint | Description |
|---|---|
| `GET /health` | status, loaded model, device, VRAM usage |
| `POST /generate` | run the model, returns PNG |

`POST /generate` (multipart/form-data) — mirrors `Flux2KleinPipeline.__call__` signature:

| field | type | default | notes |
|---|---|---|---|
| `prompt` | str | `null` | required unless `prompt_embeds` (not exposed) |
| `image` | file | — | input/reference image |
| `image_2` | file | — | second reference image (multi-reference editing) |
| `height` / `width` | int | `null` | output size; derived from input image when omitted |
| `steps` | int | `4` | distilled model, 4 steps recommended |
| `seed` | int | random | returned in `X-Seed` response header |
| `guidance` | float | `1.0` | ignored for distilled model unless > 1 |

Examples:

```bash
# text-to-image
curl -X POST http://localhost:8000/generate \
  -F "prompt=A white ceramic mug on a wooden table" \
  -F "height=1024" -F "width=1024" -F "seed=1" \
  -o out.png

# image editing
curl -X POST http://localhost:8000/generate \
  -F "prompt=Remove background, keep only hands and white background" \
  -F "image=@example.png" \
  -o out.png

# multi-reference editing
curl -X POST http://localhost:8000/generate \
  -F "prompt=Style the object in image 1 with colors from image 2" \
  -F "image=@ref1.png" -F "image_2=@ref2.png" \
  -o out.png
```

Python client:

```python
import requests

with open("example.png", "rb") as f:
    r = requests.post(
        "http://localhost:8000/generate",
        data={"prompt": "Remove background, keep only hands and white background"},
        files={"image": f},
    )
r.raise_for_status()
with open("out.png", "wb") as f:
    f.write(r.content)
```

---

## 4. Choosing the Docker base image (CUDA)

The Dockerfile uses `pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime`. The rule:

1. **Your installed NVIDIA driver is the only hard constraint** — a container image can use any CUDA
   toolkit version that is `<=` the CUDA version your driver supports.
   Check with `nvidia-smi` (top-right corner, "CUDA Version").
2. Pick the `pytorch/pytorch` tag whose CUDA matches that, e.g.:

| `nvidia-smi` CUDA | base image tag |
|---|---|
| 12.6 | `pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime` |
| 12.4 | `pytorch/pytorch:2.13.0-cuda12.4-cudnn9-runtime` |
| 13.x | `pytorch/pytorch:2.13.0-cuda13.2-cudnn9-runtime` |

3. Keep the torch minor version in sync with what the pinned dependencies were tested against
   (torchao `0.13.0`, diffusers commit in `Dockerfile`) — changing the torch major version may break
   the fp8 loading path.

`-runtime` is enough for inference; `-devel` is only needed for compiling extensions.

---

## 5. Hardware notes

- Needs an NVIDIA GPU with ~13 GB free VRAM (fp8 transformer ~4 GB + Qwen3 text encoder ~8 GB + VAE).
- Tested on RTX 4070 Ti SUPER (16 GB), Ada or newer. FP8 matmuls run through `torch._scaled_mm`.
- NVFP4 variants require Blackwell (RTX 50 series) and are not supported here.