# FLUX.2 klein 4B SDNQ 4-bit — Image Edit API

FastAPI service that runs [FLUX.2-klein-4B-SDNQ-4bit-dynamic](https://huggingface.co/Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic)
(SDNQ uint4/int5 dynamic-quantized transformer + VAE) with the SDNQ-quantized Qwen3-4B text encoder,
packaged as a Docker image.

- Transformer ~2.3 GB on GPU (SDNQ 4-bit, dequantized per layer on the fly)
- Text encoder ~2.8 GB packed on GPU (SDNQ uint4/int5, loaded from `model/text_encoder`)
- **Every model weight is loaded through SDNQ** — no llama.cpp, no GGUF
- Total VRAM footprint ~5.5 GB (+ activations)
- Single `POST /generate` endpoint for text-to-image, image editing and multi-reference editing

The image contains **dependencies only**. Project code and models are bind-mounted, so code changes never require
an image rebuild — just restart the container.

---

## 1. Download models

```bash
pip install -U "huggingface[hf_transfer]"

# full SDNQ repo: transformer, VAE, tokenizer AND the SDNQ text encoder
hf download Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic --local-dir model
```

### Expected layout

```
Flux2.0-Easy-API/
├── model/                          # Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic (complete)
│   ├── model_index.json
│   ├── scheduler/
│   ├── tokenizer/
│   ├── transformer/                # SDNQ 4-bit safetensors
│   ├── text_encoder/               # SDNQ 4-bit Qwen3-4B weights + config
│   └── vae/
├── service.py
├── client.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env
```

---

## 2. Run with Docker

```bash
cd Flux2.0-Easy-API
docker compose up -d --build
```

The first build downloads the CUDA 13.0 torch wheel (~4 GB, needed for sm_120 kernels on RTX 50 series).
Subsequent builds hit the layer cache.

Model loads on startup (~30-60 s). Watch progress:

```bash
docker logs -f flux-q4-service
```

Restart after code changes (no rebuild):

```bash
docker compose restart flux-q4
```

## 2b. Run without Docker (host venv)

```bash
python -m venv venv
venv/bin/pip install "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu130
venv/bin/pip install -r requirements.txt
venv/bin/uvicorn service:app --host 0.0.0.0 --port 8000
```

> Requires an NVIDIA GPU with sm_120 support in the torch build (RTX 50 series = Blackwell).
> torch must come from the cu130 index — the default PyPI wheel lacks sm_120 kernels.

---

## 3. API

| Endpoint | Description |
|---|---|
| `GET /health` | status, loaded model, device, VRAM usage |
| `POST /generate` | run the model, returns PNG |

`POST /generate` (multipart/form-data):

| field | type | default | notes |
|---|---|---|---|
| `prompt` | str | required | text prompt |
| `image` | file | — | input/reference image |
| `image_2` | file | — | second reference image (multi-reference editing) |
| `height` / `width` | int | `null` | output size; derived from input image when omitted; must be divisible by 16 |
| `steps` | int | `4` | distilled model, 4 steps recommended |
| `seed` | int | random | returned in `X-Seed` response header |
| `guidance` | float | `1.0` | ignored: pipeline is distilled (`is_distilled=true`) |

Examples:

```bash
# text-to-image
curl -X POST http://localhost:8000/generate \
  -F "prompt=a red car" \
  -F "height=256" -F "width=256" -F "seed=42" \
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

### Python client (`client.py`)

```python
from client import generate   # returns (PIL.Image, seed)

img, seed = generate(
    "a red car",
    images=["ref1.png"],          # optional: paths, PIL.Image or bytes
    height=256, width=256, steps=4, seed=42,
)
img.save("out.png")
```

CLI:

```bash
python client.py "a red car" --image ref1.png --image2 ref2.png \
    --height 256 --width 256 --steps 4 --seed 42 --out out.png
```

---

## 4. How the text encoder works

`Flux2KleinPipeline` natively extracts the per-token hidden states of Qwen3 layers **9, 18, 27** and
stacks them into the prompt embedding (`output_hidden_states=True`). The text encoder is loaded by
transformers from `model/text_encoder` with its SDNQ quantization config (registered via
`import sdnq` in `service.py`) — the same loading path as the transformer and VAE.

---

## 5. Docker base image and CUDA notes

- Base image: `python:3.12-slim`; torch 2.13.0 is installed from the **cu130** PyTorch index
  (`https://download.pytorch.org/whl/cu130`) because it is the build that includes **sm_120**
  kernels for RTX 50 series (Blackwell) GPUs. The `pytorch/pytorch:2.13.0-cuda12.6-*` images and
  PyPI torch wheels do **not** (their arch list stops at sm_90).
- The host driver is the only hard constraint: the container's CUDA 13.0 runtime needs
  `nvidia-smi` to report **CUDA 13.0** (driver >= 580). This was verified on driver 580.173.02.
- `gcc`/`libc6-dev` are installed in the image because triton JIT-compiles kernels at inference time.
- `network_mode: host` is used in docker-compose to avoid creating a docker network
  (the default pool may be exhausted on dev machines).

---

## 6. Hardware notes

- Tested on NVIDIA GeForce RTX 5070 Laptop GPU (8 GB VRAM), driver 580.173.02 (CUDA 13.0).
- VRAM after model load ~5.5 GB; a 256x256 generation peaks at ~6 GB.
- The distilled model runs 4-step inference; expect roughly 5-15 s per image depending on size.
- SDNQ layers are dequantized per-forward with plain bf16 matmuls (`use_quantized_matmul=False`,
  the checkpoint default).
