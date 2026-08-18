# FLUX.2 klein 4B SDNQ 4-bit — Image Edit API

FastAPI service that runs [FLUX.2-klein-4B-SDNQ-4bit-dynamic](https://huggingface.co/Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic)
(SDNQ uint4/int5 dynamic-quantized transformer + VAE) with an
[uncensored Qwen3-4B text encoder in GGUF q4_k_m format](https://huggingface.co/ponpoke/flux2-klein-4b-uncensored-text-encoder),
packaged as a Docker image.

- Transformer ~2.3 GB on GPU (SDNQ 4-bit, dequantized per layer on the fly)
- Text encoder runs in **native q4 via llama.cpp** (~2.5 GB VRAM, no bf16 dequantization)
- Total VRAM footprint ~7-8 GB
- Single `POST /generate` endpoint for text-to-image, image editing and multi-reference editing

The image contains **dependencies only**. Project code and models are bind-mounted, so code changes never require
an image rebuild — just restart the container.

---

## 1. Download models

```bash
pip install -U "huggingface[hf_transfer]"

# transformer (SDNQ 4-bit), VAE, tokenizer, configs — text encoder weights excluded
hf download Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic \
    --exclude "text_encoder/*" \
    --local-dir model

# uncensored Qwen3-4B text encoder, GGUF q4_k_m only
hf download ponpoke/flux2-klein-4b-uncensored-text-encoder \
    --exclude "*.safetensors" --exclude "*f16*" --exclude "*q6*" --exclude "*q8*" \
    --local-dir text_encoder
```

### Expected layout

```
fluxq4/
├── model/                          # Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic (no text_encoder weights)
│   ├── model_index.json
│   ├── scheduler/
│   ├── tokenizer/
│   ├── transformer/                # SDNQ 4-bit safetensors
│   ├── text_encoder/               # config only (weights come from GGUF)
│   └── vae/
├── text_encoder/
│   ├── flux2-klein-4b-uncensored-q4_k_m.gguf
│   └── flux2-klein-4b-uncensored-text-encoder/   # config + tokenizer files
├── Flux2.0-Easy-API/
│   ├── service.py
│   ├── text_encoder_llama.py
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .env
└── download.sh
```

---

## 2. Run

> Internet access is needed to **build** the image (pip, git). The proxy is **optional**:
> if your machine has normal internet, leave `WSL_PROXY` empty (or remove it from `.env`)
> and the build runs without any proxy. If you must use a proxy, set it in `.env` —
> it must be reachable from Docker build containers: with Docker Desktop on WSL2 use
> the WSL host IP (see `hostname -I`), not `127.0.0.1`.

The first build compiles `llama-cpp-python` with CUDA support for your GPU
(see `CUDA_ARCH` below), which takes ~10-20 min one time:

```bash
cd Flux2.0-Easy-API
docker compose up -d --build
```

Model loads on startup (~30-60 s). Watch progress:

```bash
docker logs -f flux-q4-service
```

Restart after code changes (no rebuild):

```bash
docker compose restart flux-q4
```

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
| `height` / `width` | int | `null` | output size; derived from input image when omitted |
| `steps` | int | `4` | distilled model, 4 steps recommended |
| `seed` | int | random | returned in `X-Seed` response header |
| `guidance` | float | `1.0` | ignored: pipeline is distilled (`is_distilled=true`) |

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

## 4. How the text encoder works

The FLUX.2 pipeline stacks per-token hidden states of Qwen3 layers **9, 18, 27** into the prompt
embedding. llama.cpp only exposes final-layer embeddings through its public API, but its extended
API (`llama_set_embeddings_layer_inp` / `llama_get_embeddings_layer_inp`) can output the residual
stream entering any layer. `text_encoder_llama.py` calls these functions directly through the
bundled `libllama.so` and feeds the stacked embeddings to the pipeline via `prompt_embeds=`, so the
GGUF never gets dequantized in bulk — VRAM stays at ~2.5 GB for the text encoder.

## 5. Choosing the Docker base image (CUDA)

The Dockerfile uses `pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime` (plus the matching `-devel`
image to compile llama-cpp-python). The rules:

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
   (`sdnq` `0.2.4`, `transformers` `5.15.0`, `llama-cpp-python` `0.3.35`, diffusers commit in
   `Dockerfile`) — changing the torch major version may break the SDNQ loading path.
4. `CUDA_ARCH` build arg must match your GPU's compute capability: `89` for Ada (RTX 40 series),
   `120` for Blackwell (RTX 50 series), `86` for Ampere (RTX 30 series). Pass with
   `docker compose build --build-arg CUDA_ARCH=...` or set it in the compose file.

`-runtime` is enough for inference; the `-devel` stage is only used at build time for `nvcc`.

---

## 6. Hardware notes

- Needs an NVIDIA GPU with ~8 GB free VRAM (SDNQ transformer ~2.3 GB packed + q4 text encoder
  ~2.5 GB + VAE + activations).
- Tested on RTX 4070 Ti SUPER (16 GB), driver 560.94 (CUDA 12.6).
- The distilled model runs 4-step inference; expect roughly 1-3 s per image depending on size.
- SDNQ layers are dequantized per-forward with plain bf16 matmuls (`use_quantized_matmul=False`,
  the checkpoint default).
