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


## 4. Hardware notes
 Requires modern hardware with native bfloat16 (bf16) support for optimal performance and numerical stability (e.g., NVIDIA Ampere architecture or newer, such as RTX 30/40/50-series, A100, H100, H200).
