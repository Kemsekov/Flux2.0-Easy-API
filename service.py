import io
import os
import sys
import threading
from contextlib import asynccontextmanager

import anyio
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
PHOTOROOM_DIR = os.path.join(HERE, "model_photoroom")
sys.path.insert(0, PHOTOROOM_DIR)

from load_torchao import load_torchao_fp8_static_model
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel

DTYPE = torch.bfloat16
MODEL_ID = "FLUX.2-klein-4b-fp8"

_lock = threading.Lock()
_pipe = None


def vram_gb():
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
        "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
    }


def build_pipeline():
    print("[service] loading fp8 transformer (torchao static fp8) ...", flush=True)
    transformer = load_torchao_fp8_static_model(
        ckpt_path=os.path.join(PHOTOROOM_DIR, "transformer_fp8_static", "model_fp8_static.pt"),
        base_model_or_factory=lambda: Flux2Transformer2DModel.from_pretrained(
            os.path.join(PHOTOROOM_DIR, "transformer_bf16"), torch_dtype=DTYPE
        ),
        device="cuda",
    ).to("cuda")

    print("[service] loading pipeline components ...", flush=True)
    pipe = Flux2KleinPipeline.from_pretrained(MODEL_DIR, transformer=transformer, torch_dtype=DTYPE)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print("[service] ready. vram:", vram_gb(), flush=True)
    return pipe


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipe
    _pipe = build_pipeline()
    yield


app = FastAPI(title="FLUX.2 klein 4B fp8 edit API", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok" if _pipe is not None else "loading",
        "model": MODEL_ID,
        "device": str(next(_pipe.transformer.parameters()).device) if _pipe is not None else None,
        "vram": vram_gb(),
    }


@app.post("/generate")
async def generate(
    prompt: str | None = Form(None),
    image: UploadFile | None = File(None),
    image_2: UploadFile | None = File(None),
    height: int | None = Form(None),
    width: int | None = Form(None),
    steps: int = Form(4),
    seed: int | None = Form(None),
    guidance: float = Form(1.0),
):
    if _pipe is None:
        raise HTTPException(503, "model still loading")

    images = []
    for f in (image, image_2):
        if f is None:
            continue
        data = await f.read()
        try:
            images.append(Image.open(io.BytesIO(data)).convert("RGB"))
        except Exception:
            raise HTTPException(400, f"cannot decode image: {f.filename}")

    if seed is None:
        seed = int(torch.randint(0, 2**63 - 1, (1,)).item())
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def run():
        with _lock:
            try:
                result = _pipe(
                    image=images or None,
                    prompt=prompt,
                    height=height,
                    width=width,
                    guidance_scale=guidance,
                    num_inference_steps=steps,
                    generator=generator,
                ).images[0]
            except ValueError as e:
                raise HTTPException(400, str(e))
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        return buf.getvalue()

    out = await anyio.to_thread.run_sync(run)
    return Response(content=out, media_type="image/png", headers={"X-Seed": str(seed)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
