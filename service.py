import io
import os
import threading
from contextlib import asynccontextmanager

import anyio
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from transformers import Qwen2Tokenizer

from diffusers import Flux2KleinPipeline

from text_encoder_llama import LlamaQwen3TextEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("FLUX_MODEL_DIR", os.path.join(HERE, "model"))
TEXT_ENCODER_DIR = os.environ.get("FLUX_TEXT_ENCODER_DIR", os.path.join(HERE, "text_encoder"))
GGUF_FILE = os.environ.get("FLUX_TEXT_ENCODER_GGUF", "flux2-klein-4b-uncensored-q4_k_m.gguf")

DTYPE = torch.bfloat16
MODEL_ID = "FLUX.2-klein-4B-SDNQ-4bit-dynamic"

_lock = threading.Lock()
_pipe = None


def vram_gb():
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
        "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
    }


def build_pipeline():
    print("[service] loading tokenizer ...", flush=True)
    tokenizer = Qwen2Tokenizer.from_pretrained(os.path.join(MODEL_DIR, "tokenizer"))

    print("[service] loading Qwen3-4B text encoder GGUF (llama.cpp, native q4, ~2.5 GB VRAM) ...", flush=True)
    text_encoder = LlamaQwen3TextEncoder(
        model_path=os.path.join(TEXT_ENCODER_DIR, GGUF_FILE),
        tokenizer=tokenizer,
    )

    print("[service] loading SDNQ 4-bit transformer + VAE ...", flush=True)
    pipe = Flux2KleinPipeline.from_pretrained(
        MODEL_DIR, text_encoder=text_encoder, tokenizer=tokenizer, torch_dtype=DTYPE
    )
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print("[service] ready. vram:", vram_gb(), flush=True)
    return pipe


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipe
    _pipe = build_pipeline()
    yield


app = FastAPI(title="FLUX.2 klein 4B SDNQ-4bit edit API", lifespan=lifespan)


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

    if prompt is None:
        raise HTTPException(400, "prompt is required")

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
            prompt_embeds = _pipe.text_encoder.encode_prompt(prompt)
            try:
                result = _pipe(
                    image=images or None,
                    prompt=None,
                    prompt_embeds=prompt_embeds,
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
