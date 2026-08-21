import gc
import io
import os
import threading
from contextlib import asynccontextmanager
import anyio
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from diffusers import Flux2KleinPipeline
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("FLUX_MODEL_DIR", os.path.join(HERE, "model"))

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
    print("[service] loading SDNQ 4-bit pipeline (transformer + text encoder + VAE) ...", flush=True)
    pipe = Flux2KleinPipeline.from_pretrained(MODEL_DIR, torch_dtype=DTYPE)
    pipe.to("cuda")

    # Enable INT8 MatMul for AMD, Intel ARC and Nvidia GPUs:
    if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
        pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
        pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)
        print("[service] SDNQ INT8 matmul enabled", flush=True)
    else:
        print("[service] SDNQ INT8 matmul disabled (triton not available)", flush=True)
    # pipe.enable_model_cpu_offload()
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
            try:
                result = _pipe(
                    image=images or None,
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    generator=generator,
                ).images[0]
            except RuntimeError as e:
                # Catch CUDA Out of Memory specifically
                if "out of memory" in str(e).lower():
                    print("[service] CUDA OOM encountered! Cleaning memory and retrying...", flush=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    try:
                        # Retry generation after cache flush
                        result = _pipe(
                            image=images or None,
                            prompt=prompt,
                            height=height,
                            width=width,
                            num_inference_steps=steps,
                            generator=generator,
                        ).images[0]
                    except RuntimeError as retry_e:
                        if "out of memory" in str(retry_e).lower():
                            print("[service] CUDA OOM failed on retry as well.", flush=True)
                            raise HTTPException(507, "CUDA Out of Memory on generation retry. Downscale your image size.")
                        raise HTTPException(500, f"Runtime error during generation retry: {str(retry_e)}")
                    except ValueError as retry_val_e:
                        raise HTTPException(400, str(retry_val_e))
                else:
                    raise HTTPException(500, f"Runtime error during generation: {str(e)}")
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
