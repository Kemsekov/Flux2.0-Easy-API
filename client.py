import argparse
import io

import requests
from PIL import Image


def generate(
    prompt,
    images=None,
    height=None,
    width=None,
    steps=4,
    seed=None,
    base_url="http://localhost:8000",
):
    files = []
    data = {"prompt": prompt, "steps": steps}
    if height is not None:
        data["height"] = height
    if width is not None:
        data["width"] = width
    if seed is not None:
        data["seed"] = seed

    for key, src in zip(("image", "image_2"), images or []):
        if isinstance(src, str):
            files.append((key, open(src, "rb")))
        elif isinstance(src, Image.Image):
            buf = io.BytesIO()
            src.convert("RGB").save(buf, format="PNG")
            files.append((key, ("img.png", buf.getvalue(), "image/png")))
        elif isinstance(src, bytes):
            files.append((key, ("img.png", src, "image/png")))
        else:
            raise TypeError(f"unsupported image type: {type(src)}")

    r = requests.post(f"{base_url}/generate", data=data, files=files)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB"), r.headers.get("X-Seed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Call the FLUX.2 /generate endpoint")
    ap.add_argument("prompt")
    ap.add_argument("--image", help="path to reference image")
    ap.add_argument("--image2", help="path to second reference image")
    ap.add_argument("--height", type=int)
    ap.add_argument("--width", type=int)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--out", default="out.png")
    a = ap.parse_args()

    imgs = [p for p in (a.image, a.image2) if p]
    img, seed = generate(a.prompt, imgs, a.height, a.width, a.steps, a.seed, a.url)
    img.save(a.out)
    print(f"saved {a.out} ({img.width}x{img.height}, seed={seed})")
