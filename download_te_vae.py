import os, sys
from huggingface_hub import snapshot_download

out = "/mnt/d/cpp-remake/flux/model"
snapshot_download(
    "black-forest-labs/FLUX.2-klein-4B",
    allow_patterns=[
        "text_encoder/model-*.safetensors",
        "text_encoder/model.safetensors.index.json",
        "vae/*.safetensors",
    ],
    local_dir=out,
)
print("DOWNLOAD_OK", flush=True)
