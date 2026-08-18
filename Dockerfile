FROM pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

ARG http_proxy
ARG https_proxy
ARG no_proxy

RUN apt-get update \
    && apt-get install -y --no-install-recommends git python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN export http_proxy="$http_proxy" https_proxy="$https_proxy" no_proxy="$no_proxy" \
    && pip install --no-cache-dir \
    "accelerate" \
    "safetensors" \
    "sentencepiece" \
    "transformers==5.15.0" \
    "torchao==0.13.0" \
    "fastapi" \
    "uvicorn[standard]" \
    "python-multipart" \
    "pillow" \
    && pip install --no-cache-dir \
    "git+https://github.com/huggingface/diffusers.git@a00d536450c6cb83824366f4b4d22426cba9165c"

WORKDIR /app

EXPOSE 8000

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]

