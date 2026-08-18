# ---- stage 1: build llama-cpp-python with CUDA support ----
FROM pytorch/pytorch:2.13.0-cuda12.6-cudnn9-devel AS llama-builder

ARG http_proxy
ARG https_proxy
ARG no_proxy
# NVIDIA GPU compute capability of the target card (89 = Ada, e.g. RTX 40 series)
ARG CUDA_ARCH=89

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FORCE_CMAKE=1 \
    CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"

RUN export http_proxy="$http_proxy" https_proxy="$https_proxy" no_proxy="$no_proxy" \
    && apt-get update \
    && apt-get install -y --no-install-recommends python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv --system-site-packages /opt/build-venv
ENV PATH="/opt/build-venv/bin:$PATH"

RUN export http_proxy="$http_proxy" https_proxy="$https_proxy" no_proxy="$no_proxy" \
    && pip install --no-cache-dir cmake ninja \
    && pip wheel --no-cache-dir --no-deps -w /wheels llama-cpp-python==0.3.35

# ---- stage 2: runtime ----
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

COPY --from=llama-builder /wheels/ /wheels/

RUN export http_proxy="$http_proxy" https_proxy="$https_proxy" no_proxy="$no_proxy" \
    && pip install --no-cache-dir /wheels/*.whl \
    && pip install --no-cache-dir \
    "accelerate" \
    "safetensors" \
    "transformers==5.15.0" \
    "sdnq==0.2.4" \
    "fastapi" \
    "uvicorn[standard]" \
    "python-multipart" \
    "pillow" \
    && pip install --no-cache-dir \
    "git+https://github.com/huggingface/diffusers.git@a00d536450c6cb83824366f4b4d22426cba9165c"

WORKDIR /app

EXPOSE 8000

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
