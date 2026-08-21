FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# torch must come from the cu130 index: it provides sm_120 kernels for
# RTX 50 series (Blackwell) GPUs. Installed first so the pinned requirements
# resolve against it.
RUN pip install --upgrade pip \
    && pip install "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu130

COPY requirements.txt .
RUN pip install -r requirements.txt

# triton JIT compiles kernels at inference time and needs a C compiler
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

EXPOSE 8000

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
