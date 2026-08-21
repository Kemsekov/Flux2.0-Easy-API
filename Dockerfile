FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# triton JIT compiles kernels at inference time and needs a C compiler
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

EXPOSE 8000

CMD ["echo","Flux2.0 API"]
