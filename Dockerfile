FROM python:3.10-slim

ARG TORCH_VERSION=2.5.1
ARG TORCHVISION_VERSION=0.20.1
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/huggingface \
    TRANSFORMERS_CACHE=/data/huggingface/transformers \
    HUGGINGFACE_HUB_CACHE=/data/huggingface/hub \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TOKENIZERS_PARALLELISM=false \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    tini \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 --shell /bin/bash appuser

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN grep -vE '^(torch|torchvision)([<>=!~].*)?$' /app/requirements.txt > /tmp/requirements.no-torch.txt \
 && python -m pip install --upgrade pip setuptools wheel \
 && pip install --index-url "${PYTORCH_INDEX_URL}" \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
 && pip install -r /tmp/requirements.no-torch.txt

COPY . /app

RUN mkdir -p /app/logs /app/gradio_logs /data/huggingface \
 && chown -R appuser:appuser /app /data

USER appuser

EXPOSE 7860

VOLUME ["/data/huggingface"]

ENTRYPOINT ["tini", "--"]
CMD ["python", "app.py"]
