# Personal Knowledge Base Agent — Docker Image
#
# Build:
#   docker build -t knowledge-agent .
#
# Run (with env vars from .env):
#   docker run -d --name kb-agent \
#     --env-file .env \
#     -v ./data:/app/data \
#     -v ./cache:/root/.cache \
#     knowledge-agent
#
# Without PaddleOCR (smaller image, ~1GB vs ~3GB):
#   docker build --build-arg WITH_OCR=false -t knowledge-agent:lite .

FROM python:3.10-slim

WORKDIR /app

# ── System dependencies ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──
COPY requirements.txt .

# PaddleOCR is HEAVY (~2GB). Install with WITH_OCR=true to include it.
ARG WITH_OCR=true

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Remove PaddleOCR if user opted out
RUN if [ "$WITH_OCR" = "false" ]; then \
        pip uninstall -y paddleocr paddlepaddle paddlepaddle-gpu 2>/dev/null || true; \
    fi

# ── Project code ──
COPY . .

RUN mkdir -p /app/data /root/.cache

# ── Volumes ──
#   /app/data          → knowledge.db, wiki pages, profiles, reasoning logs
#   /root/.cache       → fastembed/sentence-transformers model cache
VOLUME ["/app/data", "/root/.cache"]

CMD ["python", "main.py"]
