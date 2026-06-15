# ── Application image: thin layer on top of base ──
#
# This build is FAST (~5 seconds) because all Python
# dependencies are already in kb-base.
#
# Prerequisites:
#   docker build -f Dockerfile.base -t kb-base .
#
# Quick rebuild:
#   docker build -t knowledge-agent .
#   docker compose up -d

FROM kb-base:latest

WORKDIR /app
COPY . .

RUN mkdir -p /app/data /root/.cache

VOLUME ["/app/data", "/root/.cache"]
CMD ["python", "main.py"]
