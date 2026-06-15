# syntax=docker/dockerfile:1.4
#
# Personal Knowledge Base Agent — Docker Image
#
# ===== 部署方式（推荐）=====
# 第一次构建很慢（pip 要下载 ~1GB 依赖包），建议在本地构建好再传到服务器：
#
#   Windows 本地:
#     docker build -t knowledge-agent .
#     docker save knowledge-agent | gzip > kb-agent.tar.gz
#     scp kb-agent.tar.gz root@your-server:/opt/kb/
#
#   CentOS 服务器:
#     cd /opt/kb
#     docker load < kb-agent.tar.gz
#     # 创建 .env 和 data/ 目录
#     docker compose up -d
#
# ===== 服务器直接构建（慢）=====
#   docker compose build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
#   docker compose up -d

FROM python:3.10-slim

WORKDIR /app

# ── System dependencies ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies (with build cache) ──
COPY requirements.txt .

# PaddleOCR is HEAVY (~2GB). Not installed by default.
ARG WITH_OCR=false

# Optional: pip mirror for China
ARG PIP_INDEX_URL

# Install base requirements (exclude heavy PaddleOCR by default)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    grep -v "^paddle" requirements.txt > /tmp/req-base.txt && \
    pip install -r /tmp/req-base.txt ${PIP_INDEX_URL:+--index-url $PIP_INDEX_URL}

# Conditionally install PaddleOCR only when WITH_OCR=true
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$WITH_OCR" = "true" ]; then \
        pip install "paddlepaddle>=2.6.0,<3.0.0" "paddleocr>=2.8.0,<3.0.0" ${PIP_INDEX_URL:+--index-url $PIP_INDEX_URL}; \
    fi

# ── Project code ──
COPY . .
RUN mkdir -p /app/data /root/.cache

# ── Volumes ──
VOLUME ["/app/data", "/root/.cache"]

CMD ["python", "main.py"]
