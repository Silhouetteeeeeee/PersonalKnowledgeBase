#!/bin/bash
# One-command deploy: build thin image, export, upload, reload
#
# Usage: bash deploy.sh <server-host>
#   bash deploy.sh root@your-server-ip

set -euo pipefail

SERVER="${1:?Usage: bash deploy.sh root@your-server-ip}"

echo "==> Step 1: Building application image (fast) ..."
docker build -t knowledge-agent .

echo "==> Step 2: Exporting image ..."
docker save knowledge-agent | gzip > /tmp/kb-agent.tar.gz

echo "==> Step 3: Uploading image and runtime files to $SERVER ..."
ssh "$SERVER" "mkdir -p /opt/kb/data /opt/kb/cache"
scp /tmp/kb-agent.tar.gz docker-compose.yml .env "$SERVER:/opt/kb/"

echo "==> Step 4: Loading and restarting on server ..."
ssh "$SERVER" "cd /opt/kb && docker load < kb-agent.tar.gz && docker compose up -d"

echo "==> Done! Deployed successfully."
