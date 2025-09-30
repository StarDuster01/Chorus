#!/bin/bash

# Chorus RagBot - Frontend → Backend refresh and redeploy (Linux/Bash)
# Usage:
#   ./deploy-refresh.sh                 # normal cached build
#   ./deploy-refresh.sh --no-cache      # force full rebuild of backend image
#   REACT_APP_API_URL=/api ./deploy-refresh.sh

set -euo pipefail

PROJECT_ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
FRONTEND_DIR="$PROJECT_ROOT_DIR/ragbot-frontend"
BACKEND_DIR="$PROJECT_ROOT_DIR/ragbot-backend"
DEST_FRONTEND_DIR="$BACKEND_DIR/frontend"

NO_CACHE_FLAG=""
if [[ ${1-} == "--no-cache" ]]; then
  NO_CACHE_FLAG="--no-cache"
fi

# Default API base for frontend → backend calls if not provided
: "${REACT_APP_API_URL:=/api}"

echo "=== STEP 1: Building React frontend ==="
echo "REACT_APP_API_URL=${REACT_APP_API_URL}"
cd "$FRONTEND_DIR"
export REACT_APP_API_URL

if command -v npm >/dev/null 2>&1; then
  echo "Installing dependencies (npm ci)..."
  npm ci --silent || npm install --silent
  echo "Building production bundle (npm run build)..."
  npm run build
else
  echo "ERROR: npm is not installed. Please install Node.js/npm first." >&2
  exit 1
fi

echo ""
echo "=== STEP 2: Copying frontend build into backend ==="
rm -rf "$DEST_FRONTEND_DIR"
mkdir -p "$DEST_FRONTEND_DIR"
cp -r "$FRONTEND_DIR/build"/* "$DEST_FRONTEND_DIR/"
echo "Copied build → $DEST_FRONTEND_DIR"

echo ""
echo "=== STEP 3: Rebuilding backend image (docker-compose build) ==="
cd "$PROJECT_ROOT_DIR"
docker-compose build ${NO_CACHE_FLAG} backend

echo ""
echo "=== STEP 4: Restarting backend service ==="
docker-compose up -d backend

echo ""
echo "=== STEP 5: Checking health ==="
sleep 2
if command -v curl >/dev/null 2>&1; then
  set +e
  curl -f "http://localhost:50506/health" && echo "\nHealth check OK" || echo "Health endpoint not available yet (this can be normal during warm-up)"
  set -e
else
  echo "curl not installed; skipping health probe"
fi

echo ""
echo "All done. Visit your app at: http://<vm-public-ip>:50506"
echo "Tip: force a clean image rebuild with: ./deploy-refresh.sh --no-cache"


