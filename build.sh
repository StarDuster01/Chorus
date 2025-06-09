#!/bin/bash

# Build script for RagBot with version control
set -e

echo "🚀 Building RagBot with GPU-based embeddings..."

# Version info
BUILD_VERSION="1.1.7"
BUILD_DATE=$(date '+%Y-%m-%d %H:%M:%S')
VCS_REF="simplified-chroma-client"

echo "📦 Version: $BUILD_VERSION"
echo "📅 Build Date: $BUILD_DATE"
echo "🔧 VCS Ref: $VCS_REF"

# Check for no-cache flag
NO_CACHE_FLAG=""
if [[ "$1" == "--no-cache" ]]; then
    echo "🔄 Using --no-cache (slower but ensures fresh build)"
    NO_CACHE_FLAG="--no-cache"
else
    echo "⚡ Using Docker cache (faster builds)"
    echo "   💡 Use './build.sh --no-cache' for fresh build if needed"
fi

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Build with version arguments and optional no-cache
echo "🏗️ Building Docker image with version control..."
export DOCKER_BUILDKIT=1
docker-compose build \
  --build-arg BUILD_DATE="$BUILD_DATE" \
  --build-arg BUILD_VERSION="$BUILD_VERSION" \
  --build-arg VCS_REF="$VCS_REF" \
  $NO_CACHE_FLAG

echo "✅ Build complete!"
echo "🚀 Starting containers..."
docker-compose up -d

echo "📋 Checking container status..."
sleep 10
docker-compose ps

echo "📄 Viewing recent logs..."
docker logs ragbot_backend_1 --tail 40

echo "🎉 RagBot v$BUILD_VERSION is ready!"
echo "🌐 Frontend: http://localhost:50505"
echo "📊 To view logs: docker logs ragbot_backend_1 -f" 