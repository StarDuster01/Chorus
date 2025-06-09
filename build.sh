#!/bin/bash

# Build script for RagBot with version control
set -e

echo "🚀 Building RagBot with GPU-based embeddings..."

# Version info
BUILD_VERSION="1.1.0"
BUILD_DATE=$(date '+%Y-%m-%d %H:%M:%S')
VCS_REF="gpu-embeddings-preload"

echo "📦 Version: $BUILD_VERSION"
echo "📅 Build Date: $BUILD_DATE"
echo "🔧 VCS Ref: $VCS_REF"

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Build with version arguments and no cache
echo "🏗️ Building Docker image with version control..."
docker-compose build \
  --build-arg BUILD_DATE="$BUILD_DATE" \
  --build-arg BUILD_VERSION="$BUILD_VERSION" \
  --build-arg VCS_REF="$VCS_REF" \
  --no-cache

echo "✅ Build complete!"
echo "🚀 Starting containers..."
docker-compose up -d

echo "📋 Checking container status..."
sleep 5
docker-compose ps

echo "📄 Viewing recent logs..."
docker logs ragbot_backend_1 --tail 30

echo "🎉 RagBot v$BUILD_VERSION is ready!"
echo "🌐 Frontend: http://localhost:50505"
echo "📊 To view logs: docker logs ragbot_backend_1 -f" 