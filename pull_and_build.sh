cat << 'EOF' > pull_and_build.sh
#!/bin/bash
#
# pull_and_build.sh — pull latest from GitHub and then build RagBot
# Usage:
#   $ export GITHUB_USERNAME="your-username"
#   $ export GITHUB_TOKEN="your-gh-token"
#   $ ./pull_and_build.sh [--no-cache]
#

set -euo pipefail

# 1. Verify credentials
if [[ -z "${GITHUB_USERNAME:-}" ]] || [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "❗ ERROR: Please set GITHUB_USERNAME and GITHUB_TOKEN environment variables before running."
  exit 1
fi

# 2. Configure git to use your token for HTTPS pulls
echo "🔐 Configuring Git authentication..."
git config http.https://github.com/.extraheader \
     "AUTHORIZATION: basic $(printf '%s:%s' "\$GITHUB_USERNAME" "\$GITHUB_TOKEN" | base64 -w0)"

# 3. Stash local changes
echo "📂 Stashing local changes..."
git stash push --include-untracked -m "pre-pull stash @ \$(date '+%Y-%m-%d %H:%M:%S')"

# 4. Pull latest from origin on current branch
CURRENT_BRANCH=\$(git rev-parse --abbrev-ref HEAD)
echo "⬇️ Pulling latest changes from origin/\${CURRENT_BRANCH}..."
git pull origin "\${CURRENT_BRANCH}"

# 5. Re-apply your stashed changes
echo "🎯 Applying stashed changes..."
if git stash list | grep -q "pre-pull stash"; then
  git stash pop --index || {
    echo "⚠️  Could not pop stash cleanly; you may need to resolve conflicts manually."
  }
else
  echo "   (no stash entry found)"
fi

# 6. Build & deploy (same as build.sh)
echo
echo "🚀 Building RagBot with GPU-based embeddings..."
BUILD_VERSION="1.1.9"
BUILD_DATE=\$(date '+%Y-%m-%d %H:%M:%S')
VCS_REF="simplified-chroma-client"
echo "📦 Version: \${BUILD_VERSION}"
echo "📅 Build Date: \${BUILD_DATE}"
echo "🔧 VCS Ref: \${VCS_REF}"

# Handle --no-cache flag
NO_CACHE_FLAG=""
if [[ "\${1:-}" == "--no-cache" ]]; then
  echo "🔄 Using --no-cache (fresh build)"
  NO_CACHE_FLAG="--no-cache"
else
  echo "⚡ Using Docker cache (faster build)"
  echo "   💡 Pass '--no-cache' to force a fresh build"
fi

echo "🛑 Stopping existing containers..."
docker-compose down

echo "🏗️ Building Docker image..."
export DOCKER_BUILDKIT=1
docker-compose build \
  --build-arg BUILD_DATE="\${BUILD_DATE}" \
  --build-arg BUILD_VERSION="\${BUILD_VERSION}" \
  --build-arg VCS_REF="\${VCS_REF}" \
  \${NO_CACHE_FLAG}

echo "✅ Build complete!"
echo "🚀 Starting containers..."
docker-compose up -d

echo "📋 Checking container status..."
sleep 10
docker-compose ps

echo "📄 Viewing recent logs..."
docker logs ragbot_backend_1 --tail 40

echo "🎉 RagBot v\${BUILD_VERSION} is ready!"
echo "🌐 Frontend: http://localhost:50505"
echo "📊 To view logs live: docker logs ragbot_backend_1 -f"
EOF

# make it executable
chmod +x pull_and_build.sh
