#!/bin/bash
# ========================================
# RagBot VM Deployment Script (Linux)
# ========================================
# This script is designed to run ON the Azure VM
# to rebuild and redeploy the application locally
# ========================================

echo "========================================"
echo "  RagBot VM Deployment Script"
echo "========================================"
echo ""

# ========================================
# Step 1: Check for .env file
# ========================================
ENV_PATH="ragbot-backend/.env"

if [ ! -f "$ENV_PATH" ]; then
    echo "⚠️  No .env file found!"
    echo "Let's create one now. Please provide the following information:"
    echo ""
    
    # JWT Secret
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "JWT_SECRET (required for authentication)"
    echo "Leave blank to auto-generate a secure random token"
    read -p "JWT_SECRET: " JWT_SECRET
    if [ -z "$JWT_SECRET" ]; then
        echo "Generating secure JWT secret..."
        JWT_SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
        echo "✓ Generated: $JWT_SECRET"
    fi
    
    # OpenAI API Key
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "OPENAI_API_KEY (required for chat, embeddings)"
    echo "Format: sk-proj-..."
    read -p "OPENAI_API_KEY: " OPENAI_API_KEY
    
    # OpenAI Image API Key
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "OPENAI_IMAGE_API_KEY (optional - for image generation)"
    echo "Leave blank to use main OPENAI_API_KEY"
    read -p "OPENAI_IMAGE_API_KEY: " OPENAI_IMAGE_API_KEY
    
    # Anthropic API Key
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "ANTHROPIC_API_KEY (optional - for Claude models)"
    echo "Leave blank to skip"
    read -p "ANTHROPIC_API_KEY: " ANTHROPIC_API_KEY
    
    # Groq API Key
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "GROQ_API_KEY (optional - for Groq models)"
    echo "Leave blank to skip"
    read -p "GROQ_API_KEY: " GROQ_API_KEY
    
    # Port
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "PORT (application port)"
    echo "Leave blank for default: 50506"
    read -p "PORT: " PORT
    if [ -z "$PORT" ]; then
        PORT="50506"
    fi
    
    # External Data Directory
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "EXTERNAL_DATA_DIR (optional - path to external data)"
    echo "Example: ../ChorusAllData2 or /mnt/ChorusAllData2"
    echo "Leave blank to skip"
    read -p "EXTERNAL_DATA_DIR: " EXTERNAL_DATA_DIR
    
    # Create .env file
    echo ""
    echo "Creating .env file..."
    
    cat > "$ENV_PATH" << EOF
# ===================================
# RagBot Environment Variables
# ===================================
# Auto-generated on $(date '+%Y-%m-%d %H:%M:%S')

# ===================================
# REQUIRED - JWT Secret for Authentication
# ===================================
JWT_SECRET=$JWT_SECRET

# ===================================
# OpenAI API Keys
# ===================================
# Main OpenAI API key for chat, embeddings, etc.
OPENAI_API_KEY=$OPENAI_API_KEY

# Separate key for image generation
OPENAI_IMAGE_API_KEY=$OPENAI_IMAGE_API_KEY

# ===================================
# Additional AI Provider Keys
# ===================================
# For Anthropic/Claude models in Model Chorus
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY

# For Groq models in Model Chorus
GROQ_API_KEY=$GROQ_API_KEY

# ===================================
# Configuration
# ===================================
# Port
PORT=$PORT

# External data directory
EXTERNAL_DATA_DIR=$EXTERNAL_DATA_DIR
EOF
    
    echo "✅ .env file created successfully!"
    echo ""
else
    echo "✅ Found existing .env file"
    echo ""
fi

# ========================================
# Step 2: Stop and Remove Existing Containers
# ========================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Stopping existing containers..."
echo ""

# Use docker-compose down to properly clean up all containers and networks
echo "  Running docker-compose down..."
docker-compose down 2>/dev/null || true

# Kill any lingering processes on port 50506
echo "  Checking for processes on port 50506..."
PIDS=$(sudo lsof -ti:50506 2>/dev/null)
if [ ! -z "$PIDS" ]; then
    echo "  Found processes: $PIDS, killing them..."
    sudo kill -9 $PIDS 2>/dev/null || true
    sleep 1
fi

echo "  ✓ Containers stopped and removed"
echo ""

# ========================================
# Step 3: Build React Frontend
# ========================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Building React frontend..."
echo ""

cd ragbot-frontend
npm install
if [ $? -ne 0 ]; then
    echo "❌ npm install failed!"
    cd ..
    exit 1
fi

npm run build
if [ $? -ne 0 ]; then
    echo "❌ npm build failed!"
    cd ..
    exit 1
fi

cd ..
echo "  ✓ Frontend build completed"
echo ""

# ========================================
# Step 4: Copy Frontend to Backend
# ========================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Copying frontend to backend..."
echo ""

rm -rf ragbot-backend/frontend
mkdir -p ragbot-backend/frontend
cp -r ragbot-frontend/build/* ragbot-backend/frontend/

echo "  ✓ Frontend files copied"
echo ""

# ========================================
# Step 5: Start Containers with Docker Compose
# ========================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4: Starting containers with Docker Compose..."
echo ""

docker-compose up -d --build
if [ $? -ne 0 ]; then
    echo "❌ Docker Compose failed!"
    exit 1
fi

echo "  ✓ Containers started successfully"
echo ""

# ========================================
# Step 6: Wait for Services to Start
# ========================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Waiting for services to initialize..."
sleep 5

# ========================================
# Deployment Complete
# ========================================
echo ""
echo "========================================"
echo "  ✅ DEPLOYMENT COMPLETED SUCCESSFULLY"
echo "========================================"
echo ""

# Get the port from .env or use default
APP_PORT=$(grep "^PORT=" "$ENV_PATH" | cut -d '=' -f2)
if [ -z "$APP_PORT" ]; then
    APP_PORT="50506"
fi

echo "📡 Application URLs:"
echo "   Backend API:  http://localhost:$APP_PORT"
echo "   Frontend:     http://localhost:80"
echo ""

echo "📋 Useful Commands:"
echo "   View logs (backend):  docker-compose logs -f backend"
echo "   View logs (frontend): docker-compose logs -f frontend"
echo "   Stop services:        docker-compose down"
echo "   Restart services:     docker-compose restart"
echo "   View status:          docker-compose ps"
echo ""

echo "🔍 Checking container status..."
docker-compose ps
echo ""
