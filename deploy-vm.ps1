#!/usr/bin/env pwsh
# ========================================
# RagBot VM Deployment Script
# ========================================
# This script is designed to run ON the Azure VM
# to rebuild and redeploy the application locally
# ========================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  RagBot VM Deployment Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ========================================
# Step 1: Check for .env file
# ========================================
$envPath = "ragbot-backend\.env"

if (-not (Test-Path $envPath)) {
    Write-Host "⚠️  No .env file found!" -ForegroundColor Yellow
    Write-Host "Let's create one now. Please provide the following information:" -ForegroundColor Yellow
    Write-Host ""
    
    # JWT Secret
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
    Write-Host "JWT_SECRET (required for authentication)" -ForegroundColor Green
    Write-Host "Leave blank to auto-generate a secure random token" -ForegroundColor Gray
    $jwtSecret = Read-Host "JWT_SECRET"
    if ([string]::IsNullOrWhiteSpace($jwtSecret)) {
        Write-Host "Generating secure JWT secret..." -ForegroundColor Cyan
        $jwtSecret = python -c "import secrets; print(secrets.token_urlsafe(32))"
        Write-Host "✓ Generated: $jwtSecret" -ForegroundColor Green
    }
    
    # OpenAI API Key
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
    Write-Host "OPENAI_API_KEY (required for chat, embeddings)" -ForegroundColor Green
    Write-Host "Format: sk-proj-..." -ForegroundColor Gray
    $openaiKey = Read-Host "OPENAI_API_KEY"
    
    # OpenAI Image API Key
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
    Write-Host "OPENAI_IMAGE_API_KEY (optional - for image generation)" -ForegroundColor Green
    Write-Host "Leave blank to use main OPENAI_API_KEY" -ForegroundColor Gray
    $openaiImageKey = Read-Host "OPENAI_IMAGE_API_KEY"
    
    # Anthropic API Key
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
    Write-Host "ANTHROPIC_API_KEY (optional - for Claude models)" -ForegroundColor Green
    Write-Host "Leave blank to skip" -ForegroundColor Gray
    $anthropicKey = Read-Host "ANTHROPIC_API_KEY"
    
    # Groq API Key
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
    Write-Host "GROQ_API_KEY (optional - for Groq models)" -ForegroundColor Green
    Write-Host "Leave blank to skip" -ForegroundColor Gray
    $groqKey = Read-Host "GROQ_API_KEY"
    
    # Port
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
    Write-Host "PORT (application port)" -ForegroundColor Green
    Write-Host "Leave blank for default: 50506" -ForegroundColor Gray
    $port = Read-Host "PORT"
    if ([string]::IsNullOrWhiteSpace($port)) {
        $port = "50506"
    }
    
    # External Data Directory
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
    Write-Host "EXTERNAL_DATA_DIR (optional - path to external data)" -ForegroundColor Green
    Write-Host "Example: ../ChorusAllData2 or /mnt/ChorusAllData2" -ForegroundColor Gray
    Write-Host "Leave blank to skip" -ForegroundColor Gray
    $externalDataDir = Read-Host "EXTERNAL_DATA_DIR"
    
    # Create .env file
    Write-Host ""
    Write-Host "Creating .env file..." -ForegroundColor Cyan
    
    $envContent = @"
# ===================================
# RagBot Environment Variables
# ===================================
# Auto-generated on $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

# ===================================
# REQUIRED - JWT Secret for Authentication
# ===================================
JWT_SECRET=$jwtSecret

# ===================================
# OpenAI API Keys
# ===================================
# Main OpenAI API key for chat, embeddings, etc.
OPENAI_API_KEY=$openaiKey

# Separate key for image generation
OPENAI_IMAGE_API_KEY=$openaiImageKey

# ===================================
# Additional AI Provider Keys
# ===================================
# For Anthropic/Claude models in Model Chorus
ANTHROPIC_API_KEY=$anthropicKey

# For Groq models in Model Chorus
GROQ_API_KEY=$groqKey

# ===================================
# Configuration
# ===================================
# Port
PORT=$port

# External data directory
EXTERNAL_DATA_DIR=$externalDataDir
"@
    
    $envContent | Out-File -FilePath $envPath -Encoding utf8 -Force
    Write-Host "✅ .env file created successfully!" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "✅ Found existing .env file" -ForegroundColor Green
    Write-Host ""
}

# ========================================
# Step 2: Stop and Remove Existing Containers
# ========================================
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host "Step 1: Stopping existing containers..." -ForegroundColor Cyan
Write-Host ""

# Stop and remove the backend container
Write-Host "  Stopping ragbot-backend container..." -ForegroundColor White
docker stop ragbot-backend 2>$null
docker rm ragbot-backend 2>$null

# Stop and remove the frontend container
Write-Host "  Stopping ragbot-frontend container..." -ForegroundColor White
docker stop ragbot-frontend 2>$null
docker rm ragbot-frontend 2>$null

Write-Host "  ✓ Containers stopped and removed" -ForegroundColor Green
Write-Host ""

# ========================================
# Step 3: Build React Frontend
# ========================================
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host "Step 2: Building React frontend..." -ForegroundColor Cyan
Write-Host ""

Set-Location ragbot-frontend
npm install
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ npm install failed!" -ForegroundColor Red
    Set-Location ..
    exit 1
}

npm run build
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ npm build failed!" -ForegroundColor Red
    Set-Location ..
    exit 1
}

Set-Location ..
Write-Host "  ✓ Frontend build completed" -ForegroundColor Green
Write-Host ""

# ========================================
# Step 4: Copy Frontend to Backend
# ========================================
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host "Step 3: Copying frontend to backend..." -ForegroundColor Cyan
Write-Host ""

Remove-Item -Recurse -Force "ragbot-backend\frontend" -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path "ragbot-backend\frontend" -Force | Out-Null
Copy-Item -Path "ragbot-frontend\build\*" -Destination "ragbot-backend\frontend\" -Recurse -Force

Write-Host "  ✓ Frontend files copied" -ForegroundColor Green
Write-Host ""

# ========================================
# Step 5: Start Containers with Docker Compose
# ========================================
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host "Step 4: Starting containers with Docker Compose..." -ForegroundColor Cyan
Write-Host ""

docker-compose up -d --build
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker Compose failed!" -ForegroundColor Red
    exit 1
}

Write-Host "  ✓ Containers started successfully" -ForegroundColor Green
Write-Host ""

# ========================================
# Step 6: Wait for Services to Start
# ========================================
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host "Waiting for services to initialize..." -ForegroundColor Cyan
Start-Sleep -Seconds 5

# ========================================
# Deployment Complete
# ========================================
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  ✅ DEPLOYMENT COMPLETED SUCCESSFULLY" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Get the port from .env or use default
$envVars = Get-Content $envPath | ForEach-Object {
    if ($_ -match '^PORT=(.+)$') {
        return $matches[1]
    }
}
$appPort = if ($envVars) { $envVars } else { "50506" }

Write-Host "📡 Application URLs:" -ForegroundColor Cyan
Write-Host "   Backend API:  http://localhost:$appPort" -ForegroundColor White
Write-Host "   Frontend:     http://localhost:80" -ForegroundColor White
Write-Host ""

Write-Host "📋 Useful Commands:" -ForegroundColor Cyan
Write-Host "   View logs (backend):  docker logs -f ragbot-backend" -ForegroundColor Gray
Write-Host "   View logs (frontend): docker logs -f ragbot-frontend" -ForegroundColor Gray
Write-Host "   Stop services:        docker-compose down" -ForegroundColor Gray
Write-Host "   Restart services:     docker-compose restart" -ForegroundColor Gray
Write-Host "   View status:          docker-compose ps" -ForegroundColor Gray
Write-Host ""

Write-Host "🔍 Checking container status..." -ForegroundColor Cyan
docker-compose ps
Write-Host ""
