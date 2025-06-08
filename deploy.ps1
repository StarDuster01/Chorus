#!/usr/bin/env pwsh

# Simple RAG Bot Deployment Script
Write-Host "Starting RAG Bot deployment..." -ForegroundColor Green

# Set error handling
$ErrorActionPreference = "Stop"

try {
    # Step 1: Build frontend
    Write-Host "Step 1: Building frontend..." -ForegroundColor Yellow
    Set-Location "./ragbot-frontend"
    npm install
    npm run build
    Write-Host "Frontend build completed" -ForegroundColor Green

    # Step 2: Copy to backend
    Write-Host "Step 2: Copying frontend to backend..." -ForegroundColor Yellow
    Set-Location "../ragbot-backend"
    Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
    Copy-Item -Recurse -Force ../ragbot-frontend/build ./build
    Write-Host "Frontend copied to backend" -ForegroundColor Green

    # Step 3: Build and push Docker image
    Write-Host "Step 3: Building Docker image..." -ForegroundColor Yellow
    docker build -t chorusproduction.azurecr.io/ragbot:gpu .
    Write-Host "Docker image built" -ForegroundColor Green
    
    Write-Host "Step 4: Pushing to registry..." -ForegroundColor Yellow
    docker push chorusproduction.azurecr.io/ragbot:gpu
    Write-Host "Image pushed to registry" -ForegroundColor Green

    # Step 4: Deploy to Kubernetes
    Write-Host "Step 5: Deploying to Kubernetes..." -ForegroundColor Yellow
    kubectl apply -f k8s/deployment.yaml
    kubectl rollout restart deployment/ragbot -n ragbot
    kubectl rollout status deployment/ragbot -n ragbot --timeout=300s
    Write-Host "Kubernetes deployment completed" -ForegroundColor Green

    Write-Host ""
    Write-Host "DEPLOYMENT SUCCESSFUL!" -ForegroundColor Green
    Write-Host "Check pod status: kubectl get pods -n ragbot" -ForegroundColor White
    Write-Host "View logs: kubectl logs -f deployment/ragbot -n ragbot" -ForegroundColor White

} catch {
    Write-Host ""
    Write-Host "DEPLOYMENT FAILED: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} 