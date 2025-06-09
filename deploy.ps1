#!/usr/bin/env pwsh

# Simple RAG Bot Deployment Script
Write-Host "Starting RAG Bot deployment..." -ForegroundColor Green

# Set error handling
$ErrorActionPreference = "Stop"

try {
    # Step 0: Azure login and authentication
    Write-Host "Step 0: Logging into Azure..." -ForegroundColor Yellow
    az login
    Write-Host "Azure login completed" -ForegroundColor Green
    
    Write-Host "Logging into Azure Container Registry..." -ForegroundColor Yellow
    az acr login --name chorusproduction
    Write-Host "ACR login completed" -ForegroundColor Green
    
    # Step 1: Build frontend
    Write-Host "Step 1: Building frontend..." -ForegroundColor Yellow
    Set-Location "./ragbot-frontend"
    npm install
    npm run build
    Write-Host "Frontend build completed" -ForegroundColor Green

    # Step 2: Copy to backend
    Write-Host "Step 2: Copying frontend to backend..." -ForegroundColor Yellow
    Set-Location "../ragbot-backend"
    Remove-Item -Recurse -Force frontend -ErrorAction SilentlyContinue
    Copy-Item -Recurse -Force ../ragbot-frontend/build ./frontend
    Write-Host "Frontend copied to backend" -ForegroundColor Green

    # Step 3: Build and push Docker image
    Write-Host "Step 3: Building Docker image..." -ForegroundColor Yellow
    docker build -t chorusproduction.azurecr.io/ragbot:gpu .
    Write-Host "Docker image built" -ForegroundColor Green
    
    Write-Host "Step 4: Pushing to registry..." -ForegroundColor Yellow
    docker push chorusproduction.azurecr.io/ragbot:gpu
    Write-Host "Image pushed to registry" -ForegroundColor Green

    # Step 5: Deploy to Kubernetes
    Write-Host "Step 5: Deploying to Kubernetes..." -ForegroundColor Yellow
    
    # Delete any existing pods in Error state
    Write-Host "Cleaning up any failed pods..." -ForegroundColor Yellow
    kubectl delete pod -n ragbot -l app=ragbot --field-selector status.phase=Failed --force --grace-period=0
    
    # Apply the deployment
    kubectl apply -f k8s/deployment.yaml
    
    # Restart the deployment
    Write-Host "Restarting deployment..." -ForegroundColor Yellow
    kubectl rollout restart deployment/ragbot -n ragbot
    
    # Wait for rollout with increased timeout
    Write-Host "Waiting for rollout to complete (timeout: 10 minutes)..." -ForegroundColor Yellow
    kubectl rollout status deployment/ragbot -n ragbot --timeout=600s
    
    # Verify pod status
    Write-Host "Verifying pod status..." -ForegroundColor Yellow
    $pods = kubectl get pods -n ragbot -l app=ragbot -o json | ConvertFrom-Json
    $allReady = $true
    foreach ($pod in $pods.items) {
        if ($pod.status.phase -ne "Running" -or $pod.status.containerStatuses[0].ready -ne $true) {
            $allReady = $false
            Write-Host "Pod $($pod.metadata.name) is not ready. Status: $($pod.status.phase)" -ForegroundColor Red
            Write-Host "Pod logs:" -ForegroundColor Yellow
            kubectl logs -n ragbot $($pod.metadata.name) --tail=50
        }
    }
    
    if ($allReady) {
        Write-Host "All pods are running and ready!" -ForegroundColor Green
    } else {
        Write-Host "Some pods are not ready. Please check the logs above." -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "DEPLOYMENT COMPLETED!" -ForegroundColor Green
    Write-Host "Check pod status: kubectl get pods -n ragbot" -ForegroundColor White
    Write-Host "View logs: kubectl logs -f deployment/ragbot -n ragbot" -ForegroundColor White

} catch {
    Write-Host ""
    Write-Host "DEPLOYMENT FAILED: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} 