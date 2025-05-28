# RagBot Local Single-Container Deployment Script
Write-Host "=== RagBot Local Single-Container Deployment ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Stop and Remove Existing Docker Container
Write-Host "Step 1: Stopping and removing existing Docker container..." -ForegroundColor Green
docker stop ragbot-local-instance 2>$null
docker rm ragbot-local-instance 2>$null
Write-Host "  Container cleanup completed" -ForegroundColor Green

# Step 2: Rebuild the React Frontend
Write-Host ""
Write-Host "Step 2: Rebuilding React frontend..." -ForegroundColor Green
Set-Location ragbot-frontend
npm install
npm run build
Set-Location ..
Write-Host "  React build completed" -ForegroundColor Green

# Step 3: Delete Existing Frontend Files in Backend
Write-Host ""
Write-Host "Step 3: Cleaning existing frontend files in backend..." -ForegroundColor Green
Remove-Item -Recurse -Force "ragbot-backend\frontend" -ErrorAction SilentlyContinue
Write-Host "  Old frontend files removed" -ForegroundColor Green

# Step 4: Copy New Frontend Build to Backend
Write-Host ""
Write-Host "Step 4: Copying new frontend build to backend..." -ForegroundColor Green
New-Item -ItemType Directory -Path "ragbot-backend\frontend" -Force | Out-Null
Copy-Item -Path "ragbot-frontend\build\*" -Destination "ragbot-backend\frontend\" -Recurse -Force
Write-Host "  Frontend files copied successfully" -ForegroundColor Green

# Step 5: Build the New Backend Docker Image
Write-Host ""
Write-Host "Step 5: Building new backend Docker image..." -ForegroundColor Green
Set-Location ragbot-backend
docker build -t ragbot-local-unified .
Set-Location ..
Write-Host "  Docker image built successfully" -ForegroundColor Green

# Step 6: Run the New Unified Docker Container
Write-Host ""
Write-Host "Step 6: Starting new unified Docker container..." -ForegroundColor Green
docker run -d -p 50505:50505 --name ragbot-local-instance ragbot-local-unified
Start-Sleep -Seconds 3
Write-Host "  Container started successfully" -ForegroundColor Green

Write-Host ""
Write-Host "=== DEPLOYMENT COMPLETED SUCCESSFULLY ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Application is now running at: http://localhost:50505" -ForegroundColor Yellow
Write-Host ""
Write-Host "Useful commands:" -ForegroundColor White
Write-Host "  - Check container status: docker ps" -ForegroundColor Gray
Write-Host "  - View container logs:   docker logs ragbot-local-instance" -ForegroundColor Gray
Write-Host "  - Stop container:        docker stop ragbot-local-instance" -ForegroundColor Gray
Write-Host "  - Remove container:      docker rm ragbot-local-instance" -ForegroundColor Gray

# Open browser
Start-Process "http://localhost:50505"