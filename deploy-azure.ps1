Write-Host "=== STEP 1: Azure Login and Setup ===" -ForegroundColor Green
az login
az account set --subscription "Azure subscription 1"
az account show

Write-Host "=== STEP 2: Building React Frontend ===" -ForegroundColor Green
cd ragbot-frontend
npm install
npm run build
cd ..

Write-Host "=== STEP 3: Copying Frontend to Backend ===" -ForegroundColor Green
if (Test-Path "ragbot-backend\frontend") { Remove-Item -Recurse -Force "ragbot-backend\frontend" }
New-Item -ItemType Directory -Path "ragbot-backend\frontend" -Force
Copy-Item -Path "ragbot-frontend\build\*" -Destination "ragbot-backend\frontend\" -Recurse -Force

Write-Host "=== STEP 4: Setting up Azure Container Registry ===" -ForegroundColor Green
az acr update --name viriditytech --resource-group viridity_tech --admin-enabled true
az acr credential show --name viriditytech --resource-group viridity_tech
az acr login --name viriditytech

Write-Host "=== STEP 5: Building and Pushing Docker Image ===" -ForegroundColor Green
cd ragbot-backend
docker build -t viriditytech.azurecr.io/ragbot-unified:latest .
docker push viriditytech.azurecr.io/ragbot-unified:latest
cd ..

Write-Host "=== STEP 6: Updating Azure Container App ===" -ForegroundColor Green
Write-Host "Setting container registry..." -ForegroundColor Cyan
az containerapp registry set --name chorusbotfull --resource-group viridity_tech --server viriditytech.azurecr.io --identity system

Write-Host "Updating container image and environment variables..." -ForegroundColor Cyan
az containerapp update --name chorusbotfull --resource-group viridity_tech --image viriditytech.azurecr.io/ragbot-unified:latest --set-env-vars PORT=50506 EXTERNAL_DATA_DIR=/mnt/ChorusAllData2

Write-Host "Updating ingress port configuration..." -ForegroundColor Cyan
az containerapp ingress update --name chorusbotfull --resource-group viridity_tech --target-port 50506

Write-Host "=== DEPLOYMENT COMPLETE! ===" -ForegroundColor Green
Write-Host "" -ForegroundColor White
Write-Host "Application Configuration:" -ForegroundColor Cyan
Write-Host "  - Port: 50506" -ForegroundColor White
Write-Host "  - External Data Directory: /mnt/ChorusAllData2" -ForegroundColor White
Write-Host "" -ForegroundColor White
Write-Host "Application URL: https://chorusbotfull.wonderfulocean-4a90a562.westus2.azurecontainerapps.io" -ForegroundColor Yellow
Write-Host "" -ForegroundColor White
Write-Host "To view deployment status:" -ForegroundColor Cyan
Write-Host "  az containerapp revision list --name chorusbotfull --resource-group viridity_tech --output table" -ForegroundColor Gray
Write-Host "" -ForegroundColor White
Write-Host "To view logs:" -ForegroundColor Cyan
Write-Host "  az containerapp logs show --name chorusbotfull --resource-group viridity_tech --follow" -ForegroundColor Gray 