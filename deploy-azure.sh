#!/bin/bash

# Chorus RagBot v2.0 - Azure Deployment Script (Linux/Bash)
# This script builds and deploys the RagBot application to Azure Container Apps

set -e  # Exit on any error

echo "=== STEP 1: Azure Login and Setup ==="
az login
az account set --subscription "Azure subscription 1"
az account show

echo ""
echo "=== STEP 2: Building React Frontend ==="
cd ragbot-frontend
npm install
npm run build
cd ..

echo ""
echo "=== STEP 3: Copying Frontend to Backend ==="
if [ -d "ragbot-backend/frontend" ]; then
    rm -rf ragbot-backend/frontend
fi
mkdir -p ragbot-backend/frontend
cp -r ragbot-frontend/build/* ragbot-backend/frontend/

echo ""
echo "=== STEP 4: Setting up Azure Container Registry ==="
az acr update --name viriditytech --resource-group viridity_tech --admin-enabled true
az acr credential show --name viriditytech --resource-group viridity_tech
az acr login --name viriditytech

echo ""
echo "=== STEP 5: Building and Pushing Docker Image ==="
cd ragbot-backend
docker build -t viriditytech.azurecr.io/ragbot-unified:latest .
docker push viriditytech.azurecr.io/ragbot-unified:latest
cd ..

echo ""
echo "=== STEP 6: Updating Azure Container App ==="
echo "Setting container registry..."
az containerapp registry set --name chorusbotfull --resource-group viridity_tech --server viriditytech.azurecr.io --identity system

echo "Updating container image and environment variables..."
az containerapp update --name chorusbotfull --resource-group viridity_tech --image viriditytech.azurecr.io/ragbot-unified:latest --set-env-vars PORT=50506 EXTERNAL_DATA_DIR=/mnt/ChorusAllData2

echo "Updating ingress port configuration..."
az containerapp ingress update --name chorusbotfull --resource-group viridity_tech --target-port 50506

echo ""
echo "=== DEPLOYMENT COMPLETE! ==="
echo ""
echo "Application Configuration:"
echo "  - Port: 50506"
echo "  - External Data Directory: /mnt/ChorusAllData2"
echo ""
echo "Application URL: https://chorusbotfull.wonderfulocean-4a90a562.westus2.azurecontainerapps.io"
echo ""
echo "To view deployment status:"
echo "  az containerapp revision list --name chorusbotfull --resource-group viridity_tech --output table"
echo ""
echo "To view logs:"
echo "  az containerapp logs show --name chorusbotfull --resource-group viridity_tech --follow"
