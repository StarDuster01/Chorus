#!/bin/bash

# Chorus RagBot v2.0 - Azure Deployment Script (Linux/Bash)
# This script builds and deploys the RagBot v2.0 application to Azure Container Apps
# Deploys as a separate instance alongside v1.0

set -e  # Exit on any error

# Configuration
CONTAINER_APP_NAME="chorusbotfull-v2"
IMAGE_TAG="v2.0"
IMAGE_NAME="viriditytech.azurecr.io/ragbot-unified:${IMAGE_TAG}"

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
echo "Building image: ${IMAGE_NAME}"
cd ragbot-backend
docker build -t ${IMAGE_NAME} .
docker push ${IMAGE_NAME}
cd ..

echo ""
echo "=== STEP 6: Deploying to Azure Container App ==="
echo "Checking if container app '${CONTAINER_APP_NAME}' exists..."

# Check if container app exists
if az containerapp show --name ${CONTAINER_APP_NAME} --resource-group viridity_tech &> /dev/null; then
    echo "Container app exists. Updating..."
    az containerapp registry set --name ${CONTAINER_APP_NAME} --resource-group viridity_tech --server viriditytech.azurecr.io --identity system
    
    az containerapp update \
        --name ${CONTAINER_APP_NAME} \
        --resource-group viridity_tech \
        --image ${IMAGE_NAME} \
        --set-env-vars PORT=50506 EXTERNAL_DATA_DIR=/mnt/ChorusAllData2
    
    az containerapp ingress update \
        --name ${CONTAINER_APP_NAME} \
        --resource-group viridity_tech \
        --target-port 50506
else
    echo "Container app does not exist. Creating new container app..."
    
    # Get the environment name from the existing v1.0 app
    ENV_NAME=$(az containerapp show --name chorusbotfull --resource-group viridity_tech --query "properties.environmentId" -o tsv | rev | cut -d'/' -f1 | rev)
    
    az containerapp create \
        --name ${CONTAINER_APP_NAME} \
        --resource-group viridity_tech \
        --environment ${ENV_NAME} \
        --image ${IMAGE_NAME} \
        --registry-server viriditytech.azurecr.io \
        --registry-identity system \
        --target-port 50506 \
        --ingress external \
        --env-vars PORT=50506 EXTERNAL_DATA_DIR=/mnt/ChorusAllData2 \
        --cpu 2.0 \
        --memory 4.0Gi \
        --min-replicas 1 \
        --max-replicas 3
    
    echo "Container app created successfully!"
fi

echo ""
echo "=== DEPLOYMENT COMPLETE! ==="
echo ""
echo "Application Configuration:"
echo "  - Container App: ${CONTAINER_APP_NAME}"
echo "  - Image: ${IMAGE_NAME}"
echo "  - Port: 50506"
echo "  - External Data Directory: /mnt/ChorusAllData2"
echo ""

# Get the application URL
APP_URL=$(az containerapp show --name ${CONTAINER_APP_NAME} --resource-group viridity_tech --query "properties.configuration.ingress.fqdn" -o tsv)
echo "Application URL: https://${APP_URL}"
echo ""
echo "Deployed Versions:"
echo "  - v1.0: https://chorusbotfull.wonderfulocean-4a90a562.westus2.azurecontainerapps.io (port 50505)"
echo "  - v2.0: https://${APP_URL} (port 50506)"
echo ""
echo "To view deployment status:"
echo "  az containerapp revision list --name ${CONTAINER_APP_NAME} --resource-group viridity_tech --output table"
echo ""
echo "To view logs:"
echo "  az containerapp logs show --name ${CONTAINER_APP_NAME} --resource-group viridity_tech --follow"
