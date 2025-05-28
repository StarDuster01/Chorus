# RagBot: Unified Container Deployment Guide

## 1. Prerequisites

- Docker installed
- Node.js and npm installed
- Azure account with Azure CLI installed (`az`)

## 2. Project Structure

```
ragbot-backend/      # Flask backend + serves React frontend
ragbot-frontend/     # React frontend (builds into backend)
deploy-local.ps1     # Local deployment script
```

## 3. Architecture

**Unified Container Design:**
- Single Docker container runs Flask backend
- Flask serves the React frontend as static files
- Frontend build is copied into backend before Docker build
- One container = Complete application

## 4. Local Development & Testing

### Quick Local Deployment
Run the automated script:
```powershell
.\deploy-local.ps1
```

This script automatically:
1. Stops/removes existing container
2. Builds React frontend (`npm install` + `npm run build`)
3. Copies frontend build to backend
4. Builds unified Docker image
5. Runs container on `http://localhost:50505`

### Manual Local Deployment
If you prefer manual steps:

```bash
# 1. Build React frontend
cd ragbot-frontend
npm install
npm run build
cd ..

# 2. Copy frontend to backend
Remove-Item -Recurse -Force "ragbot-backend\frontend" -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path "ragbot-backend\frontend" -Force
Copy-Item -Path "ragbot-frontend\build\*" -Destination "ragbot-backend\frontend\" -Recurse -Force

# 3. Build and run unified container
cd ragbot-backend
docker build -t ragbot-local-unified .
cd ..
docker run -d -p 50505:50505 --name ragbot-local-instance ragbot-local-unified
```

## 5. Azure Deployment (Manual Steps)

### Step 1: Azure Login and Setup
```powershell
# Login to Azure
az login

# Set your subscription (replace with your subscription name/ID)
az account set --subscription "Azure subscription 1"

# Verify login
az account show
```

### Step 2: Build React Frontend
```powershell
# Navigate to frontend directory
cd ragbot-frontend

# Install dependencies and build
npm install
npm run build

# Return to root directory
cd ..
```

### Step 3: Copy Frontend to Backend
```powershell
# Remove existing frontend directory in backend
if (Test-Path "ragbot-backend\frontend") { Remove-Item -Recurse -Force "ragbot-backend\frontend" }

# Create new frontend directory
New-Item -ItemType Directory -Path "ragbot-backend\frontend" -Force

# Copy built frontend files
Copy-Item -Path "ragbot-frontend\build\*" -Destination "ragbot-backend\frontend\" -Recurse -Force
```

### Step 4: Setup Azure Container Registry (ACR)
```powershell
# Enable admin user for ACR
az acr update --name viriditytech --resource-group viridity_tech --admin-enabled true

# Get ACR credentials (optional - for verification)
az acr credential show --name viriditytech --resource-group viridity_tech

# Login to ACR
az acr login --name viriditytech
```

### Step 5: Build and Push Docker Image
```powershell
# Navigate to backend directory
cd ragbot-backend

# Build Docker image with ACR tag
docker build -t viriditytech.azurecr.io/ragbot-unified:latest .

# Push image to ACR
docker push viriditytech.azurecr.io/ragbot-unified:latest

# Return to root directory
cd ..
```

### Step 6: Update Azure Container App
```powershell
# First, configure the registry authentication using managed identity
az containerapp registry set --name chorusbotfull --resource-group viridity_tech --server viriditytech.azurecr.io --identity system

# Then update the container app with the new image
az containerapp update --name chorusbotfull --resource-group viridity_tech --image viriditytech.azurecr.io/ragbot-unified:latest
```

### Step 7: Monitor Deployment
```powershell
# Check deployment status
az containerapp revision list --name chorusbotfull --resource-group viridity_tech --output table

# Get application URL
az containerapp show --name chorusbotfull --resource-group viridity_tech --query "properties.configuration.ingress.fqdn" --output tsv

# View logs (optional)
az containerapp logs show --name chorusbotfull --resource-group viridity_tech --follow
```

### Azure Resources Used
- **Resource Group**: `viridity_tech`
- **Container App**: `chorusbotfull`
- **Container Registry**: `viriditytech.azurecr.io`
- **Environment**: `viriditytech`

## 6. Key Features

### Tesseract Loading Animations
- 3D tesseract animations using Three.js
- Dark green color for visibility
- Used in upload buttons, chat interface, and loading states

### Single Container Benefits
- ✅ Simplified deployment
- ✅ No cross-origin issues
- ✅ Single point of configuration
- ✅ Easier scaling and management

## 7. Troubleshooting

### Container Issues
```bash
# Check if container is running
docker ps

# View container logs
docker logs ragbot-local-instance

# Stop and remove container
docker stop ragbot-local-instance
docker rm ragbot-local-instance
```

### Azure Issues
```bash
# Check Container App status
az containerapp show --name chorusbotfull --resource-group viridity_tech

# View Container App logs
az containerapp logs show --name chorusbotfull --resource-group viridity_tech --follow

# Check revisions
az containerapp revision list --name chorusbotfull --resource-group viridity_tech --output table

# Check ACR images
az acr repository list --name viriditytech --output table
az acr repository show-tags --name viriditytech --repository ragbot-unified --output table
```

### Common Solutions

**"Container name already in use":**
```bash
docker stop ragbot-local-instance
docker rm ragbot-local-instance
```

**Azure authentication errors:**
```bash
# Re-login to Azure
az login

# Re-login to ACR
az acr login --name viriditytech
```

**Frontend changes not appearing:**
- Ensure you rebuild the frontend: `cd ragbot-frontend && npm run build`
- Copy frontend to backend before Docker build
- Build and push new Docker image with updated tag

**Docker push fails:**
```bash
# Check if logged into ACR
az acr login --name viriditytech

# Verify image name matches ACR format
docker images | grep viriditytech.azurecr.io
```

## 8. Development Workflow

### For Local Testing:
1. Make changes to React frontend or Flask backend
2. Run `.\deploy-local.ps1`
3. Test at `http://localhost:50505`

### For Azure Deployment:
1. Test locally first with `.\deploy-local.ps1`
2. Follow manual Azure deployment steps (Steps 1-7 above)
3. Application available at: `https://chorusbotfull.wonderfulocean-4a90a562.westus2.azurecontainerapps.io`

## 9. Application URLs

- **Local**: `http://localhost:50505`
- **Azure**: `https://chorusbotfull.wonderfulocean-4a90a562.westus2.azurecontainerapps.io`

## 10. Quick Reference Commands

### Complete Azure Deployment (Copy-Paste Ready)
```powershell
# 1. Login and setup
az login
az account set --subscription "Azure subscription 1"

# 2. Build frontend
cd ragbot-frontend; npm install; npm run build; cd ..

# 3. Copy frontend to backend
if (Test-Path "ragbot-backend\frontend") { Remove-Item -Recurse -Force "ragbot-backend\frontend" }
New-Item -ItemType Directory -Path "ragbot-backend\frontend" -Force
Copy-Item -Path "ragbot-frontend\build\*" -Destination "ragbot-backend\frontend\" -Recurse -Force

# 4. Setup ACR and build/push image
az acr update --name viriditytech --resource-group viridity_tech --admin-enabled true
az acr login --name viriditytech
cd ragbot-backend
docker build -t viriditytech.azurecr.io/ragbot-unified:latest .
docker push viriditytech.azurecr.io/ragbot-unified:latest
cd ..

# 5. Update container app
az containerapp registry set --name chorusbotfull --resource-group viridity_tech --server viriditytech.azurecr.io --identity system
az containerapp update --name chorusbotfull --resource-group viridity_tech --image viriditytech.azurecr.io/ragbot-unified:latest

# 6. Get application URL
az containerapp show --name chorusbotfull --resource-group viridity_tech --query "properties.configuration.ingress.fqdn" --output tsv
```

---

**Note**: This application uses a unified container architecture where the Flask backend serves the React frontend. The manual deployment process ensures you have full control over each step and can troubleshoot issues as they arise. 