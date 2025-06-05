# Login to Azure
Write-Host "Logging into Azure..."
az login

# Set subscription
Write-Host "Setting subscription..."
az account set --subscription "Azure subscription 1"

# Login to ACR
Write-Host "Logging into Azure Container Registry..."
az acr login --name chorusproduction

# Build frontend
Write-Host "Building frontend..."
cd ../ragbot-frontend
npm install
npm run build
cd ../ragbot-backend

# Copy frontend build
Write-Host "Copying frontend build..."
if (Test-Path "frontend") { Remove-Item -Recurse -Force "frontend" }
New-Item -ItemType Directory -Path "frontend" -Force
Copy-Item -Path "../ragbot-frontend/build/*" -Destination "frontend/" -Recurse -Force

# Build and push Docker image
Write-Host "Building and pushing Docker image..."
docker build -t chorusproduction.azurecr.io/ragbot:latest .
docker push chorusproduction.azurecr.io/ragbot:latest

# Create Kubernetes secrets
Write-Host "Creating Kubernetes secrets..."
$secretKey = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes((New-Guid).ToString()))
kubectl create namespace ragbot --dry-run=client -o yaml | kubectl apply -f -
kubectl create secret generic ragbot-secrets --namespace ragbot --from-literal=secret-key=$secretKey --dry-run=client -o yaml | kubectl apply -f -

# Apply Kubernetes configurations
Write-Host "Applying Kubernetes configurations..."
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml

# Wait for deployment
Write-Host "Waiting for deployment to complete..."
kubectl rollout status deployment/ragbot -n ragbot

# Get service URL
Write-Host "Getting service URL..."
$serviceIP = kubectl get service ragbot -n ragbot -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
Write-Host "Application is available at: http://$serviceIP" 