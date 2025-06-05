# Login to Azure
Write-Host "Logging into Azure..."
az login

# Set subscription
Write-Host "Setting subscription..."
az account set --subscription "Azure subscription 1"

# Login to ACR
Write-Host "Logging into Azure Container Registry..."
az acr login --name chorusproduction

# Build and push Docker image
Write-Host "Building and pushing Docker image..."
docker build -t chorusproduction.azurecr.io/ragbot:latest .
docker push chorusproduction.azurecr.io/ragbot:latest

# Ensure kubectl is configured with the correct context
Write-Host "Configuring kubectl context..."
az aks get-credentials --resource-group viridity_tech --name chorus-production

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

# Display deployment status
Write-Host "`nDeployment Status:"
kubectl get pods -n ragbot
kubectl get services -n ragbot 