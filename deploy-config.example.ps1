# Azure Deployment Configuration Template (PowerShell)
# Copy this file to deploy-config.ps1 and fill in your actual values
# deploy-config.ps1 is gitignored and will not be committed

# Azure Subscription
$env:AZURE_SUBSCRIPTION = "your-subscription-name"

# Azure Container Registry
$env:AZURE_REGISTRY = "your-registry.azurecr.io"
$env:AZURE_RESOURCE_GROUP = "your-resource-group"

# Container App Configuration
$env:CONTAINER_APP_NAME = "your-app-name"
$env:IMAGE_TAG = "v2.0"

# Environment (optional)
$env:AZURE_LOCATION = "eastus"
$env:CONTAINER_APP_ENV = "your-environment-name"

# To use this config:
# 1. Copy this file: Copy-Item deploy-config.example.ps1 deploy-config.ps1
# 2. Edit deploy-config.ps1 with your actual values
# 3. Run it before your deployment script: .\deploy-config.ps1
