#!/bin/bash

# Azure Deployment Configuration Template
# Copy this file to deploy-config.sh and fill in your actual values
# deploy-config.sh is gitignored and will not be committed

# Azure Subscription
export AZURE_SUBSCRIPTION="your-subscription-name"

# Azure Container Registry
export AZURE_REGISTRY="your-registry.azurecr.io"
export AZURE_RESOURCE_GROUP="your-resource-group"

# Container App Configuration
export CONTAINER_APP_NAME="your-app-name"
export IMAGE_TAG="v2.0"

# Environment (optional)
export AZURE_LOCATION="eastus"
export CONTAINER_APP_ENV="your-environment-name"

# To use this config:
# 1. Copy this file: cp deploy-config.example.sh deploy-config.sh
# 2. Edit deploy-config.sh with your actual values
# 3. Source it in your deployment script: source deploy-config.sh
