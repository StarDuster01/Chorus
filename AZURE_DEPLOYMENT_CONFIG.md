# Azure Deployment Configuration Guide

## Overview
This guide explains how to deploy RagBot to Azure with custom port and external data directory configuration.

## Configuration Changes from v1.0 to v2.0

### 1. Port Configuration
- **Old Port**: 50505
- **New Port**: 50506 (configurable via environment variable)

### 2. External Data Directory Support
- **Purpose**: Mount and use external data directories on Azure VM (e.g., ChorusAllData2)
- **Configuration**: Set via `EXTERNAL_DATA_DIR` environment variable

## Azure Container App Configuration

### Setting Environment Variables

When deploying to Azure Container Apps, you need to set the following environment variables:

```bash
# Set the port (default is 50506 if not specified)
az containerapp update \
  --name chorusbotfull \
  --resource-group viridity_tech \
  --set-env-vars PORT=50506

# Set external data directory (for ChorusAllData2)
az containerapp update \
  --name chorusbotfull \
  --resource-group viridity_tech \
  --set-env-vars EXTERNAL_DATA_DIR=/mnt/ChorusAllData2
```

### Complete Deployment Command with Environment Variables

```bash
az containerapp update \
  --name chorusbotfull \
  --resource-group viridity_tech \
  --image viriditytech.azurecr.io/ragbot-unified:latest \
  --set-env-vars \
    PORT=50506 \
    EXTERNAL_DATA_DIR=/mnt/ChorusAllData2
```

## Azure VM File Share Mount

If you're mounting an Azure File Share to `/mnt/ChorusAllData2`:

### 1. Create or Verify File Share Mount
```bash
# List existing mounts
az containerapp show \
  --name chorusbotfull \
  --resource-group viridity_tech \
  --query "properties.template.volumes"
```

### 2. Add Azure File Share as Volume
```bash
# Add storage mount
az containerapp env storage set \
  --name <your-environment-name> \
  --resource-group viridity_tech \
  --storage-name chorusdata2 \
  --azure-file-account-name <storage-account-name> \
  --azure-file-account-key <storage-account-key> \
  --azure-file-share-name ChorusAllData2 \
  --access-mode ReadWrite
```

### 3. Mount Volume to Container
Update your container app to mount the volume:

```bash
az containerapp update \
  --name chorusbotfull \
  --resource-group viridity_tech \
  --set-env-vars \
    PORT=50506 \
    EXTERNAL_DATA_DIR=/mnt/ChorusAllData2
```

## Port Configuration in Azure Container Apps

### Update Ingress Port
```bash
# Update the target port for ingress
az containerapp ingress update \
  --name chorusbotfull \
  --resource-group viridity_tech \
  --target-port 50506
```

### Verify Ingress Configuration
```bash
az containerapp ingress show \
  --name chorusbotfull \
  --resource-group viridity_tech
```

## Complete Deployment Process for Azure (Updated)

### Step 1: Build and Push Docker Image
```powershell
cd ragbot-backend
docker build -t viriditytech.azurecr.io/ragbot-unified:latest .
docker push viriditytech.azurecr.io/ragbot-unified:latest
cd ..
```

### Step 2: Update Container App with New Configuration
```bash
# Update the container app with new image and environment variables
az containerapp update \
  --name chorusbotfull \
  --resource-group viridity_tech \
  --image viriditytech.azurecr.io/ragbot-unified:latest \
  --set-env-vars \
    PORT=50506 \
    EXTERNAL_DATA_DIR=/mnt/ChorusAllData2

# Update ingress to use new port
az containerapp ingress update \
  --name chorusbotfull \
  --resource-group viridity_tech \
  --target-port 50506
```

### Step 3: Verify Deployment
```bash
# Check deployment status
az containerapp revision list \
  --name chorusbotfull \
  --resource-group viridity_tech \
  --output table

# View logs
az containerapp logs show \
  --name chorusbotfull \
  --resource-group viridity_tech \
  --follow
```

## Environment Variables Reference

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `PORT` | Application port | `50506` | `50506` |
| `EXTERNAL_DATA_DIR` | Path to external data directory | None (uses local `/code/data`) | `/mnt/ChorusAllData2` |
| `OPENAI_API_KEY` | OpenAI API key | Required | `sk-...` |
| `ANTHROPIC_API_KEY` | Anthropic API key | Required | `sk-ant-...` |
| `GROQ_API_KEY` | Groq API key | Required | `gsk_...` |
| `JWT_SECRET` | JWT secret for authentication | `default-dev-secret` | Custom secret |

## Troubleshooting

### Issue: Container fails to start with new port
**Solution**: Ensure ingress target port matches the PORT environment variable:
```bash
az containerapp ingress show --name chorusbotfull --resource-group viridity_tech
```

### Issue: External data directory not found
**Solution**: Verify the mount path exists and has correct permissions:
```bash
# Check if directory is mounted in container
az containerapp exec \
  --name chorusbotfull \
  --resource-group viridity_tech \
  --command "ls -la /mnt/ChorusAllData2"
```

### Issue: Application still uses old data directory
**Solution**: Check application logs to verify EXTERNAL_DATA_DIR is being used:
```bash
az containerapp logs show \
  --name chorusbotfull \
  --resource-group viridity_tech \
  --follow
```

Look for the log message:
- `Using external data directory: /mnt/ChorusAllData2` (correct)
- `Using local data directory: /code/data` (not using external directory)

## Migration from ChorusAllData to ChorusAllData2

If you need to migrate data from the old directory to the new one:

1. **Copy data** from `/mnt/ChorusAllData` to `/mnt/ChorusAllData2`
2. **Update environment variable** to point to new directory
3. **Verify** application is using new directory via logs
4. **Test** that all data is accessible

## Notes

- The application will automatically fall back to local data directory if `EXTERNAL_DATA_DIR` is not set or the path doesn't exist
- Application logs will indicate which data directory is being used
- Make sure file share permissions allow read/write access
- Port changes require updating both the environment variable and ingress configuration
