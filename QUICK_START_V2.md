# 🚀 Chorus RagBot v2.0 - Quick Start Guide

## Local Development

### Run Locally (One Command)
```powershell
.\deploy-local.ps1
```
**Access**: http://localhost:50506

---

## Azure Deployment

### Quick Deploy to Azure
```powershell
.\deploy-azure.ps1
```

### Manual Azure Deployment

#### 1. Build & Push
```bash
cd ragbot-backend
docker build -t viriditytech.azurecr.io/ragbot-unified:latest .
docker push viriditytech.azurecr.io/ragbot-unified:latest
```

#### 2. Update Container App
```bash
az containerapp update \
  --name chorusbotfull \
  --resource-group viridity_tech \
  --image viriditytech.azurecr.io/ragbot-unified:latest \
  --set-env-vars PORT=50506 EXTERNAL_DATA_DIR=/mnt/ChorusAllData2

az containerapp ingress update \
  --name chorusbotfull \
  --resource-group viridity_tech \
  --target-port 50506
```

---

## Configuration

### Environment Variables
```bash
PORT=50506                              # Application port (default: 50506)
EXTERNAL_DATA_DIR=/mnt/ChorusAllData2  # External data directory (optional)
```

### Docker Compose
```yaml
environment:
  - PORT=50506
  - EXTERNAL_DATA_DIR=/mnt/ChorusAllData2  # Optional
```

---

## Port Configuration

| Version | Port  |
|---------|-------|
| v1.0    | 50505 |
| v2.0    | 50506 |

---

## Data Directory Configuration

### Default (Local)
```
/code/data
```

### External (Azure File Share)
```
/mnt/ChorusAllData2
```

**Note**: Application automatically falls back to local if external directory not found.

---

## Useful Commands

### Check Container Status
```bash
docker ps
```

### View Logs (Local)
```bash
docker logs ragbot-local-instance -f
```

### View Logs (Azure)
```bash
az containerapp logs show --name chorusbotfull --resource-group viridity_tech --follow
```

### Check Azure Deployment Status
```bash
az containerapp revision list --name chorusbotfull --resource-group viridity_tech --output table
```

### Stop Local Container
```bash
docker stop ragbot-local-instance
docker rm ragbot-local-instance
```

---

## Troubleshooting

### Port Already in Use
```powershell
# Find and kill process using port 50506
Get-Process -Id (Get-NetTCPConnection -LocalPort 50506).OwningProcess | Stop-Process
```

### Application Not Starting
1. Check logs
2. Verify environment variables
3. Ensure port is not in use
4. Check Docker is running

### External Data Directory Not Working
1. Verify directory exists: `az containerapp exec --command "ls -la /mnt/ChorusAllData2"`
2. Check logs for: `Using external data directory: /mnt/ChorusAllData2`
3. Verify Azure File Share is mounted correctly

---

## URLs

### Local
- **Frontend**: http://localhost:50506
- **API**: http://localhost:50506/api

### Azure
- **Frontend**: https://chorusbotfull.wonderfulocean-4a90a562.westus2.azurecontainerapps.io
- **API**: https://chorusbotfull.wonderfulocean-4a90a562.westus2.azurecontainerapps.io/api

---

## Next Steps

1. ✅ Deploy application
2. ✅ Verify deployment
3. ✅ Configure external data directory (if needed)
4. ✅ Test functionality
5. ✅ Monitor logs

---

## Documentation

- **Full Deployment Guide**: `AZURE_DEPLOYMENT_CONFIG.md`
- **Changelog**: `CHANGELOG_V2.md`
- **README**: `README.md`
