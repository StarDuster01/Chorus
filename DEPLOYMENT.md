# 🚀 Quick Deployment Reference

## One-Command Deployment

```powershell
.\deploy-vm.ps1
```

That's it! The script handles everything.

## First Time Setup

When you run the script for the first time, it will prompt you for:

1. **JWT_SECRET** → Press Enter to auto-generate
2. **OPENAI_API_KEY** → Your OpenAI key
3. **OPENAI_IMAGE_API_KEY** → Press Enter to use main key
4. **ANTHROPIC_API_KEY** → Optional, press Enter to skip
5. **GROQ_API_KEY** → Optional, press Enter to skip
6. **PORT** → Press Enter for default (50506)
7. **EXTERNAL_DATA_DIR** → Path to external data or press Enter

## Access Your Application

After deployment completes:
- **Frontend**: http://localhost:80
- **Backend**: http://localhost:50506

## Common Commands

```bash
# View logs
docker logs -f ragbot-backend        # Backend logs
docker logs -f ragbot-frontend       # Frontend logs
docker-compose logs -f               # All logs

# Manage containers
docker-compose ps                    # Check status
docker-compose restart               # Restart services
docker-compose down                  # Stop all services

# Redeploy (rebuild everything)
.\deploy-vm.ps1
```

## Update .env File

To change environment variables:

```bash
# Option 1: Edit manually
notepad ragbot-backend\.env
docker-compose restart

# Option 2: Delete and reconfigure
rm ragbot-backend\.env
.\deploy-vm.ps1
```

## Troubleshooting

### Port conflicts?
```bash
docker-compose down
.\deploy-vm.ps1
```

### Container errors?
```bash
docker logs ragbot-backend
docker logs ragbot-frontend
```

### Start fresh?
```bash
docker-compose down
docker system prune -f
.\deploy-vm.ps1
```

---

**That's all you need to know!** 🎉
