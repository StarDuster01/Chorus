# RagBot: VM Deployment Guide

> ⚠️ **SECURITY NOTICE:** Before deploying, read [SECURITY.md](SECURITY.md) for important security best practices!

## 📋 Overview

This application uses Docker Compose to run both the backend (Flask) and frontend (React) services on an Azure VM. The deployment script handles everything automatically, including environment setup.

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose installed on the VM
- Node.js and npm installed
- Python 3.x installed
- This repository cloned to the VM

### Deploy the Application

Simply run the deployment script from the project root:

```powershell
.\deploy-vm.ps1
```

**On first run**, the script will:
1. Detect that `.env` is missing
2. Prompt you for each required environment variable
3. Auto-generate a secure JWT secret if you leave it blank
4. Create the `.env` file automatically
5. Build and deploy the application

**On subsequent runs**, the script will:
1. Use the existing `.env` file
2. Rebuild the frontend
3. Rebuild and restart the Docker containers

## 🔧 Environment Variables

The deployment script will prompt you for these variables if `.env` doesn't exist:

### Required
- **JWT_SECRET**: Leave blank to auto-generate a secure token
- **OPENAI_API_KEY**: Your OpenAI API key (format: `sk-proj-...`)

### Optional
- **OPENAI_IMAGE_API_KEY**: Separate key for image generation (falls back to OPENAI_API_KEY)
- **ANTHROPIC_API_KEY**: For Claude models in Model Chorus
- **GROQ_API_KEY**: For Groq models in Model Chorus
- **PORT**: Application port (default: 50506)
- **EXTERNAL_DATA_DIR**: Path to external data (e.g., `../ChorusAllData2` or `/mnt/ChorusAllData2`)

## 📦 What the Script Does

1. **Environment Check**: Verifies `.env` exists, creates it if needed
2. **Container Cleanup**: Stops and removes old containers
3. **Frontend Build**: Installs dependencies and builds React app
4. **File Copy**: Copies frontend build to backend directory
5. **Docker Deployment**: Starts all services with Docker Compose

## 🌐 Access the Application

After deployment:
- **Frontend**: http://localhost:80
- **Backend API**: http://localhost:50506

## 🛠️ Useful Commands

### View Logs
```bash
# Backend logs
docker logs -f ragbot-backend

# Frontend logs
docker logs -f ragbot-frontend

# All logs
docker-compose logs -f
```

### Manage Services
```bash
# Check status
docker-compose ps

# Restart services
docker-compose restart

# Stop services
docker-compose down

# Redeploy
.\deploy-vm.ps1
```

## 🏗️ Architecture

### Docker Compose Setup
The application uses two containers:
- **Backend Container**: Flask API (port 50506)
  - Serves API endpoints
  - Handles authentication, RAG, chat, image generation
  - Mounts persistent volumes for data
- **Frontend Container**: Nginx serving React build (port 80)
  - Serves the React web interface
  - Proxies API requests to backend

### Persistent Data
The following directories are mounted as volumes:
- `uploads/` - Uploaded files
- `data/` - Application data
- `conversations/` - Chat history
- `chroma_db/` - Vector database
- `datasets/` - Dataset files
- `bots/` - Bot configurations
- `image_indices/` - Image search indices

## 🔒 Security

### Important Security Practices
- ✅ `.env` file is automatically gitignored
- ✅ Never commit API keys to version control
- ✅ JWT secret is auto-generated securely
- ✅ All sensitive data is in environment variables

### If You Accidentally Expose Keys
1. Immediately rotate all API keys
2. Update the `.env` file with new keys
3. Redeploy: `.\deploy-vm.ps1`

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Stop all containers
docker-compose down

# Check what's using the port
netstat -ano | findstr :80
netstat -ano | findstr :50506

# Redeploy
.\deploy-vm.ps1
```

### Frontend Changes Not Appearing
The script automatically rebuilds the frontend on each run. Just run:
```bash
.\deploy-vm.ps1
```

### Container Won't Start
```bash
# Check logs
docker logs ragbot-backend
docker logs ragbot-frontend

# Check if .env exists
ls ragbot-backend\.env

# Rebuild from scratch
docker-compose down
docker-compose up -d --build
```

### Need to Update .env
1. Edit the file: `notepad ragbot-backend\.env`
2. Save your changes
3. Restart containers: `docker-compose restart`

Or delete `.env` and run `.\deploy-vm.ps1` to be prompted again.

## 📚 Additional Documentation

- **[SECURITY.md](SECURITY.md)** - Complete security guide
- **[ENV_SETUP.md](ragbot-backend/ENV_SETUP.md)** - Detailed environment variable documentation
- **[SECURITY_CLEANUP.md](SECURITY_CLEANUP.md)** - Security cleanup notes

## 🔄 Update Workflow

To deploy a new version:

1. **Pull latest code** (if from git):
   ```bash
   git pull origin main
   ```

2. **Run deployment script**:
   ```bash
   .\deploy-vm.ps1
   ```

The script handles everything else automatically!

## 📊 System Requirements

- **RAM**: 6GB+ recommended (backend container limit: 6GB)
- **CPU**: 4+ cores recommended
- **Storage**: Depends on dataset size
- **GPU**: Optional (NVIDIA GPU support configured in docker-compose)

## ✨ Features

- **Unified Deployment**: Single script deploys everything
- **Environment Auto-Setup**: Guided prompts for first-time setup
- **Persistent Data**: Volumes preserve data across deployments
- **Health Checks**: Automatic container health monitoring
- **GPU Support**: NVIDIA GPU acceleration when available
- **External Data**: Support for mounted external data directories

---

**Ready to deploy?** Just run `.\deploy-vm.ps1` and follow the prompts!