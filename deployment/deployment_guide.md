# RagBot Azure VM Deployment Guide

## Phase 1: VM Setup and Connection

### 1. Connect to Your VM
```bash
# SSH into your VM (replace with your private key path)
ssh -i /path/to/your/private-key.pem azureuser@20.127.202.39
```

### 2. Update System and Install Dependencies
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y python3 python3-pip python3-venv git nginx supervisor curl wget htop

# Install Node.js (for frontend building)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install Docker (optional, for containerized deployment)
sudo apt-get install -y docker.io docker-compose
sudo usermod -aG docker $USER
```

### 3. Install NVIDIA Drivers and CUDA (for GPU acceleration)
```bash
# Check if GPU is detected
lspci | grep -i nvidia

# Install NVIDIA drivers
sudo apt install -y nvidia-driver-525 nvidia-cuda-toolkit

# Verify installation (reboot may be required)
nvidia-smi
```

## Phase 2: Clone and Setup Project

### 1. Clone the Repository
```bash
# Create project directory
mkdir -p /home/azureuser/apps
cd /home/azureuser/apps

# Clone your repository (replace with your actual repo URL)
git clone https://github.com/yourusername/ragbot.git
cd ragbot
```

### 2. Setup Python Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install additional production dependencies
pip install gunicorn supervisor
```

### 3. Build Frontend
```bash
# Navigate to frontend directory
cd ragbot-frontend

# Install dependencies and build
npm install
npm run build

# Copy build to backend static folder
cp -r build/* ../ragbot-backend/frontend/
cd ..
```

## Phase 3: Configuration Changes

### 1. Environment Variables
Create `/home/azureuser/apps/ragbot/.env`:
```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic Configuration  
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Groq Configuration
GROQ_API_KEY=your_groq_api_key_here

# JWT Secret (generate a secure random string)
JWT_SECRET=your_super_secure_jwt_secret_here

# Production settings
FLASK_ENV=production
FLASK_DEBUG=False

# Database/Storage paths
DATA_FOLDER=/home/azureuser/apps/ragbot/data
UPLOAD_FOLDER=/home/azureuser/apps/ragbot/uploads
```

### 2. Create Production App Configuration
The `production_app.py` and other configuration files have been created automatically.

## Phase 4: Deploy to Azure VM

### 1. Connect to Your VM
```bash
# Connect to your Azure VM
ssh -i your-private-key.pem azureuser@20.127.202.39
```

### 2. Clone Your Repository
```bash
# Create apps directory
mkdir -p /home/azureuser/apps
cd /home/azureuser/apps

# Clone your RagBot repository
git clone https://github.com/yourusername/ragbot.git
cd ragbot
```

### 3. Run the Deployment Script
```bash
# Make the deployment script executable
chmod +x deploy.sh

# Run the deployment
./deploy.sh
```

### 4. Configure Your API Keys
```bash
# Edit the environment file
nano /home/azureuser/apps/ragbot/.env

# Add your actual API keys:
OPENAI_API_KEY=sk-your-actual-openai-key
ANTHROPIC_API_KEY=sk-ant-your-actual-anthropic-key
GROQ_API_KEY=gsk_your-actual-groq-key
```

### 5. Restart the Service
```bash
# Restart RagBot to apply the new configuration
sudo systemctl restart ragbot

# Check if it's running
sudo systemctl status ragbot
```

## Phase 5: Azure VM Network Configuration

### 1. Open Required Ports in Azure
Go to your Azure Portal → Virtual Machine → Networking → Add inbound port rule:

**HTTP Rule:**
- Source: Any
- Source port ranges: *
- Destination: Any  
- Destination port ranges: 80
- Protocol: TCP
- Action: Allow
- Priority: 1000
- Name: HTTP

**HTTPS Rule (for future SSL setup):**
- Source: Any
- Source port ranges: *
- Destination: Any
- Destination port ranges: 443
- Protocol: TCP
- Action: Allow
- Priority: 1001
- Name: HTTPS

### 2. Test Your Deployment
```bash
# Check if the application is responding
curl http://localhost:50505/health

# Check nginx is serving
curl http://localhost/health

# Check from outside (from your local machine)
curl http://20.127.202.39/health
```

## Phase 6: Monitoring and Maintenance

### 1. View Application Logs
```bash
# View RagBot application logs
sudo journalctl -u ragbot -f

# View nginx logs
sudo tail -f /var/log/nginx/ragbot_access.log
sudo tail -f /var/log/nginx/ragbot_error.log

# View application-specific logs
tail -f /home/azureuser/apps/ragbot/logs/ragbot.log
```

### 2. Common Maintenance Commands
```bash
# Restart services
sudo systemctl restart ragbot
sudo systemctl restart nginx

# Update application
cd /home/azureuser/apps/ragbot
git pull origin main
sudo systemctl restart ragbot

# Check service status
sudo systemctl status ragbot nginx

# Check disk usage
df -h
du -sh /home/azureuser/apps/ragbot/uploads
```

### 3. GPU Monitoring
```bash
# Check GPU usage
nvidia-smi

# Monitor GPU usage continuously
watch -n 1 nvidia-smi

# Check GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## Phase 7: Security Hardening (Optional)

### 1. Set up SSL with Let's Encrypt
```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate (replace with your domain)
sudo certbot --nginx -d yourdomain.com

# Auto-renewal
sudo systemctl enable certbot.timer
```

### 2. Enhanced Security
```bash
# Install fail2ban for intrusion prevention
sudo apt install fail2ban

# Configure UFW firewall more restrictively
sudo ufw delete allow 22  # Remove default SSH rule
sudo ufw allow from YOUR_IP_ADDRESS to any port 22  # Allow SSH only from your IP

# Set up automatic security updates
sudo apt install unattended-upgrades
sudo dpkg-reconfigure unattended-upgrades
```

## Troubleshooting

### Common Issues

1. **Service won't start:**
   ```bash
   sudo journalctl -u ragbot -n 50
   ```

2. **Permission errors:**
   ```bash
   sudo chown -R azureuser:azureuser /home/azureuser/apps/ragbot
   ```

3. **GPU not detected:**
   ```bash
   nvidia-smi
   sudo apt install nvidia-driver-525
   sudo reboot
   ```

4. **High memory usage:**
   ```bash
   # Restart the service to clear memory
   sudo systemctl restart ragbot
   ```

5. **Frontend not loading:**
   ```bash
   # Check nginx configuration
   sudo nginx -t
   # Check if frontend files exist
   ls -la /home/azureuser/apps/ragbot/ragbot-backend/frontend/
   ```

## Success! 🎉

Your RagBot should now be running at: **http://20.127.202.39**

The deployment includes:
- ✅ Production-optimized Flask application
- ✅ GPU acceleration for AI models
- ✅ Nginx reverse proxy for performance
- ✅ Systemd service for reliability
- ✅ Proper logging and monitoring
- ✅ Security configurations
- ✅ Automatic startup on reboot 