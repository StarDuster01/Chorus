#!/bin/bash
set -e

# RagBot Azure VM Deployment Script
# Run this script on your Azure VM to deploy RagBot

echo "🚀 Starting RagBot deployment on Azure VM..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
APP_DIR="/home/azureuser/apps/ragbot"
VENV_DIR="$APP_DIR/venv"
NGINX_SITE="ragbot"
SERVICE_NAME="ragbot"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root. Run as azureuser."
   exit 1
fi

# Update system packages
print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
print_status "Installing system dependencies..."
sudo apt install -y python3 python3-pip python3-venv git nginx curl wget htop unzip

# Install Node.js
print_status "Installing Node.js..."
if ! command -v node &> /dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi

# Install NVIDIA drivers if not present
print_status "Checking GPU support..."
if lspci | grep -i nvidia > /dev/null; then
    if ! command -v nvidia-smi &> /dev/null; then
        print_status "Installing NVIDIA drivers..."
        sudo apt install -y nvidia-driver-525 nvidia-cuda-toolkit
        print_warning "GPU drivers installed. A reboot may be required."
    else
        print_status "NVIDIA drivers already installed."
        nvidia-smi
    fi
else
    print_warning "No NVIDIA GPU detected."
fi

# Create application directory
print_status "Setting up application directory..."
mkdir -p $APP_DIR
cd $APP_DIR

# Check if this is a fresh installation or update
if [ ! -d ".git" ]; then
    print_status "Fresh installation detected. Please clone your repository first:"
    print_warning "Run: git clone <your-repo-url> $APP_DIR"
    print_warning "Then run this script again."
    exit 1
fi

# Update repository
print_status "Updating repository..."
git pull origin main || git pull origin master

# Create Python virtual environment
print_status "Setting up Python virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv $VENV_DIR
fi

# Activate virtual environment and install dependencies
print_status "Installing Python dependencies..."
source $VENV_DIR/bin/activate
pip install --upgrade pip
pip install -r ragbot-backend/requirements.txt
pip install gunicorn

# Install additional GPU-optimized packages
print_status "Installing GPU-optimized packages..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Build frontend
print_status "Building React frontend..."
cd ragbot-frontend
npm install
npm run build

# Copy frontend build to backend static folder
print_status "Copying frontend assets..."
mkdir -p ../ragbot-backend/frontend
cp -r build/* ../ragbot-backend/frontend/
cd ..

# Create necessary directories
print_status "Creating application directories..."
mkdir -p logs data uploads conversations
mkdir -p data/image_indices uploads/documents uploads/images

# Set up environment variables
print_status "Setting up environment variables..."
if [ ! -f .env ]; then
    print_warning "Creating .env file. Please edit it with your API keys!"
    cat > .env << EOL
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic Configuration  
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Groq Configuration
GROQ_API_KEY=your_groq_api_key_here

# JWT Secret (generate a secure random string)
JWT_SECRET=$(openssl rand -base64 32)

# Production settings
FLASK_ENV=production
FLASK_DEBUG=False
HOST=127.0.0.1
PORT=50505

# Paths
DATA_FOLDER=$APP_DIR/data
UPLOAD_FOLDER=$APP_DIR/uploads
EOL
    print_warning "Please edit $APP_DIR/.env with your actual API keys!"
else
    print_status "Environment file already exists."
fi

# Set up nginx configuration
print_status "Configuring nginx..."
sudo cp nginx-ragbot.conf /etc/nginx/sites-available/$NGINX_SITE

# Enable nginx site
if [ ! -L "/etc/nginx/sites-enabled/$NGINX_SITE" ]; then
    sudo ln -s /etc/nginx/sites-available/$NGINX_SITE /etc/nginx/sites-enabled/
fi

# Remove default nginx site if it exists
if [ -L "/etc/nginx/sites-enabled/default" ]; then
    sudo rm /etc/nginx/sites-enabled/default
fi

# Test nginx configuration
print_status "Testing nginx configuration..."
sudo nginx -t

# Set up systemd service
print_status "Setting up systemd service..."
sudo cp ragbot.service /etc/systemd/system/
sudo systemctl daemon-reload

# Set proper permissions
print_status "Setting file permissions..."
sudo chown -R azureuser:azureuser $APP_DIR
chmod +x ragbot-backend/production_app.py

# Configure firewall
print_status "Configuring firewall..."
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS
sudo ufw --force enable

# Start services
print_status "Starting services..."

# Start nginx
sudo systemctl enable nginx
sudo systemctl restart nginx

# Start RagBot service
sudo systemctl enable $SERVICE_NAME
sudo systemctl restart $SERVICE_NAME

# Wait a moment for services to start
sleep 5

# Check service status
print_status "Checking service status..."
if sudo systemctl is-active --quiet $SERVICE_NAME; then
    print_status "✅ RagBot service is running!"
else
    print_error "❌ RagBot service failed to start. Check logs:"
    print_error "sudo journalctl -u $SERVICE_NAME -f"
fi

if sudo systemctl is-active --quiet nginx; then
    print_status "✅ Nginx is running!"
else
    print_error "❌ Nginx failed to start. Check configuration."
fi

# Show final status
print_status "Deployment completed!"
print_status "🌐 Your RagBot application should be accessible at:"
print_status "   http://20.127.202.39"
print_status ""
print_status "📋 Useful commands:"
print_status "   Check service status: sudo systemctl status $SERVICE_NAME"
print_status "   View logs: sudo journalctl -u $SERVICE_NAME -f"
print_status "   Restart service: sudo systemctl restart $SERVICE_NAME"
print_status "   Check nginx: sudo systemctl status nginx"
print_status ""
print_warning "⚠️  Don't forget to:"
print_warning "   1. Edit $APP_DIR/.env with your actual API keys"
print_warning "   2. Restart the service after updating .env: sudo systemctl restart $SERVICE_NAME"
print_warning "   3. Set up SSL certificates for production use"
print_warning "   4. Configure your domain name if you have one"

echo ""
echo "🎉 RagBot deployment complete!" 