#!/bin/bash

# Quick Start Script for RagBot on Azure VM
# Run this immediately after cloning the repository

set -e

echo "🚀 RagBot Quick Start Script"
echo "============================"

# Check if running on the right system
if [[ $USER != "azureuser" ]]; then
    echo "⚠️  Warning: This script is designed for the azureuser account on Azure VM"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Get current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "📁 Working in: $SCRIPT_DIR"

# Make scripts executable
echo "🔧 Setting up permissions..."
chmod +x deploy.sh
chmod +x ragbot-backend/production_app.py

# Show system info
echo "💻 System Information:"
echo "   OS: $(lsb_release -d | cut -f2)"
echo "   Kernel: $(uname -r)"
echo "   Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "   CPUs: $(nproc)"

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "   GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
    echo "   GPU: Not detected (will install drivers)"
fi

echo ""
echo "📋 Pre-deployment Checklist:"
echo "   ✓ Scripts are executable"
echo "   ✓ System information gathered"

# Check if Azure VM
if curl -s -H Metadata:true "http://169.254.169.254/metadata/instance/compute/vmId?api-version=2021-02-01" &>/dev/null; then
    echo "   ✓ Running on Azure VM"
    
    # Get VM info
    VM_NAME=$(curl -s -H Metadata:true "http://169.254.169.254/metadata/instance/compute/name?api-version=2021-02-01&format=text" 2>/dev/null || echo "Unknown")
    VM_SIZE=$(curl -s -H Metadata:true "http://169.254.169.254/metadata/instance/compute/vmSize?api-version=2021-02-01&format=text" 2>/dev/null || echo "Unknown")
    echo "   📊 VM: $VM_NAME ($VM_SIZE)"
else
    echo "   ⚠️  Not detected as Azure VM (might still work)"
fi

echo ""
echo "🎯 Next Steps:"
echo ""
echo "1. **Run the full deployment script:**"
echo "   ./deploy.sh"
echo ""
echo "2. **After deployment, configure your API keys:**"
echo "   nano /home/azureuser/apps/ragbot/.env"
echo ""
echo "3. **Restart the service:**"
echo "   sudo systemctl restart ragbot"
echo ""
echo "4. **Open ports in Azure Portal:**
echo "   - Go to your VM → Networking → Add inbound port rule"
echo "   - Add rule for port 80 (HTTP)"
echo "   - Add rule for port 443 (HTTPS)"
echo ""
echo "5. **Access your application:**"
echo "   http://20.127.202.39"
echo ""

# Ask if user wants to continue with deployment
echo "🤖 Would you like to run the deployment script now?"
read -p "Run deployment? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🚀 Starting deployment..."
    sleep 2
    ./deploy.sh
else
    echo ""
    echo "📝 Deployment script not run. You can run it later with:"
    echo "   ./deploy.sh"
fi

echo ""
echo "📚 For detailed instructions, see: deployment_guide.md"
echo "🎉 Quick start setup complete!" 