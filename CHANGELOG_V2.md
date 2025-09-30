# Chorus RagBot v2.0 - Changelog

## Release Date: September 30, 2025

## Overview
Version 2.0 introduces significant UI modernization and deployment configuration improvements, including support for external data directories and configurable ports for Azure VM deployments.

---

## 🎨 Frontend Visual Enhancements

### Design System Updates
- **Color Palette Improvements**
  - Added hover gradient variants for smoother transitions
  - Enhanced shadow system with multi-level depth (card-shadow, card-shadow-hover)
  - New CSS custom properties for border radii (--border-radius-lg, --border-radius-xl)
  - Refined light background gradient

### UI Components Modernization

#### Cards & Containers
- ✨ Enhanced card shadows with more depth (0 8px 32px)
- ✨ Added hover effects with subtle lift animation (translateY(-4px))
- ✨ Gradient backgrounds with better contrast (135deg gradients)
- ✨ Border styling with subtle accent colors (rgba borders)
- ✨ Backdrop blur effects for modern glass-morphism look

#### Navigation Bar
- ✨ Improved shadow depth and clarity
- ✨ Added backdrop-filter blur for modern appearance
- ✨ Subtle border with rgba transparency

#### Buttons
- ✨ Enhanced hover states with lift animation
- ✨ Improved gradient transitions
- ✨ Added active state feedback
- ✨ Better shadow depth on interaction
- ✨ Refined letter-spacing and font-weight

#### Form Controls
- ✨ Enhanced focus states with transform animation
- ✨ Better border styling with subtle primary color tints
- ✨ Improved shadow depth on focus
- ✨ Smoother transitions with cubic-bezier easing

#### Chat Interface
- ✨ Modern message bubbles with enhanced shadows
- ✨ Better color contrast for user vs bot messages
- ✨ Improved chat input area with gradient background
- ✨ Enhanced message content spacing and padding
- ✨ Transparent background with modern aesthetics

#### Dropdowns & Menus
- ✨ Glass-morphism effect with backdrop blur
- ✨ Enhanced shadow depth
- ✨ Smooth hover animations with translateX effect
- ✨ Better item spacing and padding

#### Authentication Pages
- ✨ Improved card styling with backdrop blur
- ✨ Enhanced shadows for depth
- ✨ Better visual hierarchy

### Typography & Spacing
- Changed primary font from 'Poppins' to 'Inter' for modern readability
- Improved padding and spacing throughout
- Enhanced letter-spacing for better readability

### Background
- ✨ Added gradient background to body (f8faf9 to e6f7f0)
- ✨ Background attachment: fixed for parallax effect
- ✨ Minimum height for better mobile experience

---

## ⚙️ Backend Configuration Updates

### Port Configuration
- **Old Port**: 50505
- **New Port**: 50506 (default)
- **Configurable**: Via `PORT` environment variable
- **Updated Files**:
  - `app.py` - Added port configuration via os.getenv()
  - `gunicorn.conf.py` - Dynamic port binding
  - `Dockerfile` - Updated EXPOSE directive
  - `docker-compose.yml` - Updated port mapping
  - `deploy-local.ps1` - Updated to use new port
  - `deploy-azure.ps1` - Updated Azure deployment configuration

### External Data Directory Support
- **New Feature**: Support for external data directories (e.g., Azure File Shares)
- **Configuration**: Via `EXTERNAL_DATA_DIR` environment variable
- **Use Case**: Mount external storage like `/mnt/ChorusAllData2` on Azure VMs
- **Fallback**: Automatically falls back to local data directory if not configured
- **Logging**: Application logs which data directory is being used

#### Implementation Details
```python
external_data_dir = os.getenv("EXTERNAL_DATA_DIR")
if external_data_dir and os.path.exists(external_data_dir):
    DATA_FOLDER = external_data_dir
    print(f"Using external data directory: {DATA_FOLDER}")
else:
    DATA_FOLDER = os.path.join(app_base_dir, "data")
    print(f"Using local data directory: {DATA_FOLDER}")
```

---

## 📦 Deployment Updates

### Docker Compose
- Updated port mapping from 50505:50505 to 50506:50506
- Added PORT environment variable configuration
- Added EXTERNAL_DATA_DIR environment variable (commented with example)
- Updated healthcheck endpoint to use new port

### Local Deployment Script (`deploy-local.ps1`)
- Updated to deploy on port 50506
- Added PORT environment variable to docker run command
- Updated all user-facing messages with new port
- Updated browser auto-open URL

### Azure Deployment Script (`deploy-azure.ps1`)
- Added environment variable configuration for PORT
- Added environment variable configuration for EXTERNAL_DATA_DIR
- Added ingress port update command
- Enhanced deployment completion message with configuration details
- Added helpful commands for monitoring deployment

### New Documentation
- **AZURE_DEPLOYMENT_CONFIG.md** - Comprehensive guide for Azure deployment
  - Environment variable reference table
  - Step-by-step Azure File Share mounting instructions
  - Ingress configuration guide
  - Troubleshooting section
  - Migration guide from ChorusAllData to ChorusAllData2

---

## 🔧 Configuration Reference

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `PORT` | Application port | `50506` | No |
| `EXTERNAL_DATA_DIR` | External data directory path | None | No |
| `OPENAI_API_KEY` | OpenAI API key | None | Yes |
| `ANTHROPIC_API_KEY` | Anthropic API key | None | Yes |
| `GROQ_API_KEY` | Groq API key | None | Yes |
| `JWT_SECRET` | JWT secret for authentication | `default-dev-secret` | Recommended |

### Azure Container Apps Configuration
```bash
# Set environment variables
az containerapp update \
  --name chorusbotfull \
  --resource-group viridity_tech \
  --set-env-vars \
    PORT=50506 \
    EXTERNAL_DATA_DIR=/mnt/ChorusAllData2

# Update ingress port
az containerapp ingress update \
  --name chorusbotfull \
  --resource-group viridity_tech \
  --target-port 50506
```

---

## 📝 Files Modified

### Frontend
- `ragbot-frontend/src/index.css` - Major visual enhancements
- `ragbot-frontend/src/components/ChatInterface.css` - Chat UI improvements

### Backend
- `ragbot-backend/app.py` - Port and data directory configuration
- `ragbot-backend/gunicorn.conf.py` - Dynamic port binding
- `ragbot-backend/Dockerfile` - Updated port exposure

### Deployment
- `docker-compose.yml` - Updated configuration
- `deploy-local.ps1` - Updated local deployment
- `deploy-azure.ps1` - Enhanced Azure deployment

### Documentation
- `AZURE_DEPLOYMENT_CONFIG.md` - New comprehensive deployment guide
- `CHANGELOG_V2.md` - This file

---

## 🚀 Migration Guide

### For Local Development
1. Pull latest changes
2. Run `.\deploy-local.ps1`
3. Access application at `http://localhost:50506`

### For Azure Deployment
1. Follow updated `deploy-azure.ps1` script
2. Configure environment variables in Azure Container Apps
3. Update ingress port to 50506
4. (Optional) Mount external data directory

### For Existing Deployments
1. Update environment variables:
   ```bash
   az containerapp update --set-env-vars PORT=50506
   ```
2. Update ingress:
   ```bash
   az containerapp ingress update --target-port 50506
   ```
3. (Optional) Configure external data directory:
   ```bash
   az containerapp update --set-env-vars EXTERNAL_DATA_DIR=/mnt/ChorusAllData2
   ```

---

## 🐛 Known Issues
None at this time.

---

## 🔮 Future Enhancements
- Dark mode support
- Additional theme customization options
- Multi-region deployment support
- Enhanced monitoring and logging

---

## 📞 Support
For issues or questions, please refer to:
- `AZURE_DEPLOYMENT_CONFIG.md` for deployment troubleshooting
- Application logs for runtime issues
- Azure Container Apps documentation for infrastructure questions
