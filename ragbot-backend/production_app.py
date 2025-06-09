#!/usr/bin/env python3
"""
Production Flask application for RagBot
Optimized for deployment on Azure VM with GPU acceleration
"""

import os
import sys
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Import the main app
from app import app

# Production configuration
class ProductionConfig:
    # Security
    SECRET_KEY = os.environ.get('JWT_SECRET', 'production-secret-change-me')
    
    # Database and storage
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # File upload settings
    MAX_CONTENT_LENGTH = 20 * 1024 * 1024 * 1024  # 20GB
    
    # Logging
    LOG_LEVEL = 'INFO'
    
    # Performance
    SEND_FILE_MAX_AGE_DEFAULT = 31536000  # 1 year cache for static files

# Apply production configuration
app.config.from_object(ProductionConfig)

# Production-specific settings
app.config['DEBUG'] = False
app.config['TESTING'] = False

# Ensure directories exist
base_dir = Path("/home/azureuser/apps/ragbot")
data_dir = base_dir / "data"
upload_dir = base_dir / "uploads"
conversations_dir = base_dir / "conversations"

for directory in [data_dir, upload_dir, conversations_dir]:
    directory.mkdir(parents=True, exist_ok=True)

# Update app config with production paths
app.config['UPLOAD_FOLDER'] = str(upload_dir)
app.config['DATA_FOLDER'] = str(data_dir)

# Production logging setup
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    # File handler for application logs
    log_dir = base_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    file_handler = RotatingFileHandler(
        log_dir / 'ragbot.log', 
        maxBytes=10240000, 
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    
    app.logger.setLevel(logging.INFO)
    app.logger.info('RagBot startup')

# GPU optimization for production
def configure_gpu_settings():
    """Configure GPU settings for optimal performance"""
    try:
        import torch
        if torch.cuda.is_available():
            # Set CUDA memory fraction to prevent OOM
            torch.cuda.set_per_process_memory_fraction(0.8)
            app.logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            app.logger.info(f"CUDA memory allocated: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
        else:
            app.logger.warning("No GPU detected, using CPU")
    except ImportError:
        app.logger.warning("PyTorch not available, GPU acceleration disabled")

# Initialize GPU settings
configure_gpu_settings()

# Health check endpoint for load balancer
@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return {'status': 'healthy', 'version': '1.0.0'}, 200

if __name__ == '__main__':
    # This should not be used in production
    # Use gunicorn instead
    app.logger.warning("Running in development mode! Use gunicorn for production.")
    app.run(host='0.0.0.0', port=50505, debug=False) 