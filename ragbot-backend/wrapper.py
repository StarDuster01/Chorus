import os
import sys
import json
from pathlib import Path

# Determine if we're running from PyInstaller bundle
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # Running as PyInstaller executable
    bundle_dir = Path(sys._MEIPASS)
    
    # Set up paths
    os.environ['FLASK_APP_DIR'] = str(bundle_dir)
    
    # Load production config
    config_path = bundle_dir / 'production_config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Set up environment variables
        for key, value in config.items():
            os.environ[key] = str(value)
            
        # Set FRONTEND_PATH to the bundled frontend
        os.environ['FRONTEND_PATH'] = str(bundle_dir / 'frontend_build')
    
    # Ensure working directories exist in the current directory
    for dir_name in ['datasets', 'bots', 'conversations', 'chorus', 'uploads', 'chroma_db']:
        os.makedirs(dir_name, exist_ok=True)

# Now import app.py from the backend directory
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from app import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
