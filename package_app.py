import os
import shutil
import subprocess
import sys
import json
import platform
from pathlib import Path

# Configuration
APP_NAME = "RAGBot"
FLASK_APP = "ragbot-backend/app.py"
BACKEND_DIR = "ragbot-backend"
FRONTEND_DIR = "ragbot-frontend"
PYTHON_PATH = sys.executable

def check_requirements():
    """Check if all required tools are installed"""
    print("Checking requirements...")
    
    # Check if Node.js is installed
    try:
        node_version = subprocess.check_output(["node", "--version"]).decode().strip()
        print(f"Node.js version: {node_version}")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Error: Node.js is not installed. Please install Node.js to build the frontend.")
        sys.exit(1)
    
    # Check if npm is installed - since you already ran npm successfully, we'll skip this check
    print("Assuming npm is installed since Node.js is detected")
    
    # Check if PyInstaller is installed
    try:
        pyinstaller_output = subprocess.check_output([PYTHON_PATH, "-m", "pip", "show", "pyinstaller"]).decode()
        pyinstaller_version = next((line.split(": ")[1] for line in pyinstaller_output.split("\n") if line.startswith("Version")), "unknown")
        print(f"PyInstaller version: {pyinstaller_version}")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Error: PyInstaller is not installed. Please install PyInstaller to package the application.")
        sys.exit(1)
    
    # Check for problematic packages
    print("Checking for incompatible packages...")
    try:
        # Check if pathlib backport is installed (this can cause issues with PyInstaller)
        result = subprocess.run([PYTHON_PATH, "-m", "pip", "show", "pathlib"], 
                                capture_output=True, text=True)
        if result.returncode == 0:
            print("Warning: The 'pathlib' package is an obsolete backport and incompatible with PyInstaller.")
            print("Attempting to uninstall it automatically...")
            
            uninstall_result = subprocess.run([PYTHON_PATH, "-m", "pip", "uninstall", "pathlib", "-y"], 
                                             capture_output=True, text=True)
            
            if uninstall_result.returncode == 0:
                print("Successfully uninstalled 'pathlib' backport package.")
            else:
                print(f"Failed to uninstall 'pathlib'. Please run 'pip uninstall pathlib -y' manually.")
                print(f"Error: {uninstall_result.stderr}")
                sys.exit(1)
    except Exception as e:
        print(f"Error checking for incompatible packages: {str(e)}")
    
    print("All requirements satisfied!")

def build_frontend():
    """Build the React frontend"""
    print("Checking for built React frontend...")
    
    # Navigate to frontend directory
    frontend_dir = Path(FRONTEND_DIR)
    build_dir = frontend_dir / "build"
    
    if not frontend_dir.exists():
        print(f"Error: Frontend directory '{FRONTEND_DIR}' not found.")
        sys.exit(1)
    
    # Check if build directory already exists and contains index.html
    if build_dir.exists() and (build_dir / "index.html").exists():
        print("Frontend build directory already exists. Skipping build step.")
        return build_dir
    
    # Install dependencies
    print("Installing frontend dependencies...")
    result = subprocess.run(["npm", "install"], cwd=frontend_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error installing frontend dependencies: {result.stderr}")
        sys.exit(1)
    
    # Build for production
    print("Building frontend for production...")
    result = subprocess.run(["npm", "run", "build"], cwd=frontend_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error building frontend: {result.stderr}")
        sys.exit(1)
    
    print("Frontend built successfully!")
    return build_dir

def configure_backend_for_production(build_dir):
    """Configure Flask to serve the React frontend"""
    print("Configuring backend for production...")
    
    # Create a production config
    prod_config = {
        "SERVE_FRONTEND": True,
        "FRONTEND_PATH": str(build_dir.absolute()),
        "ENV": "production",
        "DEBUG": False
    }
    
    # Write config to a file
    with open("production_config.json", "w") as f:
        json.dump(prod_config, f, indent=2)
    
    # Make sure .env exists in the backend directory
    env_template_path = ".env.template"
    backend_env_path = os.path.join(BACKEND_DIR, ".env")
    
    if not os.path.exists(backend_env_path) and os.path.exists(env_template_path):
        print("Copying .env template to backend directory...")
        shutil.copy(env_template_path, backend_env_path)
    
    print("Backend configured for production!")

def create_executable():
    """Create an executable using PyInstaller"""
    print("Creating executable with PyInstaller...")
    
    # Define PyInstaller options
    spec_file = f"{APP_NAME}.spec"
    
    # Make sure backend directories exist
    backend_dir = Path(BACKEND_DIR)
    for dir_name in ['datasets', 'bots', 'conversations', 'chorus', 'uploads', 'chroma_db']:
        dir_path = backend_dir / dir_name
        dir_path.mkdir(exist_ok=True)
    
    # Create a spec file
    print("Creating PyInstaller spec file...")
    
    # Create a datas list with appropriate handling for files that might not exist
    datas_list = []
    
    # Add frontend build
    frontend_path = Path(f'{FRONTEND_DIR}/build').absolute()
    datas_list.append(f"(str(Path('{FRONTEND_DIR}/build').absolute()), 'frontend_build')")
    
    # Add config file
    datas_list.append(f"('production_config.json', '.')")
    
    # Add .env file if it exists
    env_path = backend_dir / ".env"
    if env_path.exists():
        datas_list.append(f"('{BACKEND_DIR}/.env', '.')")
    else:
        print(f"Warning: .env file not found in {BACKEND_DIR}. Using template instead.")
        # Use template if it exists
        if Path(".env.template").exists():
            datas_list.append(f"('.env.template', '.')")
    
    # Add users.json if it exists
    users_path = backend_dir / "users.json"
    if users_path.exists():
        datas_list.append(f"('{BACKEND_DIR}/users.json', '.')")
    else:
        print(f"Warning: users.json not found in {BACKEND_DIR}. Will be created on first run.")
    
    # Add directories
    for dir_name in ['datasets', 'bots', 'conversations', 'chorus', 'uploads', 'chroma_db']:
        dir_path = backend_dir / dir_name
        if dir_path.exists():
            datas_list.append(f"('{BACKEND_DIR}/{dir_name}', '{dir_name}')")
        else:
            print(f"Warning: {dir_name} directory not found. Will be created on first run.")
            dir_path.mkdir(exist_ok=True)
            datas_list.append(f"('{BACKEND_DIR}/{dir_name}', '{dir_name}')")
    
    # Format the datas list for the spec file
    datas_str = ",\n        ".join(datas_list)
    
    spec_content = f"""# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path
import site

# Add site-packages to path so PyInstaller can find all modules
site_packages = site.getsitepackages()
for site_package in site_packages:
    sys.path.insert(0, site_package)

a = Analysis(
    ['{FLASK_APP}'],
    pathex=[],
    binaries=[],
    datas=[
        {datas_str}
    ],
    hiddenimports=[
        'engineio.async_drivers.eventlet',
        'flask_cors',
        'anthropic',
        'openai',
        'PyPDF2',
        'chromadb',
        'pptx',
        'docx',
        'PIL',
        'bcrypt',
        'jwt'
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='{APP_NAME}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='{APP_NAME}',
)
"""
    
    with open(spec_file, "w") as f:
        f.write(spec_content)
    
    # Run PyInstaller
    print("Running PyInstaller...")
    result = subprocess.run([PYTHON_PATH, "-m", "PyInstaller", spec_file, "--noconfirm"], 
                            capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error creating executable: {result.stderr}")
        
        # Check for common error patterns and provide helpful advice
        if "pathlib" in result.stderr and "incompatible" in result.stderr:
            print("\nThis error is related to an incompatible 'pathlib' package.")
            print("Please run: pip uninstall pathlib -y")
            print("Then run this script again.")
        elif "ImportError" in result.stderr or "ModuleNotFoundError" in result.stderr:
            print("\nA required module is missing. Try installing it with pip:")
            # Try to extract the missing module name
            import re
            module_match = re.search(r"No module named '([^']+)'", result.stderr)
            if module_match:
                missing_module = module_match.group(1)
                print(f"pip install {missing_module}")
            else:
                print("Check the error message above for the specific missing module.")
        
        sys.exit(1)
    
    print("Executable created successfully!")

def create_installer():
    """Create an installer for the application"""
    print("Creating installer...")
    
    if platform.system() == "Windows":
        # Create Inno Setup script for Windows
        inno_script = f"""#define MyAppName "{APP_NAME}"
#define MyAppVersion "1.0"
#define MyAppPublisher "Your Company"
#define MyAppExeName "{APP_NAME}.exe"

[Setup]
AppId={{{{APP_NAME}}}}
AppName={{#MyAppName}}
AppVersion={{#MyAppVersion}}
AppPublisher={{#MyAppPublisher}}
DefaultDirName={{autopf}}\\{{#MyAppName}}
DefaultGroupName={{#MyAppName}}
OutputDir=installer
OutputBaseFilename={APP_NAME}_Setup
Compression=lzma
SolidCompression=yes
SetupIconFile=.\\{FRONTEND_DIR}\\public\\favicon.ico

[Files]
Source: "dist\\{APP_NAME}\\*"; DestDir: "{{app}}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{{group}}\\{{#MyAppName}}"; Filename: "{{app}}\\{{#MyAppExeName}}"
Name: "{{commondesktop}}\\{{#MyAppName}}"; Filename: "{{app}}\\{{#MyAppExeName}}"

[Run]
Filename: "{{app}}\\{{#MyAppExeName}}"; Description: "Launch {APP_NAME}"; Flags: nowait postinstall skipifsilent
"""
        
        # Write Inno Setup script to a file
        with open(f"{APP_NAME}.iss", "w") as f:
            f.write(inno_script)
        
        # Check if Inno Setup is installed
        inno_compiler = "C:\\Program Files (x86)\\Inno Setup 6\\ISCC.exe"
        if os.path.exists(inno_compiler):
            print("Running Inno Setup compiler...")
            result = subprocess.run([inno_compiler, f"{APP_NAME}.iss"], 
                                    capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Error creating installer: {result.stderr}")
                print("You'll need to compile the installer manually.")
            else:
                print("Installer created successfully!")
        else:
            print("Inno Setup not found. Please install Inno Setup and compile the .iss file manually.")
            print(f"Inno Setup script saved to {APP_NAME}.iss")
    
    elif platform.system() == "Darwin":
        # Create macOS DMG
        print("Creating macOS installer is not implemented yet.")
        print("You can use tools like create-dmg or Packages to create a macOS installer.")
    
    else:
        # Create Linux installer (deb or rpm)
        print("Creating Linux installer is not implemented yet.")
        print("You can use tools like fpm to create a Linux installer.")
    
    print("\nAlternatively, you can distribute the dist/{APP_NAME} directory directly.")

def modify_backend_for_packaging():
    """Modify the Flask app to work with PyInstaller"""
    print("Modifying backend for packaging...")
    
    # Create a wrapper script that will be executed by PyInstaller
    wrapper_script = """import os
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
"""
    
    # Save wrapper script to the backend directory
    with open(os.path.join(BACKEND_DIR, "wrapper.py"), "w") as f:
        f.write(wrapper_script)
    
    print("Backend modified for packaging!")
    
    return f"{BACKEND_DIR}/wrapper.py"  # Return the wrapper script path

def main():
    """Main function to package the application"""
    print(f"=== Packaging {APP_NAME} ===")
    
    # Check requirements
    check_requirements()
    
    # Build frontend
    build_dir = build_frontend()
    
    # Configure backend for production
    configure_backend_for_production(build_dir)
    
    # Modify backend for packaging
    wrapper_script = modify_backend_for_packaging()
    
    # Update FLASK_APP to use the wrapper script
    global FLASK_APP
    FLASK_APP = wrapper_script
    
    # Create executable
    create_executable()
    
    # Create installer
    create_installer()
    
    print(f"\n=== {APP_NAME} packaged successfully! ===")
    print(f"Executable: dist/{APP_NAME}/{APP_NAME}{'exe' if platform.system() == 'Windows' else ''}")
    print(f"Installer: installer/{APP_NAME}_Setup.exe (if Inno Setup was available)")
    print("\nIf you need to modify the application, update the source files and run this script again.")

if __name__ == "__main__":
    main()