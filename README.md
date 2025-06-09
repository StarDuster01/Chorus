# RagBot

RagBot is an advanced AI-powered chatbot application that combines the power of multiple language models with document processing capabilities. It supports both text and image interactions, making it a versatile tool for various use cases.

## Features

- Multi-model support (OpenAI, Anthropic, Groq)
- Document processing and indexing
- Image generation and processing
- Conversation management
- Dataset management
- Model chorus capabilities
- GPU-accelerated vector search

## Project Structure

```
RagBot/
├── ragbot-backend/           # Flask backend application
│   ├── app.py               # Main application file
│   ├── requirements.txt     # Python dependencies
│   ├── image_processor.py   # Image processing utilities
│   ├── dataset_handlers.py  # Dataset management
│   ├── bot_handlers.py      # Bot management
│   ├── chorus_handlers.py   # Model chorus functionality
│   └── auth_handlers.py     # Authentication utilities
├── ragbot-frontend/         # React frontend application
│   ├── src/                # Source code
│   ├── public/             # Static assets
│   └── package.json        # Node.js dependencies
├── data/                   # Data storage
│   ├── image_indices/      # Image search indices
│   └── datasets/          # Dataset storage
├── uploads/               # Upload directory
│   ├── documents/        # Document uploads
│   └── images/          # Image uploads
└── logs/                 # Application logs
```

## Azure VM Deployment

### Prerequisites

- Azure VM with:
  - Ubuntu 22.04 LTS
  - NVIDIA GPU (CUDA 11.5+)
  - At least 4GB RAM
  - At least 20GB storage

### Installation Steps

1. **Connect to the VM**
   ```bash
   ssh viriditytech@<vm-ip>
   ```

2. **Install System Dependencies**
   ```bash
   sudo apt update
   sudo apt install -y python3.11 python3.11-venv python3.11-dev
   ```

3. **Clone the Repository**
   ```bash
   cd ~/apps
   git clone https://github.com/StarDuster01/RagBot.git
   cd RagBot
   ```

4. **Set Up Python Environment**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r ragbot-backend/requirements.txt
   pip install faiss-gpu-cu11
   ```

5. **Create Required Directories**
   ```bash
   mkdir -p data uploads logs
   mkdir -p data/image_indices uploads/documents uploads/images
   ```

6. **Configure Environment**
   Create a `.env` file in the project root:
   ```
   # OpenAI Configuration
   OPENAI_API_KEY=your_openai_api_key_here

   # Anthropic Configuration  
   ANTHROPIC_API_KEY=your_anthropic_api_key_here

   # Groq Configuration
   GROQ_API_KEY=your_groq_api_key_here

   # JWT Secret (generate a secure random string)
   JWT_SECRET=your_jwt_secret_here

   # Production settings
   FLASK_ENV=production
   FLASK_DEBUG=False
   HOST=127.0.0.1
   PORT=50505

   # Paths
   DATA_FOLDER=/home/viriditytech/apps/RagBot/data
   UPLOAD_FOLDER=/home/viriditytech/apps/RagBot/uploads
   ```

7. **Set Up Systemd Service**
   Create `/etc/systemd/system/ragbot.service`:
   ```ini
   [Unit]
   Description=RagBot Application
   After=network.target

   [Service]
   User=viriditytech
   WorkingDirectory=/home/viriditytech/apps/RagBot/ragbot-backend
   Environment="PATH=/home/viriditytech/apps/RagBot/venv/bin"
   Environment="PYTHONPATH=/home/viriditytech/apps/RagBot/ragbot-backend"
   ExecStart=/home/viriditytech/apps/RagBot/venv/bin/gunicorn --workers 4 --bind 0.0.0.0:50505 --error-logfile /home/viriditytech/apps/RagBot/logs/gunicorn_error.log --access-logfile /home/viriditytech/apps/RagBot/logs/gunicorn_access.log app:app
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

8. **Start the Service**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable ragbot
   sudo systemctl start ragbot
   ```

9. **Verify Installation**
   ```bash
   curl http://localhost:50505/health
   ```

### Maintenance

- **View Logs**
  ```bash
  sudo journalctl -u ragbot
  ```

- **Restart Service**
  ```bash
  sudo systemctl restart ragbot
  ```

- **Update Application**
  ```bash
  cd ~/apps/RagBot
  git pull
  source venv/bin/activate
  pip install -r ragbot-backend/requirements.txt
  sudo systemctl restart ragbot
  ```

## Development

### Backend Development

1. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

2. Run the development server:
   ```bash
   cd ragbot-backend
   flask run
   ```

### Frontend Development

1. Install dependencies:
   ```bash
   cd ragbot-frontend
   npm install
   ```

2. Start the development server:
   ```bash
   npm start
   ```

## Security Considerations

- Keep API keys secure and never commit them to version control
- Regularly update dependencies to patch security vulnerabilities
- Use strong JWT secrets
- Implement proper access controls
- Monitor system logs for suspicious activity

## Troubleshooting

1. **Service Won't Start**
   - Check logs: `sudo journalctl -u ragbot`
   - Verify Python path and virtual environment
   - Check file permissions

2. **GPU Issues**
   - Verify CUDA installation: `nvcc --version`
   - Check GPU status: `nvidia-smi`
   - Ensure correct faiss-gpu version is installed

3. **Memory Issues**
   - Monitor memory usage: `free -h`
   - Adjust worker count in gunicorn configuration
   - Consider increasing swap space

## License

[Your License Here]

## Contributing

[Your Contributing Guidelines Here]