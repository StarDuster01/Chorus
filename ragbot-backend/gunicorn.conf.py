# Gunicorn configuration for RagBot production deployment

import multiprocessing
import os

# Server socket
bind = "0.0.0.0:50505"  # Changed to 0.0.0.0 to allow external access
backlog = 2048

# Worker processes
workers = 1  # Single worker for GPU applications
worker_class = "sync"
worker_connections = 1000
timeout = 300
keepalive = 2

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stdout
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = "ragbot"

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Performance tuning for large file uploads
worker_tmp_dir = '/dev/shm'  # Use shared memory for better performance

# Server hooks
def on_starting(server):
    print("\n=== RagBot Server Starting ===")
    print(f"Workers: {workers}")
    print(f"Worker Class: {worker_class}")
    print(f"Timeout: {timeout}")
    print(f"Log Level: {loglevel}")
    print("=============================\n")

def on_exit(server):
    print("\n=== RagBot Server Shutting Down ===\n")

# Ensure all output is unbuffered
os.environ["PYTHONUNBUFFERED"] = "1"

# SSL (uncomment and configure if using HTTPS)
# keyfile = '/path/to/keyfile'
# certfile = '/path/to/certfile'

# Graceful timeout for worker shutdown
graceful_timeout = 120

# Memory optimization - use gthread for better async support with ML models

# NOTE: Port 50505 is used for production in Azure Container Apps. Make sure this matches the exposed port in your Azure configuration. 