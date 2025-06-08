# Gunicorn configuration file
import multiprocessing

# Number of worker processes
workers = 2  # Reduced to prevent OOM with heavy ML models

# Worker class - use sync for CUDA compatibility
worker_class = 'sync'  # Required for CUDA multiprocessing compatibility

# Number of workers (single worker for CUDA)
workers = 1

# Maximum number of simultaneous clients
worker_connections = 1000

# Timeout for worker processes
timeout = 7200  # Increased for image processing operations

# Maximum number of requests a worker will process before restarting
max_requests = 1000
max_requests_jitter = 50

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Process naming
proc_name = 'ragbot'

# Server socket
bind = '0.0.0.0:50505'

# SSL (uncomment and configure if using HTTPS)
# keyfile = '/path/to/keyfile'
# certfile = '/path/to/certfile'

# Keep-alive settings
keepalive = 30

# Graceful timeout for worker shutdown
graceful_timeout = 120

# Preload app for faster worker startup
preload_app = True

# Memory optimization - use gthread for better async support with ML models

# NOTE: Port 50505 is used for production in Azure Container Apps. Make sure this matches the exposed port in your Azure configuration. 