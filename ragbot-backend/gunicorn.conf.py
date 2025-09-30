# Gunicorn configuration file
import multiprocessing

max_requests = 1000
max_requests_jitter = 50

log_file = "-"

import os

# Support configurable port via environment variable
port = os.getenv("PORT", "50506")
bind = f"0.0.0.0:{port}"

# Reduced workers to prevent OOM with heavy ML models
workers = 2
threads = 4

# Increased timeouts for image processing operations
timeout = 7200
keepalive = 30

# Memory optimization
worker_class = "sync"
worker_connections = 100
max_worker_connections = 1000

# NOTE: Port is configurable via PORT environment variable (default: 50506). Make sure this matches the exposed port in your Azure configuration. 