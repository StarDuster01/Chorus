# Gunicorn configuration file
import multiprocessing

max_requests = 1000
max_requests_jitter = 50

log_file = "-"

bind = "0.0.0.0:50505"

# Reduced workers to prevent OOM with heavy ML models
workers = 2
threads = 4

# Increased timeouts for image processing operations
timeout = 600
keepalive = 30

# Memory optimization
worker_class = "sync"
worker_connections = 100
max_worker_connections = 1000

# NOTE: Port 50505 is used for production in Azure Container Apps. Make sure this matches the exposed port in your Azure configuration. 