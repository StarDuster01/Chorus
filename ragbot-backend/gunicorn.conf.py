# Gunicorn configuration for RagBot production deployment

import multiprocessing
import os

# Server socket
bind = "127.0.0.1:50505"
backlog = 2048

# Worker processes
workers = min(multiprocessing.cpu_count(), 4)  # Limit workers due to GPU memory constraints
worker_class = "sync"
worker_connections = 1000
timeout = 300  # 5 minutes for large file uploads
keepalive = 2
max_requests = 1000
max_requests_jitter = 50

# Restart workers after processing this many requests to prevent memory leaks
preload_app = True

# Logging
accesslog = "/home/azureuser/apps/ragbot/logs/gunicorn_access.log"
errorlog = "/home/azureuser/apps/ragbot/logs/gunicorn_error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'ragbot-gunicorn'

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Performance tuning for large file uploads
worker_tmp_dir = '/dev/shm'  # Use shared memory for better performance

# Startup/shutdown
def on_starting(server):
    server.log.info("Starting RagBot server")

def on_reload(server):
    server.log.info("Reloading RagBot server")

def worker_int(worker):
    worker.log.info("Worker received INT or QUIT signal")

def pre_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def worker_abort(worker):
    worker.log.info("Worker received SIGABRT signal")

# SSL (uncomment and configure if using HTTPS)
# keyfile = '/path/to/keyfile'
# certfile = '/path/to/certfile'

# Graceful timeout for worker shutdown
graceful_timeout = 120

# Memory optimization - use gthread for better async support with ML models

# NOTE: Port 50505 is used for production in Azure Container Apps. Make sure this matches the exposed port in your Azure configuration. 