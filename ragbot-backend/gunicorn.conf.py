# Gunicorn configuration file
import multiprocessing

max_requests = 1000
max_requests_jitter = 50

log_file = "-"

bind = "0.0.0.0:50505"

workers = (multiprocessing.cpu_count() * 2) + 1
threads = workers

timeout = 300  # Increased timeout for long-running AI operations

# NOTE: Port 50505 is used for production in Azure Container Apps. Make sure this matches the exposed port in your Azure configuration. 