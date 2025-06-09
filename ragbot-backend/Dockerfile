# Frontend build stage
FROM node:16-alpine as frontend-build

WORKDIR /app

# Copy frontend package files and install dependencies
COPY ragbot-frontend/package*.json ./
RUN npm install

# Copy frontend source files
COPY ragbot-frontend/ .

# Set the API URL to use a relative path
ENV REACT_APP_API_URL=/api

# Build the frontend
RUN npm run build

# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set environment variables (static ones first)
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Install system dependencies (this should be cached)
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /code

# Copy only requirements first to leverage Docker cache
COPY ragbot-backend/requirements.txt .

# Install Python dependencies (this should be cached unless requirements.txt changes)
RUN python3.11 -m pip install -r requirements.txt

# Create necessary directories
RUN mkdir -p /code/uploads/images /code/uploads/documents /code/logs

# NOW add the build arguments that change every build (after expensive operations)
ARG BUILD_DATE
ARG BUILD_VERSION=1.1.5
ARG VCS_REF

# Labels for version tracking
LABEL maintainer="RagBot Team" \
      version="${BUILD_VERSION}" \
      build-date="${BUILD_DATE}" \
      vcs-ref="${VCS_REF}" \
      description="RagBot Application with GPU-based SentenceTransformer Embeddings"

# Add dynamic environment variable
ENV RAGBOT_VERSION=${BUILD_VERSION}

# Copy application code
COPY ragbot-backend/ .

# Copy frontend build to the correct location
COPY --from=frontend-build /app/build /code/frontend

# Expose port
EXPOSE 50505

# Create start script with version info
RUN echo '#!/bin/bash\necho "Starting RagBot v${RAGBOT_VERSION} built on ${BUILD_DATE}"\nexec python3.11 -m gunicorn -c gunicorn.conf.py app:app\n' > /code/start.sh && chmod +x /code/start.sh

# Start the application
CMD ["/code/start.sh"]