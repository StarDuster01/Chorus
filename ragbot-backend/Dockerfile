FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    wget \
    curl \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    pkg-config \
    libcairo2-dev \
    libpango1.0-dev \
    libglib2.0-dev \
    libjpeg-dev \
    libpng-dev \
    libwebp-dev \
    poppler-utils \
    imagemagick \
    libwmf-dev \
    libwmf0.2-7 \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.11
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.11 get-pip.py && \
    rm get-pip.py

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /code

# Install dependencies
COPY requirements.txt ./
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support first
# Install latest PyTorch with CUDA support + safetensors for secure model loading
RUN python3.11 -m pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
RUN python3.11 -m pip install safetensors

# Install FAISS-CPU (more reliable than GPU version for deployment)
# GPU acceleration will come from PyTorch and transformers
RUN python3.11 -m pip install faiss-cpu

# Copy backend code
COPY . .

# Expose the port Flask runs on
EXPOSE 50505

# Create Directory
RUN mkdir -p /code/uploads/images /code/uploads/documents

# Create startup script
RUN echo '#!/bin/bash\n\
# Start the Flask app\n\
exec python3.11 -m gunicorn -c gunicorn.conf.py app:app\n\
' > /code/start.sh && chmod +x /code/start.sh

# Start the Flask app
CMD ["/code/start.sh"]