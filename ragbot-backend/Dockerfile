# Use official Python image as base
FROM python:3.11-slim

WORKDIR /code

# Install system dependencies including Tesseract OCR and WMF support
RUN apt-get update && apt-get install -y \
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

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . .

# Expose the port Flask runs on
EXPOSE 50505

# Create Directory
RUN mkdir -p /code/uploads/images /code/uploads/documents

# Create startup script
RUN echo '#!/bin/bash\n\
# Start the Flask app\n\
exec gunicorn -c gunicorn.conf.py app:app\n\
' > /code/start.sh && chmod +x /code/start.sh

# Start the Flask app
CMD ["/code/start.sh"]