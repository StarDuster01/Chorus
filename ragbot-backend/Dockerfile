# Use official Python image as base
FROM python:3.11-slim

WORKDIR /code

# Install system dependencies including Tesseract OCR
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
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend codee
COPY . .

# Expose the port Flask runs on (configurable via PORT env var, default 50506)
EXPOSE 50506

# Create Directory
RUN mkdir -p /code/uploads/images /code/uploads/documents

# Start the Flask app
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]