# Build stage
FROM python:3.11-slim as builder

WORKDIR /code

# Install build dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Final stage
FROM python:3.11-slim

WORKDIR /code

# Copy Python packages and gunicorn
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/gunicorn /usr/local/bin/gunicorn

# Copy system dependencies
COPY --from=builder /usr/bin/tesseract /usr/bin/tesseract
COPY --from=builder /usr/bin/pdftoppm /usr/bin/pdftoppm
COPY --from=builder /usr/lib/x86_64-linux-gnu/libglib-2.0.so.0 /usr/lib/x86_64-linux-gnu/

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p uploads/documents uploads/images data conversations chroma_db

# Expose the port the app runs on
EXPOSE 50505

# Set environment variable for OpenMP to avoid library conflicts
ENV KMP_DUPLICATE_LIB_OK=TRUE

# Command to run the application using gunicorn with 2 workers
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:50505", "--workers", "2", "--threads", "2", "--timeout", "120", "app:app"] 