FROM python:3.11-slim

WORKDIR /code

# Install system dependencies for Pillow, Tesseract, and PDF tools
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better cache usage
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# Copy all application files
COPY . .

# Create necessary directories
RUN mkdir -p uploads/documents uploads/images data conversations chroma_db

# Expose the port the app runs on
EXPOSE 50505

# Set environment variable for OpenMP to avoid library conflicts
ENV KMP_DUPLICATE_LIB_OK=TRUE

# Command to run the application using gunicorn
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:50505", "app:app"] 