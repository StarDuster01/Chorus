# Use official Python image as base
FROM python:3.11-slim

WORKDIR /code

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . .

# Expose the port Flask runs on
EXPOSE 50505

# Create  Directory
RUN mkdir -p /code/uploads/images /code/uploads/documents
# Start the Flask app
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]