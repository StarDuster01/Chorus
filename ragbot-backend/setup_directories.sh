#!/bin/bash

# Create all required directories
mkdir -p uploads/images
mkdir -p uploads/documents
mkdir -p data
mkdir -p conversations
mkdir -p chroma_db
mkdir -p datasets
mkdir -p bots
mkdir -p image_indices

# Set permissions
chmod -R 755 uploads
chmod -R 755 data
chmod -R 755 conversations
chmod -R 755 chroma_db
chmod -R 755 datasets
chmod -R 755 bots
chmod -R 755 image_indices

echo "Directories created and permissions set" 