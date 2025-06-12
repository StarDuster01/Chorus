import os
DEFAULT_LLM_MODEL = "gpt-4.1-nano-2025-04-14"  # Default LLM model to use throughout the application
IMAGE_GENERATION_MODEL = "gpt-image-1"  
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
IMAGE_FOLDER = os.path.join(UPLOAD_FOLDER, "images")
DOCUMENT_FOLDER = os.path.join(UPLOAD_FOLDER, "documents")
VECTOR_DIMENSION = 512