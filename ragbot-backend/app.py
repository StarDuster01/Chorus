import os
import json
import base64
import uuid
import datetime
import sys  # Add sys import
from datetime import UTC, timezone  # Import UTC for timezone-aware datetime objects and timezone for timezone-aware datetime objects
import tempfile
import bcrypt
import jwt
import re
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from dotenv import load_dotenv
import openai
import anthropic
import requests  # For Groq API
# ChromaDB imports removed - using connection pool instead
from PIL import Image  # Add this import for image processing
from image_processor import ImageProcessor
from constants import VECTOR_DIMENSION
from handlers.image_handlers import remove_image_handler
import faiss
import io
import numpy as np
# Import text extraction functions from the new module


from handlers.document_handlers import get_document_content_handler
from handlers.document_handlers import get_context_snippet_handler
from handlers.document_handlers import get_original_document_handler
from handlers.document_handlers import download_document_handler
from handlers.image_handlers import search_dataset_images_handler   
# Import image handlers
from handlers.image_handlers import (
    resize_image,
    generate_image_handler,
    enhance_prompt_handler,
    get_image_handler,
    edit_image_handler,
    get_dataset_images_handler
)
# Import conversation handlers
from handlers.conversation_handlers import (
    get_conversations_handler,
    get_conversation_handler,
    delete_conversation_handler,
    delete_all_conversations_handler,
    rename_conversation_handler
)
# Import dataset handlers
from handlers.dataset_handlers import (
    find_dataset_by_id,
    sync_datasets_with_collections,
    get_mime_types_for_dataset,
    get_datasets_handler,
    create_dataset_handler,
    delete_dataset_handler,
    get_dataset_type_handler,
    remove_document_handler,
    rebuild_dataset_handler,
    upload_image_handler,
    bulk_upload_handler,
    get_upload_status_handler
)
# Import auth handlers
from handlers.auth_handlers import (
    get_token_from_header,
    verify_token,
    require_auth,
    register_handler,
    login_handler
)
# Import bot handlers
from handlers.bot_handlers import (
    get_bots_handler,
    create_bot_handler,
    delete_bot_handler,
    get_bot_datasets_handler,
    add_dataset_to_bot_handler,
    remove_dataset_from_bot_handler,
    set_bot_datasets_handler
)
# Import chorus handlers
from handlers.chorus_handlers import (
    get_chorus_config_handler,
    save_chorus_config_handler,
    set_bot_chorus_handler,
    list_choruses_handler,
    create_chorus_handler,
    get_chorus_handler,
    update_chorus_handler,
    delete_chorus_handler
)

from handlers.document_handlers import get_document_content_handler
from handlers.document_handlers import get_context_snippet_handler
from handlers.document_handlers import get_original_document_handler
from handlers.image_handlers import search_dataset_images_handler  
from handlers.dataset_handlers import upload_document_handler
from handlers.dataset_handlers import dataset_status_handler    # add near other imports
from handlers.dataset_handlers import get_dataset_documents_handler   # add near the other imports
from handlers.chat_handler import chat_with_bot_handler
from handlers.chat_handler import chat_with_image_handler
# Import constants
from constants import DEFAULT_LLM_MODEL
from constants import IMAGE_GENERATION_MODEL

# Import global model manager for pre-loading models
from image_processor import global_model_manager
# Load environment variables
load_dotenv()

# Set OpenMP environment variable to avoid library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



# Set up necessary directories
app_base_dir = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(app_base_dir, "data")
os.makedirs(DATA_FOLDER, exist_ok=True)

# Initialize Flask app
app = Flask(__name__, static_folder='frontend', static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024 * 20  # 20GB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
app.config['TEMP_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], "temp")
app.config['PROCESSING_TIMEOUT'] = 3600  # 1 hour timeout for processing

CORS(app)

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configure Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Validate Groq configuration at startup
if GROQ_API_KEY and GROQ_API_KEY.startswith("gsk_"):
    print(f"[STARTUP] Groq API configured (key: {GROQ_API_KEY[:10]}...)")
else:
    print(f"[STARTUP] Warning: Groq API key may be invalid or missing")

# Configure JWT
app.config['JWT_SECRET_KEY'] = os.getenv("JWT_SECRET", "default-dev-secret")
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = datetime.timedelta(hours=24)

# Configure upload folders
from constants import DOCUMENT_FOLDER, IMAGE_FOLDER, UPLOAD_FOLDER


# Create folders if they don't exist
os.makedirs(DOCUMENT_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Initialize ChromaDB
import chroma_client

# Pre-load AI models at startup for better performance
print("[STARTUP] Pre-loading AI models...")
try:
    global_model_manager.load_models_once()
except Exception as e:
    print(f"[STARTUP] Warning: Failed to pre-load models: {str(e)}")
    print("[STARTUP] Models will be loaded on-demand (slower performance)")

# Initialize ImageProcessor for image RAG
app_base_dir = os.path.dirname(os.path.abspath(__file__))
image_processor = ImageProcessor(app_base_dir)

# Create directories for storing conversations
CONVERSATIONS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conversations")
os.makedirs(CONVERSATIONS_FOLDER, exist_ok=True)

# Initialize ChromaDB at startup (blocking until ready)
print("[STARTUP] Initializing ChromaDB...")
chroma_client.initialize_chroma()
print("[STARTUP] ChromaDB ready!")

# Run the sync on app startup
from handlers.dataset_handlers import sync_datasets_with_collections
sync_datasets_with_collections()

# Update the authentication decorator to use the imported module
def require_auth_wrapper(f):
    return require_auth(app.config['JWT_SECRET_KEY'])(f)


# Authentication routes
@app.route('/api/auth/register', methods=['POST'])
def register():
    return register_handler(app.config['JWT_SECRET_KEY'], app.config['JWT_ACCESS_TOKEN_EXPIRES'])

@app.route('/api/auth/login', methods=['POST'])
def login():
    return login_handler(app.config['JWT_SECRET_KEY'], app.config['JWT_ACCESS_TOKEN_EXPIRES'])

# Dataset routes
@app.route('/api/datasets', methods=['GET'])
@require_auth_wrapper
def get_datasets(user_data):
    return get_datasets_handler(user_data)

@app.route('/api/datasets', methods=['POST'])
@require_auth_wrapper
def create_dataset(user_data):
    return create_dataset_handler(user_data)

@app.route('/api/datasets/<dataset_id>/documents', methods=['POST'])
@require_auth_wrapper
def upload_document(user_data, dataset_id):
    # Delegate all heavy lifting to the handler
    return upload_document_handler(user_data, dataset_id)

@app.route('/health')
def health():
    """Simple health check for Kubernetes probes - independent of model loading"""
    return "OK", 200

@app.route('/health/models')
def model_health():
    """Detailed health check including model status"""
    try:
        model_status = {
            "models_loaded": global_model_manager._models_loaded,
            "status": "healthy" if global_model_manager._models_loaded else "loading"
        }
        return jsonify(model_status), 200
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route('/api/datasets/<dataset_id>/documents/<document_id>', methods=['DELETE'])
@require_auth_wrapper
def remove_document(user_data, dataset_id, document_id):
    """Delete a document from a dataset"""
    return remove_document_handler(user_data, dataset_id, document_id)

# Admin routes
@app.route('/api/admin/rebuild_dataset/<dataset_id>', methods=['POST'])
@require_auth_wrapper
def rebuild_dataset(user_data, dataset_id):
    """Rebuild a dataset's ChromaDB collection"""
    return rebuild_dataset_handler(user_data, dataset_id)

@app.route('/api/datasets/<dataset_id>/status', methods=['GET'])
@require_auth_wrapper
def dataset_status(user_data, dataset_id):
    # delegate to the handler
    return dataset_status_handler(user_data, dataset_id)

@app.route('/api/datasets/<dataset_id>/documents', methods=['GET'])
@require_auth_wrapper
def get_dataset_documents(user_data, dataset_id):
    # delegate to handler
    return get_dataset_documents_handler(user_data, dataset_id)

@app.route('/api/datasets/<dataset_id>', methods=['DELETE'])
@require_auth_wrapper
def delete_dataset(user_data, dataset_id):
    return delete_dataset_handler(user_data, dataset_id)


@app.route('/api/bots/<bot_id>/chat', methods=['POST'])
@require_auth_wrapper
def chat_with_bot(user_data, bot_id):
    return chat_with_bot_handler(user_data, bot_id)

# New endpoints for conversation management
@app.route('/api/bots/<bot_id>/conversations', methods=['GET'])
@require_auth_wrapper
def get_conversations(user_data, bot_id):
    """Get all conversations for a specific bot"""
    return get_conversations_handler(user_data, bot_id, CONVERSATIONS_FOLDER)

@app.route('/api/bots/<bot_id>/conversations/<conversation_id>', methods=['GET'])
@require_auth_wrapper
def get_conversation(user_data, bot_id, conversation_id):
    """Get a specific conversation with all messages"""
    return get_conversation_handler(user_data, bot_id, conversation_id, CONVERSATIONS_FOLDER)

@app.route('/api/bots/<bot_id>/conversations/<conversation_id>', methods=['DELETE'])
@require_auth_wrapper
def delete_conversation(user_data, bot_id, conversation_id):
    """Delete a specific conversation"""
    return delete_conversation_handler(user_data, bot_id, conversation_id, CONVERSATIONS_FOLDER)

@app.route('/api/bots/<bot_id>/conversations', methods=['DELETE'])
@require_auth_wrapper
def delete_all_conversations(user_data, bot_id):
    """Delete all conversations for a specific bot"""
    return delete_all_conversations_handler(user_data, bot_id, CONVERSATIONS_FOLDER)

@app.route('/api/bots/<bot_id>/conversations/<conversation_id>/rename', methods=['POST'])
@require_auth_wrapper
def rename_conversation(user_data, bot_id, conversation_id):
    """Rename a conversation"""
    data = request.json
    new_title = data.get('title')
    return rename_conversation_handler(user_data, bot_id, conversation_id, new_title, CONVERSATIONS_FOLDER)

# Add model chorus API endpoints
@app.route('/api/bots/<bot_id>/chorus', methods=['GET'])
@require_auth_wrapper
def get_chorus_config(user_data, bot_id):
    return get_chorus_config_handler(user_data, bot_id)

@app.route('/api/bots/<bot_id>/chorus', methods=['POST'])
@require_auth_wrapper
def save_chorus_config(user_data, bot_id):
    return save_chorus_config_handler(user_data, bot_id)

@app.route('/api/bots/<bot_id>/set-chorus', methods=['POST'])
@require_auth_wrapper
def set_bot_chorus(user_data, bot_id):
    return set_bot_chorus_handler(user_data, bot_id)

@app.route('/api/choruses', methods=['GET'])
@require_auth_wrapper
def list_choruses(user_data):
    return list_choruses_handler(user_data)

@app.route('/api/choruses', methods=['POST'])
@require_auth_wrapper
def create_chorus(user_data):
    return create_chorus_handler(user_data)

@app.route('/api/choruses/<chorus_id>', methods=['GET'])
@require_auth_wrapper
def get_chorus(user_data, chorus_id):
    return get_chorus_handler(user_data, chorus_id)

@app.route('/api/choruses/<chorus_id>', methods=['PUT'])
@require_auth_wrapper
def update_chorus(user_data, chorus_id):
    return update_chorus_handler(user_data, chorus_id)

@app.route('/api/choruses/<chorus_id>', methods=['DELETE'])
@require_auth_wrapper
def delete_chorus(user_data, chorus_id):
    return delete_chorus_handler(user_data, chorus_id)

# Bots routes
@app.route('/api/bots', methods=['GET'])
@require_auth_wrapper
def get_bots(user_data):
    return get_bots_handler(user_data)

@app.route('/api/bots', methods=['POST'])
@require_auth_wrapper
def create_bot(user_data):
    return create_bot_handler(user_data)

@app.route('/api/bots/<bot_id>', methods=['DELETE'])
@require_auth_wrapper
def delete_bot(user_data, bot_id):
    return delete_bot_handler(user_data, bot_id)

# Add new endpoints for managing datasets in a bot
@app.route('/api/bots/<bot_id>/datasets', methods=['GET'])
@require_auth_wrapper
def get_bot_datasets(user_data, bot_id):
    return get_bot_datasets_handler(user_data, bot_id)

@app.route('/api/bots/<bot_id>/datasets', methods=['POST'])
@require_auth_wrapper
def add_dataset_to_bot(user_data, bot_id):
    return add_dataset_to_bot_handler(user_data, bot_id)

@app.route('/api/bots/<bot_id>/datasets/<dataset_id>', methods=['DELETE'])
@require_auth_wrapper
def remove_dataset_from_bot(user_data, bot_id, dataset_id):
    return remove_dataset_from_bot_handler(user_data, bot_id, dataset_id)

# Image generation route
@app.route('/api/images/generate', methods=['POST'])
@require_auth_wrapper
def generate_image(user_data):
    return generate_image_handler(user_data, IMAGE_FOLDER)

# Add a new endpoint for enhancing prompts with GPT-4o
@app.route('/api/images/enhance-prompt', methods=['POST'])
@require_auth_wrapper
def enhance_prompt(user_data):
    return enhance_prompt_handler(user_data)
        
# Route to serve generated images
@app.route('/api/images/<filename>', methods=['GET'])
def get_image(filename):
    """Serve an uploaded image file"""
    return get_image_handler(IMAGE_FOLDER, filename)


@app.route('/api/images/edit', methods=['POST'])
@require_auth_wrapper
def edit_image(user_data):
    return edit_image_handler(user_data)

@app.route('/api/bots/<bot_id>/chat-with-image', methods=['POST'])
@require_auth_wrapper
def chat_with_image(user_data, bot_id):
    return chat_with_image_handler(user_data, bot_id)

   
# This will serve the React frontend from the bundled location
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    if path.startswith('api/'):
        # Let API routes be handled as usual
        return None
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/datasets/<dataset_id>/documents/<document_id>/download/<filename>', methods=['GET'])
@require_auth_wrapper
def download_document(user_data, dataset_id, document_id, filename):
    return download_document_handler(user_data, dataset_id, document_id, filename)

@app.route('/api/datasets/<dataset_id>/images', methods=['POST'])
@require_auth_wrapper
def upload_image(user_data, dataset_id):
    """Upload an image to a dataset (already implemented in dataset_handlers)"""
    return upload_image_handler(user_data, dataset_id, IMAGE_FOLDER)

@app.route('/api/datasets/<dataset_id>/images/<image_id>', methods=['DELETE'])
@require_auth_wrapper
def remove_image(user_data, dataset_id, image_id):
    """Remove an image via image_handlers.remove_image_handler"""
    return remove_image_handler(user_data, dataset_id, image_id)

@app.route('/api/datasets/<dataset_id>/search-images', methods=['POST'])
@require_auth_wrapper
def search_dataset_images(user_data, dataset_id):
    return search_dataset_images_handler(user_data, dataset_id)

@app.route('/api/datasets/<dataset_id>/type', methods=['GET'])
@require_auth_wrapper
def get_dataset_type(user_data, dataset_id):
    """Get the type of a dataset (text, image, or mixed) to inform frontend file selection"""
    return get_dataset_type_handler(user_data, dataset_id)


@app.route('/api/documents/<document_id>/content', methods=['GET'])
@require_auth_wrapper
def get_document_content(user_data, document_id):
    return get_document_content_handler(user_data, document_id)

@app.route('/api/context/<document_id>', methods=['GET'])
@require_auth_wrapper
def get_context_snippet(user_data, document_id):
    return get_context_snippet_handler(user_data, document_id)

@app.route('/api/documents/<document_id>/original', methods=['GET'])
@require_auth_wrapper
def get_original_document(user_data, document_id):
    return get_original_document_handler(user_data, document_id)

@app.route('/api/bots/<bot_id>/set-datasets', methods=['POST'])
@require_auth_wrapper
def set_bot_datasets(user_data, bot_id):
    """Replace all datasets on a bot with a new list of datasets"""
    return set_bot_datasets_handler(user_data, bot_id)

@app.route('/api/datasets/<dataset_id>/bulk-upload', methods=['POST'])
@require_auth_wrapper
def bulk_upload(user_data, dataset_id):
    """Bulk upload a zip file of documents/images to a dataset"""
    from handlers.dataset_handlers import bulk_upload_handler
    return bulk_upload_handler(user_data, dataset_id)

@app.route('/api/datasets/<dataset_id>/upload-status/<status_id>', methods=['GET'])
@require_auth_wrapper
def get_upload_status(user_data, dataset_id, status_id):
    return get_upload_status_handler(user_data, dataset_id, status_id)

# GET dataset images listing
@app.route('/api/datasets/<dataset_id>/images', methods=['GET'])
@require_auth_wrapper
def list_dataset_images(user_data, dataset_id):
    return get_dataset_images_handler(user_data, dataset_id)

if __name__ == '__main__':
    # Development mode only - use gunicorn for production
    import os
    debug_mode = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    port = int(os.getenv('PORT', 50505))
    host = os.getenv('HOST', '0.0.0.0')
    
    print(f"Starting RagBot on {host}:{port} (debug={debug_mode})")
    app.run(host=host, port=port, debug=debug_mode)
