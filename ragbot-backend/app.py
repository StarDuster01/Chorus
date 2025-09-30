import os
import json
import base64
import uuid
import datetime
import sys  # Add sys import
from datetime import UTC  # Import UTC for timezone-aware datetime objects
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
import chromadb
from chromadb.utils import embedding_functions
from PIL import Image  # Add this import for image processing
from image_processor import ImageProcessor
import faiss
import io
import numpy as np
# Import text extraction functions from the new module
from text_extractors import (
    extract_text_from_image,
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_pptx,  # now returns (text, image_metadata)
    extract_text_from_file,
    create_semantic_chunks,
    chunk_powerpoint_content
)
# Import image handlers
from image_handlers import (
    resize_image,
    generate_image_handler,
    enhance_prompt_handler,
    get_image_handler
)
# Import conversation handlers
from conversation_handlers import (
    get_conversations_handler,
    get_conversation_handler,
    delete_conversation_handler,
    delete_all_conversations_handler,
    rename_conversation_handler
)
# Import dataset handlers
from dataset_handlers import (
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
    bulk_upload_handler
)
# Import auth handlers
from auth_handlers import (
    get_token_from_header,
    verify_token,
    require_auth,
    register_handler,
    login_handler
)
# Import bot handlers
from bot_handlers import (
    get_bots_handler,
    create_bot_handler,
    delete_bot_handler,
    get_bot_datasets_handler,
    add_dataset_to_bot_handler,
    remove_dataset_from_bot_handler,
    set_bot_datasets_handler
)
# Import chorus handlers
from chorus_handlers import (
    get_chorus_config_handler,
    save_chorus_config_handler,
    set_bot_chorus_handler,
    list_choruses_handler,
    create_chorus_handler,
    get_chorus_handler,
    update_chorus_handler,
    delete_chorus_handler
)
# Import constants
from constants import DEFAULT_LLM_MODEL
from constants import IMAGE_GENERATION_MODEL
# Load environment variables
load_dotenv()

# Set OpenMP environment variable to avoid library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Define supported file extensions
text_extensions = ['.pdf', '.docx', '.txt', '.pptx']
image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']

# Set up necessary directories
app_base_dir = os.path.dirname(os.path.abspath(__file__))

# Support external data directory via environment variable (e.g., for Azure VM mounts)
# Default to local "data" folder if EXTERNAL_DATA_DIR is not set
external_data_dir = os.getenv("EXTERNAL_DATA_DIR")
if external_data_dir and os.path.exists(external_data_dir):
    DATA_FOLDER = external_data_dir
    print(f"Using external data directory: {DATA_FOLDER}")
else:
    DATA_FOLDER = os.path.join(app_base_dir, "data")
    print(f"Using local data directory: {DATA_FOLDER}")
    
os.makedirs(DATA_FOLDER, exist_ok=True)

# Initialize Flask app
app = Flask(__name__, static_folder='frontend', static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 1000 MB upload limit
CORS(app)

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configure Anthropic
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "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"))

# Configure Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Configure JWT
app.config['JWT_SECRET_KEY'] = os.getenv("JWT_SECRET", "default-dev-secret")
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = datetime.timedelta(hours=24)

# Configure upload folders
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
DOCUMENT_FOLDER = os.path.join(UPLOAD_FOLDER, "documents")
IMAGE_FOLDER = os.path.join(UPLOAD_FOLDER, "images")

# Create folders if they don't exist
os.makedirs(DOCUMENT_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Initialize ChromaDB
chroma_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
os.makedirs(chroma_db_path, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=chroma_db_path)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-ada-002"
)

# Initialize ImageProcessor for image RAG
app_base_dir = os.path.dirname(os.path.abspath(__file__))
image_processor = ImageProcessor(app_base_dir)

# Create directories for storing conversations
CONVERSATIONS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conversations")
os.makedirs(CONVERSATIONS_FOLDER, exist_ok=True)

# Using the imported sync_datasets_with_collections function

# Run the sync on app startup
sync_datasets_with_collections(chroma_client, openai_ef)

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
    data = request.json
    
    if not data or not data.get('name'):
        return jsonify({"error": "Dataset name is required"}), 400
        
    # Get dataset type, default to "text"
    dataset_type = data.get('type', 'text')
    if dataset_type not in ['text', 'image', 'mixed']:
        return jsonify({"error": "Invalid dataset type. Must be 'text', 'image', or 'mixed'"}), 400
        
    # Create a new dataset
    dataset_id = str(uuid.uuid4())
    new_dataset = {
        "id": dataset_id,
        "name": data.get('name'),
        "description": data.get('description', ''),
        "type": dataset_type,
        "user_id": user_data['id'],
        "document_count": 0,
        "image_count": 0,
        "chunk_count": 0,
        "created_at": datetime.datetime.now(UTC).isoformat()
    }
    
    # Save the dataset
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    
    # Check if user already has datasets
    user_datasets_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")
    
    if os.path.exists(user_datasets_file):
        with open(user_datasets_file, 'r') as f:
            datasets = json.load(f)
        datasets.append(new_dataset)
    else:
        datasets = [new_dataset]
    
    with open(user_datasets_file, 'w') as f:
        json.dump(datasets, f)
    
    # Create a ChromaDB collection for this dataset
    try:
        chroma_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
        chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-ada-002"
        )
        
        # Use get_or_create_collection which handles the "already exists" case
        chroma_client.get_or_create_collection(name=dataset_id, embedding_function=openai_ef)
        print(f"Collection for dataset {dataset_id} ensured.", flush=True)
    except Exception as e:
        print(f"Error ensuring ChromaDB collection: {str(e)}", flush=True)
    
    return jsonify(new_dataset), 201

@app.route('/api/datasets/<dataset_id>/documents', methods=['POST'])
@require_auth_wrapper
def upload_document(user_data, dataset_id):
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    # Get dataset information
    dataset = find_dataset_by_id(dataset_id)
    if not dataset:
        return jsonify({"error": "Dataset not found"}), 404
        
    # Check if dataset belongs to user or user is admin
    user_id = user_data.get('id')
    is_admin = user_data.get('isAdmin', False)
    
    # Print debug information
    print(f"Dataset user_id: {dataset.get('user_id')}, Current user_id: {user_id}, isAdmin: {is_admin}", flush=True)
    
    # Handle legacy datasets that might not have user_id
    # 1. If dataset has no user_id, assume it belongs to the current user
    # 2. If dataset has user_id and it doesn't match current user (and user is not admin), deny access
    if 'user_id' in dataset and dataset['user_id'] != user_id and not is_admin:
        return jsonify({"error": "You don't have permission to access this dataset"}), 403
    
    # Check file extension
    filename = secure_filename(file.filename)
    file_extension = os.path.splitext(filename)[1].lower()
    
    # Get dataset type (text, image, or mixed)
    dataset_type = dataset.get("type", "text")
    
    # Check if file type is supported for this dataset
    if dataset_type == "text" and file_extension not in text_extensions:
        return jsonify({"error": f"Unsupported file type for text dataset. Supported types: {', '.join(text_extensions)}"}), 400
    elif dataset_type == "image" and file_extension not in image_extensions:
        return jsonify({"error": f"Unsupported file type for image dataset. Supported types: {', '.join(image_extensions)}"}), 400
    elif dataset_type == "mixed":
        if file_extension not in text_extensions and file_extension not in image_extensions:
            return jsonify({"error": f"Unsupported file type. Supported types: {', '.join(text_extensions + image_extensions)}"}), 400
    
    # If this is an image file and the dataset supports images
    if file_extension in image_extensions and dataset_type in ["image", "mixed"]:
        try:
            # Save the file with a unique name to prevent overwriting
            file_id = str(uuid.uuid4())
            original_filename = filename
            filename = f"{file_id}{file_extension}"
            file_path = os.path.join(IMAGE_FOLDER, filename)
            file.save(file_path)
            
            # Add image to dataset
            
            image_meta = image_processor.add_image_to_dataset(
                dataset_id, 
                file_path,
                {
                    "dataset_id": dataset_id,
                    "original_filename": original_filename,
                    "url": f"/api/images/{filename}",
                    "type": "image"
                }
            )
            
            # Update image count in dataset
            datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
            user_datasets_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")
            
            if os.path.exists(user_datasets_file):
                with open(user_datasets_file, 'r') as f:
                    datasets = json.load(f)
                
                for idx, ds in enumerate(datasets):
                    if ds["id"] == dataset_id:
                        if "image_count" in ds:
                            datasets[idx]["image_count"] += 1
                        else:
                            datasets[idx]["image_count"] = 1
                        break
                
                with open(user_datasets_file, 'w') as f:
                    json.dump(datasets, f)
            
            # Return success
            return jsonify({
                "id": image_meta.get("id", ""),
                "filename": original_filename,
                "type": "image",
                "caption": image_meta.get("caption", ""),
                "url": f"/api/images/{filename}"
            }), 201
            
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({"error": f"Error processing image: {str(e)}"}), 500
    
    # If this is a document file and the dataset supports text
    elif file_extension in text_extensions and dataset_type in ["text", "mixed"]:
        # Save the file
        file_path = os.path.join(DOCUMENT_FOLDER, f"{uuid.uuid4()}_{filename}")
        file.save(file_path)
        # Extract text from file
        if file_extension.lower() == '.pptx':
            text, pptx_image_metadata = extract_text_from_pptx(file_path)
        else:
            text = extract_text_from_file(file_path)
            pptx_image_metadata = []
        if not text:
            os.remove(file_path)
            return jsonify({"error": "Could not extract text from file"}), 400
        
        # For PowerPoint files, clean up the text by removing unnecessary metadata
        if file_extension.lower() == '.pptx':
            # Remove any metadata lines at the beginning
            lines = text.split('\n')
            clean_lines = []
            skip_metadata = True
            
            for line in lines:
                # Skip metadata at the beginning (like Presentation Title, Author, etc.)
                if skip_metadata and (line.startswith("Presentation Title:") or 
                                     line.startswith("Author:") or 
                                     line.startswith("Subject:") or 
                                     line.startswith("Keywords:") or 
                                     line.startswith("Category:") or 
                                     line.startswith("Comments:")):
                    continue
                
                # Once we hit actual slide content, stop skipping
                if line.startswith("## SLIDE "):
                    skip_metadata = False
                
                # Skip speaker notes
                if "Speaker Notes:" in line:
                    continue
                
                clean_lines.append(line)
            
            # Rejoin the cleaned text
            text = '\n'.join(clean_lines)
        
        # Detect PowerPoint content for chunking strategy
        is_powerpoint = file_extension.lower() == '.pptx' or "## SLIDE " in text
        
        # Set chunking parameters based on file type
        if is_powerpoint:
            # For PowerPoint files, use large chunks to keep multiple slides together
            max_chunk_size = 3000
            overlap = 500
        else:
            # Default chunking parameters for other document types
            max_chunk_size = 1000  
            overlap = 200
        
        # Create chunks from text using the semantic chunking algorithm
        chunks = create_semantic_chunks(text, max_chunk_size=max_chunk_size, overlap=overlap)
        
        # Create document
        document_id = str(uuid.uuid4())
        
        # Add chunks to vector store
        try:
            chroma_collection = chroma_client.get_or_create_collection(
                name=dataset_id,
                embedding_function=openai_ef
            )
            
            # Create unique IDs for chunks
            chunk_ids = [f"{document_id}_{i}" for i in range(len(chunks))]
            
            # Add metadata to each chunk
            metadatas = [{
                "document_id": document_id,
                "dataset_id": dataset_id,
                "filename": filename,
                "source": filename,  # Add source field for compatibility
                "file_path": file_path,
                "chunk": i,
                "total_chunks": len(chunks),
                "file_type": file_extension.lower(),
                "is_powerpoint": is_powerpoint,  # Add flag to identify PowerPoint content
                "created_at": datetime.datetime.now(UTC).isoformat(),
            } for i in range(len(chunks))]
            
            # Add chunks to vector store
            chroma_collection.add(
                ids=chunk_ids,
                documents=chunks,
                metadatas=metadatas
            )
            
            # Update document count and chunk count in dataset
            datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
            user_datasets_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")
            
            if os.path.exists(user_datasets_file):
                with open(user_datasets_file, 'r') as f:
                    datasets = json.load(f)
                
                for idx, ds in enumerate(datasets):
                    if ds["id"] == dataset_id:
                        # Update document count
                        if "document_count" in ds:
                            datasets[idx]["document_count"] += 1
                        else:
                            datasets[idx]["document_count"] = 1
                            
                        # Update chunk count
                        if "chunk_count" in ds:
                            datasets[idx]["chunk_count"] += len(chunks)
                        else:
                            datasets[idx]["chunk_count"] = len(chunks)
                        break
                
                with open(user_datasets_file, 'w') as f:
                    json.dump(datasets, f)
            
            # Return success
            return jsonify({
                "id": document_id,
                "filename": filename,
                "type": "text",
                "chunks": len(chunks)
            }), 201
            
        except Exception as e:
            # Clean up the file if there was an error
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
            return jsonify({"error": f"Error processing document: {str(e)}"}), 500
        # After chunking and before returning success, add images from pptx_image_metadata
        if pptx_image_metadata:
            for img_meta in pptx_image_metadata:
                # Add document_id and file_path to image metadata
                img_meta["document_id"] = document_id
                img_meta["file_path"] = file_path
                img_meta["dataset_id"] = dataset_id
                img_meta["type"] = "image"
                # Save image to uploads/images and update path/url
                img_filename = f"{uuid.uuid4()}.png"
                img_save_path = os.path.join(IMAGE_FOLDER, img_filename)
                import shutil
                shutil.copy(img_meta["image_path"], img_save_path)
                img_meta["path"] = img_save_path
                img_meta["url"] = f"/api/images/{img_filename}"
                # Add to image dataset/index
                image_processor.add_image_to_dataset(dataset_id, img_save_path, img_meta)
    else:
        return jsonify({"error": "Unsupported file type"}), 400

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
    # Check if dataset exists and belongs to user
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    user_datasets_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")
    
    if not os.path.exists(user_datasets_file):
        return jsonify({"error": "Dataset not found"}), 404
        
    with open(user_datasets_file, 'r') as f:
        datasets = json.load(f)
        
    dataset = None
    dataset_index = -1
    for i, d in enumerate(datasets):
        if d["id"] == dataset_id:
            dataset = d
            dataset_index = i
            break
            
    if not dataset:
        return jsonify({"error": "Dataset not found"}), 404
    
    # Get dataset type
    dataset_type = dataset.get("type", "text")
    
    # Check ChromaDB collection status for text documents
    collection_exists = False
    doc_count = 0
    chunk_count = 0
    
    if dataset_type in ["text", "mixed"]:
        existing_collections = chroma_client.list_collections()
        collection_exists = dataset_id in existing_collections
        
        if collection_exists:
            try:
                collection = chroma_client.get_collection(name=dataset_id, embedding_function=openai_ef)
                chunk_count = collection.count()
                
                # Get document count by counting unique document_ids
                try:
                    results = collection.get()
                    if results and results["metadatas"]:
                        # Extract unique document IDs
                        document_ids = set()
                        for metadata in results["metadatas"]:
                            if metadata and "document_id" in metadata:
                                document_ids.add(metadata["document_id"])
                        doc_count = len(document_ids)
                except Exception as e:
                    print(f"Error calculating document count: {str(e)}", flush=True)
                    # Fallback to existing document count in dataset
                    doc_count = dataset.get("document_count", 0)
                    
            except Exception as e:
                return jsonify({
                    "dataset": dataset,
                    "collection_exists": False,
                    "error": str(e)
                }), 200
    
    # Check image index status for images
    image_index_exists = False
    image_count = 0
    image_previews = []
    
    if dataset_type in ["image", "mixed"]:
        # Check if there's an image index for this dataset
        indices_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "image_indices")
        index_path = os.path.join(indices_dir, f"{dataset_id}_index.faiss")
        metadata_file = os.path.join(indices_dir, f"{dataset_id}_metadata.json")
        image_index_exists = os.path.exists(index_path)
        
        # Validate metadata if it exists
        if os.path.exists(metadata_file):
            try:
                # Load metadata from file to ensure it's up-to-date
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Filter to ensure we only count images that still exist on disk
                valid_metadata = []
                for img_meta in metadata:
                    # Ensure dataset_id is correct
                    if not img_meta.get('dataset_id'):
                        img_meta['dataset_id'] = dataset_id
                    
                    # Only include images for this dataset where the file still exists
                    if img_meta.get('dataset_id') == dataset_id and 'path' in img_meta:
                        if os.path.exists(img_meta['path']):
                            valid_metadata.append(img_meta)
                
                # Update processor's metadata to remove any non-existent images
                image_processor.image_metadata[dataset_id] = valid_metadata
                
                # Update the index count to match
                image_count = len(valid_metadata)
                
                # Include image previews (first 3 images that exist)
                preview_count = 0
                for img_meta in valid_metadata:
                    if preview_count >= 10:  # Limit to 10 preview images
                        break
                    
                    if 'path' in img_meta and os.path.exists(img_meta['path']):
                        image_previews.append({
                            "id": img_meta.get("id", ""),
                            "url": f"/api/images/{os.path.basename(img_meta['path'])}",
                            "caption": img_meta.get("caption", "")
                        })
                        preview_count += 1
                
                # Save updated metadata
                try:
                    with open(metadata_file, 'w') as f:
                        json.dump(valid_metadata, f)
                    print(f"Updated metadata file for dataset {dataset_id} with {len(valid_metadata)} validated images", flush=True)
                except Exception as e:
                    print(f"Error saving updated metadata: {str(e)}", flush=True)
                
            except Exception as e:
                print(f"Error validating image metadata: {str(e)}", flush=True)
        elif dataset_id in image_processor.image_metadata:
            # If no metadata file but we have metadata in memory, validate it
            valid_metadata = []
            for img_meta in image_processor.image_metadata[dataset_id]:
                if img_meta.get('dataset_id') == dataset_id and 'path' in img_meta:
                    if os.path.exists(img_meta['path']):
                        valid_metadata.append(img_meta)
            
            # Update processor's metadata
            image_processor.image_metadata[dataset_id] = valid_metadata
            image_count = len(valid_metadata)
            
            # Include image previews (first 3 images)
            preview_count = 0
            for img_meta in valid_metadata:
                if preview_count >= 3:  # Limit to 3 preview images
                    break
                
                if 'path' in img_meta and os.path.exists(img_meta['path']):
                    image_previews.append({
                        "id": img_meta.get("id", ""),
                        "url": f"/api/images/{os.path.basename(img_meta['path'])}",
                        "caption": img_meta.get("caption", "")
                    })
                    preview_count += 1
                    
            # Save metadata file
            try:
                os.makedirs(indices_dir, exist_ok=True)
                with open(metadata_file, 'w') as f:
                    json.dump(valid_metadata, f)
                print(f"Created metadata file for dataset {dataset_id} with {len(valid_metadata)} validated images", flush=True)
            except Exception as e:
                print(f"Error saving metadata: {str(e)}", flush=True)
    
    # Update dataset with accurate counts
    dataset["document_count"] = doc_count
    dataset["chunk_count"] = chunk_count
    dataset["image_count"] = image_count
    
    # Save updated dataset counts back to file
    datasets[dataset_index] = dataset
    with open(user_datasets_file, 'w') as f:
        json.dump(datasets, f)
    
    return jsonify({
        "dataset": dataset,
        "collection_exists": collection_exists,
        "document_count": doc_count,
        "chunk_count": chunk_count,
        "image_index_exists": image_index_exists,
        "image_count": image_count,
        "image_previews": image_previews
    }), 200

@app.route('/api/datasets/<dataset_id>/documents', methods=['GET'])
@require_auth_wrapper
def get_dataset_documents(user_data, dataset_id):
    # Check if dataset exists and belongs to user
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    user_datasets_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")
    
    if not os.path.exists(user_datasets_file):
        return jsonify({"error": "Dataset not found"}), 404
        
    with open(user_datasets_file, 'r') as f:
        datasets = json.load(f)
        
    dataset_exists = False
    for dataset in datasets:
        if dataset["id"] == dataset_id:
            dataset_exists = True
            break
            
    if not dataset_exists:
        return jsonify({"error": "Dataset not found"}), 404
    
    # Try to get the collection from ChromaDB
    try:
        collection = chroma_client.get_collection(name=dataset_id, embedding_function=openai_ef)
        
        # Query all documents in the collection
        results = collection.get()
        
        if not results or len(results['ids']) == 0:
            return jsonify({"documents": []}), 200
        
        # Process results to get unique documents with chunk counts
        documents = {}
        for i, metadata in enumerate(results['metadatas']):
            if metadata and 'document_id' in metadata and 'source' in metadata:
                doc_id = metadata['document_id']
                if doc_id not in documents:
                    documents[doc_id] = {
                        'id': doc_id,
                        'filename': metadata['source'],
                        'chunk_count': 1,
                        'file_type': metadata.get('file_type', ''),
                        'created_at': metadata.get('created_at', '')
                    }
                else:
                    documents[doc_id]['chunk_count'] += 1
        
        # Convert to list and sort by creation date (newest first)
        document_list = list(documents.values())
        document_list.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return jsonify({
            "documents": document_list,
            "total_documents": len(document_list),
            "total_chunks": len(results['ids'])
        }), 200
    
    except Exception as e:
        print(f"Error getting documents: {str(e)}", flush=True)
        return jsonify({"error": f"Failed to get documents: {str(e)}"}), 500

@app.route('/api/datasets/<dataset_id>', methods=['DELETE'])
@require_auth_wrapper
def delete_dataset(user_data, dataset_id):
    return delete_dataset_handler(user_data, dataset_id)

# Chat routes
@app.route('/api/bots/<bot_id>/chat', methods=['POST'])
@require_auth_wrapper
def chat_with_bot(user_data, bot_id):
    data = request.json
    message = data.get('message', '')
    debug_mode = data.get('debug_mode', False)
    use_model_chorus = data.get('use_model_chorus', False)  # User's explicit choice to use model chorus
    chorus_id = data.get('chorus_id', '')  # A specific chorus ID to use
    conversation_id = data.get('conversation_id', '')  # The conversation this message belongs to
    
    # Check for image data in base64 format
    image_data = data.get('image_data', '')
    image_path = None
    has_image = False
    
    if not message and not image_data:
        return jsonify({"error": "Message or image is required"}), 400
    
    # Create a new conversation if no conversation_id provided
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    
    # If image is provided, save it to a temporary file
    if image_data:
        has_image = True
        try:
            # Extract the base64 content and file extension
            if ';base64,' in image_data:
                header, encoded = image_data.split(';base64,')
                file_ext = header.split('/')[-1]
            else:
                encoded = image_data
                file_ext = 'png'  # Default to PNG if not specified
            
            # Decode the base64 data
            decoded_image = base64.b64decode(encoded)
            
            # Save to a temporary file
            filename = f"chat_image_{str(uuid.uuid4())}.{file_ext}"
            image_path = os.path.join(IMAGE_FOLDER, filename)
            
            with open(image_path, 'wb') as f:
                f.write(decoded_image)
                
            # Resize image if needed - use the max_dimension parameter
            image_path = resize_image(image_path, max_dimension=1024)
            
            # Update message to include reference to the image
            if not message:
                message = "[Image uploaded]"
            else:
                message = f"{message} [Image uploaded]"
                
        except Exception as img_error:
            print(f"Error processing image: {str(img_error)}", flush=True)
            return jsonify({"error": f"Failed to process image: {str(img_error)}"}), 400
    
    # Create a new message object
    user_message = {
        "id": str(uuid.uuid4()),
        "role": "user",
        "content": message,
        "timestamp": datetime.datetime.now(UTC).isoformat()
    }
    
    # If there's an image, add it to the message metadata
    if has_image:
        user_message["has_image"] = True
        user_message["image_path"] = image_path
    
    # Add message to conversation history
    conversation_file = os.path.join(CONVERSATIONS_FOLDER, f"{user_data['id']}_{bot_id}_{conversation_id}.json")
    conversation_exists = os.path.exists(conversation_file)
    
    if conversation_exists:
        # Load existing conversation
        with open(conversation_file, 'r') as f:
            conversation = json.load(f)
    else:
        # Create new conversation
        conversation = {
            "id": conversation_id,
            "bot_id": bot_id,
            "user_id": user_data['id'],
            "title": message[:40] + "..." if len(message) > 40 else message,  # Use first message as title
            "created_at": datetime.datetime.now(UTC).isoformat(),
            "updated_at": datetime.datetime.now(UTC).isoformat(),
            "messages": []
        }
    
    # Add user message to conversation
    conversation["messages"].append(user_message)
    conversation["updated_at"] = datetime.datetime.now(UTC).isoformat()
    
    # Save updated conversation
    with open(conversation_file, 'w') as f:
        json.dump(conversation, f)
        
    # Get bot info
    bots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bots")
    user_bots_file = os.path.join(bots_dir, f"{user_data['id']}_bots.json")
    
    if not os.path.exists(user_bots_file):
        return jsonify({"error": "Bot not found"}), 404
        
    with open(user_bots_file, 'r') as f:
        bots = json.load(f)
        
    bot = None
    for b in bots:
        if b["id"] == bot_id:
            bot = b
            break
            
    if not bot:
        return jsonify({"error": "Bot not found"}), 404
        
    # Get dataset IDs from the bot
    dataset_ids = bot.get("dataset_ids", [])
    
    # Process image with OpenAI Vision API if an image is present
    if has_image:
        try:
            # Read the image file and encode it as base64
            with open(image_path, 'rb') as img_file:
                encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Prepare the message with the image for OpenAI Vision API
            messages = [
                {
                    "role": "system", 
                    "content": bot.get("system_instruction", "You are a helpful assistant that can analyze images.")
                },
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": message if message != "[Image uploaded]" else "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{file_ext};base64,{encoded_image}"
                            }
                        }
                    ]
                }
            ]
            
            # Call OpenAI API with vision capabilities
            response = openai.chat.completions.create(
                model=DEFAULT_LLM_MODEL,
                messages=messages,
                max_tokens=1024
            )
            
            response_text = response.choices[0].message.content
            
            # Save the response in the conversation
            bot_response = {
                "id": str(uuid.uuid4()),
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.datetime.now(UTC).isoformat(),
                "from_image_analysis": True
            }
            conversation["messages"].append(bot_response)
            with open(conversation_file, 'w') as f:
                json.dump(conversation, f)
                
            return jsonify({
                "response": response_text,
                "conversation_id": conversation_id,
                "image_processed": True
            }), 200
            
        except Exception as vision_error:
            print(f"Error processing image with Vision API: {str(vision_error)}", flush=True)
            # If the vision API fails, continue with the regular RAG process
    
    if not dataset_ids:
        return jsonify({
            "response": "I don't have any datasets to work with. Please add a dataset to help me answer your questions.",
            "conversation_id": conversation_id
        }), 200
    
    # Check if the bot has a chorus configuration associated with it
    # If so, use model chorus by default unless explicitly turned off
    bot_has_chorus = bot.get("chorus_id", "")
    use_model_chorus = use_model_chorus or bool(bot_has_chorus)
    
    # If a specific chorus_id is provided, use it; otherwise use the bot's chorus_id if available
    specific_chorus_id = chorus_id or bot.get("chorus_id", "")
    
    # Retrieve relevant documents from all datasets
    all_contexts = []
    image_results = []
    
    try:
        for dataset_id in dataset_ids:
            try:
                # Text-based document retrieval
                collection = chroma_client.get_collection(name=dataset_id, embedding_function=openai_ef)
                
                # Check if collection has any documents
                collection_count = collection.count()
                if collection_count > 0:
                    # Determine how many results to request based on collection size
                    n_results = min(5, collection_count)  # Default to 5 results
                    
                    # First get a sample to check if we're dealing with PowerPoint content
                    sample_results = collection.query(
                        query_texts=[message],
                        n_results=1
                    )
                    
                    # Check if the sample contains PowerPoint content
                    is_powerpoint = False
                    if sample_results and sample_results["documents"] and sample_results["documents"][0]:
                        sample_text = sample_results["documents"][0][0]
                        if "## SLIDE " in sample_text or "## PRESENTATION SUMMARY ##" in sample_text:
                            is_powerpoint = True
                            # For PowerPoint content, get more chunks to ensure we have enough context
                            n_results = min(10, collection_count)  # Double the number of chunks for PowerPoint
                    
                    # Get the actual results
                    results = collection.query(
                        query_texts=[message],
                        n_results=n_results
                    )
                    
                    all_contexts.extend(results["documents"][0])
                
                # Image-based retrieval if available
                try:
                    # First, check if dataset contains images
                    has_images = False
                    dataset_type = "text"  # Default
                    
                    # Enhanced image query detection
                    image_query_terms = [
                        "image", "picture", "photo", "visual", "diagram", "graph", 
                        "chart", "illustration", "screenshot", "scan", "drawing", 
                        "artwork", "logo", "icon", "figure", "graphic", "view", 
                        "show me", "look like", "appearance", "visual", "display",
                        "find a picture", "find image", "show image", "display the visual",
                        "include the diagram", "include image", "with image", "any image",
                        "the picture", "the image", "the logo", "the diagram", "the illustration"
                    ]
                    
                    # Use a more aggressive check for image queries - partial matches and phrases
                    is_image_query = any(term in message.lower() for term in image_query_terms) or message.lower().strip().startswith('show me')
                    
                    # Debug: Log image query detection decision
                    print(f"IMAGE SEARCH DEBUG: Query '{message}' - Is image query? {is_image_query}", flush=True)
                    if is_image_query:
                        matching_terms = [term for term in image_query_terms if term in message.lower()]
                        print(f"IMAGE SEARCH DEBUG: Matched terms: {matching_terms}", flush=True)
                    
                    # Skip all image processing if this is not an image query
                    if not is_image_query:
                        print(f"IMAGE SEARCH DEBUG: Skipping image retrieval for non-image query: '{message}'", flush=True)
                        continue
                    
                    # If we get here, it is an image query, so check if the dataset has images
                    # Check dataset type and image count from metadata
                    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
                    for user_id in [user_data['id'], "shared"]:  # Check both user and shared datasets
                        user_datasets_file = os.path.join(datasets_dir, f"{user_id}_datasets.json")
                        if os.path.exists(user_datasets_file):
                            with open(user_datasets_file, 'r') as f:
                                datasets = json.load(f)
                                
                                for ds in datasets:
                                    if ds["id"] == dataset_id:
                                        dataset_type = ds.get("type", "text")
                                        if dataset_type in ["image", "mixed"] and ds.get("image_count", 0) > 0:
                                            has_images = True
                                        break
                    
                    # Direct check in image_processor metadata to confirm images exist
                    dataset_has_metadata = dataset_id in image_processor.image_metadata
                    metadata_count = len(image_processor.image_metadata.get(dataset_id, [])) if dataset_has_metadata else 0
                    
                    # If dataset has images according to either source, retrieve them
                    if has_images or metadata_count > 0:
                        print(f"Dataset {dataset_id} has images: dataset says {has_images}, metadata has {metadata_count}", flush=True)
                        
                        # Set search parameters based on whether we're using chorus mode
                        if use_model_chorus:
                            # In chorus mode, more aggressively search for images
                            top_k = 8 if is_image_query else 4
                        else:
                            # In standard mode, still search for images but with lower priority if not an image query
                            top_k = 6 if is_image_query else 3
                        
                        print(f"IMAGE SEARCH DEBUG: Will search for up to {top_k} images", flush=True)
                        
                        # Search for images with the user's query
                        img_results = []
                        try:
                            print(f"IMAGE SEARCH DEBUG: Starting semantic image search for '{message}'", flush=True)
                            img_results = image_processor.search_images(dataset_id, message, top_k=top_k)
                            print(f"IMAGE SEARCH DEBUG: Found {len(img_results)} images for query '{message}' in dataset {dataset_id}", flush=True)
                            if img_results:
                                for i, img in enumerate(img_results):
                                    print(f"IMAGE SEARCH DEBUG: Image {i+1}: {img.get('caption', 'No caption')} (score: {img.get('score', 0):.4f})", flush=True)
                        except Exception as img_search_error:
                            print(f"IMAGE SEARCH ERROR: {str(img_search_error)}", flush=True)
                        
                        # If no results or weak results, also try a more generic search, but only for image queries
                        if (not img_results or (img_results and img_results[0].get("score", 0) < 0.2)) and is_image_query:
                            # Try searching with a more generic query 
                            generic_queries = [
                                "relevant image for this topic",
                                "visual representation",
                                "image related to this subject",
                                "picture about this"
                            ]
                            
                            # Add more specific generic queries for likely image questions
                            if is_image_query:
                                generic_queries.extend([
                                    "show me images about this",
                                    "find relevant visuals",
                                    "diagrams or images for this topic",
                                    "picture showing this",
                                    "visual of this information"
                                ])
                                
                            # For chorus mode, be more aggressive in finding images
                            if use_model_chorus:
                                generic_queries.extend([
                                    "important image",
                                    "key visual",
                                    "significant illustration",
                                    "main diagram",
                                    "helpful picture for reference"
                                ])
                            
                            print(f"IMAGE SEARCH DEBUG: Using fallback generic queries: {generic_queries[:3]}...", flush=True)
                            
                            for generic_query in generic_queries:
                                try:
                                    print(f"IMAGE SEARCH DEBUG: Trying generic query: '{generic_query}'", flush=True)
                                    generic_results = image_processor.search_images(dataset_id, generic_query, top_k=4 if is_image_query else 2)
                                    if generic_results:
                                        for gen_img in generic_results:
                                            # Add to results if not already included
                                            if not any(img.get("id") == gen_img.get("id") for img in img_results):
                                                img_results.append(gen_img)
                                        if len(img_results) >= top_k:
                                            break  # Stop after finding enough results
                                except Exception as generic_search_error:
                                    print(f"Error with generic image search: {str(generic_search_error)}", flush=True)
                                    continue
                        
                        # Add relevant images to results
                        if img_results:
                            for img in img_results:
                                # Add image information to results
                                image_info = {
                                    "id": img["id"],
                                    "caption": img["caption"],
                                    "url": f"/api/images/{os.path.basename(img['path'])}",
                                    "score": img["score"],
                                    "dataset_id": dataset_id
                                }
                                image_results.append(image_info)
                                
                except Exception as img_error:
                    print(f"Error with image retrieval for dataset '{dataset_id}': {str(img_error)}", flush=True)
                    # Continue even if image retrieval fails
                
            except Exception as coll_error:
                print(f"Error with collection '{dataset_id}': {str(coll_error)}", flush=True)
                # Continue with other datasets even if one fails
                continue
                
        if not all_contexts and not image_results:
            # Save the response in the conversation
            bot_response = {
                "id": str(uuid.uuid4()),
                "role": "assistant",
                "content": "I don't have any documents or images in my knowledge base yet. Please upload some content to help me answer your questions.",
                "timestamp": datetime.datetime.now(UTC).isoformat()
            }
            conversation["messages"].append(bot_response)
            with open(conversation_file, 'w') as f:
                json.dump(conversation, f)
                
            return jsonify({
                "response": "I don't have any documents or images in my knowledge base yet. Please upload some content to help me answer your questions.",
                "debug": {"error": "No contexts found in any collections"} if debug_mode else None,
                "conversation_id": conversation_id
            }), 200
            
        # Sort contexts by relevance (they should already be sorted from query results)
        # But we need to truncate to avoid token limits
        
        # Check if we have PowerPoint content in the retrieved contexts
        has_powerpoint = any("## SLIDE " in ctx or "## PRESENTATION SUMMARY ##" in ctx for ctx in all_contexts)
        
        # Adjust the maximum number of contexts based on content type
        if has_powerpoint:
            max_contexts = 12  # Allow more contexts for PowerPoint content
        else:
            max_contexts = 8  # Default limit for regular content
            
        contexts = all_contexts[:max_contexts]
        context_text = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])
        
        # Prepare image information for context
        image_context = ""
        dataset_id_to_name = {}
        if image_results:
            # Build a mapping from dataset_id to dataset name
            for img in image_results:
                dsid = img["dataset_id"]
                if dsid not in dataset_id_to_name:
                    ds = find_dataset_by_id(dsid)
                    if ds and ds.get("name"):
                        dataset_id_to_name[dsid] = ds["name"]
                    else:
                        dataset_id_to_name[dsid] = dsid[:8]  # fallback to short id
            # Sort images by score
            image_results.sort(key=lambda x: x["score"], reverse=True)
            image_query_terms = ["image", "picture", "photo", "visual", "diagram", "graph", "chart", "illustration"]
            is_image_query = any(term in message.lower() for term in image_query_terms) or message.lower().strip().startswith('show me')
            max_images = 5 if is_image_query else 3
            top_images = image_results[:max_images]
            # If more than one dataset, prefix image refs with dataset name
            multi_dataset = len(set(img["dataset_id"] for img in top_images)) > 1
            if is_image_query:
                image_context = "\n\n## RELEVANT IMAGES: ##\n"
            else:
                image_context = "\n\nRelevant Images:\n"
            # When building image_context and image_details, prefix with document/slide info if available
            doc_id_to_name = {}
            for img in image_results:
                docid = img.get("document_id")
                if docid and docid not in doc_id_to_name:
                    docid_val = docid
                    # Try to get filename from image metadata
                    if "filename" in img:
                        docid_val = img["filename"]
                    doc_id_to_name[docid] = docid_val
            multi_doc = len(set(img.get("document_id") for img in top_images if img.get("document_id"))) > 1
            for i, img in enumerate(top_images):
                prefix = ""
                if multi_doc and img.get("document_id"):
                    docname = doc_id_to_name.get(img["document_id"], img["document_id"][:8])
                    slide = img.get("slide_number")
                    slide_title = img.get("slide_title")
                    if slide:
                        prefix = f"[{docname}, Slide {slide}] "
                    else:
                        prefix = f"[{docname}] "
                elif multi_dataset:
                    prefix = f"[{dataset_id_to_name.get(img['dataset_id'], img['dataset_id'][:8])}] "
                image_context += f"{prefix}[Image {i+1}] Caption: {img['caption']}\n"
                image_context += f"            This image is available for viewing and download.\n"
            top_image_urls = [img["url"] for img in top_images]
        
        # Combine text and image contexts
        if context_text and image_context:
            # Instead of keeping them separate, integrate image info into the main context
            # This helps the AI be aware of all information sources at once
            image_query_terms = ["image", "picture", "photo", "visual", "diagram", "graph", "chart", "illustration"]
            is_image_query = any(term in message.lower() for term in image_query_terms) or message.lower().strip().startswith('show me')
            
            # For image queries, prioritize image information
            if is_image_query and image_results:
                full_context = "## RELEVANT INFORMATION: ##\n\n" + image_context + "\n\n" + context_text
            else:
                full_context = "## RELEVANT INFORMATION: ##\n\n" + context_text + "\n\n" + image_context
        elif context_text:
            full_context = context_text
        elif image_context:
            full_context = image_context
        else:
            full_context = "No relevant information found."
        
        # Get source documents information with indices
        source_documents = []
        seen_documents = set()
        source_images = []
        
        # Add text document sources if available
        if all_contexts:
            for i, (result, metadata) in enumerate(zip(results["documents"][0][:max_contexts], results["metadatas"][0][:max_contexts])):
                if metadata and 'document_id' in metadata and 'source' in metadata:
                    doc_key = f"{metadata['document_id']}_{metadata['source']}"
                    if doc_key not in seen_documents:
                        source_documents.append({
                            'id': metadata['document_id'],
                            'filename': metadata['source'],
                            'dataset_id': dataset_id,
                            'context_index': i + 1  # Store the index used in context_text
                        })
                        seen_documents.add(doc_key)
        
        # Only add images with a decent relevance score (0.2 was the original minimum)
        # Increase threshold to ensure only more relevant images are included
        relevance_threshold = 0.25
        if image_results:
            for i, img in enumerate(top_images):
                if img.get('score', 0) >= relevance_threshold:
                    source_images.append({
                        'id': img['id'],
                        'caption': img['caption'],
                        'url': img['url'],
                        'dataset_id': img['dataset_id'],
                        'type': 'image',
                        'context_index': f"Image {i+1}"
                    })
        
        # Get previous conversation messages to add as context
        conversation_history = ""
        if len(conversation["messages"]) > 1:  # If there's more than just the current message
            # Get last 10 messages maximum
            recent_messages = conversation["messages"][-10:] if len(conversation["messages"]) > 10 else conversation["messages"]
            # Format them for context
            conversation_history = "\n".join([
                f"{msg['role'].capitalize()}: {msg['content']}" 
                for msg in recent_messages[:-1]  # Exclude current message
            ])
        
        # Prepare the system instruction and context
        system_instruction = bot.get("system_instruction", "You are a helpful assistant that answers questions based on the provided context.")
        
        # Add specific instructions for handling PowerPoint content
        system_instruction += "\n\nWhen answering questions about PowerPoint presentations, pay particular attention to slide numbers, titles, and slide content structure. Information in slide titles and speaker notes is particularly important. If information appears to be missing or incomplete in the retrieved context, explain that, but still try to provide the best possible answer."
        
        # Updated instructions for better integration of text and images
        system_instruction += "\n\nYou have access to both text documents and images in your knowledge base. Review ALL the provided context carefully, including both text documents AND image captions, before stating that information is not available. Image captions contain valuable information that should be considered as valid sources."
        
        system_instruction += "\n\nWhen referencing images, you MUST follow these exact formatting rules:"
        system_instruction += "\n1. Use the exact format '[Image X]' (where X is the image number) when citing an image"
        system_instruction += "\n2. Place the image citation AFTER describing what the image shows"
        system_instruction += "\n3. Example format: 'Here is an image of a man in a suit holding a martini [Image 1]'"
        system_instruction += "\n4. Never say you can't show or display images - they are automatically displayed when you cite them"
        system_instruction += "\n5. Don't use any other format for image citations (no parentheses, no lowercase 'image', etc.)"
        
        system_instruction += "\n\nWhen a user asks about visual content or specifically mentions images, prioritize relevant images in your response. Only reference images when they are truly relevant to the query. If there are no relevant images for the query, do not mention images at all."
        
        # Add conversation history to system instruction if available
        system_instruction_with_history = system_instruction
        if conversation_history:
            system_instruction_with_history += "\n\nThis is the conversation history so far:\n" + conversation_history
        
        # Check if using model chorus
        if use_model_chorus:
            # Check if a chorus configuration exists
            chorus_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chorus")
            os.makedirs(chorus_dir, exist_ok=True)
            
            # Get the user's chorus definitions file
            user_choruses_file = os.path.join(chorus_dir, f"{user_data['id']}_choruses.json")
            
            # Try to load the specific chorus from the user's choruses
            chorus_config = None
            
            if os.path.exists(user_choruses_file) and specific_chorus_id:
                try:
                    with open(user_choruses_file, 'r') as f:
                        choruses = json.load(f)
                        
                    # Find the requested chorus
                    chorus_config = next((c for c in choruses if c["id"] == specific_chorus_id), None)
                    
                    if debug_mode and chorus_config:
                        print(f"Using chorus configuration: {chorus_config.get('name', 'Unnamed')}", flush=True)
                except Exception as e:
                    print(f"Error loading chorus configuration: {str(e)}", flush=True)
            
            # If no chorus config found, return an error
            if not chorus_config:
                # Save the response in the conversation
                bot_response = {
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": "I couldn't find the model chorus configuration. Please check that the chorus exists and is properly configured.",
                    "timestamp": datetime.datetime.now(UTC).isoformat()
                }
                conversation["messages"].append(bot_response)
                with open(conversation_file, 'w') as f:
                    json.dump(conversation, f)
                    
                return jsonify({
                    "response": "I couldn't find the model chorus configuration. Please check that the chorus exists and is properly configured.",
                    "debug": {"error": "Chorus configuration not found"} if debug_mode else None,
                    "conversation_id": conversation_id
                }), 200
            
            # Get response and evaluator models
            response_models = chorus_config.get('response_models', [])
            evaluator_models = chorus_config.get('evaluator_models', [])
            
            # For debugging
            if debug_mode:
                print(f"Using chorus configuration: {chorus_config.get('name', 'Unnamed')}", flush=True)
                print(f"Response models: {len(response_models)}", flush=True)
                print(f"Evaluator models: {len(evaluator_models)}", flush=True)
            
            # Validate the configuration
            if not response_models or not evaluator_models:
                # Save the response in the conversation
                bot_response = {
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": "The model chorus configuration is incomplete. Please configure both response and evaluator models.",
                    "timestamp": datetime.datetime.now(UTC).isoformat()
                }
                conversation["messages"].append(bot_response)
                with open(conversation_file, 'w') as f:
                    json.dump(conversation, f)
                    
                return jsonify({
                    "response": "The model chorus configuration is incomplete. Please configure both response and evaluator models.",
                    "debug": {"error": "Incomplete chorus configuration"} if debug_mode else None,
                    "conversation_id": conversation_id
                }), 200
            
            logs = []
            logs.append(f"Using model chorus: {chorus_config.get('name', 'Unnamed')}")
            
            # Check if diverse RAG is enabled
            use_diverse_rag = chorus_config.get('use_diverse_rag', False)
            if use_diverse_rag:
                logs.append("Diverse RAG mode enabled - each model will receive different contexts")
            
            # Generate responses from all models
            all_responses = []
            anonymized_responses = []
            response_metadata = []
            
            # Process response models
            for model in response_models:
                provider = model.get('provider')
                model_name = model.get('model')
                temperature = float(model.get('temperature', 0.7))
                weight = int(model.get('weight', 1))
                
                # If diverse RAG is enabled, generate a unique context for this model
                model_specific_context = context_text
                if use_diverse_rag:
                    try:
                        # Perform a new search specifically for this model
                        model_contexts = []
                        
                        for dataset_id in dataset_ids:
                            try:
                                # Text-based document retrieval specific to this model
                                collection = chroma_client.get_collection(name=dataset_id, embedding_function=openai_ef)
                                
                                # Check if collection has any documents
                                collection_count = collection.count()
                                if collection_count > 0:
                                    # Determine how many results to request
                                    n_results = min(5, collection_count)
                                    
                                    # Add slight variation to the query for diversity
                                    model_specific_query = f"{message} relevant to {provider} {model_name}"
                                    
                                    # Get results for this specific model
                                    model_results = collection.query(
                                        query_texts=[model_specific_query],
                                        n_results=n_results
                                    )
                                    
                                    model_contexts.extend(model_results["documents"][0])
                            except Exception as e:
                                logs.append(f"Error with model-specific collection '{dataset_id}': {str(e)}")
                                continue
                        
                        # If we got contexts for this model, use them
                        if model_contexts:
                            # Limit to max_contexts
                            model_contexts = model_contexts[:max_contexts]
                            model_specific_context = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(model_contexts)])
                            logs.append(f"Generated unique context for {provider} {model_name} with {len(model_contexts)} chunks")
                    except Exception as e:
                        logs.append(f"Error generating diverse RAG for {provider} {model_name}: {str(e)}")
                        # Fall back to the common context
                        model_specific_context = context_text
                
                # Create model-specific full context with both text and image information
                model_full_context = model_specific_context
                
                # Add image context to the model prompt if available
                if image_results:
                    # Check if query is likely about images
                    image_query_terms = ["image", "picture", "photo", "visual", "diagram", "graph", "chart", "illustration", "logo", "icon"]
                    is_image_query = any(term in message.lower() for term in image_query_terms) or message.lower().strip().startswith('show me')
                    
                    # Format image information - make it more prominent for image queries
                    model_image_context = ""
                    if is_image_query:
                        model_image_context = "\n\n## RELEVANT IMAGES: ##\n"
                    else:
                        model_image_context = "\n\nRelevant Images:\n"
                        
                    # Take top images up to a limit - use more for image queries
                    max_images = 5 if is_image_query else 3
                    top_images = image_results[:max_images]
                    
                    for i, img in enumerate(top_images):
                        model_image_context += f"Available Image {i+1}:\n"
                        model_image_context += f"- Description: {img['caption']}\n"
                        model_image_context += f"- To reference this image in your response, use exactly: [Image {i+1}]\n"
                    
                    # Add image context to the full context
                    if is_image_query:
                        # For image queries, prioritize image information
                        model_full_context = "## RELEVANT INFORMATION: ##\n\n" + model_image_context + "\n\n" + model_specific_context
                    else:
                        model_full_context = "## RELEVANT INFORMATION: ##\n\n" + model_specific_context + "\n\n" + model_image_context
                
                # Add specific instructions for images to the chorus prompt
                image_prompt_instruction = ""
                if image_results:
                    image_prompt_instruction = "\n\nSeveral images were retrieved that may be relevant to this query. If the user's query relates to images or visual information, please reference the images provided in your response using [Image 1], [Image 2], etc. citations. Only reference images when they are directly relevant to answering the question. IMPORTANT: Do NOT state that you cannot display or show images to the user. Images mentioned in the context ARE available to the user for viewing."

                for i in range(weight):
                    try:
                        if provider == 'OpenAI':
                            response = openai.chat.completions.create(
                                model=model_name,
                                messages=[
                                    {"role": "system", "content": system_instruction_with_history},
                                    {"role": "user", "content": "Context:\n" + model_full_context + image_prompt_instruction + "\n\nUser question: " + message + "\n\nIf referencing information, please include citations [1], [2], etc. For images, use [Image 1], [Image 2], etc. Only reference images that are directly relevant to answering the question."}
                                ],
                                temperature=temperature
                            )
                            response_text = response.choices[0].message.content
                            anonymized_responses.append(response_text)
                            response_metadata.append({
                                "provider": provider,
                                "model": model_name,
                                "temperature": temperature,
                                "used_diverse_rag": use_diverse_rag
                            })
                            all_responses.append({
                                "provider": provider,
                                "model": model_name,
                                "response": response_text,
                                "temperature": temperature,
                                "used_diverse_rag": use_diverse_rag
                            })
                        elif provider == 'Anthropic':
                            response = anthropic_client.messages.create(
                                model=model_name,
                                system=system_instruction_with_history,
                                messages=[{"role": "user", "content": "Context:\n" + model_full_context + image_prompt_instruction + "\n\nUser question: " + message + "\n\nIf referencing information, please include citations [1], [2], etc. For images, use [Image 1], [Image 2], etc. Only reference images that are directly relevant to answering the question."}],
                                temperature=temperature,
                                max_tokens=1024
                            )
                            response_text = response.content[0].text
                            anonymized_responses.append(response_text)
                            response_metadata.append({
                                "provider": provider,
                                "model": model_name,
                                "temperature": temperature,
                                "used_diverse_rag": use_diverse_rag
                            })
                            all_responses.append({
                                "provider": provider,
                                "model": model_name,
                                "response": response_text,
                                "temperature": temperature,
                                "used_diverse_rag": use_diverse_rag
                            })
                        elif provider == 'Groq':
                            headers = {
                                "Authorization": "Bearer " + GROQ_API_KEY,
                                "Content-Type": "application/json"
                            }
                            payload = {
                                "model": model_name,
                                "messages": [
                                    {"role": "system", "content": system_instruction_with_history},
                                    {"role": "user", "content": "Context:\n" + model_full_context + image_prompt_instruction + "\n\nUser question: " + message + "\n\nIf referencing information, please include citations [1], [2], etc. For images, use [Image 1], [Image 2], etc. Only reference images that are directly relevant to answering the question."}
                                ],
                                "temperature": temperature,
                                "max_tokens": 1024
                            }
                            response = requests.post(GROQ_API_URL, json=payload, headers=headers)
                            response_json = response.json()
                            response_text = response_json["choices"][0]["message"]["content"]
                            anonymized_responses.append(response_text)
                            response_metadata.append({
                                "provider": provider,
                                "model": model_name,
                                "temperature": temperature,
                                "used_diverse_rag": use_diverse_rag
                            })
                            all_responses.append({
                                "provider": provider,
                                "model": model_name,
                                "response": response_text,
                                "temperature": temperature,
                                "used_diverse_rag": use_diverse_rag
                            })
                        elif provider == 'Mistral':
                            # Note: This would require a Mistral API implementation
                            # For now we'll use OpenAI as a fallback and log it
                            logs.append(f"Mistral API not implemented, using OpenAI fallback for {model_name}")
                            response = openai.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": system_instruction_with_history},
                                    {"role": "user", "content": "Context:\n" + model_full_context + image_prompt_instruction + "\n\nUser question: " + message + "\n\nIf referencing information, please include citations [1], [2], etc. For images, use [Image 1], [Image 2], etc. Only reference images that are directly relevant to answering the question."}
                                ],
                                temperature=temperature
                            )
                            response_text = response.choices[0].message.content
                            anonymized_responses.append(response_text)
                            response_metadata.append({
                                "provider": "OpenAI (Mistral fallback)",
                                "model": "gpt-3.5-turbo",
                                "temperature": temperature,
                                "used_diverse_rag": use_diverse_rag
                            })
                            all_responses.append({
                                "provider": "OpenAI (Mistral fallback)",
                                "model": "gpt-3.5-turbo",
                                "response": response_text,
                                "temperature": temperature,
                                "used_diverse_rag": use_diverse_rag
                            })
                    except Exception as e:
                        logs.append(f"Error with {provider} {model_name}: {str(e)}")
            
            # If no responses, use fallback
            if not all_responses:
                # Save the response in the conversation
                bot_response = {
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": "I encountered an issue with the model chorus. No models were able to generate a response.",
                    "timestamp": datetime.datetime.now(UTC).isoformat()
                }
                conversation["messages"].append(bot_response)
                with open(conversation_file, 'w') as f:
                    json.dump(conversation, f)
                    
                return jsonify({
                    "response": "I encountered an issue with the model chorus. No models were able to generate a response.",
                    "debug": {"error": "No models returned responses", "logs": logs} if debug_mode else None,
                    "conversation_id": conversation_id
                }), 200
            
            # If only one response, return it directly (but still process images)
            if len(all_responses) == 1:
                response_text = all_responses[0]["response"]
                
                # Extract which image indices were referenced in the response using robust regex
                used_image_indices = set()
                if image_results:
                    # Check which images are explicitly referenced in the response
                    print(f"CHORUS MODE (SINGLE) - IMAGE INCLUSION DEBUG: Checking for image references in response: {len(image_results)} images available", flush=True)
                    
                    # Use robust regex to find all [Image X] references
                    matches = re.findall(r'\[Image (\d+)\]', response_text)
                    for match in matches:
                        idx = int(match)
                        if 1 <= idx <= len(image_results[:5]):
                            used_image_indices.add(idx)
                            print(f"CHORUS MODE (SINGLE) - IMAGE INCLUSION DEBUG: Response explicitly references [Image {idx}]", flush=True)
                            
                    # If the response mentions images but doesn't use the exact citation format,
                    # include highly relevant images that meet our threshold
                    image_mention_terms = ["image", "picture", "photo", "logo", "diagram", "graph", "visual", "illustration", "icon"]
                    has_image_mentions = any(term.lower() in response_text.lower() for term in image_mention_terms)
                    
                    if has_image_mentions:
                        matching_terms = [term for term in image_mention_terms if term.lower() in response_text.lower()]
                        print(f"CHORUS MODE (SINGLE) - IMAGE INCLUSION DEBUG: Response contains image-related terms: {matching_terms}", flush=True)
                    
                    # Include relevant images only if the response mentions images or the query is explicitly about images
                    image_query_terms = ["image", "picture", "photo", "visual", "diagram", "graph", "chart", "illustration"]
                    is_image_query = any(term in message.lower() for term in image_query_terms) or message.lower().strip().startswith('show me')
                    
                    if (has_image_mentions or is_image_query) and not used_image_indices:
                        # Use a reasonable threshold for image relevance
                        relevance_threshold = 0.3  # Increased from 0.2 for higher quality image matches
                        print(f"CHORUS MODE (SINGLE) - IMAGE INCLUSION DEBUG: Applying relevance threshold {relevance_threshold} for image queries", flush=True)
                        for i, img in enumerate(image_results[:5]):
                            score = img.get('score', 0)
                            if score >= relevance_threshold:
                                used_image_indices.add(i+1)
                                print(f"CHORUS MODE (SINGLE) - IMAGE INCLUSION DEBUG: Including Image {i+1} with score {score:.4f} (above threshold {relevance_threshold})", flush=True)
                            else:
                                print(f"CHORUS MODE (SINGLE) - IMAGE INCLUSION DEBUG: Image {i+1} score {score:.4f} below threshold {relevance_threshold}", flush=True)
                    
                    # Only force-include an image for explicit image queries
                    if is_image_query and not used_image_indices and image_results:
                        # Include only the highest scoring image and only if it has a minimum score
                        min_score_threshold = 0.2
                        best_score = image_results[0].get('score', 0)
                        if best_score >= min_score_threshold:
                            used_image_indices.add(1)  # Add the first (highest scoring) image
                            print(f"CHORUS MODE (SINGLE) - IMAGE INCLUSION DEBUG: Force including top image with score {best_score:.4f} for image query", flush=True)
                        else:
                            print(f"CHORUS MODE (SINGLE) - IMAGE INCLUSION DEBUG: Top image score {best_score:.4f} below minimum threshold {min_score_threshold}, not including any images", flush=True)
                    
                    print(f"CHORUS MODE (SINGLE) - IMAGE INCLUSION DEBUG: Final image selection: {sorted(used_image_indices)}", flush=True)
                
                # Prepare image details for referenced images
                image_details = []
                if used_image_indices and image_results:
                    top_images = image_results[:5]  # Consider up to 5 images
                    multi_dataset = len(set(img["dataset_id"] for img in top_images)) > 1
                    for img_idx in used_image_indices:
                        if 1 <= img_idx <= len(top_images):
                            img = top_images[img_idx-1]
                            base_url = img['url']
                            download_url = f"{base_url}?download=true"
                            prefix = ""
                            if multi_dataset:
                                prefix = f"[{dataset_id_to_name.get(img['dataset_id'], img['dataset_id'][:8])}] "
                            image_details.append({
                                "index": f"{prefix}Image {img_idx}",
                                "caption": img['caption'],
                                "url": img['url'],
                                "download_url": download_url,
                                "id": img['id'],
                                "dataset_id": img['dataset_id'],
                                "document_id": img.get("document_id"),
                                "slide_number": img.get("slide_number"),
                                "slide_title": img.get("slide_title"),
                                "filename": img.get("filename")
                            })
                
                # Debug logging before response
                print(f"CHORUS MODE (SINGLE) - IMAGE DEBUG: image_results count: {len(image_results) if image_results else 0}", flush=True)
                print(f"CHORUS MODE (SINGLE) - IMAGE DEBUG: used_image_indices: {used_image_indices}", flush=True)
                print(f"CHORUS MODE (SINGLE) - IMAGE DEBUG: image_details count: {len(image_details)}", flush=True)
                if image_details:
                    for detail in image_details:
                        print(f"CHORUS MODE (SINGLE) - IMAGE DEBUG: image detail: {detail['index']} -> {detail['url']}", flush=True)
                
                # Save the response in the conversation with image details
                bot_response = {
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": response_text,
                    "timestamp": datetime.datetime.now(UTC).isoformat(),
                    "referenced_images": [img["url"] for img in image_details] if image_details else [],
                    "image_details": image_details
                }
                conversation["messages"].append(bot_response)
                with open(conversation_file, 'w') as f:
                    json.dump(conversation, f)
                    
                return jsonify({
                    "response": response_text,
                    "image_details": image_details if image_details else [],  # Always include image_details, even if empty
                    "debug": {
                        "all_responses": all_responses,
                        "anonymized_responses": anonymized_responses,
                        "response_metadata": response_metadata,
                        "logs": logs,
                        "contexts": contexts,
                        "image_results": image_results if image_results else []
                    } if debug_mode else None,
                    "conversation_id": conversation_id
                }), 200
            
            # Get voting from evaluator models
            votes = [0] * len(all_responses)
            
            for model in evaluator_models:
                provider = model.get('provider')
                model_name = model.get('model')
                temperature = float(model.get('temperature', 0.2))
                weight = int(model.get('weight', 1))
                
                voting_prompt = "Here are the " + str(len(anonymized_responses)) + " candidate responses:\n\n" + \
chr(10).join([f"Response {j+1}:\n{resp}" for j, resp in enumerate(anonymized_responses)]) + \
"\n\nWhich response provides the most accurate, helpful, and relevant answer? Return ONLY the number (1-" + \
str(len(anonymized_responses)) + ") of the best response.\n" + \
"Do not reveal any bias or preference based on writing style or approach - evaluate solely on answer quality, accuracy and helpfulness."
                # Apply weight by counting vote multiple times
                for i in range(weight):
                    try:
                        vote_text = ""
                        if provider == 'OpenAI':
                            voting_response = openai.chat.completions.create(
                                model=model_name,
                                messages=[{"role": "user", "content": voting_prompt}],
                                temperature=temperature
                            )
                            vote_text = voting_response.choices[0].message.content
                        elif provider == 'Anthropic':
                            voting_response = anthropic_client.messages.create(
                                model=model_name,
                                messages=[{"role": "user", "content": voting_prompt}],
                                temperature=temperature,
                                max_tokens=10
                            )
                            vote_text = voting_response.content[0].text
                        elif provider == 'Groq':
                            headers = {
                                "Authorization": "Bearer " + GROQ_API_KEY,
                                "Content-Type": "application/json"
                            }
                            payload = {
                                "model": model_name,
                                "messages": [{"role": "user", "content": voting_prompt}],
                                "temperature": temperature,
                                "max_tokens": 10
                            }
                            voting_response = requests.post(GROQ_API_URL, json=payload, headers=headers)
                            voting_json = voting_response.json()
                            vote_text = voting_json["choices"][0]["message"]["content"]
                        elif provider == 'Mistral':
                            # Fallback to OpenAI for Mistral
                            logs.append(f"Mistral API not implemented for evaluation, using OpenAI fallback")
                            voting_response = openai.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[{"role": "user", "content": voting_prompt}],
                                temperature=temperature
                            )
                            vote_text = voting_response.choices[0].message.content
                        
                        # Extract vote number
                        match = re.search(r'\b([1-9][0-9]*)\b', vote_text)
                        if match:
                            vote_number = int(match.group(1))
                            if 1 <= vote_number <= len(anonymized_responses):
                                votes[vote_number-1] += 1
                                logs.append(f"Evaluator {provider} {model_name} voted for response {vote_number}")
                            else:
                                logs.append(f"Invalid vote number: {vote_number}")
                        else:
                            logs.append(f"Could not parse vote from: {vote_text}")
                    except Exception as e:
                        logs.append(f"Error with evaluator {provider} {model_name}: {str(e)}")
            
            # Find winning response
            max_votes = max(votes) if votes else 0
            winning_indices = [i for i, v in enumerate(votes) if v == max_votes]
            winning_index = winning_indices[0] if winning_indices else 0
            
            winning_response = all_responses[winning_index]["response"]
            
            # Extract which image indices were referenced in the winning response using robust regex
            used_image_indices = set()
            if image_results:
                # Check which images are explicitly referenced in the response
                print(f"CHORUS MODE - IMAGE INCLUSION DEBUG: Checking for image references in response: {len(image_results)} images available", flush=True)
                
                # Use robust regex to find all [Image X] references
                matches = re.findall(r'\[Image (\d+)\]', winning_response)
                for match in matches:
                    idx = int(match)
                    if 1 <= idx <= len(image_results[:5]):
                        used_image_indices.add(idx)
                        print(f"CHORUS MODE - IMAGE INCLUSION DEBUG: Response explicitly references [Image {idx}]", flush=True)
                        
                # If the response mentions images but doesn't use the exact citation format,
                # include highly relevant images that meet our threshold
                image_mention_terms = ["image", "picture", "photo", "logo", "diagram", "graph", "visual", "illustration", "icon"]
                has_image_mentions = any(term.lower() in winning_response.lower() for term in image_mention_terms)
                
                if has_image_mentions:
                    matching_terms = [term for term in image_mention_terms if term.lower() in winning_response.lower()]
                    print(f"IMAGE INCLUSION DEBUG: Response contains image-related terms: {matching_terms}", flush=True)
                
                # Include relevant images only if the response mentions images or the query is explicitly about images
                image_query_terms = ["image", "picture", "photo", "visual", "diagram", "graph", "chart", "illustration"]
                is_image_query = any(term in message.lower() for term in image_query_terms) or message.lower().strip().startswith('show me')
                
                if (has_image_mentions or is_image_query) and not used_image_indices:
                    # Use a reasonable threshold for image relevance
                    relevance_threshold = 0.3  # Increased from 0.2 for higher quality image matches
                    print(f"IMAGE INCLUSION DEBUG: Applying relevance threshold {relevance_threshold} for image queries", flush=True)
                    for i, img in enumerate(image_results[:5]):
                        score = img.get('score', 0)
                        if score >= relevance_threshold:
                            used_image_indices.add(i+1)
                            print(f"IMAGE INCLUSION DEBUG: Including Image {i+1} with score {score:.4f} (above threshold {relevance_threshold})", flush=True)
                        else:
                            print(f"IMAGE INCLUSION DEBUG: Image {i+1} score {score:.4f} below threshold {relevance_threshold}", flush=True)
                
                # Only force-include an image for explicit image queries
                if is_image_query and not used_image_indices and image_results:
                    # Include only the highest scoring image and only if it has a minimum score
                    min_score_threshold = 0.2
                    best_score = image_results[0].get('score', 0)
                    if best_score >= min_score_threshold:
                        used_image_indices.add(1)  # Add the first (highest scoring) image
                        print(f"IMAGE INCLUSION DEBUG: Force including top image with score {best_score:.4f} for image query", flush=True)
                    else:
                        print(f"IMAGE INCLUSION DEBUG: Top image score {best_score:.4f} below minimum threshold {min_score_threshold}, not including any images", flush=True)
                
                print(f"IMAGE INCLUSION DEBUG: Final image selection: {sorted(used_image_indices)}", flush=True)
            
            # Prepare image details for referenced images
            image_details = []
            if used_image_indices and image_results:
                top_images = image_results[:5]  # Consider up to 5 images
                multi_dataset = len(set(img["dataset_id"] for img in top_images)) > 1
                for img_idx in used_image_indices:
                    if 1 <= img_idx <= len(top_images):
                        img = top_images[img_idx-1]
                        base_url = img['url']
                        download_url = f"{base_url}?download=true"
                        prefix = ""
                        if multi_dataset:
                            prefix = f"[{dataset_id_to_name.get(img['dataset_id'], img['dataset_id'][:8])}] "
                        image_details.append({
                            "index": f"{prefix}Image {img_idx}",
                            "caption": img['caption'],
                            "url": img['url'],
                            "download_url": download_url,
                            "id": img['id'],
                            "dataset_id": img['dataset_id'],
                            "document_id": img.get("document_id"),
                            "slide_number": img.get("slide_number"),
                            "slide_title": img.get("slide_title"),
                            "filename": img.get("filename")
                        })
            
            # Debug logging before response
            print(f"CHORUS MODE - IMAGE DEBUG: image_results count: {len(image_results) if image_results else 0}", flush=True)
            print(f"CHORUS MODE - IMAGE DEBUG: used_image_indices: {used_image_indices}", flush=True)
            print(f"CHORUS MODE - IMAGE DEBUG: image_details count: {len(image_details)}", flush=True)
            if image_details:
                for detail in image_details:
                    print(f"CHORUS MODE - IMAGE DEBUG: image detail: {detail['index']} -> {detail['url']}", flush=True)
            
            # Save the response in the conversation with image details
            bot_response = {
                "id": str(uuid.uuid4()),
                "role": "assistant",
                "content": winning_response,
                "timestamp": datetime.datetime.now(UTC).isoformat(),
                "referenced_images": [img["url"] for img in image_details] if image_details else [],
                "image_details": image_details
            }
            conversation["messages"].append(bot_response)
            with open(conversation_file, 'w') as f:
                json.dump(conversation, f)
                
            return jsonify({
                "response": winning_response,
                "image_details": image_details if image_details else [],  # Always include image_details, even if empty
                "debug": {
                    "all_responses": all_responses,
                    "anonymized_responses": anonymized_responses,
                    "response_metadata": response_metadata,
                    "votes": votes,
                    "logs": logs,
                    "contexts": contexts,
                    "image_results": image_results if image_results else []
                } if debug_mode else None,
                "conversation_id": conversation_id
            }), 200
                
    
        # Standard mode - just get a response from OpenAI
        # Enhance system instruction to better handle image queries in mixed datasets
        image_instruction = ""
        if image_results:
            image_query_terms = ["image", "picture", "photo", "visual", "diagram", "graph", "chart", "illustration"]
            is_image_query = any(term in message.lower() for term in image_query_terms) or message.lower().strip().startswith('show me')
            
            if is_image_query:
                image_instruction = "\n\nThe user seems to be asking about images. Please prioritize showing and describing relevant images from the provided context when appropriate. When referencing images, use the exact format '[Image X]' (where X is the image number) and be descriptive about what they show."
            else:
                image_instruction = "\n\nRelevant images are available in the context. When they may help answer the user's question, refer to them using the format '[Image X]' (where X is the image number) and briefly describe what they show."
        
        response = openai.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_instruction_with_history + "\n\nWhen you reference information from the provided context, please cite the source using the number in square brackets, e.g. [1], [2], etc. For images, use [Image 1], [Image 2], etc.\n\nIMPORTANT: Before stating that information isn't available, check ALL context including image captions. If information is only found in an image caption, still use that information and cite the image.\n\nIMPORTANT: Do NOT state that you cannot display or show images to the user. When referencing images, simply describe what they contain and cite them with [Image X]. Images mentioned in the context ARE available to the user for viewing."},
                {"role": "user", "content": "Context:\n" + full_context + "\n\nUser question: " + message + "\n\nIf referencing information, please include citations [1], [2], etc. Only reference images that are directly relevant to answering the question."}
            ]
        )
        
        response_text = response.choices[0].message.content
        
        # Extract which context indices were actually used in the response
        used_indices = set()
        for i in range(1, max_contexts + 1):
            if f"[{i}]" in response_text:
                used_indices.add(i)
        
        # Extract which image indices were referenced using robust regex detection
        used_image_indices = set()
        if image_results:
            # Always define top_images here to avoid undefined variable errors
            image_results.sort(key=lambda x: x["score"], reverse=True)
            image_query_terms = ["image", "picture", "photo", "visual", "diagram", "graph", "chart", "illustration"]
            is_image_query = any(term in message.lower() for term in image_query_terms) or message.lower().strip().startswith('show me')
            max_images = 5 if is_image_query else 3
            top_images = image_results[:max_images]
            print(f"STANDARD MODE - IMAGE INCLUSION DEBUG: Checking for image references in response: {len(image_results)} images available", flush=True)
            
            # Use robust regex to find all [Image X] references
            matches = re.findall(r'\[Image (\d+)\]', response_text)
            for match in matches:
                idx = int(match)
                if 1 <= idx <= len(top_images):
                    used_image_indices.add(idx)
                    print(f"STANDARD MODE - IMAGE INCLUSION DEBUG: Response explicitly references [Image {idx}]", flush=True)
                
            # Also check if the image caption is mentioned in the response
            for i in range(1, len(top_images) + 1):
                if top_images[i-1].get('caption'):
                    if top_images[i-1]['caption'].lower() in response_text.lower():
                        used_image_indices.add(i)
                        print(f"STANDARD MODE - IMAGE INCLUSION DEBUG: Response mentions image caption for Image {i}", flush=True)
            image_mention_terms = ["image", "picture", "photo", "logo", "diagram", "graph", "visual", "illustration", "icon"]
            has_image_mentions = any(term.lower() in response_text.lower() for term in image_mention_terms)
            if has_image_mentions:
                matching_terms = [term for term in image_mention_terms if term.lower() in response_text.lower()]
                print(f"STANDARD MODE - IMAGE INCLUSION DEBUG: Response contains image-related terms: {matching_terms}", flush=True)
            if (has_image_mentions or is_image_query) and not used_image_indices:
                relevance_threshold = 0.3
                print(f"STANDARD MODE - IMAGE INCLUSION DEBUG: Applying relevance threshold {relevance_threshold} for image queries", flush=True)
                for i, img in enumerate(top_images):
                    score = img.get('score', 0)
                    if score >= relevance_threshold:
                        used_image_indices.add(i+1)
                        print(f"STANDARD MODE - IMAGE INCLUSION DEBUG: Including Image {i+1} with score {score:.4f} (above threshold {relevance_threshold})", flush=True)
                if not used_image_indices and top_images:
                    used_image_indices.add(1)
                    print(f"STANDARD MODE - IMAGE INCLUSION DEBUG: Force including top image for image query")
            print(f"STANDARD MODE - IMAGE INCLUSION DEBUG: Final image selection: {sorted(used_image_indices)}")
        
        # Filter source_documents to only include those that were actually cited
        used_source_documents = [doc for doc in source_documents if doc['context_index'] in used_indices]
        
        # Filter source_images to only include those that were actually cited
        used_source_images = []
        if image_results:
            for img in source_images:
                img_idx = int(img['context_index'].split()[1])
                if img_idx in used_image_indices:
                    used_source_images.append(img)

        # Create detailed context information for frontend display
        context_details = []
        for i, ctx in enumerate(contexts):
            if i+1 in used_indices:  # Only include contexts that were cited
                # Find the corresponding source document
                source_doc = next((doc for doc in source_documents if doc['context_index'] == i+1), None)
                if source_doc:
                    context_details.append({
                        "index": i+1,
                        "snippet": ctx[:200] + "..." if len(ctx) > 200 else ctx,  # Truncate for display
                        "full_text": ctx,  # Include full snippet text
                        "document_id": source_doc.get('id', ''),
                        "filename": source_doc.get('filename', ''),
                        "dataset_id": source_doc.get('dataset_id', '')
                    })

        # Build image_details directly from top_images and used_image_indices
        image_details = []
        if image_results:
            top_images = image_results[:5]
            multi_dataset = len(set(img["dataset_id"] for img in top_images)) > 1
            for img_idx in used_image_indices:
                if 1 <= img_idx <= len(top_images):
                    img = top_images[img_idx-1]
                    base_url = img['url']
                    download_url = f"{base_url}?download=true"
                    prefix = ""
                    if multi_dataset:
                        prefix = f"[{dataset_id_to_name.get(img['dataset_id'], img['dataset_id'][:8])}] "
                    image_details.append({
                        "index": f"{prefix}Image {img_idx}",
                        "caption": img['caption'],
                        "url": img['url'],
                        "download_url": download_url,
                        "id": img['id'],
                        "dataset_id": img['dataset_id'],
                        "document_id": img.get("document_id"),
                        "slide_number": img.get("slide_number"),
                        "slide_title": img.get("slide_title"),
                        "filename": img.get("filename")
                    })
        
        # Debug logging before response
        print(f"STANDARD MODE - IMAGE DEBUG: image_results count: {len(image_results) if image_results else 0}", flush=True)
        print(f"STANDARD MODE - IMAGE DEBUG: used_image_indices: {used_image_indices}", flush=True)
        print(f"STANDARD MODE - IMAGE DEBUG: image_details count: {len(image_details)}", flush=True)
        if image_details:
            for detail in image_details:
                print(f"STANDARD MODE - IMAGE DEBUG: image detail: {detail['index']} -> {detail['url']}", flush=True)

        # Save the response in the conversation
        bot_response = {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.datetime.now(UTC).isoformat(),
            "referenced_images": [img["url"] for img in image_details] if image_details else [],
            "context_details": context_details,  # Add context details to the saved response
            "image_details": image_details  # Add image details to the saved response
        }
        conversation["messages"].append(bot_response)
        with open(conversation_file, 'w') as f:
            json.dump(conversation, f)
            
        # Ensure we have image_details even if no images were found
        if not image_details:
            image_details = []
            
        # Log the final response structure
        print(f"STANDARD MODE - RESPONSE DEBUG: Sending response with {len(image_details)} images", flush=True)
        print(f"STANDARD MODE - RESPONSE DEBUG: Response text contains {response_text.count('[Image')} image references", flush=True)
        
        response_data = {
            "response": response_text,
            "source_documents": used_source_documents,
            "context_details": context_details,  # Include context details in response
            "image_details": image_details,  # Always include image_details array
            "referenced_images": [img["url"] for img in image_details],  # Always include referenced_images array
            "debug": {
                "contexts": contexts,
                "image_results": image_results if image_results else [],
                "used_image_indices": list(used_image_indices) if used_image_indices else []
            } if debug_mode else None,
            "conversation_id": conversation_id
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        print(f"Error in chat_with_bot: {str(e)}")
        # Save the error response in the conversation
        bot_response = {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": "I apologize, but I encountered an error while processing your request. Please try again or contact support if the issue persists.",
            "timestamp": datetime.datetime.now(UTC).isoformat()
        }
        conversation["messages"].append(bot_response)
        with open(conversation_file, 'w') as f:
            json.dump(conversation, f)
            
        return jsonify({
            "response": "I apologize, but I encountered an error while processing your request. Please try again or contact support if the issue persists.",
            "debug": {"error": str(e)} if debug_mode else None,
            "conversation_id": conversation_id
        }), 200

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



# New route for editing images
@app.route('/api/images/edit', methods=['POST'])
@require_auth_wrapper
def edit_image(user_data):
    try:
        # Check if the request is multipart/form-data
        if not request.content_type or not request.content_type.startswith('multipart/form-data'):
            return jsonify({"error": "Request must be multipart/form-data"}), 400
            
        # Get the image file
        if 'image' not in request.files:
            return jsonify({"error": "Image file is required"}), 400
            
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"error": "No image selected"}), 400
        
        # Get the prompt
        prompt = request.form.get('prompt', '')
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
            
        # Optional parameters
        model = request.form.get('model', 'gpt-image-1')
        size = request.form.get('size', '1024x1024')
        quality = request.form.get('quality', 'medium')
        output_format = request.form.get('output_format', 'png')
        
        # Save the uploaded image
        filename = secure_filename(f"edit_source_{str(uuid.uuid4())}{os.path.splitext(image_file.filename)[1]}")
        image_path = os.path.join(IMAGE_FOLDER, filename)
        image_file.save(image_path)
        
        # Resize if needed
        image_path = resize_image(image_path)
        
        # Read image data for the API call
        with open(image_path, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Call OpenAI edit API
        response = openai.images.edit(
            image=open(image_path, 'rb'),
            prompt=prompt,
            model=model,
            size=size,
            quality=quality,
            n=1
        )
        
        # Process the result
        if not response.data or len(response.data) == 0:
            return jsonify({"error": "No image was generated"}), 500
            
        image_obj = response.data[0]
        
        # Get the image URL or base64 data
        if hasattr(image_obj, 'b64_json') and image_obj.b64_json:
            # Decode base64 content
            image_content = base64.b64decode(image_obj.b64_json)
            image_url = None
        elif hasattr(image_obj, 'url') and image_obj.url:
            # Download the image from URL
            image_response = requests.get(image_obj.url)
            if image_response.status_code != 200:
                return jsonify({"error": "Failed to download edited image"}), 500
                
            image_content = image_response.content
        else:
            return jsonify({"error": "No image data found in response"}), 500
            
        # Save the edited image
        output_filename = f"edited_{str(uuid.uuid4())}.{output_format}"
        output_path = os.path.join(IMAGE_FOLDER, output_filename)
        
        with open(output_path, 'wb') as f:
            f.write(image_content)
            
        # Create the URL for accessing the image
        api_image_url = f"/api/images/{output_filename}"
        
        return jsonify({
            "success": True,
            "image_url": api_image_url,
            "images": [{
                "image_url": api_image_url
            }],
            "params": {
                "model": model,
                "size": size,
                "quality": quality,
                "format": output_format
            }
        })
        
    except Exception as e:
        print(f"Error editing image: {str(e)}")
        return jsonify({"error": f"Image editing failed: {str(e)}"}), 500

# Add this new endpoint after the existing chat_with_bot function
@app.route('/api/bots/<bot_id>/chat-with-image', methods=['POST'])
@require_auth_wrapper
def chat_with_image(user_data, bot_id):
    # Check if the request is multipart/form-data or JSON
    if request.content_type and request.content_type.startswith('multipart/form-data'):
        # Handle multipart/form-data request
        message = request.form.get('message', '')
        debug_mode = request.form.get('debug_mode', 'false').lower() == 'true'
        conversation_id = request.form.get('conversation_id', '')
        
        # Get the image file
        if 'image' not in request.files:
            return jsonify({"error": "Image file is required"}), 400
            
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"error": "No image selected"}), 400
        
        # Create a new conversation if no conversation_id provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        # Process the image file
        try:
            # Save the image to a temporary file
            filename = f"chat_image_{str(uuid.uuid4())}{os.path.splitext(image_file.filename)[1]}"
            image_path = os.path.join(IMAGE_FOLDER, filename)
            image_file.save(image_path)
            
            # Get the file extension
            file_ext = os.path.splitext(image_file.filename)[1][1:]  # Remove the dot
            if not file_ext:
                file_ext = 'png'  # Default to PNG if no extension
        
            # Continue with image processing
        except Exception as img_error:
            print(f"Error processing image: {str(img_error)}")
            return jsonify({"error": f"Failed to process image: {str(img_error)}"}), 400
    else:
        # Handle JSON request
        data = request.json
        if not data:
            return jsonify({"error": "Invalid request format"}), 400
            
        message = data.get('message', '')
        image_data = data.get('image_data', '')
        debug_mode = data.get('debug_mode', False)
        conversation_id = data.get('conversation_id', '')
        
        if not image_data:
            return jsonify({"error": "Image data is required"}), 400
        
        # Create a new conversation if no conversation_id provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        # Process the base64 image
        try:
            # Extract the base64 content and file extension
            if ';base64,' in image_data:
                header, encoded = image_data.split(';base64,')
                file_ext = header.split('/')[-1]
            else:
                encoded = image_data
                file_ext = 'png'  # Default to PNG if not specified
            
            # Decode the base64 data
            decoded_image = base64.b64decode(encoded)
            
            # Save to a temporary file
            filename = f"chat_image_{str(uuid.uuid4())}.{file_ext}"
            image_path = os.path.join(IMAGE_FOLDER, filename)
            
            with open(image_path, 'wb') as f:
                f.write(decoded_image)
        except Exception as img_error:
            print(f"Error processing image: {str(img_error)}")
            return jsonify({"error": f"Failed to process image: {str(img_error)}"}), 400
    
    # Resize image if needed
    image_path = resize_image(image_path)
    
    # Add message text if not provided
    if not message:
        message = "[Image uploaded]"
    
    # Get bot info
    bots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bots")
    user_bots_file = os.path.join(bots_dir, f"{user_data['id']}_bots.json")
    
    if not os.path.exists(user_bots_file):
        return jsonify({"error": "Bot not found"}), 404
        
    with open(user_bots_file, 'r') as f:
        bots = json.load(f)
        
    bot = None
    for b in bots:
        if b["id"] == bot_id:
            bot = b
            break
            
    if not bot:
        return jsonify({"error": "Bot not found"}), 404
    
    # Create message object
    user_message = {
        "id": str(uuid.uuid4()),
        "role": "user",
        "content": message,
        "timestamp": datetime.datetime.now(UTC).isoformat(),
        "has_image": True,
        "image_path": image_path
    }
    
    # Add message to conversation history
    conversation_file = os.path.join(CONVERSATIONS_FOLDER, f"{user_data['id']}_{bot_id}_{conversation_id}.json")
    
    if os.path.exists(conversation_file):
        # Load existing conversation
        with open(conversation_file, 'r') as f:
            conversation = json.load(f)
    else:
        # Create new conversation
        conversation = {
            "id": conversation_id,
            "bot_id": bot_id,
            "user_id": user_data['id'],
            "title": message[:40] + "..." if len(message) > 40 else message,  # Use first message as title
            "created_at": datetime.datetime.now(UTC).isoformat(),
            "updated_at": datetime.datetime.now(UTC).isoformat(),
            "messages": []
        }
    
    # Add user message to conversation
    conversation["messages"].append(user_message)
    conversation["updated_at"] = datetime.datetime.now(UTC).isoformat()
    
    # Save updated conversation
    with open(conversation_file, 'w') as f:
        json.dump(conversation, f)
    
    # Process with Vision API
    try:
        # Read and encode the image file
        with open(image_path, 'rb') as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Get previous conversation messages to add as context
        conversation_history = ""
        if len(conversation["messages"]) > 1:  # If there's more than just the current message
            # Get last 10 messages maximum
            recent_messages = conversation["messages"][-10:] if len(conversation["messages"]) > 10 else conversation["messages"]
            # Format them for context, excluding the most recent (current) message
            conversation_history = "\n".join([
                f"{msg['role'].capitalize()}: {msg['content']}" 
                for msg in recent_messages[:-1]  # Exclude current message
            ])
        
        # Prepare system instruction with history
        system_instruction = bot.get("system_instruction", "You are a helpful assistant that can analyze images.")
        if conversation_history:
            system_instruction += "\n\nThis is the conversation history so far:\n" + conversation_history
            
        # Prepare the message with the image for OpenAI Vision API
        messages = [
            {
                "role": "system", 
                "content": system_instruction
            },
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": message if message != "[Image uploaded]" else "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{file_ext};base64,{encoded_image}"
                        }
                    }
                ]
            }
        ]
        
        # Call OpenAI API with vision capabilities
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1024
        )
        
        response_text = response.choices[0].message.content
        
        # Save the response in the conversation
        bot_response = {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.datetime.now(UTC).isoformat(),
            "from_image_analysis": True
        }
        conversation["messages"].append(bot_response)
        with open(conversation_file, 'w') as f:
            json.dump(conversation, f)
            
        # Create image details for the uploaded image
        image_details = [{
            "index": "Image 1",
            "caption": "Uploaded image",
            "url": f"/api/images/{os.path.basename(image_path)}",
            "download_url": f"/api/images/{os.path.basename(image_path)}?download=true",
            "id": str(uuid.uuid4()),
            "dataset_id": ""
        }]
            
        return jsonify({
            "response": response_text,
            "conversation_id": conversation_id,
            "image_processed": True,
            "image_details": image_details,  # Add image details to response for all requests
            "debug": {
                "image_path": image_path,
                "message": message,
                "conversation_history_length": len(conversation_history)
            } if debug_mode else None
        }), 200
        
    except Exception as e:
        print(f"Error in chat_with_image: {str(e)}")
        return jsonify({
            "error": "I apologize, but I encountered an error while processing your image. Please try again or contact support if the issue persists.",
            "details": str(e) if debug_mode else None
        }), 500
    
   
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
    # Check if dataset exists and belongs to user
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    user_datasets_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")
    
    if not os.path.exists(user_datasets_file):
        return jsonify({"error": "Dataset not found"}), 404
        
    with open(user_datasets_file, 'r') as f:
        datasets = json.load(f)
        
    dataset_exists = False
    for dataset in datasets:
        if dataset["id"] == dataset_id:
            dataset_exists = True
            break
            
    if not dataset_exists:
        return jsonify({"error": "Dataset not found"}), 404

    # Look for the document in the uploads folder
    document_path = None
    
    # First, try direct match with document_id
    direct_match = os.path.join(DOCUMENT_FOLDER, f"{document_id}_{filename}")
    if os.path.exists(direct_match):
        document_path = direct_match
    else:
        # If direct match fails, search for any file ending with the filename
        for file in os.listdir(DOCUMENT_FOLDER):
            if file.endswith(f"_{filename}"):
                document_path = os.path.join(DOCUMENT_FOLDER, file)
                break

    if not document_path or not os.path.exists(document_path):
        return jsonify({"error": "Document not found"}), 404

    return send_file(document_path, as_attachment=True, download_name=filename)

@app.route('/api/datasets/<dataset_id>/images', methods=['GET'])
@require_auth_wrapper
def get_dataset_images(user_data, dataset_id):
    """Get all images for a specific dataset"""
    # Debug information
    print(f"Getting images for dataset {dataset_id}")
    print(f"Available image metadata keys: {list(image_processor.image_metadata.keys())}")
    
    # Check if dataset exists and belongs to user
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    user_datasets_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")
    
    if not os.path.exists(user_datasets_file):
        return jsonify({"error": "Dataset not found"}), 404
        
    with open(user_datasets_file, 'r') as f:
        datasets = json.load(f)
        
    dataset_exists = False
    dataset_index = -1
    for idx, dataset in enumerate(datasets):
        if dataset["id"] == dataset_id:
            dataset_exists = True
            dataset_index = idx
            break
            
    if not dataset_exists:
        print(f"Dataset {dataset_id} not found in user datasets")
        return jsonify({"error": "Dataset not found"}), 404
    
    # Print the dataset info
    print(f"Dataset info: {datasets[dataset_index]}")
    
    # Force reload the image metadata if it's not loaded yet or might be stale
    indices_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "image_indices")
    metadata_file = os.path.join(indices_dir, f"{dataset_id}_metadata.json")
    index_path = os.path.join(indices_dir, f"{dataset_id}_index.faiss")
    
    # If metadata exists but not loaded in processor, load it now
    if os.path.exists(metadata_file):
        try:
            print(f"Loading metadata file from {metadata_file}")
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Filter to ensure we only include this dataset's images AND the image files actually exist
            valid_metadata = []
            for img_meta in metadata:
                # Add dataset_id if missing
                if not img_meta.get('dataset_id'):
                    img_meta['dataset_id'] = dataset_id
                
                # Only include if it's for this dataset and the file exists
                if img_meta.get('dataset_id') == dataset_id and 'path' in img_meta:
                    if os.path.exists(img_meta['path']):
                        valid_metadata.append(img_meta)
                    else:
                        print(f"Skipping image {img_meta.get('id')} because file doesn't exist: {img_meta.get('path')}")
            
            # Update processor's metadata with validated images
            image_processor.image_metadata[dataset_id] = valid_metadata
            
            # Also load index if it exists
            if os.path.exists(index_path):
                # We need to rebuild the index since some images may be missing
                # This ensures the index and metadata stay in sync
                if len(valid_metadata) > 0:
                    # Create a new empty index
                    index = faiss.IndexFlatIP(VECTOR_DIMENSION)
                    
                    # Add all valid images to the index
                    for img_meta in valid_metadata:
                        if 'embedding' in img_meta:
                            # If we have a pre-computed embedding
                            embedding = np.array([img_meta['embedding']], dtype=np.float32)
                            index.add(embedding)
                        else:
                            # Try to compute embedding from image
                            try:
                                embedding = image_processor.compute_image_embedding(img_meta['path'])
                                index.add(np.array([embedding], dtype=np.float32))
                            except Exception as e:
                                print(f"Error computing embedding: {str(e)}")
                    
                    # Save the new index
                    image_processor.image_indices[dataset_id] = index
                    image_processor._save_dataset_index(dataset_id)
                    print(f"Rebuilt index for dataset {dataset_id} with {len(valid_metadata)} images")
                else:
                    # Create an empty index if no valid images
                    image_processor.image_indices[dataset_id] = faiss.IndexFlatIP(VECTOR_DIMENSION)
                    image_processor._save_dataset_index(dataset_id)
                    print(f"Created empty index for dataset {dataset_id} - no valid images found")
                
            print(f"Loaded and validated {len(valid_metadata)} images for dataset {dataset_id}")
        except Exception as e:
            print(f"Error loading/validating metadata: {str(e)}")
    
    # Get images metadata from the image processor with validation
    images = []
    
    if dataset_id in image_processor.image_metadata:
        img_count = len(image_processor.image_metadata[dataset_id])
        print(f"Found {img_count} images in dataset {dataset_id}")
        
        for img_meta in image_processor.image_metadata[dataset_id]:
            # Verify this image belongs to this dataset
            if img_meta.get('dataset_id') != dataset_id:
                print(f"Skipping image {img_meta.get('id')} as it belongs to dataset {img_meta.get('dataset_id')}, not {dataset_id}")
                continue
            
            # Verify the image file exists
            if 'path' in img_meta and not os.path.exists(img_meta['path']):
                print(f"Skipping image {img_meta.get('id')} as file doesn't exist: {img_meta['path']}")
                continue
                
            # Create web-accessible URLs for each image
            img_meta_copy = img_meta.copy()
            if 'path' in img_meta_copy:
                img_meta_copy['url'] = f"/api/images/{os.path.basename(img_meta_copy['path'])}"
                # Don't expose the full path in API
                if 'path' in img_meta_copy:
                    del img_meta_copy['path']
            images.append(img_meta_copy)
        
        # Sort images by creation date if available (newest first)
        images.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        # Update the dataset's image count to match reality
        actual_count = len(images)
        if datasets[dataset_index].get("image_count", 0) != actual_count:
            print(f"Updating image count for dataset {dataset_id} from {datasets[dataset_index].get('image_count', 0)} to {actual_count}")
            datasets[dataset_index]["image_count"] = actual_count
            with open(user_datasets_file, 'w') as f:
                json.dump(datasets, f)
        
        return jsonify({
            "images": images,
            "total_images": len(images)
        }), 200
    else:
        print(f"No images found for dataset {dataset_id} in image_processor.image_metadata")
        
        # Try to recreate metadata by scanning uploaded images
        try:
            image_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads", "images")
            dataset_images = []
            
            if os.path.exists(image_folder):
                print(f"Scanning image folder: {image_folder}")
                # Create empty index for this dataset
                image_processor.image_indices[dataset_id] = faiss.IndexFlatIP(VECTOR_DIMENSION)
                image_processor.image_metadata[dataset_id] = []
                
                # Scan all image files in folder and try to load them
                for filename in os.listdir(image_folder):
                    if any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']):
                        file_path = os.path.join(image_folder, filename)
                        
                        # Try to associate the file with this dataset
                        try:
                            # Load and process image
                            image_id = str(uuid.uuid4())
                            # Create metadata
                            image_meta = {
                                "id": image_id,
                                "dataset_id": dataset_id,
                                "path": file_path,
                                "original_filename": filename,
                                "caption": f"Image: {filename}",
                                "created_at": datetime.datetime.now(UTC).isoformat(),
                                "url": f"/api/images/{filename}"
                            }
                            
                            # Try to generate caption using BLIP
                            try:
                                image_processor._load_models()
                                caption = image_processor.generate_caption(file_path)
                                image_meta["caption"] = caption
                            except Exception as e:
                                print(f"Error generating caption: {str(e)}")
                                
                            # Try to compute embedding and add to index
                            try:
                                embedding = image_processor.compute_image_embedding(file_path)
                                image_processor.image_indices[dataset_id].add(np.array([embedding], dtype=np.float32))
                            except Exception as e:
                                print(f"Error computing embedding: {str(e)}")
                            
                            # Add to metadata
                            image_processor.image_metadata[dataset_id].append(image_meta)
                            dataset_images.append(image_meta)
                            
                        except Exception as e:
                            print(f"Error processing image {filename}: {str(e)}")
                
                # If we found images, save the metadata
                if dataset_images:
                    # Ensure directory exists
                    os.makedirs(indices_dir, exist_ok=True)
                    
                    # Save index and metadata
                    try:
                        faiss.write_index(image_processor.image_indices[dataset_id], index_path)
                        with open(metadata_file, 'w') as f:
                            json.dump(image_processor.image_metadata[dataset_id], f)
                        print(f"Saved index and metadata for {len(dataset_images)} images")
                    except Exception as e:
                        print(f"Error saving index and metadata: {str(e)}")
            
            if dataset_images:
                print(f"Found {len(dataset_images)} images in uploads folder")
                
                # Update image count in dataset
                datasets[dataset_index]["image_count"] = len(dataset_images)
                with open(user_datasets_file, 'w') as f:
                    json.dump(datasets, f)
                
                # Format images for response
                images = []
                for img_meta in dataset_images:
                    img_meta_copy = img_meta.copy()
                    if 'path' in img_meta_copy:
                        del img_meta_copy['path']  # Don't expose full path
                    images.append(img_meta_copy)
                
                return jsonify({
                    "images": images,
                    "total_images": len(images),
                    "note": "Images were reconstructed from uploads folder"
                }), 200
                
        except Exception as e:
            print(f"Error reconstructing images from uploads: {str(e)}")
        
        # If we reach here, no images were found or loaded
        if datasets[dataset_index].get("image_count", 0) > 0:
            print(f"Dataset claims to have {datasets[dataset_index].get('image_count')} images but none were found")
            # Reset the count to match reality
            datasets[dataset_index]["image_count"] = 0
            with open(user_datasets_file, 'w') as f:
                json.dump(datasets, f)
        
        return jsonify({
            "images": [],
            "total_images": 0
        }), 200

@app.route('/api/datasets/<dataset_id>/images', methods=['POST'])
@require_auth_wrapper
def upload_image(user_data, dataset_id):
    """Upload an image to a dataset"""
    return upload_image_handler(user_data, dataset_id, IMAGE_FOLDER)

@app.route('/api/datasets/<dataset_id>/images/<image_id>', methods=['DELETE'])
@require_auth_wrapper
def remove_image(user_data, dataset_id, image_id):
    """Remove an image from a dataset"""
    # Check if dataset exists and belongs to user
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    user_datasets_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")
    
    if not os.path.exists(user_datasets_file):
        return jsonify({"error": "Dataset not found"}), 404
        
    with open(user_datasets_file, 'r') as f:
        datasets = json.load(f)
        
    dataset_exists = False
    dataset_index = -1
    for idx, dataset in enumerate(datasets):
        if dataset["id"] == dataset_id:
            dataset_exists = True
            dataset_index = idx
            break
            
    if not dataset_exists:
        return jsonify({"error": "Dataset not found"}), 404
    
    try:
        # Get image path before removal for cleanup
        image_path = None
        if dataset_id in image_processor.image_metadata:
            for img in image_processor.image_metadata[dataset_id]:
                if img["id"] == image_id:
                    image_path = img["path"]
                    break
        
        # Remove image from dataset
        success = image_processor.remove_image(dataset_id, image_id)
        if not success:
            return jsonify({"error": "Image not found in dataset"}), 404
            
        # Delete the image file if it exists
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
                print(f"Successfully deleted image file: {image_path}")
            except Exception as e:
                print(f"Warning: Failed to delete image file: {str(e)}")
        
        # Validate and update image metadata for dataset
        indices_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "image_indices")
        metadata_file = os.path.join(indices_dir, f"{dataset_id}_metadata.json")
        
        # Load the current metadata
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Filter to only include existing images
                valid_metadata = []
                for img_meta in metadata:
                    if img_meta.get('id') != image_id:  # Skip the deleted image
                        if 'path' in img_meta and os.path.exists(img_meta['path']):
                            valid_metadata.append(img_meta)
                
                # Update the image processor's metadata
                image_processor.image_metadata[dataset_id] = valid_metadata
                
                # Save updated metadata
                with open(metadata_file, 'w') as f:
                    json.dump(valid_metadata, f)
                
                # Rebuild the FAISS index to remove the deleted image
                try:
                    # Import VECTOR_DIMENSION from image_processor
                    # Standard CLIP embeddings are 512-dimensional
                    VECTOR_DIMENSION = 512
                    
                    # Create a new empty index
                    index = faiss.IndexFlatIP(VECTOR_DIMENSION)
                    
                    # Add all valid images to the index
                    for img in valid_metadata:
                        if 'embedding' in img and isinstance(img['embedding'], list):
                            # Convert embedding from list to numpy array
                            embedding = np.array([img['embedding']], dtype=np.float32)
                            index.add(embedding)
                        elif 'path' in img and os.path.exists(img['path']):
                            # If embedding missing but file exists, try to compute it
                            try:
                                embedding = image_processor.compute_image_embedding(img['path'])
                                if embedding is not None:
                                    index.add(np.array([embedding], dtype=np.float32))
                                    # Store embedding in metadata for future use
                                    img['embedding'] = embedding.tolist()
                            except Exception as emb_error:
                                print(f"Error computing embedding: {str(emb_error)}")
                    
                    # Replace the old index with the new one
                    image_processor.image_indices[dataset_id] = index
                    
                    # Save the updated index
                    index_path = os.path.join(indices_dir, f"{dataset_id}_index.faiss")
                    faiss.write_index(index, index_path)
                    print(f"Successfully rebuilt FAISS index for dataset {dataset_id} with {index.ntotal} images")
                except Exception as idx_error:
                    print(f"Error rebuilding FAISS index: {str(idx_error)}")
                
                # Update image count in dataset
                actual_count = len(valid_metadata)
                datasets[dataset_index]["image_count"] = actual_count
                print(f"Updated image count for dataset {dataset_id} to {actual_count}")
                
            except Exception as e:
                print(f"Error updating metadata after removal: {str(e)}")
                # Count images directly from processor
                if dataset_id in image_processor.image_metadata:
                    # Update count based on valid images in memory
                    valid_count = 0
                    for img in image_processor.image_metadata[dataset_id]:
                        if 'path' in img and os.path.exists(img['path']):
                            valid_count += 1
                    
                    datasets[dataset_index]["image_count"] = valid_count
                else:
                    # No metadata found, set count to 0
                    datasets[dataset_index]["image_count"] = 0
        else:
            # No metadata file, use in-memory count
            if dataset_id in image_processor.image_metadata:
                valid_count = 0
                for img in image_processor.image_metadata[dataset_id]:
                    if 'path' in img and os.path.exists(img['path']):
                        valid_count += 1
                datasets[dataset_index]["image_count"] = valid_count
            else:
                datasets[dataset_index]["image_count"] = 0
        
        # Save updated dataset info
        with open(user_datasets_file, 'w') as f:
            json.dump(datasets, f)
                
        return jsonify({"message": "Image removed successfully"}), 200
        
    except Exception as e:
        print(f"Error removing image: {str(e)}")
        return jsonify({"error": f"Failed to remove image: {str(e)}"}), 500

@app.route('/api/datasets/<dataset_id>/search-images', methods=['POST'])
@require_auth_wrapper
def search_dataset_images(user_data, dataset_id):
    """Search for images in a dataset using a text query"""
    # Check if dataset exists and belongs to user
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    user_datasets_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")
    
    if not os.path.exists(user_datasets_file):
        return jsonify({"error": "Dataset not found"}), 404
        
    with open(user_datasets_file, 'r') as f:
        datasets = json.load(f)
        
    dataset_exists = False
    for dataset in datasets:
        if dataset["id"] == dataset_id:
            dataset_exists = True
            break
            
    if not dataset_exists:
        return jsonify({"error": "Dataset not found"}), 404
    
    # Get query parameters
    data = request.json
    if not data or not data.get('query'):
        return jsonify({"error": "Search query is required"}), 400
        
    query = data['query']
    limit = data.get('limit', 5)  # Default to 5 results
    
    try:
        # Search for images
        results = image_processor.search_images(dataset_id, query, limit)
        
        # Format results
        formatted_results = []
        for result in results:
            result_copy = result.copy()
            if 'path' in result_copy:
                result_copy['url'] = f"/api/images/{os.path.basename(result_copy['path'])}"
                del result_copy['path']  # Don't expose full path
            formatted_results.append(result_copy)
            
        return jsonify({
            "query": query,
            "results": formatted_results
        }), 200
        
    except Exception as e:
        print(f"Error searching images: {str(e)}")
        return jsonify({"error": f"Failed to search images: {str(e)}"}), 500

# resize_image function is now imported from image_handlers.py

@app.route('/api/datasets/<dataset_id>/type', methods=['GET'])
@require_auth_wrapper
def get_dataset_type(user_data, dataset_id):
    """Get the type of a dataset (text, image, or mixed) to inform frontend file selection"""
    return get_dataset_type_handler(user_data, dataset_id)

# Using the imported get_mime_types_for_dataset function

@app.route('/api/upload-example', methods=['GET'])
def upload_example():
    """Serve a simple HTML example for uploading files that supports both text and image files"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RagBot File Upload Example</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { border: 1px solid #ccc; border-radius: 5px; padding: 20px; margin-bottom: 20px; }
            label { display: block; margin-bottom: 10px; font-weight: bold; }
            select, input, button { margin-bottom: 10px; padding: 8px; width: 100%; }
            button { background-color: #4CAF50; color: white; border: none; cursor: pointer; }
            button:hover { background-color: #45a049; }
            .result { margin-top: 20px; padding: 10px; background-color: #f5f5f5; border-radius: 5px; white-space: pre-wrap; }
            img { max-width: 100%; max-height: 300px; display: block; margin: 10px 0; }
        </style>
    </head>
    <body>
        <h1>RagBot File Upload Example</h1>
        
        <div class="container">
            <h2>Step 1: Select Dataset</h2>
            <label for="dataset">Choose a dataset:</label>
            <select id="dataset">
                <option value="">Loading datasets...</option>
            </select>
            <div id="dataset-info"></div>
        </div>
        
        <div class="container">
            <h2>Step 2: Upload File</h2>
            <div id="upload-form" style="display: none;">
                <label for="file">Select a file:</label>
                <input type="file" id="file" name="file">
                
                <label for="description">Description (optional):</label>
                <input type="text" id="description" name="description" placeholder="Enter a description">
                
                <label for="tags">Tags (comma-separated, optional):</label>
                <input type="text" id="tags" name="tags" placeholder="tag1, tag2, tag3">
                
                <button id="upload">Upload</button>
            </div>
        </div>
        
        <div class="container">
            <h2>Result</h2>
            <div id="result" class="result">No upload yet</div>
            <div id="image-preview"></div>
        </div>
        
        <script>
            // Get the authentication token from localStorage or prompt the user
            let token = localStorage.getItem('authToken');
            if (!token) {
                token = prompt('Please enter your authentication token:');
                if (token) localStorage.setItem('authToken', token);
            }
            
            // Fetch all datasets
            async function fetchDatasets() {
                try {
                    const response = await fetch('/api/datasets', {
                        headers: {
                            'Authorization': `Bearer ${token}`
                        }
                    });
                    
                    if (!response.ok) throw new Error('Failed to fetch datasets');
                    
                    const datasets = await response.json();
                    const select = document.getElementById('dataset');
                    select.innerHTML = '';
                    
                    if (datasets.length === 0) {
                        select.innerHTML = '<option value="">No datasets available</option>';
                        return;
                    }
                    
                    datasets.forEach(dataset => {
                        const option = document.createElement('option');
                        option.value = dataset.id;
                        option.textContent = `${dataset.name} (${dataset.type || 'text'})`;
                        option.dataset.type = dataset.type || 'text';
                        select.appendChild(option);
                    });
                    
                    // Show dataset info and setup file input
                    updateDatasetInfo();
                } catch (error) {
                    document.getElementById('result').textContent = `Error: ${error.message}`;
                }
            }
            
            // Update dataset info and file input accept attribute
            async function updateDatasetInfo() {
                const select = document.getElementById('dataset');
                const datasetId = select.value;
                
                if (!datasetId) {
                    document.getElementById('dataset-info').textContent = 'Please select a dataset';
                    document.getElementById('upload-form').style.display = 'none';
                    return;
                }
                
                try {
                    const response = await fetch(`/api/datasets/${datasetId}/type`, {
                        headers: {
                            'Authorization': `Bearer ${token}`
                        }
                    });
                    
                    if (!response.ok) throw new Error('Failed to fetch dataset info');
                    
                    const info = await response.json();
                    const datasetInfo = document.getElementById('dataset-info');
                    const fileInput = document.getElementById('file');
                    
                    datasetInfo.textContent = `Type: ${info.type}. Supported files: ${info.supported_extensions.join(', ')}`;
                    fileInput.accept = info.mime_types;
                    
                    document.getElementById('upload-form').style.display = 'block';
                } catch (error) {
                    document.getElementById('dataset-info').textContent = `Error: ${error.message}`;
                }
            }
            
            // Handle file upload
            async function uploadFile() {
                const datasetId = document.getElementById('dataset').value;
                const fileInput = document.getElementById('file');
                const description = document.getElementById('description').value;
                const tags = document.getElementById('tags').value;
                const resultDiv = document.getElementById('result');
                const imagePreview = document.getElementById('image-preview');
                
                if (!datasetId) {
                    resultDiv.textContent = 'Please select a dataset';
                    return;
                }
                
                if (!fileInput.files || fileInput.files.length === 0) {
                    resultDiv.textContent = 'Please select a file to upload';
                    return;
                }
                
                resultDiv.textContent = 'Uploading...';
                imagePreview.innerHTML = '';
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                if (description) formData.append('description', description);
                if (tags) formData.append('tags', tags);
                
                try {
                    const response = await fetch(`/api/datasets/${datasetId}/documents`, {
                        method: 'POST',
                        headers: {
                            'Authorization': `Bearer ${token}`
                        },
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (!response.ok) {
                        resultDiv.textContent = `Error: ${result.error}`;
                        return;
                    }
                    
                    resultDiv.textContent = JSON.stringify(result, null, 2);
                    
                    // If the response contains an image, display it
                    if (result.image && result.image.url) {
                        const img = document.createElement('img');
                        img.src = result.image.url;
                        img.alt = result.image.caption || 'Uploaded image';
                        imagePreview.appendChild(img);
                        
                        const caption = document.createElement('p');
                        caption.textContent = `Caption: ${result.image.caption || 'No caption available'}`;
                        imagePreview.appendChild(caption);
                    }
                } catch (error) {
                    resultDiv.textContent = `Error: ${error.message}`;
                }
            }
            
            // Setup event listeners
            document.addEventListener('DOMContentLoaded', () => {
                fetchDatasets();
                
                document.getElementById('dataset').addEventListener('change', updateDatasetInfo);
                document.getElementById('upload').addEventListener('click', uploadFile);
            });
        </script>
    </body>
    </html>
    """
    return html

@app.route('/api/documents/<document_id>/content', methods=['GET'])
@require_auth_wrapper
def get_document_content(user_data, document_id):
    """Get the full content of a document"""
    try:
        # Try to find the document in ChromaDB
        # First we need to search across all datasets the user has access to
        datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
        user_datasets_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")
        
        if not os.path.exists(user_datasets_file):
            return jsonify({"error": "No datasets found"}), 404
            
        with open(user_datasets_file, 'r') as f:
            datasets = json.load(f)
        
        # Search for the document in each dataset
        document_chunks = []
        document_metadata = None
        
        for dataset in datasets:
            dataset_id = dataset["id"]
            try:
                collection = chroma_client.get_collection(name=dataset_id, embedding_function=openai_ef)
                results = collection.get(where={"document_id": document_id})
                
                if results and len(results["documents"]) > 0:
                    # Found chunks from this document
                    document_chunks.extend(list(zip(results["documents"], results["metadatas"])))
                    # Get metadata from the first chunk
                    if not document_metadata and results["metadatas"] and len(results["metadatas"]) > 0:
                        document_metadata = results["metadatas"][0]
            except Exception as e:
                print(f"Error querying collection {dataset_id}: {str(e)}")
                continue
        
        if not document_chunks:
            return jsonify({"error": "Document not found"}), 404
            
        # Sort chunks by their position in the document
        document_chunks.sort(key=lambda x: x[1].get("chunk", 0) if x[1] else 0)
        
        # Combine all chunks into a single document
        full_content = "\n\n".join([chunk[0] for chunk in document_chunks])
        
        # Get document metadata
        filename = document_metadata.get("filename", "Unknown document") if document_metadata else "Unknown document"
        file_path = document_metadata.get("file_path", "") if document_metadata else ""
        
        # Try to get the original file if it exists
        original_content = ""
        if file_path and os.path.exists(file_path):
            try:
                original_content = extract_text_from_file(file_path)
            except Exception as e:
                print(f"Error extracting text from original file: {str(e)}")
                # Use the reconstructed content from chunks as fallback
                original_content = full_content
        else:
            # Use the reconstructed content from chunks as fallback
            original_content = full_content
        
        return jsonify({
            "document_id": document_id,
            "filename": filename,
            "content": original_content if original_content else full_content
        }), 200
        
    except Exception as e:
        print(f"Error retrieving document content: {str(e)}")
        return jsonify({"error": f"Failed to retrieve document content: {str(e)}"}), 500

@app.route('/api/context/<document_id>', methods=['GET'])
@require_auth_wrapper
def get_context_snippet(user_data, document_id):
    """Get the full content of a context snippet with option to view the entire document"""
    try:
        # Get snippet index from query parameter, if provided
        chunk_index = request.args.get('chunk', None)
        
        # Try to find the document in ChromaDB
        # First we need to search across all datasets the user has access to
        datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
        user_datasets_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")
        
        if not os.path.exists(user_datasets_file):
            return jsonify({"error": "No datasets found"}), 404
            
        with open(user_datasets_file, 'r') as f:
            datasets = json.load(f)
        
        # Search for the document in each dataset
        document_chunks = []
        document_metadata = None
        
        for dataset in datasets:
            dataset_id = dataset["id"]
            try:
                collection = chroma_client.get_collection(name=dataset_id, embedding_function=openai_ef)
                
                # If chunk index is provided, get only that specific chunk
                if chunk_index:
                    chunk_id = f"{document_id}_{chunk_index}"
                    results = collection.get(ids=[chunk_id])
                else:
                    # Otherwise get all chunks for this document
                    results = collection.get(where={"document_id": document_id})
                
                if results and len(results["documents"]) > 0:
                    # Found chunks from this document
                    document_chunks.extend(list(zip(results["documents"], results["metadatas"])))
                    # Get metadata from the first chunk
                    if not document_metadata and results["metadatas"] and len(results["metadatas"]) > 0:
                        document_metadata = results["metadatas"][0]
            except Exception as e:
                print(f"Error querying collection {dataset_id}: {str(e)}")
                continue
        
        if not document_chunks:
            return jsonify({"error": "Document chunk not found"}), 404
            
        # Sort chunks by their position in the document
        document_chunks.sort(key=lambda x: x[1].get("chunk", 0) if x[1] else 0)
        
        # Get snippet content - if specific chunk requested, return just that one
        if chunk_index:
            snippet_content = document_chunks[0][0] if document_chunks else ""
        else:
            # Otherwise combine all chunks
            snippet_content = "\n\n".join([chunk[0] for chunk in document_chunks])
        
        # Get document metadata
        filename = document_metadata.get("filename", "Unknown document") if document_metadata else "Unknown document"
        file_path = document_metadata.get("file_path", "") if document_metadata else ""
        source = document_metadata.get("source", filename) if document_metadata else filename
        
        # Check if the original file exists - will be used by frontend to offer "view original" option
        original_file_exists = file_path and os.path.exists(file_path)
        
        return jsonify({
            "document_id": document_id,
            "filename": filename,
            "source": source,
            "content": snippet_content,
            "original_file_exists": original_file_exists,
            "chunk_index": chunk_index
        }), 200
        
    except Exception as e:
        print(f"Error retrieving context snippet: {str(e)}")
        return jsonify({"error": f"Failed to retrieve context snippet: {str(e)}"}), 500

@app.route('/api/documents/<document_id>/original', methods=['GET'])
@require_auth_wrapper
def get_original_document(user_data, document_id):
    """Get the original file content if it exists"""
    try:
        # Try to find the document metadata in ChromaDB
        datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
        user_datasets_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")
        
        if not os.path.exists(user_datasets_file):
            return jsonify({"error": "No datasets found"}), 404
            
        with open(user_datasets_file, 'r') as f:
            datasets = json.load(f)
        
        # Search for the document in each dataset to get its metadata
        document_metadata = None
        
        for dataset in datasets:
            dataset_id = dataset["id"]
            try:
                collection = chroma_client.get_collection(name=dataset_id, embedding_function=openai_ef)
                
                # Query for just one chunk to get metadata
                results = collection.get(
                    where={"document_id": document_id},
                    limit=1
                )
                
                if results and len(results["metadatas"]) > 0:
                    document_metadata = results["metadatas"][0]
                    break
            except Exception as e:
                print(f"Error querying collection {dataset_id}: {str(e)}")
                continue
        
        if not document_metadata:
            return jsonify({"error": "Document not found"}), 404
            
        # Get file path from metadata
        file_path = document_metadata.get("file_path", "")
        filename = document_metadata.get("filename", "Unknown document")
        
        # Check if file exists
        if not file_path or not os.path.exists(file_path):
            return jsonify({"error": "Original file not found"}), 404
            
        # Extract text from the original file
        try:
            content = extract_text_from_file(file_path)
            file_extension = os.path.splitext(file_path)[1].lower()
            
            return jsonify({
                "document_id": document_id,
                "filename": filename,
                "content": content,
                "file_type": file_extension[1:] if file_extension.startswith('.') else file_extension
            }), 200
        except Exception as e:
            return jsonify({"error": f"Error extracting text from file: {str(e)}"}), 500
            
    except Exception as e:
        print(f"Error retrieving original document: {str(e)}")
        return jsonify({"error": f"Failed to retrieve original document: {str(e)}"}), 500

# After the "remove_dataset_from_bot" function, add the new endpoint:

@app.route('/api/bots/<bot_id>/set-datasets', methods=['POST'])
@require_auth_wrapper
def set_bot_datasets(user_data, bot_id):
    """Replace all datasets on a bot with a new list of datasets"""
    return set_bot_datasets_handler(user_data, bot_id)

@app.route('/api/datasets/<dataset_id>/bulk-upload', methods=['POST'])
@require_auth_wrapper
def bulk_upload(user_data, dataset_id):
    """Bulk upload a zip file of documents/images to a dataset"""
    from dataset_handlers import bulk_upload_handler
    return bulk_upload_handler(user_data, dataset_id)

if __name__ == '__main__':
    # Support configurable port via environment variable
    port = int(os.getenv("PORT", "50506"))
    app.run(host='0.0.0.0', port=port, debug=True)
