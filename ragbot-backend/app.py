import os
import json
import base64
import uuid
import datetime
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
from PyPDF2 import PdfReader
import docx
from pptx import Presentation
from PIL import Image  # Add this import for image processing

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
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

# Sync datasets with ChromaDB collections
def sync_datasets_with_collections():
    print("Syncing datasets with ChromaDB collections...")
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    
    # Get all dataset files
    dataset_files = [f for f in os.listdir(datasets_dir) if f.endswith('_datasets.json')]
    
    # In ChromaDB v0.6.0, list_collections() only returns collection names
    existing_collections = chroma_client.list_collections()
    
    for dataset_file in dataset_files:
        try:
            with open(os.path.join(datasets_dir, dataset_file), 'r') as f:
                datasets = json.load(f)
                
            for dataset in datasets:
                dataset_id = dataset["id"]
                
                # Check if collection exists in ChromaDB
                if dataset_id not in existing_collections:
                    print(f"Creating missing collection for dataset: {dataset_id}")
                    try:
                        chroma_client.create_collection(name=dataset_id, embedding_function=openai_ef)
                    except Exception as e:
                        print(f"Error creating collection for dataset {dataset_id}: {str(e)}")
        except Exception as e:
            print(f"Error processing dataset file {dataset_file}: {str(e)}")
    
    print("Dataset sync completed")

# Run the sync on app startup
sync_datasets_with_collections()

# Helper functions
def get_token_from_header():
    auth_header = request.headers.get('Authorization')
    if not auth_header or 'Bearer ' not in auth_header:
        return None
    return auth_header.split('Bearer ')[1]

def verify_token():
    token = get_token_from_header()
    if not token:
        return None
    try:
        decoded = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=["HS256"])
        return decoded
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def require_auth(f):
    def decorated(*args, **kwargs):
        user_data = verify_token()
        if not user_data:
            return jsonify({"error": "Unauthorized"}), 401
        return f(user_data, *args, **kwargs)
    decorated.__name__ = f.__name__
    return decorated

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(docx_path):
    text = ""
    doc = docx.Document(docx_path)
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def extract_text_from_pptx(pptx_path):
    text = ""
    prs = Presentation(pptx_path)
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + "\n"
    return text

def extract_text_from_file(file_path):
    extension = os.path.splitext(file_path)[1].lower()
    if extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif extension == '.docx':
        return extract_text_from_docx(file_path)
    elif extension == '.pptx':
        return extract_text_from_pptx(file_path)
    elif extension == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return ""

# Add resize_image function before your routes
def resize_image(image_path, max_dimension=2048):
    """Resize image if it's too large while maintaining aspect ratio"""
    try:
        img = Image.open(image_path)
        
        # Check if resize is needed
        width, height = img.size
        if width <= max_dimension and height <= max_dimension:
            return image_path  # No resize needed
            
        # Calculate new dimensions
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
            
        # Resize and save
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Create new filename for resized image
        filename, ext = os.path.splitext(image_path)
        resized_path = f"{filename}_resized{ext}"
        
        # Save resized image
        img.save(resized_path)
        return resized_path
    except Exception as e:
        print(f"Error resizing image: {str(e)}")
        return image_path  # Return original path if resizing fails

# Authentication routes
@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400
        
    # Check if user already exists - for demo purposes using a simple file
    users_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json")
    users = {}
    if os.path.exists(users_file):
        with open(users_file, 'r') as f:
            users = json.load(f)
    
    if username in users:
        return jsonify({"error": "Username already exists"}), 400
        
    # Hash password and store user
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    user_id = str(uuid.uuid4())
    users[username] = {
        "id": user_id,
        "username": username,
        "password": hashed_password
    }
    
    with open(users_file, 'w') as f:
        json.dump(users, f)
        
    # Generate token
    token = jwt.encode(
        {"id": user_id, "username": username, "exp": datetime.datetime.utcnow() + app.config['JWT_ACCESS_TOKEN_EXPIRES']},
        app.config['JWT_SECRET_KEY']
    )
    
    return jsonify({"token": token, "user": {"id": user_id, "username": username}}), 201

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400
        
    # Load users - for demo purposes using a simple file
    users_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json")
    if not os.path.exists(users_file):
        return jsonify({"error": "Invalid credentials"}), 401
        
    with open(users_file, 'r') as f:
        users = json.load(f)
        
    if username not in users:
        return jsonify({"error": "Invalid credentials"}), 401
        
    user = users[username]
    if not bcrypt.checkpw(password.encode('utf-8'), user["password"].encode('utf-8')):
        return jsonify({"error": "Invalid credentials"}), 401
        
    # Generate token
    token = jwt.encode(
        {"id": user["id"], "username": user["username"], "exp": datetime.datetime.utcnow() + app.config['JWT_ACCESS_TOKEN_EXPIRES']},
        app.config['JWT_SECRET_KEY']
    )
    
    return jsonify({"token": token, "user": {"id": user["id"], "username": user["username"]}}), 200

# Dataset routes
@app.route('/api/datasets', methods=['GET'])
@require_auth
def get_datasets(user_data):
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    user_datasets_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")
    
    if not os.path.exists(user_datasets_file):
        return jsonify([]), 200
        
    with open(user_datasets_file, 'r') as f:
        datasets = json.load(f)
        
    return jsonify(datasets), 200

@app.route('/api/datasets', methods=['POST'])
@require_auth
def create_dataset(user_data):
    data = request.json
    name = data.get('name')
    description = data.get('description', '')
    
    if not name:
        return jsonify({"error": "Dataset name is required"}), 400
        
    dataset_id = str(uuid.uuid4())
    
    # Save dataset metadata
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    user_datasets_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")
    
    datasets = []
    if os.path.exists(user_datasets_file):
        with open(user_datasets_file, 'r') as f:
            datasets = json.load(f)
            
    dataset = {
        "id": dataset_id,
        "name": name,
        "description": description,
        "created_at": datetime.datetime.utcnow().isoformat(),
        "document_count": 0
    }
    datasets.append(dataset)
    
    with open(user_datasets_file, 'w') as f:
        json.dump(datasets, f)
        
    # Create collection in ChromaDB
    chroma_client.create_collection(name=dataset_id, embedding_function=openai_ef)
    
    return jsonify(dataset), 201

@app.route('/api/datasets/<dataset_id>/documents', methods=['POST'])
@require_auth
def upload_document(user_data, dataset_id):
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    # Check if dataset exists and belongs to user
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    user_datasets_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")
    
    if not os.path.exists(user_datasets_file):
        return jsonify({"error": "Dataset not found"}), 404
        
    with open(user_datasets_file, 'r') as f:
        datasets = json.load(f)
        
    dataset_exists = False
    for idx, dataset in enumerate(datasets):
        if dataset["id"] == dataset_id:
            dataset_exists = True
            break
            
    if not dataset_exists:
        return jsonify({"error": "Dataset not found"}), 404
        
    # Save the file
    filename = secure_filename(file.filename)
    file_path = os.path.join(DOCUMENT_FOLDER, f"{uuid.uuid4()}_{filename}")
    file.save(file_path)
    
    # Extract text from file
    text = extract_text_from_file(file_path)
    if not text:
        os.remove(file_path)
        return jsonify({"error": "Could not extract text from file"}), 400
        
    # Create chunks from text (simplified chunking for demo)
    chunk_size = 1000
    overlap = 200
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk:
            chunks.append(chunk)
            
    # Add chunks to ChromaDB
    collection = chroma_client.get_collection(name=dataset_id, embedding_function=openai_ef)
    document_id = str(uuid.uuid4())
    
    metadata_entries = [{"source": filename, "document_id": document_id, "chunk_index": i} for i in range(len(chunks))]
    chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
    
    collection.add(
        documents=chunks,
        metadatas=metadata_entries,
        ids=chunk_ids
    )
    
    # Update document count in dataset
    for idx, dataset in enumerate(datasets):
        if dataset["id"] == dataset_id:
            dataset["document_count"] += 1
            break
            
    with open(user_datasets_file, 'w') as f:
        json.dump(datasets, f)
        
    return jsonify({"message": "Document uploaded and processed successfully"}), 200

@app.route('/api/datasets/<dataset_id>/documents/<document_id>', methods=['DELETE'])
@require_auth
def remove_document(user_data, dataset_id, document_id):
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
    
    # Try to get the collection from ChromaDB
    try:
        collection = chroma_client.get_collection(name=dataset_id, embedding_function=openai_ef)
        
        # Get all chunks with the matching document_id in their metadata
        results = collection.get(
            where={"document_id": document_id}
        )
        
        if not results or len(results['ids']) == 0:
            return jsonify({"error": "Document not found in dataset"}), 404
        
        # Delete the chunks from ChromaDB
        collection.delete(
            ids=results['ids']
        )
        
        # Update document count in dataset
        if datasets[dataset_index]["document_count"] > 0:
            datasets[dataset_index]["document_count"] -= 1
            
        with open(user_datasets_file, 'w') as f:
            json.dump(datasets, f)
            
        return jsonify({"message": "Document removed successfully"}), 200
    
    except Exception as e:
        print(f"Error removing document: {str(e)}")
        return jsonify({"error": f"Failed to remove document: {str(e)}"}), 500

# Admin routes
@app.route('/api/admin/rebuild_dataset/<dataset_id>', methods=['POST'])
@require_auth
def rebuild_dataset(user_data, dataset_id):
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
    
    # Reset document count
    datasets[dataset_index]["document_count"] = 0
    
    with open(user_datasets_file, 'w') as f:
        json.dump(datasets, f)
    
    # Try to delete the existing collection if it exists
    try:
        existing_collections = chroma_client.list_collections()
        if dataset_id in existing_collections:
            chroma_client.delete_collection(name=dataset_id)
            print(f"Deleted existing collection for dataset: {dataset_id}")
    except Exception as e:
        print(f"Error deleting collection: {str(e)}")
    
    # Create a new collection
    try:
        chroma_client.create_collection(name=dataset_id, embedding_function=openai_ef)
        return jsonify({"message": "Dataset collection has been rebuilt. Please re-upload your documents."}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to rebuild dataset: {str(e)}"}), 500

@app.route('/api/datasets/<dataset_id>/status', methods=['GET'])
@require_auth
def dataset_status(user_data, dataset_id):
    # Check if dataset exists and belongs to user
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    user_datasets_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")
    
    if not os.path.exists(user_datasets_file):
        return jsonify({"error": "Dataset not found"}), 404
        
    with open(user_datasets_file, 'r') as f:
        datasets = json.load(f)
        
    dataset = None
    for d in datasets:
        if d["id"] == dataset_id:
            dataset = d
            break
            
    if not dataset:
        return jsonify({"error": "Dataset not found"}), 404
    
    # Check ChromaDB collection status
    existing_collections = chroma_client.list_collections()
    collection_exists = dataset_id in existing_collections
    
    doc_count = 0
    if collection_exists:
        try:
            collection = chroma_client.get_collection(name=dataset_id, embedding_function=openai_ef)
            doc_count = collection.count()
        except Exception as e:
            return jsonify({
                "dataset": dataset,
                "collection_exists": False,
                "error": str(e)
            }), 200
    
    return jsonify({
        "dataset": dataset,
        "collection_exists": collection_exists,
        "document_count": doc_count
    }), 200

@app.route('/api/datasets/<dataset_id>/documents', methods=['GET'])
@require_auth
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
        
        # Process results to get unique documents
        documents = {}
        for i, metadata in enumerate(results['metadatas']):
            if metadata and 'document_id' in metadata and 'source' in metadata:
                doc_id = metadata['document_id']
                if doc_id not in documents:
                    documents[doc_id] = {
                        'id': doc_id,
                        'filename': metadata['source'],
                        'chunk_count': 1
                    }
                else:
                    documents[doc_id]['chunk_count'] += 1
        
        return jsonify({"documents": list(documents.values())}), 200
    
    except Exception as e:
        print(f"Error getting documents: {str(e)}")
        return jsonify({"error": f"Failed to get documents: {str(e)}"}), 500

@app.route('/api/datasets/<dataset_id>', methods=['DELETE'])
@require_auth
def delete_dataset(user_data, dataset_id):
    # Check if dataset exists and belongs to user
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    user_datasets_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")
    
    if not os.path.exists(user_datasets_file):
        return jsonify({"error": "Dataset not found"}), 404
        
    with open(user_datasets_file, 'r') as f:
        datasets = json.load(f)
    
    # Find and remove the dataset
    dataset_index = -1
    for idx, dataset in enumerate(datasets):
        if dataset["id"] == dataset_id:
            dataset_index = idx
            break
    
    if dataset_index == -1:
        return jsonify({"error": "Dataset not found"}), 404
    
    # Remove dataset from the list
    deleted_dataset = datasets.pop(dataset_index)
    
    # Save updated datasets list
    with open(user_datasets_file, 'w') as f:
        json.dump(datasets, f)
    
    # Delete ChromaDB collection if it exists
    try:
        existing_collections = chroma_client.list_collections()
        if dataset_id in existing_collections:
            chroma_client.delete_collection(name=dataset_id)
            print(f"Deleted ChromaDB collection for dataset: {dataset_id}")
    except Exception as e:
        print(f"Error deleting ChromaDB collection: {str(e)}")
        # Continue with deletion even if ChromaDB deletion fails
    
    # Check if any bots are using this dataset and notify in the response
    bots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bots")
    user_bots_file = os.path.join(bots_dir, f"{user_data['id']}_bots.json")
    affected_bots = []
    
    if os.path.exists(user_bots_file):
        try:
            with open(user_bots_file, 'r') as f:
                bots = json.load(f)
            
            for bot in bots:
                if bot.get("dataset_id") == dataset_id:
                    affected_bots.append(bot["name"])
        except Exception as e:
            print(f"Error checking for affected bots: {str(e)}")
    
    return jsonify({
        "message": "Dataset deleted successfully", 
        "deleted_dataset": deleted_dataset["name"],
        "affected_bots": affected_bots
    }), 200

# Chat routes
@app.route('/api/bots/<bot_id>/chat', methods=['POST'])
@require_auth
def chat_with_bot(user_data, bot_id):
    data = request.json
    message = data.get('message')
    debug_mode = data.get('debug_mode', False)
    use_model_chorus = data.get('use_model_chorus', False)  # User's explicit choice to use model chorus
    chorus_id = data.get('chorus_id', '')  # A specific chorus ID to use
    
    if not message:
        return jsonify({"error": "Message is required"}), 400
        
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
    
    if not dataset_ids:
        return jsonify({
            "response": "I don't have any datasets to work with. Please add a dataset to help me answer your questions."
        }), 200
    
    # Check if the bot has a chorus configuration associated with it
    # If so, use model chorus by default unless explicitly turned off
    bot_has_chorus = bot.get("chorus_id", "")
    use_model_chorus = use_model_chorus or bool(bot_has_chorus)
    
    # If a specific chorus_id is provided, use it; otherwise use the bot's chorus_id if available
    specific_chorus_id = chorus_id or bot.get("chorus_id", "")
    
    # Retrieve relevant documents from all datasets
    all_contexts = []
    
    try:
        for dataset_id in dataset_ids:
            try:
                collection = chroma_client.get_collection(name=dataset_id, embedding_function=openai_ef)
                
                # Check if collection has any documents
                collection_count = collection.count()
                if collection_count == 0:
                    continue
                    
                # Determine how many results to request based on collection size
                n_results = min(3, collection_count)  # Reduced from 5 to 3 since we have multiple datasets
                
                results = collection.query(
                    query_texts=[message],
                    n_results=n_results
                )
                
                all_contexts.extend(results["documents"][0])
                
            except Exception as coll_error:
                print(f"Error with collection '{dataset_id}': {str(coll_error)}")
                # Continue with other datasets even if one fails
                continue
                
        if not all_contexts:
            return jsonify({
                "response": "I don't have any documents in my knowledge base yet. Please upload some documents to help me answer your questions.",
                "debug": {"error": "No contexts found in any collections"} if debug_mode else None
            }), 200
            
        # Sort contexts by relevance (they should already be sorted from query results)
        # But we need to truncate to avoid token limits
        max_contexts = 8  # Limit the total number of contexts used
        contexts = all_contexts[:max_contexts]
        context_text = "\n\n".join(contexts)
        
        # Prepare the system instruction and context
        system_instruction = bot.get("system_instruction", "You are a helpful assistant that answers questions based on the provided context.")
        
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
                        print(f"Using chorus configuration: {chorus_config.get('name', 'Unnamed')}")
                except Exception as e:
                    print(f"Error loading chorus configuration: {str(e)}")
            
            # If no chorus config found, return an error
            if not chorus_config:
                return jsonify({
                    "response": "I couldn't find the model chorus configuration. Please check that the chorus exists and is properly configured.",
                    "debug": {"error": "Chorus configuration not found"} if debug_mode else None
                }), 200
            
            # Get response and evaluator models
            response_models = chorus_config.get('response_models', [])
            evaluator_models = chorus_config.get('evaluator_models', [])
            
            # For debugging
            if debug_mode:
                print(f"Using chorus configuration: {chorus_config.get('name', 'Unnamed')}")
                print(f"Response models: {len(response_models)}")
                print(f"Evaluator models: {len(evaluator_models)}")
            
            # Validate the configuration
            if not response_models or not evaluator_models:
                return jsonify({
                    "response": "The model chorus configuration is incomplete. Please configure both response and evaluator models.",
                    "debug": {"error": "Incomplete chorus configuration"} if debug_mode else None
                }), 200
            
            logs = []
            logs.append(f"Using model chorus: {chorus_config.get('name', 'Unnamed')}")
            
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
                
                for i in range(weight):
                    try:
                        if provider == 'OpenAI':
                            response = openai.chat.completions.create(
                                model=model_name,
                                messages=[
                                    {"role": "system", "content": system_instruction},
                                    {"role": "user", "content": f"Context:\n{context_text}\n\nUser question: {message}"}
                                ],
                                temperature=temperature
                            )
                            response_text = response.choices[0].message.content
                            anonymized_responses.append(response_text)
                            response_metadata.append({
                                "provider": provider,
                                "model": model_name,
                                "temperature": temperature
                            })
                            all_responses.append({
                                "provider": provider,
                                "model": model_name,
                                "response": response_text,
                                "temperature": temperature
                            })
                        elif provider == 'Anthropic':
                            response = anthropic_client.messages.create(
                                model=model_name,
                                system=system_instruction,
                                messages=[{"role": "user", "content": f"Context:\n{context_text}\n\nUser question: {message}"}],
                                temperature=temperature,
                                max_tokens=1024
                            )
                            response_text = response.content[0].text
                            anonymized_responses.append(response_text)
                            response_metadata.append({
                                "provider": provider,
                                "model": model_name,
                                "temperature": temperature
                            })
                            all_responses.append({
                                "provider": provider,
                                "model": model_name,
                                "response": response_text,
                                "temperature": temperature
                            })
                        elif provider == 'Groq':
                            headers = {
                                "Authorization": f"Bearer {GROQ_API_KEY}",
                                "Content-Type": "application/json"
                            }
                            payload = {
                                "model": model_name,
                                "messages": [
                                    {"role": "system", "content": system_instruction},
                                    {"role": "user", "content": f"Context:\n{context_text}\n\nUser question: {message}"}
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
                                "temperature": temperature
                            })
                            all_responses.append({
                                "provider": provider,
                                "model": model_name,
                                "response": response_text,
                                "temperature": temperature
                            })
                        elif provider == 'Mistral':
                            # Note: This would require a Mistral API implementation
                            # For now we'll use OpenAI as a fallback and log it
                            logs.append(f"Mistral API not implemented, using OpenAI fallback for {model_name}")
                            response = openai.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": system_instruction},
                                    {"role": "user", "content": f"Context:\n{context_text}\n\nUser question: {message}"}
                                ],
                                temperature=temperature
                            )
                            response_text = response.choices[0].message.content
                            anonymized_responses.append(response_text)
                            response_metadata.append({
                                "provider": "OpenAI (Mistral fallback)",
                                "model": "gpt-3.5-turbo",
                                "temperature": temperature
                            })
                            all_responses.append({
                                "provider": "OpenAI (Mistral fallback)",
                                "model": "gpt-3.5-turbo",
                                "response": response_text,
                                "temperature": temperature
                            })
                    except Exception as e:
                        logs.append(f"Error with {provider} {model_name}: {str(e)}")
            
            # If no responses, use fallback
            if not all_responses:
                return jsonify({
                    "response": "I encountered an issue with the model chorus. No models were able to generate a response.",
                    "debug": {"error": "No models returned responses", "logs": logs} if debug_mode else None
                }), 200
            
            # If only one response, return it directly
            if len(all_responses) == 1:
                return jsonify({
                    "response": all_responses[0]["response"],
                    "debug": {
                        "all_responses": all_responses,
                        "anonymized_responses": anonymized_responses,
                        "response_metadata": response_metadata,
                        "logs": logs,
                        "contexts": contexts
                    } if debug_mode else None
                }), 200
            
            # Get voting from evaluator models
            votes = [0] * len(all_responses)
            
            for model in evaluator_models:
                provider = model.get('provider')
                model_name = model.get('model')
                temperature = float(model.get('temperature', 0.2))
                weight = int(model.get('weight', 1))
                
                voting_prompt = f"""You are an expert AI response evaluator. You need to rank the following responses to the question: "{message}"

Here are the {len(anonymized_responses)} candidate responses:

{chr(10).join([f"Response {j+1}:\n{resp}" for j, resp in enumerate(anonymized_responses)])}

Which response provides the most accurate, helpful, and relevant answer? Return ONLY the number (1-{len(anonymized_responses)}) of the best response.
Do not reveal any bias or preference based on writing style or approach - evaluate solely on answer quality, accuracy and helpfulness.
"""
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
                                "Authorization": f"Bearer {GROQ_API_KEY}",
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
            
            return jsonify({
                "response": winning_response,
                "debug": {
                    "all_responses": all_responses,
                    "anonymized_responses": anonymized_responses,
                    "response_metadata": response_metadata,
                    "votes": votes,
                    "logs": logs,
                    "contexts": contexts
                } if debug_mode else None
            }), 200
                
    
        # Standard mode - just get a response from OpenAI
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Context:\n{context_text}\n\nUser question: {message}"}
            ]
        )
        
        return jsonify({
            "response": response.choices[0].message.content,
            "debug": {"contexts": contexts} if debug_mode else None
        }), 200
        
    except Exception as e:
        print(f"Error in chat_with_bot: {str(e)}")
        return jsonify({
            "response": "I apologize, but I encountered an error while processing your request. Please try again or contact support if the issue persists.",
            "debug": {"error": str(e)} if debug_mode else None
        }), 200

# Add model chorus API endpoints
@app.route('/api/bots/<bot_id>/chorus', methods=['GET'])
@require_auth
def get_chorus_config(user_data, bot_id):
    # Check if bot exists and belongs to user
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
    
    # Check for chorus_id in the bot
    chorus_id = bot.get("chorus_id", "")
    if not chorus_id:
        return jsonify({"error": "No chorus configuration found for this bot"}), 404
    
    # Get the user's chorus definitions
    chorus_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chorus")
    os.makedirs(chorus_dir, exist_ok=True)
    
    user_choruses_file = os.path.join(chorus_dir, f"{user_data['id']}_choruses.json")
    if not os.path.exists(user_choruses_file):
        return jsonify({"error": "No chorus configurations found"}), 404
    
    # Load the user's choruses and find the one with the matching ID
    with open(user_choruses_file, 'r') as f:
        choruses = json.load(f)
    
    chorus = next((c for c in choruses if c["id"] == chorus_id), None)
    if not chorus:
        return jsonify({"error": f"Chorus configuration with ID {chorus_id} not found"}), 404
    
    return jsonify(chorus), 200

@app.route('/api/bots/<bot_id>/chorus', methods=['POST'])
@require_auth
def save_chorus_config(user_data, bot_id):
    # This endpoint is now simplified to just associate a chorus with a bot
    # For backward compatibility
    
    # Check if bot exists and belongs to user
    bots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bots")
    user_bots_file = os.path.join(bots_dir, f"{user_data['id']}_bots.json")
    
    if not os.path.exists(user_bots_file):
        return jsonify({"error": "Bot not found"}), 404
        
    with open(user_bots_file, 'r') as f:
        bots = json.load(f)
        
    bot = None
    bot_index = -1
    for i, b in enumerate(bots):
        if b["id"] == bot_id:
            bot = b
            bot_index = i
            break
            
    if not bot:
        return jsonify({"error": "Bot not found"}), 404
    
    # Get the chorus data
    data = request.json
    
    # First, create or update a chorus
    try:
        # Create chorus directory if it doesn't exist
        chorus_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chorus")
        os.makedirs(chorus_dir, exist_ok=True)
        
        # Load or create user's chorus list
        user_choruses_file = os.path.join(chorus_dir, f"{user_data['id']}_choruses.json")
        
        if os.path.exists(user_choruses_file):
            with open(user_choruses_file, 'r') as f:
                choruses = json.load(f)
        else:
            choruses = []
        
        # Generate a unique ID for the chorus
        chorus_id = str(uuid.uuid4())
        
        # Create chorus data
        chorus_data = {
            "id": chorus_id,
            "name": data.get('name', 'Unnamed Chorus'),
            "description": data.get('description', ''),
            "response_models": data.get('response_models', []),
            "evaluator_models": data.get('evaluator_models', []),
            "created_at": datetime.datetime.utcnow().isoformat(),
            "updated_at": datetime.datetime.utcnow().isoformat(),
            "created_by": user_data['username']
        }
        
        # Add the chorus
        choruses.append(chorus_data)
        
        # Save the updated choruses
        with open(user_choruses_file, 'w') as f:
            json.dump(choruses, f)
            
        # Update the bot to use this chorus
        bots[bot_index]["chorus_id"] = chorus_id
        
        # Save the updated bot
        with open(user_bots_file, 'w') as f:
            json.dump(bots, f)
        
        return jsonify({
            "message": "Chorus created and assigned to bot successfully", 
            "config": chorus_data,
            "bot": bots[bot_index]
        }), 200
    except Exception as e:
        print(f"Error creating chorus: {str(e)}")
        return jsonify({"error": f"Error creating chorus: {str(e)}"}), 500

@app.route('/api/bots/<bot_id>/debug-flowchart', methods=['POST'])
@require_auth
def generate_debug_flowchart(user_data, bot_id):
    data = request.json
    message = data.get('message')
    use_model_chorus = data.get('use_model_chorus', False)
    
    if not message:
        return jsonify({"error": "Message is required"}), 400
        
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
        
    # Get dataset info
    dataset_id = bot["dataset_id"]
    
    # Retrieve relevant documents from ChromaDB
    try:
        try:
            collection = chroma_client.get_collection(name=dataset_id, embedding_function=openai_ef)
        except Exception as coll_error:
            return jsonify({
                "error": f"Collection '{dataset_id}' not found."
            }), 404
                
        # Check if collection has any documents
        collection_count = collection.count()
        if collection_count == 0:
            return jsonify({
                "error": "No documents in collection"
            }), 404
                
        # Determine how many results to request based on collection size
        n_results = min(5, collection_count)
        
        results = collection.query(
            query_texts=[message],
            n_results=n_results
        )
        
        contexts = results["documents"][0]
        context_text = "\n\n".join(contexts)
        
        # Prepare prompt
        system_instruction = bot["system_instruction"]
        
        # Generate responses using model chorus if enabled
        if use_model_chorus:
            # Get responses from multiple models
            all_responses = []
            logs = ["Model chorus mode enabled"]
            
            # Function to generate OpenAI responses
            def get_openai_responses(count=3):
                openai_responses = []
                for i in range(count):
                    try:
                        response = openai.chat.completions.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": system_instruction},
                                {"role": "user", "content": f"Context:\n{context_text}\n\nUser question: {message}"}
                            ],
                            temperature=0.5 + (i * 0.25),
                        )
                        response_text = response.choices[0].message.content
                        openai_responses.append({
                            "provider": "OpenAI",
                            "model": "GPT-4",
                            "temperature": 0.5 + (i * 0.25),
                            "response": response_text
                        })
                    except Exception as e:
                        logs.append(f"Error getting OpenAI response: {str(e)}")
                return openai_responses
                
            # Function to generate Anthropic responses
            def get_anthropic_responses(count=3):
                anthropic_responses = []
                for i in range(count):
                    try:
                        response = anthropic_client.messages.create(
                            model="claude-3-opus-20240229",
                            system=system_instruction,
                            messages=[
                                {"role": "user", "content": f"Context:\n{context_text}\n\nUser question: {message}"}
                            ],
                            temperature=0.5 + (i * 0.25),
                            max_tokens=1024
                        )
                        response_text = response.content[0].text
                        anthropic_responses.append({
                            "provider": "Anthropic",
                            "model": "Claude 3 Opus",
                            "temperature": 0.5 + (i * 0.25),
                            "response": response_text
                        })
                    except Exception as e:
                        logs.append(f"Error getting Anthropic response: {str(e)}")
                return anthropic_responses
                
            # Function to generate Groq responses
            def get_groq_responses(count=3):
                groq_responses = []
                for i in range(count):
                    try:
                        headers = {
                            "Authorization": f"Bearer {GROQ_API_KEY}",
                            "Content-Type": "application/json"
                        }
                        payload = {
                            "model": "llama3-70b-8192",
                            "messages": [
                                {"role": "system", "content": system_instruction},
                                {"role": "user", "content": f"Context:\n{context_text}\n\nUser question: {message}"}
                            ],
                            "temperature": 0.5 + (i * 0.25),
                            "max_tokens": 1024
                        }
                        response = requests.post(GROQ_API_URL, json=payload, headers=headers)
                        response_json = response.json()
                        response_text = response_json["choices"][0]["message"]["content"]
                        groq_responses.append({
                            "provider": "Groq",
                            "model": "LLaMA 3 70B",
                            "temperature": 0.5 + (i * 0.25),
                            "response": response_text
                        })
                    except Exception as e:
                        logs.append(f"Error getting Groq response: {str(e)}")
                return groq_responses
            
            # Gather responses from all providers
            openai_responses = get_openai_responses(3)
            anthropic_responses = get_anthropic_responses(3)
            groq_responses = get_groq_responses(3)
            
            # Combine all responses
            all_responses = openai_responses + anthropic_responses + groq_responses
            responses = [r["response"] for r in all_responses]
            
            # Generate Mermaid diagram code with provider information
            try:
                # Create a model chorus flowchart
                mermaid_code = "flowchart TD\n"
                
                # Define node IDs clearly to avoid conflicts
                mermaid_code += "    userQuery[\"User Query\"]\n"
                mermaid_code += "    contextRetrieval[\"Context Retrieval\"]\n"
                
                # Define provider groups
                mermaid_code += "    subgraph OpenAI\n"
                for i in range(len(openai_responses)):
                    mermaid_code += f"        openai{i+1}[\"OpenAI {i+1}\"]\n"
                mermaid_code += "    end\n"
                
                mermaid_code += "    subgraph Anthropic\n"
                for i in range(len(anthropic_responses)):
                    mermaid_code += f"        anthropic{i+1}[\"Claude {i+1}\"]\n"
                mermaid_code += "    end\n"
                
                mermaid_code += "    subgraph Groq\n"
                for i in range(len(groq_responses)):
                    mermaid_code += f"        groq{i+1}[\"LLaMA {i+1}\"]\n"
                mermaid_code += "    end\n"
                
                # Define evaluator node
                mermaid_code += "    evaluator[\"Voting Process\"]\n"
                
                # Define winner and final response
                mermaid_code += "    winner[\"Winning Response\"]\n"
                
                # Connect nodes
                mermaid_code += "    userQuery --> contextRetrieval\n"
                
                # Connect to each model
                for i in range(len(openai_responses)):
                    mermaid_code += f"    contextRetrieval --> openai{i+1}\n"
                    mermaid_code += f"    openai{i+1} --> evaluator\n"
                
                for i in range(len(anthropic_responses)):
                    mermaid_code += f"    contextRetrieval --> anthropic{i+1}\n"
                    mermaid_code += f"    anthropic{i+1} --> evaluator\n"
                
                for i in range(len(groq_responses)):
                    mermaid_code += f"    contextRetrieval --> groq{i+1}\n"
                    mermaid_code += f"    groq{i+1} --> evaluator\n"
                
                # Connect to winner
                mermaid_code += "    evaluator --> winner\n"
                
                # Add styling
                mermaid_code += "    classDef openai fill:#c1edc9,stroke:#333;\n"
                mermaid_code += "    classDef anthropic fill:#d4e1f5,stroke:#333;\n"
                mermaid_code += "    classDef groq fill:#f9d6c4,stroke:#333;\n"
                mermaid_code += "    classDef winner fill:#f9f,stroke:#333,stroke-width:4px;\n"
                
                # Apply classes
                for i in range(len(openai_responses)):
                    mermaid_code += f"    class openai{i+1} openai;\n"
                
                for i in range(len(anthropic_responses)):
                    mermaid_code += f"    class anthropic{i+1} anthropic;\n"
                
                for i in range(len(groq_responses)):
                    mermaid_code += f"    class groq{i+1} groq;\n"
                
                mermaid_code += "    class winner winner;\n"
                
                return jsonify({
                    "mermaid_code": mermaid_code,
                    "data": {
                        "responses": responses,
                        "all_responses": all_responses,
                        "logs": logs,
                        "contexts": contexts
                    }
                }), 200
            except Exception as mermaid_error:
                print(f"Error generating chorus Mermaid code: {str(mermaid_error)}")
                # Provide a simplified fallback diagram
                mermaid_code = """flowchart TD
    A[User Query] --> B[Multi-Model Processing]
    B --> C[Response]
    style C fill:#f9f,stroke:#333,stroke-width:4px"""
                
                return jsonify({
                    "mermaid_code": mermaid_code,
                    "data": {
                        "responses": responses,
                        "all_responses": all_responses,
                        "logs": logs,
                        "contexts": contexts,
                        "error": str(mermaid_error)
                    }
                }), 200
        else:
            # In debug mode - generate 5 responses and let them vote
            responses = []
            anonymized_responses = []
            response_metadata = []
            logs = ["Debug mode enabled - generating 5 different responses"]
            
            for i in range(5):
                logs.append(f"Generating response {i+1}/5...")
                response = openai.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": f"Context:\n{contexts}\n\nUser question: {message}"}
                    ],
                    temperature=0.7 + (i * 0.1),  # Vary temperature to get different responses
                )
                response_text = response.choices[0].message.content
                responses.append(response_text)
                anonymized_responses.append(response_text)
                response_metadata.append({
                    "index": i+1,
                    "temperature": 0.7 + (i * 0.1)
                })
                logs.append(f"Response {i+1}: {response_text[:100]}...")
                
            # Have the responses vote on each other
            logs.append("Having the responses evaluate each other...")
            votes = [0, 0, 0, 0, 0]
            vote_details = []
            
            for i in range(5):
                voting_prompt = f"""You are an expert evaluator. You need to rank the following responses to the question: "{message}"
                
Here are the 5 candidate responses:

Response 1:
{anonymized_responses[0]}

Response 2:
{anonymized_responses[1]}

Response 3:
{anonymized_responses[2]}

Response 4:
{anonymized_responses[3]}

Response 5:
{anonymized_responses[4]}

Which response provides the most accurate, helpful, and relevant answer? Return ONLY the number (1-5) of the best response.
Do not reveal any bias or preference based on writing style or approach - evaluate solely on answer quality, accuracy and helpfulness.
"""
                voting_response = openai.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": voting_prompt}],
                    temperature=0.2,
                )
                
                vote_text = voting_response.choices[0].message.content
                
                try:
                    # More robust vote extraction 
                    vote_number = None
                    vote_text_lower = vote_text.lower().strip()
                    
                    # Try regex first
                    match = re.search(r'\b([1-5])\b', vote_text_lower)
                    if match:
                        vote_number = int(match.group(1))
                    # If no match, look for number words
                    elif "one" in vote_text_lower or "first" in vote_text_lower:
                        vote_number = 1
                    elif "two" in vote_text_lower or "second" in vote_text_lower:
                        vote_number = 2
                    elif "three" in vote_text_lower or "third" in vote_text_lower:
                        vote_number = 3
                    elif "four" in vote_text_lower or "fourth" in vote_text_lower:
                        vote_number = 4
                    elif "five" in vote_text_lower or "fifth" in vote_text_lower:
                        vote_number = 5
                    # Fall back to the old method
                    else:
                        for j in range(1, 6):
                            if str(j) in vote_text:
                                vote_number = j
                                break
                    
                    # Record the vote
                    if vote_number and 1 <= vote_number <= 5:
                        votes[vote_number-1] += 1
                        logs.append(f"Evaluator {i+1} voted for Response {vote_number}")
                    else:
                        logs.append(f"Evaluator {i+1} vote unclear: {vote_text}")
                    
                    vote_details.append({
                        "evaluator": i+1,
                        "vote": vote_number,
                        "raw_text": vote_text
                    })
                except Exception as e:
                    logs.append(f"Couldn't parse vote from: {vote_text}. Error: {str(e)}")
                    vote_details.append({
                        "evaluator": i+1,
                        "vote": None,
                        "raw_text": vote_text
                    })
            
            # Log the final results
            logs.append(f"Voting results: {votes}")
            
            # Find the winning response
            max_votes = max(votes)
            winning_indices = [i for i, v in enumerate(votes) if v == max_votes]
            winning_index = winning_indices[0]  # Take the first one in case of a tie
            
            logs.append(f"Response {winning_index + 1} wins with {max_votes} votes")
            
            # Generate Mermaid diagram code - with safer, more reliable text escaping
            try:
                # Create a simpler, more reliable flowchart
                mermaid_code = "flowchart TD\n"
                
                # Define node IDs clearly to avoid conflicts
                mermaid_code += "    userQuery[\"User Query\"]\n"
                mermaid_code += "    contextRetrieval[\"Context Retrieval\"]\n"
                
                # Define response nodes
                for i in range(5):
                    mermaid_code += f"    response{i+1}[\"Response {i+1}\"]\n"
                
                # Define voting process node
                mermaid_code += "    votingProcess[\"Voting Process\"]\n"
                
                # Define evaluator nodes
                for i in range(5):
                    mermaid_code += f"    evaluator{i+1}[\"Evaluator {i+1}\"]\n"
                
                # Define results and final response nodes
                mermaid_code += "    results[\"Results\"]\n"
                mermaid_code += "    finalResponse[\"Final Response\"]\n"
                
                # Connect nodes
                mermaid_code += "    userQuery --> contextRetrieval\n"
                
                # Connect context to responses
                for i in range(5):
                    mermaid_code += f"    contextRetrieval --> response{i+1}\n"
                
                # Connect responses to voting process - ensure ALL responses connect to voting process
                for i in range(5):
                    mermaid_code += f"    response{i+1} --> votingProcess\n"
                
                # Connect voting process to evaluators
                for i in range(5):
                    mermaid_code += f"    votingProcess --> evaluator{i+1}\n"
                
                # Connect evaluators to results
                for i in range(5):
                    mermaid_code += f"    evaluator{i+1} --> results\n"
                
                # Connect results to final response
                mermaid_code += "    results --> finalResponse\n"
                
                # Label nodes with actual content (safely escaped)
                message_safe = message.replace('"', "'").replace('\n', ' ')[:30]
                mermaid_code += f"    userQuery[\"User Query:<br/>{message_safe}...\"]\n"
                mermaid_code += f"    contextRetrieval[\"Context Retrieval:<br/>{len(contexts)} chunks\"]\n"
                
                # Add response content
                for i in range(5):
                    resp_text = responses[i].replace('"', "'").replace('\n', ' ')[:30]
                    temp = 0.7 + (i * 0.1)
                    mermaid_code += f"    response{i+1}[\"Response {i+1} (Temp: {temp:.1f}):<br/>{resp_text}...\"]\n"
                
                # Add vote counts to results
                mermaid_code += f"    results[\"Results: {votes}<br/>Winner: Response {winning_index + 1} ({max_votes} votes)\"]\n"
                
                # Add winner content to final response
                final_text = responses[winning_index].replace('"', "'").replace('\n', ' ')[:30]
                mermaid_code += f"    finalResponse[\"Final Response:<br/>{final_text}...\"]\n"
                
                # Style nodes
                mermaid_code += "    class userQuery query;\n"
                mermaid_code += "    class contextRetrieval context;\n"
                mermaid_code += "    class response1,response2,response3,response4,response5 response;\n"
                mermaid_code += "    class evaluator1,evaluator2,evaluator3,evaluator4,evaluator5 evaluator;\n"
                mermaid_code += "    class results results;\n"
                mermaid_code += "    class finalResponse final;\n"
                mermaid_code += "    class votingProcess voting;\n"
                
                # Highlight winning response
                mermaid_code += f"    class response{winning_index + 1} winner;\n"
                
                # Add special connections for votes - only from evaluators to responses
                for i, vote_info in enumerate(vote_details):
                    vote = vote_info.get("vote")
                    if vote is not None:
                        # Use dashed lines for vote connections
                        mermaid_code += f"    evaluator{i+1} -.-> response{vote}\n"
                
                # Add styling
                mermaid_code += "    classDef query fill:#d4f1c5;\n"
                mermaid_code += "    classDef context fill:#fff4b8;\n"
                mermaid_code += "    classDef response fill:#ffcfcf;\n"
                mermaid_code += "    classDef evaluator fill:#ffd7b5;\n"
                mermaid_code += "    classDef results fill:#ffe066;\n"
                mermaid_code += "    classDef final fill:#b3e6cc;\n"
                mermaid_code += "    classDef voting fill:#ffaaaa;\n"
                mermaid_code += "    classDef winner fill:#ffcfcf,stroke:#ff0000,stroke-width:2px;\n"
            except Exception as mermaid_error:
                print(f"Error generating Mermaid code: {str(mermaid_error)}")
                # Provide a simplified fallback diagram
                mermaid_code = """flowchart TD
    A[User Query] --> B[Processing]
    B --> C[Response]
    style C fill:#f9f,stroke:#333,stroke-width:4px"""
            
            return jsonify({
                "mermaid_code": mermaid_code,
                "data": {
                    "responses": responses,
                    "anonymized_responses": anonymized_responses,
                    "response_metadata": response_metadata,
                    "votes": votes,
                    "logs": logs,
                    "contexts": contexts,
                    "vote_details": vote_details
                }
            }), 200
                
    except Exception as e:
        print(f"Error in generate_debug_flowchart: {str(e)}")
        return jsonify({"error": "Failed to generate flowchart", "details": str(e)}), 500

@app.route('/api/bots/<bot_id>/set-chorus', methods=['POST'])
@require_auth
def set_bot_chorus(user_data, bot_id):
    # Check if bot exists and belongs to user
    bots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bots")
    user_bots_file = os.path.join(bots_dir, f"{user_data['id']}_bots.json")
    
    if not os.path.exists(user_bots_file):
        return jsonify({"error": "Bot not found"}), 404
        
    with open(user_bots_file, 'r') as f:
        bots = json.load(f)
        
    bot = None
    bot_index = -1
    for i, b in enumerate(bots):
        if b["id"] == bot_id:
            bot = b
            bot_index = i
            break
            
    if not bot:
        return jsonify({"error": "Bot not found"}), 404
    
    # Get the chorus ID from the request
    data = request.json
    chorus_id = data.get('chorus_id', '')
    
    # Empty string means unassign any chorus
    if chorus_id:
        # Verify that the chorus exists if an ID was provided
        chorus_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chorus")
        user_choruses_file = os.path.join(chorus_dir, f"{user_data['id']}_choruses.json")
        
        if os.path.exists(user_choruses_file):
            with open(user_choruses_file, 'r') as f:
                choruses = json.load(f)
                
            # Find the chorus by ID
            chorus = next((c for c in choruses if c["id"] == chorus_id), None)
            if not chorus:
                return jsonify({"error": f"Chorus with ID {chorus_id} not found"}), 404
    
    # Update the bot with the new chorus ID
    bots[bot_index]["chorus_id"] = chorus_id
    
    # Save the updated bot
    with open(user_bots_file, 'w') as f:
        json.dump(bots, f)
    
    return jsonify({
        "message": chorus_id and "Chorus assigned to bot successfully" or "Chorus unassigned from bot successfully", 
        "bot": bots[bot_index]
    }), 200

@app.route('/api/choruses', methods=['GET'])
@require_auth
def list_choruses(user_data):
    # Get all chorus configurations
    chorus_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chorus")
    os.makedirs(chorus_dir, exist_ok=True)
    
    # First check if there's a user-specific choruses file
    user_choruses_file = os.path.join(chorus_dir, f"{user_data['id']}_choruses.json")
    
    # If the user choruses file doesn't exist yet, create it
    if not os.path.exists(user_choruses_file):
        with open(user_choruses_file, 'w') as f:
            json.dump([], f)
    
    # Load user's chorus definitions
    with open(user_choruses_file, 'r') as f:
        choruses = json.load(f)
    
    return jsonify(choruses), 200

@app.route('/api/choruses', methods=['POST'])
@require_auth
def create_chorus(user_data):
    # Get chorus data from request
    data = request.json
    
    # Basic validation
    if not data.get('name'):
        return jsonify({"error": "Chorus name is required"}), 400
        
    if not data.get('response_models') or not isinstance(data.get('response_models'), list) or len(data.get('response_models')) == 0:
        return jsonify({"error": "At least one response model is required"}), 400
        
    if not data.get('evaluator_models') or not isinstance(data.get('evaluator_models'), list) or len(data.get('evaluator_models')) == 0:
        return jsonify({"error": "At least one evaluator model is required"}), 400
    
    # Create chorus directory if it doesn't exist
    chorus_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chorus")
    os.makedirs(chorus_dir, exist_ok=True)
    
    # Load user's chorus definitions
    user_choruses_file = os.path.join(chorus_dir, f"{user_data['id']}_choruses.json")
    if os.path.exists(user_choruses_file):
        with open(user_choruses_file, 'r') as f:
            choruses = json.load(f)
    else:
        choruses = []
    
    # Generate a unique ID for the chorus
    chorus_id = str(uuid.uuid4())
    
    # Add the new chorus with metadata
    chorus = {
        "id": chorus_id,
        "name": data.get('name'),
        "description": data.get('description', ''),
        "created_at": datetime.datetime.utcnow().isoformat(),
        "updated_at": datetime.datetime.utcnow().isoformat(),
        "response_models": data.get('response_models', []),
        "evaluator_models": data.get('evaluator_models', []),
        "created_by": user_data['username']
    }
    
    choruses.append(chorus)
    
    # Save updated chorus list
    with open(user_choruses_file, 'w') as f:
        json.dump(choruses, f)
    
    return jsonify(chorus), 201

@app.route('/api/choruses/<chorus_id>', methods=['GET'])
@require_auth
def get_chorus(user_data, chorus_id):
    # Get chorus directory and user choruses file
    chorus_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chorus")
    user_choruses_file = os.path.join(chorus_dir, f"{user_data['id']}_choruses.json")
    
    if not os.path.exists(user_choruses_file):
        return jsonify({"error": "Chorus not found"}), 404
    
    # Load user's chorus definitions
    with open(user_choruses_file, 'r') as f:
        choruses = json.load(f)
    
    # Find the requested chorus
    chorus = next((c for c in choruses if c["id"] == chorus_id), None)
    if not chorus:
        return jsonify({"error": "Chorus not found"}), 404
    
    return jsonify(chorus), 200

@app.route('/api/choruses/<chorus_id>', methods=['PUT'])
@require_auth
def update_chorus(user_data, chorus_id):
    # Get chorus data from request
    data = request.json
    
    # Basic validation
    if not data.get('name'):
        return jsonify({"error": "Chorus name is required"}), 400
        
    if not data.get('response_models') or not isinstance(data.get('response_models'), list) or len(data.get('response_models')) == 0:
        return jsonify({"error": "At least one response model is required"}), 400
        
    if not data.get('evaluator_models') or not isinstance(data.get('evaluator_models'), list) or len(data.get('evaluator_models')) == 0:
        return jsonify({"error": "At least one evaluator model is required"}), 400
    
    # Get chorus directory and user choruses file
    chorus_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chorus")
    user_choruses_file = os.path.join(chorus_dir, f"{user_data['id']}_choruses.json")
    
    if not os.path.exists(user_choruses_file):
        return jsonify({"error": "Chorus not found"}), 404
    
    # Load user's chorus definitions
    with open(user_choruses_file, 'r') as f:
        choruses = json.load(f)
    
    # Find and update the chorus
    chorus_index = next((i for i, c in enumerate(choruses) if c["id"] == chorus_id), None)
    if chorus_index is None:
        return jsonify({"error": "Chorus not found"}), 404
    
    # Update the chorus with new data, preserving the ID and creation date
    chorus = choruses[chorus_index]
    updated_chorus = {
        "id": chorus["id"],
        "name": data.get('name'),
        "description": data.get('description', ''),
        "created_at": chorus["created_at"],
        "updated_at": datetime.datetime.utcnow().isoformat(),
        "response_models": data.get('response_models', []),
        "evaluator_models": data.get('evaluator_models', []),
        "created_by": chorus.get("created_by", user_data['username'])
    }
    
    choruses[chorus_index] = updated_chorus
    
    # Save updated chorus list
    with open(user_choruses_file, 'w') as f:
        json.dump(choruses, f)
    
    return jsonify(updated_chorus), 200

@app.route('/api/choruses/<chorus_id>', methods=['DELETE'])
@require_auth
def delete_chorus(user_data, chorus_id):
    # Get chorus directory and user choruses file
    chorus_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chorus")
    user_choruses_file = os.path.join(chorus_dir, f"{user_data['id']}_choruses.json")
    
    if not os.path.exists(user_choruses_file):
        return jsonify({"error": "Chorus not found"}), 404
    
    # Load user's chorus definitions
    with open(user_choruses_file, 'r') as f:
        choruses = json.load(f)
    
    # Find and remove the chorus
    original_length = len(choruses)
    choruses = [c for c in choruses if c["id"] != chorus_id]
    
    if len(choruses) == original_length:
        return jsonify({"error": "Chorus not found"}), 404
    
    # Save updated chorus list
    with open(user_choruses_file, 'w') as f:
        json.dump(choruses, f)
    
    # Also check if any bots are using this chorus and update them
    bots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bots")
    user_bots_file = os.path.join(bots_dir, f"{user_data['id']}_bots.json")
    
    if os.path.exists(user_bots_file):
        with open(user_bots_file, 'r') as f:
            bots = json.load(f)
        
        # Update any bots using this chorus
        updated = False
        for bot in bots:
            if bot.get("chorus_id") == chorus_id:
                bot["chorus_id"] = ""
                updated = True
        
        # Save updated bots if needed
        if updated:
            with open(user_bots_file, 'w') as f:
                json.dump(bots, f)
    
    return jsonify({"message": "Chorus deleted successfully"}), 200

# Bots routes
@app.route('/api/bots', methods=['GET'])
@require_auth
def get_bots(user_data):
    # Get bots directory and user bots file
    bots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bots")
    os.makedirs(bots_dir, exist_ok=True)
    user_bots_file = os.path.join(bots_dir, f"{user_data['id']}_bots.json")
    
    # If user doesn't have any bots yet, return empty list
    if not os.path.exists(user_bots_file):
        return jsonify([]), 200
        
    # Return user's bots
    with open(user_bots_file, 'r') as f:
        bots = json.load(f)
    
    return jsonify(bots), 200

@app.route('/api/bots', methods=['POST'])
@require_auth
def create_bot(user_data):
    # Get bot data from request
    data = request.json
    
    if not data or not data.get('name'):
        return jsonify({"error": "Bot name is required"}), 400
    
    # Process dataset IDs - can be added later or supplied in the request
    dataset_ids = []
    if data.get('dataset_id'):  # Support single dataset_id in request for backward compatibility
        dataset_ids.append(data.get('dataset_id'))
    if data.get('dataset_ids') and isinstance(data.get('dataset_ids'), list):
        dataset_ids.extend(data.get('dataset_ids'))
    
    # Get chorus_id if provided
    chorus_id = data.get('chorus_id', '')
    
    # Verify that the chorus exists if one was provided
    if chorus_id:
        chorus_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chorus")
        user_choruses_file = os.path.join(chorus_dir, f"{user_data['id']}_choruses.json")
        
        if os.path.exists(user_choruses_file):
            with open(user_choruses_file, 'r') as f:
                choruses = json.load(f)
                
            # Find the chorus by ID
            chorus = next((c for c in choruses if c["id"] == chorus_id), None)
            if not chorus:
                return jsonify({"error": f"Chorus with ID {chorus_id} not found"}), 400
    
    # Create a new bot
    new_bot = {
        "id": str(uuid.uuid4()),
        "name": data.get('name'),
        "description": data.get('description', ''),
        "dataset_ids": dataset_ids,
        "chorus_id": chorus_id,  # Set the chorus ID
        "prompt_template": data.get('prompt_template', ''),
        "system_instruction": data.get('system_instruction', 'You are a helpful AI assistant. Answer questions based on the provided context.'),
        "created_at": datetime.datetime.utcnow().isoformat()
    }
    
    # Save the bot
    bots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bots")
    os.makedirs(bots_dir, exist_ok=True)
    
    # Check if user already has bots
    user_bots_file = os.path.join(bots_dir, f"{user_data['id']}_bots.json")
    
    if os.path.exists(user_bots_file):
        with open(user_bots_file, 'r') as f:
            bots = json.load(f)
        bots.append(new_bot)
    else:
        bots = [new_bot]
    
    with open(user_bots_file, 'w') as f:
        json.dump(bots, f)
    
    return jsonify(new_bot), 201

@app.route('/api/bots/<bot_id>', methods=['DELETE'])
@require_auth
def delete_bot(user_data, bot_id):
    # Get bots directory and user bots file
    bots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bots")
    user_bots_file = os.path.join(bots_dir, f"{user_data['id']}_bots.json")
    
    if not os.path.exists(user_bots_file):
        return jsonify({"error": "Bot not found"}), 404
        
    with open(user_bots_file, 'r') as f:
        bots = json.load(f)
    
    # Find and remove the bot
    bot_found = False
    for i, bot in enumerate(bots):
        if bot["id"] == bot_id:
            bots.pop(i)
            bot_found = True
            break
    
    if not bot_found:
        return jsonify({"error": "Bot not found"}), 404
    
    # Save updated bots list
    with open(user_bots_file, 'w') as f:
        json.dump(bots, f)
    
    return jsonify({"message": "Bot deleted successfully"}), 200

# Add new endpoints for managing datasets in a bot
@app.route('/api/bots/<bot_id>/datasets', methods=['GET'])
@require_auth
def get_bot_datasets(user_data, bot_id):
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
    
    # Get dataset info for each dataset ID
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    user_datasets_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")
    
    dataset_details = []
    
    # Get dataset IDs from bot
    dataset_ids = bot.get("dataset_ids", [])
    
    if os.path.exists(user_datasets_file):
        try:
            with open(user_datasets_file, 'r') as f:
                all_datasets = json.load(f)
                
            for dataset_id in dataset_ids:
                dataset_found = False
                for d in all_datasets:
                    if d["id"] == dataset_id:
                        dataset_details.append(d)
                        dataset_found = True
                        break
                
                if not dataset_found:
                    # Add placeholder for missing dataset
                    dataset_details.append({
                        "id": dataset_id,
                        "name": "Unknown dataset",
                        "missing": True
                    })
        except Exception as e:
            print(f"Error loading datasets: {str(e)}")
    
    return jsonify({
        "bot": bot,
        "datasets": dataset_details
    }), 200

@app.route('/api/bots/<bot_id>/datasets', methods=['POST'])
@require_auth
def add_dataset_to_bot(user_data, bot_id):
    data = request.json
    dataset_id = data.get('dataset_id')
    
    if not dataset_id:
        return jsonify({"error": "Dataset ID is required"}), 400
    
    # Check if dataset exists
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    user_datasets_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")
    
    dataset_exists = False
    dataset_name = "Unknown dataset"
    if os.path.exists(user_datasets_file):
        with open(user_datasets_file, 'r') as f:
            datasets = json.load(f)
        
        for d in datasets:
            if d["id"] == dataset_id:
                dataset_exists = True
                dataset_name = d["name"]
                break
    
    if not dataset_exists:
        return jsonify({"error": "Dataset not found"}), 404
    
    # Get bot info
    bots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bots")
    user_bots_file = os.path.join(bots_dir, f"{user_data['id']}_bots.json")
    
    if not os.path.exists(user_bots_file):
        return jsonify({"error": "Bot not found"}), 404
        
    with open(user_bots_file, 'r') as f:
        bots = json.load(f)
        
    bot_index = None
    for i, b in enumerate(bots):
        if b["id"] == bot_id:
            bot_index = i
            break
            
    if bot_index is None:
        return jsonify({"error": "Bot not found"}), 404
    
    # Update bot with the new dataset
    bot = bots[bot_index]
    
    # Set up dataset_ids if it doesn't exist
    if "dataset_ids" not in bot:
        bot["dataset_ids"] = []
        
    # Add dataset if not already added
    if dataset_id not in bot["dataset_ids"]:
        bot["dataset_ids"].append(dataset_id)
        
    # Save updated bot
    bots[bot_index] = bot
    with open(user_bots_file, 'w') as f:
        json.dump(bots, f)
    
    return jsonify({
        "message": f"Dataset '{dataset_name}' added to bot successfully",
        "bot": bot
    }), 200

@app.route('/api/bots/<bot_id>/datasets/<dataset_id>', methods=['DELETE'])
@require_auth
def remove_dataset_from_bot(user_data, bot_id, dataset_id):
    # Get bot info
    bots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bots")
    user_bots_file = os.path.join(bots_dir, f"{user_data['id']}_bots.json")
    
    if not os.path.exists(user_bots_file):
        return jsonify({"error": "Bot not found"}), 404
        
    with open(user_bots_file, 'r') as f:
        bots = json.load(f)
        
    bot_index = None
    for i, b in enumerate(bots):
        if b["id"] == bot_id:
            bot_index = i
            break
            
    if bot_index is None:
        return jsonify({"error": "Bot not found"}), 404
    
    # Update bot to remove the dataset
    bot = bots[bot_index]
    
    # Remove dataset from the dataset_ids array
    dataset_removed = False
    if "dataset_ids" in bot and dataset_id in bot["dataset_ids"]:
        bot["dataset_ids"].remove(dataset_id)
        dataset_removed = True
    
    if not dataset_removed:
        return jsonify({"error": "Dataset not found in bot"}), 404
    
    # Save updated bot
    bots[bot_index] = bot
    with open(user_bots_file, 'w') as f:
        json.dump(bots, f)
    
    return jsonify({
        "message": "Dataset removed from bot successfully",
        "bot": bot
    }), 200

# Image generation route
@app.route('/api/images/generate', methods=['POST'])
@require_auth
def generate_image(user_data):
    try:
        # Get request data
        data = request.json
        prompt = data.get('prompt')
        
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
            
        # Get additional parameters with defaults for gpt-image-1
        model = "gpt-image-1"  # Always use gpt-image-1
        size = data.get('size', '1024x1024')  # Default square
        quality = data.get('quality', 'auto')  # 'low', 'medium', 'high', or 'auto'
        n = data.get('n', 1)  # Number of images to generate
        output_format = data.get('output_format', 'png')  # 'png', 'jpeg', or 'webp'
        
        # Build the API request params with only supported parameters
        generation_params = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
            "quality": quality
        }
        
        # Generate image with OpenAI
        response = openai.images.generate(**generation_params)
        
        results = []
        # Process all generated images
        for i, image_obj in enumerate(response.data):
            # Get the base64 content from OpenAI response
            if hasattr(image_obj, 'b64_json') and image_obj.b64_json:
                # Decode base64 content
                image_content = base64.b64decode(image_obj.b64_json)
                image_url = None
            elif hasattr(image_obj, 'url') and image_obj.url:
                # For backward compatibility if URL is provided
                image_url = image_obj.url
                
                # Download the image
                image_response = requests.get(image_url)
                if image_response.status_code != 200:
                    return jsonify({"error": f"Failed to download generated image {i+1}"}), 500
                    
                image_content = image_response.content
            else:
                return jsonify({"error": f"No image data found in response for image {i+1}"}), 500
                
            # Create a unique filename
            ext = output_format if output_format else "png"  # Default to png
            filename = f"generated_{str(uuid.uuid4())}.{ext}"
            filepath = os.path.join(IMAGE_FOLDER, filename)
            
            # Save the image
            with open(filepath, 'wb') as f:
                f.write(image_content)
                
            # Create the URL for accessing the image
            api_image_url = f"/api/images/{filename}"
            
            # Add to results
            results.append({
                "image_url": api_image_url,
                "original_url": image_url,
                "params": {
                    "model": model,
                    "size": size,
                    "quality": quality,
                    "format": output_format
                }
            })
        
        return jsonify({
            "success": True,
            "images": results
        })
        
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return jsonify({"error": f"Image generation failed: {str(e)}"}), 500

# Route to serve generated images
@app.route('/api/images/<filename>', methods=['GET'])
def get_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
