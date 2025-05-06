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

# Bot routes
@app.route('/api/bots', methods=['GET'])
@require_auth
def get_bots(user_data):
    bots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bots")
    os.makedirs(bots_dir, exist_ok=True)
    user_bots_file = os.path.join(bots_dir, f"{user_data['id']}_bots.json")
    
    if not os.path.exists(user_bots_file):
        return jsonify([]), 200
        
    with open(user_bots_file, 'r') as f:
        bots = json.load(f)
        
    return jsonify(bots), 200

@app.route('/api/bots', methods=['POST'])
@require_auth
def create_bot(user_data):
    data = request.json
    name = data.get('name')
    dataset_id = data.get('dataset_id')
    system_instruction = data.get('system_instruction', 'You are a helpful AI assistant.')
    
    if not name or not dataset_id:
        return jsonify({"error": "Bot name and dataset ID are required"}), 400
        
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
        
    bot_id = str(uuid.uuid4())
    
    # Save bot metadata
    bots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bots")
    os.makedirs(bots_dir, exist_ok=True)
    user_bots_file = os.path.join(bots_dir, f"{user_data['id']}_bots.json")
    
    bots = []
    if os.path.exists(user_bots_file):
        with open(user_bots_file, 'r') as f:
            bots = json.load(f)
            
    bot = {
        "id": bot_id,
        "name": name,
        "dataset_id": dataset_id,
        "system_instruction": system_instruction,
        "created_at": datetime.datetime.utcnow().isoformat()
    }
    bots.append(bot)
    
    with open(user_bots_file, 'w') as f:
        json.dump(bots, f)
        
    return jsonify(bot), 201

# Chat routes
@app.route('/api/bots/<bot_id>/chat', methods=['POST'])
@require_auth
def chat_with_bot(user_data, bot_id):
    data = request.json
    message = data.get('message')
    debug_mode = data.get('debug_mode', False)
    
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
            print(f"Error: Collection '{dataset_id}' not found. Trying to recreate it...")
            # Try to recreate the collection
            try:
                collection = chroma_client.create_collection(name=dataset_id, embedding_function=openai_ef)
                return jsonify({
                    "response": f"I've encountered an issue with my knowledge base. The collection was missing but has been recreated. Please upload documents to the dataset and try again.",
                    "debug": {
                        "error": f"Collection {dataset_id} was missing and has been recreated. Please upload documents."
                    }
                }), 200
            except Exception as create_error:
                raise Exception(f"Failed to create collection: {str(create_error)}")
                
        # Check if collection has any documents
        collection_count = collection.count()
        if collection_count == 0:
            return jsonify({
                "response": "I don't have any documents in my knowledge base yet. Please upload some documents to help me answer your questions.",
                "debug": {"error": "Empty collection"} if debug_mode else None
            }), 200
                
        # Determine how many results to request based on collection size
        n_results = min(5, collection_count)
        
        results = collection.query(
            query_texts=[message],
            n_results=n_results
        )
        
        contexts = results["documents"][0]
        
        # Prepare prompt
        system_instruction = bot["system_instruction"]
        prompt = f"{system_instruction}\n\nRelevant context:\n"
        
        for ctx in contexts:
            prompt += f"{ctx}\n\n"
            
        prompt += f"User: {message}\nAssistant:"
        
        if debug_mode:
            # In debug mode, generate 5 responses and let them vote
            responses = []
            logs = ["Debug mode enabled - generating 5 different responses"]
            
            for i in range(5):
                logs.append(f"Generating response {i+1}/5...")
                response = openai.chat.completions.create(
                    model="gpt-4",  # Using GPT-4 for higher quality responses
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": f"Context:\n{contexts}\n\nUser question: {message}"}
                    ],
                    temperature=0.7 + (i * 0.1),  # Vary temperature to get different responses
                )
                response_text = response.choices[0].message.content
                responses.append(response_text)
                logs.append(f"Response {i+1}: {response_text[:100]}...")
            
            # Have the responses vote on each other
            logs.append("Having the responses evaluate each other...")
            votes = [0, 0, 0, 0, 0]
            vote_details = []
            
            for i in range(5):
                voting_prompt = f"""You are an expert evaluator. You need to rank the following responses to the question: "{message}"
                
Here are the 5 candidate responses:

Response 1:
{responses[0]}

Response 2:
{responses[1]}

Response 3:
{responses[2]}

Response 4:
{responses[3]}

Response 5:
{responses[4]}

Which response provides the most accurate, helpful, and relevant answer? Return ONLY the number (1-5) of the best response.
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
            
            return jsonify({
                "response": responses[winning_index],
                "debug": {
                    "all_responses": responses,
                    "votes": votes,
                    "logs": logs,
                    "contexts": contexts,
                    "vote_details": vote_details
                }
            }), 200
        else:
            # Standard mode - just get one response
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": f"Context:\n{contexts}\n\nUser question: {message}"}
                ]
            )
            
            return jsonify({"response": response.choices[0].message.content}), 200
            
    except Exception as e:
        print(f"Error in chat_with_bot: {str(e)}")
        return jsonify({
            "response": "I apologize, but I encountered an error while processing your request. Please try again or contact support if the issue persists.",
            "debug": {"error": str(e)} if debug_mode else None
        }), 200  # Return 200 with error message to ensure frontend can handle it

@app.route('/api/bots/<bot_id>/chat-with-image', methods=['POST'])
@require_auth
def chat_with_image(user_data, bot_id):
    message = request.form.get('message', '')
    debug_mode = request.form.get('debug_mode', 'false').lower() == 'true'
    
    if 'image' not in request.files:
        return jsonify({"error": "Image is required"}), 400
        
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No image selected"}), 400
        
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
    
    try:
        # Save the image to a temporary file
        filename = secure_filename(image_file.filename)
        temp_image_path = os.path.join(IMAGE_FOLDER, f"{uuid.uuid4()}_{filename}")
        image_file.save(temp_image_path)
        
        # Check image file size (OpenAI limit is 20MB)
        file_size_mb = os.path.getsize(temp_image_path) / (1024 * 1024)
        if file_size_mb > 20:
            os.remove(temp_image_path)
            return jsonify({"error": "Image size exceeds 20MB limit"}), 400
            
        # Process the image if needed (resize large images)
        img_path = temp_image_path
        try:
            img = Image.open(temp_image_path)
            width, height = img.size
            
            # Resize if image is larger than 2048px in any dimension
            if width > 2048 or height > 2048:
                img_path = resize_image(temp_image_path)
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            # Continue with original image if processing fails
        
        # Encode the image to base64
        with open(img_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode('utf-8')
            
        # Clean up resized image if it exists and is different from original
        if img_path != temp_image_path and os.path.exists(img_path):
            try:
                os.remove(img_path)
            except:
                pass
        
        # Prepare the system instruction
        system_instruction = bot["system_instruction"]
        
        # Query ChromaDB for context if dataset has documents
        dataset_id = bot["dataset_id"]
        contexts = []
        
        try:
            collection = chroma_client.get_collection(name=dataset_id, embedding_function=openai_ef)
            # Only query if there are documents and we have a text message
            collection_count = collection.count()
            if collection_count > 0 and message.strip():
                # Determine how many results to request based on collection size
                n_results = min(5, collection_count)
                
                results = collection.query(
                    query_texts=[message],
                    n_results=n_results
                )
                contexts = results["documents"][0]
        except Exception as e:
            print(f"Error querying collection: {str(e)}")
            # Continue without context if there's an error
        
        # Determine if we need high detail based on the message content
        # For simple color/shape identification, low detail is sufficient
        detail_level = "low" if len(message) < 20 and any(keyword in message.lower() for keyword in ["color", "shape", "what is this"]) else "high"
        
        # Create content array with text and image
        content = [
            {"type": "text", "text": f"User message: {message}\n\n" + (f"Context from knowledge base:\n{contexts}" if contexts else "")}
        ]
        
        # Add the image to the content
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": detail_level  # Dynamically choose detail level to optimize token usage
            }
        })
        
        # Make the API call to OpenAI with vision capability
        response = openai.chat.completions.create(
            model="gpt-4o",  # Updated to use GPT-4o which has vision capabilities
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": content}
            ],
            max_tokens=1024
        )
        
        # Clean up the temporary image file
        try:
            os.remove(temp_image_path)
        except:
            pass
            
        return jsonify({
            "response": response.choices[0].message.content,
            "debug": {
                "contexts": contexts,
                "image_processed": img_path != temp_image_path,
                "detail_level": detail_level
            } if debug_mode else None
        }), 200
        
    except Exception as e:
        # Clean up the temporary image file in case of error
        try:
            if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            if 'img_path' in locals() and img_path != temp_image_path and os.path.exists(img_path):
                os.remove(img_path)
        except:
            pass
            
        print(f"Error processing image: {str(e)}")
        return jsonify({
            "response": "I'm sorry, I encountered a problem processing your image. Please try again or use a different image.",
            "debug": {"error": str(e)} if debug_mode else None
        }), 200  # Return 200 with error message for better frontend handling

# Image generation routes
@app.route('/api/images/generate', methods=['POST'])
@require_auth
def generate_image(user_data):
    data = request.json
    prompt = data.get('prompt')
    size = data.get('size', '1024x1024')
    quality = data.get('quality', 'standard')
    model = data.get('model', 'dall-e-3')
    style = data.get('style', 'vivid')
    format = data.get('format', 'url')
    
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
        
    try:
        response = openai.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
            style=style,
            response_format=format,
            n=1,
        )
        
        if format == 'url':
            image_url = response.data[0].url
            # Download and save the image for local access
            import requests
            image_data = requests.get(image_url).content
            image_filename = f"{uuid.uuid4()}.png"
            image_path = os.path.join(IMAGE_FOLDER, image_filename)
            
            with open(image_path, 'wb') as f:
                f.write(image_data)
                
            return jsonify({
                "image_url": f"/api/images/{image_filename}",
                "filename": image_filename
            }), 200
        else:  # Base64
            image_data = response.data[0].b64_json
            image_filename = f"{uuid.uuid4()}.png"
            image_path = os.path.join(IMAGE_FOLDER, image_filename)
            
            with open(image_path, 'wb') as f:
                f.write(base64.b64decode(image_data))
                
            return jsonify({
                "image_url": f"/api/images/{image_filename}",
                "filename": image_filename
            }), 200
            
    except Exception as e:
        print(f"Image generation error: {str(e)}")
        return jsonify({"error": "Failed to generate image", "details": str(e)}), 500

@app.route('/api/images/edit', methods=['POST'])
@require_auth
def edit_image(user_data):
    if 'image' not in request.files:
        return jsonify({"error": "Image is required"}), 400
        
    image_file = request.files['image']
    prompt = request.form.get('prompt')
    model = request.form.get('model', 'dall-e-3')
    
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
        
    try:
        # Save the uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_image:
            image_file.save(temp_image.name)
            
        # Handle mask if provided
        mask_path = None
        if 'mask' in request.files and request.files['mask'].filename:
            mask_file = request.files['mask']
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_mask:
                mask_file.save(temp_mask.name)
                mask_path = temp_mask.name
        
        # Open the files in binary mode
        with open(temp_image.name, "rb") as image_file:
            if mask_path:
                with open(mask_path, "rb") as mask_file:
                    response = openai.images.edit(
                        model=model,
                        image=image_file,
                        mask=mask_file,
                        prompt=prompt,
                        n=1,
                        size="1024x1024"
                    )
            else:
                response = openai.images.edit(
                    model=model,
                    image=image_file,
                    prompt=prompt,
                    n=1,
                    size="1024x1024"
                )
        
        image_url = response.data[0].url
        
        # Download and save the image for local access
        import requests
        image_data = requests.get(image_url).content
        image_filename = f"{uuid.uuid4()}.png"
        image_path = os.path.join(IMAGE_FOLDER, image_filename)
        
        with open(image_path, 'wb') as f:
            f.write(image_data)
            
        # Clean up temporary files
        os.unlink(temp_image.name)
        if mask_path:
            os.unlink(mask_path)
            
        return jsonify({
            "image_url": f"/api/images/{image_filename}",
            "filename": image_filename
        }), 200
        
    except Exception as e:
        print(f"Image editing error: {str(e)}")
        return jsonify({"error": "Failed to edit image", "details": str(e)}), 500

@app.route('/api/images/<filename>', methods=['GET'])
def get_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

@app.route('/api/bots/<bot_id>/debug-flowchart', methods=['POST'])
@require_auth
def generate_debug_flowchart(user_data, bot_id):
    data = request.json
    message = data.get('message')
    
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
        
        # Prepare prompt
        system_instruction = bot["system_instruction"]
        
        # In debug mode - generate 5 responses and let them vote
        responses = []
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
            logs.append(f"Response {i+1}: {response_text[:100]}...")
            
        # Have the responses vote on each other
        logs.append("Having the responses evaluate each other...")
        votes = [0, 0, 0, 0, 0]
        vote_details = []
        
        for i in range(5):
            voting_prompt = f"""You are an expert evaluator. You need to rank the following responses to the question: "{message}"
            
Here are the 5 candidate responses:

Response 1:
{responses[0]}

Response 2:
{responses[1]}

Response 3:
{responses[2]}

Response 4:
{responses[3]}

Response 5:
{responses[4]}

Which response provides the most accurate, helpful, and relevant answer? Return ONLY the number (1-5) of the best response.
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
                "votes": votes,
                "logs": logs,
                "contexts": contexts,
                "vote_details": vote_details
            }
        }), 200
            
    except Exception as e:
        print(f"Error in generate_debug_flowchart: {str(e)}")
        return jsonify({"error": "Failed to generate flowchart", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
