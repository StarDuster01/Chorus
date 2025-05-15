import os
import json
import uuid
import datetime
from datetime import UTC
import tempfile
from werkzeug.utils import secure_filename
from flask import request, jsonify, send_file
import chromadb
from chromadb.utils import embedding_functions
import faiss
import numpy as np
import io

from text_extractors import (
    extract_text_from_image,
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_pptx,
    extract_text_from_file,
    create_semantic_chunks,
    chunk_powerpoint_content
)

# Helper functions for dataset operations
def find_dataset_by_id(dataset_id):
    """Find a dataset by its ID across all users
    
    Args:
        dataset_id: The ID of the dataset to find
        
    Returns:
        dict: The dataset if found, None otherwise
    """
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    if not os.path.exists(datasets_dir):
        return None
    
    # Check all user dataset files
    for filename in os.listdir(datasets_dir):
        if filename.endswith("_datasets.json"):
            try:
                with open(os.path.join(datasets_dir, filename), 'r') as f:
                    datasets = json.load(f)
                
                for dataset in datasets:
                    if dataset.get("id") == dataset_id:
                        return dataset
            except Exception as e:
                print(f"Error reading dataset file {filename}: {str(e)}")
    
    return None

def sync_datasets_with_collections(chroma_client, openai_ef):
    """Syncs datasets with ChromaDB collections to ensure consistency
    
    Args:
        chroma_client: The ChromaDB client
        openai_ef: OpenAI embedding function
    """
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

def get_mime_types_for_dataset(dataset_type):
    """Get MIME types for dataset based on its type
    
    Args:
        dataset_type: The type of dataset (text, image, etc.)
    
    Returns:
        list: List of MIME types
    """
    types = {
        "text": ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/plain", "application/vnd.openxmlformats-officedocument.presentationml.presentation"],
        "image": ["image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp"]
    }
    return types.get(dataset_type, [])

# Dataset handler functions for routes
def get_datasets_handler(user_data):
    """Get all datasets for a user
    
    Args:
        user_data: User data from JWT token
    
    Returns:
        tuple: JSON response and status code
    """
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    user_datasets_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")
    
    if not os.path.exists(user_datasets_file):
        return jsonify([]), 200
        
    with open(user_datasets_file, 'r') as f:
        datasets = json.load(f)
    
    # Import the ImageProcessor here to avoid circular imports
    from image_processor import ImageProcessor
    
    # Initialize ImageProcessor
    image_processor = ImageProcessor(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    
    for dataset in datasets:
        # Update the dataset with document counts
        dataset["document_count"] = dataset.get("document_count", 0)
        
        # Update image counts from ImageProcessor
        dataset_id = dataset["id"]
        if dataset_id in image_processor.image_metadata:
            dataset["image_count"] = len(image_processor.image_metadata[dataset_id])
        else:
            dataset["image_count"] = 0
    
    return jsonify(datasets), 200

def create_dataset_handler(user_data):
    """Create a new dataset
    
    Args:
        user_data: User data from JWT token
    
    Returns:
        tuple: JSON response and status code
    """
    data = request.json
    
    if not data or not data.get('name'):
        return jsonify({"error": "Dataset name is required"}), 400
        
    # Get dataset type, default to "text"
    dataset_type = data.get('type', 'text')
    if dataset_type not in ['text', 'image']:
        return jsonify({"error": "Invalid dataset type. Must be 'text' or 'image'"}), 400
        
    # Create a new dataset
    dataset_id = str(uuid.uuid4())
    new_dataset = {
        "id": dataset_id,
        "name": data.get('name'),
        "description": data.get('description', ''),
        "type": dataset_type,
        "documents": [],
        "images": [],
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
        chroma_client.create_collection(name=dataset_id, embedding_function=openai_ef)
    except Exception as e:
        print(f"Error creating ChromaDB collection: {str(e)}")
    
    return jsonify(new_dataset), 201

def delete_dataset_handler(user_data, dataset_id):
    """Delete a dataset
    
    Args:
        user_data: User data from JWT token
        dataset_id: ID of the dataset to delete
    
    Returns:
        tuple: JSON response and status code
    """
    # Check if dataset exists and belongs to user
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    user_datasets_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")
    
    if not os.path.exists(user_datasets_file):
        return jsonify({"error": "Dataset not found"}), 404
        
    with open(user_datasets_file, 'r') as f:
        datasets = json.load(f)
    
    # Find and remove the dataset
    dataset_found = False
    dataset_to_delete = None
    for i, dataset in enumerate(datasets):
        if dataset["id"] == dataset_id:
            dataset_to_delete = datasets.pop(i)
            dataset_found = True
            break
    
    if not dataset_found:
        return jsonify({"error": "Dataset not found"}), 404
    
    # Save updated datasets list
    with open(user_datasets_file, 'w') as f:
        json.dump(datasets, f)
    
    # Delete documents from disk
    if dataset_to_delete and dataset_to_delete.get("documents"):
        DOCUMENT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads", "documents")
        for doc in dataset_to_delete["documents"]:
            doc_path = os.path.join(DOCUMENT_FOLDER, doc["filename"])
            try:
                if os.path.exists(doc_path):
                    os.remove(doc_path)
            except Exception as e:
                print(f"Error deleting document {doc['filename']}: {str(e)}")
    
    # Delete images from disk
    if dataset_to_delete and dataset_to_delete.get("images"):
        IMAGE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads", "images")
        for img in dataset_to_delete["images"]:
            img_path = os.path.join(IMAGE_FOLDER, img["filename"])
            try:
                if os.path.exists(img_path):
                    os.remove(img_path)
            except Exception as e:
                print(f"Error deleting image {img['filename']}: {str(e)}")
    
    # Delete ChromaDB collection
    try:
        chroma_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
        chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        try:
            chroma_client.delete_collection(name=dataset_id)
        except Exception as e:
            print(f"Error deleting ChromaDB collection: {str(e)}")
    except Exception as e:
        print(f"Error connecting to ChromaDB: {str(e)}")
    
    # Also remove the dataset from any bots that use it
    bots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bots")
    if os.path.exists(bots_dir):
        for filename in os.listdir(bots_dir):
            if filename.endswith("_bots.json"):
                try:
                    bots_file_path = os.path.join(bots_dir, filename)
                    with open(bots_file_path, 'r') as f:
                        bots = json.load(f)
                    
                    updated = False
                    for bot in bots:
                        if "dataset_ids" in bot and dataset_id in bot["dataset_ids"]:
                            bot["dataset_ids"].remove(dataset_id)
                            updated = True
                    
                    if updated:
                        with open(bots_file_path, 'w') as f:
                            json.dump(bots, f)
                except Exception as e:
                    print(f"Error updating bots file {filename}: {str(e)}")
    
    return jsonify({"message": "Dataset deleted successfully"}), 200

def get_dataset_type_handler(user_data, dataset_id):
    """Get the type of a dataset (text, image, or mixed) to inform frontend file selection"""
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
    
    # Get dataset type
    dataset_type = dataset.get("type", "text")
    
    # Get appropriate MIME types and extensions for this dataset type
    mime_types = get_mime_types_for_dataset(dataset_type)
    mime_types.extend(get_mime_types_for_dataset("mixed"))  # Add mixed types for flexibility
    
    supported_extensions = []
    if dataset_type in ["text", "mixed"]:
        supported_extensions.extend([".pdf", ".docx", ".txt", ".pptx"])
    if dataset_type in ["image", "mixed"]:
        supported_extensions.extend([".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"])
    
    return jsonify({
        "type": dataset_type,
        "mime_types": ",".join(mime_types),
        "supported_extensions": supported_extensions
    }), 200

def remove_document_handler(user_data, dataset_id, document_id):
    """Remove a document from a dataset
    
    Args:
        user_data: User data from JWT token
        dataset_id: ID of the dataset
        document_id: ID of the document to remove
        
    Returns:
        tuple: JSON response and status code
    """
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
        chroma_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
        chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-ada-002"
        )
        collection = chroma_client.get_collection(name=dataset_id, embedding_function=openai_ef)
        
        # Get all chunks with the matching document_id in their metadata
        results = collection.get(
            where={"document_id": document_id}
        )
        
        if not results or len(results['ids']) == 0:
            return jsonify({"error": "Document not found in dataset"}), 404
        
        # Get number of chunks to remove for updating chunk count
        num_chunks_to_remove = len(results['ids'])
        
        # Delete the chunks from ChromaDB
        collection.delete(
            ids=results['ids']
        )
        
        # Update document count and chunk count in dataset
        if datasets[dataset_index]["document_count"] > 0:
            datasets[dataset_index]["document_count"] -= 1
            
        # Update chunk count if it exists
        if "chunk_count" in datasets[dataset_index]:
            datasets[dataset_index]["chunk_count"] -= num_chunks_to_remove
            if datasets[dataset_index]["chunk_count"] < 0:
                datasets[dataset_index]["chunk_count"] = 0
            
        with open(user_datasets_file, 'w') as f:
            json.dump(datasets, f)
            
        return jsonify({
            "message": "Document removed successfully",
            "chunks_removed": num_chunks_to_remove
        }), 200
    
    except Exception as e:
        print(f"Error removing document: {str(e)}")
        return jsonify({"error": f"Failed to remove document: {str(e)}"}), 500

def rebuild_dataset_handler(user_data, dataset_id):
    """Rebuild a dataset collection in ChromaDB
    
    Args:
        user_data: User data from JWT token
        dataset_id: ID of the dataset to rebuild
    
    Returns:
        tuple: JSON response and status code
    """
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
        chroma_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
        chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        existing_collections = chroma_client.list_collections()
        if dataset_id in existing_collections:
            chroma_client.delete_collection(name=dataset_id)
            print(f"Deleted existing collection for dataset: {dataset_id}")
    except Exception as e:
        print(f"Error deleting collection: {str(e)}")
    
    # Create a new collection
    try:
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-ada-002"
        )
        chroma_client.create_collection(name=dataset_id, embedding_function=openai_ef)
        return jsonify({"message": "Dataset collection has been rebuilt. Please re-upload your documents."}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to rebuild dataset: {str(e)}"}), 500

def upload_image_handler(user_data, dataset_id, image_folder):
    """Upload an image to a dataset using ImageProcessor
    
    Args:
        user_data: User data from JWT token
        dataset_id: ID of the dataset to upload to
        image_folder: Path to image storage folder
        
    Returns:
        tuple: JSON response and status code
    """
    from flask import request, jsonify
    from werkzeug.utils import secure_filename
    import uuid
    import base64
    import os
    
    # Import resize_image from image_handlers
    from image_handlers import resize_image
    
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
    
    # Handle both form data and JSON requests
    if request.content_type and request.content_type.startswith('multipart/form-data'):
        # Handle image file from form data
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
            
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"error": "No image selected"}), 400
            
        # Save the image
        filename = secure_filename(image_file.filename)
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(filename)[1].lower()
        new_filename = f"{file_id}{file_extension}"
        image_path = os.path.join(image_folder, new_filename)
        
        try:
            image_file.save(image_path)
            print(f"Saved image to {image_path}")
        except Exception as e:
            return jsonify({"error": f"Failed to save image: {str(e)}"}), 500
        
        # Resize image if needed
        image_path = resize_image(image_path, max_dimension=1024)
        
    else:
        # Handle base64 image data from JSON
        data = request.json
        if not data or 'image_data' not in data:
            return jsonify({"error": "Image data is required"}), 400
            
        image_data = data['image_data']
        filename = data.get('filename', f"image_{uuid.uuid4()}.jpg")
        
        # Extract the base64 content
        try:
            if ';base64,' in image_data:
                header, encoded = image_data.split(';base64,')
                file_ext = header.split('/')[-1]
                if not file_ext:
                    file_ext = 'jpg'
            else:
                encoded = image_data
                file_ext = 'jpg'
                
            # Generate a unique filename
            file_id = str(uuid.uuid4())
            new_filename = f"{file_id}.{file_ext}"
            
            # Save the image
            image_path = os.path.join(image_folder, new_filename)
            with open(image_path, 'wb') as f:
                f.write(base64.b64decode(encoded))
                
            print(f"Saved base64 image to {image_path}")
                
        except Exception as e:
            return jsonify({"error": f"Failed to process image data: {str(e)}"}), 400
    
    # Get additional metadata
    custom_metadata = {
        "id": file_id,
        "dataset_id": dataset_id,
        "original_filename": filename,
        "path": image_path,
        "url": f"/api/images/{os.path.basename(image_path)}",
        "type": "image",
        "created_at": datetime.datetime.now(UTC).isoformat()
    }
    
    if request.content_type and request.content_type.startswith('multipart/form-data'):
        # Get metadata from form fields
        custom_metadata['description'] = request.form.get('description', '')
        custom_metadata['tags'] = request.form.get('tags', '').split(',') if request.form.get('tags') else []
    else:
        # Get metadata from JSON
        custom_metadata['description'] = data.get('description', '')
        custom_metadata['tags'] = data.get('tags', [])
    
    # Add user information to metadata
    custom_metadata['user_id'] = user_data['id']
    custom_metadata['username'] = user_data['username']
    
    try:
        # Access the global image_processor from app.py
        from app import image_processor
        
        # Add image to dataset in image processor
        image_metadata = image_processor.add_image_to_dataset(dataset_id, image_path, custom_metadata)
        
        # Let's double check that the image was added to the image processor
        if dataset_id in image_processor.image_metadata:
            is_indexed = False
            for img in image_processor.image_metadata[dataset_id]:
                if img.get('id') == image_metadata['id']:
                    is_indexed = True
                    break
            
            if not is_indexed:
                print(f"Warning: Image not found in image_processor metadata after add_image_to_dataset call")
                # Add it manually if the processor didn't do it
                if image_metadata not in image_processor.image_metadata[dataset_id]:
                    image_processor.image_metadata[dataset_id].append(image_metadata)
        else:
            print(f"Warning: Dataset ID {dataset_id} not in image_processor metadata after adding")
            # Initialize the dataset in the processor
            image_processor.image_metadata[dataset_id] = [image_metadata]
            
        # Generate URL for the image
        image_url = f"/api/images/{os.path.basename(image_path)}"
        
        # Update image count in dataset
        current_count = datasets[dataset_index].get("image_count", 0)
        datasets[dataset_index]["image_count"] = current_count + 1
        print(f"Updated image count for dataset {dataset_id} from {current_count} to {current_count + 1}")
        
        with open(user_datasets_file, 'w') as f:
            json.dump(datasets, f)
        
        # Also save the metadata file directly to ensure it persists
        indices_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "indices")
        os.makedirs(indices_dir, exist_ok=True)
        metadata_file = os.path.join(indices_dir, f"{dataset_id}_metadata.json")
        
        # If metadata file exists, append to it, otherwise create it
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    existing_metadata = json.load(f)
                # Append new image
                existing_metadata.append(image_metadata)
                # Write back
                with open(metadata_file, 'w') as f:
                    json.dump(existing_metadata, f)
            except Exception as e:
                print(f"Error updating metadata file: {str(e)}")
                # Create new file
                with open(metadata_file, 'w') as f:
                    json.dump([image_metadata], f)
        else:
            # Create new file
            with open(metadata_file, 'w') as f:
                json.dump([image_metadata], f)
        
        return jsonify({
            "message": "Image uploaded and processed successfully",
            "image": {
                "id": image_metadata["id"],
                "filename": os.path.basename(image_path),
                "url": image_url,
                "caption": image_metadata.get("caption", ""),
                "description": custom_metadata.get("description", ""),
                "tags": custom_metadata.get("tags", [])
            }
        }), 200
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
            except:
                pass
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

# More handlers can be added here for specific dataset operations 