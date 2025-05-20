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
    
    # Process each dataset to update counts and previews
    updated_datasets = []
    for dataset in datasets:
        # Update the dataset with document counts
        dataset["document_count"] = dataset.get("document_count", 0)
        
        # Update image counts and add image previews from ImageProcessor
        dataset_id = dataset["id"]
        image_count = 0
        image_previews = []
        
        # First check if this dataset has a metadata file
        indices_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "image_indices")
        metadata_file = os.path.join(indices_dir, f"{dataset_id}_metadata.json")
        
        # If metadata file exists, load and validate it
        if os.path.exists(metadata_file):
            try:
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
                image_count = len(valid_metadata)
                
                # Add image previews (up to 4 images)
                preview_count = 0
                for img_meta in valid_metadata:
                    if preview_count >= 4:  # Limit to 4 preview images
                        break
                    
                    if 'path' in img_meta and os.path.exists(img_meta['path']):
                        # Extract just the filename from the path
                        filename = os.path.basename(img_meta["path"])
                        # Create preview info with URL
                        preview_info = {
                            "id": img_meta.get("id", ""),
                            "url": f"/images/{filename}",
                            "caption": img_meta.get("caption", "")
                        }
                        image_previews.append(preview_info)
                        preview_count += 1
                
                # Save the updated metadata back to disk
                try:
                    with open(metadata_file, 'w') as f:
                        json.dump(valid_metadata, f)
                except Exception as e:
                    print(f"Error saving updated metadata for dataset {dataset_id}: {str(e)}")
                    
            except Exception as e:
                print(f"Error processing metadata for dataset {dataset_id}: {str(e)}")
                # Fall back to in-memory metadata if available
                if dataset_id in image_processor.image_metadata:
                    valid_metadata = []
                    for img_meta in image_processor.image_metadata[dataset_id]:
                        if 'path' in img_meta and os.path.exists(img_meta['path']):
                            valid_metadata.append(img_meta)
                    
                    image_processor.image_metadata[dataset_id] = valid_metadata
                    image_count = len(valid_metadata)
                    
                    # Add image previews (up to 4 images)
                    preview_count = 0
                    for img_meta in valid_metadata:
                        if preview_count >= 4:
                            break
                        
                        if 'path' in img_meta:
                            filename = os.path.basename(img_meta["path"])
                            preview_info = {
                                "id": img_meta.get("id", ""),
                                "url": f"/images/{filename}",
                                "caption": img_meta.get("caption", "")
                            }
                            image_previews.append(preview_info)
                            preview_count += 1
        
        # Update dataset with accurate image count and previews
        dataset["image_count"] = image_count
        dataset["image_previews"] = image_previews
        updated_datasets.append(dataset)
    
    # Save updated datasets back to file
    with open(user_datasets_file, 'w') as f:
        json.dump(updated_datasets, f)
    
    return jsonify(updated_datasets), 200

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

    # Document and file cleanup
    DOCUMENT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads", "documents")
    IMAGE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads", "images")
    document_files_deleted = 0
    image_files_deleted = 0
    
    # First try to get document file paths from ChromaDB
    try:
        chroma_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
        chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-ada-002"
        )
        
        # Get the collection for this dataset
        try:
            collection = chroma_client.get_collection(name=dataset_id, embedding_function=openai_ef)
            # Get all metadata from collection
            results = collection.get()
            
            # Extract file paths from metadata and delete files
            if results and results['metadatas'] and len(results['metadatas']) > 0:
                unique_file_paths = set()
                for metadata in results['metadatas']:
                    if metadata and 'file_path' in metadata and metadata['file_path']:
                        unique_file_paths.add(metadata['file_path'])
                
                # Delete all document files
                for file_path in unique_file_paths:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            document_files_deleted += 1
                            print(f"Deleted document file: {file_path}")
                    except Exception as e:
                        print(f"Error deleting document file {file_path}: {str(e)}")
            
            # Now delete the collection itself
            chroma_client.delete_collection(name=dataset_id)
            print(f"Deleted ChromaDB collection for dataset {dataset_id}")
            
        except Exception as e:
            print(f"Error accessing ChromaDB collection: {str(e)}")
            # Try to delete the collection anyway
            try:
                chroma_client.delete_collection(name=dataset_id)
                print(f"Deleted ChromaDB collection for dataset {dataset_id}")
            except Exception as e2:
                print(f"Error deleting ChromaDB collection: {str(e2)}")
    except Exception as e:
        print(f"Error connecting to ChromaDB: {str(e)}")
    
    # Clean up image dataset resources (FAISS index and metadata)
    indices_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "image_indices")
    index_path = os.path.join(indices_dir, f"{dataset_id}_index.faiss")
    metadata_file = os.path.join(indices_dir, f"{dataset_id}_metadata.json")
    
    # Delete image files referenced in metadata file
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                image_metadata = json.load(f)
                
            # Delete all image files referenced in metadata
            for img_meta in image_metadata:
                if 'path' in img_meta and os.path.exists(img_meta['path']):
                    try:
                        os.remove(img_meta['path'])
                        image_files_deleted += 1
                        print(f"Deleted image file: {img_meta['path']}")
                    except Exception as e:
                        print(f"Error deleting image file: {str(e)}")
            
            # Now delete the metadata file itself
            os.remove(metadata_file)
            print(f"Deleted image metadata file for dataset {dataset_id}")
        except Exception as e:
            print(f"Error processing image metadata: {str(e)}")
    
    # Delete FAISS index file if it exists
    if os.path.exists(index_path):
        try:
            os.remove(index_path)
            print(f"Deleted FAISS index for dataset {dataset_id}")
        except Exception as e:
            print(f"Error deleting FAISS index: {str(e)}")
    
    # Try to remove the dataset from image processor memory
    try:
        from app import image_processor
        if hasattr(image_processor, 'image_indices') and dataset_id in image_processor.image_indices:
            del image_processor.image_indices[dataset_id]
            print(f"Removed dataset {dataset_id} from image processor indices")
        
        if hasattr(image_processor, 'image_metadata') and dataset_id in image_processor.image_metadata:
            del image_processor.image_metadata[dataset_id]
            print(f"Removed dataset {dataset_id} from image processor metadata")
    except Exception as e:
        print(f"Error cleaning up image processor resources: {str(e)}")
    
    # Legacy cleanup for older datasets that might still use the documents array
    if dataset_to_delete and dataset_to_delete.get("documents"):
        for doc in dataset_to_delete["documents"]:
            doc_path = os.path.join(DOCUMENT_FOLDER, doc["filename"])
            try:
                if os.path.exists(doc_path):
                    os.remove(doc_path)
                    document_files_deleted += 1
            except Exception as e:
                print(f"Error deleting legacy document {doc['filename']}: {str(e)}")
    
    # Legacy cleanup for older datasets that might still use the images array
    if dataset_to_delete and dataset_to_delete.get("images"):
        for img in dataset_to_delete["images"]:
            img_path = os.path.join(IMAGE_FOLDER, img["filename"])
            try:
                if os.path.exists(img_path):
                    os.remove(img_path)
                    image_files_deleted += 1
            except Exception as e:
                print(f"Error deleting legacy image {img['filename']}: {str(e)}")
    
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
    
    print(f"Deleted dataset {dataset_id} with {document_files_deleted} document files and {image_files_deleted} image files")
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
        
        # Get file paths from metadata before deleting
        file_paths = set()
        if results and results['metadatas'] and len(results['metadatas']) > 0:
            for metadata in results['metadatas']:
                if 'file_path' in metadata and metadata['file_path']:
                    file_paths.add(metadata['file_path'])

        # Delete the chunks from ChromaDB
        collection.delete(
            ids=results['ids']
        )
        
        # Now delete the document files from disk
        deleted_files = []
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_files.append(file_path)
                    print(f"Removed document file: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {str(e)}")

        print(f"Removed {len(deleted_files)} document files")
        
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
            "chunks_removed": num_chunks_to_remove,
            "files_removed": len(deleted_files)
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
        "url": f"/images/{new_filename}",  # Use just the filename for the URL
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
        
        # Update image count in dataset
        if "image_count" not in datasets[dataset_index]:
            datasets[dataset_index]["image_count"] = 0
        datasets[dataset_index]["image_count"] += 1
        
        # Save updated dataset
        with open(user_datasets_file, 'w') as f:
            json.dump(datasets, f)
        
        # Save the metadata file
        indices_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "image_indices")
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
                "filename": new_filename,
                "url": f"/images/{new_filename}",
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