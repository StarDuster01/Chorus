import os
import json
import uuid
import datetime
from datetime import UTC
import tempfile
from werkzeug.utils import secure_filename
from flask import request, jsonify, send_file
# ChromaDB client
import chroma_client
import faiss
import numpy as np
import io
import zipfile
import shutil
import threading
import time
from image_processor import ImageProcessor

from text_extractors import (
    extract_text_from_image,
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_pptx,
    extract_text_from_file,
    create_semantic_chunks,
    chunk_powerpoint_content
)
from image_handlers import resize_image

def add_document_to_chroma(dataset_id, chunks, document_id, filename):
    """Add document chunks to ChromaDB
    
    Args:
        dataset_id: The dataset ID
        chunks: List of text chunks
        document_id: The document ID
        filename: Original filename
    """
    import datetime
    from datetime import UTC
    
    # Use simple client instead of connection pool
    chroma_collection = chroma_client.get_or_create_collection(dataset_id)
    
    # Create chunk IDs and metadata
    chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{
        "document_id": document_id,
        "filename": filename,
        "chunk": i,
        "total_chunks": len(chunks),
        "file_type": os.path.splitext(filename)[1].lower(),
        "created_at": datetime.datetime.now(UTC).isoformat(),
    } for i in range(len(chunks))]
    
    # Add chunks to vector store
    chroma_collection.add(
        ids=chunk_ids,
        documents=chunks,
        metadatas=metadatas
    )

def find_dataset_by_id(user_data, dataset_id):
    """Find a dataset by ID and return it
    
    Args:
        user_data: User data from JWT token
        dataset_id: ID of the dataset to find
        
    Returns:
        tuple: (dataset_dict, dataset_index) or (None, -1) if not found
    """
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    user_datasets_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")
    
    if not os.path.exists(user_datasets_file):
        return None, -1
        
    with open(user_datasets_file, 'r') as f:
        datasets = json.load(f)
        
    for i, dataset in enumerate(datasets):
        if dataset["id"] == dataset_id:
            return dataset, i
            
    return None, -1

def get_mimetype_from_dataset_type(dataset_type):
    """Get the appropriate MIME type based on dataset type"""
    if dataset_type == "image":
        return "image/*"
    elif dataset_type == "text":
        return "text/*"
    else:  # mixed
        return "*/*"

def sync_datasets_with_collections():
    """Syncs datasets with ChromaDB collections to ensure consistency"""
    print("Syncing datasets with ChromaDB collections...")
    
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    
    # Get all dataset files
    dataset_files = [f for f in os.listdir(datasets_dir) if f.endswith('_datasets.json')]
    
    # Get existing collections
    try:
        existing_collections = chroma_client.list_collections()
        existing_collection_names = [col.name for col in existing_collections]
    except Exception as e:
        print(f"Error listing collections: {str(e)}")
        existing_collection_names = []
    
    for dataset_file in dataset_files:
        try:
            with open(os.path.join(datasets_dir, dataset_file), 'r') as f:
                datasets = json.load(f)
                
            for dataset in datasets:
                dataset_id = dataset["id"]
                
                # Check if collection exists in ChromaDB
                if dataset_id not in existing_collection_names:
                    print(f"Creating missing collection for dataset: {dataset_id}")
                    try:
                        chroma_client.get_or_create_collection(dataset_id)
                        print(f"Collection for dataset {dataset_id} ensured.")
                    except Exception as e:
                        print(f"Error ensuring collection for dataset {dataset_id}: {str(e)}")
        except Exception as e:
            print(f"Error processing dataset file {dataset_file}: {str(e)}")
    
    print("Dataset sync completed")

# Helper functions for dataset operations
def get_mime_types_for_dataset(dataset_type):
    """Get MIME types for dataset based on its type
    
    Args:
        dataset_type: The type of dataset (text, image, etc.)
    
    Returns:
        list: List of MIME types
    """
    types = {
        "text": ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/plain", "application/vnd.openxmlformats-officedocument.presentationml.presentation"],
        "image": ["image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp", "image/wmf", "image/x-wmf"]
    }
    return types.get(dataset_type, [])

def convert_wmf_to_png(wmf_path):
    """Convert WMF file to PNG format
    
    Args:
        wmf_path: Path to WMF file
        
    Returns:
        str: Path to converted PNG file
    """
    try:
        from wand.image import Image as WandImage
        
        # Create PNG path
        png_path = os.path.splitext(wmf_path)[0] + '.png'
        
        # Convert WMF to PNG using ImageMagick
        with WandImage(filename=wmf_path) as img:
            img.format = 'png'
            img.save(filename=png_path)
            
        return png_path
    except ImportError:
        print("Warning: Wand (ImageMagick) not available. Cannot convert WMF files.")
        return None
    except Exception as e:
        print(f"Error converting WMF to PNG: {str(e)}")
        return None

# Dataset handler functions for routes
def get_datasets_handler(user_data):
    """Get all datasets for a user
    
    Args:
        user_data: User data from JWT token
    
    Returns:
        tuple: JSON response and status code
    """
    from dataset_cache import dataset_cache
    
    # Try cache first
    cached_datasets = dataset_cache.get_datasets(user_data['id'])
    if cached_datasets is not None:
        return jsonify({
            "datasets": cached_datasets,
            "status": {
                "message": "Retrieved datasets from cache",
                "cache_hit": True
            }
        }), 200
    
    # Cache miss - load from file
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    user_datasets_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")
    
    if not os.path.exists(user_datasets_file):
        dataset_cache.set_datasets(user_data['id'], [])
        return jsonify({
            "datasets": [],
            "status": {
                "message": "No datasets found - ready to create your first dataset",
                "cache_hit": False
            }
        }), 200
        
    with open(user_datasets_file, 'r') as f:
        datasets = json.load(f)
    
    print(f"[Dataset Loading] Processing {len(datasets)} datasets for user {user_data['id']}")
    
    # Process datasets efficiently - skip expensive operations for empty datasets
    for i, dataset in enumerate(datasets):
        print(f"[Dataset Loading] Processing dataset {i+1}/{len(datasets)}: {dataset.get('name', 'Unknown')}")
        
        # Ensure basic fields exist
        dataset["document_count"] = dataset.get("document_count", 0)
        dataset["chunk_count"] = dataset.get("chunk_count", 0)
        dataset["image_count"] = dataset.get("image_count", 0)
        dataset["image_previews"] = dataset.get("image_previews", [])
        
        # Only do expensive operations if the dataset might have content
        if dataset.get("document_count", 0) > 0 or dataset.get("image_count", 0) > 0:
            print(f"[Dataset Loading] Updating metadata for dataset with content: {dataset.get('name')}")
            
            # Update image info quickly by reading metadata file directly (only if needed)
            dataset_id = dataset["id"]
            indices_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "image_indices")
            metadata_file = os.path.join(indices_dir, f"{dataset_id}_metadata.json")
            
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Count valid images and create previews (first 4 only)
                    valid_count = 0
                    previews = []
                    for img_meta in metadata:
                        if img_meta.get('dataset_id') == dataset_id:
                            if os.path.exists(img_meta['path']):
                                valid_count += 1
                                if len(previews) < 4:  # Only get first 4 for previews
                                    filename = os.path.basename(img_meta["path"])
                                    previews.append({
                                        "id": img_meta.get("id", ""),
                                        "url": f"/api/images/{filename}",
                                        "caption": img_meta.get("caption", "")
                                    })
                    
                    dataset["image_count"] = valid_count
                    dataset["image_previews"] = previews
                except Exception as e:
                    print(f"[Dataset Loading] Error reading metadata for dataset {dataset_id}: {str(e)}")
                    dataset["image_count"] = dataset.get("image_count", 0)
                    dataset["image_previews"] = dataset.get("image_previews", [])
        else:
            print(f"[Dataset Loading] Skipping metadata update for empty dataset: {dataset.get('name')}")
    
    print(f"[Dataset Loading] Completed processing all datasets for user {user_data['id']}")
    
    # Cache the processed datasets
    dataset_cache.set_datasets(user_data['id'], datasets)
    
    return jsonify({
        "datasets": datasets,
        "status": {
            "message": f"Loaded {len(datasets)} datasets successfully",
            "cache_hit": False,
            "details": {
                "total_datasets": len(datasets),
                "total_documents": sum(d.get("document_count", 0) for d in datasets),
                "total_chunks": sum(d.get("chunk_count", 0) for d in datasets),
                "total_images": sum(d.get("image_count", 0) for d in datasets)
            }
        }
    }), 200

def create_dataset_handler(user_data):
    """Create a new dataset
    
    Args:
        user_data: User data from JWT token
        
    Returns:
        tuple: JSON response and status code
    """
    print("\n=== Dataset Creation Process Started ===")
    print(f"User ID: {user_data['id']}")
    
    data = request.get_json()
    print(f"Request data: {json.dumps(data, indent=2)}")
    
    # Validate required fields
    if not data.get('name'):
        print("❌ Error: Dataset name is required")
        return jsonify({"error": "Dataset name is required"}), 400
    
    if not data.get('type'):
        print("❌ Error: Dataset type is required")
        return jsonify({"error": "Dataset type is required"}), 400
    
    # Validate dataset type - now supports mixed type
    valid_types = ['text', 'image', 'mixed']
    if data.get('type') not in valid_types:
        print(f"❌ Error: Invalid dataset type '{data.get('type')}'. Must be one of: {valid_types}")
        return jsonify({"error": f"Dataset type must be one of: {', '.join(valid_types)}"}), 400
    
    print(f"\n📝 Creating dataset: {data.get('name')} (type: {data.get('type')})")
    
    # Create dataset
    dataset_id = str(uuid.uuid4())
    new_dataset = {
        "id": dataset_id,
        "name": data.get('name'),
        "description": data.get('description', ''),
        "type": data.get('type'),
        "created_at": datetime.datetime.now(UTC).isoformat(),
        "document_count": 0,
        "chunk_count": 0,
        "status": "created"  # Add status field
    }
    
    # Load existing datasets or create empty list
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    user_datasets_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")
    
    print(f"\n📂 Dataset directory: {datasets_dir}")
    print(f"📄 User datasets file: {user_datasets_file}")
    
    if os.path.exists(user_datasets_file):
        print("Found existing datasets file, loading...")
        with open(user_datasets_file, 'r') as f:
            datasets = json.load(f)
        print(f"Loaded {len(datasets)} existing datasets")
    else:
        print("No existing datasets file, creating new one")
        datasets = []
    
    datasets.append(new_dataset)
    print(f"\n✅ Added dataset to list. Total datasets: {len(datasets)}")
    
    # Save datasets to file with error handling
    try:
        print("\n💾 Saving dataset to file system...")
        with open(user_datasets_file, 'w') as f:
            json.dump(datasets, f, indent=2)
            f.flush()  # Ensure data is written to disk
            os.fsync(f.fileno())  # Force write to disk
        print("✅ Dataset saved to file system successfully")
    except Exception as e:
        print(f"❌ Error saving dataset to file: {str(e)}")
        return jsonify({
            "error": "Failed to save dataset",
            "details": str(e)
        }), 500
    
    # Create a ChromaDB collection for this dataset
    print(f"\n🔧 Creating vector database collection for dataset {dataset_id}...")
    try:
        print(f"Creating collection {dataset_id}...")
        
        # Create the collection (embedding function already initialized at startup)
        collection = chroma_client.get_or_create_collection(dataset_id)
        print(f"✅ Vector database collection created successfully: {collection.name}")
        
        # Verify collection is working by testing a simple operation
        try:
            count = collection.count()
            print(f"✅ Collection health check passed. Current count: {count}")
        except Exception as health_error:
            print(f"❌ Collection health check failed: {str(health_error)}")
            raise
            
    except Exception as e:
        print(f"\n❌ Error creating vector database collection: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Error traceback:\n{traceback.format_exc()}")
        
        # ROLLBACK: Remove the dataset from JSON file if ChromaDB failed
        print(f"\n🔄 Rolling back dataset creation due to ChromaDB failure...")
        try:
            if os.path.exists(user_datasets_file):
                with open(user_datasets_file, 'r') as f:
                    datasets = json.load(f)
                
                # Remove the dataset we just added
                datasets = [d for d in datasets if d["id"] != dataset_id]
                
                with open(user_datasets_file, 'w') as f:
                    json.dump(datasets, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                print(f"✅ Dataset rollback completed")
        except Exception as rollback_error:
            print(f"❌ Error during rollback: {str(rollback_error)}")
        
        return jsonify({
            "error": "Failed to initialize vector database",
            "details": str(e),
            "suggestion": "This may be due to GPU memory issues or embedding model problems. Try restarting the application."
        }), 500
    
    # Invalidate cache when dataset is created
    print(f"\n🔄 Invalidating cache...")
    from dataset_cache import dataset_cache
    dataset_cache.invalidate_user(user_data['id'])
    
    print(f"\n✅ Dataset creation completed successfully: {data.get('name')}")
    print("=== Dataset Creation Process Completed ===\n")
    
    return jsonify({
        **new_dataset,
        "status": "ready",
        "message": "Dataset created successfully and ready for documents"
    }), 201

def delete_dataset_handler(user_data, dataset_id):
    """Delete a dataset and all its associated data
    
    Args:
        user_data: User data from JWT token
        dataset_id: ID of the dataset to delete
        
    Returns:
        tuple: JSON response and status code
    """
    print(f"[Dataset Deletion] Starting deletion process for dataset {dataset_id}")
    
    # Check if dataset exists and belongs to user
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    user_datasets_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")
    
    print(f"[Dataset Deletion] Checking for dataset file: {user_datasets_file}")
    if not os.path.exists(user_datasets_file):
        print(f"[Dataset Deletion] Dataset file not found")
        return jsonify({"error": "Dataset not found"}), 404
        
    print(f"[Dataset Deletion] Loading datasets from file...")
    with open(user_datasets_file, 'r') as f:
        datasets = json.load(f)
        
    print(f"[Dataset Deletion] Found {len(datasets)} datasets, looking for {dataset_id}")
    dataset_exists = False
    dataset_index = -1
    for idx, dataset in enumerate(datasets):
        if dataset["id"] == dataset_id:
            dataset_exists = True
            dataset_index = idx
            print(f"[Dataset Deletion] Found dataset at index {idx}: {dataset.get('name')}")
            break
            
    if not dataset_exists:
        print(f"[Dataset Deletion] Dataset {dataset_id} not found in user's datasets")
        return jsonify({"error": "Dataset not found"}), 404
    
    document_files_deleted = 0
    
    # Delete the dataset from the file
    print(f"[Dataset Deletion] Removing dataset from file...")
    del datasets[dataset_index]
    
    print(f"[Dataset Deletion] Saving updated dataset list...")
    with open(user_datasets_file, 'w') as f:
        json.dump(datasets, f)
        f.flush()  # Ensure data is written to disk
        os.fsync(f.fileno())  # Force write to disk
    
    # Clean up ChromaDB collection and associated files
    try:
        print(f"[Dataset Deletion] Cleaning up ChromaDB collection...")
        
        # Get the collection for this dataset
        try:
            print(f"[Dataset Deletion] Getting ChromaDB collection {dataset_id}...")
            collection = chroma_client.get_collection(dataset_id)
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
                            print(f"[Dataset Deletion] Deleted document file: {file_path}")
                    except Exception as e:
                        print(f"[Dataset Deletion] Error deleting document file {file_path}: {str(e)}")
            
            # Now delete the collection itself
            print(f"[Dataset Deletion] Deleting ChromaDB collection {dataset_id}...")
            chroma_client.delete_collection(dataset_id)
            print(f"[Dataset Deletion] Successfully deleted ChromaDB collection {dataset_id}")
            
        except Exception as e:
            print(f"[Dataset Deletion] Error accessing ChromaDB collection: {str(e)}")
            # Try to delete the collection anyway
            try:
                print(f"[Dataset Deletion] Attempting to force delete ChromaDB collection {dataset_id}...")
                chroma_client.delete_collection(dataset_id)
                print(f"[Dataset Deletion] Successfully force deleted ChromaDB collection {dataset_id}")
            except Exception as e2:
                print(f"[Dataset Deletion] Error force deleting ChromaDB collection: {str(e2)}")
    except Exception as e:
        print(f"[Dataset Deletion] Error connecting to ChromaDB: {str(e)}")
    
    # Clean up image dataset resources (FAISS index and metadata)
    indices_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "image_indices")
    index_path = os.path.join(indices_dir, f"{dataset_id}_index.faiss")
    metadata_file = os.path.join(indices_dir, f"{dataset_id}_metadata.json")
    
    # Delete image files referenced in metadata file
    if os.path.exists(metadata_file):
        try:
            print(f"[Dataset Deletion] Processing image metadata file: {metadata_file}")
            with open(metadata_file, 'r') as f:
                image_metadata = json.load(f)
                
            # Delete all image files referenced in metadata
            for img_meta in image_metadata:
                if 'path' in img_meta and os.path.exists(img_meta['path']):
                    try:
                        os.remove(img_meta['path'])
                        document_files_deleted += 1
                        print(f"[Dataset Deletion] Deleted image file: {img_meta['path']}")
                    except Exception as e:
                        print(f"[Dataset Deletion] Error deleting image file: {str(e)}")
            
            # Now delete the metadata file itself
            os.remove(metadata_file)
            print(f"[Dataset Deletion] Deleted image metadata file for dataset {dataset_id}")
        except Exception as e:
            print(f"[Dataset Deletion] Error processing image metadata: {str(e)}")
    
    # Delete FAISS index file if it exists
    if os.path.exists(index_path):
        try:
            os.remove(index_path)
            print(f"[Dataset Deletion] Deleted FAISS index for dataset {dataset_id}")
        except Exception as e:
            print(f"[Dataset Deletion] Error deleting FAISS index: {str(e)}")
    
    # Try to remove the dataset from image processor memory
    try:
        print(f"[Dataset Deletion] Cleaning up image processor resources...")
        from app import image_processor
        if hasattr(image_processor, 'image_indices') and dataset_id in image_processor.image_indices:
            del image_processor.image_indices[dataset_id]
            print(f"[Dataset Deletion] Removed dataset {dataset_id} from image processor indices")
        
        if hasattr(image_processor, 'image_metadata') and dataset_id in image_processor.image_metadata:
            del image_processor.image_metadata[dataset_id]
            print(f"[Dataset Deletion] Removed dataset {dataset_id} from image processor metadata")
    except Exception as e:
        print(f"[Dataset Deletion] Error cleaning up image processor resources: {str(e)}")
    
    # Legacy cleanup for older datasets that might still use the documents array
    if dataset_exists and dataset.get("documents"):
        print(f"[Dataset Deletion] Cleaning up legacy document files...")
        for doc in dataset["documents"]:
            doc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads", "documents", doc["filename"])
            try:
                if os.path.exists(doc_path):
                    os.remove(doc_path)
                    document_files_deleted += 1
                    print(f"[Dataset Deletion] Deleted legacy document: {doc['filename']}")
            except Exception as e:
                print(f"[Dataset Deletion] Error deleting legacy document {doc['filename']}: {str(e)}")
    
    # Legacy cleanup for older datasets that might still use the images array
    if dataset_exists and dataset.get("images"):
        print(f"[Dataset Deletion] Cleaning up legacy image files...")
        for img in dataset["images"]:
            img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads", "images", img["filename"])
            try:
                if os.path.exists(img_path):
                    os.remove(img_path)
                    document_files_deleted += 1
                    print(f"[Dataset Deletion] Deleted legacy image: {img['filename']}")
            except Exception as e:
                print(f"[Dataset Deletion] Error deleting legacy image {img['filename']}: {str(e)}")
    
    # Also remove the dataset from any bots that use it
    bots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bots")
    if os.path.exists(bots_dir):
        print(f"[Dataset Deletion] Updating bot references...")
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
                            print(f"[Dataset Deletion] Removed dataset {dataset_id} from bot {bot.get('name', 'unnamed')}")
                    
                    if updated:
                        with open(bots_file_path, 'w') as f:
                            json.dump(bots, f)
                            print(f"[Dataset Deletion] Updated bot file: {filename}")
                except Exception as e:
                    print(f"[Dataset Deletion] Error updating bots file {filename}: {str(e)}")
    
    # Invalidate cache so frontend sees dataset is deleted
    try:
        from dataset_cache import dataset_cache
        dataset_cache.invalidate_user(user_data['id'])
        print(f"🔄 Cache invalidated for user {user_data['id']} after dataset deletion")
    except Exception as cache_e:
        print(f"Warning: Could not invalidate cache: {cache_e}")
    
    print(f"[Dataset Deletion] Successfully deleted dataset {dataset_id} with {document_files_deleted} document files")
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
    dataset, dataset_index = find_dataset_by_id(user_data, dataset_id)
    if not dataset:
        return jsonify({"error": "Dataset not found"}), 404
    
    # Try to get the collection from ChromaDB
    try:
        collection = chroma_client.get_collection(dataset_id)
        
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
        # Load the datasets to get the updated list
        datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
        user_datasets_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")
        
        with open(user_datasets_file, 'r') as f:
            datasets = json.load(f)
        
        # Find and update the dataset
        for i, ds in enumerate(datasets):
            if ds["id"] == dataset_id:
                if ds["document_count"] > 0:
                    ds["document_count"] -= 1
                    
                # Update chunk count if it exists
                if "chunk_count" in ds:
                    ds["chunk_count"] -= num_chunks_to_remove
                    if ds["chunk_count"] < 0:
                        ds["chunk_count"] = 0
                break
            
        # Save the updated datasets
        with open(user_datasets_file, 'w') as f:
            json.dump(datasets, f)
            f.flush()  # Ensure data is written to disk
            os.fsync(f.fileno())  # Force write to disk
            
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
    dataset, dataset_index = find_dataset_by_id(user_data, dataset_id)
    if not dataset:
        return jsonify({"error": "Dataset not found"}), 404
    
    # Load and update dataset
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    user_datasets_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")
    
    with open(user_datasets_file, 'r') as f:
        datasets = json.load(f)
    
    # Reset document count
    datasets[dataset_index]["document_count"] = 0
    
    with open(user_datasets_file, 'w') as f:
        json.dump(datasets, f)
        f.flush()  # Ensure data is written to disk
        os.fsync(f.fileno())  # Force write to disk
    
    # Try to delete the existing collection if it exists
    try:
        try:
            chroma_client.delete_collection(dataset_id)
            print(f"Deleted existing collection for dataset: {dataset_id}")
        except Exception as e:
            print(f"Collection may not exist: {str(e)}")
        
        # Create a new collection
        chroma_client.get_or_create_collection(dataset_id)
        print(f"Collection for dataset {dataset_id} ensured.")
                
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
    
    # resize_image is already imported from text_extractors at the top
    
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
        "url": f"/api/images/{new_filename}",  # Use just the filename for the URL
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
            f.flush()  # Ensure data is written to disk
            os.fsync(f.fileno())  # Force write to disk
        
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
                    f.flush()  # Ensure data is written to disk
                    os.fsync(f.fileno())  # Force write to disk
            except Exception as e:
                print(f"Error updating metadata file: {str(e)}")
                # Create new file
                with open(metadata_file, 'w') as f:
                    json.dump([image_metadata], f)
                    f.flush()  # Ensure data is written to disk
                    os.fsync(f.fileno())  # Force write to disk
        else:
            # Create new file
            with open(metadata_file, 'w') as f:
                json.dump([image_metadata], f)
                f.flush()  # Ensure data is written to disk
                os.fsync(f.fileno())  # Force write to disk
        
        # Invalidate cache so frontend sees updated counts
        try:
            from dataset_cache import dataset_cache
            dataset_cache.invalidate_user(user_data['id'])
            print(f"🔄 Cache invalidated for user {user_data['id']} after image upload")
        except Exception as cache_e:
            print(f"Warning: Could not invalidate cache: {cache_e}")
        
        return jsonify({
            "message": "Image uploaded and processed successfully",
            "image": {
                "id": image_metadata["id"],
                "filename": new_filename,
                "url": f"/api/images/{new_filename}",
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

def bulk_upload_handler(user_data, dataset_id):
    """Bulk upload a zip file of documents/images to a dataset
    
    Args:
        user_data: User data from JWT token
        dataset_id: ID of the dataset to upload to
        
    Returns:
        tuple: JSON response and status code
    """
    # resize_image is already imported from text_extractors at the top
    import threading
    import time
    
    print(f"Starting bulk upload for dataset {dataset_id}")
    
    # Check if dataset exists and belongs to user
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    user_datasets_file = os.path.join(datasets_dir, f"{user_data['id']}_datasets.json")
    if not os.path.exists(user_datasets_file):
        print(f"Dataset file not found: {user_datasets_file}")
        return jsonify({"error": "Dataset not found"}), 404
        
    with open(user_datasets_file, 'r') as f:
        datasets = json.load(f)
    dataset = next((d for d in datasets if d["id"] == dataset_id), None)
    if not dataset:
        print(f"Dataset {dataset_id} not found in user's datasets")
        return jsonify({"error": "Dataset not found"}), 404
        
    dataset_type = dataset.get("type", "mixed")
    print(f"Dataset type: {dataset_type}")
    
    # Only allow for mixed or text/image datasets
    if dataset_type not in ["mixed", "text", "image"]:
        print(f"Invalid dataset type for bulk upload: {dataset_type}")
        return jsonify({"error": "Bulk upload only supported for mixed, text, or image datasets"}), 400
        
    # Check for file in request
    if 'file' not in request.files:
        print("No file part in request")
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        print("No selected file")
        return jsonify({"error": "No selected file"}), 400
        
    filename = secure_filename(file.filename)
    file_ext = os.path.splitext(filename)[1].lower()
    if file_ext != '.zip':
        print(f"Invalid file extension: {file_ext}")
        return jsonify({"error": "Only zip files are supported for bulk upload"}), 400
        
    print(f"Processing zip file: {filename}")
    
    # Create unique temp directory for this upload
    base_temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
    os.makedirs(base_temp_dir, exist_ok=True)
    
    # Create unique subdirectory with timestamp
    upload_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    temp_dir = os.path.join(base_temp_dir, upload_id)
    os.makedirs(temp_dir, exist_ok=True)
    
    zip_path = os.path.join(temp_dir, filename)
    file.save(zip_path)
    print(f"Saved zip to: {zip_path}")
    
    # Create a processing status file
    status_file = os.path.join(temp_dir, "status.json")
    initial_status = {
        "status": "uploaded",
        "message": "File uploaded successfully, starting processing...",
        "total_files": 0,
        "processed_files": 0,
        "current_file": None,
        "successes": [],
        "errors": []
    }
    
    with open(status_file, 'w') as f:
        json.dump(initial_status, f)
    
    # Start processing in a background thread
    def process_files():
        try:
            extract_dir = os.path.join(temp_dir, "extracted")
            os.makedirs(extract_dir, exist_ok=True)
            
            # Extract zip
            print("Extracting zip file...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print("Zip extraction complete")
            
            # Count total files
            total_files = 0
            for root, dirs, files in os.walk(extract_dir):
                total_files += len(files)
            
            # Update status with total files
            with open(status_file, 'r') as f:
                status = json.load(f)
            status["total_files"] = total_files
            status["status"] = "processing"
            with open(status_file, 'w') as f:
                json.dump(status, f)
            
            # Initialize image processor for bulk upload
            image_processor = ImageProcessor(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
            
            # Process files
            successes = []
            errors = []
            text_exts = ['.pdf', '.docx', '.txt', '.pptx']
            image_exts = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']
            DOCUMENT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads", "documents")
            IMAGE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads", "images")
            os.makedirs(DOCUMENT_FOLDER, exist_ok=True)
            os.makedirs(IMAGE_FOLDER, exist_ok=True)
            
            # Collect files by type for batch processing
            text_files = []
            image_files = []
            
            # First pass: collect files by type
            for root, dirs, files in os.walk(extract_dir):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    ext = os.path.splitext(fname)[1].lower()
                    
                    if ext in text_exts and dataset_type in ["mixed", "text"]:
                        text_files.append((fpath, fname, ext))
                    elif ext in image_exts and dataset_type in ["mixed", "image"]:
                        image_files.append((fpath, fname, ext))
            
            print(f"Batch processing: {len(text_files)} text files, {len(image_files)} image files")
            
            processed_files = 0
            
            # Process text files sequentially (they're typically smaller)
            for fpath, fname, ext in text_files:
                # Update status with current file
                with open(status_file, 'r') as f:
                    status = json.load(f)
                status["current_file"] = fname
                status["processed_files"] = processed_files
                with open(status_file, 'w') as f:
                    json.dump(status, f)
                
                try:
                    # Process text document
                    doc_id = str(uuid.uuid4())
                    dest_name = f"{doc_id}{ext}"
                    dest_path = os.path.join(DOCUMENT_FOLDER, dest_name)
                    shutil.copy2(fpath, dest_path)
                    
                    # Extract text and create chunks
                    if ext == '.pptx':
                        text, pptx_image_metadata = extract_text_from_pptx(dest_path)
                    else:
                        text = extract_text_from_file(dest_path)
                        pptx_image_metadata = []
                        
                    if not text:
                        errors.append({"file": fname, "error": "Could not extract text from file"})
                        processed_files += 1
                        continue
                        
                    chunks = create_semantic_chunks(text)
                    
                    # Add to ChromaDB
                    try:
                        add_document_to_chroma(dataset_id, chunks, doc_id, fname)
                        successes.append({
                            "file": fname,
                            "type": "document",
                            "chunks": len(chunks)
                        })
                    except Exception as e:
                        print(f"ChromaDB error for {fname}: {str(e)}")
                        errors.append({"file": fname, "error": f"ChromaDB error: {str(e)}"})
                        
                    processed_files += 1
                    
                except Exception as e:
                    print(f"Error processing text file {fname}: {str(e)}")
                    errors.append({"file": fname, "error": str(e)})
                    processed_files += 1
            
            # Process images in batches for better GPU utilization
            BATCH_SIZE = 8  # Process 8 images at a time
            for i in range(0, len(image_files), BATCH_SIZE):
                batch = image_files[i:i+BATCH_SIZE]
                
                # Update status for batch
                batch_filenames = [fname for _, fname, _ in batch]
                with open(status_file, 'r') as f:
                    status = json.load(f)
                status["current_file"] = f"Processing batch: {', '.join(batch_filenames[:3])}{'...' if len(batch_filenames) > 3 else ''}"
                status["processed_files"] = processed_files
                with open(status_file, 'w') as f:
                    json.dump(status, f)
                
                # Process batch
                for fpath, fname, ext in batch:
                    try:
                        # Process image
                        img_id = str(uuid.uuid4())
                        dest_name = f"{img_id}{ext}"
                        dest_path = os.path.join(IMAGE_FOLDER, dest_name)
                        shutil.copy2(fpath, dest_path)
                        dest_path = resize_image(dest_path, max_dimension=1024)
                        
                        meta = {
                            "id": img_id,
                            "dataset_id": dataset_id,
                            "original_filename": fname,
                            "path": dest_path,
                            "url": f"/api/images/{dest_name}",
                            "type": "image",
                            "created_at": datetime.datetime.now(UTC).isoformat(),
                            "user_id": user_data['id'],
                            "username": user_data['username']
                        }
                        
                        image_processor.add_image_to_dataset(dataset_id, dest_path, meta)
                        successes.append({"file": fname, "type": "image"})
                        processed_files += 1
                        
                    except Exception as e:
                        print(f"Error processing image {fname}: {str(e)}")
                        errors.append({"file": fname, "error": str(e)})
                        processed_files += 1
            
            # Update final status
            with open(status_file, 'r') as f:
                status = json.load(f)
            status["status"] = "completed"
            status["message"] = f"Processing complete: {len(successes)} files added, {len(errors)} errors"
            status["processed_files"] = processed_files
            status["successes"] = successes
            status["errors"] = errors
            with open(status_file, 'w') as f:
                json.dump(status, f)
            
            # Update dataset counts
            try:
                with open(user_datasets_file, 'r') as f:
                    datasets = json.load(f)
                
                for idx, ds in enumerate(datasets):
                    if ds["id"] == dataset_id:
                        # Count successful documents and images
                        doc_successes = [s for s in successes if s["type"] == "document"]
                        img_successes = [s for s in successes if s["type"] == "image"]
                        
                        # Update document count
                        if doc_successes:
                            if "document_count" in ds:
                                ds["document_count"] += len(doc_successes)
                            else:
                                ds["document_count"] = len(doc_successes)
                        
                        # Update chunk count
                        total_chunks = sum(s.get("chunks", 0) for s in doc_successes)
                        if total_chunks > 0:
                            if "chunk_count" in ds:
                                ds["chunk_count"] += total_chunks
                            else:
                                ds["chunk_count"] = total_chunks
                        
                        # Update image count
                        if img_successes:
                            if "image_count" in ds:
                                ds["image_count"] += len(img_successes)
                            else:
                                ds["image_count"] = len(img_successes)
                        break
                
                with open(user_datasets_file, 'w') as f:
                    json.dump(datasets, f)
                    
                # Invalidate cache so frontend sees updated counts
                try:
                    from dataset_cache import dataset_cache
                    dataset_cache.invalidate_user(user_data['id'])
                    print(f"🔄 Cache invalidated for user {user_data['id']} after bulk upload")
                except Exception as cache_e:
                    print(f"Warning: Could not invalidate cache: {cache_e}")
                    
            except Exception as e:
                print(f"Error updating dataset counts: {str(e)}")
            
        except Exception as e:
            print(f"Error in processing thread: {str(e)}")
            with open(status_file, 'r') as f:
                status = json.load(f)
            status["status"] = "error"
            status["message"] = f"Processing failed: {str(e)}"
            with open(status_file, 'w') as f:
                json.dump(status, f)
        finally:
            # Clean up temp directory after 1 hour
            def cleanup():
                time.sleep(3600)  # Wait 1 hour
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
            
            cleanup_thread = threading.Thread(target=cleanup)
            cleanup_thread.daemon = True
            cleanup_thread.start()
    
    # Start processing thread
    process_thread = threading.Thread(target=process_files)
    process_thread.daemon = True
    process_thread.start()
    
    # Return initial response with status file path
    return jsonify({
        "message": "File uploaded successfully, processing started",
        "status_file": f"/api/datasets/{dataset_id}/upload-status/{upload_id}"
    }), 202

def diagnose_dataset_handler(user_data, dataset_id):
    """Diagnose dataset and collection health
    
    Args:
        user_data: User data from JWT token
        dataset_id: ID of the dataset to diagnose
    
    Returns:
        tuple: JSON response and status code
    """
    # Check if dataset exists in JSON
    dataset, dataset_index = find_dataset_by_id(user_data, dataset_id)
    if not dataset:
        return jsonify({"error": "Dataset not found"}), 404
    
    diagnosis = {
        "dataset_id": dataset_id,
        "dataset_name": dataset.get("name", "Unknown"),
        "dataset_type": dataset.get("type", "Unknown"),
        "json_file": {
            "exists": True,
            "document_count": dataset.get("document_count", 0),
            "chunk_count": dataset.get("chunk_count", 0),
            "image_count": dataset.get("image_count", 0)
        }
    }
    
    # Diagnose ChromaDB collection
    try:
        # For now, do a simple check since we removed the complex diagnosis from connection pool
        collection_diagnosis = {"exists": False, "accessible": False}
        try:
            collection = chroma_client.get_collection(dataset_id)
            count = collection.count()
            collection_diagnosis = {
                "exists": True,
                "count": count,
                "accessible": True,
                "embedding_compatible": True
            }
        except Exception as e:
            collection_diagnosis = {"exists": False, "error": str(e)}
        diagnosis["chromadb_collection"] = collection_diagnosis
        
        # Check for sync issues
        json_doc_count = dataset.get("document_count", 0)
        if collection_diagnosis.get("exists") and collection_diagnosis.get("count", 0) > 0:
            chroma_count = collection_diagnosis.get("count", 0)
            # Estimate document count from chunks (assuming ~5 chunks per document)
            estimated_doc_count = max(1, chroma_count // 5)
            if abs(json_doc_count - estimated_doc_count) > 2:  # Allow some variance
                diagnosis["sync_issues"] = {
                    "detected": True,
                    "json_documents": json_doc_count,
                    "estimated_from_chunks": estimated_doc_count,
                    "total_chunks": chroma_count
                }
            else:
                diagnosis["sync_issues"] = {"detected": False}
        else:
            diagnosis["sync_issues"] = {"detected": json_doc_count > 0}
            
    except Exception as e:
        diagnosis["chromadb_collection"] = {
            "error": f"Failed to diagnose: {str(e)}"
        }
    
    # Check image index for image/mixed datasets
    if dataset.get("type") in ["image", "mixed"]:
        indices_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "image_indices")
        index_path = os.path.join(indices_dir, f"{dataset_id}_index.faiss")
        metadata_file = os.path.join(indices_dir, f"{dataset_id}_metadata.json")
        
        diagnosis["image_index"] = {
            "faiss_index_exists": os.path.exists(index_path),
            "metadata_file_exists": os.path.exists(metadata_file)
        }
        
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                diagnosis["image_index"]["metadata_count"] = len(metadata)
            except Exception as e:
                diagnosis["image_index"]["metadata_error"] = str(e)
    
    # Overall health assessment
    collection_healthy = diagnosis.get("chromadb_collection", {}).get("accessible", False)
    sync_healthy = not diagnosis.get("sync_issues", {}).get("detected", True)
    
    diagnosis["overall_health"] = {
        "status": "healthy" if collection_healthy and sync_healthy else "issues_detected",
        "collection_accessible": collection_healthy,
        "data_synchronized": sync_healthy
    }
    
    return jsonify(diagnosis), 200

# More handlers can be added here for specific dataset operations 