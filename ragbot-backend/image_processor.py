import os
import json
import uuid
import base64
import datetime
from typing import List, Dict, Any, Tuple, Optional
import torch
from PIL import Image
import open_clip
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
import faiss

# Configurations
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_CLIP_MODEL = "ViT-B/32"
DEFAULT_CLIP_PRETRAINED = "openai"
DEFAULT_BLIP_MODEL = "Salesforce/blip-image-captioning-base"
VECTOR_DIMENSION = 512  # Dimension for ViT-B/32

class ImageProcessor:
    def __init__(self, data_dir: str):
        """Initialize the image processor with models for embedding and captioning
        
        Args:
            data_dir: Base directory for saving indices and metadata
        """
        self.data_dir = data_dir
        self.image_indices = {}  # Dataset ID -> FAISS index
        self.image_metadata = {}  # Dataset ID -> list of metadata dictionaries
        self.clip_model = None
        self.clip_preprocess = None
        self.blip_model = None
        self.blip_processor = None
        
        # Create directory for indices if it doesn't exist
        self.indices_dir = os.path.join(data_dir, "image_indices")
        os.makedirs(self.indices_dir, exist_ok=True)
        
        # Load indices for existing datasets
        self._load_existing_indices()
    
    def _load_models(self):
        """Load CLIP and BLIP models if not already loaded"""
        if self.clip_model is None:
            print(f"Loading CLIP model on {DEVICE}...")
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                DEFAULT_CLIP_MODEL, 
                pretrained=DEFAULT_CLIP_PRETRAINED,
                device=DEVICE
            )
            self.clip_model.eval()
        
        if self.blip_model is None:
            print(f"Loading BLIP model on {DEVICE}...")
            self.blip_processor = BlipProcessor.from_pretrained(DEFAULT_BLIP_MODEL)
            self.blip_model = BlipForConditionalGeneration.from_pretrained(DEFAULT_BLIP_MODEL).to(DEVICE)
    
    def _load_existing_indices(self):
        """Load existing FAISS indices and metadata for datasets"""
        if not os.path.exists(self.indices_dir):
            return
            
        for filename in os.listdir(self.indices_dir):
            if filename.endswith("_index.faiss"):
                dataset_id = filename.split("_index.faiss")[0]
                
                # Check if metadata file exists
                metadata_file = os.path.join(self.indices_dir, f"{dataset_id}_metadata.json")
                if not os.path.exists(metadata_file):
                    continue
                    
                try:
                    # Load FAISS index
                    index_path = os.path.join(self.indices_dir, filename)
                    index = faiss.read_index(index_path)
                    
                    # Load metadata
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    self.image_indices[dataset_id] = index
                    self.image_metadata[dataset_id] = metadata
                    print(f"Loaded image index for dataset {dataset_id} with {len(metadata)} images")
                except Exception as e:
                    print(f"Error loading index for {dataset_id}: {str(e)}")
    
    def generate_caption(self, image_path: str) -> str:
        """Generate a caption for an image using BLIP
        
        Args:
            image_path: Path to the image file
            
        Returns:
            str: Generated caption
        """
        self._load_models()
        
        try:
            # Load and preprocess image
            raw_image = Image.open(image_path).convert('RGB')
            inputs = self.blip_processor(raw_image, return_tensors="pt").to(DEVICE)
            
            # Generate caption
            with torch.no_grad():
                outputs = self.blip_model.generate(**inputs, max_new_tokens=50)
                caption = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
                
            return caption
        except Exception as e:
            print(f"Error generating caption: {str(e)}")
            return "No caption available"
    
    def compute_image_embedding(self, image_path: str) -> np.ndarray:
        """Compute CLIP embedding for an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            np.ndarray: Normalized embedding vector
        """
        self._load_models()
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.clip_preprocess(image).unsqueeze(0).to(DEVICE)
            
            # Compute embedding
            with torch.no_grad():
                embedding = self.clip_model.encode_image(image_tensor).cpu().numpy()
                
            # Normalize the embedding
            embedding = embedding / np.linalg.norm(embedding)
            return embedding[0]  # Return the first (and only) embedding
        except Exception as e:
            print(f"Error computing image embedding: {str(e)}")
            return np.zeros(VECTOR_DIMENSION, dtype=np.float32)
    
    def compute_text_embedding(self, text: str) -> np.ndarray:
        """Compute CLIP embedding for a text query
        
        Args:
            text: Text query
            
        Returns:
            np.ndarray: Normalized embedding vector
        """
        self._load_models()
        
        try:
            # Tokenize and encode text
            with torch.no_grad():
                text_tokens = open_clip.tokenize([text]).to(DEVICE)
                embedding = self.clip_model.encode_text(text_tokens).cpu().numpy()
                
            # Normalize the embedding
            embedding = embedding / np.linalg.norm(embedding)
            return embedding[0]  # Return the first (and only) embedding
        except Exception as e:
            print(f"Error computing text embedding: {str(e)}")
            return np.zeros(VECTOR_DIMENSION, dtype=np.float32)
    
    def add_image_to_dataset(self, dataset_id: str, image_path: str, metadata: Dict = None) -> Dict:
        """Add an image to a dataset's index
        
        Args:
            dataset_id: ID of the dataset
            image_path: Path to the image file
            metadata: Additional metadata for the image
            
        Returns:
            Dict: Image metadata including ID and embedding
        """
        self._load_models()
        
        # Initialize the dataset's index and metadata if needed
        if dataset_id not in self.image_indices:
            self.image_indices[dataset_id] = faiss.IndexFlatIP(VECTOR_DIMENSION)
            self.image_metadata[dataset_id] = []
        
        # Generate image embedding
        embedding = self.compute_image_embedding(image_path)
        
        # Generate caption
        caption = self.generate_caption(image_path)
        
        # Create image ID and metadata
        image_id = str(uuid.uuid4())
        image_metadata = {
            "id": image_id,
            "path": image_path,
            "caption": caption,
            "original_filename": os.path.basename(image_path),
            "created_at": datetime.datetime.utcnow().isoformat(),
            **(metadata or {})
        }
        
        # Add to index and metadata
        self.image_indices[dataset_id].add(np.array([embedding], dtype=np.float32))
        self.image_metadata[dataset_id].append(image_metadata)
        
        # Save updated index and metadata
        self._save_dataset_index(dataset_id)
        
        return image_metadata
    
    def search_images(self, dataset_id: str, query: str, top_k: int = 5) -> List[Dict]:
        """Search for images matching a text query
        
        Args:
            dataset_id: ID of the dataset to search
            query: Text query
            top_k: Maximum number of results to return
            
        Returns:
            List[Dict]: List of image metadata for matching images
        """
        # Check if dataset exists
        if dataset_id not in self.image_indices or dataset_id not in self.image_metadata:
            return []
            
        # Get the index and metadata
        index = self.image_indices[dataset_id]
        metadata = self.image_metadata[dataset_id]
        
        if len(metadata) == 0:
            return []
            
        # Compute query embedding
        query_embedding = self.compute_text_embedding(query)
        
        # Search the index
        k = min(top_k, len(metadata))
        scores, indices = index.search(np.array([query_embedding], dtype=np.float32), k)
        
        # Return results with scores
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # Valid index
                result = metadata[idx].copy()
                result["score"] = float(scores[0][i])  # Add similarity score
                results.append(result)
        
        return results
    
    def remove_image(self, dataset_id: str, image_id: str) -> bool:
        """Remove an image from a dataset
        
        Args:
            dataset_id: ID of the dataset
            image_id: ID of the image to remove
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Check if dataset exists
        if dataset_id not in self.image_indices or dataset_id not in self.image_metadata:
            return False
            
        # Find the image in metadata
        metadata = self.image_metadata[dataset_id]
        for i, item in enumerate(metadata):
            if item["id"] == image_id:
                # We need to rebuild the index without this image
                # FAISS doesn't support direct removal, so we rebuild
                
                # Remove from metadata
                metadata.pop(i)
                
                # If there are no more images, create an empty index
                if len(metadata) == 0:
                    self.image_indices[dataset_id] = faiss.IndexFlatIP(VECTOR_DIMENSION)
                else:
                    # Otherwise, rebuild the index
                    new_index = faiss.IndexFlatIP(VECTOR_DIMENSION)
                    for img_meta in metadata:
                        img_path = img_meta["path"]
                        embedding = self.compute_image_embedding(img_path)
                        new_index.add(np.array([embedding], dtype=np.float32))
                    
                    self.image_indices[dataset_id] = new_index
                
                # Save updated index and metadata
                self._save_dataset_index(dataset_id)
                return True
                
        return False  # Image not found
    
    def _save_dataset_index(self, dataset_id: str):
        """Save a dataset's index and metadata to disk
        
        Args:
            dataset_id: ID of the dataset
        """
        # Save FAISS index
        index_path = os.path.join(self.indices_dir, f"{dataset_id}_index.faiss")
        faiss.write_index(self.image_indices[dataset_id], index_path)
        
        # Save metadata
        metadata_path = os.path.join(self.indices_dir, f"{dataset_id}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.image_metadata[dataset_id], f)
    
    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset's index and metadata
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            bool: True if successful, False otherwise
        """
        if dataset_id not in self.image_indices:
            return False
            
        # Remove from memory
        if dataset_id in self.image_indices:
            del self.image_indices[dataset_id]
        if dataset_id in self.image_metadata:
            del self.image_metadata[dataset_id]
            
        # Remove files
        index_path = os.path.join(self.indices_dir, f"{dataset_id}_index.faiss")
        metadata_path = os.path.join(self.indices_dir, f"{dataset_id}_metadata.json")
        
        try:
            if os.path.exists(index_path):
                os.remove(index_path)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            return True
        except Exception as e:
            print(f"Error deleting dataset {dataset_id}: {str(e)}")
            return False 