import os
import json
import uuid
import base64
import datetime
from typing import List, Dict, Any, Tuple, Optional
import torch
import gc  # For garbage collection
from PIL import Image
import open_clip
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
import faiss

# Check for GPU FAISS availability
try:
    # Try to import GPU FAISS
    import faiss
    GPU_FAISS_AVAILABLE = hasattr(faiss, 'StandardGpuResources')
    if GPU_FAISS_AVAILABLE:
        print("[GPU] FAISS-GPU is available!")
    else:
        print("[GPU] FAISS-GPU not available, using CPU FAISS")
except ImportError:
    GPU_FAISS_AVAILABLE = False
    print("[GPU] FAISS-GPU not available, using CPU FAISS")

# Configurations
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_CLIP_MODEL = "ViT-B/32"
DEFAULT_CLIP_PRETRAINED = "openai"
DEFAULT_BLIP_MODEL = "Salesforce/blip-image-captioning-base"
VECTOR_DIMENSION = 512  # Dimension for ViT-B/32

# GPU optimization settings
USE_GPU_FAISS = GPU_FAISS_AVAILABLE and torch.cuda.is_available()
BATCH_SIZE = 32 if torch.cuda.is_available() else 8  # Larger batches on GPU

# Model cache directory for persistence
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/code/model_cache")

print(f"[GPU] Device: {DEVICE}")
print(f"[GPU] CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[GPU] GPU Count: {torch.cuda.device_count()}")
    print(f"[GPU] GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"[GPU] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"[GPU] Using GPU FAISS: {USE_GPU_FAISS}")

# Global model manager to avoid reloading models
class GlobalModelManager:
    _instance = None
    _models_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalModelManager, cls).__new__(cls)
            cls._instance.clip_model = None
            cls._instance.clip_preprocess = None
            cls._instance.blip_model = None
            cls._instance.blip_processor = None
            cls._instance.gpu_resources = None
        return cls._instance
    
    def load_models_once(self):
        """Load models once at startup with persistent caching"""
        if self._models_loaded:
            return
            
        try:
            # Create model cache directory
            os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
            
            print(f"[STARTUP] Loading CLIP model on {DEVICE} (cache: {MODEL_CACHE_DIR})...")
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                DEFAULT_CLIP_MODEL, 
                pretrained=DEFAULT_CLIP_PRETRAINED,
                device=DEVICE,
                cache_dir=MODEL_CACHE_DIR
            )
            self.clip_model.eval()
            
            # Enable mixed precision for faster inference on GPU
            if torch.cuda.is_available():
                self.clip_model = self.clip_model.half()  # Use FP16 for faster inference
            
            print(f"[STARTUP] Loading BLIP model on {DEVICE} (cache: {MODEL_CACHE_DIR})...")
            # Set cache directories for HuggingFace models
            blip_cache_dir = os.path.join(MODEL_CACHE_DIR, "blip")
            os.makedirs(blip_cache_dir, exist_ok=True)
            
            self.blip_processor = BlipProcessor.from_pretrained(
                DEFAULT_BLIP_MODEL,
                use_fast=True,
                cache_dir=blip_cache_dir
            )
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                DEFAULT_BLIP_MODEL,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                cache_dir=blip_cache_dir,
                trust_remote_code=False,  # Security: don't execute remote code
                use_safetensors=True  # Use safetensors when available
            ).to(DEVICE)
            
            # Initialize GPU resources for FAISS if available
            if USE_GPU_FAISS:
                try:
                    print(f"[STARTUP] Initializing FAISS GPU resources...")
                    self.gpu_resources = faiss.StandardGpuResources()
                    # Set temporary memory limit (1GB)
                    self.gpu_resources.setTempMemory(1024 * 1024 * 1024)
                    print(f"[STARTUP] ✅ FAISS GPU resources initialized!")
                except Exception as e:
                    print(f"[STARTUP] ⚠️  FAISS GPU initialization failed: {str(e)}, falling back to CPU")
                    self.gpu_resources = None
            
            # Clear GPU cache after loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self._models_loaded = True
            print(f"[STARTUP] ✅ All image processing models loaded successfully! (cached in {MODEL_CACHE_DIR})")
            print(f"[STARTUP] 🚀 GPU optimizations: FP16={torch.cuda.is_available()}, GPU-FAISS={USE_GPU_FAISS}")
            
        except Exception as e:
            print(f"[STARTUP] ❌ Error loading models: {str(e)}")
            self._models_loaded = False
            raise

# Initialize global model manager
global_model_manager = GlobalModelManager()

class ImageProcessor:
    def __init__(self, data_dir: str):
        """Initialize the image processor with models for embedding and captioning
        
        Args:
            data_dir: Base directory for saving indices and metadata
        """
        self.data_dir = data_dir
        self.image_indices = {}  # Dataset ID -> FAISS index
        self.image_metadata = {}  # Dataset ID -> list of metadata dictionaries
        
        # Create directory for indices if it doesn't exist
        self.indices_dir = os.path.join(data_dir, "image_indices")
        os.makedirs(self.indices_dir, exist_ok=True)
        
        # Load indices for existing datasets
        self._load_existing_indices()
    
    def _load_models(self):
        """Use pre-loaded global models"""
        # Models are loaded once at startup via global_model_manager
        pass
    
    def _clear_models(self):
        """Legacy method - models are now managed globally"""
        pass
    
    def _load_existing_indices(self):
        """Load existing FAISS indices and metadata for datasets"""
        if not os.path.exists(self.indices_dir):
            print(f"Creating indices directory at {self.indices_dir}")
            os.makedirs(self.indices_dir, exist_ok=True)
            return
            
        print(f"Loading image indices from {self.indices_dir}")
        # First check all metadata files to ensure we load all datasets
        for filename in os.listdir(self.indices_dir):
            if filename.endswith("_metadata.json"):
                dataset_id = filename.split("_metadata.json")[0]
                metadata_file = os.path.join(self.indices_dir, filename)
                index_path = os.path.join(self.indices_dir, f"{dataset_id}_index.faiss")
                
                try:
                    # Load metadata first
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Ensure each image has the correct dataset_id
                    valid_metadata = []
                    for img_meta in metadata:
                        if not img_meta.get('dataset_id'):
                            img_meta['dataset_id'] = dataset_id
                        # Only include images that belong to this dataset
                        if img_meta.get('dataset_id') == dataset_id:
                            valid_metadata.append(img_meta)
                    
                    # Store metadata even if index loading fails
                    self.image_metadata[dataset_id] = valid_metadata
                    
                    # Try to load index if it exists
                    if os.path.exists(index_path):
                        cpu_index = faiss.read_index(index_path)
                        
                        # Move to GPU if available
                        if USE_GPU_FAISS and global_model_manager.gpu_resources:
                            try:
                                gpu_index = faiss.index_cpu_to_gpu(
                                    global_model_manager.gpu_resources, 0, cpu_index
                                )
                                self.image_indices[dataset_id] = gpu_index
                                print(f"Loaded GPU image index for dataset {dataset_id} with {len(valid_metadata)} images")
                            except Exception as e:
                                print(f"Failed to load GPU index, using CPU: {str(e)}")
                                self.image_indices[dataset_id] = cpu_index
                                print(f"Loaded CPU image index for dataset {dataset_id} with {len(valid_metadata)} images")
                        else:
                            self.image_indices[dataset_id] = cpu_index
                            print(f"Loaded CPU image index for dataset {dataset_id} with {len(valid_metadata)} images")
                    else:
                        print(f"Warning: Metadata found for dataset {dataset_id} but no index file")
                        
                except Exception as e:
                    print(f"Error loading data for {dataset_id}: {str(e)}")
        
        # Check for any orphaned index files without metadata
        for filename in os.listdir(self.indices_dir):
            if filename.endswith("_index.faiss"):
                dataset_id = filename.split("_index.faiss")[0]
                
                # If we already loaded this dataset, skip
                if dataset_id in self.image_metadata:
                    continue
                    
                # Check if metadata file exists
                metadata_file = os.path.join(self.indices_dir, f"{dataset_id}_metadata.json")
                if not os.path.exists(metadata_file):
                    print(f"Warning: Found index for dataset {dataset_id} but no metadata file")
                    continue
                    
                try:
                    # Load index
                    index_path = os.path.join(self.indices_dir, filename)
                    index = faiss.read_index(index_path)
                    
                    # Load metadata
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Filter metadata to only include images for this dataset
                    valid_metadata = []
                    for img_meta in metadata:
                        if not img_meta.get('dataset_id'):
                            img_meta['dataset_id'] = dataset_id
                        if img_meta.get('dataset_id') == dataset_id:
                            valid_metadata.append(img_meta)
                    
                    self.image_indices[dataset_id] = index
                    self.image_metadata[dataset_id] = valid_metadata
                    print(f"Loaded image index for dataset {dataset_id} with {len(valid_metadata)} images")
                except Exception as e:
                    print(f"Error loading index for {dataset_id}: {str(e)}")
        
        # Print summary of loaded datasets
        if self.image_metadata:
            print(f"Loaded image metadata for {len(self.image_metadata)} datasets:")
            for dataset_id, metadata in self.image_metadata.items():
                print(f"  - Dataset {dataset_id}: {len(metadata)} images")
        else:
            print("No image datasets found")
    
    def generate_caption(self, image_path: str) -> str:
        """Generate a caption for an image using BLIP
        
        Args:
            image_path: Path to the image file
            
        Returns:
            str: Generated caption
        """
        try:
            # Use global models (already loaded at startup)
            blip_model = global_model_manager.blip_model
            blip_processor = global_model_manager.blip_processor
            
            if blip_model is None or blip_processor is None:
                return "Models not loaded"
            
            # Load and preprocess image
            raw_image = Image.open(image_path).convert('RGB')
            inputs = blip_processor(raw_image, return_tensors="pt").to(DEVICE)
            
            # Generate caption
            with torch.no_grad():
                outputs = blip_model.generate(**inputs, max_new_tokens=50)
                caption = blip_processor.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up memory after processing
            del inputs, outputs, raw_image
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
                
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
        try:
            # Use global models (already loaded at startup)
            clip_model = global_model_manager.clip_model
            clip_preprocess = global_model_manager.clip_preprocess
            
            if clip_model is None or clip_preprocess is None:
                return np.zeros(VECTOR_DIMENSION, dtype=np.float32)
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = clip_preprocess(image).unsqueeze(0).to(DEVICE)
            
            # Use mixed precision for faster inference on GPU
            if torch.cuda.is_available():
                image_tensor = image_tensor.half()
            
            # Compute embedding
            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        embedding = clip_model.encode_image(image_tensor)
                else:
                    embedding = clip_model.encode_image(image_tensor)
                    
                embedding = embedding.cpu().float().numpy()
                
            # Normalize the embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            # Clean up memory after processing
            del image, image_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
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
        try:
            # Use global models (already loaded at startup)
            clip_model = global_model_manager.clip_model
            
            if clip_model is None:
                return np.zeros(VECTOR_DIMENSION, dtype=np.float32)
            
            # Tokenize and encode text
            with torch.no_grad():
                text_tokens = open_clip.tokenize([text]).to(DEVICE)
                
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        embedding = clip_model.encode_text(text_tokens)
                else:
                    embedding = clip_model.encode_text(text_tokens)
                    
                embedding = embedding.cpu().float().numpy()
                
            # Normalize the embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            # Clean up memory after processing
            del text_tokens
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
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
        print(f"Adding image to dataset {dataset_id}: {image_path}")
        self._load_models()
        
        # Initialize the dataset's index and metadata if needed
        if dataset_id not in self.image_indices:
            print(f"Creating new index for dataset {dataset_id}")
            if USE_GPU_FAISS and global_model_manager.gpu_resources:
                try:
                    # Create GPU index
                    cpu_index = faiss.IndexFlatIP(VECTOR_DIMENSION)
                    self.image_indices[dataset_id] = faiss.index_cpu_to_gpu(
                        global_model_manager.gpu_resources, 0, cpu_index
                    )
                    print(f"Created GPU FAISS index for dataset {dataset_id}")
                except Exception as e:
                    print(f"Failed to create GPU index, falling back to CPU: {str(e)}")
                    self.image_indices[dataset_id] = faiss.IndexFlatIP(VECTOR_DIMENSION)
            else:
                self.image_indices[dataset_id] = faiss.IndexFlatIP(VECTOR_DIMENSION)
            self.image_metadata[dataset_id] = []
        
        # Ensure image path exists
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Create combined metadata
        combined_metadata = metadata.copy() if metadata else {}
        
        # Ensure the dataset_id is set in metadata
        combined_metadata['dataset_id'] = dataset_id
            
        # Use provided image ID if available, otherwise generate one
        image_id = combined_metadata.get("id", str(uuid.uuid4()))
        combined_metadata["id"] = image_id
        
        # Generate image embedding
        try:
            embedding = self.compute_image_embedding(image_path)
        except Exception as e:
            print(f"Error computing embedding for {image_path}: {str(e)}")
            raise
        
        # Generate caption if not already provided
        if "caption" not in combined_metadata:
            try:
                caption = self.generate_caption(image_path)
                combined_metadata["caption"] = caption
            except Exception as e:
                print(f"Error generating caption for {image_path}: {str(e)}")
                combined_metadata["caption"] = "No caption available"
        
        # Ensure basic metadata fields
        if "path" not in combined_metadata:
            combined_metadata["path"] = image_path
        if "original_filename" not in combined_metadata:
            combined_metadata["original_filename"] = os.path.basename(image_path)
        if "created_at" not in combined_metadata:
            combined_metadata["created_at"] = datetime.datetime.utcnow().isoformat()
        
        # Add to index and metadata
        self.image_indices[dataset_id].add(np.array([embedding], dtype=np.float32))
        self.image_metadata[dataset_id].append(combined_metadata)
        
        # Save updated index and metadata
        self._save_dataset_index(dataset_id)
        
        print(f"Successfully added image {image_id} to dataset {dataset_id}")
        return combined_metadata
    
    def search_images(self, dataset_id: str, query: str, top_k: int = 5) -> List[Dict]:
        """Search for images matching a text query
        
        Args:
            dataset_id: ID of the dataset to search
            query: Text query
            top_k: Maximum number of results to return
            
        Returns:
            List[Dict]: List of image metadata for matching images
        """
        # Always reload the latest index and metadata from disk before searching
        print(f"[Image Search] Reloading index and metadata for dataset {dataset_id} before search...")
        self._load_existing_indices()
        # Check if dataset exists
        if dataset_id not in self.image_indices or dataset_id not in self.image_metadata:
            print(f"[Image Search] Dataset {dataset_id} not found in indices or metadata after reload.")
            return []
        # Get the index and metadata
        index = self.image_indices[dataset_id]
        metadata = self.image_metadata[dataset_id]
        if len(metadata) == 0:
            print(f"[Image Search] No images in metadata for dataset {dataset_id} after reload.")
            return []
        # Load models only when needed
        try:
            self._load_models()
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
            print(f"[Image Search] Returning {len(results)} results for query '{query}' in dataset {dataset_id}.")
            return results
        except Exception as e:
            print(f"Error in image search: {str(e)}")
            return []
    
    def remove_image(self, dataset_id: str, image_id: str) -> bool:
        """Remove an image from a dataset
        
        Args:
            dataset_id: ID of the dataset
            image_id: ID of the image to remove
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Check if dataset exists
        if dataset_id not in self.image_metadata:
            print(f"Dataset {dataset_id} not found in image metadata")
            return False
            
        # Find the image in metadata
        metadata = self.image_metadata[dataset_id]
        image_index = -1
        found_img = None
        
        for i, item in enumerate(metadata):
            if item.get("id") == image_id:
                image_index = i
                found_img = item
                break
                
        if image_index == -1:
            print(f"Image {image_id} not found in dataset {dataset_id}")
            # Debug info about available images
            if metadata:
                print(f"Available image IDs: {[img.get('id') for img in metadata[:5]]}...")
            return False
            
        # We need to rebuild the index without this image
        # FAISS doesn't support direct removal, so we rebuild
                
        # Remove from metadata
        metadata.pop(image_index)
        
        # Get the image path for possible file deletion
        image_path = found_img.get("path", "")
        
        # If there are no more images, create an empty index
        if len(metadata) == 0:
            self.image_indices[dataset_id] = faiss.IndexFlatIP(VECTOR_DIMENSION)
            print(f"Removed last image from dataset {dataset_id}, creating empty index")
        else:
            # Otherwise, rebuild the index
            try:
                print(f"Rebuilding index for dataset {dataset_id} with {len(metadata)} images")
                new_index = faiss.IndexFlatIP(VECTOR_DIMENSION)
                
                # Keep track of any images with missing files
                missing_images = []
                
                for i, img_meta in enumerate(metadata):
                    img_path = img_meta.get("path", "")
                    if not img_path or not os.path.exists(img_path):
                        print(f"Warning: Image file not found: {img_path}")
                        missing_images.append(i)
                        continue
                        
                    try:
                        embedding = self.compute_image_embedding(img_path)
                        new_index.add(np.array([embedding], dtype=np.float32))
                    except Exception as e:
                        print(f"Error computing embedding for {img_path}: {str(e)}")
                        missing_images.append(i)
                
                # Remove any images with missing files from metadata
                # We need to remove in reverse order to not mess up indices
                for i in sorted(missing_images, reverse=True):
                    print(f"Removing image with missing file: {metadata[i].get('id')}")
                    metadata.pop(i)
                
                self.image_indices[dataset_id] = new_index
            except Exception as e:
                print(f"Error rebuilding index: {str(e)}")
                # If index rebuilding fails, still save the updated metadata
        
        # Save updated index and metadata
        self._save_dataset_index(dataset_id)
        
        # Try to delete the image file
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
                print(f"Deleted image file: {image_path}")
            except Exception as e:
                print(f"Warning: Could not delete image file {image_path}: {str(e)}")
        
        print(f"Successfully removed image {image_id} from dataset {dataset_id}")
        return True
    
    def _save_dataset_index(self, dataset_id: str):
        """Save a dataset's index and metadata to disk
        
        Args:
            dataset_id: ID of the dataset
        """
        # Save FAISS index
        index_path = os.path.join(self.indices_dir, f"{dataset_id}_index.faiss")
        
        # If it's a GPU index, copy to CPU before saving
        index_to_save = self.image_indices[dataset_id]
        if USE_GPU_FAISS and hasattr(index_to_save, 'index'):
            # This is a GPU index, copy to CPU
            try:
                index_to_save = faiss.index_gpu_to_cpu(index_to_save)
                print(f"Copied GPU index to CPU for saving: {dataset_id}")
            except Exception as e:
                print(f"Warning: Could not copy GPU index to CPU: {str(e)}")
        
        faiss.write_index(index_to_save, index_path)
        
        # Save metadata
        metadata_path = os.path.join(self.indices_dir, f"{dataset_id}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.image_metadata[dataset_id], f)
            f.flush()  # Ensure data is written to disk
            os.fsync(f.fileno())  # Force write to disk
    
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