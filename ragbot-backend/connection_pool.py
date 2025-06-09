import os
import chromadb
from chromadb.utils import embedding_functions
import threading
from typing import Optional
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_collection_name(name: str) -> str:
    """
    Validate and sanitize collection name according to ChromaDB requirements:
    - Length must be between 3 and 63 characters
    - Must start and end with a lowercase letter or digit
    - Can contain dots, dashes, and underscores in between
    - Must not contain two consecutive dots
    - Must not be a valid IP address
    """
    if not name:
        raise ValueError("Collection name cannot be empty")
    
    # Replace any invalid characters with underscores
    sanitized = re.sub(r'[^a-z0-9._-]', '_', name.lower())
    
    # Ensure it starts and ends with alphanumeric
    if not re.match(r'^[a-z0-9]', sanitized):
        sanitized = 'ds_' + sanitized
    if not re.match(r'[a-z0-9]$', sanitized):
        sanitized = sanitized + '_ds'
    
    # Remove consecutive dots
    sanitized = re.sub(r'\.\.+', '.', sanitized)
    
    # Ensure length constraints
    if len(sanitized) < 3:
        sanitized = sanitized + '_dataset'
    elif len(sanitized) > 63:
        sanitized = sanitized[:60] + '_ds'
    
    # Check if it's an IP address pattern and modify if so
    ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    if re.match(ip_pattern, sanitized):
        sanitized = 'dataset_' + sanitized.replace('.', '_')
    
    logger.info(f"[ChromaDB Pool] Collection name '{name}' sanitized to '{sanitized}'")
    return sanitized

class ChromaDBConnectionPool:
    """Singleton connection pool for ChromaDB to avoid creating multiple clients"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ChromaDBConnectionPool, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self._client = None
        self._embedding_function = None
        self._initialized = True
        logger.info("[ChromaDB Pool] Connection pool initialized")
        
        # MANDATORY: Pre-load embedding function during initialization to avoid blocking during dataset creation
        # App will not start until this completes successfully
        logger.info("[ChromaDB Pool] 🔄 Pre-loading embedding function (REQUIRED FOR STARTUP)...")
        try:
            self._initialize_embedding_function()
            if self._embedding_function is None:
                raise Exception("Failed to initialize embedding function")
        except Exception as e:
            logger.error(f"[ChromaDB Pool] ❌ CRITICAL: Failed to pre-load embedding function: {str(e)}")
            logger.error(f"[ChromaDB Pool] ❌ APP CANNOT START WITHOUT EMBEDDING FUNCTION")
            raise RuntimeError(f"Failed to initialize embedding function: {str(e)}")
    
    def _initialize_embedding_function(self):
        """Initialize the embedding function (called during startup)"""
        if self._embedding_function is None:
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"[ChromaDB Pool] Using device: {device}")
            
            if torch.cuda.is_available():
                logger.info(f"[ChromaDB Pool] GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"[ChromaDB Pool] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # Use a very small model for 4GB GPU - all-MiniLM-L12-v2 is smaller than L6
            # Or use paraphrase-MiniLM-L3-v2 which is even smaller (~17MB)
            model_name = "paraphrase-MiniLM-L3-v2"  # Much smaller model
            logger.info(f"[ChromaDB Pool] Loading SentenceTransformer model: {model_name}")
            logger.info(f"[ChromaDB Pool] Model size: ~17MB, optimized for limited GPU memory")
            logger.info(f"[ChromaDB Pool] ⚠️  APP WILL NOT START UNTIL MODEL IS LOADED ⚠️")
            
            try:
                import time
                
                logger.info(f"[ChromaDB Pool] Starting model download/load...")
                start_time = time.time()
                
                self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=model_name,
                    device=device
                )
                
                load_time = time.time() - start_time
                logger.info(f"[ChromaDB Pool] ✅ Successfully loaded SentenceTransformer embedding function on {device} in {load_time:.2f} seconds")
                logger.info(f"[ChromaDB Pool] 🚀 Embedding function ready - app can now start!")
                
            except Exception as e:
                logger.error(f"[ChromaDB Pool] ❌ Failed to load SentenceTransformer: {str(e)}")
                logger.info(f"[ChromaDB Pool] Falling back to default embedding function")
                self._embedding_function = embedding_functions.DefaultEmbeddingFunction()
                logger.info(f"[ChromaDB Pool] ✅ Successfully created fallback embedding function")
                logger.info(f"[ChromaDB Pool] 🚀 Fallback embedding ready - app can now start!")
    
    def get_client(self):
        """Get the ChromaDB client (creates if doesn't exist)"""
        if self._client is None:
            try:
                chroma_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
                os.makedirs(chroma_db_path, exist_ok=True)
                logger.info(f"[ChromaDB Pool] Creating new client at {chroma_db_path}")
                
                self._client = chromadb.PersistentClient(
                    path=chroma_db_path,
                    settings=chromadb.Settings(
                        anonymized_telemetry=False,
                        allow_reset=True,
                        is_persistent=True
                    )
                )
                logger.info("[ChromaDB Pool] Successfully created new client")
            except Exception as e:
                logger.error(f"[ChromaDB Pool] Error creating client: {str(e)}")
                raise
        return self._client
    
    def get_embedding_function(self):
        """Get the GPU-based sentence transformer embedding function (creates if doesn't exist)"""
        if self._embedding_function is None:
            try:
                logger.info("[ChromaDB Pool] Embedding function not pre-loaded, loading now...")
                self._initialize_embedding_function()
            except Exception as e:
                logger.error(f"[ChromaDB Pool] Error creating embedding function: {str(e)}")
                # Fall back to default embedding function if sentence transformers fail
                try:
                    logger.info("[ChromaDB Pool] Falling back to default embedding function")
                    self._embedding_function = embedding_functions.DefaultEmbeddingFunction()
                    logger.info("[ChromaDB Pool] Successfully created fallback embedding function")
                except Exception as fallback_error:
                    logger.error(f"[ChromaDB Pool] Fallback embedding function also failed: {str(fallback_error)}")
                    raise e  # Raise the original error
        return self._embedding_function
    
    def get_or_create_collection(self, collection_name: str, metadata: Optional[dict] = None):
        """Get or create a collection using the new ChromaDB API"""
        try:
            # Validate and sanitize collection name
            sanitized_name = validate_collection_name(collection_name)
            
            logger.info(f"[ChromaDB Pool] Getting/creating collection: {sanitized_name}")
            client = self.get_client()
            embedding_function = self.get_embedding_function()
            
            # Set default metadata with cosine distance
            default_metadata = {"hnsw:space": "cosine"}
            if metadata:
                default_metadata.update(metadata)
            
            # Use get_or_create_collection method which is the recommended approach
            collection = client.get_or_create_collection(
                name=sanitized_name,
                embedding_function=embedding_function,
                metadata=default_metadata
            )
            logger.info(f"[ChromaDB Pool] Successfully got/created collection: {sanitized_name}")
            return collection
            
        except Exception as e:
            logger.error(f"[ChromaDB Pool] Error getting/creating collection {collection_name}: {str(e)}")
            raise
    
    def get_collection(self, collection_name: str):
        """Get an existing collection"""
        try:
            sanitized_name = validate_collection_name(collection_name)
            logger.info(f"[ChromaDB Pool] Getting collection: {sanitized_name}")
            client = self.get_client()
            embedding_function = self.get_embedding_function()
            
            collection = client.get_collection(
                name=sanitized_name,
                embedding_function=embedding_function
            )
            logger.info(f"[ChromaDB Pool] Retrieved collection: {sanitized_name}")
            return collection
        except Exception as e:
            logger.error(f"[ChromaDB Pool] Error getting collection {collection_name}: {str(e)}")
            raise
    
    def delete_collection(self, collection_name: str):
        """Delete a collection"""
        try:
            sanitized_name = validate_collection_name(collection_name)
            logger.info(f"[ChromaDB Pool] Deleting collection: {sanitized_name}")
            client = self.get_client()
            
            client.delete_collection(name=sanitized_name)
            logger.info(f"[ChromaDB Pool] Successfully deleted collection: {sanitized_name}")
        except Exception as e:
            logger.error(f"[ChromaDB Pool] Error deleting collection {collection_name}: {str(e)}")
            raise
    
    def list_collections(self):
        """List all collections"""
        try:
            client = self.get_client()
            collections = client.list_collections()
            logger.info(f"[ChromaDB Pool] Listed {len(collections)} collections")
            return collections
        except Exception as e:
            logger.error(f"[ChromaDB Pool] Error listing collections: {str(e)}")
            raise

# Global instance
chroma_pool = ChromaDBConnectionPool() 