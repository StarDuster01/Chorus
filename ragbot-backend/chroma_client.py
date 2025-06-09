import os
import chromadb
from chromadb.utils import embedding_functions
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Global variables - initialized at startup
chroma_client = None
embedding_function = None

# Try to import torch with graceful fallback
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

def initialize_chroma():
    """Initialize ChromaDB client and embedding function at startup"""
    global chroma_client, embedding_function
    
    logger.info("[ChromaDB] Initializing ChromaDB client and embedding function...")
    
    # Initialize ChromaDB client
    try:
        chroma_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
        os.makedirs(chroma_db_path, exist_ok=True)
        
        chroma_client = chromadb.PersistentClient(
            path=chroma_db_path,
            settings=chromadb.Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        )
        logger.info(f"[ChromaDB] Client initialized at {chroma_db_path}")
    except Exception as e:
        logger.error(f"[ChromaDB] Failed to initialize client: {str(e)}")
        raise RuntimeError(f"ChromaDB client initialization failed: {str(e)}")
    
    # Initialize embedding function
    try:
        # Check device availability
        if TORCH_AVAILABLE and torch.cuda.is_available():
            device = "cuda"
            logger.info(f"[ChromaDB] Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"[ChromaDB] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            device = "cpu"
            logger.info(f"[ChromaDB] Using CPU (torch available: {TORCH_AVAILABLE})")
        
        # Load small model for 4GB GPU
        model_name = "paraphrase-MiniLM-L3-v2"  # ~17MB model
        logger.info(f"[ChromaDB] Loading embedding model: {model_name} on {device}")
        
        import time
        start_time = time.time()
        
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name,
            device=device
        )
        
        load_time = time.time() - start_time
        logger.info(f"[ChromaDB] ✅ Embedding function loaded in {load_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"[ChromaDB] Failed to load SentenceTransformer: {str(e)}")
        logger.info(f"[ChromaDB] Falling back to default embedding function")
        
        try:
            embedding_function = embedding_functions.DefaultEmbeddingFunction()
            logger.info(f"[ChromaDB] ✅ Default embedding function loaded")
        except Exception as fallback_error:
            logger.error(f"[ChromaDB] Even fallback failed: {str(fallback_error)}")
            raise RuntimeError(f"All embedding options failed: {str(e)}")
    
    logger.info("[ChromaDB] 🚀 Initialization complete!")

def get_or_create_collection(collection_name: str, metadata: dict = None):
    """Get or create a ChromaDB collection"""
    if chroma_client is None or embedding_function is None:
        raise RuntimeError("ChromaDB not initialized. Call initialize_chroma() first.")
    
    # Simple name sanitization
    sanitized_name = collection_name.lower().replace('-', '_')
    if len(sanitized_name) < 3:
        sanitized_name = sanitized_name + '_ds'
    
    # Set default metadata
    default_metadata = {"hnsw:space": "cosine"}
    if metadata:
        default_metadata.update(metadata)
    
    try:
        collection = chroma_client.get_or_create_collection(
            name=sanitized_name,
            embedding_function=embedding_function,
            metadata=default_metadata
        )
        logger.info(f"[ChromaDB] Got/created collection: {sanitized_name}")
        return collection
    except Exception as e:
        logger.error(f"[ChromaDB] Error with collection {collection_name}: {str(e)}")
        raise

def get_collection(collection_name: str):
    """Get an existing ChromaDB collection"""
    if chroma_client is None or embedding_function is None:
        raise RuntimeError("ChromaDB not initialized. Call initialize_chroma() first.")
    
    sanitized_name = collection_name.lower().replace('-', '_')
    if len(sanitized_name) < 3:
        sanitized_name = sanitized_name + '_ds'
    
    try:
        collection = chroma_client.get_collection(
            name=sanitized_name,
            embedding_function=embedding_function
        )
        logger.info(f"[ChromaDB] Retrieved collection: {sanitized_name}")
        return collection
    except Exception as e:
        logger.error(f"[ChromaDB] Error getting collection {collection_name}: {str(e)}")
        raise

def delete_collection(collection_name: str):
    """Delete a ChromaDB collection"""
    if chroma_client is None:
        raise RuntimeError("ChromaDB not initialized. Call initialize_chroma() first.")
    
    sanitized_name = collection_name.lower().replace('-', '_')
    if len(sanitized_name) < 3:
        sanitized_name = sanitized_name + '_ds'
    
    try:
        chroma_client.delete_collection(name=sanitized_name)
        logger.info(f"[ChromaDB] Deleted collection: {sanitized_name}")
    except Exception as e:
        logger.error(f"[ChromaDB] Error deleting collection {collection_name}: {str(e)}")
        raise

def list_collections():
    """List all ChromaDB collections"""
    if chroma_client is None:
        raise RuntimeError("ChromaDB not initialized. Call initialize_chroma() first.")
    
    try:
        collections = chroma_client.list_collections()
        logger.info(f"[ChromaDB] Listed {len(collections)} collections")
        return collections
    except Exception as e:
        logger.error(f"[ChromaDB] Error listing collections: {str(e)}")
        raise 