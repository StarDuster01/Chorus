import os
import chromadb
from chromadb.utils import embedding_functions
import threading
from typing import Optional

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
        print("[ChromaDB Pool] Connection pool initialized")
    
    def get_client(self):
        """Get the ChromaDB client (creates if doesn't exist)"""
        if self._client is None:
            chroma_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
            os.makedirs(chroma_db_path, exist_ok=True)
            self._client = chromadb.PersistentClient(path=chroma_db_path)
            print(f"[ChromaDB Pool] Created new client at {chroma_db_path}")
        return self._client
    
    def get_embedding_function(self):
        """Get the OpenAI embedding function (creates if doesn't exist)"""
        if self._embedding_function is None:
            self._embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="text-embedding-ada-002"
            )
            print("[ChromaDB Pool] Created embedding function")
        return self._embedding_function
    
    def get_collection(self, collection_name: str):
        """Get or create a collection"""
        client = self.get_client()
        embedding_function = self.get_embedding_function()
        return client.get_or_create_collection(
            name=collection_name, 
            embedding_function=embedding_function
        )

# Global instance
chroma_pool = ChromaDBConnectionPool() 