import os
import json
import threading
import time
from typing import Dict, List, Optional, Any

class DatasetMetadataCache:
    """Cache for dataset metadata to avoid repeated file I/O"""
    
    def __init__(self, ttl_seconds: int = 300):  # 5 minute TTL
        self.cache: Dict[str, Dict] = {}
        self.cache_timestamps: Dict[str, float] = {}
        self.ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        print(f"[Dataset Cache] Initialized with {ttl_seconds}s TTL")
    
    def _is_expired(self, cache_key: str) -> bool:
        """Check if cache entry has expired"""
        if cache_key not in self.cache_timestamps:
            return True
        return time.time() - self.cache_timestamps[cache_key] > self.ttl_seconds
    
    def _get_cache_key(self, user_id: str, dataset_id: Optional[str] = None) -> str:
        """Generate cache key"""
        if dataset_id:
            return f"dataset_{user_id}_{dataset_id}"
        return f"datasets_{user_id}"
    
    def get_datasets(self, user_id: str) -> Optional[List[Dict]]:
        """Get cached datasets for a user"""
        cache_key = self._get_cache_key(user_id)
        
        with self._lock:
            if cache_key in self.cache and not self._is_expired(cache_key):
                print(f"[Dataset Cache] Cache HIT for user {user_id} datasets")
                return self.cache[cache_key].copy()
            
        print(f"[Dataset Cache] Cache MISS for user {user_id} datasets")
        return None
    
    def set_datasets(self, user_id: str, datasets: List[Dict]) -> None:
        """Cache datasets for a user"""
        cache_key = self._get_cache_key(user_id)
        
        with self._lock:
            self.cache[cache_key] = datasets.copy()
            self.cache_timestamps[cache_key] = time.time()
            print(f"[Dataset Cache] Cached {len(datasets)} datasets for user {user_id}")
    
    def get_dataset_metadata(self, user_id: str, dataset_id: str) -> Optional[Dict]:
        """Get cached metadata for a specific dataset"""
        cache_key = self._get_cache_key(user_id, dataset_id)
        
        with self._lock:
            if cache_key in self.cache and not self._is_expired(cache_key):
                print(f"[Dataset Cache] Cache HIT for dataset {dataset_id}")
                return self.cache[cache_key].copy()
            
        print(f"[Dataset Cache] Cache MISS for dataset {dataset_id}")
        return None
    
    def set_dataset_metadata(self, user_id: str, dataset_id: str, metadata: Dict) -> None:
        """Cache metadata for a specific dataset"""
        cache_key = self._get_cache_key(user_id, dataset_id)
        
        with self._lock:
            self.cache[cache_key] = metadata.copy()
            self.cache_timestamps[cache_key] = time.time()
            print(f"[Dataset Cache] Cached metadata for dataset {dataset_id}")
    
    def invalidate_user(self, user_id: str) -> None:
        """Invalidate all cache entries for a user"""
        with self._lock:
            keys_to_remove = [key for key in self.cache.keys() if key.startswith(f"datasets_{user_id}") or key.startswith(f"dataset_{user_id}_")]
            for key in keys_to_remove:
                del self.cache[key]
                del self.cache_timestamps[key]
            print(f"[Dataset Cache] Invalidated {len(keys_to_remove)} entries for user {user_id}")
    
    def invalidate_dataset(self, user_id: str, dataset_id: str) -> None:
        """Invalidate cache for a specific dataset"""
        cache_key = self._get_cache_key(user_id, dataset_id)
        datasets_key = self._get_cache_key(user_id)
        
        with self._lock:
            # Remove specific dataset metadata
            if cache_key in self.cache:
                del self.cache[cache_key]
                del self.cache_timestamps[cache_key]
            
            # Also invalidate the datasets list for this user
            if datasets_key in self.cache:
                del self.cache[datasets_key]
                del self.cache_timestamps[datasets_key]
                
            print(f"[Dataset Cache] Invalidated cache for dataset {dataset_id}")
    
    def cleanup_expired(self) -> None:
        """Remove expired entries from cache"""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, timestamp in self.cache_timestamps.items()
                if current_time - timestamp > self.ttl_seconds
            ]
            
            for key in expired_keys:
                del self.cache[key]
                del self.cache_timestamps[key]
                
            if expired_keys:
                print(f"[Dataset Cache] Cleaned up {len(expired_keys)} expired entries")

# Global cache instance
dataset_cache = DatasetMetadataCache() 