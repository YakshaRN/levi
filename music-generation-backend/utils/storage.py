"""Local file storage utilities."""

import os
import shutil
import json
from typing import Optional, Dict
from datetime import datetime
import uuid


class LocalStorage:
    """Handle local file storage operations."""
    
    def __init__(self, base_dir: str):
        """
        Initialize local storage.
        
        Args:
            base_dir: Base directory for storage
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def save_file(self, source_path: str, filename: Optional[str] = None) -> str:
        """
        Save a file to storage.
        
        Args:
            source_path: Path to source file
            filename: Optional custom filename
        
        Returns:
            Saved file path
        """
        if filename is None:
            filename = os.path.basename(source_path)
        
        destination = os.path.join(self.base_dir, filename)
        shutil.copy2(source_path, destination)
        return destination
    
    def get_file_path(self, filename: str) -> str:
        """
        Get full path to file.
        
        Args:
            filename: Filename
        
        Returns:
            Full file path
        """
        return os.path.join(self.base_dir, filename)


class MetadataStore:
    """Store and retrieve metadata as JSON files."""
    
    def __init__(self, base_dir: str):
        """
        Initialize metadata store.
        
        Args:
            base_dir: Base directory for metadata files
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def save_metadata(self, key: str, data: Dict) -> str:
        """
        Save metadata to JSON file.
        
        Args:
            key: Unique key (will be used as filename)
            data: Dictionary data to save
        
        Returns:
            Path to saved metadata file
        """
        filename = f"{key}.json"
        filepath = os.path.join(self.base_dir, filename)
        
        # Add timestamp
        data['updated_at'] = datetime.utcnow().isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def load_metadata(self, key: str) -> Optional[Dict]:
        """
        Load metadata from JSON file.
        
        Args:
            key: Unique key
        
        Returns:
            Dictionary data or None if not found
        """
        filename = f"{key}.json"
        filepath = os.path.join(self.base_dir, filename)
        
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, 'r') as f:
            return json.load(f)


def generate_unique_id() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())
