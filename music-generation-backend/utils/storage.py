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
    
    def save_bytes(self, data: bytes, filename: str) -> str:
        """
        Save bytes data to storage.
        
        Args:
            data: Bytes data
            filename: Output filename
        
        Returns:
            Saved file path
        """
        destination = os.path.join(self.base_dir, filename)
        with open(destination, 'wb') as f:
            f.write(data)
        return destination
    
    def load_file(self, filename: str) -> bytes:
        """
        Load file as bytes.
        
        Args:
            filename: Filename to load
        
        Returns:
            File contents as bytes
        """
        file_path = os.path.join(self.base_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {filename}")
        
        with open(file_path, 'rb') as f:
            return f.read()
    
    def delete_file(self, filename: str) -> bool:
        """
        Delete a file from storage.
        
        Args:
            filename: Filename to delete
        
        Returns:
            True if deleted, False if not found
        """
        file_path = os.path.join(self.base_dir, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
    
    def file_exists(self, filename: str) -> bool:
        """
        Check if file exists.
        
        Args:
            filename: Filename to check
        
        Returns:
            True if exists
        """
        return os.path.exists(os.path.join(self.base_dir, filename))
    
    def get_file_path(self, filename: str) -> str:
        """
        Get full path to file.
        
        Args:
            filename: Filename
        
        Returns:
            Full file path
        """
        return os.path.join(self.base_dir, filename)
    
    def list_files(self, extension: Optional[str] = None) -> list:
        """
        List all files in storage.
        
        Args:
            extension: Optional file extension filter
        
        Returns:
            List of filenames
        """
        files = os.listdir(self.base_dir)
        if extension:
            files = [f for f in files if f.endswith(extension)]
        return files


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
    
    def delete_metadata(self, key: str) -> bool:
        """
        Delete metadata file.
        
        Args:
            key: Unique key
        
        Returns:
            True if deleted
        """
        filename = f"{key}.json"
        filepath = os.path.join(self.base_dir, filename)
        
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False
    
    def exists(self, key: str) -> bool:
        """
        Check if metadata exists.
        
        Args:
            key: Unique key
        
        Returns:
            True if exists
        """
        filename = f"{key}.json"
        return os.path.exists(os.path.join(self.base_dir, filename))
    
    def list_keys(self) -> list:
        """
        List all metadata keys.
        
        Returns:
            List of keys
        """
        files = os.listdir(self.base_dir)
        return [f.replace('.json', '') for f in files if f.endswith('.json')]


def generate_unique_id() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())
