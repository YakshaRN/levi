"""Audio embedding service using CLAP model."""

import torch
import numpy as np
from transformers import ClapModel, ClapProcessor
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class AudioEmbedder:
    """Generate embeddings from audio using CLAP model."""
    
    def __init__(self, model_name: str = "laion/clap-htsat-unfused", device: str = "cpu"):
        """
        Initialize audio embedder.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load CLAP model and processor."""
        try:
            logger.info(f"Loading CLAP model: {self.model_name}")
            self.processor = ClapProcessor.from_pretrained(self.model_name)
            self.model = ClapModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("CLAP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLAP model: {str(e)}")
            raise
    
    def generate_embedding(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        """
        Generate embedding from audio.
        
        Args:
            audio: Audio array
            sr: Sample rate (CLAP expects 48000 Hz)
        
        Returns:
            Embedding vector as numpy array
        """
        try:
            # Resample if necessary
            if sr != 48000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
                sr = 48000
            
            # Process audio
            inputs = self.processor(
                audios=audio,
                sampling_rate=sr,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embedding
            with torch.no_grad():
                audio_embed = self.model.get_audio_features(**inputs)
            
            # Convert to numpy and normalize
            embedding = audio_embed.cpu().numpy().flatten()
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
        
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise
    
    def generate_text_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding from text description.
        
        Args:
            text: Text description
        
        Returns:
            Embedding vector as numpy array
        """
        try:
            # Process text
            inputs = self.processor(
                text=[text],
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embedding
            with torch.no_grad():
                text_embed = self.model.get_text_features(**inputs)
            
            # Convert to numpy and normalize
            embedding = text_embed.cpu().numpy().flatten()
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
        
        except Exception as e:
            logger.error(f"Failed to generate text embedding: {str(e)}")
            raise
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
        
        Returns:
            Similarity score (0-1)
        """
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)
    
    def find_similar_embeddings(self, query_embedding: np.ndarray, 
                               candidate_embeddings: List[np.ndarray],
                               top_k: int = 5) -> List[tuple]:
        """
        Find most similar embeddings from candidates.
        
        Args:
            query_embedding: Query embedding
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return
        
        Returns:
            List of (index, similarity_score) tuples
        """
        similarities = []
        for idx, candidate in enumerate(candidate_embeddings):
            sim = self.compute_similarity(query_embedding, candidate)
            similarities.append((idx, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.model.config.projection_dim
    
    def save_embedding(self, embedding: np.ndarray, filepath: str):
        """
        Save embedding to file.
        
        Args:
            embedding: Embedding array
            filepath: Output file path
        """
        np.save(filepath, embedding)
    
    def load_embedding(self, filepath: str) -> np.ndarray:
        """
        Load embedding from file.
        
        Args:
            filepath: Input file path
        
        Returns:
            Embedding array
        """
        return np.load(filepath)
