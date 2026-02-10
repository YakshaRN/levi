"""Audio embedding service using CLAP model."""

import torch
import numpy as np
from transformers import ClapModel, ClapProcessor
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
