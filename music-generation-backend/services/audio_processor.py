"""Audio processing service for feature extraction and preprocessing."""

import numpy as np
import librosa
from typing import Dict, Tuple
import os
from utils.audio_utils import (
    load_audio, 
    get_audio_info, 
    extract_audio_features,
    validate_audio_file,
    normalize_audio
)


class AudioProcessor:
    """Handle audio processing operations."""
    
    def __init__(self, target_sr: int = 32000):
        """
        Initialize audio processor.
        
        Args:
            target_sr: Target sample rate for processing
        """
        self.target_sr = target_sr
    
    def process_audio(self, file_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Process audio file and extract features.
        
        Args:
            file_path: Path to audio file
        
        Returns:
            Tuple of (processed_audio, metadata)
        """
        # Validate file
        validate_audio_file(file_path)
        
        # Get audio info
        info = get_audio_info(file_path)
        
        # Load and resample audio
        audio, sr = load_audio(file_path, sr=self.target_sr)
        
        # Normalize audio
        audio = normalize_audio(audio)
        
        # Extract features
        features = extract_audio_features(audio, sr)
        
        # Prepare metadata
        metadata = {
            "duration": info["duration"],
            "sample_rate": sr,
            "channels": info["channels"],
            "format": info["format"],
            "features": features,
            "processed_shape": audio.shape,
        }
        
        return audio, metadata
    
    def prepare_for_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Prepare audio for embedding generation.
        
        Args:
            audio: Audio array
            sr: Sample rate
        
        Returns:
            Processed audio ready for embedding
        """
        # Ensure audio is not too long (max 30 seconds for embedding)
        max_samples = 30 * sr
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        # Pad if too short (min 5 seconds)
        min_samples = 5 * sr
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)), mode='constant')
        
        return audio
