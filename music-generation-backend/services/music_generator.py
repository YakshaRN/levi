"""Music generation service using MusicGen model."""

import torch
import numpy as np
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import logging
import os

logger = logging.getLogger(__name__)


class MusicGenerator:
    """Generate music using MusicGen model."""
    
    def __init__(self, model_name: str = "facebook/musicgen-small", device: str = "cpu"):
        """
        Initialize music generator.
        
        Args:
            model_name: Model identifier (small, medium, large, or melody)
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load MusicGen model."""
        try:
            logger.info(f"Loading MusicGen model: {self.model_name}")
            # Extract model size from name
            if "small" in self.model_name:
                size = "small"
            elif "medium" in self.model_name:
                size = "medium"
            elif "large" in self.model_name:
                size = "large"
            elif "melody" in self.model_name:
                size = "melody"
            else:
                size = "small"
            
            self.model = MusicGen.get_pretrained(size, device=self.device)
            logger.info(f"MusicGen model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load MusicGen model: {str(e)}")
            raise
    
    def generate_from_audio(self, 
                          audio: np.ndarray,
                          sr: int,
                          duration: int = 30,
                          temperature: float = 1.0,
                          top_k: int = 250,
                          top_p: float = 0.0,
                          cfg_coef: float = 3.0) -> np.ndarray:
        """
        Generate music conditioned on input audio (melody conditioning).
        
        Args:
            audio: Input audio array
            sr: Sample rate of input audio
            duration: Duration of generated audio in seconds
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            cfg_coef: Classifier-free guidance coefficient
        
        Returns:
            Generated audio array
        """
        try:
            # Set generation parameters
            self.model.set_generation_params(
                duration=duration,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                cfg_coef=cfg_coef
            )
            
            # Prepare audio for conditioning
            # MusicGen expects audio at 32kHz
            if sr != 32000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=32000)
            
            # Convert to torch tensor (batch, channels, samples) for conditioning
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
            audio_tensor = audio_tensor.to(self.device)
            
            logger.info(f"Generating similar variation (duration: {duration}s, temp: {temperature})")
            
            # Generate music: melody model uses input audio for similar variations; others fall back to text
            with torch.no_grad():
                if hasattr(self.model, 'generate_continuation'):
                    # Melody model: conditions on input audio → similar variations
                    generated = self.model.generate_continuation(
                        audio_tensor,
                        prompt_sample_rate=32000,
                        progress=True
                    )
                else:
                    generated = self.model.generate(
                        descriptions=["instrumental music"],
                        progress=True
                    )
            
            # Convert to numpy
            generated_audio = generated.cpu().numpy().squeeze()
            
            logger.info("Music generation completed")
            return generated_audio
        
        except Exception as e:
            logger.error(f"Failed to generate music: {str(e)}")
            raise
    
    def save_audio(self, audio: np.ndarray, output_path: str, sr: int = 32000):
        """
        Save generated audio to file.
        
        Args:
            audio: Audio array
            output_path: Output file path (without extension)
            sr: Sample rate
        """
        try:
            # Remove extension if present
            output_path = os.path.splitext(output_path)[0]
            
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            
            # Save using audiocraft's audio_write (saves as WAV)
            audio_write(
                output_path,
                audio_tensor,
                sr,
                strategy="loudness",
                loudness_compressor=True
            )
            
            logger.info(f"Audio saved to {output_path}.wav")
            return f"{output_path}.wav"
        
        except Exception as e:
            logger.error(f"Failed to save audio: {str(e)}")
            raise
