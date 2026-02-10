"""Audio processing utilities."""

import librosa
import soundfile as sf
import numpy as np
from typing import Tuple, Optional
import os


def load_audio(file_path: str, sr: int = 22050) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        file_path: Path to audio file
        sr: Target sample rate (default: 22050)
    
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr, mono=True)
        return audio, sample_rate
    except Exception as e:
        raise ValueError(f"Failed to load audio file: {str(e)}")


def get_audio_info(file_path: str) -> dict:
    """
    Extract basic audio file information.
    
    Args:
        file_path: Path to audio file
    
    Returns:
        Dictionary containing audio metadata
    """
    try:
        info = sf.info(file_path)
        return {
            "duration": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "format": info.format,
            "subtype": info.subtype,
            "frames": info.frames
        }
    except Exception as e:
        raise ValueError(f"Failed to get audio info: {str(e)}")


def extract_audio_features(audio: np.ndarray, sr: int = 22050) -> dict:
    """
    Extract audio features for analysis.
    
    Args:
        audio: Audio array
        sr: Sample rate
    
    Returns:
        Dictionary of audio features
    """
    features = {}
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    
    features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
    features['spectral_centroid_std'] = float(np.std(spectral_centroids))
    features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
    features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
    
    # Rhythm features
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    features['tempo'] = float(tempo)
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    features['zero_crossing_rate_mean'] = float(np.mean(zcr))
    
    # MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
    
    # Chroma features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features['chroma_mean'] = float(np.mean(chroma))
    
    # RMS energy
    rms = librosa.feature.rms(y=audio)[0]
    features['rms_mean'] = float(np.mean(rms))
    features['rms_std'] = float(np.std(rms))
    
    return features


def save_audio(audio: np.ndarray, file_path: str, sr: int = 32000):
    """
    Save audio array to file.
    
    Args:
        audio: Audio array
        file_path: Output file path
        sr: Sample rate
    """
    try:
        sf.write(file_path, audio, sr)
    except Exception as e:
        raise ValueError(f"Failed to save audio: {str(e)}")


def validate_audio_file(file_path: str, max_duration: int = 300) -> bool:
    """
    Validate audio file.
    
    Args:
        file_path: Path to audio file
        max_duration: Maximum allowed duration in seconds
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    if not os.path.exists(file_path):
        raise ValueError("File does not exist")
    
    try:
        info = get_audio_info(file_path)
        
        if info['duration'] > max_duration:
            raise ValueError(f"Audio duration ({info['duration']}s) exceeds maximum ({max_duration}s)")
        
        if info['duration'] < 1:
            raise ValueError("Audio duration must be at least 1 second")
        
        return True
    
    except Exception as e:
        raise ValueError(f"Invalid audio file: {str(e)}")


def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """
    Normalize audio to target dB level.
    
    Args:
        audio: Audio array
        target_db: Target dB level
    
    Returns:
        Normalized audio array
    """
    # Calculate current RMS
    rms = np.sqrt(np.mean(audio ** 2))
    
    if rms == 0:
        return audio
    
    # Calculate target RMS from dB
    target_rms = 10 ** (target_db / 20)
    
    # Normalize
    normalized = audio * (target_rms / rms)
    
    # Prevent clipping
    max_val = np.abs(normalized).max()
    if max_val > 1.0:
        normalized = normalized / max_val * 0.95
    
    return normalized
