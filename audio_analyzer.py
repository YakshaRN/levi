"""
Audio Analysis Module for Levitate.
Extracts musical features from audio files using librosa.
"""
import librosa
import numpy as np


def analyze_audio(path: str) -> dict:
    """
    Analyze an audio file and extract musical features.
    
    Args:
        path: Path to the audio file
        
    Returns:
        Dictionary containing extracted features:
        - tempo: BPM value
        - tempo_class: slow/moderate/upbeat/fast
        - energy: minimal/low/medium/high/intense
        - mood: melancholic/emotional/aggressive/tense/euphoric/bright/driving/epic/atmospheric
        - texture: smooth/textured/rhythmic/layered
        - harmonic_ratio: ratio of harmonic to total energy
        - dominant_pitch: 0-11 representing musical key (C to B)
        - brightness: normalized spectral centroid
        - complexity: normalized spectral bandwidth
        - rms: root mean square energy
        - centroid: spectral centroid value
    """
    y, sr = librosa.load(path)

    # Basic features
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    rms = librosa.feature.rms(y=y).mean()

    # Harmonic vs percussive separation
    y_harm, y_perc = librosa.effects.hpss(y)
    harmonic_energy = np.mean(np.abs(y_harm))
    percussive_energy = np.mean(np.abs(y_perc))
    harmonic_ratio = harmonic_energy / (harmonic_energy + percussive_energy + 1e-6)

    # Spectral features
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    
    # Chromagram for tonal analysis
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    dominant_pitch = int(np.argmax(chroma_mean))  # 0-11 representing C to B
    
    # Spectral bandwidth (complexity)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()

    # Classify tempo
    tempo_val = float(tempo)
    tempo_class = _classify_tempo(tempo_val)

    # Energy classification
    energy = _classify_energy(rms)

    # Mood classification
    mood = _classify_mood(
        harmonic_ratio=harmonic_ratio,
        centroid=centroid,
        contrast=contrast,
        zcr=zcr,
        rms=rms,
        percussive_energy=percussive_energy,
        harmonic_energy=harmonic_energy,
        tempo=tempo_val,
        bandwidth=bandwidth
    )

    # Texture classification
    texture = _classify_texture(bandwidth, harmonic_ratio, zcr, percussive_energy, harmonic_energy)

    return {
        "tempo": round(tempo_val, 1),
        "tempo_class": tempo_class,
        "energy": energy,
        "mood": mood,
        "texture": texture,
        "harmonic_ratio": round(harmonic_ratio, 2),
        "dominant_pitch": dominant_pitch,
        "brightness": round(centroid / 1000, 1),
        "complexity": round(bandwidth / 1000, 1),
        "rms": float(rms),
        "centroid": float(centroid)
    }


def _classify_tempo(tempo: float) -> str:
    """Classify tempo into categories."""
    if tempo < 80:
        return "slow"
    elif tempo < 120:
        return "moderate"
    elif tempo < 150:
        return "upbeat"
    else:
        return "fast"


def _classify_energy(rms: float) -> str:
    """Classify energy level based on RMS."""
    if rms < 0.02:
        return "minimal"
    elif rms < 0.05:
        return "low"
    elif rms < 0.15:
        return "medium"
    elif rms < 0.3:
        return "high"
    else:
        return "intense"


def _classify_mood(
    harmonic_ratio: float,
    centroid: float,
    contrast: float,
    zcr: float,
    rms: float,
    percussive_energy: float,
    harmonic_energy: float,
    tempo: float,
    bandwidth: float
) -> str:
    """Classify mood based on multiple audio features."""
    if harmonic_ratio > 0.7 and centroid < 2000 and rms < 0.1:
        return "melancholic"
    elif harmonic_ratio > 0.6 and centroid < 2500:
        return "emotional"
    elif contrast > 30 and zcr > 0.15:
        return "aggressive"
    elif contrast > 20 and zcr > 0.1:
        return "tense"
    elif centroid > 4000 and rms > 0.1:
        return "euphoric"
    elif centroid > 3000:
        return "bright"
    elif percussive_energy > harmonic_energy and tempo > 120:
        return "driving"
    elif bandwidth > 2000 and contrast > 15:
        return "epic"
    else:
        return "atmospheric"


def _classify_texture(
    bandwidth: float,
    harmonic_ratio: float,
    zcr: float,
    percussive_energy: float,
    harmonic_energy: float
) -> str:
    """Classify audio texture."""
    if bandwidth < 1500 and harmonic_ratio > 0.6:
        return "smooth"
    elif zcr > 0.12:
        return "textured"
    elif percussive_energy > harmonic_energy * 1.5:
        return "rhythmic"
    else:
        return "layered"
