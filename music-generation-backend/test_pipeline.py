#!/usr/bin/env python3
"""
Simple standalone test for audio embedding and music generation.
This script demonstrates the core functionality without the API.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.audio_processor import AudioProcessor
from services.embedder import AudioEmbedder
from services.music_generator import MusicGenerator
import numpy as np


def test_embedding_and_generation(audio_file_path: str):
    """
    Test the complete pipeline: audio -> embedding -> music generation.
    
    Args:
        audio_file_path: Path to input audio file
    """
    print("=" * 60)
    print("Music Generation Pipeline Test")
    print("=" * 60)
    
    # Step 1: Process Audio
    print("\n[1/4] Processing audio...")
    processor = AudioProcessor(target_sr=32000)
    audio, metadata = processor.process_audio(audio_file_path)
    
    print(f"  ✓ Duration: {metadata['duration']:.2f}s")
    print(f"  ✓ Sample Rate: {metadata['sample_rate']} Hz")
    print(f"  ✓ Audio shape: {audio.shape}")
    print(f"  ✓ Features extracted: {len(metadata['features'])} features")
    
    # Step 2: Generate Embedding
    print("\n[2/4] Generating audio embedding...")
    print("  (This may take a minute on first run - downloading model)")
    
    embedder = AudioEmbedder(model_name="laion/clap-htsat-unfused", device="cpu")
    
    # Prepare audio for embedding
    audio_for_embedding = processor.prepare_for_embedding(audio, metadata['sample_rate'])
    
    embedding = embedder.generate_embedding(audio_for_embedding, sr=32000)
    
    print(f"  ✓ Embedding dimension: {len(embedding)}")
    print(f"  ✓ Embedding norm: {np.linalg.norm(embedding):.4f}")
    print(f"  ✓ Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")
    
    # Save embedding
    embedding_path = "data/processed/test_embedding.npy"
    np.save(embedding_path, embedding)
    print(f"  ✓ Embedding saved to: {embedding_path}")
    
    # Step 3: Generate Music
    print("\n[3/4] Generating similar music...")
    print("  (This may take 2-3 minutes on CPU)")
    
    generator = MusicGenerator(model_name="facebook/musicgen-small", device="cpu")
    
    # Generate music using the original audio as conditioning
    generated_audio = generator.generate_from_audio(
        audio=audio,
        sr=metadata['sample_rate'],
        duration=30,  # 30 seconds
        temperature=0.8,
        top_k=250,
        cfg_coef=3.0
    )
    
    print(f"  ✓ Generated audio shape: {generated_audio.shape}")
    print(f"  ✓ Generated audio duration: ~30 seconds")
    
    # Step 4: Save Generated Music
    print("\n[4/4] Saving generated music...")
    
    output_path = "data/generated/test_generated"
    saved_path = generator.save_audio(generated_audio, output_path)
    
    print(f"  ✓ Generated music saved to: {saved_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Pipeline Test Completed Successfully!")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  - Embedding: {embedding_path}")
    print(f"  - Generated music: {saved_path}")
    print(f"\nYou can now:")
    print(f"  1. Listen to the generated music")
    print(f"  2. Compare it with the original")
    print(f"  3. Try different generation parameters")


def create_test_audio():
    """Create a simple test audio file if none exists."""
    import numpy as np
    from utils.audio_utils import save_audio
    
    print("Creating test audio file...")
    
    # Generate a simple sine wave melody
    duration = 10  # seconds
    sr = 32000
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create a simple melody (C-E-G-C major chord progression)
    frequencies = [261.63, 329.63, 392.00, 523.25]  # Hz
    audio = np.zeros_like(t)
    
    notes_per_second = 2
    samples_per_note = int(sr / notes_per_second)
    
    for i, freq in enumerate(frequencies * (duration * notes_per_second // 4)):
        start = i * samples_per_note
        end = min(start + samples_per_note, len(t))
        audio[start:end] += 0.3 * np.sin(2 * np.pi * freq * t[start:end])
    
    # Add some harmonics for richness
    audio += 0.1 * np.sin(4 * np.pi * 440 * t)
    
    # Normalize
    audio = audio / np.abs(audio).max() * 0.8
    
    output_file = "data/uploads/test_audio.wav"
    save_audio(audio, output_file, sr)
    
    print(f"✓ Test audio created: {output_file}")
    return output_file


if __name__ == "__main__":
    print("\n🎵 Music Generation Pipeline - Standalone Test 🎵\n")
    
    # Check if test audio file is provided
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        # Look for test audio in uploads
        test_files = [
            "data/uploads/test_audio.wav",
            "data/uploads/test_audio.mp3"
        ]
        
        audio_file = None
        for f in test_files:
            if os.path.exists(f):
                audio_file = f
                break
        
        if audio_file is None:
            print("No test audio file found. Creating one...")
            audio_file = create_test_audio()
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found: {audio_file}")
        print("\nUsage:")
        print(f"  python {sys.argv[0]} <path_to_audio_file>")
        print(f"\nOr place a test file at: data/uploads/test_audio.wav")
        sys.exit(1)
    
    print(f"Using audio file: {audio_file}\n")
    
    try:
        test_embedding_and_generation(audio_file)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
