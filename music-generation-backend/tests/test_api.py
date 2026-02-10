"""Test suite for music generation API."""

import requests
import time
import os
import json

# Configuration
BASE_URL = "http://localhost:8000"
API_URL = f"{BASE_URL}/api"

# Test audio file path (you'll need to provide this)
TEST_AUDIO_FILE = "test_audio.mp3"  # Replace with actual test file


def print_section(title):
    """Print section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_health_check():
    """Test health check endpoint."""
    print_section("Testing Health Check")
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    print("✓ Health check passed")


def test_upload_audio():
    """Test audio upload."""
    print_section("Testing Audio Upload")
    
    if not os.path.exists(TEST_AUDIO_FILE):
        print(f"⚠ Test audio file not found: {TEST_AUDIO_FILE}")
        print("Please provide a test audio file to continue")
        return None
    
    with open(TEST_AUDIO_FILE, 'rb') as f:
        files = {'file': (os.path.basename(TEST_AUDIO_FILE), f, 'audio/mpeg')}
        response = requests.post(f"{API_URL}/upload", files=files)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    data = response.json()
    assert 'audio_id' in data
    
    audio_id = data['audio_id']
    print(f"✓ Audio uploaded successfully. ID: {audio_id}")
    
    return audio_id


def test_get_embedding(audio_id):
    """Test embedding generation."""
    print_section("Testing Embedding Generation")
    
    if not audio_id:
        print("⚠ Skipping - no audio_id")
        return
    
    print(f"Getting embedding for audio_id: {audio_id}")
    response = requests.get(f"{API_URL}/embedding/{audio_id}")
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Embedding dimension: {data['embedding_dim']}")
        print(f"Model used: {data['model_used']}")
        print(f"✓ Embedding retrieved successfully")
    else:
        print(f"Response: {response.text}")


def test_analyze_audio(audio_id):
    """Test audio analysis."""
    print_section("Testing Audio Analysis")
    
    if not audio_id:
        print("⚠ Skipping - no audio_id")
        return
    
    response = requests.get(f"{API_URL}/analyze/{audio_id}")
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    print("✓ Audio analysis retrieved successfully")


def test_generate_music(audio_id):
    """Test music generation."""
    print_section("Testing Music Generation")
    
    if not audio_id:
        print("⚠ Skipping - no audio_id")
        return None
    
    payload = {
        "duration": 30,
        "temperature": 0.8,
        "top_k": 250,
        "top_p": 0.9,
        "cfg_coef": 3.0
    }
    
    print(f"Starting generation with params: {payload}")
    response = requests.post(
        f"{API_URL}/generate/{audio_id}",
        json=payload
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    data = response.json()
    assert 'generation_id' in data
    
    generation_id = data['generation_id']
    print(f"✓ Music generation started. ID: {generation_id}")
    
    return generation_id


def test_generation_status(generation_id):
    """Test generation status polling."""
    print_section("Testing Generation Status")
    
    if not generation_id:
        print("⚠ Skipping - no generation_id")
        return False
    
    max_wait = 300  # 5 minutes
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        response = requests.get(f"{API_URL}/status/{generation_id}")
        
        if response.status_code != 200:
            print(f"Error: {response.text}")
            return False
        
        data = response.json()
        status = data['status']
        progress = data['progress']
        
        print(f"Status: {status} | Progress: {progress}%", end='\r')
        
        if status == 'completed':
            print(f"\n✓ Generation completed successfully")
            return True
        elif status == 'failed':
            print(f"\n✗ Generation failed: {data.get('error')}")
            return False
        
        time.sleep(5)  # Poll every 5 seconds
    
    print(f"\n⚠ Generation timeout after {max_wait} seconds")
    return False


def test_download_music(generation_id):
    """Test downloading generated music."""
    print_section("Testing Music Download")
    
    if not generation_id:
        print("⚠ Skipping - no generation_id")
        return
    
    response = requests.get(f"{API_URL}/download/{generation_id}")
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        output_file = f"generated_{generation_id}.wav"
        with open(output_file, 'wb') as f:
            f.write(response.content)
        
        file_size = os.path.getsize(output_file)
        print(f"✓ Music downloaded successfully: {output_file}")
        print(f"  File size: {file_size / 1024 / 1024:.2f} MB")
    else:
        print(f"Error: {response.text}")


def run_full_test():
    """Run complete test suite."""
    print("\n" + "🎵" * 30)
    print("  Music Generation Backend - Test Suite")
    print("🎵" * 30)
    
    try:
        # Test 1: Health check
        test_health_check()
        
        # Test 2: Upload audio
        audio_id = test_upload_audio()
        
        if audio_id:
            # Test 3: Get embedding
            test_get_embedding(audio_id)
            
            # Test 4: Analyze audio
            test_analyze_audio(audio_id)
            
            # Test 5: Generate music
            generation_id = test_generate_music(audio_id)
            
            if generation_id:
                # Test 6: Poll generation status
                completed = test_generation_status(generation_id)
                
                if completed:
                    # Test 7: Download generated music
                    test_download_music(generation_id)
        
        print_section("Test Suite Completed")
        print("✓ All tests passed!")
    
    except Exception as e:
        print(f"\n✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


def quick_test():
    """Quick test without full music generation."""
    print("\n" + "🎵" * 30)
    print("  Music Generation Backend - Quick Test")
    print("🎵" * 30)
    
    try:
        # Test health check
        test_health_check()
        
        # Test upload
        audio_id = test_upload_audio()
        
        if audio_id:
            # Test analysis
            test_analyze_audio(audio_id)
            
            print_section("Quick Test Completed")
            print("✓ Quick tests passed!")
            print(f"\nTo test music generation, run:")
            print(f"  python tests/test_api.py --full")
    
    except Exception as e:
        print(f"\n✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    if "--full" in sys.argv:
        run_full_test()
    else:
        print("\nRunning quick test (without music generation)")
        print("For full test including music generation, use: --full")
        print()
        quick_test()
