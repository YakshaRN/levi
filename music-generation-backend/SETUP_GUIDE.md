# Music Generation Backend - Complete Setup & Testing Guide

## 📋 Project Overview

This backend system converts instrumental music to vector embeddings and generates similar music using AI models. It's designed to be tested locally before AWS deployment.

## 🎯 Core Features

1. **Audio to Vector Conversion**: Uses CLAP model to create embeddings
2. **Music Generation**: Uses MusicGen to create similar instrumental music
3. **REST API**: FastAPI-based endpoints for all operations
4. **Local Testing**: Complete local development environment

## 📦 Project Structure

```
music-generation-backend/
├── app/                      # FastAPI application
│   ├── main.py              # Main app entry point
│   ├── config.py            # Configuration management
│   ├── models.py            # Pydantic models
│   └── api/
│       └── routes.py        # API endpoints
├── services/                 # Core business logic
│   ├── audio_processor.py   # Audio preprocessing
│   ├── embedder.py          # CLAP embedding generation
│   └── music_generator.py   # MusicGen music generation
├── utils/                    # Utility functions
│   ├── audio_utils.py       # Audio processing utilities
│   └── storage.py           # File storage management
├── data/                     # Local data storage
│   ├── uploads/             # Uploaded audio files
│   ├── processed/           # Embeddings and metadata
│   └── generated/           # Generated music files
├── tests/                    # Test scripts
│   └── test_api.py          # API integration tests
├── requirements.txt          # Python dependencies
├── .env                      # Environment configuration
├── start.sh                  # Startup script
└── test_pipeline.py          # Standalone pipeline test
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# System dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y ffmpeg python3-pip

# Python dependencies
pip install -r requirements.txt --break-system-packages
```

### 2. Start the Server

```bash
# Using the startup script
bash start.sh

# Or manually
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Test the Pipeline

```bash
# Standalone test (creates test audio and runs full pipeline)
python test_pipeline.py

# Or with your own audio file
python test_pipeline.py path/to/your/audio.mp3
```

## 🧪 Testing Options

### Option 1: Standalone Pipeline Test (Recommended for First Test)

This tests the core functionality without the API:

```bash
python test_pipeline.py
```

**What it does:**
1. Creates a test audio file (or uses yours)
2. Processes audio and extracts features
3. Generates embedding vector
4. Generates similar music
5. Saves all outputs

**Output files:**
- `data/processed/test_embedding.npy` - Audio embedding
- `data/generated/test_generated.wav` - Generated music

### Option 2: API Testing

Start the server first, then run tests:

```bash
# Terminal 1: Start server
python -m uvicorn app.main:app --reload

# Terminal 2: Run tests
python tests/test_api.py        # Quick test
python tests/test_api.py --full  # Full test with generation
```

### Option 3: Manual API Testing

Using cURL or API client:

```bash
# 1. Upload audio
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@music.mp3"

# Response: {"audio_id": "abc-123-..."}

# 2. Get embedding
curl "http://localhost:8000/api/embedding/abc-123-..."

# 3. Generate music
curl -X POST "http://localhost:8000/api/generate/abc-123-..." \
  -H "Content-Type: application/json" \
  -d '{"duration": 30, "temperature": 0.8}'

# Response: {"generation_id": "xyz-789-..."}

# 4. Check status
curl "http://localhost:8000/api/status/xyz-789-..."

# 5. Download generated music
curl "http://localhost:8000/api/download/xyz-789-..." \
  --output generated.wav
```

## 🔧 Configuration

Edit `.env` file to customize:

```bash
# Use GPU (much faster)
DEVICE=cuda  # Requires CUDA-enabled GPU

# Use different models
GENERATION_MODEL=facebook/musicgen-medium  # Better quality, slower
EMBEDDING_MODEL=laion/clap-htsat-unfused

# Adjust limits
MAX_FILE_SIZE=104857600  # 100MB
MAX_DURATION=300         # 5 minutes
```

## 📊 Technical Details

### Audio to Vector Process

1. **Input**: Audio file (MP3, WAV, FLAC, etc.)
2. **Preprocessing**:
   - Resample to 32kHz
   - Normalize audio
   - Extract features (spectral, rhythm, timbre)
3. **Embedding**:
   - Use CLAP model (Contrastive Language-Audio Pretraining)
   - Generate 512-dimensional vector
   - Normalized L2 norm
4. **Output**: Vector embedding representing audio characteristics

### Vector to Music Process

1. **Input**: Audio file or embedding
2. **Conditioning**:
   - Use original audio as melody conditioning
   - Or generate from embedding characteristics
3. **Generation**:
   - MusicGen model generates audio tokens
   - Decoder converts tokens to waveform
   - Apply post-processing (normalization, loudness)
4. **Output**: Generated audio file (WAV, 32kHz)

### Models Used

**CLAP (Embedding)**
- Model: `laion/clap-htsat-unfused`
- Size: ~500MB
- Output: 512-dim embedding
- Purpose: Audio understanding

**MusicGen (Generation)**
- Model: `facebook/musicgen-small`
- Size: ~1.5GB
- Sample rate: 32kHz
- Purpose: Music generation

## ⚙️ Generation Parameters

- **duration** (5-120s): Length of generated music
- **temperature** (0.1-2.0): Randomness (higher = more variation)
- **top_k** (0-500): Limits vocabulary (lower = more conservative)
- **top_p** (0.0-1.0): Nucleus sampling (cumulative probability)
- **cfg_coef** (1.0-10.0): Conditioning strength (higher = more similar)

**Recommended settings:**
- **Similar music**: temp=0.7, cfg_coef=3.5
- **Creative variation**: temp=1.2, cfg_coef=2.0
- **Safe/conservative**: temp=0.5, cfg_coef=5.0

## 🎵 Example Workflow

```python
import requests
import time

BASE_URL = "http://localhost:8000/api"

# 1. Upload your instrumental track
with open("my_piano_piece.mp3", "rb") as f:
    resp = requests.post(f"{BASE_URL}/upload", files={"file": f})
audio_id = resp.json()["audio_id"]
print(f"Uploaded: {audio_id}")

# 2. Get embedding (vector representation)
resp = requests.get(f"{BASE_URL}/embedding/{audio_id}")
embedding = resp.json()["embedding"]
print(f"Embedding dimension: {len(embedding)}")

# 3. Generate similar music
resp = requests.post(
    f"{BASE_URL}/generate/{audio_id}",
    json={
        "duration": 30,
        "temperature": 0.8,
        "cfg_coef": 3.0
    }
)
gen_id = resp.json()["generation_id"]
print(f"Generation started: {gen_id}")

# 4. Wait for completion
while True:
    resp = requests.get(f"{BASE_URL}/status/{gen_id}")
    status = resp.json()
    print(f"Status: {status['status']} - {status['progress']}%")
    if status["status"] == "completed":
        break
    time.sleep(5)

# 5. Download result
resp = requests.get(f"{BASE_URL}/download/{gen_id}")
with open("generated_music.wav", "wb") as f:
    f.write(resp.content)
print("✓ Music generated and saved!")
```

## 🐛 Troubleshooting

### Models take too long to download
- First run downloads ~2GB of models
- Use a stable internet connection
- Models are cached after first download

### Out of memory errors
- Use CPU mode: `DEVICE=cpu`
- Use smaller model: `musicgen-small`
- Reduce duration: `duration=15`

### Generation too slow
- CPU generation: ~2-3 minutes for 30s
- GPU generation: ~20-30 seconds for 30s
- Consider using GPU: `DEVICE=cuda`

### Audio quality issues
- Use higher quality input (WAV > MP3)
- Try `musicgen-medium` for better quality
- Adjust `cfg_coef` (higher = closer to original)

## 📈 Performance Benchmarks

**CPU (Intel i7)**
- Upload: < 1 second
- Embedding: ~30 seconds
- Generation (30s): ~2-3 minutes

**GPU (NVIDIA RTX 3080)**
- Upload: < 1 second
- Embedding: ~5 seconds
- Generation (30s): ~20-30 seconds

## 🚢 Next Steps: AWS Deployment

Once local testing is complete:

1. **Storage**: Replace local files with S3
2. **Compute**: Deploy on ECS or Lambda
3. **ML Models**: Host on SageMaker endpoints
4. **API**: Add API Gateway
5. **Database**: Use DynamoDB for metadata
6. **Queue**: Add SQS for async processing
7. **CDN**: Use CloudFront for generated files

## 📝 API Documentation

Full interactive API docs available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🔐 Security Notes

For production deployment:
- Add authentication (JWT, API keys)
- Implement rate limiting
- Add input validation
- Enable HTTPS
- Secure API endpoints
- Add monitoring and logging

## 📚 Additional Resources

- [MusicGen Paper](https://arxiv.org/abs/2306.05284)
- [CLAP Paper](https://arxiv.org/abs/2211.06687)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [AWS SageMaker](https://aws.amazon.com/sagemaker/)
