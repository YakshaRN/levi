# Quick Start Guide

## Step 1: Installation

```bash
cd music-generation-backend

# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y ffmpeg python3-pip

# Install Python dependencies
pip install -r requirements.txt --break-system-packages
```

## Step 2: Configuration

```bash
# Copy environment template
cp .env.example .env

# Create data directories
mkdir -p data/{uploads,processed,generated}
```

## Step 3: Start the Server

### Option A: Using the startup script
```bash
bash start.sh
```

### Option B: Manual start
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The server will start at `http://localhost:8000`

## Step 4: Test the API

### Interactive Documentation
Open your browser and go to:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Using cURL

**1. Upload an audio file:**
```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@your_music.mp3"
```

Response:
```json
{
  "audio_id": "abc123...",
  "filename": "your_music.mp3",
  "file_size": 5242880,
  "duration": 180.5,
  "sample_rate": 44100,
  "status": "pending"
}
```

**2. Get audio embedding:**
```bash
curl "http://localhost:8000/api/embedding/{audio_id}"
```

**3. Generate similar music:**
```bash
curl -X POST "http://localhost:8000/api/generate/{audio_id}" \
  -H "Content-Type: application/json" \
  -d '{
    "duration": 30,
    "temperature": 0.8,
    "top_k": 250,
    "top_p": 0.9,
    "cfg_coef": 3.0
  }'
```

**4. Check generation status:**
```bash
curl "http://localhost:8000/api/status/{generation_id}"
```

**5. Download generated music:**
```bash
curl "http://localhost:8000/api/download/{generation_id}" \
  --output generated_music.wav
```

### Using Python Test Script

```bash
# Quick test (health + upload + analyze)
python tests/test_api.py

# Full test (includes music generation)
python tests/test_api.py --full
```

## Step 5: Understanding the Workflow

```
1. Upload Audio File
   ↓
2. Generate Embedding (vector representation)
   ↓
3. Request Music Generation
   ↓
4. Poll Generation Status (background task)
   ↓
5. Download Generated Music
```

## API Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload audio file |
| GET | `/api/embedding/{audio_id}` | Get/generate audio embedding |
| POST | `/api/generate/{audio_id}` | Start music generation |
| GET | `/api/status/{generation_id}` | Check generation status |
| GET | `/api/download/{generation_id}` | Download generated audio |
| GET | `/api/analyze/{audio_id}` | Get audio analysis |
| GET | `/health` | Health check |

## Generation Parameters

- **duration**: Length of generated music (5-120 seconds)
- **temperature**: Randomness (0.1-2.0, higher = more random)
- **top_k**: Top-k sampling (0-500)
- **top_p**: Nucleus sampling (0.0-1.0)
- **cfg_coef**: Guidance strength (1.0-10.0)

## Troubleshooting

### Models not loading
First run will download models (~2GB). This may take time.

### Out of memory
- Use CPU mode: Set `DEVICE=cpu` in `.env`
- Use smaller model: Set `GENERATION_MODEL=facebook/musicgen-small`

### Generation too slow
- Consider using GPU: Set `DEVICE=cuda` (requires CUDA)
- Reduce duration or use smaller model

### Port already in use
Change port in `.env`: `PORT=8001`

## Performance Tips

- **First request**: Takes longer due to model loading
- **Subsequent requests**: Much faster (models cached)
- **GPU vs CPU**: GPU is 5-10x faster for generation
- **File size**: Keep audio files under 50MB

## What's Next?

Once local testing works:
1. Deploy to AWS (Lambda/ECS)
2. Replace local storage with S3
3. Host models on SageMaker
4. Add API Gateway
5. Implement authentication

## Example Python Client

```python
import requests

BASE_URL = "http://localhost:8000/api"

# Upload
with open("song.mp3", "rb") as f:
    response = requests.post(f"{BASE_URL}/upload", 
                           files={"file": f})
audio_id = response.json()["audio_id"]

# Generate
response = requests.post(
    f"{BASE_URL}/generate/{audio_id}",
    json={"duration": 30, "temperature": 0.8}
)
generation_id = response.json()["generation_id"]

# Wait and download
import time
while True:
    status = requests.get(f"{BASE_URL}/status/{generation_id}").json()
    if status["status"] == "completed":
        break
    time.sleep(5)

# Download
response = requests.get(f"{BASE_URL}/download/{generation_id}")
with open("generated.wav", "wb") as f:
    f.write(response.content)
```

## Need Help?

Check the logs in the terminal where the server is running. Enable debug mode in `.env`:
```
DEBUG=True
```
