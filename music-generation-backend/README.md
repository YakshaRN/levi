# Music Generation Backend

Upload an **instrumental** audio track and generate **similar variations**. No GPU required by default (CPU-first; add GPU later if needed).

## What it does
- **Upload** an instrumental file (mp3, wav, flac, m4a, ogg).
- **Generate** similar variations from that audio (melody-conditioned MusicGen).
- Optional: get vector embeddings for the audio (CLAP).

Runs on **CPU** by default. Set `DEVICE=cuda` when you want to use a GPU.

## Project Structure
```
music-generation-backend/
├── app/
│   ├── main.py                 # FastAPI application
│   ├── models.py               # Pydantic models
│   ├── config.py               # Configuration
│   └── api/
│       ├── routes.py           # API endpoints
│       └── dependencies.py     # Shared dependencies
├── services/
│   ├── audio_processor.py      # Audio feature extraction
│   ├── embedder.py             # Vector embedding generation
│   └── music_generator.py      # Music generation service
├── utils/
│   ├── audio_utils.py          # Audio utility functions
│   └── storage.py              # Local file storage
├── data/
│   ├── uploads/                # Uploaded audio files
│   ├── processed/              # Processed embeddings
│   └── generated/              # Generated music files
├── tests/
│   └── test_api.py             # API tests
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
└── README.md                   # This file
```

## Installation

### Prerequisites
- Python 3.9+
- ffmpeg (for audio processing)

### Setup

1. Install ffmpeg:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y ffmpeg

# macOS
brew install ffmpeg
```

2. Install Python dependencies:

   **On EC2 or low disk space** (recommended – avoids CUDA and saves ~2GB+):
   ```bash
   pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements-cpu.txt
   ```

   Otherwise:
   ```bash
   pip install -r requirements.txt
   ```

3. Create environment file:
```bash
cp .env.example .env
```

4. Create data directories:
```bash
mkdir -p data/{uploads,processed,generated}
```

## Usage

### Start the server:
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### API Endpoints:

1. **Upload Audio**
```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@your_music.mp3"
```

2. **Get Audio Embedding**
```bash
curl "http://localhost:8000/api/embedding/{audio_id}"
```

3. **Generate Similar Music**
```bash
curl -X POST "http://localhost:8000/api/generate/{audio_id}" \
  -H "Content-Type: application/json" \
  -d '{"duration": 30, "temperature": 0.8}'
```

4. **Download Generated Music**
```bash
curl "http://localhost:8000/api/download/{generation_id}" \
  --output generated_music.wav
```

## Testing

Run the test suite:
```bash
python tests/test_api.py
```

Or use pytest:
```bash
pytest tests/ -v
```

## Models Used

1. **MusicGen Melody** – conditions on your instrumental to generate similar variations (default).
2. **CLAP** – optional, for audio embeddings.
3. **Encodec** – used by Audiocraft for audio encoding/decoding.

## Flow

```
Upload instrumental → (optional: get embedding) → Generate similar variation → Download WAV
```

## Next Steps for AWS Deployment

1. Replace local storage with S3
2. Deploy FastAPI on Lambda or ECS
3. Host models on SageMaker endpoints
4. Add DynamoDB for job tracking
5. Implement API Gateway

## Performance Notes

- First request may take 30–60s to download models.
- Subsequent requests use cached models.
- **CPU**: generation can take a few minutes for 30s audio; fine for development.
- **GPU** (optional later): set `DEVICE=cuda` for much faster generation.

## License
MIT
