"""API routes for music generation backend."""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import os
import logging
from datetime import datetime

from app.models import (
    UploadResponse, EmbeddingResponse, GenerationRequest,
    GenerationResponse, GenerationStatus, AudioAnalysis,
    JobStatus
)
from app.config import settings
from utils.storage import LocalStorage, MetadataStore, generate_unique_id
from utils.audio_utils import get_audio_info
from services.audio_processor import AudioProcessor
from services.embedder import AudioEmbedder
from services.music_generator import MusicGenerator

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services (will be lazy-loaded)
audio_processor = None
embedder = None
generator = None

# Initialize storage
upload_storage = LocalStorage(settings.upload_dir)
processed_storage = LocalStorage(settings.processed_dir)
generated_storage = LocalStorage(settings.generated_dir)
metadata_store = MetadataStore(settings.processed_dir)


def get_audio_processor():
    """Get or create audio processor instance."""
    global audio_processor
    if audio_processor is None:
        audio_processor = AudioProcessor(target_sr=32000)
    return audio_processor


def get_embedder():
    """Get or create embedder instance."""
    global embedder
    if embedder is None:
        logger.info("Initializing audio embedder...")
        embedder = AudioEmbedder(
            model_name=settings.embedding_model,
            device=settings.device
        )
    return embedder


def get_generator():
    """Get or create music generator instance."""
    global generator
    if generator is None:
        logger.info("Initializing music generator...")
        generator = MusicGenerator(
            model_name=settings.generation_model,
            device=settings.device
        )
    return generator


@router.post("/upload", response_model=UploadResponse)
async def upload_audio(file: UploadFile = File(...)):
    """
    Upload an audio file for processing.
    
    Args:
        file: Audio file to upload
    
    Returns:
        Upload response with audio_id
    """
    try:
        # Validate file extension
        file_ext = os.path.splitext(file.filename)[1].lower().replace('.', '')
        if file_ext not in settings.allowed_extensions_list:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {settings.allowed_extensions}"
            )
        
        # Generate unique ID
        audio_id = generate_unique_id()
        
        # Save file
        filename = f"{audio_id}.{file_ext}"
        temp_path = f"/tmp/{filename}"
        
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Check file size
        file_size = os.path.getsize(temp_path)
        if file_size > settings.max_file_size:
            os.remove(temp_path)
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {settings.max_file_size / 1024 / 1024}MB"
            )
        
        # Get audio info
        try:
            info = get_audio_info(temp_path)
        except Exception as e:
            os.remove(temp_path)
            raise HTTPException(status_code=400, detail=f"Invalid audio file: {str(e)}")
        
        # Move to upload storage
        upload_storage.save_file(temp_path, filename)
        os.remove(temp_path)
        
        # Save metadata
        metadata = {
            "audio_id": audio_id,
            "original_filename": file.filename,
            "filename": filename,
            "file_size": file_size,
            "duration": info["duration"],
            "sample_rate": info["sample_rate"],
            "channels": info["channels"],
            "format": info["format"],
            "status": JobStatus.PENDING,
            "created_at": datetime.utcnow().isoformat()
        }
        metadata_store.save_metadata(audio_id, metadata)
        
        logger.info(f"Audio uploaded: {audio_id} ({file.filename})")
        
        return UploadResponse(
            audio_id=audio_id,
            filename=file.filename,
            file_size=file_size,
            duration=info["duration"],
            sample_rate=info["sample_rate"],
            status=JobStatus.PENDING
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/embedding/{audio_id}", response_model=EmbeddingResponse)
async def get_embedding(audio_id: str):
    """
    Get or generate embedding for audio.
    
    Args:
        audio_id: Unique audio identifier
    
    Returns:
        Audio embedding
    """
    try:
        # Check if metadata exists
        metadata = metadata_store.load_metadata(audio_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Audio not found")
        
        # Check if embedding already exists
        embedding_path = processed_storage.get_file_path(f"{audio_id}_embedding.npy")
        
        if os.path.exists(embedding_path):
            # Load existing embedding
            embedder_service = get_embedder()
            embedding = embedder_service.load_embedding(embedding_path)
            logger.info(f"Loaded existing embedding for {audio_id}")
        else:
            # Generate new embedding
            logger.info(f"Generating embedding for {audio_id}")
            
            # Process audio
            processor = get_audio_processor()
            audio_path = upload_storage.get_file_path(metadata["filename"])
            audio, audio_metadata = processor.process_audio(audio_path)
            
            # Prepare for embedding
            audio = processor.prepare_for_embedding(audio, audio_metadata["sample_rate"])
            
            # Generate embedding
            embedder_service = get_embedder()
            embedding = embedder_service.generate_embedding(audio, sr=32000)
            
            # Save embedding
            embedder_service.save_embedding(embedding, embedding_path)
            
            # Update metadata
            metadata["embedding_generated"] = True
            metadata["embedding_dim"] = len(embedding)
            metadata_store.save_metadata(audio_id, metadata)
            
            logger.info(f"Embedding generated and saved for {audio_id}")
        
        return EmbeddingResponse(
            audio_id=audio_id,
            embedding=embedding.tolist(),
            embedding_dim=len(embedding),
            model_used=settings.embedding_model,
            created_at=datetime.utcnow()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def generate_music_task(audio_id: str, generation_id: str, params: GenerationRequest):
    """Background task for music generation."""
    try:
        logger.info(f"Starting music generation task: {generation_id}")
        
        # Load metadata
        metadata = metadata_store.load_metadata(audio_id)
        
        # Update generation status
        gen_metadata = {
            "generation_id": generation_id,
            "audio_id": audio_id,
            "status": JobStatus.PROCESSING,
            "progress": 0,
            "params": params.dict(),
            "created_at": datetime.utcnow().isoformat()
        }
        metadata_store.save_metadata(f"gen_{generation_id}", gen_metadata)
        
        # Process audio
        processor = get_audio_processor()
        audio_path = upload_storage.get_file_path(metadata["filename"])
        audio, audio_metadata = processor.process_audio(audio_path)
        
        # Update progress
        gen_metadata["progress"] = 30
        metadata_store.save_metadata(f"gen_{generation_id}", gen_metadata)
        
        # Generate music
        gen_service = get_generator()
        generated_audio = gen_service.generate_from_audio(
            audio=audio,
            sr=audio_metadata["sample_rate"],
            duration=params.duration,
            temperature=params.temperature,
            top_k=params.top_k,
            top_p=params.top_p,
            cfg_coef=params.cfg_coef
        )
        
        # Update progress
        gen_metadata["progress"] = 80
        metadata_store.save_metadata(f"gen_{generation_id}", gen_metadata)
        
        # Save generated audio
        output_path = generated_storage.get_file_path(f"{generation_id}")
        saved_path = gen_service.save_audio(generated_audio, output_path)
        
        # Update final status
        gen_metadata["status"] = JobStatus.COMPLETED
        gen_metadata["progress"] = 100
        gen_metadata["file_path"] = saved_path
        gen_metadata["completed_at"] = datetime.utcnow().isoformat()
        metadata_store.save_metadata(f"gen_{generation_id}", gen_metadata)
        
        logger.info(f"Music generation completed: {generation_id}")
    
    except Exception as e:
        logger.error(f"Music generation failed: {str(e)}")
        gen_metadata["status"] = JobStatus.FAILED
        gen_metadata["error"] = str(e)
        metadata_store.save_metadata(f"gen_{generation_id}", gen_metadata)


@router.post("/generate/{audio_id}", response_model=GenerationResponse)
async def generate_music(
    audio_id: str,
    params: GenerationRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate similar music based on uploaded audio.
    
    Args:
        audio_id: Unique audio identifier
        params: Generation parameters
        background_tasks: FastAPI background tasks
    
    Returns:
        Generation response with job ID
    """
    try:
        # Check if audio exists
        metadata = metadata_store.load_metadata(audio_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Audio not found")
        
        # Generate unique generation ID
        generation_id = generate_unique_id()
        
        # Add background task
        background_tasks.add_task(generate_music_task, audio_id, generation_id, params)
        
        # Estimate time (rough estimate)
        estimated_time = params.duration + 30  # generation time + overhead
        
        return GenerationResponse(
            generation_id=generation_id,
            audio_id=audio_id,
            status=JobStatus.PENDING,
            message="Music generation started",
            estimated_time=estimated_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{generation_id}", response_model=GenerationStatus)
async def get_generation_status(generation_id: str):
    """
    Get status of music generation job.
    
    Args:
        generation_id: Unique generation identifier
    
    Returns:
        Generation status
    """
    try:
        metadata = metadata_store.load_metadata(f"gen_{generation_id}")
        if not metadata:
            raise HTTPException(status_code=404, detail="Generation job not found")
        
        return GenerationStatus(
            generation_id=generation_id,
            audio_id=metadata["audio_id"],
            status=metadata["status"],
            progress=metadata.get("progress", 0),
            file_path=metadata.get("file_path"),
            error=metadata.get("error"),
            created_at=datetime.fromisoformat(metadata["created_at"]),
            completed_at=datetime.fromisoformat(metadata["completed_at"]) if metadata.get("completed_at") else None
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{generation_id}")
async def download_generated_music(generation_id: str):
    """
    Download generated music file.
    
    Args:
        generation_id: Unique generation identifier
    
    Returns:
        Audio file
    """
    try:
        metadata = metadata_store.load_metadata(f"gen_{generation_id}")
        if not metadata:
            raise HTTPException(status_code=404, detail="Generation job not found")
        
        if metadata["status"] != JobStatus.COMPLETED:
            raise HTTPException(status_code=400, detail="Generation not completed")
        
        file_path = metadata.get("file_path")
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Generated file not found")
        
        return FileResponse(
            file_path,
            media_type="audio/wav",
            filename=f"generated_{generation_id}.wav"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyze/{audio_id}", response_model=AudioAnalysis)
async def analyze_audio(audio_id: str):
    """
    Get detailed analysis of uploaded audio.
    
    Args:
        audio_id: Unique audio identifier
    
    Returns:
        Audio analysis
    """
    try:
        metadata = metadata_store.load_metadata(audio_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Audio not found")
        
        # Check if embedding exists
        embedding_path = processed_storage.get_file_path(f"{audio_id}_embedding.npy")
        embedding_available = os.path.exists(embedding_path)
        
        return AudioAnalysis(
            audio_id=audio_id,
            filename=metadata["original_filename"],
            duration=metadata["duration"],
            sample_rate=metadata["sample_rate"],
            channels=metadata["channels"],
            file_size=metadata["file_size"],
            format=metadata["format"],
            embedding_available=embedding_available,
            features=metadata.get("features")
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
