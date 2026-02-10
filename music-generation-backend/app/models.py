"""Pydantic models for request/response validation."""

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    """Job status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class UploadResponse(BaseModel):
    """Response model for audio upload."""
    audio_id: str
    filename: str
    file_size: int
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    status: JobStatus = JobStatus.PENDING
    message: str = "Audio uploaded successfully"


class EmbeddingResponse(BaseModel):
    """Response model for audio embedding."""
    audio_id: str
    embedding: List[float]
    embedding_dim: int
    model_used: str
    created_at: datetime


class GenerationRequest(BaseModel):
    """Request model for music generation."""
    duration: int = Field(default=30, ge=5, le=120, description="Duration in seconds")
    temperature: float = Field(default=0.8, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: int = Field(default=250, ge=0, le=500, description="Top-k sampling")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p (nucleus) sampling")
    cfg_coef: float = Field(default=3.0, ge=1.0, le=10.0, description="Classifier-free guidance coefficient")
    
    @validator('duration')
    def validate_duration(cls, v):
        """Ensure duration is a multiple of 5."""
        if v % 5 != 0:
            raise ValueError('Duration must be a multiple of 5')
        return v


class GenerationResponse(BaseModel):
    """Response model for music generation."""
    generation_id: str
    audio_id: str
    status: JobStatus
    message: str
    estimated_time: Optional[int] = None  # seconds


class GenerationStatus(BaseModel):
    """Response model for generation status check."""
    generation_id: str
    audio_id: str
    status: JobStatus
    progress: int = Field(ge=0, le=100, description="Progress percentage")
    file_path: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class AudioAnalysis(BaseModel):
    """Detailed audio analysis response."""
    audio_id: str
    filename: str
    duration: float
    sample_rate: int
    channels: int
    file_size: int
    format: str
    embedding_available: bool
    features: Optional[dict] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str = "1.0.0"
    models_loaded: bool
    timestamp: datetime
