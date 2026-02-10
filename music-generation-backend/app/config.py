"""Configuration management for the music generation backend."""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings."""
    
    # App settings
    app_name: str = "MusicGenerationBackend"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Storage paths
    upload_dir: str = "data/uploads"
    processed_dir: str = "data/processed"
    generated_dir: str = "data/generated"
    
    # Model settings
    embedding_model: str = "laion/clap-htsat-unfused"
    generation_model: str = "facebook/musicgen-small"
    device: str = "cpu"
    
    # Generation parameters
    default_duration: int = 30
    max_duration: int = 120
    default_temperature: float = 0.8
    default_top_k: int = 250
    default_top_p: float = 0.9
    
    # File upload limits
    max_file_size: int = 52428800  # 50MB
    allowed_extensions: str = "mp3,wav,flac,m4a,ogg"
    
    # API settings
    api_prefix: str = "/api"
    cors_origins: str = "*"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    @property
    def allowed_extensions_list(self) -> List[str]:
        """Return allowed extensions as a list."""
        return [ext.strip() for ext in self.allowed_extensions.split(",")]
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Return CORS origins as a list."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.upload_dir, self.processed_dir, self.generated_dir]:
            os.makedirs(directory, exist_ok=True)


# Global settings instance
settings = Settings()
settings.ensure_directories()
