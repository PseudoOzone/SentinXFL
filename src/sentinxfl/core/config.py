"""
SentinXFL Configuration Management
===================================

Centralized configuration using Pydantic Settings with environment variable support.
Optimized for RTX 3050 4GB VRAM constraint.

Author: Anshuman Bakshi
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ==========================================
    # Application
    # ==========================================
    app_name: str = "SentinXFL"
    app_version: str = "2.0.0"
    debug: bool = False
    environment: Literal["development", "staging", "production"] = "development"

    # ==========================================
    # API Server
    # ==========================================
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_reload: bool = True

    # ==========================================
    # Security
    # ==========================================
    secret_key: str = Field(default="dev-secret-key-change-in-production")
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60
    cors_origins: list[str] = ["http://localhost:3000"]

    # ==========================================
    # Paths
    # ==========================================
    base_dir: Path = Path(__file__).parent.parent.parent.parent
    data_dir: Path = Field(default=Path("data/datasets"))
    processed_dir: Path = Field(default=Path("data/processed"))
    models_dir: Path = Field(default=Path("models/checkpoints"))
    logs_dir: Path = Field(default=Path("logs"))

    # ==========================================
    # DuckDB
    # ==========================================
    duckdb_path: Path = Field(default=Path("data/sentinxfl.duckdb"))
    duckdb_memory_limit: str = "2GB"
    duckdb_threads: int = 4

    # ==========================================
    # Privacy (Differential Privacy)
    # ==========================================
    dp_epsilon: float = Field(default=1.0, ge=0.1, le=10.0)
    dp_delta: float = Field(default=1e-5, ge=1e-9, le=1e-3)
    dp_max_grad_norm: float = Field(default=1.0, ge=0.1, le=10.0)
    pii_audit_enabled: bool = True
    pii_strict_mode: bool = True

    # ==========================================
    # Federated Learning
    # ==========================================
    fl_server_address: str = "0.0.0.0:8080"
    fl_min_clients: int = Field(default=2, ge=2)
    fl_rounds: int = Field(default=10, ge=1)
    fl_local_epochs: int = Field(default=3, ge=1)
    fl_aggregation_strategy: Literal[
        "fedavg", "multi_krum", "trimmed_mean", "coordinate_median"
    ] = "fedavg"

    # ==========================================
    # ML Settings
    # ==========================================
    ml_batch_size: int = Field(default=512, ge=32)
    ml_learning_rate: float = Field(default=0.001, ge=1e-6, le=1.0)
    ml_train_split: float = Field(default=0.7, ge=0.5, le=0.9)
    ml_val_split: float = Field(default=0.15, ge=0.05, le=0.3)
    ml_test_split: float = Field(default=0.15, ge=0.05, le=0.3)
    ml_random_seed: int = 42

    # ==========================================
    # LLM Settings - Provider Selection
    # ==========================================
    # FREE OPTIONS (default):
    #   - "ollama": Local Ollama server (recommended free option)
    #   - "local": Direct HuggingFace local inference
    # PAID OPTIONS (scaffolded for future):
    #   - "openai": OpenAI API (GPT-4, etc.)
    #   - "anthropic": Anthropic API (Claude)
    #   - "groq": Groq API (fast inference)
    llm_provider: Literal["ollama", "local", "openai", "anthropic", "groq"] = "ollama"
    
    # Ollama Settings (FREE - default)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "phi3:mini"  # or llama3.2:3b, mistral, etc.
    
    # Local HuggingFace Settings (FREE)
    llm_model_id: str = "microsoft/Phi-3-mini-4k-instruct"
    llm_quantization: Literal["4bit", "8bit", "none"] = "4bit"
    llm_device: str = "cuda"
    llm_vram_gb: float = 2.0  # Reserved VRAM for LLM
    
    # Paid API Keys (SCAFFOLDED - fill when ready)
    openai_api_key: str | None = None  # Set in .env when ready
    anthropic_api_key: str | None = None
    groq_api_key: str | None = None
    
    # Common LLM Settings
    llm_max_new_tokens: int = Field(default=512, ge=64, le=2048)
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    # ==========================================
    # ChromaDB (RAG) - FREE local vector DB
    # ==========================================
    chroma_persist_dir: Path = Field(default=Path("chroma_db"))
    chroma_collection_name: str = "sentinxfl_docs"
    # FREE local embedding model (no API needed)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ==========================================
    # Logging
    # ==========================================
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "text"] = "json"
    log_file: Path = Field(default=Path("logs/sentinxfl.log"))
    log_rotation: str = "10 MB"
    log_retention: str = "30 days"

    # ==========================================
    # Hardware Constraints (RTX 3050 4GB)
    # ==========================================
    total_vram_gb: float = 4.0
    tabnet_vram_gb: float = 1.0
    max_concurrent_gpu_tasks: int = 1  # TabNet and LLM cannot run together

    @field_validator("data_dir", "processed_dir", "models_dir", "logs_dir", mode="before")
    @classmethod
    def resolve_path(cls, v: str | Path) -> Path:
        """Convert string paths to Path objects."""
        return Path(v) if isinstance(v, str) else v

    def get_absolute_path(self, relative_path: Path) -> Path:
        """Get absolute path from project root."""
        if relative_path.is_absolute():
            return relative_path
        return self.base_dir / relative_path

    @property
    def data_dir_abs(self) -> Path:
        """Absolute path to data directory."""
        return self.get_absolute_path(self.data_dir)

    @property
    def processed_dir_abs(self) -> Path:
        """Absolute path to processed data directory."""
        return self.get_absolute_path(self.processed_dir)

    @property
    def models_dir_abs(self) -> Path:
        """Absolute path to models directory."""
        return self.get_absolute_path(self.models_dir)

    @property
    def duckdb_path_abs(self) -> Path:
        """Absolute path to DuckDB file."""
        return self.get_absolute_path(self.duckdb_path)

    def available_vram_for_inference(self) -> float:
        """Calculate available VRAM after reserving for active components."""
        return self.total_vram_gb - self.tabnet_vram_gb  # When TabNet is active


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
