"""
Configuration management for HF Model Server
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import os


class ServerConfig(BaseModel):
    """Main server configuration"""
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of worker processes")
    
    # Model discovery
    skill_directories: List[str] = Field(
        default_factory=lambda: ["ipfs_accelerate_py"],
        description="Directories to scan for HF skills"
    )
    skill_pattern: str = Field(default="hf_*.py", description="Pattern for skill files")
    auto_discover: bool = Field(default=True, description="Automatically discover skills on startup")
    
    # Hardware settings
    preferred_hardware: List[str] = Field(
        default_factory=lambda: ["cuda", "rocm", "mps", "openvino", "cpu"],
        description="Preferred hardware order"
    )
    enable_hardware_detection: bool = Field(default=True, description="Detect available hardware")
    
    # Performance settings
    enable_batching: bool = Field(default=True, description="Enable request batching")
    batch_max_size: int = Field(default=32, description="Maximum batch size")
    batch_timeout_ms: int = Field(default=100, description="Batch timeout in milliseconds")
    
    enable_caching: bool = Field(default=True, description="Enable response caching")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")
    cache_max_size: int = Field(default=1000, description="Maximum cache entries")
    
    # Circuit breaker settings
    enable_circuit_breaker: bool = Field(default=True, description="Enable circuit breaker")
    circuit_breaker_threshold: int = Field(default=5, description="Failure threshold")
    circuit_breaker_timeout_seconds: int = Field(default=60, description="Circuit breaker timeout")
    
    # Model loading
    max_loaded_models: int = Field(default=3, description="Maximum number of loaded models")
    model_load_timeout_seconds: int = Field(default=300, description="Model load timeout")
    enable_model_caching: bool = Field(default=True, description="Keep models in memory")
    
    # Monitoring
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(default=9090, description="Metrics port")
    enable_health_checks: bool = Field(default=True, description="Enable health endpoints")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json or text)")
    
    # API settings
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    enable_cors: bool = Field(default=True, description="Enable CORS")
    cors_origins: List[str] = Field(default_factory=lambda: ["*"], description="CORS origins")
    
    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Load configuration from environment variables"""
        return cls(
            host=os.getenv("HF_SERVER_HOST", "0.0.0.0"),
            port=int(os.getenv("HF_SERVER_PORT", "8000")),
            workers=int(os.getenv("HF_SERVER_WORKERS", "1")),
            log_level=os.getenv("HF_SERVER_LOG_LEVEL", "INFO"),
            api_key=os.getenv("HF_SERVER_API_KEY"),
            enable_batching=os.getenv("HF_SERVER_ENABLE_BATCHING", "true").lower() == "true",
            enable_caching=os.getenv("HF_SERVER_ENABLE_CACHING", "true").lower() == "true",
            enable_circuit_breaker=os.getenv("HF_SERVER_ENABLE_CIRCUIT_BREAKER", "true").lower() == "true",
        )
    
    class Config:
        use_enum_values = True
