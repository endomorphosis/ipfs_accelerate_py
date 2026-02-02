"""
Type definitions for model loader.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
from datetime import datetime


class ModelStatus(Enum):
    """Status of a loaded model."""
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"
    UNLOADING = "unloading"


@dataclass
class LoadedModel:
    """Container for a loaded model instance."""
    
    model_id: str
    skill_instance: Any
    hardware: str
    status: ModelStatus = ModelStatus.LOADED
    loaded_at: datetime = field(default_factory=datetime.utcnow)
    last_used_at: datetime = field(default_factory=datetime.utcnow)
    use_count: int = 0
    memory_mb: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def mark_used(self):
        """Mark this model as recently used."""
        self.last_used_at = datetime.utcnow()
        self.use_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "hardware": self.hardware,
            "status": self.status.value,
            "loaded_at": self.loaded_at.isoformat(),
            "last_used_at": self.last_used_at.isoformat(),
            "use_count": self.use_count,
            "memory_mb": self.memory_mb,
            "metadata": self.metadata,
            "error": self.error,
        }
