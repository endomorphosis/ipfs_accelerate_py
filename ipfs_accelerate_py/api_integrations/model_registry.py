"""
API Model Registry

Maps API providers to their available models and capabilities.
Integrates with the model manager to provide unified model selection.
"""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

try:
    from ..common.pipeline_types import PipelineType, PipelineTypeMapper
    HAVE_PIPELINE_TYPES = True
except ImportError:
    HAVE_PIPELINE_TYPES = False
    PipelineType = None


class APIProviderType(Enum):
    """Enumeration of supported API providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"  # Gemini
    GROQ = "groq"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    REPLICATE = "replicate"
    
    # Inference engines
    VLLM = "vllm"
    HF_TGI = "hf_tgi"
    HF_TEI = "hf_tei"
    OVMS = "ovms"
    OPEA = "opea"


@dataclass
class APIModel:
    """
    Metadata for a model available via API.
    
    This complements self-hosted models tracked in the model manager.
    """
    model_id: str
    model_name: str
    provider: APIProviderType
    pipeline_types: List[str] = field(default_factory=list)
    context_length: Optional[int] = None
    supports_streaming: bool = False
    cost_per_1k_tokens: Optional[Dict[str, float]] = None  # {"input": X, "output": Y}
    description: str = ""
    is_multimodal: bool = False
    vision_capable: bool = False
    function_calling: bool = False
    json_mode: bool = False
    deprecated: bool = False


class APIModelRegistry:
    """
    Registry of models available via API providers.
    
    This allows the model manager to present a unified view of both
    self-hosted and API-based models filtered by pipeline type.
    """
    
    # OpenAI Models
    OPENAI_MODELS = [
        APIModel(
            model_id="gpt-4-turbo",
            model_name="GPT-4 Turbo",
            provider=APIProviderType.OPENAI,
            pipeline_types=["text-generation", "conversational", "text-classification"],
            context_length=128000,
            supports_streaming=True,
            cost_per_1k_tokens={"input": 0.01, "output": 0.03},
            vision_capable=True,
            is_multimodal=True,
            function_calling=True,
            json_mode=True,
        ),
        APIModel(
            model_id="gpt-4",
            model_name="GPT-4",
            provider=APIProviderType.OPENAI,
            pipeline_types=["text-generation", "conversational"],
            context_length=8192,
            supports_streaming=True,
            cost_per_1k_tokens={"input": 0.03, "output": 0.06},
            function_calling=True,
            json_mode=True,
        ),
        APIModel(
            model_id="gpt-3.5-turbo",
            model_name="GPT-3.5 Turbo",
            provider=APIProviderType.OPENAI,
            pipeline_types=["text-generation", "conversational"],
            context_length=16385,
            supports_streaming=True,
            cost_per_1k_tokens={"input": 0.0005, "output": 0.0015},
            function_calling=True,
            json_mode=True,
        ),
        APIModel(
            model_id="text-embedding-3-large",
            model_name="Text Embedding 3 Large",
            provider=APIProviderType.OPENAI,
            pipeline_types=["feature-extraction"],
            context_length=8191,
            cost_per_1k_tokens={"input": 0.00013, "output": 0},
        ),
        APIModel(
            model_id="whisper-1",
            model_name="Whisper",
            provider=APIProviderType.OPENAI,
            pipeline_types=["automatic-speech-recognition"],
            cost_per_1k_tokens={"input": 0.006, "output": 0},  # per minute
        ),
        APIModel(
            model_id="tts-1",
            model_name="Text-to-Speech 1",
            provider=APIProviderType.OPENAI,
            pipeline_types=["text-to-speech"],
            cost_per_1k_tokens={"input": 0.015, "output": 0},  # per 1k chars
        ),
    ]
    
    # Anthropic Models (Claude)
    ANTHROPIC_MODELS = [
        APIModel(
            model_id="claude-3-opus-20240229",
            model_name="Claude 3 Opus",
            provider=APIProviderType.ANTHROPIC,
            pipeline_types=["text-generation", "conversational"],
            context_length=200000,
            supports_streaming=True,
            cost_per_1k_tokens={"input": 0.015, "output": 0.075},
            vision_capable=True,
            is_multimodal=True,
        ),
        APIModel(
            model_id="claude-3-sonnet-20240229",
            model_name="Claude 3 Sonnet",
            provider=APIProviderType.ANTHROPIC,
            pipeline_types=["text-generation", "conversational"],
            context_length=200000,
            supports_streaming=True,
            cost_per_1k_tokens={"input": 0.003, "output": 0.015},
            vision_capable=True,
            is_multimodal=True,
        ),
        APIModel(
            model_id="claude-3-haiku-20240307",
            model_name="Claude 3 Haiku",
            provider=APIProviderType.ANTHROPIC,
            pipeline_types=["text-generation", "conversational"],
            context_length=200000,
            supports_streaming=True,
            cost_per_1k_tokens={"input": 0.00025, "output": 0.00125},
        ),
    ]
    
    # Google Models (Gemini)
    GOOGLE_MODELS = [
        APIModel(
            model_id="gemini-pro",
            model_name="Gemini Pro",
            provider=APIProviderType.GOOGLE,
            pipeline_types=["text-generation", "conversational"],
            context_length=30720,
            supports_streaming=True,
            cost_per_1k_tokens={"input": 0.00025, "output": 0.0005},
        ),
        APIModel(
            model_id="gemini-pro-vision",
            model_name="Gemini Pro Vision",
            provider=APIProviderType.GOOGLE,
            pipeline_types=["text-generation", "visual-question-answering", "image-to-text"],
            context_length=12288,
            supports_streaming=True,
            vision_capable=True,
            is_multimodal=True,
            cost_per_1k_tokens={"input": 0.00025, "output": 0.0005},
        ),
    ]
    
    # Groq Models
    GROQ_MODELS = [
        APIModel(
            model_id="llama3-70b-8192",
            model_name="LLaMA 3 70B",
            provider=APIProviderType.GROQ,
            pipeline_types=["text-generation", "conversational"],
            context_length=8192,
            supports_streaming=True,
            description="Fast inference on Groq LPU",
        ),
        APIModel(
            model_id="mixtral-8x7b-32768",
            model_name="Mixtral 8x7B",
            provider=APIProviderType.GROQ,
            pipeline_types=["text-generation", "conversational"],
            context_length=32768,
            supports_streaming=True,
            description="Fast inference on Groq LPU",
        ),
    ]
    
    # Cohere Models
    COHERE_MODELS = [
        APIModel(
            model_id="command",
            model_name="Command",
            provider=APIProviderType.COHERE,
            pipeline_types=["text-generation", "conversational"],
            context_length=4096,
            supports_streaming=True,
        ),
        APIModel(
            model_id="embed-english-v3.0",
            model_name="Embed English v3",
            provider=APIProviderType.COHERE,
            pipeline_types=["feature-extraction"],
            context_length=512,
        ),
    ]
    
    def __init__(self):
        """Initialize the API model registry."""
        self._models: Dict[str, APIModel] = {}
        self._provider_index: Dict[APIProviderType, List[str]] = {}
        self._pipeline_index: Dict[str, List[str]] = {}
        
        # Register all models
        self._register_models()
    
    def _register_models(self):
        """Register all API models."""
        all_models = (
            self.OPENAI_MODELS +
            self.ANTHROPIC_MODELS +
            self.GOOGLE_MODELS +
            self.GROQ_MODELS +
            self.COHERE_MODELS
        )
        
        for model in all_models:
            self._models[model.model_id] = model
            
            # Index by provider
            if model.provider not in self._provider_index:
                self._provider_index[model.provider] = []
            self._provider_index[model.provider].append(model.model_id)
            
            # Index by pipeline types
            for pipeline_type in model.pipeline_types:
                if pipeline_type not in self._pipeline_index:
                    self._pipeline_index[pipeline_type] = []
                self._pipeline_index[pipeline_type].append(model.model_id)
    
    def get_model(self, model_id: str) -> Optional[APIModel]:
        """Get model by ID."""
        return self._models.get(model_id)
    
    def get_models_by_provider(self, provider: APIProviderType) -> List[APIModel]:
        """Get all models for a specific provider."""
        model_ids = self._provider_index.get(provider, [])
        return [self._models[mid] for mid in model_ids]
    
    def get_models_by_pipeline_type(self, pipeline_type: str) -> List[APIModel]:
        """
        Get all API models that support a specific pipeline type.
        
        Args:
            pipeline_type: Pipeline type (e.g., "text-generation", "conversational")
            
        Returns:
            List of API models supporting the pipeline type
        """
        model_ids = self._pipeline_index.get(pipeline_type, [])
        return [self._models[mid] for mid in model_ids if not self._models[mid].deprecated]
    
    def get_all_models(self, include_deprecated: bool = False) -> List[APIModel]:
        """Get all registered API models."""
        if include_deprecated:
            return list(self._models.values())
        return [m for m in self._models.values() if not m.deprecated]
    
    def get_all_providers(self) -> List[APIProviderType]:
        """Get list of all API providers."""
        return list(self._provider_index.keys())
    
    def get_supported_pipeline_types(self) -> List[str]:
        """Get list of all supported pipeline types across all API models."""
        return list(self._pipeline_index.keys())
    
    def register_custom_model(self, model: APIModel):
        """
        Register a custom API model (e.g., for self-hosted inference endpoints).
        
        Args:
            model: API model to register
        """
        self._models[model.model_id] = model
        
        # Update provider index
        if model.provider not in self._provider_index:
            self._provider_index[model.provider] = []
        if model.model_id not in self._provider_index[model.provider]:
            self._provider_index[model.provider].append(model.model_id)
        
        # Update pipeline index
        for pipeline_type in model.pipeline_types:
            if pipeline_type not in self._pipeline_index:
                self._pipeline_index[pipeline_type] = []
            if model.model_id not in self._pipeline_index[pipeline_type]:
                self._pipeline_index[pipeline_type].append(model.model_id)


# Global registry instance
_global_registry: Optional[APIModelRegistry] = None


def get_global_api_model_registry() -> APIModelRegistry:
    """Get or create the global API model registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = APIModelRegistry()
    return _global_registry


def get_api_models_for_pipeline(pipeline_type: str) -> List[APIModel]:
    """
    Convenience function to get API models for a pipeline type.
    
    Args:
        pipeline_type: Pipeline type (e.g., "text-generation")
        
    Returns:
        List of API models supporting the pipeline type
    """
    registry = get_global_api_model_registry()
    return registry.get_models_by_pipeline_type(pipeline_type)


__all__ = [
    'APIProviderType',
    'APIModel',
    'APIModelRegistry',
    'get_global_api_model_registry',
    'get_api_models_for_pipeline',
]
