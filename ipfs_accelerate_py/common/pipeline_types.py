"""
HuggingFace Pipeline Types and Model Capability Mapping

This module defines all HuggingFace pipeline types and provides
utilities for mapping models to their capabilities.

Based on: https://huggingface.co/docs/transformers/main_classes/pipelines
"""

from enum import Enum
from typing import Dict, List, Set
from dataclasses import dataclass


class PipelineType(Enum):
    """
    Complete enumeration of HuggingFace pipeline types.
    
    These correspond to the task types used in transformers.pipeline()
    and HuggingFace Hub model classifications.
    """
    
    # Text Processing Pipelines
    TEXT_CLASSIFICATION = "text-classification"
    TOKEN_CLASSIFICATION = "token-classification"
    QUESTION_ANSWERING = "question-answering"
    TABLE_QUESTION_ANSWERING = "table-question-answering"
    FILL_MASK = "fill-mask"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    TEXT_GENERATION = "text-generation"  # causal_lm
    TEXT2TEXT_GENERATION = "text2text-generation"
    
    # Conversational & Feature Extraction
    CONVERSATIONAL = "conversational"
    FEATURE_EXTRACTION = "feature-extraction"
    SENTENCE_SIMILARITY = "sentence-similarity"
    ZERO_SHOT_CLASSIFICATION = "zero-shot-classification"
    
    # Vision Pipelines
    IMAGE_CLASSIFICATION = "image-classification"
    OBJECT_DETECTION = "object-detection"
    IMAGE_SEGMENTATION = "image-segmentation"
    ZERO_SHOT_IMAGE_CLASSIFICATION = "zero-shot-image-classification"
    DEPTH_ESTIMATION = "depth-estimation"
    IMAGE_TO_IMAGE = "image-to-image"
    MASK_GENERATION = "mask-generation"
    
    # Audio Pipelines
    AUTOMATIC_SPEECH_RECOGNITION = "automatic-speech-recognition"
    AUDIO_CLASSIFICATION = "audio-classification"
    TEXT_TO_SPEECH = "text-to-speech"
    TEXT_TO_AUDIO = "text-to-audio"
    ZERO_SHOT_AUDIO_CLASSIFICATION = "zero-shot-audio-classification"
    
    # Multimodal Pipelines
    DOCUMENT_QUESTION_ANSWERING = "document-question-answering"
    VISUAL_QUESTION_ANSWERING = "visual-question-answering"
    IMAGE_TO_TEXT = "image-to-text"
    VIDEO_CLASSIFICATION = "video-classification"
    
    # Specialized Pipelines
    REINFORCEMENT_LEARNING = "reinforcement-learning"
    ROBOTICS = "robotics"
    UNCONDITIONAL_IMAGE_GENERATION = "unconditional-image-generation"
    GRAPH_MACHINE_LEARNING = "graph-ml"


# Alias for common use cases
CAUSAL_LM = PipelineType.TEXT_GENERATION
SEQUENCE_CLASSIFICATION = PipelineType.TEXT_CLASSIFICATION
NER = PipelineType.TOKEN_CLASSIFICATION


@dataclass
class ModelCapability:
    """Describes a model's capability for a specific pipeline type."""
    pipeline_type: PipelineType
    supported: bool
    performance_tier: str = "unknown"  # "excellent", "good", "fair", "poor"
    notes: str = ""


class PipelineTypeMapper:
    """
    Maps models to their supported pipeline types.
    
    This class provides utilities to determine which pipeline types
    a model supports, either from HuggingFace metadata or API provider
    documentation.
    """
    
    # Map HuggingFace model architectures to likely pipeline types
    ARCHITECTURE_TO_PIPELINES: Dict[str, List[PipelineType]] = {
        # BERT-based models
        "BertForMaskedLM": [PipelineType.FILL_MASK],
        "BertForSequenceClassification": [PipelineType.TEXT_CLASSIFICATION],
        "BertForTokenClassification": [PipelineType.TOKEN_CLASSIFICATION],
        "BertForQuestionAnswering": [PipelineType.QUESTION_ANSWERING],
        
        # GPT models (causal LM)
        "GPT2LMHeadModel": [PipelineType.TEXT_GENERATION],
        "GPTNeoForCausalLM": [PipelineType.TEXT_GENERATION],
        "GPTJForCausalLM": [PipelineType.TEXT_GENERATION],
        "LlamaForCausalLM": [PipelineType.TEXT_GENERATION, PipelineType.CONVERSATIONAL],
        "MistralForCausalLM": [PipelineType.TEXT_GENERATION, PipelineType.CONVERSATIONAL],
        
        # T5 models (seq2seq)
        "T5ForConditionalGeneration": [
            PipelineType.SUMMARIZATION,
            PipelineType.TRANSLATION,
            PipelineType.TEXT2TEXT_GENERATION,
            PipelineType.QUESTION_ANSWERING
        ],
        
        # BART models
        "BartForConditionalGeneration": [
            PipelineType.SUMMARIZATION,
            PipelineType.TEXT2TEXT_GENERATION
        ],
        
        # Vision models
        "ViTForImageClassification": [PipelineType.IMAGE_CLASSIFICATION],
        "DetrForObjectDetection": [PipelineType.OBJECT_DETECTION],
        "SegformerForSemanticSegmentation": [PipelineType.IMAGE_SEGMENTATION],
        
        # Audio models
        "Wav2Vec2ForCTC": [PipelineType.AUTOMATIC_SPEECH_RECOGNITION],
        "WhisperForConditionalGeneration": [PipelineType.AUTOMATIC_SPEECH_RECOGNITION],
        
        # Multimodal models
        "VisionEncoderDecoderModel": [PipelineType.IMAGE_TO_TEXT],
        "BlipForQuestionAnswering": [PipelineType.VISUAL_QUESTION_ANSWERING],
    }
    
    # Map API providers to their supported pipeline types
    API_PROVIDER_CAPABILITIES: Dict[str, List[PipelineType]] = {
        "openai": [
            PipelineType.TEXT_GENERATION,
            PipelineType.CONVERSATIONAL,
            PipelineType.TEXT_CLASSIFICATION,
            PipelineType.FEATURE_EXTRACTION,
            PipelineType.IMAGE_GENERATION,
            PipelineType.TEXT_TO_SPEECH,
            PipelineType.AUTOMATIC_SPEECH_RECOGNITION,
        ],
        "anthropic": [
            PipelineType.TEXT_GENERATION,
            PipelineType.CONVERSATIONAL,
            PipelineType.TEXT_CLASSIFICATION,
        ],
        "google": [  # Gemini
            PipelineType.TEXT_GENERATION,
            PipelineType.CONVERSATIONAL,
            PipelineType.VISUAL_QUESTION_ANSWERING,
            PipelineType.IMAGE_TO_TEXT,
        ],
        "groq": [
            PipelineType.TEXT_GENERATION,
            PipelineType.CONVERSATIONAL,
        ],
        "cohere": [
            PipelineType.TEXT_GENERATION,
            PipelineType.TEXT_CLASSIFICATION,
            PipelineType.FEATURE_EXTRACTION,
            PipelineType.SUMMARIZATION,
        ],
        "huggingface": [  # Inference API
            # Supports all pipeline types through different models
            pt for pt in PipelineType
        ],
    }
    
    @classmethod
    def get_pipelines_for_architecture(cls, architecture: str) -> List[PipelineType]:
        """Get supported pipeline types for a model architecture."""
        return cls.ARCHITECTURE_TO_PIPELINES.get(architecture, [])
    
    @classmethod
    def get_pipelines_for_api_provider(cls, provider: str) -> List[PipelineType]:
        """Get supported pipeline types for an API provider."""
        return cls.API_PROVIDER_CAPABILITIES.get(provider.lower(), [])
    
    @classmethod
    def supports_pipeline(cls, pipeline_type: PipelineType, 
                         architecture: str = None, 
                         api_provider: str = None) -> bool:
        """
        Check if a model or API provider supports a specific pipeline type.
        
        Args:
            pipeline_type: Pipeline type to check
            architecture: Model architecture (for self-hosted models)
            api_provider: API provider name (for API models)
            
        Returns:
            True if the pipeline type is supported
        """
        if architecture:
            return pipeline_type in cls.get_pipelines_for_architecture(architecture)
        elif api_provider:
            return pipeline_type in cls.get_pipelines_for_api_provider(api_provider)
        return False


def get_all_pipeline_types() -> List[str]:
    """Get list of all pipeline type values."""
    return [pt.value for pt in PipelineType]


def get_text_pipeline_types() -> List[str]:
    """Get list of text-related pipeline types."""
    return [
        PipelineType.TEXT_CLASSIFICATION.value,
        PipelineType.TOKEN_CLASSIFICATION.value,
        PipelineType.QUESTION_ANSWERING.value,
        PipelineType.FILL_MASK.value,
        PipelineType.SUMMARIZATION.value,
        PipelineType.TRANSLATION.value,
        PipelineType.TEXT_GENERATION.value,
        PipelineType.TEXT2TEXT_GENERATION.value,
        PipelineType.CONVERSATIONAL.value,
        PipelineType.ZERO_SHOT_CLASSIFICATION.value,
    ]


def get_vision_pipeline_types() -> List[str]:
    """Get list of vision-related pipeline types."""
    return [
        PipelineType.IMAGE_CLASSIFICATION.value,
        PipelineType.OBJECT_DETECTION.value,
        PipelineType.IMAGE_SEGMENTATION.value,
        PipelineType.ZERO_SHOT_IMAGE_CLASSIFICATION.value,
        PipelineType.DEPTH_ESTIMATION.value,
        PipelineType.IMAGE_TO_IMAGE.value,
    ]


def get_audio_pipeline_types() -> List[str]:
    """Get list of audio-related pipeline types."""
    return [
        PipelineType.AUTOMATIC_SPEECH_RECOGNITION.value,
        PipelineType.AUDIO_CLASSIFICATION.value,
        PipelineType.TEXT_TO_SPEECH.value,
        PipelineType.ZERO_SHOT_AUDIO_CLASSIFICATION.value,
    ]


def get_multimodal_pipeline_types() -> List[str]:
    """Get list of multimodal pipeline types."""
    return [
        PipelineType.DOCUMENT_QUESTION_ANSWERING.value,
        PipelineType.VISUAL_QUESTION_ANSWERING.value,
        PipelineType.IMAGE_TO_TEXT.value,
        PipelineType.VIDEO_CLASSIFICATION.value,
    ]


# Export commonly used names
__all__ = [
    'PipelineType',
    'CAUSAL_LM',
    'SEQUENCE_CLASSIFICATION',
    'NER',
    'ModelCapability',
    'PipelineTypeMapper',
    'get_all_pipeline_types',
    'get_text_pipeline_types',
    'get_vision_pipeline_types',
    'get_audio_pipeline_types',
    'get_multimodal_pipeline_types',
]
