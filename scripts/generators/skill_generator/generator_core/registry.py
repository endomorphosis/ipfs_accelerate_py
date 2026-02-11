#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Component registry for the refactored generator suite.
Manages registration and lookup of templates, model information, and other components.
"""

import os
import sys
import logging
import importlib
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Type, Callable


class ComponentRegistry:
    """Registry for generator components."""

    def __init__(self):
        """Initialize the component registry."""
        self.logger = logging.getLogger(__name__)
        
        # Template registry
        self.templates = {}
        
        # Model info providers
        self.model_info_providers = {}
        
        # Hardware detectors
        self.hardware_detectors = {}
        
        # Syntax fixers
        self.syntax_fixers = {}
        
        # Architecture to model mapping
        self.architecture_mapping = {
            "encoder-only": ["bert", "roberta", "distilbert", "albert", "electra", "camembert", 
                             "xlm-roberta", "deberta", "ernie", "rembert"],
            "decoder-only": ["gpt2", "gpt-2", "gptj", "gpt-j", "gpt-neo", "gpt-neox", "llama", 
                            "llama2", "mistral", "falcon", "phi", "gemma", "opt", "mpt"],
            "encoder-decoder": ["t5", "bart", "mbart", "pegasus", "mt5", "led", "prophetnet"],
            "vision": ["vit", "swin", "resnet", "deit", "beit", "segformer", "detr", "mask2former", 
                      "yolos", "sam", "dinov2", "convnext"],
            "vision-text": ["clip", "blip", "flava", "git", "idefics", "paligemma", "imagebind", 
                           "llava", "fuyu"],
            "speech": ["whisper", "wav2vec2", "hubert", "sew", "unispeech", "clap", "musicgen", 
                      "encodec"]
        }
        
        # Default model mapping (model type to default model ID)
        self.default_models = {
            "bert": "bert-base-uncased",
            "roberta": "roberta-base",
            "gpt2": "gpt2",
            "llama": "meta-llama/Llama-2-7b-hf",
            "t5": "t5-small",
            "bart": "facebook/bart-base",
            "vit": "google/vit-base-patch16-224",
            "clip": "openai/clip-vit-base-patch32",
            "whisper": "openai/whisper-tiny"
        }
    
    def register_template(self, architecture: str, template) -> None:
        """Register a template for a specific architecture.
        
        Args:
            architecture: The architecture type (encoder-only, decoder-only, etc.)
            template: The template instance.
        """
        self.templates[architecture] = template
        self.logger.debug(f"Registered template for architecture: {architecture}")
    
    def get_template(self, model_type: str) -> Optional[Any]:
        """Get the template for a model type.
        
        Args:
            model_type: The model type (bert, gpt2, t5, etc.)
            
        Returns:
            The template instance for the model's architecture, or None if not found.
        """
        architecture = self._map_model_to_architecture(model_type)
        if architecture in self.templates:
            return self.templates[architecture]
        
        self.logger.warning(f"No template found for architecture: {architecture} (model type: {model_type})")
        return None
    
    def register_model_info_provider(self, provider_name: str, provider) -> None:
        """Register a model information provider.
        
        Args:
            provider_name: The name of the provider.
            provider: The provider instance.
        """
        self.model_info_providers[provider_name] = provider
        self.logger.debug(f"Registered model info provider: {provider_name}")
    
    def get_model_info(self, model_type: str, provider_name: Optional[str] = None) -> Dict[str, Any]:
        """Get model information for a model type.
        
        Args:
            model_type: The model type (bert, gpt2, t5, etc.)
            provider_name: Optional name of the provider to use.
            
        Returns:
            Dict containing model information.
        """
        # If provider is specified, use it
        if provider_name and provider_name in self.model_info_providers:
            provider = self.model_info_providers[provider_name]
            try:
                return provider.get_model_info(model_type)
            except Exception as e:
                self.logger.error(f"Error getting model info from provider {provider_name}: {e}")
        
        # Otherwise, try all registered providers
        for name, provider in self.model_info_providers.items():
            try:
                info = provider.get_model_info(model_type)
                if info:
                    return info
            except Exception as e:
                self.logger.debug(f"Error getting model info from provider {name}: {e}")
        
        # If no provider could get info, return basic info
        architecture = self._map_model_to_architecture(model_type)
        default_model = self.get_default_model(model_type)
        
        # Build class name from model type
        class_name = "".join(part.capitalize() for part in model_type.split("-"))
        
        return {
            "name": model_type,
            "id": default_model,
            "architecture": architecture,
            "class_name": class_name,
            "task": self._get_default_task(model_type, architecture),
            "default": True
        }
    
    def register_hardware_detector(self, hardware_type: str, detector) -> None:
        """Register a hardware detector.
        
        Args:
            hardware_type: The hardware type (cuda, rocm, mps, etc.)
            detector: The detector instance.
        """
        self.hardware_detectors[hardware_type] = detector
        self.logger.debug(f"Registered hardware detector: {hardware_type}")
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information from all registered detectors.
        
        Returns:
            Dict containing hardware information.
        """
        hardware_info = {}
        
        for hardware_type, detector in self.hardware_detectors.items():
            try:
                info = detector.detect()
                hardware_info[hardware_type] = info
            except Exception as e:
                self.logger.error(f"Error detecting hardware {hardware_type}: {e}")
                hardware_info[hardware_type] = {"available": False, "error": str(e)}
        
        return hardware_info
    
    def register_syntax_fixer(self, fixer_name: str, fixer) -> None:
        """Register a syntax fixer.
        
        Args:
            fixer_name: The name of the fixer.
            fixer: The fixer instance.
        """
        self.syntax_fixers[fixer_name] = fixer
        self.logger.debug(f"Registered syntax fixer: {fixer_name}")
    
    def get_syntax_fixer(self, fixer_name: str) -> Optional[Any]:
        """Get a syntax fixer by name.
        
        Args:
            fixer_name: The name of the fixer.
            
        Returns:
            The fixer instance, or None if not found.
        """
        if fixer_name in self.syntax_fixers:
            return self.syntax_fixers[fixer_name]
        
        self.logger.warning(f"No syntax fixer found with name: {fixer_name}")
        return None
    
    def _map_model_to_architecture(self, model_type: str) -> str:
        """Map a model type to its architecture.
        
        Args:
            model_type: The model type (bert, gpt2, t5, etc.)
            
        Returns:
            The architecture type (encoder-only, decoder-only, etc.)
        """
        # First, check for exact match in any architecture
        for architecture, models in self.architecture_mapping.items():
            if model_type in models:
                return architecture
        
        # If no exact match, check for partial matches
        model_type_lower = model_type.lower()
        for architecture, models in self.architecture_mapping.items():
            for model in models:
                if model_type_lower.startswith(model.lower()) or model.lower().startswith(model_type_lower):
                    return architecture
        
        # If all else fails, return unknown
        self.logger.warning(f"Unknown architecture for model type: {model_type}")
        return "unknown"
    
    def get_default_model(self, model_type: str) -> str:
        """Get the default model ID for a model type.
        
        Args:
            model_type: The model type (bert, gpt2, t5, etc.)
            
        Returns:
            The default model ID for the model type.
        """
        if model_type in self.default_models:
            return self.default_models[model_type]
        
        # Try to construct a default model ID
        return f"{model_type}-base"
    
    def _get_default_task(self, model_type: str, architecture: str) -> str:
        """Get the default task for a model type and architecture.
        
        Args:
            model_type: The model type (bert, gpt2, t5, etc.)
            architecture: The architecture type (encoder-only, decoder-only, etc.)
            
        Returns:
            The default task for the model type and architecture.
        """
        # Default tasks by architecture
        architecture_tasks = {
            "encoder-only": "fill-mask",
            "decoder-only": "text-generation",
            "encoder-decoder": "text2text-generation",
            "vision": "image-classification",
            "vision-text": "image-to-text",
            "speech": "automatic-speech-recognition"
        }
        
        # Specific model tasks that override the defaults
        model_tasks = {
            "bert": "fill-mask",
            "roberta": "fill-mask",
            "gpt2": "text-generation",
            "llama": "text-generation",
            "t5": "text2text-generation",
            "bart": "text2text-generation",
            "vit": "image-classification",
            "clip": "image-to-text",
            "whisper": "automatic-speech-recognition"
        }
        
        # First, check for model-specific task
        if model_type in model_tasks:
            return model_tasks[model_type]
        
        # Otherwise, use architecture-based task
        if architecture in architecture_tasks:
            return architecture_tasks[architecture]
        
        # Fallback to a generic task
        return "unknown"
    
    def discover_templates(self, templates_dir: Optional[str] = None) -> None:
        """Discover and register templates from the templates directory.
        
        Args:
            templates_dir: Optional directory to search for templates. If not provided,
                           the default templates directory will be used.
        """
        # Rather than attempting dynamic discovery, which can be problematic with templates
        # containing Jinja syntax, let's use the templates that are already registered
        # in the templates/__init__.py module
        try:
            from templates import get_all_templates
            
            # Get all templates
            templates = get_all_templates()
            
            # Register each template using its metadata
            for name, template in templates.items():
                if hasattr(template, 'get_metadata') and callable(template.get_metadata):
                    metadata = template.get_metadata()
                    
                    # Register the template for each supported architecture
                    if 'supported_architectures' in metadata:
                        for arch in metadata['supported_architectures']:
                            self.register_template(arch, template)
                    
                    self.logger.info(f"Registered template for architecture: {name}")
            
        except Exception as e:
            self.logger.error(f"Error registering templates: {e}")
            
            # Fallback: Register the built-in template mapping
            architectures = {
                "encoder-only": ["bert", "roberta", "distilbert"],
                "decoder-only": ["gpt2", "llama", "opt"],
                "encoder-decoder": ["t5", "bart"],
                "vision": ["vit", "resnet"],
                "vision-text": ["clip", "blip"],
                "speech": ["whisper", "wav2vec2"]
            }
            
            for arch, models in architectures.items():
                for model in models:
                    self.architecture_mapping[arch].append(model)
            
            self.logger.info("Registered fallback architecture mapping")
    
    def get_model_ids(self) -> List[str]:
        """Get a list of all known model IDs from the architecture mapping.
        
        Returns:
            List of model IDs.
        """
        model_ids = []
        for models in self.architecture_mapping.values():
            model_ids.extend(models)
        return sorted(set(model_ids))
    
    def get_architectures(self) -> List[str]:
        """Get a list of all known architectures.
        
        Returns:
            List of architecture types.
        """
        return sorted(self.architecture_mapping.keys())
    
    def get_models_by_architecture(self, architecture: str) -> List[str]:
        """Get a list of model types for a specific architecture.
        
        Args:
            architecture: The architecture type (encoder-only, decoder-only, etc.)
            
        Returns:
            List of model types.
        """
        if architecture in self.architecture_mapping:
            return sorted(self.architecture_mapping[architecture])
        return []