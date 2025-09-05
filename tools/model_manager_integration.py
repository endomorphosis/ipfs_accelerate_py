#!/usr/bin/env python3
"""
Integration utility for connecting the ModelManager with existing model metadata systems.

This module provides utilities to:
1. Import existing model metadata from the HuggingFace model registry
2. Convert metadata from the existing generate_model_metadata.py system
3. Populate the ModelManager with comprehensive model information
4. Export model manager data in various formats
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("model_manager_integration")

# Import the ModelManager components
try:
    from ipfs_accelerate_py.model_manager import (
        ModelManager, ModelMetadata, IOSpec, ModelType, DataType,
        create_model_from_huggingface, get_default_model_manager
    )
except ImportError:
    from model_manager import (
        ModelManager, ModelMetadata, IOSpec, ModelType, DataType,
        create_model_from_huggingface, get_default_model_manager
    )

# Try to import the existing model metadata generation system
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), 'test', 'skills'))
    from generate_model_metadata import generate_metadata, generate_model_mappings
    HAVE_EXISTING_METADATA = True
    logger.info("Successfully imported existing model metadata system")
except ImportError:
    HAVE_EXISTING_METADATA = False
    logger.warning("Could not import existing model metadata system")


class ModelManagerIntegration:
    """
    Integration utility for connecting ModelManager with existing systems.
    """
    
    def __init__(self, model_manager: Optional[ModelManager] = None):
        """
        Initialize the integration utility.
        
        Args:
            model_manager: Optional ModelManager instance. If None, creates a default one.
        """
        self.model_manager = model_manager or get_default_model_manager()
        
    def import_from_existing_metadata(self) -> int:
        """
        Import models from the existing model metadata generation system.
        
        Returns:
            Number of models imported
        """
        if not HAVE_EXISTING_METADATA:
            logger.error("Existing metadata system not available")
            return 0
        
        try:
            # Generate metadata using the existing system
            existing_metadata = generate_metadata()
            model_mappings = existing_metadata.get('model_mappings', {})
            
            imported_count = 0
            
            for model_name, mapping_data in model_mappings.items():
                try:
                    # Create a basic ModelMetadata from the mapping data
                    model_metadata = self._convert_mapping_to_metadata(model_name, mapping_data)
                    
                    if self.model_manager.add_model(model_metadata):
                        imported_count += 1
                        logger.debug(f"Imported model: {model_name}")
                    else:
                        logger.warning(f"Failed to import model: {model_name}")
                        
                except Exception as e:
                    logger.error(f"Error importing model {model_name}: {e}")
                    continue
            
            logger.info(f"Successfully imported {imported_count} models from existing metadata system")
            return imported_count
            
        except Exception as e:
            logger.error(f"Error importing from existing metadata: {e}")
            return 0
    
    def _convert_mapping_to_metadata(self, model_name: str, mapping_data: Dict[str, Any]) -> ModelMetadata:
        """
        Convert existing mapping data to ModelMetadata format.
        
        Args:
            model_name: Name of the model
            mapping_data: Mapping data from existing system
            
        Returns:
            ModelMetadata object
        """
        # Extract information from mapping data
        model_id = f"hf/{model_name}"
        architecture = mapping_data.get('architecture_type', 'unknown')
        
        # Determine model type from architecture
        model_type = self._infer_model_type(architecture, model_name)
        
        # Create basic I/O specifications based on model type and architecture
        inputs, outputs = self._create_io_specs(model_type, architecture)
        
        # Create the metadata object
        metadata = ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            model_type=model_type,
            architecture=architecture,
            inputs=inputs,
            outputs=outputs,
            tags=self._generate_tags(model_name, architecture),
            description=f"HuggingFace model: {model_name} ({architecture})",
            supported_backends=["transformers", "pytorch", "onnx"]
        )
        
        return metadata
    
    def _infer_model_type(self, architecture: str, model_name: str) -> ModelType:
        """
        Infer the model type from architecture and name.
        
        Args:
            architecture: Model architecture
            model_name: Model name
            
        Returns:
            ModelType enum value
        """
        arch_lower = architecture.lower()
        name_lower = model_name.lower()
        
        # Vision models
        if any(x in arch_lower for x in ['vit', 'vision', 'deit', 'swin', 'clip', 'dino', 'beit']):
            return ModelType.VISION_MODEL
        
        # Audio models
        if any(x in arch_lower for x in ['wav2vec', 'whisper', 'audio', 'speech', 'hubert']):
            return ModelType.AUDIO_MODEL
        
        # Multimodal models
        if any(x in arch_lower for x in ['clip', 'align', 'flava', 'layoutlm', 'donut', 'pix2struct']):
            return ModelType.MULTIMODAL
        
        # Encoder-decoder models
        if any(x in arch_lower for x in ['t5', 'bart', 'pegasus', 'marian', 'm2m', 'encoder_decoder']):
            return ModelType.ENCODER_DECODER
        
        # Encoder-only models
        if any(x in arch_lower for x in ['bert', 'roberta', 'electra', 'albert', 'deberta']):
            return ModelType.ENCODER_ONLY
        
        # Decoder-only models (language models)
        if any(x in arch_lower for x in ['gpt', 'llama', 'opt', 'bloom', 'falcon', 'mistral']):
            return ModelType.DECODER_ONLY
        
        # Default to language model
        return ModelType.LANGUAGE_MODEL
    
    def _create_io_specs(self, model_type: ModelType, architecture: str) -> tuple[List[IOSpec], List[IOSpec]]:
        """
        Create input/output specifications based on model type and architecture.
        
        Args:
            model_type: Model type
            architecture: Model architecture
            
        Returns:
            Tuple of (inputs, outputs) lists
        """
        inputs = []
        outputs = []
        
        # Standard text inputs for most models
        if model_type in [ModelType.LANGUAGE_MODEL, ModelType.ENCODER_ONLY, 
                         ModelType.DECODER_ONLY, ModelType.ENCODER_DECODER]:
            inputs.extend([
                IOSpec(name="input_ids", data_type=DataType.TOKENS, 
                      shape=(None, None), description="Input token IDs"),
                IOSpec(name="attention_mask", data_type=DataType.TOKENS, 
                      shape=(None, None), description="Attention mask", optional=True)
            ])
        
        # Vision inputs
        if model_type in [ModelType.VISION_MODEL, ModelType.MULTIMODAL]:
            inputs.append(
                IOSpec(name="pixel_values", data_type=DataType.IMAGE,
                      shape=(None, 3, 224, 224), description="Image pixel values")
            )
        
        # Audio inputs
        if model_type == ModelType.AUDIO_MODEL:
            inputs.append(
                IOSpec(name="input_values", data_type=DataType.AUDIO,
                      shape=(None, None), description="Audio input values")
            )
        
        # Standard outputs
        if "embedding" in architecture.lower() or "encoder" in architecture.lower():
            outputs.append(
                IOSpec(name="last_hidden_state", data_type=DataType.EMBEDDINGS,
                      shape=(None, None, None), description="Hidden state embeddings")
            )
        
        if any(x in architecture.lower() for x in ['classification', 'qa', 'masked', 'lm']):
            outputs.append(
                IOSpec(name="logits", data_type=DataType.LOGITS,
                      shape=(None, None, None), description="Classification/prediction logits")
            )
        
        # Multimodal outputs
        if model_type == ModelType.MULTIMODAL and "clip" in architecture.lower():
            outputs.extend([
                IOSpec(name="text_embeds", data_type=DataType.EMBEDDINGS,
                      shape=(None, None), description="Text embeddings"),
                IOSpec(name="image_embeds", data_type=DataType.EMBEDDINGS,
                      shape=(None, None), description="Image embeddings")
            ])
        
        # Default output if none specified
        if not outputs:
            outputs.append(
                IOSpec(name="logits", data_type=DataType.LOGITS,
                      shape=(None, None), description="Model output logits")
            )
        
        return inputs, outputs
    
    def _generate_tags(self, model_name: str, architecture: str) -> List[str]:
        """
        Generate appropriate tags for a model.
        
        Args:
            model_name: Model name
            architecture: Model architecture
            
        Returns:
            List of tags
        """
        tags = ["huggingface"]
        
        name_lower = model_name.lower()
        arch_lower = architecture.lower()
        
        # Task-based tags
        if any(x in arch_lower for x in ['classification', 'classifier']):
            tags.append("classification")
        if any(x in arch_lower for x in ['qa', 'question']):
            tags.append("question-answering")
        if any(x in arch_lower for x in ['masked', 'mlm']):
            tags.append("masked-lm")
        if any(x in arch_lower for x in ['generation', 'lm', 'causal']):
            tags.append("text-generation")
        if any(x in arch_lower for x in ['embedding', 'encoder']):
            tags.append("embeddings")
        
        # Domain-based tags
        if any(x in arch_lower for x in ['vision', 'image', 'vit', 'swin', 'deit']):
            tags.append("computer-vision")
        if any(x in arch_lower for x in ['audio', 'speech', 'wav2vec', 'whisper']):
            tags.append("audio")
        if any(x in name_lower for x in ['multilingual', 'multi', 'mbert']):
            tags.append("multilingual")
        
        # Architecture family tags
        if "bert" in arch_lower:
            tags.append("bert")
        if "gpt" in arch_lower:
            tags.append("gpt")
        if "t5" in arch_lower:
            tags.append("t5")
        
        return tags
    
    def populate_huggingface_models(self, model_list: List[str], 
                                   fetch_configs: bool = True) -> int:
        """
        Populate the model manager with HuggingFace models.
        
        Args:
            model_list: List of HuggingFace model names/IDs
            fetch_configs: Whether to fetch actual HuggingFace configs
            
        Returns:
            Number of models successfully added
        """
        added_count = 0
        
        for model_id in model_list:
            try:
                if fetch_configs:
                    # Try to fetch the actual config (requires transformers)
                    config = self._fetch_hf_config(model_id)
                else:
                    config = None
                
                # Create model metadata
                if config:
                    metadata = create_model_from_huggingface(
                        model_id=model_id,
                        hf_config=config
                    )
                else:
                    # Create basic metadata without config
                    metadata = self._create_basic_metadata(model_id)
                
                if self.model_manager.add_model(metadata):
                    added_count += 1
                    logger.debug(f"Added HuggingFace model: {model_id}")
                else:
                    logger.warning(f"Failed to add model: {model_id}")
                    
            except Exception as e:
                logger.error(f"Error adding HuggingFace model {model_id}: {e}")
                continue
        
        logger.info(f"Successfully added {added_count} HuggingFace models")
        return added_count
    
    def _fetch_hf_config(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch HuggingFace model configuration.
        
        Args:
            model_id: HuggingFace model ID
            
        Returns:
            Model configuration dictionary or None
        """
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_id)
            return config.to_dict()
        except ImportError:
            logger.warning("transformers library not available for config fetching")
            return None
        except Exception as e:
            logger.warning(f"Could not fetch config for {model_id}: {e}")
            return None
    
    def _create_basic_metadata(self, model_id: str) -> ModelMetadata:
        """
        Create basic metadata for a model without full config.
        
        Args:
            model_id: Model identifier
            
        Returns:
            ModelMetadata object
        """
        model_name = model_id.split('/')[-1] if '/' in model_id else model_id
        
        # Infer basic information from the model name
        model_type = self._infer_model_type_from_name(model_name)
        inputs, outputs = self._create_io_specs(model_type, model_name)
        
        return ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            model_type=model_type,
            architecture="unknown",
            inputs=inputs,
            outputs=outputs,
            tags=["huggingface"],
            description=f"HuggingFace model: {model_id}",
            source_url=f"https://huggingface.co/{model_id}"
        )
    
    def _infer_model_type_from_name(self, model_name: str) -> ModelType:
        """
        Infer model type from model name.
        
        Args:
            model_name: Model name
            
        Returns:
            ModelType enum value
        """
        name_lower = model_name.lower()
        
        if any(x in name_lower for x in ['bert', 'roberta', 'electra', 'albert', 'deberta']):
            return ModelType.ENCODER_ONLY
        if any(x in name_lower for x in ['gpt', 'llama', 'opt', 'bloom', 'falcon']):
            return ModelType.DECODER_ONLY
        if any(x in name_lower for x in ['t5', 'bart', 'pegasus']):
            return ModelType.ENCODER_DECODER
        if any(x in name_lower for x in ['vit', 'deit', 'swin', 'vision']):
            return ModelType.VISION_MODEL
        if any(x in name_lower for x in ['wav2vec', 'whisper', 'audio']):
            return ModelType.AUDIO_MODEL
        if any(x in name_lower for x in ['clip', 'align']):
            return ModelType.MULTIMODAL
        
        return ModelType.LANGUAGE_MODEL
    
    def export_model_registry(self, output_path: str, format: str = "json") -> bool:
        """
        Export the current model registry to a file.
        
        Args:
            output_path: Output file path
            format: Export format ("json" or "yaml")
            
        Returns:
            bool: True if successful
        """
        return self.model_manager.export_metadata(output_path, format)
    
    def generate_compatibility_matrix(self) -> Dict[str, Any]:
        """
        Generate a compatibility matrix showing input/output type relationships.
        
        Returns:
            Dictionary containing compatibility information
        """
        models = self.model_manager.list_models()
        
        # Count models by input/output types
        input_types = {}
        output_types = {}
        compatibility_pairs = {}
        
        for model in models:
            for inp in model.inputs:
                input_type = inp.data_type.value if hasattr(inp.data_type, 'value') else str(inp.data_type)
                input_types[input_type] = input_types.get(input_type, 0) + 1
                
            for out in model.outputs:
                output_type = out.data_type.value if hasattr(out.data_type, 'value') else str(out.data_type)
                output_types[output_type] = output_types.get(output_type, 0) + 1
                
                # Track input->output compatibility
                for inp in model.inputs:
                    inp_type = inp.data_type.value if hasattr(inp.data_type, 'value') else str(inp.data_type)
                    pair = f"{inp_type} -> {output_type}"
                    compatibility_pairs[pair] = compatibility_pairs.get(pair, 0) + 1
        
        return {
            "input_types": input_types,
            "output_types": output_types,
            "compatibility_pairs": compatibility_pairs,
            "total_models": len(models)
        }
    
    def close(self):
        """Close the integration utility and underlying model manager."""
        if self.model_manager:
            self.model_manager.close()


def populate_model_manager_from_existing_data():
    """
    Utility function to populate a model manager with existing data.
    
    Returns:
        ModelManager instance populated with existing model data
    """
    integration = ModelManagerIntegration()
    
    # Import from existing metadata system if available
    if HAVE_EXISTING_METADATA:
        imported_count = integration.import_from_existing_metadata()
        logger.info(f"Imported {imported_count} models from existing metadata system")
    
    # Add some popular HuggingFace models as examples
    popular_models = [
        "bert-base-uncased",
        "gpt2",
        "t5-small",
        "distilbert-base-uncased",
        "roberta-base"
    ]
    
    added_count = integration.populate_huggingface_models(
        popular_models, 
        fetch_configs=False  # Don't require transformers library
    )
    logger.info(f"Added {added_count} popular HuggingFace models")
    
    return integration.model_manager


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create and populate model manager
    integration = ModelManagerIntegration()
    
    # Import from existing systems
    if HAVE_EXISTING_METADATA:
        imported = integration.import_from_existing_metadata()
        print(f"Imported {imported} models from existing metadata system")
    
    # Add some example models
    example_models = ["bert-base-uncased", "gpt2", "t5-small"]
    added = integration.populate_huggingface_models(example_models, fetch_configs=False)
    print(f"Added {added} example models")
    
    # Show statistics
    stats = integration.model_manager.get_stats()
    print(f"Total models in registry: {stats['total_models']}")
    print(f"Models by type: {stats['models_by_type']}")
    
    # Generate compatibility matrix
    compatibility = integration.generate_compatibility_matrix()
    print(f"Input types: {compatibility['input_types']}")
    print(f"Output types: {compatibility['output_types']}")
    
    # Export the registry
    integration.export_model_registry("model_registry_export.json")
    print("Exported model registry to model_registry_export.json")
    
    integration.close()