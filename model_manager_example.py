#!/usr/bin/env python3
"""
Example usage of the Model Manager system.

This script demonstrates how to use the ModelManager for:
1. Creating and managing model metadata
2. Searching and filtering models
3. Working with input/output type mappings
4. Integrating with existing HuggingFace models
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("model_manager_example")

# Import the ModelManager components
try:
    from ipfs_accelerate_py.model_manager import (
        ModelManager, ModelMetadata, IOSpec, ModelType, DataType,
        create_model_from_huggingface, get_default_model_manager
    )
    from model_manager_integration import ModelManagerIntegration
except ImportError:
    from model_manager import (
        ModelManager, ModelMetadata, IOSpec, ModelType, DataType,
        create_model_from_huggingface, get_default_model_manager
    )
    from model_manager_integration import ModelManagerIntegration


def demonstrate_basic_usage():
    """Demonstrate basic model manager operations."""
    print("\n=== Basic Model Manager Usage ===")
    
    # Create a model manager (will use JSON storage by default)
    with ModelManager(storage_path="./demo_models.json", use_database=False) as manager:
        
        # Create a custom model metadata
        custom_model = ModelMetadata(
            model_id="custom/sentiment-analyzer",
            model_name="Custom Sentiment Analyzer",
            model_type=ModelType.LANGUAGE_MODEL,
            architecture="BertForSequenceClassification",
            inputs=[
                IOSpec(name="input_ids", data_type=DataType.TOKENS, 
                      shape=(None, 512), description="Input text tokens"),
                IOSpec(name="attention_mask", data_type=DataType.TOKENS, 
                      shape=(None, 512), description="Attention mask", optional=True)
            ],
            outputs=[
                IOSpec(name="logits", data_type=DataType.LOGITS, 
                      shape=(None, 3), description="Sentiment classification logits")
            ],
            huggingface_config={
                "num_labels": 3,
                "model_type": "bert",
                "task_specific_params": {
                    "text-classification": {
                        "problem_type": "single_label_classification"
                    }
                }
            },
            inference_code_location="/path/to/sentiment_analyzer.py",
            supported_backends=["pytorch", "onnx", "tensorrt"],
            hardware_requirements={
                "min_memory_gb": 2,
                "recommended_memory_gb": 4,
                "cpu_cores": 2,
                "gpu_memory_gb": 1
            },
            performance_metrics={
                "accuracy": 0.92,
                "f1_score": 0.89,
                "inference_time_ms": 45
            },
            tags=["sentiment-analysis", "classification", "nlp", "bert"],
            description="A fine-tuned BERT model for sentiment analysis with 3 classes: positive, negative, neutral"
        )
        
        # Add the model
        if manager.add_model(custom_model):
            print(f"✓ Successfully added custom model: {custom_model.model_id}")
        
        # Create a model from HuggingFace config with repository structure
        hf_config = {
            "architectures": ["GPT2LMHeadModel"],
            "model_type": "gpt2",
            "vocab_size": 50257,
            "n_positions": 1024,
            "n_ctx": 1024,
            "n_embd": 768,
            "n_layer": 12,
            "n_head": 12
        }
        
        print("\n🔍 Fetching HuggingFace repository structure...")
        gpt2_model = create_model_from_huggingface(
            model_id="gpt2",  # Use a smaller model for demo
            hf_config=hf_config,
            inference_code_location="/path/to/gpt2_inference.py",
            fetch_repo_structure=True  # Enable repository structure fetching
        )
        gpt2_model.tags.extend(["text-generation", "autoregressive"])
        gpt2_model.performance_metrics = {
            "perplexity": 29.2,
            "inference_time_ms": 120
        }
        
        manager.add_model(gpt2_model)
        print(f"✓ Successfully added HuggingFace model: {gpt2_model.model_id}")
        
        # Demonstrate repository structure features
        if gpt2_model.repository_structure:
            repo_info = gpt2_model.repository_structure
            print(f"\n📁 Repository Structure Information:")
            print(f"   - Total files: {repo_info.get('total_files', 0)}")
            print(f"   - Total size: {repo_info.get('total_size', 0):,} bytes")
            print(f"   - Branch: {repo_info.get('branch', 'unknown')}")
            
            # Show some example files and their hashes
            files = repo_info.get('files', {})
            example_files = list(files.keys())[:5]
            print(f"   - Example files:")
            for file_path in example_files:
                file_info = files[file_path]
                print(f"     * {file_path} (size: {file_info.get('size', 0):,} bytes, hash: {file_info.get('oid', 'unknown')[:8]}...)")
            
            # Test repository queries
            print(f"\n🔍 Repository Query Examples:")
            config_hash = manager.get_model_file_hash("gpt2", "config.json")
            print(f"   - config.json hash: {config_hash}")
            
            models_with_config = manager.get_models_with_file("config.json")
            print(f"   - Models with config.json: {len(models_with_config)}")
            
            models_with_py = manager.get_models_with_file(".py")
            print(f"   - Models with Python files: {len(models_with_py)}")
        else:
            print(f"⚠️  Repository structure not available (may need internet connection)")
        
        # Retrieve and display models
        print(f"\nTotal models: {len(manager.list_models())}")
        
        retrieved_model = manager.get_model("custom/sentiment-analyzer")
        if retrieved_model:
            print(f"Retrieved model: {retrieved_model.model_name}")
            print(f"  - Type: {retrieved_model.model_type.value}")
            print(f"  - Architecture: {retrieved_model.architecture}")
            print(f"  - Inputs: {[inp.name for inp in retrieved_model.inputs]}")
            print(f"  - Outputs: {[out.name for out in retrieved_model.outputs]}")
            print(f"  - Tags: {retrieved_model.tags}")


def demonstrate_search_and_filtering():
    """Demonstrate search and filtering capabilities."""
    print("\n=== Search and Filtering ===")
    
    with ModelManager(storage_path="./demo_models.json", use_database=False) as manager:
        
        # Search for models
        sentiment_models = manager.search_models("sentiment")
        print(f"Models matching 'sentiment': {len(sentiment_models)}")
        for model in sentiment_models:
            print(f"  - {model.model_id}: {model.description[:60]}...")
        
        # Filter by model type
        language_models = manager.list_models(model_type=ModelType.LANGUAGE_MODEL)
        print(f"\nLanguage models: {len(language_models)}")
        for model in language_models:
            print(f"  - {model.model_id} ({model.architecture})")
        
        # Filter by tags
        classification_models = manager.list_models(tags=["classification"])
        print(f"\nClassification models: {len(classification_models)}")
        for model in classification_models:
            print(f"  - {model.model_id}")
        
        # Find models by input/output types
        token_input_models = manager.get_models_by_input_type(DataType.TOKENS)
        print(f"\nModels accepting token input: {len(token_input_models)}")
        
        logits_output_models = manager.get_models_by_output_type(DataType.LOGITS)
        print(f"Models producing logits output: {len(logits_output_models)}")
        
        # Find compatible models
        compatible_models = manager.get_compatible_models(DataType.TOKENS, DataType.LOGITS)
        print(f"Models compatible with tokens->logits: {len(compatible_models)}")


def demonstrate_integration():
    """Demonstrate integration with existing systems."""
    print("\n=== Integration with Existing Systems ===")
    
    # Create integration utility
    integration = ModelManagerIntegration()
    
    # Import from existing metadata system (if available)
    imported_count = integration.import_from_existing_metadata()
    print(f"Imported {imported_count} models from existing metadata system")
    
    # Add some popular models
    popular_models = [
        "microsoft/DialoGPT-medium",
        "facebook/bart-large-mnli",
        "google/vit-base-patch16-224",
        "openai/whisper-base"
    ]
    
    added_count = integration.populate_huggingface_models(
        popular_models, 
        fetch_configs=False
    )
    print(f"Added {added_count} popular HuggingFace models")
    
    # Generate compatibility matrix
    compatibility = integration.generate_compatibility_matrix()
    print(f"\nCompatibility Matrix:")
    print(f"  - Input types: {compatibility['input_types']}")
    print(f"  - Output types: {compatibility['output_types']}")
    print(f"  - Total models: {compatibility['total_models']}")
    
    # Export the registry
    export_success = integration.export_model_registry("full_model_registry.json")
    if export_success:
        print("✓ Exported full model registry to full_model_registry.json")
    
    integration.close()


def demonstrate_statistics():
    """Demonstrate statistics and analytics."""
    print("\n=== Model Registry Statistics ===")
    
    with ModelManager(storage_path="./demo_models.json", use_database=False) as manager:
        stats = manager.get_stats()
        
        print(f"Total models: {stats['total_models']}")
        print(f"Models with HuggingFace config: {stats['models_with_hf_config']}")
        print(f"Models with inference code: {stats['models_with_inference_code']}")
        
        print("\nModels by type:")
        for model_type, count in stats['models_by_type'].items():
            print(f"  - {model_type}: {count}")
        
        print("\nModels by architecture:")
        for architecture, count in sorted(stats['models_by_architecture'].items()):
            print(f"  - {architecture}: {count}")
        
        print("\nCommon input types:")
        for input_type, count in stats['common_input_types'].items():
            print(f"  - {input_type}: {count}")
        
        print("\nCommon output types:")
        for output_type, count in stats['common_output_types'].items():
            print(f"  - {output_type}: {count}")
        
        print(f"\nRepository Structure Statistics:")
        print(f"  - Models with HuggingFace config: {stats.get('models_with_hf_config', 0)}")
        print(f"  - Models with inference code: {stats.get('models_with_inference_code', 0)}")
        print(f"  - Models with repository structure: {stats.get('models_with_repo_structure', 0)}")
        print(f"  - Total tracked files: {stats.get('total_tracked_files', 0):,}")
        
        # Test repository-specific queries if any models have repo structures
        if stats.get('models_with_repo_structure', 0) > 0:
            print(f"\nRepository File Examples:")
            models_with_config = manager.get_models_with_file("config.json")
            print(f"  - Models with config.json: {len(models_with_config)}")
            
            models_with_readme = manager.get_models_with_file("README")
            print(f"  - Models with README files: {len(models_with_readme)}")
            
            models_with_python = manager.get_models_with_file(".py")
            print(f"  - Models with Python files: {len(models_with_python)}")
        

def demonstrate_model_pipeline():
    """Demonstrate creating a model processing pipeline."""
    print("\n=== Model Pipeline Example ===")
    
    with ModelManager(storage_path="./demo_models.json", use_database=False) as manager:
        
        # Find models that can process text and produce embeddings
        text_to_embedding_models = manager.get_compatible_models(DataType.TEXT, DataType.EMBEDDINGS)
        print(f"Text -> Embeddings models: {len(text_to_embedding_models)}")
        
        # Find models that can process tokens and produce logits
        token_to_logits_models = manager.get_compatible_models(DataType.TOKENS, DataType.LOGITS)
        print(f"Tokens -> Logits models: {len(token_to_logits_models)}")
        
        # Create a pipeline: Text -> Tokens -> Logits
        print("\nExample pipeline: Text -> Tokens -> Logits")
        for model in token_to_logits_models[:3]:  # Show first 3
            print(f"  - {model.model_id}")
            print(f"    Architecture: {model.architecture}")
            print(f"    Description: {model.description[:60]}...")
            if model.performance_metrics:
                print(f"    Performance: {model.performance_metrics}")


def main():
    """Main example function."""
    print("Model Manager Example Usage")
    print("=" * 50)
    
    try:
        # Run demonstrations
        demonstrate_basic_usage()
        demonstrate_search_and_filtering()
        demonstrate_integration()
        demonstrate_statistics()
        demonstrate_model_pipeline()
        
        print("\n=== Example completed successfully! ===")
        print("\nFiles created:")
        print("  - demo_models.json: Basic model registry")
        print("  - model_metadata.json: Full model registry with imported models")
        print("  - full_model_registry.json: Exported full registry")
        print("  - model_registry_export.json: Integration export")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()