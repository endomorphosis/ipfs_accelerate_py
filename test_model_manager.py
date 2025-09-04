#!/usr/bin/env python3
"""
Test suite for the Model Manager functionality.

This module tests the comprehensive model metadata management system including:
- Model registration and retrieval
- Input/output type mapping
- HuggingFace configuration integration
- Storage backend functionality
- Search and filtering capabilities
"""

import os
import sys
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

# Add the parent directory to the Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import directly from the module file
try:
    from ipfs_accelerate_py.model_manager import (
        ModelManager, ModelMetadata, IOSpec, ModelType, DataType,
        create_model_from_huggingface, get_default_model_manager
    )
except ImportError:
    # Direct import if package import fails
    from model_manager import (
        ModelManager, ModelMetadata, IOSpec, ModelType, DataType,
        create_model_from_huggingface, get_default_model_manager
    )


class TestModelManager(unittest.TestCase):
    """Test cases for ModelManager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test storage
        self.temp_dir = tempfile.mkdtemp()
        self.json_path = os.path.join(self.temp_dir, "test_models.json")
        self.db_path = os.path.join(self.temp_dir, "test_models.duckdb")
        
        # Create test model metadata
        self.test_model = ModelMetadata(
            model_id="test/bert-base-uncased",
            model_name="bert-base-uncased",
            model_type=ModelType.LANGUAGE_MODEL,
            architecture="BertForMaskedLM",
            inputs=[
                IOSpec(name="input_ids", data_type=DataType.TOKENS, shape=(None, 512), description="Input token IDs"),
                IOSpec(name="attention_mask", data_type=DataType.TOKENS, shape=(None, 512), description="Attention mask")
            ],
            outputs=[
                IOSpec(name="logits", data_type=DataType.LOGITS, shape=(None, 512, 30522), description="Prediction logits")
            ],
            huggingface_config={
                "architectures": ["BertForMaskedLM"],
                "model_type": "bert",
                "vocab_size": 30522,
                "hidden_size": 768
            },
            inference_code_location="/path/to/bert_inference.py",
            supported_backends=["pytorch", "onnx"],
            tags=["nlp", "masked-lm", "bert"],
            description="BERT base model for masked language modeling"
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_json_storage_backend(self):
        """Test JSON storage backend functionality."""
        # Test with JSON backend
        manager = ModelManager(storage_path=self.json_path, use_database=False)
        
        # Add a model
        self.assertTrue(manager.add_model(self.test_model))
        
        # Verify model was added
        retrieved = manager.get_model("test/bert-base-uncased")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.model_name, "bert-base-uncased")
        self.assertEqual(retrieved.model_type, ModelType.LANGUAGE_MODEL)
        
        # Test persistence by creating new manager instance
        manager2 = ModelManager(storage_path=self.json_path, use_database=False)
        retrieved2 = manager2.get_model("test/bert-base-uncased")
        self.assertIsNotNone(retrieved2)
        self.assertEqual(retrieved2.model_name, "bert-base-uncased")
        
        manager.close()
        manager2.close()
    
    def test_database_storage_backend(self):
        """Test DuckDB storage backend functionality."""
        try:
            import duckdb
        except ImportError:
            self.skipTest("DuckDB not available for testing")
        
        # Test with database backend
        manager = ModelManager(storage_path=self.db_path, use_database=True)
        
        # Add a model
        self.assertTrue(manager.add_model(self.test_model))
        
        # Verify model was added
        retrieved = manager.get_model("test/bert-base-uncased")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.model_name, "bert-base-uncased")
        self.assertEqual(retrieved.model_type, ModelType.LANGUAGE_MODEL)
        
        # Test persistence
        manager2 = ModelManager(storage_path=self.db_path, use_database=True)
        retrieved2 = manager2.get_model("test/bert-base-uncased")
        self.assertIsNotNone(retrieved2)
        self.assertEqual(retrieved2.model_name, "bert-base-uncased")
        
        manager.close()
        manager2.close()
    
    def test_model_operations(self):
        """Test basic model operations (add, get, remove)."""
        manager = ModelManager(storage_path=self.json_path, use_database=False)
        
        # Test adding model
        self.assertTrue(manager.add_model(self.test_model))
        
        # Test getting model
        retrieved = manager.get_model("test/bert-base-uncased")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.model_id, "test/bert-base-uncased")
        
        # Test getting non-existent model
        self.assertIsNone(manager.get_model("non-existent"))
        
        # Test removing model
        self.assertTrue(manager.remove_model("test/bert-base-uncased"))
        self.assertIsNone(manager.get_model("test/bert-base-uncased"))
        
        # Test removing non-existent model
        self.assertFalse(manager.remove_model("non-existent"))
        
        manager.close()
    
    def test_model_listing_and_filtering(self):
        """Test model listing and filtering functionality."""
        manager = ModelManager(storage_path=self.json_path, use_database=False)
        
        # Add multiple test models
        vision_model = ModelMetadata(
            model_id="test/vit-base",
            model_name="vit-base",
            model_type=ModelType.VISION_MODEL,
            architecture="ViTForImageClassification",
            inputs=[IOSpec(name="pixel_values", data_type=DataType.IMAGE)],
            outputs=[IOSpec(name="logits", data_type=DataType.LOGITS)],
            tags=["vision", "classification"]
        )
        
        audio_model = ModelMetadata(
            model_id="test/wav2vec2",
            model_name="wav2vec2",
            model_type=ModelType.AUDIO_MODEL,
            architecture="Wav2Vec2ForCTC",
            inputs=[IOSpec(name="input_values", data_type=DataType.AUDIO)],
            outputs=[IOSpec(name="logits", data_type=DataType.LOGITS)],
            tags=["audio", "speech"]
        )
        
        manager.add_model(self.test_model)
        manager.add_model(vision_model)
        manager.add_model(audio_model)
        
        # Test listing all models
        all_models = manager.list_models()
        self.assertEqual(len(all_models), 3)
        
        # Test filtering by model type
        language_models = manager.list_models(model_type=ModelType.LANGUAGE_MODEL)
        self.assertEqual(len(language_models), 1)
        self.assertEqual(language_models[0].model_id, "test/bert-base-uncased")
        
        vision_models = manager.list_models(model_type=ModelType.VISION_MODEL)
        self.assertEqual(len(vision_models), 1)
        self.assertEqual(vision_models[0].model_id, "test/vit-base")
        
        # Test filtering by architecture
        bert_models = manager.list_models(architecture="BertForMaskedLM")
        self.assertEqual(len(bert_models), 1)
        
        # Test filtering by tags
        vision_tagged = manager.list_models(tags=["vision"])
        self.assertEqual(len(vision_tagged), 1)
        
        nlp_tagged = manager.list_models(tags=["nlp"])
        self.assertEqual(len(nlp_tagged), 1)
        
        manager.close()
    
    def test_search_functionality(self):
        """Test model search functionality."""
        manager = ModelManager(storage_path=self.json_path, use_database=False)
        
        # Add test models
        manager.add_model(self.test_model)
        
        gpt_model = ModelMetadata(
            model_id="test/gpt2",
            model_name="gpt2",
            model_type=ModelType.LANGUAGE_MODEL,
            architecture="GPT2LMHeadModel",
            inputs=[IOSpec(name="input_ids", data_type=DataType.TOKENS)],
            outputs=[IOSpec(name="logits", data_type=DataType.LOGITS)],
            tags=["nlp", "generation"],
            description="GPT-2 language model for text generation"
        )
        manager.add_model(gpt_model)
        
        # Test search by name
        bert_results = manager.search_models("bert")
        self.assertEqual(len(bert_results), 1)
        self.assertEqual(bert_results[0].model_id, "test/bert-base-uncased")
        
        # Test search by description
        generation_results = manager.search_models("generation")
        self.assertEqual(len(generation_results), 1)
        self.assertEqual(generation_results[0].model_id, "test/gpt2")
        
        # Test search by tags
        nlp_results = manager.search_models("nlp")
        self.assertEqual(len(nlp_results), 2)
        
        # Test case-insensitive search
        bert_upper_results = manager.search_models("BERT")
        self.assertEqual(len(bert_upper_results), 1)
        
        manager.close()
    
    def test_input_output_type_queries(self):
        """Test queries based on input/output types."""
        manager = ModelManager(storage_path=self.json_path, use_database=False)
        
        # Add models with different I/O types
        manager.add_model(self.test_model)  # TEXT -> LOGITS
        
        vision_model = ModelMetadata(
            model_id="test/clip",
            model_name="clip",
            model_type=ModelType.MULTIMODAL,
            architecture="CLIPModel",
            inputs=[
                IOSpec(name="input_ids", data_type=DataType.TEXT),
                IOSpec(name="pixel_values", data_type=DataType.IMAGE)
            ],
            outputs=[
                IOSpec(name="text_embeds", data_type=DataType.EMBEDDINGS),
                IOSpec(name="image_embeds", data_type=DataType.EMBEDDINGS)
            ]
        )
        manager.add_model(vision_model)
        
        # Test input type queries
        text_input_models = manager.get_models_by_input_type(DataType.TEXT)
        self.assertEqual(len(text_input_models), 1)
        self.assertEqual(text_input_models[0].model_id, "test/clip")
        
        token_input_models = manager.get_models_by_input_type(DataType.TOKENS)
        self.assertEqual(len(token_input_models), 1)
        self.assertEqual(token_input_models[0].model_id, "test/bert-base-uncased")
        
        image_input_models = manager.get_models_by_input_type(DataType.IMAGE)
        self.assertEqual(len(image_input_models), 1)
        
        # Test output type queries
        logits_output_models = manager.get_models_by_output_type(DataType.LOGITS)
        self.assertEqual(len(logits_output_models), 1)
        
        embeddings_output_models = manager.get_models_by_output_type(DataType.EMBEDDINGS)
        self.assertEqual(len(embeddings_output_models), 1)
        
        # Test compatibility queries
        compatible_models = manager.get_compatible_models(DataType.TEXT, DataType.EMBEDDINGS)
        self.assertEqual(len(compatible_models), 1)
        self.assertEqual(compatible_models[0].model_id, "test/clip")
        
        manager.close()
    
    def test_statistics(self):
        """Test statistics generation."""
        manager = ModelManager(storage_path=self.json_path, use_database=False)
        
        # Test empty stats
        empty_stats = manager.get_stats()
        self.assertEqual(empty_stats["total_models"], 0)
        
        # Add models and test stats
        manager.add_model(self.test_model)
        
        vision_model = ModelMetadata(
            model_id="test/vit",
            model_name="vit",
            model_type=ModelType.VISION_MODEL,
            architecture="ViTForImageClassification",
            inputs=[IOSpec(name="pixel_values", data_type=DataType.IMAGE)],
            outputs=[IOSpec(name="logits", data_type=DataType.LOGITS)],
            huggingface_config={"model_type": "vit"}
        )
        manager.add_model(vision_model)
        
        stats = manager.get_stats()
        self.assertEqual(stats["total_models"], 2)
        self.assertEqual(stats["models_by_type"]["language_model"], 1)
        self.assertEqual(stats["models_by_type"]["vision_model"], 1)
        self.assertEqual(stats["models_with_hf_config"], 2)
        self.assertEqual(stats["models_with_inference_code"], 1)
        
        manager.close()
    
    def test_huggingface_integration(self):
        """Test HuggingFace configuration integration."""
        # Test creating model from HuggingFace config
        hf_config = {
            "architectures": ["BertForMaskedLM"],
            "model_type": "bert",
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_attention_heads": 12,
            "num_hidden_layers": 12
        }
        
        model = create_model_from_huggingface(
            model_id="bert-base-uncased",
            hf_config=hf_config,
            inference_code_location="/path/to/bert.py"
        )
        
        self.assertEqual(model.model_id, "bert-base-uncased")
        self.assertEqual(model.model_name, "bert-base-uncased")
        self.assertEqual(model.model_type, ModelType.LANGUAGE_MODEL)
        self.assertEqual(model.architecture, "BertForMaskedLM")
        self.assertEqual(model.huggingface_config, hf_config)
        self.assertEqual(model.inference_code_location, "/path/to/bert.py")
        self.assertTrue(len(model.inputs) > 0)
        self.assertTrue(len(model.outputs) > 0)
    
    def test_export_functionality(self):
        """Test metadata export functionality."""
        manager = ModelManager(storage_path=self.json_path, use_database=False)
        manager.add_model(self.test_model)
        
        # Test JSON export
        export_path = os.path.join(self.temp_dir, "exported_models.json")
        self.assertTrue(manager.export_metadata(export_path, format="json"))
        self.assertTrue(os.path.exists(export_path))
        
        # Verify exported content
        with open(export_path, 'r') as f:
            exported_data = json.load(f)
        
        self.assertIn("test/bert-base-uncased", exported_data)
        exported_model = exported_data["test/bert-base-uncased"]
        self.assertEqual(exported_model["model_name"], "bert-base-uncased")
        
        manager.close()
    
    def test_context_manager(self):
        """Test context manager functionality."""
        # Test that context manager properly closes the manager
        with ModelManager(storage_path=self.json_path, use_database=False) as manager:
            manager.add_model(self.test_model)
            retrieved = manager.get_model("test/bert-base-uncased")
            self.assertIsNotNone(retrieved)
        
        # Verify persistence after context manager exit
        with ModelManager(storage_path=self.json_path, use_database=False) as manager2:
            retrieved2 = manager2.get_model("test/bert-base-uncased")
            self.assertIsNotNone(retrieved2)
    
    def test_data_types_and_enums(self):
        """Test data type and enum functionality."""
        # Test ModelType enum
        self.assertEqual(ModelType.LANGUAGE_MODEL.value, "language_model")
        self.assertEqual(ModelType.VISION_MODEL.value, "vision_model")
        
        # Test DataType enum
        self.assertEqual(DataType.TEXT.value, "text")
        self.assertEqual(DataType.IMAGE.value, "image")
        self.assertEqual(DataType.AUDIO.value, "audio")
        
        # Test IOSpec creation
        io_spec = IOSpec(
            name="test_input",
            data_type=DataType.TEXT,
            shape=(None, 512),
            dtype="int64",
            description="Test input specification"
        )
        
        self.assertEqual(io_spec.name, "test_input")
        self.assertEqual(io_spec.data_type, DataType.TEXT)
        self.assertEqual(io_spec.shape, (None, 512))
        self.assertEqual(io_spec.dtype, "int64")
        self.assertFalse(io_spec.optional)
    
    def test_default_model_manager(self):
        """Test default model manager creation."""
        # Test that we can create a default manager
        manager = get_default_model_manager()
        self.assertIsInstance(manager, ModelManager)
        manager.close()


class TestModelMetadata(unittest.TestCase):
    """Test cases for ModelMetadata functionality."""
    
    def test_model_metadata_creation(self):
        """Test ModelMetadata object creation and defaults."""
        metadata = ModelMetadata(
            model_id="test/model",
            model_name="test-model",
            model_type=ModelType.LANGUAGE_MODEL,
            architecture="TestArchitecture",
            inputs=[IOSpec(name="input", data_type=DataType.TEXT)],
            outputs=[IOSpec(name="output", data_type=DataType.TEXT)]
        )
        
        # Check required fields
        self.assertEqual(metadata.model_id, "test/model")
        self.assertEqual(metadata.model_name, "test-model")
        self.assertEqual(metadata.model_type, ModelType.LANGUAGE_MODEL)
        self.assertEqual(metadata.architecture, "TestArchitecture")
        
        # Check defaults
        self.assertIsInstance(metadata.supported_backends, list)
        self.assertIsInstance(metadata.tags, list)
        self.assertIsInstance(metadata.created_at, datetime)
        self.assertIsInstance(metadata.updated_at, datetime)
    
    def test_io_spec_creation(self):
        """Test IOSpec object creation."""
        io_spec = IOSpec(
            name="test_tensor",
            data_type=DataType.EMBEDDINGS,
            shape=(32, 768),
            dtype="float32",
            description="Test embedding tensor",
            optional=True
        )
        
        self.assertEqual(io_spec.name, "test_tensor")
        self.assertEqual(io_spec.data_type, DataType.EMBEDDINGS)
        self.assertEqual(io_spec.shape, (32, 768))
        self.assertEqual(io_spec.dtype, "float32")
        self.assertEqual(io_spec.description, "Test embedding tensor")
        self.assertTrue(io_spec.optional)


if __name__ == "__main__":
    # Run the test suite
    unittest.main(verbosity=2)