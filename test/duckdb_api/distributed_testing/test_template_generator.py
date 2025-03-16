#!/usr/bin/env python3
"""
Test script for the Template-Based Test Generator Integration in the Distributed Testing Framework.

This script tests the integration between template-based test generators and the distributed testing
framework. It verifies the functionality of the TestGeneratorIntegration class for:

1. Template database schema creation and management
2. Template storage and retrieval
3. Model-to-family mapping
4. Dynamic test generation from templates
5. Test dependency management
"""

import os
import sys
import json
import uuid
import tempfile
import unittest
from pathlib import Path

# Setup path for imports
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Conditional imports (these will be properly tested in setUp)
try:
    import duckdb
    from test_generator_integration import TestGeneratorIntegration
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False


class TestTemplateGenerator(unittest.TestCase):
    """Test the Template-Based Test Generator Integration."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Skip tests if DuckDB is not available
        if not DUCKDB_AVAILABLE:
            raise unittest.SkipTest("DuckDB or integration modules not available")
            
        # Create a temporary directory for test files
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.db_path = os.path.join(cls.temp_dir.name, "test_templates.duckdb")
        
        # Sample templates for different model families
        cls.templates = {
            "text_embedding": """
# Test for ${model_name} on ${hardware_type} with batch size ${batch_size}
import torch
from transformers import AutoModel, AutoTokenizer

def test_${model_family}_${hardware_type}():
    # Initialize model
    model_name = "${model_name}"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Move to appropriate device
    device = "${hardware_type}"
    if device != "cpu":
        model = model.to(device)
    
    # Prepare input
    text = "Example text for embedding test"
    inputs = tokenizer(text, return_tensors="pt")
    if device != "cpu":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run inference with batch size ${batch_size}
    batch = {k: v.repeat(${batch_size}, 1) for k, v in inputs.items()}
    
    # Execute model
    with torch.no_grad():
        outputs = model(**batch)
    
    # Return results
    return {
        "model_name": "${model_name}",
        "hardware_type": "${hardware_type}",
        "batch_size": ${batch_size},
        "embedding_shape": outputs.last_hidden_state.shape,
        "success": True
    }
""",
            "vision": """
# Test for ${model_name} on ${hardware_type} with batch size ${batch_size}
import torch
from transformers import AutoImageProcessor, AutoModel
import numpy as np

def test_${model_family}_${hardware_type}():
    # Initialize model
    model_name = "${model_name}"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Move to appropriate device
    device = "${hardware_type}"
    if device != "cpu":
        model = model.to(device)
    
    # Prepare input (dummy image)
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    inputs = processor(dummy_image, return_tensors="pt")
    if device != "cpu":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run inference with batch size ${batch_size}
    batch = {k: v.repeat(${batch_size}, 1, 1, 1) if v.dim() == 4 else v.repeat(${batch_size}, 1) for k, v in inputs.items()}
    
    # Execute model
    with torch.no_grad():
        outputs = model(**batch)
    
    # Return results
    return {
        "model_name": "${model_name}",
        "hardware_type": "${hardware_type}",
        "batch_size": ${batch_size},
        "output_shape": outputs.last_hidden_state.shape,
        "success": True
    }
"""
        }
        
        # Model mappings for testing
        cls.model_mappings = [
            ("bert-base-uncased", "text_embedding"),
            ("roberta-base", "text_embedding"),
            ("vit-base-patch16-224", "vision"),
            ("google/vit-base-patch16-224", "vision")
        ]
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Remove temporary directory and files
        cls.temp_dir.cleanup()
    
    def setUp(self):
        """Set up each test."""
        # Initialize generator
        self.generator = TestGeneratorIntegration(self.db_path)
    
    def tearDown(self):
        """Clean up after each test."""
        # Close generator
        self.generator.close()
    
    def test_01_schema_creation(self):
        """Test database schema creation."""
        # Schema should be created during initialization
        self.assertIsNotNone(self.generator.template_db)
        
        # Verify tables exist
        templates_table = self.generator.template_db.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='templates'
        """).fetchone()
        self.assertIsNotNone(templates_table)
        
        mappings_table = self.generator.template_db.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='model_mapping'
        """).fetchone()
        self.assertIsNotNone(mappings_table)
    
    def test_02_add_templates(self):
        """Test adding templates to the database."""
        # Add text embedding template
        template_id1 = self.generator.add_template(
            "text_embedding_template",
            "text_embedding",
            self.templates["text_embedding"],
            "cpu",
            "Template for text embedding models"
        )
        self.assertIsNotNone(template_id1)
        
        # Add vision template
        template_id2 = self.generator.add_template(
            "vision_template",
            "vision",
            self.templates["vision"],
            "cuda",
            "Template for vision models"
        )
        self.assertIsNotNone(template_id2)
        
        # Verify templates were added
        count = self.generator.template_db.execute("""
            SELECT COUNT(*) FROM templates
        """).fetchone()[0]
        self.assertEqual(count, 2)
        
        # Retrieve and verify template content
        template1 = self.generator.get_template_by_id(template_id1)
        self.assertEqual(template1["template_name"], "text_embedding_template")
        self.assertEqual(template1["model_family"], "text_embedding")
        self.assertEqual(template1["hardware_type"], "cpu")
        
        template2 = self.generator.get_template_by_id(template_id2)
        self.assertEqual(template2["template_name"], "vision_template")
        self.assertEqual(template2["model_family"], "vision")
        self.assertEqual(template2["hardware_type"], "cuda")
    
    def test_03_add_model_mappings(self):
        """Test adding model mappings to the database."""
        # Add model mappings
        for model_name, model_family in self.model_mappings:
            success = self.generator.add_model_mapping(
                model_name,
                model_family,
                description=f"Mapping for {model_name}"
            )
            self.assertTrue(success)
        
        # Verify mappings were added
        count = self.generator.template_db.execute("""
            SELECT COUNT(*) FROM model_mapping
        """).fetchone()[0]
        self.assertEqual(count, len(self.model_mappings))
        
        # Retrieve and verify mapping
        mapping = self.generator.get_mapping_by_model("bert-base-uncased")
        self.assertEqual(mapping["model_name"], "bert-base-uncased")
        self.assertEqual(mapping["model_family"], "text_embedding")
    
    def test_04_model_family_detection(self):
        """Test model family detection."""
        # Test with mapped models
        self.assertEqual(self.generator._get_model_family("bert-base-uncased"), "text_embedding")
        self.assertEqual(self.generator._get_model_family("vit-base-patch16-224"), "vision")
        
        # Test inference for unmapped models
        self.assertEqual(self.generator._get_model_family("t5-small"), "text_generation")
        self.assertEqual(self.generator._get_model_family("gpt2"), "text_generation")
        self.assertEqual(self.generator._get_model_family("facebook/wav2vec2-base"), "audio")
        self.assertEqual(self.generator._get_model_family("openai/clip-vit-base-patch32"), "multimodal")
    
    def test_05_generate_tests(self):
        """Test test generation from templates."""
        # First add necessary templates and mappings
        template_id = self.generator.add_template(
            "text_embedding_template",
            "text_embedding",
            self.templates["text_embedding"],
            None,  # No specific hardware type
            "Template for text embedding models"
        )
        self.assertIsNotNone(template_id)
        
        # Add model mapping
        success = self.generator.add_model_mapping(
            "bert-base-uncased",
            "text_embedding",
            template_id,
            "Mapping for BERT"
        )
        self.assertTrue(success)
        
        # Generate tests
        hardware_types = ["cpu", "cuda"]
        batch_sizes = [1, 4]
        success, tests = self.generator.generate_and_submit_tests(
            "bert-base-uncased",
            hardware_types,
            batch_sizes
        )
        
        # Verify test generation success
        self.assertTrue(success)
        self.assertEqual(len(tests), len(hardware_types) * len(batch_sizes))
        
        # Verify test content
        for test in tests:
            self.assertEqual(test["model_name"], "bert-base-uncased")
            self.assertIn(test["hardware_type"], hardware_types)
            self.assertIn(test["batch_size"], batch_sizes)
            self.assertEqual(test["model_family"], "text_embedding")
            self.assertIsNotNone(test["test_content"])
            self.assertTrue(test["test_content"].startswith("# Test for bert-base-uncased on"))
            
            # Verify variable substitution
            self.assertIn(f"batch_size {test['batch_size']}", test["test_content"])
            self.assertIn(f"hardware_type = \"{test['hardware_type']}\"", test["test_content"])
    
    def test_06_test_dependencies(self):
        """Test dependency management between tests."""
        # Add template and mapping
        template_id = self.generator.add_template(
            "vision_template",
            "vision",
            self.templates["vision"],
            None,
            "Template for vision models"
        )
        self.assertIsNotNone(template_id)
        
        success = self.generator.add_model_mapping(
            "vit-base-patch16-224",
            "vision",
            template_id,
            "Mapping for ViT"
        )
        self.assertTrue(success)
        
        # Generate tests with multiple batch sizes
        hardware_types = ["cuda"]
        batch_sizes = [1, 4, 16]
        success, tests = self.generator.generate_and_submit_tests(
            "vit-base-patch16-224",
            hardware_types,
            batch_sizes
        )
        
        # Verify test generation success
        self.assertTrue(success)
        self.assertEqual(len(tests), len(hardware_types) * len(batch_sizes))
        
        # Verify dependencies are set up correctly
        # Test with batch size 4 should depend on test with batch size 1
        # Test with batch size 16 should depend on test with batch size 4
        batch1_test = None
        batch4_test = None
        batch16_test = None
        
        for test in tests:
            if test["batch_size"] == 1:
                batch1_test = test
            elif test["batch_size"] == 4:
                batch4_test = test
            elif test["batch_size"] == 16:
                batch16_test = test
        
        self.assertIsNotNone(batch1_test)
        self.assertIsNotNone(batch4_test)
        self.assertIsNotNone(batch16_test)
        
        # Batch 1 test should have no dependencies
        self.assertNotIn("dependencies", batch1_test)
        
        # Batch 4 test should depend on batch 1 test
        self.assertIn("dependencies", batch4_test)
        self.assertIn(batch1_test["test_id"], batch4_test["dependencies"])
        
        # Batch 16 test should depend on batch 4 test
        self.assertIn("dependencies", batch16_test)
        self.assertIn(batch4_test["test_id"], batch16_test["dependencies"])
    
    def test_07_priority_calculation(self):
        """Test task priority calculation."""
        # Test CPU priority
        config = {"hardware_type": "cpu", "batch_size": 1}
        priority = self.generator._calculate_priority(config)
        self.assertEqual(priority, 7)  # 5 + 1 (cpu) + 1 (small batch)
        
        # Test GPU priority with large batch
        config = {"hardware_type": "cuda", "batch_size": 32}
        priority = self.generator._calculate_priority(config)
        self.assertEqual(priority, 3)  # 5 - 1 (gpu) - 1 (large batch)
        
        # Test default hardware with medium batch
        config = {"hardware_type": "tpu", "batch_size": 8}
        priority = self.generator._calculate_priority(config)
        self.assertEqual(priority, 5)  # 5 + 0 + 0
    
    def test_08_memory_estimation(self):
        """Test memory requirement estimation."""
        # Test text embedding model with small batch
        config = {"model_family": "text_embedding", "batch_size": 1}
        memory = self.generator._estimate_memory_requirement(config)
        self.assertEqual(memory, 0.5)  # Minimum 0.5GB
        
        # Test text generation model with medium batch
        config = {"model_family": "text_generation", "batch_size": 4}
        memory = self.generator._estimate_memory_requirement(config)
        self.assertEqual(memory, 4.0)  # 4GB base * (4/4) batch factor
        
        # Test multimodal model with large batch
        config = {"model_family": "multimodal", "batch_size": 8}
        memory = self.generator._estimate_memory_requirement(config)
        self.assertEqual(memory, 12.0)  # 6GB base * (8/4) batch factor


if __name__ == "__main__":
    unittest.main()