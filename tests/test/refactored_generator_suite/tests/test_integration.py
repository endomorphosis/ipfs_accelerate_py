#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integration tests for the refactored generator suite.
Tests the entire generation pipeline from end to end.
"""

import os
import sys
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from generator_core.generator import GeneratorCore
from generator_core.registry import ComponentRegistry
from generator_core.config import ConfigManager
from templates.base import TemplateBase
from templates.encoder_only import EncoderOnlyTemplate
from templates.decoder_only import DecoderOnlyTemplate
from templates.encoder_decoder import EncoderDecoderTemplate
from templates.vision import VisionTemplate
from templates.vision_text import VisionTextTemplate
from templates.speech import SpeechTemplate
from hardware.detector import HardwareDetectionManager
from dependencies.manager import DependencyManager
from model_selection.selector import ModelSelector
from syntax.validator import SyntaxValidator
from syntax.fixer import SyntaxFixer


class IntegrationTest(unittest.TestCase):
    """Integration tests for the refactored generator suite."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        
        # Initialize the registry
        self.registry = ComponentRegistry()
        
        # Initialize the config
        self.config = ConfigManager()
        
        # Set up hardware detection with mocks
        self.hardware_manager = self._setup_hardware_manager()
        
        # Set up dependency manager with mocks
        self.dependency_manager = self._setup_dependency_manager()
        
        # Set up templates
        self._register_templates()
        
        # Set up model selector
        self.model_selector = self._setup_model_selector()
        
        # Set up syntax validation
        self.syntax_validator = SyntaxValidator()
        self.syntax_fixer = SyntaxFixer()
        
        # Initialize the generator core
        self.generator = GeneratorCore(
            config=self.config,
            registry=self.registry,
            hardware_manager=self.hardware_manager,
            dependency_manager=self.dependency_manager,
            model_selector=self.model_selector,
            syntax_validator=self.syntax_validator,
            syntax_fixer=self.syntax_fixer
        )

    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()

    def _setup_hardware_manager(self):
        """Set up a mock hardware detection manager."""
        manager = HardwareDetectionManager()
        # Mock hardware detection to simulate different hardware environments
        manager.detect_all = MagicMock(return_value={
            "cuda": {"available": True, "version": "11.7", "device_count": 1},
            "rocm": {"available": False},
            "mps": {"available": False},
            "openvino": {"available": False},
            "webnn": {"available": False},
            "webgpu": {"available": False}
        })
        return manager

    def _setup_dependency_manager(self):
        """Set up a mock dependency manager."""
        manager = DependencyManager()
        # Mock dependency checks
        manager.check_all = MagicMock(return_value={
            "torch": {"available": True, "version": "2.0.0", "mocked": False},
            "transformers": {"available": True, "version": "4.30.0", "mocked": False},
            "tokenizers": {"available": True, "version": "0.13.3", "mocked": False},
            "sentencepiece": {"available": True, "version": "0.1.99", "mocked": False}
        })
        return manager

    def _register_templates(self):
        """Register test templates."""
        # Register all templates
        self.registry.register_template("encoder-only", EncoderOnlyTemplate(self.config))
        self.registry.register_template("decoder-only", DecoderOnlyTemplate(self.config))
        self.registry.register_template("encoder-decoder", EncoderDecoderTemplate(self.config))
        self.registry.register_template("vision", VisionTemplate(self.config))
        self.registry.register_template("vision-text", VisionTextTemplate(self.config))
        self.registry.register_template("speech", SpeechTemplate(self.config))

    def _setup_model_selector(self):
        """Set up a mock model selector."""
        selector = ModelSelector(self.registry)
        
        # Mock model selection methods
        def mock_select_model(model_type, **kwargs):
            # Return a mock model info based on model type
            return {
                "name": f"{model_type}-model",
                "id": f"{model_type}-base",
                "architecture": self._map_model_to_architecture(model_type),
                "default_model": f"{model_type}-base",
                "task": "text-classification" if model_type == "bert" else "text-generation" if model_type == "gpt2" else "unknown",
                "framework": "pt"  # PyTorch
            }
        
        selector.select_model = MagicMock(side_effect=mock_select_model)
        return selector

    def _map_model_to_architecture(self, model_type):
        """Map model types to architectures for testing."""
        mapping = {
            "bert": "encoder-only",
            "roberta": "encoder-only",
            "gpt2": "decoder-only",
            "llama": "decoder-only",
            "t5": "encoder-decoder",
            "bart": "encoder-decoder",
            "vit": "vision",
            "clip": "vision-text",
            "whisper": "speech"
        }
        return mapping.get(model_type, "unknown")

    def test_generate_encoder_only_model(self):
        """Test generating an encoder-only model (BERT)."""
        model_type = "bert"
        options = {
            "output_file": self.output_dir / f"test_{model_type}.py",
            "model_name": "bert-base-uncased",
            "task": "text-classification"
        }
        
        # Generate the test file
        result = self.generator.generate(model_type, options)
        
        # Verify the result
        self.assertTrue(result["success"])
        self.assertTrue(Path(options["output_file"]).exists())
        
        # Check content contains key elements
        with open(options["output_file"], "r") as f:
            content = f.read()
            self.assertIn("class BertTest", content)
            self.assertIn("from transformers import", content)
            self.assertIn("test_pipeline", content)
            self.assertIn("test_model", content)

    def test_generate_decoder_only_model(self):
        """Test generating a decoder-only model (GPT-2)."""
        model_type = "gpt2"
        options = {
            "output_file": self.output_dir / f"test_{model_type}.py",
            "model_name": "gpt2",
            "task": "text-generation"
        }
        
        # Generate the test file
        result = self.generator.generate(model_type, options)
        
        # Verify the result
        self.assertTrue(result["success"])
        self.assertTrue(Path(options["output_file"]).exists())
        
        # Check content contains key elements
        with open(options["output_file"], "r") as f:
            content = f.read()
            self.assertIn("class Gpt2Test", content)
            self.assertIn("from transformers import", content)
            self.assertIn("test_generation", content)

    def test_generate_encoder_decoder_model(self):
        """Test generating an encoder-decoder model (T5)."""
        model_type = "t5"
        options = {
            "output_file": self.output_dir / f"test_{model_type}.py",
            "model_name": "t5-small",
            "task": "text2text-generation"
        }
        
        # Generate the test file
        result = self.generator.generate(model_type, options)
        
        # Verify the result
        self.assertTrue(result["success"])
        self.assertTrue(Path(options["output_file"]).exists())
        
        # Check content contains key elements
        with open(options["output_file"], "r") as f:
            content = f.read()
            self.assertIn("class T5Test", content)
            self.assertIn("from transformers import", content)
            self.assertIn("test_translation", content)

    def test_generate_vision_model(self):
        """Test generating a vision model (ViT)."""
        model_type = "vit"
        options = {
            "output_file": self.output_dir / f"test_{model_type}.py",
            "model_name": "vit-base-patch16-224",
            "task": "image-classification"
        }
        
        # Generate the test file
        result = self.generator.generate(model_type, options)
        
        # Verify the result
        self.assertTrue(result["success"])
        self.assertTrue(Path(options["output_file"]).exists())
        
        # Check content contains key elements
        with open(options["output_file"], "r") as f:
            content = f.read()
            self.assertIn("class VitTest", content)
            self.assertIn("from transformers import", content)
            self.assertIn("from PIL import Image", content)
            self.assertIn("test_image_classification", content)

    def test_generate_vision_text_model(self):
        """Test generating a vision-text model (CLIP)."""
        model_type = "clip"
        options = {
            "output_file": self.output_dir / f"test_{model_type}.py",
            "model_name": "openai/clip-vit-base-patch32",
            "task": "image-to-text"
        }
        
        # Generate the test file
        result = self.generator.generate(model_type, options)
        
        # Verify the result
        self.assertTrue(result["success"])
        self.assertTrue(Path(options["output_file"]).exists())
        
        # Check content contains key elements
        with open(options["output_file"], "r") as f:
            content = f.read()
            self.assertIn("class ClipTest", content)
            self.assertIn("from transformers import", content)
            self.assertIn("from PIL import Image", content)
            self.assertIn("test_image_text", content)

    def test_generate_speech_model(self):
        """Test generating a speech model (Whisper)."""
        model_type = "whisper"
        options = {
            "output_file": self.output_dir / f"test_{model_type}.py",
            "model_name": "openai/whisper-tiny",
            "task": "automatic-speech-recognition"
        }
        
        # Generate the test file
        result = self.generator.generate(model_type, options)
        
        # Verify the result
        self.assertTrue(result["success"])
        self.assertTrue(Path(options["output_file"]).exists())
        
        # Check content contains key elements
        with open(options["output_file"], "r") as f:
            content = f.read()
            self.assertIn("class WhisperTest", content)
            self.assertIn("from transformers import", content)
            self.assertIn("test_speech_recognition", content)

    def test_batch_generation(self):
        """Test batch generation of multiple models."""
        models = ["bert", "gpt2", "t5", "vit", "clip", "whisper"]
        batch_options = {
            "base_output_dir": self.output_dir,
            "batch_size": len(models)
        }
        
        # Generate all models in a batch
        results = []
        for model_type in models:
            options = {
                "output_file": self.output_dir / f"test_{model_type}.py",
                "model_name": f"{model_type}-base",
                "task": "default"
            }
            result = self.generator.generate(model_type, options)
            results.append(result)
        
        # Verify all generations succeeded
        for result in results:
            self.assertTrue(result["success"])
        
        # Check all files were created
        for model_type in models:
            self.assertTrue((self.output_dir / f"test_{model_type}.py").exists())

    def test_syntax_validation_and_fixing(self):
        """Test syntax validation and fixing capability."""
        # Create a template with deliberate syntax errors
        broken_template = TemplateBase(self.config)
        broken_template.get_template_str = MagicMock(return_value="""
import os
import sys

# Missing closing parenthesis
print("Hello, world"

# Incorrect indentation
def test_function():
print("This is incorrectly indented")

# Class with methods
class TestClass:
def __init__(self):
    self.value = 42
    
def test_method(self):
    return self.value

if __name__ == "__main__":
    test_function()
    test_instance = TestClass()
    print(test_instance.test_method()
""")

        # Register the broken template
        self.registry.register_template("broken", broken_template)
        
        # Set up options
        model_type = "broken"
        options = {
            "output_file": self.output_dir / "test_broken.py",
            "model_name": "broken-model",
            "task": "default",
            "fix_syntax": True  # Enable syntax fixing
        }
        
        # Generate with syntax fixing
        result = self.generator.generate(model_type, options)
        
        # The generation should succeed despite the syntax errors because we're fixing them
        self.assertTrue(result["success"])
        self.assertTrue(Path(options["output_file"]).exists())
        
        # The generated file should have valid syntax
        with open(options["output_file"], "r") as f:
            content = f.read()
        
        # Validate the fixed content
        try:
            compile(content, options["output_file"], "exec")
            syntax_valid = True
        except SyntaxError:
            syntax_valid = False
        
        self.assertTrue(syntax_valid, "Fixed content should have valid syntax")

    def test_hardware_aware_generation(self):
        """Test hardware-aware test generation."""
        # Set up different hardware profiles for testing
        hardware_profiles = [
            # CUDA available
            {"cuda": {"available": True, "version": "11.7", "device_count": 1}, 
             "rocm": {"available": False}, "mps": {"available": False}},
            
            # ROCm available
            {"cuda": {"available": False}, 
             "rocm": {"available": True, "version": "5.2.0"}, 
             "mps": {"available": False}},
            
            # MPS available (macOS)
            {"cuda": {"available": False}, "rocm": {"available": False}, 
             "mps": {"available": True}}
        ]
        
        model_type = "bert"
        
        # Generate tests for each hardware profile
        for i, profile in enumerate(hardware_profiles):
            # Override the hardware detection
            self.hardware_manager.detect_all = MagicMock(return_value=profile)
            
            options = {
                "output_file": self.output_dir / f"test_bert_hw_{i}.py",
                "model_name": "bert-base-uncased",
                "task": "text-classification"
            }
            
            # Generate the test file
            result = self.generator.generate(model_type, options)
            
            # Verify generation succeeded
            self.assertTrue(result["success"])
            self.assertTrue(Path(options["output_file"]).exists())
            
            # Check that the hardware profile is reflected in the generated file
            with open(options["output_file"], "r") as f:
                content = f.read()
                
                # Check for hardware-specific code
                if profile["cuda"]["available"]:
                    self.assertIn("cuda", content.lower())
                    self.assertIn("torch.cuda.is_available()", content)
                
                if profile["rocm"]["available"]:
                    self.assertIn("rocm", content.lower())
                
                if profile["mps"]["available"]:
                    self.assertIn("mps", content.lower())
                    self.assertIn("torch.backends.mps.is_available()", content)

    def test_mock_environment_generation(self):
        """Test generation in a mock environment."""
        # Set up a mocked environment where dependencies are not available
        mock_dependency_manager = DependencyManager()
        mock_dependency_manager.check_all = MagicMock(return_value={
            "torch": {"available": False, "mocked": True},
            "transformers": {"available": False, "mocked": True},
            "tokenizers": {"available": False, "mocked": True},
            "sentencepiece": {"available": False, "mocked": True}
        })
        
        # Create a generator with mocked dependencies
        mock_generator = GeneratorCore(
            config=self.config,
            registry=self.registry,
            hardware_manager=self.hardware_manager,
            dependency_manager=mock_dependency_manager,
            model_selector=self.model_selector,
            syntax_validator=self.syntax_validator,
            syntax_fixer=self.syntax_fixer
        )
        
        model_type = "bert"
        options = {
            "output_file": self.output_dir / "test_bert_mocked.py",
            "model_name": "bert-base-uncased",
            "task": "text-classification"
        }
        
        # Generate the test file in mock mode
        result = mock_generator.generate(model_type, options)
        
        # Verify generation succeeded
        self.assertTrue(result["success"])
        self.assertTrue(Path(options["output_file"]).exists())
        
        # Check that mock support is included in the generated file
        with open(options["output_file"], "r") as f:
            content = f.read()
            self.assertIn("Mock", content)
            self.assertIn("patch", content)
            self.assertIn("MOCK_", content)  # Environment variable checks


if __name__ == "__main__":
    unittest.main()