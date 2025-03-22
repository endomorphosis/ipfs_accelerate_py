#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for the template system in the refactored generator suite.
Tests template base class, architecture-specific templates, and template rendering.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

# Add the parent directory to the path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from templates.base import TemplateBase
from templates.encoder_only import EncoderOnlyTemplate
from templates.decoder_only import DecoderOnlyTemplate
from templates.encoder_decoder import EncoderDecoderTemplate
from templates.vision import VisionTemplate
from templates.vision_text import VisionTextTemplate
from templates.speech import SpeechTemplate
from generator_core.config import ConfigManager


class TemplateBaseTest(unittest.TestCase):
    """Tests for the template base class."""

    def setUp(self):
        """Set up test environment."""
        self.config = ConfigManager()
        self.template = TemplateBase(self.config)

    def test_metadata(self):
        """Test template metadata."""
        metadata = self.template.get_metadata()
        self.assertIsInstance(metadata, dict)
        self.assertIn("name", metadata)
        self.assertIn("version", metadata)
        self.assertIn("description", metadata)
        self.assertIn("supported_architectures", metadata)

    def test_imports(self):
        """Test template imports."""
        imports = self.template.get_imports()
        self.assertIsInstance(imports, list)
        self.assertIn("import os", imports)
        self.assertIn("import sys", imports)
        self.assertIn("from unittest.mock import patch, MagicMock, Mock", imports)

    def test_render(self):
        """Test template rendering."""
        # Create a simple template string
        self.template.get_template_str = MagicMock(return_value="""
# {{ model_info.name }} Test
import os
import sys

# Hardware: {{ hardware_info.get('cuda', {}).get('available', False) }}
# Model: {{ model_info.name }}
# Task: {{ model_info.task }}
""")

        # Create context
        context = {
            "model_info": {
                "name": "Test Model",
                "id": "test-model",
                "task": "test-task"
            },
            "hardware_info": {
                "cuda": {
                    "available": True,
                    "version": "11.7"
                }
            }
        }

        # Render the template
        rendered = self.template.render(context)

        # Check rendering result
        self.assertIn("# Test Model Test", rendered)
        self.assertIn("# Hardware: True", rendered)
        self.assertIn("# Model: Test Model", rendered)
        self.assertIn("# Task: test-task", rendered)


class ArchitectureTemplatesTest(unittest.TestCase):
    """Tests for architecture-specific templates."""

    def setUp(self):
        """Set up test environment."""
        self.config = ConfigManager()
        
        # Initialize all templates
        self.encoder_only = EncoderOnlyTemplate(self.config)
        self.decoder_only = DecoderOnlyTemplate(self.config)
        self.encoder_decoder = EncoderDecoderTemplate(self.config)
        self.vision = VisionTemplate(self.config)
        self.vision_text = VisionTextTemplate(self.config)
        self.speech = SpeechTemplate(self.config)

    def test_template_metadata(self):
        """Test template metadata for architecture-specific templates."""
        # Check encoder-only metadata
        encoder_metadata = self.encoder_only.get_metadata()
        self.assertEqual("EncoderOnlyTemplate", encoder_metadata["name"])
        self.assertIn("encoder-only", encoder_metadata["supported_architectures"])
        
        # Check decoder-only metadata
        decoder_metadata = self.decoder_only.get_metadata()
        self.assertEqual("DecoderOnlyTemplate", decoder_metadata["name"])
        self.assertIn("decoder-only", decoder_metadata["supported_architectures"])
        
        # Check encoder-decoder metadata
        enc_dec_metadata = self.encoder_decoder.get_metadata()
        self.assertEqual("EncoderDecoderTemplate", enc_dec_metadata["name"])
        self.assertIn("encoder-decoder", enc_dec_metadata["supported_architectures"])
        
        # Check vision metadata
        vision_metadata = self.vision.get_metadata()
        self.assertEqual("VisionTemplate", vision_metadata["name"])
        self.assertIn("vision", vision_metadata["supported_architectures"])
        
        # Check vision-text metadata
        vision_text_metadata = self.vision_text.get_metadata()
        self.assertEqual("VisionTextTemplate", vision_text_metadata["name"])
        self.assertIn("vision-text", vision_text_metadata["supported_architectures"])
        
        # Check speech metadata
        speech_metadata = self.speech.get_metadata()
        self.assertEqual("SpeechTemplate", speech_metadata["name"])
        self.assertIn("speech", speech_metadata["supported_architectures"])

    def test_template_specific_imports(self):
        """Test architecture-specific imports."""
        # Check encoder-only imports
        encoder_imports = self.encoder_only.get_imports()
        self.assertIn("from transformers import AutoModelForMaskedLM, AutoTokenizer", encoder_imports)
        
        # Check decoder-only imports
        decoder_imports = self.decoder_only.get_imports()
        self.assertIn("from transformers import AutoModelForCausalLM, AutoTokenizer", decoder_imports)
        
        # Check encoder-decoder imports
        enc_dec_imports = self.encoder_decoder.get_imports()
        self.assertIn("from transformers import AutoModelForSeq2SeqLM, AutoTokenizer", enc_dec_imports)
        
        # Check vision imports
        vision_imports = self.vision.get_imports()
        self.assertIn("from transformers import AutoImageProcessor, AutoModelForImageClassification", vision_imports)
        self.assertIn("from PIL import Image", vision_imports)
        
        # Check vision-text imports
        vision_text_imports = self.vision_text.get_imports()
        self.assertIn("from transformers import CLIPProcessor, CLIPModel", vision_text_imports)
        self.assertIn("from PIL import Image", vision_text_imports)
        
        # Check speech imports
        speech_imports = self.speech.get_imports()
        self.assertIn("from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq", speech_imports)

    def test_template_renders(self):
        """Test that all templates can render with a context."""
        # Create a standard context for all templates
        context = {
            "model_info": {
                "name": "Test Model",
                "id": "test-model",
                "task": "test-task",
                "class_name": "TestModel"
            },
            "hardware_info": {
                "cuda": {"available": True, "version": "11.7"},
                "rocm": {"available": False},
                "mps": {"available": False}
            },
            "has_cuda": True,
            "has_rocm": False,
            "has_mps": False,
            "has_openvino": False,
            "has_webnn": False,
            "has_webgpu": False,
            "dependencies": {
                "torch": {"available": True, "version": "2.0.0"},
                "transformers": {"available": True, "version": "4.30.0"}
            }
        }
        
        # Test rendering for all templates
        templates = [
            self.encoder_only,
            self.decoder_only,
            self.encoder_decoder,
            self.vision,
            self.vision_text,
            self.speech
        ]
        
        for template in templates:
            rendered = template.render(context)
            self.assertIsNotNone(rendered)
            self.assertIsInstance(rendered, str)
            self.assertIn("Test Model", rendered)
            self.assertIn("import", rendered.lower())
            self.assertIn("class", rendered.lower())
            self.assertIn("def", rendered.lower())

    def test_encoder_only_template_specific_content(self):
        """Test encoder-only template specific content."""
        context = {
            "model_info": {
                "name": "BERT",
                "id": "bert-base-uncased",
                "task": "fill-mask",
                "class_name": "Bert"
            },
            "hardware_info": {"cuda": {"available": True}},
            "has_cuda": True,
            "dependencies": {"torch": {"available": True}, "transformers": {"available": True}}
        }
        
        rendered = self.encoder_only.render(context)
        
        # Check for encoder-only specific content
        self.assertIn("AutoModelForMaskedLM", rendered)
        self.assertIn("fill-mask", rendered)
        self.assertIn("class BertTest", rendered)
        self.assertIn("def test_masked_lm", rendered)

    def test_decoder_only_template_specific_content(self):
        """Test decoder-only template specific content."""
        context = {
            "model_info": {
                "name": "GPT-2",
                "id": "gpt2",
                "task": "text-generation",
                "class_name": "Gpt2"
            },
            "hardware_info": {"cuda": {"available": True}},
            "has_cuda": True,
            "dependencies": {"torch": {"available": True}, "transformers": {"available": True}}
        }
        
        rendered = self.decoder_only.render(context)
        
        # Check for decoder-only specific content
        self.assertIn("AutoModelForCausalLM", rendered)
        self.assertIn("text-generation", rendered)
        self.assertIn("class Gpt2Test", rendered)
        self.assertIn("def test_generation", rendered)

    def test_vision_template_specific_content(self):
        """Test vision template specific content."""
        context = {
            "model_info": {
                "name": "ViT",
                "id": "vit-base-patch16-224",
                "task": "image-classification",
                "class_name": "Vit"
            },
            "hardware_info": {"cuda": {"available": True}},
            "has_cuda": True,
            "dependencies": {"torch": {"available": True}, "transformers": {"available": True}}
        }
        
        rendered = self.vision.render(context)
        
        # Check for vision specific content
        self.assertIn("AutoImageProcessor", rendered)
        self.assertIn("from PIL import Image", rendered)
        self.assertIn("class VitTest", rendered)
        self.assertIn("def test_image_classification", rendered)

    def test_conditional_sections(self):
        """Test that conditional sections in templates work correctly."""
        # Test conditional CUDA section
        context_with_cuda = {
            "model_info": {"name": "Test", "id": "test", "task": "test-task", "class_name": "Test"},
            "hardware_info": {"cuda": {"available": True, "version": "11.7"}},
            "has_cuda": True,
            "has_rocm": False,
            "has_mps": False,
            "dependencies": {"torch": {"available": True}}
        }
        
        context_without_cuda = {
            "model_info": {"name": "Test", "id": "test", "task": "test-task", "class_name": "Test"},
            "hardware_info": {"cuda": {"available": False}},
            "has_cuda": False,
            "has_rocm": False,
            "has_mps": False,
            "dependencies": {"torch": {"available": True}}
        }
        
        # Test with encoder-only template
        cuda_rendered = self.encoder_only.render(context_with_cuda)
        no_cuda_rendered = self.encoder_only.render(context_without_cuda)
        
        # CUDA section should be present when CUDA is available
        self.assertIn("torch.cuda.is_available()", cuda_rendered)
        self.assertIn("device = 'cuda'", cuda_rendered)
        
        # CUDA section should handle unavailable CUDA
        self.assertIn("device = 'cpu'", no_cuda_rendered)
        
        # Test OpenVINO conditional section
        context_with_openvino = {
            "model_info": {"name": "Test", "id": "test", "task": "test-task", "class_name": "Test"},
            "hardware_info": {"cuda": {"available": False}, "openvino": {"available": True}},
            "has_cuda": False,
            "has_rocm": False,
            "has_mps": False,
            "has_openvino": True,
            "dependencies": {"torch": {"available": True}, "openvino": {"available": True}}
        }
        
        openvino_rendered = self.encoder_only.render(context_with_openvino)
        
        # OpenVINO section should be present when OpenVINO is available
        self.assertIn("openvino", openvino_rendered.lower())
        self.assertIn("test_openvino", openvino_rendered)

    def test_template_extension(self):
        """Test that templates properly extend the base template."""
        # Create a custom template extending TemplateBase
        class CustomTemplate(TemplateBase):
            def get_metadata(self):
                metadata = super().get_metadata()
                metadata.update({
                    "name": "CustomTemplate",
                    "description": "Custom template for testing",
                    "supported_architectures": ["custom"]
                })
                return metadata
                
            def get_imports(self):
                imports = super().get_imports()
                imports.extend([
                    "import custom_module",
                    "from custom_module import CustomClass"
                ])
                return imports
                
            def get_template_str(self):
                return """
# Custom Template
{{ model_info.name }} Custom Implementation

{% for imp in imports %}
{{ imp }}
{% endfor %}

class {{ model_info.class_name }}Test:
    def __init__(self):
        self.custom = True
        
    def test_custom(self):
        return True
"""
        
        # Create an instance of the custom template
        custom_template = CustomTemplate(self.config)
        
        # Test metadata
        metadata = custom_template.get_metadata()
        self.assertEqual("CustomTemplate", metadata["name"])
        self.assertEqual("Custom template for testing", metadata["description"])
        self.assertIn("custom", metadata["supported_architectures"])
        
        # Test imports
        imports = custom_template.get_imports()
        self.assertIn("import custom_module", imports)
        self.assertIn("from custom_module import CustomClass", imports)
        
        # Test rendering
        context = {
            "model_info": {
                "name": "Custom Model",
                "class_name": "CustomModel"
            },
            "imports": [
                "import numpy as np",
                "import pandas as pd"
            ]
        }
        
        rendered = custom_template.render(context)
        
        # Check custom content
        self.assertIn("Custom Model Custom Implementation", rendered)
        self.assertIn("class CustomModelTest", rendered)
        self.assertIn("def test_custom", rendered)
        self.assertIn("import numpy as np", rendered)
        self.assertIn("import pandas as pd", rendered)


if __name__ == "__main__":
    unittest.main()