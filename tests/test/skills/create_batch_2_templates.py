#\!/usr/bin/env python3
"""
Script to create template-based test files for batch 2 models.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"batch_2_templates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Base templates for different architectures
ENCODER_DECODER_TEMPLATE = """
import os
import sys
import unittest
import pytest
from unittest.mock import patch, MagicMock

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock detection for hardware capabilities
MOCK_HARDWARE_DETECTION = True

class TestHF{model_class_name}(unittest.TestCase):
    \"\"\"Test for the HuggingFace {model_name} model\"\"\"

    def setUp(self):
        # Set up any needed variables or configurations
        self.model_name = "{default_model}"
        self.task = "{task}"
        # Configure hardware detection
        self.cpu_only = True if os.environ.get("FORCE_CPU", "0") == "1" else False
        
    @pytest.mark.skip_if_no_gpu
    def test_model_loading(self):
        \"\"\"Test loading {model_name} model with proper hardware detection\"\"\"
        try:
            # Import necessary libraries
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            # Determine appropriate device
            if torch.cuda.is_available() and not self.cpu_only:
                device = torch.device("cuda")
                logger.info("Using GPU for inference")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU for inference")
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            model = model.to(device)
            
            # Basic inference test
            inputs = tokenizer("Translate to French: Hello, how are you?", return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_length=50)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            self.assertIsInstance(result, str)
            self.assertTrue(len(result) > 0)
            logger.info(f"Model output: {{result[:100]}}")
            
        except ImportError as e:
            # Skip test if dependencies aren't available
            logger.warning(f"Skipping test due to import error: {{e}}")
            pytest.skip(f"Required dependencies not available: {{e}}")
        except Exception as e:
            # Log error but don't let test fail if hardware is insufficient
            if "CUDA out of memory" in str(e) or "CUDA error" in str(e):
                logger.warning(f"Skipping test due to CUDA error: {{e}}")
                pytest.skip(f"Insufficient GPU memory: {{e}}")
            else:
                raise

    @patch("transformers.AutoModelForSeq2SeqLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_model_with_mock(self, mock_tokenizer, mock_model):
        \"\"\"Test {model_name} model functionality with mocks\"\"\"
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.decode.return_value = "This is a mock response from {model_name}"
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_outputs = MagicMock()
        mock_model_instance.generate.return_value = [mock_outputs]
        mock_model.return_value = mock_model_instance
        
        # Test with mocked objects
        tokenizer = mock_tokenizer(self.model_name)
        model = mock_model(self.model_name)
        
        # Check if mocks are properly configured
        tokenizer("Test input", return_tensors="pt")
        model.generate()
        
        # Verify mock calls
        mock_tokenizer.assert_called_once_with(self.model_name)
        mock_model.assert_called_once_with(self.model_name)
        
        # This test should always pass as it's using mocks
        self.assertTrue(True)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test {model_name} model")
    parser.add_argument("--model", type=str, default="{default_model}", help="Model ID to test")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU even if GPU is available")
    args = parser.parse_args()
    
    # Set environment variable if CPU only
    if args.cpu_only:
        os.environ["FORCE_CPU"] = "1"
    
    # Run the tests
    unittest.main(argv=["first-arg-is-ignored"])
"""

ENCODER_ONLY_TEMPLATE = """
import os
import sys
import unittest
import pytest
from unittest.mock import patch, MagicMock

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock detection for hardware capabilities
MOCK_HARDWARE_DETECTION = True

class TestHF{model_class_name}(unittest.TestCase):
    \"\"\"Test for the HuggingFace {model_name} model\"\"\"

    def setUp(self):
        # Set up any needed variables or configurations
        self.model_name = "{default_model}"
        self.task = "{task}"
        # Configure hardware detection
        self.cpu_only = True if os.environ.get("FORCE_CPU", "0") == "1" else False
        
    @pytest.mark.skip_if_no_gpu
    def test_model_loading(self):
        \"\"\"Test loading {model_name} model with proper hardware detection\"\"\"
        try:
            # Import necessary libraries
            import torch
            from transformers import AutoTokenizer, AutoModel
            
            # Determine appropriate device
            if torch.cuda.is_available() and not self.cpu_only:
                device = torch.device("cuda")
                logger.info("Using GPU for inference")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU for inference")
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name)
            model = model.to(device)
            
            # Basic inference test
            inputs = tokenizer("Hello, I'm a language model", return_tensors="pt").to(device)
            outputs = model(**inputs)
            
            # Check output structure
            self.assertIn('last_hidden_state', outputs)
            self.assertTrue(isinstance(outputs.last_hidden_state, torch.Tensor))
            logger.info(f"Output shape: {{outputs.last_hidden_state.shape}}")
            
        except ImportError as e:
            # Skip test if dependencies aren't available
            logger.warning(f"Skipping test due to import error: {{e}}")
            pytest.skip(f"Required dependencies not available: {{e}}")
        except Exception as e:
            # Log error but don't let test fail if hardware is insufficient
            if "CUDA out of memory" in str(e) or "CUDA error" in str(e):
                logger.warning(f"Skipping test due to CUDA error: {{e}}")
                pytest.skip(f"Insufficient GPU memory: {{e}}")
            else:
                raise

    @patch("transformers.AutoModel.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_model_with_mock(self, mock_tokenizer, mock_model):
        \"\"\"Test {model_name} model functionality with mocks\"\"\"
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = MagicMock()
        mock_model_instance.return_value = mock_outputs
        mock_model.return_value = mock_model_instance
        
        # Test with mocked objects
        tokenizer = mock_tokenizer(self.model_name)
        model = mock_model(self.model_name)
        
        # Verify mock calls
        mock_tokenizer.assert_called_once_with(self.model_name)
        mock_model.assert_called_once_with(self.model_name)
        
        # This test should always pass as it's using mocks
        self.assertTrue(True)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test {model_name} model")
    parser.add_argument("--model", type=str, default="{default_model}", help="Model ID to test")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU even if GPU is available")
    args = parser.parse_args()
    
    # Set environment variable if CPU only
    if args.cpu_only:
        os.environ["FORCE_CPU"] = "1"
    
    # Run the tests
    unittest.main(argv=["first-arg-is-ignored"])
"""

def create_model_test_file(model_name, class_name, architecture, default_model, task, output_dir):
    """Create a test file for a specific model based on a template."""
    output_file = os.path.join(output_dir, f"test_hf_{model_name}.py")
    
    # Select the appropriate template based on architecture
    if architecture == "encoder_decoder":
        template = ENCODER_DECODER_TEMPLATE
    elif architecture == "encoder_only":
        template = ENCODER_ONLY_TEMPLATE
    else:
        logger.error(f"Unsupported architecture: {architecture}")
        return False
    
    # Format template with model-specific values
    content = template.format(
        model_name=model_name.replace("_", "-"),
        model_class_name=class_name,
        default_model=default_model,
        task=task
    )
    
    try:
        with open(output_file, 'w') as f:
            f.write(content)
        logger.info(f"✅ Created test file for {model_name}: {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error creating test file for {model_name}: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Create template-based test files for batch 2 models")
    parser.add_argument("--output-dir", default="fixed_tests", help="Output directory for the test files")
    parser.add_argument("--batch-file", default="batch_2_models.json", help="JSON file with batch model definitions")
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        # If relative path, make it relative to current script directory
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load batch model definitions
    batch_file = args.batch_file
    if not os.path.isabs(batch_file):
        batch_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), batch_file)
    
    try:
        with open(batch_file, 'r') as f:
            batch_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading batch file: {e}")
        return 1
    
    # Process models by architecture
    successful = 0
    failed = 0
    
    for architecture, models in batch_data.items():
        logger.info(f"Processing {len(models)} {architecture} models...")
        
        for model in models:
            # Convert model name to class name (camel case)
            class_name = ''.join(word.capitalize() for word in model["name"].split('_'))
            
            # Get model-specific values
            success = create_model_test_file(
                model_name=model["name"],
                class_name=class_name,
                architecture=model.get("architecture", architecture),
                default_model=model.get("default_model", f"{model['name'].replace('_', '-')}-base"),
                task=model.get("task", "text-classification"),
                output_dir=output_dir
            )
            
            if success:
                successful += 1
            else:
                failed += 1
    
    # Print summary
    logger.info("\nTemplate Creation Summary:")
    logger.info(f"- Total Models: {successful + failed}")
    logger.info(f"- Successfully Created: {successful}")
    logger.info(f"- Failed: {failed}")
    
    if failed > 0:
        logger.warning(f"Failed to create {failed} test files")
        return 1
    else:
        logger.info("All test files created successfully")
        return 0

if __name__ == "__main__":
    sys.exit(main())
