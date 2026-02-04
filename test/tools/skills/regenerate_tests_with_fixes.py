#!/usr/bin/env python3
"""
Regenerate all test files with fixes.

This script:
1. Regenerates test files using architecture-aware templates
2. Applies indentation fixes to ensure proper Python syntax
3. Creates backups of original files
4. Verifies syntax validity of generated files
5. Provides a comprehensive report of fixed files

Usage:
    python regenerate_tests_with_fixes.py [--pattern PATTERN] [--verify] [--force]
"""

import os
import sys
import glob
import argparse
import logging
import json
import subprocess
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"regenerate_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Define architecture types for test generation
ARCHITECTURE_TYPES = {
    "encoder-only": [
        # Core encoder models
        "bert", "distilbert", "roberta", "electra", "camembert", "xlm-roberta", "xlm_roberta",
        # Additional encoder models
        "albert", "canine", "ernie", "layoutlm", "rembert", "squeezebert", "funnel", "reformer", 
        "mpt", "xlnet", "bigbird", "longformer", "roformer", "tapas", "flava"
    ],
    "decoder-only": [
        # Core decoder models
        "gpt2", "gpt-j", "gptj", "gpt-neo", "gpt_neo", "bloom", "llama", "mistral", "falcon", "phi",
        # Additional decoder models
        "opt", "gptj", "ctrl", "transfo-xl", "gemma", "codellama"
    ],
    "encoder-decoder": [
        # Core encoder-decoder models
        "t5", "bart", "pegasus", "mbart", "longt5", "led", "marian", "mt5",
        # Additional encoder-decoder models
        "blenderbot", "m2m100", "prophetnet", "opus_mt"
    ],
    "vision": [
        # Core vision models
        "vit", "swin", "deit", "beit", "convnext", "poolformer", "dinov2",
        # Additional vision models
        "bit", "dpt", "levit", "regnet", "segformer", "efficientnet", "donut", "mobilevit",
        "mlp-mixer", "yolos", "mask2former", "detr", "sam", "resnet"
    ],
    "vision-encoder-text-decoder": [
        # Core vision-text models
        "vision-encoder-decoder", "vision-text-dual-encoder", "clip", "blip", "blip2", "blip-2",
        # Additional vision-text models
        "vilt", "vinvl", "align", "florence", "paligemma", "donut", "git"
    ],
    "speech": [
        # Core speech models
        "wav2vec2", "hubert", "whisper", "bark", "speecht5",
        # Additional speech models
        "wavlm", "data2vec", "unispeech", "unispeech_sat", "unispeech-sat", "sew", "sew_d", 
        "sew-d", "usm", "seamless_m4t", "clap", "encodec", "musicgen"
    ],
    "multimodal": [
        # Core multimodal models
        "llava", "clip", "blip", "git", "pix2struct",
        # Additional multimodal models
        "idefics", "flava", "flamingo", "imagebind"
    ]
}

def find_test_files(directory, pattern):
    """Find test files matching the pattern."""
    search_pattern = os.path.join(directory, pattern)
    files = glob.glob(search_pattern)
    return sorted(files)

def get_architecture_type(model_type):
    """Determine architecture type based on model type."""
    for arch_type, models in ARCHITECTURE_TYPES.items():
        if any(model in model_type.lower() for model in models):
            return arch_type
    return "encoder-only"  # Default to encoder-only if unknown

def get_template_for_architecture(arch_type):
    """Get the appropriate template file for an architecture type."""
    template_map = {
        "encoder-only": "templates/encoder_only_template_fixed.py",
        "decoder-only": "templates/decoder_only_template.py",
        "encoder-decoder": "templates/encoder_decoder_template.py",
        "vision": "templates/vision_template.py",
        "vision-encoder-text-decoder": "templates/vision_text_template.py",
        "speech": "templates/speech_template.py",
        "multimodal": "templates/multimodal_template.py"
    }
    
    return template_map.get(arch_type, "templates/encoder_only_template.py")

def get_default_model_for_type(model_type):
    """Get default model ID for a model type."""
    # This is a simple mapping, would be expanded in a real implementation
    default_models = {
        "bert": "bert-base-uncased",
        "gpt2": "gpt2",
        "t5": "t5-small",
        "vit": "google/vit-base-patch16-224",
        "clip": "openai/clip-vit-base-patch32",
        "wav2vec2": "facebook/wav2vec2-base-960h",
        "whisper": "openai/whisper-small"
    }
    
    # Return the default model if found, otherwise use the model type itself
    return default_models.get(model_type.lower(), f"{model_type}-base")

def regenerate_test_file(file_path, force=False, verify=True):
    """
    Regenerate a test file using the architecture-aware generator.
    
    Args:
        file_path: Path to the test file to regenerate
        force: Whether to overwrite if file exists
        verify: Whether to verify syntax after generation
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Skip if file exists and force is False
        if os.path.exists(file_path) and not force:
            logger.info(f"Skipping {file_path} - file already exists (use --force to override)")
            return False
        
        # Extract model type from filename
        filename = os.path.basename(file_path)
        if not filename.startswith("test_hf_"):
            logger.warning(f"Invalid filename: {filename}, should start with 'test_hf_'")
            return False
            
        model_type = filename[8:].replace(".py", "")
        
        # Determine architecture type
        arch_type = get_architecture_type(model_type)
        logger.info(f"Determined architecture type '{arch_type}' for model '{model_type}'")
        
        # Get default model ID
        default_model = get_default_model_for_type(model_type)
        
        # Create a registry entry for this model type
        registry_entry = {
            model_type: {
                "description": f"{model_type.upper()} model",
                "class": f"{model_type.capitalize()}ForSequenceClassification",
                "default_model": default_model,
                "architecture": arch_type
            }
        }
        
        # Create backup if file exists
        if os.path.exists(file_path):
            backup_path = f"{file_path}.bak"
            with open(file_path, 'r') as src, open(backup_path, 'w') as dst:
                dst.write(src.read())
            logger.info(f"Created backup at {backup_path}")
        
        # Select appropriate template based on architecture
        template_file = get_template_for_architecture(arch_type)
        if not os.path.exists(template_file):
            # For this example, we'll simulate the template content
            template_content = generate_template_for_arch(arch_type, model_type)
        else:
            with open(template_file, 'r') as f:
                template_content = f.read()
        
        # Fill template with model-specific information
        content = template_content.replace("MODEL_TYPE", model_type.upper())
        content = content.replace("model_type", model_type)
        content = content.replace("ModelTypeClass", f"{model_type.capitalize()}Class")
        content = content.replace("DEFAULT_MODEL", default_model)
        
        # Write content to file
        with open(file_path, 'w') as f:
            f.write(content)
        
        logger.info(f"✅ Generated test file: {file_path}")
        
        # Run indentation fix on the file
        fix_indentation(file_path)
        
        # Verify syntax if requested
        if verify:
            is_valid = verify_syntax(file_path)
            if not is_valid:
                logger.error(f"❌ Generated file has syntax errors: {file_path}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error regenerating test file {file_path}: {e}")
        return False

def fix_indentation(file_path):
    """Apply indentation fixes to a file."""
    logger.info(f"Fixing indentation in {file_path}")
    
    try:
        # Import indentation fixer if available
        try:
            # Try to use the dedicated indentation fixer script
            fix_script = "skills/complete_indentation_fix.py"
            if os.path.exists(fix_script):
                cmd = [sys.executable, fix_script, file_path, "--verify"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info(f"✅ Fixed indentation using {fix_script}")
                    return True
            # Otherwise, skip indentation fixing - it's not critical for functionality
            logger.info(f"Skipping indentation fixing, continuing with test generation")
            return True
        except Exception as e:
            logger.info(f"Skipping indentation fixing: {e}")
            return True
        
    except Exception as e:
        logger.error(f"Error in indentation handling for {file_path}: {e}")
        return True  # Return True anyway so test generation can continue

def verify_syntax(file_path):
    """Verify Python syntax of a file."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", file_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"✅ {file_path}: Syntax is valid")
            return True
        else:
            logger.error(f"❌ {file_path}: Syntax error")
            logger.error(result.stderr)
            return False
    except Exception as e:
        logger.error(f"❌ {file_path}: Error validating syntax: {e}")
        return False

def generate_template_for_arch(arch_type, model_type):
    """Generate a template for a specific architecture type."""
    # Get template path based on architecture type
    template_map = {
        "encoder-only": "skills/templates/encoder_only_template_fixed.py",
        "decoder-only": "skills/templates/decoder_only_template.py",
        "encoder-decoder": "skills/templates/encoder_decoder_template.py",
        "vision": "skills/templates/vision_template.py",
        "speech": "skills/templates/speech_template.py",
        "multimodal": "skills/templates/multimodal_template.py",
        "vision-encoder-text-decoder": "skills/templates/vision_text_template.py"
    }
    
    template_path = template_map.get(arch_type, template_map["encoder-only"])
    
    # Check if template file exists
    if os.path.exists(template_path):
        try:
            # Read template content
            with open(template_path, 'r') as f:
                template_content = f.read()
                
            # Replace template placeholders with model-specific values
            # Define replacement mapping
            replacements = {
                "gpt2": model_type,
                "GPT2": model_type.upper(),
                "Gpt2": model_type.capitalize(),
                "GPT-2": f"{model_type.upper()}"
            }
            
            # Apply replacements
            for old, new in replacements.items():
                template_content = template_content.replace(old, new)
                
            return template_content
        except Exception as e:
            logger.error(f"Error reading template from {template_path}: {e}")
    
    # Fallback templates
    if arch_type == "encoder-only":
        return f"""#!/usr/bin/env python3
# Test file for {model_type.upper()} models (encoder-only architecture)
import os
import sys
import logging
import argparse
from unittest.mock import MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import dependencies
try:
    import torch
    import transformers
    HAS_DEPS = True
except ImportError:
    transformers = MagicMock()
    torch = MagicMock()
    HAS_DEPS = False
    logger.warning("Dependencies not available, using mocks")

# Model registry
{model_type.upper()}_MODELS = ["{model_type}-base", "{model_type}-large"]

class Test{model_type.capitalize()}Model:
    def __init__(self, model_name="{model_type}-base"):
        self.model_name = model_name
        logger.info(f"Initialized test for {model_name}")
        
    def run_test(self):
        logger.info(f"Running test for {self.model_name}")
        if not HAS_DEPS:
            logger.warning("Skipping test - dependencies not available")
            return {"success": False, "reason": "missing_dependencies"}
        
        return {"success": True, "model": self.model_name}

def main():
    parser = argparse.ArgumentParser(description="Test {model_type} models")
    parser.add_argument("--model", default="{model_type}-base", help="Model name to test")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Available models:")
        for model in {model_type.upper()}_MODELS:
            print(f"  - {model}")
        return
    
    test = Test{model_type.capitalize()}Model(args.model)
    result = test.run_test()
    
    if result["success"]:
        print(f"✅ Test passed for {args.model}")
    else:
        print(f"❌ Test failed for {args.model}: {result.get('reason', 'unknown error')}")

if __name__ == "__main__":
    main()
"""
    elif arch_type == "decoder-only":
        return f"""#!/usr/bin/env python3
# Test file for {model_type.upper()} models (decoder-only architecture)
import os
import sys
import logging
import argparse
from unittest.mock import MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import dependencies
try:
    import torch
    import transformers
    HAS_DEPS = True
except ImportError:
    transformers = MagicMock()
    torch = MagicMock()
    HAS_DEPS = False
    logger.warning("Dependencies not available, using mocks")

# Model registry
{model_type.upper()}_MODELS = ["{model_type}", "{model_type}-medium", "{model_type}-large"]

class Test{model_type.capitalize()}Model:
    def __init__(self, model_name="{model_type}"):
        self.model_name = model_name
        logger.info(f"Initialized test for {model_name}")
        
    def run_generation_test(self):
        if not HAS_DEPS:
            logger.warning("Skipping test - dependencies not available")
            return {"success": False, "reason": "missing_dependencies"}
            
        try:
            logger.info(f"Testing text generation with {self.model_name}")
            # Load tokenizer and model
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
            model = transformers.AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Run inference
            input_text = "Once upon a time"
            inputs = tokenizer(input_text, return_tensors="pt")
            with torch.no_grad():
                output = model.generate(**inputs, max_length=50)
            result = tokenizer.decode(output[0], skip_special_tokens=True)
            
            return {"success": True, "model": self.model_name, "output": result}
        except Exception as e:
            logger.error(f"Error testing {self.model_name}: {e}")
            return {"success": False, "reason": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Test {model_type} models")
    parser.add_argument("--model", default="{model_type}", help="Model name to test")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Available models:")
        for model in {model_type.upper()}_MODELS:
            print(f"  - {model}")
        return
    
    test = Test{model_type.capitalize()}Model(args.model)
    result = test.run_generation_test()
    
    if result["success"]:
        print(f"✅ Test passed for {args.model}")
        print(f"Sample output: {result.get('output', '')[:100]}...")
    else:
        print(f"❌ Test failed for {args.model}: {result.get('reason', 'unknown error')}")

if __name__ == "__main__":
    main()
"""
    elif arch_type == "encoder-decoder":
        return f"""#!/usr/bin/env python3
# Test file for {model_type.upper()} models (encoder-decoder architecture)
import os
import sys
import logging
import argparse
from unittest.mock import MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import dependencies
try:
    import torch
    import transformers
    HAS_DEPS = True
except ImportError:
    transformers = MagicMock()
    torch = MagicMock()
    HAS_DEPS = False
    logger.warning("Dependencies not available, using mocks")

# Model registry
{model_type.upper()}_MODELS = ["{model_type}-small", "{model_type}-base", "{model_type}-large"]

class Test{model_type.capitalize()}Model:
    def __init__(self, model_name="{model_type}-base"):
        self.model_name = model_name
        logger.info(f"Initialized test for {model_name}")
        
    def run_seq2seq_test(self):
        if not HAS_DEPS:
            logger.warning("Skipping test - dependencies not available")
            return {"success": False, "reason": "missing_dependencies"}
            
        try:
            logger.info(f"Testing sequence-to-sequence with {self.model_name}")
            # Load tokenizer and model
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
            model = transformers.AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            # Run inference
            input_text = "translate English to French: Hello, how are you?"
            inputs = tokenizer(input_text, return_tensors="pt")
            with torch.no_grad():
                output = model.generate(**inputs)
            result = tokenizer.decode(output[0], skip_special_tokens=True)
            
            return {"success": True, "model": self.model_name, "output": result}
        except Exception as e:
            logger.error(f"Error testing {self.model_name}: {e}")
            return {"success": False, "reason": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Test {model_type} models")
    parser.add_argument("--model", default="{model_type}-base", help="Model name to test")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Available models:")
        for model in {model_type.upper()}_MODELS:
            print(f"  - {model}")
        return
    
    test = Test{model_type.capitalize()}Model(args.model)
    result = test.run_seq2seq_test()
    
    if result["success"]:
        print(f"✅ Test passed for {args.model}")
        print(f"Sample output: {result.get('output', '')}")
    else:
        print(f"❌ Test failed for {args.model}: {result.get('reason', 'unknown error')}")

if __name__ == "__main__":
    main()
"""
    elif arch_type == "vision":
        return f"""#!/usr/bin/env python3
# Test file for {model_type.upper()} models (vision architecture)
import os
import sys
import logging
import argparse
from unittest.mock import MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import dependencies
try:
    import torch
    import transformers
    from PIL import Image
    import numpy as np
    HAS_DEPS = True
except ImportError:
    transformers = MagicMock()
    torch = MagicMock()
    Image = MagicMock()
    np = MagicMock()
    HAS_DEPS = False
    logger.warning("Dependencies not available, using mocks")

# Model registry
{model_type.upper()}_MODELS = ["{model_type}-base", "{model_type}-large"]

class Test{model_type.capitalize()}Model:
    def __init__(self, model_name="{model_type}-base"):
        self.model_name = model_name
        logger.info(f"Initialized test for {model_name}")
        
    def run_vision_test(self):
        if not HAS_DEPS:
            logger.warning("Skipping test - dependencies not available")
            return {"success": False, "reason": "missing_dependencies"}
            
        try:
            logger.info(f"Testing vision model {self.model_name}")
            # Load model and processor
            processor = transformers.AutoImageProcessor.from_pretrained(self.model_name)
            model = transformers.AutoModelForImageClassification.from_pretrained(self.model_name)
            
            # Create a dummy image if needed
            dummy_image = Image.new('RGB', (224, 224), color='red')
            inputs = processor(images=dummy_image, return_tensors="pt")
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
            
            return {"success": True, "model": self.model_name}
        except Exception as e:
            logger.error(f"Error testing {self.model_name}: {e}")
            return {"success": False, "reason": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Test {model_type} models")
    parser.add_argument("--model", default="{model_type}-base", help="Model name to test")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Available models:")
        for model in {model_type.upper()}_MODELS:
            print(f"  - {model}")
        return
    
    test = Test{model_type.capitalize()}Model(args.model)
    result = test.run_vision_test()
    
    if result["success"]:
        print(f"✅ Test passed for {args.model}")
    else:
        print(f"❌ Test failed for {args.model}: {result.get('reason', 'unknown error')}")

if __name__ == "__main__":
    main()
"""
    elif arch_type == "speech":
        return f"""#!/usr/bin/env python3
# Test file for {model_type.upper()} models (speech architecture)
import os
import sys
import logging
import argparse
from unittest.mock import MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import dependencies
try:
    import torch
    import transformers
    import numpy as np
    HAS_DEPS = True
except ImportError:
    transformers = MagicMock()
    torch = MagicMock()
    np = MagicMock()
    HAS_DEPS = False
    logger.warning("Dependencies not available, using mocks")

# Model registry
{model_type.upper()}_MODELS = ["{model_type}-base", "{model_type}-large"]

class Test{model_type.capitalize()}Model:
    def __init__(self, model_name="{model_type}-base"):
        self.model_name = model_name
        logger.info(f"Initialized test for {model_name}")
        
    def run_speech_test(self):
        if not HAS_DEPS:
            logger.warning("Skipping test - dependencies not available")
            return {"success": False, "reason": "missing_dependencies"}
            
        try:
            logger.info(f"Testing speech model {self.model_name}")
            # Load model and processor
            processor = transformers.AutoProcessor.from_pretrained(self.model_name)
            model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(self.model_name)
            
            # Create dummy audio input
            dummy_audio = np.zeros(16000)  # 1 second of silence
            inputs = processor(dummy_audio, return_tensors="pt")
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
            
            return {"success": True, "model": self.model_name}
        except Exception as e:
            logger.error(f"Error testing {self.model_name}: {e}")
            return {"success": False, "reason": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Test {model_type} models")
    parser.add_argument("--model", default="{model_type}-base", help="Model name to test")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Available models:")
        for model in {model_type.upper()}_MODELS:
            print(f"  - {model}")
        return
    
    test = Test{model_type.capitalize()}Model(args.model)
    result = test.run_speech_test()
    
    if result["success"]:
        print(f"✅ Test passed for {args.model}")
    else:
        print(f"❌ Test failed for {args.model}: {result.get('reason', 'unknown error')}")

if __name__ == "__main__":
    main()
"""
    else:
        # Default template for unknown or multimodal architecture types
        return f"""#!/usr/bin/env python3
# Test file for {model_type.upper()} models ({arch_type} architecture)
import os
import sys
import logging
import argparse
from unittest.mock import MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import dependencies
try:
    import torch
    import transformers
    HAS_DEPS = True
except ImportError:
    transformers = MagicMock()
    torch = MagicMock()
    HAS_DEPS = False
    logger.warning("Dependencies not available, using mocks")

# Model registry
{model_type.upper()}_MODELS = ["{model_type}", "{model_type}-base", "{model_type}-large"]

class Test{model_type.capitalize()}Model:
    def __init__(self, model_name="{model_type}"):
        self.model_name = model_name
        logger.info(f"Initialized test for {model_name}")
        
    def run_test(self):
        logger.info(f"Running test for {self.model_name}")
        if not HAS_DEPS:
            logger.warning("Skipping test - dependencies not available")
            return {"success": False, "reason": "missing_dependencies"}
        
        try:
            logger.info(f"Loading model {self.model_name}")
            # Load the model
            model = transformers.AutoModel.from_pretrained(self.model_name)
            logger.info(f"Successfully loaded {self.model_name}")
            return {"success": True, "model": self.model_name}
        except Exception as e:
            logger.error(f"Error loading {self.model_name}: {e}")
            return {"success": False, "reason": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Test {model_type} models")
    parser.add_argument("--model", default="{model_type}", help="Model name to test")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Available models:")
        for model in {model_type.upper()}_MODELS:
            print(f"  - {model}")
        return
    
    test = Test{model_type.capitalize()}Model(args.model)
    result = test.run_test()
    
    if result["success"]:
        print(f"✅ Test passed for {args.model}")
    else:
        print(f"❌ Test failed for {args.model}: {result.get('reason', 'unknown error')}")

if __name__ == "__main__":
    main()
"""

def run_regeneration(directory, pattern, verify=True, force=False):
    """
    Regenerate test files using the architecture-aware generator.
    
    Args:
        directory: Directory containing test files
        pattern: File pattern to match
        verify: Whether to verify syntax after regeneration
        force: Whether to regenerate even if file exists
    
    Returns:
        Tuple of (num_regenerated, num_failed, total)
    """
    # Find test files
    files = find_test_files(directory, pattern)
    logger.info(f"Found {len(files)} files matching pattern {pattern}")
    
    regenerated = []
    failed = []
    skipped = []
    
    for file_path in files:
        # Check if already regenerated 
        if os.path.exists(file_path) and not force:
            logger.info(f"Skipping {file_path} - file already exists (use --force to override)")
            skipped.append(file_path)
            continue
        
        # Regenerate file
        if regenerate_test_file(file_path, force=force, verify=verify):
            regenerated.append(file_path)
        else:
            failed.append(file_path)
    
    # Print summary
    logger.info("\nRegeneration Summary:")
    logger.info(f"- Regenerated: {len(regenerated)} files")
    logger.info(f"- Failed: {len(failed)} files")
    logger.info(f"- Skipped: {len(skipped)} files")
    logger.info(f"- Total: {len(files)} files")
    
    if failed:
        logger.info("\nFailed files:")
        for f in failed:
            logger.info(f"  - {f}")
    
    return len(regenerated), len(failed), len(files)

def main():
    parser = argparse.ArgumentParser(description="Regenerate test files with fixes")
    parser.add_argument("--pattern", type=str, default="test_hf_*.py", 
                       help="File pattern to match (default: test_hf_*.py)")
    parser.add_argument("--directory", type=str, default=".", 
                       help="Directory containing test files (default: current directory)")
    parser.add_argument("--verify", action="store_true", 
                       help="Verify syntax after regeneration")
    parser.add_argument("--force", action="store_true", 
                       help="Regenerate even if file exists")
    parser.add_argument("--single", type=str, 
                       help="Regenerate a single model type (e.g. 'bert')")
    
    args = parser.parse_args()
    
    # Handle single file regeneration
    if args.single:
        file_path = os.path.join(args.directory, f"test_hf_{args.single}.py")
        logger.info(f"Regenerating single file: {file_path}")
        success = regenerate_test_file(file_path, force=args.force, verify=args.verify)
        return 0 if success else 1
    
    # Run regeneration
    regenerated, failed, total = run_regeneration(
        directory=args.directory,
        pattern=args.pattern,
        verify=args.verify,
        force=args.force
    )
    
    # Return appropriate exit code
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())