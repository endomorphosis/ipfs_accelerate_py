#!/usr/bin/env python3
"""
Fix template generation issues with proper indentation.

This script:
1. Fixes indentation of custom imports
2. Fixes indentation in special handling code
3. Generates template-compliant tests for manually created models
"""

import os
import sys
import re
import logging
import shutil
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = SCRIPT_DIR.parent
SKILLS_DIR = REPO_ROOT / "skills"
TEMPLATES_DIR = SKILLS_DIR / "templates"
FIXED_TESTS_DIR = SKILLS_DIR / "fixed_tests"

# Ensure directories exist
os.makedirs(FIXED_TESTS_DIR, exist_ok=True)

# Models to fix
MODELS_TO_FIX = [
    "layoutlmv2",
    "layoutlmv3", 
    "clvp",
    "seamless_m4t_v2"
]

def get_template_for_model(model_name):
    """Get the correct template for a model."""
    model_architectures = {
        "layoutlmv2": "vision-encoder-text-decoder",
        "layoutlmv3": "vision-encoder-text-decoder",
        "clvp": "speech",
        "bigbird": "encoder-decoder",
        "seamless_m4t_v2": "speech",
        "xlm_prophetnet": "encoder-decoder"
    }
    
    architecture_templates = {
        "vision-encoder-text-decoder": "vision_text_template.py",
        "speech": "speech_template.py", 
        "encoder-decoder": "encoder_decoder_template.py"
    }
    
    architecture = model_architectures.get(model_name, "encoder-only")
    template_file = architecture_templates.get(architecture, "encoder_only_template.py")
    
    return TEMPLATES_DIR / template_file

def get_model_config(model_name):
    """Get model-specific configuration."""
    model_configs = {
        "layoutlmv2": {
            "model_id": "microsoft/layoutlmv2-base-uncased",
            "class_name": "LayoutLMv2ForSequenceClassification",
            "processor_class": "LayoutLMv2Processor",
            "custom_imports": [
                "from PIL import Image",
                "import numpy as np"
            ]
        },
        "layoutlmv3": {
            "model_id": "microsoft/layoutlmv3-base",
            "class_name": "LayoutLMv3ForSequenceClassification",
            "processor_class": "LayoutLMv3Processor",
            "custom_imports": [
                "from PIL import Image",
                "import numpy as np"
            ]
        },
        "clvp": {
            "model_id": "susnato/clvp_dev",
            "class_name": "CLVPForCausalLM",
            "processor_class": "AutoProcessor",
            "custom_imports": [
                "import numpy as np",
                "import librosa"
            ]
        },
        "bigbird": {
            "model_id": "google/bigbird-roberta-base",
            "class_name": "BigBirdForSequenceClassification",
            "processor_class": "AutoTokenizer",
            "custom_imports": [
                "import numpy as np"
            ]
        },
        "seamless_m4t_v2": {
            "model_id": "facebook/seamless-m4t-v2-large",
            "class_name": "SeamlessM4TModel",
            "processor_class": "AutoProcessor",
            "custom_imports": [
                "import numpy as np",
                "import librosa"
            ]
        },
        "xlm_prophetnet": {
            "model_id": "microsoft/xprophetnet-large-wiki100-cased",
            "class_name": "XLMProphetNetForConditionalGeneration",
            "processor_class": "AutoTokenizer",
            "custom_imports": [
                "import numpy as np"
            ]
        }
    }
    
    return model_configs.get(model_name, {})

def read_template(template_path):
    """Read a template file."""
    try:
        with open(template_path, 'r') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading template {template_path}: {e}")
        return None

def customize_template(template_path, output_path, model_params):
    """Customize a template with model-specific parameters.
    
    Args:
        template_path: Path to the template file
        output_path: Path to the output file
        model_params: Dict containing model parameters
        
    Returns:
        Bool indicating success or failure
    """
    if not os.path.exists(template_path):
        logger.error(f"Template not found: {template_path}")
        return False
    
    # Read template
    template_content = read_template(template_path)
    if not template_content:
        logger.error(f"Failed to read template: {template_path}")
        return False
    
    # Extract parameters
    model_name = model_params.get("model_name", "")
    model_id = model_params.get("model_id", model_name)
    timestamp = model_params.get("timestamp", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    architecture = model_params.get("architecture", "")
    base_class = model_params.get("base_class", "")
    
    # Get sanitized model name for class names to avoid syntax errors
    if "sanitized_model_name" in model_params:
        model_name_clean = model_params["sanitized_model_name"]
    else:
        # If model_name contains organization, strip it for class naming
        model_name_clean = model_name.split('/')[-1] if '/' in model_name else model_name
        # Replace hyphens with underscores to avoid syntax errors
        model_name_clean = model_name_clean.replace("-", "_")
    
    # Basic replacements
    content = template_content
    
    # Replace model names
    if model_name:
        content = content.replace("MODEL_NAME", model_name)
        content = content.replace("model_name", model_name_clean.lower())
        content = content.replace("ModelName", model_name_clean.capitalize())
        content = content.replace("MODELNAME", model_name_clean.upper())
        
        # For handling special strings in templates
        content = content.replace("google/vit-base-patch16-224", model_id)
        content = content.replace("Generated: TIMESTAMP", f"Generated: {timestamp}")
        
        # Update class name if it follows standard pattern
        class_name_pattern = re.compile(r'class\s+Test(\w+)(?:\([\w.]+\))?:')
        if class_name_pattern.search(content):
            content = class_name_pattern.sub(f'class Test{model_name_clean.capitalize()}\\1:', content)

    # Special parameters for refactored templates
    if base_class:
        # Update base class if it's specified
        class_pattern = re.compile(r'class\s+(\w+)(?:\([\w.]+\))?:')
        if class_pattern.search(content):
            content = class_pattern.sub(f'class \\1({base_class}):', content)
    
    # Add special imports and handling code based on architecture
    if architecture == "vision" or architecture == "vision_text":
        # Ensure PIL import is present
        if "from PIL import Image" not in content:
            import_section = content.find("import numpy as np")
            if import_section != -1:
                content = content[:import_section] + "from PIL import Image\n" + content[import_section:]
        
        # Check if we need to add image generation code
        if "def test_basic_inference" in content and "create dummy image" not in content.lower():
            inference_method = content.find("def test_basic_inference")
            if inference_method != -1:
                try_pattern = re.compile(r'\s+# (Run|Prepare) inference')
                match = try_pattern.search(content, inference_method)
                if match:
                    indent_pos = match.start()
                    indentation = content[indent_pos:match.start() + 1]
                    
                    # Create image handling code
                    image_handling = f"""
{indentation}# Create dummy image for testing if needed
{indentation}if not os.path.exists("test.jpg"):
{indentation}    dummy_image = Image.new('RGB', (224, 224), color='white')
{indentation}    dummy_image.save("test.jpg")
{indentation}
"""
                    insert_pos = match.start()
                    content = content[:insert_pos] + image_handling + content[insert_pos:]
    
    elif architecture == "speech" or architecture == "audio":
        # Ensure numpy import is present
        if "import numpy as np" not in content:
            import_section = content.find("import os")
            if import_section != -1:
                content = content[:import_section + 10] + "\nimport numpy as np" + content[import_section + 10:]
        
        # Check if we need to add audio generation code
        if "def test_basic_inference" in content and "create dummy audio" not in content.lower():
            inference_method = content.find("def test_basic_inference")
            if inference_method != -1:
                try_pattern = re.compile(r'\s+# (Run|Prepare) inference')
                match = try_pattern.search(content, inference_method)
                if match:
                    indent_pos = match.start()
                    indentation = content[indent_pos:match.start() + 1]
                    
                    # Create audio handling code
                    audio_handling = f"""
{indentation}# Create dummy audio for testing if needed
{indentation}if not os.path.exists("test.wav"):
{indentation}    sample_rate = 16000
{indentation}    dummy_audio = np.random.randn(sample_rate * 2)  # 2 seconds of random noise
{indentation}    try:
{indentation}        import scipy.io.wavfile
{indentation}        scipy.io.wavfile.write("test.wav", sample_rate, dummy_audio.astype(np.float32))
{indentation}    except ImportError:
{indentation}        # Fallback to numpy save
{indentation}        with open("test.wav", 'wb') as f:
{indentation}            np.save(f, dummy_audio.astype(np.float32))
{indentation}
"""
                    insert_pos = match.start()
                    content = content[:insert_pos] + audio_handling + content[insert_pos:]
    
    # Create backup if file exists
    if os.path.exists(output_path):
        backup_path = f"{output_path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            shutil.copy2(output_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")
    
    # Create the parent directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write the output file
    try:
        with open(output_path, 'w') as f:
            f.write(content)
        logger.info(f"Successfully created test file: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error writing test file: {e}")
        return False

def create_test_file(model_name):
    """Create a test file for a specific model."""
    template_path = get_template_for_model(model_name)
    model_config = get_model_config(model_name)
    
    if not os.path.exists(template_path):
        logger.error(f"Template not found: {template_path}")
        return False
    
    # Read template
    template_content = read_template(template_path)
    if not template_content:
        logger.error(f"Failed to read template for {model_name}")
        return False
    
    # Customize template
    model_id = model_config.get("model_id", f"{model_name}-base-uncased")
    class_name = model_config.get("class_name", f"{model_name.capitalize()}Model")
    processor_class = model_config.get("processor_class", "AutoTokenizer")
    
    # Basic replacements
    content = template_content
    content = content.replace("MODEL_TYPE", model_name.upper())
    content = content.replace("model_type", model_name)
    content = content.replace("ModelClass", class_name)
    
    # Add custom imports - correctly placed and indented
    custom_imports = model_config.get("custom_imports", [])
    if custom_imports:
        import_section = content.find("# Third-party imports")
        if import_section != -1:
            # Find the indentation level of existing imports
            lines = content.split('\n')
            import_line_idx = -1
            for i, line in enumerate(lines):
                if "# Third-party imports" in line:
                    import_line_idx = i
                    break
            
            if import_line_idx != -1 and import_line_idx + 1 < len(lines):
                # Get indentation of the next line
                next_line = lines[import_line_idx + 1]
                indentation = ""
                for char in next_line:
                    if char in (' ', '\t'):
                        indentation += char
                    else:
                        break
                
                # Insert correctly indented imports
                indented_imports = []
                for import_line in custom_imports:
                    # Skip duplicates (prevent numpy import twice)
                    if any(imp in content for imp in [import_line, import_line.strip()]):
                        continue
                    indented_imports.append(f"{indentation}{import_line}")
                
                if indented_imports:
                    import_insert = '\n'.join(indented_imports)
                    # Insert after the existing imports
                    insert_position = content.find('\n', import_section) + 1
                    content = content[:insert_position] + import_insert + '\n' + content[insert_position:]
    
    # Add special handling code for dummy file creation
    if model_name in ["layoutlmv2", "layoutlmv3"]:
        # Add image creation code to processor_class pipeline method
        # Find the right place to insert
        pipeline_method = content.find("def test_pipeline")
        if pipeline_method != -1:
            try_start = content.find("try:", pipeline_method)
            if try_start != -1:
                # Find the indentation level in the try block
                next_line_pos = content.find('\n', try_start) + 1
                if next_line_pos < len(content):
                    # Get indentation of the next line
                    next_line = content[next_line_pos:content.find('\n', next_line_pos)]
                    indentation = ""
                    for char in next_line:
                        if char in (' ', '\t'):
                            indentation += char
                        else:
                            break
                    
                    # Create properly indented code
                    image_handling = f"""
{indentation}# Create dummy image for testing if needed
{indentation}if not os.path.exists("test.jpg"):
{indentation}    dummy_image = Image.new('RGB', (224, 224), color='white')
{indentation}    dummy_image.save("test.jpg")
{indentation}
"""
                    # Insert at the right position
                    insert_pos = content.find('\n', try_start) + 1
                    content = content[:insert_pos] + image_handling + content[insert_pos:]
    
    elif model_name in ["clvp", "seamless_m4t_v2"]:
        # Add audio generation code
        # Find the right place to insert
        pipeline_method = content.find("def test_pipeline")
        if pipeline_method != -1:
            try_start = content.find("try:", pipeline_method)
            if try_start != -1:
                # Find the indentation level in the try block
                next_line_pos = content.find('\n', try_start) + 1
                if next_line_pos < len(content):
                    # Get indentation of the next line
                    next_line = content[next_line_pos:content.find('\n', next_line_pos)]
                    indentation = ""
                    for char in next_line:
                        if char in (' ', '\t'):
                            indentation += char
                        else:
                            break
                    
                    # Create properly indented code
                    audio_handling = f"""
{indentation}# Create dummy audio for testing if needed
{indentation}if not os.path.exists("test.wav"):
{indentation}    sample_rate = 16000
{indentation}    dummy_audio = np.random.randn(sample_rate * 2)  # 2 seconds of random noise
{indentation}    try:
{indentation}        import scipy.io.wavfile
{indentation}        scipy.io.wavfile.write("test.wav", sample_rate, dummy_audio.astype(np.float32))
{indentation}    except ImportError:
{indentation}        # Fallback to numpy save
{indentation}        with open("test.wav", 'wb') as f:
{indentation}            np.save(f, dummy_audio.astype(np.float32))
{indentation}
"""
                    # Insert at the right position
                    insert_pos = content.find('\n', try_start) + 1
                    content = content[:insert_pos] + audio_handling + content[insert_pos:]
    
    # Update processor class
    content = content.replace('tokenizer = transformers.AutoTokenizer.from_pretrained', 
                             f'tokenizer = transformers.{processor_class}.from_pretrained')
    
    # Determine output path
    output_path = FIXED_TESTS_DIR / f"test_hf_{model_name}.py"
    
    # Create backup if file exists
    if os.path.exists(output_path):
        backup_path = f"{output_path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            shutil.copy2(output_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")
    
    # Write the output file
    try:
        with open(output_path, 'w') as f:
            f.write(content)
        logger.info(f"Successfully created test file: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error writing test file for {model_name}: {e}")
        return False

def verify_test_file(file_path):
    """Verify that a test file has valid Python syntax and structure.
    
    Args:
        file_path: Path to the test file
        
    Returns:
        Dict containing verification results
    """
    result = {
        "valid": False,
        "error": None,
        "line_number": None,
        "file_path": str(file_path)
    }
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Try to compile to check syntax
        compile(content, file_path, 'exec')
        
        # Check for common issues
        if "indentation" not in content.lower() and "def test_" in content:
            result["valid"] = True
        else:
            # Scan for potential issues
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if "indentation" in line.lower() and "error" in line.lower():
                    result["error"] = f"Indentation error reference found at line {i+1}"
                    result["line_number"] = i+1
                    break
        
        if result["valid"]:
            logger.info(f"Verification passed: {file_path}")
        else:
            if not result["error"]:
                result["error"] = "Unknown structural issue in file"
            logger.error(f"Verification failed: {result['error']}")
            
        return result
    except SyntaxError as e:
        logger.error(f"Syntax error in {file_path}: {e}")
        logger.error(f"  Line {e.lineno}: {e.text.strip()}")
        result["error"] = f"Syntax error: {str(e)}"
        result["line_number"] = e.lineno
        return result
    except Exception as e:
        logger.error(f"Error verifying file {file_path}: {e}")
        result["error"] = f"Error: {str(e)}"
        return result

def verify_syntax(file_path):
    """Verify that a file has valid Python syntax."""
    result = verify_test_file(file_path)
    return result["valid"]

def main():
    """Main entry point."""
    success_count = 0
    error_count = 0
    
    # First, process each model
    for model_name in MODELS_TO_FIX:
        logger.info(f"Processing model: {model_name}")
        
        # Create the test file
        success = create_test_file(model_name)
        
        if success:
            # Verify syntax
            output_path = FIXED_TESTS_DIR / f"test_hf_{model_name}.py"
            if verify_syntax(output_path):
                success_count += 1
            else:
                error_count += 1
        else:
            error_count += 1
    
    # Print summary
    logger.info(f"\nTemplate Generation Summary:")
    logger.info(f"Successfully generated: {success_count} models")
    logger.info(f"Failed to generate: {error_count} models")
    
    return 0 if error_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())