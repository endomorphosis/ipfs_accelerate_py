#!/usr/bin/env python3
"""
Model-specific template fixes for manually created HuggingFace tests.

This script:
1. Defines model-specific customizations needed when regenerating tests
2. Includes test inputs, class mappings, and model-specific logic
3. Provides architecture-aware template selection
4. Handles special cases for each model architecture

Usage:
    python model_template_fixes.py [--list-models] [--verify-model MODEL]
"""

import os
import sys
import json
import argparse
import logging
import importlib.util
import re
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"model_fixes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Define paths
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = SCRIPT_DIR.parent
SKILLS_DIR = REPO_ROOT / "skills"
TEMPLATES_DIR = SKILLS_DIR / "templates"
FINAL_MODELS_DIR = REPO_ROOT / "final_models"
FIXED_TESTS_DIR = SKILLS_DIR / "fixed_tests"

# Ensure output directories exist
os.makedirs(FIXED_TESTS_DIR, exist_ok=True)
os.makedirs(SCRIPT_DIR, exist_ok=True)

# Define architecture types and their templates
ARCHITECTURE_TYPES = {
    "encoder-only": {
        "template": "encoder_only_template.py",
        "registry_name": "ENCODER_ONLY_MODELS_REGISTRY",
        "models": ["bert", "roberta", "albert", "electra", "distilbert", "deberta"]
    },
    "decoder-only": {
        "template": "decoder_only_template.py",
        "registry_name": "DECODER_ONLY_MODELS_REGISTRY",
        "models": ["gpt2", "llama", "falcon", "gpt-j", "gpt-neo", "bloom", "opt"]
    },
    "encoder-decoder": {
        "template": "encoder_decoder_template.py",
        "registry_name": "ENCODER_DECODER_MODELS_REGISTRY",
        "models": ["t5", "bart", "pegasus", "bigbird", "xlm_prophetnet", "mbart"]
    },
    "vision": {
        "template": "vision_template.py",
        "registry_name": "VISION_MODELS_REGISTRY",
        "models": ["vit", "deit", "beit", "convnext", "swin", "detr"]
    },
    "vision-encoder-text-decoder": {
        "template": "vision_text_template.py",
        "registry_name": "VISION_TEXT_MODELS_REGISTRY",
        "models": ["clip", "blip", "layoutlmv2", "layoutlmv3", "pix2struct"]
    },
    "speech": {
        "template": "speech_template.py",
        "registry_name": "SPEECH_MODELS_REGISTRY",
        "models": ["whisper", "wav2vec2", "hubert", "clvp", "seamless_m4t_v2", "speecht5"]
    },
    "multimodal": {
        "template": "multimodal_template.py",
        "registry_name": "MULTIMODAL_MODELS_REGISTRY",
        "models": ["llava", "flava", "git", "flamingo", "imagebind", "blip"]
    }
}

# Define model-specific information for each manually created model
MODEL_CONFIG = {
    "layoutlmv2": {
        "architecture": "vision-encoder-text-decoder",
        "model_id": "microsoft/layoutlmv2-base-uncased",
        "class_name": "LayoutLMv2ForSequenceClassification",
        "task": "document-question-answering",
        "test_inputs": {
            "image": "test.jpg",
            "text": "What is the title of this document?"
        },
        "processor_class": "LayoutLMv2Processor",
        "source_file": os.path.join(FINAL_MODELS_DIR, "test_layoutlmv2.py"),
        "custom_imports": [
            "from PIL import Image",
            "import numpy as np"
        ],
        "special_handling": """# Create a dummy image for testing if needed
if not os.path.exists(test_image_path):
    dummy_image = Image.new('RGB', (224, 224), color='white')
    dummy_image.save(test_image_path)"""
    },
    "layoutlmv3": {
        "architecture": "vision-encoder-text-decoder",
        "model_id": "microsoft/layoutlmv3-base",
        "class_name": "LayoutLMv3ForSequenceClassification",
        "task": "document-question-answering",
        "test_inputs": {
            "image": "test.jpg",
            "text": "What is the content of this document?"
        },
        "processor_class": "LayoutLMv3Processor",
        "source_file": os.path.join(FINAL_MODELS_DIR, "test_layoutlmv3.py"),
        "custom_imports": [
            "from PIL import Image",
            "import numpy as np"
        ],
        "special_handling": """# Create a dummy image for testing if needed
if not os.path.exists(test_image_path):
    dummy_image = Image.new('RGB', (224, 224), color='white')
    dummy_image.save(test_image_path)"""
    },
    "clvp": {
        "architecture": "speech",
        "model_id": "susnato/clvp_dev",
        "class_name": "CLVPForCausalLM",
        "task": "text-to-speech",
        "test_inputs": {
            "text": "This is a test sentence for speech synthesis.",
            "audio": "test.wav"
        },
        "processor_class": "AutoProcessor",
        "source_file": os.path.join(FINAL_MODELS_DIR, "test_clvp.py"),
        "custom_imports": [
            "import numpy as np",
            "import librosa"
        ],
        "special_handling": """# Create a dummy audio file for testing if needed
if not os.path.exists(test_audio_path):
    sample_rate = 16000
    dummy_audio = np.random.randn(sample_rate * 2)  # 2 seconds of random noise
    # Save as WAV file using scipy
    try:
        import scipy.io.wavfile
        scipy.io.wavfile.write(test_audio_path, sample_rate, dummy_audio.astype(np.float32))
    except ImportError:
        # Alternative: save using numpy directly
        with open(test_audio_path, 'wb') as f:
            np.save(f, dummy_audio.astype(np.float32))"""
    },
    "bigbird": {
        "architecture": "encoder-decoder",
        "model_id": "google/bigbird-roberta-base",
        "class_name": "BigBirdForSequenceClassification",
        "task": "text-classification",
        "test_inputs": {
            "text": "This is a long document that requires a model like BigBird that can handle long sequences efficiently."
        },
        "processor_class": "AutoTokenizer",
        "source_file": os.path.join(FINAL_MODELS_DIR, "test_hf_bigbird.py"),
        "custom_imports": [
            "import numpy as np"
        ],
        "special_handling": """
        # BigBird can handle longer sequences, so create a long input for testing
        long_input = " ".join(["This is a test sentence."] * 10)
        """
    },
    "seamless_m4t_v2": {
        "architecture": "speech",
        "model_id": "facebook/seamless-m4t-v2-large",
        "class_name": "SeamlessM4TModel",
        "task": "speech-translation",
        "test_inputs": {
            "text": "Translate this to French: Hello, how are you?",
            "audio": "test.wav"
        },
        "processor_class": "AutoProcessor",
        "source_file": os.path.join(FINAL_MODELS_DIR, "test_seamless_m4t_v2.py"),
        "custom_imports": [
            "import numpy as np",
            "import librosa"
        ],
        "special_handling": """# Create a dummy audio file for testing if needed
if not os.path.exists(test_audio_path):
    sample_rate = 16000
    dummy_audio = np.random.randn(sample_rate * 2)  # 2 seconds of random noise
    # Save as WAV file using scipy
    try:
        import scipy.io.wavfile
        scipy.io.wavfile.write(test_audio_path, sample_rate, dummy_audio.astype(np.float32))
    except ImportError:
        # Alternative: save using numpy directly
        with open(test_audio_path, 'wb') as f:
            np.save(f, dummy_audio.astype(np.float32))"""
    },
    "xlm_prophetnet": {
        "architecture": "encoder-decoder",
        "model_id": "microsoft/xprophetnet-large-wiki100-cased",
        "class_name": "XLMProphetNetForConditionalGeneration",
        "task": "text2text-generation",
        "test_inputs": {
            "text": "Translate this to German: The quick brown fox jumps over the lazy dog."
        },
        "processor_class": "AutoTokenizer",
        "source_file": os.path.join(FINAL_MODELS_DIR, "test_xlm_prophetnet.py"),
        "custom_imports": [
            "import numpy as np"
        ],
        "special_handling": """
        # XLM ProphetNet is multilingual, so test with different languages
        inputs = {
            "en": "This is a test sentence in English.",
            "de": "Dies ist ein Testsatz auf Deutsch.",
            "fr": "C'est une phrase de test en français."
        }
        """
    }
}

def get_template_path(architecture):
    """Get the path to the template file for a given architecture."""
    if architecture not in ARCHITECTURE_TYPES:
        logger.warning(f"Unknown architecture: {architecture}, defaulting to encoder-only")
        architecture = "encoder-only"
    
    template_file = ARCHITECTURE_TYPES[architecture]["template"]
    return os.path.join(TEMPLATES_DIR, template_file)

def get_registry_name(architecture):
    """Get the model registry name for a given architecture."""
    if architecture not in ARCHITECTURE_TYPES:
        logger.warning(f"Unknown architecture: {architecture}, defaulting to encoder-only")
        architecture = "encoder-only"
    
    return ARCHITECTURE_TYPES[architecture]["registry_name"]

def read_template(template_path):
    """Read a template file and return its contents."""
    try:
        with open(template_path, 'r') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading template file {template_path}: {e}")
        return None

def customize_template(template_content, model_name, model_config):
    """Customize a template for a specific model.
    
    This function applies model-specific customizations to a template file, including:
    - Basic replacements (model name, class name)
    - Registry entry addition
    - Custom imports handling
    - Special handling code insertion (with proper indentation)
    - Test input updates
    - Processor class updates
    
    IMPORTANT: Special handling code requires proper indentation to ensure valid Python syntax.
    Particular attention is needed for code blocks inside conditional statements (if/else)
    and exception handling (try/except).
    """
    # Get model information
    model_id = model_config.get("model_id", f"{model_name}-base")
    class_name = model_config.get("class_name", f"{model_name.capitalize()}Model")
    task = model_config.get("task", "text-classification")
    processor_class = model_config.get("processor_class", "AutoTokenizer")
    
    # Split content into lines for easier manipulation
    lines = template_content.split('\n')
    
    # Perform basic replacements
    for i, line in enumerate(lines):
        lines[i] = line.replace("MODEL_TYPE", model_name.upper())
        lines[i] = lines[i].replace("model_type", model_name)
        lines[i] = lines[i].replace("ModelClass", class_name)
    
    # Handle registry entry
    registry_name = get_registry_name(model_config["architecture"])
    registry_entry = [
        f'    "{model_name}": {{',
        f'        "description": "{model_name.upper()} model",',
        f'        "class": "{class_name}",',
        f'        "default_model": "{model_id}",',
        f'        "architecture": "{model_config["architecture"]}",',
        f'        "task": "{task}"',
        '    },'
    ]
    
    # Find registry and add entry
    for i, line in enumerate(lines):
        if f"{registry_name} = {{" in line:
            # Insert right after this line
            lines[i+1:i+1] = registry_entry
            break
    
    # Handle custom imports
    custom_imports = model_config.get("custom_imports", [])
    if custom_imports:
        # Process imports to remove duplicates
        existing_imports = set()
        for line in lines:
            if line.strip().startswith("import "):
                module = line.strip().split()[1].split('.')[0]
                existing_imports.add(module)
            elif line.strip().startswith("from "):
                module = line.strip().split()[1].split('.')[0]
                existing_imports.add(module)
        
        # Filter out duplicated imports
        filtered_imports = []
        for imp in custom_imports:
            if "import " in imp:
                module = imp.split()[1].split('.')[0]
                if module not in existing_imports:
                    filtered_imports.append(imp)
                    existing_imports.add(module)
        
        # Find the third-party imports section
        imports_index = -1
        for i, line in enumerate(lines):
            if "# Third-party imports" in line:
                imports_index = i
                break
        
        if imports_index >= 0:
            # Get the indentation level used for imports
            import_indent = 0
            for j in range(imports_index + 1, min(imports_index + 10, len(lines))):
                if "import" in lines[j] and lines[j].strip():
                    import_indent = len(lines[j]) - len(lines[j].lstrip())
                    break
            
            # If we couldn't find an import, use a default indentation
            if import_indent == 0:
                import_indent = 0
            
            # Format imports with correct indentation
            formatted_imports = []
            for imp in filtered_imports:
                formatted_imports.append(" " * import_indent + imp)
            
            # Insert imports after the comment
            if formatted_imports:
                lines.insert(imports_index + 1, "\n".join(formatted_imports))
    
    # Handle special handling code - This is critical
    # The indentation of special handling code is extremely important for Python syntax
    # We need to carefully detect and maintain the correct indentation level for all
    # code blocks, especially those inside conditional statements and exception handling
    special_handling = model_config.get("special_handling", "")
    if special_handling:
        # Find the location to insert in test_pipeline method
        pipeline_index = -1
        try_index = -1
        
        for i, line in enumerate(lines):
            if "def test_pipeline" in line:
                pipeline_index = i
            elif pipeline_index > 0 and "try:" in line:
                try_index = i
                break
        
        if try_index > 0:
            # Find the indentation used in the 'try' block
            indentation = 0
            for j in range(try_index + 1, min(try_index + 10, len(lines))):
                if lines[j].strip():
                    indentation = len(lines[j]) - len(lines[j].lstrip())
                    break
            
            # If we still don't have an indentation, use a typical one (12 spaces)
            if indentation == 0:
                indentation = 12
            
            # Special handling for certain models that need exact indentation
            if "layoutlmv2" in model_name or "layoutlmv3" in model_name:
                # Format for image handling with proper indentation
                image_handling = [
                    f"{' ' * indentation}# Create a dummy image for testing if needed",
                    f"{' ' * indentation}if not os.path.exists(test_image_path):",
                    f"{' ' * indentation}    dummy_image = Image.new('RGB', (224, 224), color='white')",
                    f"{' ' * indentation}    dummy_image.save(test_image_path)",
                    f"{' ' * indentation}"  # Add a blank line
                ]
                
                # Insert after the try: line
                for line in reversed(image_handling):
                    lines.insert(try_index + 1, line)
                
            elif "clvp" in model_name or "seamless_m4t_v2" in model_name:
                # Format for audio handling with proper indentation
                audio_handling = [
                    f"{' ' * indentation}# Create a dummy audio file for testing if needed",
                    f"{' ' * indentation}if not os.path.exists(test_audio_path):",
                    f"{' ' * indentation}    sample_rate = 16000",
                    f"{' ' * indentation}    dummy_audio = np.random.randn(sample_rate * 2)  # 2 seconds of random noise",
                    f"{' ' * indentation}    # Save as WAV file using scipy",
                    f"{' ' * indentation}    try:",
                    f"{' ' * indentation}        import scipy.io.wavfile",
                    f"{' ' * indentation}        scipy.io.wavfile.write(test_audio_path, sample_rate, dummy_audio.astype(np.float32))",
                    f"{' ' * indentation}    except ImportError:",
                    f"{' ' * indentation}        # Alternative: save using numpy directly",
                    f"{' ' * indentation}        with open(test_audio_path, 'wb') as f:",
                    f"{' ' * indentation}            np.save(f, dummy_audio.astype(np.float32))",
                    f"{' ' * indentation}"  # Add a blank line
                ]
                
                # Insert after the try: line
                for line in reversed(audio_handling):
                    lines.insert(try_index + 1, line)
            else:
                # For other models, use the usual approach
                # Process the special handling code line by line
                special_handling_lines = []
                indentation_level = indentation
                
                # Process each line with appropriate indentation
                for line in special_handling.strip().split('\n'):
                    stripped_line = line.strip()
                    
                    # Skip empty lines
                    if not stripped_line:
                        special_handling_lines.append("")
                        continue
                    
                    # Add with proper indentation
                    special_handling_lines.append(f"{' ' * indentation_level}{stripped_line}")
                
                # Insert after the try: line
                lines.insert(try_index + 1, "")  # Add a blank line
                for line in reversed(special_handling_lines):
                    lines.insert(try_index + 1, line)
    
    # Update test inputs
    test_inputs = model_config.get("test_inputs", {})
    for i, line in enumerate(lines):
        if 'test_input = "The quick brown fox jumps over the lazy dog."' in line and "text" in test_inputs:
            lines[i] = line.replace('test_input = "The quick brown fox jumps over the lazy dog."', 
                                   f'test_input = "{test_inputs["text"]}"')
        
        if 'test_image_path = "test.jpg"' in line and "image" in test_inputs:
            lines[i] = line.replace('test_image_path = "test.jpg"', 
                                   f'test_image_path = "{test_inputs["image"]}"')
        
        if 'test_audio_path = "test.wav"' in line and "audio" in test_inputs:
            lines[i] = line.replace('test_audio_path = "test.wav"', 
                                   f'test_audio_path = "{test_inputs["audio"]}"')
    
    # Update processor class
    for i, line in enumerate(lines):
        if 'tokenizer = transformers.AutoTokenizer.from_pretrained' in line:
            lines[i] = line.replace('tokenizer = transformers.AutoTokenizer.from_pretrained', 
                                   f'tokenizer = transformers.{processor_class}.from_pretrained')
    
    # Reassemble the content
    content = '\n'.join(lines)
    return content

def generate_test_file(model_name, output_path=None):
    """Generate a test file for a specific model."""
    if model_name not in MODEL_CONFIG:
        logger.error(f"Model '{model_name}' not found in MODEL_CONFIG")
        return False, f"Model '{model_name}' not configured"
    
    model_config = MODEL_CONFIG[model_name]
    architecture = model_config["architecture"]
    
    # Get template path
    template_path = get_template_path(architecture)
    if not os.path.exists(template_path):
        logger.error(f"Template file not found: {template_path}")
        return False, f"Template file not found: {template_path}"
    
    # Read template
    template_content = read_template(template_path)
    if not template_content:
        return False, "Failed to read template file"
    
    # Customize template
    content = customize_template(template_content, model_name, model_config)
    
    # Determine output path
    if output_path is None:
        output_path = os.path.join(FIXED_TESTS_DIR, f"test_hf_{model_name}.py")
    
    # Create backup if needed
    if os.path.exists(output_path):
        backup_path = f"{output_path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            shutil.copy2(output_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")
    
    # Write output file
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(content)
        logger.info(f"Generated test file: {output_path}")
        return True, output_path
    except Exception as e:
        logger.error(f"Error writing output file: {e}")
        return False, f"Error writing output file: {e}"

def verify_test_file(file_path):
    """Verify that a test file has valid syntax."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Detailed analysis before compiling
        lines = content.split('\n')
        indentation_issues = []
        
        # Check for common indentation issues
        for i, line in enumerate(lines):
            if line.strip() and line.strip()[0] != '#':  # Skip empty lines and comments
                # Check if line starts with correct indentation
                if line[0] == ' ' and len(line) > 1 and line[1] != ' ':
                    indentation_issues.append(f"Line {i+1}: Indentation of 1 space")
                
                # Check for inconsistent indentation with previous line
                if i > 0 and lines[i-1].strip() and lines[i-1].strip()[-1] == ':':
                    prev_indent = len(lines[i-1]) - len(lines[i-1].lstrip())
                    curr_indent = len(line) - len(line.lstrip())
                    if curr_indent <= prev_indent:  # Should be indented
                        indentation_issues.append(f"Line {i+1}: Missing indentation after colon")
        
        if indentation_issues:
            logger.warning(f"Potential indentation issues in {file_path}:")
            for issue in indentation_issues:
                logger.warning(f"  - {issue}")
        
        # Compile to check syntax
        compile(content, file_path, 'exec')
        
        # Try to import as a module to check for import errors
        spec = importlib.util.spec_from_file_location("test_module", file_path)
        if spec is not None and spec.loader is not None:
            try:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except ImportError as ie:
                # Import errors are expected in some cases, just log them
                logger.warning(f"Import warning (not critical): {ie}")
        
        # If we reach here, syntax is valid
        logger.info(f"Syntax check passed for {file_path}")
        return True, "Syntax check passed"
    except SyntaxError as e:
        # Get context of the error for better debugging
        line_num = e.lineno
        start_line = max(0, line_num - 3)
        end_line = min(len(lines), line_num + 3) if 'lines' in locals() else line_num + 3
        
        context = "\n".join([f"{i+start_line+1}: {line}" for i, line in 
                            enumerate(lines[start_line:end_line])] if 'lines' in locals() else [])
        
        error_msg = f"Syntax error at line {e.lineno}: {e.msg}\nContext:\n{context}"
        logger.error(error_msg)
        return False, error_msg
    except Exception as e:
        logger.error(f"Error verifying file: {e}")
        return False, f"Error verifying file: {e}"

def update_architecture_types(model_name):
    """Update the ARCHITECTURE_TYPES dictionary in test_generator_fixed.py."""
    if model_name not in MODEL_CONFIG:
        logger.error(f"Model '{model_name}' not found in MODEL_CONFIG")
        return False
    
    # Get model architecture
    architecture = MODEL_CONFIG[model_name]["architecture"]
    
    # Get generator file path
    generator_path = os.path.join(SKILLS_DIR, "test_generator_fixed.py")
    if not os.path.exists(generator_path):
        logger.error(f"Generator file not found: {generator_path}")
        return False
    
    try:
        # Read the file
        with open(generator_path, 'r') as f:
            content = f.read()
        
        # Find the ARCHITECTURE_TYPES dictionary
        arch_types_start = content.find("ARCHITECTURE_TYPES = {")
        if arch_types_start == -1:
            logger.error("ARCHITECTURE_TYPES not found in generator file")
            return False
        
        # Find the specific architecture type section
        arch_type_quoted = f'"{architecture}"'
        arch_pattern = rf'{arch_type_quoted}:\s*\['
        match = re.search(arch_pattern, content)
        if not match:
            logger.error(f"Architecture type '{architecture}' not found in ARCHITECTURE_TYPES")
            return False
        
        # Get the start and end of the architecture list
        list_start_pos = content.find('[', match.start())
        list_end_pos = content.find(']', list_start_pos)
        if list_start_pos == -1 or list_end_pos == -1:
            logger.error(f"Could not find list bounds for architecture '{architecture}'")
            return False
        
        # Check if model is already in the list
        architecture_list = content[list_start_pos:list_end_pos]
        model_pattern = rf'"{model_name}"'
        if re.search(model_pattern, architecture_list):
            logger.info(f"Model '{model_name}' is already in the list for architecture '{architecture}'")
            return True
        
        # Add the model to the list
        comma = "," if architecture_list.strip() != "[" else ""
        new_content = content[:list_end_pos] + f'{comma} "{model_name}"' + content[list_end_pos:]
        
        # Write the updated content
        with open(generator_path, 'w') as f:
            f.write(new_content)
        
        logger.info(f"Updated ARCHITECTURE_TYPES with model '{model_name}' in architecture '{architecture}'")
        return True
    
    except Exception as e:
        logger.error(f"Error updating ARCHITECTURE_TYPES: {e}")
        return False

def regenerate_all_models(verify=True, apply=False):
    """Regenerate all manually created model tests."""
    results = {
        "success": [],
        "failure": []
    }
    
    for model_name in MODEL_CONFIG:
        logger.info(f"Regenerating test for model: {model_name}")
        
        # Generate test file
        success, result = generate_test_file(model_name)
        
        if success:
            if verify:
                # Verify syntax
                verify_success, verify_result = verify_test_file(result)
                if verify_success:
                    logger.info(f"Verification successful for {model_name}")
                    results["success"].append(model_name)
                else:
                    logger.error(f"Verification failed for {model_name}: {verify_result}")
                    results["failure"].append((model_name, verify_result))
            else:
                results["success"].append(model_name)
            
            # Update architecture types if requested
            if apply:
                update_success = update_architecture_types(model_name)
                if not update_success:
                    logger.warning(f"Failed to update architecture types for {model_name}")
        else:
            logger.error(f"Failed to generate test for {model_name}: {result}")
            results["failure"].append((model_name, result))
    
    # Print summary
    logger.info("\nRegeneration Summary:")
    logger.info(f"- Successfully regenerated: {len(results['success'])} models")
    if results["success"]:
        logger.info(f"  Models: {', '.join(results['success'])}")
    
    logger.info(f"- Failed to regenerate: {len(results['failure'])} models")
    if results["failure"]:
        for model, error in results["failure"]:
            logger.info(f"  - {model}: {error}")
    
    return results

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Model-specific template fixes")
    parser.add_argument("--list-models", action="store_true", help="List all configured models")
    parser.add_argument("--verify-model", type=str, help="Verify a specific model test file")
    parser.add_argument("--generate-model", type=str, help="Generate a test file for a specific model")
    parser.add_argument("--generate-all", action="store_true", help="Generate test files for all models")
    parser.add_argument("--generate-specific", action="store_true", help="Generate test files for specific problematic models")
    parser.add_argument("--verify", action="store_true", help="Verify generated test files")
    parser.add_argument("--apply", action="store_true", help="Apply changes to architecture types")
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Configured models:")
        for model_name, config in MODEL_CONFIG.items():
            print(f"- {model_name}: {config['architecture']} ({config['model_id']})")
        return 0
    
    if args.verify_model:
        if args.verify_model not in MODEL_CONFIG:
            logger.error(f"Model '{args.verify_model}' not found in MODEL_CONFIG")
            return 1
        
        file_path = os.path.join(FIXED_TESTS_DIR, f"test_hf_{args.verify_model}.py")
        if not os.path.exists(file_path):
            logger.error(f"Test file not found: {file_path}")
            return 1
        
        success, result = verify_test_file(file_path)
        if success:
            logger.info(f"Verification successful for {args.verify_model}")
            return 0
        else:
            logger.error(f"Verification failed for {args.verify_model}: {result}")
            return 1
    
    if args.generate_model:
        if args.generate_model not in MODEL_CONFIG:
            logger.error(f"Model '{args.generate_model}' not found in MODEL_CONFIG")
            return 1
        
        success, result = generate_test_file(args.generate_model)
        if success:
            logger.info(f"Generated test file for {args.generate_model}: {result}")
            
            if args.verify:
                verify_success, verify_result = verify_test_file(result)
                if verify_success:
                    logger.info(f"Verification successful for {args.generate_model}")
                else:
                    logger.error(f"Verification failed for {args.generate_model}: {verify_result}")
                    return 1
            
            if args.apply:
                update_success = update_architecture_types(args.generate_model)
                if not update_success:
                    logger.warning(f"Failed to update architecture types for {args.generate_model}")
            
            return 0
        else:
            logger.error(f"Failed to generate test file for {args.generate_model}: {result}")
            return 1
    
    if args.generate_specific:
        # Generate test files for specific problematic models
        specific_models = ["layoutlmv2", "layoutlmv3", "clvp", "seamless_m4t_v2", "bigbird", "xlm_prophetnet"]
        results = {
            "success": [],
            "failure": []
        }
        
        for model_name in specific_models:
            logger.info(f"Regenerating test for model: {model_name}")
            success, result = generate_test_file(model_name)
            
            if success:
                if args.verify:
                    verify_success, verify_result = verify_test_file(result)
                    if verify_success:
                        logger.info(f"Verification successful for {model_name}")
                        results["success"].append(model_name)
                    else:
                        logger.error(f"Verification failed for {model_name}: {verify_result}")
                        results["failure"].append((model_name, verify_result))
                else:
                    results["success"].append(model_name)
                
                if args.apply:
                    update_success = update_architecture_types(model_name)
                    if not update_success:
                        logger.warning(f"Failed to update architecture types for {model_name}")
            else:
                logger.error(f"Failed to generate test for {model_name}: {result}")
                results["failure"].append((model_name, result))
        
        # Print summary
        logger.info("\nRegeneration Summary:")
        logger.info(f"- Successfully regenerated: {len(results['success'])} models")
        if results["success"]:
            logger.info(f"  Models: {', '.join(results['success'])}")
        
        logger.info(f"- Failed to regenerate: {len(results['failure'])} models")
        if results["failure"]:
            for model, error in results["failure"]:
                logger.info(f"  - {model}: {error}")
        
        if results["failure"]:
            return 1
        return 0
    
    if args.generate_all:
        results = regenerate_all_models(verify=args.verify, apply=args.apply)
        if results["failure"]:
            return 1
        return 0
    
    # If no action specified, print help
    parser.print_help()
    return 1

if __name__ == "__main__":
    sys.exit(main())