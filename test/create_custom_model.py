#!/usr/bin/env python3
"""
Utility to create custom model configurations for testing.
This tool simplifies the process of adding new model types to the enhanced_generator.
"""

import os
import sys
import json
import time
import logging
import argparse
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure the enhanced_generator is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import from enhanced_generator
try:
    from enhanced_generator import (
        MODEL_REGISTRY, 
        ARCHITECTURE_TYPES,
        get_model_architecture,
        generate_test
    )
except ImportError as e:
    logger.error(f"Failed to import from enhanced_generator: {e}")
    sys.exit(1)

# Model task types by architecture
TASK_TYPES = {
    "encoder-only": ["fill-mask", "token-classification", "sequence-classification", "question-answering"],
    "decoder-only": ["text-generation", "causal-lm"],
    "encoder-decoder": ["text2text-generation", "translation", "summarization"],
    "vision": ["image-classification", "object-detection", "instance-segmentation", "image-segmentation"],
    "vision-text": ["zero-shot-image-classification", "image-to-text", "visual-question-answering"],
    "speech": ["automatic-speech-recognition", "audio-classification", "text-to-speech", "audio-to-audio"],
    "multimodal": ["image-to-text", "video-to-text", "multimodal-embedding", "visual-question-answering"]
}

# Example model classes by architecture
MODEL_CLASSES = {
    "encoder-only": ["BertForMaskedLM", "RobertaForMaskedLM", "AlbertForMaskedLM", "ElectraForMaskedLM"],
    "decoder-only": ["GPT2LMHeadModel", "LlamaForCausalLM", "MistralForCausalLM", "FalconForCausalLM"],
    "encoder-decoder": ["T5ForConditionalGeneration", "BartForConditionalGeneration", "PegasusForConditionalGeneration"],
    "vision": ["ViTForImageClassification", "SwinForImageClassification", "DeiTForImageClassification"],
    "vision-text": ["CLIPModel", "BlipForConditionalGeneration", "LlavaForConditionalGeneration"],
    "speech": ["WhisperForConditionalGeneration", "Wav2Vec2ForCTC", "SpeechT5ForTextToSpeech"],
    "multimodal": ["LlavaForConditionalGeneration", "BlipForConditionalGeneration", "PaliGemmaForConditionalGeneration"]
}

# Example test inputs by task
TEST_INPUTS = {
    "fill-mask": "The quick brown fox jumps over the [MASK] dog.",
    "token-classification": "The quick brown fox jumps over the lazy dog.",
    "sequence-classification": "I really enjoyed this movie!",
    "question-answering": {"question": "What is the capital of France?", "context": "Paris is the capital of France."},
    "text-generation": "Once upon a time",
    "causal-lm": "def fibonacci(n):",
    "text2text-generation": "translate English to German: Hello, how are you?",
    "translation": "Hello, how are you?",
    "summarization": "The tower is 324 metres tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres on each side.",
    "image-classification": "test.jpg",
    "object-detection": "test.jpg",
    "instance-segmentation": "test.jpg",
    "image-segmentation": "test.jpg",
    "zero-shot-image-classification": ["test.jpg", ["a photo of a cat", "a photo of a dog", "a photo of a person"]],
    "image-to-text": ["test.jpg", "What is in this image?"],
    "visual-question-answering": ["test.jpg", "What is in this image?"],
    "automatic-speech-recognition": "test.mp3",
    "audio-classification": "test.wav",
    "text-to-speech": "Hello, this is a test of the Speech T5 text to speech model.",
    "audio-to-audio": "test.wav",
    "multimodal-embedding": ["test.jpg", "test.wav", "A sample text"],
    "video-to-text": ["test.mp4", "What is happening in this video?"]
}

def create_model_config(interactive: bool = True) -> Dict[str, Any]:
    """
    Create a model configuration interactively or with command line arguments.
    
    Args:
        interactive: Whether to use interactive mode or command line arguments
        
    Returns:
        Dictionary with model configuration
    """
    config = {}
    
    if interactive:
        # Get model name
        model_name = input("Enter model name (e.g., bert-base-uncased): ").strip()
        if not model_name:
            logger.error("Model name cannot be empty")
            sys.exit(1)
        
        # Convert to normalized model_id
        model_id = model_name.replace('-', '_').lower()
        
        # Determine architecture
        print("\nSelect architecture type:")
        for i, arch in enumerate(ARCHITECTURE_TYPES.keys()):
            print(f"{i+1}. {arch}")
        
        arch_index = int(input("Enter number (1-7): ").strip()) - 1
        architecture = list(ARCHITECTURE_TYPES.keys())[arch_index]
        
        # Select task
        print(f"\nSelect task type for {architecture}:")
        for i, task in enumerate(TASK_TYPES[architecture]):
            print(f"{i+1}. {task}")
        
        task_index = int(input("Enter number: ").strip()) - 1
        task = TASK_TYPES[architecture][task_index]
        
        # Select model class
        print(f"\nSelect model class for {architecture}:")
        for i, cls in enumerate(MODEL_CLASSES[architecture]):
            print(f"{i+1}. {cls}")
        
        cls_index = int(input("Enter number (or enter custom class): ").strip())
        if cls_index > 0 and cls_index <= len(MODEL_CLASSES[architecture]):
            model_class = MODEL_CLASSES[architecture][cls_index - 1]
        else:
            model_class = input("Enter custom model class: ").strip()
        
        # Get default model ID
        default_model = input(f"\nEnter default model ID (default: {model_name}): ").strip()
        if not default_model:
            default_model = model_name
        
        # Get test input
        example_input = TEST_INPUTS.get(task, "")
        print(f"\nDefault test input for {task}: {example_input}")
        use_default = input("Use default test input? (y/n): ").strip().lower() == 'y'
        
        if use_default:
            test_input = example_input
        else:
            if isinstance(example_input, list):
                test_input = []
                for i, item in enumerate(example_input):
                    new_item = input(f"Enter test input part {i+1}: ").strip()
                    if isinstance(item, list):
                        # Handle the case where an item is a list
                        parts = [p.strip() for p in new_item.split(',')]
                        test_input.append(parts)
                    else:
                        test_input.append(new_item)
            elif isinstance(example_input, dict):
                test_input = {}
                for key in example_input:
                    test_input[key] = input(f"Enter test input for {key}: ").strip()
            else:
                test_input = input("Enter test input: ").strip()
    else:
        # Use command line arguments
        args = parse_args()
        
        model_name = args.model_name
        model_id = model_name.replace('-', '_').lower()
        architecture = args.architecture
        task = args.task
        model_class = args.model_class
        default_model = args.default_model if args.default_model else model_name
        
        # Handle test input
        if args.test_input:
            test_input = args.test_input
        else:
            test_input = TEST_INPUTS.get(task, "")
    
    # Create final config
    config = {
        "model_id": model_id,
        "model_name": model_name,
        "architecture": architecture,
        "registry_entry": {
            "default_model": default_model,
            "task": task,
            "class": model_class,
            "test_input": test_input
        }
    }
    
    return config

def generate_model_code(config: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Generate code for adding to MODEL_REGISTRY and ARCHITECTURE_TYPES.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Tuple of (model_registry_code, architecture_types_code, python_import_code)
    """
    model_id = config["model_id"]
    registry_entry = config["registry_entry"]
    architecture = config["architecture"]
    
    # Generate MODEL_REGISTRY entry
    model_registry_code = f"""    "{model_id}": {{
        "default_model": "{registry_entry['default_model']}",
        "task": "{registry_entry['task']}",
        "class": "{registry_entry['class']}",
        "test_input": {repr(registry_entry['test_input'])}
    }},"""
    
    # Generate ARCHITECTURE_TYPES entry
    architecture_types_code = f"""# Add to ARCHITECTURE_TYPES["{architecture}"] list:
"{model_id}","""
    
    # Generate import code for trying the model immediately
    python_import_code = f"""
from enhanced_generator import generate_test, validate_generated_file

# Test the model generation
result = generate_test("{model_id}", "custom_model_tests")
print(f"Generated file: {{result['file_path']}}")

# Validate the file
is_valid, validation_msg = validate_generated_file(result["file_path"])
print(f"Validation result: {{is_valid}}")
if not is_valid:
    print(f"Validation message: {{validation_msg}}")
"""
    
    return model_registry_code, architecture_types_code, python_import_code

def write_model_files(config: Dict[str, Any], output_dir: str) -> Dict[str, str]:
    """
    Write model configuration to files.
    
    Args:
        config: Model configuration dictionary
        output_dir: Directory to output files
        
    Returns:
        Dictionary with file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model_id = config["model_id"]
    model_name = config["model_name"]
    
    # Generate code
    model_registry_code, architecture_types_code, python_import_code = generate_model_code(config)
    
    # Write model_registry.py
    registry_file = os.path.join(output_dir, f"{model_id}_registry.py")
    with open(registry_file, 'w') as f:
        f.write(f"""
# MODEL_REGISTRY entry for {model_name}
{model_registry_code}
        """)
    
    # Write architecture_types.py
    architecture_file = os.path.join(output_dir, f"{model_id}_architecture.py")
    with open(architecture_file, 'w') as f:
        f.write(f"""
# ARCHITECTURE_TYPES entry for {model_name}
{architecture_types_code}
        """)
    
    # Write import_test.py
    import_file = os.path.join(output_dir, f"{model_id}_test.py")
    with open(import_file, 'w') as f:
        f.write(f"""#!/usr/bin/env python3
# Test script for {model_name}
{python_import_code}
        """)
    
    # Make the test script executable
    os.chmod(import_file, 0o755)
    
    # Write config.json
    config_file = os.path.join(output_dir, f"{model_id}_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Generate a single file to add to additional_models.py
    additional_file = os.path.join(output_dir, f"{model_id}_additional_models.py")
    with open(additional_file, 'w') as f:
        f.write(f"""
# Add to additional_models.py for {model_name}

# Add to ADDITIONAL_MODELS dictionary
{model_registry_code}

# Add to ADDITIONAL_ARCHITECTURE_MAPPINGS dictionary
# Under "{config['architecture']}" key, add:
"{model_id}",
        """)
    
    return {
        "registry_file": registry_file,
        "architecture_file": architecture_file,
        "import_file": import_file,
        "config_file": config_file,
        "additional_file": additional_file
    }

def try_generate_test(config: Dict[str, Any], output_dir: str) -> Tuple[bool, str, Optional[str]]:
    """
    Try to generate a test file with the model configuration.
    
    Args:
        config: Model configuration dictionary
        output_dir: Directory to output test file
        
    Returns:
        Tuple of (success, message, file_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model_id = config["model_id"]
    model_name = config["model_name"]
    
    # Add model to temporary MODEL_REGISTRY
    temp_registry = MODEL_REGISTRY.copy()
    temp_registry[model_id] = config["registry_entry"]
    
    # Write a temporary script to generate the test
    temp_script = os.path.join(output_dir, "temp_generator.py")
    with open(temp_script, 'w') as f:
        f.write(f"""
import sys
import os
from enhanced_generator import generate_test as original_generate_test
from enhanced_generator import validate_generated_file

# Add custom entry
MODEL_REGISTRY = {repr(temp_registry)}

def modified_generate_test(model_type, output_dir, **kwargs):
    return original_generate_test(model_type, output_dir, **kwargs)

if __name__ == "__main__":
    try:
        # Generate test with custom registry
        result = modified_generate_test("{model_id}", "{output_dir}")
        print(f"SUCCESS:{{result['file_path']}}")
        
        # Validate the file
        is_valid, validation_msg = validate_generated_file(result["file_path"])
        print(f"VALID:{{is_valid}}")
        if not is_valid:
            print(f"MESSAGE:{{validation_msg}}")
        sys.exit(0)
    except Exception as e:
        print(f"ERROR:{{type(e).__name__}}:{{str(e)}}")
        sys.exit(1)
""")
    
    # Run the temporary script
    from subprocess import run, PIPE
    proc = run([sys.executable, temp_script], stdout=PIPE, stderr=PIPE, text=True)
    
    # Clean up temporary script
    os.remove(temp_script)
    
    # Parse output
    success = False
    message = ""
    file_path = None
    
    for line in proc.stdout.splitlines():
        if line.startswith("SUCCESS:"):
            file_path = line[len("SUCCESS:"):]
            success = True
        elif line.startswith("VALID:"):
            is_valid = line[len("VALID:"):].lower() == "true"
            if not is_valid:
                success = False
        elif line.startswith("MESSAGE:"):
            message = line[len("MESSAGE:"):]
        elif line.startswith("ERROR:"):
            success = False
            message = line[len("ERROR:"):]
    
    if proc.returncode != 0:
        success = False
        message = proc.stderr
    
    return success, message, file_path

def generate_model_report(config: Dict[str, Any], files: Dict[str, str], test_result: Tuple[bool, str, Optional[str]], output_file: str) -> None:
    """
    Generate a report for the model configuration and test result.
    
    Args:
        config: Model configuration dictionary
        files: Dictionary with file paths
        test_result: Tuple of (success, message, file_path)
        output_file: Output file path for the report
    """
    success, message, test_file = test_result
    model_id = config["model_id"]
    model_name = config["model_name"]
    
    with open(output_file, 'w') as f:
        f.write(f"# Model Configuration Report: {model_name}\n\n")
        f.write(f"**Generated on:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## Configuration\n\n")
        f.write(f"- **Model ID:** {model_id}\n")
        f.write(f"- **Model Name:** {model_name}\n")
        f.write(f"- **Architecture:** {config['architecture']}\n")
        f.write(f"- **Task:** {config['registry_entry']['task']}\n")
        f.write(f"- **Model Class:** {config['registry_entry']['class']}\n")
        f.write(f"- **Default Model:** {config['registry_entry']['default_model']}\n")
        f.write(f"- **Test Input:** {repr(config['registry_entry']['test_input'])}\n\n")
        
        f.write(f"## Generated Files\n\n")
        for file_type, file_path in files.items():
            f.write(f"- **{file_type}:** {file_path}\n")
        f.write("\n")
        
        f.write(f"## Test Generation Result\n\n")
        if success:
            f.write(f"✅ **Success!** Test file generated successfully.\n\n")
            f.write(f"- **Test File:** {test_file}\n")
        else:
            f.write(f"❌ **Failed!** Test file generation failed.\n\n")
            f.write(f"- **Error:** {message}\n")
        
        f.write("\n## Installation Instructions\n\n")
        f.write("To add this model to the enhanced generator, follow these steps:\n\n")
        f.write("### Option 1: Add to additional_models.py\n\n")
        f.write("1. Open `additional_models.py`\n")
        f.write("2. Add the following to the `ADDITIONAL_MODELS` dictionary:\n\n")
        f.write("```python\n")
        model_registry_code, _, _ = generate_model_code(config)
        f.write(model_registry_code + "\n")
        f.write("```\n\n")
        f.write("3. Add the model to the appropriate architecture in `ADDITIONAL_ARCHITECTURE_MAPPINGS`:\n\n")
        f.write("```python\n")
        f.write(f'"{config["architecture"]}": [\n')
        f.write(f'    # ... existing models ...\n')
        f.write(f'    "{model_id}",\n')
        f.write(f'],\n')
        f.write("```\n\n")
        
        f.write("### Option 2: Add to enhanced_generator.py\n\n")
        f.write("1. Open `enhanced_generator.py`\n")
        f.write("2. Add the following to the `MODEL_REGISTRY` dictionary:\n\n")
        f.write("```python\n")
        f.write(model_registry_code + "\n")
        f.write("```\n\n")
        f.write("3. Add the model to the appropriate architecture in `ARCHITECTURE_TYPES`:\n\n")
        f.write("```python\n")
        f.write(f'"{config["architecture"]}": [..., "{model_id}"],\n')
        f.write("```\n\n")
        
        f.write("### Test the model\n\n")
        f.write("After adding the model, test it with:\n\n")
        f.write("```python\n")
        f.write(f"from enhanced_generator import generate_test\n")
        f.write(f"result = generate_test('{model_id}', 'test_output')\n")
        f.write(f"print(f'Generated file: {{result[\"file_path\"]}}')\n")
        f.write("```\n")
    
    logger.info(f"Model report written to {output_file}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create a custom model configuration")
    parser.add_argument("--model-name", help="Model name (e.g., bert-base-uncased)")
    parser.add_argument("--architecture", choices=ARCHITECTURE_TYPES.keys(), help="Architecture type")
    parser.add_argument("--task", help="Task type")
    parser.add_argument("--model-class", help="Model class")
    parser.add_argument("--default-model", help="Default model ID")
    parser.add_argument("--test-input", help="Test input")
    parser.add_argument("--output-dir", default="custom_model_configs", help="Output directory")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Determine mode
    interactive = args.interactive or not (args.model_name and args.architecture and args.task and args.model_class)
    
    # Create configuration
    config = create_model_config(interactive)
    model_id = config["model_id"]
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Write files
    logger.info(f"Writing configuration files for {model_id}")
    files = write_model_files(config, output_dir)
    
    # Try to generate a test
    logger.info(f"Attempting to generate test for {model_id}")
    test_result = try_generate_test(config, os.path.join(output_dir, "tests"))
    
    if test_result[0]:
        logger.info(f"✅ Successfully generated test file: {test_result[2]}")
    else:
        logger.warning(f"❌ Failed to generate test file: {test_result[1]}")
    
    # Generate report
    report_file = os.path.join(output_dir, f"{model_id}_report.md")
    generate_model_report(config, files, test_result, report_file)
    
    logger.info(f"Model configuration complete!")
    logger.info(f"See the report at: {report_file}")
    
    return 0 if test_result[0] else 1

if __name__ == "__main__":
    sys.exit(main())