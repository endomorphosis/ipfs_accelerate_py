#!/usr/bin/env python3
"""
Generate Missing Hugging Face Tests

This script identifies and generates test files for Hugging Face model types
that are listed in huggingface_model_types.json but don't have corresponding
test files in the skills directory.

Usage:
  python generate_missing_hf_tests.py --list-missing    # List missing test files
  python generate_missing_hf_tests.py --generate-all    # Generate all missing test files
  python generate_missing_hf_tests.py --generate MODEL  # Generate test for specific model
  python generate_missing_hf_tests.py --report          # Generate coverage report
  python generate_missing_hf_tests.py --batch N         # Generate N test files in batch
"""

import os
import sys
import json
import argparse
import glob
import re
import datetime
from pathlib import Path

# Constants
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = CURRENT_DIR.parent
MODELS_JSON_PATH = PARENT_DIR / "huggingface_model_types.json"
COVERAGE_REPORT_PATH = CURRENT_DIR / "HF_COVERAGE_REPORT.md"
TEST_REPORT_PATH = CURRENT_DIR / "test_report.md"

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import test_generator - we'll directly use the file without importing
HAS_TEST_GENERATOR = False  # We'll work without direct import
test_generator = None

# Check if the test_generator file exists
test_generator_path = CURRENT_DIR / "test_generator.py"
if os.path.exists(test_generator_path):
    print(f"Found test_generator.py at {test_generator_path}")
    HAS_TEST_GENERATOR = True
else:
    print("Warning: Could not find test_generator.py module")

def load_model_types():
    """Load the Hugging Face model types from JSON."""
    with open(MODELS_JSON_PATH, 'r') as f:
        return json.load(f)

def find_existing_test_files():
    """Find all existing test_hf_*.py files."""
    test_files = glob.glob(str(CURRENT_DIR / "test_hf_*.py"))
    
    # Extract model names from test files
    model_names = []
    for file_path in test_files:
        file_name = os.path.basename(file_path)
        if file_name.startswith("test_hf_") and file_name.endswith(".py"):
            # Skip helper files
            if file_name in ["test_hf_\\py", "test_hf___help.py", "test_hf___list_only.py", "test_hf___model.py"]:
                continue
            
            # Extract model name (remove 'test_hf_' prefix and '.py' suffix)
            model_name = file_name[8:-3]
            model_names.append(model_name)
    
    return model_names

def find_missing_models(all_models, existing_tests):
    """Find models that don't have corresponding test files."""
    # Standardize model names for comparison
    standardized_tests = [test.replace('_', '-') for test in existing_tests]
    missing_models = []
    
    for model in all_models:
        # Check if model has a test file
        model_test_name = model.replace('-', '_')
        if model_test_name not in existing_tests and model.replace('-', '_') not in existing_tests:
            # Double check with standardized names
            if model.replace('-', '_') not in standardized_tests and model not in standardized_tests:
                missing_models.append(model)
    
    return missing_models

def create_model_registry_entry(model_type):
    """Create a model registry entry for a given model type."""
    # Default values
    family_name = model_type.upper()
    model_id = model_type.replace('_', '-')
    class_name = f"{model_type.title().replace('_', '')}Model"
    test_class_name = f"Test{model_type.title().replace('_', '')}Models"
    module_name = f"test_hf_{model_type.lower().replace('-', '_')}"
    default_task = "feature-extraction"
    
    # Special case handling for common tasks
    if model_type.endswith('lm'):
        default_task = "fill-mask"
    elif model_type.endswith('for-causal-lm') or 'gpt' in model_type.lower():
        default_task = "text-generation"
    elif model_type.endswith('seq2seq'):
        default_task = "text2text-generation"
    elif 'vision' in model_type.lower() or 'image' in model_type.lower():
        default_task = "image-classification"
    elif 'audio' in model_type.lower() or 'speech' in model_type.lower():
        default_task = "automatic-speech-recognition"
    
    # Create registry entry
    registry_entry = {
        "family_name": family_name,
        "description": f"{family_name} models",
        "default_model": f"{model_id}-base",
        "class": class_name,
        "test_class": test_class_name,
        "module_name": module_name,
        "tasks": [default_task],
        "inputs": {
            "text": "This is a test input for the model." if default_task != "image-classification" else None,
            "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg" if default_task == "image-classification" else None
        },
        "dependencies": ["transformers", "tokenizers"],
        "task_specific_args": {
            default_task: {}
        },
        "models": {
            f"{model_id}-base": {
                "description": f"{family_name} base model",
                "class": class_name
            }
        }
    }
    
    # Clean up None values
    if registry_entry["inputs"]["text"] is None:
        del registry_entry["inputs"]["text"]
    if registry_entry["inputs"]["image_url"] is None:
        del registry_entry["inputs"]["image_url"]
    
    return registry_entry

def generate_test_file(model_type):
    """Generate a test file for a model type using a template approach."""
    try:
        # Create the test file path
        model_key = model_type.lower().replace('-', '_')
        file_path = CURRENT_DIR / f"test_hf_{model_key}.py"
        
        # Check if template file exists to use as reference
        template_file = CURRENT_DIR / "test_hf_audio_spectrogram_transformer.py"
        if not os.path.exists(template_file):
            template_file = None
            for test_file in glob.glob(str(CURRENT_DIR / "test_hf_*.py")):
                if not test_file.endswith(("__help.py", "__list_only.py", "__model.py")):
                    template_file = test_file
                    break
                    
        if not template_file:
            print(f"Error: No template file found to use as reference")
            return False
            
        print(f"Using template file: {template_file}")
        
        # Read template file
        with open(template_file, 'r') as f:
            template_content = f.read()
            
        # Get model info
        model_info = create_model_registry_entry(model_type)
        
        # Replace key elements
        class_name = model_info["class"]
        family_name = model_info["family_name"]
        description = model_info["description"]
        default_model = model_info["default_model"]
        task = model_info["tasks"][0]
        
        # Convert template content
        new_content = template_content
        
        # Replace class names and variables
        new_content = re.sub(r'class Test\w+Models', f'class Test{family_name.title().replace("-", "")}Models', new_content)
        new_content = re.sub(r'audio_spectrogram_transformer', model_key, new_content, flags=re.IGNORECASE)
        new_content = re.sub(r'Audio-Spectrogram-Transformer', family_name, new_content)
        new_content = re.sub(r'AudioSpectrogramTransformer', f'{family_name.title().replace("-", "")}', new_content)
        
        # Fix indentation issues - ensure all methods are properly indented
        # Identify class definition line
        class_match = re.search(r'class Test.*Models:', new_content)
        if class_match:
            # Find all method definitions not properly indented
            pattern = r'(?<=\n)def (test_\w+|run_tests|test_with_\w+)\(self'
            # Replace with properly indented methods
            new_content = re.sub(pattern, r'    def \1(self', new_content)
        
        # Update docstring
        new_content = re.sub(r'This file provides a unified testing interface for:[^\n]*\n- .*',
                      f'This file provides a unified testing interface for:\n- {class_name}', new_content)
        
        # Create a registry definition
        registry_str = f"{model_key.lower()}_MODELS_REGISTRY"
        model_registry = f"""
# Models registry - Maps model IDs to their specific configurations
{registry_str.upper()} = {{
    "{default_model}": {{
        "description": "{description}",
        "class": "{class_name}"
    }}
}}"""
                
        # Replace registry and all references to it
        new_content = re.sub(r'# Models registry - Maps model IDs to their specific configurations[\s\S]*?MODELS_REGISTRY = {[\s\S]*?}[\s\S]*?}[\s\S]*?[}]',
                           model_registry, new_content, flags=re.DOTALL)
                           
        # Fix any potential syntax issues with extra braces
        new_content = re.sub(r'}\n}', '}\n', new_content)
                           
        # Fix the model registry init and default model
        new_content = re.sub(rf'self.model_id = model_id or "[^"]+"', 
                           f'self.model_id = model_id or "{default_model}"', new_content)
        
        # Update model_info initialization with correct registry and default model
        # This regex will match both the initialization in init and the fallback in the warning condition
        model_info_pattern = r'self\.model_info = [a-zA-Z0-9_]+_MODELS_REGISTRY\["[^"]+"\]'
        model_info_replacement = f'self.model_info = {registry_str}["{default_model}"]'
        new_content = re.sub(model_info_pattern, model_info_replacement, new_content)
         
        # Replace all other registry references - handle lowercase versions too
        new_content = re.sub(r'AUDIO_SPECTROGRAM_TRANSFORMER_MODELS_REGISTRY', f'{registry_str.upper()}', new_content)
        new_content = re.sub(r'audio_spectrogram_transformer_MODELS_REGISTRY', f'{registry_str}', new_content)
        
        # Fix case consistency issues - ensure the same case is used everywhere
        if registry_str.lower() != registry_str:
            new_content = re.sub(rf'{registry_str.lower()}_MODELS_REGISTRY', f'{registry_str}', new_content, flags=re.IGNORECASE)
            
        # Fix case consistency with model access
        new_content = re.sub(rf'\b{model_key.lower()}_MODELS_REGISTRY\b', f'{registry_str}', new_content, flags=re.IGNORECASE)
        
        # Update task and input examples based on the task
        if task == "fill-mask":
            input_str = 'self.test_text = "The quick brown fox jumps over the [MASK] dog."'
        elif task == "text-generation":
            input_str = 'self.test_text = "Once upon a time in a land far away"'
        elif task == "image-classification":
            input_str = 'self.test_image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"'
        elif task == "automatic-speech-recognition":
            input_str = 'self.test_audio = "audio_sample.mp3"'
        else:
            input_str = 'self.test_text = "This is a test input for the model."'
            
        # Replace input handling
        new_content = re.sub(r'# Define test inputs\s+[^\n]+', f'# Define test inputs\n        {input_str}', new_content)
            
        # Write to file
        with open(file_path, "w") as f:
            f.write(new_content)
        
        print(f"Created test file: {file_path}")
        return True
    
    except Exception as e:
        print(f"Error generating test file for {model_type}: {e}")
        return False

def generate_coverage_report(all_models, existing_tests, missing_models):
    """Generate a coverage report."""
    total_models = len(all_models)
    covered_models = len(existing_tests)
    missing_count = len(missing_models)
    coverage_percent = (covered_models / total_models) * 100
    
    report = f"""# Hugging Face Model Test Coverage Report

*Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*

## Coverage Summary

- **Total Model Types**: {total_models}
- **Implemented Tests**: {covered_models}
- **Missing Tests**: {missing_count}
- **Coverage**: {coverage_percent:.1f}%

## Missing Model Types

The following {missing_count} model types need test implementations:

"""
    # Sort missing models alphabetically
    missing_models.sort()
    
    # Group by prefix for better organization
    prefix_groups = {}
    for model in missing_models:
        prefix = model.split('-')[0] if '-' in model else model
        if prefix not in prefix_groups:
            prefix_groups[prefix] = []
        prefix_groups[prefix].append(model)
    
    # Add missing models to report
    for prefix, models in sorted(prefix_groups.items()):
        report += f"### {prefix.upper()} Family\n"
        for model in sorted(models):
            report += f"- `{model}`\n"
        report += "\n"
    
    report += """## Next Steps

1. **Generate Test Files**: Use this script to generate missing test files
2. **Implement Core Logic**: Complete the test implementation with appropriate model-specific logic
3. **Verify Hardware Compatibility**: Test across CPU, CUDA, and OpenVINO
4. **Document Results**: Update test_report.md with new results

## Commands

```bash
# Generate all missing test files
python generate_missing_hf_tests.py --generate-all

# Generate specific test
python generate_missing_hf_tests.py --generate MODEL_TYPE

# Generate tests in batches
python generate_missing_hf_tests.py --batch 10
```
"""
    
    # Write report to file
    with open(COVERAGE_REPORT_PATH, "w") as f:
        f.write(report)
    
    print(f"Coverage report generated: {COVERAGE_REPORT_PATH}")
    return report

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Generate missing Hugging Face test files")
    
    # Options
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--list-missing", action="store_true", help="List missing test files")
    action_group.add_argument("--generate-all", action="store_true", help="Generate all missing test files")
    action_group.add_argument("--generate", type=str, help="Generate test for specific model")
    action_group.add_argument("--batch", type=int, help="Generate a batch of N test files")
    action_group.add_argument("--report", action="store_true", help="Generate coverage report")
    
    args = parser.parse_args()
    
    # Load all models from JSON
    if not os.path.exists(MODELS_JSON_PATH):
        print(f"Error: Model types JSON file not found at {MODELS_JSON_PATH}")
        return 1
    
    all_models = load_model_types()
    print(f"Loaded {len(all_models)} model types from {MODELS_JSON_PATH}")
    
    # Find existing test files
    existing_tests = find_existing_test_files()
    print(f"Found {len(existing_tests)} existing test files")
    
    # Find missing models
    missing_models = find_missing_models(all_models, existing_tests)
    print(f"Found {len(missing_models)} missing test files")
    
    # List missing models
    if args.list_missing:
        print("\nMissing test files:")
        for model in sorted(missing_models):
            print(f"  - {model}")
    
    # Generate test file for specific model
    elif args.generate:
        model_type = args.generate
        if model_type not in all_models:
            print(f"Error: Model type '{model_type}' not found in {MODELS_JSON_PATH}")
            return 1
        
        if model_type.replace('-', '_') in existing_tests:
            print(f"Test file for '{model_type}' already exists")
            return 0
        
        print(f"Generating test file for {model_type}...")
        generate_test_file(model_type)
    
    # Generate all missing test files
    elif args.generate_all:
        print(f"Generating {len(missing_models)} missing test files...")
        successful = 0
        failed = 0
        
        for model_type in missing_models:
            print(f"Generating test file for {model_type}...")
            if generate_test_file(model_type):
                successful += 1
            else:
                failed += 1
        
        print(f"Generated {successful} test files, {failed} failed")
    
    # Generate a batch of test files
    elif args.batch:
        batch_size = min(args.batch, len(missing_models))
        print(f"Generating {batch_size} test files...")
        successful = 0
        failed = 0
        
        for model_type in missing_models[:batch_size]:
            print(f"Generating test file for {model_type}...")
            if generate_test_file(model_type):
                successful += 1
            else:
                failed += 1
        
        print(f"Generated {successful} test files, {failed} failed")
    
    # Generate coverage report
    elif args.report:
        report = generate_coverage_report(all_models, existing_tests, missing_models)
        print("\nCoverage Report Summary:")
        print(f"Total Model Types: {len(all_models)}")
        print(f"Implemented Tests: {len(existing_tests)}")
        print(f"Missing Tests: {len(missing_models)}")
        print(f"Coverage: {(len(existing_tests) / len(all_models)) * 100:.1f}%")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())