#!/usr/bin/env python3

"""
Integrate advanced model selection with the test generator.

This script:
1. Enhances the get_model_from_registry function in test_generator_fixed.py
2. Adds hardware-aware and task-specific model selection
3. Updates the integration between advanced_model_selection.py and find_models.py
4. Adds command-line options for hardware and task constraints to the test generator

Usage:
    python integrate_advanced_selection.py [--dry-run]
"""

import os
import sys
import json
import logging
import argparse
import importlib.util
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

def load_module_from_path(module_name, file_path):
    """Dynamically load a module from a file path."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.error(f"Error loading module {module_name} from {file_path}: {e}")
        return None

def update_generator_with_advanced_selection():
    """Update the test generator to use advanced model selection."""
    generator_file = CURRENT_DIR / "test_generator_fixed.py"
    
    try:
        with open(generator_file, 'r') as f:
            generator_code = f.read()
        
        # Check if advanced integration already exists
        if "from advanced_model_selection import" in generator_code:
            logger.info("Advanced model selection integration already exists in test generator")
            return True
        
        # Find the existing get_model_from_registry function
        func_start = generator_code.find("def get_model_from_registry(model_type):")
        if func_start == -1:
            logger.error("Could not find get_model_from_registry function in generator code")
            return False
        
        # Find the end of the function
        func_end = generator_code.find("# Forward declarations for indentation fixing functions", func_start)
        if func_end == -1:
            # Fallback: find the next function
            func_end = generator_code.find("def ", func_start + 10)
            # If still not found, assume it ends before the next major section
            if func_end == -1:
                func_end = generator_code.find("# Constants", func_start)
        
        if func_end == -1:
            logger.error("Could not determine end of get_model_from_registry function")
            return False
        
        # Create the enhanced function with advanced model selection
        enhanced_function = """
# Advanced model selection integration
try:
    from advanced_model_selection import select_model_advanced, get_hardware_profile
    HAS_ADVANCED_SELECTION = True
    logger.info("Advanced model selection available")
except ImportError:
    HAS_ADVANCED_SELECTION = False
    logger.warning("Advanced model selection not available")

def get_model_from_registry(model_type, task=None, hardware_profile=None, max_size_mb=None, framework=None):
    '''Get the best default model for a model type with advanced selection features.
    
    Args:
        model_type (str): The model type (e.g., 'bert', 'gpt2', 't5')
        task (str, optional): The specific task (e.g., 'text-classification')
        hardware_profile (str, optional): Hardware profile name (e.g., 'cpu-small', 'gpu-medium')
        max_size_mb (int, optional): Maximum model size in MB
        framework (str, optional): Framework compatibility (e.g., 'pytorch', 'tensorflow')
        
    Returns:
        str: The recommended model name
    '''
    # Try advanced selection if available
    if HAS_ADVANCED_SELECTION:
        try:
            default_model = select_model_advanced(
                model_type, 
                task=task, 
                hardware_profile=hardware_profile,
                max_size_mb=max_size_mb,
                framework=framework
            )
            logger.info(f"Using advanced selection for {model_type}: {default_model}")
            return default_model
        except Exception as e:
            logger.warning(f"Error using advanced model selection: {e}")
    
    # Fall back to basic model lookup if available
    if HAS_MODEL_LOOKUP:
        try:
            default_model = get_recommended_default_model(model_type)
            logger.info(f"Using recommended model for {model_type}: {default_model}")
            return default_model
        except Exception as e:
            logger.warning(f"Error getting recommended model for {model_type}: {e}")
    
    # Use the static registry as final fallback
    if model_type in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_type].get("default_model")
    
    # For unknown models, use a heuristic approach
    return f"{model_type}-base" if "-base" not in model_type else model_type
"""
        
        # Replace the existing function with the enhanced version
        updated_code = generator_code[:func_start] + enhanced_function + generator_code[func_end:]
        
        # Update the argument parser to accept advanced options
        main_func_start = updated_code.find("def main():")
        if main_func_start != -1:
            parser_section = updated_code.find("parser = argparse.ArgumentParser", main_func_start)
            if parser_section != -1:
                # Find where to insert the new arguments
                add_args_end = updated_code.find("args = parser.parse_args()", parser_section)
                if add_args_end != -1:
                    # Insert advanced arguments before parsing
                    advanced_args = """
    # Advanced model selection options
    parser.add_argument("--task", type=str, help="Specific task for model selection")
    parser.add_argument("--hardware", type=str, help="Hardware profile for model selection")
    parser.add_argument("--max-size", type=int, help="Maximum model size in MB")
    parser.add_argument("--framework", type=str, help="Framework compatibility")
"""
                    updated_code = updated_code[:add_args_end] + advanced_args + updated_code[add_args_end:]
                    logger.info("Added advanced command-line options to main function")
                else:
                    logger.warning("Could not find parser.parse_args() call, skipping advanced args")
            else:
                logger.warning("Could not find argument parser setup, skipping advanced args")
        else:
            logger.warning("Could not find main function, skipping advanced args")
        
        # Update the generate_test_file function to use advanced options
        generate_func = updated_code.find("def generate_test_file(model_family, output_dir=\".\"):")
        if generate_func != -1:
            # Find the call to get_model_from_registry
            model_lookup = updated_code.find("default_model = get_model_from_registry(model_family)", generate_func)
            if model_lookup != -1:
                # Replace with the enhanced version that passes task/hardware args
                enhanced_lookup = """    # Get default model with advanced options from args if available
    if 'args' in globals() and args is not None:
        task = getattr(args, 'task', None)
        hardware = getattr(args, 'hardware', None)
        max_size = getattr(args, 'max_size', None)
        framework = getattr(args, 'framework', None)
        default_model = get_model_from_registry(
            model_family, 
            task=task, 
            hardware_profile=hardware,
            max_size_mb=max_size,
            framework=framework
        )
    else:
        default_model = get_model_from_registry(model_family)"""
                
                updated_code = updated_code.replace(
                    "default_model = get_model_from_registry(model_family)",
                    enhanced_lookup
                )
                logger.info("Updated generate_test_file function to use advanced options")
            else:
                logger.warning("Could not find get_model_from_registry call in generate_test_file")
        else:
            logger.warning("Could not find generate_test_file function")
        
        # Write the updated code back
        with open(generator_file, 'w') as f:
            f.write(updated_code)
        
        logger.info(f"Updated {generator_file} with advanced model selection")
        return True
        
    except Exception as e:
        logger.error(f"Error updating generator with advanced selection: {e}")
        return False

def update_test_script():
    """Create a script to test the advanced features."""
    test_script_file = CURRENT_DIR / "test_model_lookup_advanced.py"
    
    try:
        # Create a simple test script
        script_content = """#!/usr/bin/env python3

\"\"\"
Test script for the enhanced model lookup system with advanced selection.

Usage:
    python test_model_lookup_advanced.py --model-type MODEL_TYPE [--task TASK] [--hardware PROFILE]
\"\"\"

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from advanced_model_selection import (
        select_model_advanced, 
        get_hardware_profile,
        TASK_TO_MODEL_TYPES,
        HARDWARE_PROFILES
    )
    HAS_ADVANCED_SELECTION = True
except ImportError:
    logger.error("Could not import advanced_model_selection.py")
    sys.exit(1)

try:
    from find_models import get_recommended_default_model, query_huggingface_api
    HAS_MODEL_LOOKUP = True
except ImportError:
    logger.error("Could not import find_models.py")
    sys.exit(1)

try:
    from test_generator_fixed import get_model_from_registry
    HAS_GENERATOR = True
except ImportError:
    logger.error("Could not import test_generator_fixed.py")
    sys.exit(1)

def main():
    \"\"\"Main entry point.\"\"\"
    parser = argparse.ArgumentParser(description="Test enhanced model lookup system")
    parser.add_argument("--model-type", type=str, required=True, help="Model type to select")
    parser.add_argument("--task", type=str, help="Specific task for model selection")
    parser.add_argument("--hardware", type=str, choices=list(HARDWARE_PROFILES.keys()), 
                      help="Hardware profile for size constraints")
    parser.add_argument("--max-size", type=int, help="Maximum model size in MB")
    parser.add_argument("--framework", type=str, help="Framework compatibility")
    parser.add_argument("--detect-hardware", action="store_true", help="Detect hardware profile")
    parser.add_argument("--list-tasks", action="store_true", help="List available tasks for model type")
    
    args = parser.parse_args()
    
    # Detect hardware if requested
    if args.detect_hardware:
        profile = get_hardware_profile()
        print(f"\\nDetected Hardware Profile:")
        print(f"  Max model size: {profile['max_size_mb']}MB")
        print(f"  Description: {profile['description']}")
        return 0
    
    # List tasks if requested
    if args.list_tasks:
        print("\\nAvailable Tasks:")
        for task, model_types in sorted(TASK_TO_MODEL_TYPES.items()):
            if args.model_type in model_types:
                print(f"  - {task}")
        return 0
    
    # Get hardware profile if specified
    hardware_profile = None
    if args.hardware:
        hardware_profile = args.hardware
        print(f"Using hardware profile: {hardware_profile}")
        print(f"  Max size: {HARDWARE_PROFILES[hardware_profile]['max_size_mb']}MB")
        print(f"  Description: {HARDWARE_PROFILES[hardware_profile]['description']}")
    
    # Show the model selection process with all methods
    print(f"\\nModel Lookup Results for '{args.model_type}':")
    
    # Method 1: Basic model lookup (find_models.py)
    try:
        basic_model = get_recommended_default_model(args.model_type)
        print(f"\\n1. Basic Model Lookup (find_models.py):")
        print(f"   Selected model: {basic_model}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Method 2: Advanced model selection (advanced_model_selection.py)
    try:
        advanced_model = select_model_advanced(
            args.model_type,
            task=args.task,
            hardware_profile=args.hardware,
            max_size_mb=args.max_size,
            framework=args.framework
        )
        print(f"\\n2. Advanced Model Selection (advanced_model_selection.py):")
        print(f"   Selected model: {advanced_model}")
        print(f"   Task: {args.task if args.task else 'Not specified'}")
        print(f"   Hardware profile: {args.hardware if args.hardware else 'Auto-detected'}")
        print(f"   Max size: {args.max_size if args.max_size else 'Not constrained'}")
        print(f"   Framework: {args.framework if args.framework else 'Any'}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Method 3: Integrated model selection (test_generator_fixed.py)
    try:
        integrated_model = get_model_from_registry(
            args.model_type,
            task=args.task,
            hardware_profile=args.hardware,
            max_size_mb=args.max_size,
            framework=args.framework
        )
        print(f"\\n3. Integrated Model Selection (test_generator_fixed.py):")
        print(f"   Selected model: {integrated_model}")
    except Exception as e:
        print(f"   Error: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
"""
        
        with open(test_script_file, 'w') as f:
            f.write(script_content)
        
        # Make the script executable
        os.chmod(test_script_file, 0o755)
        
        logger.info(f"Created test script: {test_script_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating test script: {e}")
        return False

def update_advanced_selection_with_lookup():
    """Update advanced_model_selection.py to better integrate with find_models.py."""
    advanced_selection_file = CURRENT_DIR / "advanced_model_selection.py"
    
    try:
        with open(advanced_selection_file, 'r') as f:
            advanced_code = f.read()
        
        # Check if integration already exists
        if "from find_models import get_recommended_default_model" in advanced_code:
            logger.info("Integration with find_models.py already exists in advanced_model_selection.py")
            return True
        
        # Create the integration code
        integration_code = """
# Integration with find_models.py
try:
    from find_models import get_recommended_default_model, query_huggingface_api as find_models_query
    HAS_FIND_MODELS = True
    logger.info("find_models.py integration available")
except ImportError:
    HAS_FIND_MODELS = False
    logger.warning("find_models.py not available, using built-in API query")
    find_models_query = None
"""
        
        # Insert the integration after imports
        imports_end = advanced_code.find("# Configure logging")
        if imports_end != -1:
            updated_code = advanced_code[:imports_end] + integration_code + advanced_code[imports_end:]
            
            # Update query_huggingface_api to leverage find_models.py if available
            query_func_start = updated_code.find("def query_huggingface_api(model_type, limit=10, task=None, size_mb=None, framework=None):")
            if query_func_start != -1:
                query_func_end = updated_code.find("def ", query_func_start + 10)
                
                # Create enhanced query function
                enhanced_query = """def query_huggingface_api(model_type, limit=10, task=None, size_mb=None, framework=None):
    \"\"\"Query the HuggingFace API for models with advanced filtering.\"\"\"
    try:
        logger.info(f"Querying HuggingFace API for {model_type} models (task: {task}, size: {size_mb}MB)")
        
        # Use find_models.py query function if available
        if HAS_FIND_MODELS and find_models_query:
            try:
                # Basic query using find_models.py
                models = find_models_query(model_type, limit=limit)
                
                # Apply additional filtering here for task, size, and framework
                filtered_models = []
                for model in models:
                    # Get model size if available, otherwise estimate
                    model_size = estimate_model_size(model)
                    
                    # Skip models that exceed the size limit
                    if size_mb and model_size and model_size > size_mb:
                        logger.info(f"Skipping {model['id']} (size: ~{model_size}MB, limit: {size_mb}MB)")
                        continue
                    
                    # Skip models that don't match the framework
                    if framework and not model_matches_framework(model, framework):
                        logger.info(f"Skipping {model['id']} (framework mismatch)")
                        continue
                    
                    # Skip models that don't match the task (if specified)
                    if task and not task_matches_model(model, task):
                        logger.info(f"Skipping {model['id']} (task mismatch)")
                        continue
                    
                    filtered_models.append(model)
                
                logger.info(f"Found {len(filtered_models)}/{len(models)} models matching criteria")
                return filtered_models
            except Exception as e:
                logger.warning(f"Error using find_models query: {e}, falling back to direct API query")
        
        # Fall back to direct API query
        import requests
        url = f"https://huggingface.co/api/models?filter={model_type}&sort=downloads&direction=-1&limit={limit}"
        
        # Add task filter if specified
        if task:
            url += f"&filter={task}"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        models = response.json()
        
        # If no size limit, return all models
        if not size_mb:
            return models
        
        # Filter models by size if size_mb is specified
        filtered_models = []
        for model in models:
            # Get model size if available, otherwise estimate
            model_size = estimate_model_size(model)
            
            # Skip models that exceed the size limit
            if model_size and model_size > size_mb:
                logger.info(f"Skipping {model['id']} (size: ~{model_size}MB, limit: {size_mb}MB)")
                continue
            
            # Skip models that don't match the framework
            if framework and not model_matches_framework(model, framework):
                logger.info(f"Skipping {model['id']} (framework mismatch)")
                continue
            
            filtered_models.append(model)
        
        logger.info(f"Found {len(filtered_models)}/{len(models)} models matching criteria")
        return filtered_models
        
    except Exception as e:
        logger.error(f"Error querying HuggingFace API: {e}")
        return []

def task_matches_model(model, task):
    \"\"\"Check if a model is suitable for a specific task.\"\"\"
    # Extract tags and model ID
    tags = model.get("tags", [])
    model_id = model.get("id", "").lower()
    
    # Check if task is mentioned in tags
    if any(task.lower() in tag.lower() for tag in tags):
        return True
    
    # Check model type based on model_id parts
    model_parts = model_id.split("/")[-1].split("-")
    model_type = model_parts[0].lower()
    
    # See if model_type is in the list for this task
    for model_list in TASK_TO_MODEL_TYPES.get(task, []):
        if model_type in model_list:
            return True
    
    return False
"""
                
                # Replace the existing function with the enhanced version
                if query_func_end != -1:
                    updated_code = updated_code[:query_func_start] + enhanced_query + updated_code[query_func_end:]
                else:
                    # Find the next section if function end not found
                    next_section = updated_code.find("def estimate_model_size(model_info):", query_func_start)
                    if next_section != -1:
                        updated_code = updated_code[:query_func_start] + enhanced_query + updated_code[next_section:]
                
                # Write the updated code back
                with open(advanced_selection_file, 'w') as f:
                    f.write(updated_code)
                
                logger.info(f"Updated {advanced_selection_file} with find_models.py integration")
                return True
            else:
                logger.error("Could not find query_huggingface_api function")
                return False
        else:
            logger.error("Could not find appropriate insertion point for imports")
            return False
            
    except Exception as e:
        logger.error(f"Error updating advanced_model_selection.py: {e}")
        return False

def create_github_workflow():
    """Create a GitHub Actions workflow for model registry updates."""
    workflow_file = CURRENT_DIR / "github-workflow-model-lookup.yml"
    
    try:
        workflow_content = """name: Update Model Registry

on:
  # Run weekly to keep the model registry fresh
  schedule:
    - cron: '0 0 * * 0'  # Run at midnight every Sunday
  
  # Allow manual trigger
  workflow_dispatch:
    inputs:
      update_mode:
        description: 'Update mode'
        required: true
        default: 'all'
        type: choice
        options:
          - all
          - popular
          - minimal
      force_update:
        description: 'Force update of registry'
        required: false
        default: false
        type: boolean

jobs:
  update-registry:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install requests huggingface_hub
      
      - name: Update model registry
        run: |
          cd test/skills
          python find_models.py --update-registry
          
          # If using advanced_model_selection.py
          if [ -f advanced_model_selection.py ]; then
            # Update with task-specific, hardware-aware defaults
            python advanced_model_selection.py --model-type bert --task text-classification
            python advanced_model_selection.py --model-type gpt2 --task text-generation
            python advanced_model_selection.py --model-type t5 --task text2text-generation
            python advanced_model_selection.py --model-type vit --task image-classification
            python advanced_model_selection.py --model-type whisper --task automatic-speech-recognition
          fi
          
          # Update test generator with new registry
          python integrate_model_lookup.py --update-all
      
      - name: Create PR if changes detected
        uses: peter-evans/create-pull-request@v5
        with:
          commit-message: 'chore: Update HuggingFace model registry'
          title: 'chore: Update HuggingFace model registry'
          body: |
            This PR updates the HuggingFace model registry with the latest popular models.
            
            - Updates default model selections based on current download counts
            - Refreshes the huggingface_model_types.json registry
            - Updates the MODEL_REGISTRY in test_generator_fixed.py
            
            This is an automated PR generated by the weekly model registry update workflow.
          branch: update-model-registry
          base: main
          delete-branch: true
  
  generate-test-samples:
    needs: update-registry
    runs-on: ubuntu-latest
    steps:
      - name: Checkout updated branch
        uses: actions/checkout@v3
        with:
          ref: update-model-registry
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install requests
      
      - name: Generate test samples
        run: |
          cd test/skills
          mkdir -p test_output
          
          # Generate test files for key model types
          python test_generator_fixed.py --generate bert --output-dir test_output --verify
          python test_generator_fixed.py --generate gpt2 --output-dir test_output --verify
          python test_generator_fixed.py --generate t5 --output-dir test_output --verify
          python test_generator_fixed.py --generate vit --output-dir test_output --verify
          
          # Test with hardware-aware selection if available
          if [ -f advanced_model_selection.py ]; then
            python test_generator_fixed.py --generate bert --output-dir test_output --hardware cpu-small --verify
          fi
      
      - name: Upload test samples
        uses: actions/upload-artifact@v3
        with:
          name: test-samples
          path: test/skills/test_output/
          retention-days: 7
"""
        
        with open(workflow_file, 'w') as f:
            f.write(workflow_content)
        
        logger.info(f"Created GitHub Actions workflow: {workflow_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating GitHub Actions workflow: {e}")
        return False

def create_documentation():
    """Create documentation for the advanced model selection system."""
    docs_file = CURRENT_DIR / "MODEL_LOOKUP_ADVANCED_README.md"
    
    try:
        docs_content = """# Advanced Model Selection for HuggingFace Test Generator

This document describes the enhanced model selection system that provides task-specific,
hardware-aware, and framework-compatible model recommendations for the HuggingFace test generator.

## Overview

The advanced model selection system:

1. Selects the most appropriate HuggingFace models based on multiple criteria:
   - Task compatibility (e.g., text-classification, image-segmentation)
   - Hardware constraints (e.g., CPU/GPU, memory limits)
   - Framework compatibility (PyTorch, TensorFlow, JAX)
   - Download popularity (trending and well-maintained models)
   - Benchmark performance (when available)

2. Provides robust fallback mechanisms:
   - API queries for fresh, popularity-based selection
   - Registry-based caching for offline/CI environments
   - Size-based fallbacks (e.g., large → base → small → tiny)
   - Static defaults as final fallbacks

## Using Advanced Selection

### Command Line Options

The `test_generator_fixed.py` now supports the following advanced options:

```bash
# Basic model generation with automatic model selection
python test_generator_fixed.py --generate bert

# Task-specific model selection
python test_generator_fixed.py --generate bert --task text-classification

# Hardware-constrained model selection
python test_generator_fixed.py --generate gpt2 --hardware cpu-small

# Size-constrained model selection
python test_generator_fixed.py --generate t5 --max-size 500

# Framework-specific model selection
python test_generator_fixed.py --generate bert --framework pytorch
```

### Testing Advanced Features

Use the provided test script to explore different selection options:

```bash
# Test all selection methods for a model type
python test_model_lookup_advanced.py --model-type bert

# Test with task constraints
python test_model_lookup_advanced.py --model-type bert --task text-classification

# Detect available hardware
python test_model_lookup_advanced.py --detect-hardware

# List suitable tasks for a model type
python test_model_lookup_advanced.py --model-type bert --list-tasks
```

## Supported Tasks

The system supports the following tasks mapped to appropriate model types:

| Task | Description | Suitable Model Types |
|------|-------------|---------------------|
| text-classification | Text classification/categorization | bert, roberta, distilbert, xlm-roberta |
| token-classification | Token-level tasks (NER, POS) | bert, roberta, distilbert |
| question-answering | Question answering | bert, roberta, distilbert |
| text-generation | Open-ended text generation | gpt2, gpt-j, gpt-neo, llama |
| summarization | Text summarization | t5, bart, pegasus |
| translation | Machine translation | t5, mbart, mt5 |
| image-classification | Image classification | vit, resnet, swin |
| image-segmentation | Image segmentation | mask2former, segformer |
| automatic-speech-recognition | Speech recognition | whisper, wav2vec2, hubert |
| visual-question-answering | VQA | llava, blip |

## Hardware Profiles

The system includes predefined hardware profiles for common scenarios:

| Profile | Description | Maximum Model Size |
|---------|-------------|-------------------|
| cpu-small | Limited CPU environments | 500 MB |
| cpu-medium | Standard CPU environments | 2,000 MB |
| cpu-large | High-memory CPU environments | 10,000 MB |
| gpu-small | Entry-level GPUs | 5,000 MB |
| gpu-medium | Mid-range GPUs | 15,000 MB |
| gpu-large | High-end GPUs | 50,000 MB |

## Integration with CI/CD

The system includes a GitHub Actions workflow for automated registry updates:

1. Weekly updates of model popularity data
2. Verification of new model selections
3. Pull request creation for registry changes
4. Test sample generation for verification

Use the following environment variables to control behavior in CI/CD:

- `USE_STATIC_REGISTRY=true` - Disable API calls and use cached registry
- `FORCE_API_CALLS=true` - Force API calls even in CI environments
- `MAX_REGISTRY_AGE_DAYS=30` - Control maximum age of registry before refresh

## Extending the System

### Adding New Tasks

To add a new task, update the `TASK_TO_MODEL_TYPES` dictionary in `advanced_model_selection.py`:

```python
TASK_TO_MODEL_TYPES = {
    # Existing tasks...
    "new-task-name": ["model-type1", "model-type2"],
}
```

### Adding New Hardware Profiles

To add a new hardware profile, update the `HARDWARE_PROFILES` dictionary:

```python
HARDWARE_PROFILES = {
    # Existing profiles...
    "new-profile-name": {
        "max_size_mb": 8000,
        "description": "Description of the profile"
    },
}
```

## Troubleshooting

### API Rate Limiting

If you encounter HuggingFace API rate limiting:

1. Set a longer delay between requests
2. Use the cached registry data with `USE_STATIC_REGISTRY=true`
3. Update specific model types individually rather than all at once

### Size Estimation Errors

If model size estimation is inaccurate:

1. Provide an explicit size constraint with `--max-size`
2. Add size indicators in the model registry
3. Update the size estimation heuristics in `estimate_model_size()`

## Components

The advanced model selection system consists of:

- `advanced_model_selection.py` - Core selection logic
- `find_models.py` - HuggingFace API interaction
- Integration with `test_generator_fixed.py`
- Support files (registry, benchmarks, etc.)

## Development

When developing the system:

1. Run tests with `python test_advanced_model_selection.py`
2. Verify integration with `python test_model_lookup_advanced.py`
3. Update documentation when adding new features
"""
        
        with open(docs_file, 'w') as f:
            f.write(docs_content)
        
        logger.info(f"Created documentation: {docs_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating documentation: {e}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Integrate advanced model selection with test generator")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without applying them")
    
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("Running in dry-run mode, no changes will be applied")
    
    # Update advanced_model_selection.py to integrate with find_models.py
    if not args.dry_run:
        success = update_advanced_selection_with_lookup()
        if not success:
            logger.error("Failed to update advanced_model_selection.py with find_models.py integration")
    
    # Update test_generator_fixed.py to use advanced model selection
    if not args.dry_run:
        success = update_generator_with_advanced_selection()
        if not success:
            logger.error("Failed to update test_generator_fixed.py with advanced model selection")
    
    # Create test script
    if not args.dry_run:
        success = update_test_script()
        if not success:
            logger.error("Failed to create test script")
    
    # Create GitHub workflow
    if not args.dry_run:
        success = create_github_workflow()
        if not success:
            logger.error("Failed to create GitHub workflow")
    
    # Create documentation
    if not args.dry_run:
        success = create_documentation()
        if not success:
            logger.error("Failed to create documentation")
    
    logger.info("Integration completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())