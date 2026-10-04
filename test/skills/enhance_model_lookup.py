#!/usr/bin/env python3

"""
Enhance the HuggingFace model lookup integration with the test generator.

This script:
1. Updates all template files to support dynamic model selection
2. Enhances the test generator's template processing to properly handle model lookup
3. Creates a test suite for the model lookup system
4. Updates documentation with comprehensive usage examples

Usage:
    python enhance_model_lookup.py [--templates] [--test-suite] [--docs]
"""

import os
import sys
import re
import json
import logging
import argparse
import glob
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = CURRENT_DIR / "templates"
REGISTRY_FILE = CURRENT_DIR / "huggingface_model_types.json"
DOCS_FILE = CURRENT_DIR / "MODEL_LOOKUP_SUMMARY.md"

def update_template_files():
    """Update all template files to support dynamic model selection."""
    logger.info("Updating template files for model lookup integration")
    
    # Find all template files
    template_files = glob.glob(str(TEMPLATES_DIR / "*_template.py"))
    
    if not template_files:
        logger.warning(f"No template files found in {TEMPLATES_DIR}")
        return False
    
    logger.info(f"Found {len(template_files)} template files to update")
    
    # Read model registry for default models
    try:
        with open(REGISTRY_FILE, 'r') as f:
            registry_data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading model registry: {e}")
        registry_data = {}
    
    # Default values for templates, mapped by architecture
    default_values = {
        "encoder_only": {
            "model_type": "bert",
            "default_model": registry_data.get("bert", {}).get("default_model", "google-bert/bert-base-uncased"),
            "model_class": "BertForMaskedLM",
            "task": "fill-mask",
            "input_text": "The quick brown fox jumps over the [MASK] dog.",
        },
        "decoder_only": {
            "model_type": "gpt2",
            "default_model": registry_data.get("gpt2", {}).get("default_model", "gpt2"),
            "model_class": "GPT2LMHeadModel",
            "task": "text-generation",
            "input_text": "GPT-2 is a transformer model that",
        },
        "encoder_decoder": {
            "model_type": "t5",
            "default_model": registry_data.get("t5", {}).get("default_model", "t5-small"),
            "model_class": "T5ForConditionalGeneration",
            "task": "text2text-generation",
            "input_text": "translate English to German: The house is wonderful.",
        },
        "vision": {
            "model_type": "vit",
            "default_model": registry_data.get("vit", {}).get("default_model", "google/vit-base-patch16-224"),
            "model_class": "ViTForImageClassification",
            "task": "image-classification",
            "input_text": None,
        },
        "vision_text": {
            "model_type": "clip",
            "default_model": registry_data.get("clip", {}).get("default_model", "openai/clip-vit-base-patch32"),
            "model_class": "CLIPModel",
            "task": "zero-shot-image-classification",
            "input_text": None,
        },
        "speech": {
            "model_type": "whisper",
            "default_model": registry_data.get("whisper", {}).get("default_model", "openai/whisper-base.en"),
            "model_class": "WhisperForConditionalGeneration",
            "task": "automatic-speech-recognition",
            "input_text": None,
        },
        "multimodal": {
            "model_type": "blip",
            "default_model": registry_data.get("blip", {}).get("default_model", "Salesforce/blip-image-captioning-base"),
            "model_class": "BlipForConditionalGeneration",
            "task": "image-to-text",
            "input_text": None,
        },
    }
    
    # Process each template file
    updated_count = 0
    for template_file in template_files:
        try:
            # Determine template type
            template_name = os.path.basename(template_file)
            template_type = template_name.split("_")[0]
            
            # Find the appropriate default values
            for t, values in default_values.items():
                if template_type in t:
                    defaults = values
                    break
            else:
                # Fallback to encoder_only if no match
                defaults = default_values["encoder_only"]
            
            with open(template_file, 'r') as f:
                content = f.read()
            
            # Insert model lookup code if not already present
            if "get_model_from_registry" not in content:
                # Find a suitable insertion point after imports
                import_section_end = re.search(r'import\s+[^\n]+\n\s*(\n|$)', content)
                if import_section_end:
                    insert_pos = import_section_end.end()
                    
                    # Model lookup code to insert
                    lookup_code = f"""
# Model lookup integration
try:
    from find_models import get_recommended_default_model
    HAS_MODEL_LOOKUP = True
    logger.info("Model lookup available")
except ImportError:
    HAS_MODEL_LOOKUP = False
    logger.warning("Model lookup not available, using static defaults")

def get_model_from_registry(model_type="{defaults['model_type']}"):
    '''Get the best default model for a model type.'''
    if HAS_MODEL_LOOKUP:
        try:
            default_model = get_recommended_default_model(model_type)
            logger.info(f"Using recommended model for {{model_type}}: {{default_model}}")
            return default_model
        except Exception as e:
            logger.warning(f"Error getting recommended model for {{model_type}}: {{e}}")
    
    # Fallback to static defaults
    defaults = {{
        "{defaults['model_type']}": "{defaults['default_model']}",
        # Add more defaults here as needed
    }}
    
    return defaults.get(model_type, f"{{model_type}}-base" if "-base" not in model_type else model_type)
"""
                    
                    # Insert the code
                    updated_content = content[:insert_pos] + lookup_code + content[insert_pos:]
                    
                    # Replace hardcoded model references with dynamic lookup
                    # Pattern for model path strings like "bert-base-uncased" or "google/vit-base-patch16-224"
                    model_pattern = re.compile(r'(["\'])([a-zA-Z0-9-_/]+(?:bert|gpt2|t5|vit|clip|whisper|blip)[a-zA-Z0-9-_/]*)(["\'])')
                    
                    # Replace with get_model_from_registry call
                    def replace_model(match):
                        quote = match.group(1)
                        model_path = match.group(2)
                        
                        # Skip certain patterns that shouldn't be replaced
                        if any(x in model_path for x in ['import', 'from', 'path', 'module', 'class']):
                            return match.group(0)
                            
                        # Extract model type from path
                        model_type = defaults['model_type']
                        for t in default_values.keys():
                            type_value = default_values[t]['model_type']
                            if type_value in model_path.lower():
                                model_type = type_value
                                break
                            
                        return f"{quote}{{get_model_from_registry('{model_type}')}}{quote}"
                    
                    # Apply replacements - carefully to avoid false positives
                    # Only replace the default model references in specific contexts
                    for model_ref in [
                        f'"{defaults["default_model"]}"', 
                        f"'{defaults['default_model']}'"
                    ]:
                        if model_ref in updated_content:
                            # Only replace in contexts like model=".." or path=".." to avoid false positives
                            contexts = [
                                f'model={model_ref}',
                                f'model_id={model_ref}',
                                f'model_name={model_ref}',
                                f'path={model_ref}',
                                f'MODEL_PATH = {model_ref}'
                            ]
                            
                            for ctx in contexts:
                                if ctx in updated_content:
                                    model_type = defaults['model_type']
                                    updated_content = updated_content.replace(
                                        ctx, 
                                        ctx.replace(model_ref, f"get_model_from_registry('{model_type}')")
                                    )
                    
                    # Write the updated content
                    with open(template_file, 'w') as f:
                        f.write(updated_content)
                    
                    logger.info(f"✅ Updated template file: {template_file}")
                    updated_count += 1
                else:
                    logger.warning(f"Could not find insertion point in {template_file}")
            else:
                logger.info(f"Template file already has model lookup integration: {template_file}")
        
        except Exception as e:
            logger.error(f"Error updating template file {template_file}: {e}")
    
    logger.info(f"Updated {updated_count}/{len(template_files)} template files")
    return updated_count > 0

def create_test_suite():
    """Create a test suite for the model lookup system."""
    logger.info("Creating test suite for model lookup system")
    
    test_dir = CURRENT_DIR / "model_lookup_tests"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create test suite file
    test_suite_file = test_dir / "test_model_lookup_suite.py"
    
    content = """#!/usr/bin/env python3

"""
    content += '"""'
    content += """
Test suite for the HuggingFace model lookup system.

This test suite verifies:
1. The direct API integration
2. The test generator integration 
3. The template file integration
4. Fallback mechanisms when the API is unavailable

Usage:
    python test_model_lookup_suite.py [--model-type TYPE] [--all]
"""
    content += '"""'
    content += """

import os
import sys
import json
import logging
import argparse
import importlib.util
from pathlib import Path
from unittest import mock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

# Try to import required modules
try:
    from find_models import get_recommended_default_model, query_huggingface_api
    HAS_FIND_MODELS = True
except ImportError:
    HAS_FIND_MODELS = False
    logger.warning("Could not import find_models.py")

# Import test_generator_fixed.py if available
try:
    from test_generator_fixed import get_model_from_registry
    HAS_GENERATOR = True
except ImportError:
    HAS_GENERATOR = False
    logger.warning("Could not import test_generator_fixed.py")

def test_direct_api(model_type="bert"):
    """Test direct API integration."""
    if not HAS_FIND_MODELS:
        logger.error("find_models.py module not available")
        return False
    
    try:
        logger.info(f"Testing direct API for model type: {model_type}")
        
        # Call the API
        default_model = get_recommended_default_model(model_type)
        logger.info(f"Default model for {model_type}: {default_model}")
        
        # Basic validation
        if not default_model or not isinstance(default_model, str):
            logger.error(f"Invalid default model: {default_model}")
            return False
        
        # Check if model_type is in the result
        if model_type in default_model.lower():
            logger.info(f"✅ Model type found in default model")
        
        return True
    
    except Exception as e:
        logger.error(f"Error in direct API test: {e}")
        return False

def test_api_fallback(model_type="bert"):
    """Test API fallback mechanism."""
    if not HAS_FIND_MODELS:
        logger.error("find_models.py module not available")
        return False
    
    try:
        logger.info(f"Testing API fallback for model type: {model_type}")
        
        # Mock query_huggingface_api to simulate API failure
        with mock.patch('find_models.query_huggingface_api', return_value=[]):
            default_model = get_recommended_default_model(model_type)
            logger.info(f"Fallback model for {model_type}: {default_model}")
            
            # Basic validation
            if not default_model or not isinstance(default_model, str):
                logger.error(f"Invalid fallback model: {default_model}")
                return False
            
            logger.info(f"✅ Fallback mechanism working correctly")
            return True
    
    except Exception as e:
        logger.error(f"Error in API fallback test: {e}")
        return False

def test_generator_integration(model_type="bert"):
    """Test test generator integration."""
    if not HAS_GENERATOR:
        logger.error("test_generator_fixed.py module not available")
        return False
    
    try:
        logger.info(f"Testing generator integration for model type: {model_type}")
        
        # Call the generator's function
        default_model = get_model_from_registry(model_type)
        logger.info(f"Generator default model for {model_type}: {default_model}")
        
        # Basic validation
        if not default_model or not isinstance(default_model, str):
            logger.error(f"Invalid generator model: {default_model}")
            return False
        
        logger.info(f"✅ Generator integration working correctly")
        return True
    
    except Exception as e:
        logger.error(f"Error in generator integration test: {e}")
        return False

def test_full_integration(model_type="bert"):
    """Test full integration including test file generation."""
    try:
        logger.info(f"Testing full integration for model type: {model_type}")
        
        # Import the generate_minimal_test.py module
        sys.path.insert(0, PARENT_DIR)
        try:
            from generate_minimal_test import generate_minimal_test
            HAS_GENERATOR_MINIMAL = True
        except ImportError:
            HAS_GENERATOR_MINIMAL = False
            logger.warning("Could not import generate_minimal_test.py")
            return False
        
        # Create a test file
        temp_dir = os.path.join(CURRENT_DIR, "temp_tests")
        os.makedirs(temp_dir, exist_ok=True)
        
        result = generate_minimal_test(model_type, temp_dir)
        
        if result:
            logger.info(f"✅ Full integration test passed")
            return True
        else:
            logger.error(f"❌ Failed to generate test file")
            return False
    
    except Exception as e:
        logger.error(f"Error in full integration test: {e}")
        return False

def run_all_tests(model_type="bert"):
    """Run all tests with the specified model type."""
    results = {
        "direct_api": test_direct_api(model_type),
        "api_fallback": test_api_fallback(model_type),
        "generator_integration": test_generator_integration(model_type),
        "full_integration": test_full_integration(model_type)
    }
    
    # Print results
    print(f"\n=== Test Results for {model_type} ===\n")
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(results.values())
    print(f"\nOverall Result: {'✅ PASS' if all_passed else '❌ FAIL'}\n")
    
    return all_passed

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test suite for model lookup system")
    parser.add_argument("--model-type", type=str, default="bert", help="Model type to test")
    parser.add_argument("--all", action="store_true", help="Test with all core model types")
    parser.add_argument("--test", choices=["direct", "fallback", "generator", "full", "all"], 
                      default="all", help="Specific test to run")
    
    args = parser.parse_args()
    
    if args.all:
        # Test with core model types
        core_types = ["bert", "gpt2", "t5", "vit"]
        results = {}
        
        for model_type in core_types:
            logger.info(f"\n=== Testing {model_type} ===\n")
            results[model_type] = run_all_tests(model_type)
        
        # Print overall summary
        print("\n=== Overall Test Summary ===\n")
        for model_type, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} - {model_type}")
        
        all_passed = all(results.values())
        print(f"\nFinal Result: {'✅ PASS' if all_passed else '❌ FAIL'}")
        
        return 0 if all_passed else 1
    else:
        # Run specific test or all tests for single model type
        if args.test == "direct":
            success = test_direct_api(args.model_type)
        elif args.test == "fallback":
            success = test_api_fallback(args.model_type)
        elif args.test == "generator":
            success = test_generator_integration(args.model_type)
        elif args.test == "full":
            success = test_full_integration(args.model_type)
        else:  # all
            success = run_all_tests(args.model_type)
        
        return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
"""
    
    # Write the test suite
    with open(test_suite_file, 'w') as f:
        f.write(content)
    
    logger.info(f"✅ Created test suite at {test_suite_file}")
    
    # Create a minimal benchmark file
    benchmark_file = test_dir / "benchmark_model_lookup.py"
    
    benchmark_content = """#!/usr/bin/env python3

"""
    benchmark_content += '"""'
    benchmark_content += """
Benchmark the performance of the HuggingFace model lookup system.

This script measures:
1. API query time for different model types
2. Time savings from using the registry cache
3. Performance comparison between direct API calls and get_model_from_registry

Usage:
    python benchmark_model_lookup.py [--iterations N] [--reset-cache]
"""
    benchmark_content += '"""'
    benchmark_content += """

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

# Try to import required modules
try:
    from find_models import get_recommended_default_model, query_huggingface_api
    HAS_FIND_MODELS = True
except ImportError:
    HAS_FIND_MODELS = False
    logger.warning("Could not import find_models.py")

# Import test_generator_fixed.py if available
try:
    from test_generator_fixed import get_model_from_registry
    HAS_GENERATOR = True
except ImportError:
    HAS_GENERATOR = False
    logger.warning("Could not import test_generator_fixed.py")

def reset_registry_cache():
    """Reset the registry cache file."""
    registry_file = os.path.join(PARENT_DIR, "huggingface_model_types.json")
    if os.path.exists(registry_file):
        # Backup the registry
        import shutil
        backup_file = f"{registry_file}.bak"
        shutil.copy2(registry_file, backup_file)
        logger.info(f"Backed up registry to {backup_file}")
        
        # Remove or empty the registry
        with open(registry_file, 'w') as f:
            f.write("{}")
        
        logger.info(f"Reset registry cache: {registry_file}")
        return True
    else:
        logger.warning(f"Registry file not found: {registry_file}")
        return False

def measure_api_query_time(model_type, iterations=5):
    """Measure the time to query the HuggingFace API."""
    if not HAS_FIND_MODELS:
        logger.error("find_models.py module not available")
        return None
    
    logger.info(f"Measuring API query time for {model_type} ({iterations} iterations)")
    
    times = []
    for i in range(iterations):
        start_time = time.time()
        try:
            models = query_huggingface_api(model_type, limit=10)
            elapsed = time.time() - start_time
            times.append(elapsed)
            logger.info(f"  Iteration {i+1}: {elapsed:.4f}s - Found {len(models)} models")
        except Exception as e:
            logger.error(f"  Iteration {i+1}: Error - {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        logger.info(f"Average API query time: {avg_time:.4f}s")
        return avg_time
    else:
        logger.error("No successful API queries")
        return None

def measure_model_lookup_time(model_type, iterations=5, use_generator=False):
    """Measure the time to get a recommended default model."""
    if (not HAS_FIND_MODELS) or (use_generator and not HAS_GENERATOR):
        logger.error("Required modules not available")
        return None
    
    logger.info(f"Measuring {'generator' if use_generator else 'direct'} model lookup time for {model_type} ({iterations} iterations)")
    
    times = []
    for i in range(iterations):
        start_time = time.time()
        try:
            if use_generator:
                model = get_model_from_registry(model_type)
            else:
                model = get_recommended_default_model(model_type)
            
            elapsed = time.time() - start_time
            times.append(elapsed)
            logger.info(f"  Iteration {i+1}: {elapsed:.4f}s - Model: {model}")
        except Exception as e:
            logger.error(f"  Iteration {i+1}: Error - {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        logger.info(f"Average lookup time: {avg_time:.4f}s")
        return avg_time
    else:
        logger.error("No successful lookups")
        return None

def run_benchmark(model_types=None, iterations=5, reset_cache=False):
    """Run the full benchmark."""
    if model_types is None:
        model_types = ["bert", "gpt2", "t5", "vit"]
    
    if reset_cache:
        reset_registry_cache()
    
    results = {}
    
    for model_type in model_types:
        logger.info(f"\n=== Benchmarking {model_type} ===\n")
        
        # First run: Direct API (no cache)
        direct_time = measure_model_lookup_time(model_type, iterations, use_generator=False)
        
        # Second run: Test generator (with cache from first run)
        generator_time = measure_model_lookup_time(model_type, iterations, use_generator=True)
        
        # Store results
        results[model_type] = {
            "direct_time": direct_time,
            "generator_time": generator_time,
            "speedup": (direct_time / generator_time) if direct_time and generator_time else None
        }
    
    # Print summary
    print("\n=== Benchmark Summary ===\n")
    print(f"{'Model Type':<10} | {'Direct (s)':<10} | {'Generator (s)':<12} | {'Speedup':<10}")
    print("-" * 50)
    
    for model_type, data in results.items():
        direct = f"{data['direct_time']:.4f}" if data['direct_time'] else "N/A"
        generator = f"{data['generator_time']:.4f}" if data['generator_time'] else "N/A"
        speedup = f"{data['speedup']:.2f}x" if data['speedup'] else "N/A"
        
        print(f"{model_type:<10} | {direct:<10} | {generator:<12} | {speedup:<10}")
    
    return results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Benchmark model lookup system")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations for each test")
    parser.add_argument("--reset-cache", action="store_true", help="Reset the registry cache before benchmarking")
    parser.add_argument("--models", type=str, help="Comma-separated list of model types to benchmark")
    
    args = parser.parse_args()
    
    # Parse model types
    if args.models:
        model_types = [m.strip() for m in args.models.split(",")]
    else:
        model_types = ["bert", "gpt2", "t5", "vit"]
    
    # Run the benchmark
    run_benchmark(model_types, args.iterations, args.reset_cache)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
"""
    
    # Write the benchmark file
    with open(benchmark_file, 'w') as f:
        f.write(benchmark_content)
    
    logger.info(f"✅ Created benchmark file at {benchmark_file}")
    return True

def update_documentation():
    """Update documentation with comprehensive usage examples."""
    logger.info("Updating documentation with comprehensive usage examples")
    
    # Create content for MODEL_LOOKUP_SUMMARY.md
    content = """# HuggingFace Model Lookup System

## Overview

The HuggingFace Model Lookup System provides dynamic selection of appropriate models for testing and development. It integrates with the test generator to ensure tests use reliable, popular models that are appropriate for the task.

## Key Features

- **Dynamic Model Selection**: Automatically selects appropriate models based on popularity and size
- **API Integration**: Queries the HuggingFace API to find the most popular models of each type
- **Intelligent Fallbacks**: Gracefully degrades from API → registry → hardcoded defaults
- **Smart Selection**: Prefers "base" or "small" models that are still popular enough to be reliable
- **Caching**: Stores model information to minimize API calls
- **Template Integration**: Works with all template types for test generation

## Implementation Details

### Core Components

1. **find_models.py**: Core module with:
   - `query_huggingface_api(model_type, limit=5, min_downloads=1000)`: Fetches popular models
   - `get_recommended_default_model(model_type)`: Selects the best model based on criteria
   - `try_model_access(model_name)`: Verifies model availability

2. **huggingface_model_types.json**: Registry file with:
   ```json
   {
     "bert": {
       "default_model": "google-bert/bert-base-uncased",
       "models": ["list", "of", "models"],
       "downloads": {"model": download_count},
       "updated_at": "timestamp"
     }
   }
   ```

3. **test_generator_fixed.py**: Generator integration through:
   ```python
   def get_model_from_registry(model_type):
       '''Get the best default model for a model type.'''
       if HAS_MODEL_LOOKUP:
           try:
               default_model = get_recommended_default_model(model_type)
               return default_model
           except Exception:
               # Fall back to registry lookup
               pass
       
       # Use static registry as fallback
       if model_type in MODEL_REGISTRY:
           return MODEL_REGISTRY[model_type].get("default_model")
       
       # Heuristic approach for unknown models
       return f"{model_type}-base" if "-base" not in model_type else model_type
   ```

## Usage Examples

### Basic Usage

```python
from find_models import get_recommended_default_model

# Get the default model for a specific type
model = get_recommended_default_model("bert")
print(f"Using model: {model}")  # e.g., "google-bert/bert-base-uncased"
```

### With Test Generator

```python
from test_generator_fixed import generate_test_file

# Generate a test file for a specific model type
# The function will automatically use the lookup system
generate_test_file("bert", output_dir="my_tests")
```

### Minimal Test Generation

```python
from generate_minimal_test import generate_minimal_test

# Generate a minimal test file with model lookup
generate_minimal_test("gpt2", output_dir="minimal_tests")
```

### Command-line Usage

```bash
# Update the registry with new model information
python expand_model_registry.py --model-type bert

# Generate a test with dynamic model selection
python test_generator_fixed.py --generate bert

# Verify the model lookup integration
python verify_model_lookup.py --model-type bert --generate
```

## Model Selection Process

1. **API Query**: The system first queries the HuggingFace API to get popular models of the requested type.

2. **Smart Filtering**: It prioritizes models with "base" or "small" in the name to favor appropriately-sized models.

3. **Downloads Check**: It verifies models have sufficient downloads to ensure reliability.

4. **Registry Fallback**: If the API is unavailable, it uses cached information from the registry file.

5. **Static Fallback**: If all else fails, it uses hardcoded defaults like "bert-base-uncased" or "gpt2".

## Future Enhancements

- **Periodic Registry Updates**: Automatically refresh the registry in CI/CD environments
- **Model Verification**: Pre-check model availability before returning
- **Architecture-Aware Selection**: Better model selection based on architecture requirements
- **Task-Specific Selection**: Select models that are optimized for specific tasks

## Troubleshooting

If you encounter issues with the model lookup system:

1. **API Access**: Ensure you have internet access for API queries
2. **Registry File**: Check if huggingface_model_types.json exists and is valid
3. **Module Imports**: Verify find_models.py is in the Python path
4. **Logs**: Check the logs for specific error messages
5. **Fallbacks**: The system should still work with fallbacks even if API fails
"""
    
    # Write the documentation
    with open(DOCS_FILE, 'w') as f:
        f.write(content)
    
    logger.info(f"✅ Updated documentation at {DOCS_FILE}")
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Enhance model lookup integration")
    parser.add_argument("--templates", action="store_true", help="Update template files")
    parser.add_argument("--test-suite", action="store_true", help="Create test suite")
    parser.add_argument("--docs", action="store_true", help="Update documentation")
    parser.add_argument("--all", action="store_true", help="Perform all enhancements")
    
    args = parser.parse_args()
    
    if args.all or args.templates:
        update_template_files()
    
    if args.all or args.test_suite:
        create_test_suite()
    
    if args.all or args.docs:
        update_documentation()
    
    logger.info("Enhancement of model lookup integration complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())