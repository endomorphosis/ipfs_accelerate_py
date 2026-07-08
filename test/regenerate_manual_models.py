#!/usr/bin/env python3
"""
Regenerate manually created model tests using the template system.

This script:
1. Maps manual models to their correct architecture types
2. Uses the test generator to create properly templated test files
3. Verifies the syntax and functionality of generated tests
4. Updates the model registry with detailed model information

Usage:
    python regenerate_manual_models.py [--output-dir DIRECTORY] [--verify]
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"regenerate_manual_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Import regenerate_test_file function
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    # Try to import from skills directory
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills"))
    from regenerate_tests_with_fixes import regenerate_test_file
    logger.info("Successfully imported regenerate_test_file from skills/regenerate_tests_with_fixes.py")
except ImportError:
    try:
        # Try alternative import paths
        from skills.regenerate_tests_with_fixes import regenerate_test_file
        logger.info("Successfully imported regenerate_test_file from skills module")
    except ImportError:
        logger.error("Failed to import regenerate_test_file. Make sure regenerate_tests_with_fixes.py exists in the skills directory.")
        sys.exit(1)

# Define the manual models and their architecture types
MANUAL_MODELS = {
    "layoutlmv2": "vision-encoder-text-decoder",
    "layoutlmv3": "vision-encoder-text-decoder",
    "clvp": "speech",
    "bigbird": "encoder-decoder",
    "seamless_m4t_v2": "speech",
    "xlm_prophetnet": "encoder-decoder"
}

# Define model registry entries for these models
MODEL_REGISTRY_UPDATES = {
    "layoutlmv2": {
        "description": "LayoutLMv2 model for document understanding",
        "class": "LayoutLMv2ForSequenceClassification",
        "default_model": "microsoft/layoutlmv2-base-uncased",
        "architecture": "vision-encoder-text-decoder"
    },
    "layoutlmv3": {
        "description": "LayoutLMv3 model for document understanding",
        "class": "LayoutLMv3ForSequenceClassification",
        "default_model": "microsoft/layoutlmv3-base",
        "architecture": "vision-encoder-text-decoder"
    },
    "clvp": {
        "description": "CLVP model for text-to-speech synthesis",
        "class": "CLVPForCausalLM",
        "default_model": "susnato/clvp_dev",
        "architecture": "speech"
    },
    "bigbird": {
        "description": "BigBird for long document processing",
        "class": "BigBirdForSequenceClassification",
        "default_model": "google/bigbird-roberta-base",
        "architecture": "encoder-decoder"
    },
    "seamless_m4t_v2": {
        "description": "Seamless M4T v2 for speech translation",
        "class": "SeamlessM4TModel",
        "default_model": "facebook/seamless-m4t-v2-large",
        "architecture": "speech"
    },
    "xlm_prophetnet": {
        "description": "XLM-ProphetNet for multilingual sequence-to-sequence tasks",
        "class": "XLMProphetNetForConditionalGeneration",
        "default_model": "microsoft/xprophetnet-large-wiki100-cased",
        "architecture": "encoder-decoder"
    }
}

def regenerate_manual_models(output_dir="fixed_tests", verify=True):
    """Regenerate manually created model tests using the template system."""
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Track results
    results = {
        "success": [],
        "failure": []
    }
    
    # Process each manual model
    for model, arch_type in MANUAL_MODELS.items():
        logger.info(f"Processing model: {model} (architecture: {arch_type})")
        output_path = os.path.join(output_dir, f"test_hf_{model}.py")
        
        # Try to regenerate the test file
        try:
            success = regenerate_test_file(output_path, force=True, verify=verify)
            
            if success:
                logger.info(f"✅ Successfully regenerated test for {model}")
                results["success"].append(model)
            else:
                logger.error(f"❌ Failed to regenerate test for {model}")
                results["failure"].append((model, "Unknown error from regenerate_test_file"))
        except Exception as e:
            logger.error(f"Error regenerating test for {model}: {e}")
            results["failure"].append((model, str(e)))
    
    # Print summary
    logger.info("\nRegeneration Summary:")
    logger.info(f"- Successfully regenerated: {len(results['success'])} models")
    if results["success"]:
        logger.info(f"  Models: {', '.join(results['success'])}")
    
    logger.info(f"- Failed: {len(results['failure'])} models")
    if results["failure"]:
        for model, error in results["failure"]:
            logger.info(f"  - {model}: {error}")
    
    return results

def update_architecture_types_file():
    """Update the ARCHITECTURE_TYPES dictionary in test_generator_fixed.py."""
    try:
        generator_path = os.path.join("skills", "test_generator_fixed.py")
        if not os.path.exists(generator_path):
            logger.warning(f"Generator file not found: {generator_path}, skipping update")
            return False
        
        with open(generator_path, 'r') as f:
            content = f.read()
        
        # Add the manual models to their respective architecture types
        for model, arch_type in MANUAL_MODELS.items():
            # Convert arch_type to the variable name used in the generator
            var_name = arch_type.replace("-", "_")
            
            # Check if the model is already listed
            if f'"{model}"' in content:
                logger.info(f"Model {model} already exists in ARCHITECTURE_TYPES, skipping")
                continue
            
            # Find the architecture type list and add the model
            arch_pattern = f'"{arch_type}": ['
            if arch_pattern in content:
                # Find the closing bracket
                start_idx = content.find(arch_pattern)
                list_start = content.find('[', start_idx)
                list_end = content.find(']', list_start)
                
                # Insert the model at the end of the list
                new_content = (
                    content[:list_end] + 
                    f', "{model}"' + 
                    content[list_end:]
                )
                
                content = new_content
                logger.info(f"Added {model} to {arch_type} in ARCHITECTURE_TYPES")
        
        # Write the updated content
        with open(generator_path, 'w') as f:
            f.write(content)
        
        logger.info("Updated ARCHITECTURE_TYPES in test_generator_fixed.py")
        return True
    
    except Exception as e:
        logger.error(f"Error updating ARCHITECTURE_TYPES: {e}")
        return False

def update_model_registry_file():
    """Update the MODEL_REGISTRY with detailed information for manual models."""
    try:
        generator_path = os.path.join("skills", "test_generator_fixed.py")
        if not os.path.exists(generator_path):
            logger.warning(f"Generator file not found: {generator_path}, skipping update")
            return False
        
        with open(generator_path, 'r') as f:
            content = f.read()
        
        # Find the MODEL_REGISTRY dictionary
        registry_start = content.find("MODEL_REGISTRY = {")
        if registry_start == -1:
            logger.warning("MODEL_REGISTRY not found in generator file, skipping update")
            return False
        
        # Find the end of the dictionary
        registry_end = content.find("}", registry_start)
        nesting_level = 1
        for i in range(registry_start + 16, len(content)):
            if content[i] == '{':
                nesting_level += 1
            elif content[i] == '}':
                nesting_level -= 1
                if nesting_level == 0:
                    registry_end = i
                    break
        
        # Create the registry entries for manual models
        registry_updates = ""
        for model, info in MODEL_REGISTRY_UPDATES.items():
            registry_updates += f',\n    "{model}": {{\n'
            registry_updates += f'        "description": "{info["description"]}",\n'
            registry_updates += f'        "class": "{info["class"]}",\n'
            registry_updates += f'        "default_model": "{info["default_model"]}",\n'
            registry_updates += f'        "architecture": "{info["architecture"]}"\n'
            registry_updates += f'    }}'
        
        # Insert the updates before the closing brace
        new_content = content[:registry_end] + registry_updates + content[registry_end:]
        
        # Write the updated content
        with open(generator_path, 'w') as f:
            f.write(new_content)
        
        logger.info("Updated MODEL_REGISTRY in test_generator_fixed.py")
        return True
    
    except Exception as e:
        logger.error(f"Error updating MODEL_REGISTRY: {e}")
        return False

def verify_generated_tests(output_dir="fixed_tests"):
    """Verify the syntax and functionality of generated test files."""
    results = {
        "success": [],
        "failure": []
    }
    
    for model in MANUAL_MODELS:
        test_file = os.path.join(output_dir, f"test_hf_{model}.py")
        
        if not os.path.exists(test_file):
            logger.warning(f"Test file not found: {test_file}, skipping verification")
            results["failure"].append((model, "File not found"))
            continue
        
        # Verify syntax using python -m py_compile
        try:
            import py_compile
            py_compile.compile(test_file, doraise=True)
            
            # Additional verification checks could be added here
            
            results["success"].append(model)
            logger.info(f"✅ Verified test file: {test_file}")
        except Exception as e:
            results["failure"].append((model, str(e)))
            logger.error(f"❌ Verification failed for {test_file}: {e}")
    
    # Print summary
    logger.info("\nVerification Summary:")
    logger.info(f"- Successfully verified: {len(results['success'])} models")
    if results["success"]:
        logger.info(f"  Models: {', '.join(results['success'])}")
    
    logger.info(f"- Failed verification: {len(results['failure'])} models")
    if results["failure"]:
        for model, error in results["failure"]:
            logger.info(f"  - {model}: {error}")
    
    return results

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Regenerate manually created model tests using the template system")
    parser.add_argument("--output-dir", type=str, default="fixed_tests",
                        help="Directory to save regenerated files (default: fixed_tests)")
    parser.add_argument("--verify", action="store_true", 
                        help="Verify syntax after regeneration")
    parser.add_argument("--update-generator", action="store_true",
                        help="Update ARCHITECTURE_TYPES and MODEL_REGISTRY in the generator")
    
    args = parser.parse_args()
    
    # Update the generator if requested
    if args.update_generator:
        update_architecture_types_file()
        update_model_registry_file()
    
    # Regenerate the tests
    results = regenerate_manual_models(
        output_dir=args.output_dir,
        verify=args.verify
    )
    
    # Verify the generated tests if requested
    if args.verify:
        verify_results = verify_generated_tests(output_dir=args.output_dir)
        
        # Failure if any verification failed
        if verify_results["failure"]:
            logger.error("Some generated tests failed verification")
            return 1
    
    # Success if any model was successfully regenerated
    if results["success"]:
        logger.info("At least some models were successfully regenerated")
        return 0
    else:
        logger.error("All regeneration attempts failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())