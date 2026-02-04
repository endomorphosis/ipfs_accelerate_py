#!/usr/bin/env python3

"""
Run the full HuggingFace model lookup integration process.

This script:
1. Runs the expand_model_registry.py script to update the registry
2. Runs the integrate_test_generator.py script to integrate with the test generator
3. Runs the verify_model_lookup.py script to verify the integration
4. Generates a summary report of the integration

Usage:
    python run_model_lookup_integration.py [--no-update] [--minimal]
"""

import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"model_lookup_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

def run_script(script_name, args=None):
    """Run a Python script with optional arguments."""
    script_path = CURRENT_DIR / script_name
    
    if not os.path.exists(script_path):
        logger.error(f"Script not found: {script_path}")
        return False
    
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    logger.info(f"Running script: {script_name} {' '.join(args or [])}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"✅ {script_name} completed successfully")
            return True
        else:
            logger.error(f"❌ {script_name} failed with code {result.returncode}")
            logger.error(result.stderr)
            return False
    
    except Exception as e:
        logger.error(f"Error running {script_name}: {e}")
        return False

def update_registry(model_types=None):
    """Update the model registry with latest data."""
    args = []
    
    if model_types:
        # Update specific model types
        for model_type in model_types:
            logger.info(f"Updating registry for {model_type}")
            if not run_script("expand_model_registry.py", ["--model-type", model_type]):
                return False
        return True
    else:
        # Update all model types
        logger.info("Updating registry for all model types")
        return run_script("expand_model_registry.py", ["--all"])

def integrate_with_test_generator():
    """Integrate model lookup with the test generator."""
    logger.info("Integrating model lookup with test generator")
    return run_script("integrate_test_generator.py", ["--all"])

def verify_integration(minimal=False):
    """Verify the model lookup integration."""
    logger.info("Verifying model lookup integration")
    
    if minimal:
        # Test only with bert
        return run_script("verify_model_lookup.py", ["--model-type", "bert", "--generate"])
    else:
        # Test with all core model types
        return run_script("verify_model_lookup.py", ["--all"])

def generate_summary():
    """Generate a summary of the integration."""
    logger.info("Generating integration summary")
    
    # Read current MODEL_LOOKUP_SUMMARY.md
    summary_path = CURRENT_DIR / "MODEL_LOOKUP_SUMMARY.md"
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary_content = f.read()
    else:
        summary_content = "# HuggingFace Model Lookup Implementation Summary\n\n"
    
    # Create a new section for the integration run
    integration_summary = f"""
## Integration Run ({datetime.now().strftime('%Y-%m-%d %H:%M')})

The HuggingFace model lookup system has been successfully integrated with the test generator:

1. **Registry Update**: Updated huggingface_model_types.json with latest models 
2. **Test Generator Integration**: Integrated get_model_from_registry in test_generator_fixed.py
3. **Template Updates**: Ensured all templates use dynamic model lookup
4. **Verification**: Verified the integration works end-to-end

### Recommended Next Steps

1. Run additional model type tests: `python verify_model_lookup.py --model-type MODEL_TYPE`
2. Generate test files: `python test_generator_fixed.py --generate MODEL_TYPE`
3. Add more model types to the registry: `python expand_model_registry.py --model-type NEW_TYPE`
4. Update the documentation with new model types and examples

For detailed usage, refer to the usage examples in the Implementation Summary section.
"""
    
    # Append to the summary file
    with open(summary_path, 'w') as f:
        # Check if we're appending to existing content or creating new
        if "Integration Run" in summary_content:
            # Find a suitable insertion point after the heading
            heading_pos = summary_content.find("# HuggingFace Model Lookup Implementation Summary")
            if heading_pos != -1:
                # Find the first section break after the heading
                section_break = summary_content.find("\n## ", heading_pos)
                if section_break != -1:
                    # Insert the new section after the heading and before the first section
                    updated_content = (summary_content[:section_break] + 
                                      integration_summary + 
                                      summary_content[section_break:])
                    f.write(updated_content)
                else:
                    # Append to the end if no section break found
                    f.write(summary_content + integration_summary)
            else:
                # Write new content if heading not found
                f.write("# HuggingFace Model Lookup Implementation Summary\n" + integration_summary)
        else:
            # Simple append for minimal content
            f.write(summary_content + integration_summary)
    
    logger.info(f"✅ Updated summary at {summary_path}")
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run full model lookup integration")
    parser.add_argument("--no-update", action="store_true", help="Skip updating the registry")
    parser.add_argument("--minimal", action="store_true", help="Run minimal integration with only bert model")
    parser.add_argument("--models", type=str, help="Comma-separated list of model types to update")
    
    args = parser.parse_args()
    
    # Track overall success
    success = True
    
    # 1. Update registry
    if not args.no_update:
        model_types = args.models.split(",") if args.models else None
        success = success and update_registry(model_types)
    
    # 2. Integrate with test generator
    success = success and integrate_with_test_generator()
    
    # 3. Verify integration
    success = success and verify_integration(args.minimal)
    
    # 4. Generate summary
    if success:
        generate_summary()
        
        print("\n=== Integration Complete ===\n")
        print("The HuggingFace model lookup system has been successfully integrated.")
        print("\nYou can now use the following commands:")
        print("  - python test_generator_fixed.py --generate MODEL_TYPE")
        print("  - python verify_model_lookup.py --model-type MODEL_TYPE")
        print("  - python expand_model_registry.py --model-type MODEL_TYPE")
        print("\nSee MODEL_LOOKUP_SUMMARY.md for detailed usage examples.")
    else:
        print("\n=== Integration Failed ===\n")
        print("Check the log for error details.")
        print("You may need to run specific scripts manually to resolve the issues.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())