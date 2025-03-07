#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Availability Checker for Benchmarking System

This script checks if a given model is available for benchmarking
and helps diagnose issues with model definitions in the benchmark system.

Usage:
    python check_model_availability.py --model MODEL_NAME
    python check_model_availability.py --list-available
    python check_model_availability.py --verify-all

Example:
    python check_model_availability.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
    python check_model_availability.py --model laion/clap-htsat-unfused
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get the model maps from execute_comprehensive_benchmarks.py
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from execute_comprehensive_benchmarks import MODEL_MAP, SMALL_MODEL_MAP
    logger.info("Successfully imported model maps from execute_comprehensive_benchmarks.py")
except ImportError:
    logger.error("Failed to import model maps from execute_comprehensive_benchmarks.py")
    # Define fallback model maps based on what we know
    MODEL_MAP = {
        "bert": "bert-base-uncased",
        "t5": "t5-small",
        "llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "clip": "openai/clip-vit-base-patch32",
        "vit": "google/vit-base-patch16-224",
        "clap": "laion/clap-htsat-unfused",
        "wav2vec2": "facebook/wav2vec2-base",
        "whisper": "openai/whisper-tiny",
        "llava": "llava-hf/llava-1.5-7b-hf",
        "llava-next": "llava-hf/llava-v1.6-mistral-7b",
        "xclip": "microsoft/xclip-base-patch32",
        "qwen2": "Qwen/Qwen2-0.5B-Instruct",
        "detr": "facebook/detr-resnet-50"
    }
    SMALL_MODEL_MAP = {
        "bert": "prajjwal1/bert-tiny",
        "t5": "google/t5-efficient-tiny",
        "llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "clip": "openai/clip-vit-base-patch16-224",
        "vit": "facebook/deit-tiny-patch16-224",
        "clap": "laion/clap-htsat-unfused", 
        "wav2vec2": "facebook/wav2vec2-base",
        "whisper": "openai/whisper-tiny",
        "llava": "llava-hf/llava-1.5-7b-hf",
        "llava-next": "llava-hf/llava-v1.6-mistral-7b",
        "xclip": "microsoft/xclip-base-patch32",
        "qwen2": "Qwen/Qwen2-0.5B-Instruct",
        "detr": "facebook/detr-resnet-50"
    }

# Import necessary libraries for model checking
try:
    import transformers
    from transformers import AutoModel, AutoConfig
    HAS_TRANSFORMERS = True
    logger.info("Successfully imported transformers library")
except ImportError:
    logger.warning("Transformers library not available. Will perform basic checks only.")
    HAS_TRANSFORMERS = False


def get_model_key_from_path(model_path: str) -> Optional[str]:
    """Get the model key (short name) from the model path."""
    reverse_map = {v: k for k, v in MODEL_MAP.items()}
    reverse_small_map = {v: k for k, v in SMALL_MODEL_MAP.items()}
    
    return reverse_map.get(model_path) or reverse_small_map.get(model_path)


def check_model_availability(model_name_or_path: str, use_small: bool = False) -> Dict:
    """
    Check if a model is available in the system.
    
    Args:
        model_name_or_path: Name or path of the model to check
        use_small: Whether to use small model variants
        
    Returns:
        Dict with availability information
    """
    result = {
        "model_name_or_path": model_name_or_path,
        "available_in_model_map": False,
        "available_in_small_model_map": False,
        "model_key": None,
        "normalized_model_path": None,
        "transformers_available": HAS_TRANSFORMERS,
        "can_load_config": False,
        "can_load_model": False,
        "error_message": None
    }
    
    # Check if model is in MODEL_MAP or SMALL_MODEL_MAP
    model_key = get_model_key_from_path(model_name_or_path)
    if model_key:
        result["model_key"] = model_key
        if model_name_or_path in MODEL_MAP.values():
            result["available_in_model_map"] = True
        if model_name_or_path in SMALL_MODEL_MAP.values():
            result["available_in_small_model_map"] = True
    else:
        # Check if it's a short name that exists in the maps
        if model_name_or_path in MODEL_MAP:
            result["model_key"] = model_name_or_path
            result["available_in_model_map"] = True
            result["normalized_model_path"] = MODEL_MAP[model_name_or_path]
        if model_name_or_path in SMALL_MODEL_MAP:
            result["model_key"] = model_name_or_path
            result["available_in_small_model_map"] = True
            if not result["normalized_model_path"]:
                result["normalized_model_path"] = SMALL_MODEL_MAP[model_name_or_path]
    
    # If we have transformers, try to load the model config and model
    if HAS_TRANSFORMERS:
        model_path = result["normalized_model_path"] or model_name_or_path
        try:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            result["can_load_config"] = True
            
            # Don't try to load the actual model unless explicitly requested
            # as it could be large and time-consuming
            result["can_load_model"] = "Would attempt to load model (skipped for efficiency)"
        except Exception as e:
            result["error_message"] = str(e)
    
    return result


def list_available_models() -> Dict:
    """List all available models in the benchmark system."""
    result = {
        "model_map": MODEL_MAP,
        "small_model_map": SMALL_MODEL_MAP,
        "model_keys": sorted(MODEL_MAP.keys()),
        "model_paths": sorted(MODEL_MAP.values()),
        "small_model_paths": sorted(SMALL_MODEL_MAP.values())
    }
    return result


def verify_all_models() -> Dict:
    """Verify all models in the model maps."""
    results = {
        "standard_models": {},
        "small_models": {},
        "summary": {
            "total": len(MODEL_MAP),
            "standard_available": 0,
            "small_available": 0,
            "standard_issues": [],
            "small_issues": []
        }
    }
    
    # Check standard models
    for model_key, model_path in MODEL_MAP.items():
        logger.info(f"Checking standard model: {model_key} ({model_path})")
        result = check_model_availability(model_key)
        results["standard_models"][model_key] = result
        if result.get("can_load_config", False):
            results["summary"]["standard_available"] += 1
        else:
            results["summary"]["standard_issues"].append({
                "model_key": model_key,
                "model_path": model_path,
                "error": result.get("error_message")
            })
    
    # Check small models
    for model_key, model_path in SMALL_MODEL_MAP.items():
        logger.info(f"Checking small model: {model_key} ({model_path})")
        result = check_model_availability(model_key, use_small=True)
        results["small_models"][model_key] = result
        if result.get("can_load_config", False):
            results["summary"]["small_available"] += 1
        else:
            results["summary"]["small_issues"].append({
                "model_key": model_key,
                "model_path": model_path,
                "error": result.get("error_message")
            })
    
    return results


def create_model_fix_script(issues: List[Dict]) -> str:
    """Create a script to fix model issues."""
    script_lines = [
        "#!/usr/bin/env python",
        "# -*- coding: utf-8 -*-",
        "\"\"\"",
        "Model Definition Fix Script",
        "",
        "This script updates the model definitions in execute_comprehensive_benchmarks.py",
        "to fix issues identified by check_model_availability.py.",
        "\"\"\"",
        "",
        "import os",
        "import sys",
        "import re",
        "from pathlib import Path",
        "",
        "def fix_model_definitions():",
        "    \"\"\"Fix model definitions in execute_comprehensive_benchmarks.py\"\"\"",
        "    # Path to the script",
        "    script_path = Path(__file__).parent / 'execute_comprehensive_benchmarks.py'",
        "    ",
        "    if not script_path.exists():",
        "        print(f\"Error: Could not find {script_path}\")",
        "        return False",
        "    ",
        "    # Read the script content",
        "    with open(script_path, 'r') as f:",
        "        content = f.read()",
        "    ",
        "    # Create backup",
        "    backup_path = script_path.with_suffix('.py.bak')",
        "    with open(backup_path, 'w') as f:",
        "        f.write(content)",
        "    print(f\"Created backup at {backup_path}\")",
        "    ",
        "    # Apply fixes",
        "    fixed_content = content"
    ]
    
    for issue in issues:
        model_key = issue["model_key"]
        # Add logic to fix each issue
        script_lines.extend([
            f"    ",
            f"    # Fix for {model_key}",
            f"    # Original error: {issue.get('error', 'Unknown error')}",
            f"    fixed_content = re.sub(",
            f"        r'(\"{model_key}\"\\s*:\\s*)\"[^\"]*\"',",
            f"        f'\\\\1\"{issue['model_path']}\"',",
            f"        fixed_content",
            f"    )"
        ])
    
    script_lines.extend([
        "    ",
        "    # Write the fixed content",
        "    with open(script_path, 'w') as f:",
        "        f.write(fixed_content)",
        "    ",
        "    print(f\"Fixed model definitions in {script_path}\")",
        "    return True",
        "",
        "",
        "if __name__ == '__main__':",
        "    success = fix_model_definitions()",
        "    sys.exit(0 if success else 1)",
    ])
    
    return "\n".join(script_lines)


def main():
    parser = argparse.ArgumentParser(description="Check model availability for benchmarking")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="Model name or path to check")
    group.add_argument("--list-available", action="store_true", help="List all available models")
    group.add_argument("--verify-all", action="store_true", help="Verify all models in the model maps")
    parser.add_argument("--use-small", action="store_true", help="Use small model variants")
    parser.add_argument("--output", type=str, help="Output file path (JSON)")
    parser.add_argument("--create-fix-script", action="store_true", help="Create a script to fix model definition issues")
    
    args = parser.parse_args()
    
    if args.model:
        result = check_model_availability(args.model, use_small=args.use_small)
        logger.info(f"Check result for model '{args.model}':")
        logger.info(json.dumps(result, indent=2))
        
        # Print user-friendly summary
        print("\nModel Availability Check Results:")
        print(f"Model: {args.model}")
        
        if result["model_key"]:
            print(f"Model Key: {result['model_key']}")
        else:
            print("Model Key: Not found in model maps")
        
        if result["normalized_model_path"]:
            print(f"Normalized Path: {result['normalized_model_path']}")
        
        print(f"Available in MODEL_MAP: {result['available_in_model_map']}")
        print(f"Available in SMALL_MODEL_MAP: {result['available_in_small_model_map']}")
        
        if HAS_TRANSFORMERS:
            print(f"Can Load Config: {result['can_load_config']}")
            print(f"Can Load Model: {result['can_load_model']}")
            
            if result["error_message"]:
                print(f"Error: {result['error_message']}")
        else:
            print("Note: Transformers library not available for full model validation")
            
        # Provide recommendations
        print("\nRecommendations:")
        if not result["model_key"] and not result["normalized_model_path"]:
            print("- This model is not recognized in the benchmarking system.")
            print("- Add it to MODEL_MAP or SMALL_MODEL_MAP in execute_comprehensive_benchmarks.py")
        elif not result["available_in_model_map"] and not result["available_in_small_model_map"]:
            print("- The model key is recognized but the path seems incorrect.")
            print("- Update the model path in MODEL_MAP or SMALL_MODEL_MAP")
        elif result["error_message"]:
            print("- There was an error loading the model configuration.")
            print("- Check that the model path is correct and the model is available")
        else:
            print("- The model appears to be properly configured and available.")
            print("- You should be able to use it in benchmarks.")
    
    elif args.list_available:
        result = list_available_models()
        logger.info("Available models in the benchmark system:")
        logger.info(f"Model keys: {', '.join(result['model_keys'])}")
        
        # Print user-friendly output
        print("\nAvailable Models in Benchmark System:")
        print("\nStandard Models:")
        for key, path in MODEL_MAP.items():
            print(f"- {key}: {path}")
        
        print("\nSmall Models:")
        for key, path in SMALL_MODEL_MAP.items():
            print(f"- {key}: {path}")
    
    elif args.verify_all:
        results = verify_all_models()
        logger.info("Verification results summary:")
        logger.info(json.dumps(results["summary"], indent=2))
        
        # Print user-friendly summary
        print("\nModel Verification Results:")
        print(f"Total models: {results['summary']['total']}")
        print(f"Standard models available: {results['summary']['standard_available']}/{len(MODEL_MAP)}")
        print(f"Small models available: {results['summary']['small_available']}/{len(SMALL_MODEL_MAP)}")
        
        if results["summary"]["standard_issues"]:
            print("\nIssues with standard models:")
            for issue in results["summary"]["standard_issues"]:
                print(f"- {issue['model_key']} ({issue['model_path']}): {issue['error']}")
        
        if results["summary"]["small_issues"]:
            print("\nIssues with small models:")
            for issue in results["summary"]["small_issues"]:
                print(f"- {issue['model_key']} ({issue['model_path']}): {issue['error']}")
        
        # Create fix script if requested
        if args.create_fix_script:
            all_issues = results["summary"]["standard_issues"] + results["summary"]["small_issues"]
            if all_issues:
                script_content = create_model_fix_script(all_issues)
                fix_script_path = "fix_model_definitions.py"
                with open(fix_script_path, "w") as f:
                    f.write(script_content)
                os.chmod(fix_script_path, 0o755)  # Make executable
                print(f"\nCreated fix script: {fix_script_path}")
            else:
                print("\nNo issues to fix.")
    
    # Save results to file if requested
    if args.output:
        output_data = None
        if args.model:
            output_data = check_model_availability(args.model, use_small=args.use_small)
        elif args.list_available:
            output_data = list_available_models()
        elif args.verify_all:
            output_data = verify_all_models()
        
        if output_data:
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()