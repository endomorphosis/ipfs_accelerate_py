#!/usr/bin/env python
"""
ONNX File Checker

A command-line utility to check the availability of ONNX files on HuggingFace and
optionally convert PyTorch models to ONNX format when needed.

Usage:
    python check_onnx_files.py --model bert-base-uncased --onnx-path model.onnx
    python check_onnx_files.py --models bert-base-uncased t5-small --onnx-path model.onnx --convert
    python check_onnx_files.py --model-file models.txt --convert --cache-dir ./onnx_cache
"""

import os
import sys
import logging
import argparse
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("check_onnx_files")

# Import the ONNX verification utility
try:
    from onnx_verification import OnnxVerifier, verify_and_get_onnx_model
    from onnx_verification import OnnxVerificationError, OnnxConversionError
except ImportError:
    logger.error("Failed to import onnx_verification module. Make sure it's in your Python path.")
    sys.exit(1)

def load_models_from_file(file_path: str) -> List[str]:
    """Load model IDs from a file, one per line."""
    try:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except Exception as e:
        logger.error(f"Error loading models from file {file_path}: {e}")
        sys.exit(1)

def check_onnx_file(model_id: str, onnx_path: str, convert: bool = False, 
                  cache_dir: Optional[str] = None, model_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Check the availability of an ONNX file on HuggingFace.
    
    Args:
        model_id: HuggingFace model ID
        onnx_path: Path to the ONNX file within the repository
        convert: Whether to convert the model to ONNX if not found
        cache_dir: Optional cache directory for converted models
        model_type: Optional model type for conversion
        
    Returns:
        Dictionary with check results
    """
    # Create OnnxVerifier
    verifier = OnnxVerifier(cache_dir=cache_dir)
    
    # Initialize result dictionary
    result = {
        "model_id": model_id,
        "onnx_path": onnx_path,
        "timestamp": datetime.now().isoformat(),
        "success": None,
        "message": None,
        "converted": False,
        "local_path": None,
        "error": None
    }
    
    try:
        # Check if the ONNX file exists on HuggingFace
        logger.info(f"Checking ONNX file for {model_id}: {onnx_path}")
        success, message = verifier.verify_onnx_file(model_id, onnx_path)
        
        result["success"] = success
        result["message"] = message
        
        # If not found and convert flag is True, convert from PyTorch
        if not success and convert:
            logger.info(f"ONNX file not found for {model_id}, attempting conversion")
            
            # Create conversion configuration
            conversion_config = {"model_type": model_type} if model_type else None
            
            try:
                # Convert from PyTorch
                local_path = verifier.get_onnx_model(
                    model_id=model_id,
                    onnx_path=onnx_path,
                    conversion_config=conversion_config
                )
                
                result["success"] = True
                result["converted"] = True
                result["local_path"] = local_path
                result["message"] = f"Successfully converted to ONNX at {local_path}"
                
                logger.info(f"Successfully converted {model_id} to ONNX at {local_path}")
            
            except OnnxConversionError as e:
                result["success"] = False
                result["error"] = str(e)
                result["message"] = f"Conversion failed: {str(e)}"
                logger.error(f"Failed to convert {model_id} to ONNX: {e}")
        
    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        logger.error(f"Error checking ONNX file for {model_id}: {e}")
    
    return result

def main():
    """Main function for the ONNX file checker."""
    parser = argparse.ArgumentParser(description='Check ONNX file availability on HuggingFace')
    
    # Model selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model', type=str, help='HuggingFace model ID')
    group.add_argument('--models', type=str, nargs='+', help='Multiple HuggingFace model IDs')
    group.add_argument('--model-file', type=str, help='File containing model IDs (one per line)')
    
    # ONNX path
    parser.add_argument('--onnx-path', type=str, default='model.onnx', help='Path to the ONNX file within the repository')
    
    # Conversion options
    parser.add_argument('--convert', action='store_true', help='Convert PyTorch models to ONNX if not found')
    parser.add_argument('--cache-dir', type=str, help='Cache directory for converted models')
    parser.add_argument('--model-type', type=str, help='Model type for conversion (bert, t5, gpt, vit, clip, whisper, wav2vec2)')
    
    # Output options
    parser.add_argument('--output', type=str, help='Output file for results (JSON format)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get model IDs
    if args.model:
        models = [args.model]
    elif args.models:
        models = args.models
    else:
        models = load_models_from_file(args.model_file)
    
    logger.info(f"Checking ONNX files for {len(models)} models")
    
    # Check ONNX files for each model
    results = []
    for model_id in models:
        result = check_onnx_file(
            model_id=model_id,
            onnx_path=args.onnx_path,
            convert=args.convert,
            cache_dir=args.cache_dir,
            model_type=args.model_type
        )
        results.append(result)
        
        # Print result
        if result["success"]:
            if result["converted"]:
                print(f"✅ {model_id}: ONNX file converted at {result['local_path']}")
            else:
                print(f"✅ {model_id}: ONNX file available at {result['message']}")
        else:
            print(f"❌ {model_id}: {result['message']}")
    
    # Save results to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "onnx_path": args.onnx_path,
                "convert": args.convert,
                "results": results
            }, f, indent=2)
        logger.info(f"Results saved to {args.output}")
    
    # Print summary
    success_count = sum(1 for result in results if result["success"])
    converted_count = sum(1 for result in results if result["converted"])
    print(f"\nSummary: {success_count}/{len(models)} models have ONNX files available")
    if args.convert:
        print(f"Converted: {converted_count}/{len(models)} models converted from PyTorch to ONNX")
    
    # Return non-zero exit code if any model failed
    if success_count < len(models):
        sys.exit(1)

if __name__ == "__main__":
    main()