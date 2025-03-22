#!/usr/bin/env python3
"""
Script to generate sample mock detection test results for visualization.
This produces example JSON files that can be used to test the visualization system.
"""

import os
import json
import random
import datetime
import argparse
from typing import List, Dict, Any

# Constants
RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")
DEFAULT_NUM_RESULTS = 5
DEFAULT_MODELS_PER_RESULT = 10

# Ensure results directory exists
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

def generate_sample_result(num_models: int = 10) -> Dict[str, Any]:
    """
    Generate a sample mock detection test result.
    
    Args:
        num_models: Number of models to include in the result
        
    Returns:
        Dictionary with sample test result data
    """
    # Commonly used model families for realistic test data
    model_families = [
        "bert", "gpt2", "t5", "vit", "clip", "blip", 
        "roberta", "llama", "whisper", "wav2vec2"
    ]
    
    # Generate specific model IDs based on families
    model_ids = []
    for family in model_families:
        if family == "bert":
            model_ids.extend([
                "bert-base-uncased", 
                "bert-large-uncased",
                "bert-base-cased", 
                "bert-base-chinese"
            ])
        elif family == "gpt2":
            model_ids.extend([
                "gpt2", 
                "gpt2-medium", 
                "gpt2-large", 
                "gpt2-xl"
            ])
        elif family == "t5":
            model_ids.extend([
                "t5-small", 
                "t5-base", 
                "t5-large"
            ])
        elif family == "vit":
            model_ids.extend([
                "vit-base-patch16-224", 
                "vit-large-patch16-224"
            ])
        elif family == "clip":
            model_ids.extend([
                "openai/clip-vit-base-patch32", 
                "openai/clip-vit-large-patch14"
            ])
        elif family == "blip":
            model_ids.extend([
                "Salesforce/blip-image-captioning-base", 
                "Salesforce/blip-image-captioning-large"
            ])
        elif family == "roberta":
            model_ids.extend([
                "roberta-base", 
                "roberta-large"
            ])
        elif family == "llama":
            model_ids.extend([
                "meta-llama/Llama-2-7b-hf", 
                "meta-llama/Llama-2-13b-hf"
            ])
        elif family == "whisper":
            model_ids.extend([
                "openai/whisper-tiny", 
                "openai/whisper-base", 
                "openai/whisper-small"
            ])
        elif family == "wav2vec2":
            model_ids.extend([
                "facebook/wav2vec2-base-960h", 
                "facebook/wav2vec2-large-960h"
            ])
    
    # Select random subset if we have too many
    if len(model_ids) > num_models:
        model_ids = random.sample(model_ids, num_models)
    
    # Add prefixes like "hf_" to some models
    for i in range(len(model_ids)):
        if random.random() < 0.3:  # 30% chance to add prefix
            model_ids[i] = f"hf_{model_ids[i]}"
    
    # Generate test results for each model
    test_results = []
    for model_id in model_ids:
        # Decide if this test uses mocks or real dependencies
        using_mocks = random.random() < 0.5  # 50% chance of using mocks
        
        # Success rate depends on whether using mocks
        success_probability = 0.95 if using_mocks else 0.85  # Mocks slightly more reliable
        success = random.random() < success_probability
        
        # Duration also depends on whether using mocks
        duration_base = 200 if using_mocks else 1200  # Real dependencies take longer
        duration_variation = duration_base * 0.3  # 30% variation
        duration_ms = int(random.uniform(
            duration_base - duration_variation,
            duration_base + duration_variation
        ))
        
        # Generate error message if test failed
        error = None
        if not success:
            error_types = [
                "ImportError: No module named 'transformers'",
                "RuntimeError: CUDA out of memory",
                "ValueError: Model not found",
                "ConnectionError: Failed to download model",
                "TypeError: Expected tensor for argument 'input'"
            ]
            error = random.choice(error_types)
        
        # Create test result entry
        result = {
            "test_name": f"test_{model_id.split('/')[-1]}",
            "model_id": model_id,
            "success": success,
            "using_mocks": using_mocks,
            "duration_ms": duration_ms,
            "error": error
        }
        
        # Add dependency status
        if using_mocks:
            # At least one key dependency must be missing
            has_transformers = random.random() < 0.3
            has_torch = not has_transformers or random.random() < 0.3
            
            # Support libraries might be present even with mocks
            has_tokenizers = random.random() < 0.5
            has_sentencepiece = random.random() < 0.5
        else:
            # All dependencies must be present for real inference
            has_transformers = True
            has_torch = True
            has_tokenizers = True
            has_sentencepiece = True
        
        result["dependencies"] = {
            "transformers": has_transformers,
            "torch": has_torch,
            "tokenizers": has_tokenizers,
            "sentencepiece": has_sentencepiece
        }
        
        test_results.append(result)
    
    # Create complete result
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return {
        "timestamp": timestamp,
        "environment": {
            "python_version": f"3.{random.randint(8, 11)}.{random.randint(0, 9)}",
            "platform": random.choice(["linux", "darwin", "win32"]),
            "cpu_only": random.random() < 0.7  # 70% chance of CPU only
        },
        "test_results": test_results
    }

def generate_multiple_results(num_results: int = 5, 
                             models_per_result: int = 10) -> List[Dict[str, Any]]:
    """
    Generate multiple sample test results.
    
    Args:
        num_results: Number of result files to generate
        models_per_result: Average number of models per result
        
    Returns:
        List of generated result dictionaries
    """
    results = []
    
    for i in range(num_results):
        # Vary the number of models slightly for each result
        num_models = max(1, int(models_per_result * random.uniform(0.7, 1.3)))
        result = generate_sample_result(num_models)
        results.append(result)
    
    return results

def save_results(results: List[Dict[str, Any]]) -> List[str]:
    """
    Save generated results to JSON files.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        List of saved file paths
    """
    saved_files = []
    
    for i, result in enumerate(results):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mock_detection_{timestamp}_{i}.json"
        filepath = os.path.join(RESULT_DIR, filename)
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
        
        saved_files.append(filepath)
    
    return saved_files

def main():
    """Main function to parse arguments and generate sample results."""
    parser = argparse.ArgumentParser(description="Generate sample mock detection test results")
    parser.add_argument('--num-results', type=int, default=DEFAULT_NUM_RESULTS,
                      help=f"Number of result files to generate (default: {DEFAULT_NUM_RESULTS})")
    parser.add_argument('--models-per-result', type=int, default=DEFAULT_MODELS_PER_RESULT,
                      help=f"Average number of models per result (default: {DEFAULT_MODELS_PER_RESULT})")
    
    args = parser.parse_args()
    
    print(f"Generating {args.num_results} sample result files...")
    results = generate_multiple_results(args.num_results, args.models_per_result)
    
    saved_files = save_results(results)
    print(f"Generated {len(saved_files)} sample result files:")
    for filepath in saved_files:
        print(f"  - {os.path.basename(filepath)}")
    
    print(f"\nResults saved to: {RESULT_DIR}")
    print("\nYou can now run the visualization tool:")
    print("python mock_detection_visualization.py")

if __name__ == "__main__":
    main()