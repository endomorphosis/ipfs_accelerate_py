#!/usr/bin/env python3

"""
Simple test script to test a BERT model using HuggingFace transformers.
This is a simplified version for testing the basic transformers functionality.
"""

import os
import sys
import time
import logging
import argparse
import json
import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import required libraries
try:
    import torch
    import transformers
    from transformers import BertModel, BertTokenizer, pipeline

from refactored_test_suite.model_test import ModelTest
    
    HAS_REQUIREMENTS = True
except ImportError as e:
    logger.error(f"Missing dependency: {e}")
    HAS_REQUIREMENTS = False

def test_bert_model(model_id="bert-base-uncased", cpu_only=False):
    """Test a BERT model using HuggingFace transformers."""
    if not HAS_REQUIREMENTS:
        logger.error("Missing required dependencies. Please install transformers and torch.")
        return {"success": False, "error": "Missing dependencies"}
    
    results = {"model_id": model_id, "timestamp": datetime.datetime.now().isoformat()}
    
    # Set device
    if cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    results["device"] = device
    logger.info(f"Using device: {device}")
    
    try:
        # Test with pipeline API
        logger.info(f"Testing {model_id} with pipeline...")
        start_time = time.time()
        nlp = pipeline("fill-mask", model=model_id, device=0 if device == "cuda" else -1)
        load_time = time.time() - start_time
        
        test_text = "The man worked as a [MASK]."
        
        # Run inference
        inference_start = time.time()
        outputs = nlp(test_text)
        inference_time = time.time() - inference_start
        
        # Record results
        results["pipeline_success"] = True
        results["pipeline_load_time"] = load_time
        results["pipeline_inference_time"] = inference_time
        top_prediction = outputs[0]["token_str"] if isinstance(outputs, list) and len(outputs) > 0 else "N/A"
        results["top_prediction"] = top_prediction
        
        logger.info(f"Pipeline test succeeded. Top prediction: {top_prediction}")
        logger.info(f"Load time: {load_time:.2f}s, Inference time: {inference_time:.2f}s")
        
        # Test with direct model loading (optional)
        logger.info(f"Testing {model_id} with direct model loading...")
        model_start = time.time()
        tokenizer = BertTokenizer.from_pretrained(model_id)
        model = BertModel.from_pretrained(model_id)
        if device == "cuda":
            model = model.to(device)
        model_load_time = time.time() - model_start
        
        # Tokenize and run inference
        inputs = tokenizer(test_text, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        direct_start = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        direct_time = time.time() - direct_start
        
        results["direct_success"] = True
        results["direct_load_time"] = model_load_time
        results["direct_inference_time"] = direct_time
        
        logger.info(f"Direct model test succeeded")
        logger.info(f"Model load time: {model_load_time:.2f}s, Inference time: {direct_time:.2f}s")
        
        results["success"] = True
        
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        results["success"] = False
        results["error"] = str(e)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Test BERT models with HuggingFace transformers")
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="Model ID to test")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU-only mode")
    parser.add_argument("--output", type=str, help="Path to save results JSON")
    
    args = parser.parse_args()
    
    results = test_bert_model(args.model, args.cpu_only)
    
    # Print summary
    if results.get("success", False):
        print("\nTest Summary:")
        print(f"Model: {args.model}")
        print(f"Device: {results.get('device', 'unknown')}")
        print("\nPipeline API:")
        print(f"  Load time: {results.get('pipeline_load_time', 'N/A'):.2f}s")
        print(f"  Inference time: {results.get('pipeline_inference_time', 'N/A'):.2f}s")
        print(f"  Top prediction: {results.get('top_prediction', 'N/A')}")
        
        print("\nDirect API:")
        print(f"  Load time: {results.get('direct_load_time', 'N/A'):.2f}s")
        print(f"  Inference time: {results.get('direct_inference_time', 'N/A'):.2f}s")
    else:
        print("\nTest Failed:")
        print(f"Error: {results.get('error', 'Unknown error')}")
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    return 0 if results.get("success", False) else 1

if __name__ == "__main__":
    sys.exit(main())