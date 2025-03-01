"""
This script searches for LLaVA models on HuggingFace and tests whether they require API tokens.
It will help identify suitable models for the hf_llava and hf_llava_next test files.
"""

import sys
import os
import time
import json
import requests
import torch
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, "/home/barberb/ipfs_accelerate_py")

# Try to import necessary libraries
try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    transformers_available = True
    print("Transformers library is available.")
except ImportError:
    transformers_available = False
    print("Transformers library is not available. Will use mock testing only.")

# Function to search HuggingFace for LLaVA models
def search_huggingface_models(search_term="llava", limit=20):
    """Search for models on HuggingFace with pagination"""
    models = []
    url = f"https://huggingface.co/api/models?search={search_term}&sort=downloads&direction=-1&limit={limit}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Filter for models with 'llava' in the name
        llava_models = [model for model in data if "llava" in model["modelId"].lower()]
        
        # Extract relevant information
        for model in llava_models:
            models.append({
                "name": model["modelId"],
                "downloads": model.get("downloads", 0),
                "lastModified": model.get("lastModified", ""),
                "tags": model.get("tags", [])
            })
            
        return models
    except Exception as e:
        print(f"Error searching HuggingFace models: {e}")
        return []

# Function to check model info using API without downloading
def check_model_info(model_name):
    """Check model existence and access without downloading"""
    try:
        # Use the model info API instead of downloading
        url = f"https://huggingface.co/api/models/{model_name}"
        response = requests.get(url)
        
        if response.status_code == 200:
            # Model exists and is accessible
            data = response.json()
            
            # Check if the model is gated (requires token)
            is_gated = data.get("cardData", {}).get("gated", False)
            requires_token = is_gated or "token" in str(data).lower()
            
            # Get model details
            tags = data.get("tags", [])
            private = data.get("private", False)
            
            return {
                "name": model_name,
                "exists": True,
                "requires_token": requires_token or private,
                "private": private,
                "tags": tags,
                "card_data": data.get("cardData", {})
            }
        elif response.status_code == 401:
            # Model exists but requires authentication
            return {
                "name": model_name,
                "exists": True,
                "requires_token": True,
                "error": "Authentication required (401)"
            }
        elif response.status_code == 404:
            # Model doesn't exist
            return {
                "name": model_name,
                "exists": False,
                "requires_token": False,
                "error": "Model not found (404)"
            }
        else:
            # Other error
            return {
                "name": model_name,
                "exists": "unknown",
                "requires_token": "unknown",
                "error": f"Unexpected status code: {response.status_code}"
            }
    except Exception as e:
        return {
            "name": model_name,
            "exists": "unknown",
            "requires_token": "unknown",
            "error": str(e)
        }

# Function to test if a model requires API token
def test_model_access(model_name, resources=None, metadata=None):
    """Test if a model can be accessed without API token"""
    print(f"Checking model info for {model_name}...")
    
    # First, check model info without downloading
    model_info = check_model_info(model_name)
    
    # If model doesn't exist or we know it requires tokens, return early
    if not model_info.get("exists") or model_info.get("requires_token"):
        return model_info
    
    # If transformers isn't available, we can't properly test
    if not transformers_available:
        print(f"Skipping download test for {model_name} as transformers is not available")
        model_info["download_tested"] = False
        return model_info
    
    # If we get here, the model exists and might be accessible
    try:
        # Try to get config only first - this is lighter than processor or model
        from transformers import AutoConfig
        print(f"Testing config access to {model_name}...")
        
        start_time = time.time()
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config_time = time.time() - start_time
        
        model_info.update({
            "config_loaded": True,
            "config_time": config_time,
            "download_tested": True,
            "actual_requires_token": False
        })
        
        return model_info
        
    except Exception as e:
        error_str = str(e)
        requires_token = "401" in error_str or "authentication" in error_str.lower() or "token" in error_str.lower()
        
        model_info.update({
            "download_tested": True,
            "actual_requires_token": requires_token,
            "download_error": error_str
        })
        
        return model_info

# Main execution
if __name__ == "__main__":
    # Search for LLaVA models
    print("Searching for LLaVA models on HuggingFace...")
    models = search_huggingface_models(search_term="llava", limit=30)
    
    if not models:
        print("No models found or error occurred during search.")
        sys.exit(1)
    
    # Print top models by downloads
    print(f"\nFound {len(models)} LLaVA models. Top models by downloads:")
    for i, model in enumerate(models[:10]):
        print(f"{i+1}. {model['name']} - {model.get('downloads', 'unknown')} downloads")
    
    # Test a subset of models for token requirements
    print("\nTesting models for API token requirements...")
    results = []
    
    # Define specific models to test
    # Include smaller models, demos, and ones likely to be accessible
    specific_models = [
        # Tiny/random/demo models
        "katuni4ka/tiny-random-llava",
        "katuni4ka/tiny-random-llava-next",
        "RahulSChand/llava-tiny-random-safety",
        "RahulSChand/llava-tiny-random-1.5",
        "merlkuo/llava-tiny-for-test",
        "maywell/llava-tiny-hf-26",
        "maywell/llava-tiny-hf-25",
        "k2-enterprises/tiny-random-llava",
        "TrungTnguyen/llava-next-tiny-demo",
        
        # University/research models (potentially more accessible)
        "cvssp/LLaVA-7B",
        "cvssp/LLaVA-NeXT-Video",
        "cvssp/Uni-LLaVA-7B",
        "Edinburgh-University/hedgehog-llava-stable", 
        
        # Other potentially accessible models
        "llava-hf/bakLlava-v1-hf",
        "NousResearch/Nous-Hermes-llava-QA-7B", 
        "hysts/LLaVA-NeXT-7B",
        "LanguageBind/LLaVA-Pretrain-LLaMA-2-7B",
        "farleyknight-org-username/llava-v1.5-7b"
    ]
    
    # Add top models that aren't already in the specific list
    for model in models[:10]:
        if model["name"] not in specific_models:
            specific_models.append(model["name"])
    
    print(f"Testing {len(specific_models)} models...")
    for model_name in specific_models:
        print(f"Testing: {model_name}")
        result = test_model_access(model_name)
        results.append(result)
        # Brief pause to avoid rate limiting
        time.sleep(2)
    
    # Save results to file
    output_dir = Path("collected_results")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "llava_model_access_results.json"
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\nResults summary:")
    
    # Categorize models
    accessible_models = []
    token_models = []
    nonexistent_models = []
    unknown_models = []
    
    for result in results:
        # Check for models that exist and are accessible
        if result.get("exists") and not result.get("requires_token") and not result.get("actual_requires_token", False):
            accessible_models.append(result)
        # Check for models that require tokens
        elif result.get("requires_token") or result.get("actual_requires_token", False):
            token_models.append(result)
        # Check for models that don't exist
        elif result.get("exists") is False:
            nonexistent_models.append(result)
        # Everything else has unknown status
        else:
            unknown_models.append(result)
    
    # Print accessible models
    print(f"Models that exist and don't require tokens: {len(accessible_models)}")
    for model in accessible_models:
        print(f"  - {model['name']}")
        if model.get("config_loaded"):
            print(f"    ✓ Config loaded successfully in {model.get('config_time', 0):.2f}s")
        if model.get("tags"):
            print(f"    Tags: {', '.join(model.get('tags', []))}")
    
    # Print token-requiring models
    print(f"\nModels that require tokens: {len(token_models)}")
    for model in token_models:
        print(f"  - {model['name']}")
    
    # Print nonexistent models
    if nonexistent_models:
        print(f"\nModels that don't exist: {len(nonexistent_models)}")
        for model in nonexistent_models:
            print(f"  - {model['name']}")
    
    # Print unknown models
    if unknown_models:
        print(f"\nModels with unknown status: {len(unknown_models)}")
        for model in unknown_models:
            print(f"  - {model['name']}")
            print(f"    Error: {model.get('error', 'Unknown error')}")
    
    # Recommend models to use
    if accessible_models:
        print("\nRecommended models for testing:")
        for model in accessible_models:
            print(f"  - {model['name']}")
    else:
        print("\nNo accessible models found for testing. Consider using a custom mock model for testing.")
    
    print(f"\nDetailed results saved to {output_path}")