#!/usr/bin/env python
import os
import sys
import json
import requests
import time
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

# Get API key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("Error: GROQ_API_KEY not found in environment variables")
    sys.exit(1)

# Base URL for Groq API
GROQ_API_BASE = "https://api.groq.com"

# Add system path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Model lists from our implementation
from ipfs_accelerate_py.api_backends.groq import PRODUCTION_MODELS, PREVIEW_MODELS

def check_model(model_name):
    """Check if a model exists in the Groq API by making a minimal request"""
    url = f"{GROQ_API_BASE}/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 1  # Minimize token usage
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=5)
        
        if response.status_code == 200:
            model_info = {
                "status": "available",
                "official": model_name in PRODUCTION_MODELS or model_name in PREVIEW_MODELS
            }
            
            # Check if the response contains additional model info
            if response.json().get("model") != model_name:
                model_info["actual_model"] = response.json().get("model")
                
            return model_info
        elif response.status_code == 404:
            # Model not found
            return {"status": "not_found", "error": response.json().get("error", {}).get("message", "Model not found")}
        else:
            # Other errors
            return {"status": "error", "code": response.status_code, "message": response.text}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}

def fetch_model_list():
    """Attempt to fetch a list of models from the Groq API"""
    url = f"{GROQ_API_BASE}/openai/v1/models"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            return response.json().get("data", [])
        else:
            print(f"Error fetching models: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"Failed to fetch models: {str(e)}")
        return []

def check_model_variants(base_name, known_models):
    """Check for potential model variants based on common patterns"""
    variants = []
    
    # Common variant patterns
    patterns = [
        f"{base_name}-latest",
        f"{base_name}-beta",
        f"{base_name}-preview",
        f"{base_name}-turbo",
        f"{base_name}-mini",
        f"{base_name}-small",
        f"{base_name}-medium",
        f"{base_name}-large",
        f"{base_name}-xl",
        f"{base_name}-xxl"
    ]
    
    for pattern in patterns:
        if pattern not in known_models:
            print(f"Checking potential model variant: {pattern}")
            result = check_model(pattern)
            
            if result["status"] == "available":
                variants.append({"name": pattern, "result": result})
                print(f"  Found undocumented model: {pattern} ✓")
            else:
                print(f"  Not available: {pattern} ✗")
    
    return variants

def check_model_customizations(base_models, known_models):
    """Check for model customizations with various suffixes"""
    customizations = []
    suffixes = [
        "vision", "audio", "code", "json", "python", "instruct", "chat",
        "32k", "16k", "8k", "4k", "en", "fr", "de", "es", "multilingual"
    ]
    
    for base in base_models:
        for suffix in suffixes:
            candidate = f"{base}-{suffix}"
            if candidate not in known_models:
                print(f"Checking potential customization: {candidate}")
                result = check_model(candidate)
                
                if result["status"] == "available":
                    customizations.append({"name": candidate, "result": result})
                    print(f"  Found undocumented model: {candidate} ✓")
                else:
                    print(f"  Not available: {candidate} ✗")
                    
    return customizations

def main():
    # Combine our known models
    known_models = list(PRODUCTION_MODELS.keys()) + list(PREVIEW_MODELS.keys())
    print(f"Known models in our implementation: {len(known_models)}")
    for model in known_models:
        print(f"  - {model}")
    
    print("\nAttempting to fetch models from Groq API...")
    api_models = fetch_model_list()
    
    if api_models:
        api_model_names = [model.get("id") for model in api_models]
        print(f"Models returned by API: {len(api_model_names)}")
        for model in api_model_names:
            print(f"  - {model}")
        
        # Find models in API but not in our lists
        missing_models = [model for model in api_model_names if model not in known_models]
        if missing_models:
            print(f"\nModels missing from our implementation: {len(missing_models)}")
            for model in missing_models:
                print(f"  - {model}")
        else:
            print("\nAll API-returned models are already in our implementation.")
    else:
        print("Could not fetch models from API, checking known models only.")
    
    # Verify all our known models actually exist
    print("\nVerifying all known models...")
    invalid_models = []
    for model in known_models:
        print(f"Checking {model}...")
        result = check_model(model)
        if result["status"] != "available":
            invalid_models.append({"name": model, "result": result})
            print(f"  ✗ Model not available: {model}")
        else:
            print(f"  ✓ Available{' (unofficial)' if not result.get('official', True) else ''}")
    
    if invalid_models:
        print("\nInvalid models in our implementation:")
        for model in invalid_models:
            print(f"  - {model['name']}: {model['result']}")
    
    # Check for undocumented variants of base models
    print("\nChecking for undocumented model variants...")
    base_models = ["llama3", "llama-3", "gemma2", "mixtral", "whisper"]
    undocumented_variants = []
    
    for base in base_models:
        variants = check_model_variants(base, known_models)
        undocumented_variants.extend(variants)
    
    # Check for model customizations
    print("\nChecking for model customizations...")
    customizations = check_model_customizations(base_models, known_models)
    undocumented_variants.extend(customizations)
    
    # Save results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "known_models": known_models,
        "api_models": api_model_names if api_models else [],
        "missing_models": missing_models if api_models and missing_models else [],
        "invalid_models": [{"name": m["name"], "reason": m["result"]} for m in invalid_models],
        "undocumented_models": [{"name": m["name"], "details": m["result"]} for m in undocumented_variants]
    }
    
    with open("groq_model_check_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate summary
    print("\n=== Summary ===")
    print(f"Known models in implementation: {len(known_models)}")
    print(f"Models returned by API: {len(api_model_names) if api_models else 'Unknown'}")
    print(f"Models missing from implementation: {len(missing_models) if api_models and missing_models else 0}")
    print(f"Invalid models in implementation: {len(invalid_models)}")
    print(f"Undocumented models discovered: {len(undocumented_variants)}")
    print(f"\nDetailed results saved to: groq_model_check_results.json")

if __name__ == "__main__":
    main()