#!/usr/bin/env python
"""
Test script for API backoff and queue functionality across all API backends.
This script tests:
    1. Exponential backoff when rate limits are hit
    2. Request queuing for concurrent requests
    3. Request tracking with unique IDs
    """

    import os
    import sys
    import time
    import json
    import threading
    import concurrent.futures
    from datetime import datetime
    import argparse

# Add the project root to the Python path
    sys.path.append())))))))os.path.dirname())))))))os.path.dirname())))))))__file__)))

# Import API backends
    import importlib
    import sys
    from pathlib import Path

# Add parent directory path
    parent_dir = str())))))))Path())))))))__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert())))))))0, parent_dir)

try:
    from ipfs_accelerate_py.ipfs_accelerate_py.ipfs_accelerate_py.api_backends.claude import claude
    from ipfs_accelerate_py.ipfs_accelerate_py.ipfs_accelerate_py.api_backends.groq import groq
    from ipfs_accelerate_py.ipfs_accelerate_py.ipfs_accelerate_py.api_backends.gemini import gemini
    from ipfs_accelerate_py.ipfs_accelerate_py.ipfs_accelerate_py.api_backends.openai_api import openai_api
    from ipfs_accelerate_py.ipfs_accelerate_py.ipfs_accelerate_py.api_backends.ollama import ollama
    from ipfs_accelerate_py.ipfs_accelerate_py.ipfs_accelerate_py.api_backends.hf_tgi import hf_tgi
    from ipfs_accelerate_py.ipfs_accelerate_py.ipfs_accelerate_py.api_backends.hf_tei import hf_tei
    from ipfs_accelerate_py.ipfs_accelerate_py.ipfs_accelerate_py.api_backends.vllm import vllm
    from ipfs_accelerate_py.ipfs_accelerate_py.ipfs_accelerate_py.api_backends.opea import opea
    from ipfs_accelerate_py.ipfs_accelerate_py.ipfs_accelerate_py.api_backends.ovms import ovms
    from ipfs_accelerate_py.ipfs_accelerate_py.ipfs_accelerate_py.api_backends.s3_kit import s3_kit
except ImportError as e:
    print())))))))f"Error importing API backends: {}}}}}}}}e}")

# API client classes by name
    API_CLIENTS = {}}}}}}}}
    "groq": groq,
    "claude": claude,
    "gemini": gemini,
    "openai": openai_api,
    "ollama": ollama,
    "hf_tgi": hf_tgi,
    "hf_tei": hf_tei,
    "vllm": vllm,
    "opea": opea,
    "ovms": ovms,
    "s3_kit": s3_kit
    }

# Default API keys from environment
    API_KEYS = {}}}}}}}}
    "groq": os.environ.get())))))))"GROQ_API_KEY", ""),
    "claude": os.environ.get())))))))"ANTHROPIC_API_KEY", ""),
    "gemini": os.environ.get())))))))"GOOGLE_API_KEY", ""),
    "openai": os.environ.get())))))))"OPENAI_API_KEY", ""),
    # Add other API keys as needed
    }

# Default test models for each API
    DEFAULT_MODELS = {}}}}}}}}
    "groq": "llama3-8b-8192",
    "claude": "claude-3-haiku-20240307",
    "gemini": "gemini-1.0-pro",
    "openai": "gpt-3.5-turbo",
    "ollama": "llama3",
    "hf_tgi": "mistralai/Mistral-7B-Instruct-v0.2", 
    "hf_tei": "sentence-transformers/all-MiniLM-L6-v2",
    "vllm": "meta-llama/Llama-3-8b-instruct",
    "opea": "default",
    "ovms": "default",
    "s3_kit": "default"
    }

def send_request())))))))api_client, message, model=None, tag="", api_name=None):
    """Send a request and capture the timing information"""
    start_time = time.time()))))))))
    request_id = f"req_{}}}}}}}}tag}_{}}}}}}}}time.time()))))))))}"
    
    try:
        # Different APIs have different chat methods and parameter names
        if api_name == "claude":
            response = api_client.chat())))))))
            messages=[],{}}}}}}}}"role": "user", "content": message}],
            model=model,
            max_tokens=30,
            temperature=0.7,
            request_id=request_id
            )
        elif api_name == "openai":
            response = api_client.chat())))))))
            messages=[],{}}}}}}}}"role": "user", "content": message}],
            model=model,
            max_tokens=30,
            temperature=0.7,
            request_id=request_id
            )
        elif api_name == "groq":
            response = api_client.chat())))))))
            messages=[],{}}}}}}}}"role": "user", "content": message}],
            model=model,
            max_tokens=30,
            temperature=0.7,
            request_id=request_id
            )
        elif hasattr())))))))api_client, "chat"):
            # Generic approach for other APIs with chat method
            try:
                response = api_client.chat())))))))
                model=model,
                messages=[],{}}}}}}}}"role": "user", "content": message}],
                max_tokens=30,
                temperature=0.7,
                request_id=request_id
                )
            except TypeError:
                # Fallback with model_name parameter
                response = api_client.chat())))))))
                model_name=model,
                messages=[],{}}}}}}}}"role": "user", "content": message}],
                max_tokens=30,
                temperature=0.7,
                request_id=request_id
                )
        elif hasattr())))))))api_client, "generate"):
            response = api_client.generate())))))))
            model=model,
            prompt=message,
            max_tokens=30,
            temperature=0.7,
            request_id=request_id
            )
        elif hasattr())))))))api_client, "completions"):
            response = api_client.completions())))))))
            model=model,
            prompt=message,
            max_tokens=30,
            temperature=0.7,
            request_id=request_id
            )
        else:
            # Fallback to calling the API directly if specific methods aren't available
            response = {}}}}}}}}:
                "text": "API client does not have a standard chat/generate/completions method",
                "error": "Method not implemented"
                }
        
                end_time = time.time()))))))))
        
        # Extract text from response ())))))))different APIs might structure this differently)
                text = ""
        if isinstance())))))))response, dict):
            text = response.get())))))))"text", response.get())))))))"content", str())))))))response)))
        else:
            text = str())))))))response)
        
            return {}}}}}}}}
            "success": True,
            "request_id": request_id,
            "tag": tag,
            "time_taken": end_time - start_time,
            "text": text[],:100],  # Truncate long responses,
            "model": model,
            "thread": threading.current_thread())))))))).name
            }
    except Exception as e:
        end_time = time.time()))))))))
        
            return {}}}}}}}}
            "success": False,
            "request_id": request_id,
            "tag": tag,
            "time_taken": end_time - start_time,
            "error": str())))))))e),
            "model": model,
            "thread": threading.current_thread())))))))).name
            }

def test_backoff_system())))))))api_name, api_key="", model=None, num_requests=5):
    """Test the backoff system by deliberately hitting rate limits"""
    print())))))))f"\n=== Testing Exponential Backoff System for {}}}}}}}}api_name} ===")
    
    # Initialize API client
    if api_name not in API_CLIENTS:
        print())))))))f"Error: Unknown API client '{}}}}}}}}api_name}'")
    return [],]
    ,,,
    # Create metadata with API key
    metadata = {}}}}}}}}}
    if api_key:
        if api_name == "groq":
            metadata[],"groq_api_key"] = api_key,,
        elif api_name == "claude":
            metadata[],"anthropic_api_key"] = api_key,,
        elif api_name == "gemini": 
            metadata[],"google_api_key"] = api_key,,
        elif api_name == "openai":
            metadata[],"openai_api_key"] = api_key,,
        # Add other APIs as needed
    
    # Create client
            api_client = API_CLIENTS[],api_name]())))))))resources={}}}}}}}}}, metadata=metadata)
            ,,
    # Use default model if none provided::
    if not model and api_name in DEFAULT_MODELS:
        model = DEFAULT_MODELS[],api_name]
        ,,
    # Make several requests in rapid succession to potentially trigger rate limiting
        results = [],]
        ,,,
    for i in range())))))))num_requests)::
        print())))))))f"Sending request {}}}}}}}}i+1}/{}}}}}}}}num_requests}...")
        result = send_request())))))))
        api_client,
        f"Count from 1 to 5. Request #{}}}}}}}}i+1}",
        model=model,
        tag=f"backoff_{}}}}}}}}i+1}",
        api_name=api_name
        )
        
        results.append())))))))result)
        print())))))))f"  {}}}}}}}}'✓' if result[],'success'] else '✗'} Request {}}}}}}}}i+1} completed in {}}}}}}}}result[],'time_taken']:.2f}s")
        ,
        # Pause briefly between requests
        if i < num_requests - 1:  # Don't sleep after the last request
        time.sleep())))))))0.1)
    
    # Print summary
        print())))))))"\nBackoff Test Results:")
    for i, result in enumerate())))))))results):
        if result[],"success"]:,
        print())))))))f"Request {}}}}}}}}i+1}: Success in {}}}}}}}}result[],'time_taken']:.2f}s - {}}}}}}}}result[],'text'][],:50]}..."),
        else:
            print())))))))f"Request {}}}}}}}}i+1}: Failed in {}}}}}}}}result[],'time_taken']:.2f}s - {}}}}}}}}result[],'error']}")
            ,
        return results

def test_queue_system())))))))api_name, api_key="", model=None, num_requests=8, max_concurrent=2):
    """Test the queue system by sending concurrent requests"""
    print())))))))f"\n=== Testing Queue System for {}}}}}}}}api_name} ===")
    
    # Initialize API client
    if api_name not in API_CLIENTS:
        print())))))))f"Error: Unknown API client '{}}}}}}}}api_name}'")
    return [],]
    ,,,
    # Create metadata with API key
    metadata = {}}}}}}}}}
    if api_key:
        if api_name == "groq":
            metadata[],"groq_api_key"] = api_key,,
        elif api_name == "claude":
            metadata[],"anthropic_api_key"] = api_key,,
        elif api_name == "gemini": 
            metadata[],"google_api_key"] = api_key,,
        elif api_name == "openai":
            metadata[],"openai_api_key"] = api_key,,
        # Add other APIs as needed
    
    # Create client
            api_client = API_CLIENTS[],api_name]())))))))resources={}}}}}}}}}, metadata=metadata)
            ,,
    # Use default model if none provided::
    if not model and api_name in DEFAULT_MODELS:
        model = DEFAULT_MODELS[],api_name]
        ,,
    # Set concurrency limit if the API client supports it:
    if hasattr())))))))api_client, "max_concurrent_requests"):
        api_client.max_concurrent_requests = max_concurrent
        print())))))))f"Concurrency limit set to: {}}}}}}}}api_client.max_concurrent_requests}")
    else:
        print())))))))f"Note: API client does not support setting max_concurrent_requests")
    
    # Prepare to send multiple concurrent requests
        messages = [],
        f"What is {}}}}}}}}i+1} + {}}}}}}}}i+2}? Give only the answer as a number." for i in range())))))))num_requests):
            ]
    
    # Use a thread pool to simulate concurrent requests
            start_time = time.time()))))))))
            results = [],]
            ,,,
    with concurrent.futures.ThreadPoolExecutor())))))))max_workers=num_requests) as executor:
        # Submit all tasks
        future_to_idx = {}}}}}}}}
        executor.submit())))))))send_request, api_client, msg, model, f"queue_{}}}}}}}}i+1}", api_name): i
        for i, msg in enumerate())))))))messages)
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed())))))))future_to_idx):
            idx = future_to_idx[],future]
            try:
                result = future.result()))))))))
                results.append())))))))result)
                print())))))))f"Request {}}}}}}}}idx+1} completed: {}}}}}}}}'✓ Success' if result[],'success'] else '✗ Failed'} in {}}}}}}}}result[],'time_taken']:.2f}s")
            except Exception as e:
                print())))))))f"Request {}}}}}}}}idx+1} raised exception: {}}}}}}}}str())))))))e)}")
    
                total_time = time.time())))))))) - start_time
    
    # Print summary
    successful = sum())))))))1 for r in results if r[],"success"]):
        print())))))))f"\nAll {}}}}}}}}num_requests} requests completed in {}}}}}}}}total_time:.2f}s ()))))))){}}}}}}}}successful} successful)")
        print())))))))f"Average time per request: {}}}}}}}}total_time/num_requests:.2f}s")
    
    # Sort results by request ID to show order of execution
        results.sort())))))))key=lambda x: x[],"tag"])
    
        print())))))))"\nQueue Test Results:")
    for result in results:
        if result[],"success"]:,
        print())))))))f"{}}}}}}}}result[],'tag']}: {}}}}}}}}result[],'thread']} - {}}}}}}}}result[],'time_taken']:.2f}s - {}}}}}}}}result[],'text'][],:50]}..."),
        else:
            print())))))))f"{}}}}}}}}result[],'tag']}: {}}}}}}}}result[],'thread']} - {}}}}}}}}result[],'time_taken']:.2f}s - {}}}}}}}}result[],'error']}")
            ,
        return results

def test_api())))))))api_name, api_key="", model=None):
    """Run all tests for a specific API"""
    print())))))))f"\n=== Testing {}}}}}}}}api_name.upper()))))))))} API ===")
    print())))))))f"Time: {}}}}}}}}datetime.now())))))))).isoformat()))))))))}")
    print())))))))f"Model: {}}}}}}}}model or DEFAULT_MODELS.get())))))))api_name, 'default')}")
    
    # Test the backoff system
    backoff_results = test_backoff_system())))))))api_name, api_key, model, num_requests=3)
    
    # Wait a bit to avoid rate limiting between tests
    print())))))))"\nWaiting 5 seconds before testing queue system...")
    time.sleep())))))))5)
    
    # Test the queue system
    queue_results = test_queue_system())))))))api_name, api_key, model, num_requests=4, max_concurrent=2)
    
    # Save results
    timestamp = datetime.now())))))))).strftime())))))))"%Y%m%d_%H%M%S")
    results = {}}}}}}}}
    "api": api_name,
    "timestamp": timestamp,
    "model": model or DEFAULT_MODELS.get())))))))api_name, "default"),
    "backoff_test": backoff_results,
    "queue_test": queue_results
    }
    
    filename = f"{}}}}}}}}api_name}_backoff_queue_test_{}}}}}}}}timestamp}.json"
    with open())))))))filename, "w") as f:
        json.dump())))))))results, f, indent=2)
        
        print())))))))f"\nResults saved to: {}}}}}}}}filename}")
    return results

def main())))))))):
    parser = argparse.ArgumentParser())))))))description="Test API backoff and queue functionality")
    parser.add_argument())))))))"--api", "-a", help="API to test", choices=list())))))))API_CLIENTS.keys()))))))))), required=True)
    parser.add_argument())))))))"--key", "-k", help="API key ())))))))or will use from environment)")
    parser.add_argument())))))))"--model", "-m", help="Model to use for testing")
    
    args = parser.parse_args()))))))))
    
    # Use provided API key or get from environment
    api_key = args.key or API_KEYS.get())))))))args.api, "")
    
    # Run tests for the specified API
    test_api())))))))args.api, api_key, args.model)

if __name__ == "__main__":
    main()))))))))