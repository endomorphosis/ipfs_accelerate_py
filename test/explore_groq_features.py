#!/usr/bin/env python
import os
import sys
import json
import time
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ipfs_accelerate_py'))

# Import the groq implementation
from api_backends import groq as groq_module

def format_response(response):
    """Format API response for readability"""
    formatted = {}}}}}}}}
    "text": response.get("text", ""),
    "model": response.get("model", "unknown"),
    "usage": response.get("usage", {}}}}}}}}}),
    "metadata": response.get("metadata", {}}}}}}}}})
    }
return json.dumps(formatted, indent=2)

def test_temperature_settings(groq_api, model="llama3-8b-8192"):
    """Test how different temperature settings affect responses"""
    print("\n=== Testing Temperature Settings ===")
    
    prompt = "List 3 possible names for a tech startup focused on sustainable energy"
    messages = []],,{}}}}}}}}"role": "user", "content": prompt}]
    ,
    # Test different temperature values
    temps = []],,0.0, 0.5, 1.0],
    results = {}}}}}}}}}
    
    for temp in temps:
        print(f"\nTemperature: {}}}}}}}}temp}")
        try:
            response = groq_api.chat(model, messages, temperature=temp)
            print(f"Response: {}}}}}}}}response[]],,'text'][]],,:100]}..."),
            results[]],,str(temp)] = response[]],,"text"],
        except Exception as e:
            print(f"Error: {}}}}}}}}str(e)}")
            results[]],,str(temp)] = f"Error: {}}}}}}}}str(e)}"
            ,
            return results

def test_system_prompts(groq_api, model="llama3-8b-8192"):
    """Test the effect of system prompts"""
    print("\n=== Testing System Prompts ===")
    
    # Setup for testing system prompts through the chat function
    user_message = "What is the future of renewable energy?"
    
    # Test different system prompts
    system_prompts = []],,
    "You are a friendly assistant",
    "You are a technical expert who provides detailed technical answers",
    "You are a concise assistant who responds in bullet points"
    ]
    
    results = {}}}}}}}}}
    
    for i, system in enumerate(system_prompts):
        print(f"\nSystem Prompt {}}}}}}}}i+1}: \"{}}}}}}}}system}\"")
        try:
            # Create messages with system prompt
            messages = []],,
            {}}}}}}}}"role": "system", "content": system},
            {}}}}}}}}"role": "user", "content": user_message}
            ]
            
            response = groq_api.chat(model, messages)
            print(f"Response: {}}}}}}}}response[]],,'text'][]],,:100]}..."),
            results[]],,f"system_{}}}}}}}}i+1}"] = {}}}}}}}}
            "system": system,
            "response": response[]],,"text"]
            }
        except Exception as e:
            print(f"Error: {}}}}}}}}str(e)}")
            results[]],,f"system_{}}}}}}}}i+1}"] = {}}}}}}}}
            "system": system,
            "error": str(e)
            }
    
            return results

def test_model_comparison(groq_api):
    """Compare responses across different models"""
    print("\n=== Testing Model Comparison ===")
    
    # Complex reasoning prompt
    prompt = "Explain the concept of quantum entanglement and its implications for quantum computing in simple terms"
    messages = []],,{}}}}}}}}"role": "user", "content": prompt}]
    ,
    # Models to test (small, medium, large)
    models = []],,"llama3-8b-8192", "gemma2-9b-it", "llama3-70b-8192"]
    
    results = {}}}}}}}}}
    
    for model in models:
        print(f"\nModel: {}}}}}}}}model}")
        try:
            start_time = time.time()
            response = groq_api.chat(model, messages)
            end_time = time.time()
            
            # Calculate metrics
            duration = end_time - start_time
            tokens = response.get("usage", {}}}}}}}}}).get("total_tokens", 0)
            
            print(f"Response time: {}}}}}}}}duration:.2f}s")
            print(f"Tokens: {}}}}}}}}tokens}")
            print(f"Response: {}}}}}}}}response[]],,'text'][]],,:100]}..."),
            
            results[]],,model] = {}}}}}}}}
            "response": response[]],,"text"],
            "time": duration,
            "tokens": tokens
            }
        except Exception as e:
            print(f"Error: {}}}}}}}}str(e)}")
            results[]],,model] = {}}}}}}}}
            "error": str(e)
            }
    
            return results

def test_streaming(groq_api, model="llama3-8b-8192"):
    """Test the streaming functionality"""
    print("\n=== Testing Streaming API ===")
    
    messages = []],,{}}}}}}}}"role": "user", "content": "Count from 1 to 5 and explain why each number is interesting"}]
    
    try:
        # Due to sseclient potentially not being installed, we'll handle the potential error
        print("Attempting to stream response...")
        chunks_received = 0
        total_length = 0
        
        # This will use the streaming API if available, or fallback to regular chat:
        for chunk in groq_api.stream_chat(model, messages):
            chunks_received += 1
            chunk_text = chunk.get("text", "")
            total_length += len(chunk_text)
            
            # Print progress every 5 chunks
            if chunks_received % 5 == 0:
                print(f"  Received {}}}}}}}}chunks_received} chunks ({}}}}}}}}total_length} chars)")
        
                print(f"\nStreaming complete: {}}}}}}}}chunks_received} chunks, {}}}}}}}}total_length} total characters")
            return {}}}}}}}}
            "success": True,
            "chunks": chunks_received,
            "total_chars": total_length
            }
    except Exception as e:
        print(f"Streaming error: {}}}}}}}}str(e)}")
            return {}}}}}}}}
            "success": False,
            "error": str(e)
            }

def main():
    # Get API key from environment
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: No GROQ_API_KEY found in environment")
        sys.exit(1)
    
    # Initialize API client
        groq_api = groq_module(resources={}}}}}}}}}, metadata={}}}}}}}}"groq_api_key": api_key})
    
    # Run tests and collect results
        results = {}}}}}}}}
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "temperature_results": test_temperature_settings(groq_api),
        "system_prompt_results": test_system_prompts(groq_api),
        "model_comparison": test_model_comparison(groq_api),
        "streaming_results": test_streaming(groq_api)
        }
    
    # Save results
        output_file = "groq_feature_exploration.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
        print(f"\nResults saved to {}}}}}}}}output_file}")
    
    # Identify documentation updates needed
        print("\n=== Documentation Update Recommendations ===")
        print("1. Add information about model capabilities:")
        print("   - Throughput and response time differences between models")
        print("   - Typical token generation speeds for planning")
    
        print("\n2. Update implementation with:")
        print("   - Support for system messages in the chat method")
        print("   - Temperature control options")
        print("   - Detailed usage statistics metrics (queue time, processing time)")
    
        print("\n3. Additional Features to Document:")
        print("   - Token prediction control (top_p, top_k parameters)")
        print("   - Logging support for tracking API usage")
        print("   - Response format options")

if __name__ == "__main__":
    main()