#!/usr/bin/env python
"""
Integration test for API key multiplexing in the IPFS Accelerate framework.
This test should be run after all other API tests to verify that
multiple API keys can be used simultaneously with separate queues.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import the multiplexing example
from api_key_multiplexing_example import test_in_ipfs_accelerate_environment

class TestApiMultiplexing:
    def __init__(self, resources=None, metadata=None):
        """
        Initialize the test with optional resources and metadata.
        These will be passed in when run from test_ipfs_accelerate.py
        """
        self.resources = resources or {}
        self.metadata = metadata or {}
        
        # Ensure API keys are loaded from environment if not in metadata
        if "openai_api_key" not in self.metadata:
            # Load .env file if python-dotenv is available
            try:
                from dotenv import load_dotenv
                load_dotenv()
                
                # Add API keys from environment to metadata
                self.metadata.update({
                    # OpenAI keys
                    "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
                    "openai_api_key_1": os.environ.get("OPENAI_API_KEY_1", ""),
                    "openai_api_key_2": os.environ.get("OPENAI_API_KEY_2", ""),
                    "openai_api_key_3": os.environ.get("OPENAI_API_KEY_3", ""),
                    
                    # Groq keys
                    "groq_api_key": os.environ.get("GROQ_API_KEY", ""),
                    "groq_api_key_1": os.environ.get("GROQ_API_KEY_1", ""),
                    "groq_api_key_2": os.environ.get("GROQ_API_KEY_2", ""),
                    
                    # Claude keys
                    "claude_api_key": os.environ.get("CLAUDE_API_KEY", ""),
                    "anthropic_api_key_1": os.environ.get("ANTHROPIC_API_KEY_1", ""),
                    "anthropic_api_key_2": os.environ.get("ANTHROPIC_API_KEY_2", ""),
                    
                    # Gemini keys
                    "gemini_api_key": os.environ.get("GEMINI_API_KEY", ""),
                    "gemini_api_key_1": os.environ.get("GEMINI_API_KEY_1", ""),
                    "gemini_api_key_2": os.environ.get("GEMINI_API_KEY_2", "")
                })
            except ImportError:
                print("python-dotenv not available, using environment variables directly")
    
    def test(self):
        """Run the multiplexing test"""
        print("Starting API Key Multiplexing test...")
        start_time = time.time()
        
        # Run the test
        results = test_in_ipfs_accelerate_environment(
            resources=self.resources,
            metadata=self.metadata
        )
        
        # Add timing information
        elapsed = time.time() - start_time
        results["api_key_multiplexing"]["elapsed_time"] = f"{elapsed:.2f} seconds"
        
        # Print summary
        key_counts = results["api_key_multiplexing"].get("key_counts", {})
        total_keys = key_counts.get("total", 0)
        
        print(f"API Key Multiplexing test completed in {elapsed:.2f} seconds")
        print(f"Found {total_keys} API keys for testing")
        print(f"Status: {results['api_key_multiplexing']['status']}")
        
        # If we have usage stats, print them
        if "results" in results["api_key_multiplexing"] and "usage_stats" in results["api_key_multiplexing"]["results"]:
            usage_stats = results["api_key_multiplexing"]["results"]["usage_stats"]
            
            for provider, keys in usage_stats.items():
                if keys:
                    print(f"\n{provider.upper()} API Keys:")
                    for key_name, data in keys.items():
                        print(f"  {key_name}: {data['usage']} requests, {data['queue_size']} queued")
        
        return results
    
    def __test__(self):
        """Run test and save results to JSON file"""
        test_results = {}
        try:
            test_results = self.test()
        except Exception as e:
            import traceback
            test_results = {
                "api_key_multiplexing": {
                    "status": "Error",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            }
        
        # Create a timestamp for the results file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results to file
        results_file = f"api_multiplexing_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"Results saved to {results_file}")
        return test_results

if __name__ == "__main__":
    # Run the test directly
    test = TestApiMultiplexing()
    test.__test__()