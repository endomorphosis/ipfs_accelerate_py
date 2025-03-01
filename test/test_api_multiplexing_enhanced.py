#!/usr/bin/env python
"""
Enhanced test for API key multiplexing in IPFS Accelerate.
Tests multiplexing with different key selection strategies and load scenarios.

This test:
1. Tests standard round-robin API key selection
2. Tests least-loaded selection strategy
3. Tests high concurrency with automatic fallback
4. Compares performance between multiplexed and non-multiplexed APIs
"""

import os
import sys
import json
import time
import threading
import concurrent.futures
import random
from pathlib import Path
from datetime import datetime
import argparse

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import the multiplexing example
from api_key_multiplexing_example import (
    ApiKeyMultiplexer, 
    example_openai_request,
    example_groq_request
)

class EnhancedApiMultiplexingTest:
    def __init__(self, resources=None, metadata=None, verbose=True):
        """
        Initialize the enhanced multiplexing test.
        
        Args:
            resources: Optional resources dict from ipfs_accelerate test environment
            metadata: Optional metadata dict containing API keys
            verbose: Whether to print detailed logs during testing
        """
        self.resources = resources or {}
        self.metadata = metadata or {}
        self.verbose = verbose
        self.results = {}
        self.multiplexer = None
        
        # Load API keys from environment if not in metadata
        self._load_api_keys()
        
        # Initialize test prompts
        self.test_prompts = [
            "What is the capital of France?",
            "Count from 1 to 5",
            "Write a haiku about programming",
            "What is 2+2?",
            "Name three planets in our solar system",
            "Explain what an API is in one sentence",
            "What day comes after Tuesday?",
            "Tell me a fun fact about elephants",
            "What is the main ingredient in bread?",
            "How many legs does a spider have?"
        ]
    
    def _load_api_keys(self):
        """Load API keys from environment if not in metadata"""
        if "openai_api_key" not in self.metadata:
            # Try to load from dotenv first
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                if self.verbose:
                    print("Note: python-dotenv not available")
            
            # Add keys from environment
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
                "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
                "anthropic_api_key_1": os.environ.get("ANTHROPIC_API_KEY_1", ""),
                "anthropic_api_key_2": os.environ.get("ANTHROPIC_API_KEY_2", ""),
                
                # Gemini keys
                "gemini_api_key": os.environ.get("GOOGLE_API_KEY", ""),
                "gemini_api_key_1": os.environ.get("GEMINI_API_KEY_1", ""),
                "gemini_api_key_2": os.environ.get("GEMINI_API_KEY_2", "")
            })
    
    def _setup_multiplexer(self):
        """Initialize the API key multiplexer with available keys"""
        self.multiplexer = ApiKeyMultiplexer()
        
        # Add all available keys
        keys_added = 0
        
        # Add OpenAI keys
        for key_name, env_var in [
            ("openai_main", "openai_api_key"),
            ("openai_1", "openai_api_key_1"),
            ("openai_2", "openai_api_key_2"),
            ("openai_3", "openai_api_key_3")
        ]:
            if env_var in self.metadata and self.metadata[env_var]:
                self.multiplexer.add_openai_key(key_name, self.metadata[env_var])
                keys_added += 1
        
        # Add Groq keys
        for key_name, env_var in [
            ("groq_main", "groq_api_key"),
            ("groq_1", "groq_api_key_1"),
            ("groq_2", "groq_api_key_2")
        ]:
            if env_var in self.metadata and self.metadata[env_var]:
                self.multiplexer.add_groq_key(key_name, self.metadata[env_var])
                keys_added += 1
        
        # Add Claude keys
        for key_name, env_var in [
            ("claude_main", "claude_api_key"),
            ("claude_main_alt", "anthropic_api_key"),
            ("claude_1", "anthropic_api_key_1"),
            ("claude_2", "anthropic_api_key_2")
        ]:
            if env_var in self.metadata and self.metadata[env_var]:
                self.multiplexer.add_claude_key(key_name, self.metadata[env_var])
                keys_added += 1
        
        # Add Gemini keys
        for key_name, env_var in [
            ("gemini_main", "gemini_api_key"),
            ("gemini_1", "gemini_api_key_1"),
            ("gemini_2", "gemini_api_key_2")
        ]:
            if env_var in self.metadata and self.metadata[env_var]:
                self.multiplexer.add_gemini_key(key_name, self.metadata[env_var])
                keys_added += 1
        
        return keys_added
    
    def test_round_robin_selection(self, num_requests=10):
        """Test round-robin key selection strategy"""
        if self.verbose:
            print("\n=== Testing Round-Robin API Key Selection ===")
        
        results = {
            "strategy": "round-robin",
            "num_requests": num_requests,
            "start_time": datetime.now().isoformat(),
            "requests": [],
            "key_distribution": {},
            "success_rate": 0
        }
        
        # Get available API types
        available_apis = []
        if self.multiplexer.openai_clients:
            available_apis.append("openai")
        if self.multiplexer.groq_clients:
            available_apis.append("groq")
        
        if not available_apis:
            results["status"] = "Skipped - No compatible APIs available"
            return results
        
        # Choose a random API type to test
        api_type = random.choice(available_apis)
        results["api_type"] = api_type
        
        # Get the client function based on API type
        if api_type == "openai":
            request_fn = lambda prompt: example_openai_request(self.multiplexer, key_name=None, prompt=prompt)
            key_counts = {key: 0 for key in self.multiplexer.openai_clients.keys()}
        elif api_type == "groq":
            request_fn = lambda prompt: example_groq_request(self.multiplexer, key_name=None, prompt=prompt)
            key_counts = {key: 0 for key in self.multiplexer.groq_clients.keys()}
        
        # Run the requests
        for i in range(num_requests):
            prompt = random.choice(self.test_prompts)
            
            try:
                if self.verbose:
                    print(f"Request {i+1}/{num_requests}...")
                
                result = request_fn(prompt)
                
                # Update key distribution if successful
                if result["success"]:
                    # Get the key that was used (this is a simplification, in a real
                    # implementation we would need to track which key was used)
                    stats = self.multiplexer.get_usage_stats()
                    for key, data in stats[api_type].items():
                        if data["usage"] > key_counts.get(key, 0):
                            key_counts[key] = data["usage"]
                            used_key = key
                            break
                    else:
                        used_key = "unknown"
                    
                    result["key_used"] = used_key
                
                results["requests"].append(result)
            
            except Exception as e:
                if self.verbose:
                    print(f"Error in request {i+1}: {str(e)}")
                results["requests"].append({
                    "success": False,
                    "error": str(e)
                })
        
        # Calculate success rate
        successful = sum(1 for r in results["requests"] if r.get("success", False))
        results["success_rate"] = successful / num_requests if num_requests > 0 else 0
        results["successful_requests"] = successful
        
        # Get key distribution
        stats = self.multiplexer.get_usage_stats()
        results["key_distribution"] = {
            key: data["usage"] for key, data in stats[api_type].items()
        }
        
        if self.verbose:
            print(f"Round-Robin Test Results: {successful}/{num_requests} successful")
            print(f"Key distribution: {results['key_distribution']}")
        
        return results
    
    def test_least_loaded_selection(self, num_requests=10, concurrent=5):
        """Test least-loaded key selection strategy with concurrent requests"""
        if self.verbose:
            print("\n=== Testing Least-Loaded API Key Selection ===")
        
        results = {
            "strategy": "least-loaded",
            "num_requests": num_requests,
            "concurrent_requests": concurrent,
            "start_time": datetime.now().isoformat(),
            "requests": [],
            "key_distribution": {},
            "success_rate": 0
        }
        
        # Get available API types
        available_apis = []
        if self.multiplexer.openai_clients:
            available_apis.append("openai")
        if self.multiplexer.groq_clients:
            available_apis.append("groq")
        
        if not available_apis:
            results["status"] = "Skipped - No compatible APIs available"
            return results
        
        # Choose a random API type to test
        api_type = random.choice(available_apis)
        results["api_type"] = api_type
        
        # Get the client function based on API type
        if api_type == "openai":
            request_fn = lambda prompt: example_openai_request(
                self.multiplexer, key_name=None, prompt=prompt
            )
            clients = self.multiplexer.openai_clients
        elif api_type == "groq":
            request_fn = lambda prompt: example_groq_request(
                self.multiplexer, key_name=None, prompt=prompt
            )
            clients = self.multiplexer.groq_clients
        
        # Run the requests concurrently
        request_results = []
        start_time = time.time()
        
        # Use ThreadPoolExecutor for concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent) as executor:
            # Submit all tasks
            futures = []
            for i in range(num_requests):
                prompt = random.choice(self.test_prompts)
                futures.append(executor.submit(request_fn, prompt))
            
            # Process results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    result = future.result()
                    request_results.append(result)
                    if self.verbose:
                        status = "✓" if result["success"] else "✗"
                        print(f"Request {i+1}/{num_requests}: {status}")
                except Exception as e:
                    if self.verbose:
                        print(f"Request {i+1}/{num_requests} error: {str(e)}")
                    request_results.append({
                        "success": False,
                        "error": str(e)
                    })
        
        total_time = time.time() - start_time
        results["total_time"] = total_time
        results["requests"] = request_results
        
        # Calculate success rate
        successful = sum(1 for r in request_results if r.get("success", False))
        results["success_rate"] = successful / num_requests if num_requests > 0 else 0
        results["successful_requests"] = successful
        
        # Get key distribution
        stats = self.multiplexer.get_usage_stats()
        results["key_distribution"] = {
            key: data["usage"] for key, data in stats[api_type].items()
        }
        
        if self.verbose:
            print(f"Least-Loaded Test Results: {successful}/{num_requests} successful in {total_time:.2f}s")
            print(f"Key distribution: {results['key_distribution']}")
        
        return results
    
    def test_high_concurrency(self, num_requests=20, concurrent=10):
        """Test the multiplexer under high concurrency"""
        if self.verbose:
            print("\n=== Testing High Concurrency Multiplexing ===")
        
        results = {
            "strategy": "high-concurrency",
            "num_requests": num_requests,
            "concurrent_requests": concurrent,
            "start_time": datetime.now().isoformat(),
            "requests": [],
            "api_distribution": {},
            "success_rate": 0
        }
        
        # Determine which APIs are available
        available_apis = []
        if len(self.multiplexer.openai_clients) > 0:
            available_apis.append("openai")
        if len(self.multiplexer.groq_clients) > 0:
            available_apis.append("groq")
        if len(self.multiplexer.claude_clients) > 0:
            available_apis.append("claude")
        if len(self.multiplexer.gemini_clients) > 0:
            available_apis.append("gemini")
        
        if len(available_apis) < 1:
            results["status"] = "Skipped - Not enough APIs available"
            return results
        
        api_counts = {api: 0 for api in available_apis}
        
        # Define request functions for each API type
        api_functions = {
            "openai": lambda prompt: example_openai_request(self.multiplexer, prompt=prompt),
            "groq": lambda prompt: example_groq_request(self.multiplexer, prompt=prompt),
            # Add equivalent functions for claude and gemini here
        }
        
        # Define a function that will try multiple APIs with fallback
        def make_request_with_fallback(prompt):
            # Try available APIs in a random order
            apis_to_try = available_apis.copy()
            random.shuffle(apis_to_try)
            
            result = None
            used_api = None
            
            for api in apis_to_try:
                if api in api_functions:
                    try:
                        if self.verbose:
                            print(f"Trying {api} API...")
                        
                        result = api_functions[api](prompt)
                        
                        if result and result.get("success", False):
                            used_api = api
                            break
                    except Exception as e:
                        if self.verbose:
                            print(f"Error with {api} API: {str(e)}")
            
            if result:
                result["api_used"] = used_api
                api_counts[used_api] = api_counts.get(used_api, 0) + 1
                return result
            else:
                return {
                    "success": False,
                    "error": "All available APIs failed",
                    "api_used": None
                }
        
        # Run requests concurrently
        request_results = []
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent) as executor:
            futures = []
            for i in range(num_requests):
                prompt = random.choice(self.test_prompts)
                futures.append(executor.submit(make_request_with_fallback, prompt))
            
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    result = future.result()
                    request_results.append(result)
                    if self.verbose:
                        status = "✓" if result.get("success", False) else "✗"
                        api = result.get("api_used", "unknown")
                        print(f"Request {i+1}/{num_requests}: {status} using {api}")
                except Exception as e:
                    if self.verbose:
                        print(f"Request {i+1}/{num_requests} error: {str(e)}")
                    request_results.append({
                        "success": False,
                        "error": str(e),
                        "api_used": None
                    })
        
        total_time = time.time() - start_time
        results["total_time"] = total_time
        results["requests"] = request_results
        
        # Calculate success rate
        successful = sum(1 for r in request_results if r.get("success", False))
        results["success_rate"] = successful / num_requests if num_requests > 0 else 0
        results["successful_requests"] = successful
        
        # Get API distribution
        results["api_distribution"] = api_counts
        
        if self.verbose:
            print(f"High Concurrency Test Results: {successful}/{num_requests} successful in {total_time:.2f}s")
            print(f"API distribution: {results['api_distribution']}")
        
        return results
    
    def test_multiplexed_vs_single(self, requests_per_test=5):
        """Compare multiplexed vs. single API key performance"""
        if self.verbose:
            print("\n=== Comparing Multiplexed vs. Single API Key Performance ===")
        
        results = {
            "strategy": "comparative",
            "requests_per_test": requests_per_test,
            "start_time": datetime.now().isoformat(),
            "multiplexed": {"requests": [], "time": 0, "success_rate": 0},
            "single": {"requests": [], "time": 0, "success_rate": 0}
        }
        
        # Find an API with multiple keys
        api_type = None
        for api in ["openai", "groq"]:
            clients = getattr(self.multiplexer, f"{api}_clients", {})
            if len(clients) > 1:
                api_type = api
                break
        
        if not api_type:
            results["status"] = "Skipped - No API with multiple keys found"
            return results
        
        results["api_type"] = api_type
        
        # Get a list of all prompts to use (for consistency between tests)
        prompts = random.choices(self.test_prompts, k=requests_per_test)
        
        # Test 1: Using multiplexed keys
        if self.verbose:
            print(f"\nTesting with multiplexed {api_type.upper()} API keys...")
        
        # Get the client function
        if api_type == "openai":
            test_fn = lambda prompt: example_openai_request(self.multiplexer, prompt=prompt)
        elif api_type == "groq":
            test_fn = lambda prompt: example_groq_request(self.multiplexer, prompt=prompt)
        
        # Run the multiplexed test
        multiplexed_results = []
        multiplexed_start = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=requests_per_test) as executor:
            futures = [executor.submit(test_fn, prompt) for prompt in prompts]
            
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    result = future.result()
                    multiplexed_results.append(result)
                    if self.verbose:
                        status = "✓" if result.get("success", False) else "✗"
                        print(f"Multiplexed request {i+1}/{requests_per_test}: {status}")
                except Exception as e:
                    if self.verbose:
                        print(f"Multiplexed request {i+1}/{requests_per_test} error: {str(e)}")
                    multiplexed_results.append({"success": False, "error": str(e)})
        
        multiplexed_time = time.time() - multiplexed_start
        
        # Test 2: Using a single key
        if self.verbose:
            print(f"\nTesting with single {api_type.upper()} API key...")
        
        # Get a single key
        if api_type == "openai":
            first_key = list(self.multiplexer.openai_clients.keys())[0]
            single_test_fn = lambda prompt: example_openai_request(
                self.multiplexer, key_name=first_key, prompt=prompt
            )
        elif api_type == "groq":
            first_key = list(self.multiplexer.groq_clients.keys())[0]
            single_test_fn = lambda prompt: example_groq_request(
                self.multiplexer, key_name=first_key, prompt=prompt
            )
        
        # Run the single-key test
        single_results = []
        single_start = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=requests_per_test) as executor:
            futures = [executor.submit(single_test_fn, prompt) for prompt in prompts]
            
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    result = future.result()
                    single_results.append(result)
                    if self.verbose:
                        status = "✓" if result.get("success", False) else "✗"
                        print(f"Single-key request {i+1}/{requests_per_test}: {status}")
                except Exception as e:
                    if self.verbose:
                        print(f"Single-key request {i+1}/{requests_per_test} error: {str(e)}")
                    single_results.append({"success": False, "error": str(e)})
        
        single_time = time.time() - single_start
        
        # Process results
        multiplexed_success = sum(1 for r in multiplexed_results if r.get("success", False))
        single_success = sum(1 for r in single_results if r.get("success", False))
        
        results["multiplexed"]["requests"] = multiplexed_results
        results["multiplexed"]["time"] = multiplexed_time
        results["multiplexed"]["success_rate"] = multiplexed_success / requests_per_test
        
        results["single"]["requests"] = single_results
        results["single"]["time"] = single_time
        results["single"]["success_rate"] = single_success / requests_per_test
        
        # Calculate speedup factor
        if single_time > 0:
            results["speedup_factor"] = single_time / multiplexed_time
        else:
            results["speedup_factor"] = 0
        
        if self.verbose:
            print("\nComparison Results:")
            print(f"Multiplexed: {multiplexed_success}/{requests_per_test} successful in {multiplexed_time:.2f}s")
            print(f"Single key: {single_success}/{requests_per_test} successful in {single_time:.2f}s")
            
            if results["speedup_factor"] > 1:
                print(f"Multiplexing is {results['speedup_factor']:.2f}x faster")
            elif results["speedup_factor"] < 1:
                print(f"Single key is {1/results['speedup_factor']:.2f}x faster")
            else:
                print("No significant difference in speed")
        
        return results
    
    def run_all_tests(self):
        """Run all multiplexing tests and collect results"""
        if self.verbose:
            print("=== Starting Enhanced API Multiplexing Tests ===")
            print(f"Time: {datetime.now().isoformat()}")
        
        start_time = time.time()
        
        # Set up the multiplexer
        keys_added = self._setup_multiplexer()
        
        if keys_added == 0:
            if self.verbose:
                print("No API keys available for testing.")
                print("Please set API keys in your environment or provide them in metadata.")
            
            self.results = {
                "status": "Skipped - No API keys available",
                "timestamp": datetime.now().isoformat()
            }
            return self.results
        
        # Run the test suites
        self.results = {
            "status": "Running",
            "timestamp": datetime.now().isoformat(),
            "keys_added": keys_added,
            "tests": {}
        }
        
        # Get available API types for summary
        api_counts = {
            "openai": len(self.multiplexer.openai_clients),
            "groq": len(self.multiplexer.groq_clients),
            "claude": len(self.multiplexer.claude_clients),
            "gemini": len(self.multiplexer.gemini_clients)
        }
        self.results["available_apis"] = api_counts
        
        if self.verbose:
            print(f"\nAvailable API keys:")
            for api, count in api_counts.items():
                if count > 0:
                    print(f"  {api}: {count} keys")
        
        # Test 1: Round-robin key selection
        test_results = self.test_round_robin_selection(num_requests=5)
        self.results["tests"]["round_robin"] = test_results
        
        # Test 2: Least-loaded key selection
        test_results = self.test_least_loaded_selection(num_requests=6, concurrent=3)
        self.results["tests"]["least_loaded"] = test_results
        
        # Test 3: High concurrency
        test_results = self.test_high_concurrency(num_requests=8, concurrent=4)
        self.results["tests"]["high_concurrency"] = test_results
        
        # Test 4: Multiplexed vs single key
        if any(count > 1 for api, count in api_counts.items() if api in ["openai", "groq"]):
            test_results = self.test_multiplexed_vs_single(requests_per_test=4)
            self.results["tests"]["comparative"] = test_results
        
        # Calculate overall stats
        total_time = time.time() - start_time
        total_requests = sum(
            len(test.get("requests", [])) 
            for test in self.results["tests"].values()
            if isinstance(test, dict)
        )
        
        successful_requests = sum(
            test.get("successful_requests", 0)
            for test in self.results["tests"].values()
            if isinstance(test, dict) and "successful_requests" in test
        )
        
        self.results["total_time"] = total_time
        self.results["total_requests"] = total_requests
        self.results["successful_requests"] = successful_requests
        self.results["success_rate"] = successful_requests / total_requests if total_requests > 0 else 0
        self.results["status"] = "Completed"
        
        if self.verbose:
            print(f"\n=== API Multiplexing Tests Completed ===")
            print(f"Total time: {total_time:.2f}s")
            print(f"Total requests: {total_requests}")
            print(f"Successful requests: {successful_requests}")
            print(f"Success rate: {self.results['success_rate']:.2%}")
        
        return self.results
    
    def __test__(self):
        """Run tests and save results"""
        try:
            results = self.run_all_tests()
            
            # Save results to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"api_multiplexing_enhanced_{timestamp}.json"
            
            with open(filename, "w") as f:
                json.dump(results, f, indent=2)
            
            if self.verbose:
                print(f"\nResults saved to: {filename}")
            
            return results
        
        except Exception as e:
            import traceback
            error_results = {
                "status": "Error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            }
            
            # Still try to save the error results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"api_multiplexing_enhanced_error_{timestamp}.json"
            
            with open(filename, "w") as f:
                json.dump(error_results, f, indent=2)
            
            if self.verbose:
                print(f"Error: {str(e)}")
                print(f"Error details saved to: {filename}")
            
            return error_results

def main():
    parser = argparse.ArgumentParser(description="Enhanced tests for API key multiplexing")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress detailed output")
    
    args = parser.parse_args()
    
    tester = EnhancedApiMultiplexingTest(verbose=not args.quiet)
    tester.__test__()

if __name__ == "__main__":
    main()