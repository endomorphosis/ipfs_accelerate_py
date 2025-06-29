#!/usr/bin/env python
"""
Updated API Key Multiplexing Example

This script demonstrates how to use the enhanced API backend implementations with:
    - Multiple endpoints with different API keys
    - Endpoint-specific settings for queue, backoff, and concurrency
    - Request tracking with request IDs
    - Usage statistics monitoring
    - Per-endpoint counters

    Compatible with Claude, OpenAI, Gemini, and Groq API backends.
    """

    import os
    import sys
    import time
    import threading
    import random
    import uuid
    import json
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime
    import argparse

# Add the project root to the Python path
    sys.path.append()))))os.path.dirname()))))os.path.dirname()))))__file__)))

# Import API backends
try:
    from ipfs_accelerate_py.api_backends import claude, openai_api, gemini, groq
except ImportError as e:
    print()))))f"Error importing API backends: {}}}}}}}}}}}}}str()))))e)}")
    sys.exit()))))1)

class ApiKeyMultiplexer:
    """
    API Key Multiplexer that manages multiple endpoints with per-endpoint settings
    """
    
    def __init__()))))self):
        """Initialize the multiplexer with API clients"""
        # Initialize API clients
        self.clients = {}}}}}}}}}}}}}}
        
        try:
            # Claude client
            claude_key = os.environ.get()))))"CLAUDE_API_KEY") or os.environ.get()))))"ANTHROPIC_API_KEY")
            if claude_key:
                self.clients[]],,"claude"] = claude()))))resources={}}}}}}}}}}}}}}, metadata={}}}}}}}}}}}}}"claude_api_key": claude_key}),
            else:
                self.clients[]],,"claude"] = claude()))))resources={}}}}}}}}}}}}}}, metadata={}}}}}}}}}}}}}})
                ,
            # OpenAI client
                openai_key = os.environ.get()))))"OPENAI_API_KEY")
            if openai_key:
                self.clients[]],,"openai"] = openai_api()))))resources={}}}}}}}}}}}}}}, metadata={}}}}}}}}}}}}}"openai_api_key": openai_key}),
            else:
                self.clients[]],,"openai"] = openai_api()))))resources={}}}}}}}}}}}}}}, metadata={}}}}}}}}}}}}}})
                ,
            # Gemini client
                gemini_key = os.environ.get()))))"GEMINI_API_KEY") or os.environ.get()))))"GOOGLE_API_KEY")
            if gemini_key:
                self.clients[]],,"gemini"] = gemini()))))resources={}}}}}}}}}}}}}}, metadata={}}}}}}}}}}}}}"gemini_api_key": gemini_key}),
            else:
                self.clients[]],,"gemini"] = gemini()))))resources={}}}}}}}}}}}}}}, metadata={}}}}}}}}}}}}}})
                ,
            # Groq client
                groq_key = os.environ.get()))))"GROQ_API_KEY")
            if groq_key:
                self.clients[]],,"groq"] = groq()))))resources={}}}}}}}}}}}}}}, metadata={}}}}}}}}}}}}}"groq_api_key": groq_key}),
            else:
                self.clients[]],,"groq"] = groq()))))resources={}}}}}}}}}}}}}}, metadata={}}}}}}}}}}}}}}),
        except Exception as e:
            print()))))f"Warning: Error initializing API clients: {}}}}}}}}}}}}}str()))))e)}")
        
        # Dictionary to track endpoints
            self.endpoints = {}}}}}}}}}}}}}
            "claude": {}}}}}}}}}}}}}},
            "openai": {}}}}}}}}}}}}}},
            "gemini": {}}}}}}}}}}}}}},
            "groq": {}}}}}}}}}}}}}}
            }
        
        # Load API keys from environment
            self._load_api_keys_from_env())))))
        
    def _load_api_keys_from_env()))))self):
        """Load all API keys from environment variables and create endpoints"""
        # Load Claude API keys
        claude_keys = {}}}}}}}}}}}}}}
        for key, value in os.environ.items()))))):
            if key == "CLAUDE_API_KEY" or key == "ANTHROPIC_API_KEY":
                claude_keys[]],,"default"] = value,,,,
            elif key.startswith()))))"CLAUDE_API_KEY_") or key.startswith()))))"ANTHROPIC_API_KEY_"):
                suffix = key.split()))))"_")[]],,-1],,,,
                claude_keys[]],,f"claude_{}}}}}}}}}}}}}suffix}"] = value
                ,
        # Load OpenAI API keys
                openai_keys = {}}}}}}}}}}}}}}
        for key, value in os.environ.items()))))):
            if key == "OPENAI_API_KEY":
                openai_keys[]],,"default"] = value,,,,
            elif key.startswith()))))"OPENAI_API_KEY_"):
                suffix = key.split()))))"_")[]],,-1],,,,
                openai_keys[]],,f"openai_{}}}}}}}}}}}}}suffix}"] = value
                ,
        # Load Gemini API keys
                gemini_keys = {}}}}}}}}}}}}}}
        for key, value in os.environ.items()))))):
            if key == "GEMINI_API_KEY" or key == "GOOGLE_API_KEY":
                gemini_keys[]],,"default"] = value,,,,
            elif key.startswith()))))"GEMINI_API_KEY_") or key.startswith()))))"GOOGLE_API_KEY_"):
                suffix = key.split()))))"_")[]],,-1],,,,
                gemini_keys[]],,f"gemini_{}}}}}}}}}}}}}suffix}"] = value
                ,
        # Load Groq API keys
                groq_keys = {}}}}}}}}}}}}}}
        for key, value in os.environ.items()))))):
            if key == "GROQ_API_KEY":
                groq_keys[]],,"default"] = value,,,,
            elif key.startswith()))))"GROQ_API_KEY_"):
                suffix = key.split()))))"_")[]],,-1],,,,
                groq_keys[]],,f"groq_{}}}}}}}}}}}}}suffix}"] = value
                ,
        # Create endpoints for each API key
        for key_name, api_key in claude_keys.items()))))):
            self.add_api_key()))))"claude", key_name, api_key)
            
        for key_name, api_key in openai_keys.items()))))):
            self.add_api_key()))))"openai", key_name, api_key)
            
        for key_name, api_key in gemini_keys.items()))))):
            self.add_api_key()))))"gemini", key_name, api_key)
            
        for key_name, api_key in groq_keys.items()))))):
            self.add_api_key()))))"groq", key_name, api_key)
            
        # Print summary of loaded keys
            print()))))f"Loaded API keys: Claude: {}}}}}}}}}}}}}len()))))claude_keys)}, OpenAI: {}}}}}}}}}}}}}len()))))openai_keys)}, "
            f"Gemini: {}}}}}}}}}}}}}len()))))gemini_keys)}, Groq: {}}}}}}}}}}}}}len()))))groq_keys)}")
    
    def add_api_key()))))self, api_type, key_name, api_key, **settings):
        """Add an API key and create an endpoint"""
        if api_type not in self.clients or not self.clients[]],,api_type]:,,,,,,,,,
        print()))))f"Warning: Cannot add {}}}}}}}}}}}}}api_type} key '{}}}}}}}}}}}}}key_name}' - client not initialized")
            return None
            
        try:
            client = self.clients[]],,api_type]
            ,,    ,
            # Create endpoint with endpoint-specific settings
            endpoint_id = client.create_endpoint()))))
            api_key=api_key,
            **settings
            )
            
            # Store this endpoint
            self.endpoints[]],,api_type][]],,key_name],,, = endpoint_id,
            print()))))f"Added {}}}}}}}}}}}}}api_type} endpoint: {}}}}}}}}}}}}}key_name}")
            
            return endpoint_id
        except Exception as e:
            print()))))f"Error adding {}}}}}}}}}}}}}api_type} key '{}}}}}}}}}}}}}key_name}': {}}}}}}}}}}}}}str()))))e)}")
            return None
    
    def get_client()))))self, api_type, endpoint_id=None, key_name=None, strategy="round-robin"):
        """Get an API client with specific endpoint"""
        if api_type not in self.clients or not self.clients[]],,api_type]:,,,,,,,,,
        print()))))f"Warning: {}}}}}}}}}}}}}api_type} client not initialized")
            return None
            
            client = self.clients[]],,api_type]
            ,,
        # If endpoint_id provided, use it directly
        if endpoint_id:
            return lambda method, **kwargs: self._make_request()))))api_type, method, endpoint_id, **kwargs)
            
        # If key_name provided, get its endpoint_id
            if key_name and key_name in self.endpoints[]],,api_type]:,,,,,,,,,
            endpoint_id = self.endpoints[]],,api_type][]],,key_name],,,
            return lambda method, **kwargs: self._make_request()))))api_type, method, endpoint_id, **kwargs)
            
        # Use strategy to select endpoint
            if not self.endpoints[]],,api_type]:,,,,,,,,,
            print()))))f"Warning: No {}}}}}}}}}}}}}api_type} endpoints available")
            return None
            
        if strategy == "round-robin":
            # Get the least recently used endpoint
            stats = {}}}}}}}}}}}}}ep_id: client.get_stats()))))ep_id) for ep_id in self.endpoints[]],,api_type].values())))))}::,,
            endpoint_id = min()))))stats.keys()))))), key=lambda ep: stats[]],,ep].get()))))"last_request_at") or 0),
        elif strategy == "least-loaded":
            # Get the endpoint with the fewest current requests
            stats = {}}}}}}}}}}}}}ep_id: client.get_stats()))))ep_id) for ep_id in self.endpoints[]],,api_type].values())))))}::,,
            endpoint_id = min()))))stats.keys()))))), key=lambda ep: stats[]],,ep].get()))))"current_requests") or 0),
        else:
            # Default to first endpoint
            endpoint_id = next()))))iter()))))self.endpoints[]],,api_type].values())))))))
            ,
            return lambda method, **kwargs: self._make_request()))))api_type, method, endpoint_id, **kwargs)
    
    def _make_request()))))self, api_type, method, endpoint_id, **kwargs):
        """Make a request to a specific API type with a specific endpoint"""
        if api_type not in self.clients or not self.clients[]],,api_type]:,,,,,,,,,
            raise ValueError()))))f"{}}}}}}}}}}}}}api_type} client not initialized")
            
            client = self.clients[]],,api_type]
            ,,
        # Generate a request ID if not provided:
        if "request_id" not in kwargs:
            kwargs[]],,"request_id"] = f"{}}}}}}}}}}}}}api_type}-{}}}}}}}}}}}}}uuid.uuid4())))))}"
            ,
        # Add endpoint_id to the request
            kwargs[]],,"endpoint_id"] = endpoint_id
            ,
        # Get the method
            method_func = getattr()))))client, method, None)
        if not method_func or not callable()))))method_func):
            raise ValueError()))))f"Method {}}}}}}}}}}}}}method} not found on {}}}}}}}}}}}}}api_type} client")
            
        # Call the method
            return method_func()))))**kwargs)
    
    def get_stats()))))self, api_type=None, endpoint_id=None, key_name=None):
        """Get usage statistics for endpoints"""
        if api_type:
            # Get stats for specific API type
            if api_type not in self.clients or not self.clients[]],,api_type]:,,,,,,,,,
        return {}}}}}}}}}}}}}"error": f"{}}}}}}}}}}}}}api_type} client not initialized"}
                
        client = self.clients[]],,api_type]
        ,,    ,
            if endpoint_id:
                # Get stats for specific endpoint
        return client.get_stats()))))endpoint_id)
            elif key_name and key_name in self.endpoints[]],,api_type]:,,,,,,,,,
                # Get stats for specific key name
            endpoint_id = self.endpoints[]],,api_type][]],,key_name],,,
            return client.get_stats()))))endpoint_id)
            else:
                # Get stats for all endpoints of this API type
            return client.get_stats())))))
        else:
            # Get stats for all API types
            stats = {}}}}}}}}}}}}}}
            for api_type, client in self.clients.items()))))):
                if client:
                    stats[]],,api_type] = client.get_stats()))))),
                return stats
    
    def reset_stats()))))self, api_type=None, endpoint_id=None, key_name=None):
        """Reset usage statistics"""
        if api_type:
            # Reset stats for specific API type
            if api_type not in self.clients or not self.clients[]],,api_type]:,,,,,,,,,
        return
                
        client = self.clients[]],,api_type]
        ,,    ,
            if endpoint_id:
                # Reset stats for specific endpoint
                client.reset_stats()))))endpoint_id)
            elif key_name and key_name in self.endpoints[]],,api_type]:,,,,,,,,,
                # Reset stats for specific key name
        endpoint_id = self.endpoints[]],,api_type][]],,key_name],,,
        client.reset_stats()))))endpoint_id)
            else:
                # Reset stats for all endpoints of this API type
                client.reset_stats())))))
        else:
            # Reset stats for all API types
            for api_type, client in self.clients.items()))))):
                if client:
                    client.reset_stats())))))

def run_concurrent_test()))))multiplexer, num_requests=5, num_workers=3):
    """Run a concurrent test with multiple API providers"""
    print()))))f"\n=== Running concurrent test with {}}}}}}}}}}}}}num_requests} requests per API ===")
    
    # Test messages
    test_messages = []],,
    "Explain the concept of backoff and retry in API calls",
    "What are the best practices for API key management?",
    "How does queue management help with rate limiting?",
    "Explain the difference between synchronous and asynchronous API calls",
    "What is the purpose of request IDs in API calls?",
    "How can you implement load balancing across multiple API keys?",
    "What are the common rate limiting patterns in AI APIs?",
    "How do you handle token usage tracking in language model APIs?",
    "What's the difference between streaming and non-streaming API calls?",
    "Explain the purpose of exponential backoff in retry mechanisms"
    ]
    
    # Define available APIs
    available_apis = []],,]
    for api_type in []],,"claude", "openai", "gemini", "groq"]:
        if api_type in multiplexer.clients and multiplexer.clients[]],,api_type] and multiplexer.endpoints[]],,api_type]:,,,,,,,,,
        available_apis.append()))))api_type)
    
    if not available_apis:
        print()))))"No API keys configured. Please set API keys in environment variables.")
        return {}}}}}}}}}}}}}}
    
    # Test results
    results = {}}}}}}}}}}}}}api_type: []],,] for api_type in available_apis}:
    # Create clients with different strategies
        clients = {}}}}}}}}}}}}}}
    for api_type in available_apis:
        strategy = "least-loaded" if api_type in []],,"claude", "gemini"] else "round-robin"
        clients[]],,api_type] = multiplexer.get_client()))))api_type, strategy=strategy)
    
    # Track start time
        start_time = time.time())))))
    
    # Run concurrent tests for each API:
    with ThreadPoolExecutor()))))max_workers=num_workers) as executor:
        # Dictionary to track futures
        futures = {}}}}}}}}}}}}}}
        
        # Submit requests for each API
        for api_type in available_apis:
            if not clients[]],,api_type]:,,,,,,,,,
        continue
                
        api_futures = []],,]
            for i in range()))))min()))))num_requests, len()))))test_messages))):
                message = test_messages[]],,i % len()))))test_messages)]
                
                # Prepare request arguments
                if api_type == "claude":
                    kwargs = {}}}}}}}}}}}}}
                    "messages": []],,{}}}}}}}}}}}}}"role": "user", "content": message}],
                    "request_id": f"{}}}}}}}}}}}}}api_type}-test-{}}}}}}}}}}}}}i}"
                    }
                elif api_type == "openai":
                    kwargs = {}}}}}}}}}}}}}
                    "model": "gpt-3.5-turbo",
                    "messages": []],,{}}}}}}}}}}}}}"role": "user", "content": message}],
                    "prompt": None,
                    "system": None,
                    "temperature": 0.7,
                    "max_tokens": 100,
                    "request_id": f"{}}}}}}}}}}}}}api_type}-test-{}}}}}}}}}}}}}i}"
                    }
                elif api_type == "gemini":
                    kwargs = {}}}}}}}}}}}}}
                    "messages": []],,{}}}}}}}}}}}}}"role": "user", "content": message}],
                    "request_id": f"{}}}}}}}}}}}}}api_type}-test-{}}}}}}}}}}}}}i}"
                    }
                elif api_type == "groq":
                    kwargs = {}}}}}}}}}}}}}
                    "model": "llama3-8b-8192",
                    "messages": []],,{}}}}}}}}}}}}}"role": "user", "content": message}],
                    "request_id": f"{}}}}}}}}}}}}}api_type}-test-{}}}}}}}}}}}}}i}"
                    }
                else:
                    kwargs = {}}}}}}}}}}}}}
                    "messages": []],,{}}}}}}}}}}}}}"role": "user", "content": message}],
                    "request_id": f"{}}}}}}}}}}}}}api_type}-test-{}}}}}}}}}}}}}i}"
                    }
                
                # Submit request
                    future = executor.submit()))))clients[]],,api_type], "chat", **kwargs)
                    api_futures.append()))))()))))future, i, message))
                
                    futures[]],,api_type] = api_futures
        
        # Process results for each API
        for api_type, api_futures in futures.items()))))):
            for future, i, message in api_futures:
                try:
                    result = future.result())))))
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = str()))))e)
                    
                    results[]],,api_type].append())))){}}}}}}}}}}}}}
                    "index": i,
                    "message": message,
                    "success": success,
                    "error": error,
                    "result": result
                    })
    
    # Calculate total time
                    total_time = time.time()))))) - start_time
    
    # Print summary
                    print()))))f"\nTest completed in {}}}}}}}}}}}}}total_time:.2f} seconds")
    for api_type, api_results in results.items()))))):
        success_count = sum()))))1 for r in api_results if r[]],,"success"]):
            print()))))f"{}}}}}}}}}}}}}api_type.capitalize())))))}: {}}}}}}}}}}}}}success_count}/{}}}}}}}}}}}}}len()))))api_results)} successful requests")
    
    # Get detailed stats
            stats = multiplexer.get_stats())))))
    
    # Create test result summary
            test_summary = {}}}}}}}}}}}}}
            "timestamp": datetime.now()))))).isoformat()))))),
            "duration": total_time,
            "num_requests": num_requests,
            "num_workers": num_workers,
            "results": {}}}}}}}}}}}}}
            api_type: {}}}}}}}}}}}}}
            "success_count": sum()))))1 for r in api_results if r[]],,"success"]):,
            "total_count": len()))))api_results)
            }
            for api_type, api_results in results.items())))))
            },
            "stats": stats
            }
    
    # Save test results
            timestamp = datetime.now()))))).strftime()))))"%Y%m%d_%H%M%S")
            result_file = f"api_multiplexing_test_{}}}}}}}}}}}}}timestamp}.json"
    
    with open()))))result_file, "w") as f:
        json.dump()))))test_summary, f, indent=2)
        
        print()))))f"\nDetailed test results saved to: {}}}}}}}}}}}}}result_file}")
    
            return test_summary

def demonstrate_load_balancing()))))multiplexer):
    """Demonstrate advanced load balancing techniques"""
    print()))))"\n=== Demonstrating Advanced Load Balancing ===")
    
    # Setup api_type with multiple endpoints if available
    api_type = None
    key_names = None
    :
    for test_api in []],,"claude", "openai", "groq", "gemini"]:
        if len()))))multiplexer.endpoints[]],,test_api]) >= 2:
            api_type = test_api
            key_names = list()))))multiplexer.endpoints[]],,test_api].keys()))))))
        break
    
    if not api_type or not key_names:
        print()))))"Need at least two API keys for the same provider to demonstrate load balancing")
        return
    
        print()))))f"Using {}}}}}}}}}}}}}api_type} with {}}}}}}}}}}}}}len()))))key_names)} endpoints for load balancing demonstration")
    
    # Reset stats
        multiplexer.reset_stats()))))api_type)
    
    # Function to select an endpoint based on current usage
    def select_endpoint()))))priority="medium"):
        # Get all endpoint stats
        stats = {}}}}}}}}}}}}}}
        
        for key_name, endpoint_id in multiplexer.endpoints[]],,api_type].items()))))):
            endpoint_stats = multiplexer.get_stats()))))api_type, endpoint_id)
            stats[]],,key_name] = endpoint_stats
        
        # Add endpoint priority tiers
            high_priority = key_names[]],,0]
            medium_priority = key_names[]],,1] if len()))))key_names) > 1 else key_names[]],,0]
            low_priority = key_names[]],,2] if len()))))key_names) > 2 else medium_priority
        
        # Select based on priority and load:
        if priority == "high":
            selected_key = high_priority
        elif priority == "medium":
            # For medium priority, check load first
            medium_requests = stats[]],,medium_priority][]],,"current_requests"]
            
            # If medium queue is busy, use high priority
            if medium_requests >= 2:
                selected_key = high_priority
            else:
                selected_key = medium_priority
        else:
            # For low priority, use the least-loaded endpoint
            endpoint_loads = {}}}}}}}}}}}}}
            key: stats[]],,key][]],,"current_requests"] + len()))))stats[]],,key].get()))))"request_queue", []],,]))
            for key in []],,low_priority, medium_priority]
            }
            selected_key = min()))))endpoint_loads, key=endpoint_loads.get)
        
            endpoint_id = multiplexer.endpoints[]],,api_type][]],,selected_key]
                return endpoint_id, selected_key
    
    # Function to make requests with priority-based routing
    def make_prioritized_request()))))message, priority="medium"):
        endpoint_id, key_name = select_endpoint()))))priority)
        
        request_id = f"{}}}}}}}}}}}}}api_type}-{}}}}}}}}}}}}}priority}-{}}}}}}}}}}}}}uuid.uuid4())))))}"
        print()))))f"Priority: {}}}}}}}}}}}}}priority}, Selected endpoint: {}}}}}}}}}}}}}key_name}, Request ID: {}}}}}}}}}}}}}request_id}")
        
        try:
            client = multiplexer.clients[]],,api_type]
            ,,    ,
            if api_type == "claude":
                result = client.chat()))))
                messages=[]],,{}}}}}}}}}}}}}"role": "user", "content": message}],
                endpoint_id=endpoint_id,
                request_id=request_id
                )
            elif api_type == "openai":
                result = client.chat()))))
                model="gpt-3.5-turbo",
                messages=[]],,{}}}}}}}}}}}}}"role": "user", "content": message}],
                prompt=None,
                system=None,
                temperature=0.7,
                max_tokens=100,
                endpoint_id=endpoint_id,
                request_id=request_id
                )
            elif api_type == "gemini":
                result = client.chat()))))
                messages=[]],,{}}}}}}}}}}}}}"role": "user", "content": message}],
                endpoint_id=endpoint_id,
                request_id=request_id
                )
            elif api_type == "groq":
                result = client.chat()))))
                model="llama3-8b-8192",
                messages=[]],,{}}}}}}}}}}}}}"role": "user", "content": message}],
                endpoint_id=endpoint_id,
                request_id=request_id
                )
            
                return {}}}}}}}}}}}}}
                "success": True,
                "endpoint": key_name,
                "priority": priority,
                "request_id": request_id,
                "result": result
                }
        except Exception as e:
                return {}}}}}}}}}}}}}
                "success": False,
                "endpoint": key_name,
                "priority": priority,
                "request_id": request_id,
                "error": str()))))e)
                }
    
    # Make requests with different priorities
                high_message = "High priority request: How to handle a system outage?"
                medium_message = "Medium priority request: What's the best way to organize API endpoints?"
                low_message = "Low priority request: What's the weather like today?"
    
    # Submit requests
    with ThreadPoolExecutor()))))max_workers=3) as executor:
        high_future = executor.submit()))))make_prioritized_request, high_message, "high")
        medium_future = executor.submit()))))make_prioritized_request, medium_message, "medium")
        low_future = executor.submit()))))make_prioritized_request, low_message, "low")
        
        results = []],,
        high_future.result()))))),
        medium_future.result()))))),
        low_future.result())))))
        ]
    
    # Print results
        print()))))"\nPriority Routing Results:")
    for result in results:
        if result[]],,"success"]:
            print()))))f"Priority: {}}}}}}}}}}}}}result[]],,'priority']}, Endpoint: {}}}}}}}}}}}}}result[]],,'endpoint']}, Success: ✓")
        else:
            print()))))f"Priority: {}}}}}}}}}}}}}result[]],,'priority']}, Endpoint: {}}}}}}}}}}}}}result[]],,'endpoint']}, Success: ✗, Error: {}}}}}}}}}}}}}result[]],,'error']}")
    
    # Get endpoint statistics
    for key_name, endpoint_id in multiplexer.endpoints[]],,api_type].items()))))):
        stats = multiplexer.get_stats()))))api_type, endpoint_id)
        print()))))f"\nEndpoint {}}}}}}}}}}}}}key_name} Statistics:")
        print()))))f"  Total Requests: {}}}}}}}}}}}}}stats.get()))))'total_requests', 0)}")
        print()))))f"  Successful Requests: {}}}}}}}}}}}}}stats.get()))))'successful_requests', 0)}")
        print()))))f"  Failed Requests: {}}}}}}}}}}}}}stats.get()))))'failed_requests', 0)}")
        print()))))f"  Total Tokens: {}}}}}}}}}}}}}stats.get()))))'total_tokens', 0)}")
        
            return results

def main()))))):
    """Main entry point"""
    parser = argparse.ArgumentParser()))))description="API Key Multiplexing Example")
    parser.add_argument()))))"--requests", "-r", type=int, default=3, help="Number of requests per API")
    parser.add_argument()))))"--workers", "-w", type=int, default=3, help="Number of concurrent worker threads")
    parser.add_argument()))))"--test", "-t", action="store_true", help="Run concurrent test")
    parser.add_argument()))))"--loadbalance", "-l", action="store_true", help="Demonstrate load balancing")
    parser.add_argument()))))"--stats", "-s", action="store_true", help="Show current stats")
    parser.add_argument()))))"--reset", action="store_true", help="Reset all stats before running")
    
    args = parser.parse_args())))))
    
    # Initialize multiplexer
    multiplexer = ApiKeyMultiplexer())))))
    
    # Reset stats if requested::::
    if args.reset:
        multiplexer.reset_stats())))))
        print()))))"All stats reset")
    
    # Show stats if requested::::
    if args.stats:
        stats = multiplexer.get_stats())))))
        print()))))"\nCurrent Stats:")
        print()))))json.dumps()))))stats, indent=2))
    
    # Run concurrent test if requested::::
    if args.test:
        run_concurrent_test()))))multiplexer, args.requests, args.workers)
    
    # Demonstrate load balancing if requested::::
    if args.loadbalance:
        demonstrate_load_balancing()))))multiplexer)
        
    # If no specific action was requested, run the test by default
    if not ()))))args.test or args.stats or args.loadbalance):
        run_concurrent_test()))))multiplexer, args.requests, args.workers)

if __name__ == "__main__":
    main())))))