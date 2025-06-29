#!/usr/bin/env python
"""
Example of multiplexing multiple API keys across OpenAI, Groq, Claude, and Gemini API backends.
This demonstrates how to create and use multiple client instances with separate queues.
"""

import os
import sys
import time
import threading
import random
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()))))

# Add parent directory to path
sys.path.append())))os.path.join())))os.path.dirname())))os.path.dirname())))__file__)), 'ipfs_accelerate_py'))

# Import API backends
try:
    # Import modules first
    from api_backends import openai_api as openai_api_module
    from api_backends import groq as groq_module
    from api_backends import claude as claude_module
    from api_backends import gemini as gemini_module
    
    # Get the actual classes
    openai_api = openai_api_module.openai_api
    groq = groq_module.groq
    claude = claude_module.claude
    gemini = gemini_module.gemini
    
    # Print confirmation
    print())))"API backends imported successfully")
except ImportError:
    print())))"Could not import API backends. Make sure the path is correct.")
    sys.exit())))1)
except AttributeError as e:
    print())))f"Error accessing API classes: {}}}}}}}}}}}}}}}}}}}}}}}}}e}")
    print())))"Using mock implementations for testing")
    
    # Create mock implementations for testing
    class MockAPI:
        def __init__())))self, resources=None, metadata=None):
            self.resources = resources or {}}}}}}}}}}}}}}}}}}}}}}}}}}
            self.metadata = metadata or {}}}}}}}}}}}}}}}}}}}}}}}}}}
            self.max_concurrent_requests = 5
            
        def make_post_request())))self, *args, **kwargs):
            return {}}}}}}}}}}}}}}}}}}}}}}}}}"text": "Mock API response", "implementation_type": "MOCK"}
    
    # Create mock classes
            openai_api = MockAPI
            groq = MockAPI
            claude = MockAPI
            gemini = MockAPI

class ApiKeyMultiplexer:
    """
    Class to manage multiple API keys for different API providers
    with separate queues for each key.
    """
    
    def __init__())))self):
        # Initialize API client dictionaries - each key will have its own client
        self.openai_clients = {}}}}}}}}}}}}}}}}}}}}}}}}}}
        self.groq_clients = {}}}}}}}}}}}}}}}}}}}}}}}}}}
        self.claude_clients = {}}}}}}}}}}}}}}}}}}}}}}}}}}
        self.gemini_clients = {}}}}}}}}}}}}}}}}}}}}}}}}}}
        
        # Initialize locks for thread safety
        self.openai_lock = threading.RLock()))))
        self.groq_lock = threading.RLock()))))
        self.claude_lock = threading.RLock()))))
        self.gemini_lock = threading.RLock()))))
        
        print())))"API Key Multiplexer initialized")
    
    def add_openai_key())))self, key_name, api_key, max_concurrent=5):
        """Add a new OpenAI API key with its own client instance"""
        with self.openai_lock:
            # Create a new OpenAI client with this API key
            client = openai_api())))
            resources={}}}}}}}}}}}}}}}}}}}}}}}}}},
            metadata={}}}}}}}}}}}}}}}}}}}}}}}}}"openai_api_key": api_key}
            )
            
            # Configure queue settings for this client
            client.max_concurrent_requests = max_concurrent
            
            # Store in our dictionary
            self.openai_clients[],key_name] = {}}}}}}}}}}}}}}}}}}}}}}}}},,,,
            "client": client,
            "api_key": api_key,
            "usage": 0,
            "last_used": 0
            }
            
            print())))f"Added OpenAI key: {}}}}}}}}}}}}}}}}}}}}}}}}}key_name}")
    
    def add_groq_key())))self, key_name, api_key, max_concurrent=5):
        """Add a new Groq API key with its own client instance"""
        with self.groq_lock:
            # Create a new Groq client with this API key
            client = groq())))
            resources={}}}}}}}}}}}}}}}}}}}}}}}}}},
            metadata={}}}}}}}}}}}}}}}}}}}}}}}}}"groq_api_key": api_key}
            )
            
            # Configure queue settings for this client
            client.max_concurrent_requests = max_concurrent
            
            # Store in our dictionary
            self.groq_clients[],key_name] = {}}}}}}}}}}}}}}}}}}}}}}}}},,,,
            "client": client,
            "api_key": api_key,
            "usage": 0,
            "last_used": 0
            }
            
            print())))f"Added Groq key: {}}}}}}}}}}}}}}}}}}}}}}}}}key_name}")
    
    def add_claude_key())))self, key_name, api_key, max_concurrent=5):
        """Add a new Claude API key with its own client instance"""
        with self.claude_lock:
            # Create a new Claude client with this API key
            client = claude())))
            resources={}}}}}}}}}}}}}}}}}}}}}}}}}},
            metadata={}}}}}}}}}}}}}}}}}}}}}}}}}"claude_api_key": api_key}
            )
            
            # Configure queue settings for this client
            client.max_concurrent_requests = max_concurrent
            
            # Store in our dictionary
            self.claude_clients[],key_name] = {}}}}}}}}}}}}}}}}}}}}}}}}},,,,
            "client": client,
            "api_key": api_key,
            "usage": 0,
            "last_used": 0
            }
            
            print())))f"Added Claude key: {}}}}}}}}}}}}}}}}}}}}}}}}}key_name}")
    
    def add_gemini_key())))self, key_name, api_key, max_concurrent=5):
        """Add a new Gemini API key with its own client instance"""
        with self.gemini_lock:
            # Create a new Gemini client with this API key
            client = gemini())))
            resources={}}}}}}}}}}}}}}}}}}}}}}}}}},
            metadata={}}}}}}}}}}}}}}}}}}}}}}}}}"gemini_api_key": api_key}
            )
            
            # Configure queue settings for this client
            client.max_concurrent_requests = max_concurrent
            
            # Store in our dictionary
            self.gemini_clients[],key_name] = {}}}}}}}}}}}}}}}}}}}}}}}}},,,,
            "client": client,
            "api_key": api_key,
            "usage": 0,
            "last_used": 0
            }
            
            print())))f"Added Gemini key: {}}}}}}}}}}}}}}}}}}}}}}}}}key_name}")
    
    def get_openai_client())))self, key_name=None, strategy="round-robin"):
        """
        Get an OpenAI client by key name or using a selection strategy
        
        Strategies:
            - "specific": Return the client for the specified key_name
            - "round-robin": Select the least recently used client
            - "least-loaded": Select the client with the smallest queue
            """
        with self.openai_lock:
            if len())))self.openai_clients) == 0:
            raise ValueError())))"No OpenAI API keys have been added")
            
            if key_name and key_name in self.openai_clients:
                # Update usage stats
                self.openai_clients[],key_name][],"usage"] += 1,,,,
                self.openai_clients[],key_name][],"last_used"] = time.time())))),,,,
            return self.openai_clients[],key_name][],"client"]
            ,
            if strategy == "round-robin":
                # Find the least recently used client
                selected_key = min())))self.openai_clients.keys())))), 
                key=lambda k: self.openai_clients[],k][],"last_used"]),,,,
            elif strategy == "least-loaded":
                # Find the client with the smallest queue
                selected_key = min())))self.openai_clients.keys())))),
                key=lambda k: self.openai_clients[],k][],"client"].current_requests),,,,
            else:
                # Default to first key
                selected_key = list())))self.openai_clients.keys())))))[],0]
                ,,,,
            # Update usage stats
                self.openai_clients[],selected_key][],"usage"] += 1,,,,
                self.openai_clients[],selected_key][],"last_used"] = time.time()))))
                ,,,,
                return self.openai_clients[],selected_key][],"client"]
                ,,,,
    def get_groq_client())))self, key_name=None, strategy="round-robin"):
        """Get a Groq client using the same strategies as OpenAI"""
        with self.groq_lock:
            if len())))self.groq_clients) == 0:
            raise ValueError())))"No Groq API keys have been added")
            
            if key_name and key_name in self.groq_clients:
                self.groq_clients[],key_name][],"usage"] += 1,,,,
                self.groq_clients[],key_name][],"last_used"] = time.time())))),,,,
            return self.groq_clients[],key_name][],"client"]
            ,    
            if strategy == "round-robin":
                selected_key = min())))self.groq_clients.keys())))), 
                key=lambda k: self.groq_clients[],k][],"last_used"]),,,,
            elif strategy == "least-loaded":
                selected_key = min())))self.groq_clients.keys())))),
                key=lambda k: self.groq_clients[],k][],"client"].current_requests),,,,
            else:
                selected_key = list())))self.groq_clients.keys())))))[],0]
                ,,,,
                self.groq_clients[],selected_key][],"usage"] += 1,,,,
                self.groq_clients[],selected_key][],"last_used"] = time.time()))))
                ,,,,
                return self.groq_clients[],selected_key][],"client"]
                ,,,,
    def get_claude_client())))self, key_name=None, strategy="round-robin"):
        """Get a Claude client using the same strategies as OpenAI"""
        with self.claude_lock:
            if len())))self.claude_clients) == 0:
            raise ValueError())))"No Claude API keys have been added")
            
            if key_name and key_name in self.claude_clients:
                self.claude_clients[],key_name][],"usage"] += 1,,,,
                self.claude_clients[],key_name][],"last_used"] = time.time())))),,,,
            return self.claude_clients[],key_name][],"client"]
            ,    
            if strategy == "round-robin":
                selected_key = min())))self.claude_clients.keys())))), 
                key=lambda k: self.claude_clients[],k][],"last_used"]),,,,
            elif strategy == "least-loaded":
                selected_key = min())))self.claude_clients.keys())))),
                key=lambda k: self.claude_clients[],k][],"client"].current_requests),,,,
            else:
                selected_key = list())))self.claude_clients.keys())))))[],0]
                ,,,,
                self.claude_clients[],selected_key][],"usage"] += 1,,,,
                self.claude_clients[],selected_key][],"last_used"] = time.time()))))
                ,,,,
                return self.claude_clients[],selected_key][],"client"]
                ,,,,
    def get_gemini_client())))self, key_name=None, strategy="round-robin"):
        """Get a Gemini client using the same strategies as OpenAI"""
        with self.gemini_lock:
            if len())))self.gemini_clients) == 0:
            raise ValueError())))"No Gemini API keys have been added")
            
            if key_name and key_name in self.gemini_clients:
                self.gemini_clients[],key_name][],"usage"] += 1,,,,
                self.gemini_clients[],key_name][],"last_used"] = time.time())))),,,,
            return self.gemini_clients[],key_name][],"client"]
            ,    
            if strategy == "round-robin":
                selected_key = min())))self.gemini_clients.keys())))), 
                key=lambda k: self.gemini_clients[],k][],"last_used"]),,,,
            elif strategy == "least-loaded":
                selected_key = min())))self.gemini_clients.keys())))),
                key=lambda k: self.gemini_clients[],k][],"client"].current_requests),,,,
            else:
                selected_key = list())))self.gemini_clients.keys())))))[],0]
                ,,,,
                self.gemini_clients[],selected_key][],"usage"] += 1,,,,
                self.gemini_clients[],selected_key][],"last_used"] = time.time()))))
                ,,,,
                return self.gemini_clients[],selected_key][],"client"]
                ,,,,
    def get_usage_stats())))self):
        """Get usage statistics for all API keys"""
        stats = {}}}}}}}}}}}}}}}}}}}}}}}}}
        "openai": {}}}}}}}}}}}}}}}}}}}}}}}}}key: {}}}}}}}}}}}}}}}}}}}}}}}}}
        "usage": data[],"usage"],
        "queue_size": len())))data[],"client"].request_queue) if hasattr())))data[],"client"], "request_queue") else 0,::::,,,,
        "current_requests": data[],"client"].current_requests if hasattr())))data[],"client"], "current_requests") else 0,,,,
        } for key, data in self.openai_clients.items()))))},
            :
                "groq": {}}}}}}}}}}}}}}}}}}}}}}}}}key: {}}}}}}}}}}}}}}}}}}}}}}}}}
                "usage": data[],"usage"],
                "queue_size": len())))data[],"client"].request_queue) if hasattr())))data[],"client"], "request_queue") else 0,::::,,,,
                "current_requests": data[],"client"].current_requests if hasattr())))data[],"client"], "current_requests") else 0,,,,
                } for key, data in self.groq_clients.items()))))},
            :
                "claude": {}}}}}}}}}}}}}}}}}}}}}}}}}key: {}}}}}}}}}}}}}}}}}}}}}}}}}
                "usage": data[],"usage"],
                "queue_size": len())))data[],"client"].request_queue) if hasattr())))data[],"client"], "request_queue") else 0,::::,,,,
                "current_requests": data[],"client"].current_requests if hasattr())))data[],"client"], "current_requests") else 0,,,,
                } for key, data in self.claude_clients.items()))))},
            :
                "gemini": {}}}}}}}}}}}}}}}}}}}}}}}}}key: {}}}}}}}}}}}}}}}}}}}}}}}}}
                "usage": data[],"usage"],
                "queue_size": len())))data[],"client"].request_queue) if hasattr())))data[],"client"], "request_queue") else 0,::::,,,,
                "current_requests": data[],"client"].current_requests if hasattr())))data[],"client"], "current_requests") else 0,,,,
                } for key, data in self.gemini_clients.items()))))}
                }
        
                return stats

# Example usage:
def example_openai_request())))multiplexer, key_name=None, prompt="Tell me a joke"):
    """Run an example OpenAI chat request"""
    try:
        # Get a client - either specific or using round-robin
        client = multiplexer.get_openai_client())))key_name, strategy="round-robin")
        
        # Make the request
        response = client())))"chat", messages=[],{}}}}}}}}}}}}}}}}}}}}}}}}}"role": "user", "content": prompt}])
        ,,
    return {}}}}}}}}}}}}}}}}}}}}}}}}}
    "success": True,
    "content": response.get())))"text", "No text returned"),
            "queue_size": len())))client.request_queue) if hasattr())))client, "request_queue") else 0,::::
                "current_requests": client.current_requests if hasattr())))client, "current_requests") else 0
        }::
    except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str())))e)
            }

def example_groq_request())))multiplexer, key_name=None, prompt="Tell me a joke"):
    """Run an example Groq chat request"""
    try:
        client = multiplexer.get_groq_client())))key_name, strategy="least-loaded")
        
        response = client())))"chat", messages=[],{}}}}}}}}}}}}}}}}}}}}}}}}}"role": "user", "content": prompt}])
        ,,
    return {}}}}}}}}}}}}}}}}}}}}}}}}}
    "success": True,
    "content": response.get())))"text", "No text returned"),
            "queue_size": len())))client.request_queue) if hasattr())))client, "request_queue") else 0,::::
                "current_requests": client.current_requests if hasattr())))client, "current_requests") else 0
        }::
    except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str())))e)
            }

def load_example_keys())))):
    """Load example API keys from environment variables"""
    multiplexer = ApiKeyMultiplexer()))))
    
    # Load OpenAI keys ())))these are separate API keys from different accounts)
    openai_keys = {}}}}}}}}}}}}}}}}}}}}}}}}}
    "openai_key_1": os.environ.get())))"OPENAI_API_KEY_1", ""),
    "openai_key_2": os.environ.get())))"OPENAI_API_KEY_2", ""),
    "openai_key_3": os.environ.get())))"OPENAI_API_KEY_3", "")
    }
    
    # Load Groq keys
    groq_keys = {}}}}}}}}}}}}}}}}}}}}}}}}}
    "groq_key_1": os.environ.get())))"GROQ_API_KEY_1", ""),
    "groq_key_2": os.environ.get())))"GROQ_API_KEY_2", "")
    }
    
    # Load Claude keys
    claude_keys = {}}}}}}}}}}}}}}}}}}}}}}}}}
    "claude_key_1": os.environ.get())))"ANTHROPIC_API_KEY_1", ""),
    "claude_key_2": os.environ.get())))"ANTHROPIC_API_KEY_2", "")
    }
    
    # Load Gemini keys
    gemini_keys = {}}}}}}}}}}}}}}}}}}}}}}}}}
    "gemini_key_1": os.environ.get())))"GEMINI_API_KEY_1", ""),
    "gemini_key_2": os.environ.get())))"GEMINI_API_KEY_2", "")
    }
    
    # Add keys to multiplexer
    for name, key in openai_keys.items())))):
        if key:
            multiplexer.add_openai_key())))name, key)
    
    for name, key in groq_keys.items())))):
        if key:
            multiplexer.add_groq_key())))name, key)
    
    for name, key in claude_keys.items())))):
        if key:
            multiplexer.add_claude_key())))name, key)
    
    for name, key in gemini_keys.items())))):
        if key:
            multiplexer.add_gemini_key())))name, key)
    
        return multiplexer

def simulate_concurrent_requests())))multiplexer, num_requests=10, include_apis=None):
    """
    Simulate many concurrent requests using multiple API keys
    
    Args:
        multiplexer: The ApiKeyMultiplexer instance
        num_requests: Number of concurrent requests to make
        include_apis: List of API types to include, e.g. [],"openai", "groq", "claude", "gemini"],
        If None, will use all available APIs with keys
        """
        results = [],]
        ,
    # Generate some test prompts
        prompts = [],
        "What is the capital of France?",
        "Write a short poem about a cat",
        "Explain quantum computing in simple terms",
        "Give me a recipe for chocolate cake",
        "Tell me a fun fact about space"
        ]
    
    # Determine which APIs to use
        available_apis = [],]
    ,if include_apis:
        # Use only the specified APIs that have keys
        for api in include_apis:
            if api == "openai" and multiplexer.openai_clients:
                available_apis.append())))"openai")
            elif api == "groq" and multiplexer.groq_clients:
                available_apis.append())))"groq")
            elif api == "claude" and multiplexer.claude_clients:
                available_apis.append())))"claude")
            elif api == "gemini" and multiplexer.gemini_clients:
                available_apis.append())))"gemini")
    else:
        # Use all APIs that have keys
        if multiplexer.openai_clients:
            available_apis.append())))"openai")
        if multiplexer.groq_clients:
            available_apis.append())))"groq")
        if multiplexer.claude_clients:
            available_apis.append())))"claude")
        if multiplexer.gemini_clients:
            available_apis.append())))"gemini")
    
    # If no APIs are available, return early
    if not available_apis:
            return [],"No API keys available for testing"]
    
    def example_claude_request())))multiplexer, key_name=None, prompt="Tell me a joke"):
        """Run an example Claude chat request"""
        try:
            client = multiplexer.get_claude_client())))key_name, strategy="round-robin")
            response = client())))"chat", messages=[],{}}}}}}}}}}}}}}}}}}}}}}}}}"role": "user", "content": prompt}])
            ,,    return {}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "content": response.get())))"text", "No text returned"),
                "queue_size": len())))client.request_queue) if hasattr())))client, "request_queue") else 0,::::
                    "current_requests": client.current_requests if hasattr())))client, "current_requests") else 0
            }::
        except Exception as e:
                return {}}}}}}}}}}}}}}}}}}}}}}}}}
                "success": False,
                "error": str())))e)
                }
    
    def example_gemini_request())))multiplexer, key_name=None, prompt="Tell me a joke"):
        """Run an example Gemini chat request"""
        try:
            client = multiplexer.get_gemini_client())))key_name, strategy="round-robin")
            response = client())))"chat", messages=[],{}}}}}}}}}}}}}}}}}}}}}}}}}"role": "user", "content": prompt}])
            ,,    return {}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "content": response.get())))"text", "No text returned"),
                "queue_size": len())))client.request_queue) if hasattr())))client, "request_queue") else 0,::::
                    "current_requests": client.current_requests if hasattr())))client, "current_requests") else 0
            }::
        except Exception as e:
                return {}}}}}}}}}}}}}}}}}}}}}}}}}
                "success": False,
                "error": str())))e)
                }
    
    def make_random_request())))):
        # Randomly select an API from available ones
        api_type = random.choice())))available_apis)
        prompt = random.choice())))prompts)
        
        if api_type == "openai":
            result = example_openai_request())))multiplexer, prompt=prompt)
            result[],"api"] = "openai"
        elif api_type == "groq":
            result = example_groq_request())))multiplexer, prompt=prompt)
            result[],"api"] = "groq"
        elif api_type == "claude":
            result = example_claude_request())))multiplexer, prompt=prompt)
            result[],"api"] = "claude"
        elif api_type == "gemini":
            result = example_gemini_request())))multiplexer, prompt=prompt)
            result[],"api"] = "gemini"
            
            return result
    
    # Use ThreadPoolExecutor to run requests concurrently
    with ThreadPoolExecutor())))max_workers=num_requests) as executor:
        futures = [],executor.submit())))make_random_request) for _ in range())))num_requests)]:
        for future in futures:
            results.append())))future.result())))))
    
            return results

def main())))):
    # Check if we have any API keys configured
    if ())))not os.environ.get())))"OPENAI_API_KEY_1") and 
    not os.environ.get())))"GROQ_API_KEY_1") and
        not os.environ.get())))"ANTHROPIC_API_KEY_1") and :
        not os.environ.get())))"GEMINI_API_KEY_1")):
            print())))"No API keys found in environment variables.")
            print())))"Please set API keys in your .env file. Example:")
            print())))"OPENAI_API_KEY_1=sk-your-key")
            print())))"OPENAI_API_KEY_2=sk-your-second-key")
            print())))"GROQ_API_KEY_1=gsk-your-key")
            return
    
    # Load multiplexer with keys
            multiplexer = load_example_keys()))))
    
    # Get stats before running tests
            print())))"\nAPI Key Counts:")
            stats = multiplexer.get_usage_stats()))))
            print())))f"OpenAI keys: {}}}}}}}}}}}}}}}}}}}}}}}}}len())))stats[],'openai'])}")
            print())))f"Groq keys: {}}}}}}}}}}}}}}}}}}}}}}}}}len())))stats[],'groq'])}")
            print())))f"Claude keys: {}}}}}}}}}}}}}}}}}}}}}}}}}len())))stats[],'claude'])}")
            print())))f"Gemini keys: {}}}}}}}}}}}}}}}}}}}}}}}}}len())))stats[],'gemini'])}")
    
    # If we have at least one key for any provider, run example requests
    total_keys = sum())))len())))keys) for keys in stats.values())))))::
    if total_keys > 0:
        print())))"\nRunning example requests...")
        
        # Run simulation with concurrent requests
        num_requests = min())))20, total_keys * 4)  # 4 requests per key, max 20
        results = simulate_concurrent_requests())))multiplexer, num_requests=num_requests)
        
        # Count successes and failures
        if isinstance())))results, list) and all())))isinstance())))r, dict) for r in results):
            successes = sum())))1 for r in results if r.get())))"success"))
            failures = len())))results) - successes:
                print())))f"\nResults: {}}}}}}}}}}}}}}}}}}}}}}}}}successes} successful requests, {}}}}}}}}}}}}}}}}}}}}}}}}}failures} failures")
        else:
            print())))f"\nResults: {}}}}}}}}}}}}}}}}}}}}}}}}}results}")
        
        # Show usage stats after tests
            print())))"\nUsage Statistics:")
            final_stats = multiplexer.get_usage_stats()))))
        
        for provider, keys in final_stats.items())))):
            if keys:
                print())))f"\n{}}}}}}}}}}}}}}}}}}}}}}}}}provider.upper()))))} API Keys:")
                for key_name, data in keys.items())))):
                    print())))f"  {}}}}}}}}}}}}}}}}}}}}}}}}}key_name}: {}}}}}}}}}}}}}}}}}}}}}}}}}data[],'usage']} requests, {}}}}}}}}}}}}}}}}}}}}}}}}}data[],'current_requests']} active, {}}}}}}}}}}}}}}}}}}}}}}}}}data[],'queue_size']} queued")
    else:
        print())))"\nNo API keys available for testing. Please add keys to your .env file.")


def test_in_ipfs_accelerate_environment())))resources=None, metadata=None):
    """
    Integration function for testing within the test_ipfs_accelerate environment.
    This is intended to be called as the final test when testing the API backends.
    
    Args:
        resources: Resources dictionary from test_ipfs_accelerate
        metadata: Metadata dictionary from test_ipfs_accelerate
        
    Returns:
        dict: Test results and statistics
        """
        results = {}}}}}}}}}}}}}}}}}}}}}}}}}
        "api_key_multiplexing": {}}}}}}}}}}}}}}}}}}}}}}}}}
        "status": "Testing",
        "timestamp": time.strftime())))"%Y-%m-%d %H:%M:%S"),
        "results": {}}}}}}}}}}}}}}}}}}}}}}}}}}
        }
        }
    
    try:
        # Initialize the multiplexer
        multiplexer = ApiKeyMultiplexer()))))
        
        # Add keys from metadata if present:
        if metadata:
            # OpenAI keys
            if "openai_api_key" in metadata and metadata[],"openai_api_key"]:
                multiplexer.add_openai_key())))"openai_main", metadata[],"openai_api_key"])
            if "openai_api_key_1" in metadata and metadata[],"openai_api_key_1"]:
                multiplexer.add_openai_key())))"openai_1", metadata[],"openai_api_key_1"])
            if "openai_api_key_2" in metadata and metadata[],"openai_api_key_2"]:
                multiplexer.add_openai_key())))"openai_2", metadata[],"openai_api_key_2"])
                
            # Groq keys
            if "groq_api_key" in metadata and metadata[],"groq_api_key"]:
                multiplexer.add_groq_key())))"groq_main", metadata[],"groq_api_key"])
            if "groq_api_key_1" in metadata and metadata[],"groq_api_key_1"]:
                multiplexer.add_groq_key())))"groq_1", metadata[],"groq_api_key_1"])
            if "groq_api_key_2" in metadata and metadata[],"groq_api_key_2"]:
                multiplexer.add_groq_key())))"groq_2", metadata[],"groq_api_key_2"])
                
            # Claude keys
            if "claude_api_key" in metadata and metadata[],"claude_api_key"]:
                multiplexer.add_claude_key())))"claude_main", metadata[],"claude_api_key"])
            if "anthropic_api_key_1" in metadata and metadata[],"anthropic_api_key_1"]:
                multiplexer.add_claude_key())))"claude_1", metadata[],"anthropic_api_key_1"])
            if "anthropic_api_key_2" in metadata and metadata[],"anthropic_api_key_2"]:
                multiplexer.add_claude_key())))"claude_2", metadata[],"anthropic_api_key_2"])
                
            # Gemini keys
            if "gemini_api_key" in metadata and metadata[],"gemini_api_key"]:
                multiplexer.add_gemini_key())))"gemini_main", metadata[],"gemini_api_key"])
            if "gemini_api_key_1" in metadata and metadata[],"gemini_api_key_1"]:
                multiplexer.add_gemini_key())))"gemini_1", metadata[],"gemini_api_key_1"])
            if "gemini_api_key_2" in metadata and metadata[],"gemini_api_key_2"]:
                multiplexer.add_gemini_key())))"gemini_2", metadata[],"gemini_api_key_2"])
        
        # Load from environment if nothing in metadata
                if ())))len())))multiplexer.openai_clients) == 0 and
                len())))multiplexer.groq_clients) == 0 and
            len())))multiplexer.claude_clients) == 0 and:
            len())))multiplexer.gemini_clients) == 0):
            # Try to load from environment
                multiplexer = load_example_keys()))))
        
        # Get the API key statistics
                stats = multiplexer.get_usage_stats()))))
                results[],"api_key_multiplexing"][],"key_counts"] = {}}}}}}}}}}}}}}}}}}}}}}}}}
                "openai": len())))stats[],"openai"]),
                "groq": len())))stats[],"groq"]),
                "claude": len())))stats[],"claude"]),
                "gemini": len())))stats[],"gemini"]),
            "total": sum())))len())))keys) for keys in stats.values())))))::
                }
        
        # Test multiplexing if we have any keys
        total_keys = results[],"api_key_multiplexing"][],"key_counts"][],"total"]:
        if total_keys > 0:
            # Run a few concurrent requests to test multiplexing
            num_requests = min())))10, total_keys * 2)  # 2 requests per key, max 10
            test_results = simulate_concurrent_requests())))multiplexer, num_requests=num_requests)
            
            # Process results
            if isinstance())))test_results, list) and all())))isinstance())))r, dict) for r in test_results):
                successes = sum())))1 for r in test_results if r.get())))"success"))
                failures = len())))test_results) - successes
                
                # Record results
                results[],"api_key_multiplexing"][],"results"][],"success_count"] = successes
                results[],"api_key_multiplexing"][],"results"][],"failure_count"] = failures
                results[],"api_key_multiplexing"][],"results"][],"total_requests"] = len())))test_results)
                
                # Get final usage statistics
                final_stats = multiplexer.get_usage_stats()))))
                results[],"api_key_multiplexing"][],"results"][],"usage_stats"] = {}}}}}}}}}}}}}}}}}}}}}}}}}}
                :
                for provider, keys in final_stats.items())))):
                    if keys:
                        provider_stats = {}}}}}}}}}}}}}}}}}}}}}}}}}}
                        for key_name, data in keys.items())))):
                            provider_stats[],key_name] = {}}}}}}}}}}}}}}}}}}}}}}}}},,,,
                            "usage": data[],"usage"],
                            "queue_size": data[],"queue_size"],
                            "current_requests": data[],"current_requests"]
                            }
                            results[],"api_key_multiplexing"][],"results"][],"usage_stats"][],provider] = provider_stats
                
                # Set overall status
                if successes > 0:
                    results[],"api_key_multiplexing"][],"status"] = "Success"
                else:
                    results[],"api_key_multiplexing"][],"status"] = "Failed - No successful requests"
            else:
                results[],"api_key_multiplexing"][],"status"] = "Failed - No valid results"
                results[],"api_key_multiplexing"][],"results"][],"message"] = str())))test_results)
        else:
            results[],"api_key_multiplexing"][],"status"] = "Skipped - No API keys available"
    
    except Exception as e:
        results[],"api_key_multiplexing"][],"status"] = "Error"
        results[],"api_key_multiplexing"][],"error"] = str())))e)
        import traceback
        results[],"api_key_multiplexing"][],"traceback"] = traceback.format_exc()))))
    
            return results

if __name__ == "__main__":
    main()))))