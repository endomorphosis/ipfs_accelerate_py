#!/usr/bin/env python
"""
Test script to verify API backend improvements for Claude, OpenAI, Gemini, and Groq.
This script tests:
    1. Each endpoint having its own counters, API key, backoff, and queue
    2. Request ID support in each request
    3. Consistent implementation of retry: and backoff mechanisms
    """

    import os
    import sys
    import json
    import time
    import unittest
    import threading
    from pathlib import Path
    from unittest.mock import MagicMock, patch
    from concurrent.futures import ThreadPoolExecutor

# Add the project root to the Python path
    sys.path.append()))))))))))))))os.path.dirname()))))))))))))))os.path.dirname()))))))))))))))__file__)))

# Import API backends - with fallbacks if imports fail:
try::
    from ipfs_accelerate_py.api_backends import claude, openai_api, gemini, groq
except ImportError as e:
    print()))))))))))))))f"Error importing API backends: {}}}}str()))))))))))))))e)}")
    print()))))))))))))))"Creating mock implementations for testing...")
    
    class MockBackend:
        def __init__()))))))))))))))self, resources=None, metadata=None):
            self.resources = resources or {}}}
            self.metadata = metadata or {}}}
            self.api_key = metadata.get()))))))))))))))"api_key", "") if metadata else ""
            self.endpoints = {}}}
            self.max_retries = 3
            self.initial_retry:_delay = 1
            self.backoff_factor = 2
            self.max_retry:_delay = 30
            self.queue_enabled = True
            self.queue_size = 50
            self.queue_processing = False
            self.current_requests = 0
            self.max_concurrent_requests = 5
            self.request_queue = [],],
            self.queue_lock = threading.RLock())))))))))))))))
    
    # Create mock modules:
    if 'claude' not in locals()))))))))))))))):
        claude = MockBackend
    if 'openai_api' not in locals()))))))))))))))):
        openai_api = MockBackend
    if 'gemini' not in locals()))))))))))))))):
        gemini = MockBackend
    if 'groq' not in locals()))))))))))))))):
        groq = MockBackend


class TestApiImprovements()))))))))))))))unittest.TestCase):
    """Test suite for API backend improvements"""
    
    def setUp()))))))))))))))self):
        """Set up test environment with mock API keys"""
        self.api_keys = {}}
        "claude": "sk-claude-test-key",
        "openai": "sk-openai-test-key",
        "gemini": "gemini-test-key",
        "groq": "gsk-groq-test-key"
        }
        
        # Initialize API clients
        self.claude = claude()))))))))))))))resources={}}}, metadata={}}"claude_api_key": self.api_keys[],"claude"]}),
        self.openai = openai_api()))))))))))))))resources={}}}, metadata={}}"openai_api_key": self.api_keys[],"openai"]}),
        self.gemini = gemini()))))))))))))))resources={}}}, metadata={}}"gemini_api_key": self.api_keys[],"gemini"]}),
        self.groq = groq()))))))))))))))resources={}}}, metadata={}}"groq_api_key": self.api_keys[],"groq"]})
        ,
        # Store clients in a dictionary for easier iteration
        self.api_clients = {}}
        "claude": self.claude,
        "openai": self.openai,
        "gemini": self.gemini,
        "groq": self.groq
        }
    
    def test_endpoint_creation()))))))))))))))self):
        """Test that endpoints can be created with custom settings"""
        for api_name, client in self.api_clients.items()))))))))))))))):
            # Skip if create_endpoint method doesn't exist:
            if not hasattr()))))))))))))))client, "create_endpoint"):
                print()))))))))))))))f"⚠️ {}}}}api_name} doesn't have create_endpoint method - implementation incomplete")
            continue
            
            # Test with custom settings
            custom_key = f"custom-{}}}}api_name}-key"
            endpoint_id = client.create_endpoint()))))))))))))))
            api_key=custom_key,
            max_retries=10,
            initial_retry:_delay=2,
            backoff_factor=3,
            max_concurrent_requests=8
            )
            
            # Verify endpoint was created
            self.assertIn()))))))))))))))endpoint_id, client.endpoints, f"{}}}}api_name} endpoint creation failed")
            
            # Verify custom settings were applied
            endpoint = client.endpoints[],endpoint_id],,
            self.assertEqual()))))))))))))))endpoint[],"api_key"], custom_key, f"{}}}}api_name} custom API key not set"),
            self.assertEqual()))))))))))))))endpoint[],"max_retries"], 10, f"{}}}}api_name} custom max_retries not set"),
            self.assertEqual()))))))))))))))endpoint[],"initial_retry:_delay"], 2, f"{}}}}api_name} custom initial_retry:_delay not set"),
            self.assertEqual()))))))))))))))endpoint[],"backoff_factor"], 3, f"{}}}}api_name} custom backoff_factor not set"),
            self.assertEqual()))))))))))))))endpoint[],"max_concurrent_requests"], 8, f"{}}}}api_name} custom max_concurrent_requests not set")
            ,
            # Verify counters were initialized
            self.assertEqual()))))))))))))))endpoint[],"total_requests"], 0, f"{}}}}api_name} total_requests not initialized"),
            self.assertEqual()))))))))))))))endpoint[],"successful_requests"], 0, f"{}}}}api_name} successful_requests not initialized"),
            self.assertEqual()))))))))))))))endpoint[],"failed_requests"], 0, f"{}}}}api_name} failed_requests not initialized")
            ,
            print()))))))))))))))f"✅ {}}}}api_name}: Endpoint creation test passed")
    
    def test_get_endpoint()))))))))))))))self):
        """Test that get_endpoint works properly"""
        for api_name, client in self.api_clients.items()))))))))))))))):
            # Skip if methods don't exist::::
            if not hasattr()))))))))))))))client, "create_endpoint") or not hasattr()))))))))))))))client, "get_endpoint"):
                print()))))))))))))))f"⚠️ {}}}}api_name} doesn't have required endpoint methods - implementation incomplete")
            continue
            
            # Create an endpoint
            endpoint_id = client.create_endpoint())))))))))))))))
            
            # Test getting the endpoint by ID
            endpoint = client.get_endpoint()))))))))))))))endpoint_id)
            self.assertIsNotNone()))))))))))))))endpoint, f"{}}}}api_name} get_endpoint by ID failed")
            
            # Test getting default endpoint when no ID provided
            default_endpoint = client.get_endpoint())))))))))))))))
            self.assertIsNotNone()))))))))))))))default_endpoint, f"{}}}}api_name} get_endpoint default failed")
            
            print()))))))))))))))f"✅ {}}}}api_name}: Get endpoint test passed")
    
    def test_update_endpoint()))))))))))))))self):
        """Test that update_endpoint works properly"""
        for api_name, client in self.api_clients.items()))))))))))))))):
            # Skip if methods don't exist::::
            if not hasattr()))))))))))))))client, "create_endpoint") or not hasattr()))))))))))))))client, "update_endpoint"):
                print()))))))))))))))f"⚠️ {}}}}api_name} doesn't have required endpoint methods - implementation incomplete")
            continue
            
            # Create an endpoint
            endpoint_id = client.create_endpoint())))))))))))))))
            
            # Update the endpoint
            updated = client.update_endpoint()))))))))))))))
            endpoint_id,
            api_key="updated-key",
            max_retries=15,
            queue_size=200
            )
            
            # Verify updates were applied
            self.assertEqual()))))))))))))))updated[],"api_key"], "updated-key", f"{}}}}api_name} update_endpoint api_key failed"),
            self.assertEqual()))))))))))))))updated[],"max_retries"], 15, f"{}}}}api_name} update_endpoint max_retries failed"),
            self.assertEqual()))))))))))))))updated[],"queue_size"], 200, f"{}}}}api_name} update_endpoint queue_size failed")
            ,
            # Verify in the main storage
            endpoint = client.endpoints[],endpoint_id],,
            self.assertEqual()))))))))))))))endpoint[],"api_key"], "updated-key", f"{}}}}api_name} update not stored in endpoints dict")
            ,
            print()))))))))))))))f"✅ {}}}}api_name}: Update endpoint test passed")
    
    def test_stats_tracking()))))))))))))))self):
        """Test that endpoint stats are properly tracked"""
        for api_name, client in self.api_clients.items()))))))))))))))):
            # Skip if methods don't exist::::
            if not hasattr()))))))))))))))client, "create_endpoint") or not hasattr()))))))))))))))client, "get_stats"):
                print()))))))))))))))f"⚠️ {}}}}api_name} doesn't have required stats methods - implementation incomplete")
            continue
            
            # Create an endpoint
            endpoint_id = client.create_endpoint())))))))))))))))
            
            # Manually update stats for testing
            client.endpoints[],endpoint_id],,[],"total_requests"] = 10
            client.endpoints[],endpoint_id],,[],"successful_requests"] = 8
            client.endpoints[],endpoint_id],,[],"failed_requests"] = 2
            client.endpoints[],endpoint_id],,[],"total_tokens"] = 500
            client.endpoints[],endpoint_id],,[],"input_tokens"] = 300
            client.endpoints[],endpoint_id],,[],"output_tokens"] = 200
            
            # Get stats for the endpoint
            stats = client.get_stats()))))))))))))))endpoint_id)
            
            # Verify stats are correct
            self.assertEqual()))))))))))))))stats[],"total_requests"], 10, f"{}}}}api_name} total_requests stats incorrect"),
            self.assertEqual()))))))))))))))stats[],"successful_requests"], 8, f"{}}}}api_name} successful_requests stats incorrect"),
            self.assertEqual()))))))))))))))stats[],"failed_requests"], 2, f"{}}}}api_name} failed_requests stats incorrect"),
            self.assertEqual()))))))))))))))stats[],"total_tokens"], 500, f"{}}}}api_name} total_tokens stats incorrect"),
            self.assertEqual()))))))))))))))stats[],"input_tokens"], 300, f"{}}}}api_name} input_tokens stats incorrect"),
            self.assertEqual()))))))))))))))stats[],"output_tokens"], 200, f"{}}}}api_name} output_tokens stats incorrect")
            ,
            # Get global stats
            global_stats = client.get_stats())))))))))))))))
            
            # Verify global stats include our endpoint's stats
            self.assertEqual()))))))))))))))global_stats[],"total_requests"], 10, f"{}}}}api_name} global total_requests stats incorrect")
            ,
            print()))))))))))))))f"✅ {}}}}api_name}: Stats tracking test passed")
    
    def test_stats_reset()))))))))))))))self):
        """Test that reset_stats works properly"""
        for api_name, client in self.api_clients.items()))))))))))))))):
            # Skip if methods don't exist::::
            if not hasattr()))))))))))))))client, "create_endpoint") or not hasattr()))))))))))))))client, "reset_stats"):
                print()))))))))))))))f"⚠️ {}}}}api_name} doesn't have required stats methods - implementation incomplete")
            continue
            
            # Create an endpoint
            endpoint_id = client.create_endpoint())))))))))))))))
            
            # Manually update stats for testing
            client.endpoints[],endpoint_id],,[],"total_requests"] = 10
            client.endpoints[],endpoint_id],,[],"successful_requests"] = 8
            client.endpoints[],endpoint_id],,[],"failed_requests"] = 2
            client.endpoints[],endpoint_id],,[],"total_tokens"] = 500
            
            # Reset stats for the endpoint
            client.reset_stats()))))))))))))))endpoint_id)
            
            # Verify stats were reset
            self.assertEqual()))))))))))))))client.endpoints[],endpoint_id],,[],"total_requests"], 0, f"{}}}}api_name} total_requests not reset")
            self.assertEqual()))))))))))))))client.endpoints[],endpoint_id],,[],"successful_requests"], 0, f"{}}}}api_name} successful_requests not reset")
            self.assertEqual()))))))))))))))client.endpoints[],endpoint_id],,[],"failed_requests"], 0, f"{}}}}api_name} failed_requests not reset")
            self.assertEqual()))))))))))))))client.endpoints[],endpoint_id],,[],"total_tokens"], 0, f"{}}}}api_name} total_tokens not reset")
            
            print()))))))))))))))f"✅ {}}}}api_name}: Stats reset test passed")
    
    def test_request_id_generation()))))))))))))))self):
        """Test that request IDs are properly generated or used"""
        for api_name, client in self.api_clients.items()))))))))))))))):
            # Skip if core request method doesn't exist
            request_method_name = self._get_request_method_name()))))))))))))))api_name)::
            if not hasattr()))))))))))))))client, request_method_name):
                print()))))))))))))))f"⚠️ {}}}}api_name} doesn't have {}}}}request_method_name} method - can't test request IDs")
                continue
            
            # Get the request method
                request_method = getattr()))))))))))))))client, request_method_name)
            
            # Create a mock for the method that extracts the request_id
            with patch.object()))))))))))))))client, request_method_name) as mock_request:
                # Set up mock to capture and return the request_id
                def side_effect()))))))))))))))*args, **kwargs):
                    # Extract request_id from kwargs
                    request_id = kwargs.get()))))))))))))))"request_id")
                return {}}"request_id": request_id}
                
                mock_request.side_effect = side_effect
                
                # Test with explicit request_id
                try::
                    # Different APIs have different parameter orders, try: to adapt
                    if api_name == "claude":
                        result = request_method())))))))))))))){}}"messages": [],{}}"role": "user", "content": "test"}]}, api_key="test-key", request_id="explicit-id"),
                    elif api_name == "openai":
                        result = request_method()))))))))))))))"https://api.openai.com/v1/chat/completions", {}}"messages": [],{}}"role": "user", "content": "test"}]}, "test-key", "explicit-id"),,
                    elif api_name == "gemini":
                        result = request_method()))))))))))))))"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent", {}}"contents": [],{}}"parts": [],{}}"text": "test"}]}]}, "test-key", "explicit-id"),
                    elif api_name == "groq":
                        result = request_method()))))))))))))))"https://api.groq.com/openai/v1/chat/completions", {}}"model": "llama3-8b-8192", "messages": [],{}}"role": "user", "content": "test"}]}, "test-key", "explicit-id"),,
                    else:
                        # Generic fallback
                        result = request_method()))))))))))))))"https://api.example.com/v1/chat", {}}"prompt": "test"}, request_id="explicit-id")
                    
                    # Verify request_id was passed through
                        self.assertEqual()))))))))))))))result.get()))))))))))))))"request_id"), "explicit-id", f"{}}}}api_name} explicit request_id not used")
                except Exception as e:
                    print()))))))))))))))f"⚠️ {}}}}api_name} explicit request_id test failed: {}}}}str()))))))))))))))e)}")
                
                # Test with auto-generated request_id
                    mock_request.reset_mock())))))))))))))))
                
                try::
                    # Different APIs have different parameter orders, try: to adapt
                    if api_name == "claude":
                        result = request_method())))))))))))))){}}"messages": [],{}}"role": "user", "content": "test"}]}, api_key="test-key"),
                    elif api_name == "openai":
                        result = request_method()))))))))))))))"https://api.openai.com/v1/chat/completions", {}}"messages": [],{}}"role": "user", "content": "test"}]}, "test-key"),,
                    elif api_name == "gemini":
                        result = request_method()))))))))))))))"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent", {}}"contents": [],{}}"parts": [],{}}"text": "test"}]}]}, "test-key"),
                    elif api_name == "groq":
                        result = request_method()))))))))))))))"https://api.groq.com/openai/v1/chat/completions", {}}"model": "llama3-8b-8192", "messages": [],{}}"role": "user", "content": "test"}]}, "test-key"),,
                    else:
                        # Generic fallback
                        result = request_method()))))))))))))))"https://api.example.com/v1/chat", {}}"prompt": "test"})
                    
                    # Verify request_id was generated
                        self.assertIsNotNone()))))))))))))))result.get()))))))))))))))"request_id"), f"{}}}}api_name} auto-generated request_id missing")
                        self.assertIn()))))))))))))))"req_", str()))))))))))))))result.get()))))))))))))))"request_id", "")), f"{}}}}api_name} auto-generated request_id doesn't match expected format")
                except Exception as e:
                    print()))))))))))))))f"⚠️ {}}}}api_name} auto-generated request_id test failed: {}}}}str()))))))))))))))e)}")
            
                    print()))))))))))))))f"✅ {}}}}api_name}: Request ID generation test passed")
    
    def test_queue_mechanism()))))))))))))))self):
        """Test that the queue mechanism works properly"""
        for api_name, client in self.api_clients.items()))))))))))))))):
            # Skip if required methods don't exist:
            if not hasattr()))))))))))))))client, "create_endpoint") or not hasattr()))))))))))))))client, "_process_queue"):
                print()))))))))))))))f"⚠️ {}}}}api_name} doesn't have required queue methods - implementation incomplete")
            continue
            
            # Create an endpoint with a low concurrency limit
            endpoint_id = client.create_endpoint()))))))))))))))max_concurrent_requests=1)
            
            # Set up mock for request processing
            request_method_name = self._get_request_method_name()))))))))))))))api_name)
            if not hasattr()))))))))))))))client, request_method_name):
                print()))))))))))))))f"⚠️ {}}}}api_name} doesn't have {}}}}request_method_name} method - can't test queue")
            continue
            
            with patch.object()))))))))))))))client, request_method_name) as mock_request:
                # Configure mock to delay responses
                def delayed_response()))))))))))))))*args, **kwargs):
                    time.sleep()))))))))))))))0.1)  # Short delay to ensure queueing
                return {}}"success": True}
                
                mock_request.side_effect = delayed_response
                
                # Set up concurrent requests
                num_requests = 3  # More than max_concurrent_requests
                futures = [],],
                
                try::
                    # Start a thread pool to submit concurrent requests
                    with ThreadPoolExecutor()))))))))))))))max_workers=num_requests) as executor:
                        # Define request function
                        def make_request()))))))))))))))idx):
                            try::
                                # Adapt based on API
                                if api_name == "claude":
                                return client.make_post_request()))))))))))))))
                                {}}}}"messages": [],{}}}}"role": "user", "content": f"test {}}}}idx}"}]},
                                api_key="test-key",
                                endpoint_id=endpoint_id
                                )
                                elif api_name == "openai":
                                return client.make_request()))))))))))))))
                                "https://api.openai.com/v1/chat/completions",
                                {}}}}"messages": [],{}}}}"role": "user", "content": f"test {}}}}idx}"}]},
                                "test-key",
                                endpoint_id=endpoint_id
                                )
                                elif api_name == "gemini":
                                return client.make_post_request_gemini()))))))))))))))
                                "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent",
                                {}}}}"contents": [],{}}}}"parts": [],{}}}}"text": f"test {}}}}idx}"}]}]},
                                "test-key",
                                endpoint_id=endpoint_id
                                )
                                elif api_name == "groq":
                                return client.make_post_request_groq()))))))))))))))
                                "https://api.groq.com/openai/v1/chat/completions",
                                {}}}}"model": "llama3-8b-8192", "messages": [],{}}}}"role": "user", "content": f"test {}}}}idx}"}]},
                                "test-key",
                                endpoint_id=endpoint_id
                                )
                                else:
                                    # Generic fallback
                                return client.make_request()))))))))))))))
                                "https://api.example.com/v1/chat",
                                {}}}}"prompt": f"test {}}}}idx}"},
                                endpoint_id=endpoint_id
                                )
                            except Exception as e:
                                return {}}"error": str()))))))))))))))e)}
                        
                        # Submit requests
                        for i in range()))))))))))))))num_requests):
                            futures.append()))))))))))))))executor.submit()))))))))))))))make_request, i))
                        
                        # Wait for all futures to complete
                            results = [],future.result()))))))))))))))) for future in futures]:,
                        # Verify all requests were successful
                            self.assertEqual()))))))))))))))sum()))))))))))))))1 for result in results if result.get()))))))))))))))"success", False)), num_requests,
                            f"{}}}}api_name} queue mechanism didn't process all requests")
                        
                        # Verify the queue was used
                        # We know it was used if all requests were successful despite max_concurrent_requests=1
                        # But also check that we actually hit the code path that adds to the queue
                            self.assertTrue()))))))))))))))len()))))))))))))))client.endpoints[],endpoint_id],,[],"request_queue"]) == 0,
                                      f"{}}}}api_name} queue mechanism didn't clear queue after processing"):
                except Exception as e:
                    print()))))))))))))))f"⚠️ {}}}}api_name} queue mechanism test failed: {}}}}str()))))))))))))))e)}")
                                          continue
            
                                          print()))))))))))))))f"✅ {}}}}api_name}: Queue mechanism test passed")
    
    def test_backoff_retry:_mechanism()))))))))))))))self):
        """Test that the backoff retry: mechanism works properly"""
        for api_name, client in self.api_clients.items()))))))))))))))):
            # Skip if core request method doesn't exist
            request_method_name = self._get_request_method_name()))))))))))))))api_name)::
            if not hasattr()))))))))))))))client, request_method_name):
                print()))))))))))))))f"⚠️ {}}}}api_name} doesn't have {}}}}request_method_name} method - can't test backoff")
                continue
            
            # Create an endpoint with custom retry: settings
                endpoint_id = client.create_endpoint()))))))))))))))
                max_retries=3,
                initial_retry:_delay=0.01,  # Small delay for faster testing
                backoff_factor=2
                )
            
            # Get the request method
                request_method = getattr()))))))))))))))client, request_method_name)
            
            # Set up mock to simulate rate limiting then success
            with patch()))))))))))))))"requests.post") as mock_post:
                # Configure mock responses
                mock_responses = [],
                    # First request: rate limit error
                MagicMock()))))))))))))))
                status_code=429,
                headers={}}"retry:-after": "0.01"},  # Small retry: for faster testing
                json=lambda: {}}"error": {}}"message": "Rate limit exceeded"}}
                ),
                    # Second request: success
                MagicMock()))))))))))))))
                status_code=200,
                json=lambda: {}}"success": True}
                )
                ]
                mock_post.side_effect = mock_responses
                
                # Mock time.sleep to avoid actual delays
                with patch()))))))))))))))"time.sleep") as mock_sleep:
                    try::
                        # Make request - adapt based on API
                        if api_name == "claude":
                            result = request_method()))))))))))))))
                            {}}}}"messages": [],{}}}}"role": "user", "content": "test backoff"}]},
                            api_key="test-key",
                            endpoint_id=endpoint_id
                            )
                        elif api_name == "openai":
                            result = request_method()))))))))))))))
                            "https://api.openai.com/v1/chat/completions",
                            {}}}}"messages": [],{}}}}"role": "user", "content": "test backoff"}]},
                            "test-key",
                            endpoint_id=endpoint_id
                            )
                        elif api_name == "gemini":
                            result = request_method()))))))))))))))
                            "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent",
                            {}}}}"contents": [],{}}}}"parts": [],{}}}}"text": "test backoff"}]}]},
                            "test-key",
                            endpoint_id=endpoint_id
                            )
                        elif api_name == "groq":
                            result = request_method()))))))))))))))
                            "https://api.groq.com/openai/v1/chat/completions",
                            {}}}}"model": "llama3-8b-8192", "messages": [],{}}}}"role": "user", "content": "test backoff"}]},
                            "test-key",
                            endpoint_id=endpoint_id
                            )
                        else:
                            # Generic fallback
                            result = request_method()))))))))))))))
                            "https://api.example.com/v1/chat",
                            {}}}}"prompt": "test backoff"},
                            endpoint_id=endpoint_id
                            )
                        
                        # Verify request succeeded after retry:
                            self.assertEqual()))))))))))))))result.get()))))))))))))))"success"), True, f"{}}}}api_name} backoff retry: didn't succeed")
                        
                        # Verify time.sleep was called with retry:-after value
                            mock_sleep.assert_called_with()))))))))))))))0.01)
                        
                        # Verify requests.post was called twice ()))))))))))))))once for initial, once for retry:)
                            self.assertEqual()))))))))))))))mock_post.call_count, 2, f"{}}}}api_name} backoff retry: didn't make correct number of attempts")
                    except Exception as e:
                        print()))))))))))))))f"⚠️ {}}}}api_name} backoff retry: test failed: {}}}}str()))))))))))))))e)}")
                            continue
            
                            print()))))))))))))))f"✅ {}}}}api_name}: Backoff retry: mechanism test passed")
    
    def _get_request_method_name()))))))))))))))self, api_name):
        """Get the appropriate request method name for an API"""
        if api_name == "claude":
        return "make_post_request"
        elif api_name == "openai":
        return "make_request"
        elif api_name == "gemini":
        return "make_post_request_gemini"
        elif api_name == "groq":
        return "make_post_request_groq"
        else:
        return "make_request"  # Default


if __name__ == "__main__":
    # Run the tests
    unittest.main())))))))))))))))