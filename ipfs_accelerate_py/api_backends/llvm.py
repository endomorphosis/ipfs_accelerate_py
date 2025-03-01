import os
import json
import time
import threading
import hashlib
import uuid
import requests
from concurrent.futures import Future
from queue import Queue
from dotenv import load_dotenv

class llvm:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources if resources else {}
        self.metadata = metadata if metadata else {}
        
        # Get LLVM API endpoint from metadata or environment
        self.api_endpoint = self._get_api_endpoint()
        
        # Initialize queue and backoff systems
        self.max_concurrent_requests = 5
        self.queue_size = 100
        self.request_queue = Queue(maxsize=self.queue_size)
        self.active_requests = 0
        self.queue_lock = threading.RLock()
        
        # Start queue processor
        self.queue_processor = threading.Thread(target=self._process_queue)
        self.queue_processor.daemon = True
        self.queue_processor.start()
        
        # Initialize backoff configuration
        self.max_retries = 5
        self.initial_retry_delay = 1
        self.backoff_factor = 2
        self.max_retry_delay = 16
        
        # Request tracking
        self.request_tracking = True
        self.recent_requests = {}
        
        
        # Retry and backoff settings
        self.max_retries = 5
        self.initial_retry_delay = 1
        self.backoff_factor = 2
        self.max_retry_delay = 60  # Maximum delay in seconds
        
        # Request queue settings
        self.queue_enabled = True
        self.queue_size = 100
        self.queue_processing = False
        self.current_requests = 0
        self.max_concurrent_requests = 5
        self.request_queue = []
        self.queue_lock = threading.RLock()
        return None

    def _get_api_endpoint(self):
        """Get LLVM API endpoint from metadata or environment"""
        # Try to get from metadata
        api_endpoint = self.metadata.get("llvm_endpoint")
        if api_endpoint:
            return api_endpoint
        
        # Try to get from environment
        env_endpoint = os.environ.get("LLVM_API_ENDPOINT")
        if env_endpoint:
            return env_endpoint
        
        # Try to load from dotenv
        try:
            load_dotenv()
            env_endpoint = os.environ.get("LLVM_API_ENDPOINT")
            if env_endpoint:
                return env_endpoint
        except ImportError:
            pass
        
        # Return default if no endpoint found
        return "http://localhost:8080/v1"
        
    def _process_queue(self):
        """Process queued requests with proper concurrency management"""
        while True:
            try:
                future, endpoint_url, data, request_id = self.request_queue.get()
                
                with self.queue_lock:
                    self.active_requests += 1
                
                # Process with retry logic
                retry_count = 0
                while retry_count <= self.max_retries:
                    try:
                        # Construct headers
                        headers = {"Content-Type": "application/json"}
                        
                        # Make request with proper error handling
                        response = requests.post(
                            endpoint_url,
                            json=data,
                            headers=headers,
                            timeout=self.metadata.get("timeout", 30)
                        )
                        
                        # Check for HTTP errors
                        response.raise_for_status()
                        
                        # Parse JSON response
                        result = response.json()
                        
                        # Update tracking with response
                        if self.request_tracking:
                            self.recent_requests[request_id] = {
                                "timestamp": time.time(),
                                "endpoint": endpoint_url,
                                "status": "success",
                                "response_code": response.status_code
                            }
                        
                        future.set_result(result)
                        break
                        
                    except requests.RequestException as e:
                        retry_count += 1
                        
                        if retry_count > self.max_retries:
                            # Update tracking with error
                            if self.request_tracking:
                                self.recent_requests[request_id] = {
                                    "timestamp": time.time(),
                                    "endpoint": endpoint_url,
                                    "status": "error",
                                    "error": str(e)
                                }
                            
                            future.set_exception(e)
                            break
                        
                        # Calculate backoff delay
                        delay = min(
                            self.initial_retry_delay * (self.backoff_factor ** (retry_count - 1)),
                            self.max_retry_delay
                        )
                        
                        # Sleep with backoff
                        time.sleep(delay)
                    
                    except Exception as e:
                        # Update tracking with error
                        if self.request_tracking:
                            self.recent_requests[request_id] = {
                                "timestamp": time.time(),
                                "endpoint": endpoint_url,
                                "status": "error",
                                "error": str(e)
                            }
                        
                        future.set_exception(e)
                        break
                
                with self.queue_lock:
                    self.active_requests -= 1
                
                self.request_queue.task_done()
                
            except Exception as e:
                print(f"Error in queue processor: {e}")
    
    def make_post_request_llvm(self, endpoint_url, data, request_id=None):
        """Make a request to LLVM API with queue and backoff"""
        # Generate unique request ID if not provided
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        # Queue system with proper concurrency management
        future = Future()
        
        # Add to queue
        self.request_queue.put((future, endpoint_url, data, request_id))
        
        # Get result (blocks until request is processed)
        return future.result()
        
    def execute_code(self, code, options=None):
        """Execute code using LLVM JIT compiler"""
        # Construct the proper endpoint URL
        endpoint_url = f"{self.api_endpoint}/execute"
        
        # Prepare request data
        data = {
            "code": code,
            "options": options or {}
        }
        
        # Make request with queue and backoff
        response = self.make_post_request_llvm(endpoint_url, data)
        
        # Process and normalize response
        return {
            "result": response.get("result", ""),
            "output": response.get("output", ""),
            "errors": response.get("errors", []),
            "execution_time": response.get("execution_time", 0),
            "implementation_type": "(REAL)"
        }

    def optimize_code(self, code, optimization_level=None, options=None):
        """Optimize code using LLVM optimizer"""
        # Construct the proper endpoint URL
        endpoint_url = f"{self.api_endpoint}/optimize"
        
        # Prepare request data
        data = {
            "code": code,
            "optimization_level": optimization_level or "O2",
            "options": options or {}
        }
        
        # Make request with queue and backoff
        response = self.make_post_request_llvm(endpoint_url, data)
        
        # Process and normalize response
        return {
            "optimized_code": response.get("optimized_code", ""),
            "optimization_passes": response.get("optimization_passes", []),
            "errors": response.get("errors", []),
            "implementation_type": "(REAL)"
        }
    
    def batch_execute(self, code_batch, options=None):
        """Execute multiple code snippets in batch"""
        # Construct the proper endpoint URL
        endpoint_url = f"{self.api_endpoint}/batch_execute"
        
        # Prepare request data
        data = {
            "code_batch": code_batch,
            "options": options or {}
        }
        
        # Make request with queue and backoff
        response = self.make_post_request_llvm(endpoint_url, data)
        
        # Process and normalize response
        return {
            "results": response.get("results", []),
            "errors": response.get("errors", []),
            "execution_times": response.get("execution_times", []),
            "implementation_type": "(REAL)"
        }
            
    def create_llvm_endpoint_handler(self, endpoint_url=None):
        """Create an endpoint handler for LLVM"""
        async def endpoint_handler(code, **kwargs):
            """Handle requests to LLVM endpoint"""
            # Use provided endpoint or default
            if not endpoint_url:
                actual_endpoint = self.api_endpoint
            else:
                actual_endpoint = endpoint_url
                
            # Extract options from kwargs
            options = {k: v for k, v in kwargs.items() if k not in ["optimization_level", "batch"]}
            
            # Check if batch operation
            if kwargs.get("batch", False) and isinstance(code, list):
                response = self.batch_execute(code, options)
            # Check if optimization request
            elif "optimization_level" in kwargs:
                response = self.optimize_code(code, kwargs["optimization_level"], options)
            else:
                # Standard execution
                response = self.execute_code(code, options)
            
            return response
        
        return endpoint_handler
        
    def test_llvm_endpoint(self, endpoint_url=None):
        """Test the LLVM endpoint"""
        if not endpoint_url:
            endpoint_url = f"{self.api_endpoint}/execute"
            
        # Simple C code to test execution
        test_code = """
        #include <stdio.h>
        int main() {
            printf("Hello from LLVM\n");
            return 0;
        }
        """
        
        try:
            response = self.execute_code(test_code)
            return "result" in response and response.get("implementation_type") == "(REAL)"
        except Exception as e:
            print(f"Error testing LLVM endpoint: {e}")
            return False