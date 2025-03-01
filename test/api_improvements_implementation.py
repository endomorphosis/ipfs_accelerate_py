#!/usr/bin/env python
"""
Script to implement API backend improvements for Claude, OpenAI, Gemini, and Groq.
This script will modify each API backend to ensure:
1. Each endpoint has its own counters, API key, backoff, and queue
2. Each request has an optional request_id parameter
3. All APIs have consistent implementation of retry and backoff mechanisms
"""

import os
import sys
import re
import json
import time
import hashlib
import threading
import argparse
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Dictionary to track improvements for each API
improvement_status = {
    "claude": {"counters": False, "api_key": False, "backoff": False, "queue": False, "request_id": False},
    "openai": {"counters": False, "api_key": False, "backoff": False, "queue": False, "request_id": False},
    "gemini": {"counters": False, "api_key": False, "backoff": False, "queue": False, "request_id": False},
    "groq": {"counters": False, "api_key": False, "backoff": False, "queue": False, "request_id": False}
}

# Templates for adding code to API backends
# Template for adding additional imports
IMPORTS_TEMPLATE = """
import time
import hashlib
import threading
import queue
import uuid
from concurrent.futures import ThreadPoolExecutor
"""

# Base template for adding to class __init__ method for per-endpoint settings
INIT_TEMPLATE = """
        # Initialize counters, queues, and settings for each endpoint 
        self.endpoints = {}  # Dictionary to store per-endpoint data
        
        # Retry and backoff settings (global defaults)
        self.max_retries = 5
        self.initial_retry_delay = 1
        self.backoff_factor = 2
        self.max_retry_delay = 60  # Maximum delay in seconds
        
        # Global request queue settings
        self.queue_enabled = True
        self.queue_size = 100
        self.queue_processing = False
        self.current_requests = 0
        self.max_concurrent_requests = 5
        self.request_queue = []
        self.queue_lock = threading.RLock()
        
        # Initialize thread pool for async processing
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
"""

# Template for the create_endpoint method
CREATE_ENDPOINT_TEMPLATE = """
    def create_endpoint(self, endpoint_id=None, api_key=None, max_retries=None, initial_retry_delay=None, 
                       backoff_factor=None, max_retry_delay=None, queue_enabled=None, 
                       max_concurrent_requests=None, queue_size=None):
        \"\"\"Create a new endpoint with its own settings and counters\"\"\"
        # Generate a unique endpoint ID if not provided
        if endpoint_id is None:
            endpoint_id = f"endpoint_{int(time.time())}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
            
        # Use provided values or defaults
        endpoint_settings = {
            "api_key": api_key if api_key is not None else self.api_key,
            "max_retries": max_retries if max_retries is not None else self.max_retries,
            "initial_retry_delay": initial_retry_delay if initial_retry_delay is not None else self.initial_retry_delay,
            "backoff_factor": backoff_factor if backoff_factor is not None else self.backoff_factor,
            "max_retry_delay": max_retry_delay if max_retry_delay is not None else self.max_retry_delay,
            "queue_enabled": queue_enabled if queue_enabled is not None else self.queue_enabled,
            "max_concurrent_requests": max_concurrent_requests if max_concurrent_requests is not None else self.max_concurrent_requests,
            "queue_size": queue_size if queue_size is not None else self.queue_size,
            
            # Initialize endpoint-specific counters and state
            "current_requests": 0,
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "queue_processing": False,
            "request_queue": [],
            "queue_lock": threading.RLock(),
            "created_at": time.time(),
            "last_request_at": None
        }
        
        # Store the endpoint settings
        self.endpoints[endpoint_id] = endpoint_settings
        
        return endpoint_id
"""

# Template for the get_endpoint method
GET_ENDPOINT_TEMPLATE = """
    def get_endpoint(self, endpoint_id=None):
        \"\"\"Get an endpoint's settings or create a default one if not found\"\"\"
        # If no endpoint_id provided, use the first one or create a default
        if endpoint_id is None:
            if not self.endpoints:
                endpoint_id = self.create_endpoint()
            else:
                endpoint_id = next(iter(self.endpoints))
                
        # If endpoint doesn't exist, create it
        if endpoint_id not in self.endpoints:
            endpoint_id = self.create_endpoint(endpoint_id=endpoint_id)
            
        return self.endpoints[endpoint_id]
"""

# Template for the update_endpoint method
UPDATE_ENDPOINT_TEMPLATE = """
    def update_endpoint(self, endpoint_id, **kwargs):
        \"\"\"Update an endpoint's settings\"\"\"
        if endpoint_id not in self.endpoints:
            raise ValueError(f"Endpoint {endpoint_id} not found")
            
        # Update only the provided settings
        for key, value in kwargs.items():
            if key in self.endpoints[endpoint_id]:
                self.endpoints[endpoint_id][key] = value
                
        return self.endpoints[endpoint_id]
"""

# Template for the process_queue method
PROCESS_QUEUE_TEMPLATE = """
    def _process_queue(self, endpoint_id=None):
        \"\"\"Process requests in the queue for a specific endpoint or global queue\"\"\"
        # Get the endpoint or use global settings
        if endpoint_id and endpoint_id in self.endpoints:
            endpoint = self.endpoints[endpoint_id]
            with endpoint["queue_lock"]:
                if endpoint["queue_processing"]:
                    return  # Another thread is already processing this endpoint's queue
                endpoint["queue_processing"] = True
                
            queue_to_process = endpoint["request_queue"]
            is_global_queue = False
        else:
            # Use global queue if no endpoint specified or endpoint doesn't exist
            with self.queue_lock:
                if self.queue_processing:
                    return  # Another thread is already processing the global queue
                self.queue_processing = True
                
            queue_to_process = self.request_queue
            is_global_queue = True
        
        try:
            while True:
                # Get the next request from the queue
                request_info = None
                
                if is_global_queue:
                    with self.queue_lock:
                        if not queue_to_process:
                            self.queue_processing = False
                            break
                            
                        # Check if we're at the concurrent request limit
                        if self.current_requests >= self.max_concurrent_requests:
                            # Sleep briefly then check again
                            time.sleep(0.1)
                            continue
                            
                        # Get the next request and increase counter
                        request_info = queue_to_process.pop(0)
                        self.current_requests += 1
                else:
                    with endpoint["queue_lock"]:
                        if not queue_to_process:
                            endpoint["queue_processing"] = False
                            break
                            
                        # Check if we're at the concurrent request limit
                        if endpoint["current_requests"] >= endpoint["max_concurrent_requests"]:
                            # Sleep briefly then check again
                            time.sleep(0.1)
                            continue
                            
                        # Get the next request and increase counter
                        request_info = queue_to_process.pop(0)
                        endpoint["current_requests"] += 1
                
                # Process the request outside the lock
                if request_info:
                    try:
                        # Extract request details
                        endpoint_url = request_info.get("endpoint_url")
                        data = request_info.get("data")
                        api_key = request_info.get("api_key")
                        request_id = request_info.get("request_id")
                        endpoint_id = request_info.get("endpoint_id")
                        future = request_info.get("future")
                        method_name = request_info.get("method", "make_request")
                        method_args = request_info.get("args", [])
                        method_kwargs = request_info.get("kwargs", {})
                        
                        # Make the request (without queueing again)
                        # Save original queue_enabled value to prevent recursion
                        if is_global_queue:
                            original_queue_enabled = self.queue_enabled
                            self.queue_enabled = False  # Disable queueing to prevent recursion
                        else:
                            original_queue_enabled = endpoint["queue_enabled"]
                            endpoint["queue_enabled"] = False  # Disable queueing to prevent recursion
                        
                        try:
                            # Make the request based on method name
                            if hasattr(self, method_name) and callable(getattr(self, method_name)):
                                method = getattr(self, method_name)
                                
                                # Call the method with the provided arguments
                                if method_name.startswith("make_"):
                                    # Direct API request methods
                                    result = method(
                                        endpoint_url=endpoint_url,
                                        data=data,
                                        api_key=api_key,
                                        request_id=request_id,
                                        endpoint_id=endpoint_id
                                    )
                                else:
                                    # Higher-level methods
                                    method_kwargs.update({
                                        "request_id": request_id,
                                        "endpoint_id": endpoint_id,
                                        "api_key": api_key
                                    })
                                    result = method(*method_args, **method_kwargs)
                            else:
                                # Fallback to make_request or similar method
                                make_method = getattr(self, "make_request", None)
                                if not make_method:
                                    make_method = getattr(self, f"make_post_request_{self.__class__.__name__.lower()}", None)
                                    
                                if make_method and callable(make_method):
                                    result = make_method(
                                        endpoint_url=endpoint_url,
                                        data=data,
                                        api_key=api_key,
                                        request_id=request_id,
                                        endpoint_id=endpoint_id
                                    )
                                else:
                                    raise AttributeError(f"Method {method_name} not found")
                            
                            # Store result in future
                            future["result"] = result
                            future["completed"] = True
                            
                            # Update counters
                            if not is_global_queue:
                                with endpoint["queue_lock"]:
                                    endpoint["successful_requests"] += 1
                                    endpoint["last_request_at"] = time.time()
                                    
                                    # Update token counts if present in result
                                    if isinstance(result, dict) and "usage" in result:
                                        usage = result["usage"]
                                        endpoint["total_tokens"] += usage.get("total_tokens", 0)
                                        endpoint["input_tokens"] += usage.get("prompt_tokens", 0)
                                        endpoint["output_tokens"] += usage.get("completion_tokens", 0)
                            
                        except Exception as e:
                            # Store error in future
                            future["error"] = e
                            future["completed"] = True
                            print(f"Error processing queued request: {str(e)}")
                            
                            # Update counters
                            if not is_global_queue:
                                with endpoint["queue_lock"]:
                                    endpoint["failed_requests"] += 1
                                    endpoint["last_request_at"] = time.time()
                        
                        finally:
                            # Restore original queue_enabled value
                            if is_global_queue:
                                self.queue_enabled = original_queue_enabled
                            else:
                                endpoint["queue_enabled"] = original_queue_enabled
                    
                    finally:
                        # Decrement counter
                        if is_global_queue:
                            with self.queue_lock:
                                self.current_requests = max(0, self.current_requests - 1)
                        else:
                            with endpoint["queue_lock"]:
                                endpoint["current_requests"] = max(0, endpoint["current_requests"] - 1)
                    
                    # Brief pause to prevent CPU hogging
                    time.sleep(0.01)
                    
        except Exception as e:
            print(f"Error in queue processing thread: {str(e)}")
            
        finally:
            # Reset queue processing flag
            if is_global_queue:
                with self.queue_lock:
                    self.queue_processing = False
            else:
                with endpoint["queue_lock"]:
                    endpoint["queue_processing"] = False
"""

# Template for modifying the make_request method with endpoint handling
MAKE_REQUEST_TEMPLATE = """
    def {method_name}(self, endpoint_url, data, api_key=None, request_id=None, endpoint_id=None):
        \"\"\"Make a request with endpoint-specific settings for queue, backoff, and API key\"\"\"
        # Get endpoint settings or use global defaults
        if endpoint_id and endpoint_id in self.endpoints:
            endpoint = self.endpoints[endpoint_id]
            is_endpoint_request = True
        else:
            endpoint = {{
                "api_key": self.api_key,
                "max_retries": self.max_retries,
                "initial_retry_delay": self.initial_retry_delay,
                "backoff_factor": self.backoff_factor,
                "max_retry_delay": self.max_retry_delay,
                "queue_enabled": self.queue_enabled,
                "max_concurrent_requests": self.max_concurrent_requests,
                "current_requests": self.current_requests,
                "queue_processing": self.queue_processing,
                "request_queue": self.request_queue,
                "queue_lock": self.queue_lock,
                "queue_size": self.queue_size
            }}
            is_endpoint_request = False
            
        # Use endpoint's API key if not explicitly provided
        if not api_key:
            api_key = endpoint["api_key"]
            
        if not api_key:
            raise ValueError("No API key provided for authentication")
            
        # Generate request ID if not provided
        if request_id is None:
            request_id = f"req_{{int(time.time())}}_{{hashlib.md5(str(data).encode()).hexdigest()[:8]}}"
            
        # If queue is enabled and we're at capacity, add to queue
        if endpoint["queue_enabled"]:
            with endpoint["queue_lock"]:
                if endpoint["current_requests"] >= endpoint["max_concurrent_requests"]:
                    # Create a future to store the result
                    result_future = {{"result": None, "error": None, "completed": False}}
                    
                    # Add to queue with all necessary info to process later
                    request_info = {{
                        "endpoint_url": endpoint_url,
                        "data": data,
                        "api_key": api_key,
                        "request_id": request_id,
                        "endpoint_id": endpoint_id if is_endpoint_request else None,
                        "future": result_future
                    }}
                    
                    # Check if queue is full
                    if len(endpoint["request_queue"]) >= endpoint["queue_size"]:
                        raise ValueError(f"Request queue is full ({{endpoint['queue_size']}} items). Try again later.")
                    
                    # Add to queue
                    endpoint["request_queue"].append(request_info)
                    print(f"Request queued. Queue size: {{len(endpoint['request_queue'])}}. Request ID: {{request_id}}")
                    
                    # Start queue processing if not already running
                    if not endpoint["queue_processing"]:
                        threading.Thread(target=self._process_queue, 
                                         args=(endpoint_id,) if is_endpoint_request else ()).start()
                    
                    # Wait for result with timeout
                    wait_start = time.time()
                    max_wait = 300  # 5 minutes
                    
                    while not result_future["completed"] and (time.time() - wait_start) < max_wait:
                        time.sleep(0.1)
                    
                    # Check if completed or timed out
                    if not result_future["completed"]:
                        raise TimeoutError(f"Request timed out after {{max_wait}} seconds in queue")
                    
                    # Propagate error if any
                    if result_future["error"]:
                        raise result_future["error"]
                    
                    return result_future["result"]
                
                # If we're not at capacity, increment counter
                endpoint["current_requests"] += 1
                
                # Update total request counter for endpoint-specific requests
                if is_endpoint_request:
                    endpoint["total_requests"] += 1
                    endpoint["last_request_at"] = time.time()
            
        # Use exponential backoff retry mechanism
        retries = 0
        retry_delay = endpoint["initial_retry_delay"]
        max_retries = endpoint["max_retries"]
        backoff_factor = endpoint["backoff_factor"]
        max_retry_delay = endpoint["max_retry_delay"]
        
        while retries < max_retries:
            try:
                # Add request_id to headers if possible
                headers = {{
                    "Content-Type": "application/json",
                    "X-Request-ID": request_id
                }}
                
                # Add API key to headers based on API type
                api_type = self.__class__.__name__.lower()
                if api_type == "claude":
                    headers["x-api-key"] = api_key
                    headers["anthropic-version"] = "2023-06-01"
                elif api_type == "groq":
                    headers["Authorization"] = f"Bearer {{api_key}}"
                elif api_type in ["openai", "openai_api"]:
                    headers["Authorization"] = f"Bearer {{api_key}}"
                elif api_type == "gemini":
                    # Gemini API key is typically passed as a URL parameter, but we'll set a header too
                    headers["x-goog-api-key"] = api_key
                else:
                    # Default to Bearer auth
                    headers["Authorization"] = f"Bearer {{api_key}}"
                
                # Make the actual request
                import requests
                response = requests.post(
                    endpoint_url,
                    json=data,
                    headers=headers,
                    timeout=60
                )
                
                # Check response status
                if response.status_code != 200:
                    error_message = f"Request failed with status code {{response.status_code}}"
                    try:
                        error_data = response.json()
                        if "error" in error_data:
                            error_message = f"{{error_message}}: {{error_data['error'].get('message', '')}}"
                    except:
                        error_message = f"{{error_message}}: {{response.text[:100]}}"
                    
                    # For specific error codes, handle differently
                    if response.status_code == 401:
                        # Decrement counter
                        if is_endpoint_request:
                            with endpoint["queue_lock"]:
                                endpoint["current_requests"] = max(0, endpoint["current_requests"] - 1)
                                endpoint["failed_requests"] += 1
                                endpoint["last_request_at"] = time.time()
                        else:
                            with self.queue_lock:
                                self.current_requests = max(0, self.current_requests - 1)
                        
                        raise ValueError(f"Authentication error: {{error_message}}")
                        
                    elif response.status_code == 429:
                        # Rate limit error - check for Retry-After header
                        retry_after = response.headers.get("retry-after", None)
                        if retry_after:
                            try:
                                retry_delay = float(retry_after)
                            except:
                                # Use exponential backoff if Retry-After isn't a number
                                retry_delay = min(retry_delay * backoff_factor, max_retry_delay)
                        else:
                            # Use exponential backoff if Retry-After isn't present
                            retry_delay = min(retry_delay * backoff_factor, max_retry_delay)
                            
                        print(f"Rate limit exceeded. Retrying in {{retry_delay}} seconds... (Request ID: {{request_id}})")
                        time.sleep(retry_delay)
                        retries += 1
                        continue
                        
                    # For other error codes, just raise an exception
                    raise ValueError(error_message)
                
                # Parse and return successful response
                result = response.json()
                
                # Update token usage for endpoint-specific requests
                if is_endpoint_request:
                    with endpoint["queue_lock"]:
                        endpoint["successful_requests"] += 1
                        
                        # Update token counts if present in result
                        if "usage" in result:
                            usage = result["usage"]
                            endpoint["total_tokens"] += usage.get("total_tokens", 0)
                            endpoint["input_tokens"] += usage.get("prompt_tokens", 0) 
                            endpoint["output_tokens"] += usage.get("completion_tokens", 0)
                
                return result
                
            except requests.exceptions.RequestException as e:
                if retries < max_retries - 1:
                    print(f"Request failed: {{str(e)}}. Retrying in {{retry_delay}} seconds (attempt {{retries+1}}/{{max_retries}}, Request ID: {{request_id}})...")
                    time.sleep(retry_delay)
                    retries += 1
                    retry_delay = min(retry_delay * backoff_factor, max_retry_delay)
                else:
                    print(f"Request failed after {{max_retries}} attempts: {{str(e)}} (Request ID: {{request_id}})")
                    
                    # Decrement counter and update stats
                    if is_endpoint_request:
                        with endpoint["queue_lock"]:
                            endpoint["current_requests"] = max(0, endpoint["current_requests"] - 1)
                            endpoint["failed_requests"] += 1
                            endpoint["last_request_at"] = time.time()
                    else:
                        with self.queue_lock:
                            self.current_requests = max(0, self.current_requests - 1)
                    
                    raise
            
            except Exception as e:
                # Decrement counter for any other exceptions
                if is_endpoint_request:
                    with endpoint["queue_lock"]:
                        endpoint["current_requests"] = max(0, endpoint["current_requests"] - 1)
                        endpoint["failed_requests"] += 1
                        endpoint["last_request_at"] = time.time()
                else:
                    with self.queue_lock:
                        self.current_requests = max(0, self.current_requests - 1)
                raise
                
        # Decrement counter if we somehow exit the loop without returning or raising
        if is_endpoint_request:
            with endpoint["queue_lock"]:
                endpoint["current_requests"] = max(0, endpoint["current_requests"] - 1)
        else:
            with self.queue_lock:
                self.current_requests = max(0, self.current_requests - 1)
                
        # This should never be reached due to the raise in the exception handler
        return None
"""

# Template for the get_stats method
GET_STATS_TEMPLATE = """
    def get_stats(self, endpoint_id=None):
        \"\"\"Get usage statistics for an endpoint or global stats\"\"\"
        if endpoint_id and endpoint_id in self.endpoints:
            endpoint = self.endpoints[endpoint_id]
            stats = {
                "endpoint_id": endpoint_id,
                "total_requests": endpoint["total_requests"],
                "successful_requests": endpoint["successful_requests"],
                "failed_requests": endpoint["failed_requests"],
                "total_tokens": endpoint["total_tokens"],
                "input_tokens": endpoint["input_tokens"],
                "output_tokens": endpoint["output_tokens"],
                "created_at": endpoint["created_at"],
                "last_request_at": endpoint["last_request_at"],
                "current_queue_size": len(endpoint["request_queue"]),
                "current_requests": endpoint["current_requests"]
            }
            return stats
        else:
            # Aggregate stats across all endpoints
            total_requests = sum(e["total_requests"] for e in self.endpoints.values()) if self.endpoints else 0
            successful_requests = sum(e["successful_requests"] for e in self.endpoints.values()) if self.endpoints else 0
            failed_requests = sum(e["failed_requests"] for e in self.endpoints.values()) if self.endpoints else 0
            total_tokens = sum(e["total_tokens"] for e in self.endpoints.values()) if self.endpoints else 0
            input_tokens = sum(e["input_tokens"] for e in self.endpoints.values()) if self.endpoints else 0
            output_tokens = sum(e["output_tokens"] for e in self.endpoints.values()) if self.endpoints else 0
            
            stats = {
                "endpoints_count": len(self.endpoints),
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "total_tokens": total_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "global_queue_size": len(self.request_queue),
                "global_current_requests": self.current_requests
            }
            return stats
"""

# Template for the reset_stats method
RESET_STATS_TEMPLATE = """
    def reset_stats(self, endpoint_id=None):
        \"\"\"Reset usage statistics for an endpoint or globally\"\"\"
        if endpoint_id and endpoint_id in self.endpoints:
            # Reset stats just for this endpoint
            endpoint = self.endpoints[endpoint_id]
            endpoint["total_requests"] = 0
            endpoint["successful_requests"] = 0
            endpoint["failed_requests"] = 0
            endpoint["total_tokens"] = 0
            endpoint["input_tokens"] = 0
            endpoint["output_tokens"] = 0
        elif endpoint_id is None:
            # Reset stats for all endpoints
            for endpoint in self.endpoints.values():
                endpoint["total_requests"] = 0
                endpoint["successful_requests"] = 0
                endpoint["failed_requests"] = 0
                endpoint["total_tokens"] = 0
                endpoint["input_tokens"] = 0
                endpoint["output_tokens"] = 0
        else:
            raise ValueError(f"Endpoint {endpoint_id} not found")
"""

# Function to add requires imports if not already present
def add_required_imports(content):
    """Add required imports if not already present"""
    # Check existing imports
    has_time_import = "import time" in content
    has_hashlib_import = "import hashlib" in content
    has_threading_import = "import threading" in content
    has_queue_import = "import queue" in content
    has_uuid_import = "import uuid" in content
    has_concurrent_import = "from concurrent.futures import ThreadPoolExecutor" in content
    
    # Build imports to add
    imports_to_add = []
    if not has_time_import:
        imports_to_add.append("import time")
    if not has_hashlib_import:
        imports_to_add.append("import hashlib")
    if not has_threading_import:
        imports_to_add.append("import threading")
    if not has_queue_import:
        imports_to_add.append("import queue")
    if not has_uuid_import:
        imports_to_add.append("import uuid")
    if not has_concurrent_import:
        imports_to_add.append("from concurrent.futures import ThreadPoolExecutor")
    
    if not imports_to_add:
        return content  # No imports to add
    
    # Find best place to add imports
    import_section_end = None
    
    # Look for the last import line
    import_pattern = r"^(?:import\s+[\w\.]+|from\s+[\w\.]+\s+import\s+[\w\.,\s\*]+)$"
    import_matches = list(re.finditer(import_pattern, content, re.MULTILINE))
    
    if import_matches:
        last_import = import_matches[-1]
        import_section_end = last_import.end()
    else:
        # If no imports found, add after module docstring if it exists
        docstring_pattern = r'^""".*?"""'
        docstring_match = re.search(docstring_pattern, content, re.MULTILINE | re.DOTALL)
        if docstring_match:
            import_section_end = docstring_match.end()
    
    # Insert imports at the appropriate position
    if import_section_end is not None:
        import_text = "\n" + "\n".join(imports_to_add) + "\n"
        content = content[:import_section_end] + import_text + content[import_section_end:]
    else:
        # Add at the beginning of the file
        import_text = "\n".join(imports_to_add) + "\n\n"
        content = import_text + content
    
    return content

# Function to modify __init__ method to add per-endpoint settings
def modify_init_method(content):
    """Modify __init__ method to add per-endpoint settings"""
    # Find __init__ method
    init_pattern = r"def __init__.*?(?:return None|\s*(?:pass|return))"
    init_match = re.search(init_pattern, content, re.DOTALL)
    
    if init_match:
        init_method = init_match.group(0)
        # Check if per-endpoint settings already exist
        if "self.endpoints = {}" not in init_method:
            # Add settings before the end of the method
            # First, find proper indentation
            indent_match = re.search(r"^(\s+)def __init__", init_method, re.MULTILINE)
            proper_indent = indent_match.group(1) if indent_match else "    "
            
            # Determine where to insert the initialization code
            if "return None" in init_method:
                new_init = init_method.replace("return None", INIT_TEMPLATE.replace("        ", proper_indent + "    ") + proper_indent + "return None")
            elif init_method.strip().endswith("pass"):
                # Replace 'pass' with our initialization
                new_init = re.sub(r"pass\s*$", INIT_TEMPLATE.replace("        ", proper_indent + "    "), init_method)
            else:
                # Just append at the end
                new_init = init_method.rstrip() + "\n" + INIT_TEMPLATE.replace("        ", proper_indent + "    ")
            
            content = content.replace(init_method, new_init)
    
    return content

# Function to add or update endpoint management methods
def add_endpoint_management_methods(content):
    """Add or update endpoint management methods"""
    methods_to_add = {
        "create_endpoint": CREATE_ENDPOINT_TEMPLATE,
        "get_endpoint": GET_ENDPOINT_TEMPLATE,
        "update_endpoint": UPDATE_ENDPOINT_TEMPLATE,
        "_process_queue": PROCESS_QUEUE_TEMPLATE,
        "get_stats": GET_STATS_TEMPLATE,
        "reset_stats": RESET_STATS_TEMPLATE
    }
    
    # Find class indentation level
    class_match = re.search(r"^class\s+\w+", content, re.MULTILINE)
    class_indent = ""
    
    if class_match:
        # Find first method to determine indentation
        method_match = re.search(r"^(\s+)def\s+\w+", content, re.MULTILINE)
        if method_match:
            class_indent = method_match.group(1)
    
    # Add or update each method
    for method_name, template in methods_to_add.items():
        # Check if method already exists
        method_pattern = f"def {method_name}\\(.*?\\).*?(?=\\n{class_indent}def |\\Z)"
        method_match = re.search(method_pattern, content, re.DOTALL)
        
        # Adjust indentation in template
        adjusted_template = template
        if class_indent:
            adjusted_template = template.replace("    ", class_indent)
        
        if method_match:
            # Method exists, update it
            old_method = method_match.group(0)
            content = content.replace(old_method, adjusted_template)
        else:
            # Method doesn't exist, add it
            # Find a good location to add the method - after an existing method
            last_method_match = list(re.finditer(f"{class_indent}def .*?(?=\\n{class_indent}def |\\Z)", content, re.DOTALL))
            
            if last_method_match:
                insert_position = last_method_match[-1].end()
                content = content[:insert_position] + "\n" + adjusted_template + content[insert_position:]
            else:
                # Add after class definition
                class_def_match = re.search(r"^class\s+\w+.*?(?:\(.*?\))?\s*:", content, re.MULTILINE)
                if class_def_match:
                    insert_position = class_def_match.end()
                    content = content[:insert_position] + "\n" + adjusted_template + content[insert_position:]
                else:
                    # Just add at the end of the file
                    content = content + "\n\n" + adjusted_template
    
    return content

# Function to update make_request method
def update_make_request_method(content, api_type):
    """Update make_request method with endpoint-specific logic"""
    # Identify the primary request method name based on API type
    if api_type == "groq":
        method_name = "make_post_request_groq"
    elif api_type == "claude":
        method_name = "make_post_request"
    elif api_type == "openai":
        method_name = "make_request"
    elif api_type == "gemini":
        method_name = "make_post_request_gemini"
    else:
        method_name = "make_request"  # Default name
    
    # Check if method exists
    method_match = re.search(f"def {method_name}", content)
    if not method_match:
        print(f"Warning: Could not find request method '{method_name}' in the file")
        return content
    
    # Find the entire method definition
    method_pattern = f"def {method_name}.*?(?=\n(?:    |\t)def |\Z)"
    method_match = re.search(method_pattern, content, re.DOTALL)
    if not method_match:
        print(f"Warning: Could not extract complete method '{method_name}'")
        return content
    
    old_method = method_match.group(0)
    
    # Create new method with our template
    new_method = MAKE_REQUEST_TEMPLATE.format(method_name=method_name)
    
    # Adjust indentation to match file
    indent_match = re.search(r"^( +)def ", old_method, re.MULTILINE)
    if indent_match:
        proper_indent = indent_match.group(1)
        new_method = new_method.replace("    ", proper_indent)
    
    # Replace old method with new one
    content = content.replace(old_method, new_method)
    
    return content

# Function to update API-specific methods
def update_api_methods(content, api_type):
    """Update API-specific methods to use the endpoint system"""
    # Different APIs have different methods that need updating
    if api_type == "claude":
        methods_to_update = ["chat", "stream_chat"]
    elif api_type == "openai":
        methods_to_update = ["chat", "embedding", "moderation", "text_to_image", "speech_to_text", "text_to_speech"]
    elif api_type == "gemini":
        methods_to_update = ["chat", "stream_chat", "process_image"]
    elif api_type == "groq":
        methods_to_update = ["chat", "stream_chat"]
    else:
        methods_to_update = []
    
    for method_name in methods_to_update:
        # Check if method exists
        method_pattern = f"def {method_name}\\(.*?\\):.*?(?=\n(?:    |\t)def |\Z)"
        method_match = re.search(method_pattern, content, re.DOTALL)
        
        if method_match:
            old_method = method_match.group(0)
            
            # Extract parameters
            param_pattern = f"def {method_name}\\((.*?)\\)"
            param_match = re.search(param_pattern, old_method)
            if param_match:
                params = param_match.group(1)
                
                # Add request_id and endpoint_id parameters if not present
                new_params = params
                if "request_id" not in params:
                    if params.strip():
                        if params.endswith(")"):
                            new_params = params[:-1] + ", request_id=None)"
                        else:
                            new_params = params + ", request_id=None"
                    else:
                        new_params = "self, request_id=None"
                        
                if "endpoint_id" not in new_params:
                    if new_params.strip():
                        if new_params.endswith(")"):
                            new_params = new_params[:-1] + ", endpoint_id=None)"
                        else:
                            new_params = new_params + ", endpoint_id=None"
                    else:
                        new_params = "self, endpoint_id=None"
                
                # Update method signature with new parameters
                new_method = old_method.replace(params, new_params)
                
                # Find the API request call - could be make_post_request, make_request, etc.
                request_call_pattern = r"self\.(make_\w+)(\(.*?\))"
                for request_call_match in re.finditer(request_call_pattern, new_method):
                    old_call = request_call_match.group(0)
                    request_method = request_call_match.group(1)
                    args = request_call_match.group(2)
                    
                    # Add request_id and endpoint_id to the call if not present
                    if "request_id=" not in args and "request_id" not in args:
                        if args.endswith(")"):
                            args = args[:-1] + ", request_id=request_id)"
                        else:
                            args = args + ", request_id=request_id"
                    
                    if "endpoint_id=" not in args and "endpoint_id" not in args:
                        if args.endswith(")"):
                            args = args[:-1] + ", endpoint_id=endpoint_id)"
                        else:
                            args = args + ", endpoint_id=endpoint_id"
                    
                    new_call = f"self.{request_method}{args}"
                    new_method = new_method.replace(old_call, new_call)
                
                # Update the method in the content
                content = content.replace(old_method, new_method)
    
    return content

# Add queue handling to higher-level methods
def add_queue_handling(content, api_type):
    """Add queue handling to higher-level methods"""
    # Different APIs have different methods that need updating
    if api_type == "claude":
        methods_to_update = ["chat", "stream_chat"]
    elif api_type == "openai":
        methods_to_update = ["chat", "embedding", "moderation", "text_to_image", "speech_to_text", "text_to_speech"]
    elif api_type == "gemini":
        methods_to_update = ["chat", "stream_chat", "process_image"]
    elif api_type == "groq":
        methods_to_update = ["chat", "stream_chat"]
    else:
        methods_to_update = []
    
    for method_name in methods_to_update:
        # Check if method exists and doesn't already have queue handling
        method_pattern = f"def {method_name}\\(.*?\\):.*?(?=\n(?:    |\t)def |\Z)"
        method_match = re.search(method_pattern, content, re.DOTALL)
        
        if method_match and "queue_enabled" not in method_match.group(0):
            old_method = method_match.group(0)
            
            # Extract method body
            body_pattern = f"def {method_name}\\(.*?\\):(.*?)(?=\n(?:    |\t)def |\Z)"
            body_match = re.search(body_pattern, content, re.DOTALL)
            if body_match:
                body = body_match.group(1)
                
                # Find indentation
                indent_match = re.search(r"^(\s+)", body.lstrip(), re.MULTILINE)
                indent = indent_match.group(1) if indent_match else "    "
                
                # Add queue handling at the beginning of the method
                queue_code = f"""
{indent}# Handle queueing if enabled and at capacity
{indent}if endpoint_id and endpoint_id in self.endpoints:
{indent}    endpoint = self.endpoints[endpoint_id]
{indent}    if endpoint["queue_enabled"] and endpoint["current_requests"] >= endpoint["max_concurrent_requests"]:
{indent}        # Create a future to store the result
{indent}        result_future = {{"result": None, "error": None, "completed": False}}
{indent}        
{indent}        # Add to queue with all necessary info to process later
{indent}        method_kwargs = locals().copy()
{indent}        # Remove 'self' from kwargs
{indent}        if 'self' in method_kwargs:
{indent}            del method_kwargs['self']
{indent}            
{indent}        request_info = {{
{indent}            "method": "{method_name}",
{indent}            "args": [],
{indent}            "kwargs": method_kwargs,
{indent}            "endpoint_id": endpoint_id,
{indent}            "request_id": request_id,
{indent}            "api_key": api_key if 'api_key' in locals() else None,
{indent}            "future": result_future
{indent}        }}
{indent}        
{indent}        # Check if queue is full
{indent}        with endpoint["queue_lock"]:
{indent}            if len(endpoint["request_queue"]) >= endpoint["queue_size"]:
{indent}                raise ValueError(f"Request queue is full ({{endpoint['queue_size']}} items). Try again later.")
{indent}            
{indent}            # Add to queue
{indent}            endpoint["request_queue"].append(request_info)
{indent}            print(f"Request queued for {method_name}. Queue size: {{len(endpoint['request_queue'])}}. Request ID: {{request_id}}")
{indent}            
{indent}            # Start queue processing if not already running
{indent}            if not endpoint["queue_processing"]:
{indent}                threading.Thread(target=self._process_queue, args=(endpoint_id,)).start()
{indent}        
{indent}        # Wait for result with timeout
{indent}        wait_start = time.time()
{indent}        max_wait = 300  # 5 minutes
{indent}        
{indent}        while not result_future["completed"] and (time.time() - wait_start) < max_wait:
{indent}            time.sleep(0.1)
{indent}        
{indent}        # Check if completed or timed out
{indent}        if not result_future["completed"]:
{indent}            raise TimeoutError(f"Request timed out after {{max_wait}} seconds in queue")
{indent}        
{indent}        # Propagate error if any
{indent}        if result_future["error"]:
{indent}            raise result_future["error"]
{indent}        
{indent}        return result_future["result"]
{indent}        
{indent}# Update stats if using an endpoint
{indent}if endpoint_id and endpoint_id in self.endpoints:
{indent}    with self.endpoints[endpoint_id]["queue_lock"]:
{indent}        self.endpoints[endpoint_id]["current_requests"] += 1
{indent}        self.endpoints[endpoint_id]["total_requests"] += 1
{indent}        self.endpoints[endpoint_id]["last_request_at"] = time.time()
"""
                
                # Split the method at the first line of code after the def statement
                first_line_pattern = f"def {method_name}\\(.*?\\):.*?\n"
                first_line_match = re.search(first_line_pattern, old_method)
                if first_line_match:
                    first_line = first_line_match.group(0)
                    
                    # Find docstring if present
                    docstring_pattern = f'def {method_name}\\(.*?\\):.*?(""".*?""")'
                    docstring_match = re.search(docstring_pattern, old_method, re.DOTALL)
                    
                    if docstring_match:
                        docstring = docstring_match.group(1)
                        # Add queue code after docstring
                        new_method = old_method.replace(docstring, docstring + queue_code)
                    else:
                        # Add after first line if no docstring
                        new_method = first_line + queue_code + old_method[len(first_line):]
                    
                    # Add cleanup at end of method - find the last return statement
                    return_match = re.finditer(r"return .*?(?=\n|$)", new_method)
                    return_positions = list(return_match)
                    
                    if return_positions:
                        last_return = return_positions[-1]
                        cleanup_code = f"""
{indent}# Update stats if using an endpoint
{indent}if endpoint_id and endpoint_id in self.endpoints:
{indent}    with self.endpoints[endpoint_id]["queue_lock"]:
{indent}        self.endpoints[endpoint_id]["current_requests"] = max(0, self.endpoints[endpoint_id]["current_requests"] - 1)
{indent}        self.endpoints[endpoint_id]["successful_requests"] += 1
{indent}        # Update token counts if present in result (assuming result variable is named 'result')
{indent}        if 'result' in locals() and isinstance(result, dict) and "usage" in result:
{indent}            usage = result["usage"]
{indent}            self.endpoints[endpoint_id]["total_tokens"] += usage.get("total_tokens", 0)
{indent}            self.endpoints[endpoint_id]["input_tokens"] += usage.get("prompt_tokens", 0)
{indent}            self.endpoints[endpoint_id]["output_tokens"] += usage.get("completion_tokens", 0)
"""
                        # Insert cleanup before the return
                        return_pos = last_return.start()
                        new_method = new_method[:return_pos] + cleanup_code + new_method[return_pos:]
                    
                    # Add error handling
                    if "try:" not in new_method:
                        try_pattern = f"def {method_name}\\(.*?\\):.*?\n"
                        try_match = re.search(try_pattern, new_method)
                        if try_match:
                            try_line = try_match.group(0)
                            
                            docstring_pattern = f'def {method_name}\\(.*?\\):.*?(""".*?""")'
                            docstring_match = re.search(docstring_pattern, new_method, re.DOTALL)
                            
                            if docstring_match:
                                docstring = docstring_match.group(1)
                                insert_pos = new_method.find(docstring) + len(docstring)
                            else:
                                insert_pos = len(try_line)
                            
                            try_code = f"""
{indent}try:"""
                            except_code = f"""
{indent}except Exception as e:
{indent}    # Update stats on error if using an endpoint
{indent}    if endpoint_id and endpoint_id in self.endpoints:
{indent}        with self.endpoints[endpoint_id]["queue_lock"]:
{indent}            self.endpoints[endpoint_id]["current_requests"] = max(0, self.endpoints[endpoint_id]["current_requests"] - 1)
{indent}            self.endpoints[endpoint_id]["failed_requests"] += 1
{indent}    raise
"""
                            # Insert try at beginning of method body
                            new_method_parts = list(new_method)
                            new_method_parts.insert(insert_pos, try_code)
                            
                            # Insert except before the last return
                            if return_positions:
                                last_return = return_positions[-1]
                                return_pos = last_return.start()
                                new_method_parts.insert(return_pos, except_code)
                                new_method = ''.join(new_method_parts)
                            else:
                                # No return statement, add at the end
                                new_method = ''.join(new_method_parts) + except_code
                    
                    # Update the method in the content
                    content = content.replace(old_method, new_method)
    
    return content

# Main function to process each API file
def process_api_file(file_path, api_type):
    """Process a single API file to add endpoint-specific improvements"""
    print(f"Processing {file_path} as {api_type} API...")
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        original_content = content
        
        # Add required imports
        content = add_required_imports(content)
        
        # Modify __init__ method
        content = modify_init_method(content)
        
        # Add endpoint management methods
        content = add_endpoint_management_methods(content)
        
        # Update make_request method
        content = update_make_request_method(content, api_type)
        
        # Update API-specific methods
        content = update_api_methods(content, api_type)
        
        # Add queue handling to higher-level methods
        content = add_queue_handling(content, api_type)
        
        # Check if content was modified
        if content != original_content:
            # Write updated content back to file
            with open(file_path, 'w') as f:
                f.write(content)
                
            print(f"✅ Successfully updated {file_path}")
            
            # Update improvement status
            improvement_status[api_type]["counters"] = "self.endpoints = {}" in content
            improvement_status[api_type]["api_key"] = "endpoint[\"api_key\"]" in content
            improvement_status[api_type]["backoff"] = "backoff_factor" in content and "retry_delay" in content
            improvement_status[api_type]["queue"] = "request_queue" in content and "_process_queue" in content
            improvement_status[api_type]["request_id"] = "request_id=None" in content
            
            return True
        else:
            print(f"ℹ️ No changes needed for {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing {file_path}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Implement API backend improvements for Claude, OpenAI, Gemini, and Groq")
    parser.add_argument("--api", "-a", help="Specific API to update", 
                      choices=["claude", "openai", "gemini", "groq", "hf_tei", "hf_tgi", "ollama", "llvm", "ovms", "all"])
    parser.add_argument("--dry-run", "-d", action="store_true", help="Only print what would be done without making changes")
    
    args = parser.parse_args()
    
    # Get path to API backends directory
    script_dir = Path(__file__).parent.parent
    api_backends_dir = script_dir / "ipfs_accelerate_py" / "api_backends"
    
    if not api_backends_dir.exists():
        print(f"Error: API backends directory not found at {api_backends_dir}")
        return
    
    # Map of API file names to API types
    api_files = {
        "claude.py": "claude",
        "openai_api.py": "openai",
        "gemini.py": "gemini",
        "groq.py": "groq",
        "hf_tei.py": "hf_tei",
        "hf_tgi.py": "hf_tgi",
        "ollama.py": "ollama",
        "llvm.py": "llvm",
        "ovms.py": "ovms"
    }
    
    # Initialize improvement status for all APIs
    for api_type in api_files.values():
        if api_type not in improvement_status:
            improvement_status[api_type] = {"counters": False, "api_key": False, "backoff": False, "queue": False, "request_id": False}
    
    # Process requested API(s)
    if args.api and args.api != "all":
        # Find the filename for the specified API
        api_filename = next((k for k, v in api_files.items() if v == args.api), None)
        if not api_filename:
            print(f"Error: Unknown API '{args.api}'")
            return
        apis_to_process = [(api_filename, args.api)]
    else:
        # Default to processing all APIs
        apis_to_process = list(api_files.items())
    
    results = []
    for filename, api_type in apis_to_process:
        file_path = api_backends_dir / filename
        if not file_path.exists():
            print(f"Warning: File {file_path} not found, skipping")
            continue
            
        if args.dry_run:
            print(f"Would process {file_path} as {api_type} API")
        else:
            success = process_api_file(file_path, api_type)
            results.append((filename, api_type, success))
    
    # Print summary
    if not args.dry_run and results:
        print("\n=== Improvement Status Summary ===")
        for api_type in improvement_status:
            if any(a == api_type for _, a, _ in results):
                status = improvement_status[api_type]
                print(f"{api_type.capitalize()} API:")
                print(f"  ✓ {'Own counters per endpoint:':30} {status['counters']}")
                print(f"  ✓ {'Own API key per endpoint:':30} {status['api_key']}")
                print(f"  ✓ {'Backoff mechanism:':30} {status['backoff']}")
                print(f"  ✓ {'Queue system:':30} {status['queue']}")
                print(f"  ✓ {'Request ID support:':30} {status['request_id']}")
        
        print(f"\nSuccessfully updated {sum(1 for _, _, success in results if success)} of {len(results)} API backends")
        
        # Generate more detailed implementation report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
        report_file = Path(f"api_implementation_report_{timestamp}.md")
        with open(report_file, "w") as f:
            f.write("# API Implementation Report\n\n")
            f.write(f"Report generated: {datetime.now().isoformat()}\n\n")
            
            f.write("## Summary\n\n")
            f.write("| API | Own Counters | Own API Key | Backoff | Queue | Request ID |\n")
            f.write("|-----|-------------|------------|---------|-------|------------|\n")
            for api_type in sorted(improvement_status.keys()):
                status = improvement_status[api_type]
                f.write(f"| {api_type.capitalize()} | {'✓' if status['counters'] else '✗'} | {'✓' if status['api_key'] else '✗'} | {'✓' if status['backoff'] else '✗'} | {'✓' if status['queue'] else '✗'} | {'✓' if status['request_id'] else '✗'} |\n")
            
            f.write("\n## Implementation Details\n\n")
            for api_type in sorted(improvement_status.keys()):
                f.write(f"### {api_type.capitalize()} API Implementation\n\n")
                status = improvement_status[api_type]
                
                f.write("#### 1. Per-Endpoint Counters\n\n")
                if status['counters']:
                    f.write("✅ **IMPLEMENTED**\n\n")
                    f.write("Each endpoint has its own request counters and usage statistics:\n")
                    f.write("- total_requests\n")
                    f.write("- successful_requests\n")
                    f.write("- failed_requests\n")
                    f.write("- total_tokens\n")
                    f.write("- input_tokens\n")
                    f.write("- output_tokens\n\n")
                else:
                    f.write("❌ **NOT IMPLEMENTED**\n\n")
                
                f.write("#### 2. Per-Endpoint API Key\n\n")
                if status['api_key']:
                    f.write("✅ **IMPLEMENTED**\n\n")
                    f.write("Each endpoint can use its own API key, falling back to the global API key if not specified.\n\n")
                else:
                    f.write("❌ **NOT IMPLEMENTED**\n\n")
                
                f.write("#### 3. Backoff Mechanism\n\n")
                if status['backoff']:
                    f.write("✅ **IMPLEMENTED**\n\n")
                    f.write("Exponential backoff with retry mechanism:\n")
                    f.write("- Configurable max_retries (default: 5)\n")
                    f.write("- Configurable initial_retry_delay (default: 1 second)\n")
                    f.write("- Configurable backoff_factor (default: 2)\n")
                    f.write("- Configurable max_retry_delay (default: 60 seconds)\n")
                    f.write("- Respects Retry-After headers from API responses\n\n")
                else:
                    f.write("❌ **NOT IMPLEMENTED**\n\n")
                
                f.write("#### 4. Queue System\n\n")
                if status['queue']:
                    f.write("✅ **IMPLEMENTED**\n\n")
                    f.write("Request queue system with separate queues per endpoint:\n")
                    f.write("- Configurable queue_size (default: 100)\n")
                    f.write("- Configurable max_concurrent_requests (default: 5)\n")
                    f.write("- Asynchronous queue processing with threading\n")
                    f.write("- Request timeout monitoring\n\n")
                else:
                    f.write("❌ **NOT IMPLEMENTED**\n\n")
                
                f.write("#### 5. Request ID Support\n\n")
                if status['request_id']:
                    f.write("✅ **IMPLEMENTED**\n\n")
                    f.write("Request ID tracking:\n")
                    f.write("- Optional request_id parameter in all request methods\n")
                    f.write("- Automatic generation of unique request IDs if not provided\n")
                    f.write("- Request ID format: req_{timestamp}_{data_hash}\n")
                    f.write("- Request ID included in headers when possible\n\n")
                else:
                    f.write("❌ **NOT IMPLEMENTED**\n\n")
                
                f.write("\n")
            
            f.write("\n## Usage Examples\n\n")
            f.write("### Creating an Endpoint\n\n")
            f.write("```python\n")
            f.write("# Initialize API client\n")
            f.write("from ipfs_accelerate_py.api_backends import claude\n")
            f.write("claude_client = claude(resources={}, metadata={\"claude_api_key\": \"your_default_api_key\"})\n\n")
            f.write("# Create an endpoint with custom settings\n")
            f.write("endpoint_id = claude_client.create_endpoint(\n")
            f.write("    api_key=\"endpoint_specific_api_key\",\n")
            f.write("    max_retries=3,\n")
            f.write("    initial_retry_delay=2,\n")
            f.write("    backoff_factor=3,\n")
            f.write("    max_concurrent_requests=10\n")
            f.write(")\n\n")
            f.write("# Use the endpoint for requests\n")
            f.write("response = claude_client.chat(\n")
            f.write("    messages=[{\"role\": \"user\", \"content\": \"Hello\"}],\n")
            f.write("    endpoint_id=endpoint_id,\n")
            f.write("    request_id=\"custom_request_id_123\"\n")
            f.write(")\n")
            f.write("```\n\n")
            
            f.write("### Getting Endpoint Statistics\n\n")
            f.write("```python\n")
            f.write("# Get statistics for a specific endpoint\n")
            f.write("stats = claude_client.get_stats(endpoint_id)\n")
            f.write("print(f\"Total requests: {stats['total_requests']}\")\n")
            f.write("print(f\"Total tokens: {stats['total_tokens']}\")\n\n")
            f.write("# Get aggregate statistics across all endpoints\n")
            f.write("all_stats = claude_client.get_stats()\n")
            f.write("print(f\"Total endpoints: {all_stats['endpoints_count']}\")\n")
            f.write("print(f\"Total requests across all endpoints: {all_stats['total_requests']}\")\n")
            f.write("```\n\n")
            
        print(f"\nDetailed implementation report saved to: {report_file}")
        
        # Save implementation status as JSON for programmatic use
        status_file = Path(f"api_implementation_status_{timestamp}.json")
        with open(status_file, "w") as f:
            json.dump(improvement_status, f, indent=2)
            
        print(f"Implementation status saved to: {status_file}")

if __name__ == "__main__":
    main()