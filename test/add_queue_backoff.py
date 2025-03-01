#!/usr/bin/env python
"""
Script to add queue and backoff mechanisms to all API backends.
This script will:
1. Add exponential backoff retry mechanism to all API backends
2. Add request queue system for handling concurrent requests
3. Add request tracking with unique IDs
"""

import os
import sys
import re
import glob
import argparse
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Base template for adding to class __init__ method
INIT_TEMPLATE = """
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
"""

# Function to add import for threading if not already present
def add_threading_import(content):
    """Add import for threading if not already there"""
    if "import threading" not in content:
        # Add after existing imports
        import_section_end = re.search(r"(^import.*?$|^from.*?$)", content, re.MULTILINE | re.DOTALL)
        if import_section_end:
            position = import_section_end.end()
            return content[:position] + "\nimport threading" + content[position:]
    return content

# Function to add queue processing method
def add_queue_processing(content):
    """Add the queue processing method to the API class"""
    # First check if the method already exists
    if "_process_queue" in content:
        return content
        
    # Find a good location to add the method - after reset_usage_stats if it exists
    match = re.search(r"def reset_usage_stats.*?return.*?\n", content, re.DOTALL)
    if match:
        insert_position = match.end()
    else:
        # Or after the __init__ method
        match = re.search(r"def __init__.*?return None\n", content, re.DOTALL)
        if match:
            insert_position = match.end()
        else:
            # Just add it at the end of the file
            insert_position = len(content)
    
    queue_method = """
    def _process_queue(self):
        # Process requests in the queue in FIFO order
        with self.queue_lock:
            if self.queue_processing:
                return  # Another thread is already processing the queue
            self.queue_processing = True
        
        logger.info("Starting queue processing thread")
        
        try:
            while True:
                # Get the next request from the queue
                with self.queue_lock:
                    if not self.request_queue:
                        self.queue_processing = False
                        break
                        
                    # Check if we're at the concurrent request limit
                    if self.current_requests >= self.max_concurrent_requests:
                        # Sleep briefly then check again
                        time.sleep(0.1)
                        continue
                        
                    # Get the next request and increase counter
                    request_info = self.request_queue.pop(0)
                    self.current_requests += 1
                
                # Process the request outside the lock
                try:
                    # Extract request details
                    endpoint_url = request_info["endpoint_url"]
                    data = request_info["data"]
                    api_key = request_info["api_key"]
                    request_id = request_info["request_id"]
                    future = request_info["future"]
                    
                    # Make the request (without queueing again)
                    # Save original queue_enabled value
                    original_queue_enabled = self.queue_enabled
                    self.queue_enabled = False  # Disable queueing to prevent recursion
                    
                    try:
                        # Make the request
                        result = self.make_post_request(
                            endpoint_url=endpoint_url,
                            data=data,
                            api_key=api_key,
                            request_id=request_id
                        )
                        
                        # Store result in future
                        future["result"] = result
                        future["completed"] = True
                        
                    except Exception as e:
                        # Store error in future
                        future["error"] = e
                        future["completed"] = True
                        logger.error(f"Error processing queued request: {str(e)}")
                    
                    finally:
                        # Restore original queue_enabled value
                        self.queue_enabled = original_queue_enabled
                
                finally:
                    # Decrement counter
                    with self.queue_lock:
                        self.current_requests = max(0, self.current_requests - 1)
                
                # Brief pause to prevent CPU hogging
                time.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Error in queue processing thread: {str(e)}")
            
        finally:
            with self.queue_lock:
                self.queue_processing = False
                
            logger.info("Queue processing thread finished")"""
            
    # Replace indentation - the string literal has 4 spaces but we need to match file's style
    # Extract indentation from another method in the file
    indent_match = re.search(r"^( +)def ", content, re.MULTILINE)
    if indent_match:
        proper_indent = indent_match.group(1)
        queue_method = queue_method.replace("    ", proper_indent)
    
    return content[:insert_position] + queue_method + content[insert_position:]

# Function to add or update request method with backoff and queue
def add_backoff_queue_to_request_method(content, api_type):
    """Add or update request method with backoff and queue functionality"""
    # Identify the primary request method name based on API type
    if api_type == "groq":
        method_name = "make_post_request_groq"
    elif api_type == "claude":
        method_name = "make_post_request"
    elif api_type == "openai":
        method_name = "make_request"
    else:
        method_name = "make_post_request"  # Default name
    
    # Check if method exists
    method_match = re.search(f"def {method_name}", content)
    if not method_match:
        print(f"Warning: Could not find request method '{method_name}' in the file")
        return content
    
    # Find the entire method definition
    method_pattern = f"def {method_name}.*?(?=\n    def |\n\nclass |$)"
    method_match = re.search(method_pattern, content, re.DOTALL)
    if not method_match:
        print(f"Warning: Could not extract complete method '{method_name}'")
        return content
    
    old_method = method_match.group(0)
    
    # Create new method with backoff and queue
    # Extract parameters from old method signature
    param_pattern = f"def {method_name}\\((.*?)\\)"
    param_match = re.search(param_pattern, old_method)
    if not param_match:
        print(f"Warning: Could not extract parameters for method '{method_name}'")
        return content
    
    params = param_match.group(1)
    
    # Add request_id parameter if not present
    if "request_id" not in params:
        if params.strip():
            params = params + ", request_id=None"
        else:
            params = "self, request_id=None"
    
    # Create the new method with backoff and queue
    queue_code = f"""
    def {method_name}({params}):
        \"\"\"Make a request with exponential backoff and queue\"\"\"
        if not api_key:
            api_key = self.api_key
            
        if not api_key:
            raise ValueError("No API key provided for authentication")
        
        # If queue is enabled and we're at capacity, add to queue
        if hasattr(self, "queue_enabled") and self.queue_enabled:
            with self.queue_lock:
                if self.current_requests >= self.max_concurrent_requests:
                    # Create a future to store the result
                    result_future = {{"result": None, "error": None, "completed": False}}
                    
                    # Add to queue with all necessary info to process later
                    request_info = {{
                        "endpoint_url": endpoint_url,
                        "data": data,
                        "api_key": api_key,
                        "request_id": request_id,
                        "future": result_future
                    }}
                    
                    # Check if queue is full
                    if len(self.request_queue) >= self.queue_size:
                        raise ValueError(f"Request queue is full ({{self.queue_size}} items). Try again later.")
                    
                    # Add to queue
                    self.request_queue.append(request_info)
                    logger.info(f"Request queued. Queue size: {{len(self.request_queue)}}")
                    
                    # Start queue processing if not already running
                    if not self.queue_processing:
                        threading.Thread(target=self._process_queue).start()
                    
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
                self.current_requests += 1
            
        # Generate request ID if not provided
        if request_id is None:
            request_id = f"req_{{int(time.time())}}_{{hashlib.md5(str(data).encode()).hexdigest()[:8]}}"
        
        # Use exponential backoff retry mechanism
        retries = 0
        retry_delay = self.initial_retry_delay if hasattr(self, "initial_retry_delay") else 1
        max_retries = self.max_retries if hasattr(self, "max_retries") else 3
        backoff_factor = self.backoff_factor if hasattr(self, "backoff_factor") else 2
        max_retry_delay = self.max_retry_delay if hasattr(self, "max_retry_delay") else 60
        
        while retries < max_retries:
            try:"""
    
    # Extract the inner logic of the method
    inner_code_pattern = r"(?<=try:)(.*?)(?=except|return\s+None\Z)"
    inner_code_match = re.search(inner_code_pattern, old_method, re.DOTALL)
    
    if inner_code_match:
        inner_code = inner_code_match.group(1)
    else:
        # Fallback to extracting everything after the method signature
        signature_end = re.search(f"def {method_name}.*?:", old_method).end()
        inner_code = old_method[signature_end:].split("try:")[1].split("except")[0]
    
    # Check if there's an existing except block
    except_pattern = r"except.*?:(.*?)(?=\n\s*return None|\Z)"
    except_match = re.search(except_pattern, old_method, re.DOTALL)
    
    if except_match:
        except_code = except_match.group(0)
        # Modify to add retries
        new_except_code = """
                except requests.exceptions.RequestException as e:
                    if retries < max_retries - 1:
                        logger.warning(f"Request failed: {str(e)}. Retrying in {retry_delay} seconds (attempt {retries+1}/{max_retries})...")
                        time.sleep(retry_delay)
                        retries += 1
                        retry_delay = min(retry_delay * backoff_factor, max_retry_delay)
                    else:
                        logger.error(f"Request failed after {max_retries} attempts: {str(e)}")
                        
                        # Decrement counter if queue enabled
                        if hasattr(self, "queue_enabled") and self.queue_enabled:
                            with self.queue_lock:
                                self.current_requests = max(0, self.current_requests - 1)
                        
                        raise
                
                except Exception as e:
                    # Decrement counter if queue enabled for any other exceptions
                    if hasattr(self, "queue_enabled") and self.queue_enabled:
                        with self.queue_lock:
                            self.current_requests = max(0, self.current_requests - 1)
                    raise"""
    else:
        new_except_code = """
                except Exception as e:
                    # Decrement counter if queue enabled
                    if hasattr(self, "queue_enabled") and self.queue_enabled:
                        with self.queue_lock:
                            self.current_requests = max(0, self.current_requests - 1)
                    raise"""
    
    # Assembly the final method
    new_method = queue_code + inner_code + new_except_code + """
                        
            # Decrement counter if we somehow exit the loop without returning or raising
            if hasattr(self, "queue_enabled") and self.queue_enabled:
                with self.queue_lock:
                    self.current_requests = max(0, self.current_requests - 1)
                    
            # This should never be reached due to the raise in the exception handler
            return None"""
    
    # Adjust indentation
    # Find the indentation from the original method
    indent_match = re.search(r"^( +)def " + method_name, old_method, re.MULTILINE)
    if indent_match:
        proper_indent = indent_match.group(1)
        # Adjust the indentation in the new method
        new_method = new_method.replace("    ", proper_indent)
    
    # Replace old method with new one
    return content.replace(old_method, new_method)

def process_file(file_path, api_type):
    """Process a single API file to add queue and backoff"""
    print(f"Processing {file_path} as {api_type} API...")
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Add threading import if needed
        content = add_threading_import(content)
        
        # Find __init__ method and add queue/backoff settings
        init_pattern = r"def __init__.*?return None"
        init_match = re.search(init_pattern, content, re.DOTALL)
        
        if init_match:
            init_method = init_match.group(0)
            # Check if settings already exist
            if "queue_enabled" not in init_method:
                # Add settings before "return None"
                new_init = init_method.replace("return None", INIT_TEMPLATE + "        return None")
                content = content.replace(init_method, new_init)
        
        # Add queue processing method
        content = add_queue_processing(content)
        
        # Add backoff and queue to request method
        content = add_backoff_queue_to_request_method(content, api_type)
        
        # Write updated content back to file
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"Successfully updated {file_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Add queue and backoff mechanisms to API backends")
    parser.add_argument("--api", "-a", help="Specific API to update", 
                       choices=["groq", "claude", "gemini", "openai", "ollama", "hf_tgi", "hf_tei", "llvm", "opea", "ovms", "s3_kit", "all"])
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
        "groq.py": "groq",
        "claude.py": "claude",
        "gemini.py": "gemini",
        "openai_api.py": "openai",
        "ollama.py": "ollama",
        "hf_tgi.py": "hf_tgi",
        "hf_tei.py": "hf_tei",
        "llvm.py": "llvm",
        "opea.py": "opea",
        "ovms.py": "ovms",
        "s3_kit.py": "s3_kit"
    }
    
    # Process requested API(s)
    if args.api == "all":
        apis_to_process = list(api_files.items())
    elif args.api:
        # Find the filename for the specified API
        api_filename = next((k for k, v in api_files.items() if v == args.api), None)
        if not api_filename:
            print(f"Error: Unknown API '{args.api}'")
            return
        apis_to_process = [(api_filename, args.api)]
    else:
        # Default to processing all
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
            success = process_file(file_path, api_type)
            results.append((filename, api_type, success))
    
    # Print summary
    if not args.dry_run:
        print("\n=== Summary ===")
        for filename, api_type, success in results:
            print(f"{filename}: {'✓ Success' if success else '✗ Failed'}")
        
        success_count = sum(1 for _, _, success in results if success)
        print(f"\nSuccessfully updated {success_count} of {len(results)} API backends")

if __name__ == "__main__":
    main()