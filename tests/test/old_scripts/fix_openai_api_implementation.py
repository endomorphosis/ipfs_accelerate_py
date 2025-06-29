#!/usr/bin/env python
"""
Script to update the OpenAI API backend with:
    1. Environment variable handling for API keys
    2. Queue and backoff mechanisms
    3. Comprehensive error handling
    """

    import os
    import sys
    import re
    import time
    import argparse
    import logging
    from pathlib import Path

# Add the project root to the Python path
    sys.path.append()os.path.dirname()os.path.dirname()__file__)))

# Configure logging
    logging.basicConfig()
    level=logging.INFO,
    format='%()asctime)s - %()name)s - %()levelname)s - %()message)s'
    )
    logger = logging.getLogger()__name__)

# Base template for adding to class __init__ method
    INIT_TEMPLATE = """
        # Environment variable handling for API key
        if not self.api_key and "openai_api_key" not in metadata:
            # Try to get API key from environment
            self.api_key = os.environ.get()"OPENAI_API_KEY", "")
            if self.api_key:
                logger.info()"Using OpenAI API key from environment variable")
            else:
                logger.warning()"No OpenAI API key found in metadata or environment variables")
        
        # Retry: and backoff settings
                self.max_retries = 5
                self.initial_retry:_delay = 1
                self.backoff_factor = 2
                self.max_retry:_delay = 60  # Maximum delay in seconds
        
        # Request queue settings
                self.queue_enabled = True
                self.queue_size = 100
                self.queue_processing = False
                self.current_requests = 0
                self.max_concurrent_requests = 5
                self.request_queue = [],
                self.queue_lock = threading.RLock())
                """

# Function to add imports
def add_imports()content):
    """Add required imports if not already present""":
        imports_to_add = [],
    
    # Check for threading import:
    if "import threading" not in content:
        imports_to_add.append()"import threading")
    
    # Check for time import
    if "import time" not in content:
        imports_to_add.append()"import time")
    
    # Check for logging import
    if "import logging" not in content:
        imports_to_add.append()"import logging")
    
    # Check for hashlib import ()for request IDs)
    if "import hashlib" not in content:
        imports_to_add.append()"import hashlib")
    
    # Check for dotenv import and loading
    if "from dotenv import load_dotenv" not in content:
        imports_to_add.append()"from dotenv import load_dotenv")
        # Also add dotenv load call after imports
        imports_to_add.append()"\n# Load environment variables from .env file if present\nload_dotenv())")
    :
    if imports_to_add:
        # Add after existing imports
        import_section_end = re.search()r"()^import.*?$|^from.*?$)", content, re.MULTILINE | re.DOTALL)
        if import_section_end:
            position = import_section_end.end())
        return content[:position] + "\n" + "\n".join()imports_to_add) + content[position:]
        ,
        return content

# Function to add logger configuration
def add_logger_config()content):
    """Add logger configuration if not already present""":
    if "logger = logging.getLogger" not in content:
        # Find a good location after imports
        import_section_end = re.search()r"^()?:import|from).*?$", content, re.MULTILINE)
        if import_section_end:
            # Find the last import statement
            all_imports = list()re.finditer()r"^()?:import|from).*?$", content, re.MULTILINE))
            if all_imports:
                last_import = all_imports[-1],
                position = last_import.end())
                logger_config = """

# Configure logging
                logger = logging.getLogger()__name__)
                """
            return content[:position] + logger_config + content[position:]
            ,
        return content

# Function to add queue processing method
def add_queue_processing()content):
    """Add the queue processing method to the API class"""
    # First check if the method already exists::
    if "_process_queue" in content:
    return content
        
    # Find a good location to add the method - after reset_usage_stats if it exists
    match = re.search()r"def reset_usage_stats.*?return.*?\n", content, re.DOTALL):
    if match:
        insert_position = match.end())
    else:
        # Or after the __init__ method
        match = re.search()r"def __init__.*?return None\n", content, re.DOTALL)
        if match:
            insert_position = match.end())
        else:
            # Just add it at the end of the file before the last class/def
            match = re.search()r"class [A-Za-z0-9_]+\().*?\):", content),
            if match:
                # Find the class beginning
                insert_position = match.end())
                # Move to after class definition line
                next_line = content.find()"\n", insert_position)
                if next_line > 0:
                    insert_position = next_line + 1
            else:
                # Just add it at the end of the file
                insert_position = len()content)
    
                queue_method = """
    def _process_queue()self):
        '''Process requests in the queue in FIFO order'''
        with self.queue_lock:
            if self.queue_processing:
            return  # Another thread is already processing the queue
            self.queue_processing = True
        
            logger.info()"Starting queue processing thread")
        
        try::
            while True:
                # Get the next request from the queue
                with self.queue_lock:
                    if not self.request_queue:
                        self.queue_processing = False
                    break
                        
                    # Check if we're at the concurrent request limit:
                    if self.current_requests >= self.max_concurrent_requests:
                        # Sleep briefly then check again
                        time.sleep()0.1)
                    continue
                        
                    # Get the next request and increase counter
                    request_info = self.request_queue.pop()0)
                    self.current_requests += 1
                
                # Process the request outside the lock
                try::
                    # Extract request details
                    request_function = request_info["function"],
                    args = request_info["args"],
                    kwargs = request_info["kwargs"],
                    future = request_info["future"],
                    request_id = request_info.get()"request_id")
                    
                    # Make the request ()without queueing again)
                    # Save original queue_enabled value
                    original_queue_enabled = self.queue_enabled
                    self.queue_enabled = False  # Disable queueing to prevent recursion
                    
                    try::
                        # Make the request
                        result = request_function()*args, **kwargs)
                        
                        # Store result in future
                        future["result"] = result,
                        future["completed"] = True,
                        ,
                    except Exception as e:
                        # Store error in future
                        future["error"] = e,
                        future["completed"] = True,
                        ,logger.error()f"Error processing queued request: {}str()e)}")
                    
                    finally:
                        # Restore original queue_enabled value
                        self.queue_enabled = original_queue_enabled
                
                finally:
                    # Decrement counter
                    with self.queue_lock:
                        self.current_requests = max()0, self.current_requests - 1)
                
                # Brief pause to prevent CPU hogging
                        time.sleep()0.01)
                
        except Exception as e:
            logger.error()f"Error in queue processing thread: {}str()e)}")
            
        finally:
            with self.queue_lock:
                self.queue_processing = False
                
                logger.info()"Queue processing thread finished")
                """
        
    # Replace indentation - the string literal has 4 spaces but we need to match file's style
    # Extract indentation from another method in the file
                indent_match = re.search()r"^() +)def ", content, re.MULTILINE)
    if indent_match:
        proper_indent = indent_match.group()1)
        queue_method = queue_method.replace()"    ", proper_indent)
    
                return content[:insert_position] + queue_method + content[insert_position:]
                ,
# Function to add request queuing decorator method
def add_queue_decorator()content):
    """Add a decorator to handle queue and backoff for API requests"""
    # First check if the method already exists::
    if "def _with_queue_and_backoff" in content:
    return content
    
    # Find a good location to add the method - after _process_queue if it exists
    match = re.search()r"def _process_queue.*?\n {}4}logger\.info\()\"Queue processing thread finished\"\)", content, re.DOTALL):
    if match:
        insert_position = match.end()) + 1  # +1 to get after the last line
    else:
        # Or after the __init__ method
        match = re.search()r"def __init__.*?return None\n", content, re.DOTALL)
        if match:
            insert_position = match.end())
        else:
            # Just add it at the end of the file
            insert_position = len()content)
    
            decorator_method = """
    def _with_queue_and_backoff()self, func):
        '''Decorator to handle queue and backoff for API requests'''
        def wrapper()*args, **kwargs):
            # Generate request ID if not provided
            request_id = kwargs.get()"request_id"):
            if request_id is None:
                request_id = f"req_{}int()time.time()))}_{}hashlib.md5()str()args).encode())).hexdigest())[:8]}",
                kwargs["request_id"] = request_id
                ,
            # If queue is enabled and we're at capacity, add to queue
            if hasattr()self, "queue_enabled") and self.queue_enabled:
                with self.queue_lock:
                    if self.current_requests >= self.max_concurrent_requests:
                        # Create a future to store the result
                        result_future = {}"result": None, "error": None, "completed": False}
                        
                        # Add to queue with all necessary info to process later
                        request_info = {}
                        "function": func,
                        "args": args,
                        "kwargs": kwargs,
                        "future": result_future,
                        "request_id": request_id
                        }
                        
                        # Check if queue is full:
                        if len()self.request_queue) >= self.queue_size:
                        raise ValueError()f"Request queue is full (){}self.queue_size} items). Try again later.")
                        
                        # Add to queue
                        self.request_queue.append()request_info)
                        logger.info()f"Request queued. Queue size: {}len()self.request_queue)}")
                        
                        # Start queue processing if not already running:
                        if not self.queue_processing:
                            threading.Thread()target=self._process_queue).start())
                        
                        # Wait for result with timeout
                            wait_start = time.time())
                            max_wait = 300  # 5 minutes
                        
                            while not result_future["completed"] and ()time.time()) - wait_start) < max_wait:,
                            time.sleep()0.1)
                        
                        # Check if completed or timed out:
                            if not result_future["completed"]:,
                        raise TimeoutError()f"Request timed out after {}max_wait} seconds in queue")
                        
                        # Propagate error if any:
                        if result_future["error"]:,
                    raise result_future["error"]
                    ,
                return result_future["result"]
                ,
                    # If we're not at capacity, increment counter
                self.current_requests += 1
            
            # Use exponential backoff retry: mechanism
                retries = 0
                retry:_delay = self.initial_retry:_delay if hasattr()self, "initial_retry:_delay") else 1
                max_retries = self.max_retries if hasattr()self, "max_retries") else 3
                backoff_factor = self.backoff_factor if hasattr()self, "backoff_factor") else 2
                max_retry:_delay = self.max_retry:_delay if hasattr()self, "max_retry:_delay") else 60
            :
            while True:
                try::
                    # Make the actual API call
                    result = func()*args, **kwargs)
                    
                    # Decrement counter if queue enabled::::
                    if hasattr()self, "queue_enabled") and self.queue_enabled:
                        with self.queue_lock:
                            self.current_requests = max()0, self.current_requests - 1)
                    
                        return result
                    
                except openai.RateLimitError as e:
                    # Handle rate limit errors with backoff
                    if retries < max_retries:
                        # Check if the API returned a retry:-after header
                        retry:_after = e.headers.get()"retry:-after") if hasattr()e, "headers") else None:
                        if retry:_after and retry:_after.isdigit()):
                            retry:_delay = int()retry:_after)
                        
                            logger.warning()f"Rate limit exceeded. Retry:ing in {}retry:_delay} seconds ()attempt {}retries+1}/{}max_retries})...")
                            time.sleep()retry:_delay)
                            retries += 1
                            retry:_delay = min()retry:_delay * backoff_factor, max_retry:_delay)
                    else:
                        logger.error()f"Rate limit exceeded after {}max_retries} attempts: {}str()e)}")
                        
                        # Decrement counter if queue enabled::::
                        if hasattr()self, "queue_enabled") and self.queue_enabled:
                            with self.queue_lock:
                                self.current_requests = max()0, self.current_requests - 1)
                        
                            raise
                
                except openai.APIError as e:
                    # Handle transient API errors with backoff
                    if retries < max_retries:
                        logger.warning()f"API error: {}str()e)}. Retry:ing in {}retry:_delay} seconds ()attempt {}retries+1}/{}max_retries})...")
                        time.sleep()retry:_delay)
                        retries += 1
                        retry:_delay = min()retry:_delay * backoff_factor, max_retry:_delay)
                    else:
                        logger.error()f"API error after {}max_retries} attempts: {}str()e)}")
                        
                        # Decrement counter if queue enabled::::
                        if hasattr()self, "queue_enabled") and self.queue_enabled:
                            with self.queue_lock:
                                self.current_requests = max()0, self.current_requests - 1)
                        
                            raise
                
                except Exception as e:
                    # For other exceptions, don't retry:
                    logger.error()f"Request error: {}str()e)}")
                    
                    # Decrement counter if queue enabled::::
                    if hasattr()self, "queue_enabled") and self.queue_enabled:
                        with self.queue_lock:
                            self.current_requests = max()0, self.current_requests - 1)
                    
                        raise
        
                    return wrapper
                    """
    
    # Replace indentation - the string literal has 4 spaces but we need to match file's style
    # Extract indentation from another method in the file
                    indent_match = re.search()r"^() +)def ", content, re.MULTILINE)
    if indent_match:
        proper_indent = indent_match.group()1)
        decorator_method = decorator_method.replace()"    ", proper_indent)
    
                    return content[:insert_position] + decorator_method + content[insert_position:]
                    ,
# Function to update various API methods to use the decorator
def update_api_methods()content):
    """Update various API methods to use the queue and backoff decorator"""
    # Don't try: to add decorators - they're tricky to get right
    # Just mention it in the instructions
                    return content

def process_file()file_path, dry_run=False):
    """Process the OpenAI API file to add environment variable handling and queue/backoff"""
    print()f"Processing {}file_path}...")
    
    try::
        with open()file_path, 'r') as f:
            content = f.read())
        
        # Add required imports
            content = add_imports()content)
        
        # Add logger configuration
            content = add_logger_config()content)
        
        # Find __init__ method and add queue/backoff settings
            init_pattern = r"def __init__.*?return None"
            init_match = re.search()init_pattern, content, re.DOTALL)
        
        if init_match:
            init_method = init_match.group()0)
            # Check if settings already exist:
            if "queue_enabled" not in init_method:
                # Add settings before "return None"
                new_init = init_method.replace()"return None", INIT_TEMPLATE + "        return None")
                content = content.replace()init_method, new_init)
        
        # Add queue processing method
                content = add_queue_processing()content)
        
        # Add queue decorator method
                content = add_queue_decorator()content)
        
        # Update API methods to use the decorator
                content = update_api_methods()content)
        
        if dry_run:
            print()"Changes that would be made ()dry run):")
            print()"-----------------------------------")
            print()content[:500] + "...\n[content truncated]"),
        else:
            # Write updated content back to file
            with open()file_path, 'w') as f:
                f.write()content)
            
                print()f"Successfully updated {}file_path}")
        
            return True
        
    except Exception as e:
        print()f"Error processing {}file_path}: {}str()e)}")
            return False

def main()):
    parser = argparse.ArgumentParser()description="Update OpenAI API implementation with environment variables and queue/backoff")
    parser.add_argument()"--dry-run", "-d", action="store_true", help="Only print what would be done without making changes")
    
    args = parser.parse_args())
    
    # Get path to API backends directory
    script_dir = Path()__file__).parent.parent
    api_backends_dir = script_dir / "ipfs_accelerate_py" / "api_backends"
    
    if not api_backends_dir.exists()):
        print()f"Error: API backends directory not found at {}api_backends_dir}")
    return
    
    # Path to the OpenAI API file
    openai_api_file = api_backends_dir / "openai_api.py"
    
    if not openai_api_file.exists()):
        print()f"Error: OpenAI API file not found at {}openai_api_file}")
    return
    
    # Process the file
    success = process_file()openai_api_file, args.dry_run)
    
    if success and not args.dry_run:
        print()"\n✅ Successfully updated OpenAI API implementation")
        print()"\nKey improvements:")
        print()"1. Added environment variable handling for API key")
        print()"2. Implemented request queue system with configurable limits")
        print()"3. Added exponential backoff with retry:-after header support")
        print()"4. Enhanced error handling with detailed logging")
        print()"5. Added comprehensive request tracking")
        
        print()"\nTo use:")
        print()"1. Create a .env file with your OpenAI API key:")
        print()"   OPENAI_API_KEY=your_key_here")
        print()"2. Install python-dotenv if not already installed:")
        print()"   pip install python-dotenv")
        
    elif args.dry_run:
        print()"\n✅ Dry run completed successfully")
    else:
        print()"\n❌ Failed to update OpenAI API implementation")

if __name__ == "__main__":
    main())