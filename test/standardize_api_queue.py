#!/usr/bin/env python
"""
Standardize API Queue Implementation

This script standardizes the queue implementation across all API backends by:
1. Ensuring all APIs use list-based queues consistently
2. Fixing queue processing methods to work with the list implementation
3. Resolving syntax and indentation errors in queue-related code

The script will modify all API backend files to use a consistent pattern.
"""

import os
import sys
import re
from pathlib import Path
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("standardize_api_queue")

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Standard queue initialization template
QUEUE_INIT_TEMPLATE = """
        # Standard queue implementation (list-based)
        self.request_queue = []  # List-based queue for simplicity
        self.queue_size = 100  # Maximum queue size
        self.queue_lock = threading.RLock()  # Thread-safe access
        self.queue_processing = False
        self.max_concurrent_requests = 5
        self.active_requests = 0
"""

# Standard queue processing template
QUEUE_PROCESSING_TEMPLATE = """
    def _process_queue(self):
        \"\"\"Process requests in the queue with standard pattern\"\"\"
        with self.queue_lock:
            if self.queue_processing:
                return  # Another thread is already processing
            self.queue_processing = True
        
        try:
            while True:
                # Get the next request from the queue
                request_info = None
                
                with self.queue_lock:
                    if not self.request_queue:
                        self.queue_processing = False
                        break
                        
                    # Check if we're at capacity
                    if self.active_requests >= self.max_concurrent_requests:
                        time.sleep(0.1)  # Brief pause
                        continue
                        
                    # Get next request and increment counter
                    request_info = self.request_queue.pop(0)
                    self.active_requests += 1
                
                # Process the request outside the lock
                if request_info:
                    try:
                        # Extract request details
                        future = request_info.get("future")
                        func = request_info.get("func")
                        args = request_info.get("args", [])
                        kwargs = request_info.get("kwargs", {})
                        
                        # Special handling for different request formats
                        if func and callable(func):
                            # Function-based request
                            try:
                                result = func(*args, **kwargs)
                                if future:
                                    future["result"] = result
                                    future["completed"] = True
                            except Exception as e:
                                if future:
                                    future["error"] = e
                                    future["completed"] = True
                                logger.error(f"Error executing queued function: {e}")
                        else:
                            # Direct API request format
                            endpoint_url = request_info.get("endpoint_url")
                            data = request_info.get("data")
                            api_key = request_info.get("api_key")
                            request_id = request_info.get("request_id")
                            
                            if hasattr(self, "make_request"):
                                method = self.make_request
                            elif hasattr(self, "make_post_request"):
                                method = self.make_post_request
                            else:
                                raise AttributeError("No request method found")
                            
                            # Temporarily disable queueing to prevent recursion
                            original_queue_enabled = getattr(self, "queue_enabled", True)
                            setattr(self, "queue_enabled", False)
                            
                            try:
                                result = method(
                                    endpoint_url=endpoint_url,
                                    data=data,
                                    api_key=api_key,
                                    request_id=request_id
                                )
                                
                                if future:
                                    future["result"] = result
                                    future["completed"] = True
                            except Exception as e:
                                if future:
                                    future["error"] = e
                                    future["completed"] = True
                                logger.error(f"Error processing queued request: {e}")
                            finally:
                                # Restore original queue_enabled
                                setattr(self, "queue_enabled", original_queue_enabled)
                    
                    finally:
                        # Decrement counter
                        with self.queue_lock:
                            self.active_requests = max(0, self.active_requests - 1)
                
                # Brief pause to prevent CPU hogging
                time.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Error in queue processing thread: {e}")
            
        finally:
            # Reset queue processing flag
            with self.queue_lock:
                self.queue_processing = False
"""

def find_and_fix_queue_init(content):
    """Find and fix queue initialization code"""
    # Look for existing queue initialization in __init__ method
    init_pattern = r"def __init__.*?(?:return None|\s*(?:pass|return))"
    init_match = re.search(init_pattern, content, re.DOTALL)
    
    if not init_match:
        logger.warning("Could not find __init__ method")
        return content
        
    init_method = init_match.group(0)
    
    # Check if queue initialization already exists
    if "self.request_queue = []" in init_method or "self.request_queue = Queue" in init_method:
        # Replace existing queue initialization code
        # Look for patterns indicating the queue initialization section
        queue_pattern = r"(?:# Queue (?:settings|initialization).*?(?=\n\s*#|\n\s*self\.\w+\s*=(?!\s*\[|\s*Queue)|\n\s*return|\n\s*$))"
        queue_match = re.search(queue_pattern, init_method, re.DOTALL)
        
        if queue_match:
            # Replace existing queue initialization
            old_queue_init = queue_match.group(0)
            indent = re.match(r"^(\s+)", old_queue_init)
            proper_indent = indent.group(1) if indent else "        "
            new_queue_init = QUEUE_INIT_TEMPLATE.replace("        ", proper_indent)
            
            updated_init = init_method.replace(old_queue_init, new_queue_init)
            content = content.replace(init_method, updated_init)
        else:
            # Add standardized queue initialization if no pattern found
            updated_init = init_method.replace("return None", QUEUE_INIT_TEMPLATE + "        return None")
            content = content.replace(init_method, updated_init)
    else:
        # Add standardized queue initialization
        updated_init = init_method.replace("return None", QUEUE_INIT_TEMPLATE + "        return None")
        content = content.replace(init_method, updated_init)
    
    return content

def find_and_fix_process_queue(content):
    """Find and fix the _process_queue method"""
    # Check if _process_queue already exists
    queue_pattern = r"def _process_queue\(self.*?\).*?(?=\n    def |\n\nclass |\Z)"
    queue_match = re.search(queue_pattern, content, re.DOTALL)
    
    if queue_match:
        # Replace existing implementation
        old_queue_method = queue_match.group(0)
        
        # Determine proper indentation
        indent_match = re.search(r"^(\s+)def ", old_queue_method, re.MULTILINE)
        proper_indent = indent_match.group(1) if indent_match else "    "
        
        # Adjust indentation in template
        new_queue_method = QUEUE_PROCESSING_TEMPLATE.replace("    ", proper_indent)
        
        # Replace old method with new one
        content = content.replace(old_queue_method, new_queue_method)
    else:
        # Add new implementation after __init__ method
        init_pattern = r"def __init__.*?(?:return None|\s*(?:pass|return))"
        init_match = re.search(init_pattern, content, re.DOTALL)
        
        if init_match:
            # Insert after __init__ method
            insert_pos = init_match.end()
            content = content[:insert_pos] + "\n" + QUEUE_PROCESSING_TEMPLATE + content[insert_pos:]
        else:
            # Add at the end of the class
            class_pattern = r"class .*?:"
            class_match = re.search(class_pattern, content)
            
            if class_match:
                # Find the end of the class
                class_pos = class_match.end()
                content = content[:class_pos] + "\n" + QUEUE_PROCESSING_TEMPLATE + content[class_pos:]
            else:
                # Just append to the end
                content += "\n" + QUEUE_PROCESSING_TEMPLATE
    
    return content

def fix_queue_usage_in_methods(content):
    """Fix queue usage in other methods like make_request"""
    # Find make_request or similar methods
    request_pattern = r"def (?:make_request|make_post_request|make_post_request_\w+)\(self.*?\).*?(?=\n    def |\n\nclass |\Z)"
    request_matches = re.finditer(request_pattern, content, re.DOTALL)
    
    for match in request_matches:
        method = match.group(0)
        updated_method = method
        
        # Fix queue-related operations
        # 1. Replace any Queue.get() operations
        updated_method = re.sub(
            r"self\.request_queue\.get\(.*?\)",
            "self.request_queue.pop(0) if self.request_queue else None",
            updated_method
        )
        
        # 2. Replace Queue.put() operations
        updated_method = re.sub(
            r"self\.request_queue\.put\((.+?)\)",
            r"self.request_queue.append(\1)",
            updated_method
        )
        
        # 3. Replace qsize() checks
        updated_method = re.sub(
            r"self\.request_queue\.qsize\(\)",
            "len(self.request_queue)",
            updated_method
        )
        
        # Update the method
        if updated_method != method:
            content = content.replace(method, updated_method)
    
    return content

def process_api_file(file_path):
    """Process a single API file to standardize queue implementation"""
    logger.info(f"Processing {file_path}...")
    
    try:
        # Create backup
        backup_path = f"{file_path}.bak"
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup at {backup_path}")
        
        # Read file content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Apply fixes
        content = find_and_fix_queue_init(content)
        content = find_and_fix_process_queue(content)
        content = fix_queue_usage_in_methods(content)
        
        # Ensure imports
        if "import threading" not in content:
            # Add threading import
            content = "import threading\n" + content
        
        if "import time" not in content:
            # Add time import
            content = "import time\n" + content
        
        # Write updated content
        with open(file_path, 'w') as f:
            f.write(content)
            
        logger.info(f"✅ Successfully updated {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"❌ Error processing {file_path}: {e}")
        return False

def main():
    """Main function to process all API backend files"""
    # Find API backends directory
    script_dir = Path(__file__).parent.parent
    api_backends_dir = script_dir / "ipfs_accelerate_py" / "api_backends"
    
    if not api_backends_dir.exists():
        logger.error(f"API backends directory not found at {api_backends_dir}")
        return 1
    
    # API files to process
    api_files = [
        "claude.py",
        "openai_api.py",
        "groq.py",
        "gemini.py",
        "ollama.py",
        "hf_tgi.py",
        "hf_tei.py",
        "llvm.py",
        "opea.py",
        "ovms.py",
        "s3_kit.py"
    ]
    
    # Process each API file
    results = []
    for api_file in api_files:
        file_path = api_backends_dir / api_file
        if not file_path.exists():
            logger.warning(f"File {file_path} not found, skipping")
            continue
            
        success = process_api_file(file_path)
        results.append((api_file, success))
    
    # Print summary
    logger.info("\n=== Queue Standardization Summary ===")
    for file_name, success in results:
        logger.info(f"{file_name}: {'✅ Success' if success else '❌ Failed'}")
    
    success_count = sum(1 for _, success in results if success)
    logger.info(f"\nSuccessfully updated {success_count} of {len(results)} API backends")
    
    if success_count < len(results):
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())