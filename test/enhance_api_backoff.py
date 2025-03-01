#!/usr/bin/env python
"""
Advanced API Backend Enhancement Script
This script enhances API backends with:
1. Priority queue functionality
2. Circuit breaker pattern
3. Enhanced monitoring and reporting
4. Request batching optimization
"""

import os
import sys
import re
import time
import glob
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("enhance_api_backoff")

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Base template for priority queue implementation
PRIORITY_QUEUE_TEMPLATE = """
        # Priority levels
        self.PRIORITY_HIGH = 0
        self.PRIORITY_NORMAL = 1
        self.PRIORITY_LOW = 2
        
        # Change request queue to priority-based
        self.request_queue = []  # Will store (priority, request_info) tuples
"""

# Base template for circuit breaker implementation
CIRCUIT_BREAKER_TEMPLATE = """
        # Circuit breaker settings
        self.circuit_state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.failure_threshold = 5  # Number of failures before opening circuit
        self.reset_timeout = 30  # Seconds to wait before trying half-open
        self.failure_count = 0
        self.last_failure_time = 0
        self.circuit_lock = threading.RLock()
"""

# Base template for monitoring implementation
MONITORING_TEMPLATE = """
        # Request monitoring
        self.request_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "retried_requests": 0,
            "average_response_time": 0,
            "total_response_time": 0,
            "requests_by_model": {},
            "errors_by_type": {},
            "queue_wait_times": [],
            "backoff_delays": []
        }
        self.stats_lock = threading.RLock()
        
        # Enable metrics collection
        self.collect_metrics = True
"""

# Base template for request batching implementation
BATCHING_TEMPLATE = """
        # Batching settings
        self.batching_enabled = True
        self.max_batch_size = 10
        self.batch_timeout = 0.5  # Max seconds to wait for more requests
        self.batch_queue = {}  # Keyed by model name
        self.batch_timers = {}  # Timers for each batch
        self.batch_lock = threading.RLock()
        
        # Models that support batching
        self.embedding_models = []  # Models supporting batched embeddings
        self.completion_models = []  # Models supporting batched completions
        self.supported_batch_models = []  # All models supporting batching
"""

# Function to update an API module with priority queue implementation
def add_priority_queue(content):
    """Add priority queue capability to API backend"""
    # Check if already implemented
    if "PRIORITY_HIGH" in content:
        logger.info("Priority queue already implemented, skipping...")
        return content
    
    # Find the initialization section
    init_match = re.search(r"def __init__.*?self.queue_lock = threading.RLock\(\)", content, re.DOTALL)
    if not init_match:
        logger.warning("Could not find appropriate initialization section for priority queue")
        return content
    
    # Insert priority queue template
    init_end = init_match.end()
    new_content = content[:init_end] + PRIORITY_QUEUE_TEMPLATE + content[init_end:]
    
    # Update the queue processing method
    queue_process_pattern = r"def _process_queue\(self\):.*?while True:.*?self.request_queue.pop\(0\)"
    queue_process_match = re.search(queue_process_pattern, new_content, re.DOTALL)
    
    if queue_process_match:
        old_queue_code = queue_process_match.group(0)
        
        # Replace simple queue pop with priority queue processing
        new_queue_code = old_queue_code.replace(
            "self.request_queue.pop(0)",
            "self.request_queue.sort(key=lambda x: x[0])\n" + 
            "                    priority, request_info = self.request_queue.pop(0)"
        )
        
        # Update all references to request_info
        new_queue_code = re.sub(
            r"request_info\[(.*?)\]", 
            r"request_info[\1]", 
            new_queue_code
        )
        
        new_content = new_content.replace(old_queue_code, new_queue_code)
    
    # Add method to queue with priority
    queue_with_priority_method = """
    def queue_with_priority(self, request_info, priority=None):
        """Queue a request with a specific priority level."""
        if priority is None:
            priority = self.PRIORITY_NORMAL
            
        with self.queue_lock:
            # Check if queue is full
            if len(self.request_queue) >= self.queue_size:
                raise ValueError(f"Request queue is full ({self.queue_size} items). Try again later.")
            
            # Record queue entry time for metrics
            request_info["queue_entry_time"] = time.time()
            
            # Add to queue with priority
            self.request_queue.append((priority, request_info))
            
            # Sort queue by priority (lower numbers = higher priority)
            self.request_queue.sort(key=lambda x: x[0])
            
            logger.info(f"Request queued with priority {priority}. Queue size: {len(self.request_queue)}")
            
            # Start queue processing if not already running
            if not self.queue_processing:
                threading.Thread(target=self._process_queue).start()
                
            # Create future to track result
            future = {"result": None, "error": None, "completed": False}
            request_info["future"] = future
            return future
    """
    
    # Add method after _process_queue
    queue_method_pattern = r"def _process_queue\(self\):.*?(?=\n    def |\n\nclass |$)"
    queue_method_match = re.search(queue_method_pattern, new_content, re.DOTALL)
    
    if queue_method_match:
        queue_method_end = queue_method_match.end()
        new_content = new_content[:queue_method_end] + queue_with_priority_method + new_content[queue_method_end:]
    
    return new_content

# Function to add circuit breaker pattern
def add_circuit_breaker(content):
    """Add circuit breaker pattern to API backend"""
    # Check if already implemented
    if "circuit_state" in content:
        logger.info("Circuit breaker already implemented, skipping...")
        return content
    
    # Find the initialization section
    init_match = re.search(r"def __init__.*?self.queue_lock = threading.RLock\(\)", content, re.DOTALL)
    if not init_match:
        logger.warning("Could not find appropriate initialization section for circuit breaker")
        return content
    
    # Insert circuit breaker template
    init_end = init_match.end()
    new_content = content[:init_end] + CIRCUIT_BREAKER_TEMPLATE + content[init_end:]
    
    # Add circuit breaker methods
    circuit_breaker_methods = """
    def check_circuit_breaker(self):
        """Check if circuit breaker allows requests"""
        with self.circuit_lock:
            now = time.time()
            
            if self.circuit_state == "OPEN":
                # Check if enough time has passed to try again
                if now - self.last_failure_time > self.reset_timeout:
                    logger.info("Circuit breaker transitioning from OPEN to HALF-OPEN")
                    self.circuit_state = "HALF_OPEN"
                    return True
                else:
                    # Circuit is open, fail fast
                    return False
                    
            elif self.circuit_state == "HALF_OPEN":
                # In half-open state, we allow a single request to test the service
                return True
                
            else:  # CLOSED
                # Normal operation, allow requests
                return True

    def track_request_result(self, success, error_type=None):
        """Track the result of a request for circuit breaker logic"""
        with self.circuit_lock:
            if success:
                # Successful request
                if self.circuit_state == "HALF_OPEN":
                    # Service is working again, close the circuit
                    logger.info("Circuit breaker transitioning from HALF-OPEN to CLOSED")
                    self.circuit_state = "CLOSED"
                    self.failure_count = 0
                elif self.circuit_state == "CLOSED":
                    # Reset failure count on success
                    self.failure_count = 0
            else:
                # Failed request
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                # Update error statistics
                if error_type and hasattr(self, "collect_metrics") and self.collect_metrics:
                    with self.stats_lock:
                        if error_type not in self.request_stats["errors_by_type"]:
                            self.request_stats["errors_by_type"][error_type] = 0
                        self.request_stats["errors_by_type"][error_type] += 1
                
                if self.circuit_state == "CLOSED" and self.failure_count >= self.failure_threshold:
                    # Too many failures, open the circuit
                    logger.warning(f"Circuit breaker transitioning from CLOSED to OPEN after {self.failure_count} failures")
                    self.circuit_state = "OPEN"
                    
                    # Update circuit breaker statistics
                    if hasattr(self, "stats_lock") and hasattr(self, "request_stats"):
                        with self.stats_lock:
                            if "circuit_breaker_trips" not in self.request_stats:
                                self.request_stats["circuit_breaker_trips"] = 0
                            self.request_stats["circuit_breaker_trips"] += 1
                    
                elif self.circuit_state == "HALF_OPEN":
                    # Failed during test request, back to open
                    logger.warning("Circuit breaker transitioning from HALF-OPEN to OPEN after test request failure")
                    self.circuit_state = "OPEN"
    """
    
    # Add methods at end of class
    class_end_pattern = r"\nclass .*?:\n.*?(?=\nclass |\n# |\Z)"
    class_match = re.search(class_end_pattern, new_content, re.DOTALL)
    
    if class_match:
        class_end = class_match.end()
        new_content = new_content[:class_end] + circuit_breaker_methods + new_content[class_end:]
    
    # Modify make_request method to use circuit breaker
    request_method_pattern = r"def make_(?:post_)?request\(self.*?\):"
    request_method_match = re.search(request_method_pattern, new_content)
    
    if request_method_match:
        # Add check at beginning of method
        request_method_start = request_method_match.end()
        circuit_check_code = """
        # Check circuit breaker first
        if hasattr(self, "check_circuit_breaker") and not self.check_circuit_breaker():
            raise Exception(f"Circuit breaker is OPEN. Service appears to be unavailable. Try again in {self.reset_timeout} seconds.")
        """
        
        new_content = new_content[:request_method_start] + circuit_check_code + new_content[request_method_start:]
        
        # Update try/except to track result
        try_pattern = r"try:.*?except.*?(?=\n\s*return |\Z)"
        try_match = re.search(try_pattern, new_content, re.DOTALL)
        
        if try_match:
            try_block = try_match.group(0)
            
            # Add success tracking at end of try block
            success_track = """
                    # Track successful request for circuit breaker
                    if hasattr(self, "track_request_result"):
                        self.track_request_result(True)
            """
            
            # Add error tracking in except block
            error_track = """
                    # Track failed request for circuit breaker
                    if hasattr(self, "track_request_result"):
                        error_type = type(e).__name__
                        self.track_request_result(False, error_type)
            """
            
            # Insert tracking code
            try_block_modified = re.sub(
                r"try:(.*?)except (.*?):(.*?)(?=\n\s*return |\Z)",
                r"try:\1" + success_track + r"except \2:\3" + error_track,
                try_block,
                flags=re.DOTALL
            )
            
            new_content = new_content.replace(try_block, try_block_modified)
    
    return new_content

# Function to add enhanced monitoring
def add_monitoring(content):
    """Add enhanced monitoring to API backend"""
    # Check if already implemented
    if "request_stats" in content:
        logger.info("Monitoring already implemented, skipping...")
        return content
    
    # Find the initialization section
    init_match = re.search(r"def __init__.*?self.queue_lock = threading.RLock\(\)", content, re.DOTALL)
    if not init_match:
        logger.warning("Could not find appropriate initialization section for monitoring")
        return content
    
    # Insert monitoring template
    init_end = init_match.end()
    new_content = content[:init_end] + MONITORING_TEMPLATE + content[init_end:]
    
    # Add monitoring methods
    monitoring_methods = """
    def update_stats(self, stats_update):
        """Update request statistics in a thread-safe way"""
        if not hasattr(self, "collect_metrics") or not self.collect_metrics:
            return
            
        with self.stats_lock:
            for key, value in stats_update.items():
                if key in self.request_stats:
                    if isinstance(self.request_stats[key], dict) and isinstance(value, dict):
                        # Update nested dictionary
                        for k, v in value.items():
                            if k in self.request_stats[key]:
                                self.request_stats[key][k] += v
                            else:
                                self.request_stats[key][k] = v
                    elif isinstance(self.request_stats[key], list) and not isinstance(value, dict):
                        # Append to list
                        self.request_stats[key].append(value)
                    elif key == "average_response_time":
                        # Special handling for average calculation
                        total = self.request_stats["total_response_time"] + stats_update.get("response_time", 0)
                        count = self.request_stats["total_requests"]
                        if count > 0:
                            self.request_stats["average_response_time"] = total / count
                    else:
                        # Simple addition for counters
                        self.request_stats[key] += value

    def get_stats(self):
        """Get a copy of the current request statistics"""
        if not hasattr(self, "stats_lock") or not hasattr(self, "request_stats"):
            return {}
            
        with self.stats_lock:
            # Return a copy to avoid thread safety issues
            return dict(self.request_stats)

    def reset_stats(self):
        """Reset all statistics"""
        if not hasattr(self, "stats_lock") or not hasattr(self, "request_stats"):
            return
            
        with self.stats_lock:
            self.request_stats = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "retried_requests": 0,
                "average_response_time": 0,
                "total_response_time": 0,
                "requests_by_model": {},
                "errors_by_type": {},
                "queue_wait_times": [],
                "backoff_delays": []
            }

    def generate_report(self, include_details=False):
        """Generate a report of API usage and performance"""
        if not hasattr(self, "get_stats") or not callable(self.get_stats):
            return {"error": "Statistics not available"}
            
        stats = self.get_stats()
        
        # Build report
        report = {
            "summary": {
                "total_requests": stats.get("total_requests", 0),
                "success_rate": (stats.get("successful_requests", 0) / stats.get("total_requests", 1)) * 100 if stats.get("total_requests", 0) > 0 else 0,
                "average_response_time": stats.get("average_response_time", 0),
                "retry_rate": (stats.get("retried_requests", 0) / stats.get("total_requests", 1)) * 100 if stats.get("total_requests", 0) > 0 else 0,
            },
            "models": stats.get("requests_by_model", {}),
            "errors": stats.get("errors_by_type", {})
        }
        
        # Add circuit breaker info if available
        if hasattr(self, "circuit_state"):
            report["circuit_breaker"] = {
                "state": self.circuit_state,
                "failure_count": self.failure_count,
                "failure_threshold": self.failure_threshold,
                "reset_timeout": self.reset_timeout,
                "trips": stats.get("circuit_breaker_trips", 0)
            }
        
        if include_details:
            # Add detailed metrics
            queue_wait_times = stats.get("queue_wait_times", [])
            backoff_delays = stats.get("backoff_delays", [])
            
            report["details"] = {
                "queue_wait_times": {
                    "min": min(queue_wait_times) if queue_wait_times else 0,
                    "max": max(queue_wait_times) if queue_wait_times else 0,
                    "avg": sum(queue_wait_times) / len(queue_wait_times) if queue_wait_times else 0,
                    "count": len(queue_wait_times)
                },
                "backoff_delays": {
                    "min": min(backoff_delays) if backoff_delays else 0,
                    "max": max(backoff_delays) if backoff_delays else 0,
                    "avg": sum(backoff_delays) / len(backoff_delays) if backoff_delays else 0,
                    "count": len(backoff_delays)
                }
            }
        
        return report
    """
    
    # Add methods at end of class
    class_end_pattern = r"\nclass .*?:\n.*?(?=\nclass |\n# |\Z)"
    class_match = re.search(class_end_pattern, new_content, re.DOTALL)
    
    if class_match:
        class_end = class_match.end()
        new_content = new_content[:class_end] + monitoring_methods + new_content[class_end:]
    
    # Update request method to track metrics
    request_method_pattern = r"def make_(?:post_)?request\(self.*?\):.*?try:.*?except"
    request_method_match = re.search(request_method_pattern, new_content, re.DOTALL)
    
    if request_method_match:
        # Add timing code at beginning of try block
        request_method = request_method_match.group(0)
        
        # Add start timing
        request_with_timing = request_method.replace(
            "try:",
            "try:\n            start_time = time.time()\n            model = data.get('model', '')\n            # Update request count\n            if hasattr(self, 'update_stats'):\n                self.update_stats({\n                    'total_requests': 1,\n                    'requests_by_model': {model: 1} if model else {}\n                })"
        )
        
        # Add end timing and success stats in try block
        result_pattern = r"(return.*?response.*?)\n"
        request_with_success = re.sub(
            result_pattern,
            r"\1\n            # Update success stats\n            if hasattr(self, 'update_stats'):\n                end_time = time.time()\n                self.update_stats({\n                    'successful_requests': 1,\n                    'total_response_time': end_time - start_time,\n                    'response_time': end_time - start_time\n                })\n",
            request_with_timing,
            flags=re.DOTALL
        )
        
        # Add failure stats in except block
        request_with_failure = request_with_success.replace(
            "except",
            "            # Update failure stats\n            if hasattr(self, 'update_stats'):\n                end_time = time.time()\n                error_type = type(e).__name__\n                self.update_stats({\n                    'failed_requests': 1,\n                    'errors_by_type': {error_type: 1}\n                })\n            except"
        )
        
        new_content = new_content.replace(request_method, request_with_failure)
    
    # Update retry section to track retries
    retry_pattern = r"retries \+= 1.*?time\.sleep\((?:retry_)?delay\)"
    retry_match = re.search(retry_pattern, new_content, re.DOTALL)
    
    if retry_match:
        retry_code = retry_match.group(0)
        
        # Add retry tracking
        retry_with_tracking = retry_code.replace(
            "retries += 1",
            "retries += 1\n                    # Track retry in stats\n                    if hasattr(self, 'update_stats'):\n                        self.update_stats({\n                            'retried_requests': 1,\n                            'backoff_delays': retry_delay\n                        })"
        )
        
        new_content = new_content.replace(retry_code, retry_with_tracking)
    
    return new_content

# Function to add request batching
def add_batching(content):
    """Add request batching capability to API backend"""
    # Check if already implemented
    if "batching_enabled" in content:
        logger.info("Request batching already implemented, skipping...")
        return content
    
    # Find the initialization section
    init_match = re.search(r"def __init__.*?self.queue_lock = threading.RLock\(\)", content, re.DOTALL)
    if not init_match:
        logger.warning("Could not find appropriate initialization section for batching")
        return content
    
    # Insert batching template
    init_end = init_match.end()
    new_content = content[:init_end] + BATCHING_TEMPLATE + content[init_end:]
    
    # Add batching methods
    batching_methods = """
    def add_to_batch(self, model, request_info):
        """Add a request to the batch queue for the specified model"""
        if not hasattr(self, "batching_enabled") or not self.batching_enabled or model not in self.supported_batch_models:
            # Either batching is disabled or model doesn't support it
            return False
            
        with self.batch_lock:
            # Initialize batch queue for this model if needed
            if model not in self.batch_queue:
                self.batch_queue[model] = []
                
            # Add request to batch
            self.batch_queue[model].append(request_info)
            
            # Check if we need to start a timer for this batch
            if len(self.batch_queue[model]) == 1:
                # First item in batch, start timer
                if model in self.batch_timers and self.batch_timers[model] is not None:
                    self.batch_timers[model].cancel()
                
                self.batch_timers[model] = threading.Timer(
                    self.batch_timeout, 
                    self._process_batch,
                    args=[model]
                )
                self.batch_timers[model].daemon = True
                self.batch_timers[model].start()
                
            # Check if batch is full and should be processed immediately
            if len(self.batch_queue[model]) >= self.max_batch_size:
                # Cancel timer since we're processing now
                if model in self.batch_timers and self.batch_timers[model] is not None:
                    self.batch_timers[model].cancel()
                    self.batch_timers[model] = None
                    
                # Process batch immediately
                threading.Thread(target=self._process_batch, args=[model]).start()
                return True
                
            return True
    
    def _process_batch(self, model):
        """Process a batch of requests for the specified model"""
        with self.batch_lock:
            # Get all requests for this model
            if model not in self.batch_queue:
                return
                
            batch_requests = self.batch_queue[model]
            self.batch_queue[model] = []
            
            # Clear timer reference
            if model in self.batch_timers:
                self.batch_timers[model] = None
        
        if not batch_requests:
            return
            
        # Update batch statistics
        if hasattr(self, "collect_metrics") and self.collect_metrics and hasattr(self, "update_stats"):
            self.update_stats({"batched_requests": len(batch_requests)})
        
        try:
            # Check which type of batch processing to use
            if model in self.embedding_models:
                self._process_embedding_batch(model, batch_requests)
            elif model in self.completion_models:
                self._process_completion_batch(model, batch_requests)
            else:
                logger.warning(f"Unknown batch processing type for model {model}")
                # Fail all requests in the batch
                for req in batch_requests:
                    future = req.get("future")
                    if future:
                        future["error"] = Exception(f"No batch processing available for model {model}")
                        future["completed"] = True
                
        except Exception as e:
            logger.error(f"Error processing batch for model {model}: {e}")
            
            # Set error for all futures in the batch
            for req in batch_requests:
                future = req.get("future")
                if future:
                    future["error"] = e
                    future["completed"] = True
    
    def _process_embedding_batch(self, model, batch_requests):
        """Process a batch of embedding requests - override in subclasses"""
        try:
            # Extract texts from requests
            texts = []
            for req in batch_requests:
                data = req.get("data", {})
                text = data.get("text", data.get("input", ""))
                texts.append(text)
            
            # This is a placeholder - subclasses should implement this
            # with the actual batched embedding API call
            batch_result = {"embeddings": [[0.1, 0.2] * 50] * len(texts)}
            
            # Distribute results to individual futures
            for i, req in enumerate(batch_requests):
                future = req.get("future")
                if future and i < len(batch_result.get("embeddings", [])):
                    future["result"] = {
                        "embedding": batch_result["embeddings"][i],
                        "model": model,
                        "implementation_type": "MOCK-BATCHED"
                    }
                    future["completed"] = True
                elif future:
                    future["error"] = Exception("Batch embedding result index out of range")
                    future["completed"] = True
                    
        except Exception as e:
            # Propagate error to all futures
            for req in batch_requests:
                future = req.get("future")
                if future:
                    future["error"] = e
                    future["completed"] = True
    
    def _process_completion_batch(self, model, batch_requests):
        """Process a batch of completion requests - override in subclasses"""
        try:
            # Extract prompts from requests
            prompts = []
            for req in batch_requests:
                data = req.get("data", {})
                prompt = data.get("prompt", data.get("input", ""))
                prompts.append(prompt)
            
            # This is a placeholder - subclasses should implement this
            # with the actual batched completion API call
            batch_result = {"completions": [f"Mock response for prompt {i}" for i in range(len(prompts))]}
            
            # Distribute results to individual futures
            for i, req in enumerate(batch_requests):
                future = req.get("future")
                if future and i < len(batch_result.get("completions", [])):
                    future["result"] = {
                        "text": batch_result["completions"][i],
                        "model": model,
                        "implementation_type": "MOCK-BATCHED"
                    }
                    future["completed"] = True
                elif future:
                    future["error"] = Exception("Batch completion result index out of range")
                    future["completed"] = True
                    
        except Exception as e:
            # Propagate error to all futures
            for req in batch_requests:
                future = req.get("future")
                if future:
                    future["error"] = e
                    future["completed"] = True
    """
    
    # Add methods at end of class
    class_end_pattern = r"\nclass .*?:\n.*?(?=\nclass |\n# |\Z)"
    class_match = re.search(class_end_pattern, new_content, re.DOTALL)
    
    if class_match:
        class_end = class_match.end()
        new_content = new_content[:class_end] + batching_methods + new_content[class_end:]
    
    # Modify process_queue to check for batching
    process_queue_pattern = r"def _process_queue\(self\):.*?with self.queue_lock:.*?request_info = self.request_queue.pop\(0\)|request_info = self.queue.get\(.*?\)"
    process_match = re.search(process_queue_pattern, new_content, re.DOTALL)
    
    if process_match:
        process_queue = process_match.group(0)
        
        # Add batching check after getting request from queue
        queue_with_batching = process_queue.replace(
            "request_info =",
            "# Check for priority queue\n                if isinstance(item, tuple) and len(item) == 2:\n                    priority, request_info = item\n                else:\n                    request_info = item\n                \n                # Check if this request can be batched\n                model = request_info.get('model')\n                if hasattr(self, 'batching_enabled') and self.batching_enabled and model and model in getattr(self, 'supported_batch_models', []):\n                    # Try to add to batch\n                    if self.add_to_batch(model, request_info):\n                        # Successfully added to batch, move to next request\n                        self.request_queue.task_done()\n                        continue\n                \n                # Process normally if not batched\n                "
        )
        
        new_content = new_content.replace(process_queue, queue_with_batching)
    
    return new_content

def process_file(file_path, api_type):
    """Process a single API file to add advanced features"""
    logger.info(f"Processing {file_path} as {api_type} API...")
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Make a backup of the original file
        backup_path = f"{file_path}.bak"
        with open(backup_path, 'w') as f:
            f.write(content)
        logger.info(f"Created backup at {backup_path}")
        
        # Apply enhancements in order
        content = add_priority_queue(content)
        content = add_circuit_breaker(content)
        content = add_monitoring(content)
        content = add_batching(content)
        
        # Write updated content back to file
        with open(file_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Successfully updated {file_path} with advanced features")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Enhance API backends with advanced features: priority queue, circuit breaker, " +
                   "monitoring, and request batching"
    )
    parser.add_argument("--api", "-a", help="Specific API to update", 
                       choices=["groq", "claude", "gemini", "openai", "ollama", 
                                "hf_tgi", "hf_tei", "llvm", "opea", "ovms", "s3_kit", "all"])
    parser.add_argument("--dry-run", "-d", action="store_true", 
                       help="Only print what would be done without making changes")
    parser.add_argument("--features", "-f", nargs="+",
                       choices=["priority", "circuit-breaker", "monitoring", "batching", "all"],
                       default=["all"],
                       help="Specific features to add (default: all)")
    
    args = parser.parse_args()
    
    # Get path to API backends directory
    script_dir = Path(__file__).parent.parent
    api_backends_dir = script_dir / "ipfs_accelerate_py" / "api_backends"
    
    if not api_backends_dir.exists():
        logger.error(f"Error: API backends directory not found at {api_backends_dir}")
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
            logger.error(f"Error: Unknown API '{args.api}'")
            return
        apis_to_process = [(api_filename, args.api)]
    else:
        # Default to processing all
        apis_to_process = list(api_files.items())
    
    results = []
    for filename, api_type in apis_to_process:
        file_path = api_backends_dir / filename
        if not file_path.exists():
            logger.warning(f"Warning: File {file_path} not found, skipping")
            continue
            
        if args.dry_run:
            logger.info(f"Would process {file_path} as {api_type} API")
        else:
            success = process_file(file_path, api_type)
            results.append((filename, api_type, success))
    
    # Print summary
    if not args.dry_run:
        logger.info("\n=== Enhancement Summary ===")
        for filename, api_type, success in results:
            logger.info(f"{filename}: {'✓ Success' if success else '✗ Failed'}")
        
        success_count = sum(1 for _, _, success in results if success)
        logger.info(f"\nSuccessfully enhanced {success_count} of {len(results)} API backends")
        
        # Create an example test script
        create_test_script()

def create_test_script():
    """Create a test script for the enhanced features"""
    test_script_path = Path(__file__).parent / "test_enhanced_api_features.py"
    
    test_script_content = """#!/usr/bin/env python
\"\"\"
Test script for enhanced API features including:
- Priority queue
- Circuit breaker pattern
- Enhanced monitoring
- Request batching
\"\"\"

import os
import sys
import time
import json
import threading
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import backend APIs
try:
    from ipfs_accelerate_py.api_backends import (
        openai_api, groq, claude, gemini, ollama,
        hf_tgi, hf_tei, llvm, opea, ovms, s3_kit
    )
except ImportError:
    print("Could not import API backends. Make sure the path is correct.")
    sys.exit(1)

# API client classes by name
API_CLIENTS = {
    "groq": groq,
    "claude": claude,
    "gemini": gemini,
    "openai": openai_api,
    "ollama": ollama,
    "hf_tgi": hf_tgi,
    "hf_tei": hf_tei,
    "llvm": llvm,
    "opea": opea,
    "ovms": ovms,
    "s3_kit": s3_kit
}

def test_priority_queue(api_client):
    \"\"\"Test priority queue functionality\"\"\"
    print("\\n=== Testing Priority Queue ===")
    
    if not hasattr(api_client, "PRIORITY_HIGH"):
        print("Priority queue not implemented for this API")
        return False
    
    # Test high, normal and low priority requests
    results = []
    
    # Queue a low priority request first
    print("Queuing LOW priority request...")
    low_req = {
        "data": {"prompt": "This is a low priority request"},
        "queue_entry_time": time.time(),
        "model": "test-model"
    }
    low_future = api_client.queue_with_priority(low_req, api_client.PRIORITY_LOW)
    
    # Queue a normal priority request second
    print("Queuing NORMAL priority request...")
    normal_req = {
        "data": {"prompt": "This is a normal priority request"},
        "queue_entry_time": time.time(),
        "model": "test-model"
    }
    normal_future = api_client.queue_with_priority(normal_req, api_client.PRIORITY_NORMAL)
    
    # Queue a high priority request last
    print("Queuing HIGH priority request...")
    high_req = {
        "data": {"prompt": "This is a high priority request"},
        "queue_entry_time": time.time(),
        "model": "test-model"
    }
    high_future = api_client.queue_with_priority(high_req, api_client.PRIORITY_HIGH)
    
    # Check queue ordering
    with api_client.queue_lock:
        queue_ordering = [item[0] for item in api_client.request_queue]
    
    print(f"Queue priority ordering: {queue_ordering}")
    
    # Check if high priority is first
    if queue_ordering and queue_ordering[0] == api_client.PRIORITY_HIGH:
        print("✓ Priority queue working correctly")
        return True
    else:
        print("✗ Priority queue not working as expected")
        return False
        
def test_circuit_breaker(api_client):
    \"\"\"Test circuit breaker functionality\"\"\"
    print("\\n=== Testing Circuit Breaker ===")
    
    if not hasattr(api_client, "circuit_state"):
        print("Circuit breaker not implemented for this API")
        return False
    
    # Get initial state
    initial_state = api_client.circuit_state
    print(f"Initial circuit state: {initial_state}")
    
    # Test tracking failures
    for i in range(api_client.failure_threshold + 1):
        print(f"Simulating failure {i+1}/{api_client.failure_threshold+1}")
        api_client.track_request_result(False, "TestError")
        
        if i < api_client.failure_threshold:
            print(f"  Failure count: {api_client.failure_count}")
        else:
            print(f"  Circuit state: {api_client.circuit_state}")
    
    # Check if circuit opened
    if api_client.circuit_state == "OPEN":
        print("✓ Circuit breaker opened correctly after failures")
        
        # Test circuit check
        if not api_client.check_circuit_breaker():
            print("✓ Circuit breaker correctly preventing requests")
        else:
            print("✗ Circuit breaker should prevent requests when open")
        
        # Reset circuit state for further tests
        api_client.circuit_state = "CLOSED"
        api_client.failure_count = 0
        
        return True
    else:
        print("✗ Circuit breaker did not open as expected")
        return False

def test_monitoring(api_client):
    \"\"\"Test monitoring and reporting functionality\"\"\"
    print("\\n=== Testing Monitoring and Reporting ===")
    
    if not hasattr(api_client, "request_stats"):
        print("Monitoring not implemented for this API")
        return False
    
    # Reset stats
    if hasattr(api_client, "reset_stats"):
        api_client.reset_stats()
    
    # Simulate successful requests
    for i in range(5):
        print(f"Simulating successful request {i+1}/5")
        api_client.update_stats({
            "total_requests": 1,
            "successful_requests": 1,
            "total_response_time": 0.5,
            "response_time": 0.5,
            "requests_by_model": {"test-model": 1}
        })
    
    # Simulate failed requests
    for i in range(2):
        print(f"Simulating failed request {i+1}/2")
        api_client.update_stats({
            "total_requests": 1,
            "failed_requests": 1,
            "errors_by_type": {"TestError": 1}
        })
    
    # Get stats
    stats = api_client.get_stats()
    print("\\nStatistics:")
    print(f"Total requests: {stats.get('total_requests', 0)}")
    print(f"Successful requests: {stats.get('successful_requests', 0)}")
    print(f"Failed requests: {stats.get('failed_requests', 0)}")
    print(f"Average response time: {stats.get('average_response_time', 0):.3f}s")
    
    # Generate report
    if hasattr(api_client, "generate_report"):
        report = api_client.generate_report(include_details=True)
        print("\\nGenerated Report:")
        print(f"Success rate: {report['summary']['success_rate']:.1f}%")
        print(f"Error types: {report.get('errors', {})}")
    
    # Check if stats are working
    if stats.get('total_requests', 0) == 7 and stats.get('successful_requests', 0) == 5:
        print("✓ Monitoring system working correctly")
        return True
    else:
        print("✗ Monitoring system not working as expected")
        return False

def run_all_tests(api_name, api_key=None):
    \"\"\"Run all enhanced feature tests for an API\"\"\"
    print(f"\\n=== Testing Enhanced Features for {api_name.upper()} API ===")
    print(f"Time: {datetime.now().isoformat()}")
    
    # Initialize API client
    if api_name not in API_CLIENTS:
        print(f"Error: Unknown API client '{api_name}'")
        return {}
    
    # Create metadata with API key
    metadata = {}
    if api_key:
        if api_name == "groq":
            metadata["groq_api_key"] = api_key
        elif api_name == "claude":
            metadata["anthropic_api_key"] = api_key
        elif api_name == "gemini": 
            metadata["google_api_key"] = api_key
        elif api_name == "openai":
            metadata["openai_api_key"] = api_key
    
    # Create client
    try:
        api_client = API_CLIENTS[api_name](resources={}, metadata=metadata)
    except Exception as e:
        print(f"Error creating API client: {e}")
        return {
            "api": api_name,
            "status": "Error",
            "error": str(e),
            "results": {}
        }
    
    results = {
        "api": api_name,
        "timestamp": datetime.now().isoformat(),
        "results": {}
    }
    
    # Test priority queue
    try:
        results["results"]["priority_queue"] = {
            "status": "Success" if test_priority_queue(api_client) else "Failed"
        }
    except Exception as e:
        results["results"]["priority_queue"] = {
            "status": "Error",
            "error": str(e)
        }
    
    # Test circuit breaker
    try:
        results["results"]["circuit_breaker"] = {
            "status": "Success" if test_circuit_breaker(api_client) else "Failed"
        }
    except Exception as e:
        results["results"]["circuit_breaker"] = {
            "status": "Error",
            "error": str(e)
        }
    
    # Test monitoring
    try:
        results["results"]["monitoring"] = {
            "status": "Success" if test_monitoring(api_client) else "Failed"
        }
    except Exception as e:
        results["results"]["monitoring"] = {
            "status": "Error",
            "error": str(e)
        }
    
    # Calculate success rate
    successful = sum(1 for feature, data in results["results"].items() 
                     if data.get("status") == "Success")
    total = len(results["results"])
    
    results["success_rate"] = successful / total if total > 0 else 0
    results["status"] = "Success" if successful == total else "Partial" if successful > 0 else "Failed"
    
    # Print summary
    print("\\n=== Test Results Summary ===")
    print(f"Status: {results['status']}")
    print(f"Success Rate: {results['success_rate']*100:.1f}%")
    
    for feature, data in results["results"].items():
        print(f"{feature}: {data.get('status')}")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"{api_name}_enhanced_features_{timestamp}.json"
    
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\\nResults saved to: {result_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Test enhanced API features")
    parser.add_argument("--api", "-a", help="API to test", choices=list(API_CLIENTS.keys()), required=True)
    parser.add_argument("--key", "-k", help="API key (or will use from environment)")
    
    args = parser.parse_args()
    
    # Run tests for the specified API
    run_all_tests(args.api, args.key)

if __name__ == "__main__":
    main()
"""
    
    with open(test_script_path, 'w') as f:
        f.write(test_script_content)
    
    logger.info(f"\nCreated test script at {test_script_path}")
    logger.info(f"Run with: python {test_script_path} --api [openai|groq|claude|etc]")

if __name__ == "__main__":
    main()