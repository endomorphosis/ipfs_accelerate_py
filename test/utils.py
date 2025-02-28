import os
import time
import json
import fcntl
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FileLock:
    """A file-based lock to ensure thread-safe access to shared resources."""
    
    def __init__(self, lock_file: str):
        """Initialize with the path to the lock file.
        
        Args:
            lock_file: Path to the lock file
        """
        self.lock_file = lock_file
        self.lock_handle = None
        
    def __enter__(self):
        """Acquire the lock when entering a with block."""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.lock_file), exist_ok=True)
        
        # Open or create the lock file
        self.lock_handle = open(self.lock_file, 'w')
        
        # Try to acquire the lock with retry
        max_attempts = 10
        attempt = 0
        while attempt < max_attempts:
            try:
                fcntl.flock(self.lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except IOError:
                attempt += 1
                logger.info(f"Lock {self.lock_file} is held by another process. "
                           f"Waiting... (attempt {attempt}/{max_attempts})")
                time.sleep(1)
        
        # If we get here, we couldn't acquire the lock
        raise TimeoutError(f"Could not acquire lock on {self.lock_file} after {max_attempts} attempts")
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release the lock when exiting a with block."""
        if self.lock_handle:
            fcntl.flock(self.lock_handle, fcntl.LOCK_UN)
            self.lock_handle.close()
            self.lock_handle = None

def find_model_path(model_name: str) -> str:
    """Find a model's path with multiple fallback strategies.
    
    Args:
        model_name: The name of the model to find
        
    Returns:
        The path to the model if found, or the model name itself
    """
    try:
        # Try HF cache first
        cache_path = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", "models")
        if os.path.exists(cache_path):
            model_dirs = [x for x in os.listdir(cache_path) if model_name in x]
            if model_dirs:
                return os.path.join(cache_path, model_dirs[0])
                
        # Try alternate paths
        alt_paths = [
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
            os.path.join("/tmp", "huggingface")
        ]
        for path in alt_paths:
            if os.path.exists(path):
                for root, dirs, _ in os.walk(path):
                    if model_name in root:
                        return root
                        
        # Try downloading if online
        try:
            from huggingface_hub import snapshot_download
            return snapshot_download(model_name)
        except Exception as e:
            logger.warning(f"Failed to download model: {e}")
            
        # Last resort - return the model name and hope for the best
        return model_name
    except Exception as e:
        logger.error(f"Error finding model path: {e}")
        return model_name

def validate_parameters(device_label: str, task_type: Optional[str] = None) -> Tuple[str, int, Optional[str]]:
    """Validate and extract device information from device label.
    
    Args:
        device_label: Device specification (format: "device:index")
        task_type: Optional task type for model initialization
        
    Returns:
        Tuple of (device_type, device_index, task_type)
    """
    try:
        # Parse device label (format: "device:index")
        parts = device_label.split(":")
        device_type = parts[0].lower()
        device_index = int(parts[1]) if len(parts) > 1 else 0
        
        # Validate task type based on model family
        valid_tasks = [
            "text-generation", 
            "text2text-generation", 
            "text2text-generation-with-past",
            "image-classification",
            "image-text-to-text",
            "audio-classification",
            "automatic-speech-recognition"
        ]
        
        if task_type and task_type not in valid_tasks:
            logger.warning(f"Unknown task type '{task_type}', defaulting to 'text-generation'")
            task_type = "text-generation"
            
        return device_type, device_index, task_type
    except Exception as e:
        logger.error(f"Error parsing parameters: {e}, using defaults")
        return "cpu", 0, task_type or "text-generation"

def report_status(results_dict: Dict[str, Any], 
                 platform: str, 
                 operation: str, 
                 success: bool, 
                 using_mock: bool = False, 
                 error: Optional[Exception] = None) -> Dict[str, Any]:
    """Add consistent status reporting to results dictionary.
    
    Args:
        results_dict: The dictionary to add status to
        platform: Platform identifier (cpu, cuda, openvino, etc.)
        operation: Operation being performed (init, infer, etc.)
        success: Whether the operation succeeded
        using_mock: Whether a mock implementation was used
        error: Optional error message if operation failed
        
    Returns:
        Updated results dictionary with status information
    """
    implementation = "(MOCK)" if using_mock else "(REAL)"
    
    if not success:
        status = f"Error {implementation}: {str(error)}"
    else:
        status = f"Success {implementation}"
        
    results_dict[f"{platform}_{operation}"] = status
    
    # Add implementation type marker to dictionary
    if "implementation_type" not in results_dict:
        results_dict["implementation_type"] = "MOCK" if using_mock else "REAL"
    
    # Log for debugging
    logger.info(f"{platform} {operation}: {status}")
    
    return results_dict

def get_model_cache_lock_path(model_name: str) -> str:
    """Get the path to the lock file for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Path to the lock file
    """
    # Create a directory for lock files if it doesn't exist
    lock_dir = os.path.join(os.path.expanduser("~"), ".cache", "ipfs_accelerate", "locks")
    os.makedirs(lock_dir, exist_ok=True)
    
    # Create a lock file name based on the model name
    sanitized_name = model_name.replace('/', '_').replace('\\', '_')
    return os.path.join(lock_dir, f"{sanitized_name}.lock")