import os
import threading
import logging
import platform
import re
from datetime import datetime

class ResourcePool:
    """
    Centralized resource management to avoid duplicate loading of models and resources.
    
    This class provides efficient resource sharing across test execution and implementation
    validation, avoiding duplicate model loading and optimizing memory usage.
    
    Attributes:
        resources (dict): Dictionary of shared resources
        models (dict): Dictionary of loaded models
        tokenizers (dict): Dictionary of loaded tokenizers
        _lock (threading.RLock): Lock for thread safety
        _stats (dict): Usage statistics
        low_memory_mode (bool): Whether to operate in low-memory mode
    """
    
    def __init__(self):
        self.resources = {}
        self.models = {}
        self.tokenizers = {}
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "memory_usage": 0,
            "creation_timestamps": {},
            "last_accessed": {}
        }
        
        # Check for low memory mode
        self.low_memory_mode = os.environ.get("RESOURCE_POOL_LOW_MEMORY", "0").lower() in ("1", "true", "yes")
        
        # Setup logging
        self.logger = logging.getLogger("ResourcePool")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Try to detect available memory for better resource management
        self.available_memory_mb = self._detect_available_memory()
        
        # If very low memory, force low memory mode
        if self.available_memory_mb < 4096 and not self.low_memory_mode:
            self.logger.warning(f"Low memory detected ({self.available_memory_mb:.2f} MB). Enabling low memory mode.")
            self.low_memory_mode = True
        
        self.logger.info(f"ResourcePool initialized (low memory mode: {self.low_memory_mode}, available memory: {self.available_memory_mb} MB)")
    
    def _detect_available_memory(self):
        """Detect available system memory in MB for better resource management"""
        # Try using hardware_detection module first
        try:
            # Import locally to avoid circular imports
            from generators.hardware.hardware_detection import detect_hardware_with_comprehensive_checks
            hardware_info = detect_hardware_with_comprehensive_checks()
            
            if "system" in hardware_info and "available_memory" in hardware_info["system"]:
                return float(hardware_info["system"]["available_memory"])
        except (ImportError, KeyError, AttributeError, Exception) as e:
            self.logger.debug(f"Could not use hardware_detection module: {str(e)}")
        
        # Fall back to psutil if available
        try:
            import psutil
            vm = psutil.virtual_memory()
            available_mb = vm.available / (1024 * 1024)
            return available_mb
        except ImportError:
            # If psutil is not available, try platform-specific approaches
            if platform.system() == "Linux":
                try:
                    with open('/proc/meminfo', 'r') as f:
                        meminfo = f.read()
                    # Extract available memory
                    match = re.search(r'MemAvailable:\s+(\d+)', meminfo)
                    if match:
                        return int(match.group(1)) / 1024  # Convert from KB to MB
                except:
                    pass
            # Default if we can't detect
            return 8192  # Assume 8GB as default
    
    def get_resource(self, resource_type, resource_id=None, constructor=None):
        """
        Get or create a resource from the pool
        
        Args:
            resource_type (str): The type of resource (e.g., 'torch', 'transformers')
            resource_id (str, optional): Optional identifier for the resource
            constructor (callable, optional): Function to create the resource if not present
            
        Returns:
            The requested resource, or None if it couldn't be created
        """
        with self._lock:
            key = f"{resource_type}:{resource_id}" if resource_id else resource_type
            
            # Check if resource exists
            if key in self.resources:
                # Resource hit - reusing existing
                self._stats["hits"] += 1
                self._stats["last_accessed"][key] = datetime.now().isoformat()
                self.logger.debug(f"Resource hit: {key}")
                return self.resources[key]
            
            # Resource miss - need to create it
            if constructor:
                self._stats["misses"] += 1
                try:
                    self.logger.info(f"Creating resource: {key}")
                    self.resources[key] = constructor()
                    self._stats["creation_timestamps"][key] = datetime.now().isoformat()
                    self._stats["last_accessed"][key] = datetime.now().isoformat()
                    
                    # Optionally track memory usage if it's a PyTorch model
                    if hasattr(self.resources[key], "get_memory_footprint"):
                        memory_usage = self.resources[key].get_memory_footprint()
                        self._stats["memory_usage"] += memory_usage
                        self.logger.info(f"Resource {key} uses {memory_usage} bytes")
                    
                    return self.resources[key]
                except Exception as e:
                    self.logger.error(f"Error creating resource {key}: {str(e)}")
                    return None
            else:
                self.logger.warning(f"Resource not found and no constructor provided: {key}")
                return None
    
    def get_model(self, model_type, model_name, constructor=None, hardware_preferences=None):
        """
        Get or create a model from the pool with hardware awareness
        
        Args:
            model_type (str): The type of model (e.g., 'bert', 't5')
            model_name (str): The specific model name (e.g., 'bert-base-uncased')
            constructor (callable, optional): Function to create the model if not present
            hardware_preferences (dict, optional): Hardware preferences for model loading
            
        Returns:
            The requested model, or None if it couldn't be created
        """
        with self._lock:
            key = f"{model_type}:{model_name}"
            
            # Check if model exists
            if key in self.models:
                # Model hit - reusing existing
                self._stats["hits"] += 1
                self._stats["last_accessed"][key] = datetime.now().isoformat()
                self.logger.debug(f"Model hit: {key}")
                return self.models[key]
            
            # Model miss - need to create it
            if constructor:
                self._stats["misses"] += 1
                
                # Check hardware compatibility if we're creating a new model
                target_device = self._get_optimal_device(model_type, model_name, hardware_preferences)
                if target_device:
                    self.logger.info(f"Selected device for {key}: {target_device}")
                
                try:
                    self.logger.info(f"Loading model: {key}")
                    start_time = datetime.now()
                    
                    # Create the model
                    model = constructor()
                    load_time = (datetime.now() - start_time).total_seconds()
                    
                    # Store in cache
                    self.models[key] = model
                    self._stats["creation_timestamps"][key] = datetime.now().isoformat()
                    self._stats["last_accessed"][key] = datetime.now().isoformat()
                    self.logger.info(f"Model {key} loaded in {load_time:.2f} seconds")
                    
                    # Track memory usage if possible
                    try:
                        import torch
                        if hasattr(self.models[key], "get_memory_footprint"):
                            memory_usage = self.models[key].get_memory_footprint()
                        elif torch.is_tensor(self.models[key]) or hasattr(self.models[key], "parameters"):
                            # For PyTorch models
                            memory_usage = sum(p.nelement() * p.element_size() for p in self.models[key].parameters())
                        else:
                            memory_usage = 0
                            
                        self._stats["memory_usage"] += memory_usage
                        self.logger.info(f"Model {key} uses approximately {memory_usage/1024/1024:.2f} MB")
                        
                        # If in low memory mode and memory usage is high, move to CPU to free GPU memory
                        if self.low_memory_mode and hasattr(model, "to") and memory_usage > (500 * 1024 * 1024):  # Over 500MB
                            if hasattr(torch, "cuda") and torch.cuda.is_available() and next(model.parameters()).device.type == "cuda":
                                self.logger.info(f"Low memory mode active - moving {key} to CPU after initialization")
                                model.to("cpu")
                                if hasattr(torch.cuda, "empty_cache"):
                                    torch.cuda.empty_cache()
                    except (ImportError, AttributeError, Exception) as e:
                        self.logger.debug(f"Could not calculate memory usage for {key}: {str(e)}")
                    
                    return self.models[key]
                except Exception as e:
                    self.logger.error(f"Error loading model {key}: {str(e)}")
                    return None
            else:
                self.logger.warning(f"Model not found and no constructor provided: {key}")
                return None
                
    def _get_optimal_device(self, model_type, model_name, hardware_preferences=None):
        """
        Determine the optimal device for a model based on hardware detection and preferences
        
        Args:
            model_type: Type of model
            model_name: Name of model
            hardware_preferences: Optional user hardware preferences
            
        Returns:
            String with recommended device or None if not applicable
        """
        # Honor user preferences first if provided
        if hardware_preferences and "device" in hardware_preferences:
            if hardware_preferences["device"] != "auto":
                self.logger.info(f"Using user-specified device: {hardware_preferences['device']}")
                return hardware_preferences["device"]
            
        # Check if hardware_detection module is available
        import os.path
        hardware_detection_path = os.path.join(os.path.dirname(__file__), "hardware_detection.py")
        if not os.path.exists(hardware_detection_path):
            self.logger.debug("hardware_detection.py file not found - using basic device detection")
            # Fall back to basic PyTorch detection
            return self._basic_device_detection()
            
        # Use hardware_detection if available
        try:
            # Check if model_family_classifier is available 
            model_classifier_path = os.path.join(os.path.dirname(__file__), "model_family_classifier.py")
            has_model_classifier = os.path.exists(model_classifier_path)
            
            # Import hardware detection (should be available since we checked file existence)
            from generators.hardware.hardware_detection import detect_available_hardware
            
            # Get hardware info
            hardware_info = detect_available_hardware()
            best_device = hardware_info.get("torch_device", "cpu")
            
            # Get model family info if classifier is available
            model_family = None
            if has_model_classifier:
                try:
                    from model_family_classifier import classify_model
                    model_info = classify_model(model_name=model_name)
                    model_family = model_info.get("family")
                    self.logger.debug(f"Model {model_name} classified as {model_family}")
                except (ImportError, Exception) as e:
                    self.logger.debug(f"Error using model family classifier: {str(e)}")
            else:
                # Use model_type as fallback if provided
                model_family = model_type if model_type != "default" else None
                self.logger.debug(f"Using model_type '{model_type}' as family (model_family_classifier not available)")
            
            # Special case handling based on model family
            if model_family == "multimodal" and best_device == "mps":
                self.logger.warning(f"Model {model_name} is multimodal and may not work well on MPS. Using CPU instead.")
                return "cpu"
                
            # Check device against available memory for large language models
            if model_family == "text_generation" and best_device == "cuda":
                # Large language models need more memory - check against available CUDA memory
                try:
                    import torch
                    if torch.cuda.is_available():
                        # Get total GPU memory
                        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                        # Get free GPU memory
                        free_gpu_memory = (torch.cuda.get_device_properties(0).total_memory - 
                                          torch.cuda.memory_allocated() -
                                          torch.cuda.memory_reserved()) / (1024**3)  # GB
                        
                        # Certain large models need specific amounts of VRAM
                        large_model_patterns = [
                            "llama-7b", "llama-13b", "llama2-7b", "llama2-13b", 
                            "stable-diffusion", "bloom-7b1", "mistral-7b", "falcon-7b", "mixtral"
                        ]
                        
                        # Check if model name matches any large model patterns
                        is_large_model = any(pattern in model_name.lower() for pattern in large_model_patterns)
                        if is_large_model and free_gpu_memory < 7.5:  # Need at least 8GB for 7B models
                            self.logger.warning(f"Insufficient GPU memory for large model {model_name}. Available: {free_gpu_memory:.2f}GB. Using CPU instead.")
                            return "cpu"
                except (ImportError, AttributeError, Exception) as e:
                    self.logger.debug(f"Error checking GPU memory: {str(e)}")
            
            return best_device
            
        except (ImportError, Exception) as e:
            self.logger.debug(f"Could not determine optimal device using hardware_detection: {str(e)}")
            # Fall back to basic detection
            return self._basic_device_detection()
    
    def _basic_device_detection(self):
        """
        Perform basic device detection using PyTorch directly
        Used as a fallback when hardware_detection module is not available
        
        Returns:
            String with recommended device
        """
        try:
            import torch
            if torch.cuda.is_available():
                self.logger.info("Using basic CUDA detection: cuda")
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.logger.info("Using basic MPS detection: mps")
                return "mps"
            else:
                self.logger.info("No GPU detected, using CPU")
                return "cpu"
        except ImportError:
            self.logger.warning("PyTorch not available, defaulting to CPU")
            return "cpu"
        except Exception as e:
            self.logger.warning(f"Error in basic device detection: {str(e)}")
            return "cpu"
    
    def get_tokenizer(self, model_type, model_name, constructor=None):
        """
        Get or create a tokenizer from the pool
        
        Args:
            model_type (str): The type of model (e.g., 'bert', 't5')
            model_name (str): The specific model name (e.g., 'bert-base-uncased')
            constructor (callable, optional): Function to create the tokenizer if not present
            
        Returns:
            The requested tokenizer, or None if it couldn't be created
        """
        with self._lock:
            key = f"tokenizer:{model_type}:{model_name}"
            
            # Check if tokenizer exists
            if key in self.tokenizers:
                # Tokenizer hit - reusing existing
                self._stats["hits"] += 1
                self._stats["last_accessed"][key] = datetime.now().isoformat()
                self.logger.debug(f"Tokenizer hit: {key}")
                return self.tokenizers[key]
            
            # Tokenizer miss - need to create it
            if constructor:
                self._stats["misses"] += 1
                try:
                    self.logger.info(f"Loading tokenizer: {key}")
                    self.tokenizers[key] = constructor()
                    self._stats["creation_timestamps"][key] = datetime.now().isoformat()
                    self._stats["last_accessed"][key] = datetime.now().isoformat()
                    
                    return self.tokenizers[key]
                except Exception as e:
                    self.logger.error(f"Error loading tokenizer {key}: {str(e)}")
                    return None
            else:
                self.logger.warning(f"Tokenizer not found and no constructor provided: {key}")
                return None
    
    def cleanup_unused_resources(self, max_age_minutes=30):
        """
        Clean up resources that haven't been used in a while  
        Args:
            max_age_minutes (int): Maximum time in minutes since last access before cleaning up
        """
        with self._lock:
            current_time = datetime.now()
            resources_to_remove = []
            models_to_remove = []
            tokenizers_to_remove = []
            
            # In low memory mode, use more aggressive timeouts
            if self.low_memory_mode:
                max_age_minutes = min(max_age_minutes, 10)  # Max 10 minutes in low memory mode
                self.logger.info(f"Using aggressive cleanup timeout of {max_age_minutes} minutes (low memory mode)")
            
            # Check if available memory is below threshold (20% of total)
            memory_pressure = False
            try:
                import psutil
                vm = psutil.virtual_memory()
                available_percent = vm.available / vm.total * 100
                if available_percent < 20:
                    memory_pressure = True
                    self.logger.warning(f"Memory pressure detected: {available_percent:.1f}% available. Using aggressive cleanup.")
                    max_age_minutes = min(max_age_minutes, 5)  # Even more aggressive timeout
            except ImportError:
                pass
            
            # Check resources
            for key, resource in self.resources.items():
                if key in self._stats["last_accessed"]:
                    last_accessed = datetime.fromisoformat(self._stats["last_accessed"][key])
                    age_minutes = (current_time - last_accessed).total_seconds() / 60
                    
                    # In low memory mode, prioritize keeping smaller resources
                    if age_minutes > max_age_minutes:
                        resources_to_remove.append(key)
            
            # Check models
            for key, model in self.models.items():
                if key in self._stats["last_accessed"]:
                    last_accessed = datetime.fromisoformat(self._stats["last_accessed"][key])
                    age_minutes = (current_time - last_accessed).total_seconds() / 60
                    
                    # In low memory mode or under pressure, more aggressively clean up large models
                    if age_minutes > max_age_minutes:
                        models_to_remove.append(key)
                    elif (self.low_memory_mode or memory_pressure) and age_minutes > max_age_minutes/2:
                        # Try to estimate model size
                        model_size_mb = 0
                        try:
                            if hasattr(model, "get_memory_footprint"):
                                model_size_mb = model.get_memory_footprint() / (1024*1024)
                            elif hasattr(model, "parameters"):
                                # Rough estimate based on parameters
                                model_size_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024*1024)
                            
                            # Remove larger models more aggressively
                            if model_size_mb > 100:  # If larger than 100MB
                                models_to_remove.append(key)
                                self.logger.info(f"Removing large model {key} ({model_size_mb:.1f} MB) due to memory pressure")
                        except:
                            pass
            
            # Check tokenizers
            for key, tokenizer in self.tokenizers.items():
                if key in self._stats["last_accessed"]:
                    last_accessed = datetime.fromisoformat(self._stats["last_accessed"][key])
                    age_minutes = (current_time - last_accessed).total_seconds() / 60
                    
                    if age_minutes > max_age_minutes:
                        tokenizers_to_remove.append(key)
            
            # Remove resources
            for key in resources_to_remove:
                self.logger.info(f"Cleaning up unused resource: {key}")
                del self.resources[key]
                
            # Remove models - with special handling for CUDA models
            for key in models_to_remove:
                self.logger.info(f"Cleaning up unused model: {key}")
                try:
                    # Try to move model to CPU before deletion if it's a PyTorch model
                    if hasattr(self.models[key], "to") and hasattr(self.models[key], "cpu"):
                        self.models[key].to("cpu")
                except Exception:
                    pass
                
                del self.models[key]
                
            # Remove tokenizers
            for key in tokenizers_to_remove:
                self.logger.info(f"Cleaning up unused tokenizer: {key}")
                del self.tokenizers[key]
                
            # Force garbage collection
            try:
                import gc
                gc.collect()
                
                # Try to clear CUDA cache if available
                try:
                    import torch
                    if hasattr(torch, "cuda") and hasattr(torch.cuda, "empty_cache"):
                        torch.cuda.empty_cache()
                        self.logger.debug("CUDA cache cleared")
                except ImportError:
                    pass
            except Exception as e:
                self.logger.debug(f"Error during garbage collection: {str(e)}")
            
            removed_count = len(resources_to_remove) + len(models_to_remove) + len(tokenizers_to_remove)
            self.logger.info(f"Cleaned up {removed_count} unused resources")
            
            # If in low memory mode and under memory pressure, consider more aggressive cleanup
            if (self.low_memory_mode or memory_pressure) and removed_count == 0:
                self.logger.warning("No resources removed but memory pressure exists. Consider manual clearing.")
                
            return removed_count
    
    def get_stats(self):
        """
        Get resource pool usage statistics
        
        Returns:
            dict: Statistics about resource usage
        """
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_ratio = self._stats["hits"] / max(1, total_requests)
            
            # Get system memory information if possible
            system_memory = {}
            try:
                import psutil
                vm = psutil.virtual_memory()
                system_memory = {
                    "total_mb": vm.total / (1024 * 1024),
                    "available_mb": vm.available / (1024 * 1024),
                    "percent_used": vm.percent,
                    "under_pressure": vm.percent > 80  # Consider > 80% as pressure
                }
            except ImportError:
                # Try platform-specific fallbacks
                if platform.system() == "Linux":
                    try:
                        with open('/proc/meminfo', 'r') as f:
                            meminfo = f.read()
                            total_match = re.search(r'MemTotal:\s+(\d+)', meminfo)
                            avail_match = re.search(r'MemAvailable:\s+(\d+)', meminfo)
                        if total_match and avail_match:
                            total_kb = int(total_match.group(1))
                            avail_kb = int(avail_match.group(1))
                            system_memory = {
                                "total_mb": total_kb / 1024,
                                "available_mb": avail_kb / 1024,
                                "percent_used": 100 - (avail_kb / total_kb * 100),
                                "under_pressure": (avail_kb / total_kb * 100) < 20
                            }
                    except:
                        pass
            
            # Get CUDA memory information if possible
            cuda_memory = {}
            try:
                import torch
                if torch.cuda.is_available():
                    device_count = torch.cuda.device_count()
                    cuda_memory = {
                        "device_count": device_count,
                        "devices": []
                    }
                    
                    for i in range(device_count):
                        props = torch.cuda.get_device_properties(i)
                        allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)
                        reserved = torch.cuda.memory_reserved(i) / (1024 * 1024)
                        total = props.total_memory / (1024 * 1024)
                        
                        cuda_memory["devices"].append({
                            "id": i,
                            "name": props.name,
                            "total_mb": total,
                            "allocated_mb": allocated,
                            "reserved_mb": reserved,
                            "free_mb": total - allocated,
                            "percent_used": (allocated / total) * 100,
                            "under_pressure": (allocated / total) > 0.8  # Over 80% utilization
                        })
            except ImportError:
                pass
            except Exception as e:
                cuda_memory["error"] = str(e)
            
            return {
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "total_requests": total_requests,
                "hit_ratio": hit_ratio,
                "memory_usage": self._stats["memory_usage"],
                "memory_usage_mb": self._stats["memory_usage"] / (1024 * 1024),
                "cached_resources": len(self.resources),
                "cached_models": len(self.models),
                "cached_tokenizers": len(self.tokenizers),
                "timestamp": datetime.now().isoformat(),
                "low_memory_mode": self.low_memory_mode,
                "system_memory": system_memory,
                "cuda_memory": cuda_memory
            }
    
    def clear(self):
        """Clear all cached resources"""
        with self._lock:
            # First try to clean up PyTorch resources properly
            try:
                # Move models to CPU before deletion if possible
                for key, model in self.models.items():
                    if hasattr(model, "to") and hasattr(model, "cpu"):
                        try:
                            model.to("cpu")
                        except Exception as e:
                            self.logger.debug(f"Error moving model {key} to CPU: {str(e)}")
                
                # Try to clear CUDA cache if available
                try:
                    import torch
                    if hasattr(torch, "cuda") and hasattr(torch.cuda, "empty_cache"):
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
            except Exception as e:
                self.logger.debug(f"Error during torch cleanup: {str(e)}")
            
            # Clear all dictionaries
            count = len(self.resources) + len(self.models) + len(self.tokenizers)
            self.resources.clear()
            self.models.clear()
            self.tokenizers.clear()
            
            # Reset stats but keep structure
            self._stats = {
                "hits": 0, 
                "misses": 0, 
                "memory_usage": 0,
                "creation_timestamps": {},
                "last_accessed": {}
            }
            
            # Force garbage collection
            try:
                import gc
                gc.collect()
            except Exception:
                pass
            
            self.logger.info(f"ResourcePool cleared - removed {count} cached objects")
            
    def generate_error_report(self, model_name: str, hardware_type: str,
                             error_message: str, stack_trace: str = None) -> dict:
        """
        Generate a structured error report for hardware compatibility issues
        
        Args:
            model_name: Name of the model
            hardware_type: Hardware platform (cuda, rocm, etc.)
            error_message: Error message
            stack_trace: Optional stack trace
            
        Returns:
            Dictionary containing structured error report
        """
        from datetime import datetime
        import os.path
        
        # Initialize report with basic information
        report = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "hardware_type": hardware_type,
            "error_message": error_message,
            "stack_trace": stack_trace,
            "recommendations": []
        }
        
        # Try to get model family information if available
        model_classifier_path = os.path.join(os.path.dirname(__file__), "model_family_classifier.py")
        if os.path.exists(model_classifier_path):
            try:
                from model_family_classifier import classify_model
                model_info = classify_model(model_name=model_name)
                
                # Add model family information to report
                report["model_family"] = model_info.get("family")
                if model_info.get("subfamily"):
                    report["subfamily"] = model_info.get("subfamily")
                
                # Get hardware priority list from model family
                if "hardware_priorities" in model_info:
                    # Add alternatives for this hardware type
                    priorities = model_info.get("hardware_priorities", [])
                    if hardware_type in priorities:
                        idx = priorities.index(hardware_type)
                        report["alternatives"] = priorities[idx+1:] if idx+1 < len(priorities) else []
                    else:
                        report["alternatives"] = priorities
                
                self.logger.debug(f"Added model family information to error report: {report['model_family']}")
            except (ImportError, Exception) as e:
                self.logger.debug(f"Error getting model family information: {str(e)}")
                # Continue without model family information
        
        # Generate specific recommendations based on error type and hardware
        report["recommendations"] = self._generate_recommendations(model_name, hardware_type, error_message)
        
        return report
    
    def _generate_recommendations(self, model_name: str, hardware_type: str, error_message: str) -> list:
        """
        Generate recommendations based on error type and hardware platform
        
        Args:
            model_name: Name of the model
            hardware_type: Hardware platform
            error_message: Error message
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        error_lower = error_message.lower()
        
        # Handle out of memory errors
        if "out of memory" in error_lower or "oom" in error_lower:
            recommendations.append(f"The model {model_name} requires more memory than available on {hardware_type}.")
            recommendations.append("Consider using a smaller model variant if available.")
            recommendations.append("Reduce batch size or sequence length to decrease memory requirements.")
            
            if hardware_type in ["cuda", "rocm", "mps"]:
                recommendations.append("Try running on CPU with 'device=cpu'.")
                
            if hardware_type == "cuda" and "openvino" in self._get_available_hardware():
                recommendations.append("Try OpenVINO with 'device=openvino'.")
        
        # Handle unsupported operation errors
        elif "not implemented" in error_lower or "not supported" in error_lower or "unsupported" in error_lower or "operation" in error_lower:
            recommendations.append(f"The model {model_name} contains operations not supported on {hardware_type} platform.")
            recommendations.append("This is typically due to hardware-specific limitations or missing driver functionality.")
            
            alternatives = self._suggest_alternative_hardware(hardware_type, model_name)
            if alternatives:
                recommendations.append(f"Try running on {alternatives[0]} with 'device={alternatives[0]}'.")
            else:
                recommendations.append("Consider using a different model that's compatible with your hardware.")
        
        # Handle driver version mismatches
        elif "driver version" in error_lower or "cuda version" in error_lower:
            if hardware_type == "cuda":
                recommendations.append("Update your NVIDIA drivers to the latest version compatible with your CUDA toolkit.")
            elif hardware_type == "rocm":
                recommendations.append("Update your AMD drivers to the latest version compatible with your ROCm toolkit.")
            else:
                recommendations.append(f"Update your {hardware_type} drivers to the latest version.")
        
        # General recommendations
        else:
            recommendations.append("Check the model's compatibility with the hardware platform.")
            recommendations.append("Try running on a different hardware platform if available.")
            
            alternatives = self._suggest_alternative_hardware(hardware_type, model_name)
            if alternatives:
                recommendations.append(f"Recommended alternative hardware: {', '.join(alternatives)}")
        
        return recommendations
    
    def _suggest_alternative_hardware(self, current_hardware: str, model_name: str) -> list:
        """
        Suggest alternative hardware based on model type and available hardware
        
        Args:
            current_hardware: Current hardware platform
            model_name: Name of the model
            
        Returns:
            List of suggested hardware alternatives
        """
        import os.path
        
        # Default fallback priority
        default_priority = ["cuda", "mps", "rocm", "openvino", "cpu"]
        
        # Get available hardware
        available_hardware = self._get_available_hardware()
        
        # Try to classify model for better suggestions
        model_classifier_path = os.path.join(os.path.dirname(__file__), "model_family_classifier.py")
        if os.path.exists(model_classifier_path):
            try:
                from model_family_classifier import classify_model
                model_info = classify_model(model_name=model_name)
                
                if "hardware_priorities" in model_info:
                    # Use model family specific priorities
                    priorities = model_info.get("hardware_priorities")
                    self.logger.debug(f"Using model family specific hardware priorities: {priorities}")
                    
                    # Filter out current hardware and unavailable platforms
                    alternatives = [hw for hw in priorities if hw != current_hardware and hw in available_hardware]
                    
                    if alternatives:
                        return alternatives
            except (ImportError, Exception) as e:
                self.logger.debug(f"Error getting model family specific hardware suggestions: {str(e)}")
        
        # Fallback to default priorities if model classification fails
        alternatives = [hw for hw in default_priority if hw != current_hardware and hw in available_hardware]
        return alternatives
    
    def _get_available_hardware(self) -> list:
        """
        Get list of available hardware platforms
        
        Returns:
            List of available hardware platform strings
        """
        available = ["cpu"]  # CPU is always available
        
        # Try to detect other hardware
        try:
            import torch
            if torch.cuda.is_available():
                available.append("cuda")
                
            if hasattr(torch, 'mps') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                available.append("mps")
        except ImportError:
            pass
            
        # Check for OpenVINO
        try:
            import importlib.util
            if importlib.util.find_spec("openvino") is not None:
                available.append("openvino")
        except ImportError:
            pass
            
        # Check for ROCm (HIP) - this is a simplified check
        try:
            import torch
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                available.append("rocm")
        except ImportError:
            pass
            
        return available
    
    def save_error_report(self, report: dict, output_dir: str = "./hardware_reports") -> str:
        """
        Save error report to file
        
        Args:
            report: Error report dictionary
            output_dir: Directory to save report
            
        Returns:
            Path to saved report file
        """
        import os
        import json
        from datetime import datetime
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = report["model_name"].replace("/", "_")
        filename = f"{output_dir}/hardware_error_{model_name}_{report['hardware_type']}_{timestamp}.json"
        
        # Save report
        with open(filename, "w") as f:
            json.dump(report, f, indent=2)
            
        self.logger.info(f"Error report saved to {filename}")
        
        return filename

# Create a global instance for shared use
global_resource_pool = ResourcePool()

def get_global_resource_pool():
    """Get the global resource pool instance"""
    return global_resource_pool