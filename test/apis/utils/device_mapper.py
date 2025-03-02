"""
Device Mapper module for multi-GPU support and custom device mapping.

This module provides functions for:
1. Detecting available GPU hardware
2. Mapping model parts to specific devices
3. Implementing various mapping strategies (auto, balanced, sequential)
4. Estimating memory requirements for model layers
5. Optimizing device mapping based on model architecture and available hardware
"""

import os
import json
import math
import threading
from typing import Dict, List, Union, Optional, Tuple, Any
import logging

# Setup logger
logger = logging.getLogger(__name__)

# Global lock for thread safety
device_lock = threading.RLock()

class DeviceMapper:
    """
    Class for mapping model parts to specific devices with various strategies.
    Supports multi-GPU configurations with custom mapping rules.
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 prefer_cuda: bool = True,
                 prefer_rocm: bool = False,
                 enable_mps: bool = True):
        """
        Initialize the DeviceMapper with hardware detection and configuration.
        
        Args:
            config_path: Path to a JSON configuration file for device mapping rules
            prefer_cuda: Whether to prefer CUDA devices over others
            prefer_rocm: Whether to prefer AMD ROCm devices over others
            enable_mps: Whether to enable Apple Silicon MPS devices
        """
        self.device_info = {}
        self.available_devices = []
        self.device_memory = {}
        self.device_capabilities = {}
        self.model_memory_requirements = {}
        self.config_path = config_path
        self.prefer_cuda = prefer_cuda
        self.prefer_rocm = prefer_rocm
        self.enable_mps = enable_mps
        
        # Detect hardware on initialization
        self.detect_hardware()
        
        # Load custom configuration if available
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        
    def detect_hardware(self) -> Dict[str, Any]:
        """
        Detect available hardware devices (CPU, CUDA, ROCm, MPS).
        
        Returns:
            Dictionary with detected hardware information
        """
        with device_lock:
            self.device_info = {
                "cpu": {"available": True, "name": "CPU", "count": 1},
                "cuda": {"available": False, "count": 0, "devices": []},
                "rocm": {"available": False, "count": 0, "devices": []},
                "mps": {"available": False, "count": 0},
                "preferred": "cpu"
            }
            
            # Try to import torch
            try:
                import torch
                
                # Check for CUDA
                if hasattr(torch, 'cuda') and torch.cuda.is_available():
                    cuda_count = torch.cuda.device_count()
                    self.device_info["cuda"]["available"] = True
                    self.device_info["cuda"]["count"] = cuda_count
                    
                    # Get detailed info for each CUDA device
                    for i in range(cuda_count):
                        device_name = torch.cuda.get_device_name(i)
                        device_mem = torch.cuda.get_device_properties(i).total_memory
                        # Convert to GB
                        device_mem_gb = device_mem / (1024**3)
                        
                        self.device_info["cuda"]["devices"].append({
                            "id": i,
                            "name": device_name,
                            "memory": device_mem_gb,
                            "capability": f"{torch.cuda.get_device_capability(i)[0]}.{torch.cuda.get_device_capability(i)[1]}"
                        })
                        
                        # Update device memory map
                        self.device_memory[f"cuda:{i}"] = device_mem_gb
                        
                    # Mark CUDA as preferred if available and preferred
                    if self.prefer_cuda and cuda_count > 0:
                        self.device_info["preferred"] = "cuda"
                
                # Check for ROCm (AMD GPUs)
                if hasattr(torch, '_C') and hasattr(torch._C, '_rocm_is_available') and torch._C._rocm_is_available():
                    # ROCm uses the CUDA API in PyTorch, so we need to check this way
                    rocm_count = torch.cuda.device_count()
                    self.device_info["rocm"]["available"] = True
                    self.device_info["rocm"]["count"] = rocm_count
                    
                    # Get detailed info for each ROCm device
                    for i in range(rocm_count):
                        device_name = torch.cuda.get_device_name(i)
                        device_mem = torch.cuda.get_device_properties(i).total_memory
                        # Convert to GB
                        device_mem_gb = device_mem / (1024**3)
                        
                        self.device_info["rocm"]["devices"].append({
                            "id": i,
                            "name": device_name,
                            "memory": device_mem_gb
                        })
                        
                        # Update device memory map
                        self.device_memory[f"rocm:{i}"] = device_mem_gb
                    
                    # Mark ROCm as preferred if available and preferred
                    if self.prefer_rocm and rocm_count > 0:
                        self.device_info["preferred"] = "rocm"
                
                # Check for MPS (Apple Silicon)
                if self.enable_mps and hasattr(torch, 'mps') and hasattr(torch.mps, 'is_available') and torch.mps.is_available():
                    self.device_info["mps"]["available"] = True
                    self.device_info["mps"]["count"] = 1  # MPS is always a single device
                    
                    # For Apple Silicon, we don't have a direct way to get memory
                    # Use a conservative estimate based on system memory
                    try:
                        import psutil
                        system_memory = psutil.virtual_memory().total / (1024**3)
                        # Estimate 70% of system memory is available for MPS
                        estimated_mem = system_memory * 0.7
                        self.device_memory["mps:0"] = estimated_mem
                    except ImportError:
                        # Default to 4GB if we can't determine
                        self.device_memory["mps:0"] = 4.0
            
            except ImportError:
                logger.warning("PyTorch not available. Hardware detection limited to CPU only.")
            
            # Build list of available devices
            self.available_devices = ["cpu"]
            
            if self.device_info["cuda"]["available"]:
                for i in range(self.device_info["cuda"]["count"]):
                    self.available_devices.append(f"cuda:{i}")
            
            if self.device_info["rocm"]["available"]:
                for i in range(self.device_info["rocm"]["count"]):
                    self.available_devices.append(f"rocm:{i}")
            
            if self.device_info["mps"]["available"]:
                self.available_devices.append("mps:0")
                
            return self.device_info
    
    def load_config(self, config_path: str) -> bool:
        """
        Load device mapping configuration from a JSON file.
        
        Args:
            config_path: Path to the JSON configuration file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Process configuration
            if "model_memory_requirements" in config:
                self.model_memory_requirements = config["model_memory_requirements"]
            
            # Additional configuration can be added here
            
            return True
        except Exception as e:
            logger.error(f"Failed to load device mapper configuration: {str(e)}")
            return False
    
    def save_config(self, config_path: str) -> bool:
        """
        Save current device mapping configuration to a JSON file.
        
        Args:
            config_path: Path to save the JSON configuration
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            config = {
                "device_info": self.device_info,
                "model_memory_requirements": self.model_memory_requirements
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Failed to save device mapper configuration: {str(e)}")
            return False
    
    def estimate_model_memory(self, model_id: str, layers: Optional[int] = None) -> Dict[str, float]:
        """
        Estimate memory requirements for model parts.
        
        Args:
            model_id: Hugging Face model ID or local model path
            layers: Number of layers in the model (if known)
            
        Returns:
            Dictionary with memory estimates for different model parts
        """
        # Check if we have cached estimates
        if model_id in self.model_memory_requirements:
            return self.model_memory_requirements[model_id]
        
        # Default estimates based on model type
        if "gpt2" in model_id.lower():
            base_size = 0.5
            per_layer = 0.1
        elif "bert" in model_id.lower():
            base_size = 0.4
            per_layer = 0.05
        elif "t5" in model_id.lower():
            base_size = 0.8
            per_layer = 0.15
        elif "llama" in model_id.lower() or "mistral" in model_id.lower():
            base_size = 1.2
            per_layer = 0.25
        else:
            # Default estimates
            base_size = 0.5
            per_layer = 0.1
        
        # If layers not specified, estimate based on model ID
        if layers is None:
            if "small" in model_id.lower():
                layers = 6
            elif "base" in model_id.lower():
                layers = 12
            elif "large" in model_id.lower():
                layers = 24
            elif "xl" in model_id.lower() or "xxl" in model_id.lower():
                layers = 36
            else:
                layers = 12  # Default
        
        # Calculate memory requirements
        total_mem = base_size + (layers * per_layer)
        
        # Create memory requirements dictionary
        memory_req = {
            "total": total_mem,
            "embeddings": base_size * 0.3,
            "layers": [(per_layer * 0.8) for _ in range(layers)],
            "head": base_size * 0.2
        }
        
        # Cache the result
        self.model_memory_requirements[model_id] = memory_req
        
        return memory_req
    
    def get_recommended_device(self, model_id: str) -> str:
        """
        Get the recommended device for a model based on memory requirements.
        
        Args:
            model_id: Hugging Face model ID or local model path
            
        Returns:
            Device string (e.g., "cuda:0", "cpu")
        """
        memory_req = self.estimate_model_memory(model_id)
        total_req = memory_req["total"]
        
        # Find devices with enough memory
        suitable_devices = []
        
        for device in self.available_devices:
            if device.startswith("cpu"):
                # CPU always works but is last choice
                suitable_devices.append((device, 0))
                continue
                
            # Check memory requirement
            if device in self.device_memory and self.device_memory[device] >= total_req:
                # Higher memory gets higher priority
                priority = self.device_memory[device]
                suitable_devices.append((device, priority))
        
        # Sort by priority (descending)
        suitable_devices.sort(key=lambda x: x[1], reverse=True)
        
        if suitable_devices:
            return suitable_devices[0][0]
        else:
            return "cpu"  # Fallback to CPU
    
    def create_device_map(self, 
                         model_id: str, 
                         strategy: str = "auto",
                         target_devices: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Create a device map for distributing model across devices.
        
        Args:
            model_id: Hugging Face model ID or local model path
            strategy: Mapping strategy ("auto", "balanced", "sequential")
            target_devices: List of devices to use (if None, use all available)
            
        Returns:
            Dictionary mapping model parts to devices
        """
        memory_req = self.estimate_model_memory(model_id)
        
        # Filter target devices
        if target_devices is None:
            # Use all available, except CPU if we have GPUs
            if any(device != "cpu" for device in self.available_devices):
                devices = [d for d in self.available_devices if d != "cpu"]
            else:
                devices = self.available_devices
        else:
            devices = [d for d in target_devices if d in self.available_devices]
            if not devices:
                logger.warning("None of the specified target devices are available. Falling back to all devices.")
                devices = self.available_devices
        
        # Create device map based on strategy
        if strategy == "sequential":
            return self._create_sequential_map(model_id, memory_req, devices)
        elif strategy == "balanced":
            return self._create_balanced_map(model_id, memory_req, devices)
        else:  # Auto strategy
            return self._create_auto_map(model_id, memory_req, devices)
    
    def _create_sequential_map(self, 
                              model_id: str, 
                              memory_req: Dict[str, Any], 
                              devices: List[str]) -> Dict[str, str]:
        """
        Create a sequential device map that fills one device before moving to the next.
        
        Args:
            model_id: Hugging Face model ID or local model path
            memory_req: Memory requirements dictionary
            devices: List of target devices
            
        Returns:
            Dictionary mapping model parts to devices
        """
        device_map = {}
        
        # Start with embeddings on first device
        current_device_idx = 0
        current_device = devices[current_device_idx]
        
        # Map embeddings
        device_map["embeddings"] = current_device
        
        # Distribute layers
        for i, layer_mem in enumerate(memory_req["layers"]):
            # Check if we need to move to next device
            if current_device != "cpu" and current_device in self.device_memory:
                device_mem = self.device_memory[current_device]
                total_used = sum(memory_req["layers"][j] for j in range(i) if device_map.get(f"layer.{j}", current_device) == current_device)
                
                # If adding this layer would exceed memory, try next device
                if total_used + layer_mem > device_mem * 0.9 and current_device_idx < len(devices) - 1:
                    current_device_idx += 1
                    current_device = devices[current_device_idx]
            
            device_map[f"layer.{i}"] = current_device
        
        # Map the head to last device used
        device_map["head"] = current_device
        
        return device_map
    
    def _create_balanced_map(self, 
                            model_id: str, 
                            memory_req: Dict[str, Any], 
                            devices: List[str]) -> Dict[str, str]:
        """
        Create a balanced device map that distributes layers evenly across devices.
        
        Args:
            model_id: Hugging Face model ID or local model path
            memory_req: Memory requirements dictionary
            devices: List of target devices
            
        Returns:
            Dictionary mapping model parts to devices
        """
        device_map = {}
        
        # Count total layers
        num_layers = len(memory_req["layers"])
        
        # Calculate layers per device (rounded up)
        layers_per_device = math.ceil(num_layers / len(devices))
        
        # Map embeddings to first device
        device_map["embeddings"] = devices[0]
        
        # Distribute layers
        for i in range(num_layers):
            device_idx = min(i // layers_per_device, len(devices) - 1)
            device_map[f"layer.{i}"] = devices[device_idx]
        
        # Map head to last device
        device_map["head"] = devices[-1]
        
        return device_map
    
    def _create_auto_map(self, 
                        model_id: str, 
                        memory_req: Dict[str, Any], 
                        devices: List[str]) -> Dict[str, str]:
        """
        Create an auto device map based on memory constraints.
        
        Args:
            model_id: Hugging Face model ID or local model path
            memory_req: Memory requirements dictionary
            devices: List of target devices
            
        Returns:
            Dictionary mapping model parts to devices
        """
        device_map = {}
        
        # If only one device, put everything there
        if len(devices) == 1:
            return {"": devices[0]}
        
        # Get device memory capacities
        device_capacities = []
        for device in devices:
            if device == "cpu":
                capacity = float('inf')  # CPU has no hard limit
            elif device in self.device_memory:
                capacity = self.device_memory[device]
            else:
                capacity = 8.0  # Default to 8GB if unknown
            
            device_capacities.append((device, capacity))
        
        # Sort devices by capacity (descending)
        device_capacities.sort(key=lambda x: x[1], reverse=True)
        sorted_devices = [d[0] for d in device_capacities]
        
        # Track memory usage on each device
        device_usage = {device: 0.0 for device in sorted_devices}
        
        # Assign embeddings to first device
        device_map["embeddings"] = sorted_devices[0]
        device_usage[sorted_devices[0]] += memory_req["embeddings"]
        
        # Distribute layers
        for i, layer_mem in enumerate(memory_req["layers"]):
            # Find device with least used memory percentage
            best_device = sorted_devices[0]
            best_ratio = device_usage[best_device] / (self.device_memory.get(best_device, float('inf')) if best_device != "cpu" else float('inf'))
            
            for device in sorted_devices:
                device_capacity = self.device_memory.get(device, float('inf')) if device != "cpu" else float('inf')
                usage_ratio = device_usage[device] / device_capacity
                
                if usage_ratio < best_ratio:
                    best_device = device
                    best_ratio = usage_ratio
            
            # Assign layer to best device
            device_map[f"layer.{i}"] = best_device
            device_usage[best_device] += layer_mem
        
        # Assign head to device with most layers
        layer_counts = {}
        for i in range(len(memory_req["layers"])):
            device = device_map[f"layer.{i}"]
            layer_counts[device] = layer_counts.get(device, 0) + 1
        
        head_device = max(layer_counts.items(), key=lambda x: x[1])[0]
        device_map["head"] = head_device
        
        return device_map
    
    def apply_device_map(self, model, device_map: Dict[str, str]) -> None:
        """
        Apply a device map to a PyTorch model.
        
        Args:
            model: PyTorch model object
            device_map: Dictionary mapping model parts to devices
            
        Returns:
            None (modifies model in-place)
        """
        try:
            import torch
            
            # If we have a single device for the whole model
            if "" in device_map:
                device = device_map[""]
                model.to(device)
                return
            
            # Handle special case for HF models with .parallelize() method
            if hasattr(model, 'parallelize') and hasattr(model, 'deparallelize'):
                model.deparallelize()  # Ensure model is not already parallelized
                
                # If using HF's .parallelize(), convert our device map to their format
                hf_device_map = {}
                
                # Map standard layer patterns to HF-specific ones
                for key, device in device_map.items():
                    if key == "embeddings":
                        hf_device_map["word_embeddings"] = device
                        hf_device_map["position_embeddings"] = device
                        hf_device_map["token_type_embeddings"] = device
                    elif key.startswith("layer."):
                        layer_idx = int(key.split(".")[1])
                        hf_device_map[f"h.{layer_idx}"] = device
                    elif key == "head":
                        hf_device_map["ln_f"] = device
                        hf_device_map["lm_head"] = device
                
                # Apply to model
                model.parallelize(hf_device_map)
                return
            
            # Apply manually to standard PyTorch models
            for name, module in model.named_children():
                # Find the right device for this module
                target_device = None
                
                for key, device in device_map.items():
                    if key == "" or name == key or (key.endswith(".") and name.startswith(key)):
                        target_device = device
                        break
                
                if target_device:
                    module.to(target_device)
                else:
                    # If no specific mapping, apply recursively
                    self.apply_device_map(module, device_map)
        
        except ImportError:
            logger.error("Failed to apply device map: PyTorch not available")
        except Exception as e:
            logger.error(f"Failed to apply device map: {str(e)}")
    
    def get_tensor_parallel_config(self, model_id: str, target_devices: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get tensor parallel configuration for models that support it (like VLLM).
        
        Args:
            model_id: Hugging Face model ID or local model path
            target_devices: List of devices to use (if None, use all available)
            
        Returns:
            Dictionary with tensor parallel configuration
        """
        # Filter target devices
        if target_devices is None:
            # Use all available devices except CPU
            devices = [d for d in self.available_devices if d != "cpu"]
        else:
            devices = [d for d in target_devices if d in self.available_devices and d != "cpu"]
        
        # Get device indices
        device_indices = []
        for device in devices:
            parts = device.split(":")
            if len(parts) == 2 and parts[1].isdigit():
                device_indices.append(int(parts[1]))
        
        # Default configuration
        config = {
            "tensor_parallel_size": len(device_indices),
            "gpu_ids": device_indices,
            "max_parallel_loading_workers": min(8, len(device_indices) * 2)
        }
        
        return config
    
    def get_docker_gpu_args(self, target_devices: Optional[List[str]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Get Docker GPU arguments for container deployment.
        
        Args:
            target_devices: List of devices to use (if None, use all available)
            
        Returns:
            Tuple of (gpu_arg_string, environment_variables)
        """
        # Filter target devices
        if target_devices is None:
            # Use all available devices except CPU
            devices = [d for d in self.available_devices if d != "cpu"]
        else:
            devices = [d for d in target_devices if d in self.available_devices and d != "cpu"]
        
        # Get device indices
        device_indices = []
        for device in devices:
            parts = device.split(":")
            if len(parts) == 2 and parts[1].isdigit():
                device_indices.append(int(parts[1]))
        
        # Sort device indices
        device_indices.sort()
        
        # Create GPU argument string
        if not device_indices:
            gpu_arg = ""
        elif len(device_indices) == 1:
            gpu_arg = f"--gpus device={device_indices[0]}"
        else:
            gpu_arg = f"--gpus all"
        
        # Create environment variables
        env_vars = {
            "NUM_SHARD": len(device_indices) if device_indices else 1
        }
        
        # If specific devices, add CUDA_VISIBLE_DEVICES
        if device_indices:
            env_vars["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_indices))
        
        return gpu_arg, env_vars