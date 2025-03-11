/**
 * Converted from Python: device_mapper.py
 * Conversion date: 2025-03-11 04:08:55
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";


export interface Props {
  model_memory_requirements: return;
  available_devices: if;
  device_memory: device_mem;
  device_memory: capacity;
}

"""
Device Mapper module for multi-GPU support && custom device mapping.

This module provides functions for:
  1. Detecting available GPU hardware
  2. Mapping model parts to specific devices
  3. Implementing various mapping strategies (auto, balanced, sequential)
  4. Estimating memory requirements for model layers
  5. Optimizing device mapping based on model architecture && available hardware
  """

  import * as $1
  import * as $1
  import * as $1
  import * as $1
  import ${$1} from "$1"
  import * as $1

# Setup logger
  logger = logging.getLogger(__name__)

# Global lock for thread safety
  device_lock = threading.RLock()

class $1 extends $2 {
  """
  Class for mapping model parts to specific devices with various strategies.
  Supports multi-GPU configurations with custom mapping rules.
  """
  
}
  def __init__(self, 
  $1: $2 | null = null,
  $1: boolean = true,
  $1: boolean = false,
        $1: boolean = true):
          """
          Initialize the DeviceMapper with hardware detection && configuration.
    
    Args:
      config_path: Path to a JSON configuration file for device mapping rules
      prefer_cuda: Whether to prefer CUDA devices over others
      prefer_rocm: Whether to prefer AMD ROCm devices over others
      enable_mps: Whether to enable Apple Silicon MPS devices
      """
      this.device_info = {}}}}}}}}
      this.available_devices = [],,
      this.device_memory = {}}}}}}}}
      this.device_capabilities = {}}}}}}}}
      this.model_memory_requirements = {}}}}}}}}
      this.config_path = config_path
      this.prefer_cuda = prefer_cuda
      this.prefer_rocm = prefer_rocm
      this.enable_mps = enable_mps
    
    # Detect hardware on initialization
      this.detect_hardware()
    
    # Load custom configuration if ($1) {
    if ($1) {
      this.load_config(config_path)
    
    }
      def detect_hardware(self) -> Dict[str, Any]:,
      """
      Detect available hardware devices (CPU, CUDA, ROCm, MPS).
    
    }
    Returns:
      Dictionary with detected hardware information
      """
    with device_lock:
      this.device_info = {}}}}}}}
      "cpu": {}}}}}}}"available": true, "name": "CPU", "count": 1},
      "cuda": {}}}}}}}"available": false, "count": 0, "devices": [],,},
      "rocm": {}}}}}}}"available": false, "count": 0, "devices": [],,},
      "mps": {}}}}}}}"available": false, "count": 0},
      "preferred": "cpu"
      }
      
      # Try to import * as $1
      try {
        import * as $1
        
      }
        # Check for CUDA
        if ($1) {
          cuda_count = torch.cuda.device_count()
          this.device_info["cuda"]["available"] = true,
          this.device_info["cuda"]["count"] = cuda_count
          ,
          # Get detailed info for each CUDA device
          for (let $1 = 0; $1 < $2; $1++) {
            device_name = torch.cuda.get_device_name(i)
            device_mem = torch.cuda.get_device_properties(i).total_memory
            # Convert to GB
            device_mem_gb = device_mem / (1024**3)
            
          }
            this.device_info["cuda"]["devices"].append({}}}}}}},
            "id": i,
            "name": device_name,
            "memory": device_mem_gb,
            "capability": `$1`,
            })
            
        }
            # Update device memory map
            this.device_memory[`$1`] = device_mem_gb
            ,
          # Mark CUDA as preferred if ($1) { && preferred
          if ($1) {
            this.device_info["preferred"] = "cuda"
            ,
        # Check for ROCm (AMD GPUs)
          }
        if ($1) {
          # ROCm uses the CUDA API in PyTorch, so we need to check this way
          rocm_count = torch.cuda.device_count()
          this.device_info["rocm"]["available"] = true,
          this.device_info["rocm"]["count"] = rocm_count
          ,
          # Get detailed info for each ROCm device
          for (let $1 = 0; $1 < $2; $1++) {
            device_name = torch.cuda.get_device_name(i)
            device_mem = torch.cuda.get_device_properties(i).total_memory
            # Convert to GB
            device_mem_gb = device_mem / (1024**3)
            
          }
            this.device_info["rocm"]["devices"].append({}}}}}}},
            "id": i,
            "name": device_name,
            "memory": device_mem_gb
            })
            
        }
            # Update device memory map
            this.device_memory[`$1`] = device_mem_gb
            ,
          # Mark ROCm as preferred if ($1) { && preferred
          if ($1) {
            this.device_info["preferred"] = "rocm"
            ,
        # Check for MPS (Apple Silicon)
          }
        if ($1) {
          this.device_info["mps"]["available"] = true,
          this.device_info["mps"]["count"] = 1  # MPS is always a single device
          ,
          # For Apple Silicon, we don't have a direct way to get memory
          # Use a conservative estimate based on system memory
          try ${$1} catch($2: $1) {
            # Default to 4GB if ($1) ${$1} catch($2: $1) {
        logger.warning("PyTorch !available. Hardware detection limited to CPU only.")
            }
      
          }
      # Build list of available devices
        }
        this.available_devices = ["cpu"]
        ,
        if ($1) {,
        for i in range(this.device_info["cuda"]["count"]):,
        this.$1.push($2)
      
        if ($1) {,
        for i in range(this.device_info["rocm"]["count"]):,
        this.$1.push($2)
      
        if ($1) {,
        this.$1.push($2)
        
            return this.device_info
  
  $1($2): $3 {
    """
    Load device mapping configuration from a JSON file.
    
  }
    Args:
      config_path: Path to the JSON configuration file
      
    Returns:
      true if loaded successfully, false otherwise
    """:
    try {
      with open(config_path, 'r') as f:
        config = json.load(f)
      
    }
      # Process configuration
      if ($1) ${$1} catch($2: $1) {
      logger.error(`$1`)
      }
        return false
  
  $1($2): $3 {
    """
    Save current device mapping configuration to a JSON file.
    
  }
    Args:
      config_path: Path to save the JSON configuration
      
    Returns:
      true if saved successfully, false otherwise
    """:
    try {
      config = {}}}}}}}
      "device_info": this.device_info,
      "model_memory_requirements": this.model_memory_requirements
      }
      
    }
      with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
      
      return true
    } catch($2: $1) {
      logger.error(`$1`)
      return false
  
    }
      def estimate_model_memory(self, $1: string, $1: $2 | null = null) -> Dict[str, float]:,
      """
      Estimate memory requirements for model parts.
    
    Args:
      model_id: Hugging Face model ID || local model path
      layers: Number of layers in the model (if known)
      :
    Returns:
      Dictionary with memory estimates for different model parts
      """
    # Check if ($1) {
    if ($1) {
      return this.model_memory_requirements[model_id]
      ,
    # Default estimates based on model type
    }
    if ($1) {
      base_size = 0.5
      per_layer = 0.1
    elif ($1) {
      base_size = 0.4
      per_layer = 0.05
    elif ($1) {
      base_size = 0.8
      per_layer = 0.15
    elif ($1) ${$1} else {
      # Default estimates
      base_size = 0.5
      per_layer = 0.1
    
    }
    # If layers !specified, estimate based on model ID
    }
    if ($1) {
      if ($1) {
        layers = 6
      elif ($1) {
        layers = 12
      elif ($1) {
        layers = 24
      elif ($1) ${$1} else {
        layers = 12  # Default
    
      }
    # Calculate memory requirements
      }
        total_mem = base_size + (layers * per_layer)
    
      }
    # Create memory requirements dictionary
      }
        memory_req = {}}}}}}}
        "total": total_mem,
        "embeddings": base_size * 0.3,
        "layers": $3.map(($2) => $1),:,
        "head": base_size * 0.2
        }
    
    }
    # Cache the result
    }
        this.model_memory_requirements[model_id] = memory_req
        ,
        return memory_req
  
    }
  $1($2): $3 {
    """
    Get the recommended device for a model based on memory requirements.
    
  }
    Args:
    }
      model_id: Hugging Face model ID || local model path
      
    Returns:
      Device string (e.g., "cuda:0", "cpu")
      """
      memory_req = this.estimate_model_memory(model_id)
      total_req = memory_req["total"]
      ,
    # Find devices with enough memory
      suitable_devices = [],,
    
    for device in this.available_devices:
      if ($1) {
        # CPU always works but is last choice
        $1.push($2))
      continue
      }
        
      # Check memory requirement
      if ($1) {,
        # Higher memory gets higher priority
      priority = this.device_memory[device],,
      $1.push($2))
    
    # Sort by priority (descending)
      suitable_devices.sort(key=lambda x: x[1], reverse=true),
      ,
    if ($1) ${$1} else {
      return "cpu"  # Fallback to CPU
  
    }
      def create_device_map(self,
      $1: string,
      $1: string = "auto",
      target_devices: Optional[List[str]] = null) -> Dict[str, str]:,
      """
      Create a device map for distributing model across devices.
    
    Args:
      model_id: Hugging Face model ID || local model path
      strategy: Mapping strategy ("auto", "balanced", "sequential")
      target_devices: List of devices to use (if null, use all available)
      :::
    Returns:
      Dictionary mapping model parts to devices
      """
      memory_req = this.estimate_model_memory(model_id)
    
    # Filter target devices
    if ($1) ${$1} catch($2: $1) {: if ($1) {
      if ($1) {
        devices = [d for d in this.available_devices if ($1) ${$1} else ${$1} else {
      devices = [d for d in target_devices if ($1) {,
        }
      if ($1) {
        logger.warning("null of the specified target devices are available. Falling back to all devices.")
        devices = this.available_devices
    
      }
    # Create device map based on strategy
      }
    if ($1) {
        return this._create_sequential_map(model_id, memory_req, devices)
    elif ($1) ${$1} else {  # Auto strategy
    }
        return this._create_auto_map(model_id, memory_req, devices)
  
    }
        def _create_sequential_map(self,
        $1: string,
        $1: Record<$2, $3>,
        $1: $2[]) -> Dict[str, str]:,,,
        """
        Create a sequential device map that fills one device before moving to the next.
    
    Args:
      model_id: Hugging Face model ID || local model path
      memory_req: Memory requirements dictionary
      devices: List of target devices
      
    Returns:
      Dictionary mapping model parts to devices
      """
      device_map = {}}}}}}}}
    
    # Start with embeddings on first device
      current_device_idx = 0
      current_device = devices[current_device_idx]
      ,
    # Map embeddings
      device_map["embeddings"] = current_device
      ,
    # Distribute layers
      for i, layer_mem in enumerate(memory_req["layers"]):,,
      # Check if ($1) {
      if ($1) {
        device_mem = this.device_memory[current_device],
        total_used = sum(memory_req["layers"][j] for j in range(i) if device_map.get(`$1`, current_device) == current_device)
        ,
        # If adding this layer would exceed memory, try next device:
        if ($1) {
          current_device_idx += 1
          current_device = devices[current_device_idx]
          ,
          device_map[`$1`], = current_device
          ,
    # Map the head to last device used
        }
          device_map["head"] = current_device
          ,
        return device_map
  
      }
        def _create_balanced_map(self,
        $1: string,
        $1: Record<$2, $3>,
        $1: $2[]) -> Dict[str, str]:,,,
        """
        Create a balanced device map that distributes layers evenly across devices.
    
      }
    Args:
      model_id: Hugging Face model ID || local model path
      memory_req: Memory requirements dictionary
      devices: List of target devices
      
    Returns:
      Dictionary mapping model parts to devices
      """
      device_map = {}}}}}}}}
    
    # Count total layers
      num_layers = len(memory_req["layers"])
      ,
    # Calculate layers per device (rounded up)
      layers_per_device = math.ceil(num_layers / len(devices))
    
    # Map embeddings to first device
      device_map["embeddings"] = devices[0],
      ,
    # Distribute layers
    for (let $1 = 0; $1 < $2; $1++) {
      device_idx = min(i // layers_per_device, len(devices) - 1)
      device_map[`$1`], = devices[device_idx]
      ,
    # Map head to last device
    }
      device_map["head"] = devices[-1]
      ,
      return device_map
  
      def _create_auto_map(self,
      $1: string,
      $1: Record<$2, $3>,
      $1: $2[]) -> Dict[str, str]:,,,
      """
      Create an auto device map based on memory constraints.
    
    Args:
      model_id: Hugging Face model ID || local model path
      memory_req: Memory requirements dictionary
      devices: List of target devices
      
    Returns:
      Dictionary mapping model parts to devices
      """
      device_map = {}}}}}}}}
    
    # If only one device, put everything there
    if ($1) {
      return {}}}}}}}"": devices[0],}
      ,
    # Get device memory capacities
    }
      device_capacities = [],,
    for (const $1 of $2) {
      if ($1) {
        capacity = float('inf')  # CPU has no hard limit
      elif ($1) ${$1} else {
        capacity = 8.0  # Default to 8GB if unknown
      
      }
        $1.push($2))
    
      }
    # Sort devices by capacity (descending):
    }
        device_capacities.sort(key=lambda x: x[1], reverse=true),
        ,sorted_devices = $3.map(($2) => $1):,
    # Track memory usage on each device
    device_usage = {}}}}}}}device: 0.0 for device in sorted_devices}:
    # Assign embeddings to first device
      device_map["embeddings"] = sorted_devices[0],,
      device_usage[sorted_devices[0],] += memory_req["embeddings"]
      ,
    # Distribute layers
      for i, layer_mem in enumerate(memory_req["layers"]):,,
      # Find device with least used memory percentage
      best_device = sorted_devices[0],
      best_ratio = device_usage[best_device] / (this.device_memory.get(best_device, float('in`$1`cpu" else float('inf')),
      :
      for (const $1 of $2) {
        device_capacity = this.device_memory.get(device, float('in`$1`cpu" else float('inf')
        usage_ratio = device_usage[device],, / device_capacity
        :
        if ($1) {
          best_device = device
          best_ratio = usage_ratio
      
        }
      # Assign layer to best device
      }
          device_map[`$1`], = best_device,
          device_usage[best_device] += layer_mem
          ,
    # Assign head to device with most layers
          layer_counts = {}}}}}}}}
          for i in range(len(memory_req["layers"])):,
          device = device_map[`$1`],
          layer_counts[device],, = layer_counts.get(device, 0) + 1
    
          head_device = max(Object.entries($1), key=lambda x: x[1]),[0],,
          device_map["head"] = head_device
          ,
          return device_map
  
          $1($2): $3 {,
          """
          Apply a device map to a PyTorch model.
    
    Args:
      model: PyTorch model object
      device_map: Dictionary mapping model parts to devices
      
    Returns:
      null (modifies model in-place)
      """
    try {
      import * as $1
      
    }
      # If we have a single device for the whole model
      if ($1) {
        device = device_map[""],
        model.to(device)
      return
      }
      
      # Handle special case for HF models with .parallelize() method
      if ($1) {
        model.deparallelize()  # Ensure model is !already parallelized
        
      }
        # If using HF's .parallelize(), convert our device map to their format
        hf_device_map = {}}}}}}}}
        
        # Map standard layer patterns to HF-specific ones
        for key, device in Object.entries($1):
          if ($1) {
            hf_device_map["word_embeddings"] = device,
            hf_device_map["position_embeddings"] = device,
            hf_device_map["token_type_embeddings"] = device,
          elif ($1) {
            layer_idx = int(key.split(".")[1]),
            hf_device_map[`$1`] = device,
          elif ($1) {
            hf_device_map["ln_f"] = device,
            hf_device_map["lm_head"] = device
            ,
        # Apply to model
          }
            model.parallelize(hf_device_map)
            return
      
          }
      # Apply manually to standard PyTorch models
          }
      for name, module in model.named_children():
        # Find the right device for this module
        target_device = null
        
        for key, device in Object.entries($1):
          if ($1) {
            target_device = device
          break
          }
        
        if ($1) ${$1} else ${$1} catch($2: $1) ${$1} catch($2: $1) {
      logger.error(`$1`)
        }
  
      def get_tensor_parallel_config(self, $1: string, target_devices: Optional[List[str]] = null) -> Dict[str, Any]:,,
      """
      Get tensor parallel configuration for models that support it (like VLLM).
    
    Args:
      model_id: Hugging Face model ID || local model path
      target_devices: List of devices to use (if null, use all available)
      :::
    Returns:
      Dictionary with tensor parallel configuration
      """
    # Filter target devices
    if ($1) ${$1} catch($2: $1) {:
      devices = [d for d in this.available_devices if ($1) ${$1} else {
      devices = $3.map(($2) => $1)
      }
      ,,
    # Get device indices
    device_indices = [],,::
    for (const $1 of $2) {
      parts = device.split(":")
      if ($1) {,,
      $1.push($2),)
    
    }
    # Default configuration
      config = {}}}}}}}
      "tensor_parallel_size": len(device_indices),
      "gpu_ids": device_indices,
      "max_parallel_loading_workers": min(8, len(device_indices) * 2)
      }
    
      return config
  
      def get_docker_gpu_args(self, target_devices: Optional[List[str]] = null) -> Tuple[str, Dict[str, Any]]:,
      """
      Get Docker GPU arguments for container deployment.
    
    Args:
      target_devices: List of devices to use (if null, use all available)
      :::
    Returns:
      Tuple of (gpu_arg_string, environment_variables)
      """
    # Filter target devices
    if ($1) ${$1} catch($2: $1) {:
      devices = [d for d in this.available_devices if ($1) ${$1} else {
      devices = $3.map(($2) => $1)
      }
      ,,
    # Get device indices
    device_indices = [],,::
    for (const $1 of $2) {
      parts = device.split(":")
      if ($1) {,,
      $1.push($2),)
    
    }
    # Sort device indices
      device_indices.sort()
    
    # Create GPU argument string
    if ($1) {
      gpu_arg = ""
    elif ($1) ${$1} else {
      gpu_arg = `$1`
    
    }
    # Create environment variables
    }
      env_vars = {}}}}}}}
      "NUM_SHARD": len(device_indices) if device_indices else 1
      }
    
    # If specific devices, add CUDA_VISIBLE_DEVICES:
    if ($1) {
      env_vars["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_indices))
      ,
      return gpu_arg, env_vars