/**
 * Converted from Python: test_hf_xclip.py
 * Conversion date: 2025-03-11 04:09:33
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";

#!/usr/bin/env python3
"""
Class-based test file for all X-CLIP-family models.
This file provides a unified testing interface for:
- XCLIPModel
"""

import * as $1
import * as $1
import * as $1
import * as $1
import * as $1
import * as $1
import * as $1
import * as $1
from unittest.mock import * as $1, MagicMock, Mock
import ${$1} from "$1"
import ${$1} from "$1"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party imports
import * as $1 as np

# Try to import * as $1
try ${$1} catch($2: $1) {
  torch = MagicMock()
  HAS_TORCH = false
  logger.warning("torch !available, using mock")

}
# Try to import * as $1
try ${$1} catch($2: $1) {
  transformers = MagicMock()
  HAS_TRANSFORMERS = false
  logger.warning("transformers !available, using mock")

}

# Try to import * as $1
try {
  import ${$1} from "$1"
  import * as $1
  import ${$1} from "$1"
  HAS_PIL = true
} catch($2: $1) {
  Image = MagicMock()
  requests = MagicMock()
  BytesIO = MagicMock()
  HAS_PIL = false
  logger.warning("PIL || requests !available, using mock")

}

}
if ($1) {
  
}
class $1 extends $2 {
$1($2) {
    this.model_path = model_path
    this.platform = platform
    console.log($1)

}
  $1($2) {
    """Initialize for CPU platform."""
    
  }
    this.platform = "CPU"
    this.device = "cpu"
    this.device_name = "cpu"
    return true
  
}
    """Mock handler for platforms that don't have real implementations."""
    
    
  
  $1($2) {
    """Initialize for CUDA platform."""
    import * as $1
    this.platform = "CUDA"
    this.device = "cuda"
    this.device_name = "cuda" if torch.cuda.is_available() else "cpu"
    return true
  
  }
  
  $1($2) {
    """Initialize for MPS platform."""
    import * as $1
    this.platform = "MPS"
    this.device = "mps"
    this.device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    return true
  
  }
  
  $1($2) {
    """Initialize for OPENVINO platform."""
    import * as $1
    this.platform = "OPENVINO"
    this.device = "openvino"
    this.device_name = "openvino"
    return true
  
  }
  
  $1($2) {
    """Initialize for ROCM platform."""
    import * as $1
    this.platform = "ROCM"
    this.device = "rocm"
    this.device_name = "cuda" if torch.cuda.is_available() && torch.version.hip is !null else "cpu"
    return true
  
  }

  $1($2) {
    """Create handler for CPU platform."""
    model_path = this.get_model_path_or_name()
      handler = AutoModel.from_pretrained(model_path).to(this.device_name)
    return handler
  
  }
  
  
  $1($2) {
    """Create handler for CUDA platform."""
    model_path = this.get_model_path_or_name()
      handler = AutoModel.from_pretrained(model_path).to(this.device_name)
    return handler
  
  }
  
  $1($2) {
    """Create handler for MPS platform."""
    model_path = this.get_model_path_or_name()
      handler = AutoModel.from_pretrained(model_path).to(this.device_name)
    return handler
  
  }
  
  $1($2) {
    """Create handler for OPENVINO platform."""
    model_path = this.get_model_path_or_name()
      from openvino.runtime import * as $1
      import * as $1 as np
      ie = Core()
      compiled_model = ie.compile_model(model_path, "CPU")
      handler = lambda input_data: compiled_model(np.array(input_data))[0]
    return handler
  
  }
  
  $1($2) {
    """Create handler for ROCM platform."""
    model_path = this.get_model_path_or_name()
      handler = AutoModel.from_pretrained(model_path).to(this.device_name)
    return handler
  
  }

  $1($2) {
    """Test the model using OpenVINO integration."""
    results = ${$1}
    
  }
    # Check for OpenVINO support
    if ($1) {
      results["openvino_error_type"] = "missing_dependency"
      results["openvino_missing_core"] = ["openvino"]
      results["openvino_success"] = false
      return results
    
    }
    # Check for transformers
    if ($1) {
      results["openvino_error_type"] = "missing_dependency"
      results["openvino_missing_core"] = ["transformers"]
      results["openvino_success"] = false
      return results
    
    }
    try {
      from optimum.intel import * as $1
      logger.info(`$1`)
      
    }
      # Time tokenizer loading
      tokenizer_load_start = time.time()
      tokenizer = transformers.AutoTokenizer.from_pretrained(this.model_id)
      tokenizer_load_time = time.time() - tokenizer_load_start
      
      # Time model loading
      model_load_start = time.time()
      model = OVModelForVision.from_pretrained(
        this.model_id,
        export=true,
        provider="CPU"
      )
      model_load_time = time.time() - model_load_start
      
      # Prepare input
      test_input = this.test_image_url
      
      # Process image
      if ($1) ${$1} else {
        # Mock inputs
        inputs = ${$1}
      
      }
      # Run inference
      start_time = time.time()
      outputs = model(**inputs)
      inference_time = time.time() - start_time
      
      # Process classification output
      if ($1) {
        logits = outputs.logits_per_image[0]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
      }
        predictions = []
        for i, (label, prob) in enumerate(zip(this.candidate_labels, probs)):
          predictions.append(${$1})
      } else {
        predictions = [${$1}]
      
      }
      # Store results
      results["openvino_success"] = true
      results["openvino_load_time"] = model_load_time
      results["openvino_inference_time"] = inference_time
      results["openvino_tokenizer_load_time"] = tokenizer_load_time
      
      # Add predictions if available
      if ($1) {
        results["openvino_predictions"] = predictions
      
      }
      results["openvino_error_type"] = "none"
      
      # Add to examples
      example_data = ${$1}
      
      if ($1) {
        example_data["predictions"] = predictions
      
      }
      this.$1.push($2)
      
      # Store in performance stats
      this.performance_stats["openvino"] = ${$1}
      
    } catch($2: $1) {
      # Store error information
      results["openvino_success"] = false
      results["openvino_error"] = str(e)
      results["openvino_traceback"] = traceback.format_exc()
      logger.error(`$1`)
      
    }
      # Classify error
      error_str = str(e).lower()
      if ($1) ${$1} else {
        results["openvino_error_type"] = "other"
    
      }
    # Add to overall results
    this.results["openvino"] = results
    return results
  
    
    $1($2) {
      """
      Run all tests for this model.
      
    }
      Args:
        all_hardware: If true, tests on all available hardware (CPU, CUDA, OpenVINO)
      
      Returns:
        Dict containing test results
      """
      # Always test on default device
      this.test_pipeline()
      this.test_from_pretrained()
      
      # Test on all available hardware if requested
      if ($1) {
        # Always test on CPU
        if ($1) {
          this.test_pipeline(device="cpu")
          this.test_from_pretrained(device="cpu")
        
        }
        # Test on CUDA if available
        if ($1) {
          this.test_pipeline(device="cuda")
          this.test_from_pretrained(device="cuda")
        
        }
        # Test on OpenVINO if available
        if ($1) {
          this.test_with_openvino()
      
        }
      # Build final results
      }
      return {
        "results": this.results,
        "examples": this.examples,
        "performance": this.performance_stats,
        "hardware": HW_CAPABILITIES,
        "metadata": ${$1}
      }
      }
  
  


  
  $1($2) {
    """Return mock output."""
    console.log($1)
    return ${$1}
class $1 extends $2 {
    @staticmethod
    $1($2) {
      class $1 extends $2 {
        $1($2) {
          this.size = (224, 224)
        $1($2) {
          return self
        $1($2) {
          return self
      return MockImg()
        }
      
        }
  class $1 extends $2 {
    @staticmethod
    $1($2) {
      class $1 extends $2 {
        $1($2) {
          this.content = b"mock image data"
        $1($2) {
          pass
      return MockResponse()
        }

        }
  Image.open = MockImage.open
      }
  requests.get = MockRequests.get
    }

  }

        }
# Hardware detection
      }
$1($2) {
  """Check available hardware && return capabilities."""
  capabilities = ${$1}
  
}
  # Check CUDA
    }
  if ($1) {
    capabilities["cuda"] = torch.cuda.is_available()
    if ($1) {
      capabilities["cuda_devices"] = torch.cuda.device_count()
      capabilities["cuda_version"] = torch.version.cuda
  
    }
  # Check MPS (Apple Silicon)
  }
  if ($1) {
    capabilities["mps"] = torch.mps.is_available()
  
  }
  # Check OpenVINO
  try ${$1} catch($2: $1) {
    pass
  
  }
  return capabilities

}
# Get hardware capabilities
  }
HW_CAPABILITIES = check_hardware()

# Models registry - Maps model IDs to their specific configurations
X-CLIP_MODELS_REGISTRY = {
  "microsoft/xclip-base-patch32": ${$1},
}
}

class $1 extends $2 {
  """Base test class for all X-CLIP-family models."""
  
}
  $1($2) {
    """Initialize the test class for a specific model || default."""
    this.model_id = model_id || "microsoft/xclip-base-patch32"
    
  }
    # Verify model exists in registry
    if ($1) ${$1} else {
      this.model_info = X-CLIP_MODELS_REGISTRY[this.model_id]
    
    }
    # Define model parameters
    this.task = "zero-shot-image-classification"
    this.class_name = this.model_info["class"]
    this.description = this.model_info["description"]
    
    # Define test inputs
    this.test_text = "['a photo of a cat', 'a photo of a dog']"
    this.test_texts = [
      "['a photo of a cat', 'a photo of a dog']",
      "['a photo of a cat', 'a photo of a dog'] (alternative)"
    ]
    this.test_image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    
    # Configure hardware preference
    if ($1) {
      this.preferred_device = "cuda"
    elif ($1) ${$1} else {
      this.preferred_device = "cpu"
    
    }
    logger.info(`$1`)
    }
    
    # Results storage
    this.results = {}
    this.examples = []
    this.performance_stats = {}
  
  

  $1($2) {
    """Initialize vision model for WebNN inference."""
    try {
      console.log($1)
      model_name = model_name || this.model_name
      
    }
      # Check for WebNN support
      webnn_support = false
      try {
        # In browser environments, check for WebNN API
        import * as $1
        if ($1) ${$1} catch($2: $1) {
        # Not in a browser environment
        }
        pass
        
      }
      # Create queue for inference requests
      import * as $1
      queue = asyncio.Queue(16)
      
  }
      if ($1) {
        # Create a WebNN simulation using CPU implementation for vision models
        console.log($1)
        
      }
        # Initialize with CPU for simulation
        endpoint, processor, _, _, batch_size = this.init_cpu(model_name=model_name)
        
        # Wrap the CPU function to simulate WebNN
  $1($2) {
          try {
            # Process image input (path || PIL Image)
            if ($1) {
              import ${$1} from "$1"
              image = Image.open(image_input).convert("RGB")
            elif ($1) {
              if ($1) {
                import ${$1} from "$1"
                image = $3.map(($2) => $1)
              } else ${$1} else {
              image = image_input
              }
              
              }
            # Process with processor
            }
            inputs = processor(images=image, return_tensors="pt")
            }
            
          }
            # Run inference
            with torch.no_grad():
              outputs = endpoint(**inputs)
            
  }
            # Add WebNN-specific metadata
            return ${$1}
          } catch($2: $1) {
            console.log($1)
            return ${$1}
        
          }
        return endpoint, processor, webnn_handler, queue, batch_size
      } else {
        # Use actual WebNN implementation when available
        # (This would use the WebNN API in browser environments)
        console.log($1)
        
      }
        # Since WebNN API access depends on browser environment,
        # implementation details would involve JS interop
        
        # Create mock implementation for now (replace with real implementation)
        return null, null, lambda x: ${$1}, queue, 1
        
    } catch($2: $1) {
      console.log($1)
      # Fallback to a minimal mock
      import * as $1
      queue = asyncio.Queue(16)
      return null, null, lambda x: ${$1}, queue, 1

    }
  $1($2) {
    """Initialize multimodal model for WebGPU inference with advanced optimizations."""
    try {
      console.log($1)
      model_name = model_name || this.model_name
      
    }
      # Check for WebGPU support
      webgpu_support = false
      try {
        # In browser environments, check for WebGPU API
        import * as $1
        if ($1) ${$1} catch($2: $1) {
        # Not in a browser environment
        }
        pass
        
      }
      # Create queue for inference requests
      import * as $1
      queue = asyncio.Queue(16)
      
  }
      if ($1) {
        # Create a WebGPU simulation using CPU implementation for multimodal models
        console.log($1)
        
      }
        # Initialize with CPU for simulation
        endpoint, processor, _, _, batch_size = this.init_cpu(model_name=model_name)
        
        # Multimodal-specific optimizations
        use_parallel_loading = true
        use_4bit_quantization = true
        use_kv_cache_optimization = true
        console.log($1)
        
        # Wrap the CPU function to simulate WebGPU/transformers.js for multimodal
  $1($2) {
          try {
            # Process multimodal input (image + text)
            if ($1) {
              # Handle dictionary input with multiple modalities
              image = input_data.get("image")
              text = input_data.get("text")
              
            }
              # Load image if path is provided
              if ($1) {
                import ${$1} from "$1"
                image = Image.open(image).convert("RGB")
            elif ($1) {
              # Handle image path as direct input
              import ${$1} from "$1"
              image = Image.open(input_data).convert("RGB")
              text = kwargs.get("text", "")
            } else {
              # Default handling for text input
              image = null
              text = input_data
              
            }
            # Process with processor
            }
            if ($1) {
              # Apply parallel loading optimization if enabled
              if ($1) ${$1} else {
              inputs = processor(input_data, return_tensors="pt")
              }
            
            }
            # Apply 4-bit quantization if enabled
              }
            if ($1) {
              console.log($1)
              # In real implementation, weights would be quantized here
            
            }
            # Apply KV cache optimization if enabled
            if ($1) {
              console.log($1)
              # In real implementation, KV cache would be used here
            
            }
            # Run inference with optimizations
            with torch.no_grad():
              outputs = endpoint(**inputs)
            
          }
            # Add WebGPU-specific metadata including optimization flags
            return {
              "output": outputs,
              "implementation_type": "SIMULATION_WEBGPU_TRANSFORMERS_JS",
              "model": model_name,
              "backend": "webgpu-simulation",
              "device": "webgpu",
              "optimizations": ${$1},
              "transformers_js": ${$1}
            }
          } catch($2: $1) {
            console.log($1)
            return ${$1}
        
          }
        return endpoint, processor, webgpu_handler, queue, batch_size
      } else {
        # Use actual WebGPU implementation when available
        console.log($1)
        
      }
        # Since WebGPU API access depends on browser environment,
            }
        # implementation details would involve JS interop
        
  }
        # Create mock implementation for now (replace with real implementation)
        return null, null, lambda x: ${$1}, queue, 1
        
    } catch($2: $1) {
      console.log($1)
      # Fallback to a minimal mock
      import * as $1
      queue = asyncio.Queue(16)
      return null, null, lambda x: ${$1}, queue, 1
$1($2) {
  """Test the model using transformers pipeline API."""
  if ($1) {
    device = this.preferred_device
  
  }
  results = ${$1}
  
}
  # Check for dependencies
    }
  if ($1) {
    results["pipeline_error_type"] = "missing_dependency"
    results["pipeline_missing_core"] = ["transformers"]
    results["pipeline_success"] = false
    return results
    
  }
  if ($1) {
    results["pipeline_error_type"] = "missing_dependency"
    results["pipeline_missing_deps"] = ["pillow>=8.0.0", "requests>=2.25.0"]
    results["pipeline_success"] = false
    return results
  
  }
  try {
    logger.info(`$1`)
    
  }
    # Create pipeline with appropriate parameters
    pipeline_kwargs = ${$1}
    
    # Time the model loading
    load_start_time = time.time()
    pipeline = transformers.pipeline(**pipeline_kwargs)
    load_time = time.time() - load_start_time
    
    # Prepare test input
    pipeline_input = this.test_text
    
    # Run warmup inference if on CUDA
    if ($1) {
      try ${$1} catch($2: $1) {
        pass
    
      }
    # Run multiple inference passes
    }
    num_runs = 3
    times = []
    outputs = []
    
    for (let $1 = 0; $1 < $2; $1++) {
      start_time = time.time()
      output = pipeline(pipeline_input)
      end_time = time.time()
      $1.push($2)
      $1.push($2)
    
    }
    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    # Store results
    results["pipeline_success"] = true
    results["pipeline_avg_time"] = avg_time
    results["pipeline_min_time"] = min_time
    results["pipeline_max_time"] = max_time
    results["pipeline_load_time"] = load_time
    results["pipeline_error_type"] = "none"
    
    # Add to examples
    this.examples.append(${$1})
    
    # Store in performance stats
    this.performance_stats[`$1`] = ${$1}
    
  } catch($2: $1) {
    # Store error information
    results["pipeline_success"] = false
    results["pipeline_error"] = str(e)
    results["pipeline_traceback"] = traceback.format_exc()
    logger.error(`$1`)
    
  }
    # Classify error type
    error_str = str(e).lower()
    traceback_str = traceback.format_exc().lower()
    
    if ($1) {
      results["pipeline_error_type"] = "cuda_error"
    elif ($1) {
      results["pipeline_error_type"] = "out_of_memory"
    elif ($1) ${$1} else {
      results["pipeline_error_type"] = "other"
  
    }
  # Add to overall results
    }
  this.results[`$1`] = results
    }
  return results

  
  
$1($2) {
  """Test the model using direct from_pretrained loading."""
  if ($1) {
    device = this.preferred_device
  
  }
  results = ${$1}
  
}
  # Check for dependencies
  if ($1) {
    results["from_pretrained_error_type"] = "missing_dependency"
    results["from_pretrained_missing_core"] = ["transformers"]
    results["from_pretrained_success"] = false
    return results
    
  }
  if ($1) {
    results["from_pretrained_error_type"] = "missing_dependency"
    results["from_pretrained_missing_deps"] = ["pillow>=8.0.0", "requests>=2.25.0"]
    results["from_pretrained_success"] = false
    return results
  
  }
  try {
    logger.info(`$1`)
    
  }
    # Common parameters for loading
    pretrained_kwargs = ${$1}
    
    # Time tokenizer loading
    tokenizer_load_start = time.time()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
      this.model_id,
      **pretrained_kwargs
    )
    tokenizer_load_time = time.time() - tokenizer_load_start
    
    # Use appropriate model class based on model type
    model_class = null
    if ($1) ${$1} else {
      # Fallback to Auto class
      model_class = transformers.AutoModel
    
    }
    # Time model loading
    model_load_start = time.time()
    model = model_class.from_pretrained(
      this.model_id,
      **pretrained_kwargs
    )
    model_load_time = time.time() - model_load_start
    
    # Move model to device
    if ($1) {
      model = model.to(device)
    
    }
    # Prepare test input
    test_input = this.test_image_url
    
    # Get image
    if ($1) ${$1} else {
      # Mock image
      image = null
      
    }
    # Get text features
    inputs = tokenizer(this.candidate_labels, padding=true, return_tensors="pt")
    
    if ($1) {
      # Get image features
      processor = transformers.AutoProcessor.from_pretrained(this.model_id)
      image_inputs = processor(images=image, return_tensors="pt")
      inputs.update(image_inputs)
    
    }
    # Move inputs to device
    if ($1) {
      inputs = ${$1}
    
    }
    # Run warmup inference if using CUDA
    if ($1) {
      try ${$1} catch($2: $1) {
        pass
    
      }
    # Run multiple inference passes
    }
    num_runs = 3
    times = []
    outputs = []
    
    for (let $1 = 0; $1 < $2; $1++) {
      start_time = time.time()
      with torch.no_grad():
        output = model(**inputs)
      end_time = time.time()
      $1.push($2)
      $1.push($2)
    
    }
    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    # Process classification output
    if ($1) {
      logits = outputs.logits_per_image[0]
      probs = torch.nn.functional.softmax(logits, dim=-1)
      predictions = []
      for i, (label, prob) in enumerate(zip(this.candidate_labels, probs)):
        predictions.append(${$1})
    } else {
      predictions = [${$1}]
    
    }
    # Calculate model size
    }
    param_count = sum(p.numel() for p in model.parameters())
    model_size_mb = (param_count * 4) / (1024 * 1024)  # Rough size in MB
    
    # Store results
    results["from_pretrained_success"] = true
    results["from_pretrained_avg_time"] = avg_time
    results["from_pretrained_min_time"] = min_time
    results["from_pretrained_max_time"] = max_time
    results["tokenizer_load_time"] = tokenizer_load_time
    results["model_load_time"] = model_load_time
    results["model_size_mb"] = model_size_mb
    results["from_pretrained_error_type"] = "none"
    
    # Add predictions if available
    if ($1) {
      results["predictions"] = predictions
    
    }
    # Add to examples
    example_data = ${$1}
    
    if ($1) {
      example_data["predictions"] = predictions
    
    }
    this.$1.push($2)
    
    # Store in performance stats
    this.performance_stats[`$1`] = ${$1}
    
  } catch($2: $1) {
    # Store error information
    results["from_pretrained_success"] = false
    results["from_pretrained_error"] = str(e)
    results["from_pretrained_traceback"] = traceback.format_exc()
    logger.error(`$1`)
    
  }
    # Classify error type
    error_str = str(e).lower()
    traceback_str = traceback.format_exc().lower()
    
    if ($1) {
      results["from_pretrained_error_type"] = "cuda_error"
    elif ($1) {
      results["from_pretrained_error_type"] = "out_of_memory"
    elif ($1) ${$1} else {
      results["from_pretrained_error_type"] = "other"
  
    }
  # Add to overall results
    }
  this.results[`$1`] = results
    }
  return results

  
  
$1($2) ${$1}.json"
  output_path = os.path.join(output_dir, filename)
  
  # Save results
  with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
  
  logger.info(`$1`)
  return output_path

$1($2) {
  """Get a list of all available X-CLIP models in the registry."""
  return list(X-Object.keys($1))

}
$1($2) {
  """Test all registered X-CLIP models."""
  models = get_available_models()
  results = {}
  
}
  for (const $1 of $2) {
    logger.info(`$1`)
    tester = TestXCLIPModels(model_id)
    model_results = tester.run_tests(all_hardware=all_hardware)
    
  }
    # Save individual results
    save_results(model_id, model_results, output_dir=output_dir)
    
    # Add to summary
    results[model_id] = ${$1}
  
  # Save summary
  summary_path = os.path.join(output_dir, `$1`%Y%m%d_%H%M%S')}.json")
  with open(summary_path, "w") as f:
    json.dump(results, f, indent=2)
  
  logger.info(`$1`)
  return results

$1($2) {
  """Command-line entry point."""
  parser = argparse.ArgumentParser(description="Test X-CLIP-family models")
  
}
  # Model selection
  model_group = parser.add_mutually_exclusive_group()
  model_group.add_argument("--model", type=str, help="Specific model to test")
  model_group.add_argument("--all-models", action="store_true", help="Test all registered models")
  
  # Hardware options
  parser.add_argument("--all-hardware", action="store_true", help="Test on all available hardware")
  parser.add_argument("--cpu-only", action="store_true", help="Test only on CPU")
  
  # Output options
  parser.add_argument("--output-dir", type=str, default="collected_results", help="Directory for output files")
  parser.add_argument("--save", action="store_true", help="Save results to file")
  
  # List options
  parser.add_argument("--list-models", action="store_true", help="List all available models")
  
  args = parser.parse_args()
  
  # List models if requested
  if ($1) {
    models = get_available_models()
    console.log($1)
    for (const $1 of $2) ${$1}): ${$1}")
    return
  
  }
  # Create output directory if needed
  if ($1) {
    os.makedirs(args.output_dir, exist_ok=true)
  
  }
  # Test all models if requested
  if ($1) {
    results = test_all_models(output_dir=args.output_dir, all_hardware=args.all_hardware)
    
  }
    # Print summary
    console.log($1)
    total = len(results)
    successful = sum(1 for r in Object.values($1) if r["success"])
    console.log($1)
    return
  
  # Test single model (default || specified)
  model_id = args.model || "microsoft/xclip-base-patch32"
  logger.info(`$1`)
  
  # Override preferred device if CPU only
  if ($1) {
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  
  }
  # Run test
  tester = TestXCLIPModels(model_id)
  results = tester.run_tests(all_hardware=args.all_hardware)
  
  # Save results if requested
  if ($1) {
    save_results(model_id, results, output_dir=args.output_dir)
  
  }
  # Print summary
  success = any(r.get("pipeline_success", false) for r in results["results"].values()
        if r.get("pipeline_success") is !false)
  
  console.log($1)
  if ($1) {
    console.log($1)
    
  }
    # Print performance highlights
    for device, stats in results["performance"].items():
      if ($1) ${$1}s average inference time")
    
    # Print example outputs if available
    if ($1) {
      console.log($1)
      example = results["examples"][0]
      if ($1) ${$1}")
        console.log($1)
      elif ($1) ${$1}")
        console.log($1)
  } else {
    console.log($1)
    
  }
    # Print error information
    }
    for test_name, result in results["results"].items():
      if ($1) ${$1}")
        console.log($1)}")
  
  console.log($1)

if ($1) {
  main()
