/**
 * Converted from Python: test_hardware_selection.py
 * Conversion date: 2025-03-11 04:08:34
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";

#!/usr/bin/env python
"""
Test the hardware selection system.

This module tests the hardware selector class to ensure it correctly recommends
hardware for various models && scenarios, including fallback functionality when
prediction models aren't available.
"""

import * as $1
import * as $1
import * as $1
import * as $1
import * as $1
import * as $1
import ${$1} from "$1"
from unittest.mock import * as $1, MagicMock

# Add parent directory to path
sys.$1.push($2)))))))os.path.dirname()))))))os.path.dirname()))))))os.path.abspath()))))))__file__))))

# Import hardware selector
import ${$1} from "$1"

# Configure logging
logging.basicConfig()))))))level=logging.INFO)


class TestHardwareSelector()))))))unittest.TestCase):
  """Test cases for the hardware selector class."""

  $1($2) {
    """Set up test fixtures."""
    # Create a temporary directory for benchmark data
    this.temp_dir = tempfile.TemporaryDirectory())))))))
    this.benchmark_path = os.path.join()))))))this.temp_dir.name, "benchmark_results")
    os.makedirs()))))))this.benchmark_path, exist_ok=true)
    
  }
    # Create empty benchmark files
    os.makedirs()))))))os.path.join()))))))this.benchmark_path, "raw_results"), exist_ok=true)
    os.makedirs()))))))os.path.join()))))))this.benchmark_path, "processed_results"), exist_ok=true)
    
    # Create compatibility matrix
    this.compatibility_matrix = {}}}}}}
    "timestamp": "2025-03-01T00:00:00Z",
    "hardware_types": ["cpu", "cuda", "rocm", "mps", "openvino", "webnn", "webgpu"],
    "model_families": {}}}}}}
    "embedding": {}}}}}}
    "hardware_compatibility": {}}}}}}
    "cpu": {}}}}}}"compatible": true, "performance_rating": "medium"},
    "cuda": {}}}}}}"compatible": true, "performance_rating": "high"},
    "rocm": {}}}}}}"compatible": true, "performance_rating": "high"},
    "mps": {}}}}}}"compatible": true, "performance_rating": "high"},
    "openvino": {}}}}}}"compatible": true, "performance_rating": "medium"},
    "webnn": {}}}}}}"compatible": true, "performance_rating": "high"},
    "webgpu": {}}}}}}"compatible": true, "performance_rating": "medium"}
    }
    },
    "text_generation": {}}}}}}
    "hardware_compatibility": {}}}}}}
    "cpu": {}}}}}}"compatible": true, "performance_rating": "low"},
    "cuda": {}}}}}}"compatible": true, "performance_rating": "high"},
    "rocm": {}}}}}}"compatible": true, "performance_rating": "medium"},
    "mps": {}}}}}}"compatible": true, "performance_rating": "medium"},
    "openvino": {}}}}}}"compatible": true, "performance_rating": "low"},
    "webnn": {}}}}}}"compatible": false, "performance_rating": "unknown"},
    "webgpu": {}}}}}}"compatible": true, "performance_rating": "low"}
    }
    }
    }
    }
    
    with open()))))))os.path.join()))))))this.benchmark_path, "hardware_compatibility_matrix.json"), "w") as f:
      json.dump()))))))this.compatibility_matrix, f)
  
  $1($2) {
    """Tear down test fixtures."""
    this.temp_dir.cleanup())))))))
  
  }
  $1($2) {
    """Test basic initialization of the hardware selector."""
    selector = HardwareSelector()))))))database_path=this.benchmark_path)
    this.assertIsNotnull()))))))selector)
    this.assertEqual()))))))selector.database_path, Path()))))))this.benchmark_path))
    this.assertIn()))))))"embedding", selector.compatibility_matrix["model_families"]),
    ,
  $1($2) {
    """Test basic hardware selection without prediction models."""
    selector = HardwareSelector()))))))database_path=this.benchmark_path)
    
  }
    # Test with embedding model
    result = selector.select_hardware()))))))
    model_family="embedding",
    model_name="bert-base-uncased",
    batch_size=1,
    mode="inference",
    available_hardware=["cpu", "cuda", "openvino"],,,,,
    )
    
  }
    this.assertIn()))))))"primary_recommendation", result)
    this.assertIn()))))))"fallback_options", result)
    this.assertIn()))))))"compatible_hardware", result)
    this.assertGreater()))))))len()))))))result["compatible_hardware"]), 0),
    this.assertEqual()))))))len()))))))result["fallback_options"]), 2)
    ,
    # For embedding models, CUDA should be recommended for inference
    this.assertEqual()))))))result["primary_recommendation"], "cuda")
    ,,,    ,
    # Test with text generation model
    result = selector.select_hardware()))))))
    model_family="text_generation",
    model_name="gpt2",
    batch_size=1,
    mode="inference",
    available_hardware=["cpu", "cuda", "openvino"],,,,,
    )
    
    # For text generation models, CUDA should also be recommended
    this.assertEqual()))))))result["primary_recommendation"], "cuda")
    ,,,
  $1($2) {
    """Test hardware selection when scikit-learn is unavailable."""
    # Mock sklearn import * as $1 simulate unavailability
    with patch.dict()))))))"sys.modules", {}}}}}}"sklearn": null}):
      selector = HardwareSelector()))))))database_path=this.benchmark_path)
      
  }
      # Test with embedding model
      result = selector.select_hardware()))))))
      model_family="embedding",
      model_name="bert-base-uncased",
      batch_size=1,
      mode="inference",
      available_hardware=["cpu", "cuda", "openvino"],,,,,
      )
      
      # Even without sklearn, we should still get recommendations
      this.assertIn()))))))"primary_recommendation", result)
      this.assertEqual()))))))result["primary_recommendation"], "cuda")
      ,,,
  $1($2) {
    """Test hardware selection with fallback prediction models."""
    # Create a selector with fallback models instead of trained models
    selector = HardwareSelector()))))))database_path=this.benchmark_path)
    
  }
    # Directly initialize fallback models
    selector._initialize_fallback_models()))))))"inference")
    selector._initialize_fallback_models()))))))"training")
    
    # Test with embedding model
    result = selector.select_hardware()))))))
    model_family="embedding",
    model_name="bert-base-uncased",
    batch_size=1,
    mode="inference",
    available_hardware=["cpu", "cuda", "openvino"],,,,,
    )
    
    # We should still get recommendations
    this.assertIn()))))))"primary_recommendation", result)
    this.assertEqual()))))))result["primary_recommendation"], "cuda")
    ,,,    ,
    # Test with different batch sizes
    result_large_batch = selector.select_hardware()))))))
    model_family="embedding",
    model_name="bert-base-uncased",
    batch_size=64,
    mode="inference",
    available_hardware=["cpu", "cuda", "openvino"],,,,,
    )
    
    # Larger batch sizes should still recommend cuda
    this.assertEqual()))))))result_large_batch["primary_recommendation"], "cuda")
    ,,,
  $1($2) {
    """Test generation of distributed training configuration."""
    selector = HardwareSelector()))))))database_path=this.benchmark_path)
    
  }
    # Test with small model
    config = selector.get_distributed_training_config()))))))
    model_family="text_generation",
    model_name="gpt2",
    gpu_count=4,
    batch_size=8
    )
    
    # Check that config has the expected fields
    this.assertEqual()))))))config["model_family"], "text_generation"),
    this.assertEqual()))))))config["model_name"], "gpt2"),
    this.assertEqual()))))))config["gpu_count"], 4),
    this.assertEqual()))))))config["per_gpu_batch_size"], 8),
    this.assertEqual()))))))config["global_batch_size"], 32),
    this.assertIn()))))))"distributed_strategy", config)
    this.assertIn()))))))"estimated_memory", config)
    
    # Test with large model && memory constraints
    config = selector.get_distributed_training_config()))))))
    model_family="text_generation",
    model_name="llama-7b",
    gpu_count=4,
    batch_size=8,
    max_memory_gb=16
    )
    
    # Should include memory optimizations
    this.assertIn()))))))"memory_optimizations", config)
    this.asserttrue()))))))len()))))))config["memory_optimizations"]) > 0)
    ,
  $1($2) {
    """Test creation of hardware selection map."""
    selector = HardwareSelector()))))))database_path=this.benchmark_path)
    
  }
    # Create selection map
    selection_map = selector.create_hardware_selection_map()))))))["embedding"])
    ,
    # Check that map has the expected structure
    this.assertIn()))))))"model_families", selection_map)
    this.assertIn()))))))"embedding", selection_map["model_families"]),
    ,    this.assertIn()))))))"model_sizes", selection_map["model_families"]["embedding"]),,
    ,this.assertIn()))))))"inference", selection_map["model_families"]["embedding"]),,
    ,this.assertIn()))))))"training", selection_map["model_families"]["embedding"]),,


if ($1) {
  unittest.main())))))))