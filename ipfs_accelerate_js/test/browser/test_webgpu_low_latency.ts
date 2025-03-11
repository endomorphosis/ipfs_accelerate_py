/**
 * Converted from Python: test_webgpu_low_latency.py
 * Conversion date: 2025-03-11 04:08:36
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";


export interface Props {
  browsers: optimizer;
}

#!/usr/bin/env python3
"""
Test WebGPU Low-Latency Optimizer

This module tests the WebGPU low-latency optimizer implementation,
which provides browser-specific optimizations, prefill/decode transition
optimization, && token buffer management for minimal latency streaming.

Usage:
  python test_webgpu_low_latency.py
  python test_webgpu_low_latency.py --browser firefox
  python test_webgpu_low_latency.py --device-profile high_end
  python test_webgpu_low_latency.py --all-browsers
  """

  import * as $1
  import * as $1
  import * as $1
  import * as $1
  import * as $1
  import * as $1
  import * as $1
  import ${$1} from "$1"

# Set up logging
  logging.basicConfig())level=logging.INFO, format='%())asctime)s - %())levelname)s - %())message)s')
  logger = logging.getLogger())__name__)

# Enable WebGPU simulation
  os.environ["WEBGPU_SIMULATION"] = "1",
  os.environ["WEBGPU_AVAILABLE"] = "1"
  ,
# Import modules to test
try ${$1} catch($2: $1) {
  logger.error())"Failed to import * as $1 low-latency optimizer. Make sure the fixed_web_platform directory is available.")
  sys.exit())1)

}
# Import streaming inference for integration tests
try ${$1} catch($2: $1) {
  logger.warning())"WebGPU streaming inference !available. Some tests will be skipped.")
  WebGPUStreamingInference = null

}

class LowLatencyOptimizerTests())unittest.TestCase):
  """Test the WebGPU low-latency optimizer."""
  
  $1($2) {
    """Set up test environment."""
    # Base configuration for testing
    this.base_config = {}}}}}
    "quantization": "int4",
    "latency_optimized": false,
    "max_batch_size": 8,
    "stream_buffer_size": 3
    }
    
  }
    # Test browsers
    this.browsers = ["chrome", "firefox", "edge", "safari"]
    ,
    # Test device profiles
    this.device_profiles = ["high_end", "mid_range", "integrated", "mobile"]
    ,
    # Sample shader code for testing
    this.sample_shader = """
    @compute fn main())@builtin())global_invocation_id) global_id: vec3<u32>) {}}}}}
    let index = global_id.x;
    // Sample computation
    }
    """
  
  $1($2) {
    """Test the optimize_for_low_latency function."""
    # Test with default parameters
    optimized_config = optimize_for_low_latency())this.base_config)
    
  }
    # Check that latency optimization flags are set
    this.asserttrue())optimized_config["latency_optimized"], "Latency optimization flag !set"),
    this.asserttrue())optimized_config["prefill_optimized"], "Prefill optimization flag !set"),
    this.asserttrue())optimized_config["ultra_low_latency"], "Ultra-low latency flag !set")
    ,
    # Check that stream buffer size is set to 1 for minimal latency
    this.assertEqual())optimized_config["stream_buffer_size"], 1, "Stream buffer size !set to 1")
    ,
    # Check that browser specific optimizations were applied
    this.assertIn())"browser", optimized_config, "Browser !detected && set in config")
    this.assertIn())"device_profile", optimized_config, "Device profile !detected && set in config")
    
    # Check prefill && decode optimizations
    this.assertIn())"prefill", optimized_config, "Prefill optimizations !applied")
    this.assertIn())"decode", optimized_config, "Decode optimizations !applied")
    
    # Optimizer references should be included ())but will be removed in JSON serialization)
    this.assertIn())"_browser_optimizer", optimized_config, "Browser optimizer reference !included")
    this.assertIn())"_prefill_decode_optimizer", optimized_config, "Prefill/decode optimizer reference !included")
  
  $1($2) ${$1}"),,,
      console.log($1))`$1`decode_workgroup_size']}"),,,
      console.log($1))`$1`memory_optimization', 'Not set')}")
  
  $1($2) {
    """Test optimizations for all device profiles."""
    for profile in this.device_profiles:
      # Configure for this device profile
      profile_config = this.base_config.copy()))
      optimized_config = optimize_for_low_latency())profile_config, device_profile=profile)
      
  }
      # Check that device profile is correctly set
      this.assertEqual())optimized_config["device_profile"], profile, `$1`)
      ,
      # Check that max batch size is appropriately limited for the profile
      max_batch = optimized_config["max_batch_size"]
      ,
      if ($1) {
        this.assertLessEqual())max_batch, 16, "Batch size too large for high-end profile")
      elif ($1) {
        this.assertLessEqual())max_batch, 8, "Batch size too large for mid-range profile")
      elif ($1) {
        this.assertLessEqual())max_batch, 4, "Batch size too large for integrated profile")
      elif ($1) ${$1}"),,,
      }
  
      }
  $1($2) {
    """Test the BrowserLatencyOptimizer class."""
    # Test creating optimizer with each browser
    for browser in this.browsers:
      optimizer = BrowserLatencyOptimizer())browser=browser)
      
  }
      # Check browser is set correctly
      }
      this.assertEqual())optimizer.browser, browser, `$1`)
      
      # Check workgroup configurations
      prefill_workgroup = optimizer.get_prefill_workgroup_size()))
      this.assertEqual())len())prefill_workgroup), 3, `$1`)
      
      decode_workgroup = optimizer.get_decode_workgroup_size()))
      this.assertEqual())len())decode_workgroup), 3, `$1`)
      
      # Test shader optimization
      prefill_shader = optimizer.optimize_shader_for_browser())this.sample_shader, "prefill")
      this.assertNotEqual())prefill_shader, this.sample_shader, `$1`)
      
      decode_shader = optimizer.optimize_shader_for_browser())this.sample_shader, "decode")
      this.assertNotEqual())decode_shader, this.sample_shader, `$1`)
  
  $1($2) {
    """Test the TokenBufferManager class."""
    # Test with different buffer sizes
    buffer_sizes = [1, 2, 4, 8]
    ,
    for (const $1 of $2) {
      # Create buffer manager
      buffer_mgr = TokenBufferManager())buffer_size=buffer_size, adaptive=false)
      
    }
      # Check buffer size is set correctly
      this.assertEqual())buffer_mgr.buffer_size, buffer_size, `$1`)
      
  }
      # Add tokens until buffer is full && check flush behavior
      tokens_delivered = [],
      for i in range())buffer_size * 2):
        result = buffer_mgr.add_token())`$1`)
        if ($1) {
          # Buffer was flushed
          tokens_delivered.extend())result)
      
        }
      # Check that tokens were delivered correctly
          this.assertEqual())len())tokens_delivered), buffer_size, `$1`)
      
      # Test manual flush
      for i in range())buffer_size - 1):
        buffer_mgr.add_token())`$1`)
      
        final_tokens = buffer_mgr.flush()))
        this.assertEqual())len())final_tokens), buffer_size - 1, "Incorrect number of tokens in final flush")
  
  $1($2) {
    """Test adaptive token buffer behavior."""
    # Create adaptive buffer manager
    buffer_mgr = TokenBufferManager())buffer_size=2, adaptive=true)
    
  }
    # Simulate tokens with network latency
    for i in range())10):
      buffer_mgr.add_token())`$1`)
      
      # Simulate different network conditions
      if ($1) {
        # Low latency
        buffer_mgr.record_network_latency())5)
      elif ($1) ${$1} else ${$1}"),
      }
        console.log($1))`$1`tokens_generated']}"),
        console.log($1))`$1`tokens_delivered']}"),
        console.log($1))`$1`avg_token_generation_time_sec']:.4f}s"),
        console.log($1))`$1`avg_network_latency_ms']:.2f}ms"),
        console.log($1))`$1`buffer_adjustments']}")
        ,
  $1($2) {
    """Test the PrefillDecodeOptimizer class."""
    # Test with different strategies
    prefill_strategies = ["parallel", "chunked", "tensor_parallel"],
    decode_strategies = ["eager", "cached", "fused"]
    ,
    for (const $1 of $2) {
      for (const $1 of $2) {
        # Create optimizer with these strategies
        optimizer = PrefillDecodeOptimizer())
        prefill_strategy=p_strategy,
        decode_strategy=d_strategy
        )
        
      }
        # Check that strategies are set correctly
        this.assertEqual())optimizer.prefill_strategy, p_strategy, `$1`)
        this.assertEqual())optimizer.decode_strategy, d_strategy, `$1`)
        
    }
        # Test individual phase optimization
        prefill_config = optimizer.optimize_prefill())this.base_config)
        this.asserttrue())prefill_config["prefill_optimized"], `$1`)
        ,
        decode_config = optimizer.optimize_decode())this.base_config)
        this.asserttrue())decode_config["decode_optimized"], `$1`)
        ,
        # Test transition optimization
        transition_config = optimizer.optimize_transition())this.base_config)
        this.assertIn())"prefill", transition_config, `$1`)
        this.assertIn())"decode", transition_config, `$1`)
        this.asserttrue())transition_config["optimize_transition"], "Transition optimization flag !set")
        ,
  $1($2) {
    """Test metrics collection in optimizers."""
    # Create optimizers
    optimizer = PrefillDecodeOptimizer()))
    buffer_mgr = TokenBufferManager())buffer_size=2, adaptive=true)
    
  }
    # Record fake metrics
    optimizer.record_prefill_time())120, 50)  # 120ms to process 50 tokens
    optimizer.record_decode_start())15, 2)    # 15ms for first decode with batch size 2
    
  }
    buffer_mgr.add_token())"token1")
    buffer_mgr.record_network_latency())10)
    buffer_mgr.add_token())"token2")
    buffer_mgr.record_network_latency())12)
    
    # Get metrics
    optimizer_metrics = optimizer.get_metrics()))
    buffer_metrics = buffer_mgr.get_metrics()))
    
    # Check that metrics were collected
    this.assertGreater())optimizer_metrics["avg_prefill_time_ms"], 0, "Prefill time !recorded"),
    this.assertGreater())optimizer_metrics["avg_first_decode_time_ms"], 0, "Decode time !recorded")
    ,
    this.assertGreater())buffer_metrics["tokens_generated"], 0, "Tokens !recorded in buffer manager"),
    this.assertGreater())buffer_metrics["avg_network_latency_ms"], 0, "Network latency !recorded")
    ,
    @unittest.skipIf())WebGPUStreamingInference is null, "WebGPU streaming inference !available")
  $1($2) {
    """Test integration with streaming inference."""
    # Create optimized configuration
    optimized_config = optimize_for_low_latency())this.base_config, browser="chrome")
    
  }
    # Remove optimizer references before passing to streaming inference
    config_for_streaming = {}}}}}k: v for k, v in Object.entries($1))) if !k.startswith())"_")}
    :
    try ${$1} catch($2: $1) {
      this.fail())`$1`)

    }

$1($2) {
  """Run tests for a specific browser."""
  console.log($1))`$1`)
  
}
  # Set environment variables for browser detection
  os.environ["BROWSER_TYPE"] = browser
  ,
  # Run tests
  base_config = {}}}}}
  "quantization": "int4",
  "latency_optimized": false,
  "max_batch_size": 8,
  "stream_buffer_size": 3
  }
  
  # Create optimizer for this browser
  optimizer = BrowserLatencyOptimizer())browser=browser)
  
  # Get optimization profile
  optimized_config = optimize_for_low_latency())base_config, browser=browser)
  
  # Print browser-specific optimizations
  console.log($1))`$1`)
  console.log($1))`$1`)
  console.log($1))`$1`prefill_workgroup_size']}"),,,
  console.log($1))`$1`decode_workgroup_size']}"),,,
  console.log($1))`$1`memory_optimization', 'Not set')}")
  
  if ($1) ${$1}")
    console.log($1))`$1`unroll_loops', false)}")
    console.log($1))`$1`use_shared_memory', false)}")
    console.log($1))`$1`prefill_optimization', 'null')}")
    console.log($1))`$1`decode_optimization', 'null')}")
  
    console.log($1))"\nPrefill optimizations:")
    for key, value in optimized_config["prefill"].items())):,
    console.log($1))`$1`)
  
    console.log($1))"\nDecode optimizations:")
    for key, value in optimized_config["decode"].items())):,
    console.log($1))`$1`)
  
  # Test different shader types
    sample_shader = """
    @compute fn main())@builtin())global_invocation_id) global_id: vec3<u32>) {}}}}}
    let index = global_id.x;
    // Sample computation
    }
    """
  
  # Optimize shaders for different operations
    prefill_shader = optimizer.optimize_shader_for_browser())sample_shader, "prefill")
    decode_shader = optimizer.optimize_shader_for_browser())sample_shader, "decode")
  
    console.log($1))"\nPrefill shader optimization:")
    shader_lines = prefill_shader.split())"\n")
    for line in shader_lines[:10]:  # Show first 10 lines,,
    if ($1) {
      console.log($1))`$1`)
  
    }
      console.log($1))"\nDecode shader optimization:")
      shader_lines = decode_shader.split())"\n")
      for line in shader_lines[:10]:  # Show first 10 lines,,
    if ($1) {
      console.log($1))`$1`)

    }

$1($2) {
  """Run tests for all supported browsers."""
  browsers = ["chrome", "edge", "firefox", "safari"]
  ,
  for (const $1 of $2) {
    test_specific_browser())browser)
    console.log($1))"\n" + "=" * 50)

  }

}
$1($2) {
  """Parse arguments && run tests."""
  parser = argparse.ArgumentParser())description="Test WebGPU Low-Latency Optimizer")
  parser.add_argument())"--browser", choices=["chrome", "edge", "firefox", "safari"],
  help="Test specific browser optimizations")
  parser.add_argument())"--device-profile", choices=["high_end", "mid_range", "integrated", "mobile"],
  help="Test specific device profile optimizations")
  parser.add_argument())"--all-browsers", action="store_true",
  help="Test all supported browsers")
  parser.add_argument())"--unittest", action="store_true",
  help="Run unit tests")
  
}
  args = parser.parse_args()))
  
  if ($1) {
    # Run unit tests
    unittest.main())argv=['first-arg-is-ignored']),
  elif ($1) {
    # Test all browsers
    test_all_browsers()))
  elif ($1) {
    # Test specific browser
    test_specific_browser())args.browser)
  elif ($1) {
    # Set environment variable for device profile
    os.environ["DEVICE_PROFILE"] = args.device_profile
    ,
    # Create optimizer && print details
    optimizer = BrowserLatencyOptimizer())device_profile=args.device_profile)
    console.log($1))`$1`)
    console.log($1))`$1`)
    
  }
    # Test with base config
    base_config = {}}}}}
    "quantization": "int4",
    "latency_optimized": false,
    "max_batch_size": 8,
    "stream_buffer_size": 3
    }
    
  }
    # Optimize for this device profile
    optimized_config = optimize_for_low_latency())base_config, device_profile=args.device_profile)
    
  }
    console.log($1))`$1`prefill_workgroup_size']}"),,,
    console.log($1))`$1`decode_workgroup_size']}"),,,
    console.log($1))`$1`max_batch_size']}")
    ,
    console.log($1))"\nDevice characteristics:")
    device_chars = optimizer.device_characteristics
    for key, value in Object.entries($1))):
      console.log($1))`$1`)
  } else {
    # Default to unittest
    unittest.main())argv=['first-arg-is-ignored']),

  }

  }
if ($1) {
  main()))