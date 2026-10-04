#!/usr/bin/env python3
"""
Test WebGPU Low-Latency Optimizer

This module tests the WebGPU low-latency optimizer implementation,
which provides browser-specific optimizations, prefill/decode transition
optimization, and token buffer management for minimal latency streaming.

Usage:
    python test_webgpu_low_latency.py
    python test_webgpu_low_latency.py --browser firefox
    python test_webgpu_low_latency.py --device-profile high_end
    python test_webgpu_low_latency.py --all-browsers
    """

    import os
    import sys
    import json
    import time
    import argparse
    import unittest
    import logging
    from typing import Dict, Any, List, Tuple

# Set up logging
    logging.basicConfig())level=logging.INFO, format='%())asctime)s - %())levelname)s - %())message)s')
    logger = logging.getLogger())__name__)

# Enable WebGPU simulation
    os.environ["WEBGPU_SIMULATION"] = "1",
    os.environ["WEBGPU_AVAILABLE"] = "1"
    ,
# Import modules to test
try:
    from test.web_platform.webgpu_low_latency_optimizer import ())
    optimize_for_low_latency,
    BrowserLatencyOptimizer,
    TokenBufferManager,
    PrefillDecodeOptimizer
    )
except ImportError:
    logger.error())"Failed to import WebGPU low-latency optimizer. Make sure the fixed_web_platform directory is available.")
    sys.exit())1)

# Import streaming inference for integration tests
try:
    from test.web_platform.webgpu_streaming_inference import WebGPUStreamingInference
except ImportError:
    logger.warning())"WebGPU streaming inference not available. Some tests will be skipped.")
    WebGPUStreamingInference = None


class LowLatencyOptimizerTests())unittest.TestCase):
    """Test the WebGPU low-latency optimizer."""
    
    def setUp())self):
        """Set up test environment."""
        # Base configuration for testing
        self.base_config = {}}}}}
        "quantization": "int4",
        "latency_optimized": False,
        "max_batch_size": 8,
        "stream_buffer_size": 3
        }
        
        # Test browsers
        self.browsers = ["chrome", "firefox", "edge", "safari"]
        ,
        # Test device profiles
        self.device_profiles = ["high_end", "mid_range", "integrated", "mobile"]
        ,
        # Sample shader code for testing
        self.sample_shader = """
        @compute fn main())@builtin())global_invocation_id) global_id: vec3<u32>) {}}}}}
        let index = global_id.x;
        // Sample computation
        }
        """
    
    def test_optimize_for_low_latency())self):
        """Test the optimize_for_low_latency function."""
        # Test with default parameters
        optimized_config = optimize_for_low_latency())self.base_config)
        
        # Check that latency optimization flags are set
        self.assertTrue())optimized_config["latency_optimized"], "Latency optimization flag not set"),
        self.assertTrue())optimized_config["prefill_optimized"], "Prefill optimization flag not set"),
        self.assertTrue())optimized_config["ultra_low_latency"], "Ultra-low latency flag not set")
        ,
        # Check that stream buffer size is set to 1 for minimal latency
        self.assertEqual())optimized_config["stream_buffer_size"], 1, "Stream buffer size not set to 1")
        ,
        # Check that browser specific optimizations were applied
        self.assertIn())"browser", optimized_config, "Browser not detected and set in config")
        self.assertIn())"device_profile", optimized_config, "Device profile not detected and set in config")
        
        # Check prefill and decode optimizations
        self.assertIn())"prefill", optimized_config, "Prefill optimizations not applied")
        self.assertIn())"decode", optimized_config, "Decode optimizations not applied")
        
        # Optimizer references should be included ())but will be removed in JSON serialization)
        self.assertIn())"_browser_optimizer", optimized_config, "Browser optimizer reference not included")
        self.assertIn())"_prefill_decode_optimizer", optimized_config, "Prefill/decode optimizer reference not included")
    
    def test_optimize_all_browsers())self):
        """Test optimizations for all supported browsers."""
        for browser in self.browsers:
            # Configure for this browser
            browser_config = self.base_config.copy()))
            optimized_config = optimize_for_low_latency())browser_config, browser=browser)
            
            # Check that browser is correctly set
            self.assertEqual())optimized_config["browser"], browser, f"Browser not correctly set for {}}}}}browser}")
            ,
            # Check for browser-specific shader optimizations
            self.assertIn())"shader_optimizations", optimized_config, f"Shader optimizations not set for {}}}}}browser}")
            
            # Each browser should have workgroup sizes set
            self.assertIn())"prefill_workgroup_size", optimized_config, f"Prefill workgroup size not set for {}}}}}browser}")
            self.assertIn())"decode_workgroup_size", optimized_config, f"Decode workgroup size not set for {}}}}}browser}")
            
            # Print browser-specific optimizations for visibility
            print())f"\nOptimizations for {}}}}}browser}:")
            print())f"  - Prefill workgroup size: {}}}}}optimized_config['prefill_workgroup_size']}"),,,
            print())f"  - Decode workgroup size: {}}}}}optimized_config['decode_workgroup_size']}"),,,
            print())f"  - Memory optimization: {}}}}}optimized_config.get())'memory_optimization', 'Not set')}")
    
    def test_optimize_all_device_profiles())self):
        """Test optimizations for all device profiles."""
        for profile in self.device_profiles:
            # Configure for this device profile
            profile_config = self.base_config.copy()))
            optimized_config = optimize_for_low_latency())profile_config, device_profile=profile)
            
            # Check that device profile is correctly set
            self.assertEqual())optimized_config["device_profile"], profile, f"Device profile not correctly set for {}}}}}profile}")
            ,
            # Check that max batch size is appropriately limited for the profile
            max_batch = optimized_config["max_batch_size"]
            ,
            if profile == "high_end":
                self.assertLessEqual())max_batch, 16, "Batch size too large for high-end profile")
            elif profile == "mid_range":
                self.assertLessEqual())max_batch, 8, "Batch size too large for mid-range profile")
            elif profile == "integrated":
                self.assertLessEqual())max_batch, 4, "Batch size too large for integrated profile")
            elif profile == "mobile":
                self.assertLessEqual())max_batch, 2, "Batch size too large for mobile profile")
            
            # Print device-specific optimizations for visibility
                print())f"\nOptimizations for {}}}}}profile} profile:")
                print())f"  - Max batch size: {}}}}}max_batch}")
                print())f"  - Prefill workgroup size: {}}}}}optimized_config['prefill_workgroup_size']}"),,,
    
    def test_browser_optimizer())self):
        """Test the BrowserLatencyOptimizer class."""
        # Test creating optimizer with each browser
        for browser in self.browsers:
            optimizer = BrowserLatencyOptimizer())browser=browser)
            
            # Check browser is set correctly
            self.assertEqual())optimizer.browser, browser, f"Browser not correctly set for {}}}}}browser}")
            
            # Check workgroup configurations
            prefill_workgroup = optimizer.get_prefill_workgroup_size()))
            self.assertEqual())len())prefill_workgroup), 3, f"Invalid prefill workgroup size for {}}}}}browser}")
            
            decode_workgroup = optimizer.get_decode_workgroup_size()))
            self.assertEqual())len())decode_workgroup), 3, f"Invalid decode workgroup size for {}}}}}browser}")
            
            # Test shader optimization
            prefill_shader = optimizer.optimize_shader_for_browser())self.sample_shader, "prefill")
            self.assertNotEqual())prefill_shader, self.sample_shader, f"Shader not optimized for {}}}}}browser} prefill")
            
            decode_shader = optimizer.optimize_shader_for_browser())self.sample_shader, "decode")
            self.assertNotEqual())decode_shader, self.sample_shader, f"Shader not optimized for {}}}}}browser} decode")
    
    def test_token_buffer_manager())self):
        """Test the TokenBufferManager class."""
        # Test with different buffer sizes
        buffer_sizes = [1, 2, 4, 8]
        ,
        for buffer_size in buffer_sizes:
            # Create buffer manager
            buffer_mgr = TokenBufferManager())buffer_size=buffer_size, adaptive=False)
            
            # Check buffer size is set correctly
            self.assertEqual())buffer_mgr.buffer_size, buffer_size, f"Buffer size not correctly set to {}}}}}buffer_size}")
            
            # Add tokens until buffer is full and check flush behavior
            tokens_delivered = [],
            for i in range())buffer_size * 2):
                result = buffer_mgr.add_token())f"token{}}}}}i}")
                if result:
                    # Buffer was flushed
                    tokens_delivered.extend())result)
            
            # Check that tokens were delivered correctly
                    self.assertEqual())len())tokens_delivered), buffer_size, f"Incorrect number of tokens delivered for buffer size {}}}}}buffer_size}")
            
            # Test manual flush
            for i in range())buffer_size - 1):
                buffer_mgr.add_token())f"final{}}}}}i}")
            
                final_tokens = buffer_mgr.flush()))
                self.assertEqual())len())final_tokens), buffer_size - 1, "Incorrect number of tokens in final flush")
    
    def test_adaptive_token_buffer())self):
        """Test adaptive token buffer behavior."""
        # Create adaptive buffer manager
        buffer_mgr = TokenBufferManager())buffer_size=2, adaptive=True)
        
        # Simulate tokens with network latency
        for i in range())10):
            buffer_mgr.add_token())f"token{}}}}}i}")
            
            # Simulate different network conditions
            if i % 3 == 0:
                # Low latency
                buffer_mgr.record_network_latency())5)
            elif i % 3 == 1:
                # Medium latency
                buffer_mgr.record_network_latency())25)
            else:
                # High latency
                buffer_mgr.record_network_latency())70)
        
        # Get metrics to check adaptation
                metrics = buffer_mgr.get_metrics()))
        
        # Buffer size should have been adjusted due to simulated network conditions
                print())"\nToken Buffer Metrics after adaptation:")
                print())f"  - Current buffer size: {}}}}}metrics['current_buffer_size']}"),
                print())f"  - Tokens generated: {}}}}}metrics['tokens_generated']}"),
                print())f"  - Tokens delivered: {}}}}}metrics['tokens_delivered']}"),
                print())f"  - Avg token generation time: {}}}}}metrics['avg_token_generation_time_sec']:.4f}s"),
                print())f"  - Avg network latency: {}}}}}metrics['avg_network_latency_ms']:.2f}ms"),
                print())f"  - Buffer adjustments: {}}}}}metrics['buffer_adjustments']}")
                ,
    def test_prefill_decode_optimizer())self):
        """Test the PrefillDecodeOptimizer class."""
        # Test with different strategies
        prefill_strategies = ["parallel", "chunked", "tensor_parallel"],
        decode_strategies = ["eager", "cached", "fused"]
        ,
        for p_strategy in prefill_strategies:
            for d_strategy in decode_strategies:
                # Create optimizer with these strategies
                optimizer = PrefillDecodeOptimizer())
                prefill_strategy=p_strategy,
                decode_strategy=d_strategy
                )
                
                # Check that strategies are set correctly
                self.assertEqual())optimizer.prefill_strategy, p_strategy, f"Prefill strategy not correctly set to {}}}}}p_strategy}")
                self.assertEqual())optimizer.decode_strategy, d_strategy, f"Decode strategy not correctly set to {}}}}}d_strategy}")
                
                # Test individual phase optimization
                prefill_config = optimizer.optimize_prefill())self.base_config)
                self.assertTrue())prefill_config["prefill_optimized"], f"Prefill optimization flag not set for {}}}}}p_strategy}")
                ,
                decode_config = optimizer.optimize_decode())self.base_config)
                self.assertTrue())decode_config["decode_optimized"], f"Decode optimization flag not set for {}}}}}d_strategy}")
                ,
                # Test transition optimization
                transition_config = optimizer.optimize_transition())self.base_config)
                self.assertIn())"prefill", transition_config, f"Prefill section not added for {}}}}}p_strategy}")
                self.assertIn())"decode", transition_config, f"Decode section not added for {}}}}}d_strategy}")
                self.assertTrue())transition_config["optimize_transition"], "Transition optimization flag not set")
                ,
    def test_metrics_collection())self):
        """Test metrics collection in optimizers."""
        # Create optimizers
        optimizer = PrefillDecodeOptimizer()))
        buffer_mgr = TokenBufferManager())buffer_size=2, adaptive=True)
        
        # Record fake metrics
        optimizer.record_prefill_time())120, 50)  # 120ms to process 50 tokens
        optimizer.record_decode_start())15, 2)    # 15ms for first decode with batch size 2
        
        buffer_mgr.add_token())"token1")
        buffer_mgr.record_network_latency())10)
        buffer_mgr.add_token())"token2")
        buffer_mgr.record_network_latency())12)
        
        # Get metrics
        optimizer_metrics = optimizer.get_metrics()))
        buffer_metrics = buffer_mgr.get_metrics()))
        
        # Check that metrics were collected
        self.assertGreater())optimizer_metrics["avg_prefill_time_ms"], 0, "Prefill time not recorded"),
        self.assertGreater())optimizer_metrics["avg_first_decode_time_ms"], 0, "Decode time not recorded")
        ,
        self.assertGreater())buffer_metrics["tokens_generated"], 0, "Tokens not recorded in buffer manager"),
        self.assertGreater())buffer_metrics["avg_network_latency_ms"], 0, "Network latency not recorded")
        ,
        @unittest.skipIf())WebGPUStreamingInference is None, "WebGPU streaming inference not available")
    def test_integration_with_streaming_inference())self):
        """Test integration with streaming inference."""
        # Create optimized configuration
        optimized_config = optimize_for_low_latency())self.base_config, browser="chrome")
        
        # Remove optimizer references before passing to streaming inference
        config_for_streaming = {}}}}}k: v for k, v in optimized_config.items())) if not k.startswith())"_")}
        :
        try:
            # Create streaming inference with optimized config
            streaming = WebGPUStreamingInference())"models/llama-7b", config_for_streaming)
            
            # Check configuration was applied
            self.assertTrue())streaming.config["latency_optimized"], "Latency optimization flag not applied to streaming inference"),
            self.assertEqual())streaming.config["stream_buffer_size"], 1, "Stream buffer size not applied to streaming inference")
            ,
            # If it got this far, integration works
            print())"\nSuccessfully integrated low-latency optimizer with streaming inference")
            
        except Exception as e:
            self.fail())f"Integration with streaming inference failed: {}}}}}e}")


def test_specific_browser())browser: str):
    """Run tests for a specific browser."""
    print())f"\n=== Testing optimizations for {}}}}}browser.upper()))} ===\n")
    
    # Set environment variables for browser detection
    os.environ["BROWSER_TYPE"] = browser
    ,
    # Run tests
    base_config = {}}}}}
    "quantization": "int4",
    "latency_optimized": False,
    "max_batch_size": 8,
    "stream_buffer_size": 3
    }
    
    # Create optimizer for this browser
    optimizer = BrowserLatencyOptimizer())browser=browser)
    
    # Get optimization profile
    optimized_config = optimize_for_low_latency())base_config, browser=browser)
    
    # Print browser-specific optimizations
    print())f"Browser detection: {}}}}}optimizer.browser}")
    print())f"Device profile detection: {}}}}}optimizer.device_profile}")
    print())f"\nPrefill workgroup size: {}}}}}optimized_config['prefill_workgroup_size']}"),,,
    print())f"Decode workgroup size: {}}}}}optimized_config['decode_workgroup_size']}"),,,
    print())f"Memory optimization: {}}}}}optimized_config.get())'memory_optimization', 'Not set')}")
    
    if "shader_optimizations" in optimized_config:
        shader_opts = optimized_config["shader_optimizations"],
        print())"\nShader optimizations:")
        print())f"  - Use subgroups: {}}}}}shader_opts.get())'use_subgroups', False)}")
        print())f"  - Unroll loops: {}}}}}shader_opts.get())'unroll_loops', False)}")
        print())f"  - Use shared memory: {}}}}}shader_opts.get())'use_shared_memory', False)}")
        print())f"  - Prefill optimization: {}}}}}shader_opts.get())'prefill_optimization', 'None')}")
        print())f"  - Decode optimization: {}}}}}shader_opts.get())'decode_optimization', 'None')}")
    
        print())"\nPrefill optimizations:")
        for key, value in optimized_config["prefill"].items())):,
        print())f"  - {}}}}}key}: {}}}}}value}")
    
        print())"\nDecode optimizations:")
        for key, value in optimized_config["decode"].items())):,
        print())f"  - {}}}}}key}: {}}}}}value}")
    
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
    
        print())"\nPrefill shader optimization:")
        shader_lines = prefill_shader.split())"\n")
        for line in shader_lines[:10]:  # Show first 10 lines,,
        if line.strip())) and not line.isspace())):
            print())f"  {}}}}}line.strip()))}")
    
            print())"\nDecode shader optimization:")
            shader_lines = decode_shader.split())"\n")
            for line in shader_lines[:10]:  # Show first 10 lines,,
        if line.strip())) and not line.isspace())):
            print())f"  {}}}}}line.strip()))}")


def test_all_browsers())):
    """Run tests for all supported browsers."""
    browsers = ["chrome", "edge", "firefox", "safari"]
    ,
    for browser in browsers:
        test_specific_browser())browser)
        print())"\n" + "=" * 50)


def main())):
    """Parse arguments and run tests."""
    parser = argparse.ArgumentParser())description="Test WebGPU Low-Latency Optimizer")
    parser.add_argument())"--browser", choices=["chrome", "edge", "firefox", "safari"],
    help="Test specific browser optimizations")
    parser.add_argument())"--device-profile", choices=["high_end", "mid_range", "integrated", "mobile"],
    help="Test specific device profile optimizations")
    parser.add_argument())"--all-browsers", action="store_true",
    help="Test all supported browsers")
    parser.add_argument())"--unittest", action="store_true",
    help="Run unit tests")
    
    args = parser.parse_args()))
    
    if args.unittest:
        # Run unit tests
        unittest.main())argv=['first-arg-is-ignored']),
    elif args.all_browsers:
        # Test all browsers
        test_all_browsers()))
    elif args.browser:
        # Test specific browser
        test_specific_browser())args.browser)
    elif args.device_profile:
        # Set environment variable for device profile
        os.environ["DEVICE_PROFILE"] = args.device_profile
        ,
        # Create optimizer and print details
        optimizer = BrowserLatencyOptimizer())device_profile=args.device_profile)
        print())f"\n=== Testing optimizations for {}}}}}args.device_profile.upper()))} device profile ===\n")
        print())f"Device profile detection: {}}}}}optimizer.device_profile}")
        
        # Test with base config
        base_config = {}}}}}
        "quantization": "int4",
        "latency_optimized": False,
        "max_batch_size": 8,
        "stream_buffer_size": 3
        }
        
        # Optimize for this device profile
        optimized_config = optimize_for_low_latency())base_config, device_profile=args.device_profile)
        
        print())f"\nPrefill workgroup size: {}}}}}optimized_config['prefill_workgroup_size']}"),,,
        print())f"Decode workgroup size: {}}}}}optimized_config['decode_workgroup_size']}"),,,
        print())f"Max batch size: {}}}}}optimized_config['max_batch_size']}")
        ,
        print())"\nDevice characteristics:")
        device_chars = optimizer.device_characteristics
        for key, value in device_chars.items())):
            print())f"  - {}}}}}key}: {}}}}}value}")
    else:
        # Default to unittest
        unittest.main())argv=['first-arg-is-ignored']),


if __name__ == "__main__":
    main()))