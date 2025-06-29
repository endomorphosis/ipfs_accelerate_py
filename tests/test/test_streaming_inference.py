#!/usr/bin/env python3
"""
Test Streaming Inference Implementation

This module tests the streaming inference implementation with focus on:
    1. AdaptiveBatchSizeController - Tests for adaptive batch sizing
    2. LowLatencyOptimizer - Tests for latency optimization
    3. StreamingTelemetryCollector - Tests for telemetry collection
    4. MemoryPressureMonitor - Tests for memory pressure handling
    5. StreamingInferencePipeline - Tests for the full pipeline

Usage:
    python test_streaming_inference.py
    python test_streaming_inference.py --verbose
    """

    import os
    import sys
    import json
    import time
    import asyncio
    import logging
    import unittest
    from typing import Dict, List, Any, Optional

# Configure logging
    logging.basicConfig()))))))))))level=logging.INFO, format='%()))))))))))asctime)s - %()))))))))))levelname)s - %()))))))))))message)s')
    logger = logging.getLogger()))))))))))"streaming_inference_test")

# Add parent directory to path for imports
    sys.path.append()))))))))))os.path.dirname()))))))))))os.path.dirname()))))))))))os.path.abspath()))))))))))__file__))))

# Import streaming inference modules
try:
    from fixed_web_platform.streaming_inference import ()))))))))))
    AdaptiveBatchSizeController,
    LowLatencyOptimizer,
    StreamingTelemetryCollector,
    MemoryPressureMonitor,
    StreamingInferencePipeline
    )
except ImportError:
    logger.error()))))))))))"Could not import streaming_inference module. Make sure it exists.")
    sys.exit()))))))))))1)


class TestAdaptiveBatchSizeController()))))))))))unittest.TestCase):
    """Test the AdaptiveBatchSizeController class."""
    
    def setUp()))))))))))self):
        """Set up test case."""
        self.controller = AdaptiveBatchSizeController()))))))))))
        min_batch_size=1,
        max_batch_size=8,
        config={}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "target_latency_ms": 50,
        "max_latency_ms": 150
        }
        )
    
    def test_initialization()))))))))))self):
        """Test initialization of the controller."""
        self.assertEqual()))))))))))self.controller.min_batch_size, 1)
        self.assertEqual()))))))))))self.controller.max_batch_size, 8)
        self.assertEqual()))))))))))self.controller.current_batch_size, 1)
        self.assertEqual()))))))))))self.controller.config["target_latency_ms"], 50),
        self.assertEqual()))))))))))self.controller.config["max_latency_ms"], 150)
        ,
    def test_device_initialization()))))))))))self):
        """Test initialization based on device capabilities."""
        # High-end GPU
        high_end_device = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "gpu_available": True,
        "gpu_type": "NVIDIA RTX 3090",
        "gpu_memory_mb": 16000,
        "cpu_cores": 16
        }
        batch_size = self.controller.initialize_for_device()))))))))))high_end_device)
        self.assertEqual()))))))))))batch_size, 8)
        self.assertEqual()))))))))))self.controller.device_profile["performance_tier"], "high"),
        ,
        # Mid-range GPU
        self.controller.current_batch_size = 1  # Reset
        mid_range_device = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "gpu_available": True,
        "gpu_type": "NVIDIA GTX 1660",
        "gpu_memory_mb": 6000,
        "cpu_cores": 8
        }
        batch_size = self.controller.initialize_for_device()))))))))))mid_range_device)
        self.assertEqual()))))))))))batch_size, 4)
        self.assertEqual()))))))))))self.controller.device_profile["performance_tier"], "medium")
        ,
        # Low-end GPU
        self.controller.current_batch_size = 1  # Reset
        low_end_device = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "gpu_available": True,
        "gpu_type": "NVIDIA MX450",
        "gpu_memory_mb": 2048,
        "cpu_cores": 4
        }
        batch_size = self.controller.initialize_for_device()))))))))))low_end_device)
        self.assertEqual()))))))))))batch_size, 1)
        self.assertEqual()))))))))))self.controller.device_profile["performance_tier"], "low")
        ,
        # CPU only
        self.controller.current_batch_size = 1  # Reset
        cpu_device = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "gpu_available": False,
        "gpu_type": "none",
        "gpu_memory_mb": 0,
        "cpu_cores": 4
        }
        batch_size = self.controller.initialize_for_device()))))))))))cpu_device)
        self.assertEqual()))))))))))batch_size, 1)
        self.assertEqual()))))))))))self.controller.device_profile["performance_tier"], "cpu_only")
        ,
    def test_network_conditions()))))))))))self):
        """Test adaptation based on network conditions."""
        # Stable network
        stable_network = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "latency_ms": 50,
        "bandwidth_mbps": 10.0,
        "stability": 0.95
        }
        self.controller.current_batch_size = 4  # Start with batch size 4
        batch_size = self.controller.update_network_conditions()))))))))))stable_network)
        self.assertEqual()))))))))))batch_size, 4)  # Should remain the same
        
        # Unstable network
        unstable_network = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "latency_ms": 200,
        "bandwidth_mbps": 2.0,
        "stability": 0.6
        }
        batch_size = self.controller.update_network_conditions()))))))))))unstable_network)
        self.assertEqual()))))))))))batch_size, 2)  # Should reduce to half
    
    def test_batch_size_adjustment()))))))))))self):
        """Test batch size adjustment based on performance."""
        # Initialize with some history
        self.controller.current_batch_size = 2
        
        # Good performance - low latency
        good_stats = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "tokens_per_second": 50,
        "latency_ms": 30,  # Below target latency
        "memory_usage_mb": 2000
        }
        
        # Update multiple times to build history
        for _ in range()))))))))))5):
            self.controller.update_after_batch()))))))))))good_stats)
        
        # Batch size should increase
            self.assertEqual()))))))))))self.controller.current_batch_size, 3)
        
        # Poor performance - high latency
            poor_stats = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "tokens_per_second": 20,
            "latency_ms": 200,  # Above max latency
            "memory_usage_mb": 2500
            }
        
        # Update multiple times to build history
        for _ in range()))))))))))5):
            self.controller.update_after_batch()))))))))))poor_stats)
        
        # Batch size should decrease
            self.assertEqual()))))))))))self.controller.current_batch_size, 2)
    
    def test_memory_pressure_handling()))))))))))self):
        """Test adaptation under memory pressure."""
        self.controller.current_batch_size = 8
        
        # Apply memory pressure
        changed = self.controller.handle_memory_pressure()))))))))))True)
        
        # Batch size should be halved
        self.assertEqual()))))))))))self.controller.current_batch_size, 4)
        self.assertTrue()))))))))))changed)
        
        # Apply memory pressure again
        changed = self.controller.handle_memory_pressure()))))))))))True)
        
        # Batch size should be halved again
        self.assertEqual()))))))))))self.controller.current_batch_size, 2)
        self.assertTrue()))))))))))changed)
        
        # Apply memory pressure again
        changed = self.controller.handle_memory_pressure()))))))))))True)
        
        # Batch size should be halved again
        self.assertEqual()))))))))))self.controller.current_batch_size, 1)
        self.assertTrue()))))))))))changed)
        
        # Apply memory pressure once more
        changed = self.controller.handle_memory_pressure()))))))))))True)
        
        # Batch size should remain at min_batch_size
        self.assertEqual()))))))))))self.controller.current_batch_size, 1)
        self.assertFalse()))))))))))changed)  # No change this time


class TestLowLatencyOptimizer()))))))))))unittest.TestCase):
    """Test the LowLatencyOptimizer class."""
    
    def setUp()))))))))))self):
        """Set up test case."""
        self.optimizer = LowLatencyOptimizer()))))))))))
        config={}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "optimization_level": "balanced",
        "enable_prefetch": True
        }
        )
    
    def test_initialization()))))))))))self):
        """Test initialization of the optimizer."""
        self.assertEqual()))))))))))self.optimizer.optimization_level, "balanced")
        self.assertTrue()))))))))))self.optimizer.prefetch_enabled)
    
    def test_browser_initialization()))))))))))self):
        """Test initialization for different browsers."""
        # Chrome
        chrome_info = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "name": "chrome",
        "version": 120
        }
        profile = self.optimizer.initialize_for_browser()))))))))))chrome_info)
        self.assertTrue()))))))))))profile["supports_transfer_overlap"]),,,,,
        self.assertTrue()))))))))))profile["supports_worker_threads"]),,,,,
        self.assertTrue()))))))))))profile["supports_stream_optimization"]),,
        self.assertEqual()))))))))))profile["optimal_chunk_size"], 8)
        ,
        # Firefox
        firefox_info = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "name": "firefox",
        "version": 115
        }
        profile = self.optimizer.initialize_for_browser()))))))))))firefox_info)
        self.assertTrue()))))))))))profile["supports_transfer_overlap"]),,,,,
        self.assertTrue()))))))))))profile["supports_worker_threads"]),,,,,
        self.assertTrue()))))))))))profile["supports_stream_optimization"]),,
        self.assertEqual()))))))))))profile["optimal_chunk_size"], 4)
        ,
        # Older Firefox
        old_firefox_info = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "name": "firefox",
        "version": 110
        }
        profile = self.optimizer.initialize_for_browser()))))))))))old_firefox_info)
        self.assertTrue()))))))))))profile["supports_transfer_overlap"]),,,,,
        self.assertTrue()))))))))))profile["supports_worker_threads"]),,,,,
        self.assertFalse()))))))))))profile["supports_stream_optimization"]),,
        
        # Safari
        safari_info = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "name": "safari",
        "version": 17
        }
        profile = self.optimizer.initialize_for_browser()))))))))))safari_info)
        self.assertTrue()))))))))))profile["supports_transfer_overlap"]),,,,,
        self.assertTrue()))))))))))profile["supports_worker_threads"]),,,,,
        self.assertTrue()))))))))))profile["supports_stream_optimization"]),,
        
        # Older Safari
        old_safari_info = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "name": "safari",
        "version": 14
        }
        profile = self.optimizer.initialize_for_browser()))))))))))old_safari_info)
        self.assertFalse()))))))))))profile["supports_transfer_overlap"]),,,,,
        self.assertFalse()))))))))))profile["supports_worker_threads"]),,,,,
        self.assertFalse()))))))))))profile["supports_stream_optimization"]),,
    
    def test_token_generation_optimization()))))))))))self):
        """Test optimization of token generation."""
        # Setup Chrome profile
        chrome_info = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "name": "chrome",
        "version": 120
        }
        self.optimizer.initialize_for_browser()))))))))))chrome_info)
        
        # Test optimizations for first token ()))))))))))prompt processing)
        optimizations = self.optimizer.optimize_token_generation()))))))))))
        model=None,
        inputs=list()))))))))))range()))))))))))1000)),  # 1000 input tokens
        generated_tokens=[],  # No tokens generated yet,
        )
        
        self.assertTrue()))))))))))optimizations["use_kv_cache"]),
        self.assertTrue()))))))))))optimizations["prompt_chunking"]),
        self.assertTrue()))))))))))optimizations["prefetch_first_token"]),
        self.assertEqual()))))))))))optimizations["prompt_chunk_size"], 512)
        ,
        # Test optimizations for early tokens
        optimizations = self.optimizer.optimize_token_generation()))))))))))
        model=None,
        inputs=list()))))))))))range()))))))))))100)),
        generated_tokens=list()))))))))))range()))))))))))2))  # 2 tokens generated
        )
        
        self.assertTrue()))))))))))optimizations["reduce_batch_size"]),
        self.assertTrue()))))))))))optimizations["aggressive_prefetch"])
        ,
        # Test optimizations for later tokens
        optimizations = self.optimizer.optimize_token_generation()))))))))))
        model=None,
        inputs=list()))))))))))range()))))))))))100)),
        generated_tokens=list()))))))))))range()))))))))))10))  # 10 tokens generated
        )
        
        self.assertTrue()))))))))))optimizations["enable_batch_processing"]),
        self.assertTrue()))))))))))optimizations["adaptive_prefetch"])
        ,
    def test_update_after_token()))))))))))self):
        """Test update after token generation."""
        # Compute-bound scenario
        compute_stats = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "compute_time_ms": 100,
        "transfer_time_ms": 10
        }
        self.optimizer.update_after_token()))))))))))compute_stats)
        self.assertEqual()))))))))))self.optimizer.compute_transfer_ratio, 10.0)
        self.assertEqual()))))))))))self.optimizer.optimization_level, "compute_focused")
        
        # Transfer-bound scenario
        transfer_stats = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "compute_time_ms": 10,
        "transfer_time_ms": 100
        }
        self.optimizer.update_after_token()))))))))))transfer_stats)
        self.assertEqual()))))))))))self.optimizer.compute_transfer_ratio, 0.1)
        self.assertEqual()))))))))))self.optimizer.optimization_level, "transfer_focused")
        
        # Balanced scenario
        balanced_stats = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "compute_time_ms": 50,
        "transfer_time_ms": 50
        }
        self.optimizer.update_after_token()))))))))))balanced_stats)
        self.assertEqual()))))))))))self.optimizer.compute_transfer_ratio, 1.0)
        self.assertEqual()))))))))))self.optimizer.optimization_level, "balanced")


class TestStreamingTelemetryCollector()))))))))))unittest.TestCase):
    """Test the StreamingTelemetryCollector class."""
    
    def setUp()))))))))))self):
        """Set up test case."""
        self.collector = StreamingTelemetryCollector()))))))))))
        config={}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "enabled": True,
        "sampling_rate": 1.0
        }
        )
    
    def test_initialization()))))))))))self):
        """Test initialization of the collector."""
        self.assertTrue()))))))))))self.collector.enabled)
        self.assertEqual()))))))))))self.collector.sampling_rate, 1.0)
        self.assertIsNone()))))))))))self.collector.start_time)
    
    def test_session_management()))))))))))self):
        """Test session start and metrics."""
        self.collector.start_session())))))))))))
        self.assertIsNotNone()))))))))))self.collector.start_time)
        self.assertEqual()))))))))))len()))))))))))self.collector.metrics["token_latency"]), 0)
        ,
        # Add some metrics
        for i in range()))))))))))5):
            self.collector.record_token_generated())))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "latency_ms": 50 + i * 10,
            "tokens_per_second": 20 - i,
            "memory_usage_mb": 1000 + i * 100,
            "batch_size": 4
            })
        
        # Check metrics were recorded
            self.assertEqual()))))))))))len()))))))))))self.collector.metrics["token_latency"]), 5),
            self.assertEqual()))))))))))len()))))))))))self.collector.metrics["throughput"]), 5),
            self.assertEqual()))))))))))len()))))))))))self.collector.metrics["memory_usage"]), 5),
            self.assertEqual()))))))))))len()))))))))))self.collector.metrics["batch_sizes"]), 5)
            ,
    def test_error_recording()))))))))))self):
        """Test error recording."""
        self.collector.start_session())))))))))))
        
        # Record an error
        self.collector.record_error())))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "type": "RuntimeError",
        "message": "Test error",
        "token_position": 10,
        "recovered": False
        })
        
        # Check error was recorded
        self.assertEqual()))))))))))len()))))))))))self.collector.metrics["errors"]), 1),
        self.assertEqual()))))))))))self.collector.metrics["errors"][0],["type"], "RuntimeError"),
        self.assertEqual()))))))))))self.collector.metrics["errors"][0],["message"], "Test error")
        ,
    def test_session_summary()))))))))))self):
        """Test session summary generation."""
        self.collector.start_session())))))))))))
        
        # Add some metrics
        for i in range()))))))))))10):
            self.collector.record_token_generated())))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "latency_ms": 50 + i * 5,
            "tokens_per_second": 20 - i * 0.5,
            "memory_usage_mb": 1000 + i * 50,
            "batch_size": 4 if i < 5 else 8
            })
        
        # Record an error
        self.collector.record_error())))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}:
            "type": "RuntimeError",
            "message": "Test error",
            "token_position": 5,
            "recovered": False
            })
        
        # Get summary
            time.sleep()))))))))))0.1)  # Ensure some session duration
            summary = self.collector.get_session_summary())))))))))))
        
        # Check summary values
            self.assertEqual()))))))))))summary["total_tokens"], 10),
            self.assertTrue()))))))))))summary["session_duration_sec"] > 0),
            self.assertTrue()))))))))))summary["average_token_latency_ms"] > 0),
            self.assertTrue()))))))))))summary["p95_token_latency_ms"] > 0),
            self.assertTrue()))))))))))summary["average_throughput_tokens_per_sec"] > 0),
            self.assertEqual()))))))))))summary["error_count"], 1),
            self.assertEqual()))))))))))summary["error_rate"], 0.1)  # 1 error in 10 tokens,
            self.assertEqual()))))))))))summary["most_common_batch_size"], 4)
            ,
    def test_percentile_calculation()))))))))))self):
        """Test percentile calculation."""
        # Skip test if not implemented:
        if not hasattr()))))))))))self.collector, '_percentile'):
            self.skipTest()))))))))))"_percentile method not implemented")
            
        # Empty list
            self.assertEqual()))))))))))self.collector._percentile()))))))))))[],, 95), 0)
            ,
        # Single value
            self.assertEqual()))))))))))self.collector._percentile()))))))))))[10], 50), 10)
            ,
        # Multiple values
            values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            self.assertEqual()))))))))))self.collector._percentile()))))))))))values, 50), 55)  # Median
            self.assertEqual()))))))))))self.collector._percentile()))))))))))values, 90), 95)  # 90th percentile
    
    def test_most_common_calculation()))))))))))self):
        """Test most common element calculation."""
        # Empty list
        self.assertIsNone()))))))))))self.collector._most_common()))))))))))[],))
        ,
        # Single value
        self.assertEqual()))))))))))self.collector._most_common()))))))))))[5]), 5)
        ,
        # Multiple values with clear mode
        self.assertEqual()))))))))))self.collector._most_common()))))))))))[1, 2, 2, 3, 2, 4]), 2)
        ,
        # Multiple values with tie
        result = self.collector._most_common()))))))))))[1, 2, 3, 1, 2, 3]),
        self.assertTrue()))))))))))result in [1, 2, 3])

        ,
class TestMemoryPressureMonitor()))))))))))unittest.TestCase):
    """Test the MemoryPressureMonitor class."""
    
    def setUp()))))))))))self):
        """Set up test case."""
        self.monitor = MemoryPressureMonitor()))))))))))
        config={}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "warning_threshold": 0.75,  # 75% memory usage
        "critical_threshold": 0.85,  # 85% memory usage
        "memory_limit_mb": 8192,  # 8GB
        "check_interval_ms": 100  # Check every 100ms
        }
        )
    
    def test_initialization()))))))))))self):
        """Test initialization of the monitor."""
        self.assertEqual()))))))))))self.monitor.warning_threshold, 0.75)
        self.assertEqual()))))))))))self.monitor.critical_threshold, 0.85)
        self.assertEqual()))))))))))self.monitor.memory_limit_mb, 8192)
        self.assertEqual()))))))))))self.monitor.check_interval_ms, 100)
        self.assertEqual()))))))))))self.monitor.current_memory_mb, 0)
        self.assertEqual()))))))))))self.monitor.peak_memory_mb, 0)
        self.assertFalse()))))))))))self.monitor.pressure_detected)
    
    def test_device_initialization()))))))))))self):
        """Test initialization with device info."""
        device_info = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "gpu_memory_mb": 16000,
        "model_size_mb": 4000
        }
        self.monitor.initialize()))))))))))device_info)
        
        # Memory limit should be 90% of GPU memory
        self.assertAlmostEqual()))))))))))self.monitor.memory_limit_mb, 16000 * 0.9, delta=1)
        
        # Current memory should be model size plus overhead
        self.assertEqual()))))))))))self.monitor.current_memory_mb, 4000 + 100)
        
        # Peak memory should match current memory
        self.assertEqual()))))))))))self.monitor.peak_memory_mb, self.monitor.current_memory_mb)
    
    def test_memory_usage_update()))))))))))self):
        """Test updating memory usage."""
        # Initialize with some usage
        self.monitor.current_memory_mb = 1000
        self.monitor.peak_memory_mb = 1000
        
        # Update to higher usage
        self.monitor.update_memory_usage()))))))))))2000)
        self.assertEqual()))))))))))self.monitor.current_memory_mb, 2000)
        self.assertEqual()))))))))))self.monitor.peak_memory_mb, 2000)
        
        # Update to lower usage
        self.monitor.update_memory_usage()))))))))))1500)
        self.assertEqual()))))))))))self.monitor.current_memory_mb, 1500)
        self.assertEqual()))))))))))self.monitor.peak_memory_mb, 2000)  # Peak should remain
    
    def test_memory_pressure_detection()))))))))))self):
        """Test memory pressure detection."""
        # Initialize with device info
        self.monitor.initialize())))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"gpu_memory_mb": 10000, "model_size_mb": 3000})
        
        # Set up callbacks
        warning_called = False
        critical_called = False
        
        def warning_callback()))))))))))):
            nonlocal warning_called
            warning_called = True
        
        def critical_callback()))))))))))):
            nonlocal critical_called
            critical_called = True
        
            self.monitor.set_warning_callback()))))))))))warning_callback)
            self.monitor.set_critical_callback()))))))))))critical_callback)
        
        # No pressure initially
            self.assertFalse()))))))))))self.monitor.check_memory_pressure()))))))))))))
            self.assertFalse()))))))))))warning_called)
            self.assertFalse()))))))))))critical_called)
        
        # Update to warning level
            self.monitor.update_memory_usage()))))))))))self.monitor.memory_limit_mb * 0.8)
            self.assertTrue()))))))))))self.monitor.check_memory_pressure()))))))))))))
            self.assertTrue()))))))))))warning_called)
            self.assertFalse()))))))))))critical_called)
        
        # Reset flags
            warning_called = False
        
        # Update to critical level
            self.monitor.update_memory_usage()))))))))))self.monitor.memory_limit_mb * 0.9)
            self.assertTrue()))))))))))self.monitor.check_memory_pressure()))))))))))))
            self.assertFalse()))))))))))warning_called)  # Only critical should be called
            self.assertTrue()))))))))))critical_called)
    
    def test_memory_percentage()))))))))))self):
        """Test memory percentage calculation."""
        self.monitor.memory_limit_mb = 8000
        self.monitor.current_memory_mb = 4000
        
        percentage = self.monitor.get_memory_percentage())))))))))))
        self.assertEqual()))))))))))percentage, 50.0)  # 4000 / 8000 * 100


class TestStreamingInferencePipeline()))))))))))unittest.TestCase):
    """Test the StreamingInferencePipeline class."""
    
    def setUp()))))))))))self):
        """Set up test case."""
        # Create a mock model for testing
        mock_model = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "name": "test_model",
        "type": "language_model"
        }
        
        self.pipeline = StreamingInferencePipeline()))))))))))
        model=mock_model,
        config={}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "min_batch_size": 1,
        "max_batch_size": 8,
        "latency_optimizer_config": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "optimization_level": "balanced",
        "enable_prefetch": True
        },
        "memory_monitor_config": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "warning_threshold": 0.75,
        "critical_threshold": 0.85,
        "memory_limit_mb": 8192
        },
        "telemetry_config": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "enabled": True,
        "sampling_rate": 1.0
        }
        }
        )
    
    def test_initialization()))))))))))self):
        """Test initialization of the pipeline."""
        self.assertIsNotNone()))))))))))self.pipeline.batch_size_controller)
        self.assertIsNotNone()))))))))))self.pipeline.latency_optimizer)
        self.assertIsNotNone()))))))))))self.pipeline.memory_monitor)
        self.assertIsNotNone()))))))))))self.pipeline.telemetry_collector)
        self.assertFalse()))))))))))self.pipeline.initialized)
        self.assertFalse()))))))))))self.pipeline.is_generating)
    
    def test_pipeline_initialization()))))))))))self):
        """Test pipeline initialization with device and browser info."""
        device_info = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "gpu_available": True,
        "gpu_type": "NVIDIA RTX 3090",
        "gpu_memory_mb": 16000,
        "cpu_cores": 16,
        "model_size_mb": 4000
        }
        
        browser_info = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "name": "chrome",
        "version": 120
        }
        
        self.pipeline.initialize()))))))))))device_info, browser_info)
        
        self.assertTrue()))))))))))self.pipeline.initialized)
        self.assertEqual()))))))))))self.pipeline.batch_size_controller.current_batch_size, 8)
        self.assertEqual()))))))))))self.pipeline.batch_size_controller.device_profile["performance_tier"], "high"),
        ,self.assertTrue()))))))))))self.pipeline.latency_optimizer.browser_profile["supports_transfer_overlap"]),,,,,
    
    def test_memory_callbacks()))))))))))self):
        """Test memory pressure callbacks."""
        # Initialize first
        device_info = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "gpu_available": True,
        "gpu_type": "NVIDIA RTX 3090",
        "gpu_memory_mb": 16000,
        "cpu_cores": 16,
        "model_size_mb": 4000
        }
        
        browser_info = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "name": "chrome",
        "version": 120
        }
        
        self.pipeline.initialize()))))))))))device_info, browser_info)
        
        # Set batch size to maximum
        self.pipeline.batch_size_controller.current_batch_size = 8
        
        # Simulate memory warning ()))))))))))no action should be taken)
        self.pipeline._on_memory_warning())))))))))))
        self.assertEqual()))))))))))self.pipeline.batch_size_controller.current_batch_size, 8)
        
        # Simulate memory critical event ()))))))))))batch size should be reduced)
        self.pipeline._on_memory_critical())))))))))))
        self.assertEqual()))))))))))self.pipeline.batch_size_controller.current_batch_size, 4)
        
        # Simulate another critical event ()))))))))))batch size should be reduced again)
        self.pipeline._on_memory_critical())))))))))))
        self.assertEqual()))))))))))self.pipeline.batch_size_controller.current_batch_size, 2)
    
    def test_tokenize_and_decode()))))))))))self):
        """Test tokenization and decoding ()))))))))))simulation)."""
        tokens = self.pipeline._tokenize()))))))))))"This is a test")
        self.assertEqual()))))))))))len()))))))))))tokens), 4)
        self.assertEqual()))))))))))tokens[0],, 1000),
        self.assertEqual()))))))))))tokens[3], 1003)
        ,
        text = self.pipeline._decode_token()))))))))))1000)
        self.assertEqual()))))))))))text, "<token_1000>")
    
    def test_token_generation()))))))))))self):
        """Test token generation ()))))))))))simulation)."""
        input_tokens = [1000, 1001, 1002, 1003],
        tokens_generated = 0
        batch_size = 4
        
        next_tokens = self.pipeline._generate_tokens()))))))))))
        input_tokens=input_tokens,
        tokens_generated=tokens_generated,
        batch_size=batch_size
        )
        
        self.assertEqual()))))))))))len()))))))))))next_tokens), batch_size)
        self.assertEqual()))))))))))next_tokens[0],, 2000),
        self.assertEqual()))))))))))next_tokens[3], 2003)

        ,
class AsyncTestCase()))))))))))unittest.TestCase):
    """Base class for asynchronous tests."""
    
    def run_async()))))))))))self, coroutine):
        """Run an asynchronous coroutine in the event loop."""
    return asyncio.get_event_loop()))))))))))).run_until_complete()))))))))))coroutine)


class TestStreamingInferencePipelineAsync()))))))))))AsyncTestCase):
    """Test asynchronous methods of the StreamingInferencePipeline class."""
    
    def setUp()))))))))))self):
        """Set up test case."""
        # Create a mock model for testing
        mock_model = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "name": "test_model",
        "type": "language_model"
        }
        
        self.pipeline = StreamingInferencePipeline()))))))))))
        model=mock_model,
        config={}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "min_batch_size": 1,
        "max_batch_size": 8,
        "latency_optimizer_config": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "optimization_level": "balanced",
        "enable_prefetch": True
        },
        "memory_monitor_config": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "warning_threshold": 0.75,
        "critical_threshold": 0.85,
        "memory_limit_mb": 8192
        },
        "telemetry_config": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "enabled": True,
        "sampling_rate": 1.0
        }
        }
        )
        
        # Initialize the pipeline
        device_info = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "gpu_available": True,
        "gpu_type": "NVIDIA RTX 3090",
        "gpu_memory_mb": 16000,
        "cpu_cores": 16,
        "model_size_mb": 4000
        }
        
        browser_info = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        "name": "chrome",
        "version": 120
        }
        
        self.pipeline.initialize()))))))))))device_info, browser_info)
    
    def test_generate_stream()))))))))))self):
        """Test the generate_stream method."""
        async def _test_generate()))))))))))):
            prompt = "This is a test prompt for streaming inference"
            max_tokens = 20
            
            # Collect tokens
            tokens = [],
            summary = None
            
            async for token_info in self.pipeline.generate_stream()))))))))))prompt, max_tokens):
                if "is_summary" in token_info:
                    summary = token_info["session_summary"],
                else:
                    tokens.append()))))))))))token_info)
            
            # Check results
                    self.assertEqual()))))))))))len()))))))))))tokens), max_tokens)
                    self.assertIsNotNone()))))))))))summary)
                    self.assertEqual()))))))))))summary["total_tokens"], max_tokens)
                    ,
            # Check token info structure
                    first_token = tokens[0],
                    self.assertIn()))))))))))"token", first_token)
                    self.assertIn()))))))))))"text", first_token)
                    self.assertIn()))))))))))"position", first_token)
                    self.assertIn()))))))))))"latency_ms", first_token)
                    self.assertIn()))))))))))"tokens_per_second", first_token)
                    self.assertIn()))))))))))"batch_size", first_token)
                    self.assertIn()))))))))))"memory_usage_mb", first_token)
            
            # Check positions are correct
            for i, token_info in enumerate()))))))))))tokens):
                self.assertEqual()))))))))))token_info["position"], i + 1)
                ,
                self.run_async()))))))))))_test_generate()))))))))))))
    
    def test_pipeline_errors()))))))))))self):
        """Test error handling in the pipeline."""
        # Test initialization error
        with self.assertRaises()))))))))))ValueError):
            self.run_async()))))))))))self.pipeline.generate_stream()))))))))))"test", 10))
        
        # Reset pipeline state
            self.pipeline.initialized = True
        
        # Test concurrent generation error
        async def _test_concurrent()))))))))))):
            self.pipeline.is_generating = True
            with self.assertRaises()))))))))))RuntimeError):
                async for _ in self.pipeline.generate_stream()))))))))))"test", 10):
                pass
        
                self.run_async()))))))))))_test_concurrent()))))))))))))


def main()))))))))))):
    """Run the tests."""
    unittest.main())))))))))))


if __name__ == "__main__":
    main())))))))))))