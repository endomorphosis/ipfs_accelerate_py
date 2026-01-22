#!/usr/bin/env python3
"""
Test WebGPU Streaming Inference Compute/Transfer Overlap

This script tests the enhanced WebGPU streaming inference pipeline with
compute/transfer overlap implementation and browser-specific optimizations.

The key improvements being tested:
    1. Compute/transfer overlap reducing effective latency
    2. Browser-specific optimizations for Chrome, Firefox, and Safari
    3. Adaptive prefetching based on recent performance metrics
    4. Token prediction functionality for optimized prefetching

To run:
    python test_webgpu_compute_transfer_overlap.py --browser chrome
    python test_webgpu_compute_transfer_overlap.py --browser firefox
    python test_webgpu_compute_transfer_overlap.py --compare-browsers
    python test_webgpu_compute_transfer_overlap.py --test-prediction
    """

    import os
    import sys
    import time
    import json
    import argparse
    import logging
    from typing import Dict, List, Any, Optional, Union

# Configure logging
    logging.basicConfig())))))))))))level=logging.INFO, format='%())))))))))))asctime)s - %())))))))))))levelname)s - %())))))))))))message)s')
    logger = logging.getLogger())))))))))))__name__)

# Add parent directory to path
    sys.path.append())))))))))))os.path.dirname())))))))))))os.path.dirname())))))))))))os.path.abspath())))))))))))__file__))))

# Import required modules
try:
    from fixed_web_platform.webgpu_streaming_inference import WebGPUStreamingInference
except ImportError:
    logger.error())))))))))))"Could not import WebGPU streaming inference module. Make sure it exists.")
    sys.exit())))))))))))1)


    def test_compute_transfer_overlap())))))))))))browser_info: Dict[]],,str, Any], precision: str = "int4"):,,
    """
    Test the compute/transfer overlap implementation.
    
    Args:
        browser_info: Browser information dictionary
        precision: Quantization precision ())))))))))))int2, int3, int4)
    
    Returns:
        Dictionary with test results
        """
        logger.info())))))))))))f"Testing compute/transfer overlap with {}}}}}}}}}}}}}}browser_info[]],,'name']} and {}}}}}}}}}}}}}}precision} precision")
        ,,
    # Configure environment based on browser
        os.environ[]],,"WEBGPU_SIMULATION"] = "1"  # Use simulation mode for testing,,
        os.environ[]],,"WEBGPU_AVAILABLE"] = "1"
        ,,
        if browser_info[]],,"name"].lower())))))))))))) == "firefox":,,
        os.environ[]],,"WEBGPU_COMPUTE_SHADERS_ENABLED"] = "1"
        ,,
    # Run tests with and without overlap for comparison
        results = {}}}}}}}}}}}}}}
        "browser": browser_info[]],,"name"],
        "precision": precision,
        "with_overlap": test_with_overlap())))))))))))browser_info, precision),
        "without_overlap": test_without_overlap())))))))))))browser_info, precision)
        }
    
    # Calculate performance improvement
        if "tokens_per_second" in results[]],,"with_overlap"] and "tokens_per_second" in results[]],,"without_overlap"]:,
        with_tps = results[]],,"with_overlap"][]],,"tokens_per_second"],
        without_tps = results[]],,"without_overlap"][]],,"tokens_per_second"]
        ,
        if without_tps > 0:
            improvement = ())))))))))))with_tps - without_tps) / without_tps * 100
            results[]],,"throughput_improvement_percent"] = improvement,
            logger.info())))))))))))f"Performance improvement: {}}}}}}}}}}}}}}improvement:.2f}%")
    
    # Calculate latency improvement
            if "avg_token_latency_ms" in results[]],,"with_overlap"] and "avg_token_latency_ms" in results[]],,"without_overlap"]:,
            with_latency = results[]],,"with_overlap"][]],,"avg_token_latency_ms"],
            without_latency = results[]],,"without_overlap"][]],,"avg_token_latency_ms"]
            ,
        if without_latency > 0:
            improvement = ())))))))))))without_latency - with_latency) / without_latency * 100
            results[]],,"latency_improvement_percent"] = improvement,
            logger.info())))))))))))f"Latency improvement: {}}}}}}}}}}}}}}improvement:.2f}%")
    
            return results


            def test_with_overlap())))))))))))browser_info: Dict[]],,str, Any], precision: str):,,,,,
            """
            Test streaming inference with compute/transfer overlap enabled.
    
    Args:
        browser_info: Browser information dictionary
        precision: Quantization precision
        
    Returns:
        Dictionary with test results
        """
    # Configure with overlap enabled
        config = {}}}}}}}}}}}}}}
        "quantization": precision,
        "optimize_kv_cache": True,
        "latency_optimized": True,
        "adaptive_batch_size": True,
        "browser_info": browser_info,
        # Enable compute/transfer overlap
        "overlap_enabled": True,
        "prefetch_enabled": True
        }
    
    # Create streaming inference handler
        streaming = WebGPUStreamingInference())))))))))))
        model_path="models/llama-7b",
        config=config
        )
    
    # Collect tokens and timing info
        tokens = []],,],,,,,,,,
        timings = []],,],,,,,,,,
    
    # Test generation with callback for timing information
    def token_callback())))))))))))token, is_last=False):
        tokens.append())))))))))))token)
        if hasattr())))))))))))streaming, "_token_timing"):
            timings.append())))))))))))streaming._token_timing.copy())))))))))))))
    
    # Run generation
            start_time = time.time()))))))))))))
            prompt = "Explain the concept of compute/transfer overlap in the context of streaming inference"
    
            streaming.generate())))))))))))
            prompt=prompt,
            max_tokens=20,
            temperature=0.7,
            callback=token_callback
            )
    
            generation_time = time.time())))))))))))) - start_time
    
    # Get performance stats
            stats = streaming.get_performance_stats()))))))))))))
    
    # Prepare results
            results = {}}}}}}}}}}}}}}
            "tokens_generated": len())))))))))))tokens),
            "generation_time_sec": generation_time,
        "tokens_per_second": len())))))))))))tokens) / generation_time if generation_time > 0 else 0,::::
            "optimization_usage": getattr())))))))))))streaming, "_optimization_usage", {}}}}}}}}}}}}}}})
            }
    
    # Calculate average compute and transfer times
    if timings:
        compute_times = []],,t.get())))))))))))"compute_time_ms", 0) for t in timings if "compute_time_ms" in t],
        transfer_times = []],,t.get())))))))))))"transfer_time_ms", 0) for t in timings if "transfer_time_ms" in t],
        prefetch_times = []],,t.get())))))))))))"prefetch_time_ms", 0) for t in timings if "prefetch_time_ms" in t],
        :
        if compute_times:
            results[]],,"avg_compute_time_ms"] = sum())))))))))))compute_times) / len())))))))))))compute_times)
            ,
        if transfer_times:
            results[]],,"avg_transfer_time_ms"] = sum())))))))))))transfer_times) / len())))))))))))transfer_times)
            ,
        if prefetch_times:
            results[]],,"avg_prefetch_time_ms"] = sum())))))))))))prefetch_times) / len())))))))))))prefetch_times)
            ,
        # Calculate overlap efficiency
            overlap_efficiencies = []],,t.get())))))))))))"overlap_efficiency", 0) for t in timings if "overlap_efficiency" in t]:,
        if overlap_efficiencies:
            results[]],,"avg_overlap_efficiency"] = sum())))))))))))overlap_efficiencies) / len())))))))))))overlap_efficiencies)
            ,
    # Add latency metrics
    if hasattr())))))))))))streaming, "_latency_tracker"):
        results[]],,"avg_token_latency_ms"] = sum())))))))))))streaming._latency_tracker) / len())))))))))))streaming._latency_tracker)
        ,,,,,
            return results


            def test_without_overlap())))))))))))browser_info: Dict[]],,str, Any], precision: str):,,,,,
            """
            Test streaming inference with compute/transfer overlap disabled.
    
    Args:
        browser_info: Browser information dictionary
        precision: Quantization precision
        
    Returns:
        Dictionary with test results
        """
    # Configure with overlap disabled
        config = {}}}}}}}}}}}}}}
        "quantization": precision,
        "optimize_kv_cache": True,
        "latency_optimized": True,
        "adaptive_batch_size": True,
        "browser_info": browser_info,
        # Disable compute/transfer overlap
        "overlap_enabled": False,
        "prefetch_enabled": False
        }
    
    # Create streaming inference handler
        streaming = WebGPUStreamingInference())))))))))))
        model_path="models/llama-7b",
        config=config
        )
    
    # Collect tokens and timing info
        tokens = []],,],,,,,,,,
    
    # Test generation with callback for timing information
    def token_callback())))))))))))token, is_last=False):
        tokens.append())))))))))))token)
    
    # Run generation
        start_time = time.time()))))))))))))
        prompt = "Explain the concept of compute/transfer overlap in the context of streaming inference"
    
        streaming.generate())))))))))))
        prompt=prompt,
        max_tokens=20,
        temperature=0.7,
        callback=token_callback
        )
    
        generation_time = time.time())))))))))))) - start_time
    
    # Get performance stats
        stats = streaming.get_performance_stats()))))))))))))
    
    # Prepare results
        results = {}}}}}}}}}}}}}}
        "tokens_generated": len())))))))))))tokens),
        "generation_time_sec": generation_time,
        "tokens_per_second": len())))))))))))tokens) / generation_time if generation_time > 0 else 0
        }
    
    # Add latency metrics:
    if hasattr())))))))))))streaming, "_latency_tracker"):
        results[]],,"avg_token_latency_ms"] = sum())))))))))))streaming._latency_tracker) / len())))))))))))streaming._latency_tracker)
        ,,,,,
        return results


        def test_token_prediction())))))))))))browser_info: Dict[]],,str, Any], precision: str = "int4"):,,
        """
        Test token prediction functionality in the compute/transfer overlap implementation.
    
    Args:
        browser_info: Browser information dictionary
        precision: Quantization precision
        
    Returns:
        Dictionary with test results
        """
        logger.info())))))))))))f"Testing token prediction with {}}}}}}}}}}}}}}browser_info[]],,'name']} and {}}}}}}}}}}}}}}precision} precision")
        ,,
    # Configure environment based on browser
        os.environ[]],,"WEBGPU_SIMULATION"] = "1"  # Use simulation mode for testing,,
        os.environ[]],,"WEBGPU_AVAILABLE"] = "1"
        ,,
        if browser_info[]],,"name"].lower())))))))))))) == "firefox":,,
        os.environ[]],,"WEBGPU_COMPUTE_SHADERS_ENABLED"] = "1"
        ,,
    # Test with different prompt types to evaluate prediction adaptation
        results = {}}}}}}}}}}}}}}
        "browser": browser_info[]],,"name"],
        "precision": precision,
        "standard_text": test_prediction_with_standard_text())))))))))))browser_info, precision),
        "list_pattern": test_prediction_with_list_pattern())))))))))))browser_info, precision),
        "random_text": test_prediction_with_random_text())))))))))))browser_info, precision)
        }
    
    # Calculate overall token prediction metrics
        prefetch_sizes = []],,],,,,,,,,
        prediction_success_rates = []],,],,,,,,,,
    
    for test_name, test_result in results.items())))))))))))):
        if isinstance())))))))))))test_result, dict):
            if "avg_prefetch_size" in test_result:
                prefetch_sizes.append())))))))))))test_result[]],,"avg_prefetch_size"]),
            if "prediction_success_rate" in test_result:
                prediction_success_rates.append())))))))))))test_result[]],,"prediction_success_rate"])
                ,
    if prefetch_sizes:
        results[]],,"overall_avg_prefetch_size"] = sum())))))))))))prefetch_sizes) / len())))))))))))prefetch_sizes),
        logger.info())))))))))))f"Overall average prefetch size: {}}}}}}}}}}}}}}results[]],,'overall_avg_prefetch_size']:.2f}")
        ,
    if prediction_success_rates:
        results[]],,"overall_prediction_success_rate"] = sum())))))))))))prediction_success_rates) / len())))))))))))prediction_success_rates),
        logger.info())))))))))))f"Overall prediction success rate: {}}}}}}}}}}}}}}results[]],,'overall_prediction_success_rate']*100:.2f}%")
        ,
    # Calculate adaptation metrics
        if ())))))))))))"standard_text" in results and isinstance())))))))))))results[]],,"standard_text"], dict) and:,
            "random_text" in results and isinstance())))))))))))results[]],,"random_text"], dict)):
                ,
                standard_prefetch = results[]],,"standard_text"].get())))))))))))"avg_prefetch_size", 0),
                random_prefetch = results[]],,"random_text"].get())))))))))))"avg_prefetch_size", 0)
                ,
        if standard_prefetch > 0 and random_prefetch > 0:
            # Calculate adaptation ratio ())))))))))))how much did prefetch size adapt between text types)
            results[]],,"prefetch_adaptation_ratio"] = standard_prefetch / random_prefetch,
            logger.info())))))))))))f"Prefetch adaptation ratio ())))))))))))standard/random): {}}}}}}}}}}}}}}results[]],,'prefetch_adaptation_ratio']:.2f}")
            ,
                return results


                def test_prediction_with_standard_text())))))))))))browser_info: Dict[]],,str, Any], precision: str):,,,,,
                """
                Test token prediction with standard text.
    
    Args:
        browser_info: Browser information dictionary
        precision: Quantization precision
        
    Returns:
        Dictionary with test results
        """
    # Configure with prediction enabled
        config = {}}}}}}}}}}}}}}
        "quantization": precision,
        "optimize_kv_cache": True,
        "latency_optimized": True,
        "adaptive_batch_size": True,
        "browser_info": browser_info,
        # Enable compute/transfer overlap with token prediction
        "overlap_enabled": True,
        "prefetch_enabled": True,
        "token_prediction_enabled": True
        }
    
    # Create streaming inference handler
        streaming = WebGPUStreamingInference())))))))))))
        model_path="models/llama-7b",
        config=config
        )
    
    # Collect tokens, prefetch sizes and prediction info
        tokens = []],,],,,,,,,,
        prefetch_sizes = []],,],,,,,,,,
    
    # Test generation with callback for timing information
    def token_callback())))))))))))token, is_last=False):
        tokens.append())))))))))))token)
        
        # Capture prefetch size from optimization config if available:::
        if hasattr())))))))))))streaming, "_last_optimization_config") and "compute_stage" in streaming._last_optimization_config:
            compute_stage = streaming._last_optimization_config[]],,"compute_stage"],,,
            if "prefetch_size" in compute_stage:
                prefetch_sizes.append())))))))))))compute_stage[]],,"prefetch_size"])
                ,,,
    # Run generation
                start_time = time.time()))))))))))))
                prompt = "Explain the concept of token prediction in language models and how it improves performance."
    
                streaming.generate())))))))))))
                prompt=prompt,
                max_tokens=30,
                temperature=0.7,
                callback=token_callback
                )
    
                generation_time = time.time())))))))))))) - start_time
    
    # Extract prediction metrics
                prediction_success_rate = 0.0
    if hasattr())))))))))))streaming, "_prediction_success_rate") and streaming._prediction_success_rate:
        prediction_success_rate = sum())))))))))))streaming._prediction_success_rate) / len())))))))))))streaming._prediction_success_rate)
    
    # Extract token confidence and entropy values if available:::
        confidence_values = []],,],,,,,,,,
        entropy_values = []],,],,,,,,,,
    
    if hasattr())))))))))))streaming, "_token_confidence_history"):
        confidence_values = streaming._token_confidence_history
    
    if hasattr())))))))))))streaming, "_token_entropy_history"):
        entropy_values = streaming._token_entropy_history
    
    # Calculate average prefetch size
        avg_prefetch_size = sum())))))))))))prefetch_sizes) / len())))))))))))prefetch_sizes) if prefetch_sizes else 0
    
    # Prepare results
    results = {}}}}}}}}}}}}}}:::
        "tokens_generated": len())))))))))))tokens),
        "generation_time_sec": generation_time,
        "tokens_per_second": len())))))))))))tokens) / generation_time if generation_time > 0 else 0,::::
            "prefetch_sizes": prefetch_sizes,
            "avg_prefetch_size": avg_prefetch_size,
            "prediction_success_rate": prediction_success_rate,
        "avg_confidence": sum())))))))))))confidence_values) / len())))))))))))confidence_values) if confidence_values else 0,:
            "avg_entropy": sum())))))))))))entropy_values) / len())))))))))))entropy_values) if entropy_values else 0
            }
    
    # Add latency metrics:
    if hasattr())))))))))))streaming, "_latency_tracker"):
        results[]],,"avg_token_latency_ms"] = sum())))))))))))streaming._latency_tracker) / len())))))))))))streaming._latency_tracker)
        ,,,,,
        logger.info())))))))))))f"Standard text - Average prefetch size: {}}}}}}}}}}}}}}avg_prefetch_size:.2f}")
        logger.info())))))))))))f"Standard text - Prediction success rate: {}}}}}}}}}}}}}}prediction_success_rate*100:.2f}%")
    
            return results


            def test_prediction_with_list_pattern())))))))))))browser_info: Dict[]],,str, Any], precision: str):,,,,,
            """
            Test token prediction with highly predictable list pattern text.
    
    Args:
        browser_info: Browser information dictionary
        precision: Quantization precision
        
    Returns:
        Dictionary with test results
        """
    # Configure with prediction enabled
        config = {}}}}}}}}}}}}}}
        "quantization": precision,
        "optimize_kv_cache": True,
        "latency_optimized": True,
        "adaptive_batch_size": True,
        "browser_info": browser_info,
        # Enable compute/transfer overlap with token prediction
        "overlap_enabled": True,
        "prefetch_enabled": True,
        "token_prediction_enabled": True
        }
    
    # Create streaming inference handler
        streaming = WebGPUStreamingInference())))))))))))
        model_path="models/llama-7b",
        config=config
        )
    
    # Collect tokens, prefetch sizes and prediction info
        tokens = []],,],,,,,,,,
        prefetch_sizes = []],,],,,,,,,,
    
    # Test generation with callback for timing information
    def token_callback())))))))))))token, is_last=False):
        tokens.append())))))))))))token)
        
        # Capture prefetch size from optimization config if available:::
        if hasattr())))))))))))streaming, "_last_optimization_config") and "compute_stage" in streaming._last_optimization_config:
            compute_stage = streaming._last_optimization_config[]],,"compute_stage"],,,
            if "prefetch_size" in compute_stage:
                prefetch_sizes.append())))))))))))compute_stage[]],,"prefetch_size"])
                ,,,
    # Run generation with a predictable list prompt
                start_time = time.time()))))))))))))
                prompt = ())))))))))))
                "Here is a numbered list of programming languages:\n"
                "1. Python\n"
                "2. JavaScript\n"
                "3. Java\n"
                "4. C++\n"
                "5. Go\n"
                "6. Rust\n"
                "7. TypeScript\n"
                "8. Swift\n"
                "9. Kotlin\n"
                "10. "
                )
    
                streaming.generate())))))))))))
                prompt=prompt,
                max_tokens=20,
                temperature=0.7,
                callback=token_callback
                )
    
                generation_time = time.time())))))))))))) - start_time
    
    # Extract prediction metrics
                prediction_success_rate = 0.0
    if hasattr())))))))))))streaming, "_prediction_success_rate") and streaming._prediction_success_rate:
        prediction_success_rate = sum())))))))))))streaming._prediction_success_rate) / len())))))))))))streaming._prediction_success_rate)
    
    # Calculate pattern predictability
        pattern_predictability = 0.0
    if hasattr())))))))))))streaming, "_analyze_sentence_patterns"):
        pattern_samples = []],,],,,,,,,,
        # Take multiple samples to get a better average
        for _ in range())))))))))))5):
            pattern_samples.append())))))))))))streaming._analyze_sentence_patterns())))))))))))))
        
        if pattern_samples:
            pattern_predictability = sum())))))))))))pattern_samples) / len())))))))))))pattern_samples)
    
    # Calculate average prefetch size
            avg_prefetch_size = sum())))))))))))prefetch_sizes) / len())))))))))))prefetch_sizes) if prefetch_sizes else 0
    
    # Prepare results
    results = {}}}}}}}}}}}}}}:::
        "tokens_generated": len())))))))))))tokens),
        "generation_time_sec": generation_time,
        "tokens_per_second": len())))))))))))tokens) / generation_time if generation_time > 0 else 0,::::
            "prefetch_sizes": prefetch_sizes,
            "avg_prefetch_size": avg_prefetch_size,
            "prediction_success_rate": prediction_success_rate,
            "pattern_predictability": pattern_predictability
            }
    
    # Add latency metrics
    if hasattr())))))))))))streaming, "_latency_tracker"):
        results[]],,"avg_token_latency_ms"] = sum())))))))))))streaming._latency_tracker) / len())))))))))))streaming._latency_tracker)
        ,,,,,
        logger.info())))))))))))f"List pattern - Average prefetch size: {}}}}}}}}}}}}}}avg_prefetch_size:.2f}")
        logger.info())))))))))))f"List pattern - Pattern predictability: {}}}}}}}}}}}}}}pattern_predictability:.2f}")
        logger.info())))))))))))f"List pattern - Prediction success rate: {}}}}}}}}}}}}}}prediction_success_rate*100:.2f}%")
    
            return results


            def test_prediction_with_random_text())))))))))))browser_info: Dict[]],,str, Any], precision: str):,,,,,
            """
            Test token prediction with unpredictable random text.
    
    Args:
        browser_info: Browser information dictionary
        precision: Quantization precision
        
    Returns:
        Dictionary with test results
        """
    # Configure with prediction enabled
        config = {}}}}}}}}}}}}}}
        "quantization": precision,
        "optimize_kv_cache": True,
        "latency_optimized": True,
        "adaptive_batch_size": True,
        "browser_info": browser_info,
        # Enable compute/transfer overlap with token prediction
        "overlap_enabled": True,
        "prefetch_enabled": True,
        "token_prediction_enabled": True
        }
    
    # Create streaming inference handler
        streaming = WebGPUStreamingInference())))))))))))
        model_path="models/llama-7b",
        config=config
        )
    
    # Collect tokens, prefetch sizes and prediction info
        tokens = []],,],,,,,,,,
        prefetch_sizes = []],,],,,,,,,,
    
    # Test generation with callback for timing information
    def token_callback())))))))))))token, is_last=False):
        tokens.append())))))))))))token)
        
        # Capture prefetch size from optimization config if available:::
        if hasattr())))))))))))streaming, "_last_optimization_config") and "compute_stage" in streaming._last_optimization_config:
            compute_stage = streaming._last_optimization_config[]],,"compute_stage"],,,
            if "prefetch_size" in compute_stage:
                prefetch_sizes.append())))))))))))compute_stage[]],,"prefetch_size"])
                ,,,
    # Run generation with an unpredictable prompt
                start_time = time.time()))))))))))))
                prompt = ())))))))))))
                "Generate a random sequence of words without any patterns or predictable "
                "structure. Include unusual combinations and avoid typical sentence structures."
                )
    
                streaming.generate())))))))))))
                prompt=prompt,
                max_tokens=20,
                temperature=0.9,  # Higher temperature for more randomness
                callback=token_callback
                )
    
                generation_time = time.time())))))))))))) - start_time
    
    # Extract prediction metrics
                prediction_success_rate = 0.0
    if hasattr())))))))))))streaming, "_prediction_success_rate") and streaming._prediction_success_rate:
        prediction_success_rate = sum())))))))))))streaming._prediction_success_rate) / len())))))))))))streaming._prediction_success_rate)
    
    # Calculate pattern predictability
        pattern_predictability = 0.0
    if hasattr())))))))))))streaming, "_analyze_sentence_patterns"):
        pattern_samples = []],,],,,,,,,,
        # Take multiple samples to get a better average
        for _ in range())))))))))))5):
            pattern_samples.append())))))))))))streaming._analyze_sentence_patterns())))))))))))))
        
        if pattern_samples:
            pattern_predictability = sum())))))))))))pattern_samples) / len())))))))))))pattern_samples)
    
    # Calculate average prefetch size
            avg_prefetch_size = sum())))))))))))prefetch_sizes) / len())))))))))))prefetch_sizes) if prefetch_sizes else 0
    
    # Prepare results
    results = {}}}}}}}}}}}}}}:::
        "tokens_generated": len())))))))))))tokens),
        "generation_time_sec": generation_time,
        "tokens_per_second": len())))))))))))tokens) / generation_time if generation_time > 0 else 0,::::
            "prefetch_sizes": prefetch_sizes,
            "avg_prefetch_size": avg_prefetch_size,
            "prediction_success_rate": prediction_success_rate,
            "pattern_predictability": pattern_predictability
            }
    
    # Add latency metrics
    if hasattr())))))))))))streaming, "_latency_tracker"):
        results[]],,"avg_token_latency_ms"] = sum())))))))))))streaming._latency_tracker) / len())))))))))))streaming._latency_tracker)
        ,,,,,
        logger.info())))))))))))f"Random text - Average prefetch size: {}}}}}}}}}}}}}}avg_prefetch_size:.2f}")
        logger.info())))))))))))f"Random text - Pattern predictability: {}}}}}}}}}}}}}}pattern_predictability:.2f}")
        logger.info())))))))))))f"Random text - Prediction success rate: {}}}}}}}}}}}}}}prediction_success_rate*100:.2f}%")
    
            return results


def compare_browsers())))))))))))):
    """
    Compare compute/transfer overlap performance across browsers.
    
    Returns:
        Dictionary with comparison data
        """
    # Test with different browsers
        browsers = []],,
        {}}}}}}}}}}}}}}"name": "chrome", "version": 120},
        {}}}}}}}}}}}}}}"name": "firefox", "version": 115},
        {}}}}}}}}}}}}}}"name": "safari", "version": 17}
        ]
    
        precision = "int4"  # Use 4-bit for comparison
    
        results = {}}}}}}}}}}}}}}}
        comparison = {}}}}}}}}}}}}}}
        "browsers": []],,],,,,,,,,,
        "throughput_improvement": {}}}}}}}}}}}}}}},
        "latency_improvement": {}}}}}}}}}}}}}}},
        "overlap_efficiency": {}}}}}}}}}}}}}}}
        }
    
    for browser in browsers:
        try:
            # Run test for this browser
            browser_results = test_compute_transfer_overlap())))))))))))browser, precision)
            results[]],,browser[]],,"name"]] = browser_results
            
            # Add to comparison data
            comparison[]],,"browsers"].append())))))))))))browser[]],,"name"])
            
            if "throughput_improvement_percent" in browser_results:
                comparison[]],,"throughput_improvement"][]],,browser[]],,"name"]] = browser_results[]],,"throughput_improvement_percent"]
            
            if "latency_improvement_percent" in browser_results:
                comparison[]],,"latency_improvement"][]],,browser[]],,"name"]] = browser_results[]],,"latency_improvement_percent"]
            
            if "with_overlap" in browser_results and "avg_overlap_efficiency" in browser_results[]],,"with_overlap"]:
                comparison[]],,"overlap_efficiency"][]],,browser[]],,"name"]] = browser_results[]],,"with_overlap"][]],,"avg_overlap_efficiency"]
                
        except Exception as e:
            logger.error())))))))))))f"Error testing {}}}}}}}}}}}}}}browser[]],,'name']}: {}}}}}}}}}}}}}}e}")
    
                return comparison


def compare_token_prediction())))))))))))):
    """
    Compare token prediction functionality across browsers.
    
    Returns:
        Dictionary with comparison data
        """
    # Test with different browsers
        browsers = []],,
        {}}}}}}}}}}}}}}"name": "chrome", "version": 120},
        {}}}}}}}}}}}}}}"name": "firefox", "version": 115},
        {}}}}}}}}}}}}}}"name": "safari", "version": 17}
        ]
    
        precision = "int4"  # Use 4-bit for comparison
    
        results = {}}}}}}}}}}}}}}}
        comparison = {}}}}}}}}}}}}}}
        "browsers": []],,],,,,,,,,,
        "avg_prefetch_size": {}}}}}}}}}}}}}}},
        "prediction_success_rate": {}}}}}}}}}}}}}}},
        "prefetch_adaptation_ratio": {}}}}}}}}}}}}}}}
        }
    
    for browser in browsers:
        try:
            # Run token prediction test for this browser
            browser_results = test_token_prediction())))))))))))browser, precision)
            results[]],,browser[]],,"name"]] = browser_results
            
            # Add to comparison data
            comparison[]],,"browsers"].append())))))))))))browser[]],,"name"])
            
            if "overall_avg_prefetch_size" in browser_results:
                comparison[]],,"avg_prefetch_size"][]],,browser[]],,"name"]] = browser_results[]],,"overall_avg_prefetch_size"]
            
            if "overall_prediction_success_rate" in browser_results:
                comparison[]],,"prediction_success_rate"][]],,browser[]],,"name"]] = browser_results[]],,"overall_prediction_success_rate"]
            
            if "prefetch_adaptation_ratio" in browser_results:
                comparison[]],,"prefetch_adaptation_ratio"][]],,browser[]],,"name"]] = browser_results[]],,"prefetch_adaptation_ratio"]
                
        except Exception as e:
            logger.error())))))))))))f"Error testing token prediction for {}}}}}}}}}}}}}}browser[]],,'name']}: {}}}}}}}}}}}}}}e}")
    
                return comparison


def main())))))))))))):
    """Main function to run tests."""
    parser = argparse.ArgumentParser())))))))))))description="Test WebGPU Compute/Transfer Overlap and Token Prediction")
    parser.add_argument())))))))))))"--browser", default="chrome", help="Browser to test ())))))))))))chrome, firefox, safari)")
    parser.add_argument())))))))))))"--precision", default="int4", help="Quantization precision ())))))))))))int2, int3, int4)")
    parser.add_argument())))))))))))"--compare-browsers", action="store_true", help="Compare all browsers")
    parser.add_argument())))))))))))"--test-prediction", action="store_true", help="Test token prediction functionality")
    parser.add_argument())))))))))))"--compare-prediction", action="store_true", help="Compare token prediction across browsers")
    parser.add_argument())))))))))))"--output", help="Output file for results")
    
    args = parser.parse_args()))))))))))))
    
    if args.compare_browsers:
        logger.info())))))))))))"Comparing compute/transfer overlap across browsers")
        comparison = compare_browsers()))))))))))))
        
        logger.info())))))))))))"Browser Comparison Results:")
        
        logger.info())))))))))))"Throughput Improvement:")
        for browser, improvement in comparison[]],,"throughput_improvement"].items())))))))))))):
            logger.info())))))))))))f"  {}}}}}}}}}}}}}}browser}: {}}}}}}}}}}}}}}improvement:.2f}%")
        
            logger.info())))))))))))"Latency Improvement:")
        for browser, improvement in comparison[]],,"latency_improvement"].items())))))))))))):
            logger.info())))))))))))f"  {}}}}}}}}}}}}}}browser}: {}}}}}}}}}}}}}}improvement:.2f}%")
        
            logger.info())))))))))))"Overlap Efficiency:")
        for browser, efficiency in comparison[]],,"overlap_efficiency"].items())))))))))))):
            logger.info())))))))))))f"  {}}}}}}}}}}}}}}browser}: {}}}}}}}}}}}}}}efficiency:.2f}")
        
        # Save results if output specified::::
        if args.output:
            with open())))))))))))args.output, "w") as f:
                json.dump())))))))))))comparison, f, indent=2)
            
                logger.info())))))))))))f"Results saved to {}}}}}}}}}}}}}}args.output}")
    
    elif args.compare_prediction:
        logger.info())))))))))))"Comparing token prediction across browsers")
        comparison = compare_token_prediction()))))))))))))
        
        logger.info())))))))))))"Token Prediction Comparison Results:")
        
        logger.info())))))))))))"Average Prefetch Size:")
        for browser, size in comparison[]],,"avg_prefetch_size"].items())))))))))))):
            logger.info())))))))))))f"  {}}}}}}}}}}}}}}browser}: {}}}}}}}}}}}}}}size:.2f}")
        
            logger.info())))))))))))"Prediction Success Rate:")
        for browser, rate in comparison[]],,"prediction_success_rate"].items())))))))))))):
            logger.info())))))))))))f"  {}}}}}}}}}}}}}}browser}: {}}}}}}}}}}}}}}rate*100:.2f}%")
        
            logger.info())))))))))))"Prefetch Adaptation Ratio ())))))))))))standard/random):")
        for browser, ratio in comparison[]],,"prefetch_adaptation_ratio"].items())))))))))))):
            logger.info())))))))))))f"  {}}}}}}}}}}}}}}browser}: {}}}}}}}}}}}}}}ratio:.2f}")
        
        # Save results if output specified::::
        if args.output:
            with open())))))))))))args.output, "w") as f:
                json.dump())))))))))))comparison, f, indent=2)
            
                logger.info())))))))))))f"Results saved to {}}}}}}}}}}}}}}args.output}")
    
    elif args.test_prediction:
        # Test token prediction with specific browser
        browser_info = {}}}}}}}}}}}}}}"name": args.browser, "version": 120}
        results = test_token_prediction())))))))))))browser_info, args.precision)
        
        logger.info())))))))))))"Token Prediction Test Results:")
        logger.info())))))))))))f"  Browser: {}}}}}}}}}}}}}}results[]],,'browser']}")
        logger.info())))))))))))f"  Precision: {}}}}}}}}}}}}}}results[]],,'precision']}")
        
        if "overall_avg_prefetch_size" in results:
            logger.info())))))))))))f"  Overall average prefetch size: {}}}}}}}}}}}}}}results[]],,'overall_avg_prefetch_size']:.2f}")
            ,
        if "overall_prediction_success_rate" in results:
            logger.info())))))))))))f"  Overall prediction success rate: {}}}}}}}}}}}}}}results[]],,'overall_prediction_success_rate']*100:.2f}%")
            ,
        if "prefetch_adaptation_ratio" in results:
            logger.info())))))))))))f"  Prefetch adaptation ratio: {}}}}}}}}}}}}}}results[]],,'prefetch_adaptation_ratio']:.2f}")
            ,
        # Save results if output specified::::
        if args.output:
            with open())))))))))))args.output, "w") as f:
                json.dump())))))))))))results, f, indent=2)
            
                logger.info())))))))))))f"Results saved to {}}}}}}}}}}}}}}args.output}")
    
    else:
        # Test compute/transfer overlap with specific browser
        browser_info = {}}}}}}}}}}}}}}"name": args.browser, "version": 120}
        results = test_compute_transfer_overlap())))))))))))browser_info, args.precision)
        
        logger.info())))))))))))"Test Results:")
        logger.info())))))))))))f"  Browser: {}}}}}}}}}}}}}}results[]],,'browser']}")
        logger.info())))))))))))f"  Precision: {}}}}}}}}}}}}}}results[]],,'precision']}")
        
        if "throughput_improvement_percent" in results:
            logger.info())))))))))))f"  Throughput improvement: {}}}}}}}}}}}}}}results[]],,'throughput_improvement_percent']:.2f}%")
        
        if "latency_improvement_percent" in results:
            logger.info())))))))))))f"  Latency improvement: {}}}}}}}}}}}}}}results[]],,'latency_improvement_percent']:.2f}%")
        
        # Save results if output specified::::
        if args.output:
            with open())))))))))))args.output, "w") as f:
                json.dump())))))))))))results, f, indent=2)
            
                logger.info())))))))))))f"Results saved to {}}}}}}}}}}}}}}args.output}")


if __name__ == "__main__":
    main()))))))))))))