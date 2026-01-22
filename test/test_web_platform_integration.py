#!/usr/bin/env python3
"""
Test script for validating web platform integration.

This script tests the integration of WebNN and WebGPU platforms with the
ResourcePool and model generation system, verifying proper implementation
type reporting and simulation behavior.

Usage:
    python test_web_platform_integration.py --platform webnn
    python test_web_platform_integration.py --platform webgpu
    python test_web_platform_integration.py --platform both --verbose
    """

    import os
    import sys
    import json
    import time
    import argparse
    import logging
    import datetime
    import asyncio
    from pathlib import Path
    from typing import Dict, List, Any, Optional, Tuple, Callable, Union

# Configure logging
    logging.basicConfig()))))))))))))))))))))
    level=logging.INFO,
    format='%()))))))))))))))))))))asctime)s - %()))))))))))))))))))))levelname)s - %()))))))))))))))))))))message)s'
    )
    logger = logging.getLogger()))))))))))))))))))))"web_platform_test")

# Constants for WebNN and WebGPU implementation types
    WEBNN_IMPL_TYPE = "REAL_WEBNN"
    WEBGPU_IMPL_TYPE = "REAL_WEBGPU"

# Test models for different modalities
    TEST_MODELS = {}}}}}}}}}}}}}}}}}}}
    "text": "bert-base-uncased",
    "vision": "google/vit-base-patch16-224",
    "audio": "openai/whisper-tiny",
    "multimodal": "openai/clip-vit-base-patch32"
    }

def setup_web_environment()))))))))))))))))))))platform: str, verbose: bool = False) -> bool:
    """
    Set up the environment variables for web platform testing.
    
    Args:
        platform: Which platform to enable ()))))))))))))))))))))'webnn', 'webgpu', or 'both')
        verbose: Whether to print verbose output
        
    Returns:
        True if successful, False otherwise
        """
    # Check for the helper script
    helper_script = "./run_web_platform_tests.sh":
    if not Path()))))))))))))))))))))helper_script).exists()))))))))))))))))))))):
        helper_script = "test/run_web_platform_tests.sh"
        if not Path()))))))))))))))))))))helper_script).exists()))))))))))))))))))))):
            logger.error()))))))))))))))))))))f"Helper script not found: {}}}}}}}}}}}}}}}}}}}helper_script}")
            logger.error()))))))))))))))))))))"Please run this script from the project root directory")
        return False
    
    # Set appropriate environment variables based on platform
    if platform.lower()))))))))))))))))))))) == "webnn":
        os.environ["WEBNN_ENABLED"] = "1",,,
        os.environ["WEBNN_SIMULATION"] = "1",,,
        os.environ["WEBNN_AVAILABLE"] = "1",,
        if verbose:
            logger.info()))))))))))))))))))))"WebNN simulation enabled")
    elif platform.lower()))))))))))))))))))))) == "webgpu":
        os.environ["WEBGPU_ENABLED"] = "1",,,,,
        os.environ["WEBGPU_SIMULATION"] = "1",,,, ,
        os.environ["WEBGPU_AVAILABLE"] = "1",,,
        if verbose:
            logger.info()))))))))))))))))))))"WebGPU simulation enabled")
    elif platform.lower()))))))))))))))))))))) == "both":
        os.environ["WEBNN_ENABLED"] = "1",,,
        os.environ["WEBNN_SIMULATION"] = "1",,,
        os.environ["WEBNN_AVAILABLE"] = "1",,
        os.environ["WEBGPU_ENABLED"] = "1",,,,,
        os.environ["WEBGPU_SIMULATION"] = "1",,,,
        os.environ["WEBGPU_AVAILABLE"] = "1",,,
        if verbose:
            logger.info()))))))))))))))))))))"Both WebNN and WebGPU simulation enabled")
    else:
        logger.error()))))))))))))))))))))f"Unknown platform: {}}}}}}}}}}}}}}}}}}}platform}")
            return False
    
    # Enable shader precompilation and compute shaders for WebGPU
            if platform.lower()))))))))))))))))))))) in ["webgpu", "both"]:,,
            os.environ["WEBGPU_SHADER_PRECOMPILE_ENABLED"] = "1",
            os.environ["WEBGPU_COMPUTE_SHADERS_ENABLED"] = "1",
        if verbose:
            logger.info()))))))))))))))))))))"WebGPU shader precompilation and compute shaders enabled")
    
    # Enable parallel loading for both platforms if applicable:
            if platform.lower()))))))))))))))))))))) in ["webnn", "both"]:,
            os.environ["WEBNN_PARALLEL_LOADING_ENABLED"] = "1",
        if verbose:
            logger.info()))))))))))))))))))))"WebNN parallel loading enabled")
            if platform.lower()))))))))))))))))))))) in ["webgpu", "both"]:,,
            os.environ["WEBGPU_PARALLEL_LOADING_ENABLED"] = "1",
        if verbose:
            logger.info()))))))))))))))))))))"WebGPU parallel loading enabled")
    
            return True

            def test_web_platform()))))))))))))))))))))platform: str, model_modality: str = "text", verbose: bool = False,
            model_size: str = "base", performance_iterations: int = 1) -> Dict[str, Any]:,,,,,
            """
            Test the web platform integration for a specific model modality.
    
    Args:
        platform: Which platform to test ()))))))))))))))))))))'webnn' or 'webgpu')
        model_modality: Which model modality to test ()))))))))))))))))))))'text', 'vision', 'audio', 'multimodal')
        verbose: Whether to print verbose output
        model_size: Model size to test ()))))))))))))))))))))'tiny', 'small', 'base', 'large')
        performance_iterations: Number of inference iterations for performance measurement
        
    Returns:
        Dictionary with test results
        """
    # Get model name for the modality based on size
        model_name = TEST_MODELS.get()))))))))))))))))))))model_modality, TEST_MODELS["text"])
        ,,
    # Adjust model name based on size if smaller variants are requested:
    if model_size == "tiny" and model_modality == "text":
        model_name = "prajjwal1/bert-tiny" if model_modality == "text" else model_name:
    elif model_size == "small":
        # Use smaller model variants
        if model_modality == "text":
            model_name = "prajjwal1/bert-mini"
        elif model_modality == "vision":
            model_name = "facebook/deit-tiny-patch16-224"
        elif model_modality == "audio":
            model_name = "openai/whisper-tiny"
    
    if verbose:
        logger.info()))))))))))))))))))))f"Testing {}}}}}}}}}}}}}}}}}}}platform} with {}}}}}}}}}}}}}}}}}}}model_modality} model '{}}}}}}}}}}}}}}}}}}}model_name}' ()))))))))))))))))))))size: {}}}}}}}}}}}}}}}}}}}model_size})")
    
    # Import the fixed_web_platform module ()))))))))))))))))))))from current directory)
    try:
        # Try to import fixed_web_platform from the current directory
        sys.path.append()))))))))))))))))))))'.')
        # Import traditional platform handler
        from fixed_web_platform.web_platform_handler import ()))))))))))))))))))))
        process_for_web, init_webnn, init_webgpu, create_mock_processors
        )
        
        # Try to import new unified framework components
        try:
            from fixed_web_platform.unified_web_framework import WebPlatformAccelerator
            from fixed_web_platform.webgpu_streaming_inference import WebGPUStreamingInference
            has_unified_framework = True
        except ImportError:
            has_unified_framework = False
            
        if verbose:
            logger.info()))))))))))))))))))))"Successfully imported web platform handler from fixed_web_platform")
            if has_unified_framework:
                logger.info()))))))))))))))))))))"Successfully imported unified framework components")
    except ImportError:
        # Try to import from the test directory
        try:
            sys.path.append()))))))))))))))))))))'test')
            # Import traditional platform handler
            from fixed_web_platform.web_platform_handler import ()))))))))))))))))))))
            process_for_web, init_webnn, init_webgpu, create_mock_processors
            )
            
            # Try to import new unified framework components
            try:
                from fixed_web_platform.unified_web_framework import WebPlatformAccelerator
                from fixed_web_platform.webgpu_streaming_inference import WebGPUStreamingInference
                has_unified_framework = True
            except ImportError:
                has_unified_framework = False
                
            if verbose:
                logger.info()))))))))))))))))))))"Successfully imported web platform handler from test/fixed_web_platform")
                if has_unified_framework:
                    logger.info()))))))))))))))))))))"Successfully imported unified framework components")
        except ImportError:
            logger.error()))))))))))))))))))))"Failed to import web platform handler from fixed_web_platform")
                    return {}}}}}}}}}}}}}}}}}}}
                    "success": False,
                    "error": "Failed to import web platform handler from fixed_web_platform",
                    "platform": platform,
                    "model_modality": model_modality
                    }
    
    # Create a test class to use the web platform handlers
    class TestModelHandler:
        def __init__()))))))))))))))))))))self):
            self.model_name = model_name
            self.mode = model_modality
            self.device = platform.lower())))))))))))))))))))))
            self.processors = create_mock_processors())))))))))))))))))))))
            
        def test_platform()))))))))))))))))))))self):
            # Initialize the platform-specific handler
            if platform.lower()))))))))))))))))))))) == "webnn":
                result = init_webnn()))))))))))))))))))))
                self,
                model_name=self.model_name,
                model_type=self.mode,
                device=self.device,
                web_api_mode="simulation",
                create_mock_processor=self.processors["image_processor"] ,,
                if self.mode == "vision" else None
                )::
            elif platform.lower()))))))))))))))))))))) == "webgpu":
                result = init_webgpu()))))))))))))))))))))
                self,
                model_name=self.model_name,
                model_type=self.mode,
                device=self.device,
                web_api_mode="simulation",
                create_mock_processor=self.processors["image_processor"] ,,
                if self.mode == "vision" else None
                )::
            else:
                    return {}}}}}}}}}}}}}}}}}}}
                    "success": False,
                    "error": f"Unknown platform: {}}}}}}}}}}}}}}}}}}}platform}"
                    }
            
            # Verify the result
            if not result or not isinstance()))))))))))))))))))))result, dict):
                    return {}}}}}}}}}}}}}}}}}}}
                    "success": False,
                    "error": f"Failed to initialize {}}}}}}}}}}}}}}}}}}}platform} for {}}}}}}}}}}}}}}}}}}}model_name}"
                    }
            
            # Extract key components
                    endpoint = result.get()))))))))))))))))))))"endpoint")
                    processor = result.get()))))))))))))))))))))"processor")
                    batch_supported = result.get()))))))))))))))))))))"batch_supported", False)
                    implementation_type = result.get()))))))))))))))))))))"implementation_type", "UNKNOWN")
            
            if not endpoint:
                    return {}}}}}}}}}}}}}}}}}}}
                    "success": False,
                    "error": f"No endpoint returned for {}}}}}}}}}}}}}}}}}}}platform}"
                    }
            
            # Create test input based on modality
            if self.mode == "text":
                test_input = "This is a test input for text models"
            elif self.mode == "vision":
                test_input = "test.jpg"
            elif self.mode == "audio":
                test_input = "test.mp3"
            elif self.mode == "multimodal":
                test_input = {}}}}}}}}}}}}}}}}}}}"image": "test.jpg", "text": "What is in this image?"}
            else:
                test_input = "Generic test input"
            
            # Process input for web platform
                processed_input = process_for_web()))))))))))))))))))))self.mode, test_input, batch_supported)
            
            # Run inference with performance measurement
            try:
                # Initial inference to warm up
                inference_result = endpoint()))))))))))))))))))))processed_input)
                
                # Run multiple iterations for performance testing
                inference_times = [],,,,,,,,,,,,
                total_inference_time = 0
                iterations = performance_iterations if performance_iterations > 0 else 1
                :
                for i in range()))))))))))))))))))))iterations):
                    start_time = time.time())))))))))))))))))))))
                    inference_result = endpoint()))))))))))))))))))))processed_input)
                    end_time = time.time())))))))))))))))))))))
                    elapsed_time = ()))))))))))))))))))))end_time - start_time) * 1000  # Convert to ms
                    inference_times.append()))))))))))))))))))))elapsed_time)
                    total_inference_time += elapsed_time
                
                # Calculate performance metrics
                    avg_inference_time = total_inference_time / iterations if iterations > 0 else 0
                    min_inference_time = min()))))))))))))))))))))inference_times) if inference_times else 0
                    max_inference_time = max()))))))))))))))))))))inference_times) if inference_times else 0
                    std_dev = ()))))))))))))))))))))
                    ()))))))))))))))))))))sum()))))))))))))))))))))()))))))))))))))))))))t - avg_inference_time) ** 2 for t in inference_times) / iterations) ** 0.5 
                    if iterations > 1 else 0
                    )
                
                # Extract metrics from result if available:::
                if isinstance()))))))))))))))))))))inference_result, dict) and "performance_metrics" in inference_result:
                    result_metrics = inference_result["performance_metrics"],
                else:
                    result_metrics = {}}}}}}}}}}}}}}}}}}}}
                
                # Check implementation type in the result
                    result_impl_type = ()))))))))))))))))))))
                    inference_result.get()))))))))))))))))))))"implementation_type") 
                    if isinstance()))))))))))))))))))))inference_result, dict) else None
                    )
                
                # Verify implementation type from both sources
                    expected_impl_type = ()))))))))))))))))))))
                    WEBNN_IMPL_TYPE if platform.lower()))))))))))))))))))))) == "webnn" else WEBGPU_IMPL_TYPE
                    )
                
                # Create enhanced result with performance metrics
                return {}}}}}}}}}}}}}}}}}}}:
                    "success": True,
                    "platform": platform,
                    "model_name": self.model_name,
                    "model_modality": self.mode,
                    "batch_supported": batch_supported,
                    "initialization_type": implementation_type,
                    "result_type": result_impl_type,
                    "expected_type": expected_impl_type,
                    "type_match": ()))))))))))))))))))))
                    result_impl_type == "SIMULATION" or
                    result_impl_type == expected_impl_type
                    ),
                    "has_metrics": ()))))))))))))))))))))
                    "performance_metrics" in inference_result
                    if isinstance()))))))))))))))))))))inference_result, dict) else False
                    ),:
                        "performance": {}}}}}}}}}}}}}}}}}}}
                        "iterations": iterations,
                        "avg_inference_time_ms": avg_inference_time,
                        "min_inference_time_ms": min_inference_time,
                        "max_inference_time_ms": max_inference_time,
                        "std_dev_ms": std_dev,
                        "reported_metrics": result_metrics
                        }
                        }
            except Exception as e:
                        return {}}}}}}}}}}}}}}}}}}}
                        "success": False,
                        "error": f"Error during inference: {}}}}}}}}}}}}}}}}}}}str()))))))))))))))))))))e)}",
                        "platform": platform,
                        "model_name": self.model_name,
                        "model_modality": self.mode
                        }
    
    # Run the test
                        test_handler = TestModelHandler())))))))))))))))))))))
                    return test_handler.test_platform())))))))))))))))))))))

                    def print_test_results()))))))))))))))))))))results: Dict[str, Dict[str, Dict[str, Any]]], verbose: bool = False) -> bool:,
                    """
                    Print test results and return overall success status.
    
    Args:
        results: Dictionary with test results
        verbose: Whether to print verbose output
        
    Returns:
        True if all tests passed, False otherwise
        """
        all_success = True
    
    # Print header
        print()))))))))))))))))))))"\nWeb Platform Integration Test Results")
        print()))))))))))))))))))))"===================================\n")
    
    # Process and print results by platform and modality:
    for platform, modality_results in results.items()))))))))))))))))))))):
        print()))))))))))))))))))))f"\n{}}}}}}}}}}}}}}}}}}}platform.upper())))))))))))))))))))))} Platform:")
        print()))))))))))))))))))))"-" * ()))))))))))))))))))))len()))))))))))))))))))))platform) + 10))
        
        platform_success = True
        
        for modality, result in modality_results.items()))))))))))))))))))))):
            success = result.get()))))))))))))))))))))"success", False)
            platform_success = platform_success and success
            
            if success:
                model_name = result.get()))))))))))))))))))))"model_name", "Unknown")
                init_type = result.get()))))))))))))))))))))"initialization_type", "Unknown")
                result_type = result.get()))))))))))))))))))))"result_type", "Unknown")
                expected_type = result.get()))))))))))))))))))))"expected_type", "Unknown")
                type_match = result.get()))))))))))))))))))))"type_match", False)
                has_metrics = result.get()))))))))))))))))))))"has_metrics", False)
                
                status = "✅ PASS" if type_match else "❌ FAIL":
                    print()))))))))))))))))))))f"  {}}}}}}}}}}}}}}}}}}}modality.capitalize())))))))))))))))))))))} ())))))))))))))))))))){}}}}}}}}}}}}}}}}}}}model_name}): {}}}}}}}}}}}}}}}}}}}status}")
                
                # Extract performance metrics if available::
                    performance = result.get()))))))))))))))))))))"performance", {}}}}}}}}}}}}}}}}}}}})
                :
                if verbose or not type_match:
                    print()))))))))))))))))))))f"    - Init Type: {}}}}}}}}}}}}}}}}}}}init_type}")
                    print()))))))))))))))))))))f"    - Result Type: {}}}}}}}}}}}}}}}}}}}result_type}")
                    print()))))))))))))))))))))f"    - Expected: {}}}}}}}}}}}}}}}}}}}expected_type}")
                    print()))))))))))))))))))))f"    - Has Metrics: {}}}}}}}}}}}}}}}}}}}'Yes' if has_metrics else 'No'}")
                    
                    # Print performance information if available:::
                    if performance:
                        avg_time = performance.get()))))))))))))))))))))"avg_inference_time_ms", 0)
                        min_time = performance.get()))))))))))))))))))))"min_inference_time_ms", 0)
                        max_time = performance.get()))))))))))))))))))))"max_inference_time_ms", 0)
                        iterations = performance.get()))))))))))))))))))))"iterations", 0)
                        
                        print()))))))))))))))))))))f"    - Performance ())))))))))))))))))))){}}}}}}}}}}}}}}}}}}}iterations} iterations):")
                        print()))))))))))))))))))))f"      * Average: {}}}}}}}}}}}}}}}}}}}avg_time:.2f} ms")
                        print()))))))))))))))))))))f"      * Min: {}}}}}}}}}}}}}}}}}}}min_time:.2f} ms")
                        print()))))))))))))))))))))f"      * Max: {}}}}}}}}}}}}}}}}}}}max_time:.2f} ms")
                        
                        # Print advanced metrics if available::
                        reported_metrics = performance.get()))))))))))))))))))))"reported_metrics", {}}}}}}}}}}}}}}}}}}}}):
                        if reported_metrics:
                            print()))))))))))))))))))))f"    - Advanced Metrics:")
                            for key, value in reported_metrics.items()))))))))))))))))))))):
                                if isinstance()))))))))))))))))))))value, ()))))))))))))))))))))int, float)):
                                    print()))))))))))))))))))))f"      * {}}}}}}}}}}}}}}}}}}}key}: {}}}}}}}}}}}}}}}}}}}value}")
                                elif isinstance()))))))))))))))))))))value, dict):
                                    print()))))))))))))))))))))f"      * {}}}}}}}}}}}}}}}}}}}key}:")
                                    for subkey, subvalue in value.items()))))))))))))))))))))):
                                        print()))))))))))))))))))))f"        - {}}}}}}}}}}}}}}}}}}}subkey}: {}}}}}}}}}}}}}}}}}}}subvalue}")
                
                                        all_success = all_success and type_match
            else:
                error = result.get()))))))))))))))))))))"error", "Unknown error")
                print()))))))))))))))))))))f"  {}}}}}}}}}}}}}}}}}}}modality.capitalize())))))))))))))))))))))}: ❌ FAIL - {}}}}}}}}}}}}}}}}}}}error}")
                platform_success = False
                all_success = False
        
                print()))))))))))))))))))))f"\n  {}}}}}}}}}}}}}}}}}}}platform.upper())))))))))))))))))))))} Summary: {}}}}}}}}}}}}}}}}}}}'✅ PASS' if platform_success else '❌ FAIL'}")
    
    # Print overall summary:
                print()))))))))))))))))))))"\nOverall Test Result:", "✅ PASS" if all_success else "❌ FAIL")
    
                                        return all_success
:
    def run_tests()))))))))))))))))))))platforms: List[str], modalities: List[str], verbose: bool = False,
    model_size: str = "base", performance_iterations: int = 1) -> Dict[str, Dict[str, Dict[str, Any]]]:,
    """
    Run tests for specified platforms and modalities.
    
    Args:
        platforms: List of platforms to test
        modalities: List of modalities to test
        verbose: Whether to print verbose output
        model_size: Size of models to test ()))))))))))))))))))))'tiny', 'small', 'base', 'large')
        performance_iterations: Number of iterations for performance testing
        
    Returns:
        Dictionary with test results
        """
        results = {}}}}}}}}}}}}}}}}}}}}
    
    for platform in platforms:
        # Set up environment for this platform
        if not setup_web_environment()))))))))))))))))))))platform, verbose):
            logger.error()))))))))))))))))))))f"Failed to set up environment for {}}}}}}}}}}}}}}}}}}}platform}")
        continue
        
        platform_results = {}}}}}}}}}}}}}}}}}}}}
        
        for modality in modalities:
            if verbose:
                logger.info()))))))))))))))))))))f"Testing {}}}}}}}}}}}}}}}}}}}platform} platform with {}}}}}}}}}}}}}}}}}}}modality} modality")
            
            # Run the test with size and performance parameters
                result = test_web_platform()))))))))))))))))))))
                platform=platform, 
                model_modality=modality, 
                verbose=verbose,
                model_size=model_size,
                performance_iterations=performance_iterations
                )
                platform_results[modality] = result,
                ,
                results[platform] = platform_results,
                ,
            return results

            def test_unified_framework()))))))))))))))))))))platform: str, model_modality: str, verbose: bool = False) -> Dict[str, Any]:,,,,,
            """
            Test the unified web framework implementation.
    
    Args:
        platform: Which platform to test ()))))))))))))))))))))'webnn' or 'webgpu')
        model_modality: Which model modality to test ()))))))))))))))))))))'text', 'vision', 'audio', 'multimodal')
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary with test results
        """
    # Import unified framework components
    try:
        sys.path.append()))))))))))))))))))))'.')
        from fixed_web_platform.unified_web_framework import WebPlatformAccelerator
        
        if verbose:
            logger.info()))))))))))))))))))))"Successfully imported unified framework from fixed_web_platform")
            
    except ImportError:
        try:
            sys.path.append()))))))))))))))))))))'test')
            from fixed_web_platform.unified_web_framework import WebPlatformAccelerator
            
            if verbose:
                logger.info()))))))))))))))))))))"Successfully imported unified framework from test/fixed_web_platform")
                
        except ImportError:
            logger.error()))))))))))))))))))))"Failed to import unified framework from fixed_web_platform")
                return {}}}}}}}}}}}}}}}}}}}
                "success": False,
                "error": "Failed to import unified framework",
                "platform": platform,
                "model_modality": model_modality
                }
            
    # Get model name for the modality
                model_name = TEST_MODELS.get()))))))))))))))))))))model_modality, TEST_MODELS["text"])
                ,,
    # Set environment for platform
    if platform.lower()))))))))))))))))))))) == "webgpu":
        os.environ["WEBGPU_ENABLED"] = "1",,,,,
        os.environ["WEBGPU_SIMULATION"] = "1",,,,
        os.environ["WEBGPU_AVAILABLE"] = "1",,,
    elif platform.lower()))))))))))))))))))))) == "webnn":
        os.environ["WEBNN_ENABLED"] = "1",,,
        os.environ["WEBNN_SIMULATION"] = "1",,,
        os.environ["WEBNN_AVAILABLE"] = "1",,
    
    try:
        # Create accelerator with auto-detection
        accelerator = WebPlatformAccelerator()))))))))))))))))))))
        model_path=model_name,
        model_type=model_modality,
        auto_detect=True
        )
        
        # Get configuration
        config = accelerator.get_config())))))))))))))))))))))
        
        # Create endpoint
        endpoint = accelerator.create_endpoint())))))))))))))))))))))
        
        # Create test input based on modality
        if model_modality == "text":
            test_input = "This is a test input for text models"
        elif model_modality == "vision":
            test_input = {}}}}}}}}}}}}}}}}}}}"image": "test.jpg"}
        elif model_modality == "audio":
            test_input = {}}}}}}}}}}}}}}}}}}}"audio": "test.mp3"}
        elif model_modality == "multimodal":
            test_input = {}}}}}}}}}}}}}}}}}}}"image": "test.jpg", "text": "What is in this image?"}
        else:
            test_input = "Generic test input"
        
        # Run inference with performance measurement
            start_time = time.time())))))))))))))))))))))
            inference_result = endpoint()))))))))))))))))))))test_input)
            inference_time = ()))))))))))))))))))))time.time()))))))))))))))))))))) - start_time) * 1000  # ms
        
        # Get performance metrics
            metrics = accelerator.get_performance_metrics())))))))))))))))))))))
        
        # Get feature usage
            feature_usage = accelerator.get_feature_usage())))))))))))))))))))))
        
        # Check if appropriate feature is in use
            expected_feature = "4bit_quantization" if config.get()))))))))))))))))))))"quantization", 16) <= 4 else None
        
        return {}}}}}}}}}}}}}}}}}}}:
            "success": True,
            "platform": platform,
            "model_name": model_name,
            "model_modality": model_modality,
            "config": config,
            "feature_usage": feature_usage,
            "has_expected_feature": expected_feature in feature_usage if expected_feature else True,:
                "metrics": metrics,
                "inference_time_ms": inference_time
                }
    except Exception as e:
                return {}}}}}}}}}}}}}}}}}}}
                "success": False,
                "error": f"Error with unified framework: {}}}}}}}}}}}}}}}}}}}str()))))))))))))))))))))e)}",
                "platform": platform,
                "model_modality": model_modality
                }

                def test_streaming_inference()))))))))))))))))))))verbose: bool = False) -> Dict[str, Any]:,,,,,
                """
                Test streaming inference implementation.
    
    Args:
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary with test results
        """
    # Import streaming inference component
    try:
        sys.path.append()))))))))))))))))))))'.')
        from fixed_web_platform.webgpu_streaming_inference import ()))))))))))))))))))))
        WebGPUStreamingInference,
        optimize_for_streaming
        )
        
        if verbose:
            logger.info()))))))))))))))))))))"Successfully imported streaming inference from fixed_web_platform")
            
    except ImportError:
        try:
            sys.path.append()))))))))))))))))))))'test')
            from fixed_web_platform.webgpu_streaming_inference import ()))))))))))))))))))))
            WebGPUStreamingInference,
            optimize_for_streaming
            )
            
            if verbose:
                logger.info()))))))))))))))))))))"Successfully imported streaming inference from test/fixed_web_platform")
                
        except ImportError:
            logger.error()))))))))))))))))))))"Failed to import streaming inference from fixed_web_platform")
                return {}}}}}}}}}}}}}}}}}}}
                "success": False,
                "error": "Failed to import streaming inference"
                }
    
    # Enable WebGPU simulation
                os.environ["WEBGPU_ENABLED"] = "1",,,,,
                os.environ["WEBGPU_SIMULATION"] = "1",,,,
                os.environ["WEBGPU_AVAILABLE"] = "1",,,
    
    try:
        # Configure for streaming
        config = optimize_for_streaming())))))))))))))))))))){}}}}}}}}}}}}}}}}}}}
        "quantization": "int4",
        "latency_optimized": True,
        "adaptive_batch_size": True
        })
        
        # Create streaming handler
        streaming_handler = WebGPUStreamingInference()))))))))))))))))))))
        model_path=TEST_MODELS["text"],
        config=config
        )
        
        # Test with callback
        tokens_received = [],,,,,,,,,,,,
        
        def token_callback()))))))))))))))))))))token, is_last=False):
            tokens_received.append()))))))))))))))))))))token)
        
        # Run streaming generation
            prompt = "This is a test prompt for streaming inference"
        
        # Measure generation time
            start_time = time.time())))))))))))))))))))))
            result = streaming_handler.generate()))))))))))))))))))))
            prompt,
            max_tokens=20,
            temperature=0.7,
            callback=token_callback
            )
            generation_time = time.time()))))))))))))))))))))) - start_time
        
        # Get performance stats
            stats = streaming_handler.get_performance_stats())))))))))))))))))))))
        
        # Verify results
            has_batch_size_history = "batch_size_history" in stats and len()))))))))))))))))))))stats["batch_size_history"]) > 0
            ,
        return {}}}}}}}}}}}}}}}}}}}
        "success": True,
        "tokens_generated": stats.get()))))))))))))))))))))"tokens_generated", 0),
        "tokens_per_second": stats.get()))))))))))))))))))))"tokens_per_second", 0),
        "tokens_received": len()))))))))))))))))))))tokens_received),
        "generation_time_sec": generation_time,
        "batch_size_history": stats.get()))))))))))))))))))))"batch_size_history", [],,,,,,,,,,,,),
        "has_batch_size_adaptation": has_batch_size_history,
        "adaptive_batch_size_enabled": config.get()))))))))))))))))))))"adaptive_batch_size", False),
        "result_length": len()))))))))))))))))))))result) if result else 0
        }:
    except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": f"Error with streaming inference: {}}}}}}}}}}}}}}}}}}}str()))))))))))))))))))))e)}"
            }

            async def test_async_streaming_inference()))))))))))))))))))))verbose: bool = False) -> Dict[str, Any]:,,,,,
            """
            Test async streaming inference implementation.
    
    Args:
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary with test results
        """
    # Import streaming inference component
    try:
        sys.path.append()))))))))))))))))))))'.')
        from fixed_web_platform.webgpu_streaming_inference import ()))))))))))))))))))))
        WebGPUStreamingInference,
        optimize_for_streaming
        )
        
        if verbose:
            logger.info()))))))))))))))))))))"Successfully imported streaming inference from fixed_web_platform")
            
    except ImportError:
        try:
            sys.path.append()))))))))))))))))))))'test')
            from fixed_web_platform.webgpu_streaming_inference import ()))))))))))))))))))))
            WebGPUStreamingInference,
            optimize_for_streaming
            )
            
            if verbose:
                logger.info()))))))))))))))))))))"Successfully imported streaming inference from test/fixed_web_platform")
                
        except ImportError:
            logger.error()))))))))))))))))))))"Failed to import streaming inference from fixed_web_platform")
                return {}}}}}}}}}}}}}}}}}}}
                "success": False,
                "error": "Failed to import streaming inference"
                }
    
    # Enable WebGPU simulation
                os.environ["WEBGPU_ENABLED"] = "1",,,,,
                os.environ["WEBGPU_SIMULATION"] = "1",,,,
                os.environ["WEBGPU_AVAILABLE"] = "1",,,
    
    try:
        # Configure for streaming with enhanced latency options
        config = optimize_for_streaming())))))))))))))))))))){}}}}}}}}}}}}}}}}}}}
        "quantization": "int4",
        "latency_optimized": True,
        "adaptive_batch_size": True,
        "ultra_low_latency": True,   # New option for extreme low latency
        "stream_buffer_size": 1      # Smallest buffer for lowest latency
        })
        
        # Create streaming handler
        streaming_handler = WebGPUStreamingInference()))))))))))))))))))))
        model_path=TEST_MODELS["text"],
        config=config
        )
        
        # Run async streaming generation
        prompt = "This is a test prompt for async streaming inference with enhanced latency optimization"
        
        # Measure generation time
        start_time = time.time())))))))))))))))))))))
        result = await streaming_handler.generate_async()))))))))))))))))))))
        prompt,
        max_tokens=20,
        temperature=0.7
        )
        generation_time = time.time()))))))))))))))))))))) - start_time
        
        # Get performance stats
        stats = streaming_handler.get_performance_stats())))))))))))))))))))))
        
        # Calculate per-token latency metrics
        tokens_generated = stats.get()))))))))))))))))))))"tokens_generated", 0)
        avg_token_latency = generation_time * 1000 / tokens_generated if tokens_generated > 0 else 0
        
        # Test if adaptive batch sizing worked
        batch_size_history = stats.get()))))))))))))))))))))"batch_size_history", [],,,,,,,,,,,,)
        batch_adaptation_occurred = len()))))))))))))))))))))batch_size_history) > 1 and len()))))))))))))))))))))set()))))))))))))))))))))batch_size_history)) > 1
        
        return {}}}}}}}}}}}}}}}}}}}:
            "success": True,
            "tokens_generated": tokens_generated,
            "tokens_per_second": stats.get()))))))))))))))))))))"tokens_per_second", 0),
            "generation_time_sec": generation_time,
            "avg_token_latency_ms": avg_token_latency,
            "batch_size_history": batch_size_history,
            "batch_adaptation_occurred": batch_adaptation_occurred,
            "result_length": len()))))))))))))))))))))result) if result else 0,:
                "latency_optimized": config.get()))))))))))))))))))))"latency_optimized", False),
                "ultra_low_latency": config.get()))))))))))))))))))))"ultra_low_latency", False),
                "is_async": True
                }
    except Exception as e:
                return {}}}}}}}}}}}}}}}}}}}
                "success": False,
                "error": f"Error with async streaming inference: {}}}}}}}}}}}}}}}}}}}str()))))))))))))))))))))e)}"
                }

                def run_async_test()))))))))))))))))))))verbose: bool = False) -> Dict[str, Any]:,,,,,
                """
                Run async test using asyncio.
    
    Args:
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary with test results
        """
        loop = asyncio.get_event_loop())))))))))))))))))))))
        return loop.run_until_complete()))))))))))))))))))))test_async_streaming_inference()))))))))))))))))))))verbose))

def main()))))))))))))))))))))):
    """Parse arguments and run the tests."""
    parser = argparse.ArgumentParser()))))))))))))))))))))description="Test web platform integration")
    parser.add_argument()))))))))))))))))))))"--platform", choices=["webnn", "webgpu", "both"], default="both",
    help="Which platform to test")
    parser.add_argument()))))))))))))))))))))"--modality", choices=["text", "vision", "audio", "multimodal", "all"], default="all",
    help="Which model modality to test")
    parser.add_argument()))))))))))))))))))))"--verbose", action="store_true",
    help="Enable verbose output")
    
    # Add performance testing options
    performance_group = parser.add_argument_group()))))))))))))))))))))"Performance Testing")
    performance_group.add_argument()))))))))))))))))))))"--iterations", type=int, default=1,
    help="Number of inference iterations for performance testing")
    performance_group.add_argument()))))))))))))))))))))"--benchmark", action="store_true",
    help="Run in benchmark mode with 10 iterations")
    performance_group.add_argument()))))))))))))))))))))"--benchmark-intensive", action="store_true",
    help="Run intensive benchmark with 100 iterations")
    
    # Add model size options
    size_group = parser.add_argument_group()))))))))))))))))))))"Model Size")
    size_group.add_argument()))))))))))))))))))))"--size", choices=["tiny", "small", "base", "large"], default="base",
    help="Model size to test")
    size_group.add_argument()))))))))))))))))))))"--test-all-sizes", action="store_true",
    help="Test all available sizes for each model")
    
    # Add comparison options
    comparison_group = parser.add_argument_group()))))))))))))))))))))"Comparison")
    comparison_group.add_argument()))))))))))))))))))))"--compare-platforms", action="store_true",
    help="Generate detailed platform comparison")
    comparison_group.add_argument()))))))))))))))))))))"--compare-sizes", action="store_true",
    help="Compare different model sizes")
    
    # Add feature tests
    feature_group = parser.add_argument_group()))))))))))))))))))))"Feature Tests")
    feature_group.add_argument()))))))))))))))))))))"--test-unified-framework", action="store_true",
    help="Test unified web framework")
    feature_group.add_argument()))))))))))))))))))))"--test-streaming", action="store_true",
    help="Test streaming inference")
    feature_group.add_argument()))))))))))))))))))))"--test-async-streaming", action="store_true",
    help="Test async streaming inference")
    feature_group.add_argument()))))))))))))))))))))"--test-all-features", action="store_true",
    help="Test all new features")
    
    # Add output options
    output_group = parser.add_argument_group()))))))))))))))))))))"Output")
    output_group.add_argument()))))))))))))))))))))"--output-json", type=str,
    help="Save results to JSON file")
    output_group.add_argument()))))))))))))))))))))"--output-markdown", type=str,
    help="Save results to Markdown file")
                               
    args = parser.parse_args())))))))))))))))))))))
    
    # Determine platforms to test
    platforms = [],,,,,,,,,,,,
    if args.platform == "both":
        platforms = ["webnn", "webgpu"],
    else:
        platforms = [args.platform]
        ,
    # Determine modalities to test
        modalities = [],,,,,,,,,,,,
    if args.modality == "all":
        modalities = ["text", "vision", "audio", "multimodal"],
    else:
        modalities = [args.modality]
        ,
    # Determine performance iterations
        iterations = args.iterations
    if args.benchmark:
        iterations = 10
    elif args.benchmark_intensive:
        iterations = 100
    
    # Determine model sizes to test
        sizes = [],,,,,,,,,,,,
    if args.test_all_sizes or args.compare_sizes:
        sizes = ["tiny", "small", "base"],
    else:
        sizes = [args.size]
        ,
    # Run the tests
        all_results = {}}}}}}}}}}}}}}}}}}}}
    
    # Run feature tests if requested:
        feature_results = {}}}}}}}}}}}}}}}}}}}}
    :
    if args.test_unified_framework or args.test_all_features:
        # Test unified framework for each platform and modality
        unified_results = {}}}}}}}}}}}}}}}}}}}}
        for platform in platforms:
            platform_results = {}}}}}}}}}}}}}}}}}}}}
            for modality in modalities:
                if args.verbose:
                    logger.info()))))))))))))))))))))f"Testing unified framework with {}}}}}}}}}}}}}}}}}}}platform} platform and {}}}}}}}}}}}}}}}}}}}modality} modality")
                    result = test_unified_framework()))))))))))))))))))))platform, modality, args.verbose)
                    platform_results[modality] = result,
                    ,    unified_results[platform] = platform_results,
                    ,    feature_results["unified_framework"] = unified_results
                    ,
        # Print unified framework results
                    print()))))))))))))))))))))"\nUnified Framework Test Results:")
                    print()))))))))))))))))))))"===============================")
        for platform, platform_results in unified_results.items()))))))))))))))))))))):
            print()))))))))))))))))))))f"\n{}}}}}}}}}}}}}}}}}}}platform.upper())))))))))))))))))))))} Platform:")
            for modality, result in platform_results.items()))))))))))))))))))))):
                if result.get()))))))))))))))))))))"success", False):
                    print()))))))))))))))))))))f"  {}}}}}}}}}}}}}}}}}}}modality.capitalize())))))))))))))))))))))}: ✅ PASS")
                    if args.verbose:
                        # Print feature usage
                        feature_usage = result.get()))))))))))))))))))))"feature_usage", {}}}}}}}}}}}}}}}}}}}})
                        print()))))))))))))))))))))"  Feature Usage:")
                        for feature, used in feature_usage.items()))))))))))))))))))))):
                            print()))))))))))))))))))))f"    - {}}}}}}}}}}}}}}}}}}}feature}: {}}}}}}}}}}}}}}}}}}}'✅' if used else '❌'}")
                        
                        # Print performance metrics
                        metrics = result.get()))))))))))))))))))))"metrics", {}}}}}}}}}}}}}}}}}}}}):
                            print()))))))))))))))))))))"  Performance Metrics:")
                            print()))))))))))))))))))))f"    - Initialization: {}}}}}}}}}}}}}}}}}}}metrics.get()))))))))))))))))))))'initialization_time_ms', 0):.2f} ms")
                            print()))))))))))))))))))))f"    - First Inference: {}}}}}}}}}}}}}}}}}}}metrics.get()))))))))))))))))))))'first_inference_time_ms', 0):.2f} ms")
                            print()))))))))))))))))))))f"    - Inference: {}}}}}}}}}}}}}}}}}}}result.get()))))))))))))))))))))'inference_time_ms', 0):.2f} ms")
                else:
                    error = result.get()))))))))))))))))))))"error", "Unknown error")
                    print()))))))))))))))))))))f"  {}}}}}}}}}}}}}}}}}}}modality.capitalize())))))))))))))))))))))}: ❌ FAIL - {}}}}}}}}}}}}}}}}}}}error}")
    
    if args.test_streaming or args.test_all_features:
        # Test streaming inference
        if args.verbose:
            logger.info()))))))))))))))))))))"Testing streaming inference")
            streaming_result = test_streaming_inference()))))))))))))))))))))args.verbose)
            feature_results["streaming_inference"] = streaming_result
            ,
        # Print streaming inference results
            print()))))))))))))))))))))"\nStreaming Inference Test Results:")
            print()))))))))))))))))))))"================================")
        if streaming_result.get()))))))))))))))))))))"success", False):
            print()))))))))))))))))))))"  Streaming Inference: ✅ PASS")
            print()))))))))))))))))))))f"  Tokens Generated: {}}}}}}}}}}}}}}}}}}}streaming_result.get()))))))))))))))))))))'tokens_generated', 0)}")
            print()))))))))))))))))))))f"  Tokens/Second: {}}}}}}}}}}}}}}}}}}}streaming_result.get()))))))))))))))))))))'tokens_per_second', 0):.2f}")
            print()))))))))))))))))))))f"  Generation Time: {}}}}}}}}}}}}}}}}}}}streaming_result.get()))))))))))))))))))))'generation_time_sec', 0):.2f} seconds")
            if args.verbose:
                print()))))))))))))))))))))f"  Batch Size History: {}}}}}}}}}}}}}}}}}}}streaming_result.get()))))))))))))))))))))'batch_size_history', [],,,,,,,,,,,,)}")
                print()))))))))))))))))))))f"  Adaptive Batch Size: {}}}}}}}}}}}}}}}}}}}'✅' if streaming_result.get()))))))))))))))))))))'has_batch_size_adaptation', False) else '❌'}"):
                    print()))))))))))))))))))))f"  Result Length: {}}}}}}}}}}}}}}}}}}}streaming_result.get()))))))))))))))))))))'result_length', 0)} characters")
        else:
            error = streaming_result.get()))))))))))))))))))))"error", "Unknown error")
            print()))))))))))))))))))))f"  Streaming Inference: ❌ FAIL - {}}}}}}}}}}}}}}}}}}}error}")
    
    if args.test_async_streaming or args.test_all_features:
        # Test async streaming inference
        if args.verbose:
            logger.info()))))))))))))))))))))"Testing async streaming inference")
        try:
            async_result = run_async_test()))))))))))))))))))))args.verbose)
            feature_results["async_streaming"] = async_result
            ,
            # Print async streaming results
            print()))))))))))))))))))))"\nAsync Streaming Inference Test Results:")
            print()))))))))))))))))))))"=======================================")
            if async_result.get()))))))))))))))))))))"success", False):
                print()))))))))))))))))))))"  Async Streaming: ✅ PASS")
                print()))))))))))))))))))))f"  Tokens Generated: {}}}}}}}}}}}}}}}}}}}async_result.get()))))))))))))))))))))'tokens_generated', 0)}")
                print()))))))))))))))))))))f"  Tokens/Second: {}}}}}}}}}}}}}}}}}}}async_result.get()))))))))))))))))))))'tokens_per_second', 0):.2f}")
                print()))))))))))))))))))))f"  Generation Time: {}}}}}}}}}}}}}}}}}}}async_result.get()))))))))))))))))))))'generation_time_sec', 0):.2f} seconds")
                if args.verbose:
                    print()))))))))))))))))))))f"  Batch Size History: {}}}}}}}}}}}}}}}}}}}async_result.get()))))))))))))))))))))'batch_size_history', [],,,,,,,,,,,,)}")
                    print()))))))))))))))))))))f"  Result Length: {}}}}}}}}}}}}}}}}}}}async_result.get()))))))))))))))))))))'result_length', 0)} characters")
            else:
                error = async_result.get()))))))))))))))))))))"error", "Unknown error")
                print()))))))))))))))))))))f"  Async Streaming: ❌ FAIL - {}}}}}}}}}}}}}}}}}}}error}")
        except Exception as e:
            print()))))))))))))))))))))f"\nAsync Streaming Inference Test: ❌ FAIL - {}}}}}}}}}}}}}}}}}}}e}")
            feature_results["async_streaming"] = {}}}}}}}}}}}}}}}}}}}"success": False, "error": str()))))))))))))))))))))e)}
            ,
    # Add feature results to overall results
    if feature_results:
        all_results["feature_tests"] = feature_results
        ,
    # Run standard tests for each size
    for size in sizes:
        # Create a result entry for this size
        size_key = f"size_{}}}}}}}}}}}}}}}}}}}size}"
        all_results[size_key] = run_tests())))))))))))))))))))),
        platforms=platforms,
        modalities=modalities,
        verbose=args.verbose,
        model_size=size,
        performance_iterations=iterations
        )
    
    # Print results
    if args.compare_sizes and len()))))))))))))))))))))sizes) > 1:
        # Print comparison between sizes
        print()))))))))))))))))))))"\nSize Comparison:")
        print()))))))))))))))))))))"===============")
        
        # Compare model sizes and print metrics
        print()))))))))))))))))))))"Size comparison for different model variants:")
        print()))))))))))))))))))))"---------------------------------------------")
        
        # Create comparison table
        print()))))))))))))))))))))f"{}}}}}}}}}}}}}}}}}}}'Model Size':<10} {}}}}}}}}}}}}}}}}}}}'Avg Inference ()))))))))))))))))))))ms)':<20} {}}}}}}}}}}}}}}}}}}}'Min Time ()))))))))))))))))))))ms)':<15} {}}}}}}}}}}}}}}}}}}}'Max Time ()))))))))))))))))))))ms)':<15} {}}}}}}}}}}}}}}}}}}}'Memory ()))))))))))))))))))))MB)':<15} {}}}}}}}}}}}}}}}}}}}'Size ()))))))))))))))))))))MB)':<15} {}}}}}}}}}}}}}}}}}}}'Size Reduction %':<15}")
        print()))))))))))))))))))))"-" * 120)
        
        # Track base size for calculating reduction percentages
        base_model_size = 0
        
        for size in sizes:
            size_key = f"size_{}}}}}}}}}}}}}}}}}}}size}"
            if size_key in all_results:
                # Calculate average metrics across all models and platforms
                avg_times = [],,,,,,,,,,,,
                min_times = [],,,,,,,,,,,,
                max_times = [],,,,,,,,,,,,
                memory_usage = [],,,,,,,,,,,,
                model_sizes = [],,,,,,,,,,,,
                
                # Collect metrics from all results
                for platform, platform_results in all_results[size_key].items()))))))))))))))))))))):,,,
                    for modality, result in platform_results.items()))))))))))))))))))))):
                        if result.get()))))))))))))))))))))"success", False) and "performance" in result:
                            perf = result["performance"],,,
                            avg_times.append()))))))))))))))))))))perf.get()))))))))))))))))))))"avg_inference_time_ms", 0))
                            min_times.append()))))))))))))))))))))perf.get()))))))))))))))))))))"min_inference_time_ms", 0))
                            max_times.append()))))))))))))))))))))perf.get()))))))))))))))))))))"max_inference_time_ms", 0))
                            
                            # Extract memory usage from reported metrics if available::
                            reported_metrics = perf.get()))))))))))))))))))))"reported_metrics", {}}}}}}}}}}}}}}}}}}}}):
                            if "memory_usage_mb" in reported_metrics:
                                memory_usage.append()))))))))))))))))))))reported_metrics["memory_usage_mb"])
                                ,
                            # Extract model size if available::
                            if "model_size_mb" in reported_metrics:
                                model_sizes.append()))))))))))))))))))))reported_metrics["model_size_mb"])
                                ,
                # Calculate averages
                                avg_time = sum()))))))))))))))))))))avg_times) / len()))))))))))))))))))))avg_times) if avg_times else 0
                                min_time = sum()))))))))))))))))))))min_times) / len()))))))))))))))))))))min_times) if min_times else 0
                                max_time = sum()))))))))))))))))))))max_times) / len()))))))))))))))))))))max_times) if max_times else 0
                                avg_memory = sum()))))))))))))))))))))memory_usage) / len()))))))))))))))))))))memory_usage) if memory_usage else 0
                                avg_model_size = sum()))))))))))))))))))))model_sizes) / len()))))))))))))))))))))model_sizes) if model_sizes else 0
                
                # Store base size for reduction calculation:
                if size == "base":
                    base_model_size = avg_model_size
                
                # Calculate size reduction percentage
                    size_reduction = 0
                if base_model_size > 0 and avg_model_size > 0:
                    size_reduction = ()))))))))))))))))))))1 - ()))))))))))))))))))))avg_model_size / base_model_size)) * 100
                
                # Print results with size information
                    print()))))))))))))))))))))f"{}}}}}}}}}}}}}}}}}}}size:<10} {}}}}}}}}}}}}}}}}}}}avg_time:<20.2f} {}}}}}}}}}}}}}}}}}}}min_time:<15.2f} {}}}}}}}}}}}}}}}}}}}max_time:<15.2f} {}}}}}}}}}}}}}}}}}}}avg_memory:<15.2f} {}}}}}}}}}}}}}}}}}}}avg_model_size:<15.2f} {}}}}}}}}}}}}}}}}}}}size_reduction:<15.2f}")
        
        # Always return success for size comparison since it's informational
                    success = True
    else:
        # Print regular results ()))))))))))))))))))))using the first/only size)
        first_size = f"size_{}}}}}}}}}}}}}}}}}}}sizes[0]}",
        success = print_test_results()))))))))))))))))))))all_results[first_size], args.verbose)
        ,
    # Save results if requested:
    if args.output_json:
        with open()))))))))))))))))))))args.output_json, 'w') as f:
            json.dump()))))))))))))))))))))all_results, f, indent=2)
            print()))))))))))))))))))))f"\nResults saved to {}}}}}}}}}}}}}}}}}}}args.output_json}")
        
    if args.output_markdown:
        # Generate markdown report
        try:
            with open()))))))))))))))))))))args.output_markdown, 'w') as f:
                # Write markdown header
                f.write()))))))))))))))))))))"# Web Platform Integration Test Report\n\n")
                f.write()))))))))))))))))))))f"Generated on {}}}}}}}}}}}}}}}}}}}datetime.datetime.now()))))))))))))))))))))).strftime()))))))))))))))))))))'%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Write test configuration
                f.write()))))))))))))))))))))"## Test Configuration\n\n")
                f.write()))))))))))))))))))))f"- Platforms: {}}}}}}}}}}}}}}}}}}}', '.join()))))))))))))))))))))platforms)}\n")
                f.write()))))))))))))))))))))f"- Modalities: {}}}}}}}}}}}}}}}}}}}', '.join()))))))))))))))))))))modalities)}\n")
                f.write()))))))))))))))))))))f"- Model Size: {}}}}}}}}}}}}}}}}}}}args.size}\n")
                f.write()))))))))))))))))))))f"- Performance Iterations: {}}}}}}}}}}}}}}}}}}}iterations}\n\n")
                
                # Write test results
                f.write()))))))))))))))))))))"## Test Results\n\n")
                
                for size in sizes:
                    size_key = f"size_{}}}}}}}}}}}}}}}}}}}size}"
                    if size_key in all_results:
                        f.write()))))))))))))))))))))f"### Size: {}}}}}}}}}}}}}}}}}}}size}\n\n")
                        
                        for platform, platform_results in all_results[size_key].items()))))))))))))))))))))):,,,
                        f.write()))))))))))))))))))))f"#### {}}}}}}}}}}}}}}}}}}}platform.upper())))))))))))))))))))))} Platform\n\n")
                            
                            # Create results table
                        f.write()))))))))))))))))))))"| Modality | Model | Status | Avg Time ()))))))))))))))))))))ms) | Memory ()))))))))))))))))))))MB) |\n")
                        f.write()))))))))))))))))))))"|----------|-------|--------|--------------|-------------|\n")
                            
                            for modality, result in platform_results.items()))))))))))))))))))))):
                                status = "✅ PASS" if result.get()))))))))))))))))))))"success", False) else "❌ FAIL"
                                model_name = result.get()))))))))))))))))))))"model_name", "Unknown")
                                
                                # Extract performance metrics
                                avg_time = "N/A"
                                memory = "N/A"
                                :
                                if result.get()))))))))))))))))))))"success", False) and "performance" in result:
                                    perf = result["performance"],,,
                                    avg_time = f"{}}}}}}}}}}}}}}}}}}}perf.get()))))))))))))))))))))'avg_inference_time_ms', 0):.2f}"
                                    
                                    # Extract memory usage if available::
                                    reported_metrics = perf.get()))))))))))))))))))))"reported_metrics", {}}}}}}}}}}}}}}}}}}}})
                                    if "memory_usage_mb" in reported_metrics:
                                        memory = f"{}}}}}}}}}}}}}}}}}}}reported_metrics['memory_usage_mb']:.2f}"
                                        ,
                                        f.write()))))))))))))))))))))f"| {}}}}}}}}}}}}}}}}}}}modality.capitalize())))))))))))))))))))))} | {}}}}}}}}}}}}}}}}}}}model_name} | {}}}}}}}}}}}}}}}}}}}status} | {}}}}}}}}}}}}}}}}}}}avg_time} | {}}}}}}}}}}}}}}}}}}}memory} |\n")
                            
                                        f.write()))))))))))))))))))))"\n")
                        
                                        f.write()))))))))))))))))))))"\n")
                
                # Write size comparison if multiple sizes were tested:
                if args.compare_sizes and len()))))))))))))))))))))sizes) > 1:
                    f.write()))))))))))))))))))))"## Size Comparison\n\n")
                    f.write()))))))))))))))))))))"| Model Size | Avg Inference ()))))))))))))))))))))ms) | Min Time ()))))))))))))))))))))ms) | Max Time ()))))))))))))))))))))ms) | Memory ()))))))))))))))))))))MB) | Size ()))))))))))))))))))))MB) | Size Reduction % |\n")
                    f.write()))))))))))))))))))))"|------------|-------------------|---------------|---------------|-------------|-----------|------------------|\n")
                    
                    # Track base size for calculating reduction percentages
                    base_model_size = 0
                    
                    for size in sizes:
                        size_key = f"size_{}}}}}}}}}}}}}}}}}}}size}"
                        if size_key in all_results:
                            # Calculate average metrics across all models and platforms
                            avg_times = [],,,,,,,,,,,,
                            min_times = [],,,,,,,,,,,,
                            max_times = [],,,,,,,,,,,,
                            memory_usage = [],,,,,,,,,,,,
                            model_sizes = [],,,,,,,,,,,,
                            
                            # Collect metrics from all results
                            for platform, platform_results in all_results[size_key].items()))))))))))))))))))))):,,,
                                for modality, result in platform_results.items()))))))))))))))))))))):
                                    if result.get()))))))))))))))))))))"success", False) and "performance" in result:
                                        perf = result["performance"],,,
                                        avg_times.append()))))))))))))))))))))perf.get()))))))))))))))))))))"avg_inference_time_ms", 0))
                                        min_times.append()))))))))))))))))))))perf.get()))))))))))))))))))))"min_inference_time_ms", 0))
                                        max_times.append()))))))))))))))))))))perf.get()))))))))))))))))))))"max_inference_time_ms", 0))
                                        
                                        # Extract memory usage from reported metrics if available::
                                        reported_metrics = perf.get()))))))))))))))))))))"reported_metrics", {}}}}}}}}}}}}}}}}}}}})
                                        if "memory_usage_mb" in reported_metrics:
                                            memory_usage.append()))))))))))))))))))))reported_metrics["memory_usage_mb"])
                                            ,
                                        # Extract model size if available::
                                        if "model_size_mb" in reported_metrics:
                                            model_sizes.append()))))))))))))))))))))reported_metrics["model_size_mb"])
                                            ,
                            # Calculate averages
                                            avg_time = sum()))))))))))))))))))))avg_times) / len()))))))))))))))))))))avg_times) if avg_times else 0
                                            min_time = sum()))))))))))))))))))))min_times) / len()))))))))))))))))))))min_times) if min_times else 0
                                            max_time = sum()))))))))))))))))))))max_times) / len()))))))))))))))))))))max_times) if max_times else 0
                                            avg_memory = sum()))))))))))))))))))))memory_usage) / len()))))))))))))))))))))memory_usage) if memory_usage else 0
                                            avg_model_size = sum()))))))))))))))))))))model_sizes) / len()))))))))))))))))))))model_sizes) if model_sizes else 0
                            
                            # Store base size for reduction calculation:
                            if size == "base":
                                base_model_size = avg_model_size
                            
                            # Calculate size reduction percentage
                                size_reduction = 0
                            if base_model_size > 0 and avg_model_size > 0:
                                size_reduction = ()))))))))))))))))))))1 - ()))))))))))))))))))))avg_model_size / base_model_size)) * 100
                            
                            # Write to markdown
                                f.write()))))))))))))))))))))f"| {}}}}}}}}}}}}}}}}}}}size} | {}}}}}}}}}}}}}}}}}}}avg_time:.2f} | {}}}}}}}}}}}}}}}}}}}min_time:.2f} | {}}}}}}}}}}}}}}}}}}}max_time:.2f} | {}}}}}}}}}}}}}}}}}}}avg_memory:.2f} | {}}}}}}}}}}}}}}}}}}}avg_model_size:.2f} | {}}}}}}}}}}}}}}}}}}}size_reduction:.2f} |\n")
                    
                                f.write()))))))))))))))))))))"\n")
                
                # Write summary
                                f.write()))))))))))))))))))))"## Summary\n\n")
                                f.write()))))))))))))))))))))f"Overall test result: **{}}}}}}}}}}}}}}}}}}}'PASS' if success else 'FAIL'}**\n\n")
                
                # Write recommendations based on results
                f.write()))))))))))))))))))))"## Recommendations\n\n"):
                    f.write()))))))))))))))))))))"Based on the test results, here are some recommendations:\n\n")
                
                # Analyze results and provide recommendations
                for platform in platforms:
                    platform_success = True
                    platform_issues = [],,,,,,,,,,,,
                    
                    for size_key in all_results:
                        if platform in all_results[size_key]:,
                        for modality, result in all_results[size_key][platform].items()))))))))))))))))))))):,
                                if not result.get()))))))))))))))))))))"success", False):
                                    platform_success = False
                                    error = result.get()))))))))))))))))))))"error", "Unknown error")
                                    platform_issues.append()))))))))))))))))))))f"- {}}}}}}}}}}}}}}}}}}}modality.capitalize())))))))))))))))))))))}: {}}}}}}}}}}}}}}}}}}}error}")
                    
                    if platform_success:
                        f.write()))))))))))))))))))))f"- {}}}}}}}}}}}}}}}}}}}platform.upper())))))))))))))))))))))}: All tests passed. Platform is fully compatible.\n")
                    else:
                        f.write()))))))))))))))))))))f"- {}}}}}}}}}}}}}}}}}}}platform.upper())))))))))))))))))))))}: Some tests failed. Issues to address:\n")
                        for issue in platform_issues:
                            f.write()))))))))))))))))))))f"  {}}}}}}}}}}}}}}}}}}}issue}\n")
                    
                            f.write()))))))))))))))))))))"\n")
                
                # Write next steps
                            f.write()))))))))))))))))))))"## Next Steps\n\n")
                            f.write()))))))))))))))))))))"1. Fix any failing tests identified in this report\n")
                            f.write()))))))))))))))))))))"2. Run comprehensive benchmarks with the database integration\n")
                            f.write()))))))))))))))))))))"3. Test with real browser environments using browser automation\n")
                            f.write()))))))))))))))))))))"4. Implement fixes for any platform-specific issues\n")
                
                            print()))))))))))))))))))))f"\nMarkdown report saved to {}}}}}}}}}}}}}}}}}}}args.output_markdown}")
        except Exception as e:
            print()))))))))))))))))))))f"\nError generating markdown report: {}}}}}}}}}}}}}}}}}}}e}")
    
                            return 0 if success else 1
:
if __name__ == "__main__":
    sys.exit()))))))))))))))))))))main()))))))))))))))))))))))