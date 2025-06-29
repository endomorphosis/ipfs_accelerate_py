#!/usr/bin/env python3
"""
Test enhanced OpenVINO backend integration with optimum.intel and INT8 quantization.

This script demonstrates the enhanced capabilities of the OpenVINO backend, including:
    1. Improved optimum.intel integration for HuggingFace models
    2. Enhanced INT8 quantization with calibration data
    3. Model format conversion and optimization
    4. Precision control ()))))))FP32, FP16, INT8)
    """

    import os
    import sys
    import logging
    import argparse
    import time
    from typing import Dict, Any, List, Optional

# Configure logging
    logging.basicConfig()))))))level=logging.INFO, format='%()))))))asctime)s - %()))))))name)s - %()))))))levelname)s - %()))))))message)s')
    logger = logging.getLogger()))))))"test_enhanced_openvino")

# Add parent directory to path for imports
    sys.path.insert()))))))0, os.path.dirname()))))))os.path.dirname()))))))os.path.abspath()))))))__file__))))

# Import the OpenVINO backend
try:
    sys.path.insert()))))))0, os.path.join()))))))os.path.dirname()))))))os.path.abspath()))))))__file__))))
    from ipfs_accelerate_py.hardware.backends.openvino_backend import OpenVINOBackend
    BACKEND_IMPORTED = True
except ImportError as e:
    logger.error()))))))f"Failed to import OpenVINO backend: {}}}}}}}e}")
    logger.error()))))))f"Current sys.path: {}}}}}}}sys.path}")
    BACKEND_IMPORTED = False

def test_optimum_integration()))))))model_name="bert-base-uncased", device="CPU"):
    """Test optimum.intel integration with OpenVINO backend."""
    if not BACKEND_IMPORTED:
        logger.error()))))))"OpenVINO backend not imported, skipping optimum.intel test")
    return False
    
    logger.info()))))))f"Testing optimum.intel integration with {}}}}}}}model_name} on device {}}}}}}}device}...")
    
    try:
        backend = OpenVINOBackend())))))))
        
        if not backend.is_available()))))))):
            logger.warning()))))))"OpenVINO is not available on this system, skipping test")
        return False
        
        # Check for optimum.intel integration
        optimum_info = backend.get_optimum_integration())))))))
        if not optimum_info.get()))))))"available", False):
            logger.warning()))))))"optimum.intel is not available, skipping test")
        return False
        
        # Log optimum.intel info
        logger.info()))))))f"optimum.intel version: {}}}}}}}optimum_info.get()))))))'version', 'Unknown')}")
        logger.info()))))))f"Supported models: {}}}}}}}len()))))))optimum_info.get()))))))'supported_models', [],]))}")
        ,
        for model_info in optimum_info.get()))))))'supported_models', [],]):,
        logger.info()))))))f"  - {}}}}}}}model_info.get()))))))'type')}: {}}}}}}}model_info.get()))))))'class_name')} ()))))))Available: {}}}}}}}model_info.get()))))))'available', False)})")
        
        # Test loading a model with optimum.intel
        config = {}}}}}}}
        "device": device,
        "model_type": "text",
        "precision": "FP32",
        "use_optimum": True
        }
        
        # Load the model
        logger.info()))))))f"Loading model {}}}}}}}model_name} with optimum.intel...")
        load_result = backend.load_model()))))))model_name, config)
        
        if load_result.get()))))))"status") != "success":
            logger.error()))))))f"Failed to load model: {}}}}}}}load_result.get()))))))'message', 'Unknown error')}")
        return False
        
        logger.info()))))))f"Model {}}}}}}}model_name} loaded successfully with optimum.intel")
        
        # Test inference
        logger.info()))))))f"Running inference with {}}}}}}}model_name} using optimum.intel...")
        
        # Sample input text
        input_text = "This is a test sentence for OpenVINO inference."
        
        inference_result = backend.run_inference()))))))
        model_name,
        input_text,
        {}}}}}}}"device": device, "model_type": "text"}
        )
        
        if inference_result.get()))))))"status") != "success":
            logger.error()))))))f"Inference failed: {}}}}}}}inference_result.get()))))))'message', 'Unknown error')}")
        return False
        
        # Print inference metrics
        logger.info()))))))f"Inference completed successfully with optimum.intel")
        logger.info()))))))f"  Latency: {}}}}}}}inference_result.get()))))))'latency_ms', 0):.2f} ms")
        logger.info()))))))f"  Throughput: {}}}}}}}inference_result.get()))))))'throughput_items_per_sec', 0):.2f} items/sec")
        logger.info()))))))f"  Memory usage: {}}}}}}}inference_result.get()))))))'memory_usage_mb', 0):.2f} MB")
        
        # Unload the model
        logger.info()))))))f"Unloading model {}}}}}}}model_name}...")
        backend.unload_model()))))))model_name, device)
        
    return True
    except Exception as e:
        logger.error()))))))f"Error during optimum.intel test: {}}}}}}}e}")
    return False

def test_int8_quantization()))))))model_name="bert-base-uncased", device="CPU"):
    """Test INT8 quantization with OpenVINO backend."""
    if not BACKEND_IMPORTED:
        logger.error()))))))"OpenVINO backend not imported, skipping INT8 quantization test")
    return False
    
    logger.info()))))))f"Testing INT8 quantization with {}}}}}}}model_name} on device {}}}}}}}device}...")
    
    try:
        backend = OpenVINOBackend())))))))
        
        if not backend.is_available()))))))):
            logger.warning()))))))"OpenVINO is not available on this system, skipping test")
        return False
        
        # Import required libraries
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            logger.error()))))))f"Failed to import required libraries: {}}}}}}}e}")
            return False
        
        # Load model with PyTorch
            logger.info()))))))f"Loading {}}}}}}}model_name} with PyTorch...")
            tokenizer = AutoTokenizer.from_pretrained()))))))model_name)
            pt_model = AutoModel.from_pretrained()))))))model_name)
        
        # Export to ONNX
            import tempfile
            import os
            from transformers.onnx import export as onnx_export
        
        # Create temporary directory for export
        with tempfile.TemporaryDirectory()))))))) as temp_dir:
            # Create ONNX export path
            onnx_path = os.path.join()))))))temp_dir, "model.onnx")
            
            # Export model to ONNX
            logger.info()))))))f"Exporting {}}}}}}}model_name} to ONNX format...")
            input_sample = tokenizer()))))))"Sample text for export", return_tensors="pt")
            
            # Export the model
            onnx_export()))))))
            preprocessor=tokenizer,
            model=pt_model,
            config=pt_model.config,
            opset=13,
            output=onnx_path
            )
            
            logger.info()))))))f"Model exported to ONNX: {}}}}}}}onnx_path}")
            
            # Generate calibration data
            logger.info()))))))"Generating calibration data...")
            
            calibration_texts = [],
            "The quick brown fox jumps over the lazy dog.",
            "OpenVINO provides hardware acceleration for deep learning models.",
            "INT8 quantization can significantly improve performance.",
            "Deep learning frameworks optimize inference on various hardware platforms.",
            "Model compression techniques reduce memory footprint while maintaining accuracy."
            ]
            
            calibration_data = [],]:
            for text in calibration_texts:
                inputs = tokenizer()))))))text, return_tensors="pt")
                sample = {}}}}}}}
                "input_ids": inputs[],"input_ids"].numpy()))))))),
                "attention_mask": inputs[],"attention_mask"].numpy())))))))
                }
                calibration_data.append()))))))sample)
            
                logger.info()))))))f"Generated {}}}}}}}len()))))))calibration_data)} calibration samples")
            
            # Test FP32 inference
                logger.info()))))))"Testing FP32 inference with ONNX model...")
            
                fp32_config = {}}}}}}}
                "device": device,
                "model_type": "text",
                "precision": "FP32",
                "model_path": onnx_path,
                "model_format": "ONNX"
                }
            
            # Load FP32 model
                fp32_load_result = backend.load_model()))))))
                "bert_fp32",
                fp32_config
                )
            
            if fp32_load_result.get()))))))"status") != "success":
                logger.error()))))))f"Failed to load FP32 model: {}}}}}}}fp32_load_result.get()))))))'message', 'Unknown error')}")
                return False
            
            # Run FP32 inference
                fp32_inference_result = backend.run_inference()))))))
                "bert_fp32",
                calibration_data[],0],
                {}}}}}}}"device": device, "model_type": "text"}
                )
            
            if fp32_inference_result.get()))))))"status") != "success":
                logger.error()))))))f"FP32 inference failed: {}}}}}}}fp32_inference_result.get()))))))'message', 'Unknown error')}")
                return False
            
                logger.info()))))))f"FP32 inference completed:")
                logger.info()))))))f"  Latency: {}}}}}}}fp32_inference_result.get()))))))'latency_ms', 0):.2f} ms")
            
            # Test INT8 inference
                logger.info()))))))"Testing INT8 inference with ONNX model and calibration data...")
            
                int8_config = {}}}}}}}
                "device": device,
                "model_type": "text",
                "precision": "INT8",
                "model_path": onnx_path,
                "model_format": "ONNX",
                "calibration_data": calibration_data
                }
            
            # Load INT8 model
                int8_load_result = backend.load_model()))))))
                "bert_int8",
                int8_config
                )
            
            if int8_load_result.get()))))))"status") != "success":
                logger.error()))))))f"Failed to load INT8 model: {}}}}}}}int8_load_result.get()))))))'message', 'Unknown error')}")
                return False
            
            # Run INT8 inference
                int8_inference_result = backend.run_inference()))))))
                "bert_int8",
                calibration_data[],0],
                {}}}}}}}"device": device, "model_type": "text"}
                )
            
            if int8_inference_result.get()))))))"status") != "success":
                logger.error()))))))f"INT8 inference failed: {}}}}}}}int8_inference_result.get()))))))'message', 'Unknown error')}")
                return False
            
                logger.info()))))))f"INT8 inference completed:")
                logger.info()))))))f"  Latency: {}}}}}}}int8_inference_result.get()))))))'latency_ms', 0):.2f} ms")
            
            # Compare performance
                fp32_latency = fp32_inference_result.get()))))))'latency_ms', 0)
                int8_latency = int8_inference_result.get()))))))'latency_ms', 0)
            
            if fp32_latency > 0 and int8_latency > 0:
                speedup = fp32_latency / int8_latency
                logger.info()))))))f"INT8 speedup: {}}}}}}}speedup:.2f}x faster than FP32")
            
            # Unload models
                backend.unload_model()))))))"bert_fp32", device)
                backend.unload_model()))))))"bert_int8", device)
            
                return True
            
    except Exception as e:
        logger.error()))))))f"Error during INT8 quantization test: {}}}}}}}e}")
                return False

def compare_precisions()))))))model_name="bert-base-uncased", device="CPU", iterations=5):
    """Compare FP32, FP16, and INT8 precision performance."""
    if not BACKEND_IMPORTED:
        logger.error()))))))"OpenVINO backend not imported, skipping comparison")
    return False
    
    logger.info()))))))f"Comparing precision performance for {}}}}}}}model_name} on {}}}}}}}device}...")
    
    try:
        backend = OpenVINOBackend())))))))
        
        if not backend.is_available()))))))):
            logger.warning()))))))"OpenVINO is not available on this system, skipping comparison")
        return False
        
        # Import required libraries
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            import numpy as np
        except ImportError as e:
            logger.error()))))))f"Failed to import required libraries: {}}}}}}}e}")
            return False
        
        # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained()))))))model_name)
        
        # Create sample input
            test_text = "This is a test sentence for benchmarking different precisions."
            inputs = tokenizer()))))))test_text, return_tensors="pt")
        
        # Convert to numpy
            input_dict = {}}}}}}}
            "input_ids": inputs[],"input_ids"].numpy()))))))),
            "attention_mask": inputs[],"attention_mask"].numpy())))))))
            }
        
        # Prepare to store results
            results = {}}}}}}}}
        
        # Test with optimum.intel if available
        optimum_info = backend.get_optimum_integration()))))))):
        if optimum_info.get()))))))"available", False):
            logger.info()))))))"Testing optimum.intel integration with different precisions")
            
            # Prepare configurations for each precision
            precisions = [],"FP32", "FP16", "INT8"]
            
            for precision in precisions:
                # Create configuration
                config = {}}}}}}}
                "device": device,
                "model_type": "text",
                "precision": precision,
                "use_optimum": True
                }
                
                # Clean model name with precision
                model_key = f"{}}}}}}}model_name}_{}}}}}}}precision}_optimum"
                
                # Load the model
                logger.info()))))))f"Loading {}}}}}}}model_name} with optimum.intel in {}}}}}}}precision} precision...")
                load_result = backend.load_model()))))))model_key, config)
                
                if load_result.get()))))))"status") != "success":
                    logger.warning()))))))f"Failed to load model in {}}}}}}}precision} precision: {}}}}}}}load_result.get()))))))'message', 'Unknown error')}")
                continue
                
                # Run warmup inference
                backend.run_inference()))))))model_key, test_text, {}}}}}}}"device": device, "model_type": "text"})
                
                # Collect latencies
                latencies = [],]
                
                logger.info()))))))f"Running {}}}}}}}iterations} iterations for {}}}}}}}precision}...")
                
                for i in range()))))))iterations):
                    inference_result = backend.run_inference()))))))
                    model_key,
                    test_text,
                    {}}}}}}}"device": device, "model_type": "text"}
                    )
                    
                    if inference_result.get()))))))"status") == "success":
                        latencies.append()))))))inference_result.get()))))))"latency_ms", 0))
                
                # Calculate average metrics
                if latencies:
                    avg_latency = sum()))))))latencies) / len()))))))latencies)
                    min_latency = min()))))))latencies)
                    max_latency = max()))))))latencies)
                    
                    # Store results
                    results[],f"{}}}}}}}precision} ()))))))optimum.intel)"] = {}}}}}}}
                    "avg_latency_ms": avg_latency,
                    "min_latency_ms": min_latency,
                    "max_latency_ms": max_latency,
                    "throughput_items_per_sec": 1000 / avg_latency
                    }
                    
                    # Log results
                    logger.info()))))))f"{}}}}}}}precision} ()))))))optimum.intel) Results:")
                    logger.info()))))))f"  Average Latency: {}}}}}}}avg_latency:.2f} ms")
                    logger.info()))))))f"  Min Latency: {}}}}}}}min_latency:.2f} ms")
                    logger.info()))))))f"  Max Latency: {}}}}}}}max_latency:.2f} ms")
                    logger.info()))))))f"  Throughput: {}}}}}}}1000 / avg_latency:.2f} items/sec")
                
                # Unload the model
                    backend.unload_model()))))))model_key, device)
        
        # Print comparison
        if results:
            logger.info()))))))"\nPerformance Comparison:")
            logger.info()))))))"=" * 60)
            logger.info()))))))f"{}}}}}}}'Precision':<20} {}}}}}}}'Avg Latency ()))))))ms)':<20} {}}}}}}}'Throughput ()))))))items/sec)':<20}")
            logger.info()))))))"-" * 60)
            
            # Find the baseline for normalization ()))))))using FP32 if available, otherwise first in results)
            baseline_key = next()))))))()))))))k for k in results if "FP32" in k), next()))))))iter()))))))results.keys()))))))))))
            baseline_latency = results[],baseline_key][],"avg_latency_ms"]
            :
            for precision, metrics in results.items()))))))):
                speedup = baseline_latency / metrics[],"avg_latency_ms"] if metrics[],"avg_latency_ms"] > 0 else 0:
                    logger.info()))))))f"{}}}}}}}precision:<20} {}}}}}}}metrics[],'avg_latency_ms']:<20.2f} {}}}}}}}metrics[],'throughput_items_per_sec']:<20.2f} ())))))){}}}}}}}speedup:.2f}x)")
            
                    logger.info()))))))"=" * 60)
        
                return True
    except Exception as e:
        logger.error()))))))f"Error during precision comparison: {}}}}}}}e}")
                return False

def main()))))))):
    """Command-line entry point."""
    parser = argparse.ArgumentParser()))))))description="Test enhanced OpenVINO backend integration")
    
    # Test options
    parser.add_argument()))))))"--test-optimum", action="store_true", help="Test optimum.intel integration")
    parser.add_argument()))))))"--test-int8", action="store_true", help="Test INT8 quantization")
    parser.add_argument()))))))"--compare-precisions", action="store_true", help="Compare FP32, FP16, and INT8 precision performance")
    parser.add_argument()))))))"--run-all", action="store_true", help="Run all tests")
    
    # Configuration options
    parser.add_argument()))))))"--model", type=str, default="bert-base-uncased", help="Model name to use for tests")
    parser.add_argument()))))))"--device", type=str, default="CPU", help="OpenVINO device to use ()))))))CPU, GPU, AUTO, etc.)")
    parser.add_argument()))))))"--iterations", type=int, default=5, help="Number of iterations for performance comparison")
    
    args = parser.parse_args())))))))
    
    # If no specific test is selected, print help
    if not ()))))))args.test_optimum or args.test_int8 or args.compare_precisions or args.run_all):
        parser.print_help())))))))
    return 1
    
    # Run tests based on arguments
    results = {}}}}}}}}
    
    if args.test_optimum or args.run_all:
        results[],"optimum_integration"] = test_optimum_integration()))))))args.model, args.device)
    
    if args.test_int8 or args.run_all:
        results[],"int8_quantization"] = test_int8_quantization()))))))args.model, args.device)
    
    if args.compare_precisions or args.run_all:
        results[],"precision_comparison"] = compare_precisions()))))))args.model, args.device, args.iterations)
    
    # Print overall test results
        logger.info()))))))"\nOverall Test Results:")
    for test_name, result in results.items()))))))):
        status = "PASSED" if result else "FAILED":
            logger.info()))))))f"  {}}}}}}}test_name}: {}}}}}}}status}")
    
    # Check if any test failed:
    if False in results.values()))))))):
            return 1
    
        return 0

if __name__ == "__main__":
    sys.exit()))))))main()))))))))