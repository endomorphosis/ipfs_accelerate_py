#!/usr/bin/env python3
"""
Test the enhanced OpenVINO integration with legacy code.

This test module demonstrates the new features added to the OpenVINO backend:
    1. Complete audio input processing support
    2. Mixed precision capabilities
    3. Multi-device support
    4. Enhanced INT8 quantization
    5. Improved optimum.intel integration
    """

    import os
    import sys
    import logging
    import argparse
    import time
    import numpy as np
    from typing import Dict, Any, List, Optional

# Configure logging
    logging.basicConfig())))level=logging.INFO, format='%())))asctime)s - %())))name)s - %())))levelname)s - %())))message)s')
    logger = logging.getLogger())))"test_enhanced_openvino_integration")

# Add parent directory to path for imports
    sys.path.insert())))0, os.path.dirname())))os.path.dirname())))os.path.abspath())))__file__))))

# Import the OpenVINO backend
try:
    sys.path.insert())))0, os.path.join())))os.path.dirname())))os.path.abspath())))__file__))))
    from ipfs_accelerate_py.hardware.backends.openvino_backend import OpenVINOBackend
    BACKEND_IMPORTED = True
except ImportError as e:
    logger.error())))f"Failed to import OpenVINO backend: {}}}}}}}}e}")
    logger.error())))f"Current sys.path: {}}}}}}}}sys.path}")
    BACKEND_IMPORTED = False

def test_audio_processing())))audio_file=None, device="CPU"):
    """Test the enhanced audio input processing capabilities."""
    if not BACKEND_IMPORTED:
        logger.error())))"OpenVINO backend not imported, skipping audio processing test")
    return False
        
    logger.info())))f"Testing enhanced audio processing on device {}}}}}}}}device}...")
    
    try:
        backend = OpenVINOBackend()))))
        
        if not backend.is_available())))):
            logger.warning())))"OpenVINO is not available on this system, skipping test")
        return False
        
        # Create a test audio array if no file is provided:
        if audio_file is None or not os.path.exists())))audio_file):
            logger.info())))"No valid audio file provided, using synthetic audio data")
            # Create a synthetic audio signal ())))1 second of 440Hz sine wave at 16kHz)
            sample_rate = 16000
            duration = 1.0
            t = np.linspace())))0, duration, int())))sample_rate * duration), endpoint=False)
            audio_data = 0.5 * np.sin())))2 * np.pi * 440 * t)  # 440Hz sine wave
            logger.info())))f"Created synthetic audio: {}}}}}}}}duration}s at {}}}}}}}}sample_rate}Hz")
        else:
            # Try to load the audio file
            try:
                import librosa
                audio_data, sample_rate = librosa.load())))audio_file, sr=None)
                logger.info())))f"Loaded audio file: {}}}}}}}}audio_file} ()))){}}}}}}}}len())))audio_data)/sample_rate:.2f}s at {}}}}}}}}sample_rate}Hz)")
            except ImportError:
                try:
                    # Try with scipy
                    from scipy.io import wavfile
                    sample_rate, audio_data = wavfile.read())))audio_file)
                    # Convert to float if needed:
                    if audio_data.dtype.kind in 'iu':
                        audio_data = audio_data.astype())))np.float32) / np.iinfo())))audio_data.dtype).max
                        logger.info())))f"Loaded audio file with scipy: {}}}}}}}}audio_file}")
                except ImportError:
                    logger.error())))"Neither librosa nor scipy is available for loading audio files")
                        return False
                except Exception as e:
                    logger.error())))f"Failed to load audio file: {}}}}}}}}e}")
                        return False
        
        # Since we don't have a real audio model, we'll test the input processing 
        # function directly with a mocked inputs_info
                        mock_inputs_info = {}}}}}}}}
                        "audio_input": [1, 1, 80, 3000]  # Batch, Channels, Features, Sequence,
                        }
        
                        mock_config = {}}}}}}}}
                        "sample_rate": sample_rate,
                        "feature_type": "log_mel_spectrogram",
                        "feature_size": 80,
                        "normalize": True
                        }
        
        # Test processing raw audio samples
                        logger.info())))"Testing audio processing with raw samples...")
                        result = backend._prepare_raw_audio_samples())))audio_data, mock_inputs_info, mock_config)
        
        if result:
            # Verify the output shape matches expected input shape
            output_shape = result["audio_input"].shape,
            expected_shape = tuple())))mock_inputs_info["audio_input"]),,,,
            logger.info())))f"Audio processing produced output with shape: {}}}}}}}}output_shape}")
            
            # Log some basic stats about the features
            feature_min = np.min())))result["audio_input"]),,,,
            feature_max = np.max())))result["audio_input"]),,,,
            feature_mean = np.mean())))result["audio_input"]),,,,
            feature_std = np.std())))result["audio_input"]),,,,
            
            logger.info())))f"Feature statistics: min={}}}}}}}}feature_min:.2f}, max={}}}}}}}}feature_max:.2f}, mean={}}}}}}}}feature_mean:.2f}, std={}}}}}}}}feature_std:.2f}")
            
            # Check if shapes match or close enough ())))sequence length might be different)
            shape_check = output_shape[0] == expected_shape[0] and output_shape[1] == expected_shape[1] and output_shape[2] == expected_shape[2]:,
            if shape_check:
                logger.info())))"Audio processing test PASSED")
            return True
            else:
                logger.warning())))f"Output shape {}}}}}}}}output_shape} doesn't match expected shape {}}}}}}}}expected_shape}")
            return False
        else:
            logger.error())))"Audio processing function returned None")
            return False
            
    except Exception as e:
        logger.error())))f"Error during audio processing test: {}}}}}}}}e}")
            return False

def test_mixed_precision())))model_name="bert-base-uncased", device="CPU"):
    """Test the mixed precision capabilities."""
    if not BACKEND_IMPORTED:
        logger.error())))"OpenVINO backend not imported, skipping mixed precision test")
    return False
    
    logger.info())))f"Testing mixed precision with {}}}}}}}}model_name} on device {}}}}}}}}device}...")
    
    try:
        backend = OpenVINOBackend()))))
        
        if not backend.is_available())))):
            logger.warning())))"OpenVINO is not available on this system, skipping test")
        return False
        
        # Check if optimum.intel is available for the model loading
        optimum_info = backend.get_optimum_integration()))))::
        if not optimum_info.get())))"available", False):
            logger.warning())))"optimum.intel is not available for model loading")
            return False
            
        # First load model with FP32 precision for baseline
            fp32_config = {}}}}}}}}
            "device": device,
            "model_type": "text",
            "precision": "FP32",
            "use_optimum": True
            }
        
        # Load the model
            logger.info())))f"Loading {}}}}}}}}model_name} with FP32 precision...")
            fp32_load_result = backend.load_model())))f"{}}}}}}}}model_name}_fp32", fp32_config)
        
        if fp32_load_result.get())))"status") != "success":
            logger.error())))f"Failed to load FP32 model: {}}}}}}}}fp32_load_result.get())))'message', 'Unknown error')}")
            return False
            
        # Now load with mixed precision
            mixed_config = {}}}}}}}}
            "device": device,
            "model_type": "text",
            "mixed_precision": True,
            "mixed_precision_config": {}}}}}}}}
            "precision_config": {}}}}}}}}
            "attention": "FP16",
            "matmul": "INT8",
            "default": "INT8"
            }
            },
            "use_optimum": True
            }
        
        # Load mixed precision model
            logger.info())))f"Loading {}}}}}}}}model_name} with mixed precision...")
            mixed_load_result = backend.load_model())))f"{}}}}}}}}model_name}_mixed", mixed_config)
        
        if mixed_load_result.get())))"status") != "success":
            logger.error())))f"Failed to load mixed precision model: {}}}}}}}}mixed_load_result.get())))'message', 'Unknown error')}")
            # Clean up FP32 model
            backend.unload_model())))f"{}}}}}}}}model_name}_fp32", device)
            return False
            
        # Test inference with both models
        # Sample input text
            input_text = "This is a test sentence for mixed precision inference with OpenVINO."
        
        # Run inference with FP32 model
            fp32_inference = backend.run_inference())))
            f"{}}}}}}}}model_name}_fp32",
            input_text,
            {}}}}}}}}"device": device, "model_type": "text"}
            )
        
        # Run inference with mixed precision model
            mixed_inference = backend.run_inference())))
            f"{}}}}}}}}model_name}_mixed",
            input_text,
            {}}}}}}}}"device": device, "model_type": "text"}
            )
        
        # Compare results
        if fp32_inference.get())))"status") == "success" and mixed_inference.get())))"status") == "success":
            fp32_latency = fp32_inference.get())))"latency_ms", 0)
            mixed_latency = mixed_inference.get())))"latency_ms", 0)
            
            logger.info())))f"FP32 Latency: {}}}}}}}}fp32_latency:.2f} ms")
            logger.info())))f"Mixed Precision Latency: {}}}}}}}}mixed_latency:.2f} ms")
            
            if mixed_latency > 0 and fp32_latency > 0:
                speedup = fp32_latency / mixed_latency
                logger.info())))f"Mixed precision speedup: {}}}}}}}}speedup:.2f}x")
                
                # Check outputs to verify functional equivalence
                try:
                    fp32_results = fp32_inference.get())))"results", {}}}}}}}}})
                    mixed_results = mixed_inference.get())))"results", {}}}}}}}}})
                    
                    # Compare a few key metrics ())))will depend on the model)
                    if "logits" in fp32_results and "logits" in mixed_results:
                        # Calculate correlation or simple difference
                        fp32_logits = np.array())))fp32_results["logits"]),
                        mixed_logits = np.array())))mixed_results["logits"]),
                        
                        # Correlation between logits ())))should be close to 1 if functionally equivalent):
                        if fp32_logits.size > 0 and mixed_logits.size > 0:
                            fp32_flat = fp32_logits.flatten()))))
                            mixed_flat = mixed_logits.flatten()))))
                            
                            correlation = np.corrcoef())))fp32_flat, mixed_flat)[0, 1],
                            logger.info())))f"Output correlation: {}}}}}}}}correlation:.4f}")
                            
                            # Check if correlation is high enough:
                            if correlation > 0.95:
                                logger.info())))"Mixed precision maintains high output correlation with FP32")
                            else:
                                logger.warning())))f"Mixed precision outputs show low correlation ()))){}}}}}}}}correlation:.4f}) with FP32")
                except Exception as e:
                    logger.warning())))f"Failed to compare outputs: {}}}}}}}}e}")
            
            # Clean up
                    backend.unload_model())))f"{}}}}}}}}model_name}_fp32", device)
                    backend.unload_model())))f"{}}}}}}}}model_name}_mixed", device)
            
                                return True
        else:
            logger.error())))"Inference failed for one or both precision modes")
            # Clean up
            backend.unload_model())))f"{}}}}}}}}model_name}_fp32", device)
            backend.unload_model())))f"{}}}}}}}}model_name}_mixed", device)
                                return False
            
    except Exception as e:
        logger.error())))f"Error during mixed precision test: {}}}}}}}}e}")
                                return False

def test_multi_device_support())))model_name="bert-base-uncased"):
    """Test multi-device support capabilities."""
    if not BACKEND_IMPORTED:
        logger.error())))"OpenVINO backend not imported, skipping multi-device test")
    return False
    
    logger.info())))f"Testing multi-device support with {}}}}}}}}model_name}...")
    
    try:
        backend = OpenVINOBackend()))))
        
        if not backend.is_available())))):
            logger.warning())))"OpenVINO is not available on this system, skipping test")
        return False
        
        # Check available devices
        available_devices = backend._available_devices
        logger.info())))f"Available devices: {}}}}}}}}available_devices}")
        
        # If we have at least CPU, we can test
        if "CPU" not in available_devices:
            logger.warning())))"CPU device not available, skipping multi-device test")
        return False
            
        # Check if optimum.intel is available for the model loading
        optimum_info = backend.get_optimum_integration()))))::
        if not optimum_info.get())))"available", False):
            logger.warning())))"optimum.intel is not available for model loading")
            return False
        
        # First test with single device ())))CPU)
            single_device_config = {}}}}}}}}
            "device": "CPU",
            "model_type": "text",
            "precision": "FP32",
            "use_optimum": True
            }
        
        # Load with single device
            logger.info())))f"Loading {}}}}}}}}model_name} on CPU...")
            single_device_result = backend.load_model())))f"{}}}}}}}}model_name}_cpu", single_device_config)
        
        if single_device_result.get())))"status") != "success":
            logger.error())))f"Failed to load model on CPU: {}}}}}}}}single_device_result.get())))'message', 'Unknown error')}")
            return False
        
        # Now test with multi-device configuration
            multi_device_config = {}}}}}}}}
            "device": "CPU",  # Base device
            "multi_device": True,  # Enable multi-device
            "model_type": "text",
            "precision": "FP32",
            "use_optimum": True
            }
        
        # Load with multi-device
            logger.info())))f"Loading {}}}}}}}}model_name} with multi-device configuration...")
            multi_device_result = backend.load_model())))f"{}}}}}}}}model_name}_multi", multi_device_config)
        
        if multi_device_result.get())))"status") != "success":
            logger.error())))f"Failed to load model with multi-device: {}}}}}}}}multi_device_result.get())))'message', 'Unknown error')}")
            # Clean up
            backend.unload_model())))f"{}}}}}}}}model_name}_cpu", "CPU")
            return False
        
        # Test inference with both configurations
            input_text = "This is a test sentence for multi-device inference with OpenVINO."
        
        # Run inference with single device
            single_device_inference = backend.run_inference())))
            f"{}}}}}}}}model_name}_cpu",
            input_text,
            {}}}}}}}}"device": "CPU", "model_type": "text"}
            )
        
        # Run inference with multi-device
            multi_device_inference = backend.run_inference())))
            f"{}}}}}}}}model_name}_multi",
            input_text,
            {}}}}}}}}"device": "CPU", "model_type": "text"}  # Device doesn't matter here, stored during load
            )
        
        # Compare results
        if single_device_inference.get())))"status") == "success" and multi_device_inference.get())))"status") == "success":
            single_latency = single_device_inference.get())))"latency_ms", 0)
            multi_latency = multi_device_inference.get())))"latency_ms", 0)
            
            logger.info())))f"Single Device Latency: {}}}}}}}}single_latency:.2f} ms")
            logger.info())))f"Multi-Device Latency: {}}}}}}}}multi_latency:.2f} ms")
            
            # Multi-device performance can vary dramatically based on available hardware
            # and may not always be faster in simple tests

            # Clean up
            backend.unload_model())))f"{}}}}}}}}model_name}_cpu", "CPU")
            backend.unload_model())))f"{}}}}}}}}model_name}_multi", "CPU")
            
            return True
        else:
            logger.error())))"Inference failed for one or both device configurations")
            # Clean up
            backend.unload_model())))f"{}}}}}}}}model_name}_cpu", "CPU")
            backend.unload_model())))f"{}}}}}}}}model_name}_multi", "CPU")
            return False
            
    except Exception as e:
        logger.error())))f"Error during multi-device test: {}}}}}}}}e}")
            return False

def main())))):
    """Command-line entry point."""
    parser = argparse.ArgumentParser())))description='Test enhanced OpenVINO integration with legacy code')
    
    # Test options
    parser.add_argument())))'--test-audio', action='store_true', help='Test audio input processing')
    parser.add_argument())))'--test-mixed-precision', action='store_true', help='Test mixed precision capabilities')
    parser.add_argument())))'--test-multi-device', action='store_true', help='Test multi-device support')
    parser.add_argument())))'--run-all', action='store_true', help='Run all tests')
    
    # Configuration
    parser.add_argument())))'--model', default='bert-base-uncased', help='Model name to use for tests')
    parser.add_argument())))'--device', default='CPU', help='Device to use for tests')
    parser.add_argument())))'--audio-file', default=None, help='Audio file to use for testing')
    
    args = parser.parse_args()))))
    
    # If no tests specified, print help
    if not ())))args.test_audio or args.test_mixed_precision or args.test_multi_device or args.run_all):
        parser.print_help()))))
    return 1
    
    # Run selected tests
    results = {}}}}}}}}}
    
    if args.test_audio or args.run_all:
        logger.info())))"\n=== Running Audio Processing Test ===")
        results['audio_processing'] = test_audio_processing())))args.audio_file, args.device)
        ,
    if args.test_mixed_precision or args.run_all:
        logger.info())))"\n=== Running Mixed Precision Test ===")
        results['mixed_precision'] = test_mixed_precision())))args.model, args.device)
        ,
    if args.test_multi_device or args.run_all:
        logger.info())))"\n=== Running Multi-Device Support Test ===")
        results['multi_device'] = test_multi_device_support())))args.model)
        ,
    # Print summary
        logger.info())))"\n=== Test Results Summary ===")
    for test, result in results.items())))):
        status = "PASSED" if result else "FAILED":
            logger.info())))f"{}}}}}}}}test}: {}}}}}}}}status}")
    
    # Return success if all tests passed
        return 0 if all())))results.values()))))) else 1
:
if __name__ == '__main__':
    sys.exit())))main())))))