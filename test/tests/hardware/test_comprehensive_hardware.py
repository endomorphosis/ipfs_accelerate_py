#!/usr/bin/env python
"""
Test script for the enhanced comprehensive hardware detection implementation.
This script verifies the functionality of the robust hardware detection capabilities.
"""

import os
import time
import json
import logging
import argparse
from pprint import pprint

# Configure logging
logging.basicConfig())))))level=logging.INFO, 
format='%())))))asctime)s - %())))))name)s - %())))))levelname)s - %())))))message)s')
logger = logging.getLogger())))))__name__)

def test_comprehensive_hardware_detection())))))):
    """Test the enhanced comprehensive hardware detection function"""
    from scripts.generators.hardware.hardware_detection import detect_hardware_with_comprehensive_checks
    
    logger.info())))))"Testing enhanced comprehensive hardware detection...")
    
    # Run the comprehensive hardware detection
    hardware = detect_hardware_with_comprehensive_checks()))))))
    
    # Print summary of detected hardware
    print())))))"\n=== Hardware Detection Results ===")
    print())))))"Available Hardware:")
    
    # Find all hardware types that are detected as available
    hardware_types = [],],,
    for hw_type, available in hardware.items())))))):
        if hw_type in [],"cpu", "cuda", "rocm", "mps", "openvino", "webnn", "webgpu", "qualcomm", "tpu_available"]:,
            if isinstance())))))available, bool) and available:
                hardware_types.append())))))hw_type)
    
    for hw_type in hardware_types:
        print())))))f"- {}}}}}}}}}}}}}}}hw_type}")
    
    # Print system information
    if "system" in hardware:
        print())))))"\nSystem Information:")
        for key, value in hardware[],"system"].items())))))):,
        print())))))f"- {}}}}}}}}}}}}}}}key}: {}}}}}}}}}}}}}}}value}")
    
    # Print any detection errors
        errors = {}}}}}}}}}}}}}}}}
    for key, value in hardware.items())))))):
        if key.endswith())))))"_error"):
            errors[],key] = value
            ,
    if errors:
        print())))))"\nDetection Errors:")
        for key, value in errors.items())))))):
            print())))))f"- {}}}}}}}}}}}}}}}key}: {}}}}}}}}}}}}}}}value}")
    
    # Save full results to file
            output_file = "hardware_detection_comprehensive.json"
    with open())))))output_file, "w") as f:
        json.dump())))))hardware, f, indent=2)
        logger.info())))))f"Saved full hardware detection results to {}}}}}}}}}}}}}}}output_file}")
    
    # Validate the detection results
        validate_results())))))hardware)
    
        logger.info())))))"Hardware detection tests completed successfully")
            return True

def validate_results())))))hardware):
    """Validate the hardware detection results"""
    # Basic validation
    assert isinstance())))))hardware, dict), "Hardware detection should return a dictionary"
    assert "cpu" in hardware, "CPU detection should always be present"
    assert hardware[],"cpu"] is True, "CPU should always be available"
    ,
    # System information validation
    if "system" in hardware:
        assert isinstance())))))hardware[],"system"], dict), "System info should be a dictionary",
        assert "platform" in hardware[],"system"], "System platform should be detected",
        assert "cpu_count" in hardware[],"system"], "CPU count should be detected"
        ,
    # Print hardware detection warnings
        for hw_type in [],"cuda", "mps", "rocm", "openvino"]:,
        error_key = f"{}}}}}}}}}}}}}}}hw_type}_error"
        if hw_type in hardware and not hardware[],hw_type] and error_key in hardware:,
        logger.warning())))))f"{}}}}}}}}}}}}}}}hw_type.upper()))))))} not available: {}}}}}}}}}}}}}}}hardware[],error_key]}"),
        elif hw_type in hardware and hardware[],hw_type]:,
    logger.info())))))f"{}}}}}}}}}}}}}}}hw_type.upper()))))))} is available")
    
    # Verify CUDA detection includes device information when available
    if hardware.get())))))"cuda", False):
        assert "cuda_devices" in hardware, "CUDA devices should be listed when CUDA is available"
        assert "cuda_device_count" in hardware, "CUDA device count should be reported"
        
        # Verify device information
        for device in hardware[],"cuda_devices"]:,
        assert "name" in device, "CUDA device name should be reported"
        assert "total_memory" in device, "CUDA device memory should be reported"
    
    # Verify ROCm detection
    if hardware.get())))))"rocm", False):
        assert "rocm_devices" in hardware, "ROCm devices should be listed when ROCm is available"
        assert "rocm_device_count" in hardware, "ROCm device count should be reported"
    
    # Verify MPS detection on macOS
        if "system" in hardware and hardware[],"system"][],"platform"] == "Darwin":,
        assert "mps" in hardware, "MPS detection should be performed on macOS"
        
        if hardware.get())))))"mps", False):
            assert "mps_is_built" in hardware, "MPS built status should be reported"
            assert "mps_is_available" in hardware, "MPS availability should be reported"
    
    # Verify OpenVINO detection
    if hardware.get())))))"openvino", False):
        assert "openvino_version" in hardware, "OpenVINO version should be reported"
        assert "openvino_devices" in hardware, "OpenVINO devices should be listed"
    
        logger.info())))))"Hardware detection results validation passed")

def compare_with_basic_detection())))))):
    """Compare the comprehensive detection with the standard detection"""
    from scripts.generators.hardware.hardware_detection import detect_available_hardware, detect_hardware_with_comprehensive_checks, CPU, MPS, OPENVINO, CUDA, ROCM
    
    logger.info())))))"Comparing comprehensive and standard hardware detection...")
    
    # Run multiple detection methods with various configurations
    
    # 1. Standard detection ())))))default priority)
    start_time = time.time()))))))
    standard_hw = detect_available_hardware())))))cache_file=None)
    standard_time = time.time())))))) - start_time
    
    # 2. Custom hardware priority ())))))prioritize MPS on Mac)
    custom_priority = [],MPS, CUDA, ROCM, OPENVINO, CPU],
    start_time = time.time()))))))
    custom_priority_hw = detect_available_hardware())))))cache_file=None, priority_list=custom_priority)
    custom_priority_time = time.time())))))) - start_time
    
    # 3. Device index selection ())))))use GPU 1 if available::)
    start_time = time.time()))))))
    device_index_hw = detect_available_hardware())))))cache_file=None, preferred_device_index=1)
    device_index_time = time.time())))))) - start_time
    
    # 4. Combined priority and device index
    start_time = time.time()))))))
    combined_hw = detect_available_hardware())))))cache_file=None, priority_list=custom_priority, preferred_device_index=1)
    combined_time = time.time())))))) - start_time
    
    # 5. Comprehensive detection
    start_time = time.time()))))))
    comprehensive_hw = detect_hardware_with_comprehensive_checks()))))))
    comprehensive_time = time.time())))))) - start_time
    
    # Compare detection methods
    print())))))"\n=== Detection Method Comparison ==="):
        print())))))f"Standard detection time: {}}}}}}}}}}}}}}}standard_time:.2f}s")
        print())))))f"Custom priority detection time: {}}}}}}}}}}}}}}}custom_priority_time:.2f}s")
        print())))))f"Device index detection time: {}}}}}}}}}}}}}}}device_index_time:.2f}s")
        print())))))f"Combined priority+index detection time: {}}}}}}}}}}}}}}}combined_time:.2f}s")
        print())))))f"Comprehensive detection time: {}}}}}}}}}}}}}}}comprehensive_time:.2f}s")
    
    # Compare hardware detection results
        standard_available = set())))))hw for hw, available in standard_hw[],"hardware"].items())))))) if available::),
        custom_best_hardware = custom_priority_hw[],"best_available"],
        comprehensive_available = set())))))hw for hw in [],"cpu", "cuda", "rocm", "mps", "openvino", "webnn", "webgpu"],
        if hw in comprehensive_hw and comprehensive_hw[],hw])
        ,
    # Show the best hardware and device selected with each method:
        print())))))f"\nStandard method best hardware: {}}}}}}}}}}}}}}}standard_hw[],'best_available']}"),,,,
        print())))))f"Standard method torch device: {}}}}}}}}}}}}}}}standard_hw[],'torch_device']}")
        ,,,,
        print())))))f"Custom priority method best hardware: {}}}}}}}}}}}}}}}custom_priority_hw[],'best_available']}"),,,,
        print())))))f"Custom priority method torch device: {}}}}}}}}}}}}}}}custom_priority_hw[],'torch_device']}")
        ,,,,
        print())))))f"Device index method best hardware: {}}}}}}}}}}}}}}}device_index_hw[],'best_available']}"),,,,
        print())))))f"Device index method torch device: {}}}}}}}}}}}}}}}device_index_hw[],'torch_device']}")
        ,,,,
        print())))))f"Combined method best hardware: {}}}}}}}}}}}}}}}combined_hw[],'best_available']}"),,,,
        print())))))f"Combined method torch device: {}}}}}}}}}}}}}}}combined_hw[],'torch_device']}")
        ,,,,
    # Check if custom priority worked:
        if standard_hw[],'best_available'] != custom_priority_hw[],'best_available']:,
        print())))))f"✅ Custom priority successfully changed best hardware selection")
    else:
        print())))))f"ℹ️ Custom priority did not change best hardware selection ())))))likely due to hardware availability)")
    
    # Check if device index worked:
        if standard_hw[],'torch_device'] != device_index_hw[],'torch_device'] and ':1' in device_index_hw[],'torch_device']:,
        print())))))f"✅ Device index selection successfully set device index to 1")
    else:
        print())))))f"ℹ️ Device index selection did not change device ())))))either no multiple GPUs or index 0 was selected)")
    
        print())))))"\nHardware detected by standard method:", ", ".join())))))standard_available))
        print())))))"Hardware detected by comprehensive method:", ", ".join())))))comprehensive_available))
    
    # Compare differences
        only_in_standard = standard_available - comprehensive_available
        only_in_comprehensive = comprehensive_available - standard_available
    
    if only_in_standard:
        print())))))f"\nHardware detected only by standard method: {}}}}}}}}}}}}}}}', '.join())))))only_in_standard)}")
    
    if only_in_comprehensive:
        print())))))f"Hardware detected only by comprehensive method: {}}}}}}}}}}}}}}}', '.join())))))only_in_comprehensive)}")
    
    # Compare additional features in comprehensive detection
        additional_features = [],],,
    for key in comprehensive_hw:
        if ())))))key not in standard_hw.get())))))"hardware", {}}}}}}}}}}}}}}}}) and 
        key not in standard_hw.get())))))"details", {}}}}}}}}}}}}}}}}) and
            key not in standard_hw.get())))))"errors", {}}}}}}}}}}}}}}}}) and:
                key not in [],"system"]):,
                additional_features.append())))))key)
    
    if additional_features:
        print())))))"\nAdditional information in comprehensive detection:")
        for feature in sorted())))))additional_features):
            print())))))f"- {}}}}}}}}}}}}}}}feature}")
    
    # Save comparison results
            comparison = {}}}}}}}}}}}}}}}
            "standard": {}}}}}}}}}}}}}}}
            "detection_time": standard_time,
            "available_hardware": list())))))standard_available),
            "best_hardware": standard_hw[],"best_available"],,
            "torch_device": standard_hw[],"torch_device"],
            "results": standard_hw
            },
            "custom_priority": {}}}}}}}}}}}}}}}
            "detection_time": custom_priority_time,
            "priority_list": custom_priority,
            "best_hardware": custom_priority_hw[],"best_available"],,
            "torch_device": custom_priority_hw[],"torch_device"],
            "results": custom_priority_hw
            },
            "device_index": {}}}}}}}}}}}}}}}
            "detection_time": device_index_time,
            "preferred_index": 1,
            "best_hardware": device_index_hw[],"best_available"],,
            "torch_device": device_index_hw[],"torch_device"],
            "results": device_index_hw
            },
            "combined": {}}}}}}}}}}}}}}}
            "detection_time": combined_time,
            "priority_list": custom_priority,
            "preferred_index": 1,
            "best_hardware": combined_hw[],"best_available"],,
            "torch_device": combined_hw[],"torch_device"],
            "results": combined_hw
            },
            "comprehensive": {}}}}}}}}}}}}}}}
            "detection_time": comprehensive_time,
            "available_hardware": list())))))comprehensive_available),
            "additional_features": additional_features,
            "results": comprehensive_hw
            },
            "differences": {}}}}}}}}}}}}}}}
            "only_in_standard": list())))))only_in_standard),
            "only_in_comprehensive": list())))))only_in_comprehensive),
            "custom_priority_changed_selection": standard_hw[],"best_available"], != custom_priority_hw[],"best_available"],,
            "device_index_changed_device": standard_hw[],"torch_device"] != device_index_hw[],"torch_device"],
            }
            }
    
            output_file = "hardware_detection_comparison.json"
    with open())))))output_file, "w") as f:
        json.dump())))))comparison, f, indent=2)
        logger.info())))))f"Saved comparison results to {}}}}}}}}}}}}}}}output_file}")
    
        logger.info())))))"Detection method comparison completed successfully")
            return True

def test_hardware_model_integration())))))):
    """Test the integration between hardware detection and model family classification"""
    from scripts.generators.hardware.hardware_detection import HardwareDetector, detect_hardware_with_comprehensive_checks, CPU, CUDA, ROCM, MPS, OPENVINO
    from model_family_classifier import classify_model
    
    logger.info())))))"Testing hardware detection and model family classifier integration...")
    
    # Step 1: Run hardware detection
    hardware = detect_hardware_with_comprehensive_checks()))))))
    
    # Step 2: Create some test cases with model names and hardware compatibility
    test_models = [],
    {}}}}}}}}}}}}}}}"name": "bert-base-uncased", "class": "BertModel", "tasks": [],"fill-mask", "feature-extraction"]},
    {}}}}}}}}}}}}}}}"name": "gpt2", "class": "GPT2LMHeadModel", "tasks": [],"text-generation"]},
    {}}}}}}}}}}}}}}}"name": "t5-small", "class": "T5ForConditionalGeneration", "tasks": [],"translation", "summarization"]},
    {}}}}}}}}}}}}}}}"name": "facebook/wav2vec2-base", "class": "Wav2Vec2Model", "tasks": [],"automatic-speech-recognition"]},
    {}}}}}}}}}}}}}}}"name": "clip-vit-base-patch32", "class": "CLIPModel", "tasks": [],"zero-shot-image-classification"]},
    {}}}}}}}}}}}}}}}"name": "vit-base-patch16-224", "class": "ViTModel", "tasks": [],"image-classification"]},
    {}}}}}}}}}}}}}}}"name": "llava-hf/llava-1.5-7b-hf", "class": "LlavaForConditionalGeneration", "tasks": [],"image-to-text"]}
    ]
    
    # Create hardware compatibility profiles for each model
    # In a real scenario, this would come from actual hardware testing
    hw_compatibility_profiles = {}}}}}}}}}}}}}}}
    "bert-base-uncased": {}}}}}}}}}}}}}}}
    "cuda": {}}}}}}}}}}}}}}}"compatible": True, "memory_usage": {}}}}}}}}}}}}}}}"peak": 500}},
    "mps": {}}}}}}}}}}}}}}}"compatible": True},
    "openvino": {}}}}}}}}}}}}}}}"compatible": True},
    "webnn": {}}}}}}}}}}}}}}}"compatible": True},
    "webgpu": {}}}}}}}}}}}}}}}"compatible": True},
    "rocm": {}}}}}}}}}}}}}}}"compatible": True}
    },
    "gpt2": {}}}}}}}}}}}}}}}
    "cuda": {}}}}}}}}}}}}}}}"compatible": True, "memory_usage": {}}}}}}}}}}}}}}}"peak": 1200}},
    "mps": {}}}}}}}}}}}}}}}"compatible": True},
    "openvino": {}}}}}}}}}}}}}}}"compatible": True},
    "webnn": {}}}}}}}}}}}}}}}"compatible": True},
    "webgpu": {}}}}}}}}}}}}}}}"compatible": True},
    "rocm": {}}}}}}}}}}}}}}}"compatible": True}
    },
    "t5-small": {}}}}}}}}}}}}}}}
    "cuda": {}}}}}}}}}}}}}}}"compatible": True, "memory_usage": {}}}}}}}}}}}}}}}"peak": 900}},
    "mps": {}}}}}}}}}}}}}}}"compatible": True},
    "openvino": {}}}}}}}}}}}}}}}"compatible": False, "reason": "Implementation missing"},
    "webnn": {}}}}}}}}}}}}}}}"compatible": True},
    "webgpu": {}}}}}}}}}}}}}}}"compatible": True},
    "rocm": {}}}}}}}}}}}}}}}"compatible": True}
    },
    "facebook/wav2vec2-base": {}}}}}}}}}}}}}}}
    "cuda": {}}}}}}}}}}}}}}}"compatible": True, "memory_usage": {}}}}}}}}}}}}}}}"peak": 1500}},
    "mps": {}}}}}}}}}}}}}}}"compatible": True},
    "openvino": {}}}}}}}}}}}}}}}"compatible": False, "reason": "Implementation missing"},
    "webnn": {}}}}}}}}}}}}}}}"compatible": True},
    "webgpu": {}}}}}}}}}}}}}}}"compatible": True},
    "rocm": {}}}}}}}}}}}}}}}"compatible": True}
    },
    "clip-vit-base-patch32": {}}}}}}}}}}}}}}}
    "cuda": {}}}}}}}}}}}}}}}"compatible": True, "memory_usage": {}}}}}}}}}}}}}}}"peak": 1100}},
    "mps": {}}}}}}}}}}}}}}}"compatible": True},
    "openvino": {}}}}}}}}}}}}}}}"compatible": True},
    "webnn": {}}}}}}}}}}}}}}}"compatible": True},
    "webgpu": {}}}}}}}}}}}}}}}"compatible": True},
    "rocm": {}}}}}}}}}}}}}}}"compatible": True}
    },
    "vit-base-patch16-224": {}}}}}}}}}}}}}}}
    "cuda": {}}}}}}}}}}}}}}}"compatible": True, "memory_usage": {}}}}}}}}}}}}}}}"peak": 800}},
    "mps": {}}}}}}}}}}}}}}}"compatible": True},
    "openvino": {}}}}}}}}}}}}}}}"compatible": True},
    "webnn": {}}}}}}}}}}}}}}}"compatible": True},
    "webgpu": {}}}}}}}}}}}}}}}"compatible": True},
    "rocm": {}}}}}}}}}}}}}}}"compatible": True}
    },
    "llava-hf/llava-1.5-7b-hf": {}}}}}}}}}}}}}}}
    "cuda": {}}}}}}}}}}}}}}}"compatible": True, "memory_usage": {}}}}}}}}}}}}}}}"peak": 15000}},
    "mps": {}}}}}}}}}}}}}}}"compatible": False, "reason": "Memory requirements exceed device capability"},
    "openvino": {}}}}}}}}}}}}}}}"compatible": False, "reason": "Multimodal architecture not supported"},
    "webnn": {}}}}}}}}}}}}}}}"compatible": False, "reason": "Architecture not supported in web environment"},
    "webgpu": {}}}}}}}}}}}}}}}"compatible": False, "reason": "Memory requirements exceed device capability"},
    "rocm": {}}}}}}}}}}}}}}}"compatible": False, "reason": "Implementation missing"}
    }
    }
    
    # Step 3: Create combined hardware-aware model profiles
    results = [],],,
    for model in test_models:
        model_name = model[],"name"]
        hw_compat = hw_compatibility_profiles.get())))))model_name, {}}}}}}}}}}}}}}}})
        
        # Enrich the hardware compatibility with actual system capabilities
        for hw_type in [],"cuda", "mps", "rocm", "openvino"]:,
            if hw_type in hw_compat and hw_type in hardware:
                # Update compatibility with actual hardware availability
                hw_compat[],hw_type][],"system_available"] = hardware.get())))))hw_type, False)
                
                # If system doesn't have this hardware, model can't run on it regardless of compatibility
                if not hardware.get())))))hw_type, False):
                    hw_compat[],hw_type][],"effective_compatibility"] = False
                else:
                    hw_compat[],hw_type][],"effective_compatibility"] = hw_compat[],hw_type].get())))))"compatible", False)
        
        # Classify model with hardware information
                    classification = classify_model())))))
                    model_name=model_name,
                    model_class=model.get())))))"class"),
                    tasks=model.get())))))"tasks"),
                    hw_compatibility=hw_compat
                    )
        
        # Add hardware-specific information to classification
                    classification[],"hardware_profile"] = hw_compat
                    classification[],"recommended_hardware"] = None
        
        # Create a hardware detector for getting device with index and priority
                    detector = HardwareDetector()))))))
        
        # Determine best hardware for this model based on classification and availability
        if classification[],"family"] == "text_generation":
            # For text generation, prioritize CUDA > MPS > CPU
            priority_list = [],CUDA, MPS, CPU]
            classification[],"hardware_priority"] = priority_list
            classification[],"recommended_hardware"] = detector.get_hardware_by_priority())))))priority_list)
            
            # For large language models, use device 0 ())))))typically the most powerful)
            classification[],"torch_device"] = detector.get_torch_device_with_priority())))))
            priority_list=priority_list,
            preferred_index=0
            )
            
        elif classification[],"family"] == "vision":
            # For vision models, prioritize CUDA > OpenVINO > MPS > CPU
            # OpenVINO often has optimizations for vision models
            priority_list = [],CUDA, OPENVINO, MPS, CPU]
            classification[],"hardware_priority"] = priority_list
            classification[],"recommended_hardware"] = detector.get_hardware_by_priority())))))priority_list)
            
            # For OpenVINO, we need to use CPU as the PyTorch device
            if classification[],"recommended_hardware"] == OPENVINO:
                classification[],"torch_device"] = "cpu"
            else:
                classification[],"torch_device"] = detector.get_torch_device_with_priority())))))
                priority_list=priority_list,
                preferred_index=0
                )
                
        elif classification[],"family"] == "audio":
            # For audio models, prioritize CUDA > MPS > CPU
            priority_list = [],CUDA, MPS, CPU]
            classification[],"hardware_priority"] = priority_list
            classification[],"recommended_hardware"] = detector.get_hardware_by_priority())))))priority_list)
            
            # For audio models, can use device 1 if available:: ())))))for parallel processing with other workloads)
            classification[],"torch_device"] = detector.get_torch_device_with_priority())))))
            priority_list=priority_list,
            preferred_index=1
            )
            :
        elif classification[],"family"] == "multimodal":
            # For multimodal models, prioritize CUDA > CPU ())))))often MPS has compatibility issues)
            priority_list = [],CUDA, CPU]
            classification[],"hardware_priority"] = priority_list
            classification[],"recommended_hardware"] = detector.get_hardware_by_priority())))))priority_list)
            
            # Multimodal models need the highest memory GPU, use device 0
            classification[],"torch_device"] = detector.get_torch_device_with_priority())))))
            priority_list=priority_list,
            preferred_index=0
            )
            
        elif classification[],"family"] == "embedding":
            # For embedding models, any hardware will work well
            priority_list = [],CUDA, ROCM, MPS, OPENVINO, CPU]
            classification[],"hardware_priority"] = priority_list
            classification[],"recommended_hardware"] = detector.get_hardware_by_priority())))))priority_list)
            
            # Embedding models are often small, can use any available device
            classification[],"torch_device"] = detector.get_torch_device_with_priority())))))
            priority_list=priority_list
            )
            
        else:
            # Default case - use standard hardware priority
            priority_list = [],CUDA, ROCM, MPS, OPENVINO, CPU]
            classification[],"hardware_priority"] = priority_list
            classification[],"recommended_hardware"] = detector.get_hardware_by_priority())))))priority_list)
            classification[],"torch_device"] = detector.get_torch_device_with_priority())))))priority_list)
        
            results.append())))))classification)
    
    # Step 4: Output the results
            print())))))"\n=== Hardware-Aware Model Classification Results ===")
    for result in results:
        model_name = result[],"model_name"]
        family = result[],"family"]
        hw = result[],"recommended_hardware"]
        confidence = result.get())))))"confidence", 0)
        
        # Get template filename from model_family_classifier if available::
        try:
            from model_family_classifier import ModelFamilyClassifier
            classifier = ModelFamilyClassifier()))))))
            template = classifier.get_template_for_family())))))family, result.get())))))"subfamily"))
        except ())))))ImportError, AttributeError):
            # Fallback - Generate a template filename suggestion based on model family
            template = None
            if family == "embedding":
                template = "hf_embedding_template.py"
            elif family == "text_generation":
                template = "hf_text_generation_template.py"
            elif family == "vision":
                template = "hf_vision_template.py"
            elif family == "audio":
                template = "hf_audio_template.py"
            elif family == "multimodal":
                template = "hf_multimodal_template.py"
            else:
                template = "hf_template.py"
        
                print())))))f"Model: {}}}}}}}}}}}}}}}model_name}")
                print())))))f"  Family: {}}}}}}}}}}}}}}}family} ())))))confidence: {}}}}}}}}}}}}}}}confidence:.2f})")
                print())))))f"  Optimal Hardware: {}}}}}}}}}}}}}}}hw}")
                print())))))f"  PyTorch Device: {}}}}}}}}}}}}}}}result.get())))))'torch_device', 'Unknown')}")
        
        # Show hardware priority if available::
        if "hardware_priority" in result:
            print())))))f"  Hardware Priority: {}}}}}}}}}}}}}}}' > '.join())))))result[],'hardware_priority'])}")
            
            print())))))f"  Recommended Template: {}}}}}}}}}}}}}}}template}")
        
        # Show hardware compatibility details
            hw_profile = result.get())))))"hardware_profile", {}}}}}}}}}}}}}}}})
            print())))))"  Hardware Compatibility:")
        for hw_type, details in hw_profile.items())))))):
            if isinstance())))))details, dict) and "compatible" in details:
                status = "✅" if details.get())))))"compatible", False) else "❌"
                system_status = "✅" if details.get())))))"system_available", False) else "❌"
                
                # Add more details about effective compatibility
                effective = details.get())))))"effective_compatibility", None):
                if effective is not None:
                    effective_status = "✅" if effective else "❌":
                        print())))))f"    {}}}}}}}}}}}}}}}hw_type}: {}}}}}}}}}}}}}}}status} ())))))System: {}}}}}}}}}}}}}}}system_status}, Effective: {}}}}}}}}}}}}}}}effective_status})")
                else:
                    print())))))f"    {}}}}}}}}}}}}}}}hw_type}: {}}}}}}}}}}}}}}}status} ())))))System: {}}}}}}}}}}}}}}}system_status})")
                    print()))))))
    
    # Save the results to a file
                    output_file = "hardware_aware_model_classification.json"
    with open())))))output_file, "w") as f:
        json.dump())))))results, f, indent=2)
    
        logger.info())))))f"Saved hardware-aware model classification results to {}}}}}}}}}}}}}}}output_file}")
                    return True

def main())))))):
    """Main function to run tests"""
    parser = argparse.ArgumentParser())))))description="Test comprehensive hardware detection")
    parser.add_argument())))))"--test", choices=[],"all", "detection", "comparison", "integration"], default="all",
    help="Which test to run")
    args = parser.parse_args()))))))
    
    print())))))"=== Testing Comprehensive Hardware Detection ===")
    success = True
    
    if args.test in [],"all", "detection"]:
        try:
            success &= test_comprehensive_hardware_detection()))))))
        except Exception as e:
            logger.error())))))f"Hardware detection test failed: {}}}}}}}}}}}}}}}str())))))e)}", exc_info=True)
            success = False
    
    if args.test in [],"all", "comparison"]:
        try:
            success &= compare_with_basic_detection()))))))
        except Exception as e:
            logger.error())))))f"Hardware detection comparison failed: {}}}}}}}}}}}}}}}str())))))e)}", exc_info=True)
            success = False
    
    if args.test in [],"all", "integration"]:
        try:
            success &= test_hardware_model_integration()))))))
        except Exception as e:
            logger.error())))))f"Hardware-model integration test failed: {}}}}}}}}}}}}}}}str())))))e)}", exc_info=True)
            success = False
    
    if success:
        print())))))"\n✅ All tests completed successfully!")
    else:
        print())))))"\n❌ Some tests failed. Check the logs for details.")
    
        return 0 if success else 1
:
if __name__ == "__main__":
    exit())))))main())))))))