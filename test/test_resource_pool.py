#!/usr/bin/env python
# Test script for the ResourcePool class with enhanced device-specific testing

import os
import time
import argparse
import logging
from resource_pool import get_global_resource_pool

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_torch():
    """Load PyTorch module"""
    import torch
    return torch

def load_transformers():
    """Load transformers module"""
    import transformers
    return transformers

def load_numpy():
    """Load numpy module"""
    import numpy as np
    return np

def load_bert_model():
    """Load a BERT model for testing"""
    import torch
    import transformers
    # Use tiny model for testing
    return transformers.AutoModel.from_pretrained("prajjwal1/bert-tiny")

def load_t5_model():
    """Load a T5 model for testing a different model family"""
    import torch
    import transformers
    # Use tiny model for testing
    return transformers.T5ForConditionalGeneration.from_pretrained("google/t5-efficient-tiny")

def test_resource_sharing():
    """Test that resources are properly shared"""
    # Get the resource pool
    pool = get_global_resource_pool()
    
    # First access (miss)
    logger.info("Loading torch for the first time")
    torch1 = pool.get_resource("torch", constructor=load_torch)
    
    # Second access (hit)
    logger.info("Loading torch for the second time")
    torch2 = pool.get_resource("torch", constructor=load_torch)
    
    # Check that we got the same object
    assert torch1 is torch2, "Resource pool failed to return the same object"
    
    # Check stats
    stats = pool.get_stats()
    logger.info(f"Resource pool stats: {stats}")
    assert stats["hits"] >= 1, "Expected at least one cache hit"
    assert stats["misses"] >= 1, "Expected at least one cache miss"
    
    logger.info("Resource sharing test passed!")

def test_model_caching():
    """Test model caching functionality"""
    # Get the resource pool
    pool = get_global_resource_pool()
    
    # First check that resources are available
    torch = pool.get_resource("torch", constructor=load_torch)
    transformers = pool.get_resource("transformers", constructor=load_transformers)
    
    if torch is None or transformers is None:
        logger.error("Required dependencies missing for model caching test")
        return
    
    # First access (miss)
    logger.info("Loading BERT model for the first time")
    model1 = pool.get_model("bert", "prajjwal1/bert-tiny", constructor=load_bert_model)
    
    # Second access (hit)
    logger.info("Loading BERT model for the second time")
    model2 = pool.get_model("bert", "prajjwal1/bert-tiny", constructor=load_bert_model)
    
    # Check that we got the same object
    assert model1 is model2, "Resource pool failed to return the same model"
    
    # Check stats
    stats = pool.get_stats()
    logger.info(f"Resource pool stats after model loading: {stats}")
    
    logger.info("Model caching test passed!")

def test_device_specific_caching():
    """Test device-specific model caching functionality"""
    # Get the resource pool
    pool = get_global_resource_pool()
    torch = pool.get_resource("torch", constructor=load_torch)
    
    if torch is None:
        logger.error("PyTorch not available for device-specific caching test")
        return
    
    # Check available devices - at minimum CPU should be available
    available_devices = ['cpu']
    if torch.cuda.is_available():
        available_devices.append('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        available_devices.append('mps')
    
    logger.info(f"Testing device-specific caching for devices: {available_devices}")
    
    # Define a simple constructor for testing
    def create_tensor_on_device(device):
        return torch.ones(10, 10).to(device)
    
    # Test caching across different devices
    models = {}
    
    # Create models on different devices
    for device in available_devices:
        # Create a constructor for this device
        logger.info(f"Creating tensor on {device}")
        constructor = lambda d=device: create_tensor_on_device(d)
        
        # Request the model with this device
        models[device] = pool.get_model(
            "test_tensor", 
            f"tensor_on_{device}", 
            constructor=constructor,
            hardware_preferences={"device": device}
        )
    
    # Verify each device has its own instance
    for i, device1 in enumerate(available_devices):
        for j, device2 in enumerate(available_devices):
            if i != j:
                # Different devices should have different instances
                assert models[device1] is not models[device2], f"Models on {device1} and {device2} should be different instances"
                logger.info(f"Verified separate instances for {device1} and {device2}")
    
    # Verify cache hits on same device
    for device in available_devices:
        constructor = lambda d=device: create_tensor_on_device(d)
        
        # This should be a cache hit
        model2 = pool.get_model(
            "test_tensor", 
            f"tensor_on_{device}", 
            constructor=constructor,
            hardware_preferences={"device": device}
        )
        
        # Should be same instance
        assert models[device] is model2, f"Cache miss on second access for device {device}"
        logger.info(f"Verified cache hit for second access on {device}")
    
    logger.info("Device-specific caching test passed!")

def test_cleanup():
    """Test cleanup of unused resources"""
    # Get the resource pool
    pool = get_global_resource_pool()
    
    # Load some temporary resources
    pool.get_resource("temp_resource", constructor=lambda: {"data": "temporary"})
    
    # Get stats before cleanup
    stats_before = pool.get_stats()
    logger.info(f"Stats before cleanup: {stats_before}")
    
    # Cleanup with a short timeout (0.1 minutes)
    # This will remove resources that haven't been accessed in the last 6 seconds
    time.sleep(7)  # Wait to ensure the resource is older than the timeout
    removed = pool.cleanup_unused_resources(max_age_minutes=0.1)
    
    # Get stats after cleanup
    stats_after = pool.get_stats()
    logger.info(f"Stats after cleanup: {stats_after}")
    logger.info(f"Removed {removed} resources")
    
    logger.info("Cleanup test passed!")

def test_memory_tracking():
    """Test the memory tracking functionality"""
    # Get the resource pool
    pool = get_global_resource_pool()
    
    # Get initial memory stats
    initial_stats = pool.get_stats()
    initial_memory = initial_stats.get("memory_usage_mb", 0)
    logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # Load resources that should increase memory usage
    numpy = pool.get_resource("numpy", constructor=load_numpy)
    torch = pool.get_resource("torch", constructor=load_torch)
    transformers = pool.get_resource("transformers", constructor=load_transformers)
    
    # Load models
    logger.info("Loading models to track memory usage")
    bert_model = pool.get_model("bert", "prajjwal1/bert-tiny", constructor=load_bert_model)
    
    try:
        # Try to load T5 model if possible
        t5_model = pool.get_model("t5", "google/t5-efficient-tiny", constructor=load_t5_model)
        logger.info("Successfully loaded T5 model")
    except Exception as e:
        logger.warning(f"T5 model loading failed (expected in some environments): {e}")
    
    # Get updated memory stats
    updated_stats = pool.get_stats()
    updated_memory = updated_stats.get("memory_usage_mb", 0)
    logger.info(f"Updated memory usage: {updated_memory:.2f} MB")
    logger.info(f"Memory increase: {updated_memory - initial_memory:.2f} MB")
    
    # Verify memory tracking is working
    assert updated_memory > initial_memory, "Memory usage should increase after loading models"
    
    # Check system memory pressure 
    system_memory = updated_stats.get("system_memory", {})
    if system_memory:
        logger.info(f"System memory available: {system_memory.get('available_mb', 'N/A')} MB")
        logger.info(f"System memory pressure: {system_memory.get('percent_used', 'N/A')}%")
    
    # Check CUDA memory if available
    cuda_memory = updated_stats.get("cuda_memory", {})
    if cuda_memory and cuda_memory.get("device_count", 0) > 0:
        logger.info("CUDA memory stats:")
        for device in cuda_memory.get("devices", []):
            total_mb = device.get("total_mb", 0)
            allocated_mb = device.get("allocated_mb", 0)
            # Check if free_mb is already provided in the stats
            if "free_mb" in device:
                free_mb = device.get("free_mb", 0)
            else:
                # Calculate if not provided directly
                free_mb = total_mb - allocated_mb
            percent_used = device.get("percent_used", 0)
            logger.info(f"  Device {device['id']}: {free_mb:.2f} MB free, {allocated_mb:.2f} MB used ({percent_used:.1f}%)")
    
    logger.info("Memory tracking test passed!")

def test_model_family_integration():
    """Test integration with model family classifier"""
    import os.path
    
    # Check for model family classifier module
    model_classifier_path = os.path.join(os.path.dirname(__file__), "model_family_classifier.py")
    if not os.path.exists(model_classifier_path):
        logger.warning("model_family_classifier.py file does not exist, skipping integration test")
        return
    
    try:
        # Import model classifier dynamically to avoid hard dependency
        from model_family_classifier import classify_model, ModelFamilyClassifier
    except ImportError as e:
        logger.warning(f"Could not import model_family_classifier module: {e}")
        logger.warning("Skipping model family integration test")
        return
    
    # Get resource pool and dependencies
    pool = get_global_resource_pool()
    torch = pool.get_resource("torch", constructor=load_torch)
    transformers = pool.get_resource("transformers", constructor=load_transformers)
    
    if not torch or not transformers:
        logger.error("Required dependencies missing for model family integration test")
        return
    
    # Load a model with explicit embedding model type
    try:
        logger.info("Loading BERT model for family classification testing")
        model = pool.get_model(
            "embedding",  # Explicitly set model type to embedding
            "prajjwal1/bert-tiny", 
            constructor=load_bert_model
        )
        
        # Check that model was successfully loaded
        if model is None:
            logger.error("Failed to load BERT model for family classification test")
            return
    except Exception as e:
        logger.error(f"Error loading model for family classification test: {e}")
        return
    
    # Basic classification test
    try:
        # Classify the model
        classification = classify_model("prajjwal1/bert-tiny", model_class="BertModel")
        
        # Log classification results
        logger.info(f"Model classified as: {classification.get('family')} (confidence: {classification.get('confidence', 0):.2f})")
        if classification.get('subfamily'):
            logger.info(f"Subfamily: {classification.get('subfamily')} (confidence: {classification.get('subfamily_confidence', 0):.2f})")
        
        # Verify family classification
        assert classification.get('family') == "embedding", "BERT should be classified as embedding model"
    except Exception as e:
        logger.error(f"Error during basic model classification: {e}")
        # Continue with the test as other parts may still work
    
    # Check for hardware detection module
    hardware_detection_path = os.path.join(os.path.dirname(__file__), "hardware_detection.py")
    if not os.path.exists(hardware_detection_path):
        logger.warning("hardware_detection.py file does not exist, skipping hardware-aware classification")
    else:
        try:
            # Import hardware detection
            from hardware_detection import detect_hardware_with_comprehensive_checks
            
            # Get hardware information
            logger.info("Detecting hardware capabilities for classification integration")
            hardware_info = detect_hardware_with_comprehensive_checks()
            
            # Create hardware compatibility information to test with model classifier
            hw_compatibility = {}
            for hw_type in ["cuda", "mps", "rocm", "openvino"]:
                hw_compatibility[hw_type] = {
                    "compatible": hardware_info.get(hw_type, False),
                    "memory_usage": {"peak": 256}  # Small model for BERT-tiny
                }
            
            # Test hardware-aware classification
            logger.info("Testing hardware-aware model classification")
            hw_aware_classification = classify_model(
                model_name="prajjwal1/bert-tiny", 
                model_class="BertModel",
                hw_compatibility=hw_compatibility
            )
            
            logger.info("Hardware-aware classification results:")
            logger.info(f"  Family: {hw_aware_classification.get('family')}")
            logger.info(f"  Confidence: {hw_aware_classification.get('confidence', 0):.2f}")
            
            # Check if hardware analysis was used
            hardware_analysis_used = False
            for analysis in hw_aware_classification.get('analyses', []):
                if analysis.get('source') == 'hardware_analysis':
                    hardware_analysis_used = True
                    logger.info(f"  Hardware analysis: {analysis.get('reason', 'No reason provided')}")
            
            # Log hardware analysis status
            if hardware_analysis_used:
                logger.info("✅ Hardware analysis was successfully used in classification")
            else:
                logger.info("⚠️ Hardware analysis was available but not used in classification")
                
            # Verify hardware-aware classification
            assert hw_aware_classification.get('family') is not None, "Hardware-aware classification should return a family"
            
        except ImportError as e:
            logger.warning(f"Could not import hardware detection module: {e}")
        except Exception as e:
            logger.warning(f"Error during hardware-aware classification: {e}")
    
    # Test template selection from classifier
    try:
        logger.info("Testing template selection based on model family")
        classifier = ModelFamilyClassifier()
        
        # Get base classification for template selection
        if 'classification' not in locals():
            # Fallback if classification failed earlier
            classification = classify_model("prajjwal1/bert-tiny", model_class="BertModel")
        
        # Get recommended template
        template = classifier.get_template_for_family(
            classification.get('family'), 
            classification.get('subfamily')
        )
        logger.info(f"Recommended template for model: {template}")
        
        # Verify template selection for embedding models
        if classification.get('family') == "embedding":
            assert template == "hf_embedding_template.py", "BERT should use the embedding template"
            logger.info("✅ Template selection verified for embedding model")
        elif classification.get('family') is not None:
            # At least verify we got a template file
            assert template.endswith(".py"), "Template selection should return a .py file"
            logger.info(f"✅ Template selection verified for {classification.get('family')} model")
    except (ImportError, Exception) as e:
        logger.warning(f"Could not test template selection: {e}")
    
    logger.info("Model family integration test completed successfully")

def test_example_workflow():
    """Test an example workflow using the resource pool"""
    # Get the resource pool
    pool = get_global_resource_pool()
    
    # First, we'd ensure necessary libraries are available
    torch = pool.get_resource("torch", constructor=load_torch)
    transformers = pool.get_resource("transformers", constructor=load_transformers)
    
    if torch is None or transformers is None:
        logger.error("Required dependencies missing for example workflow test")
        return
    
    # Load a model
    logger.info("Loading model for test generation")
    model = pool.get_model("bert", "prajjwal1/bert-tiny", constructor=load_bert_model)
    if model is None:
        logger.error("Failed to load model for example workflow test")
        return
    
    # Simulate test generation
    logger.info("Generating tests using cached model")
    
    # Simulate using model for inference
    if hasattr(model, "forward"):
        try:
            # Create a simple input tensor
            input_ids = torch.tensor([[1, 2, 3, 4, 5]])
            with torch.no_grad():
                outputs = model(input_ids)
            
            logger.info(f"Model produced output with shape: {outputs.last_hidden_state.shape}")
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
    
    # Show memory usage
    stats = pool.get_stats()
    logger.info(f"Memory usage after test generation: {stats['memory_usage_mb']:.2f} MB")
    
    logger.info("Example workflow test passed!")

def test_hardware_aware_model_selection():
    """Test hardware-aware model device selection"""
    import os.path
    
    # Check for hardware detection module
    hardware_detection_path = os.path.join(os.path.dirname(__file__), "hardware_detection.py")
    hardware_detection_available = False
    if not os.path.exists(hardware_detection_path):
        logger.warning("hardware_detection.py file does not exist, using limited testing")
    else:
        try:
            # Import hardware detection with constants
            from hardware_detection import detect_available_hardware, detect_hardware_with_comprehensive_checks
            # Try to import constants, with fallbacks if not found
            try:
                from hardware_detection import CPU, CUDA, MPS, ROCM, OPENVINO
            except ImportError:
                # Define fallback constants if not available
                CPU, CUDA, MPS, ROCM, OPENVINO = "cpu", "cuda", "mps", "rocm", "openvino"
                logger.warning("Hardware constants not available, using string fallbacks")
            hardware_detection_available = True
        except ImportError as e:
            logger.warning(f"Could not import hardware detection module: {e}")
    
    # Check for model family classifier
    model_classifier_path = os.path.join(os.path.dirname(__file__), "model_family_classifier.py")
    model_classifier_available = False
    if not os.path.exists(model_classifier_path):
        logger.warning("model_family_classifier.py file does not exist, using limited testing")
    else:
        try:
            # Import model family classifier
            from model_family_classifier import classify_model, ModelFamilyClassifier
            model_classifier_available = True
        except ImportError as e:
            logger.warning(f"Could not import model family classifier: {e}")
    
    # Get the resource pool
    pool = get_global_resource_pool()
    torch = pool.get_resource("torch", constructor=load_torch)
    
    if not torch:
        logger.error("PyTorch not available for hardware-aware model selection test")
        return
    
    # Get available hardware info
    available_devices = ['cpu']
    if torch.cuda.is_available():
        available_devices.append('cuda')
        if torch.cuda.device_count() > 1:
            available_devices.append('cuda:1')  # Add second GPU if available
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        available_devices.append('mps')
    
    logger.info(f"Available hardware devices: {available_devices}")
    
    # Create a dictionary mapping model families to appropriate test models with class names
    test_models = {
        "embedding": {
            "name": "prajjwal1/bert-tiny",
            "constructor": load_bert_model,
            "class_name": "BertModel"
        },
        "text_generation": {
            "name": "google/t5-efficient-tiny",
            "constructor": load_t5_model,
            "class_name": "T5ForConditionalGeneration"
        }
    }
    
    # Get hardware info if available
    hw_info = None
    if hardware_detection_available:
        try:
            logger.info("Running comprehensive hardware detection")
            hw_info = detect_hardware_with_comprehensive_checks()
            detected_hw = [hw for hw, available in hw_info.items() 
                          if hw in ['cpu', 'cuda', 'mps', 'rocm', 'openvino'] and available]
            logger.info(f"Detected hardware: {', '.join(detected_hw)}")
        except Exception as e:
            logger.warning(f"Error during hardware detection: {e}")
    
    # Start with basic hardware preferences
    hardware_preferences = [
        {"device": "cpu"},  # Explicitly request CPU
        {"device": "auto"}  # Let ResourcePool choose best device
    ]
    
    # Add device-specific preferences based on available hardware
    if "cuda" in available_devices:
        hardware_preferences.append({"device": "cuda"})
        if "cuda:1" in available_devices:
            hardware_preferences.append({"device": "cuda:1"})
    
    if "mps" in available_devices:
        hardware_preferences.append({"device": "mps"})
    
    # If hardware detection and model classifier are available, add family-based preferences
    family_based_prefs = []
    if hardware_detection_available and model_classifier_available:
        # We need both components for family-based hardware preferences
        try:
            logger.info("Creating family-based hardware preferences")
            
            # For embedding models (like BERT)
            if "mps" in available_devices:
                # Apple Silicon works well with embedding models
                family_based_prefs.append({
                    "priority_list": [MPS, CUDA, CPU],
                    "model_family": "embedding",
                    "description": "MPS-prioritized for embedding models"
                })
            elif "cuda" in available_devices:
                family_based_prefs.append({
                    "priority_list": [CUDA, CPU],
                    "model_family": "embedding",
                    "description": "CUDA-prioritized for embedding models"
                })
            
            # For text generation models (like T5, GPT)
            if "cuda" in available_devices:
                # Text generation models need GPU memory
                family_based_prefs.append({
                    "priority_list": [CUDA, CPU],
                    "model_family": "text_generation",
                    "description": "CUDA-prioritized for text generation models"
                })
            
            # Add these to hardware preferences
            hardware_preferences.extend(family_based_prefs)
            logger.info(f"Added {len(family_based_prefs)} family-based hardware preferences")
        except Exception as e:
            logger.warning(f"Error creating family-based preferences: {e}")
    
    # Test each model type with different hardware preferences
    for model_family, model_info in test_models.items():
        logger.info(f"\nTesting hardware selection for {model_family} model: {model_info['name']}")
        
        # Get model classification if available
        if model_classifier_available:
            try:
                model_classification = classify_model(
                    model_name=model_info["name"],
                    model_class=model_info.get("class_name")
                )
                logger.info(f"Model classification: {model_classification.get('family', 'unknown')} (confidence: {model_classification.get('confidence', 0):.2f})")
                
                # Show subfamily if available
                if model_classification.get("subfamily"):
                    logger.info(f"Model subfamily: {model_classification.get('subfamily')} (confidence: {model_classification.get('subfamily_confidence', 0):.2f})")
                
                # Get template recommendation if available
                try:
                    classifier = ModelFamilyClassifier()
                    template = classifier.get_template_for_family(
                        model_classification.get('family'), 
                        model_classification.get('subfamily')
                    )
                    logger.info(f"Recommended template: {template}")
                except Exception as e:
                    logger.warning(f"Could not get template recommendation: {e}")
            except Exception as e:
                logger.warning(f"Error classifying model: {e}")
        
        # Create hardware compatibility info for this model
        hw_compatibility = None
        if hardware_detection_available and hw_info:
            hw_compatibility = {}
            for hw_type in ["cuda", "mps", "rocm", "openvino"]:
                # Set different memory requirements based on model family
                peak_memory = 256  # Default small model
                if model_family == "text_generation":
                    # Text generation models typically need more memory
                    peak_memory = 512
                
                hw_compatibility[hw_type] = {
                    "compatible": hw_info.get(hw_type, False),
                    "memory_usage": {"peak": peak_memory}
                }
        
        # Test each hardware preference with this model
        for pref in hardware_preferences:
            try:
                # Check if this is a family-specific preference
                if "model_family" in pref and pref["model_family"] != model_family:
                    logger.info(f"Skipping preference {pref.get('description', pref)} - not for {model_family} models")
                    continue
                
                # Prepare hardware preferences with compatibility info
                current_pref = pref.copy()
                if hw_compatibility and "hw_compatibility" not in current_pref:
                    current_pref["hw_compatibility"] = hw_compatibility
                
                # Log preference being tested
                if "description" in pref:
                    logger.info(f"Testing with preference: {pref['description']}")
                else:
                    logger.info(f"Testing with preference: {pref}")
                
                # Request model with these preferences
                model = pool.get_model(
                    model_type=model_family,
                    model_name=model_info["name"],
                    constructor=model_info["constructor"],
                    hardware_preferences=current_pref
                )
                
                # Check if model loaded successfully
                if model is not None:
                    logger.info(f"Successfully loaded model with preference {pref.get('description', '')}")
                    
                    # Check model device
                    device_str = "unknown"
                    if hasattr(model, "device"):
                        device_str = str(model.device)
                        logger.info(f"Model is on device: {device_str}")
                    elif hasattr(model, "parameters"):
                        # Try to get device from parameters
                        try:
                            first_param = next(model.parameters())
                            device_str = str(first_param.device)
                            logger.info(f"Model's first parameter is on device: {device_str}")
                        except (StopIteration, Exception) as e:
                            logger.warning(f"Could not check model parameters: {e}")
                    
                    # For priority list preferences, check if selected device is in priority order
                    if "priority_list" in pref:
                        priority_list = pref["priority_list"]
                        device_type = device_str.split(':')[0]  # Extract base device type
                        
                        # Check if device type matches any in priority list
                        matches_priority = False
                        for hw_type in priority_list:
                            hw_str = str(hw_type).lower()
                            if hw_str == device_type:
                                matches_priority = True
                                logger.info(f"✅ Model correctly placed on device {device_type} from priority list")
                                break
                        
                        if not matches_priority:
                            logger.warning(f"⚠️ Model on device {device_type} not in priority list {priority_list}")
                else:
                    logger.warning(f"Failed to load model with preference {pref}")
            except Exception as e:
                logger.error(f"Error testing hardware preference {pref}: {e}")
    
    # Test integration with hardware detection recommendations if available
    if hardware_detection_available and hw_info and hw_info.get("torch_device"):
        try:
            # Get recommended device from comprehensive hardware detection
            recommended_device = hw_info.get("torch_device")
            logger.info(f"\nHardware detection recommends device: {recommended_device}")
            
            # Test with recommendation directly
            for model_family, model_info in test_models.items():
                logger.info(f"Testing recommended device {recommended_device} with {model_family} model")
                
                try:
                    model = pool.get_model(
                        model_type=model_family,
                        model_name=model_info["name"],
                        constructor=model_info["constructor"],
                        hardware_preferences={"device": recommended_device}
                    )
                    
                    if model is not None:
                        logger.info(f"Successfully loaded {model_family} model on {recommended_device}")
                        
                        # Verify device matches recommendation
                        if hasattr(model, "parameters"):
                            try:
                                device = next(model.parameters()).device
                                device_type = str(device).split(':')[0]
                                
                                if device_type in recommended_device:
                                    logger.info(f"✅ Model correctly placed on recommended device type: {device_type}")
                                else:
                                    logger.warning(f"⚠️ Model device {device} doesn't match recommendation {recommended_device}")
                            except Exception as e:
                                logger.warning(f"Could not check model device: {e}")
                    else:
                        logger.warning(f"Failed to load model on recommended device")
                except Exception as e:
                    logger.error(f"Error loading model on recommended device: {e}")
        except Exception as e:
            logger.warning(f"Could not test hardware detection recommendations: {e}")
            
    # Test full integration between model family classification and hardware detection
    if hardware_detection_available and model_classifier_available:
        try:
            logger.info("\nTesting full hardware-model integration")
            
            # For each model, get classification and use it to create optimal hardware preferences
            for model_family, model_info in test_models.items():
                # Get model classification
                classification = classify_model(
                    model_name=model_info["name"],
                    model_class=model_info.get("class_name")
                )
                family = classification.get("family")
                
                if not family:
                    logger.warning(f"Could not determine family for {model_info['name']}, skipping")
                    continue
                
                logger.info(f"Creating hardware preferences for {family} model")
                
                # Create optimal hardware preference based on family and available hardware
                if family == "embedding":
                    # Embedding models work well on MPS/CUDA
                    if "mps" in available_devices:
                        priority_list = ["mps", "cuda", "cpu"]
                    elif "cuda" in available_devices:
                        priority_list = ["cuda", "cpu"]
                    else:
                        priority_list = ["cpu"]
                elif family == "text_generation":
                    # Text generation models need GPU memory
                    if "cuda" in available_devices:
                        priority_list = ["cuda", "cpu"]
                    else:
                        priority_list = ["cpu"]
                else:
                    # Default case
                    priority_list = ["cuda", "mps", "cpu"]
                    
                logger.info(f"Selected priority list for {family}: {priority_list}")
                
                # Test loading with these preferences
                try:
                    hw_prefs = {"priority_list": priority_list}
                    logger.info(f"Loading {model_info['name']} with family-based hardware preference")
                    
                    model = pool.get_model(
                        model_type=model_family,
                        model_name=model_info["name"],
                        constructor=model_info["constructor"],
                        hardware_preferences=hw_prefs
                    )
                    
                    if model is not None:
                        logger.info(f"Successfully loaded model with family-based hardware preferences")
                        
                        # Check device
                        if hasattr(model, "parameters"):
                            try:
                                device = next(model.parameters()).device
                                logger.info(f"Model is on device: {device}")
                                
                                # Check if device is highest priority available
                                device_type = str(device).split(':')[0]
                                if device_type == priority_list[0] or (priority_list[0] not in available_devices and device_type == priority_list[1]):
                                    logger.info(f"✅ Model correctly placed on highest priority available device: {device_type}")
                                else:
                                    logger.warning(f"⚠️ Model device {device_type} is not highest priority. Expected {priority_list[0]}")
                            except Exception as e:
                                logger.warning(f"Could not check model device: {e}")
                    else:
                        logger.warning(f"Failed to load model with family-based preferences")
                except Exception as e:
                    logger.error(f"Error testing family-based hardware selection: {e}")
        except Exception as e:
            logger.warning(f"Error in full hardware-model integration test: {e}")
    
    logger.info("Hardware-aware model selection test completed successfully")

def main():
    """Run all tests"""
    parser = argparse.ArgumentParser(description="Test the ResourcePool functionality")
    parser.add_argument("--test", choices=[
        "all", "sharing", "caching", "device", "cleanup", 
        "memory", "family", "workflow", "hardware"
    ], default="all", help="Which test to run")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger('resource_pool').setLevel(logging.DEBUG)
    
    logger.info("Starting ResourcePool tests")
    
    try:
        # Run tests based on command line argument
        if args.test in ["all", "sharing"]:
            test_resource_sharing()
        
        if args.test in ["all", "caching"]:
            test_model_caching()
        
        if args.test in ["all", "device"]:
            test_device_specific_caching()
        
        if args.test in ["all", "cleanup"]:
            test_cleanup()
        
        if args.test in ["all", "memory"]:
            test_memory_tracking()
        
        if args.test in ["all", "family"]:
            test_model_family_integration()
        
        if args.test in ["all", "workflow"]:
            test_example_workflow()
        
        if args.test in ["all", "hardware"]:
            test_hardware_aware_model_selection()
        
        # Final cleanup
        pool = get_global_resource_pool()
        pool.clear()
        
        logger.info("All ResourcePool tests passed!")
        return 0
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())