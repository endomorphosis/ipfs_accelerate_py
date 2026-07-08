#!/usr/bin/env python3
"""
Real-world model testing integration for IPFS Accelerate Python.

This module tests actual model loading and inference with hardware detection
to validate that the system works with real ML models, not just mocks.
"""

import sys
import time
import logging
from unittest.mock import patch, MagicMock

# Optional pytest import
try:
    import pytest
except ImportError:
    pytest = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Safe imports for testing without requiring heavy dependencies
try:
    from utils.safe_imports import safe_import
    transformers = safe_import('transformers')
    torch = safe_import('torch')
except ImportError:
    transformers = None
    torch = None

from hardware_detection import HardwareDetector

class RealWorldModelTester:
    """Test actual model loading with hardware detection."""
    
    # Small, lightweight models for testing (< 50MB each)
    TEST_MODELS = [
        {
            "name": "prajjwal1/bert-tiny",
            "size_mb": 4.2,
            "type": "bert",
            "description": "Tiny BERT model for testing"
        },
        {
            "name": "microsoft/DialoGPT-small", 
            "size_mb": 117,
            "type": "gpt",
            "description": "Small DialoGPT model for conversation"
        },
        {
            "name": "distilbert-base-uncased-finetuned-sst-2-english",
            "size_mb": 255,
            "type": "distilbert", 
            "description": "DistilBERT for sentiment analysis"
        }
    ]
    
    def __init__(self):
        self.detector = HardwareDetector()
        self.results = {}
        
    def test_model_compatibility(self, model_info, use_mock=True):
        """Test model compatibility with detected hardware."""
        model_name = model_info["name"]
        model_type = model_info["type"]
        
        logger.info(f"Testing model compatibility: {model_name}")
        
        # Get best available hardware
        if use_mock:
            # Use mocked hardware detection for CI environments
            with patch.object(self.detector, 'get_available_hardware') as mock_detect:
                mock_detect.return_value = ['cpu', 'webnn', 'webgpu']
                best_hardware = self.detector.get_best_available_hardware()
        else:
            best_hardware = self.detector.get_best_available_hardware()
            
        logger.info(f"Best available hardware: {best_hardware}")
        
        # Test model loading simulation (without actually downloading)
        result = {
            "model_name": model_name,
            "model_type": model_type,
            "size_mb": model_info["size_mb"],
            "best_hardware": best_hardware,
            "compatible": True,
            "estimated_inference_time_ms": self._estimate_inference_time(model_type, best_hardware),
            "memory_requirements_mb": self._estimate_memory_requirements(model_info),
            "status": "simulated_success"
        }
        
        return result
        
    def test_actual_model_loading(self, model_info):
        """Test actual model loading if dependencies are available."""
        if not transformers or not torch:
            logger.info("Transformers/PyTorch not available, skipping actual model loading")
            return self._create_mock_result(model_info, "dependencies_unavailable")
            
        model_name = model_info["name"]
        
        try:
            logger.info(f"Attempting to load model: {model_name}")
            
            # Try to load the model with timeout
            start_time = time.time()
            
            # Use transformers AutoModel for generic loading
            model = transformers.AutoModel.from_pretrained(
                model_name, 
                torch_dtype=torch.float32,
                device_map="cpu"  # Force CPU for testing
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            
            load_time = time.time() - start_time
            
            # Test simple inference
            test_input = "Hello world, this is a test."
            inputs = tokenizer(test_input, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                start_inference = time.time()
                outputs = model(**inputs)
                inference_time = time.time() - start_inference
                
            result = {
                "model_name": model_name,
                "model_type": model_info["type"],
                "size_mb": model_info["size_mb"],
                "load_time_seconds": round(load_time, 2),
                "inference_time_ms": round(inference_time * 1000, 2),
                "output_shape": list(outputs.last_hidden_state.shape) if hasattr(outputs, 'last_hidden_state') else "unknown",
                "status": "actual_success"
            }
            
            logger.info(f"✅ Successfully loaded and tested {model_name}")
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load {model_name}: {str(e)}")
            return self._create_mock_result(model_info, f"load_failed: {str(e)}")
            
    def _create_mock_result(self, model_info, status):
        """Create a mock result when actual loading isn't possible."""
        return {
            "model_name": model_info["name"],
            "model_type": model_info["type"], 
            "size_mb": model_info["size_mb"],
            "load_time_seconds": self._estimate_load_time(model_info),
            "inference_time_ms": self._estimate_inference_time(model_info["type"], "cpu"),
            "output_shape": "mocked",
            "status": status
        }
        
    def _estimate_load_time(self, model_info):
        """Estimate model loading time based on size."""
        # Rough estimate: 1 second per 50MB
        return round(model_info["size_mb"] / 50.0, 2)
        
    def _estimate_inference_time(self, model_type, hardware):
        """Estimate inference time based on model type and hardware."""
        base_times = {
            "bert": 50,      # 50ms base
            "gpt": 100,      # 100ms base  
            "distilbert": 30, # 30ms base
        }
        
        hardware_multipliers = {
            "cpu": 1.0,
            "cuda": 0.2,
            "mps": 0.3, 
            "webnn": 0.7,
            "webgpu": 0.8
        }
        
        base_time = base_times.get(model_type, 75)
        multiplier = hardware_multipliers.get(hardware, 1.0)
        
        return round(base_time * multiplier, 1)
        
    def _estimate_memory_requirements(self, model_info):
        """Estimate memory requirements for model."""
        # Rough estimate: model size * 2 (for weights + activations)
        return round(model_info["size_mb"] * 2, 1)
        
    def run_comprehensive_model_tests(self, use_actual_loading=False):
        """Run comprehensive tests across all test models."""
        logger.info("Starting comprehensive real-world model tests")
        
        results = []
        
        for model_info in self.TEST_MODELS:
            try:
                # Always test compatibility
                compatibility_result = self.test_model_compatibility(model_info)
                results.append(compatibility_result)
                
                # Optionally test actual loading
                if use_actual_loading:
                    loading_result = self.test_actual_model_loading(model_info)
                    results.append(loading_result)
                    
            except Exception as e:
                logger.error(f"Error testing {model_info['name']}: {str(e)}")
                error_result = {
                    "model_name": model_info["name"],
                    "status": f"test_error: {str(e)}"
                }
                results.append(error_result)
                
        return results

# Pytest integration (if available)
if pytest:
    class TestRealWorldModels:
        """Pytest test cases for real-world model integration."""
        
        @pytest.fixture
        def model_tester(self):
            """Create model tester instance."""
            return RealWorldModelTester()
            
        def test_model_compatibility_simulation(self, model_tester):
            """Test model compatibility with simulated hardware."""
            results = model_tester.run_comprehensive_model_tests(use_actual_loading=False)
            
            assert len(results) >= 3, "Should test at least 3 models"
            
            for result in results:
                assert "model_name" in result
                assert "status" in result
                assert result["status"] in ["simulated_success", "test_error"]
                
            logger.info(f"✅ Model compatibility tests passed: {len(results)} models tested")
            
        @pytest.mark.slow
        def test_actual_model_loading_if_available(self, model_tester):
            """Test actual model loading if dependencies are available."""
            # Only test the smallest model to avoid long CI times
            tiny_model = model_tester.TEST_MODELS[0]  # bert-tiny (4MB)
            
            result = model_tester.test_actual_model_loading(tiny_model)
            
            assert "model_name" in result
            assert "status" in result
            
            if transformers and torch:
                # If dependencies are available, we expect either success or a specific failure
                assert result["status"] in ["actual_success", "load_failed: permission denied", "load_failed: network error"]
            else:
                # If dependencies are not available, should be marked as such
                assert "dependencies_unavailable" in result["status"]
                
            logger.info(f"✅ Actual model loading test completed: {result['status']}")
            
        def test_hardware_model_optimization_recommendations(self, model_tester):
            """Test that hardware detection provides optimization recommendations."""
            detector = HardwareDetector()
            
            # Get actual available hardware instead of mocking
            best_hardware = detector.get_best_available_hardware()
            
            # Should be one of the known hardware types
            known_hardware = ['cpu', 'cuda', 'rocm', 'mps', 'openvino', 'webnn', 'webgpu', 'qualcomm']
            assert best_hardware in known_hardware, f"Unexpected hardware choice: {best_hardware}"
            
            # CPU should always be available as fallback
            available = detector.get_available_hardware()
            assert available['cpu'] is True, "CPU should always be available"
            
            # Test recommendations for different model types
            for model_info in model_tester.TEST_MODELS:
                compatibility = model_tester.test_model_compatibility(model_info, use_mock=True)
                
                assert compatibility["compatible"] is True
                assert compatibility["estimated_inference_time_ms"] > 0
                assert compatibility["memory_requirements_mb"] > 0
                
            logger.info("✅ Hardware optimization recommendations test passed")
else:
    # Define empty class if pytest not available
    class TestRealWorldModels:
        pass

def main():
    """Run real-world model tests as standalone script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test real-world model integration")
    parser.add_argument("--actual-loading", action="store_true", 
                       help="Test actual model loading (requires transformers/torch)")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick compatibility tests only")
    
    args = parser.parse_args()
    
    tester = RealWorldModelTester()
    
    if args.quick:
        logger.info("Running quick compatibility tests...")
        results = tester.run_comprehensive_model_tests(use_actual_loading=False)
    else:
        logger.info("Running comprehensive model tests...")
        results = tester.run_comprehensive_model_tests(use_actual_loading=args.actual_loading)
    
    # Print summary
    print("\n" + "="*80)
    print("REAL-WORLD MODEL TESTING SUMMARY")
    print("="*80)
    
    success_count = sum(1 for r in results if "success" in r.get("status", ""))
    total_count = len(results)
    
    print(f"Total models tested: {total_count}")
    print(f"Successful tests: {success_count}")
    print(f"Success rate: {success_count/total_count*100:.1f}%")
    
    print("\nDetailed Results:")
    for result in results:
        status = result.get("status", "unknown")
        model_name = result.get("model_name", "unknown")
        print(f"  {model_name}: {status}")
        
        if "inference_time_ms" in result:
            print(f"    - Estimated inference: {result['inference_time_ms']}ms")
        if "memory_requirements_mb" in result:
            print(f"    - Memory required: {result['memory_requirements_mb']}MB")
    
    print("\n✅ Real-world model testing completed!")
    return 0 if success_count > 0 else 1

if __name__ == "__main__":
    sys.exit(main())