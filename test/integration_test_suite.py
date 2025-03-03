#!/usr/bin/env python3
"""
Comprehensive Integration Test Suite for IPFS Accelerate Python

This test suite verifies that all components of the system work together
properly across different hardware platforms, model types, and APIs.
"""

import os
import sys
import json
import time
import datetime
import argparse
import unittest
import logging
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import traceback
import importlib

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 
                                        f"integration_test_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    ]
)
logger = logging.getLogger("integration_test")

# Try to import necessary modules
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.warning("NumPy not available")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    logger.warning("tqdm not available, progress bars will be disabled")

# Define integration test categories
INTEGRATION_CATEGORIES = [
    "hardware_detection",
    "resource_pool",
    "model_loading",
    "api_backends",
    "web_platforms",
    "multimodal",
    "endpoint_lifecycle",
    "batch_processing",
    "queue_management",
    "hardware_compatibility",  # New category for automated hardware compatibility testing
    "cross_platform"           # New category for cross-platform validation
]

@dataclass
class TestResult:
    """Class to store a single test result"""
    category: str
    test_name: str
    status: str  # "pass", "fail", "skip", "error"
    execution_time: float = 0.0
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    hardware_platform: Optional[str] = None
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert test result to a dictionary for JSON serialization"""
        return {
            "category": self.category,
            "test_name": self.test_name,
            "status": self.status,
            "execution_time": round(self.execution_time, 3),
            "error_message": self.error_message,
            "details": self.details,
            "hardware_platform": self.hardware_platform,
            "timestamp": datetime.datetime.now().isoformat()
        }


@dataclass
class TestSuiteResults:
    """Class to store all test results from a test suite run"""
    results: List[TestResult] = field(default_factory=list)
    start_time: datetime.datetime = field(default_factory=datetime.datetime.now)
    end_time: Optional[datetime.datetime] = None
    
    def add_result(self, result: TestResult) -> None:
        """Add a test result to the collection"""
        self.results.append(result)
    
    def finish(self) -> None:
        """Mark the test suite as finished and record the end time"""
        self.end_time = datetime.datetime.now()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the test results"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == "pass")
        failed = sum(1 for r in self.results if r.status == "fail")
        skipped = sum(1 for r in self.results if r.status == "skip")
        errors = sum(1 for r in self.results if r.status == "error")
        
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 0}
            
            categories[result.category]["total"] += 1
            if result.status == "pass":
                categories[result.category]["passed"] += 1
            elif result.status == "fail":
                categories[result.category]["failed"] += 1
            elif result.status == "skip":
                categories[result.category]["skipped"] += 1
            elif result.status == "error":
                categories[result.category]["errors"] += 1
        
        execution_time = 0
        if self.end_time:
            execution_time = (self.end_time - self.start_time).total_seconds()
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "errors": errors,
            "pass_rate": passed / total if total > 0 else 0,
            "categories": categories,
            "execution_time": execution_time,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None
        }
    
    def save_results(self, filename: str) -> None:
        """Save the test results to a JSON file"""
        data = {
            "summary": self.get_summary(),
            "results": [r.as_dict() for r in self.results]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Test results saved to {filename}")
    
    def print_summary(self) -> None:
        """Print a summary of the test results"""
        summary = self.get_summary()
        
        print("\n===== INTEGRATION TEST RESULTS =====")
        print(f"Total tests: {summary['total']}")
        print(f"Passed: {summary['passed']} ({summary['pass_rate']:.1%})")
        print(f"Failed: {summary['failed']}")
        print(f"Skipped: {summary['skipped']}")
        print(f"Errors: {summary['errors']}")
        print(f"Execution time: {summary['execution_time']:.2f} seconds")
        
        print("\nResults by category:")
        for category, stats in summary['categories'].items():
            print(f"  {category}: {stats['passed']}/{stats['total']} passed ({stats['passed']/stats['total']:.1%})")
            
        if summary['failed'] > 0 or summary['errors'] > 0:
            print("\nFailed tests:")
            for result in self.results:
                if result.status in ["fail", "error"]:
                    print(f"  {result.category} - {result.test_name}: {result.error_message}")


class IntegrationTestSuite:
    """Comprehensive integration test suite for IPFS Accelerate Python"""
    
    def __init__(self, 
                categories: Optional[List[str]] = None,
                hardware_platforms: Optional[List[str]] = None,
                timeout: int = 300,
                skip_slow_tests: bool = False):
        """Initialize the test suite"""
        self.categories = categories or INTEGRATION_CATEGORIES
        self.hardware_platforms = hardware_platforms or self._detect_available_hardware()
        self.timeout = timeout
        self.skip_slow_tests = skip_slow_tests
        self.results = TestSuiteResults()
        
        # Import test modules
        self.test_modules = self._import_test_modules()
        
        # Set up paths for results
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.results_dir = os.path.join(self.test_dir, "integration_results")
        os.makedirs(self.results_dir, exist_ok=True)
    
    def _detect_available_hardware(self) -> List[str]:
        """Detect available hardware platforms"""
        hardware = ["cpu"]
        
        # Check for CUDA
        if HAS_TORCH and torch.cuda.is_available():
            hardware.append("cuda")
        
        # Check for MPS (Apple Silicon)
        if HAS_TORCH and hasattr(torch, "mps") and torch.backends.mps.is_available():
            hardware.append("mps")
        
        # Check for ROCm
        try:
            if HAS_TORCH and hasattr(torch.utils, "hip") and torch.utils.hip.is_available():
                hardware.append("rocm")
        except:
            pass
        
        # Check for OpenVINO
        try:
            import openvino
            hardware.append("openvino")
        except ImportError:
            pass
        
        # Web platforms are always included in simulation mode
        hardware.extend(["webnn", "webgpu"])
        
        return hardware
    
    def _import_test_modules(self) -> Dict[str, Any]:
        """Import test modules for the integration test suite"""
        modules = {}
        
        # Import test_comprehensive_hardware for hardware detection tests
        try:
            modules["hardware_detection"] = importlib.import_module("test.test_comprehensive_hardware")
            logger.info("Imported hardware detection module")
        except ImportError as e:
            logger.warning(f"Could not import hardware detection module: {e}")
        
        # Import test_resource_pool for resource pool tests
        try:
            modules["resource_pool"] = importlib.import_module("test.test_resource_pool")
            logger.info("Imported resource pool module")
        except ImportError as e:
            logger.warning(f"Could not import resource pool module: {e}")
        
        # Import test_api_backend for API backend tests
        try:
            modules["api_backends"] = importlib.import_module("test.test_api_backend")
            logger.info("Imported API backend module")
        except ImportError as e:
            logger.warning(f"Could not import API backend module: {e}")
        
        # Import web platform testing module
        try:
            modules["web_platforms"] = importlib.import_module("test.web_platform_testing")
            logger.info("Imported web platform testing module")
        except ImportError as e:
            logger.warning(f"Could not import web platform testing module: {e}")
        
        # Import endpoint lifecycle test module
        try:
            modules["endpoint_lifecycle"] = importlib.import_module("test.test_endpoint_lifecycle")
            logger.info("Imported endpoint lifecycle module")
        except ImportError as e:
            logger.warning(f"Could not import endpoint lifecycle module: {e}")
        
        # Import batch inference test module
        try:
            modules["batch_processing"] = importlib.import_module("test.test_batch_inference")
            logger.info("Imported batch inference module")
        except ImportError as e:
            logger.warning(f"Could not import batch inference module: {e}")
        
        # Import queue management test module
        try:
            modules["queue_management"] = importlib.import_module("test.test_api_backoff_queue")
            logger.info("Imported queue management module")
        except ImportError as e:
            logger.warning(f"Could not import queue management module: {e}")
        
        return modules
    
    def _run_hardware_detection_tests(self) -> None:
        """Run hardware detection integration tests"""
        category = "hardware_detection"
        
        if category not in self.categories:
            logger.info(f"Skipping {category} tests (not in selected categories)")
            return
        
        if "hardware_detection" not in self.test_modules:
            logger.warning(f"Skipping {category} tests (module not imported)")
            return
        
        logger.info(f"Running {category} integration tests")
        
        # Test hardware detection functionality
        test_name = "test_detect_all_hardware"
        start_time = time.time()
        
        try:
            module = self.test_modules["hardware_detection"]
            
            # Create a detector instance
            if hasattr(module, "HardwareDetector"):
                detector = module.HardwareDetector()
                hardware_info = detector.detect_all()
                
                # Verify that hardware detection returns expected structure
                if not isinstance(hardware_info, dict):
                    raise ValueError("Hardware detection did not return a dictionary")
                
                if "cpu" not in hardware_info:
                    raise ValueError("CPU info missing from hardware detection")
                
                # Test passed
                end_time = time.time()
                self.results.add_result(TestResult(
                    category=category,
                    test_name=test_name,
                    status="pass",
                    execution_time=end_time - start_time,
                    details={"detected_hardware": list(hardware_info.keys())}
                ))
                logger.info(f"✓ {test_name} passed")
                
            else:
                # If no HardwareDetector class found, try the functional approach
                if hasattr(module, "detect_all_hardware"):
                    hardware_info = module.detect_all_hardware()
                    
                    # Verify that hardware detection returns expected structure
                    if not isinstance(hardware_info, dict):
                        raise ValueError("Hardware detection did not return a dictionary")
                    
                    if "cpu" not in hardware_info:
                        raise ValueError("CPU info missing from hardware detection")
                    
                    # Test passed
                    end_time = time.time()
                    self.results.add_result(TestResult(
                        category=category,
                        test_name=test_name,
                        status="pass",
                        execution_time=end_time - start_time,
                        details={"detected_hardware": list(hardware_info.keys())}
                    ))
                    logger.info(f"✓ {test_name} passed")
                else:
                    raise ImportError("No hardware detection functionality found")
                
        except Exception as e:
            end_time = time.time()
            self.results.add_result(TestResult(
                category=category,
                test_name=test_name,
                status="error",
                execution_time=end_time - start_time,
                error_message=str(e),
                details={"traceback": traceback.format_exc()}
            ))
            logger.error(f"✗ {test_name} failed: {str(e)}")
        
        # Test hardware-specific detection for each platform
        for platform in self.hardware_platforms:
            test_name = f"test_detect_{platform}_hardware"
            start_time = time.time()
            
            try:
                module = self.test_modules["hardware_detection"]
                
                # Skip web platforms for individual hardware tests
                if platform in ["webnn", "webgpu"]:
                    self.results.add_result(TestResult(
                        category=category,
                        test_name=test_name,
                        status="skip",
                        execution_time=0,
                        hardware_platform=platform,
                        details={"reason": "Web platforms not tested individually"}
                    ))
                    continue
                
                # Create a detector instance
                if hasattr(module, "HardwareDetector"):
                    detector = module.HardwareDetector()
                    
                    # Call the appropriate detection method
                    if platform == "cpu":
                        info = detector.detect_cpu()
                    elif platform == "cuda":
                        info = detector.detect_cuda()
                    elif platform == "mps":
                        info = detector.detect_mps()
                    elif platform == "rocm":
                        info = detector.detect_rocm()
                    elif platform == "openvino":
                        info = detector.detect_openvino()
                    else:
                        raise ValueError(f"Unknown hardware platform: {platform}")
                    
                    # Test passed
                    end_time = time.time()
                    self.results.add_result(TestResult(
                        category=category,
                        test_name=test_name,
                        status="pass",
                        execution_time=end_time - start_time,
                        hardware_platform=platform,
                        details={"info": str(info)}
                    ))
                    logger.info(f"✓ {test_name} passed")
                    
                else:
                    # If no HardwareDetector class found, try the functional approach
                    if hasattr(module, f"detect_{platform}"):
                        detect_func = getattr(module, f"detect_{platform}")
                        info = detect_func()
                        
                        # Test passed
                        end_time = time.time()
                        self.results.add_result(TestResult(
                            category=category,
                            test_name=test_name,
                            status="pass",
                            execution_time=end_time - start_time,
                            hardware_platform=platform,
                            details={"info": str(info)}
                        ))
                        logger.info(f"✓ {test_name} passed")
                    else:
                        raise ImportError(f"No hardware detection function for {platform}")
                    
            except Exception as e:
                end_time = time.time()
                self.results.add_result(TestResult(
                    category=category,
                    test_name=test_name,
                    status="error",
                    execution_time=end_time - start_time,
                    hardware_platform=platform,
                    error_message=str(e),
                    details={"traceback": traceback.format_exc()}
                ))
                logger.error(f"✗ {test_name} failed: {str(e)}")
    
    def _run_resource_pool_tests(self) -> None:
        """Run resource pool integration tests"""
        category = "resource_pool"
        
        if category not in self.categories:
            logger.info(f"Skipping {category} tests (not in selected categories)")
            return
        
        if "resource_pool" not in self.test_modules:
            logger.warning(f"Skipping {category} tests (module not imported)")
            return
        
        logger.info(f"Running {category} integration tests")
        
        # Test ResourcePool initialization
        test_name = "test_resource_pool_init"
        start_time = time.time()
        
        try:
            module = self.test_modules["resource_pool"]
            
            # Import ResourcePool class
            if hasattr(module, "ResourcePool"):
                ResourcePool = module.ResourcePool
                
                # Create a resource pool instance
                pool = ResourcePool()
                
                # Verify that pool is correctly initialized
                if not hasattr(pool, "get_device"):
                    raise AttributeError("ResourcePool missing get_device method")
                
                if not hasattr(pool, "allocate"):
                    raise AttributeError("ResourcePool missing allocate method")
                
                if not hasattr(pool, "release"):
                    raise AttributeError("ResourcePool missing release method")
                
                # Test passed
                end_time = time.time()
                self.results.add_result(TestResult(
                    category=category,
                    test_name=test_name,
                    status="pass",
                    execution_time=end_time - start_time
                ))
                logger.info(f"✓ {test_name} passed")
                
            else:
                raise ImportError("ResourcePool class not found")
                
        except Exception as e:
            end_time = time.time()
            self.results.add_result(TestResult(
                category=category,
                test_name=test_name,
                status="error",
                execution_time=end_time - start_time,
                error_message=str(e),
                details={"traceback": traceback.format_exc()}
            ))
            logger.error(f"✗ {test_name} failed: {str(e)}")
        
        # Test device allocation
        test_name = "test_resource_pool_device_allocation"
        start_time = time.time()
        
        try:
            module = self.test_modules["resource_pool"]
            
            # Import ResourcePool class
            if hasattr(module, "ResourcePool"):
                ResourcePool = module.ResourcePool
                
                # Create a resource pool instance
                pool = ResourcePool()
                
                # Allocate CPU device
                cpu_device = pool.get_device(device_type="cpu")
                
                # Check that the device exists
                if cpu_device is None:
                    raise ValueError("Could not allocate CPU device")
                
                # Release the device
                pool.release(cpu_device)
                
                # Test passed
                end_time = time.time()
                self.results.add_result(TestResult(
                    category=category,
                    test_name=test_name,
                    status="pass",
                    execution_time=end_time - start_time,
                    details={"device": str(cpu_device)}
                ))
                logger.info(f"✓ {test_name} passed")
                
            else:
                raise ImportError("ResourcePool class not found")
                
        except Exception as e:
            end_time = time.time()
            self.results.add_result(TestResult(
                category=category,
                test_name=test_name,
                status="error",
                execution_time=end_time - start_time,
                error_message=str(e),
                details={"traceback": traceback.format_exc()}
            ))
            logger.error(f"✗ {test_name} failed: {str(e)}")
        
        # Test model family integration with resource pool
        test_name = "test_resource_pool_model_family"
        start_time = time.time()
        
        try:
            # Skip if model_family_classifier is not available
            try:
                model_family_module = importlib.import_module("test.model_family_classifier")
                logger.info("Imported model family classifier module")
            except ImportError:
                logger.warning("Skipping model family test (module not imported)")
                self.results.add_result(TestResult(
                    category=category,
                    test_name=test_name,
                    status="skip",
                    details={"reason": "model_family_classifier not available"}
                ))
                return
            
            module = self.test_modules["resource_pool"]
            
            # Import ResourcePool class
            if hasattr(module, "ResourcePool"):
                ResourcePool = module.ResourcePool
                
                # Create a resource pool instance with model family integration
                pool = ResourcePool(use_model_family=True)
                
                # Get device for text model family
                text_device = pool.get_device(model_family="text")
                
                # Check that the device exists
                if text_device is None:
                    raise ValueError("Could not allocate device for text model family")
                
                # Release the device
                pool.release(text_device)
                
                # Test passed
                end_time = time.time()
                self.results.add_result(TestResult(
                    category=category,
                    test_name=test_name,
                    status="pass",
                    execution_time=end_time - start_time,
                    details={"device": str(text_device)}
                ))
                logger.info(f"✓ {test_name} passed")
                
            else:
                raise ImportError("ResourcePool class not found")
                
        except Exception as e:
            end_time = time.time()
            self.results.add_result(TestResult(
                category=category,
                test_name=test_name,
                status="error",
                execution_time=end_time - start_time,
                error_message=str(e),
                details={"traceback": traceback.format_exc()}
            ))
            logger.error(f"✗ {test_name} failed: {str(e)}")
    
    def _run_model_loading_tests(self) -> None:
        """Run model loading integration tests"""
        category = "model_loading"
        
        if category not in self.categories:
            logger.info(f"Skipping {category} tests (not in selected categories)")
            return
        
        logger.info(f"Running {category} integration tests")
        
        # Skip if torch is not available
        if not HAS_TORCH:
            logger.warning("Skipping model loading tests (torch not available)")
            self.results.add_result(TestResult(
                category=category,
                test_name="test_model_loading",
                status="skip",
                details={"reason": "torch not available"}
            ))
            return
        
        # Try to import transformers
        try:
            import transformers
            from transformers import AutoTokenizer, AutoModel
            logger.info("Imported transformers module")
        except ImportError:
            logger.warning("Skipping model loading tests (transformers not available)")
            self.results.add_result(TestResult(
                category=category,
                test_name="test_model_loading",
                status="skip",
                details={"reason": "transformers not available"}
            ))
            return
        
        # Test basic model loading
        test_name = "test_basic_model_loading"
        start_time = time.time()
        
        try:
            # Use a small model for testing
            model_name = "prajjwal1/bert-tiny"
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            # Verify model and tokenizer
            assert tokenizer is not None, "Tokenizer is None"
            assert model is not None, "Model is None"
            
            # Test tokenizer
            tokens = tokenizer("Hello world", return_tensors="pt")
            assert "input_ids" in tokens, "Tokenizer did not return input_ids"
            
            # Test model inference
            with torch.no_grad():
                outputs = model(**tokens)
            
            assert "last_hidden_state" in outputs, "Model outputs missing last_hidden_state"
            
            # Test passed
            end_time = time.time()
            self.results.add_result(TestResult(
                category=category,
                test_name=test_name,
                status="pass",
                execution_time=end_time - start_time,
                details={
                    "model_name": model_name,
                    "tokenizer_type": type(tokenizer).__name__,
                    "model_type": type(model).__name__
                }
            ))
            logger.info(f"✓ {test_name} passed")
            
        except Exception as e:
            end_time = time.time()
            self.results.add_result(TestResult(
                category=category,
                test_name=test_name,
                status="error",
                execution_time=end_time - start_time,
                error_message=str(e),
                details={"traceback": traceback.format_exc()}
            ))
            logger.error(f"✗ {test_name} failed: {str(e)}")
        
        # Test model loading on different hardware platforms
        for platform in self.hardware_platforms:
            # Skip web platforms for model loading tests
            if platform in ["webnn", "webgpu"]:
                continue
                
            test_name = f"test_model_loading_{platform}"
            start_time = time.time()
            
            try:
                # Use a small model for testing
                model_name = "prajjwal1/bert-tiny"
                
                # Skip if platform is not available
                if platform == "cuda" and not torch.cuda.is_available():
                    self.results.add_result(TestResult(
                        category=category,
                        test_name=test_name,
                        status="skip",
                        hardware_platform=platform,
                        details={"reason": "CUDA not available"}
                    ))
                    continue
                
                if platform == "mps" and not (hasattr(torch, "mps") and torch.backends.mps.is_available()):
                    self.results.add_result(TestResult(
                        category=category,
                        test_name=test_name,
                        status="skip",
                        hardware_platform=platform,
                        details={"reason": "MPS not available"}
                    ))
                    continue
                
                if platform == "rocm" and not (hasattr(torch.utils, "hip") and torch.utils.hip.is_available()):
                    self.results.add_result(TestResult(
                        category=category,
                        test_name=test_name,
                        status="skip",
                        hardware_platform=platform,
                        details={"reason": "ROCm not available"}
                    ))
                    continue
                
                if platform == "openvino":
                    try:
                        import openvino
                    except ImportError:
                        self.results.add_result(TestResult(
                            category=category,
                            test_name=test_name,
                            status="skip",
                            hardware_platform=platform,
                            details={"reason": "OpenVINO not available"}
                        ))
                        continue
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # Map platform to device
                device_map = {
                    "cpu": "cpu",
                    "cuda": "cuda",
                    "mps": "mps",
                    "rocm": "cuda"  # ROCm uses CUDA device
                }
                
                # Special handling for OpenVINO
                if platform == "openvino":
                    try:
                        from optimum.intel import OVModelForSequenceClassification
                        model = OVModelForSequenceClassification.from_pretrained(
                            model_name, 
                            export=True,
                            device="CPU"
                        )
                    except ImportError:
                        self.results.add_result(TestResult(
                            category=category,
                            test_name=test_name,
                            status="skip",
                            hardware_platform=platform,
                            details={"reason": "optimum.intel not available"}
                        ))
                        continue
                else:
                    # Load model to device
                    device = device_map.get(platform, "cpu")
                    model = AutoModel.from_pretrained(model_name).to(device)
                
                # Test tokenizer
                tokens = tokenizer("Hello world", return_tensors="pt")
                
                # Move tokens to device
                if platform != "openvino":
                    tokens = {k: v.to(device) for k, v in tokens.items()}
                
                # Test model inference
                with torch.no_grad():
                    outputs = model(**tokens)
                
                # Test passed
                end_time = time.time()
                self.results.add_result(TestResult(
                    category=category,
                    test_name=test_name,
                    status="pass",
                    execution_time=end_time - start_time,
                    hardware_platform=platform,
                    details={
                        "model_name": model_name,
                        "device": device if platform != "openvino" else "openvino"
                    }
                ))
                logger.info(f"✓ {test_name} passed")
                
            except Exception as e:
                end_time = time.time()
                self.results.add_result(TestResult(
                    category=category,
                    test_name=test_name,
                    status="error",
                    execution_time=end_time - start_time,
                    hardware_platform=platform,
                    error_message=str(e),
                    details={"traceback": traceback.format_exc()}
                ))
                logger.error(f"✗ {test_name} failed: {str(e)}")
    
    def _run_api_backend_tests(self) -> None:
        """Run API backend integration tests"""
        category = "api_backends"
        
        if category not in self.categories:
            logger.info(f"Skipping {category} tests (not in selected categories)")
            return
        
        if "api_backends" not in self.test_modules:
            logger.warning(f"Skipping {category} tests (module not imported)")
            return
        
        logger.info(f"Running {category} integration tests")
        
        # Test API backend initialization
        test_name = "test_api_backend_init"
        start_time = time.time()
        
        try:
            module = self.test_modules["api_backends"]
            
            # Check for initialization function
            if hasattr(module, "init_backends") or hasattr(module, "APIBackendManager"):
                # Test passed (we can't actually initialize without credentials)
                end_time = time.time()
                self.results.add_result(TestResult(
                    category=category,
                    test_name=test_name,
                    status="pass",
                    execution_time=end_time - start_time,
                    details={"note": "API backend initialization function found"}
                ))
                logger.info(f"✓ {test_name} passed")
            else:
                # Test API backend registry
                if hasattr(module, "test_api_backend_registry"):
                    # Run registry test
                    registry_result = module.test_api_backend_registry()
                    
                    # Test passed
                    end_time = time.time()
                    self.results.add_result(TestResult(
                        category=category,
                        test_name=test_name,
                        status="pass" if registry_result else "fail",
                        execution_time=end_time - start_time,
                        details={"registry_test": registry_result}
                    ))
                    logger.info(f"{'✓' if registry_result else '✗'} {test_name} {'passed' if registry_result else 'failed'}")
                else:
                    raise ImportError("No API backend initialization functionality found")
                
        except Exception as e:
            end_time = time.time()
            self.results.add_result(TestResult(
                category=category,
                test_name=test_name,
                status="error",
                execution_time=end_time - start_time,
                error_message=str(e),
                details={"traceback": traceback.format_exc()}
            ))
            logger.error(f"✗ {test_name} failed: {str(e)}")
        
        # Test API multiplexing
        test_name = "test_api_multiplexing"
        start_time = time.time()
        
        try:
            # Look for API multiplexing test functions
            if hasattr(module, "test_api_multiplexing") or hasattr(module, "test_multiplexing"):
                multiplex_func = getattr(module, "test_api_multiplexing", None) or getattr(module, "test_multiplexing")
                
                # Run multiplexing test in mock mode if available
                if callable(multiplex_func):
                    multiplex_result = multiplex_func(use_mock=True)
                    
                    # Test passed
                    end_time = time.time()
                    self.results.add_result(TestResult(
                        category=category,
                        test_name=test_name,
                        status="pass" if multiplex_result else "fail",
                        execution_time=end_time - start_time,
                        details={"multiplexing_test": multiplex_result}
                    ))
                    logger.info(f"{'✓' if multiplex_result else '✗'} {test_name} {'passed' if multiplex_result else 'failed'}")
                else:
                    # Try importing API multiplexing module directly
                    try:
                        multiplex_module = importlib.import_module("test.test_api_multiplexing")
                        logger.info("Imported API multiplexing module")
                        
                        if hasattr(multiplex_module, "test_multiplexing"):
                            multiplex_result = multiplex_module.test_multiplexing(use_mock=True)
                            
                            # Test passed
                            end_time = time.time()
                            self.results.add_result(TestResult(
                                category=category,
                                test_name=test_name,
                                status="pass" if multiplex_result else "fail",
                                execution_time=end_time - start_time,
                                details={"multiplexing_test": multiplex_result}
                            ))
                            logger.info(f"{'✓' if multiplex_result else '✗'} {test_name} {'passed' if multiplex_result else 'failed'}")
                        else:
                            raise AttributeError("No test_multiplexing function found")
                            
                    except (ImportError, AttributeError) as e:
                        raise ImportError(f"No API multiplexing test functionality found: {e}")
            else:
                raise ImportError("No API multiplexing test functionality found")
                
        except Exception as e:
            end_time = time.time()
            self.results.add_result(TestResult(
                category=category,
                test_name=test_name,
                status="error",
                execution_time=end_time - start_time,
                error_message=str(e),
                details={"traceback": traceback.format_exc()}
            ))
            logger.error(f"✗ {test_name} failed: {str(e)}")
    
    def _run_web_platform_tests(self) -> None:
        """Run web platform integration tests"""
        category = "web_platforms"
        
        if category not in self.categories:
            logger.info(f"Skipping {category} tests (not in selected categories)")
            return
        
        if "web_platforms" not in self.test_modules:
            logger.warning(f"Skipping {category} tests (module not imported)")
            return
        
        logger.info(f"Running {category} integration tests")
        
        # Test web platform testing functionality
        test_name = "test_web_platform_testing_init"
        start_time = time.time()
        
        try:
            module = self.test_modules["web_platforms"]
            
            # Check for WebPlatformTesting class
            if hasattr(module, "WebPlatformTesting"):
                # Create testing instance
                web_tester = module.WebPlatformTesting()
                
                # Verify that the tester is correctly initialized
                if not hasattr(web_tester, "web_platforms"):
                    raise AttributeError("WebPlatformTesting missing web_platforms attribute")
                
                if not hasattr(web_tester, "test_model_on_web_platform"):
                    raise AttributeError("WebPlatformTesting missing test_model_on_web_platform method")
                
                # Test passed
                end_time = time.time()
                self.results.add_result(TestResult(
                    category=category,
                    test_name=test_name,
                    status="pass",
                    execution_time=end_time - start_time,
                    details={"web_platforms": web_tester.web_platforms}
                ))
                logger.info(f"✓ {test_name} passed")
                
            else:
                raise ImportError("WebPlatformTesting class not found")
                
        except Exception as e:
            end_time = time.time()
            self.results.add_result(TestResult(
                category=category,
                test_name=test_name,
                status="error",
                execution_time=end_time - start_time,
                error_message=str(e),
                details={"traceback": traceback.format_exc()}
            ))
            logger.error(f"✗ {test_name} failed: {str(e)}")
        
        # Test WebNN simulation mode
        test_name = "test_webnn_simulation"
        start_time = time.time()
        
        try:
            module = self.test_modules["web_platforms"]
            
            # Skip if slow tests are disabled
            if self.skip_slow_tests:
                self.results.add_result(TestResult(
                    category=category,
                    test_name=test_name,
                    status="skip",
                    details={"reason": "Slow tests disabled"}
                ))
                logger.info(f"Skipping {test_name} (slow tests disabled)")
                return
            
            # Check for WebPlatformTesting class
            if hasattr(module, "WebPlatformTesting"):
                # Create testing instance
                web_tester = module.WebPlatformTesting()
                
                # Try to detect modality of "bert"
                modality = web_tester.detect_model_modality("bert")
                
                # Check detection result
                if modality != "text":
                    raise ValueError(f"Incorrect modality detection: got {modality}, expected 'text'")
                
                # Test passed
                end_time = time.time()
                self.results.add_result(TestResult(
                    category=category,
                    test_name=test_name,
                    status="pass",
                    execution_time=end_time - start_time,
                    details={"bert_modality": modality}
                ))
                logger.info(f"✓ {test_name} passed")
                
            else:
                raise ImportError("WebPlatformTesting class not found")
                
        except Exception as e:
            end_time = time.time()
            self.results.add_result(TestResult(
                category=category,
                test_name=test_name,
                status="error",
                execution_time=end_time - start_time,
                error_message=str(e),
                details={"traceback": traceback.format_exc()}
            ))
            logger.error(f"✗ {test_name} failed: {str(e)}")
        
        # Test WebGPU simulation mode
        test_name = "test_webgpu_simulation"
        start_time = time.time()
        
        try:
            # Skip if slow tests are disabled
            if self.skip_slow_tests:
                self.results.add_result(TestResult(
                    category=category,
                    test_name=test_name,
                    status="skip",
                    details={"reason": "Slow tests disabled"}
                ))
                logger.info(f"Skipping {test_name} (slow tests disabled)")
                return
                
            # Try importing web platform benchmark module
            try:
                bench_module = importlib.import_module("test.web_platform_benchmark")
                logger.info("Imported web platform benchmark module")
                
                if hasattr(bench_module, "WebPlatformBenchmark"):
                    # Create benchmarking instance
                    web_bench = bench_module.WebPlatformBenchmark()
                    
                    # Test passed
                    end_time = time.time()
                    self.results.add_result(TestResult(
                        category=category,
                        test_name=test_name,
                        status="pass",
                        execution_time=end_time - start_time,
                        details={"web_platforms": web_bench.web_platforms}
                    ))
                    logger.info(f"✓ {test_name} passed")
                else:
                    raise ImportError("WebPlatformBenchmark class not found")
                    
            except ImportError as e:
                # Fall back to web_platforms module
                module = self.test_modules["web_platforms"]
                
                # Create testing instance
                web_tester = module.WebPlatformTesting()
                
                # Try to detect modality of "vit"
                modality = web_tester.detect_model_modality("vit")
                
                # Check detection result
                if modality != "vision":
                    raise ValueError(f"Incorrect modality detection: got {modality}, expected 'vision'")
                
                # Test passed
                end_time = time.time()
                self.results.add_result(TestResult(
                    category=category,
                    test_name=test_name,
                    status="pass",
                    execution_time=end_time - start_time,
                    details={"vit_modality": modality}
                ))
                logger.info(f"✓ {test_name} passed")
                
        except Exception as e:
            end_time = time.time()
            self.results.add_result(TestResult(
                category=category,
                test_name=test_name,
                status="error",
                execution_time=end_time - start_time,
                error_message=str(e),
                details={"traceback": traceback.format_exc()}
            ))
            logger.error(f"✗ {test_name} failed: {str(e)}")
    
    def _run_multimodal_tests(self) -> None:
        """Run multimodal integration tests"""
        category = "multimodal"
        
        if category not in self.categories:
            logger.info(f"Skipping {category} tests (not in selected categories)")
            return
        
        logger.info(f"Running {category} integration tests")
        
        # Skip if torch is not available
        if not HAS_TORCH:
            logger.warning("Skipping multimodal tests (torch not available)")
            self.results.add_result(TestResult(
                category=category,
                test_name="test_multimodal_integration",
                status="skip",
                details={"reason": "torch not available"}
            ))
            return
        
        # Try to import transformers
        try:
            import transformers
            logger.info("Imported transformers module")
        except ImportError:
            logger.warning("Skipping multimodal tests (transformers not available)")
            self.results.add_result(TestResult(
                category=category,
                test_name="test_multimodal_integration",
                status="skip",
                details={"reason": "transformers not available"}
            ))
            return
        
        # Test CLIP model loading
        test_name = "test_clip_model_loading"
        start_time = time.time()
        
        try:
            # Skip if slow tests are disabled
            if self.skip_slow_tests:
                self.results.add_result(TestResult(
                    category=category,
                    test_name=test_name,
                    status="skip",
                    details={"reason": "Slow tests disabled"}
                ))
                logger.info(f"Skipping {test_name} (slow tests disabled)")
                return
                
            # Use a small CLIP model for testing
            model_name = "openai/clip-vit-base-patch32"
            
            # Import processor and model
            from transformers import CLIPProcessor, CLIPModel
            
            # Load processor and model
            processor = CLIPProcessor.from_pretrained(model_name)
            model = CLIPModel.from_pretrained(model_name)
            
            # Verify processor and model
            assert processor is not None, "Processor is None"
            assert model is not None, "Model is None"
            
            # Test processor
            # Skip actual processing since we don't have an image
            
            # Test model architecture
            assert hasattr(model, "text_model"), "Model missing text_model component"
            assert hasattr(model, "vision_model"), "Model missing vision_model component"
            
            # Test passed
            end_time = time.time()
            self.results.add_result(TestResult(
                category=category,
                test_name=test_name,
                status="pass",
                execution_time=end_time - start_time,
                details={
                    "model_name": model_name,
                    "processor_type": type(processor).__name__,
                    "model_type": type(model).__name__
                }
            ))
            logger.info(f"✓ {test_name} passed")
            
        except Exception as e:
            end_time = time.time()
            self.results.add_result(TestResult(
                category=category,
                test_name=test_name,
                status="error",
                execution_time=end_time - start_time,
                error_message=str(e),
                details={"traceback": traceback.format_exc()}
            ))
            logger.error(f"✗ {test_name} failed: {str(e)}")
    
    def _run_endpoint_lifecycle_tests(self) -> None:
        """Run endpoint lifecycle integration tests"""
        category = "endpoint_lifecycle"
        
        if category not in self.categories:
            logger.info(f"Skipping {category} tests (not in selected categories)")
            return
        
        if "endpoint_lifecycle" not in self.test_modules:
            logger.warning(f"Skipping {category} tests (module not imported)")
            return
        
        logger.info(f"Running {category} integration tests")
        
        # Test endpoint creation and destruction
        test_name = "test_endpoint_lifecycle"
        start_time = time.time()
        
        try:
            module = self.test_modules["endpoint_lifecycle"]
            
            # Check for test function
            if hasattr(module, "test_endpoint_lifecycle") or hasattr(module, "test_lifecycle"):
                # Get test function
                test_func = getattr(module, "test_endpoint_lifecycle", None) or getattr(module, "test_lifecycle")
                
                # Run test in mock mode if possible
                if callable(test_func):
                    try:
                        # Try with mock mode parameter
                        lifecycle_result = test_func(use_mock=True)
                    except TypeError:
                        # Parameter not supported, try without
                        lifecycle_result = test_func()
                    
                    # Test passed
                    end_time = time.time()
                    self.results.add_result(TestResult(
                        category=category,
                        test_name=test_name,
                        status="pass" if lifecycle_result else "fail",
                        execution_time=end_time - start_time,
                        details={"lifecycle_test": lifecycle_result}
                    ))
                    logger.info(f"{'✓' if lifecycle_result else '✗'} {test_name} {'passed' if lifecycle_result else 'failed'}")
                else:
                    raise TypeError("Test function is not callable")
            else:
                # Check for EndpointManager class
                if hasattr(module, "EndpointManager") or hasattr(module, "EndpointLifecycleManager"):
                    # Get manager class
                    manager_class = getattr(module, "EndpointManager", None) or getattr(module, "EndpointLifecycleManager")
                    
                    # Create manager instance
                    manager = manager_class()
                    
                    # Verify manager methods
                    methods_to_check = ["create_endpoint", "destroy_endpoint", "get_endpoint"]
                    missing_methods = [method for method in methods_to_check if not hasattr(manager, method)]
                    
                    if missing_methods:
                        raise AttributeError(f"EndpointManager missing methods: {missing_methods}")
                    
                    # Test passed
                    end_time = time.time()
                    self.results.add_result(TestResult(
                        category=category,
                        test_name=test_name,
                        status="pass",
                        execution_time=end_time - start_time,
                        details={"note": "EndpointManager class found with required methods"}
                    ))
                    logger.info(f"✓ {test_name} passed")
                else:
                    raise ImportError("No endpoint lifecycle functionality found")
                
        except Exception as e:
            end_time = time.time()
            self.results.add_result(TestResult(
                category=category,
                test_name=test_name,
                status="error",
                execution_time=end_time - start_time,
                error_message=str(e),
                details={"traceback": traceback.format_exc()}
            ))
            logger.error(f"✗ {test_name} failed: {str(e)}")
    
    def _run_batch_processing_tests(self) -> None:
        """Run batch processing integration tests"""
        category = "batch_processing"
        
        if category not in self.categories:
            logger.info(f"Skipping {category} tests (not in selected categories)")
            return
        
        if "batch_processing" not in self.test_modules:
            logger.warning(f"Skipping {category} tests (module not imported)")
            return
        
        logger.info(f"Running {category} integration tests")
        
        # Test batch inference
        test_name = "test_batch_inference"
        start_time = time.time()
        
        try:
            module = self.test_modules["batch_processing"]
            
            # Check for test function
            if hasattr(module, "test_batch_inference") or hasattr(module, "run_batch_test"):
                # Get test function
                test_func = getattr(module, "test_batch_inference", None) or getattr(module, "run_batch_test")
                
                # Run test in mock mode if possible
                if callable(test_func):
                    try:
                        # Try with mock mode parameter
                        batch_result = test_func(use_mock=True)
                    except TypeError:
                        # Parameter not supported, try without
                        batch_result = test_func()
                    
                    # Test passed
                    end_time = time.time()
                    self.results.add_result(TestResult(
                        category=category,
                        test_name=test_name,
                        status="pass" if batch_result else "fail",
                        execution_time=end_time - start_time,
                        details={"batch_test": batch_result}
                    ))
                    logger.info(f"{'✓' if batch_result else '✗'} {test_name} {'passed' if batch_result else 'failed'}")
                else:
                    raise TypeError("Test function is not callable")
            else:
                # Check for BatchProcessor class
                if hasattr(module, "BatchProcessor") or hasattr(module, "BatchInferenceProcessor"):
                    # Get processor class
                    processor_class = getattr(module, "BatchProcessor", None) or getattr(module, "BatchInferenceProcessor")
                    
                    # Create processor instance
                    processor = processor_class()
                    
                    # Verify processor methods
                    methods_to_check = ["process_batch", "get_results"]
                    missing_methods = [method for method in methods_to_check if not hasattr(processor, method)]
                    
                    if missing_methods:
                        raise AttributeError(f"BatchProcessor missing methods: {missing_methods}")
                    
                    # Test passed
                    end_time = time.time()
                    self.results.add_result(TestResult(
                        category=category,
                        test_name=test_name,
                        status="pass",
                        execution_time=end_time - start_time,
                        details={"note": "BatchProcessor class found with required methods"}
                    ))
                    logger.info(f"✓ {test_name} passed")
                else:
                    raise ImportError("No batch processing functionality found")
                
        except Exception as e:
            end_time = time.time()
            self.results.add_result(TestResult(
                category=category,
                test_name=test_name,
                status="error",
                execution_time=end_time - start_time,
                error_message=str(e),
                details={"traceback": traceback.format_exc()}
            ))
            logger.error(f"✗ {test_name} failed: {str(e)}")
    
    def _run_queue_management_tests(self) -> None:
        """Run queue management integration tests"""
        category = "queue_management"
        
        if category not in self.categories:
            logger.info(f"Skipping {category} tests (not in selected categories)")
            return
        
        if "queue_management" not in self.test_modules:
            logger.warning(f"Skipping {category} tests (module not imported)")
            return
        
        logger.info(f"Running {category} integration tests")
        
        # Test backoff queue
        test_name = "test_backoff_queue"
        start_time = time.time()
        
        try:
            module = self.test_modules["queue_management"]
            
            # Check for test function
            if hasattr(module, "test_backoff_queue") or hasattr(module, "test_queue_backoff"):
                # Get test function
                test_func = getattr(module, "test_backoff_queue", None) or getattr(module, "test_queue_backoff")
                
                # Run test in mock mode if possible
                if callable(test_func):
                    try:
                        # Try with mock mode parameter
                        queue_result = test_func(use_mock=True)
                    except TypeError:
                        # Parameter not supported, try without
                        queue_result = test_func()
                    
                    # Test passed
                    end_time = time.time()
                    self.results.add_result(TestResult(
                        category=category,
                        test_name=test_name,
                        status="pass" if queue_result else "fail",
                        execution_time=end_time - start_time,
                        details={"queue_test": queue_result}
                    ))
                    logger.info(f"{'✓' if queue_result else '✗'} {test_name} {'passed' if queue_result else 'failed'}")
                else:
                    raise TypeError("Test function is not callable")
            else:
                # Try to import queue backoff module directly
                try:
                    backoff_module = importlib.import_module("test.test_queue_backoff")
                    logger.info("Imported queue backoff module")
                    
                    # Check for test function
                    if hasattr(backoff_module, "test_queue_backoff"):
                        # Run test
                        queue_result = backoff_module.test_queue_backoff()
                        
                        # Test passed
                        end_time = time.time()
                        self.results.add_result(TestResult(
                            category=category,
                            test_name=test_name,
                            status="pass" if queue_result else "fail",
                            execution_time=end_time - start_time,
                            details={"queue_test": queue_result}
                        ))
                        logger.info(f"{'✓' if queue_result else '✗'} {test_name} {'passed' if queue_result else 'failed'}")
                    else:
                        # Check for BackoffQueue class
                        if hasattr(backoff_module, "BackoffQueue") or hasattr(backoff_module, "APIBackoffQueue"):
                            # Get queue class
                            queue_class = getattr(backoff_module, "BackoffQueue", None) or getattr(backoff_module, "APIBackoffQueue")
                            
                            # Create queue instance
                            queue = queue_class()
                            
                            # Verify queue methods
                            methods_to_check = ["add_request", "get_next", "handle_response"]
                            missing_methods = [method for method in methods_to_check if not hasattr(queue, method)]
                            
                            if missing_methods:
                                raise AttributeError(f"BackoffQueue missing methods: {missing_methods}")
                            
                            # Test passed
                            end_time = time.time()
                            self.results.add_result(TestResult(
                                category=category,
                                test_name=test_name,
                                status="pass",
                                execution_time=end_time - start_time,
                                details={"note": "BackoffQueue class found with required methods"}
                            ))
                            logger.info(f"✓ {test_name} passed")
                        else:
                            raise ImportError("No BackoffQueue class found")
                            
                except ImportError as e:
                    # Check for BackoffQueue class in the current module
                    if hasattr(module, "BackoffQueue") or hasattr(module, "APIBackoffQueue"):
                        # Get queue class
                        queue_class = getattr(module, "BackoffQueue", None) or getattr(module, "APIBackoffQueue")
                        
                        # Create queue instance
                        queue = queue_class()
                        
                        # Verify queue methods
                        methods_to_check = ["add_request", "get_next", "handle_response"]
                        missing_methods = [method for method in methods_to_check if not hasattr(queue, method)]
                        
                        if missing_methods:
                            raise AttributeError(f"BackoffQueue missing methods: {missing_methods}")
                        
                        # Test passed
                        end_time = time.time()
                        self.results.add_result(TestResult(
                            category=category,
                            test_name=test_name,
                            status="pass",
                            execution_time=end_time - start_time,
                            details={"note": "BackoffQueue class found with required methods"}
                        ))
                        logger.info(f"✓ {test_name} passed")
                    else:
                        raise ImportError(f"No queue management functionality found: {e}")
                
        except Exception as e:
            end_time = time.time()
            self.results.add_result(TestResult(
                category=category,
                test_name=test_name,
                status="error",
                execution_time=end_time - start_time,
                error_message=str(e),
                details={"traceback": traceback.format_exc()}
            ))
            logger.error(f"✗ {test_name} failed: {str(e)}")
    
    def _run_hardware_compatibility_tests(self) -> None:
        """Run hardware compatibility matrix validation tests
        
        These tests verify that models work as expected on all claimed compatible hardware platforms.
        The tests check against the hardware compatibility matrix defined in documentation,
        and validate actual compatibility through empirical testing.
        """
        category = "hardware_compatibility"
        
        if category not in self.categories:
            logger.info(f"Skipping {category} tests (not in selected categories)")
            return
        
        logger.info(f"Running {category} integration tests")
        
        # Skip if torch is not available
        if not HAS_TORCH:
            logger.warning("Skipping hardware compatibility tests (torch not available)")
            self.results.add_result(TestResult(
                category=category,
                test_name="test_hardware_compatibility",
                status="skip",
                details={"reason": "torch not available"}
            ))
            return
        
        # Try to import transformers
        try:
            import transformers
            logger.info("Imported transformers module")
        except ImportError:
            logger.warning("Skipping hardware compatibility tests (transformers not available)")
            self.results.add_result(TestResult(
                category=category,
                test_name="test_hardware_compatibility",
                status="skip",
                details={"reason": "transformers not available"}
            ))
            return
        
        # Try importing hardware_detection and model_family_classifier modules
        try:
            # Import hardware detection
            hardware_detection_module = importlib.import_module("test.hardware_detection")
            model_family_module = importlib.import_module("test.model_family_classifier")
            logger.info("Successfully imported hardware detection and model family modules")
        except ImportError as e:
            logger.warning(f"Skipping hardware compatibility tests (required modules not available): {e}")
            self.results.add_result(TestResult(
                category=category,
                test_name="test_hardware_compatibility",
                status="skip",
                details={"reason": f"Required modules not available: {e}"}
            ))
            return
        
        # Create test matrix - model families and their representative models
        compatibility_matrix = {
            "embedding": {
                "name": "prajjwal1/bert-tiny",
                "class": "BertModel",
                "constructor": lambda: transformers.AutoModel.from_pretrained("prajjwal1/bert-tiny")
            },
            "text_generation": {
                "name": "google/t5-efficient-tiny",
                "class": "T5ForConditionalGeneration",
                "constructor": lambda: transformers.T5ForConditionalGeneration.from_pretrained("google/t5-efficient-tiny")
            },
            "vision": {
                "name": "google/vit-base-patch16-224",
                "class": "ViTModel",
                "constructor": lambda: transformers.ViTModel.from_pretrained("google/vit-base-patch16-224", 
                                                                             ignore_mismatched_sizes=True)
            }
        }
        
        # Try to test audio model if available (this might be too large for some CI environments)
        try:
            if not self.skip_slow_tests:
                compatibility_matrix["audio"] = {
                    "name": "openai/whisper-tiny",
                    "class": "WhisperModel",
                    "constructor": lambda: transformers.WhisperModel.from_pretrained("openai/whisper-tiny")
                }
        except Exception as e:
            logger.warning(f"Skipping audio model in compatibility matrix: {e}")
        
        # Get detected hardware
        try:
            # Use hardware detection to get available hardware
            if hasattr(hardware_detection_module, "detect_hardware_with_comprehensive_checks"):
                hardware_info = hardware_detection_module.detect_hardware_with_comprehensive_checks()
                available_hardware = [hw for hw, available in hardware_info.items() 
                                    if hw in ['cpu', 'cuda', 'mps', 'rocm', 'openvino', 'webnn', 'webgpu'] 
                                    and available]
                logger.info(f"Detected hardware: {available_hardware}")
            else:
                # Fallback to basic hardware detection
                available_hardware = self.hardware_platforms
        except Exception as e:
            logger.error(f"Error detecting hardware for compatibility tests: {e}")
            available_hardware = ["cpu"]  # Fallback to CPU only
        
        # Import model_family_classifier to classify models
        if hasattr(model_family_module, "classify_model"):
            classify_model = model_family_module.classify_model
        else:
            # Fallback to basic classification
            classify_model = lambda model_name, **kwargs: {"family": None, "confidence": 0}
        
        # Test each model family on each hardware platform
        for family, model_info in compatibility_matrix.items():
            test_name = f"test_{family}_hardware_compatibility"
            model_name = model_info["name"]
            
            # Get expected compatibility for this family
            try:
                # Try to read compatibility matrix from hardware_detection module
                matrix_found = False
                expected_compatibility = {}
                
                if hasattr(hardware_detection_module, "MODEL_FAMILY_HARDWARE_COMPATIBILITY"):
                    compatibility_data = hardware_detection_module.MODEL_FAMILY_HARDWARE_COMPATIBILITY
                    if family in compatibility_data:
                        expected_compatibility = compatibility_data[family]
                        matrix_found = True
                
                if not matrix_found:
                    # Fallback to default expectations based on common knowledge
                    expected_compatibility = {
                        "cpu": True,  # CPU should always work
                        "cuda": True,  # CUDA should work for all families
                        "mps": family != "multimodal",  # MPS has issues with multimodal
                        "rocm": family in ["embedding", "text_generation"],  # ROCm works best with text
                        "openvino": family in ["embedding", "vision"],  # OpenVINO works best with vision
                        "webnn": family in ["embedding", "vision"],  # WebNN supports simpler models
                        "webgpu": family in ["embedding", "vision"]  # WebGPU similar to WebNN
                    }
            except Exception as e:
                logger.warning(f"Error reading compatibility matrix, using defaults: {e}")
                # Use defaults
                expected_compatibility = {
                    "cpu": True,  # CPU should always work
                    "cuda": True,  # CUDA should work for all families
                    "mps": family != "multimodal",  # MPS has issues with multimodal
                    "rocm": family in ["embedding", "text_generation"],  # ROCm works best with text
                    "openvino": family in ["embedding", "vision"],  # OpenVINO works best with vision
                    "webnn": family in ["embedding", "vision"],  # WebNN supports simpler models
                    "webgpu": family in ["embedding", "vision"]  # WebGPU similar to WebNN
                }
            
            # Test results for this model
            compatibility_results = {}
            
            # Test model on each hardware platform
            for platform in available_hardware:
                # Skip web platforms for actual model loading (simulation only)
                if platform in ["webnn", "webgpu"]:
                    # Only test classification for web platforms
                    try:
                        # Classify model
                        classification = classify_model(
                            model_name=model_name,
                            model_class=model_info["class"],
                            hw_compatibility={
                                platform: {"compatible": expected_compatibility.get(platform, False)}
                            }
                        )
                        
                        # Check if classification works
                        is_compatible = classification.get("family") == family
                        
                        # Add result for this platform
                        compatibility_results[platform] = {
                            "expected": expected_compatibility.get(platform, False),
                            "actual": is_compatible,
                            "matches_expected": is_compatible == expected_compatibility.get(platform, False),
                            "classification": classification.get("family"),
                            "classification_confidence": classification.get("confidence", 0)
                        }
                        
                        logger.info(f"Model {model_name} classified for {platform}: {is_compatible}")
                    except Exception as e:
                        logger.error(f"Error classifying model {model_name} for {platform}: {e}")
                        compatibility_results[platform] = {
                            "expected": expected_compatibility.get(platform, False),
                            "actual": False,
                            "matches_expected": False,
                            "error": str(e)
                        }
                    continue
                
                # For real hardware, try loading the model
                platform_start_time = time.time()
                
                try:
                    # Skip if hardware not actually available
                    if platform == "cuda" and not torch.cuda.is_available():
                        compatibility_results[platform] = {
                            "expected": expected_compatibility.get(platform, False),
                            "actual": False,
                            "skipped": True,
                            "reason": "CUDA not available"
                        }
                        continue
                    
                    if platform == "mps" and not (hasattr(torch, "mps") and torch.backends.mps.is_available()):
                        compatibility_results[platform] = {
                            "expected": expected_compatibility.get(platform, False),
                            "actual": False,
                            "skipped": True,
                            "reason": "MPS not available"
                        }
                        continue
                    
                    if platform == "rocm" and not (hasattr(torch.utils, "hip") and torch.utils.hip.is_available()):
                        compatibility_results[platform] = {
                            "expected": expected_compatibility.get(platform, False),
                            "actual": False,
                            "skipped": True,
                            "reason": "ROCm not available"
                        }
                        continue
                    
                    if platform == "openvino":
                        try:
                            import openvino
                        except ImportError:
                            compatibility_results[platform] = {
                                "expected": expected_compatibility.get(platform, False),
                                "actual": False,
                                "skipped": True,
                                "reason": "OpenVINO not available"
                            }
                            continue
                    
                    # Set timeout to reasonable value for model loading
                    model_timeout = 120  # 2 minutes
                    model_loaded = False
                    
                    # Map platform to device
                    device_map = {
                        "cpu": "cpu",
                        "cuda": "cuda",
                        "mps": "mps",
                        "rocm": "cuda"  # ROCm uses CUDA device
                    }
                    
                    # Special handling for OpenVINO
                    if platform == "openvino":
                        try:
                            from optimum.intel import OVModelForSequenceClassification
                            model = OVModelForSequenceClassification.from_pretrained(
                                model_info["name"], 
                                export=True,
                                device="CPU"
                            )
                            model_loaded = True
                        except ImportError:
                            compatibility_results[platform] = {
                                "expected": expected_compatibility.get(platform, False),
                                "actual": False,
                                "skipped": True,
                                "reason": "optimum.intel not available"
                            }
                            continue
                    else:
                        # Load model to device with timeout
                        import signal
                        
                        def timeout_handler(signum, frame):
                            raise TimeoutError(f"Model loading timed out after {model_timeout} seconds")
                        
                        # Set signal handler
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(model_timeout)
                        
                        try:
                            # Load model to device
                            device = device_map.get(platform, "cpu")
                            model = model_info["constructor"]().to(device)
                            model_loaded = True
                            
                            # Cancel alarm
                            signal.alarm(0)
                        except Exception as load_error:
                            # Cancel alarm
                            signal.alarm(0)
                            raise load_error
                    
                    # Run a basic inference test
                    try:
                        # Based on model family, create appropriate test input
                        if family == "embedding":
                            # Create a simple input for BERT-like models
                            if platform == "openvino":
                                # OpenVINO may need special handling
                                inputs = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
                            else:
                                inputs = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]]).to(device)}
                        
                        elif family == "text_generation":
                            # Create input for text generation models
                            if platform == "openvino":
                                inputs = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
                            else:
                                inputs = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]]).to(device)}
                        
                        elif family == "vision":
                            # Create input for vision models
                            if platform == "openvino":
                                # OpenVINO may need special handling
                                inputs = {"pixel_values": torch.randn(1, 3, 224, 224)}
                            else:
                                inputs = {"pixel_values": torch.randn(1, 3, 224, 224).to(device)}
                        
                        elif family == "audio":
                            # Create input for audio models
                            if platform == "openvino":
                                # OpenVINO may need special handling
                                inputs = {"input_features": torch.randn(1, 80, 3000)}
                            else:
                                inputs = {"input_features": torch.randn(1, 80, 3000).to(device)}
                        
                        else:
                            # Generic fallback
                            if platform == "openvino":
                                inputs = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
                            else:
                                inputs = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]]).to(device)}
                        
                        # Run model inference
                        with torch.no_grad():
                            outputs = model(**inputs)
                        
                        # Success - model works on this platform
                        inference_success = True
                    except Exception as infer_error:
                        logger.warning(f"Inference error for {model_name} on {platform}: {infer_error}")
                        inference_success = False
                    
                    # Record compatibility results
                    is_compatible = model_loaded and inference_success
                    platform_end_time = time.time()
                    
                    compatibility_results[platform] = {
                        "expected": expected_compatibility.get(platform, False),
                        "actual": is_compatible,
                        "matches_expected": is_compatible == expected_compatibility.get(platform, False),
                        "model_loaded": model_loaded,
                        "inference_success": inference_success,
                        "execution_time": platform_end_time - platform_start_time
                    }
                    
                    logger.info(f"Model {model_name} compatibility on {platform}: {is_compatible} " +
                               f"(expected: {expected_compatibility.get(platform, False)})")
                    
                except Exception as e:
                    platform_end_time = time.time()
                    logger.error(f"Error testing {model_name} on {platform}: {e}")
                    compatibility_results[platform] = {
                        "expected": expected_compatibility.get(platform, False),
                        "actual": False,
                        "matches_expected": not expected_compatibility.get(platform, False),
                        "error": str(e),
                        "execution_time": platform_end_time - platform_start_time
                    }
            
            # Calculate overall compatibility score for this model
            matches = sum(1 for p, r in compatibility_results.items() 
                         if r.get("matches_expected", False) and not r.get("skipped", False))
            total = sum(1 for p, r in compatibility_results.items() if not r.get("skipped", False))
            compatibility_score = matches / total if total > 0 else 0
            
            # Add test result for this model family
            end_time = time.time()
            self.results.add_result(TestResult(
                category=category,
                test_name=test_name,
                status="pass" if compatibility_score >= 0.8 else "fail",
                execution_time=end_time - time.time(),
                details={
                    "model_name": model_name,
                    "model_family": family,
                    "compatibility_score": compatibility_score,
                    "platform_results": compatibility_results
                }
            ))
            
            logger.info(f"Compatibility test for {family} models: " +
                       f"{'PASS' if compatibility_score >= 0.8 else 'FAIL'} " +
                       f"(score: {compatibility_score:.2f})")

    def _run_cross_platform_tests(self) -> None:
        """Run cross-platform validation tests
        
        These tests verify that the entire stack works consistently across different platforms,
        including web platforms like WebNN and WebGPU.
        """
        category = "cross_platform"
        
        if category not in self.categories:
            logger.info(f"Skipping {category} tests (not in selected categories)")
            return
        
        logger.info(f"Running {category} integration tests")
        
        # First check if we have all required components
        try:
            # Import hardware detection and resource pool
            hardware_detection_module = importlib.import_module("test.hardware_detection")
            resource_pool_module = importlib.import_module("test.resource_pool")
            logger.info("Successfully imported hardware detection and resource pool modules")
        except ImportError as e:
            logger.warning(f"Skipping cross platform tests (required modules not available): {e}")
            self.results.add_result(TestResult(
                category=category,
                test_name="test_cross_platform",
                status="skip",
                details={"reason": f"Required modules not available: {e}"}
            ))
            return
        
        # Test platforms - these are the platforms we want to test across
        test_platforms = ["cpu", "cuda", "mps", "rocm", "openvino", "webnn", "webgpu"]
        
        # Filter for actually detected platforms
        try:
            # Use hardware detection to get available hardware
            if hasattr(hardware_detection_module, "detect_hardware_with_comprehensive_checks"):
                hardware_info = hardware_detection_module.detect_hardware_with_comprehensive_checks()
                available_platforms = [hw for hw, available in hardware_info.items() 
                                    if hw in test_platforms and available]
                logger.info(f"Available platforms for cross-platform testing: {available_platforms}")
            else:
                # Fallback to basic hardware detection
                available_platforms = [p for p in self.hardware_platforms if p in test_platforms]
        except Exception as e:
            logger.error(f"Error detecting platforms for cross-platform tests: {e}")
            available_platforms = ["cpu"]  # Fallback to CPU only
        
        # Add simulated web platforms if not detected but requested
        for web_platform in ["webnn", "webgpu"]:
            if web_platform not in available_platforms and web_platform in test_platforms:
                logger.info(f"Adding simulated {web_platform} platform for testing")
                available_platforms.append(web_platform)
        
        # Test resource pool integration across platforms
        test_name = "test_resource_pool_cross_platform"
        start_time = time.time()
        
        try:
            # Get ResourcePool class from module
            if not hasattr(resource_pool_module, "ResourcePool") and hasattr(resource_pool_module, "get_global_resource_pool"):
                # Try to get the pool instance directly
                pool = resource_pool_module.get_global_resource_pool()
            elif hasattr(resource_pool_module, "ResourcePool"):
                # Create resource pool instance
                pool = resource_pool_module.ResourcePool()
            else:
                raise ImportError("ResourcePool not found in module")
            
            # Results for this test
            platform_results = {}
            
            # Test each platform with resource pool
            for platform in available_platforms:
                platform_start_time = time.time()
                
                try:
                    # For web platforms, test in simulation mode
                    if platform in ["webnn", "webgpu"]:
                        # Check if pool has web platform support methods
                        if hasattr(pool, "supports_web_platform"):
                            support_result = pool.supports_web_platform(platform)
                            platform_results[platform] = {
                                "success": support_result,
                                "device": platform,
                                "execution_time": time.time() - platform_start_time
                            }
                            logger.info(f"ResourcePool web platform support for {platform}: {support_result}")
                        elif hasattr(pool, "get_device"):
                            # Try to get device with web platform preference
                            device = pool.get_device(hardware_preferences={"web_platform": platform})
                            platform_results[platform] = {
                                "success": device is not None,
                                "device": str(device) if device else None,
                                "execution_time": time.time() - platform_start_time
                            }
                            logger.info(f"ResourcePool device for {platform}: {device}")
                        else:
                            platform_results[platform] = {
                                "success": False,
                                "error": "ResourcePool missing web platform support methods",
                                "execution_time": time.time() - platform_start_time
                            }
                    else:
                        # Real hardware platforms
                        # Skip if platform is not available
                        if platform == "cuda" and not torch.cuda.is_available():
                            platform_results[platform] = {
                                "success": False,
                                "skipped": True,
                                "reason": "CUDA not available"
                            }
                            continue
                        
                        if platform == "mps" and not (hasattr(torch, "mps") and torch.backends.mps.is_available()):
                            platform_results[platform] = {
                                "success": False,
                                "skipped": True,
                                "reason": "MPS not available"
                            }
                            continue
                        
                        if platform == "rocm" and not (hasattr(torch.utils, "hip") and torch.utils.hip.is_available()):
                            platform_results[platform] = {
                                "success": False,
                                "skipped": True,
                                "reason": "ROCm not available"
                            }
                            continue
                        
                        if platform == "openvino":
                            try:
                                import openvino
                            except ImportError:
                                platform_results[platform] = {
                                    "success": False,
                                    "skipped": True,
                                    "reason": "OpenVINO not available"
                                }
                                continue
                        
                        # For available hardware, try getting a device
                        if hasattr(pool, "get_device"):
                            device = pool.get_device(device_type=platform)
                            platform_results[platform] = {
                                "success": device is not None,
                                "device": str(device) if device else None,
                                "execution_time": time.time() - platform_start_time
                            }
                            logger.info(f"ResourcePool device for {platform}: {device}")
                        else:
                            platform_results[platform] = {
                                "success": False,
                                "error": "ResourcePool missing get_device method",
                                "execution_time": time.time() - platform_start_time
                            }
                    
                except Exception as e:
                    logger.error(f"Error testing ResourcePool with {platform}: {e}")
                    platform_results[platform] = {
                        "success": False,
                        "error": str(e),
                        "execution_time": time.time() - platform_start_time
                    }
            
            # Calculate overall success rate
            successes = sum(1 for p, r in platform_results.items() 
                          if r.get("success", False) and not r.get("skipped", False))
            total = sum(1 for p, r in platform_results.items() if not r.get("skipped", False))
            success_rate = successes / total if total > 0 else 0
            
            # Add test result
            end_time = time.time()
            self.results.add_result(TestResult(
                category=category,
                test_name=test_name,
                status="pass" if success_rate >= 0.8 else "fail",
                execution_time=end_time - start_time,
                details={
                    "success_rate": success_rate,
                    "platforms_tested": len(platform_results),
                    "platform_results": platform_results
                }
            ))
            
            logger.info(f"ResourcePool cross-platform test: " +
                       f"{'PASS' if success_rate >= 0.8 else 'FAIL'} " +
                       f"(success rate: {success_rate:.2f})")
            
        except Exception as e:
            end_time = time.time()
            self.results.add_result(TestResult(
                category=category,
                test_name=test_name,
                status="error",
                execution_time=end_time - start_time,
                error_message=str(e),
                details={"traceback": traceback.format_exc()}
            ))
            logger.error(f"✗ {test_name} failed: {str(e)}")

    def run_tests(self) -> TestSuiteResults:
        """Run all integration tests"""
        logger.info(f"Starting integration test suite with categories: {self.categories}")
        logger.info(f"Detected hardware platforms: {self.hardware_platforms}")
        
        # Run tests for each category
        self._run_hardware_detection_tests()
        self._run_resource_pool_tests()
        self._run_model_loading_tests()
        self._run_api_backend_tests()
        self._run_web_platform_tests()
        self._run_multimodal_tests()
        self._run_endpoint_lifecycle_tests()
        self._run_batch_processing_tests()
        self._run_queue_management_tests()
        self._run_hardware_compatibility_tests()  # New test category
        self._run_cross_platform_tests()          # New test category
        
        # Mark test suite as finished
        self.results.finish()
        
        # Print summary
        self.results.print_summary()
        
        # Save results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_dir, f"integration_test_results_{timestamp}.json")
        self.results.save_results(results_file)
        
        return self.results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run integration tests for IPFS Accelerate Python")
    
    parser.add_argument("--categories", nargs="+", choices=INTEGRATION_CATEGORIES,
                        help="Categories of tests to run")
    parser.add_argument("--hardware", nargs="+", 
                        help="Hardware platforms to test")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout for tests in seconds")
    parser.add_argument("--skip-slow", action="store_true",
                        help="Skip slow tests")
    parser.add_argument("--output", type=str,
                        help="Custom output file for test results")
    parser.add_argument("--web-platforms", action="store_true",
                        help="Focus testing on WebNN/WebGPU platforms")
    parser.add_argument("--hardware-compatibility", action="store_true",
                        help="Run hardware compatibility matrix validation tests")
    parser.add_argument("--cross-platform", action="store_true",
                        help="Run cross-platform validation tests")
    parser.add_argument("--ci-mode", action="store_true",
                        help="Enable CI mode with smaller models and faster tests")
    
    return parser.parse_args()


def main():
    """Main entry point for the integration test suite."""
    args = parse_args()
    
    # Process special category flags
    categories = args.categories
    
    # If specific category flags are set, add them to test categories
    if args.web_platforms and "web_platforms" not in (categories or []):
        if categories is None:
            categories = ["web_platforms"]
        else:
            categories.append("web_platforms")
    
    if args.hardware_compatibility and "hardware_compatibility" not in (categories or []):
        if categories is None:
            categories = ["hardware_compatibility"]
        else:
            categories.append("hardware_compatibility")
    
    if args.cross_platform and "cross_platform" not in (categories or []):
        if categories is None:
            categories = ["cross_platform"]
        else:
            categories.append("cross_platform")
    
    # Add required dependencies for special categories
    if "hardware_compatibility" in (categories or []) or "cross_platform" in (categories or []):
        # These tests need hardware detection, so add it if not already there
        if categories is None:
            categories = ["hardware_detection", "hardware_compatibility", "cross_platform"]
        elif "hardware_detection" not in categories:
            categories.append("hardware_detection")
    
    # Process hardware platforms
    hardware_platforms = args.hardware
    
    # If we're testing web platforms specifically, add them if not already there
    if args.web_platforms and hardware_platforms is not None:
        if "webnn" not in hardware_platforms:
            hardware_platforms.append("webnn")
        if "webgpu" not in hardware_platforms:
            hardware_platforms.append("webgpu")
    
    # Set up CI mode if requested
    skip_slow = args.skip_slow or args.ci_mode
    timeout = min(args.timeout, 180) if args.ci_mode else args.timeout
    
    # Create and run test suite
    test_suite = IntegrationTestSuite(
        categories=categories,
        hardware_platforms=hardware_platforms,
        timeout=timeout,
        skip_slow_tests=skip_slow
    )
    
    # Run all tests
    results = test_suite.run_tests()
    
    # Save results to custom output file if specified
    if args.output:
        results.save_results(args.output)
    else:
        # In CI mode, always save results with a consistent filename
        if args.ci_mode:
            results.save_results("integration_test_results_ci.json")
    
    # Return exit code based on test results
    summary = results.get_summary()
    
    # Print a final summary for CI environments
    if args.ci_mode:
        print(f"\n==== CI TEST SUMMARY ====")
        print(f"Tests: {summary['total']} | Passed: {summary['passed']} | Failed: {summary['failed']} | Errors: {summary['errors']} | Skipped: {summary['skipped']}")
        print(f"Pass rate: {summary['pass_rate']:.1%}")
        print(f"Categories: {', '.join(categories) if categories else 'all'}")
        print(f"Platforms: {', '.join(hardware_platforms) if hardware_platforms else 'auto-detected'}")
    
    # In CI mode, only consider failures in the explicitly requested categories as true failures
    if args.ci_mode and categories:
        critical_failures = 0
        for result in results.results:
            if result.status in ["fail", "error"] and result.category in categories:
                critical_failures += 1
                
        if critical_failures > 0:
            print(f"Found {critical_failures} critical failures in requested categories")
            sys.exit(1)
        else:
            print("No critical failures in requested categories")
            sys.exit(0)
    else:
        # Standard mode - any failure causes a non-zero exit code
        if summary["failed"] > 0 or summary["errors"] > 0:
            sys.exit(1)
        else:
            sys.exit(0)


if __name__ == "__main__":
    main()