#!/usr/bin/env python3
"""
Script to set up the refactored test infrastructure.

This script:
1. Creates the directory structure for refactored tests
2. Creates base test classes and utilities
3. Creates sample migrated test files
4. Updates pytest.ini to support both original and refactored tests
"""

import os
import sys
import shutil
from pathlib import Path

# Base paths
TEST_DIR = Path('test')
REFACTORED_DIR = TEST_DIR / 'refactored_tests'
COMMON_DIR = REFACTORED_DIR / 'common'

# Test category directories
UNIT_DIR = REFACTORED_DIR / 'unit'
INTEGRATION_DIR = REFACTORED_DIR / 'integration'
MODELS_DIR = REFACTORED_DIR / 'models'
HARDWARE_DIR = REFACTORED_DIR / 'hardware'
BROWSER_DIR = REFACTORED_DIR / 'browser'
API_DIR = REFACTORED_DIR / 'api'
E2E_DIR = REFACTORED_DIR / 'e2e'

# Model type directories
TEXT_DIR = MODELS_DIR / 'text'
VISION_DIR = MODELS_DIR / 'vision'
AUDIO_DIR = MODELS_DIR / 'audio'

# Hardware type directories
WEBGPU_DIR = HARDWARE_DIR / 'webgpu'
WEBNN_DIR = HARDWARE_DIR / 'webnn'
PLATFORM_DIR = HARDWARE_DIR / 'platform'

def create_directories():
    """Create the directory structure for refactored tests."""
    print("Creating directory structure...")
    
    # Create main directories
    os.makedirs(COMMON_DIR, exist_ok=True)
    os.makedirs(UNIT_DIR, exist_ok=True)
    os.makedirs(INTEGRATION_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(HARDWARE_DIR, exist_ok=True)
    os.makedirs(BROWSER_DIR, exist_ok=True)
    os.makedirs(API_DIR, exist_ok=True)
    os.makedirs(E2E_DIR, exist_ok=True)
    
    # Create model type directories
    os.makedirs(TEXT_DIR, exist_ok=True)
    os.makedirs(VISION_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)
    
    # Create hardware type directories
    os.makedirs(WEBGPU_DIR, exist_ok=True)
    os.makedirs(WEBNN_DIR, exist_ok=True)
    os.makedirs(PLATFORM_DIR, exist_ok=True)
    
    # Create __init__.py files
    for directory in [
        REFACTORED_DIR, COMMON_DIR, UNIT_DIR, INTEGRATION_DIR, 
        MODELS_DIR, TEXT_DIR, VISION_DIR, AUDIO_DIR,
        HARDWARE_DIR, WEBGPU_DIR, WEBNN_DIR, PLATFORM_DIR,
        BROWSER_DIR, API_DIR, E2E_DIR
    ]:
        init_file = directory / '__init__.py'
        if not init_file.exists():
            with open(init_file, 'w') as f:
                f.write('"""Test module."""\n')

def create_base_test_class():
    """Create the BaseTest class."""
    print("Creating BaseTest class...")
    
    content = """
import pytest
import os
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

class BaseTest:
    """Base class for all test classes.
    
    Provides common functionality for test setup, teardown, and utilities.
    """
    
    @pytest.fixture(autouse=True)
    def setup_test(self):
        """Set up test environment before each test method."""
        self.setup_logging()
        self.test_start_time = self.get_current_time()
        yield
        self.cleanup()
        
    def setup_logging(self, level=logging.INFO):
        """Configure logging for tests."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
    def get_current_time(self) -> float:
        """Get current time for performance measurements."""
        import time
        return time.time()
        
    def measure_execution_time(self, start_time: float) -> float:
        """Measure execution time since start_time."""
        return self.get_current_time() - start_time
        
    def cleanup(self):
        """Clean up resources after test execution."""
        pass  # Override in subclasses as needed
        
    def assert_structure_matches(self, obj: Any, expected_structure: Dict[str, type]):
        """Assert that object has expected structure of attributes and types."""
        for attr, expected_type in expected_structure.items():
            assert hasattr(obj, attr), f"Object missing attribute: {attr}"
            if expected_type is not None:
                assert isinstance(getattr(obj, attr), expected_type), \
                    f"Attribute {attr} has wrong type. Expected {expected_type}, got {type(getattr(obj, attr))}"
                    
    def assert_lists_equal_unordered(self, list1: List, list2: List):
        """Assert that two lists contain the same elements, regardless of order."""
        assert len(list1) == len(list2), f"Lists have different lengths: {len(list1)} vs {len(list2)}"
        for item in list1:
            assert item in list2, f"Item {item} in first list but not in second list"
"""
    
    with open(COMMON_DIR / 'base_test.py', 'w') as f:
        f.write(content.lstrip())

def create_model_test_class():
    """Create the ModelTest class."""
    print("Creating ModelTest class...")
    
    content = """
from .base_test import BaseTest
import pytest
import os
import json
from typing import Any, Dict, List, Optional, Tuple, Union

class ModelTest(BaseTest):
    """Base class for model tests.
    
    Provides common functionality for testing machine learning models.
    """
    
    model_name: str = None
    model_type: str = None
    
    @pytest.fixture(autouse=True)
    def setup_model_test(self):
        """Set up test environment for model testing."""
        super().setup_test()
        self.verify_model_attributes()
        self.model = self.load_model()
        yield
        self.unload_model()
        
    def verify_model_attributes(self):
        """Verify that required model attributes are set."""
        assert self.model_name is not None, "model_name must be defined in the test class"
        assert self.model_type is not None, "model_type must be defined in the test class"
        
    def load_model(self):
        """Load the model for testing.
        
        Override in subclasses with specific model loading logic.
        """
        self.logger.info(f"Loading model: {self.model_name} (type: {self.model_type})")
        return None
        
    def unload_model(self):
        """Unload the model after testing.
        
        Override in subclasses with specific model unloading logic.
        """
        self.logger.info(f"Unloading model: {self.model_name}")
        self.model = None
        
    def assert_model_outputs_match_expected(self, outputs: Any, expected_outputs: Any, 
                                         tolerance: float = 1e-5):
        """Assert that model outputs match expected outputs within tolerance."""
        # Implement comparison logic based on output type
        # This is a placeholder for actual implementation
        pass
"""
    
    with open(COMMON_DIR / 'model_test.py', 'w') as f:
        f.write(content.lstrip())

def create_hardware_test_class():
    """Create the HardwareTest class."""
    print("Creating HardwareTest class...")
    
    content = """
from .base_test import BaseTest
import pytest
import os
import platform
from typing import Any, Dict, List, Optional, Set, Tuple, Union

class HardwareTest(BaseTest):
    """Base class for hardware compatibility tests.
    
    Provides common functionality for testing hardware compatibility.
    """
    
    required_hardware: Set[str] = set()
    
    @pytest.fixture(autouse=True)
    def setup_hardware_test(self):
        """Set up test environment for hardware testing."""
        super().setup_test()
        self.detect_available_hardware()
        self.verify_required_hardware()
        yield
        
    def detect_available_hardware(self):
        """Detect available hardware for testing."""
        self.available_hardware = set()
        
        # Basic system information
        self.system_info = {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
        }
        
        # Add CPU info
        self.available_hardware.add("cpu")
        
        # Detect GPU if available
        # This is a placeholder for actual implementation
        # Would use platform-specific methods to detect GPUs
        
        self.logger.info(f"Detected hardware: {self.available_hardware}")
        
    def verify_required_hardware(self):
        """Verify that required hardware is available."""
        if self.required_hardware:
            missing_hardware = self.required_hardware - self.available_hardware
            if missing_hardware:
                pytest.skip(f"Required hardware not available: {missing_hardware}")
                
    def assert_hardware_compatibility(self, feature: str, expected_compatibility: bool = True):
        """Assert that a specific hardware feature is compatible as expected."""
        # This is a placeholder for actual implementation
        pass
"""
    
    with open(COMMON_DIR / 'hardware_test.py', 'w') as f:
        f.write(content.lstrip())

def create_api_test_class():
    """Create the APITest class."""
    print("Creating APITest class...")
    
    content = """
from .base_test import BaseTest
import pytest
import requests
import json
from typing import Any, Dict, List, Optional, Tuple, Union

class APITest(BaseTest):
    """Base class for API tests.
    
    Provides common functionality for testing APIs.
    """
    
    api_base_url: str = None
    
    @pytest.fixture(autouse=True)
    def setup_api_test(self):
        """Set up test environment for API testing."""
        super().setup_test()
        self.verify_api_attributes()
        self.setup_api_client()
        yield
        self.teardown_api_client()
        
    def verify_api_attributes(self):
        """Verify that required API attributes are set."""
        assert self.api_base_url is not None, "api_base_url must be defined in the test class"
        
    def setup_api_client(self):
        """Set up API client for testing."""
        self.session = requests.Session()
        
    def teardown_api_client(self):
        """Clean up API client after testing."""
        if hasattr(self, 'session'):
            self.session.close()
            
    def make_api_request(self, method: str, endpoint: str, 
                       params: Optional[Dict] = None, 
                       data: Optional[Dict] = None, 
                       headers: Optional[Dict] = None) -> requests.Response:
        """Make an API request and return the response."""
        url = f"{self.api_base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        return self.session.request(method, url, params=params, json=data, headers=headers)
        
    def assert_successful_response(self, response: requests.Response):
        """Assert that an API response is successful."""
        assert response.ok, f"API request failed with status {response.status_code}: {response.text}"
"""
    
    with open(COMMON_DIR / 'api_test.py', 'w') as f:
        f.write(content.lstrip())

def create_browser_test_class():
    """Create the BrowserTest class."""
    print("Creating BrowserTest class...")
    
    content = """
from .base_test import BaseTest
import pytest
import os
from typing import Any, Dict, List, Optional, Tuple, Union

class BrowserTest(BaseTest):
    """Base class for browser tests.
    
    Provides common functionality for browser-specific testing.
    """
    
    browser_type: str = None
    
    @pytest.fixture(autouse=True)
    def setup_browser_test(self):
        """Set up test environment for browser testing."""
        super().setup_test()
        self.verify_browser_attributes()
        self.setup_browser()
        yield
        self.teardown_browser()
        
    def verify_browser_attributes(self):
        """Verify that required browser attributes are set."""
        assert self.browser_type is not None, "browser_type must be defined in the test class"
        
    def setup_browser(self):
        """Set up browser environment for testing."""
        self.logger.info(f"Setting up browser: {self.browser_type}")
        # This is a placeholder for actual browser setup
        # Would use selenium or similar tools in actual implementation
        
    def teardown_browser(self):
        """Clean up browser environment after testing."""
        self.logger.info(f"Tearing down browser: {self.browser_type}")
        # This is a placeholder for actual browser teardown
"""
    
    with open(COMMON_DIR / 'browser_test.py', 'w') as f:
        f.write(content.lstrip())

def create_test_utilities():
    """Create test utility modules."""
    print("Creating test utilities...")
    
    # Test fixtures
    fixtures_content = """
import pytest
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir

@pytest.fixture
def temp_file():
    """Create a temporary file for tests."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        file_path = tmp_file.name
    yield file_path
    os.unlink(file_path)

@pytest.fixture
def sample_model_outputs():
    """Provide sample model outputs for testing."""
    return {
        "text": ["Sample text output 1", "Sample text output 2"],
        "vision": [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]],
        "audio": [[[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]]],
    }

@pytest.fixture
def mock_api_response():
    """Provide mock API response for testing."""
    return {
        "status": "success",
        "data": {
            "results": [
                {"id": 1, "name": "Result 1"},
                {"id": 2, "name": "Result 2"},
            ]
        }
    }
"""
    
    with open(COMMON_DIR / 'test_fixtures.py', 'w') as f:
        f.write(fixtures_content.lstrip())
    
    # Test assertions
    assertions_content = """
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

def assert_tensors_equal(tensor1: np.ndarray, tensor2: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8):
    """Assert that two tensors are equal within tolerance."""
    assert np.allclose(tensor1, tensor2, rtol=rtol, atol=atol), \
        f"Tensors not equal within tolerance. Max difference: {np.max(np.abs(tensor1 - tensor2))}"

def assert_json_structure_matches(json_obj: Dict, expected_structure: Dict):
    """Assert that a JSON object matches the expected structure."""
    for key, expected_type in expected_structure.items():
        assert key in json_obj, f"JSON missing key: {key}"
        
        if isinstance(expected_type, dict):
            assert isinstance(json_obj[key], dict), f"Expected dict for key {key}, got {type(json_obj[key])}"
            assert_json_structure_matches(json_obj[key], expected_type)
        elif isinstance(expected_type, list) and len(expected_type) > 0:
            assert isinstance(json_obj[key], list), f"Expected list for key {key}, got {type(json_obj[key])}"
            if json_obj[key]:  # Only check if list is not empty
                assert_json_structure_matches(json_obj[key][0], expected_type[0])
        else:
            assert isinstance(json_obj[key], expected_type), \
                f"Type mismatch for key {key}. Expected {expected_type}, got {type(json_obj[key])}"

def assert_api_success(response_json: Dict):
    """Assert that an API response indicates success."""
    assert "status" in response_json, "Response missing 'status' field"
    assert response_json["status"] == "success", f"API returned non-success status: {response_json['status']}"

def assert_model_performance(execution_time: float, max_time: float):
    """Assert that model execution time is within acceptable range."""
    assert execution_time <= max_time, f"Model execution time ({execution_time:.4f}s) exceeds maximum ({max_time:.4f}s)"
"""
    
    with open(COMMON_DIR / 'test_assertions.py', 'w') as f:
        f.write(assertions_content.lstrip())
    
    # Test mocks
    mocks_content = """
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

class MockModel:
    """Mock model for testing."""
    
    def __init__(self, model_type: str = "text"):
        self.model_type = model_type
        self.initialized = True
        
    def predict(self, inputs: Any) -> Any:
        """Mock prediction method."""
        if self.model_type == "text":
            return ["Mock text output for: " + str(input) for input in inputs]
        elif self.model_type == "vision":
            # Return mock image classification results
            batch_size = len(inputs) if isinstance(inputs, list) else 1
            return np.random.rand(batch_size, 10)  # 10 classes
        elif self.model_type == "audio":
            # Return mock audio processing results
            batch_size = len(inputs) if isinstance(inputs, list) else 1
            return np.random.rand(batch_size, 5, 100)  # 5 segments, 100 features
        else:
            return None

class MockAPIClient:
    """Mock API client for testing."""
    
    def __init__(self, base_url: str = "https://api.example.com"):
        self.base_url = base_url
        self.requests = []
        
    def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Mock GET request."""
        self.requests.append({"method": "GET", "endpoint": endpoint, "params": params})
        return self._mock_response(endpoint)
        
    def post(self, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Mock POST request."""
        self.requests.append({"method": "POST", "endpoint": endpoint, "data": data})
        return self._mock_response(endpoint)
        
    def _mock_response(self, endpoint: str) -> Dict:
        """Generate mock response based on endpoint."""
        if endpoint == "models":
            return {
                "status": "success",
                "data": {
                    "models": [
                        {"id": 1, "name": "model1", "type": "text"},
                        {"id": 2, "name": "model2", "type": "vision"},
                    ]
                }
            }
        elif endpoint == "predict":
            return {
                "status": "success",
                "data": {
                    "predictions": ["Mock prediction 1", "Mock prediction 2"]
                }
            }
        else:
            return {
                "status": "error",
                "message": f"Unknown endpoint: {endpoint}"
            }
"""
    
    with open(COMMON_DIR / 'test_mocks.py', 'w') as f:
        f.write(mocks_content.lstrip())
    
    # Hardware detection
    hardware_detection_content = """
import platform
import os
import subprocess
import re
from typing import Dict, List, Optional, Set

def get_system_info() -> Dict[str, str]:
    """Get basic system information."""
    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
    }

def detect_available_hardware() -> Set[str]:
    """Detect available hardware for testing."""
    available_hardware = set(["cpu"])
    
    system = platform.system()
    
    # Check for CUDA GPUs on Linux/Windows
    if system in ("Linux", "Windows"):
        try:
            # Try to get NVIDIA GPU info (will fail if no NVIDIA GPU or driver installed)
            nvidia_smi_output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                universal_newlines=True
            )
            if nvidia_smi_output.strip():
                available_hardware.add("cuda")
                available_hardware.add("gpu")
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
            
    # Check for Metal on macOS
    if system == "Darwin":
        try:
            # Get macOS GPU info
            system_profiler_output = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType"],
                universal_newlines=True
            )
            if "Chipset Model" in system_profiler_output:
                available_hardware.add("metal")
                available_hardware.add("gpu")
        except subprocess.SubprocessError:
            pass
            
    # Check for WebGPU support (this would be browser-specific in reality)
    # This is a placeholder for actual detection logic
    
    # Check for WebNN support (this would be browser-specific in reality)
    # This is a placeholder for actual detection logic
    
    return available_hardware

def get_cpu_info() -> Dict[str, Any]:
    """Get detailed CPU information."""
    cpu_info = {
        "processor": platform.processor(),
        "cores": os.cpu_count(),
    }
    
    # For Linux, try to get more detailed info from /proc/cpuinfo
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpu_info_text = f.read()
                
            # Extract model name
            model_match = re.search(r"model name\s+:\s+(.*)", cpu_info_text)
            if model_match:
                cpu_info["model_name"] = model_match.group(1)
                
            # Extract CPU MHz
            mhz_match = re.search(r"cpu MHz\s+:\s+(.*)", cpu_info_text)
            if mhz_match:
                cpu_info["mhz"] = float(mhz_match.group(1))
        except:
            pass
            
    return cpu_info
"""
    
    with open(COMMON_DIR / 'hardware_detection.py', 'w') as f:
        f.write(hardware_detection_content.lstrip())

def create_sample_migrated_test():
    """Create a sample migrated test file."""
    print("Creating sample migrated test file...")
    
    content = """
import pytest
import numpy as np
from test.refactored_tests.common.model_test import ModelTest

@pytest.mark.refactored
class TestBertModel(ModelTest):
    """Tests for BERT model functionality."""
    
    model_name = "bert-base-uncased"
    model_type = "text"
    
    def load_model(self):
        """Load BERT model for testing."""
        try:
            from transformers import AutoModel, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name)
            return {"model": model, "tokenizer": tokenizer}
        except ImportError:
            pytest.skip("transformers package not installed")
        except Exception as e:
            pytest.skip(f"Failed to load model: {str(e)}")
        
    def test_should_encode_text_successfully(self):
        """Test that BERT model can encode text successfully."""
        if not self.model:
            pytest.skip("Model not loaded")
            
        # Prepare input
        text = ["Hello world", "Testing BERT model"]
        inputs = self.model["tokenizer"](text, return_tensors="pt", padding=True)
        
        # Run model
        start_time = self.get_current_time()
        outputs = self.model["model"](**inputs)
        execution_time = self.measure_execution_time(start_time)
        
        # Verify outputs
        self.logger.info(f"Model execution time: {execution_time:.4f}s")
        assert outputs.last_hidden_state is not None
        assert outputs.last_hidden_state.shape[0] == len(text)
        
    def test_should_handle_empty_input(self):
        """Test that BERT model handles empty input appropriately."""
        if not self.model:
            pytest.skip("Model not loaded")
            
        # Empty input should raise a specific exception
        with pytest.raises(ValueError):
            inputs = self.model["tokenizer"]([], return_tensors="pt", padding=True)
            self.model["model"](**inputs)
"""
    
    with open(TEXT_DIR / 'test_bert_model.py', 'w') as f:
        f.write(content.lstrip())

def create_base_classes_test():
    """Create a test for the base classes."""
    print("Creating base classes test...")
    
    content = """
import pytest
import time
from test.refactored_tests.common.base_test import BaseTest
from test.refactored_tests.common.model_test import ModelTest
from test.refactored_tests.common.hardware_test import HardwareTest
from test.refactored_tests.common.api_test import APITest
from test.refactored_tests.common.browser_test import BrowserTest

@pytest.mark.refactored
class TestBaseTestClass:
    """Tests for BaseTest class functionality."""
    
    def test_should_setup_logging(self):
        """Test that logging setup works correctly."""
        test_instance = BaseTest()
        test_instance.setup_test()
        assert hasattr(test_instance, 'logger')
        assert test_instance.logger.name == 'BaseTest'
        
    def test_should_measure_execution_time(self):
        """Test that execution time measurement works correctly."""
        test_instance = BaseTest()
        start_time = test_instance.get_current_time()
        time.sleep(0.1)  # Sleep for 100ms
        execution_time = test_instance.measure_execution_time(start_time)
        assert execution_time >= 0.1
        
    def test_should_assert_structure_matches(self):
        """Test that structure assertion works correctly."""
        test_instance = BaseTest()
        
        # Create a test object
        class TestObj:
            def __init__(self):
                self.attr1 = "value1"
                self.attr2 = 42
                
        obj = TestObj()
        
        # Test with matching structure
        test_instance.assert_structure_matches(obj, {
            "attr1": str,
            "attr2": int,
        })
        
        # Test with missing attribute
        with pytest.raises(AssertionError):
            test_instance.assert_structure_matches(obj, {
                "attr1": str,
                "attr3": str,
            })
            
        # Test with wrong type
        with pytest.raises(AssertionError):
            test_instance.assert_structure_matches(obj, {
                "attr1": int,
                "attr2": int,
            })

@pytest.mark.refactored
class TestModelTestClass:
    """Tests for ModelTest class functionality."""
    
    def test_should_require_model_attributes(self):
        """Test that ModelTest requires model_name and model_type."""
        class TestModelSubclass(ModelTest):
            pass
            
        test_instance = TestModelSubclass()
        with pytest.raises(AssertionError):
            test_instance.verify_model_attributes()
            
    def test_should_accept_valid_model_attributes(self):
        """Test that ModelTest accepts valid model_name and model_type."""
        class TestModelSubclass(ModelTest):
            model_name = "test_model"
            model_type = "test_type"
            
        test_instance = TestModelSubclass()
        test_instance.verify_model_attributes()  # Should not raise
"""
    
    with open(UNIT_DIR / 'test_base_classes.py', 'w') as f:
        f.write(content.lstrip())

def create_run_script():
    """Create a script to run the refactored tests."""
    print("Creating run script...")
    
    content = """#!/usr/bin/env python3
"""
Run refactored tests.
"""

import os
import sys
import pytest

def main():
    """Run refactored tests."""
    print("Running refactored tests...")
    
    # Add argument to identify refactored tests
    pytest_args = ["-m", "refactored"]
    
    # Add any command line args passed to this script
    pytest_args.extend(sys.argv[1:])
    
    # Add refactored tests directory
    pytest_args.append("test/refactored_tests")
    
    # Run pytest with the specified args
    return pytest.main(pytest_args)

if __name__ == "__main__":
    sys.exit(main())
"""
    
    with open(TEST_DIR / 'run_refactored_tests.py', 'w') as f:
        f.write(content)
    
    # Make the script executable
    os.chmod(TEST_DIR / 'run_refactored_tests.py', 0o755)

def update_pytest_ini():
    """Update pytest.ini for parallel test runs."""
    print("Updating pytest.ini...")
    
    pytest_ini_path = Path('pytest.ini')
    
    if pytest_ini_path.exists():
        # Backup existing file
        shutil.copy(pytest_ini_path, pytest_ini_path.with_suffix('.bak'))
        
        # Read existing content
        with open(pytest_ini_path, 'r') as f:
            content = f.read()
        
        # Check if markers section exists
        if 'markers =' in content:
            # Add our markers
            lines = content.splitlines()
            for i, line in enumerate(lines):
                if line.strip().startswith('markers ='):
                    # Find the end of the markers section
                    j = i
                    while j < len(lines) and (lines[j].strip().endswith(',') or j == i):
                        j += 1
                    
                    # Insert our markers
                    lines.insert(j, '    original: marks tests as original test suite')
                    lines.insert(j + 1, '    refactored: marks tests as refactored test suite')
                    
                    # Update content
                    content = '\n'.join(lines)
                    break
        else:
            # Add markers section
            content += '\nmarkers =\n    original: marks tests as original test suite\n    refactored: marks tests as refactored test suite\n'
        
        # Update testpaths if it exists
        if 'testpaths =' in content:
            lines = content.splitlines()
            for i, line in enumerate(lines):
                if line.strip().startswith('testpaths ='):
                    # Replace with our testpaths
                    lines[i] = 'testpaths = test test/refactored_tests'
                    
                    # Update content
                    content = '\n'.join(lines)
                    break
        else:
            # Add testpaths
            content += '\ntestpaths = test test/refactored_tests\n'
        
        # Write updated content
        with open(pytest_ini_path, 'w') as f:
            f.write(content)
    else:
        # Create new pytest.ini
        content = """[pytest]
testpaths = test test/refactored_tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    original: marks tests as original test suite
    refactored: marks tests as refactored test suite
"""
        
        with open(pytest_ini_path, 'w') as f:
            f.write(content)

def main():
    """Set up the refactored test infrastructure."""
    create_directories()
    
    # Create base classes
    create_base_test_class()
    create_model_test_class()
    create_hardware_test_class()
    create_api_test_class()
    create_browser_test_class()
    
    # Create test utilities
    create_test_utilities()
    
    # Create sample tests
    create_sample_migrated_test()
    create_base_classes_test()
    
    # Create run script
    create_run_script()
    
    # Update pytest.ini
    update_pytest_ini()
    
    print("\nRefactored test infrastructure set up successfully!")
    print("\nTo run the refactored tests:")
    print("  python test/run_refactored_tests.py")

if __name__ == "__main__":
    main()