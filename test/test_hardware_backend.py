import os
import sys
import importlib

# Add the parent directory to sys.path to import modules correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestHardwareBackend:
    def __init__(self, resources, metadata):
        self.resources = resources
        self.metadata = metadata
        self.hardware_platforms = ["cpu", "webnn", "cuda", "openvino", "qualcomm", "apple"]
        # Import all skill tests from skills folder
        self.skill_modules = self._import_skill_modules()
        return None
    
    def _import_skill_modules(self):
        """Import all skill test modules from the skills folder"""
        skills_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")
        skill_modules = {}
        if not os.path.exists(skills_dir):
            print(f"Warning: Skills directory not found at {skills_dir}")
            return skill_modules

        for filename in os.listdir(skills_dir):
            if filename.startswith("test_") and filename.endswith(".py"):
                module_name = filename[:-3]  # Remove .py extension
                try:
                    module = importlib.import_module(f"test.skills.{module_name}")
                    skill_modules[module_name] = module
                except ImportError as e:
                    print(f"Error importing {module_name}: {e}")
                    
        return skill_modules

    def test_cpu(self):
        """Test all skills on CPU hardware"""
        print("\n=== Testing skills on CPU ===")
        results = {}
        
        for module_name, module in self.skill_modules.items():
            print(f"Testing {module_name} on CPU...")
            try:
                # Get the test class from the module
                test_class_name = next((name for name in dir(module) if name.startswith('test_') and not name.startswith('test_on_')), None)
                if not test_class_name:
                    print(f"  Warning: No test class found in {module_name}")
                    continue
                    
                test_class = getattr(module, test_class_name)
                # Initialize the test class with CPU configuration
                test_instance = test_class(
                    resources={"hardware": "cpu", **self.resources}, 
                    metadata={"platform": "cpu", **self.metadata}
                )
                # Run the test
                if hasattr(test_instance, "__test__"):
                    result = test_instance.__test__()
                    results[module_name] = result
                    print(f"  {module_name} on CPU: {'SUCCESS' if result else 'FAILED'}")
                else:
                    print(f"  Warning: {module_name} has no __test__ method")
            except Exception as e:
                print(f"  Error testing {module_name} on CPU: {e}")
                results[module_name] = str(e)
                
        return results

    def test_webnn(self):
        """Test all skills on WebNN hardware"""
        print("\n=== Testing skills on WebNN ===")
        print("WebNN tests not implemented yet")
        # Test implementation would be similar to test_cpu but with WebNN configuration
        return {"status": "not implemented"}
    
    def test_cuda(self):
        """Test all skills on CUDA hardware"""
        print("\n=== Testing skills on CUDA ===")
        print("CUDA tests not implemented yet")
        # Test implementation would be similar to test_cpu but with CUDA configuration
        return {"status": "not implemented"}

    def test_openvino(self):
        """Test all skills on OpenVINO hardware"""
        print("\n=== Testing skills on OpenVINO ===")
        print("OpenVINO tests not implemented yet")
        # Test implementation would be similar to test_cpu but with OpenVINO configuration
        return {"status": "not implemented"}
    
    def test_qualcomm(self):
        """Test all skills on Qualcomm hardware"""
        print("\n=== Testing skills on Qualcomm ===")
        print("Qualcomm tests not implemented yet")
        # Test implementation would be similar to test_cpu but with Qualcomm configuration
        return {"status": "not implemented"}
    
    def test_apple(self):
        """Test all skills on Apple hardware"""
        print("\n=== Testing skills on Apple ===") 
        print("Apple tests not implemented yet")
        # Test implementation would be similar to test_cpu but with Apple configuration
        return {"status": "not implemented"}
    
    def __test__(self):
        """Run tests on all hardware platforms"""
        results = {}
        
        # Start with CPU tests which should work on all platforms
        results["cpu"] = self.test_cpu()
        
        # Run other hardware tests based on availability
        # We could add platform detection here in the future
        
        if os.environ.get("TEST_WEBNN", "").lower() in ("1", "true", "yes"):
            results["webnn"] = self.test_webnn()
        
        if os.environ.get("TEST_CUDA", "").lower() in ("1", "true", "yes"):
            results["cuda"] = self.test_cuda()
        
        if os.environ.get("TEST_OPENVINO", "").lower() in ("1", "true", "yes"):
            results["openvino"] = self.test_openvino()
        
        if os.environ.get("TEST_QUALCOMM", "").lower() in ("1", "true", "yes"):
            results["qualcomm"] = self.test_qualcomm()
        
        if os.environ.get("TEST_APPLE", "").lower() in ("1", "true", "yes"):
            results["apple"] = self.test_apple()
        
        return results

# Backwards compatibility - keep old name available
test_hardware_backend = TestHardwareBackend

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Test hardware backends')
    parser.add_argument('--platform', choices=['cpu', 'webnn', 'cuda', 'openvino', 'qualcomm', 'apple', 'all'], 
                        default='cpu', help='Hardware platform to test')
    args = parser.parse_args()
    
    # Initialize tester
    tester = TestHardwareBackend(resources={}, metadata={})
    
    # Run tests based on selected platform
    if args.platform == 'all':
        results = tester.__test__()
        print("\n=== Summary ===")
        for platform, platform_results in results.items():
            print(f"{platform}: {len(platform_results)} skills tested")
    else:
        test_method = getattr(tester, f"test_{args.platform}")
        results = test_method()
        print(f"\n=== Summary for {args.platform} ===")
        print(f"{len(results)} skills tested")
    
    print("\nHardware backend tests completed")