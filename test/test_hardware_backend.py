import os
import sys
import json
import importlib
from datetime import datetime

# Add the parent directory to sys.path to import modules correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestHardwareBackend:
    def __init__(self, resources, metadata):
        self.resources = resources
        self.metadata = metadata
        self.hardware_platforms = ["cpu", "webnn", "cuda", "openvino", "qualcomm", "apple"]
        # Import all skill tests from skills folder
        self.skill_modules = self._import_skill_modules()
        
        # Setup paths for results
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.collected_results_dir = os.path.join(self.test_dir, "collected_results")
        self.expected_results_dir = os.path.join(self.test_dir, "expected_results")
        
        # Create results directory if it doesn't exist
        os.makedirs(self.collected_results_dir, exist_ok=True)
        
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

    def _save_results(self, platform, results):
        """Save test results to a JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hardware_{platform}_results_{timestamp}.json"
        filepath = os.path.join(self.collected_results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        print(f"Results saved to {filepath}")
        return filepath
    
    def _compare_with_expected(self, platform, results, results_file):
        """Compare test results with expected results"""
        expected_file = os.path.join(self.expected_results_dir, f"expected_{platform}_results.json")
        
        if not os.path.exists(expected_file):
            print(f"No expected results file found at {expected_file}")
            return False
            
        try:
            with open(expected_file, 'r') as f:
                expected_results = json.load(f)
                
            # Count matches, mismatches, and missing tests
            matches = 0
            mismatches = 0
            missing = 0
            
            for module_name, expected in expected_results.items():
                if module_name in results:
                    if results[module_name] == expected:
                        matches += 1
                    else:
                        mismatches += 1
                        print(f"Mismatch in {module_name}: expected {expected}, got {results[module_name]}")
                else:
                    missing += 1
                    print(f"Missing test result for {module_name}")
            
            # Check for extra tests not in expected results
            extra = 0
            for module_name in results:
                if module_name not in expected_results:
                    extra += 1
                    print(f"Extra test result for {module_name}")
                    
            print(f"\n=== Comparison with expected results ===")
            print(f"Matches: {matches}")
            print(f"Mismatches: {mismatches}")
            print(f"Missing: {missing}")
            print(f"Extra: {extra}")
            
            # Save comparison results alongside the test results
            comparison = {
                "matches": matches,
                "mismatches": mismatches,
                "missing": missing,
                "extra": extra,
                "total_expected": len(expected_results),
                "total_actual": len(results)
            }
            
            comparison_file = results_file.replace(".json", "_comparison.json")
            with open(comparison_file, 'w') as f:
                json.dump(comparison, f, indent=2)
                
            return matches == len(expected_results) and mismatches == 0 and missing == 0
            
        except Exception as e:
            print(f"Error comparing with expected results: {e}")
            return False

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
        
        # Save and compare results
        results_file = self._save_results("cpu", results)
        self._compare_with_expected("cpu", results, results_file)
                
        return results

    def test_webnn(self):
        """Test all skills on WebNN hardware"""
        print("\n=== Testing skills on WebNN ===")
        print("WebNN tests not implemented yet")
        # Test implementation would be similar to test_cpu but with WebNN configuration
        results = {"status": "not implemented"}
        
        # Save and compare results
        results_file = self._save_results("webnn", results)
        self._compare_with_expected("webnn", results, results_file)
        
        return results
    
    def test_cuda(self):
        """Test all skills on CUDA hardware"""
        print("\n=== Testing skills on CUDA ===")
        print("CUDA tests not implemented yet")
        # Test implementation would be similar to test_cpu but with CUDA configuration
        results = {"status": "not implemented"}
        
        # Save and compare results
        results_file = self._save_results("cuda", results)
        self._compare_with_expected("cuda", results, results_file)
        
        return results

    def test_openvino(self):
        """Test all skills on OpenVINO hardware"""
        print("\n=== Testing skills on OpenVINO ===")
        print("OpenVINO tests not implemented yet")
        # Test implementation would be similar to test_cpu but with OpenVINO configuration
        results = {"status": "not implemented"}
        
        # Save and compare results
        results_file = self._save_results("openvino", results)
        self._compare_with_expected("openvino", results, results_file)
        
        return results
    
    def test_qualcomm(self):
        """Test all skills on Qualcomm hardware"""
        print("\n=== Testing skills on Qualcomm ===")
        print("Qualcomm tests not implemented yet")
        # Test implementation would be similar to test_cpu but with Qualcomm configuration
        results = {"status": "not implemented"}
        
        # Save and compare results
        results_file = self._save_results("qualcomm", results)
        self._compare_with_expected("qualcomm", results, results_file)
        
        return results
    
    def test_apple(self):
        """Test all skills on Apple hardware"""
        print("\n=== Testing skills on Apple ===") 
        print("Apple tests not implemented yet")
        # Test implementation would be similar to test_cpu but with Apple configuration
        results = {"status": "not implemented"}
        
        # Save and compare results
        results_file = self._save_results("apple", results)
        self._compare_with_expected("apple", results, results_file)
        
        return results
    
    def __test__(self):
        """Run tests on all hardware platforms"""
        all_results = {}
        overall_success = True
        
        # Start with CPU tests which should work on all platforms
        cpu_results = self.test_cpu()
        all_results["cpu"] = cpu_results
        
        # Run other hardware tests based on availability
        # We could add platform detection here in the future
        
        if os.environ.get("TEST_WEBNN", "").lower() in ("1", "true", "yes"):
            webnn_results = self.test_webnn()
            all_results["webnn"] = webnn_results
        
        if os.environ.get("TEST_CUDA", "").lower() in ("1", "true", "yes"):
            cuda_results = self.test_cuda()
            all_results["cuda"] = cuda_results
        
        if os.environ.get("TEST_OPENVINO", "").lower() in ("1", "true", "yes"):
            openvino_results = self.test_openvino()
            all_results["openvino"] = openvino_results
        
        if os.environ.get("TEST_QUALCOMM", "").lower() in ("1", "true", "yes"):
            qualcomm_results = self.test_qualcomm()
            all_results["qualcomm"] = qualcomm_results
        
        if os.environ.get("TEST_APPLE", "").lower() in ("1", "true", "yes"):
            apple_results = self.test_apple()
            all_results["apple"] = apple_results
        
        # Save combined results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_filename = f"hardware_all_results_{timestamp}.json"
        combined_filepath = os.path.join(self.collected_results_dir, combined_filename)
        
        with open(combined_filepath, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
            
        print(f"Combined results saved to {combined_filepath}")
        
        return all_results

# Backwards compatibility - keep old name available
test_hardware_backend = TestHardwareBackend

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Test hardware backends')
    parser.add_argument('--platform', choices=['cpu', 'webnn', 'cuda', 'openvino', 'qualcomm', 'apple', 'all'], 
                        default='cpu', help='Hardware platform to test')
    parser.add_argument('--compare', action='store_true', help='Compare with expected results')
    parser.add_argument('--save-expected', action='store_true', help='Save current results as expected results')
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
    
    # Save as expected results if requested
    if args.save_expected:
        expected_file = os.path.join(tester.expected_results_dir, f"expected_{args.platform}_results.json")
        os.makedirs(tester.expected_results_dir, exist_ok=True)
        with open(expected_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Saved as expected results: {expected_file}")
    
    print("\nHardware backend tests completed")