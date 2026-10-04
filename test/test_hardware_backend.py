import os
import sys
import json
import importlib
from datetime import datetime

# Add the parent directory to sys.path to import modules correctly
sys.path.insert()))0, os.path.dirname()))os.path.dirname()))os.path.abspath()))__file__))))

class TestHardwareBackend:
    def __init__()))self, resources, metadata):
        self.resources = resources
        self.metadata = metadata
        self.hardware_platforms = ["cpu", "webnn", "webgpu", "cuda", "openvino", "qualcomm", "apple"],
        # Import all skill tests from skills folder
        self.skill_modules = self._import_skill_modules())))
        
        # Setup paths for results
        self.test_dir = os.path.dirname()))os.path.abspath()))__file__))
        self.collected_results_dir = os.path.join()))self.test_dir, "collected_results")
        self.expected_results_dir = os.path.join()))self.test_dir, "expected_results")
        
        # Create results directory if it doesn't exist
        os.makedirs()))self.collected_results_dir, exist_ok=True)
        
    return None
    :
    def _import_skill_modules()))self):
        """Import all skill test modules from the skills folder"""
        skills_dir = os.path.join()))os.path.dirname()))os.path.abspath()))__file__)), "skills")
        skill_modules = {}}}}
        if not os.path.exists()))skills_dir):
            print()))f"Warning: Skills directory not found at {}}}skills_dir}")
        return skill_modules

        for filename in os.listdir()))skills_dir):
            if filename.startswith()))"test_") and filename.endswith()))".py"):
                module_name = filename[:-3]  # Remove .py extension,
                try:
                    module = importlib.import_module()))f"test.skills.{}}}module_name}")
                    skill_modules[module_name] = module,
                except ImportError as e:
                    print()))f"Error importing {}}}module_name}: {}}}e}")
                    
                    return skill_modules

    def _save_results()))self, platform, results):
        """Save test results to a JSON file"""
        timestamp = datetime.now()))).strftime()))"%Y%m%d_%H%M%S")
        filename = f"hardware_{}}}platform}_results_{}}}timestamp}.json"
        filepath = os.path.join()))self.collected_results_dir, filename)
        
        with open()))filepath, 'w') as f:
            json.dump()))results, f, indent=2, default=str)
            
            print()))f"Results saved to {}}}filepath}")
        return filepath
    
    def _compare_with_expected()))self, platform, results, results_file):
        """Compare test results with expected results"""
        expected_file = os.path.join()))self.expected_results_dir, f"expected_{}}}platform}_results.json")
        
        if not os.path.exists()))expected_file):
            print()))f"No expected results file found at {}}}expected_file}")
        return False
            
        try:
            with open()))expected_file, 'r') as f:
                expected_results = json.load()))f)
                
            # Count matches, mismatches, and missing tests
                matches = 0
                mismatches = 0
                missing = 0
            
            for module_name, expected in expected_results.items()))):
                if module_name in results:
                    if results[module_name] == expected:,
                    matches += 1
                    else:
                        mismatches += 1
                        print()))f"Mismatch in {}}}module_name}: expected {}}}expected}, got {}}}results[module_name]}"),
                else:
                    missing += 1
                    print()))f"Missing test result for {}}}module_name}")
            
            # Check for extra tests not in expected results
                    extra = 0
            for module_name in results:
                if module_name not in expected_results:
                    extra += 1
                    print()))f"Extra test result for {}}}module_name}")
                    
                    print()))f"\n=== Comparison with expected results ===")
                    print()))f"Matches: {}}}matches}")
                    print()))f"Mismatches: {}}}mismatches}")
                    print()))f"Missing: {}}}missing}")
                    print()))f"Extra: {}}}extra}")
            
            # Save comparison results alongside the test results
                    comparison = {}}}
                    "matches": matches,
                    "mismatches": mismatches,
                    "missing": missing,
                    "extra": extra,
                    "total_expected": len()))expected_results),
                    "total_actual": len()))results)
                    }
            
                    comparison_file = results_file.replace()))".json", "_comparison.json")
            with open()))comparison_file, 'w') as f:
                json.dump()))comparison, f, indent=2)
                
                    return matches == len()))expected_results) and mismatches == 0 and missing == 0
            
        except Exception as e:
            print()))f"Error comparing with expected results: {}}}e}")
                    return False

    def test_cpu()))self):
        """Test all skills on CPU hardware"""
        print()))"\n=== Testing skills on CPU ===")
        results = {}}}}
        
        for module_name, module in self.skill_modules.items()))):
            print()))f"Testing {}}}module_name} on CPU...")
            try:
                # Get the test class from the module
                test_class_name = next()))()))name for name in dir()))module) if name.startswith()))'test_') and not name.startswith()))'test_on_')), None):::
                if not test_class_name:
                    print()))f"  Warning: No test class found in {}}}module_name}")
                    continue
                    
                    test_class = getattr()))module, test_class_name)
                # Initialize the test class with CPU configuration
                    test_instance = test_class()))
                    resources={}}}"hardware": "cpu", **self.resources}, 
                    metadata={}}}"platform": "cpu", **self.metadata}
                    )
                # Run the test
                if hasattr()))test_instance, "__test__"):
                    result = test_instance.__test__())))
                    results[module_name] = result,,,
                    print()))f"  {}}}module_name} on CPU: {}}}'SUCCESS' if result else 'FAILED'}"):
                else:
                    print()))f"  Warning: {}}}module_name} has no __test__ method")
            except Exception as e:
                print()))f"  Error testing {}}}module_name} on CPU: {}}}e}")
                results[module_name] = str()))e)
                ,        ,
        # Save and compare results
                results_file = self._save_results()))"cpu", results)
                self._compare_with_expected()))"cpu", results, results_file)
                
                    return results

    def test_webnn()))self):
        """Test all skills on WebNN hardware"""
        print()))"\n=== Testing skills on WebNN ===")
        print()))"WebNN tests not implemented yet")
        # Test implementation would be similar to test_cpu but with WebNN configuration
        results = {}}}"status": "not implemented"}
        
        # Save and compare results
        results_file = self._save_results()))"webnn", results)
        self._compare_with_expected()))"webnn", results, results_file)
        
                    return results
    
    def test_cuda()))self):
        """Test all skills on CUDA hardware"""
        print()))"\n=== Testing skills on CUDA ===")
        results = {}}}}
        
        # Check if CUDA is available
        cuda_available = False:
        try:
            import torch
            cuda_available = torch.cuda.is_available())))
            if cuda_available:
                device_count = torch.cuda.device_count())))
                device_name = torch.cuda.get_device_name()))0) if device_count > 0 else "Unknown":
                    print()))f"CUDA is available: {}}}device_count} device()))s) found")
                    print()))f"Using device: {}}}device_name}")
            else:
                print()))"CUDA is not available on this system")
                results["cuda_status"] = "CUDA not available",
        except ImportError:
            print()))"PyTorch not installed, CUDA tests will be skipped")
            results["cuda_status"] = "PyTorch not installed"
            ,
        if not cuda_available:
            # Skip tests but provide meaningful result
            results["status"] = "skipped",,
            results_file = self._save_results()))"cuda", results)
            self._compare_with_expected()))"cuda", results, results_file)
            return results
        
        # Clean up GPU memory before starting tests
            torch.cuda.empty_cache())))
        
        # Run tests for each skill module
        for module_name, module in self.skill_modules.items()))):
            print()))f"Testing {}}}module_name} on CUDA...")
            try:
                # Get the test class from the module
                test_class_name = next()))()))name for name in dir()))module) if name.startswith()))'test_') and not name.startswith()))'test_on_')), None):::
                if not test_class_name:
                    print()))f"  Warning: No test class found in {}}}module_name}")
                    continue
                    
                # Initialize the test class
                    test_class = getattr()))module, test_class_name)
                    test_instance = test_class()))
                    resources={}}}"hardware": "cuda", "torch": torch, **self.resources}, 
                    metadata={}}}"platform": "cuda", "device": "cuda:0", **self.metadata}
                    )
                
                # Run the test with timeout protection
                if hasattr()))test_instance, "__test__"):
                    # Set a timeout for CUDA tests ()))they can hang sometimes)
                    import threading
                    import time
                    
                    # Define a worker function that runs the test
                    def worker()))):
                        nonlocal result_container
                        try:
                            result_container["result"],, = test_instance.__test__()))),
                            result_container["completed"] = True,,
                        except Exception as e:
                            result_container["error"] = str()))e),,
                            result_container["completed"] = True,,
                    
                    # Initialize result container
                            result_container = {}}}"result": None, "error": None, "completed": False}
                    
                    # Start worker thread
                            thread = threading.Thread()))target=worker)
                            thread.daemon = True
                            thread.start())))
                    
                    # Wait for completion or timeout
                            timeout = 300  # 5 minutes timeout
                            start_time = time.time())))
                            while not result_container["completed"] and time.time()))) - start_time < timeout:,,
                            time.sleep()))1)
                    
                    # Check result
                            if result_container["completed"]:,,
                            if result_container["error"] is not None:,,
                            results[module_name] = f"Error: {}}}result_container['error']}",,
                            print()))f"  {}}}module_name} on CUDA: FAILED ())){}}}result_container['error']})"),,
                        else:
                            result = result_container["result"],,
                            results[module_name] = result,,,
                            status = "SUCCESS" if result else "FAILED"::
                                print()))f"  {}}}module_name} on CUDA: {}}}status}")
                    else:
                        results[module_name] = "Timeout: Test took too long",,
                        print()))f"  {}}}module_name} on CUDA: TIMEOUT ()))>300s)")
                        
                        # Try to clean up after timeout
                        try:
                            # Force CUDA cleanup
                            torch.cuda.empty_cache())))
                            print()))"  Cleaned CUDA memory after timeout")
                        except Exception as e:
                            print()))f"  Error cleaning up CUDA memory: {}}}e}")
                else:
                    print()))f"  Warning: {}}}module_name} has no __test__ method")
                    results[module_name] = "No __test__ method"
                    ,
                # Clean up after each test to avoid memory leaks
                    torch.cuda.empty_cache())))
                
            except Exception as e:
                print()))f"  Error testing {}}}module_name} on CUDA: {}}}e}")
                results[module_name] = str()))e)
                ,        ,
                # Clean up after errors as well
                try:
                    torch.cuda.empty_cache())))
                except Exception:
                    pass
        
        # Save and compare results
                    results_file = self._save_results()))"cuda", results)
                    self._compare_with_expected()))"cuda", results, results_file)
        
        # Final cleanup
        try:
            torch.cuda.empty_cache())))
            print()))"Final CUDA memory cleanup done")
        except Exception:
            pass
            
                    return results

    def test_openvino()))self):
        """Test all skills on OpenVINO hardware"""
        print()))"\n=== Testing skills on OpenVINO ===")
        results = {}}}}
        
        # Check if OpenVINO is available
        openvino_available = False:
        try:
            import openvino as ov
            ie = ov.Core())))
            openvino_available = True
            devices = ie.available_devices
            print()))f"OpenVINO is available: {}}}devices}")
            results["openvino_status"] = f"Available devices: {}}}devices}",
        except ImportError:
            print()))"OpenVINO not installed, tests will be skipped")
            results["openvino_status"] = "OpenVINO not installed",
        except Exception as e:
            print()))f"Error initializing OpenVINO: {}}}e}")
            results["openvino_status"] = f"Error: {}}}str()))e)}"
            ,
        if not openvino_available:
            # Skip tests but provide meaningful result
            results["status"] = "skipped",,
            results_file = self._save_results()))"openvino", results)
            self._compare_with_expected()))"openvino", results, results_file)
            return results
            
        # Check if we have a proper device to run OpenVINO
            has_device = False
        target_device = "CPU"  # Default to CPU:
        try:
            # Check if we have more than just CPU available:
            if "GPU" in devices:
                target_device = "GPU"
                has_device = True
                print()))"Using GPU device for OpenVINO")
            elif "CPU" in devices:
                has_device = True
                print()))"Using CPU device for OpenVINO")
                
            if not has_device:
                print()))"No suitable OpenVINO device found")
                results["status"] = "No suitable device",
                results_file = self._save_results()))"openvino", results)
                return results
                
        except Exception as e:
            print()))f"Error checking OpenVINO devices: {}}}e}")
            results["device_error"] = str()))e)
            ,
        # Create directory for OpenVINO IR models if it doesn't exist
            import os
            openvino_dir = os.path.join()))os.path.dirname()))os.path.abspath()))__file__)), "skills", "openvino_model")
            os.makedirs()))openvino_dir, exist_ok=True)
            
        # Run tests for each skill module:
        for module_name, module in self.skill_modules.items()))):
            print()))f"Testing {}}}module_name} on OpenVINO...")
            try:
                # Get the test class from the module
                test_class_name = next()))()))name for name in dir()))module) if name.startswith()))'test_') and not name.startswith()))'test_on_')), None):::
                if not test_class_name:
                    print()))f"  Warning: No test class found in {}}}module_name}")
                    continue
                    
                # Initialize the test class with OpenVINO configuration
                    test_class = getattr()))module, test_class_name)
                    test_instance = test_class()))
                    resources={}}}
                    "hardware": "openvino",
                    "openvino": ov,
                    "openvino_core": ie,
                    "openvino_device": target_device,
                    "openvino_model_dir": openvino_dir,
                    **self.resources
                    }, 
                    metadata={}}}
                    "platform": "openvino",
                    "device": target_device,
                    "model_dir": openvino_dir,
                    **self.metadata
                    }
                    )
                
                # Run the test with timeout protection
                if hasattr()))test_instance, "__test__"):
                    # Set a timeout for OpenVINO tests
                    import threading
                    import time
                    
                    # Define a worker function that runs the test
                    def worker()))):
                        nonlocal result_container
                        try:
                            result_container["result"],, = test_instance.__test__()))),
                            result_container["completed"] = True,,
                        except Exception as e:
                            result_container["error"] = str()))e),,
                            result_container["completed"] = True,,
                    
                    # Initialize result container
                            result_container = {}}}"result": None, "error": None, "completed": False}
                    
                    # Start worker thread
                            thread = threading.Thread()))target=worker)
                            thread.daemon = True
                            thread.start())))
                    
                    # Wait for completion or timeout
                            timeout = 300  # 5 minutes timeout
                            start_time = time.time())))
                            while not result_container["completed"] and time.time()))) - start_time < timeout:,,
                            time.sleep()))1)
                    
                    # Check result
                            if result_container["completed"]:,,
                            if result_container["error"] is not None:,,
                            results[module_name] = f"Error: {}}}result_container['error']}",,
                            print()))f"  {}}}module_name} on OpenVINO: FAILED ())){}}}result_container['error']})"),,
                        else:
                            result = result_container["result"],,
                            results[module_name] = result,,,
                            status = "SUCCESS" if result else "FAILED"::
                                print()))f"  {}}}module_name} on OpenVINO: {}}}status}")
                    else:
                        results[module_name] = "Timeout: Test took too long",,
                        print()))f"  {}}}module_name} on OpenVINO: TIMEOUT ()))>300s)")
                else:
                    print()))f"  Warning: {}}}module_name} has no __test__ method")
                    results[module_name] = "No __test__ method"
                    ,
            except Exception as e:
                print()))f"  Error testing {}}}module_name} on OpenVINO: {}}}e}")
                results[module_name] = str()))e)
                ,
        # Add OpenVINO version info to results
        try:
            results["openvino_version"] = ov.__version__,
        except Exception:
            results["openvino_version"] = "Unknown"
            ,
        # Save and compare results
            results_file = self._save_results()))"openvino", results)
            self._compare_with_expected()))"openvino", results, results_file)
        
            return results
    
    def test_qualcomm()))self):
        """Test all skills on Qualcomm hardware"""
        print()))"\n=== Testing skills on Qualcomm ===")
        print()))"Qualcomm tests not implemented yet")
        # Test implementation would be similar to test_cpu but with Qualcomm configuration
        results = {}}}"status": "not implemented"}
        
        # Save and compare results
        results_file = self._save_results()))"qualcomm", results)
        self._compare_with_expected()))"qualcomm", results, results_file)
        
            return results
    
    def test_apple()))self):
        """Test all skills on Apple hardware"""
        print()))"\n=== Testing skills on Apple ===") 
        print()))"Apple tests not implemented yet")
        # Test implementation would be similar to test_cpu but with Apple configuration
        results = {}}}"status": "not implemented"}
        
        # Save and compare results
        results_file = self._save_results()))"apple", results)
        self._compare_with_expected()))"apple", results, results_file)
        
            return results
    
    def __test__()))self, resources=None, metadata=None):
        """Run tests on all hardware platforms
        
        Args:
            resources ()))dict, optional): Dictionary containing resources. Defaults to None.
            metadata ()))dict, optional): Dictionary containing metadata. Defaults to None.
            
        Returns:
            dict: Test results for all hardware platforms
            """
        # Update resources and metadata if provided:
        if resources is not None:
            self.resources = resources
        if metadata is not None:
            self.metadata = metadata
            
            all_results = {}}}}
            overall_success = True
        
        # Start with CPU tests which should work on all platforms
            cpu_results = self.test_cpu())))
            all_results["cpu"] = cpu_results
            ,
        # Run other hardware tests based on availability
        # We could add platform detection here in the future
        
        if os.environ.get()))"TEST_WEBNN", "").lower()))) in ()))"1", "true", "yes"):
            webnn_results = self.test_webnn())))
            all_results["webnn"] = webnn_results
            ,
        if os.environ.get()))"TEST_CUDA", "").lower()))) in ()))"1", "true", "yes"):
            cuda_results = self.test_cuda())))
            all_results["cuda"] = cuda_results
            ,
        if os.environ.get()))"TEST_OPENVINO", "").lower()))) in ()))"1", "true", "yes"):
            openvino_results = self.test_openvino())))
            all_results["openvino"] = openvino_results
            ,
        if os.environ.get()))"TEST_QUALCOMM", "").lower()))) in ()))"1", "true", "yes"):
            qualcomm_results = self.test_qualcomm())))
            all_results["qualcomm"] = qualcomm_results
            ,
        if os.environ.get()))"TEST_APPLE", "").lower()))) in ()))"1", "true", "yes"):
            apple_results = self.test_apple())))
            all_results["apple"] = apple_results
            ,
        # Save combined results
            timestamp = datetime.now()))).strftime()))"%Y%m%d_%H%M%S")
            combined_filename = f"hardware_all_results_{}}}timestamp}.json"
            combined_filepath = os.path.join()))self.collected_results_dir, combined_filename)
        
        with open()))combined_filepath, 'w') as f:
            json.dump()))all_results, f, indent=2, default=str)
            
            print()))f"Combined results saved to {}}}combined_filepath}")
        
            return all_results

# Backwards compatibility - keep old name available
            test_hardware_backend = TestHardwareBackend

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser()))description='Test hardware backends')
    parser.add_argument()))'--platform', choices=['cpu', 'webnn', 'cuda', 'openvino', 'qualcomm', 'apple', 'all'], 
    default='cpu', help='Hardware platform to test')
    parser.add_argument()))'--compare', action='store_true', help='Compare with expected results')
    parser.add_argument()))'--save-expected', action='store_true', help='Save current results as expected results')
    args = parser.parse_args())))
    
    # Initialize tester
    tester = TestHardwareBackend()))resources={}}}}, metadata={}}}})
    
    # Run tests based on selected platform
    if args.platform == 'all':
        results = tester.__test__())))
        print()))"\n=== Summary ===")
        for platform, platform_results in results.items()))):
            print()))f"{}}}platform}: {}}}len()))platform_results)} skills tested")
    else:
        test_method = getattr()))tester, f"test_{}}}args.platform}")
        results = test_method())))
        print()))f"\n=== Summary for {}}}args.platform} ===")
        print()))f"{}}}len()))results)} skills tested")
    
    # Save as expected results if requested:
    if args.save_expected:
        expected_file = os.path.join()))tester.expected_results_dir, f"expected_{}}}args.platform}_results.json")
        os.makedirs()))tester.expected_results_dir, exist_ok=True)
        with open()))expected_file, 'w') as f:
            json.dump()))results, f, indent=2, default=str)
            print()))f"Saved as expected results: {}}}expected_file}")
    
            print()))"\nHardware backend tests completed")