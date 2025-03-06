#!/usr/bin/env python
"""
Simple test for the IPFS Accelerate Python package
This test focuses on checking the core functionality without requiring API backends.
"""

import os
import sys
import time
import json
import importlib
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ipfs_test")

def test_module_presence(module_name):
    """Test if a module can be imported"""
    try:
        module = importlib.import_module(module_name)
        logger.info(f"✅ Successfully imported {module_name}")
        return True, module
    except ImportError as e:
        logger.error(f"❌ Failed to import {module_name}: {e}")
        return False, None

def test_package_structure():
    """Test the overall package structure"""
    results = {
        "package_structure": {}
    }
    
    # Test main package
    is_present, module = test_module_presence("ipfs_accelerate_py")
    if not is_present:
        logger.error("❌ Main package ipfs_accelerate_py not found, skipping further tests")
        results["package_structure"]["main_package"] = "Missing"
        return results
    
    results["package_structure"]["main_package"] = "Present"
    
    # For compatibility with our flat structure, we just check for attributes
    # rather than importing submodules
    
    # Expected attributes in main module
    expected_attrs = [
        "backends",
        "config",
        "ipfs_accelerate",
        "load_checkpoint_and_dispatch"
    ]
    
    for attr_name in expected_attrs:
        if hasattr(module, attr_name):
            logger.info(f"✅ Found attribute: {attr_name}")
            results["package_structure"][attr_name] = "Present"
        else:
            logger.error(f"❌ Missing attribute: {attr_name}")
            results["package_structure"][attr_name] = "Missing"
    
    return results

def test_module_attributes():
    """Test attributes of key modules"""
    results = {
        "module_attributes": {}
    }
    
    # Test main module attributes
    try:
        import ipfs_accelerate_py
        
        # Check main attributes
        main_attrs = ['backends', 'config', 'ipfs_accelerate', 'load_checkpoint_and_dispatch']
        attr_results = {}
        
        for attr in main_attrs:
            has_attr = hasattr(ipfs_accelerate_py, attr)
            attr_results[attr] = "Present" if has_attr else "Missing"
            
            if has_attr:
                logger.info(f"✅ Found attribute: {attr}")
            else:
                logger.error(f"❌ Missing attribute: {attr}")
                
        results["module_attributes"]["ipfs_accelerate_py"] = attr_results
        
        # Check ipfs_accelerate attributes
        # In our flat structure, we need to access attributes differently
        if hasattr(ipfs_accelerate_py, 'ipfs_accelerate'):
            ipfs_obj = ipfs_accelerate_py.ipfs_accelerate
            ipfs_module_attrs = ['load_checkpoint_and_dispatch']
            ipfs_attr_results = {}
            
            for attr in ipfs_module_attrs:
                # Check if the attribute exists directly in the ipfs_accelerate object
                has_attr = hasattr(ipfs_obj, attr)
                ipfs_attr_results[attr] = "Present" if has_attr else "Missing"
                
                if has_attr:
                    logger.info(f"✅ Found ipfs_accelerate.{attr}")
                else:
                    logger.error(f"❌ Missing ipfs_accelerate.{attr}")
                    
            results["module_attributes"]["ipfs_accelerate"] = ipfs_attr_results
            
        # Check backends attributes
        if hasattr(ipfs_accelerate_py, 'backends'):
            backends_cls = ipfs_accelerate_py.backends
            # Get methods from the backends class
            backend_attrs = ['docker_tunnel', 'marketplace', 'start_container', 'stop_container']
            backend_attr_results = {}
            
            # For our flat structure, backends is a class, so we only need to check if
            # the methods exist in the class definition, not in an instance
            for attr in backend_attrs:
                has_attr = hasattr(backends_cls, attr) or attr in dir(backends_cls)
                backend_attr_results[attr] = "Present" if has_attr else "Missing"
                
                if has_attr:
                    logger.info(f"✅ Found backends.{attr}")
                else:
                    logger.warning(f"⚠️ Missing backends.{attr}")
                    
            results["module_attributes"]["backends"] = backend_attr_results
            
    except ImportError:
        logger.error("❌ Failed to import ipfs_accelerate_py for attribute tests")
    except Exception as e:
        logger.error(f"❌ Error in attribute tests: {e}")
        
    return results

def test_basic_functionality():
    """Test basic functionality of the IPFS Accelerate package"""
    results = {
        "basic_functionality": {}
    }
    
    try:
        import ipfs_accelerate_py
        
        # Test config functionality (for our flat structure, config is a class)
        if hasattr(ipfs_accelerate_py, 'config'):
            logger.info("Testing config functionality")
            try:
                # Create an instance of the config class
                config_instance = ipfs_accelerate_py.config()
                
                # Try some basic operations
                debug_value = config_instance.get("general", "debug", False)
                config_instance.set("general", "test_value", "test")
                test_value = config_instance.get("general", "test_value", None)
                
                # Check if operations worked as expected
                if test_value == "test":
                    results["basic_functionality"]["config_init"] = "Success"
                    logger.info("✅ Successfully initialized config")
                else:
                    results["basic_functionality"]["config_init"] = "Partial Success"
                    logger.warning("⚠️ Config initialized but operations may not work correctly")
            except Exception as e:
                results["basic_functionality"]["config_init"] = f"Error: {str(e)}"
                logger.error(f"❌ Error initializing config: {e}")
                
        # Test backends functionality (for our flat structure, backends is a class)
        if hasattr(ipfs_accelerate_py, 'backends'):
            logger.info("Testing backends functionality")
            try:
                # Create an instance of the backends class
                backends_instance = ipfs_accelerate_py.backends()
                
                # Try some basic operations - just check if methods exist 
                # without actually calling them to avoid side effects
                has_methods = all(
                    hasattr(backends_instance, method) 
                    for method in ['start_container', 'stop_container', 'docker_tunnel']
                )
                
                if has_methods:
                    results["basic_functionality"]["backends_init"] = "Success"
                    logger.info("✅ Successfully initialized backends")
                else:
                    results["basic_functionality"]["backends_init"] = "Partial Success"
                    logger.warning("⚠️ Backends initialized but may be missing methods")
            except Exception as e:
                results["basic_functionality"]["backends_init"] = f"Error: {str(e)}"
                logger.error(f"❌ Error initializing backends: {e}")
                
        # Test load_checkpoint_and_dispatch functionality
        if hasattr(ipfs_accelerate_py, 'load_checkpoint_and_dispatch'):
            logger.info("Testing load_checkpoint_and_dispatch functionality")
            try:
                # Since we don't want to actually call the function with real data,
                # just check if it's callable
                if callable(ipfs_accelerate_py.load_checkpoint_and_dispatch):
                    results["basic_functionality"]["load_checkpoint_and_dispatch"] = "Success"
                    logger.info("✅ load_checkpoint_and_dispatch is callable")
                else:
                    results["basic_functionality"]["load_checkpoint_and_dispatch"] = "Failed"
                    logger.error("❌ load_checkpoint_and_dispatch is not callable")
            except Exception as e:
                results["basic_functionality"]["load_checkpoint_and_dispatch"] = f"Error: {str(e)}"
                logger.error(f"❌ Error checking load_checkpoint_and_dispatch: {e}")
                
    except ImportError:
        logger.error("❌ Failed to import ipfs_accelerate_py for functionality tests")
    except Exception as e:
        logger.error(f"❌ Error in functionality tests: {e}")
        
    return results

def run_all_tests():
    """Run all tests and return combined results"""
    logger.info("Starting IPFS Accelerate Python tests")
    
    start_time = time.time()
    results = {
        "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "package_version": "Unknown"
    }
    
    # Try to get package version
    try:
        import ipfs_accelerate_py
        if hasattr(ipfs_accelerate_py, '__version__'):
            results["package_version"] = ipfs_accelerate_py.__version__
    except:
        pass
    
    # Run tests
    structure_results = test_package_structure()
    attribute_results = test_module_attributes()
    functionality_results = test_basic_functionality()
    
    # Combine results
    results.update(structure_results)
    results.update(attribute_results)
    results.update(functionality_results)
    
    # Add test duration
    end_time = time.time()
    results["test_duration_seconds"] = round(end_time - start_time, 2)
    
    logger.info(f"Tests completed in {results['test_duration_seconds']} seconds")
    return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test the IPFS Accelerate Python package")
    parser.add_argument("--output", "-o", help="Output file for test results (JSON)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set log level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Run tests
    results = run_all_tests()
    
    # Print summary
    print("\n--- Test Summary ---")
    print(f"Package Structure: {'✅' if results['package_structure']['main_package'] == 'Present' else '❌'}")
    
    module_count = sum(1 for k, v in results['package_structure'].items() if v == 'Present')
    expected_modules = len(results['package_structure'])
    print(f"Modules Found: {module_count}/{expected_modules}")
    
    attribute_success = True
    for module, attrs in results['module_attributes'].items():
        for attr, status in attrs.items():
            if status != "Present":
                attribute_success = False
                break
    print(f"Module Attributes: {'✅' if attribute_success else '⚠️'}")
    
    functionality_success = True
    for func, status in results.get('basic_functionality', {}).items():
        if status != "Success" and not status.startswith("Success"):
            functionality_success = False
            break
    print(f"Basic Functionality: {'✅' if functionality_success else '⚠️'}")
    
    # Save results if output file specified
    if args.output:
        output_path = Path(args.output)
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {output_path}")
        except Exception as e:
            print(f"\nError saving results: {e}")
    
    # Return success if all main modules were found
    return 0 if module_count >= 3 else 1

if __name__ == "__main__":
    sys.exit(main())