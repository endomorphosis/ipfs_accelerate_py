#!/usr/bin/env python
"""
Minimal test for the IPFS Accelerate Python package
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

def test_basic_imports():
    """Test if basic modules can be imported"""
    results = {}}}}:
        "imports": {}}}}}
        }
    
    # Our implementation uses a flat structure, so we only need to import the main module
    # and then check for attributes rather than importing submodules directly
    
    try:
        # Import the main module
        module = importlib.import_module("ipfs_accelerate_py")
        logger.info(f"✅ Successfully imported ipfs_accelerate_py")
        results["imports"]["ipfs_accelerate_py"] = "Success"
        ,
        # Check for required attributes instead of submodules
        required_attrs = {}}}}
        "config": "Configuration management",
        "backends": "Backend operations",
        "ipfs_accelerate": "Core IPFS functionality",
        "load_checkpoint_and_dispatch": "Checkpoint loading function"
        }
        
        for attr_name, description in required_attrs.items():
            if hasattr(module, attr_name):
                attr_type = type(getattr(module, attr_name)).__name__
                logger.info(f"✅ Found required attribute: {}}}}attr_name} ({}}}}attr_type})")
                results["imports"][f"ipfs_accelerate_py.{}}}}attr_name}"] = "Success",
            else:
                logger.error(f"❌ Missing required attribute: {}}}}attr_name}")
                results["imports"][f"ipfs_accelerate_py.{}}}}attr_name}"] = f"Failed: attribute not found",
    except ImportError as e:
        logger.error(f"❌ Failed to import ipfs_accelerate_py: {}}}}e}")
        results["imports"]["ipfs_accelerate_py"] = f"Failed: {}}}}str(e)}"
        ,
        # Mark all submodules as failed since the main module couldn't be imported
        for attr_name in ["config", "backends", "ipfs_accelerate", "load_checkpoint_and_dispatch"]:,
        results["imports"][f"ipfs_accelerate_py.{}}}}attr_name}"] = f"Failed: main module not imported"
        ,
                return results

def analyze_package():
    """Analyze the IPFS Accelerate Python package"""
    results = {}}}}
    "analysis": {}}}}}
    }
    
    try:
        import ipfs_accelerate_py
        
        # Check the package path
        package_path = getattr(ipfs_accelerate_py, '__file__', None)
        if package_path:
            results["analysis"]["package_path"] = package_path,
            logger.info(f"Package path: {}}}}package_path}")
        else:
            results["analysis"]["package_path"] = "Unknown",
            logger.warning("Package path unknown")
        
        # Check main attributes
            attributes = {}}}}}
        for attr_name in dir(ipfs_accelerate_py):
            if not attr_name.startswith('__'):
                try:
                    attr = getattr(ipfs_accelerate_py, attr_name)
                    attr_type = type(attr).__name__
                    attributes[attr_name] = attr_type,
                    logger.info(f"Attribute: {}}}}attr_name} ({}}}}attr_type})")
                except Exception as e:
                    attributes[attr_name] = f"Error: {}}}}str(e)}",
                    logger.error(f"Error getting attribute {}}}}attr_name}: {}}}}e}")
        
                    results["analysis"]["attributes"] = attributes
                    ,
    except ImportError:
        logger.error("❌ Failed to import ipfs_accelerate_py for analysis")
    except Exception as e:
        logger.error(f"❌ Error in analysis: {}}}}e}")
        
        return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Minimal test for IPFS Accelerate Python")
    parser.add_argument("--output", "-o", help="Output file for test results (JSON)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set log level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Run tests
        logger.info("Starting minimal IPFS Accelerate Python tests")
    
        results = {}}}}
        "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    
    # Run tests
        import_results = test_basic_imports()
        analysis_results = analyze_package()
    
    # Combine results
        results.update(import_results)
        results.update(analysis_results)
    
    # Print summary
        print("\n--- Test Summary ---")
        import_success = all(status == "Success" for status in results.get("imports", {}}}}}).values())
        print(f"Basic Imports: {}}}}'✅' if import_success else '❌'}")
    
    num_attributes = len(results.get("analysis", {}}}}}).get("attributes", {}}}}})):
        print(f"Package Attributes Found: {}}}}num_attributes}")
    
    # Print known attributes
        attributes = results.get("analysis", {}}}}}).get("attributes", {}}}}})
    if "backends" in attributes:
        print(f"- backends: {}}}}attributes['backends']}"),
    if "config" in attributes:
        print(f"- config: {}}}}attributes['config']}"),
    if "ipfs_accelerate" in attributes:
        print(f"- ipfs_accelerate: {}}}}attributes['ipfs_accelerate']}"),
    if "load_checkpoint_and_dispatch" in attributes:
        print(f"- load_checkpoint_and_dispatch: {}}}}attributes['load_checkpoint_and_dispatch']}")
        ,
    # Save results if output file specified:
    if args.output:
        output_path = Path(args.output)
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
                print(f"\nResults saved to {}}}}output_path}")
        except Exception as e:
            print(f"\nError saving results: {}}}}e}")
    
    # Return success if all main modules were imported
                return 0 if import_success else 1
:
if __name__ == "__main__":
    sys.exit(main())