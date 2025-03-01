#!/usr/bin/env python
import os
import sys
import json
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ipfs_accelerate_py'))

# Import the API implementation tester
from test_api_real_implementation import APIImplementationTester, CredentialManager

def main():
    """Test a single API implementation"""
    parser = argparse.ArgumentParser(description="Test a single API backend implementation")
    parser.add_argument("api", choices=[
        "openai", "claude", "gemini", "groq", "hf_tgi", "hf_tei", 
        "llvm", "ovms", "ollama", "s3_kit", "opea"
    ], help="The API to test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    parser.add_argument("--output", "-o", help="Output file for results (JSON format)")
    
    args = parser.parse_args()
    
    # Initialize the tester
    tester = APIImplementationTester()
    
    # Set up resources and metadata
    resources = {}
    metadata = tester.setup_metadata()
    
    # Import the specific API test module
    try:
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        module_name = f"apis.test_{args.api}"
        if args.api == "openai":
            module_name = "apis.test_openai_api"
            
        api_module = __import__(module_name, fromlist=["*"])
        test_class_name = f"test_{args.api}"
        if args.api == "openai":
            test_class_name = "test_openai_api"
            
        test_class = getattr(api_module, test_class_name)
    except ImportError:
        print(f"Error: Could not import test module for {args.api}")
        sys.exit(1)
    except AttributeError:
        print(f"Error: Could not find test class for {args.api}")
        sys.exit(1)
    
    print(f"\n=== Testing {args.api} API implementation ===\n")
    
    # Run the standard tests
    try:
        api_instance = test_class(resources=resources, metadata=metadata)
        print("Running standard API tests...")
        std_results = api_instance.test()
        
        if args.verbose:
            print("\nStandard test results:")
            print(json.dumps(std_results, indent=2))
            
        # Test real implementation
        print("\nChecking if implementation is real or mock...")
        impl_results = tester._test_implementation(api_instance, test_class_name)
        
        # Determine implementation type
        is_real = impl_results.get("is_real", False)
        status = impl_results.get("status", "ERROR")
        
        if is_real:
            impl_type = "REAL"
        elif status == "ERROR" and not is_real:
            impl_type = "ERROR/UNDETERMINED"
        else:
            impl_type = "MOCK"
            
        print(f"\nImplementation type: {impl_type}")
        print(f"Status: {status}")
        
        if "error" in impl_results:
            print(f"Error: {impl_results['error']}")
            
        # Save results if output file specified
        if args.output:
            results = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "api": args.api,
                "standard_tests": std_results,
                "implementation_check": impl_results,
                "summary": {
                    "type": impl_type,
                    "status": status
                }
            }
            
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
                
            print(f"\nResults saved to {args.output}")
            
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()