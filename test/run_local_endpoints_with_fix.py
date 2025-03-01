#!/usr/bin/env python3
"""
This script applies the endpoint_handler fix and runs tests on all local endpoints.
It will test all models defined in mapped_models.json with the fixed implementation.
"""

import os
import sys
import json
import asyncio
import traceback
from datetime import datetime

# Add the parent directory to sys.path for proper imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variable to avoid tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import the needed modules
try:
    from fix_endpoint_handler import EndpointHandlerFixer
except ImportError as e:
    print(f"Error importing EndpointHandlerFixer: {e}")
    sys.exit(1)

async def main():
    print("=== Running Local Endpoints Tests with Fix ===\n")
    print("This script will apply the endpoint_handler fix to make handlers callable")
    print("and test all models defined in mapped_models.json\n")
    
    # Run the endpoint handler fixer
    fixer = EndpointHandlerFixer()
    
    # Create the persistent fix file
    fix_path = fixer.create_persistent_fix()
    print(f"Created persistent fix in {fix_path}")
    print("You can apply this fix permanently by updating the ipfs_accelerate_py module\n")
    
    # Run tests with the dynamic fix
    print("Running tests with dynamic fix...")
    results = await fixer.run_tests()
    
    # Calculate statistics
    success_count = sum(1 for result in results["model_results"].values() if result["status"] == "Success")
    total_count = len(results["model_results"])
    
    print(f"\nTested {total_count} models:")
    print(f"- {success_count} successful ({success_count/total_count*100:.1f}%)")
    print(f"- {total_count - success_count} failed ({(total_count - success_count)/total_count*100:.1f}%)")
    
    # Show instructions for applying the fix permanently
    print("\n=== How to Apply the Fix Permanently ===")
    print("1. The fix has been saved to endpoint_handler_fix.py")
    print("2. You can apply the fix to the ipfs_accelerate_py module by:")
    print("   - Finding the ipfs_accelerate.py file in your installed package")
    print("   - Replacing the endpoint_handler property with the implementation in endpoint_handler_fix.py")
    print("   - Adding the _create_mock_handler method to the class")
    print("3. Or you can use the dynamic fix approach in your own code:")
    print("   ```python")
    print("   from fix_endpoint_handler import EndpointHandlerFixer")
    print("   fixer = EndpointHandlerFixer()")
    print("   # Apply to your ipfs_accelerate_py instance")
    print("   fixer.apply_endpoint_handler_fix(your_ipfs_accelerate_instance)")
    print("   ```")

if __name__ == "__main__":
    asyncio.run(main())