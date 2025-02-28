#!/usr/bin/env python3
"""
Fix the LLAMA test to use actual testing rather than predefined results.
This script will modify the test_hf_llama.py file to use a local test model
and to run real tests instead of using predefined results.
"""

import os
import re

def fix_llama_test():
    """Fix the test_hf_llama.py file to use real tests"""
    
    # Path to the LLAMA test file
    test_file = '/home/barberb/ipfs_accelerate_py/test/skills/test_hf_llama.py'
    
    # Read the current content
    with open(test_file, 'r') as f:
        content = f.read()
    
    # Add the same output formatting as BERT test for better parsing
    if_main_pattern = re.compile(r'if __name__ == "__main__":\s+try:')
    if_main_block_start = if_main_pattern.search(content)
    
    if if_main_block_start:
        # Get the indentation level
        start_pos = if_main_block_start.start()
        indentation = ""
        for i in range(start_pos-1, 0, -1):
            if content[i] != ' ' and content[i] != '\t':
                indentation = content[i+1:start_pos]
                break
        
        # Build the new if __name__ == "__main__" block
        new_if_main_block = f'''if __name__ == "__main__":
    try:
        print("Starting LLaMA test...")
        this_llama = test_hf_llama()
        results = this_llama.__test__()
        print("LLaMA test completed")
        
        # Print test results in detailed format for better parsing
        status_dict = results.get("status", {{}})
        examples = results.get("examples", [])
        metadata = results.get("metadata", {{}})
        
        # Extract implementation status
        cpu_status = "UNKNOWN"
        cuda_status = "UNKNOWN"
        openvino_status = "UNKNOWN"
        
        for key, value in status_dict.items():
            if "cpu_" in key and "REAL" in value:
                cpu_status = "REAL"
            elif "cpu_" in key and "MOCK" in value:
                cpu_status = "MOCK"
                
            if "cuda_" in key and "REAL" in value:
                cuda_status = "REAL"
            elif "cuda_" in key and "MOCK" in value:
                cuda_status = "MOCK"
                
            if "openvino_" in key and "REAL" in value:
                openvino_status = "REAL"
            elif "openvino_" in key and "MOCK" in value:
                openvino_status = "MOCK"
                
        # Also look in examples
        for example in examples:
            platform = example.get("platform", "")
            impl_type = example.get("implementation_type", "")
            
            if platform == "CPU" and "REAL" in impl_type:
                cpu_status = "REAL"
            elif platform == "CPU" and "MOCK" in impl_type:
                cpu_status = "MOCK"
                
            if platform == "CUDA" and "REAL" in impl_type:
                cuda_status = "REAL"
            elif platform == "CUDA" and "MOCK" in impl_type:
                cuda_status = "MOCK"
                
            if platform == "OpenVINO" and "REAL" in impl_type:
                openvino_status = "REAL"
            elif platform == "OpenVINO" and "MOCK" in impl_type:
                openvino_status = "MOCK"
        
        # Print summary in a parser-friendly format
        print("\\nLLAMA TEST RESULTS SUMMARY")
        print(f"MODEL: {{metadata.get('model_name', 'Unknown')}}")
        print(f"CPU_STATUS: {{cpu_status}}")
        print(f"CUDA_STATUS: {{cuda_status}}")
        print(f"OPENVINO_STATUS: {{openvino_status}}")
        
        # Print performance information if available
        for example in examples:
            platform = example.get("platform", "")
            output = example.get("output", {{}})
            elapsed_time = example.get("elapsed_time", 0)
            
            print(f"\\n{{platform}} PERFORMANCE METRICS:")
            print(f"  Elapsed time: {{elapsed_time:.4f}}s")
            
            if "generated_text" in output:
                text = output["generated_text"]
                print(f"  Generated text sample: {{text[:50]}}...")
                
            # Check for detailed metrics
            if "performance_metrics" in output:
                metrics = output["performance_metrics"]
                for k, v in metrics.items():
                    print(f"  {{k}}: {{v}}")
        
        # Print a JSON representation to make it easier to parse
        print("\\nstructured_results")
        print(json.dumps({{
            "status": {{
                "cpu": cpu_status,
                "cuda": cuda_status,
                "openvino": openvino_status
            }},
            "model_name": metadata.get("model_name", "Unknown"),
            "examples": examples
        }}))
        
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during testing: {{str(e)}}")
        traceback.print_exc()
        sys.exit(1)'''
        
        # Replace the if __name__ == "__main__" block
        content = if_main_pattern.sub(new_if_main_block, content)
    
    # Fix the __test__ method to use real tests
    test_method_pattern = re.compile(r'def __test__\(self\):.*?# Create directories if they don\'t exist', re.DOTALL)
    test_method_match = test_method_pattern.search(content)
    
    if test_method_match:
        new_test_method = '''def __test__(self):
        """
        Run tests and compare/save results.
        Handles result collection, comparison with expected results, and storage.
        
        Returns:
            dict: Test results
        """
        # Run actual tests instead of using predefined results
        test_results = {}
        try:
            test_results = self.test()
        except Exception as e:
            test_results = {
                "status": {"test_error": str(e)},
                "examples": [],
                "metadata": {
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            }
        
        # Create directories if they don't exist'''
        
        # Replace the __test__ method up to the directory creation
        content = test_method_pattern.sub(new_test_method, content)
    
    # Write the updated file
    with open(test_file, 'w') as f:
        f.write(content)
    
    print(f"Updated {test_file} to use real tests")

if __name__ == "__main__":
    fix_llama_test()