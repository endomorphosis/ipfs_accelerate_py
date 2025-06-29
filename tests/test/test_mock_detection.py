#!/usr/bin/env python3
"""
Script to test mock detection in our test files by running them with different
combinations of mocked dependencies and capturing output.
"""

import os
import sys
import subprocess
import tempfile
import re
from typing import List, Dict, Any, Tuple

def create_mock_script(dependencies_to_mock: List[str]) -> str:
    """
    Create a temporary Python script that runs a test with mocked dependencies.
    
    Args:
        dependencies_to_mock: List of dependency names to mock
        
    Returns:
        Path to the created script
    """
    script = """#!/usr/bin/env python3
# Auto-generated mock test script
import sys
from unittest.mock import MagicMock

# Mock specified dependencies
{mocks}

# Now import and run the test module
import importlib.util
import os
import sys

# Load the test file as a module
test_path = os.path.join("skills", "fixed_tests", "test_hf_bert.py")
spec = importlib.util.spec_from_file_location("test_hf_bert", test_path)
test_module = importlib.util.module_from_spec(spec)

# Run the test module's main function
sys.argv = [test_path, "--cpu-only"]
spec.loader.exec_module(test_module)
"""
    
    mock_lines = []
    for dep in dependencies_to_mock:
        mock_lines.append(f'sys.modules["{dep}"] = MagicMock()')
        if dep == 'torch':
            # Add specific mock for torch.cuda
            mock_lines.append('sys.modules["torch"].cuda = MagicMock()')
            mock_lines.append('sys.modules["torch"].cuda.is_available = lambda: False')
    
    script = script.format(mocks="\n".join(mock_lines))
    
    # Write to temporary file
    fd, script_path = tempfile.mkstemp(suffix='.py', prefix='mock_test_')
    with os.fdopen(fd, 'w') as f:
        f.write(script)
    
    return script_path

def run_test_with_mocks(dependencies_to_mock: List[str]) -> Tuple[int, str]:
    """
    Run a test with specific dependencies mocked.
    
    Args:
        dependencies_to_mock: List of dependency names to mock
        
    Returns:
        Tuple of (return_code, output)
    """
    script_path = create_mock_script(dependencies_to_mock)
    
    try:
        print(f"Running test with mocked dependencies: {', '.join(dependencies_to_mock)}")
        process = subprocess.Popen(
            [sys.executable, script_path], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        output, _ = process.communicate()
        return process.returncode, output
    finally:
        # Clean up temporary script
        os.unlink(script_path)

def check_indicators_in_output(output: str) -> Dict[str, Any]:
    """
    Check for mock indicators in the test output.
    
    Args:
        output: Test output text
        
    Returns:
        Dict with detection results
    """
    results = {
        "has_real_inference_indicator": False,
        "has_mock_indicator": False,
        "output_snippet": "",
        "error": None
    }
    
    # Extract the TEST RESULTS SUMMARY section
    summary_match = re.search(r'TEST RESULTS SUMMARY:(.*?)(?:\n\n|\Z)', output, re.DOTALL)
    if summary_match:
        summary = summary_match.group(1)
        results["output_snippet"] = summary.strip()
        
        # Check for indicators
        results["has_real_inference_indicator"] = "🚀 Using REAL INFERENCE" in summary
        results["has_mock_indicator"] = "🔷 Using MOCK OBJECTS" in summary
    else:
        results["error"] = "Could not find TEST RESULTS SUMMARY section"
    
    return results

def test_mock_combinations() -> None:
    """Test various combinations of mocked dependencies."""
    combinations = [
        [],  # No mocks (baseline)
        ["transformers"],  # Mock just transformers
        ["torch"],  # Mock just torch
        ["tokenizers"],  # Mock just tokenizers
        ["transformers", "torch"],  # Mock both core libraries
        ["transformers", "torch", "tokenizers"],  # Mock all three
    ]
    
    results = []
    
    for deps in combinations:
        return_code, output = run_test_with_mocks(deps)
        indicator_results = check_indicators_in_output(output)
        
        results.append({
            "mocked_dependencies": deps,
            "return_code": return_code,
            "indicators": indicator_results
        })
    
    # Print summary of results
    print("\n" + "="*70)
    print("MOCK DETECTION TEST RESULTS SUMMARY")
    print("="*70)
    
    for result in results:
        deps = result["mocked_dependencies"]
        deps_str = ", ".join(deps) if deps else "None (baseline)"
        
        if result["indicators"]["error"]:
            status = f"⚠️ ERROR: {result['indicators']['error']}"
        elif result["indicators"]["has_mock_indicator"]:
            status = "✅ CORRECT: Mock indicator shown"
        elif result["indicators"]["has_real_inference_indicator"] and not deps:
            status = "✅ CORRECT: Real inference indicator shown (no mocks)"
        elif result["indicators"]["has_real_inference_indicator"] and deps:
            status = "❌ INCORRECT: Real inference indicator shown with mocks"
        else:
            status = "❓ UNKNOWN: No indicators found"
        
        print(f"\nTest with mocked dependencies: {deps_str}")
        print(f"Status: {status}")
        if result["indicators"]["output_snippet"]:
            print("Output snippet:")
            for line in result["indicators"]["output_snippet"].split("\n"):
                print(f"  {line}")

if __name__ == "__main__":
    test_mock_combinations()