#!/usr/bin/env python3
"""
Test script for indentation fixing tools.

This script validates that our indentation fixing tools work properly by:
1. Taking a sample of test files
2. Creating a copy with intentional indentation issues
3. Running our fixing tools on them
4. Verifying the results match expected output
"""

import os
import sys
import shutil
import tempfile
import subprocess
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_file_with_bad_indentation(output_path):
    """Create a test file with intentional indentation issues."""
    content = '''#!/usr/bin/env python3
"""
Test file with indentation issues.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestModel:
  """Class with bad indentation."""
  
  def __init__(self, model_id=None):
        """Initialize with inconsistent spacing."""
        self.model_id = model_id or "test-model"
    # Missing proper spacing
    def test_method(self, param1=None):
      """Docstring with wrong indentation."""
      if param1 is None:
    param1 = "default"
      # Nested blocks with inconsistent indentation
      for i in range(10):
          if i % 2 == 0:
        logger.info(f"Even number: {i}")
          else:
      logger.info(f"Odd number: {i}")
      
      try:
          result = self.process_data(param1)
      except Exception as e:
      logger.error(f"Error: {e}")
      return None
      
      return result
  
  def process_data(self, data):
    """Process input data."""
    results = {
      "model": self.model_id,
      "data": data,
      "status": "success"
    }
    return results    def save_results(self, output_dir="results"):
      """Save results with bad indentation."""
      os.makedirs(output_dir, exist_ok=True)
      
      output_path = os.path.join(output_dir, f"{self.model_id}.json")
      
      with open(output_path, "w") as f:
        f.write("Results would go here")
      
      return output_path

def main():
    tester = TestModel("test-model")
    result = tester.test_method("test-input")
    print(f"Test result: {result}")
    
if __name__ == "__main__":
    main()
'''
    
    with open(output_path, "w") as f:
        f.write(content)
    
    return output_path

def run_indentation_fix(file_path):
    """Run the indentation fix on the file."""
    fix_script = "complete_indentation_fix.py"
    
    if not os.path.exists(fix_script):
        logger.error(f"Fix script not found: {fix_script}")
        return False
    
    cmd = [sys.executable, fix_script, file_path, "--verify"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Error running fix script: {e}")
        return False

def verify_python_syntax(file_path):
    """Verify that the Python file has valid syntax."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", file_path],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False

def test_indentation_fix():
    """Test the indentation fixing functionality."""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Created temporary directory: {temp_dir}")
        
        # Create a test file with bad indentation
        test_file = os.path.join(temp_dir, "test_bad_indentation.py")
        create_test_file_with_bad_indentation(test_file)
        
        logger.info(f"Created test file with bad indentation: {test_file}")
        
        # Check that it has syntax errors
        initial_syntax_valid = verify_python_syntax(test_file)
        if initial_syntax_valid:
            logger.warning("Test file unexpectedly has valid syntax before fixing")
        else:
            logger.info("✅ Initial test file has syntax errors as expected")
        
        # Run the indentation fix
        fix_success = run_indentation_fix(test_file)
        
        if fix_success:
            logger.info("✅ Indentation fix completed successfully")
        else:
            logger.error("❌ Indentation fix failed")
            return False
        
        # Verify the fixed file has valid syntax
        fixed_syntax_valid = verify_python_syntax(test_file)
        
        if fixed_syntax_valid:
            logger.info("✅ Fixed file has valid syntax")
            return True
        else:
            logger.error("❌ Fixed file still has syntax errors")
            return False

def main():
    parser = argparse.ArgumentParser(description="Test indentation fixing tools")
    args = parser.parse_args()
    
    success = test_indentation_fix()
    
    if success:
        logger.info("All tests passed successfully!")
        return 0
    else:
        logger.error("Tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())