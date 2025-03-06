#!/usr/bin/env python
"""
Script to add Qualcomm support to all template files.

This script updates all template files in the hardware_test_templates directory
to add Qualcomm platform support consistently.
"""

import os
import sys
import re
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("qualcomm_update")

# Path to hardware test templates
TEMPLATES_DIR = Path(__file__).parent / "hardware_test_templates"

def add_qualcomm_to_docstring(file_path: str) -> bool:
    """
    Add Qualcomm to the template docstring.
    
    Args:
        file_path: Path to the template file
        
    Returns:
        True if changes were made, False otherwise
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except (IOError, UnicodeDecodeError) as e:
        logger.error(f"Failed to read file: {str(e)}")
        return False
    
    # Look for docstring
    docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
    if not docstring_match:
        logger.warning(f"No docstring found in {file_path}")
        return False
    
    docstring = docstring_match.group(1)
    
    # Check if Qualcomm is already mentioned
    if "Qualcomm" in docstring:
        logger.info(f"Qualcomm already in docstring for {file_path}")
        return False
    
    # Look for the platforms list
    platforms_pattern = r'(- (CPU|CUDA|OpenVINO|MPS|ROCm|WebNN|WebGPU)[^\n]*\n)+'
    platforms_match = re.search(platforms_pattern, docstring)
    
    if platforms_match:
        # Find the last platform in the list
        platforms_text = platforms_match.group(0)
        # Insert Qualcomm before WebNN (which is usually near the end)
        if "- WebNN" in platforms_text:
            new_platforms_text = platforms_text.replace(
                "- WebNN", 
                "- Qualcomm: Qualcomm AI Engine/Hexagon DSP implementation\n- WebNN"
            )
        # Or insert at the end if WebNN isn't found
        else:
            # Find the last platform line
            lines = platforms_text.strip().split('\n')
            last_line = lines[-1]
            # Insert Qualcomm after the last platform
            new_platforms_text = platforms_text.replace(
                last_line,
                f"{last_line}\n- Qualcomm: Qualcomm AI Engine/Hexagon DSP implementation"
            )
        
        # Replace in the full docstring
        new_docstring = docstring.replace(platforms_text, new_platforms_text)
        
        # Replace in the full content
        new_content = content.replace(docstring_match.group(1), new_docstring)
        
        # Write back to file
        with open(file_path, 'w') as f:
            f.write(new_content)
        
        logger.info(f"Added Qualcomm to docstring in {file_path}")
        return True
    else:
        logger.warning(f"Could not find platforms list in docstring for {file_path}")
        return False

def add_qualcomm_to_test_cases(file_path: str) -> bool:
    """
    Add Qualcomm to the test_cases list.
    
    Args:
        file_path: Path to the template file
        
    Returns:
        True if changes were made, False otherwise
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except (IOError, UnicodeDecodeError) as e:
        logger.error(f"Failed to read file: {str(e)}")
        return False
    
    # Check if Qualcomm is already in test cases
    if '"QUALCOMM"' in content or "'QUALCOMM'" in content:
        logger.info(f"Qualcomm already in test cases for {file_path}")
        return False
    
    # Look for test cases section
    test_cases_pattern = r'(self\.test_cases\s*=\s*\[\s*\{[^}]*"platform":\s*"[^"]+")[^}]*\}.*?(\s*\]\s*)'
    test_cases_match = re.search(test_cases_pattern, content, re.DOTALL)
    
    if test_cases_match:
        # Find the last test case
        test_cases_text = test_cases_match.group(0)
        
        # Extract WebNN test case to use as template
        webnn_pattern = r'(\{\s*"description":\s*"Test on WEBNN platform"[^}]*\})'
        webnn_match = re.search(webnn_pattern, test_cases_text)
        
        if webnn_match:
            webnn_case = webnn_match.group(1)
            # Create Qualcomm case by replacing WEBNN with QUALCOMM
            qualcomm_case = webnn_case.replace("WEBNN", "QUALCOMM")
            
            # Insert Qualcomm case before WebNN
            new_test_cases = test_cases_text.replace(webnn_case, qualcomm_case + ",\n            " + webnn_case)
            
            # Replace in the full content
            new_content = content.replace(test_cases_text, new_test_cases)
            
            # Write back to file
            with open(file_path, 'w') as f:
                f.write(new_content)
            
            logger.info(f"Added Qualcomm to test cases in {file_path}")
            return True
        else:
            logger.warning(f"Could not find WebNN test case to use as template in {file_path}")
            return False
    else:
        logger.warning(f"Could not find test_cases list in {file_path}")
        return False

def add_qualcomm_init_method(file_path: str) -> bool:
    """
    Add init_qualcomm method to the template.
    
    Args:
        file_path: Path to the template file
        
    Returns:
        True if changes were made, False otherwise
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except (IOError, UnicodeDecodeError) as e:
        logger.error(f"Failed to read file: {str(e)}")
        return False
    
    # Check if init_qualcomm already exists
    if "def init_qualcomm" in content:
        logger.info(f"init_qualcomm already exists in {file_path}")
        return False
    
    # Look for init_rocm method to use as reference
    rocm_pattern = r'def init_rocm\(self\):(.*?)def init_webnn\(self\):'
    rocm_match = re.search(rocm_pattern, content, re.DOTALL)
    
    if rocm_match:
        rocm_method = rocm_match.group(1)
        
        # Create Qualcomm init method
        qualcomm_method = """
    def init_qualcomm(self):
        # Initialize for Qualcomm platform
        try:
            # Try to import Qualcomm-specific libraries
            import importlib.util
            has_qnn = importlib.util.find_spec("qnn_wrapper") is not None
            has_qti = importlib.util.find_spec("qti") is not None
            has_qualcomm_env = "QUALCOMM_SDK" in os.environ
            
            if has_qnn or has_qti or has_qualcomm_env:
                self.platform = "QUALCOMM"
                self.device = "qualcomm"
            else:
                print("Qualcomm SDK not available, falling back to CPU")
                self.platform = "CPU"
                self.device = "cpu"
        except Exception as e:
            print(f"Error initializing Qualcomm platform: {e}")
            self.platform = "CPU"
            self.device = "cpu"
            
        return self.load_tokenizer()
        """
        
        # Insert Qualcomm init method before WebNN init method
        new_content = content.replace("def init_webnn(self):", qualcomm_method + "\n    def init_webnn(self):")
        
        # Write back to file
        with open(file_path, 'w') as f:
            f.write(new_content)
        
        logger.info(f"Added init_qualcomm method to {file_path}")
        return True
    else:
        logger.warning(f"Could not find init_rocm method in {file_path}")
        return False

def add_qualcomm_handler_method(file_path: str) -> bool:
    """
    Add create_qualcomm_handler method to the template.
    
    Args:
        file_path: Path to the template file
        
    Returns:
        True if changes were made, False otherwise
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except (IOError, UnicodeDecodeError) as e:
        logger.error(f"Failed to read file: {str(e)}")
        return False
    
    # Check if create_qualcomm_handler already exists
    if "def create_qualcomm_handler" in content:
        logger.info(f"create_qualcomm_handler already exists in {file_path}")
        return False
    
    # Look for create_rocm_handler method
    rocm_pattern = r'def create_rocm_handler\(self\):(.*?)def create_webnn_handler\(self\):'
    rocm_match = re.search(rocm_pattern, content, re.DOTALL)
    
    if rocm_match:
        # Create Qualcomm handler method
        qualcomm_handler = """
    def create_qualcomm_handler(self):
        # Create handler for Qualcomm platform
        try:
            model_path = self.get_model_path_or_name()
            if self.tokenizer is None:
                self.load_tokenizer()
                
            # Check if Qualcomm QNN SDK is available
            import importlib.util
            has_qnn = importlib.util.find_spec("qnn_wrapper") is not None
            
            if has_qnn:
                try:
                    # Import QNN wrapper (in a real implementation)
                    import qnn_wrapper as qnn
                    
                    # QNN implementation would look something like this:
                    # 1. Convert model to QNN format
                    # 2. Load the model on the Hexagon DSP
                    # 3. Set up the inference handler
                    
                    def handler(input_text):
                        # Tokenize input
                        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
                        
                        # Convert to numpy for QNN input
                        input_ids_np = inputs["input_ids"].numpy()
                        attention_mask_np = inputs["attention_mask"].numpy()
                        
                        # This would call the QNN model in a real implementation
                        # result = qnn_model.execute([input_ids_np, attention_mask_np])
                        # embedding = result[0]
                        
                        # Using mock embedding for demonstration
                        embedding = np.random.rand(1, 768)
                        
                        return {
                            "embedding": embedding,
                            "success": True,
                            "platform": "qualcomm"
                        }
                    
                    return handler
                except ImportError:
                    print("QNN wrapper available but failed to import, using mock implementation")
                    return MockHandler(self.model_path, "qualcomm")
            else:
                # Check for QTI AI Engine
                has_qti = importlib.util.find_spec("qti") is not None
                
                if has_qti:
                    try:
                        # Import QTI AI Engine
                        import qti.aisw.dlc_utils as qti_utils
                        
                        # Mock implementation
                        def handler(input_text):
                            # Tokenize input
                            inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
                            
                            # Mock QTI execution
                            embedding = np.random.rand(1, 768)
                            
                            return {
                                "embedding": embedding,
                                "success": True,
                                "platform": "qualcomm-qti"
                            }
                        
                        return handler
                    except ImportError:
                        print("QTI available but failed to import, using mock implementation")
                        return MockHandler(self.model_path, "qualcomm")
                else:
                    # Fall back to mock implementation
                    return MockHandler(self.model_path, "qualcomm")
        except Exception as e:
            print(f"Error creating Qualcomm handler: {e}")
            return MockHandler(self.model_path, "qualcomm")
            """
        
        # Insert Qualcomm handler method before WebNN handler method
        new_content = content.replace("def create_webnn_handler(self):", qualcomm_handler + "\n    def create_webnn_handler(self):")
        
        # Write back to file
        with open(file_path, 'w') as f:
            f.write(new_content)
        
        logger.info(f"Added create_qualcomm_handler method to {file_path}")
        return True
    else:
        logger.warning(f"Could not find create_rocm_handler method in {file_path}")
        return False

def add_qualcomm_to_template(file_path: str) -> bool:
    """
    Add Qualcomm support to a template file.
    
    Args:
        file_path: Path to the template file
        
    Returns:
        True if changes were made, False otherwise
    """
    logger.info(f"Processing {file_path}")
    
    changes_made = False
    
    # Step 1: Add Qualcomm to docstring
    if add_qualcomm_to_docstring(file_path):
        changes_made = True
    
    # Step 2: Add Qualcomm to test_cases
    if add_qualcomm_to_test_cases(file_path):
        changes_made = True
    
    # Step 3: Add init_qualcomm method
    if add_qualcomm_init_method(file_path):
        changes_made = True
    
    # Step 4: Add create_qualcomm_handler method
    if add_qualcomm_handler_method(file_path):
        changes_made = True
    
    if changes_made:
        logger.info(f"✅ Added Qualcomm support to {file_path}")
    else:
        logger.info(f"No changes needed for {file_path}")
    
    return changes_made

def process_all_templates(backup: bool = False) -> Dict[str, bool]:
    """
    Process all templates in the hardware_test_templates directory.
    
    Args:
        backup: Whether to create backups before modifying files
        
    Returns:
        Dictionary mapping file paths to whether changes were made
    """
    results = {}
    
    # Find all template files
    template_files = list(TEMPLATES_DIR.glob("template_*.py"))
    logger.info(f"Found {len(template_files)} template files")
    
    # Skip template_database.py
    template_files = [f for f in template_files if f.name != "template_database.py"]
    logger.info(f"Processing {len(template_files)} template files (excluding template_database.py)")
    
    # Create backups if requested
    if backup:
        for file_path in template_files:
            backup_path = str(file_path) + '.bak'
            try:
                with open(file_path, 'r') as src, open(backup_path, 'w') as dst:
                    dst.write(src.read())
                logger.info(f"Created backup: {backup_path}")
            except Exception as e:
                logger.error(f"Failed to create backup for {file_path}: {e}")
    
    # Process each template
    for file_path in template_files:
        results[str(file_path)] = add_qualcomm_to_template(str(file_path))
    
    # Summarize results
    changed_count = sum(1 for changed in results.values() if changed)
    logger.info(f"Added Qualcomm support to {changed_count} out of {len(results)} template files")
    
    return results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Add Qualcomm support to template files")
    parser.add_argument("--file", type=str, help="Process a specific template file")
    parser.add_argument("--backup", action="store_true", help="Create backups before modifying files")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    if args.file:
        # Process a specific file
        file_path = args.file
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return 1
        
        if args.backup:
            backup_path = file_path + '.bak'
            try:
                with open(file_path, 'r') as src, open(backup_path, 'w') as dst:
                    dst.write(src.read())
                logger.info(f"Created backup: {backup_path}")
            except Exception as e:
                logger.error(f"Failed to create backup: {e}")
        
        result = add_qualcomm_to_template(file_path)
        
        if result:
            logger.info(f"✅ Successfully added Qualcomm support to {file_path}")
            return 0
        else:
            logger.warning(f"No changes made to {file_path}")
            return 1
    else:
        # Process all templates
        results = process_all_templates(args.backup)
        
        changed_count = sum(1 for changed in results.values() if changed)
        if changed_count > 0:
            logger.info(f"✅ Successfully added Qualcomm support to {changed_count} template files")
            return 0
        else:
            logger.warning("No changes made to any template files")
            return 1

if __name__ == "__main__":
    sys.exit(main())