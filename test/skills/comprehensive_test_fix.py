#!/usr/bin/env python3
"""
Comprehensive HuggingFace Test Fix Script

This script addresses multiple issues in HuggingFace test files:
1. Adds proper mock detection for tokenizers and sentencepiece
2. Adds colorized output for better visual indication
3. Fixes environment variable controls for mocking
4. Ensures consistent mock detection across all test files
5. Fixes syntax errors in problematic files

Usage:
    python comprehensive_test_fix.py [--file FILE_PATH] [--dir DIRECTORY] [--check-only]
"""

import os
import sys
import re
import argparse
import logging
import subprocess
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"comprehensive_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# ANSI color codes for terminal output
GREEN = "\033[32m"
BLUE = "\033[34m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"

def check_mock_detection(content):
    """
    Check if mock detection system is properly implemented in content.
    
    Args:
        content: File content as string
        
    Returns:
        dict: Dictionary with check results
    """
    results = {
        "has_env_vars": {
            "torch": "MOCK_TORCH" in content,
            "transformers": "MOCK_TRANSFORMERS" in content,
            "tokenizers": "MOCK_TOKENIZERS" in content,
            "sentencepiece": "MOCK_SENTENCEPIECE" in content
        },
        "has_imports": {
            "torch": "import torch" in content,
            "transformers": "import transformers" in content,
            "tokenizers": "import tokenizers" in content,
            "sentencepiece": "import sentencepiece" in content
        },
        "has_detection": {
            "torch": "HAS_TORCH" in content,
            "transformers": "HAS_TRANSFORMERS" in content,
            "tokenizers": "HAS_TOKENIZERS" in content,
            "sentencepiece": "HAS_SENTENCEPIECE" in content
        },
        "has_mock_control": {
            "torch": "if MOCK_TORCH:" in content,
            "transformers": "if MOCK_TRANSFORMERS:" in content,
            "tokenizers": "if MOCK_TOKENIZERS:" in content,
            "sentencepiece": "if MOCK_SENTENCEPIECE:" in content
        },
        "has_inference_detection": "using_real_inference =" in content,
        "has_mock_detection": "using_mocks =" in content,
        "has_color_indicators": "GREEN =" in content and "BLUE =" in content,
        "has_visual_indicators": "🚀 Using REAL INFERENCE" in content and "🔷 Using MOCK OBJECTS" in content,
        "has_metadata": '"using_real_inference":' in content and '"using_mocks":' in content
    }
    
    # Derive overall results
    results["complete"] = (
        all(results["has_env_vars"].values()) and
        all(results["has_detection"].values()) and
        all(results["has_mock_control"].values()) and
        results["has_inference_detection"] and
        results["has_mock_detection"] and
        results["has_visual_indicators"] and
        results["has_metadata"]
    )
    
    return results

def fix_syntax_errors(content):
    """
    Fix common syntax errors in test files.
    
    Args:
        content: File content as string
        
    Returns:
        str: Fixed content
    """
    # Fix unterminated string literals
    pattern = r'print\("\s*$'
    if re.search(pattern, content, re.MULTILINE):
        content = re.sub(pattern, 'print("", ', content)
        logger.info("Fixed unterminated string literal")
    
    # Fix indentation errors in try-except blocks
    try_except_pattern = r'try:\s+.*?except\s+.*?:'
    for match in re.finditer(try_except_pattern, content, re.DOTALL):
        try_block = match.group(0)
        if re.search(r'\n\s+\n', try_block):
            fixed_block = re.sub(r'\n\s+\n', '\n', try_block)
            content = content.replace(try_block, fixed_block)
            logger.info("Fixed indentation in try-except block")
    
    return content

def add_color_definitions(content):
    """
    Add color definitions for terminal output if missing.
    
    Args:
        content: File content as string
        
    Returns:
        str: Updated content with color definitions
    """
    if "GREEN =" in content and "BLUE =" in content and "RESET =" in content:
        return content
    
    # Add after imports section
    import_section_end = 0
    for match in re.finditer(r'^import|^from\s+\w+\s+import', content, re.MULTILINE):
        import_line = match.group(0)
        line_end = content.find('\n', match.end())
        if line_end > import_section_end:
            import_section_end = line_end
    
    # If no imports found, add at beginning
    insert_position = import_section_end + 1 if import_section_end > 0 else 0
    
    color_definitions = """
# ANSI color codes for terminal output
GREEN = "\\033[32m"
BLUE = "\\033[34m"
YELLOW = "\\033[33m"
RED = "\\033[31m"
RESET = "\\033[0m"
"""
    
    content = content[:insert_position] + color_definitions + content[insert_position:]
    logger.info("Added color definitions")
    
    return content

def add_env_vars(content):
    """
    Add environment variable declarations for mock control if missing.
    
    Args:
        content: File content as string
        
    Returns:
        str: Updated content with environment variable declarations
    """
    env_vars_to_add = []
    
    # Check which env vars are missing
    if "MOCK_TORCH" not in content:
        env_vars_to_add.append("MOCK_TORCH = os.environ.get('MOCK_TORCH', 'False').lower() == 'true'")
    
    if "MOCK_TRANSFORMERS" not in content:
        env_vars_to_add.append("MOCK_TRANSFORMERS = os.environ.get('MOCK_TRANSFORMERS', 'False').lower() == 'true'")
    
    if "MOCK_TOKENIZERS" not in content:
        env_vars_to_add.append("MOCK_TOKENIZERS = os.environ.get('MOCK_TOKENIZERS', 'False').lower() == 'true'")
    
    if "MOCK_SENTENCEPIECE" not in content:
        env_vars_to_add.append("MOCK_SENTENCEPIECE = os.environ.get('MOCK_SENTENCEPIECE', 'False').lower() == 'true'")
    
    if not env_vars_to_add:
        return content
    
    # Find insertion point: after imports or after MagicMock import
    if "from unittest.mock import MagicMock" in content:
        insert_after = "from unittest.mock import MagicMock"
        insert_point = content.find(insert_after) + len(insert_after)
        insert_point = content.find('\n', insert_point) + 1
    else:
        # Add after imports
        import_section_end = 0
        for match in re.finditer(r'^import|^from\s+\w+\s+import', content, re.MULTILINE):
            import_line = match.group(0)
            line_end = content.find('\n', match.end())
            if line_end > import_section_end:
                import_section_end = line_end
        
        insert_point = import_section_end + 1 if import_section_end > 0 else 0
    
    # Insert env vars
    env_vars_block = "\n# Check if we should mock specific dependencies\n" + "\n".join(env_vars_to_add) + "\n"
    content = content[:insert_point] + env_vars_block + content[insert_point:]
    logger.info(f"Added environment variables: {', '.join([var.split(' =')[0] for var in env_vars_to_add])}")
    
    return content

def add_mock_imports(content):
    """
    Add missing mock imports for tokenizers and sentencepiece.
    
    Args:
        content: File content as string
        
    Returns:
        str: Updated content with mock imports added
    """
    # Check if necessary imports are missing
    needs_tokenizers = 'import tokenizers' not in content or 'HAS_TOKENIZERS' not in content
    needs_sentencepiece = 'import sentencepiece' not in content or 'HAS_SENTENCEPIECE' not in content
    
    if not needs_tokenizers and not needs_sentencepiece:
        return content
    
    # Find appropriate insertion point after transformers import
    transformers_import = re.search(r'# Try to import transformers.*?logger\.warning\("transformers not available, using mock"\)', content, re.DOTALL)
    
    if not transformers_import:
        logger.warning("Couldn't find transformers import section to add mock imports")
        return content
    
    insert_point = transformers_import.end()
    
    # Add tokenizers import if needed
    if needs_tokenizers:
        tokenizers_import = """

# Try to import tokenizers
try:
    if MOCK_TOKENIZERS:
        raise ImportError("Mocked tokenizers import failure")
    import tokenizers
    HAS_TOKENIZERS = True
except ImportError:
    tokenizers = MagicMock()
    HAS_TOKENIZERS = False
    logger.warning("tokenizers not available, using mock")"""
        
        content = content[:insert_point] + tokenizers_import + content[insert_point:]
        insert_point += len(tokenizers_import)
        logger.info("Added tokenizers mock import")
    
    # Add sentencepiece import if needed
    if needs_sentencepiece:
        sentencepiece_import = """

# Try to import sentencepiece
try:
    if MOCK_SENTENCEPIECE:
        raise ImportError("Mocked sentencepiece import failure")
    import sentencepiece
    HAS_SENTENCEPIECE = True
except ImportError:
    sentencepiece = MagicMock()
    HAS_SENTENCEPIECE = False
    logger.warning("sentencepiece not available, using mock")"""
        
        content = content[:insert_point] + sentencepiece_import + content[insert_point:]
        logger.info("Added sentencepiece mock import")
    
    return content

def add_mock_detection(content):
    """
    Add mock detection logic if missing.
    
    Args:
        content: File content as string
        
    Returns:
        str: Updated content with mock detection added
    """
    if "using_real_inference =" in content and "using_mocks =" in content:
        return content
    
    # Find run_tests method
    run_tests_match = re.search(r'def run_tests\([^)]*\):', content)
    if not run_tests_match:
        logger.warning("Could not find run_tests method to add mock detection")
        return content
    
    # Find return statement in run_tests method
    start_pos = run_tests_match.end()
    # Handle indentation by finding the first non-empty line after the function definition
    indentation = ""
    next_line_pos = content.find('\n', start_pos) + 1
    while next_line_pos < len(content):
        line_start = next_line_pos
        line_end = content.find('\n', line_start)
        if line_end == -1:
            line_end = len(content)
        line = content[line_start:line_end]
        if line.strip():
            indentation = re.match(r'(\s*)', line).group(1)
            break
        next_line_pos = line_end + 1
    
    # Find an appropriate insertion point before return statement
    return_match = re.search(r'\s+return\s+{', content[start_pos:])
    if not return_match:
        logger.warning("Could not find return statement in run_tests method")
        return content
    
    insert_point = start_pos + return_match.start()
    
    # Create mock detection code
    mock_detection_code = f"""
{indentation}# Determine if real inference or mock objects were used
{indentation}using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
{indentation}using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE

"""
    
    # Insert mock detection code
    content = content[:insert_point] + mock_detection_code + content[insert_point:]
    logger.info("Added mock detection logic")
    
    # Now add detection to the metadata in the return statement
    return_pos = content.find("return {", insert_point)
    metadata_match = re.search(r'"metadata"\s*:\s*{', content[return_pos:])
    
    if not metadata_match:
        logger.warning("Could not find metadata in return statement to add detection")
        return content
    
    metadata_pos = return_pos + metadata_match.end()
    closing_brace = content.find("}", metadata_pos)
    
    if closing_brace == -1:
        logger.warning("Could not find closing brace for metadata")
        return content
    
    # Create metadata addition
    mock_metadata = f"""
{indentation}        "has_transformers": HAS_TRANSFORMERS,
{indentation}        "has_torch": HAS_TORCH,
{indentation}        "has_tokenizers": HAS_TOKENIZERS,
{indentation}        "has_sentencepiece": HAS_SENTENCEPIECE,
{indentation}        "using_real_inference": using_real_inference,
{indentation}        "using_mocks": using_mocks,
{indentation}        "test_type": "REAL INFERENCE" if (using_real_inference and not using_mocks) else "MOCK OBJECTS (CI/CD)"
"""
    
    # Insert metadata
    content = content[:closing_brace] + mock_metadata + content[closing_brace:]
    logger.info("Added mock detection metadata")
    
    return content

def add_visual_indicators(content):
    """
    Add visual indicators for real vs. mock inference if missing.
    
    Args:
        content: File content as string
        
    Returns:
        str: Updated content with visual indicators added
    """
    if "🚀 Using REAL INFERENCE" in content and "🔷 Using MOCK OBJECTS" in content:
        # Check if they're colorized
        if not (f"{GREEN}🚀" in content and f"{BLUE}🔷" in content):
            # Add colorization
            content = content.replace(
                'print(f"🚀 Using REAL INFERENCE with actual models")',
                'print(f"{GREEN}🚀 Using REAL INFERENCE with actual models{RESET}")'
            )
            content = content.replace(
                'print(f"🔷 Using MOCK OBJECTS for CI/CD testing only")',
                'print(f"{BLUE}🔷 Using MOCK OBJECTS for CI/CD testing only{RESET}")'
            )
            logger.info("Colorized existing visual indicators")
        return content
    
    # Find main method
    main_match = re.search(r'def main\(\):', content)
    if not main_match:
        logger.warning("Could not find main method to add visual indicators")
        return content
    
    # Look for "Test Results Summary" section or similar
    start_pos = main_match.end()
    summary_patterns = [
        "TEST RESULTS SUMMARY",
        "print(f\"Successfully tested",
        "print(results)"
    ]
    
    insert_pos = None
    for pattern in summary_patterns:
        pattern_pos = content.find(pattern, start_pos)
        if pattern_pos != -1:
            # Find the beginning of the line
            line_start = content.rfind('\n', start_pos, pattern_pos) + 1
            # Get indentation
            indentation = re.match(r'(\s*)', content[line_start:pattern_pos]).group(1)
            insert_pos = line_start
            break
    
    if insert_pos is None:
        logger.warning("Could not find appropriate location for visual indicators")
        return content
    
    # Create visual indicators code
    visual_indicators = f"""
{indentation}# Indicate real vs mock inference clearly
{indentation}if using_real_inference and not using_mocks:
{indentation}    print(f"{{GREEN}}🚀 Using REAL INFERENCE with actual models{{RESET}}")
{indentation}else:
{indentation}    print(f"{{BLUE}}🔷 Using MOCK OBJECTS for CI/CD testing only{{RESET}}")
{indentation}    print(f"   Dependencies: transformers={{HAS_TRANSFORMERS}}, torch={{HAS_TORCH}}, tokenizers={{HAS_TOKENIZERS}}, sentencepiece={{HAS_SENTENCEPIECE}}")

"""
    
    # Insert visual indicators
    content = content[:insert_pos] + visual_indicators + content[insert_pos:]
    logger.info("Added visual indicators")
    
    return content

def fix_test_file(file_path, check_only=False):
    """
    Apply all fixes to a test file.
    
    Args:
        file_path: Path to the test file
        check_only: If True, only check without modifying
        
    Returns:
        tuple: (success, message)
    """
    try:
        # Read the file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check current state
        check_results = check_mock_detection(content)
        
        if check_results["complete"]:
            logger.info(f"✅ {file_path}: Mock detection is already complete")
            return True, "Mock detection already complete"
        
        if check_only:
            missing = []
            
            # Check environment variables
            for key, has_it in check_results["has_env_vars"].items():
                if not has_it:
                    missing.append(f"MOCK_{key.upper()}")
            
            # Check imports
            for key, has_it in check_results["has_detection"].items():
                if not has_it:
                    missing.append(f"HAS_{key.upper()}")
            
            # Check mock control
            for key, has_it in check_results["has_mock_control"].items():
                if not has_it:
                    missing.append(f"Mock control for {key}")
            
            # Check other elements
            if not check_results["has_inference_detection"]:
                missing.append("Real inference detection")
            
            if not check_results["has_mock_detection"]:
                missing.append("Mock detection")
            
            if not check_results["has_visual_indicators"]:
                missing.append("Visual indicators")
            
            if not check_results["has_metadata"]:
                missing.append("Metadata")
            
            return False, f"Missing: {', '.join(missing)}"
        
        # Create backup
        backup_path = f"{file_path}.fix.bak"
        with open(backup_path, 'w') as f:
            f.write(content)
        logger.info(f"Created backup at {backup_path}")
        
        # Apply fixes one by one
        # First, fix any syntax errors
        content = fix_syntax_errors(content)
        
        # Add color definitions
        content = add_color_definitions(content)
        
        # Add environment variables
        content = add_env_vars(content)
        
        # Add missing mock imports
        content = add_mock_imports(content)
        
        # Add mock detection
        content = add_mock_detection(content)
        
        # Add visual indicators
        content = add_visual_indicators(content)
        
        # Write changes
        with open(file_path, 'w') as f:
            f.write(content)
        
        # Verify syntax
        try:
            compile(content, file_path, 'exec')
            logger.info(f"✅ Syntax check passed for {file_path}")
        except SyntaxError as e:
            logger.error(f"❌ Syntax error in {file_path} after fixes: {e}")
            # Restore backup
            with open(backup_path, 'r') as f:
                original = f.read()
            with open(file_path, 'w') as f:
                f.write(original)
            logger.info(f"Restored backup for {file_path}")
            return False, f"Syntax error after fixes: {e}"
        
        # Verify mock detection is now complete
        with open(file_path, 'r') as f:
            updated_content = f.read()
        
        check_results = check_mock_detection(updated_content)
        
        if check_results["complete"]:
            logger.info(f"✅ {file_path}: Successfully fixed mock detection")
            return True, "Successfully fixed mock detection"
        else:
            logger.warning(f"⚠️ {file_path}: Fixes applied but mock detection still incomplete")
            return False, "Fixes applied but mock detection still incomplete"
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return False, f"Error: {e}"

def verify_test_file(file_path):
    """
    Verify that a test file works with different mock configurations.
    
    Args:
        file_path: Path to the test file
        
    Returns:
        tuple: (success, results)
    """
    logger.info(f"Verifying test file: {file_path}")
    
    # Run with different mock configurations
    configurations = [
        {"MOCK_TORCH": "False", "MOCK_TRANSFORMERS": "False", "MOCK_TOKENIZERS": "False", "MOCK_SENTENCEPIECE": "False"},
        {"MOCK_TORCH": "True", "MOCK_TRANSFORMERS": "False", "MOCK_TOKENIZERS": "False", "MOCK_SENTENCEPIECE": "False"},
        {"MOCK_TORCH": "False", "MOCK_TRANSFORMERS": "True", "MOCK_TOKENIZERS": "False", "MOCK_SENTENCEPIECE": "False"},
        {"MOCK_TORCH": "False", "MOCK_TRANSFORMERS": "False", "MOCK_TOKENIZERS": "True", "MOCK_SENTENCEPIECE": "False"},
        {"MOCK_TORCH": "False", "MOCK_TRANSFORMERS": "False", "MOCK_TOKENIZERS": "False", "MOCK_SENTENCEPIECE": "True"},
    ]
    
    results = {}
    all_passed = True
    
    for config in configurations:
        # Create configuration description
        config_desc = ", ".join([f"{k}={v}" for k, v in config.items()])
        logger.info(f"Testing configuration: {config_desc}")
        
        # Set environment variables
        env = os.environ.copy()
        env.update(config)
        
        try:
            # Run test file with this configuration
            command = [sys.executable, file_path]
            process = subprocess.run(
                command,
                env=env,
                timeout=60,
                capture_output=True,
                text=True
            )
            
            # Check if test passed
            success = process.returncode == 0
            output = process.stdout + process.stderr
            
            # Check if the mock detection is working
            mock_detected = "🔷 Using MOCK OBJECTS" in output
            real_detected = "🚀 Using REAL INFERENCE" in output
            
            # Determine expected detection based on configuration
            expected_mock = any(v == "True" for v in config.values())
            
            # Check if detection matches expectation
            detection_match = (expected_mock and mock_detected) or (not expected_mock and real_detected)
            
            # Save results
            results[config_desc] = {
                "success": success,
                "output": output,
                "mock_detected": mock_detected,
                "real_detected": real_detected,
                "detection_match": detection_match
            }
            
            if not success or not detection_match:
                all_passed = False
                
            logger.info(f"Configuration {config_desc}: {'✅ Passed' if success and detection_match else '❌ Failed'}")
            
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout while testing configuration: {config_desc}")
            results[config_desc] = {
                "success": False,
                "output": "Timeout expired",
                "detection_match": False
            }
            all_passed = False
            
        except Exception as e:
            logger.error(f"Error while testing configuration {config_desc}: {e}")
            results[config_desc] = {
                "success": False,
                "output": str(e),
                "detection_match": False
            }
            all_passed = False
    
    return all_passed, results

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Fix for HuggingFace Test Files")
    parser.add_argument("--file", type=str, help="Process a specific file")
    parser.add_argument("--dir", type=str, default="fixed_tests", help="Directory containing test files")
    parser.add_argument("--check-only", action="store_true", help="Only check for issues without fixing")
    parser.add_argument("--verify", action="store_true", help="Verify tests with different mock configurations after fixing")
    
    args = parser.parse_args()
    
    # Process a single file
    if args.file:
        file_path = args.file
        if not os.path.exists(file_path):
            print(f"{RED}File not found: {file_path}{RESET}")
            return 1
        
        print(f"Processing file: {file_path}")
        success, message = fix_test_file(file_path, args.check_only)
        
        if args.check_only:
            status = f"{GREEN}✅ Complete" if success else f"{RED}❌ Incomplete"
            print(f"Mock detection check: {status}{RESET}")
            print(f"Details: {message}")
            
        else:
            status = f"{GREEN}✅ Success" if success else f"{RED}❌ Failed"
            print(f"Fix result: {status}{RESET}")
            print(f"Details: {message}")
            
            if success and args.verify:
                print("\nVerifying test file with different mock configurations...")
                verify_success, verify_results = verify_test_file(file_path)
                
                status = f"{GREEN}✅ All passed" if verify_success else f"{RED}❌ Some failed"
                print(f"Verification result: {status}{RESET}")
                
                for config, result in verify_results.items():
                    config_status = f"{GREEN}✅ Passed" if result["success"] and result["detection_match"] else f"{RED}❌ Failed"
                    print(f"  Configuration {config}: {config_status}{RESET}")
                    
                    if not result["success"] or not result["detection_match"]:
                        print(f"    Output excerpt: {result['output'][:200]}...")
        
        return 0 if success else 1
    
    # Process all files in directory
    if not os.path.exists(args.dir):
        print(f"{RED}Directory not found: {args.dir}{RESET}")
        return 1
    
    test_files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) 
                 if f.startswith("test_hf_") and f.endswith(".py")]
    
    print(f"Found {len(test_files)} test files in {args.dir}")
    
    success_count = 0
    failure_count = 0
    results = {}
    
    for file_path in test_files:
        print(f"Processing {os.path.basename(file_path)}...")
        success, message = fix_test_file(file_path, args.check_only)
        
        if success:
            success_count += 1
        else:
            failure_count += 1
            
        results[file_path] = {
            "success": success,
            "message": message
        }
    
    # Generate summary
    if args.check_only:
        print(f"\n{GREEN}Check Summary:{RESET}")
        print(f"- Files with complete mock detection: {success_count}/{len(test_files)}")
        print(f"- Files with incomplete mock detection: {failure_count}/{len(test_files)}")
    else:
        print(f"\n{GREEN}Fix Summary:{RESET}")
        print(f"- Files successfully fixed: {success_count}/{len(test_files)}")
        print(f"- Files with fix failures: {failure_count}/{len(test_files)}")
    
    # Create detailed summary file
    summary_file = f"fix_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(summary_file, 'w') as f:
        if args.check_only:
            f.write("Mock Detection Check Summary\n")
            f.write("===========================\n\n")
        else:
            f.write("Mock Detection Fix Summary\n")
            f.write("=========================\n\n")
        
        f.write(f"Total files processed: {len(test_files)}\n")
        f.write(f"Success count: {success_count}\n")
        f.write(f"Failure count: {failure_count}\n\n")
        
        f.write("Details:\n")
        for file_path, result in results.items():
            status = "✅ " if result["success"] else "❌ "
            f.write(f"{status}{os.path.basename(file_path)}: {result['message']}\n")
    
    print(f"Detailed summary written to {summary_file}")
    
    return 0 if failure_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())