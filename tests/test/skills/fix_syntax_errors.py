#!/usr/bin/env python3
"""
Fix common syntax errors in test files.

This script identifies and fixes:
1. Unterminated string literals
2. Missing color code definitions
3. Missing mock checks for dependencies

Usage:
    python fix_syntax_errors.py --file FILE_PATH
"""

import os
import sys
import re
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"syntax_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def fix_unterminated_strings(content):
    """
    Fix unterminated string literals in the content.
    
    Args:
        content: File content as string
        
    Returns:
        str: Fixed content
    """
    # Find lines with unterminated string literals
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if line has an odd number of quote characters without escape
        single_quotes = line.count("'") - line.count("\\'")
        double_quotes = line.count('"') - line.count('\\"')
        
        # Check for f-strings with unterminated quotes
        if line.strip().startswith('print(f"') and not line.strip().endswith('")'):
            # This is likely an incomplete f-string
            if i + 1 < len(lines) and "):" in lines[i+1]:
                # Next line contains the closing parenthesis, probably meant to be a single line
                fixed_line = line + " " + lines[i+1].strip()
                fixed_lines.append(fixed_line)
                i += 2
                continue
            elif line.strip() == 'print(f"':
                # This is just the start of an f-string, likely missing the content and closing quote
                if "GREEN" in content and "BLUE" in content:
                    # Assume this is for mock detection
                    if "using_real_inference" in content and "using_mocks" in content:
                        # Add the typical mock detection visual indicators
                        fixed_lines.append('    # Indicate real vs mock inference clearly')
                        fixed_lines.append('    if using_real_inference and not using_mocks:')
                        fixed_lines.append('        print(f"{GREEN}🚀 Using REAL INFERENCE with actual models{RESET}")')
                        fixed_lines.append('    else:')
                        fixed_lines.append('        print(f"{BLUE}🔷 Using MOCK OBJECTS for CI/CD testing only{RESET}")')
                        fixed_lines.append('        print(f"   Dependencies: transformers={HAS_TRANSFORMERS}, torch={HAS_TORCH}, tokenizers={HAS_TOKENIZERS}, sentencepiece={HAS_SENTENCEPIECE}")')
                        i += 1
                        continue
                    else:
                        # Just complete the string with a general message
                        fixed_lines.append('print(f"Test completed successfully")')
                        i += 1
                        continue
                else:
                    # Just complete the string with a general message
                    fixed_lines.append('print(f"Test completed successfully")')
                    i += 1
                    continue
        
        # Check for regular unterminated strings
        if (single_quotes % 2 != 0) or (double_quotes % 2 != 0):
            if i + 1 < len(lines):
                # Try to combine with next line
                next_line = lines[i+1]
                combined = line + " " + next_line
                
                # Check if combining fixes the issue
                combined_single_quotes = combined.count("'") - combined.count("\\'")
                combined_double_quotes = combined.count('"') - combined.count('\\"')
                
                if (combined_single_quotes % 2 == 0) and (combined_double_quotes % 2 == 0):
                    # Combining fixed the issue
                    fixed_lines.append(combined)
                    i += 2
                    continue
                else:
                    # Just fix it by adding a closing quote
                    if single_quotes % 2 != 0:
                        fixed_lines.append(line + "'")
                    else:
                        fixed_lines.append(line + '"')
                    i += 1
                    continue
            else:
                # Last line with unterminated string
                if single_quotes % 2 != 0:
                    fixed_lines.append(line + "'")
                else:
                    fixed_lines.append(line + '"')
                i += 1
                continue
        
        # Regular line, keep as is
        fixed_lines.append(line)
        i += 1
    
    return '\n'.join(fixed_lines)

def add_color_codes(content):
    """
    Add color code definitions if missing.
    
    Args:
        content: File content as string
        
    Returns:
        str: Updated content with color codes
    """
    if "GREEN =" not in content and "BLUE =" not in content:
        # Add color code definitions after imports
        color_definitions = """
# ANSI color codes for terminal output
GREEN = "\\033[32m"
BLUE = "\\033[34m"
RESET = "\\033[0m"
"""
        # Find a good place to insert them (after imports)
        import_pattern = r"import\s+\w+|from\s+\w+\s+import"
        last_import = 0
        
        for match in re.finditer(import_pattern, content):
            last_import = max(last_import, match.end())
        
        if last_import > 0:
            line_end = content.find("\n", last_import)
            if line_end != -1:
                return content[:line_end+1] + color_definitions + content[line_end+1:]
        
        # Fallback - just add at the beginning
        return color_definitions + content
    
    return content

def add_missing_mock_checks(content):
    """
    Add missing mock checks for tokenizers and sentencepiece.
    
    Args:
        content: File content as string
        
    Returns:
        str: Updated content with mock checks
    """
    # Check if we need to add tokenizers check
    if "MOCK_TOKENIZERS" in content and "if MOCK_TOKENIZERS:" not in content:
        # Find the transformers mock check and add tokenizers after it
        transformers_check_pattern = r"try:\s+if MOCK_TRANSFORMERS:[^}]*?except ImportError:"
        match = re.search(transformers_check_pattern, content, re.DOTALL)
        
        if match:
            end_idx = match.end()
            tokenizers_check = """

# Try to import tokenizers
try:
    if MOCK_TOKENIZERS:
        raise ImportError("Mocked tokenizers import failure")
    import tokenizers
    HAS_TOKENIZERS = True
except ImportError:
    tokenizers = MagicMock()
    HAS_TOKENIZERS = False
    logger.warning("tokenizers not available, using mock")
"""
            content = content[:end_idx] + tokenizers_check + content[end_idx:]
    
    # Check if we need to add sentencepiece check
    if "MOCK_SENTENCEPIECE" in content and "if MOCK_SENTENCEPIECE:" not in content:
        # Find either the tokenizers mock check or transformers mock check and add after it
        if "MOCK_TOKENIZERS" in content:
            check_pattern = r"try:\s+if MOCK_TOKENIZERS:[^}]*?except ImportError:"
        else:
            check_pattern = r"try:\s+if MOCK_TRANSFORMERS:[^}]*?except ImportError:"
        
        match = re.search(check_pattern, content, re.DOTALL)
        
        if match:
            end_idx = match.end()
            sentencepiece_check = """

# Try to import sentencepiece
try:
    if MOCK_SENTENCEPIECE:
        raise ImportError("Mocked sentencepiece import failure")
    import sentencepiece
    HAS_SENTENCEPIECE = True
except ImportError:
    sentencepiece = MagicMock()
    HAS_SENTENCEPIECE = False
    logger.warning("sentencepiece not available, using mock")
"""
            content = content[:end_idx] + sentencepiece_check + content[end_idx:]
    
    return content

def fix_missing_mock_vars(content):
    """
    Add missing mock environment variables.
    
    Args:
        content: File content as string
        
    Returns:
        str: Updated content with mock environment variables
    """
    env_vars_needed = []
    
    if "MOCK_TORCH" not in content:
        env_vars_needed.append("MOCK_TORCH = os.environ.get('MOCK_TORCH', 'False').lower() == 'true'")
    
    if "MOCK_TRANSFORMERS" not in content:
        env_vars_needed.append("MOCK_TRANSFORMERS = os.environ.get('MOCK_TRANSFORMERS', 'False').lower() == 'true'")
    
    if "MOCK_TOKENIZERS" not in content:
        env_vars_needed.append("MOCK_TOKENIZERS = os.environ.get('MOCK_TOKENIZERS', 'False').lower() == 'true'")
    
    if "MOCK_SENTENCEPIECE" not in content:
        env_vars_needed.append("MOCK_SENTENCEPIECE = os.environ.get('MOCK_SENTENCEPIECE', 'False').lower() == 'true'")
    
    if env_vars_needed:
        env_vars_block = "# Check if we should mock specific dependencies\n" + "\n".join(env_vars_needed)
        
        # Find a good place to insert them (before torch import)
        torch_import_idx = content.find("# Try to import torch")
        if torch_import_idx != -1:
            return content[:torch_import_idx] + env_vars_block + "\n" + content[torch_import_idx:]
        
        # Alternative: find after imports
        import_pattern = r"import\s+\w+|from\s+\w+\s+import"
        last_import = 0
        
        for match in re.finditer(import_pattern, content):
            last_import = max(last_import, match.end())
        
        if last_import > 0:
            line_end = content.find("\n", last_import)
            if line_end != -1:
                return content[:line_end+1] + "\n" + env_vars_block + "\n" + content[line_end+1:]
        
        # Fallback - just add at the beginning
        return env_vars_block + "\n" + content
    
    return content

def add_mock_detection_logic(content):
    """
    Add mock detection logic if missing.
    
    Args:
        content: File content as string
        
    Returns:
        str: Updated content with mock detection logic
    """
    if "using_real_inference =" not in content or "using_mocks =" not in content:
        # Find the run_tests function
        run_tests_pattern = r"def\s+run_tests\s*\([^)]*\):"
        match = re.search(run_tests_pattern, content)
        
        if match:
            # Find the return statement within run_tests
            func_start = match.end()
            return_pattern = r"(\s+)return\s+\{"
            return_match = re.search(return_pattern, content[func_start:])
            
            if return_match:
                indent = return_match.group(1)
                return_pos = func_start + return_match.start()
                
                # Add the detection logic before return
                detection_logic = f"""
{indent}# Determine if real inference or mock objects were used
{indent}using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
{indent}using_mocks = not using_real_inference"""
                
                if "HAS_TOKENIZERS" in content:
                    detection_logic += " or not HAS_TOKENIZERS"
                
                if "HAS_SENTENCEPIECE" in content:
                    detection_logic += " or not HAS_SENTENCEPIECE"
                
                detection_logic += "\n"
                
                content = content[:return_pos] + detection_logic + content[return_pos:]
        
        # Also add the metadata in the return statement if needed
        metadata_pattern = r'"metadata":\s*\{'
        match = re.search(metadata_pattern, content)
        
        if match:
            metadata_start = match.end()
            metadata_end = content.find("}", metadata_start)
            
            if metadata_end != -1:
                # Check if the metadata already includes the mock detection info
                metadata_section = content[metadata_start:metadata_end]
                
                if '"has_transformers":' not in metadata_section:
                    mock_metadata = """
                "has_transformers": HAS_TRANSFORMERS,
                "has_torch": HAS_TORCH,"""
                    
                    if "HAS_TOKENIZERS" in content:
                        mock_metadata += """
                "has_tokenizers": HAS_TOKENIZERS,"""
                    
                    if "HAS_SENTENCEPIECE" in content:
                        mock_metadata += """
                "has_sentencepiece": HAS_SENTENCEPIECE,"""
                    
                    mock_metadata += """
                "using_real_inference": using_real_inference,
                "using_mocks": using_mocks,
                "test_type": "REAL INFERENCE" if (using_real_inference and not using_mocks) else "MOCK OBJECTS (CI/CD)"
"""
                    content = content[:metadata_start] + mock_metadata + content[metadata_start:]
    
    return content

def add_visual_indicators(content):
    """
    Add visual indicators if missing.
    
    Args:
        content: File content as string
        
    Returns:
        str: Updated content with visual indicators
    """
    if "print(f\"{GREEN}🚀" not in content and "print(f\"{BLUE}🔷" not in content:
        # Try to find a good location - main function before TEST RESULTS SUMMARY
        main_pattern = r"def\s+main\s*\(\s*\):"
        match = re.search(main_pattern, content)
        
        if match:
            main_start = match.end()
            summary_pattern = r"(\s+)print\s*\(\s*[\"'].*TEST RESULTS SUMMARY.*[\"']\s*\)"
            summary_match = re.search(summary_pattern, content[main_start:])
            
            if summary_match:
                indent = summary_match.group(1)
                indicators_pos = main_start + summary_match.start()
                
                # Add visual indicators before summary
                indicators_code = f"""
{indent}# Indicate real vs mock inference clearly
{indent}using_real_inference = results["metadata"]["using_real_inference"]
{indent}using_mocks = results["metadata"]["using_mocks"]
{indent}
{indent}if using_real_inference and not using_mocks:
{indent}    print(f"{{GREEN}}🚀 Using REAL INFERENCE with actual models{{RESET}}")
{indent}else:
{indent}    print(f"{{BLUE}}🔷 Using MOCK OBJECTS for CI/CD testing only{{RESET}}")
{indent}    print(f"   Dependencies: transformers={{HAS_TRANSFORMERS}}, torch={{HAS_TORCH}}, tokenizers={{HAS_TOKENIZERS}}, sentencepiece={{HAS_SENTENCEPIECE}}")

"""
                content = content[:indicators_pos] + indicators_code + content[indicators_pos:]
    
    return content

def fix_file(file_path):
    """
    Fix common syntax errors in a file.
    
    Args:
        file_path: Path to the file to fix
        
    Returns:
        bool: True if file was fixed successfully, False otherwise
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Create a backup
        backup_path = f"{file_path}.syntax.bak"
        with open(backup_path, 'w') as f:
            f.write(content)
        logger.info(f"Created backup at {backup_path}")
        
        # Apply fixes
        fixed_content = content
        fixed_content = fix_unterminated_strings(fixed_content)
        fixed_content = add_color_codes(fixed_content)
        fixed_content = fix_missing_mock_vars(fixed_content)
        fixed_content = add_missing_mock_checks(fixed_content)
        fixed_content = add_mock_detection_logic(fixed_content)
        fixed_content = add_visual_indicators(fixed_content)
        
        # Write the fixed content
        with open(file_path, 'w') as f:
            f.write(fixed_content)
        
        # Verify the syntax
        try:
            compile(fixed_content, file_path, 'exec')
            logger.info(f"✅ Successfully fixed {file_path}")
            return True
        except SyntaxError as e:
            logger.error(f"❌ Syntax error after fixes in {file_path}: {e}")
            
            # Restore from backup
            with open(backup_path, 'r') as src, open(file_path, 'w') as dst:
                dst.write(src.read())
            logger.info(f"Restored backup for {file_path}")
            return False
        
    except Exception as e:
        logger.error(f"Error fixing {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fix common syntax errors in test files")
    parser.add_argument("--file", type=str, required=True, help="Path to the file to fix")
    args = parser.parse_args()
    
    file_path = args.file
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return 1
    
    success = fix_file(file_path)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())