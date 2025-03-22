#!/usr/bin/env python3

import os
import sys
import re
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_dependency_blocks(file_path):
    """Fix torch and transformers dependency blocks completely."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Define correct torch import block
        torch_import_block = """
# Check if dependencies are available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = MagicMock()
    HAS_TORCH = False
    logger.warning("torch not available, using mock")
"""

        # Define correct transformers import block
        transformers_import_block = """
try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = MagicMock()
    HAS_TRANSFORMERS = False
    logger.warning("transformers not available, using mock")
"""

        # First, fix any remaining indentation issues
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            # Skip empty lines
            if not line.strip():
                fixed_lines.append(line)
                continue
                
            # Check for common indentation issues
            stripped = line.strip()
            
            # Fix model registry indentation
            if stripped.startswith('XLM_') and 'MODELS_REGISTRY' in stripped:
                fixed_lines.append(stripped)
            # Fix main function indentation
            elif stripped.startswith('def main'):
                fixed_lines.append(stripped)
            # Fix indented import statements that should be at the top level
            elif stripped.startswith('import ') and line.startswith(' '):
                fixed_lines.append(stripped)
            # Fix indented registry entries
            elif stripped.startswith('"') and ':' in stripped and '{' in line:
                # This is likely a registry entry - preserve its indentation level relative to parent
                if len(fixed_lines) > 0:
                    prev_line = fixed_lines[-1]
                    if '{' in prev_line:
                        # This is the first entry after the opening brace
                        prev_indent = len(prev_line) - len(prev_line.lstrip())
                        fixed_lines.append(' ' * (prev_indent + 4) + stripped)
                    else:
                        # Match the indentation of the previous line
                        fixed_lines.append(line)
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        # Rebuild content
        content = '\n'.join(fixed_lines)

        # Look for problematic import sections and replace them completely
        import_section_start = content.find("# Check if dependencies are available")
        if import_section_start >= 0:
            # Find the end of the mocked imports section
            mocked_section_end = content.find("# Simple registry", import_section_start)
            if mocked_section_end < 0:
                mocked_section_end = content.find("def ", import_section_start)
            
            if mocked_section_end > 0:
                # Extract everything before and after the dependency blocks
                content_before = content[:import_section_start]
                content_after = content[mocked_section_end:]
                
                # Replace the entire dependencies section with correct blocks
                content = content_before + torch_import_block + transformers_import_block + content_after
                logger.info(f"Replaced dependency import blocks in {file_path}")
        
        # Check if the content has changed
        if content != original_content:
            # Write the fixed content
            with open(file_path, 'w') as f:
                f.write(content)
            
            # Verify the syntax
            try:
                compile(content, file_path, 'exec')
                logger.info(f"✅ {file_path}: Syntax is valid after fixes")
                return True
            except SyntaxError as e:
                logger.error(f"❌ {file_path}: Syntax errors remain: {e}")
                return False
        else:
            logger.info(f"No changes needed in {file_path}")
            return True
        
    except Exception as e:
        logger.error(f"Error fixing file {file_path}: {e}")
        return False

def fix_model_mappings(file_path):
    """Fix common template mapping issues."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Fix model tester reference issues
        model_tester_pattern = re.compile(r'(\w+)_tester = Test(\w+)Models')
        for match in model_tester_pattern.finditer(content):
            model_type = match.group(1)
            referenced_type = match.group(2)
            
            # Try to determine the correct class name
            if '-' in model_type:
                # Convert hyphenated names to PascalCase
                parts = model_type.split('_')
                pascal_case = ''.join(part.capitalize() for part in parts)
            else:
                pascal_case = model_type.capitalize()
            
            if model_type.lower() != referenced_type.lower():
                logger.info(f"Fixing model tester reference: {model_type}_tester = Test{referenced_type}Models -> Test{pascal_case}Models")
                content = content.replace(
                    f"{model_type}_tester = Test{referenced_type}Models",
                    f"{model_type}_tester = Test{pascal_case}Models"
                )
        
        # Fix tester references in print statements
        tester_ref_pattern = re.compile(r'Device: {(\w+)_tester\.device}')
        for match in tester_ref_pattern.finditer(content):
            referenced_tester = match.group(1)
            model_type_ref = re.search(r'(\w+)_tester = Test', content)
            if model_type_ref and referenced_tester != model_type_ref.group(1):
                model_type = model_type_ref.group(1)
                logger.info(f"Fixing tester device reference: {referenced_tester}_tester -> {model_type}_tester")
                content = content.replace(
                    f"Device: {{{referenced_tester}_tester.device}}",
                    f"Device: {{{model_type}_tester.device}}"
                )
        
        # Check if the content has changed
        if content != original_content:
            # Write the fixed content
            with open(file_path, 'w') as f:
                f.write(content)
            logger.info(f"Fixed model mappings in {file_path}")
        else:
            logger.info(f"No model mapping changes needed in {file_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error fixing model mappings in {file_path}: {e}")
        return False

def fix_method_definitions(file_path):
    """Fix method definition issues."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Fix method definition attached to the end of another method's return statement
        content = re.sub(
            r'return {"success": False, "error": str\(e\)}\s*def run_tests\(',
            r'return {"success": False, "error": str(e)}\n\n    def run_tests(',
            content
        )
        
        # Fix malformed return statements
        content = re.sub(
            r'}\s*return results',
            r'}\n        return results',
            content
        )
        
        # Fix missing blank line before if __name__ block
        content = re.sub(
            r'return 0 if success else 1\s*\n\s*if __name__ ==',
            r'return 0 if success else 1\n\nif __name__ ==',
            content
        )
        
        # Check if the content has changed
        if content != original_content:
            # Write the fixed content
            with open(file_path, 'w') as f:
                f.write(content)
            logger.info(f"Fixed method definitions in {file_path}")
        else:
            logger.info(f"No method definition changes needed in {file_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error fixing method definitions in {file_path}: {e}")
        return False

def fix_test_file(file_path):
    """Apply all fixes to a test file."""
    # Check if the file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False
    
    # Apply dependency block fixes
    if not fix_dependency_blocks(file_path):
        logger.error(f"Failed to fix dependency blocks in {file_path}")
        return False
    
    # Apply model mapping fixes
    if not fix_model_mappings(file_path):
        logger.error(f"Failed to fix model mappings in {file_path}")
        return False
    
    # Apply method definition fixes
    if not fix_method_definitions(file_path):
        logger.error(f"Failed to fix method definitions in {file_path}")
        return False
    
    logger.info(f"Successfully fixed {file_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Fix template import issues in test files")
    parser.add_argument("--file", type=str, help="Fix a single file")
    parser.add_argument("--directory", type=str, help="Fix all Python files in a directory")
    parser.add_argument("--pattern", type=str, default="test_hf_*.py", help="File pattern to match (default: test_hf_*.py)")
    
    args = parser.parse_args()
    
    if not (args.file or args.directory):
        parser.error("Either --file or --directory must be specified")
    
    success_count = 0
    failed_count = 0
    
    if args.file:
        if fix_test_file(args.file):
            success_count += 1
        else:
            failed_count += 1
    
    if args.directory:
        directory = Path(args.directory)
        files = list(directory.glob(args.pattern))
        logger.info(f"Found {len(files)} files matching pattern '{args.pattern}' in {directory}")
        
        for file_path in files:
            if fix_test_file(str(file_path)):
                success_count += 1
            else:
                failed_count += 1
    
    logger.info(f"Fixed {success_count} files successfully, {failed_count} files failed")
    
    return 0 if failed_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())