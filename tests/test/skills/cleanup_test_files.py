#!/usr/bin/env python3
"""
Enhanced script to clean up the generated test files.
This script performs targeted string replacements and regex-based fixes for common indentation issues.
"""

import os
import sys
import re
from pathlib import Path

def cleanup_test_file(file_path):
    """
    Apply targeted fixes to clean up the most critical indentation issues.
    
    Args:
        file_path: Path to the file to clean up
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 1. Fix method boundaries with proper spacing
    content = re.sub(r'return results\s+def', 'return results\n\n    def', content)
    content = re.sub(r'self\.performance_stats = \{\}\s+def', 
                    'self.performance_stats = {}\n\n    def', content)
    
    # 2. Fix dependency check indentation - normalize to 8 spaces
    content = re.sub(r'(\s+)if not HAS_(\w+):', r'        if not HAS_\2:', content)
    
    # 3. Fix nested indentation in control structures (if/else/elif)
    content = re.sub(r'(\s+)else:\n\s+results', r'\1else:\n\1    results', content)
    content = re.sub(r'(\s+)elif .+:\n\s+results', r'\1elif \2:\n\1    results', content)
    
    # 4. Fix nested block indentation (try/except/finally)
    content = re.sub(r'(\s+)try:\n\s+', r'\1try:\n\1    ', content)
    content = re.sub(r'(\s+)except .+:\n\s+', r'\1except \2:\n\1    ', content)
    content = re.sub(r'(\s+)finally:\n\s+', r'\1finally:\n\1    ', content)
    
    # 5. Normalize method indentation - ensure all method declarations have 4 spaces
    content = re.sub(r'^(\s*)def test_(\w+)', r'    def test_\2', content, flags=re.MULTILINE)
    content = re.sub(r'^(\s*)def run_tests', r'    def run_tests', content, flags=re.MULTILINE)
    
    # 6. Fix loops, conditionals, and other blocks inside methods to have 8 spaces
    method_indentation_fixes = [
        # Conditionals
        (r'(\s{4}def .+\n\s+.+\n)(\s+)if (.+):', r'\1        if \3:'),
        (r'(\s{4}def .+\n\s+.+\n\s+.+\n)(\s+)if (.+):', r'\1        if \3:'),
        
        # Loops
        (r'(\s{4}def .+\n\s+.+\n)(\s+)for (.+):', r'\1        for \3:'),
        (r'(\s{4}def .+\n\s+.+\n\s+.+\n)(\s+)for (.+):', r'\1        for \3:'),
        
        # Try blocks
        (r'(\s{4}def .+\n\s+.+\n)(\s+)try:', r'\1        try:'),
        (r'(\s{4}def .+\n\s+.+\n\s+.+\n)(\s+)try:', r'\1        try:'),
    ]
    
    # 7. Apply string replacement for common patterns
    string_fixes = [
        # Fix method declarations
        ('return results    def test_from_pretrained', 'return results\n\n    def test_from_pretrained'),
        ('return results    def run_tests', 'return results\n\n    def run_tests'),
        
        # Fix error handling indentation
        ('            else:\n            results', '            else:\n                results'),
        ('            elif', '            elif'),
        
        # Fix common statement indentation
        ('    if device', '        if device'),
        ('    for _ in range', '        for _ in range'),
        ('    try:', '        try:'),
        ('    logger.', '        logger.'),
        
        # Fix nested indentation in try/except blocks
        ('        try:\n        with', '        try:\n            with'),
        ('        except Exception:\n        pass', '        except Exception:\n            pass'),
        
        # Fix trailing content
        ('}\n\ndef', '}\n\n\ndef'),
        (']\n\ndef', ']\n\n\ndef'),
        
        # Fix tokenizer initialization indentation
        ('            # First load the tokenizer', '        # First load the tokenizer'),
        ('            tokenizer =', '        tokenizer ='),
        ('            if tokenizer.pad_token', '        if tokenizer.pad_token'),
        ('            tokenizer.pad_token', '        tokenizer.pad_token'),
        ('            logger.info', '        logger.info'),
        ('            # Create pipeline', '        # Create pipeline'),
        ('            pipeline_kwargs', '        pipeline_kwargs'),
    ]
    
    # Apply all regex fixes
    for pattern, replacement in method_indentation_fixes:
        content = re.sub(pattern, replacement, content)
    
    # Apply all string fixes
    for old, new in string_fixes:
        content = content.replace(old, new)
    
    # 8. Add proper docstring indentation
    content = re.sub(r'(\s{4}def [^:]+:)\s+"""(.+?)"""', r'\1\n        """\2"""', content, flags=re.DOTALL)
    
    # 9. Fix input preparation indentation (common issue in test files)
    if "# Prepare test input" in content:
        content = re.sub(r'(\s+)# Prepare test input', r'        # Prepare test input', content)
        content = re.sub(r'(\s+)(test_input|pipeline_input) =', r'        \2 =', content)
    
    # 10. Fix output processing indentation (another common issue)
    if "# Process output" in content:
        content = re.sub(r'(\s+)# Process output', r'        # Process output', content)
    
    # Write the cleaned content back
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Enhanced cleanup applied to {file_path}")
    return file_path

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Clean up indentation issues in generated test files")
    parser.add_argument("file_path", help="Path to the test file to clean up")
    parser.add_argument("--output", "-o", help="Output path (defaults to overwriting the input file)")
    parser.add_argument("--backup", "-b", action="store_true", help="Create a backup of the original file")
    
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args()
    
    file_path = args.file_path
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    # Create backup if requested
    if args.backup:
        backup_path = f"{file_path}.bak"
        import shutil
        shutil.copy2(file_path, backup_path)
        print(f"Created backup at {backup_path}")
    
    # Run cleanup
    cleaned_path = cleanup_test_file(file_path)
    
    # Output to different file if specified
    if args.output:
        with open(cleaned_path, 'r') as src:
            with open(args.output, 'w') as dst:
                dst.write(src.read())
        print(f"Saved cleaned file to {args.output}")
    
    print(f"Successfully cleaned up {file_path}")

if __name__ == "__main__":
    import argparse
    main()