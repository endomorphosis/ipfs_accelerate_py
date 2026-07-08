#\!/usr/bin/env python3
"""
Simple indentation fixer for test files.
"""
import os
import re
import sys

# Example use: python simple_fixer.py fixed_files_manual/test_hf_bert.py fixed_files_manual/test_hf_gpt2.py

def fix_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Apply a series of pattern-based fixes
    
    # Fix indentation in try/except blocks
    content = re.sub(r'(\s+)HAS_\w+ = False\n\s+logger\.warning', r'\1HAS_\w+ = False\n\1logger.warning', content)
    
    # Fix method boundaries
    content = re.sub(r'(\s+)return results\s+def', r'\1return results\n\n\1def', content)
    
    # Fix indentation in local_files_only block
    content = re.sub(r'(\s+)}\s+# Time tokenizer loading', r'\1}\n\n\1# Time tokenizer loading', content)
    
    # Fix various indentation patterns
    content = content.replace('    HAS_TORCH = False\n        logger.warning', '    HAS_TORCH = False\n    logger.warning')
    content = content.replace('    HAS_TRANSFORMERS = False\n        logger.warning', '    HAS_TRANSFORMERS = False\n    logger.warning')
    content = content.replace('    HAS_TOKENIZERS = False\n        logger.warning', '    HAS_TOKENIZERS = False\n    logger.warning')
    content = content.replace('    HAS_SENTENCEPIECE = False\n        logger.warning', '    HAS_SENTENCEPIECE = False\n    logger.warning')
    content = content.replace('    HAS_PIL = False\n        logger.warning', '    HAS_PIL = False\n    logger.warning')
    
    # Fix import blocks
    content = content.replace('\n        try:\n        import', '\n    try:\n        import')
    
    # Fix class methods
    content = content.replace('self.performance_stats = {}    def', 'self.performance_stats = {}\n\n    def')
    content = content.replace('return results    def', 'return results\n\n    def')
    
    # Write fixed content back
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed indentation in {file_path}")

# Process command line arguments
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_fixer.py file1.py file2.py ...")
        sys.exit(1)
    
    for file_path in sys.argv[1:]:
        if os.path.exists(file_path):
            fix_file(file_path)
        else:
            print(f"Error: File {file_path} not found")
