#!/usr/bin/env python3
"""
Comprehensive cleanup for test files.
This script performs a series of targeted replacements to fix common issues.
"""
import os
import re
import sys
import shutil
import argparse

def backup_file(file_path):
    """Create a backup of the file."""
    backup_path = f"{file_path}.bak"
    shutil.copy2(file_path, backup_path)
    print(f"Created backup at {backup_path}")
    return backup_path

def fix_imports(content):
    """Fix import section indentation issues."""
    # Fix torch import block
    content = re.sub(
        r'try:\n\s+import torch\n\s+HAS_TORCH = True\nexcept ImportError:\n\s+torch = MagicMock\(\)\n\s+HAS_TORCH = False\n\s+logger\.warning',
        'try:\n    import torch\n    HAS_TORCH = True\nexcept ImportError:\n    torch = MagicMock()\n    HAS_TORCH = False\n    logger.warning',
        content
    )
    
    # Fix transformers import block
    content = re.sub(
        r'try:\n\s+import transformers\n\s+HAS_TRANSFORMERS = True\nexcept ImportError:\n\s+transformers = MagicMock\(\)\n\s+HAS_TRANSFORMERS = False\n\s+logger\.warning',
        'try:\n    import transformers\n    HAS_TRANSFORMERS = True\nexcept ImportError:\n    transformers = MagicMock()\n    HAS_TRANSFORMERS = False\n    logger.warning',
        content
    )
    
    # Fix tokenizers import block
    content = re.sub(
        r'try:\n\s+import tokenizers\n\s+HAS_TOKENIZERS = True\nexcept ImportError:\n\s+tokenizers = MagicMock\(\)\n\s+HAS_TOKENIZERS = False\n\s+logger\.warning',
        'try:\n    import tokenizers\n    HAS_TOKENIZERS = True\nexcept ImportError:\n    tokenizers = MagicMock()\n    HAS_TOKENIZERS = False\n    logger.warning',
        content
    )
    
    # Fix sentencepiece import block
    content = re.sub(
        r'try:\n\s+import sentencepiece\n\s+HAS_SENTENCEPIECE = True\nexcept ImportError:\n\s+sentencepiece = MagicMock\(\)\n\s+HAS_SENTENCEPIECE = False\n\s+logger\.warning',
        'try:\n    import sentencepiece\n    HAS_SENTENCEPIECE = True\nexcept ImportError:\n    sentencepiece = MagicMock()\n    HAS_SENTENCEPIECE = False\n    logger.warning',
        content
    )
    
    return content

def fix_class_methods(content):
    """Fix class method indentation issues."""
    # Fix method boundaries
    content = re.sub(r'(\s+)return results\s+(\s+)def', r'\1return results\n\n\2def', content)
    
    # Fix performance_stats method boundary
    content = re.sub(r'self\.performance_stats = {}\s+(\s+)def', r'self.performance_stats = {}\n\n\1def', content)
    
    # Fix method declarations to have 4 spaces
    content = re.sub(r'^(\s*)def (test_\w+|run_tests)\(', r'    def \2(', content, flags=re.MULTILINE)
    
    # Fix indentation of method body - this is a more complex operation
    # We'll need to identify method blocks and re-indent them properly
    
    return content

def fix_dependency_checks(content):
    """Fix dependency check indentation issues."""
    # Fix dependency check blocks
    fixed_content = re.sub(
        r'(\s+)if not HAS_TOKENIZERS:\n\s+results\["(pipeline|from_pretrained)_error_type"\] = "missing_dependency"',
        r'\1if not HAS_TOKENIZERS:\n\1    results["\2_error_type"] = "missing_dependency"',
        content
    )
    
    fixed_content = re.sub(
        r'(\s+)if not HAS_SENTENCEPIECE:\n\s+results\["(pipeline|from_pretrained)_error_type"\] = "missing_dependency"',
        r'\1if not HAS_SENTENCEPIECE:\n\1    results["\2_error_type"] = "missing_dependency"',
        fixed_content
    )
    
    fixed_content = re.sub(
        r'(\s+)if not HAS_PIL:\n\s+results\["(pipeline|from_pretrained)_error_type"\] = "missing_dependency"',
        r'\1if not HAS_PIL:\n\1    results["\2_error_type"] = "missing_dependency"',
        fixed_content
    )
    
    return fixed_content

def fix_mock_definitions(content):
    """Fix mock class definition indentation issues."""
    # Fix mock class definitions
    fixed_content = re.sub(
        r'if not HAS_TOKENIZERS:\n(\s+)class MockTokenizer:',
        r'if not HAS_TOKENIZERS:\n    class MockTokenizer:',
        content
    )
    
    fixed_content = re.sub(
        r'if not HAS_SENTENCEPIECE:\n(\s+)class MockSentencePieceProcessor:',
        r'if not HAS_SENTENCEPIECE:\n    class MockSentencePieceProcessor:',
        fixed_content
    )
    
    fixed_content = re.sub(
        r'if not HAS_PIL:\n(\s+)class MockImage:',
        r'if not HAS_PIL:\n    class MockImage:',
        fixed_content
    )
    
    # Fix tokenizer assignment indentation
    fixed_content = re.sub(
        r'(\s+)tokenizers\.Tokenizer = MockTokenizer',
        r'    tokenizers.Tokenizer = MockTokenizer',
        fixed_content
    )
    
    # Fix sentencepiece assignment indentation
    fixed_content = re.sub(
        r'(\s+)sentencepiece\.SentencePieceProcessor = MockSentencePieceProcessor',
        r'    sentencepiece.SentencePieceProcessor = MockSentencePieceProcessor',
        fixed_content
    )
    
    return fixed_content

def fix_try_except_blocks(content):
    """Fix try/except block indentation."""
    # Fix try/except blocks
    fixed_content = re.sub(
        r'(\s+)try:\n\s+(\w)',
        r'\1try:\n\1    \2',
        content
    )
    
    fixed_content = re.sub(
        r'(\s+)except Exception:\n\s+(\w)',
        r'\1except Exception:\n\1    \2',
        fixed_content
    )
    
    fixed_content = re.sub(
        r'(\s+)except ImportError:\n\s+(\w)',
        r'\1except ImportError:\n\1    \2',
        fixed_content
    )
    
    # Fix try/except within methods (deeper indentation)
    fixed_content = re.sub(
        r'(\s{8})try:\n\s+(\w)',
        r'\1try:\n\1    \2',
        fixed_content
    )
    
    fixed_content = re.sub(
        r'(\s{8})except Exception:\n\s+(\w)',
        r'\1except Exception:\n\1    \2',
        fixed_content
    )
    
    return fixed_content

def fix_if_blocks(content):
    """Fix if block indentation."""
    # Fix if blocks - especially device checks
    fixed_content = re.sub(
        r'(\s+)if device == "cuda":\n\s+(\w)',
        r'\1if device == "cuda":\n\1    \2',
        content
    )
    
    fixed_content = re.sub(
        r'(\s+)if device != "cpu":\n\s+(\w)',
        r'\1if device != "cpu":\n\1    \2',
        fixed_content
    )
    
    # Handle if/else, if/elif structures
    fixed_content = re.sub(
        r'(\s+)if (.+):\n\s+(.+)\n\s+else:\n\s+(\w)',
        r'\1if \2:\n\1    \3\n\1else:\n\1    \4',
        fixed_content
    )
    
    fixed_content = re.sub(
        r'(\s+)elif (.+):\n\s+(\w)',
        r'\1elif \2:\n\1    \3',
        fixed_content
    )
    
    return fixed_content

def fix_method_body(content):
    """Fix method body indentation (should be 8 spaces)."""
    # This is a more complex operation that would require parsing the Python structure.
    # For simplicity, we'll use pattern matching for common issues
    
    # Fix pipeline kwargs indentation
    fixed_content = re.sub(
        r'(\s+)pipeline_kwargs = {\n\s+"task": self\.task,',
        r'\1pipeline_kwargs = {\n\1    "task": self.task,',
        content
    )
    
    # Fix tokenizer loading indentation
    fixed_content = re.sub(
        r'(\s+)tokenizer = transformers\.\w+\.from_pretrained\(\n\s+self\.model_id,',
        r'\1tokenizer = transformers.\w+.from_pretrained(\n\1    self.model_id,',
        fixed_content
    )
    
    # Fix model loading indentation
    fixed_content = re.sub(
        r'(\s+)model = model_class\.from_pretrained\(\n\s+self\.model_id,',
        r'\1model = model_class.from_pretrained(\n\1    self.model_id,',
        fixed_content
    )
    
    return fixed_content

def fix_nested_blocks(content):
    """Fix nested block indentation (should be 12 spaces)."""
    # Most nested blocks should have 12 spaces
    # For simplicity, we'll focus on common patterns
    
    # Fix for loops inside try blocks
    fixed_content = re.sub(
        r'(\s{8})for _ in range\(num_runs\):\n\s+(\w)',
        r'\1for _ in range(num_runs):\n\1    \2',
        content
    )
    
    # Fix with blocks inside try blocks
    fixed_content = re.sub(
        r'(\s{8})with torch\.no_grad\(\):\n\s+(\w)',
        r'\1with torch.no_grad():\n\1    \2',
        fixed_content
    )
    
    return fixed_content

def fix_method_spacing(content):
    """Fix spacing between methods."""
    # Ensure proper spacing between methods
    fixed_content = re.sub(
        r'(\s+def \w+\(self,.+\):.*?return [^}]+})\s*(\s+def)',
        r'\1\n\n\2',
        content,
        flags=re.DOTALL
    )
    
    return fixed_content

def fix_indentation(file_path, create_backup=True):
    """Apply all indentation fixes to a file."""
    if create_backup:
        backup_file(file_path)
        
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Apply each fix in sequence
    content = fix_imports(content)
    content = fix_class_methods(content)
    content = fix_dependency_checks(content)
    content = fix_mock_definitions(content)
    content = fix_try_except_blocks(content)
    content = fix_if_blocks(content)
    content = fix_method_body(content)
    content = fix_nested_blocks(content)
    content = fix_method_spacing(content)
    
    # Write the fixed content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
        
    print(f"Applied comprehensive indentation fixes to: {file_path}")
    return file_path

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fix indentation issues in test files")
    parser.add_argument("files", nargs="+", help="Files to fix")
    parser.add_argument("--no-backup", action="store_true", help="Don't create backups")
    
    args = parser.parse_args()
    
    for file_path in args.files:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            continue
        
        fix_indentation(file_path, create_backup=not args.no_backup)
    
    print(f"Fixed {len(args.files)} files")

if __name__ == "__main__":
    main()