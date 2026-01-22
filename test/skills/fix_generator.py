#!/usr/bin/env python3
"""
This script fixes key issues in the test generator to correct the syntax errors in generated test files.
"""

import os
import re
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_template_indentation(template_path):
    """Fix indentation issues in template files."""
    if not os.path.exists(template_path):
        logger.error(f"Template file not found: {template_path}")
        return False
    
    logger.info(f"Fixing indentation in template: {template_path}")
    
    with open(template_path, 'r') as f:
        content = f.read()
    
    # Fix indentation in the template's test_from_pretrained method
    # This is one of the key areas with indentation issues
    content = fix_from_pretrained_method(content)
    
    # Fix other indentation and syntax issues
    content = fix_try_except_blocks(content)
    content = fix_docstring_issues(content)
    content = fix_extra_markers(content)
    
    # Write fixed template back
    with open(template_path, 'w') as f:
        f.write(content)
    
    logger.info(f"Fixed template saved: {template_path}")
    return True

def fix_from_pretrained_method(content):
    """Fix indentation issues in the test_from_pretrained method."""
    # The key issue is in the cuda inference loop indentation
    pattern = r"if device == \"cuda\":\s+try:[^\n]*with torch\.no_grad\(\):[^\n]*_ = model\(\*\*inputs\)[^\n]*except Exception:[^\n]*pass\s+(# Run multiple inference passes\s+num_runs = \d+\s+times = \[\]\s+outputs = \[\]\s+for _ in range\(num_runs\):)"
    
    # Fix the indentation to move code outside the CUDA block
    replacement = r"if device == \"cuda\":\n        try:\n            with torch.no_grad():\n                _ = model(**inputs)\n        except Exception:\n            pass\n\n    # Run multiple inference passes\n    num_runs = 3\n    times = []\n    outputs = []\n    \n    for _ in range(num_runs):"
    
    fixed_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    return fixed_content

def fix_try_except_blocks(content):
    """Fix missing or malformed try-except blocks."""
    # Add proper except blocks where missing
    pattern = r"try:([^\n]+)(?!\s*except)"
    replacement = r"try:\1\n    except Exception:\n        pass"
    
    content = re.sub(pattern, replacement, content)
    
    # Ensure consistent indentation in try-except blocks
    lines = content.split('\n')
    fixed_lines = []
    in_try_block = False
    try_indent = 0
    
    for i, line in enumerate(lines):
        if re.match(r'\s*try\s*:', line):
            in_try_block = True
            try_indent = len(line) - len(line.lstrip())
            fixed_lines.append(line)
            continue
        
        if in_try_block and re.match(r'\s*except\s+', line):
            in_try_block = False
            # Check if except block has consistent indentation
            except_indent = len(line) - len(line.lstrip())
            if except_indent != try_indent:
                # Fix indentation to match try block
                fixed_lines.append(' ' * try_indent + line.lstrip())
                continue
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_docstring_issues(content):
    """Fix issues with docstrings."""
    # Fix triple quote issues
    content = content.replace('""""', '"""')
    content = content.replace("''''", "'''")
    
    # Fix unterminated triple quotes
    triple_quote_count = content.count('"""')
    if triple_quote_count % 2 != 0:
        logger.info(f"Fixing unterminated triple quotes (found {triple_quote_count} instances)")
        
        # Simple fix: find the last non-terminated docstring and add closing quotes
        lines = content.split('\n')
        in_docstring = False
        docstring_start = -1
        docstring_indent = 0
        
        for i, line in enumerate(lines):
            if '"""' in line:
                if line.count('"""') % 2 == 1:  # Odd number of triple quotes on this line
                    if not in_docstring:
                        in_docstring = True
                        docstring_start = i
                        docstring_indent = len(line) - len(line.lstrip())
                    else:
                        in_docstring = False
        
        # If we're still in a docstring at the end, add closing quotes
        if in_docstring:
            logger.info(f"Adding missing closing triple quotes at line {docstring_start}")
            indent = ' ' * docstring_indent
            lines.append(f"{indent}\"\"\"")
            content = '\n'.join(lines)
    
    return content

def fix_extra_markers(content):
    """Fix duplicate repository markers and template markers."""
    # Fix duplicate or corrupted registry names
    content = re.sub(r"GPT_GPT_GPT_GPT_J_MODELS_REGISTRY", "GPT_J_MODELS_REGISTRY", content)
    content = re.sub(r"hf_gpt_j_j_j_j_j_j_j_", "hf_gpt_j_", content)
    
    # Remove template markers that might cause issues
    content = re.sub(r"# <TEMPLATE:.*?>", "", content)
    
    return content

def fix_test_generator(generator_path="/home/barberb/ipfs_accelerate_py/test/skills/test_generator_fixed.py"):
    """Fix the test generator file."""
    if not os.path.exists(generator_path):
        logger.error(f"Generator file not found: {generator_path}")
        return False
    
    logger.info(f"Fixing test generator: {generator_path}")
    
    with open(generator_path, 'r') as f:
        content = f.read()
    
    # 1. Fix the token_based_replace function to handle special cases better
    token_based_replace_fix = """
def token_based_replace(template, replacements):
    """
    Replace tokens in the template using a more robust approach that preserves code structure.
    
    Args:
        template (str): The template content
        replacements (dict): Dictionary of token -> replacement mappings
        
    Returns:
        str: The processed template with replacements applied
    """
    # Sort replacements by length (longest first) to avoid partial replacements
    sorted_replacements = sorted(replacements.items(), key=lambda x: len(x[0]), reverse=True)
    
    # First pass: Replace tokens in a context-aware manner
    lines = template.split('\\n')
    processed_lines = []
    
    # Track state
    in_string = False
    string_delimiter = None
    in_comment = False
    
    for line in lines:
        # Process the line character by character
        processed_line = ''
        i = 0
        while i < len(line):
            # Check if we're at the start of a comment
            if line[i:i+1] == '#' and not in_string:
                in_comment = True
                processed_line += line[i]
                i += 1
                continue
                
            # Check if we're at the start of a string
            if line[i:i+3] == '\"\"\"' and not in_string:
                in_string = True
                string_delimiter = '\"\"\"'
                processed_line += line[i:i+3]
                i += 3
                continue
            elif line[i:i+1] in ['"', "'"] and not in_string:
                in_string = True
                string_delimiter = line[i:i+1]
                processed_line += line[i]
                i += 1
                continue
                
            # Check if we're at the end of a string
            if in_string and string_delimiter and i + len(string_delimiter) <= len(line) and line[i:i+len(string_delimiter)] == string_delimiter:
                in_string = False
                processed_line += line[i:i+len(string_delimiter)]
                i += len(string_delimiter)
                string_delimiter = None
                continue
            
            # If we're in a string or comment, don't do replacements
            if in_string or in_comment:
                processed_line += line[i]
                i += 1
                continue
                
            # Check for token replacements
            replaced = False
            for token, replacement in sorted_replacements:
                if i + len(token) <= len(line) and line[i:i+len(token)] == token:
                    # Only replace whole words/identifiers
                    next_char = line[i+len(token):i+len(token)+1] if i+len(token) < len(line) else None
                    prev_char = line[i-1:i] if i > 0 else None
                    
                    # Check if this is a whole word/identifier
                    is_whole_word = (
                        (not next_char or not (next_char.isalnum() or next_char == '_')) and
                        (not prev_char or not (prev_char.isalnum() or prev_char == '_'))
                    )
                    
                    if is_whole_word:
                        processed_line += replacement
                        i += len(token)
                        replaced = True
                        break
            
            # If no replacement was made, keep the original character
            if not replaced:
                processed_line += line[i]
                i += 1
                
        # Add the processed line to the result
        processed_lines.append(processed_line)
        
        # Reset comment state at the end of the line
        in_comment = False
    
    content = '\\n'.join(processed_lines)
    
    # Second pass: Fix known problematic patterns that might result from replacements
    # Fix duplicate imports - common issue with token replacements
    content = re.sub(r"import torch\\nimport torch", "import torch", content)
    content = re.sub(r"import transformers\\nimport transformers", "import transformers", content)
    
    # Fix registry names that might get corrupted during replacement
    content = re.sub(r"(\w+)_(\w+)_\\1_\\2", r"\\1_\\2", content)
    
    return content
"""
    
    # Replace the token_based_replace function with our improved version
    content = re.sub(r"def token_based_replace\(template, replacements\):.*?return '\\n'\.join\(processed_lines\)",
                    token_based_replace_fix, content, flags=re.DOTALL)
    
    # 2. Fix the indentation handling in templates
    preprocess_template_fix = """
def preprocess_template(template_content):
    """
    Preprocess a template by normalizing indentation and adding special markers.
    
    This function ensures consistent indentation in the template and adds special
    markers for sections that require careful handling during replacement.
    """
    lines = template_content.split('\\n')
    processed_lines = []
    
    # Track indentation levels for syntax structures
    stack = []  # Stack of (type, indent_level) pairs
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        
        # Process the line based on its content
        if stripped.startswith(('class ', 'def ')) and stripped.endswith(':'):
            # This is a class or function definition
            while stack and stack[-1][1] >= indent:
                stack.pop()  # Pop anything at the same or deeper indent level
            
            # Add the definition to our stack
            def_type = 'class' if stripped.startswith('class ') else 'function'
            stack.append((def_type, indent))
            processed_lines.append(line)
            continue
            
        elif stripped == 'try:':
            # This is a try block
            stack.append(('try', indent))
            processed_lines.append(line)
            continue
            
        elif stripped.startswith(('except ', 'finally:')) and stack and stack[-1][0] == 'try':
            # This is part of a try-except-finally structure
            try_indent = stack[-1][1]
            if indent != try_indent:
                # Fix indentation to match the try statement
                line = ' ' * try_indent + stripped
                indent = try_indent
            
            if stripped.startswith('except '):
                # Keep track of the except block
                stack.append(('except', indent))
            elif stripped == 'finally:':
                # Keep track of the finally block
                stack.append(('finally', indent))
                
            processed_lines.append(line)
            continue
            
        # Handle content based on its position in our syntax structure stack
        if stack:
            # Check if we need to adjust indentation based on context
            context_type, context_indent = stack[-1]
            
            if context_type in ('try', 'except', 'finally'):
                # Content inside a try/except/finally block should be indented +4
                expected_indent = context_indent + 4
                
                # Only fix indentation if this isn't a new block at the same level
                if stripped and not stripped.endswith(':') and indent <= context_indent:
                    # This should be indented content but isn't
                    line = ' ' * expected_indent + stripped
                    
            elif context_type in ('class', 'function'):
                # Content inside a class or function should be indented +4
                expected_indent = context_indent + 4
                
                # Only fix indentation for content, not new definitions at same level
                if stripped and not stripped.endswith(':') and indent <= context_indent:
                    # This should be indented content but isn't
                    line = ' ' * expected_indent + stripped
        
        # Add the potentially modified line
        processed_lines.append(line)
    
    # Fix unterminated triple quotes
    content = '\\n'.join(processed_lines)
    triple_quote_count = content.count('"""')
    
    if triple_quote_count % 2 != 0:
        # We have unterminated triple quotes, fix it
        lines = processed_lines
        in_docstring = False
        docstring_start = -1
        docstring_indent = 0
        
        for i, line in enumerate(lines):
            if '"""' in line:
                if line.count('"""') % 2 == 1:  # Odd number of triple quotes on this line
                    if not in_docstring:
                        in_docstring = True
                        docstring_start = i
                        docstring_indent = len(line) - len(line.lstrip())
                    else:
                        in_docstring = False
        
        # If we're still in a docstring at the end, add closing quotes
        if in_docstring:
            indent = ' ' * docstring_indent
            lines.append(f"{indent}\"\"\"")
            processed_lines = lines
    
    return '\\n'.join(processed_lines)
"""
    
    # Replace the preprocess_template function with our improved version
    content = re.sub(r"def preprocess_template\(template_content\):.*?return '\\n'\.join\(processed_lines\)",
                    preprocess_template_fix, content, flags=re.DOTALL)
    
    # 3. Fix the post_process_generated_file function for better error handling
    post_process_fix = """
def post_process_generated_file(content):
    """Perform post-processing on the generated file to ensure it's valid Python."""
    # 0. Fix template dependencies (especially for hyphenated models)
    content = fix_template_dependencies(content)
    
    # 1. Fix indentation in try/except blocks
    content = fix_try_except_blocks(content)
    
    # 2. Fix unterminated triple quotes
    content = fix_unterminated_triple_quotes(content)
    
    # 3. Fix docstring-method definition issues
    content = fix_docstring_method_definition_issues(content)
    
    # 4. Fix general indentation issues
    content = fix_indentation_issues(content)
    
    # 5. Fix method boundaries
    content = fix_method_boundaries(content)
    
    # 6. Fix unbalanced delimiters
    content = fix_unbalanced_delimiters(content)
    
    # 7. Final cleanup
    content = final_cleanup(content)
    
    # 8. Special fix for hyphenated model templates with duplicated import statements
    content = re.sub(r"import torch\\nimport torch", "import torch", content)
    content = re.sub(r"import transformers\\nimport transformers", "import transformers", content)
    
    # 9. Fix typical indentation patterns in the test_from_pretrained method
    content = fix_from_pretrained_indentation(content)
    
    # 10. Validate syntax
    try:
        compile(content, "<string>", 'exec')
        return content, True, None
    except SyntaxError as e:
        # If there's still a syntax error, return the error details
        error_message = f"Syntax error on line {e.lineno}: {e.msg}"
        if hasattr(e, 'text') and e.text:
            error_message += f"\\n{e.text}"
            if hasattr(e, 'offset') and e.offset:
                error_message += "\\n" + " " * (e.offset - 1) + "^"
        
        # If the error is about a try block, attempt a specific fix
        if "expected 'except' or 'finally' block" in e.msg and hasattr(e, 'lineno'):
            line_no = e.lineno
            lines = content.split('\\n')
            
            # Look for a try block structure to fix
            if line_no > 0 and line_no < len(lines):
                try_line_no = None
                
                # Find the closest preceding "try:" line
                for i in range(line_no-1, max(0, line_no-10), -1):
                    if lines[i].strip() == "try:":
                        try_line_no = i
                        break
                
                if try_line_no is not None:
                    # Get the indentation of the try block
                    try_indent = len(lines[try_line_no]) - len(lines[try_line_no].lstrip())
                    
                    # Add a proper except block after the problematic line
                    lines.insert(line_no, ' ' * try_indent + "except Exception:")
                    lines.insert(line_no + 1, ' ' * (try_indent + 4) + "pass")
                    
                    # Join the modified lines
                    content = '\\n'.join(lines)
                    
                    # Try to validate again
                    try:
                        compile(content, "<string>", 'exec')
                        return content, True, None
                    except SyntaxError as new_e:
                        # Return the updated content with the new error
                        new_error_message = f"Syntax error on line {new_e.lineno}: {new_e.msg}"
                        if hasattr(new_e, 'text') and new_e.text:
                            new_error_message += f"\\n{new_e.text}"
                            if hasattr(new_e, 'offset') and new_e.offset:
                                new_error_message += "\\n" + " " * (new_e.offset - 1) + "^"
                        return content, False, new_error_message
        
        return content, False, error_message
"""
    
    # Add the fix_from_pretrained_indentation function
    fix_from_pretrained_indentation = """
def fix_from_pretrained_indentation(content):
    """Fix indentation issues specifically in the test_from_pretrained method."""
    # Look for the problematic pattern (cuda if block with run inference code)
    pattern = r"(if device == \"cuda\":\\s+try:[^\\n]*with torch\\.no_grad\\(\\):[^\\n]*_ = model\\(\\*\\*inputs\\)[^\\n]*except Exception:[^\\n]*pass)\\s+(# Run multiple inference passes\\s+num_runs = \\d+\\s+times = \\[\\]\\s+outputs = \\[\\]\\s+for _ in range\\(num_runs\\):)"
    
    # Fix the indentation
    replacement = r"\\1\\n\\n    \\2"
    
    # Apply the fix
    fixed_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Fix missing indentation after the for loop
    pattern2 = r"(for _ in range\\(num_runs\\):[^\\n]*start_time = time\\.time\\(\\)[^\\n]*with torch\\.no_grad\\(\\):[^\\n]*output = model\\(\\*\\*inputs\\)[^\\n]*end_time = time\\.time\\(\\)[^\\n]*times\\.append\\(end_time - start_time\\)[^\\n]*outputs\\.append\\(output\\))\s+# Calculate statistics"
    
    replacement2 = r"\\1\\n\\n    # Calculate statistics"
    fixed_content = re.sub(pattern2, replacement2, fixed_content, flags=re.DOTALL)
    
    return fixed_content
"""
    
    # Add the new function to the content
    content = content.replace("def fix_indentation_issues(content):", fix_from_pretrained_indentation + "\n\ndef fix_indentation_issues(content):")
    
    # Replace the post_process_generated_file function with our improved version
    content = re.sub(r"def post_process_generated_file\(content\):.*?return content, False, error_message",
                    post_process_fix, content, flags=re.DOTALL)
    
    # Save the updated generator
    with open(generator_path, 'w') as f:
        f.write(content)
    
    logger.info(f"Fixed generator saved: {generator_path}")
    return True

def fix_templates_directory(templates_dir="/home/barberb/ipfs_accelerate_py/test/skills/templates"):
    """Fix all templates in the templates directory."""
    if not os.path.exists(templates_dir):
        logger.error(f"Templates directory not found: {templates_dir}")
        return False
    
    templates_fixed = 0
    for template_file in os.listdir(templates_dir):
        if template_file.endswith(".py") and not template_file.endswith(".bak"):
            template_path = os.path.join(templates_dir, template_file)
            if fix_template_indentation(template_path):
                templates_fixed += 1
    
    logger.info(f"Fixed {templates_fixed} templates in {templates_dir}")
    return templates_fixed > 0

def main():
    """Main function to fix the test generator and templates."""
    # 1. First fix the test generator
    generator_fixed = fix_test_generator()
    
    # 2. Fix all templates
    templates_fixed = fix_templates_directory()
    
    # Report results
    if generator_fixed:
        logger.info("Successfully fixed the test generator.")
    else:
        logger.error("Failed to fix the test generator.")
    
    if templates_fixed:
        logger.info("Successfully fixed the templates.")
    else:
        logger.error("Failed to fix the templates.")
    
    return generator_fixed and templates_fixed

if __name__ == "__main__":
    main()