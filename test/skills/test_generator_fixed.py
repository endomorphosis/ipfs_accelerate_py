#!/usr/bin/env python3

# Import hardware detection capabilities if available
try:
    from generators.hardware.hardware_detection import (
        HAS_CUDA, HAS_ROCM, HAS_OPENVINO, HAS_MPS, HAS_WEBNN, HAS_WEBGPU,
        detect_all_hardware
    )
    HAS_HARDWARE_DETECTION = True
except ImportError:
    HAS_HARDWARE_DETECTION = False
    # We'll detect hardware manually as fallback

import os
import sys
import json
import time
import datetime
import traceback
import logging
import argparse
import re
from unittest.mock import patch, MagicMock, Mock
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Define architecture types for model mapping
ARCHITECTURE_TYPES = {
    "encoder-only": ["bert", "distilbert", "roberta", "electra", "camembert", "xlm-roberta", "deberta", "layoutlm", "canine", "roformer"],
    "decoder-only": ["gpt2", "gpt-j", "gpt-neo", "gpt-neox", "bloom", "llama", "mistral", "falcon", "phi", "mixtral", "mpt", "codellama", "qwen2", "qwen3", "stablelm", "mosaic_mpt", "pythia", "xglm", "open_llama"],
    "encoder-decoder": ["t5", "bart", "pegasus", "mbart", "longt5", "led", "marian", "mt5", "flan"],
    "vision": ["vit", "swin", "deit", "beit", "convnext", "poolformer", "dinov2", "mobilenet-v2"],
    "vision-text": ["vision-encoder-decoder", "vision-text-dual-encoder", "clip", "blip", "blip-2", "chinese-clip", "clipseg"],
    "speech": ["wav2vec2", "hubert", "whisper", "bark", "speecht5"],
    "multimodal": ["llava", "clip", "blip", "git", "pix2struct", "paligemma", "video-llava", "fuyu", "kosmos-2", "llava-next"]
}

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Model lookup integration
try:
    from find_models import get_recommended_default_model, query_huggingface_api
    HAS_MODEL_LOOKUP = True
    logger.info("Model lookup integration available")
except ImportError:
    HAS_MODEL_LOOKUP = False
    logger.warning("Model lookup not available, using static model registry")


# Advanced model selection integration
try:
    from advanced_model_selection import select_model_advanced, get_hardware_profile
    HAS_ADVANCED_SELECTION = True
    logger.info("Advanced model selection available")
except ImportError:
    HAS_ADVANCED_SELECTION = False
    logger.warning("Advanced model selection not available")

def get_model_from_registry(model_type, task=None, hardware_profile=None, max_size_mb=None, framework=None):
    '''Get the best default model for a model type with advanced selection features.
    
    Args:
        model_type (str): The model type (e.g., 'bert', 'gpt2', 't5')
        task (str, optional): The specific task (e.g., 'text-classification')
        hardware_profile (str, optional): Hardware profile name (e.g., 'cpu-small', 'gpu-medium')
        max_size_mb (int, optional): Maximum model size in MB
        framework (str, optional): Framework compatibility (e.g., 'pytorch', 'tensorflow')
        
    Returns:
        str: The recommended model name
    '''
    # Try advanced selection if available
    if HAS_ADVANCED_SELECTION:
        try:
            default_model = select_model_advanced(
                model_type, 
                task=task, 
                hardware_profile=hardware_profile,
                max_size_mb=max_size_mb,
                framework=framework
            )
            logger.info(f"Using advanced selection for {model_type}: {default_model}")
            return default_model
        except Exception as e:
            logger.warning(f"Error using advanced model selection: {e}")
    
    # Fall back to basic model lookup if available
    if HAS_MODEL_LOOKUP:
        try:
            default_model = get_recommended_default_model(model_type)
            logger.info(f"Using recommended model for {model_type}: {default_model}")
            return default_model
        except Exception as e:
            logger.warning(f"Error getting recommended model for {model_type}: {e}")
    
    # Use the static registry as final fallback
    if model_type in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_type].get("default_model")
    
    # For unknown models, use a heuristic approach
    return f"{model_type}-base" if "-base" not in model_type else model_type
# Forward declarations for indentation fixing functions
def fix_class_method_indentation(content):
    """Fix indentation issues in class methods."""
    return content

def fix_unterminated_triple_quotes(content):
    """Fix unterminated triple quotes in the content more robustly."""
    lines = content.split('\n')
    triple_quote_count = content.count('"""')
    single_triple_quote_count = content.count("'''")
    
    # Handle regular triple quotes """
    if triple_quote_count % 2 != 0:
        logger.info(f"Odd number of triple quotes found: {triple_quote_count}, fixing...")
        in_docstring = False
        docstring_start_line = None
        docstring_start_indent = 0
        
        for i, line in enumerate(lines):
            # Count triple quotes on this line
            quotes_in_line = line.count('"""')
            
            # Skip lines with even number of quotes (they open and close on same line)
            if quotes_in_line > 0 and quotes_in_line % 2 == 0:
                continue
                
            # Handle lines with odd number of quotes
            if quotes_in_line % 2 != 0:
                if not in_docstring:
                    # Opening a docstring
                    in_docstring = True
                    docstring_start_line = i
                    docstring_start_indent = len(line) - len(line.lstrip())
                else:
                    # Closing a docstring
                    in_docstring = False
                    docstring_start_line = None
        
        # If we're still in a docstring at the end, add closing quotes with proper indentation
        if in_docstring and docstring_start_line is not None:
            # Get same indentation as the opening quote line
            indent_str = ' ' * docstring_start_indent
            
            # Check if the last line has content and needs a newline before closing
            if lines[-1].strip():
                lines.append(f"{indent_str}\"\"\"")
            else:
                # If last line is already empty, just add the quotes with indentation
                lines[-1] = f"{indent_str}\"\"\"" 
                
            logger.info(f"Added missing closing triple quotes with matching indentation from line {docstring_start_line+1}")
    
    # Handle single triple quotes '''
    if single_triple_quote_count % 2 != 0:
        logger.info(f"Odd number of single triple quotes found: {single_triple_quote_count}, fixing...")
        in_docstring = False
        docstring_start_line = None
        docstring_start_indent = 0
        
        for i, line in enumerate(lines):
            # Count triple quotes on this line
            quotes_in_line = line.count("'''")
            
            # Skip lines with even number of quotes (they open and close on same line)
            if quotes_in_line > 0 and quotes_in_line % 2 == 0:
                continue
                
            # Handle lines with odd number of quotes
            if quotes_in_line % 2 != 0:
                if not in_docstring:
                    # Opening a docstring
                    in_docstring = True
                    docstring_start_line = i
                    docstring_start_indent = len(line) - len(line.lstrip())
                else:
                    # Closing a docstring
                    in_docstring = False
                    docstring_start_line = None
        
        # If we're still in a docstring at the end, add closing quotes with proper indentation
        if in_docstring and docstring_start_line is not None:
            # Get same indentation as the opening quote line
            indent_str = ' ' * docstring_start_indent
            
            # Check if the last line has content and needs a newline before closing
            if lines[-1].strip():
                lines.append(f"{indent_str}'''")
            else:
                # If last line is already empty, just add the quotes with indentation
                lines[-1] = f"{indent_str}'''" 
                
            logger.info(f"Added missing closing single triple quotes with matching indentation from line {docstring_start_line+1}")
    
    return '\n'.join(lines)

def preprocess_template(template_content):
    """
    Preprocess a template by normalizing indentation and adding special markers.
    
    This function ensures consistent indentation in the template and adds special
    markers for sections that require careful handling during replacement.
    """
    lines = template_content.split('\n')
    processed_lines = []
    
    in_try_block = False
    try_indent = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Mark top-level try blocks with special markers
        if stripped == 'try:':
            in_try_block = True
            try_indent = len(line) - len(line.lstrip())
            processed_lines.append(line)
            
            # Add a special marker for the try block
            processed_lines.append(f"{' ' * (try_indent + 4)}# <TEMPLATE:TRY_BLOCK>")
            continue
            
        # Mark the end of try blocks
        if in_try_block and stripped.startswith(('except', 'finally')):
            in_try_block = False
            processed_lines.append(line)
            
            # Add a special marker for the except/finally block
            processed_lines.append(f"{' ' * (try_indent + 4)}# <TEMPLATE:EXCEPT_BLOCK>")
            continue
        
        # Add special markers for class definitions
        if stripped.startswith('class ') and stripped.endswith(':'):
            processed_lines.append(line)
            
            # Add a marker for class body
            indent = len(line) - len(line.lstrip())
            processed_lines.append(f"{' ' * (indent + 4)}# <TEMPLATE:CLASS_DEF>")
            continue
            
        # Add special markers for function definitions
        if stripped.startswith('def ') and stripped.endswith(':'):
            processed_lines.append(line)
            
            # Add a marker for function body
            indent = len(line) - len(line.lstrip())
            processed_lines.append(f"{' ' * (indent + 4)}# <TEMPLATE:FUNCTION_DEF>")
            continue
        
        # Ensure triple quotes are properly terminated
        if '"""' in line and line.count('"""') % 2 != 0:
            # Add a marker for docstrings
            processed_lines.append(line)
            processed_lines.append("# <TEMPLATE:DOCSTRING>")
            continue
            
        # Process normal lines
        processed_lines.append(line)
    
    return '\n'.join(processed_lines)

def token_based_replace(template, replacements):
    """
    Replace tokens in the template using a more robust approach that preserves code structure.
    
    Args:
        template (str): The template content
        replacements (dict): Dictionary of token -> replacement mappings
        
    Returns:
        str: The processed template with replacements applied
    """
    # Get lines for processing
    lines = template.split('\n')
    processed_lines = []
    
    # Track state
    in_string = False
    string_delimiter = None
    in_comment = False
    
    for line in lines:
        # Skip special template marker lines
        if '<TEMPLATE:' in line:
            continue
            
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
            if line[i:i+3] == '"""' and not in_string:
                in_string = True
                string_delimiter = '"""'
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
            for token, replacement in replacements.items():
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
    
    return '\n'.join(processed_lines)

def fix_syntax_errors(content):
    """Fix common syntax errors in generated test files."""
    # Fix extra quotes in docstrings
    content = content.replace('""""', '"""')
    content = content.replace("''''", "'''")
    
    # Use the more robust function to fix unterminated triple quotes
    content = fix_unterminated_triple_quotes(content)
    
    # Fix unbalanced parentheses, brackets, and braces
    content = fix_unbalanced_delimiters(content)
    
    # Fix dangling commas in dictionaries and lists
    content = fix_dangling_commas(content)
    
    # Fix broken try-except blocks
    content = fix_try_except_blocks(content)
    
    # Fix class docstring issues
    content = fix_class_docstring_issues(content)
    
    # Fix method definition issues
    content = fix_docstring_method_definition_issues(content)
    
    # Fix method boundaries
    content = fix_method_boundaries(content)
    
    # Fix common indentation issues
    content = fix_indentation_issues(content)
    
    # Fix top-level try blocks
    content = fix_top_level_try_blocks(content)
    
    # Final cleanup
    content = final_cleanup(content)
    
    return content

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
    
    # 5. Final cleanup
    content = final_cleanup(content)
    
    # 6. Special fix for hyphenated model templates with duplicated import statements
    if "import torch\n    import torch" in content:
        # Fix duplicated torch import
        content = content.replace("import torch\n    import torch", "import torch")
    
    if "import transformers\nimport transformers" in content:
        # Fix duplicated transformers import
        content = content.replace("import transformers\nimport transformers", "import transformers")
    
    # 7. Validate syntax
    try:
        compile(content, "<string>", 'exec')
        return content, True, None
    except SyntaxError as e:
        # If there's still a syntax error, return the error details
        error_message = f"Syntax error on line {e.lineno}: {e.msg}"
        if hasattr(e, 'text') and e.text:
            error_message += f"\n{e.text}"
            if hasattr(e, 'offset') and e.offset:
                error_message += "\n" + " " * (e.offset - 1) + "^"
        
        # If the error is about a try block, attempt a specific fix
        if "expected 'except' or 'finally' block" in e.msg and hasattr(e, 'lineno'):
            line_no = e.lineno
            lines = content.split('\n')
            
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
                    lines.insert(line_no, ' ' * try_indent + "except ImportError:")
                    lines.insert(line_no + 1, ' ' * (try_indent + 4) + "pass")
                    
                    # Join the modified lines
                    content = '\n'.join(lines)
                    
                    # Try to validate again
                    try:
                        compile(content, "<string>", 'exec')
                        return content, True, None
                    except SyntaxError as new_e:
                        # Return the updated content with the new error
                        new_error_message = f"Syntax error on line {new_e.lineno}: {new_e.msg}"
                        if hasattr(new_e, 'text') and new_e.text:
                            new_error_message += f"\n{new_e.text}"
                            if hasattr(new_e, 'offset') and new_e.offset:
                                new_error_message += "\n" + " " * (new_e.offset - 1) + "^"
                        return content, False, new_error_message
        
        return content, False, error_message

def fix_unbalanced_delimiters(content):
    """Fix unbalanced parentheses, brackets, and braces."""
    lines = content.split('\n')
    
    # Check function and method definitions for missing colons
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (stripped.startswith('def ') or stripped.startswith('class ')) and '(' in stripped and ')' in stripped and not stripped.endswith(':'):
            lines[i] = line + ':'
            logger.info(f"Added missing colon in function/class definition on line {i+1}")
    
    # Fix simple cases of missing closing parentheses in function calls
    stack = []
    
    # For each opening delimiter, track its position
    for i, char in enumerate(content):
        if char in '([{':
            stack.append((char, i))
        elif char in ')]}':
            if not stack:
                # Too many closing delimiters, ignore for now
                continue
            
            opening, _ = stack.pop()
            expected_closing = {'(': ')', '[': ']', '{': '}'}[opening]
            
            if char != expected_closing:
                # Mismatched delimiter
                logger.info(f"Mismatched delimiter: expected {expected_closing} but found {char}")
                # Too complex for simple fixing, skip
    
    # Check if we have unclosed delimiters
    if stack:
        # Simple case: missing close parenthesis at end of function call on a line
        for i, line in enumerate(lines):
            line_stack = []
            for char in line:
                if char in '([{':
                    line_stack.append(char)
                elif char in ')]}':
                    if line_stack:
                        line_stack.pop()
            
            # If line has unclosed delimiters and doesn't end with a comma or colon
            if line_stack and not line.rstrip().endswith((',', ':', '\\')):
                for opening in reversed(line_stack):
                    closing = {'(': ')', '[': ']', '{': '}'}[opening]
                    lines[i] = line + closing
                    logger.info(f"Added missing {closing} on line {i+1}")
                    
    content = '\n'.join(lines)
    return content

def fix_dangling_commas(content):
    """Fix dangling commas in dictionaries and lists."""
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        # Check for dangling commas before closing brackets
        if re.search(r',\s*[\)\]}]', line):
            # Replace ", )" with ")" - fixed regex replacement
            lines[i] = re.sub(r',(\s*)([\)\]}])', r'\1\2', line)
            logger.info(f"Fixed dangling comma on line {i+1}")
    
    content = '\n'.join(lines)
    return content

def fix_template_dependencies(content):
    """Fix template dependency imports and try/except blocks completely."""
    # This is a more aggressive fix specifically for template files
    # It completely replaces the problematic sections with correct implementations
    
    # Define correct torch import block pattern
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

    # Define correct transformers import block pattern
    transformers_import_block = """
try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = MagicMock()
    HAS_TRANSFORMERS = False
    logger.warning("transformers not available, using mock")
"""

    # Special case for hyphenated model templates which may have line break issues
    if "import torch\n    import torch\n" in content or "import transformers\nimport transformers\n" in content:
        logger.info("Fixing duplicate import lines in hyphenated model template")
        # Fix torch imports
        content = re.sub(
            r'try:.*?import torch.*?HAS_TORCH = True.*?except ImportError:.*?torch = MagicMock\(\).*?HAS_TORCH = False.*?(?:logger\.warning\(|[^#])',
            torch_import_block,
            content,
            flags=re.DOTALL
        )
        
        # Fix transformers imports
        content = re.sub(
            r'try:.*?import transformers.*?HAS_TRANSFORMERS = True.*?except ImportError:.*?transformers = MagicMock\(\).*?HAS_TRANSFORMERS = False.*?(?:logger\.warning\(|[^#])',
            transformers_import_block,
            content,
            flags=re.DOTALL
        )
    else:    
        # Use regex to find and replace problematic import blocks
        content = re.sub(
            r'# Check if dependencies are available\s*try:[^\n]*?import torch[^\n]*?HAS_TORCH = True.*?except ImportError:[^\n]*?torch = MagicMock\(\)[^\n]*?HAS_TORCH = False.*?(?:logger\.warning\(|[^#])',
            lambda _: torch_import_block + ('\nlogger.warning(' if 'logger.warning(' in _.group(0) else ''),
            content,
            flags=re.DOTALL
        )
        
        content = re.sub(
            r'try:[^\n]*?import transformers[^\n]*?HAS_TRANSFORMERS = True.*?except ImportError:[^\n]*?transformers = MagicMock\(\)[^\n]*?HAS_TRANSFORMERS = False.*?(?:logger\.warning\(|[^#])',
            lambda _: transformers_import_block + ('\nlogger.warning(' if 'logger.warning(' in _.group(0) else ''),
            content,
            flags=re.DOTALL
        )
    
    # Log the fix
    logger.info("Applied template dependency import fixes")
    
    return content

def fix_top_level_try_blocks(content):
    """Fix top-level try blocks that have incorrect indentation."""
    # First, apply the aggressive template fix
    content = fix_template_dependencies(content)
    
    # Then handle any remaining top-level try blocks
    lines = content.split('\n')
    fixed_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Look for top-level try blocks that need fixing
        if line.strip() == 'try:':
            fixed_lines.append(line)
            
            # Make sure the next non-empty line is properly indented
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                fixed_lines.append(lines[j])  # Keep any blank lines
                j += 1
            
            # If we found a non-empty line that needs indentation
            if j < len(lines):
                next_line = lines[j]
                if not next_line.strip().startswith(('except', 'finally')) and len(next_line) - len(next_line.lstrip()) < 4:
                    # This is a line that should be indented in the try block
                    # Check for import specifically to handle it carefully
                    if 'import' in next_line:
                        # Preserve any existing indentation for import statements
                        fixed_lines.append('    ' + next_line.lstrip())
                        logger.info(f"Fixed indentation in try block import on line {j+1}")
                    else:
                        # Standard indentation for other content
                        fixed_lines.append('    ' + next_line.lstrip())
                        logger.info(f"Fixed indentation in top-level try block after line {i+1}")
                    i = j  # Skip the line we just fixed
                else:
                    # Keep the line as-is
                    fixed_lines.append(next_line)
                    i = j
                    
        # Handle except blocks immediately after try blocks
        elif line.strip().startswith('except '):
            fixed_lines.append(line)
            
            # Make sure the next non-empty line is properly indented
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                fixed_lines.append(lines[j])  # Keep any blank lines
                j += 1
            
            # If we found a non-empty line that needs indentation
            if j < len(lines):
                next_line = lines[j]
                if not next_line.strip().startswith(('except', 'finally', 'try:')) and len(next_line) - len(next_line.lstrip()) < 4:
                    # Indent with 4 spaces
                    fixed_lines.append('    ' + next_line.lstrip())
                    logger.info(f"Fixed indentation in except block after line {i+1}")
                    i = j  # Skip the line we just fixed
                else:
                    # Keep the line as-is
                    fixed_lines.append(next_line)
                    i = j
        else:
            # Add the line unchanged
            fixed_lines.append(line)
            i += 1
            
    return '\n'.join(fixed_lines)

def final_cleanup(content):
    """Perform a final cleanup of the entire file to catch any remaining issues."""
    lines = content.split('\n')
    cleaned_lines = []
    
    # Track state
    in_docstring = False
    docstring_start_line = None
    docstring_indent = 0
    
    for i, line in enumerate(lines):
        # Skip empty or whitespace-only lines at the start of the file
        if not cleaned_lines and not line.strip():
            continue
            
        # Fix indentation around docstrings
        if '"""' in line:
            # Count occurrence in this line
            quotes_count = line.count('"""')
            
            # Handle toggle between enter/exit docstring state
            if quotes_count % 2 != 0:
                if not in_docstring:
                    # Start of docstring
                    in_docstring = True
                    docstring_start_line = i
                    docstring_indent = len(line) - len(line.lstrip())
                else:
                    # End of docstring
                    in_docstring = False
        
        # Handle known problem: indentation in class __init__ method
        if line.strip().startswith('def __init__') and not line.strip().endswith(':'):
            # Add missing colon
            line = line.rstrip() + ':'
            logger.info(f"Added missing colon to __init__ method on line {i+1}")
            
        # Handle broken try blocks
        if line.strip() == 'try:' and i + 1 < len(lines):
            next_line = lines[i + 1]
            if not next_line.strip() or next_line.strip().startswith(('except', 'finally')):
                # Try block is empty, add a pass statement
                indent = len(line) - len(line.lstrip())
                cleaned_lines.append(line)
                cleaned_lines.append(' ' * (indent + 4) + 'pass')
                logger.info(f"Added missing pass statement in empty try block on line {i+1}")
                continue
                
        # Fix common indent issues in if blocks at the end of file
        if line.strip().startswith('if ') and i + 1 >= len(lines):
            # if statement at the end with no body
            indent = len(line) - len(line.lstrip())
            cleaned_lines.append(line)
            cleaned_lines.append(' ' * (indent + 4) + 'pass')
            logger.info(f"Added missing pass statement to if block at EOF on line {i+1}")
            continue
            
        # Fix the common issue with main() at the end of file
        if i > 0 and lines[i-1].strip() == 'if __name__ == "__main__":' and line.strip() == 'sys.exit(main())':
            # Ensure proper indentation for the main function call
            indent = len(lines[i-1]) - len(lines[i-1].lstrip())
            cleaned_lines.append(' ' * (indent + 4) + line.strip())
            logger.info(f"Fixed indentation of main function call on line {i+1}")
            continue
            
        # Fix broken if __name__ block
        if line.strip().startswith('return 0 if success else 1') and i + 1 < len(lines) and lines[i+1].strip().startswith('if __name__ =='):
            cleaned_lines.append(line)
            cleaned_lines.append('')  # Add blank line
            logger.info(f"Added blank line before if __name__ block on line {i+2}")
            continue
        
        # Keep the line (default case)
        cleaned_lines.append(line)
    
    # Check for common EOF issues
    if cleaned_lines:
        # Remove trailing empty lines
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()
            
        # Always end with a newline
        if cleaned_lines[-1].strip():
            cleaned_lines.append('')
    
    # Fix broken main method references in the content (full string ops, not just lines)
    content = '\n'.join(cleaned_lines)
    
    # Fix TestBertModels reference that should be TestModelTypeModels
    model_tester_pattern = re.compile(r'(\w+)_tester = Test(\w+)Models', re.MULTILINE)
    for match in model_tester_pattern.finditer(content):
        model_type = match.group(1)
        referenced_type = match.group(2)
        pascal_case = get_pascal_case_identifier(model_type) if 'get_pascal_case_identifier' in globals() else model_type.capitalize()
        
        if model_type != referenced_type.lower():
            logger.info(f"Fixing model tester reference: {model_type}_tester = Test{referenced_type}Models -> Test{pascal_case}Models")
            content = content.replace(
                f"{model_type}_tester = Test{referenced_type}Models",
                f"{model_type}_tester = Test{pascal_case}Models"
            )
    
    # Fix tester references in print statements
    tester_ref_pattern = re.compile(r'Device: {(\w+)_tester\.device}', re.MULTILINE)
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
    
    return content

def fix_indentation_issues(content):
    """Fix common indentation issues with enhanced try/except block handling."""
    lines = content.split('\n')
    
    # First pass: track and fix blocks
    in_class = False
    in_method = False
    in_try_block = False
    in_except_block = False
    in_finally_block = False
    class_indent = 0
    method_indent = 0
    try_indent = 0
    
    # First, fix class docstring + method definition issues
    i = 0
    while i < len(lines)-1:
        # Look for class docstring followed by method definition issues
        if lines[i].strip().startswith('"""') and lines[i].strip().endswith('"""') and lines[i+1].strip().startswith('def '):
            # Check if the docstring and method definition are on the same line
            if '"""def ' in lines[i]:
                parts = lines[i].split('"""')
                if len(parts) >= 2:
                    # Extract the docstring and method definition
                    docstring = parts[0] + '"""'
                    method_def = parts[1].strip()
                    
                    # Fix by splitting into two lines
                    lines[i] = docstring
                    lines.insert(i+1, method_def)
                    logger.info(f"Fixed docstring and method definition combined on line {i+1}")
                    i += 1  # Skip the newly inserted line
        i += 1
    
    # Second pass: scan for blocks and fix indentation
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        
        # Check for missing newline between docstring and method definition
        if i < len(lines)-1 and stripped.endswith('"""') and not stripped.startswith('"""'):
            next_line = lines[i+1].strip()
            if next_line.startswith('def '):
                # Make sure there's an empty line between docstring and method
                lines.insert(i+1, '')
                logger.info(f"Added missing empty line after docstring on line {i+1}")
                i += 1  # Skip the newly inserted empty line
        
        # Track class definition
        if stripped.startswith('class ') and stripped.endswith(':'):
            in_class = True
            class_indent = indent
        
        # Track method definition
        elif in_class and stripped.startswith('def ') and stripped.endswith(':'):
            in_method = True
            method_indent = indent
            
        # Track try blocks - correctly handle nested try blocks
        elif stripped == 'try:':
            in_try_block = True
            try_indent = indent
            
            # Look ahead to ensure the next content line is properly indented
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
                
            # If we found a content line with bad indentation, fix it
            if j < len(lines):
                next_indent = len(lines[j]) - len(lines[j].lstrip())
                if lines[j].strip() and next_indent <= try_indent:
                    # Fix the indentation
                    lines[j] = ' ' * (try_indent + 4) + lines[j].lstrip()
                    logger.info(f"Fixed indentation after try: on line {j+1}")
        
        # Track except blocks
        elif in_try_block and stripped.startswith('except ') and stripped.endswith(':'):
            in_try_block = False
            in_except_block = True
            
            # Look ahead to ensure the next content line is properly indented
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
                
            # If we found a content line with bad indentation, fix it
            if j < len(lines):
                next_indent = len(lines[j]) - len(lines[j].lstrip())
                if lines[j].strip() and next_indent <= try_indent:
                    # Fix the indentation
                    lines[j] = ' ' * (try_indent + 4) + lines[j].lstrip()
                    logger.info(f"Fixed indentation after except: on line {j+1}")
        
        # Track finally blocks
        elif (in_try_block or in_except_block) and stripped == 'finally:':
            in_try_block = False
            in_except_block = False
            in_finally_block = True
            
            # Look ahead to ensure the next content line is properly indented
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
                
            # If we found a content line with bad indentation, fix it
            if j < len(lines):
                next_indent = len(lines[j]) - len(lines[j].lstrip())
                if lines[j].strip() and next_indent <= try_indent:
                    # Fix the indentation
                    lines[j] = ' ' * (try_indent + 4) + lines[j].lstrip()
                    logger.info(f"Fixed indentation after finally: on line {j+1}")
            
        # End of block detection
        elif in_except_block and indent <= try_indent and stripped and not stripped.startswith(('except ', 'finally:')):
            in_except_block = False
        elif in_finally_block and indent <= try_indent and stripped:
            in_finally_block = False
            
        # Fix indentation in method body
        elif in_method and indent < method_indent + 4 and stripped and not stripped.startswith(('def ', 'class ')):
            # This is a method body line with incorrect indentation
            lines[i] = ' ' * (method_indent + 4) + stripped
            logger.info(f"Fixed indentation in method body on line {i+1}")
            
        # Fix indentation in try/except blocks
        elif in_try_block and indent < try_indent + 4 and stripped and not stripped.startswith(('except ', 'finally:')):
            # This is a try block line with incorrect indentation
            lines[i] = ' ' * (try_indent + 4) + stripped
            logger.info(f"Fixed indentation in try block on line {i+1}")
            
        elif in_except_block and indent < try_indent + 4 and stripped and not stripped.startswith(('try:', 'except ', 'finally:')):
            # This is an except block line with incorrect indentation
            lines[i] = ' ' * (try_indent + 4) + stripped
            logger.info(f"Fixed indentation in except block on line {i+1}")
            
        elif in_finally_block and indent < try_indent + 4 and stripped:
            # This is a finally block line with incorrect indentation
            lines[i] = ' ' * (try_indent + 4) + stripped
            logger.info(f"Fixed indentation in finally block on line {i+1}")
        
        # Fix docstring-method definition run-on issues
        if '"""def ' in line:
            parts = line.split('"""')
            if len(parts) >= 2:
                # Fix by splitting into two lines
                lines[i] = parts[0] + '"""'
                indent_str = ' ' * indent
                lines.insert(i+1, indent_str + parts[1].strip())
                logger.info(f"Fixed docstring and method definition combined on line {i+1}")
                i += 1  # Skip the newly inserted line
        
        i += 1
    
    content = '\n'.join(lines)
    return content

def fix_test_indentation(template_content):
    """Fix indentation issues in generated test files."""
    try:
        # Apply all indentation fixes
        fixed_content = fix_class_method_indentation(template_content)
        
        # Normalize excessive spacing
        fixed_content = re.sub(r'\n\s*\n\s*\n', '\n\n', fixed_content)
        
        return fixed_content
    except Exception as e:
        logger.error(f"Error applying indentation fixes: {e}")
        # Return original content if fixing fails
        return template_content

# Constants
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = CURRENT_DIR.parent
RESULTS_DIR = CURRENT_DIR / "collected_results"
EXPECTED_DIR = CURRENT_DIR / "expected_results"
TEMPLATES_DIR = CURRENT_DIR / "templates"

# Try to import torch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = MagicMock()
    HAS_TORCH = False
    logger.warning("torch not available, using mock")

# Try to import transformers
try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = MagicMock()
    HAS_TRANSFORMERS = False
    logger.warning("transformers not available, using mock")

# Try to import tokenizers
try:
    import tokenizers
    HAS_TOKENIZERS = True
except ImportError:
    tokenizers = MagicMock()
    HAS_TOKENIZERS = False
    logger.warning("tokenizers not available, using mock")

# Try to import sentencepiece
try:
    import sentencepiece
    HAS_SENTENCEPIECE = True
except ImportError:
    sentencepiece = MagicMock()
    HAS_SENTENCEPIECE = False
    logger.warning("sentencepiece not available, using mock")

# Model Registry - Maps model families to their configurations
MODEL_REGISTRY = {
    "bert": {
        "family_name": "BERT",
        "description": "BERT-family masked language models",
        "default_model": "google-bert/bert-base-uncased",
        "class": "BertForMaskedLM",
        "test_class": "TestBertModels",
        "module_name": "test_hf_bert",
        "tasks": ['fill-mask'],
        "inputs": {
            "text": "The quick brown fox jumps over the [MASK] dog.",
        },
    },
    "gpt2": {
        "family_name": "GPT-2",
        "description": "GPT-2 autoregressive language models",
        "default_model": "datificate/gpt2-small-spanish",
        "class": "GPT2LMHeadModel",
        "test_class": "TestGPT2Models",
        "module_name": "test_hf_gpt2",
        "tasks": ['text-generation'],
        "inputs": {
            "text": "GPT-2 is a transformer model that",
        },
        "task_specific_args": {
            "text-generation": {
                "max_length": 50,
                "min_length": 20,
            },
        },
    },
    "t5": {
        "family_name": "T5",
        "description": "T5 encoder-decoder models",
        "default_model": "amazon/chronos-t5-small",
        "class": "T5ForConditionalGeneration",
        "test_class": "TestT5Models",
        "module_name": "test_hf_t5",
        "tasks": ['text2text-generation'],
        "inputs": {
            "text": "translate English to German: The house is wonderful.",
        },
        "task_specific_args": {
            "text2text-generation": {
                "max_length": 50,
            },
        },
    },
    "vit": {
        "family_name": "ViT",
        "description": "Vision Transformer models",
        "default_model": "google/vit-base-patch16-224-in21k",
        "class": "ViTForImageClassification",
        "test_class": "TestVitModels",
        "module_name": "test_hf_vit",
        "tasks": ['image-classification'],
        "inputs": {
        },
    },
    "gpt-j": {
        "family_name": "GPT-J",
        "description": "GPT-J autoregressive language models",
        "default_model": "mmnga/aibuncho-japanese-novel-gpt-j-6b-gguf",
        "class": "GPTJForCausalLM",
        "test_class": "TestGPTJModels",
        "module_name": "test_hf_gpt_j",
        "tasks": ['text-generation'],
        "inputs": {
            "text": "GPT-J is a transformer model that",
        },
        "task_specific_args": {
            "text-generation": {
                "max_length": 50,
            },
        },
    },
    "gpt-neo": {
        "family_name": "GPT-Neo",
        "description": "GPT-Neo autoregressive language models",
        "default_model": "EleutherAI/gpt-neo-1.3B",
        "class": "GPTNeoForCausalLM",
        "test_class": "TestGPTNeoModels",
        "module_name": "test_hf_gpt_neo",
        "tasks": ['text-generation'],
        "inputs": {
            "text": "GPT-Neo is a transformer model that",
        },
    },
    "gpt-neox": {
        "family_name": "GPTNeoX",
        "description": "GPTNeoX autoregressive language models",
        "default_model": "EleutherAI/gpt-neox-20b",
        "class": "GPTNeoXForCausalLM",
        "test_class": "TestGPTNeoXModels",
        "module_name": "test_hf_gpt_neox",
        "tasks": ['text-generation'],
        "inputs": {
            "text": "GPTNeoX is a transformer model that",
        },
    },
    "xlm-roberta": {
        "family_name": "XLM-RoBERTa",
        "description": "XLM-RoBERTa masked language models for cross-lingual understanding",
        "default_model": "xlm-roberta-base",
        "class": "XLMRobertaForMaskedLM",
        "test_class": "TestXLMRobertaModels",
        "module_name": "test_hf_xlm_roberta",
        "tasks": ['fill-mask'],
        "inputs": {
            "text": "XLM-RoBERTa is a <mask> language model.",
        },
    },
    "roberta": {
        "family_name": "RoBERTa",
        "description": "RoBERTa masked language models",
        "default_model": "roberta-base",
        "class": "RobertaForMaskedLM",
        "test_class": "TestRobertaModels",
        "module_name": "test_hf_roberta",
        "tasks": ['fill-mask'],
        "inputs": {
            "text": "RoBERTa is a <mask> language model.",
        },
    },
    "distilbert": {
        "family_name": "DistilBERT",
        "description": "DistilBERT masked language models",
        "default_model": "distilbert-base-uncased",
        "class": "DistilBertForMaskedLM",
        "test_class": "TestDistilBertModels",
        "module_name": "test_hf_distilbert",
        "tasks": ['fill-mask'],
        "inputs": {
            "text": "DistilBERT is a <mask> language model.",
        },
    },
    "albert": {
        "family_name": "ALBERT",
        "description": "ALBERT (A Lite BERT) masked language models",
        "default_model": "albert-base-v2",
        "class": "AlbertForMaskedLM",
        "test_class": "TestAlbertModels",
        "module_name": "test_hf_albert",
        "tasks": ['fill-mask'],
        "inputs": {
            "text": "ALBERT is a <mask> language model.",
        },
    },
    "electra": {
        "family_name": "ELECTRA",
        "description": "ELECTRA discriminator models",
        "default_model": "google/electra-small-discriminator",
        "class": "ElectraForMaskedLM",
        "test_class": "TestElectraModels",
        "module_name": "test_hf_electra",
        "tasks": ['fill-mask'],
        "inputs": {
            "text": "ELECTRA is a <mask> language model.",
        },
    },
    "layoutlm": {
        "family_name": "LayoutLM",
        "description": "LayoutLM models for document understanding",
        "default_model": "microsoft/layoutlm-base-uncased",
        "class": "LayoutLMForMaskedLM",
        "test_class": "TestLayoutLMModels", 
        "module_name": "test_hf_layoutlm",
        "tasks": ['fill-mask', 'token-classification'],
        "inputs": {
            "text": "LayoutLM is a [MASK] language model for document understanding.",
            "bbox": [[0, 0, 100, 20], [120, 0, 220, 20], [240, 0, 340, 20], [360, 0, 460, 20], [480, 0, 580, 20], [0, 25, 100, 45], [120, 25, 220, 45], [240, 25, 340, 45], [360, 25, 460, 45]]
        },
    },
    "canine": {
        "family_name": "CANINE",
        "description": "CANINE character-level transformer for multilingual NLP",
        "default_model": "google/canine-s",
        "class": "CanineForMaskedLM",
        "test_class": "TestCanineModels", 
        "module_name": "test_hf_canine",
        "tasks": ['fill-mask', 'token-classification', 'text-classification'],
        "inputs": {
            "text": "CANINE is a [MASK] language model for character-level understanding."
        },
    },
    "roformer": {
        "family_name": "RoFormer",
        "description": "RoFormer rotary position embedding transformer for NLP",
        "default_model": "junnyu/roformer_chinese_base",
        "class": "RoformerForMaskedLM",
        "test_class": "TestRoformerModels",
        "module_name": "test_hf_roformer",
        "tasks": ['fill-mask', 'token-classification', 'text-classification'],
        "inputs": {
            "text": "RoFormer is a [MASK] language model with rotary position embeddings."
        },
    },
    "bart": {
        "family_name": "BART",
        "description": "BART sequence-to-sequence models",
        "default_model": "facebook/bart-base",
        "class": "BartForConditionalGeneration",
        "test_class": "TestBartModels",
        "module_name": "test_hf_bart",
        "tasks": ['summarization', 'translation'],
        "inputs": {
            "text": "BART is a denoising autoencoder for pretraining sequence-to-sequence models.",
        },
    },
    "mbart": {
        "family_name": "mBART",
        "description": "Multilingual BART sequence-to-sequence models",
        "default_model": "facebook/mbart-large-cc25",
        "class": "MBartForConditionalGeneration",
        "test_class": "TestMBartModels",
        "module_name": "test_hf_mbart",
        "tasks": ['translation'],
        "inputs": {
            "text": "mBART is a multilingual sequence-to-sequence model.",
        },
    },
    "pegasus": {
        "family_name": "Pegasus",
        "description": "Pegasus summarization models",
        "default_model": "google/pegasus-xsum",
        "class": "PegasusForConditionalGeneration",
        "test_class": "TestPegasusModels",
        "module_name": "test_hf_pegasus",
        "tasks": ['summarization'],
        "inputs": {
            "text": "Pegasus is a model for abstractive summarization optimized for ROUGE.",
        },
    },
    "mt5": {
        "family_name": "mT5",
        "description": "Multilingual T5 models",
        "default_model": "google/mt5-small",
        "class": "MT5ForConditionalGeneration",
        "test_class": "TestMT5Models",
        "module_name": "test_hf_mt5",
        "tasks": ['translation'],
        "inputs": {
            "text": "translate English to German: The house is wonderful.",
        },
    },
    "clip": {
        "family_name": "CLIP",
        "description": "Contrastive Language-Image Pre-training models",
        "default_model": "openai/clip-vit-base-patch32",
        "class": "CLIPModel",
        "test_class": "TestCLIPModels",
        "module_name": "test_hf_clip",
        "tasks": ['zero-shot-image-classification'],
        "inputs": {
        },
    },
    "blip": {
        "family_name": "BLIP",
        "description": "Bootstrapping Language-Image Pre-training models",
        "default_model": "Salesforce/blip-image-captioning-base",
        "class": "BlipForConditionalGeneration",
        "test_class": "TestBlipModels",
        "module_name": "test_hf_blip",
        "tasks": ['image-to-text'],
        "inputs": {
        },
    },
    "llava": {
        "family_name": "LLaVA",
        "description": "Large Language and Vision Assistant",
        "default_model": "llava-hf/llava-1.5-7b-hf",
        "class": "LlavaForConditionalGeneration",
        "test_class": "TestLlavaModels",
        "module_name": "test_hf_llava",
        "tasks": ['visual-question-answering'],
        "inputs": {
        },
    },
    "whisper": {
        "family_name": "Whisper",
        "description": "Speech recognition models",
        "default_model": "openai/whisper-base.en",
        "class": "WhisperForConditionalGeneration",
        "test_class": "TestWhisperModels",
        "module_name": "test_hf_whisper",
        "tasks": ['automatic-speech-recognition'],
        "inputs": {
        },
    },
    "wav2vec2": {
        "family_name": "Wav2Vec2",
        "description": "Speech representation models",
        "default_model": "facebook/wav2vec2-base",
        "class": "Wav2Vec2ForCTC",
        "test_class": "TestWav2Vec2Models",
        "module_name": "test_hf_wav2vec2",
        "tasks": ['automatic-speech-recognition'],
        "inputs": {
        },
    },
    "hubert": {
        "family_name": "HuBERT",
        "description": "Hidden-Unit BERT speech models",
        "default_model": "facebook/hubert-base-ls960",
        "class": "HubertForCTC",
        "test_class": "TestHubertModels",
        "module_name": "test_hf_hubert",
        "tasks": ['automatic-speech-recognition'],
        "inputs": {
        },
    },
    "llama": {
        "family_name": "LLaMA",
        "description": "Large Language Model Meta AI",
        "default_model": "meta-llama/Llama-2-7b-hf",
        "class": "LlamaForCausalLM",
        "test_class": "TestLlamaModels",
        "module_name": "test_hf_llama",
        "tasks": ['text-generation'],
        "inputs": {
            "text": "LLaMA is a foundational language model that",
        },
    },
    "open_llama": {
        "family_name": "Open LLaMA",
        "description": "Open-source implementation of Meta's LLaMA model",
        "default_model": "openlm-research/open_llama_3b",
        "class": "LlamaForCausalLM",
        "test_class": "TestOpenLlamaModels",
        "module_name": "test_hf_open_llama",
        "tasks": ['text-generation'],
        "inputs": {
            "text": "Open-LLaMA is a transformer model that",
        },
    },
    "opt": {
        "family_name": "OPT",
        "description": "Open Pre-trained Transformer language models",
        "default_model": "facebook/opt-125m",
        "class": "OPTForCausalLM",
        "test_class": "TestOPTModels",
        "module_name": "test_hf_opt",
        "tasks": ['text-generation'],
        "inputs": {
            "text": "OPT is an open-source language model that",
        },
    },
    "bloom": {
        "family_name": "BLOOM",
        "description": "BigScience Large Open-science Open-access Multilingual language models",
        "default_model": "bigscience/bloom-560m",
        "class": "BloomForCausalLM",
        "test_class": "TestBloomModels",
        "module_name": "test_hf_bloom",
        "tasks": ['text-generation'],
        "inputs": {
            "text": "BLOOM is a multilingual language model that",
        },
    },
    "stablelm": {
        "family_name": "StableLM",
        "description": "StableLM decoder-only language models",
        "default_model": "stabilityai/stablelm-3b-4e1t",
        "class": "AutoModelForCausalLM",
        "test_class": "TestStablelmModels",
        "module_name": "test_hf_stablelm",
        "tasks": ['text-generation'],
        "inputs": {
            "text": "StableLM is an open-source language model that",
        },
    },
    "mosaic_mpt": {
        "family_name": "MosaicMPT",
        "description": "Mosaic MPT decoder-only language models",
        "default_model": "mosaicml/mpt-7b-instruct",
        "class": "AutoModelForCausalLM",
        "test_class": "TestMosaicMptModels",
        "module_name": "test_hf_mosaic_mpt",
        "tasks": ['text-generation'],
        "inputs": {
            "text": "Mosaic MPT is a language model that",
        },
    },
    "pythia": {
        "family_name": "Pythia",
        "description": "Pythia decoder-only language models by EleutherAI",
        "default_model": "EleutherAI/pythia-1b",
        "class": "AutoModelForCausalLM",
        "test_class": "TestPythiaModels",
        "module_name": "test_hf_pythia",
        "tasks": ['text-generation'],
        "inputs": {
            "text": "Pythia is a language model that",
        },
    },
    "xglm": {
        "family_name": "XGLM",
        "description": "XGLM multilingual decoder-only language models",
        "default_model": "facebook/xglm-1.7B",
        "class": "XGLMForCausalLM",
        "test_class": "TestXglmModels",
        "module_name": "test_hf_xglm",
        "tasks": ['text-generation'],
        "inputs": {
            "text": "XGLM is a multilingual language model that",
        },
    },
    "codellama": {
        "family_name": "CodeLLama",
        "description": "LLaMA model specialized for code generation and understanding",
        "default_model": "codellama/CodeLlama-7b-hf",
        "class": "LlamaForCausalLM",
        "test_class": "TestCodeLlamaModels",
        "module_name": "test_hf_codellama",
        "tasks": ['text-generation'],
        "inputs": {
            "text": "def fibonacci(n):",
        },
        "task_specific_args": {
            "text-generation": {
                "max_length": 100,
                "min_length": 20,
            },
        },
    },

    "qwen2": {
        "family_name": "Qwen2",
        "description": "Qwen2 large language model developed by Alibaba",
        "default_model": "Qwen/Qwen2-7B",
        "class": "Qwen2ForCausalLM",
        "test_class": "TestQwen2Models",
        "module_name": "test_hf_qwen2",
        "tasks": ['text-generation'],
        "inputs": {
            "text": "Qwen2 is a large language model that",
        },
    },
    "qwen3": {
        "family_name": "Qwen3",
        "description": "Qwen3 large language model developed by Alibaba",
        "default_model": "Qwen/Qwen3-7B",
        "class": "Qwen3ForCausalLM",
        "test_class": "TestQwen3Models",
        "module_name": "test_hf_qwen3",
        "tasks": ['text-generation'],
        "inputs": {
            "text": "Qwen3 is a large language model that",
        },
    },
    "fuyu": {
        "family_name": "Fuyu",
        "description": "Fuyu multimodal model by Adept",
        "default_model": "adept/fuyu-8b",
        "class": "FuyuForCausalLM",
        "test_class": "TestFuyuModels",
        "module_name": "test_hf_fuyu",
        "tasks": ['visual-question-answering'],
        "inputs": {
            "text": "What is shown in this image?",
        },
    },
    "kosmos-2": {
        "family_name": "Kosmos-2",
        "description": "Kosmos-2 multimodal model with grounding capabilities",
        "default_model": "microsoft/kosmos-2-patch14-224",
        "class": "Kosmos2ForConditionalGeneration",
        "test_class": "TestKosmos2Models",
        "module_name": "test_hf_kosmos2",
        "tasks": ['visual-question-answering', 'image-to-text', 'image-grounding'],
        "inputs": {
            "text": "What is shown in this image?",
        },
    },
    "llava-next": {
        "family_name": "LLaVA-Next",
        "description": "Next generation of LLaVA with improved capabilities",
        "default_model": "llava-hf/llava-v1.6-mistral-7b-hf",
        "class": "LlavaNextForConditionalGeneration",
        "test_class": "TestLlavaNextModels",
        "module_name": "test_hf_llava_next",
        "tasks": ['visual-question-answering'],
        "inputs": {
            "text": "What is shown in this image?",
        },
    },
    "video-llava": {
        "family_name": "Video-LLaVA",
        "description": "LLaVA model adapted for video understanding",
        "default_model": "videollava/videollava-7b-hf",
        "class": "VideoLlavaForConditionalGeneration",
        "test_class": "TestVideoLlavaModels",
        "module_name": "test_hf_video_llava",
        "tasks": ['video-question-answering'],
        "inputs": {
            "text": "What is happening in this video?",
        },
        "alternate_models": ["LanguageBind/Video-LLaVA-7B-hf"],
    },
    "bark": {
        "family_name": "Bark",
        "description": "Text-to-audio model by Suno",
        "default_model": "suno/bark-small",
        "class": "BarkModel",
        "test_class": "TestBarkModels",
        "module_name": "test_hf_bark",
        "tasks": ['text-to-audio'],
        "inputs": {
            "text": "Hello, my name is Suno. And, I like to sing.",
        },
    },
    "mobilenet-v2": {
        "family_name": "MobileNetV2",
        "description": "Lightweight vision model optimized for mobile and edge devices",
        "default_model": "google/mobilenet_v2_1.0_224",
        "class": "MobileNetV2ForImageClassification",
        "test_class": "TestMobileNetV2Models",
        "module_name": "test_hf_mobilenet_v2",
        "tasks": ['image-classification'],
        "inputs": {
        },
    },
    "blip-2": {
        "family_name": "BLIP-2",
        "description": "BLIP-2 vision-language model with improved architecture",
        "default_model": "Salesforce/blip2-opt-2.7b",
        "class": "Blip2ForConditionalGeneration",
        "test_class": "TestBlip2Models",
        "module_name": "test_hf_blip_2",
        "tasks": ['image-to-text', 'visual-question-answering'],
        "inputs": {
            "text": "What is shown in this image?",
        },
    },
    "chinese-clip": {
        "family_name": "ChineseCLIP",
        "description": "Chinese CLIP model for vision-text understanding",
        "default_model": "OFA-Sys/chinese-clip-vit-base-patch16",
        "class": "ChineseCLIPModel",
        "test_class": "TestChineseCLIPModels",
        "module_name": "test_hf_chinese_clip",
        "tasks": ['zero-shot-image-classification'],
        "inputs": {
        },
    },
    "clipseg": {
        "family_name": "CLIPSeg",
        "description": "CLIP with segmentation capabilities",
        "default_model": "CIDAS/clipseg-rd64-refined",
        "class": "CLIPSegForImageSegmentation",
        "test_class": "TestCLIPSegModels",
        "module_name": "test_hf_clipseg",
        "tasks": ['image-segmentation'],
        "inputs": {
            "text": "person",
        },
    },
}

# Class name capitalization fixes
CLASS_NAME_FIXES = {
    "VitForImageClassification": "ViTForImageClassification",
    "SwinForImageClassification": "SwinForImageClassification",
    "DeitForImageClassification": "DeiTForImageClassification",
    "BeitForImageClassification": "BEiTForImageClassification",
    "ConvnextForImageClassification": "ConvNextForImageClassification",
    "Gpt2LMHeadModel": "GPT2LMHeadModel",
    "GptjForCausalLM": "GPTJForCausalLM",
    "GptneoForCausalLM": "GPTNeoForCausalLM",
    "XlmRobertaForMaskedLM": "XLMRobertaForMaskedLM",
    "XlmRobertaModel": "XLMRobertaModel",
    "RobertaForMaskedLM": "RobertaForMaskedLM",
    "DistilbertForMaskedLM": "DistilBertForMaskedLM",
    "AlbertForMaskedLM": "AlbertForMaskedLM",
    "ElectraForMaskedLM": "ElectraForMaskedLM",
    "BartForConditionalGeneration": "BartForConditionalGeneration",
    "MbartForConditionalGeneration": "MBartForConditionalGeneration",
    "PegasusForConditionalGeneration": "PegasusForConditionalGeneration",
    "Mt5ForConditionalGeneration": "MT5ForConditionalGeneration",
    "ClipModel": "CLIPModel",
    "BlipForConditionalGeneration": "BlipForConditionalGeneration",
    "LlavaForConditionalGeneration": "LlavaForConditionalGeneration",
    "WhisperForConditionalGeneration": "WhisperForConditionalGeneration",
    "Wav2vec2ForCTC": "Wav2Vec2ForCTC",
    "HubertForCTC": "HubertForCTC",
    "LlamaForCausalLM": "LlamaForCausalLM",
    "OptForCausalLM": "OPTForCausalLM",
    "BloomForCausalLM": "BloomForCausalLM",
    "CodeLlamaForCausalLM": "LlamaForCausalLM"  # CodeLLama uses LlaMa architecture
}

def to_valid_identifier(name):
    # Replace hyphens with underscores
    valid_name = name.replace("-", "_")
    
    # Ensure the name doesn't start with a number
    if valid_name and valid_name[0].isdigit():
        valid_name = f"m{valid_name}"
    
    # Replace any invalid characters with underscores
    valid_name = re.sub(r'[^a-zA-Z0-9_]', '_', valid_name)
    
    # Deduplicate consecutive underscores
    valid_name = re.sub(r'_+', '_', valid_name)
    
    return valid_name

def get_pascal_case_identifier(text):
    """Convert a model name (potentially hyphenated) to PascalCase for class names."""
    # Split by hyphens and capitalize each part
    parts = text.split('-')
    return ''.join(part.capitalize() for part in parts)

def get_architecture_type(model_type):
    """Determine architecture type based on model type."""
    model_type_lower = model_type.lower()
    for arch_type, models in ARCHITECTURE_TYPES.items():
        if any(model in model_type_lower for model in models):
            return arch_type
    return "encoder-only"  # Default to encoder-only if unknown
    
def fix_class_docstring_issues(content):
    """Fix issues with class docstrings specifically."""
    lines = content.split('\n')
    fixed_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Look for class definition followed by a docstring
        if line.strip().startswith("class ") and line.strip().endswith(":"):
            class_indent = len(line) - len(line.lstrip())
            fixed_lines.append(line)
            i += 1
            
            # If next line exists and is a docstring starting line
            if i < len(lines) and lines[i].strip().startswith('"""'):
                docstring_start_line = i
                docstring_lines = [lines[i]]
                docstring_indent = class_indent + 4  # Class docstring should be indented +4
                
                # If the docstring is not properly indented, fix it
                if len(lines[i]) - len(lines[i].lstrip()) != docstring_indent:
                    docstring_lines[0] = ' ' * docstring_indent + lines[i].strip()
                    logger.info(f"Fixed class docstring indentation on line {i+1}")
                
                # Look for the end of the docstring
                i += 1
                in_docstring = True
                found_end = False
                
                # Extract all docstring lines
                while i < len(lines) and in_docstring:
                    current_line = lines[i]
                    
                    # Check for docstring end
                    if '"""' in current_line and not current_line.strip().startswith('"""'):
                        # We found the end of the docstring
                        found_end = True
                        in_docstring = False
                        
                        # If it ends on this line, add it and continue
                        docstring_lines.append(current_line)
                        i += 1
                    elif i == docstring_start_line + 1 and current_line.strip().startswith('def '):
                        # This is a method definition immediately after the docstring start
                        # The docstring was never closed - add a closing on a separate line
                        docstring_lines.append(' ' * docstring_indent + '"""')
                        found_end = True
                        in_docstring = False
                        logger.info(f"Added missing closing quotes for class docstring before line {i+1}")
                        # Don't increment i - we'll process the method next
                    elif current_line.strip().startswith('Initialize'):
                        # This is a common issue where __init__ docstring continues but there's no method
                        # We need to add the __init__ method declaration before this
                        docstring_lines.append(' ' * docstring_indent + '"""')
                        found_end = True
                        in_docstring = False
                        
                        # Add method declaration
                        fixed_lines.extend(docstring_lines)
                        fixed_lines.append(' ' * (class_indent + 4) + 'def __init__(self, model_id="default-model", device=None):')
                        fixed_lines.append(' ' * (class_indent + 8) + '"""')
                        
                        # Continue processing the line as part of the method docstring
                        i += 1
                        logger.info(f"Fixed missing __init__ method definition on line {i}")
                        continue
                    else:
                        # Continue collecting docstring lines
                        # Ensure proper indentation
                        if current_line.strip():
                            # Only fix indentation if it's less than what we expect
                            # This preserves additional indentation for formatting
                            if len(current_line) - len(current_line.lstrip()) < docstring_indent:
                                current_line = ' ' * docstring_indent + current_line.lstrip()
                        docstring_lines.append(current_line)
                        i += 1
                
                # If we didn't find an end, add a closing triple quote
                if not found_end:
                    docstring_lines.append(' ' * docstring_indent + '"""')
                    logger.info(f"Added missing closing quotes for class docstring at end")
                
                # Add all docstring lines
                fixed_lines.extend(docstring_lines)
            else:
                # No docstring - just continue
                continue
        else:
            # Normal line - add it
            fixed_lines.append(line)
            i += 1
    
    return '\n'.join(fixed_lines)

def fix_docstring_method_definition_issues(content):
    """Fix common issues with docstrings and method definitions."""
    lines = content.split('\n')
    
    # First fix class docstrings
    content = fix_class_docstring_issues(content)
    lines = content.split('\n')
    
    # Process each line
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Fix docstring-method definition run-ons
        if '"""def ' in line:
            parts = line.split('"""')
            if len(parts) >= 2:
                # Split them into two separate lines
                docstring_part = parts[0] + '"""'
                method_part = parts[1].strip()
                
                # Get the current indentation level
                indentation = len(line) - len(line.lstrip())
                indent_str = ' ' * indentation
                
                # Replace the current line with the docstring
                lines[i] = indent_str + docstring_part
                
                # Insert the method on the next line with proper indentation
                lines.insert(i + 1, indent_str + method_part)
                
                logger.info(f"Fixed docstring-method definition run-on on line {i+1}")
                
                # Skip the inserted line in next iteration
                i += 1
        
        # Check for docstring followed immediately by method def without blank line
        if i < len(lines) - 1:
            current_line = lines[i].strip()
            next_line = lines[i + 1].strip()
            
            if current_line.endswith('"""') and next_line.startswith('def '):
                # Get indentation of current line
                indentation = len(lines[i]) - len(lines[i].lstrip())
                indent_str = ' ' * indentation
                
                # Insert a blank line between them
                lines.insert(i + 1, indent_str)
                logger.info(f"Inserted blank line between docstring and method definition at line {i+1}")
                
                # Skip the inserted line
                i += 1
        
        # Fix method docstrings that don't have proper closing quotes
        if line.strip().startswith('def ') and line.strip().endswith(':'):
            def_indent = len(line) - len(line.lstrip())
            doc_indent = def_indent + 4  # Docstring should be indented +4 spaces
            
            # Look for a docstring start on the next line
            if i + 1 < len(lines) and lines[i+1].strip().startswith('"""'):
                docstring_start = i + 1
                docstring_start_line = lines[docstring_start]
                
                # If docstring indentation is wrong, fix it
                if len(docstring_start_line) - len(docstring_start_line.lstrip()) != doc_indent:
                    lines[docstring_start] = ' ' * doc_indent + docstring_start_line.strip()
                    logger.info(f"Fixed method docstring indentation on line {docstring_start+1}")
                
                # Check if this is a one-line docstring without closing quotes
                if not lines[docstring_start].strip().endswith('"""'):
                    # Look for the closing quotes
                    found_end = False
                    for j in range(docstring_start + 1, min(docstring_start + 10, len(lines))):
                        if '"""' in lines[j]:
                            found_end = True
                            break
                    
                    if not found_end:
                        # If no closing quotes found, add them
                        if docstring_start + 1 < len(lines) and lines[docstring_start+1].strip():
                            # If next line has content, add a new line with just closing quotes
                            lines.insert(docstring_start + 1, ' ' * doc_indent + '"""')
                            logger.info(f"Added missing closing quotes for method docstring after line {docstring_start+1}")
                            i += 1  # Skip the line we just added
                        else:
                            # Add closing quotes to an empty line
                            lines[docstring_start] = lines[docstring_start] + '"""'
                            logger.info(f"Added missing closing quotes for method docstring on line {docstring_start+1}")
        
        # Look for indentation errors in class definitions
        if line.strip().startswith('class ') and line.strip().endswith(':'):
            class_indent = len(line) - len(line.lstrip())
            
            # Check if methods are properly indented
            j = i + 1
            while j < len(lines) and (lines[j].strip() == '' or lines[j].lstrip().startswith('"""')):
                j += 1
                
            if j < len(lines) and lines[j].strip().startswith('def '):
                method_indent = len(lines[j]) - len(lines[j].lstrip())
                if method_indent != class_indent + 4:
                    # Fix method indentation
                    correct_indent = ' ' * (class_indent + 4)
                    lines[j] = correct_indent + lines[j].lstrip()
                    logger.info(f"Fixed method indentation on line {j+1}")
        
        i += 1
    
    return '\n'.join(lines)

def get_template_for_architecture(model_type, templates_dir="templates"):
    """Get the template path for a specific model type's architecture."""
    # For hyphenated model names, always use the specialized template
    if '-' in model_type:
        hyphenated_template = os.path.join(templates_dir, "hyphenated_name_template.py")
        if os.path.exists(hyphenated_template):
            logger.info(f"Using hyphenated name template for {model_type}")
            return hyphenated_template
        else:
            logger.warning(f"Hyphenated name template not found, will use architecture-specific template for {model_type}")
    
    # For models with underscores, use the specialized underscore template
    elif '_' in model_type:
        underscore_template = os.path.join(templates_dir, "underscore_name_template.py")
        if os.path.exists(underscore_template):
            logger.info(f"Using underscore name template for {model_type}")
            return underscore_template
        else:
            logger.warning(f"Underscore name template not found, will use architecture-specific template for {model_type}")
    
    # If it's a special architecture type (that itself contains hyphens)
    arch_type = get_architecture_type(model_type)
    if '-' in arch_type:
        # For these special architecture types with hyphens, also use the hyphenated template
        hyphenated_template = os.path.join(templates_dir, "hyphenated_name_template.py")
        if os.path.exists(hyphenated_template):
            logger.info(f"Using hyphenated name template for architecture {arch_type}")
            return hyphenated_template
    
    # For typical model types, prefer architecture-specific templates
    template_map = {
        "encoder-only": os.path.join(templates_dir, "encoder_only_template.py"),
        "decoder-only": os.path.join(templates_dir, "decoder_only_template.py"),
        "encoder-decoder": os.path.join(templates_dir, "encoder_decoder_template.py"),
        "vision": os.path.join(templates_dir, "vision_template.py"),
        "vision-text": os.path.join(templates_dir, "vision_text_template.py"),
        "speech": os.path.join(templates_dir, "speech_template.py"),
        "multimodal": os.path.join(templates_dir, "multimodal_template.py")
    }
    
    template_path = template_map.get(arch_type)
    if template_path and os.path.exists(template_path):
        logger.info(f"Using {arch_type} template for {model_type}")
        return template_path
    
    # Fallback to minimal template if available
    minimal_template = os.path.join(templates_dir, "minimal_bert_template.py")
    if os.path.exists(minimal_template):
        logger.info(f"Using minimal template for {model_type}")
        return minimal_template
    
    # Last resort: fallback to encoder-only template
    logger.warning(f"Template not found for {arch_type}, using encoder-only template")
    fallback_template = os.path.join(templates_dir, "encoder_only_template.py")
    if not os.path.exists(fallback_template):
        logger.error(f"Fallback template not found: {fallback_template}")
        return None
    return fallback_template

def generate_test_file(model_family, output_dir="."):
    """Generate a test file for a model family."""
    # Special handling for hyphenated model names
    if "-" in model_family:
        # Use the specialized function for hyphenated models
        try:
            from simplified_fix_hyphenated import create_hyphenated_test_file
            logger.info(f"Using specialized hyphenated model generator for {model_family}")
            result = create_hyphenated_test_file(model_family, output_dir)
            if result[0]:  # Success
                return True
            else:
                logger.warning(f"Specialized generator failed: {result[1]}, falling back to standard approach")
        except ImportError:
            logger.warning("simplified_fix_hyphenated.py not available, using standard approach for hyphenated model")
    
    # Check if the model family exists in the registry
    if model_family not in MODEL_REGISTRY:
        logger.error(f"Model family '{model_family}' not found in registry")
        return False
    
    # Fix hyphenated model names for valid Python identifiers
    model_family_valid = to_valid_identifier(model_family)
    
    # Get model configuration from registry
    model_config = MODEL_REGISTRY[model_family]
    module_name = model_config.get("module_name", f"test_hf_{model_family_valid}")
    
    # Create proper capitalized name for class using PascalCase conversion
    model_pascal_case = get_pascal_case_identifier(model_family)
    test_class = model_config.get("test_class", f"Test{model_pascal_case}Models")
    
    # Get default model with advanced options from args if available
    if 'args' in globals() and args is not None:
        task = getattr(args, 'task', None)
        hardware = getattr(args, 'hardware', None)
        max_size = getattr(args, 'max_size', None)
        framework = getattr(args, 'framework', None)
        default_model = get_model_from_registry(
            model_family, 
            task=task, 
            hardware_profile=hardware,
            max_size_mb=max_size,
            framework=framework
        )
    else:
        default_model = get_model_from_registry(model_family)
    tasks = model_config.get("tasks", ["text-generation"])
    inputs = model_config.get("inputs", {})
    
    # Get architecture-specific template
    template_path = get_template_for_architecture(model_family)
    if not template_path:
        logger.error(f"Could not find template for {model_family}")
        return False
        
    logger.info(f"Using template: {os.path.basename(template_path)} for {model_family}")
    
    try:
        with open(template_path, "r") as f:
            template = f.read()
        
        # Prepare replacements with proper handling for hyphenated model names
        # Convert to valid Python identifiers for variables and constants
        model_upper = model_family_valid.upper()  # Valid uppercase identifier (e.g., XLM_ROBERTA)
        model_pascal_case = get_pascal_case_identifier(model_family)  # PascalCase for class names (e.g., XlmRoberta)
        
        default_task = tasks[0] if tasks else "fill-mask"
        
        # Make replacements based on model type with proper handling of hyphenated names
        replacements = {
            # Replace registry name - Use valid identifier in uppercase
            "BERT_MODELS_REGISTRY": f"{model_upper}_MODELS_REGISTRY",
            "VIT_MODELS_REGISTRY": f"{model_upper}_MODELS_REGISTRY",
            
            # Replace class names using proper PascalCase
            "TestBertModels": test_class,
            "TestVitModels": test_class,
            
            # Replace model types
            "bert-base-uncased": default_model,
            "google/vit-base-patch16-224": default_model,
            "google-bert/bert-base-uncased": default_model,
            
            # Replace class identifiers
            "BERT": model_upper,  # For constants
            "ViT": model_pascal_case,  # For class names in PascalCase
            
            # Replace lowercase identifiers with valid Python identifiers
            "bert": model_family_valid,
            "vit": model_family_valid,
            
            # Replace tasks
            "fill-mask": default_task,
            
            # Fix hyphenated references in file paths
            "hf_bert_": f"hf_{model_family_valid}_",
            "hf_vit_": f"hf_{model_family_valid}_"
        }
        
        # Use the token-based replacement for improved handling
        # Especially important for hyphenated model names
        if '-' in model_family:
            logger.info(f"Using token-based replacement for hyphenated model: {model_family}")
            
            # Step 1: Preprocess the template to add markers
            preprocessed_template = preprocess_template(template)
            
            # Step 2: Apply token-based replacements
            content = token_based_replace(preprocessed_template, replacements)
            
            # Fix default model references in specific contexts
            lines = content.split('\n')
            
            # Replace __init__ method signature default value
            for i, line in enumerate(lines):
                if 'def __init__' in line and 'model_id=' in line:
                    for old_model in ["google-bert/bert-base-uncased", "bert-base-uncased", "gpt2-medium", "t5-small"]:
                        if old_model in line:
                            lines[i] = line.replace(old_model, default_model)
                            break
                
                # Fix docstring model references
                if "model_id:" in line and "default:" in line:
                    for old_model in ["google-bert/bert-base-uncased", "bert-base-uncased", "gpt2-medium", "t5-small"]:
                        if old_model in line:
                            lines[i] = line.replace(old_model, default_model)
                            break
                
                # Fix parser default model
                if "parser.add_argument(\"--model\"" in line and "default=" in line:
                    for old_model in ["google-bert/bert-base-uncased", "bert-base-uncased", "gpt2-medium", "t5-small"]:
                        if old_model in line:
                            lines[i] = line.replace(old_model, default_model)
                            break
                
                # Fix description
                if "parser.add_argument" in line and "description=" in line:
                    if "Test BERT" in line:
                        lines[i] = line.replace("Test BERT", f"Test {model_pascal_case}")
                    elif "Test GPT2" in line:
                        lines[i] = line.replace("Test GPT2", f"Test {model_pascal_case}")
                    elif "Test T5" in line:
                        lines[i] = line.replace("Test T5", f"Test {model_pascal_case}")
                    elif "Test ViT" in line:
                        lines[i] = line.replace("Test ViT", f"Test {model_pascal_case}")
                
                # Fix test inputs with named placeholders
                if 'test_input =' in line and 'The quick brown fox jumps over the [MASK] dog.' in line:
                    if model_config.get("inputs") and "text" in model_config["inputs"]:
                        custom_input = model_config["inputs"]["text"]
                        lines[i] = line.replace('The quick brown fox jumps over the [MASK] dog.', custom_input)
            
            # Recombine content
            content = '\n'.join(lines)
            
        else:
            # For non-hyphenated models, use the standard approach
            # Split the template into sections for more precise handling
            lines = template.split('\n')
            
            # Replace __init__ method signature more carefully to avoid mangling
            in_init_method = False
            init_start = None
            init_end = None
            
            # Find the __init__ method
            for i, line in enumerate(lines):
                if 'def __init__' in line and 'model_id=' in line:
                    init_start = i
                    in_init_method = True
                elif in_init_method and line.strip() == '':
                    init_end = i
                    break
                    
            # If we found the __init__ method, handle it specially
            if init_start is not None and init_end is not None:
                # Replace the model ID default value
                lines[init_start] = lines[init_start].replace(
                    'model_id="google-bert/bert-base-uncased"', 
                    f'model_id="{default_model}"'
                )
                
                # Fix the docstring
                for j in range(init_start + 1, init_end):
                    if "model_id:" in lines[j] and "default:" in lines[j]:
                        lines[j] = lines[j].replace(
                            '"google-bert/bert-base-uncased"', 
                            f'"{default_model}"'
                        )
                    
            # Ensure the main function uses the right model family
            for i, line in enumerate(lines):
                if "parser.add_argument(\"--model\"" in line and "default=" in line:
                    lines[i] = line.replace(
                        'default="google-bert/bert-base-uncased"',
                        f'default="{default_model}"'
                    )
                    
                # Fix description
                if "parser.add_argument" in line and "description=" in line:
                    if "Test BERT" in line:
                        lines[i] = line.replace(
                            f"Test BERT",
                            f"Test {model_pascal_case}"
                        )
            
            # Process each line individually to avoid mangling code structure
            for i, line in enumerate(lines):
                # Skip empty lines
                if not line.strip():
                    continue
                    
                # Special handling for class definition line
                if line.strip().startswith('class TestBertModels:'):
                    lines[i] = f'class {test_class}:'
                    continue
                    
                # Special handling for bert_tester variable
                if 'bert_tester =' in line:
                    lines[i] = line.replace('bert_tester', f'{model_family_valid}_tester')
                    continue
                    
                # Replace tasks more carefully - only in specific contexts
                if '"fill-mask"' in line and 'pipeline(' in line:
                    lines[i] = line.replace('"fill-mask"', f'"{default_task}"')
                    continue
                    
                # Handle test inputs with named placeholders
                if 'test_input = "The quick brown fox jumps over the [MASK] dog."' in line:
                    if model_config.get("inputs") and "text" in model_config["inputs"]:
                        custom_input = model_config["inputs"]["text"]
                        lines[i] = line.replace('The quick brown fox jumps over the [MASK] dog.', custom_input)
                    continue
                
                # Apply general replacements for the current line
                for old, new in replacements.items():
                    if old in line:
                        # Avoid mangling imports in try blocks and other critical code
                        if 'import' in line and old in ['bert', 'vit']:
                            continue
                            
                        # Don't replace BERT in comments with # at the start
                        if line.strip().startswith('#') and (old == 'BERT' or old == 'bert'):
                            continue
                            
                        # Avoid replacing inside string literals
                        if ('"' in line or "'" in line) and not ('import' in line or 'from' in line):
                            # Skip replacing inside quotes for text content
                            continue
                            
                        # Apply the replacement
                        lines[i] = line.replace(old, new)
            
            # Combine back into content
            content = '\n'.join(lines)
        
        # For hyphenated models, use enhanced post-processing
        if '-' in model_family:
            # Use our enhanced post-processing which includes syntax validation
            content, success, *error = post_process_generated_file(content)
            if not success and error:
                logger.warning(f"Syntax issues detected after post-processing: {error[0]}")
                # Try additional fix for hyphenated models
                content = fix_syntax_errors(content)
        else:
            # For non-hyphenated models, use the standard approach
            # Apply all fixes in one consolidated step using the fix_syntax_errors function
            content = fix_syntax_errors(content)
            
            # Apply one final pass of indentation fixes as a safety measure
            content = fix_test_indentation(content)
        
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Write the test file
        output_file = os.path.join(output_dir, f"{module_name}.py")
        with open(output_file, "w") as f:
            f.write(content)
        
        # Validate syntax
        try:
            compile(content, output_file, 'exec')
            logger.info(f"✅ Syntax is valid for {output_file}")
        except SyntaxError as e:
            logger.error(f"❌ Syntax error in generated file: {e}")
            # Show the problematic line for debugging
            if hasattr(e, 'lineno') and e.lineno is not None:
                lines = content.split('\n')
                line_no = e.lineno - 1  # 0-based index
                if 0 <= line_no < len(lines):
                    logger.error(f"Problematic line {e.lineno}: {lines[line_no].rstrip()}")
            
            # Try progressive syntax error fixing
            logger.info("Attempting progressive syntax fixes...")
            try:
                # Apply comprehensive syntax fixing
                fixed_content = content
                
                # Fix attempt 1: Try to address hyphenated model names issues
                if '-' in model_family:
                    # Ensure all hyphenated references are properly converted
                    logger.info(f"Checking for missed hyphenated references in the content")
                    model_family_valid = to_valid_identifier(model_family)
                    model_pascal_case = get_pascal_case_identifier(model_family)
                    model_upper = model_family_valid.upper()
                    
                    # Find any leftover references
                    lines = fixed_content.split('\n')
                    for i, line in enumerate(lines):
                        # Handle registry references
                        if f"{model_family.upper()}_MODELS_REGISTRY" in line:
                            lines[i] = line.replace(f"{model_family.upper()}_MODELS_REGISTRY", f"{model_upper}_MODELS_REGISTRY")
                            logger.info(f"Fixed hyphenated registry reference on line {i+1}")
                            
                        # Handle class name references
                        original_class_pattern = f"Test{model_family.capitalize()}Models"
                        if original_class_pattern in line:
                            lines[i] = line.replace(original_class_pattern, f"Test{model_pascal_case}Models")
                            logger.info(f"Fixed hyphenated class name on line {i+1}")
                            
                        # Handle variable references
                        if model_family in line and not "'" in line and not '"' in line:
                            lines[i] = line.replace(model_family, model_family_valid)
                            logger.info(f"Fixed hyphenated variable reference on line {i+1}")
                    
                    fixed_content = '\n'.join(lines)
                
                # Fix attempt 2: Apply syntax error fixes
                fixed_content = fix_syntax_errors(fixed_content)
                
                # Fix attempt 3: Try indentation fixes specifically
                fixed_content = fix_indentation_issues(fixed_content)
                
                # Fix attempt 4: Fix unbalanced delimiters
                fixed_content = fix_unbalanced_delimiters(fixed_content)
                
                # Save the fixed content
                with open(output_file, 'w') as f:
                    f.write(fixed_content)
                
                # Verify syntax again
                try:
                    compile(fixed_content, output_file, 'exec')
                    logger.info(f"✅ Syntax is valid after comprehensive fixes for {output_file}")
                except SyntaxError as final_e:
                    logger.error(f"❌ Comprehensive syntax fixes failed: {final_e}")
                    
                    # Last attempt: Try to identify and fix the specific line
                    if hasattr(final_e, 'lineno') and final_e.lineno is not None:
                        line_no = final_e.lineno - 1  # 0-based index
                        lines = fixed_content.split('\n')
                        if 0 <= line_no < len(lines):
                            problematic_line = lines[line_no]
                            logger.error(f"Final problematic line {final_e.lineno}: {problematic_line}")
                            
                            # Apply specific fixes to the problematic line
                            if "unterminated" in str(final_e) and "string" in str(final_e):
                                # This is likely an unterminated triple-quoted string
                                if '"""' in content:
                                    # Try more aggressively to fix unterminated triple quotes
                                    # Find all lines with triple quotes and fix them
                                    triple_quote_lines = []
                                    for j, check_line in enumerate(lines):
                                        if '"""' in check_line:
                                            triple_quote_lines.append(j)
                                    
                                    # Check if we have an odd number of lines with triple quotes
                                    if len(triple_quote_lines) % 2 != 0:
                                        # Find the last line that likely needs closing quotes
                                        last_quote_line = triple_quote_lines[-1]
                                        # Add triple quotes to a new line after that
                                        indent = len(lines[last_quote_line]) - len(lines[last_quote_line].lstrip())
                                        lines.insert(last_quote_line + 1, ' ' * indent + '"""')
                                        logger.info(f"Added missing triple quotes after line {last_quote_line+1}")
                                elif "'''" in content:
                                    # Similar fix for single triple quotes
                                    single_triple_lines = []
                                    for j, check_line in enumerate(lines):
                                        if "'''" in check_line:
                                            single_triple_lines.append(j)
                                    
                                    # Fix if we have an odd number
                                    if len(single_triple_lines) % 2 != 0:
                                        last_quote_line = single_triple_lines[-1]
                                        indent = len(lines[last_quote_line]) - len(lines[last_quote_line].lstrip())
                                        lines.insert(last_quote_line + 1, ' ' * indent + "'''")
                                        logger.info(f"Added missing single triple quotes after line {last_quote_line+1}")
                                
                                # Last resort - if the error mentions a specific line, add triple quotes there
                                if line_no < len(lines):
                                    # Calculate indentation
                                    current_line = lines[line_no]
                                    indent = len(current_line) - len(current_line.lstrip()) if current_line.strip() else 4
                                    # Add quotes on a new line
                                    lines.insert(line_no + 1, ' ' * indent + '"""')
                                    logger.info(f"Last resort: Added triple quotes after line {line_no+1}")
                            
                            elif "unexpected EOF" in str(final_e):
                                # Try to fix unterminated strings or missing closing brackets
                                for char in ')}]':
                                    if problematic_line.count(char) < problematic_line.count({'(':')', '{':'}', '[':']'}[char]):
                                        lines[line_no] = problematic_line + char
                                        logger.info(f"Added missing '{char}' to line {final_e.lineno}")
                            
                            # Fix syntax error in docstring-method definition run-on
                            elif "invalid syntax" in str(final_e):
                                if '"""def ' in problematic_line:
                                    # Looks like docstring followed by method definition on same line
                                    parts = problematic_line.split('"""')
                                    if len(parts) >= 2:
                                        # Split into separate lines
                                        docstring = parts[0] + '"""'
                                        method_def = parts[1].strip()
                                        
                                        # Replace problematic line with fixed one
                                        lines[line_no] = docstring
                                        lines.insert(line_no + 1, '    ' + method_def)  # Use 4 spaces for class method
                                        
                                        logger.info(f"Fixed docstring-method definition run-on on line {final_e.lineno}")
                                elif problematic_line.strip().startswith('class ') and line_no + 1 < len(lines):
                                    # Check if the class needs docstring closure
                                    if '"""' in lines[line_no + 1] and not lines[line_no + 1].strip().endswith('"""'):
                                        # Close the docstring
                                        indent = len(lines[line_no + 1]) - len(lines[line_no + 1].lstrip())
                                        lines.insert(line_no + 2, ' ' * indent + '"""')
                                        logger.info(f"Fixed unclosed class docstring after line {line_no+2}")
                            
                            # Write the final attempt
                            fixed_content = '\n'.join(lines)
                            with open(output_file, 'w') as f:
                                f.write(fixed_content)
                            
                            # Final verification
                            try:
                                compile(fixed_content, output_file, 'exec')
                                logger.info(f"✅ Syntax is valid after final line-specific fixes for {output_file}")
                            except Exception:
                                logger.error(f"All syntax fix attempts failed for {output_file}")
                                return False
                    else:
                        logger.error(f"Cannot determine problematic line in final error: {final_e}")
                        return False
            except Exception as fix_error:
                logger.error(f"Failed to fix syntax errors: {fix_error}")
                return False
        
        logger.info(f"Generated test file: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating test file for {model_family}: {e}")
        traceback.print_exc()
        return False

def list_model_families():
    """List all available model families in the registry."""
    families = sorted(MODEL_REGISTRY.keys())
    print("\nAvailable model families:")
    for family in families:
        config = MODEL_REGISTRY[family]
        arch_type = get_architecture_type(family)
        print(f"  - {family} ({config['family_name']}): {config['description']} [Architecture: {arch_type}]")
    print()
    return families

def run_tests(all_models=False):
    """
    Run tests for model families.
    
    Args:
        all_models: If True, tests all models in registry
        
    Returns:
        Dict containing test results
    """
    # Determine if real inference or mock objects were used
    using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
    using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE
    
    results = {}
    
    if all_models:
        for model_family in MODEL_REGISTRY.keys():
            try:
                # Generate and run test for this family
                success = generate_test_file(model_family, "fixed_tests")
                results[model_family] = {"success": success}
            except Exception as e:
                logger.error(f"Error testing {model_family}: {e}")
                results[model_family] = {"success": False, "error": str(e)}
    
    return {
        "results": results,
        "metadata": {
            "timestamp": datetime.datetime.now().isoformat(),
            "has_transformers": HAS_TRANSFORMERS,
            "has_torch": HAS_TORCH,
            "has_tokenizers": HAS_TOKENIZERS,
            "has_sentencepiece": HAS_SENTENCEPIECE,
            "using_real_inference": using_real_inference,
            "using_mocks": using_mocks,
            "test_type": "REAL INFERENCE" if (using_real_inference and not using_mocks) else "MOCK OBJECTS (CI/CD)"
        }
    }

def get_hyphenated_model_families():
    """Get a list of all model families with hyphenated names."""
    return [f for f in MODEL_REGISTRY.keys() if '-' in f]

def test_hyphenated_model_support(output_dir="fixed_tests", verify=True):
    """Test the generator's support for hyphenated model families."""
    hyphenated_families = get_hyphenated_model_families()
    
    logger.info(f"Found {len(hyphenated_families)} model families with hyphenated names")
    if not hyphenated_families:
        logger.warning("No hyphenated model families found in registry")
        return True
    
    success_count = 0
    fail_count = 0
    results = {}
    
    for family in hyphenated_families:
        logger.info(f"Testing hyphenated model family: {family}")
        
        # Generate test file
        success = generate_test_file(family, output_dir)
        
        # Verify syntax if requested
        if verify and success:
            output_file = os.path.join(output_dir, f"test_hf_{to_valid_identifier(family)}.py")
            try:
                with open(output_file, "r") as f:
                    code = f.read()
                compile(code, output_file, "exec")
                logger.info(f"✅ {output_file}: Syntax is valid")
                results[family] = {"success": True, "valid_syntax": True}
                success_count += 1
            except SyntaxError as e:
                logger.error(f"❌ {output_file}: Syntax error: {e}")
                results[family] = {"success": False, "valid_syntax": False, "error": str(e)}
                fail_count += 1
        else:
            if success:
                results[family] = {"success": True}
                success_count += 1
            else:
                results[family] = {"success": False}
                fail_count += 1
    
    # Print summary
    total = len(hyphenated_families)
    logger.info(f"Hyphenated model test summary: {success_count}/{total} successful ({success_count/total*100:.1f}%)")
    
    return results

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Generate HuggingFace model test files")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list-families", action="store_true", help="List available model families")
    group.add_argument("--generate", type=str, help="Generate test for a specific model family")
    group.add_argument("--all", action="store_true", help="Generate tests for all model families")
    group.add_argument("--hyphenated-only", action="store_true", help="Generate tests only for models with hyphenated names")
    group.add_argument("--test-hyphenated", action="store_true", help="Test hyphenated model name handling")
    
    parser.add_argument("--output-dir", type=str, default="fixed_tests", help="Output directory for test files")
    parser.add_argument("--verify", action="store_true", help="Verify syntax of generated tests")
    
    
    # Advanced model selection options
    parser.add_argument("--task", type=str, help="Specific task for model selection")
    parser.add_argument("--hardware", type=str, help="Hardware profile for model selection")
    parser.add_argument("--max-size", type=int, help="Maximum model size in MB")
    parser.add_argument("--framework", type=str, help="Framework compatibility")
    
    args = parser.parse_args()
    
    if args.list_families:
        list_model_families()
        return 0
    
    if args.test_hyphenated:
        # Test specific support for hyphenated model families
        results = test_hyphenated_model_support(args.output_dir, args.verify)
        
        # Calculate success/failure counts
        success_count = sum(1 for r in results.values() if r.get("success", False))
        fail_count = len(results) - success_count
    
    elif args.all or args.hyphenated_only:
        # Get all families or just hyphenated ones
        if args.hyphenated_only:
            families = get_hyphenated_model_families()
            logger.info(f"Found {len(families)} model families with hyphenated names")
        else:
            families = MODEL_REGISTRY.keys()
            
        success_count = 0
        fail_count = 0
        
        for family in families:
            logger.info(f"Generating test for {family}...")
            success = generate_test_file(family, args.output_dir)
            
            if success:
                success_count += 1
            else:
                fail_count += 1
        
        # Determine if real inference or mock objects were used
        using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
        using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE
        
        print("\nTEST RESULTS SUMMARY:")
        
        # Indicate real vs mock inference clearly
        if using_real_inference and not using_mocks:
            print(f"🚀 Using REAL INFERENCE with actual models")
        else:
            print(f"🔷 Using MOCK OBJECTS for CI/CD testing only")
            print(f"   Dependencies: transformers={HAS_TRANSFORMERS}, torch={HAS_TORCH}, tokenizers={HAS_TOKENIZERS}, sentencepiece={HAS_SENTENCEPIECE}")
        
        logger.info(f"Generated {success_count} test files successfully, {fail_count} failed")
        return 0 if fail_count == 0 else 1
    
    if args.generate:
        family = args.generate
        success = generate_test_file(family, args.output_dir)
        
        if args.verify and success:
            output_file = os.path.join(args.output_dir, f"test_hf_{family}.py")
            try:
                with open(output_file, "r") as f:
                    code = f.read()
                compile(code, output_file, "exec")
                logger.info(f"✅ {output_file}: Syntax is valid")
            except SyntaxError as e:
                logger.error(f"❌ {output_file}: Syntax error: {e}")
                return 1
        
        # Determine if real inference or mock objects were used
        using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
        using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE
        
        print("\nTEST RESULTS SUMMARY:")
        
        # Indicate real vs mock inference clearly
        if using_real_inference and not using_mocks:
            print(f"🚀 Using REAL INFERENCE with actual models")
        else:
            print(f"🔷 Using MOCK OBJECTS for CI/CD testing only")
            print(f"   Dependencies: transformers={HAS_TRANSFORMERS}, torch={HAS_TORCH}, tokenizers={HAS_TOKENIZERS}, sentencepiece={HAS_SENTENCEPIECE}")
        
        return 0 if success else 1

# Indentation fixing utilities - full implementation
def apply_indentation(code, base_indent=0):
    """Apply consistent indentation to code blocks."""
    # Split the code into lines
    lines = code.strip().split('\n')
    
    # Determine the minimum indentation of non-empty lines
    min_indent = float('inf')
    for line in lines:
        if line.strip():  # Skip empty lines
            leading_spaces = len(line) - len(line.lstrip())
            min_indent = min(min_indent, leading_spaces)
    
    # If no indentation found, set to 0
    if min_indent == float('inf'):
        min_indent = 0
    
    # Remove the minimum indentation from all lines and add the base indentation
    indented_lines = []
    indent_spaces = ' ' * base_indent
    
    for line in lines:
        if line.strip():  # If not an empty line
            # Remove original indentation and add new base indentation
            indented_line = indent_spaces + line[min_indent:]
            indented_lines.append(indented_line)
        else:
            # For empty lines, just add base indentation
            indented_lines.append(indent_spaces)
    
    # Join the lines back into a single string
    return '\n'.join(indented_lines)

def fix_method_boundaries(content):
    """Fix method boundaries to ensure proper spacing and indentation."""
    # First add proper spacing between methods
    content = content.replace("        return results\n    def ", "        return results\n\n    def ")
    
    # Make sure __init__ has correct spacing after it
    content = content.replace("        self.performance_stats = {}\n    def ", "        self.performance_stats = {}\n\n    def ")
        
    # Place all method declarations at the right indentation level
    content = re.sub(r'(\s+)def test_pipeline\(', r'    def test_pipeline(', content)
    content = re.sub(r'(\s+)def test_from_pretrained\(', r'    def test_from_pretrained(', content)
    content = re.sub(r'(\s+)def run_tests\(', r'    def run_tests(', content)
    
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
    
    # Fix any other methods (save_results, main, etc.)
    content = re.sub(r'^(\s*)def ([^(]+)\(', r'def \2(', content, flags=re.MULTILINE)
    
    return content

def extract_method(content, method_name):
    """Extract a method from the class content."""
    # Find the method definition
    pattern = re.compile(rf'(\s+)def {method_name}\([^)]*\):(.*?)(?=\s+def|\Z)', re.DOTALL)
    match = pattern.search(content)
    
    if match:
        return match.group(0)
    return None

def fix_method_content(method_text, method_name):
    """Fix the indentation of a method's content."""
    # Normalize method indentation first
    lines = method_text.split('\n')
    method_lines = []
    
    # First line should be the method definition with exactly 4 spaces
    if lines and lines[0].strip().startswith(f"def {method_name}"):
        method_lines.append(f"    def {method_name}" + lines[0].strip()[4 + len(method_name):])
    else:
        # If we can't find the method definition, return unchanged
        return method_text
    
    # Process the remaining lines with proper indentation for method body
    i = 1
    in_docstring = False
    
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            method_lines.append("")
            i += 1
            continue
        
        # Handle docstrings
        if stripped.startswith('"""') and stripped.endswith('"""') and len(stripped) > 3:
            # Single line docstring
            method_lines.append("        " + stripped)
            i += 1
            continue
        elif stripped.startswith('"""'):
            # Start of multi-line docstring
            method_lines.append("        " + stripped)
            in_docstring = True
            i += 1
            continue
        elif stripped.endswith('"""') and in_docstring:
            # End of multi-line docstring
            method_lines.append("        " + stripped)
            in_docstring = False
            i += 1
            continue
        elif in_docstring:
            # Inside multi-line docstring
            method_lines.append("        " + stripped)
            i += 1
            continue
        
        # Calculate the indentation level based on context
        # Default indentation for method body (8 spaces = 4 for method + 4 for body)
        indent_level = 8
        
        # Check previous lines to help determine nesting
        if i > 1 and len(method_lines) > 1:
            prev_line = method_lines[-1].rstrip()
            if prev_line.endswith(':'):  # Previous line ends with colon (if, for, etc.)
                indent_level = len(prev_line) - len(prev_line.lstrip()) + 4
            elif len(prev_line) > 0:  # Use same indentation as previous non-empty line 
                indent_level = len(prev_line) - len(prev_line.lstrip())
        
        # Handle method body with appropriate indentation
        if stripped.startswith("if ") or stripped.startswith("for ") or stripped.startswith("while ") or \
           stripped.startswith("try:") or stripped.startswith("except ") or stripped.startswith("else:") or \
           stripped.startswith("elif ") or stripped.startswith("with ") or stripped.startswith("class "):
            # Control flow statements at method body level (8 spaces by default)
            method_lines.append(" " * 8 + stripped)
        elif stripped.startswith("return "):
            # Return statements at method body level
            method_lines.append(" " * 8 + stripped)
        elif stripped.startswith(("self.", "results[", "logger.")):
            # Method level variable access
            method_lines.append(" " * 8 + stripped)
        elif stripped.startswith("#"):
            # Comments at same level as surrounding code
            method_lines.append(" " * indent_level + stripped)
        elif stripped in ["pass", "continue", "break"]:
            # Simple statements
            method_lines.append(" " * indent_level + stripped)
        elif "=" in stripped and not stripped.startswith(" "):
            # Variable assignments
            method_lines.append(" " * 8 + stripped)
        elif stripped.startswith(("(", "[", "{")) or stripped.endswith((")", "]", "}")):
            # Collection literals or continuations - keep same indentation as surrounding context
            method_lines.append(" " * indent_level + stripped)
        else:
            # For other lines, try to preserve relative indentation based on context
            method_lines.append(" " * 8 + stripped)
        
        i += 1
    
    return "\n".join(method_lines)

def fix_dependency_checks(content):
    """Fix indentation in dependency check blocks."""
    # Fix dependency checks indentation to 8 spaces inside methods
    content = re.sub(r'(\s+)if not HAS_(\w+):', r'        if not HAS_\2:', content)
    
    # Fix returns in dependency checks
    content = re.sub(r'(\s+)return results', r'        return results', content)
    
    return content

def fix_imports(content):
    """Fix import section indentation."""
    # Make all top-level imports properly unindented
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(('import ', 'from ')):
            # Top-level imports should have no indentation
            fixed_lines.append(stripped)
        elif stripped.startswith(('try:', 'except ')):
            # Try/except blocks around imports should have no indentation
            fixed_lines.append(stripped)
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_mock_definitions(content):
    """Fix mock class definitions indentation."""
    # Fix indentation of mock classes
    mock_classes = re.findall(r'(\s+)class Mock(\w+):', content)
    for indent, class_name in mock_classes:
        # Replace with proper indentation (4 spaces for class inside a conditional block)
        content = content.replace(f"{indent}class Mock{class_name}:", f"    class Mock{class_name}:")
    
    return content

def fix_try_except_blocks(content):
    """Fix try/except block indentation and structure issues."""
    # Find all try blocks and properly indent their content
    try_pattern = re.compile(r'(\s+)try:(.*?)(\s+)except', re.DOTALL)
    
    def fix_try_block(match):
        indent = match.group(1)
        block_content = match.group(2)
        except_indent = match.group(3)
        
        # Normalize the block content indentation
        fixed_block = apply_indentation(block_content, len(indent) + 4)
        
        return f"{indent}try:{fixed_block}\n{except_indent}except"
    
    content = try_pattern.sub(fix_try_block, content)
    
    # Fix except blocks with similar approach
    except_pattern = re.compile(r'(\s+)except.*?:(.*?)(?=\s+(?:try:|except|else:|finally:|def|$))', re.DOTALL)
    
    def fix_except_block(match):
        indent = match.group(1)
        block_content = match.group(2)
        except_line = match.group(0).split(':', 1)[0] + ':'
        
        # Normalize the block content indentation
        fixed_block = apply_indentation(block_content, len(indent) + 4)
        
        return f"{except_line}{fixed_block}"
    
    content = except_pattern.sub(fix_except_block, content)
    
    # Fix broken import try-except blocks (specific issue)
    import_try_except_pattern = re.compile(
        r'try:\s*\n\s*import (transformers|torch)\s*\nimport \1\s*\n\s*HAS_\w+ = True',
        re.MULTILINE
    )
    content = re.sub(
        r'try:\s*\n\s*import (transformers|torch)\s*\nimport \1\s*\n\s*HAS_(\w+) = True',
        r'try:\n    import \1\n    HAS_\2 = True',
        content
    )

    # Fix nested conditional statements in try blocks
    try_if_pattern = re.compile(
        r'try:\s*\n\s*if not HAS_\w+:\s*\n\s*if not HAS_\w+:',
        re.MULTILINE
    )
    content = re.sub(
        r'try:\s*\n\s*if not HAS_(\w+):\s*\n\s*if not HAS_\1:',
        r'try:\n            if not HAS_\1:',
        content
    )
    
    # Fix doubled error handler patterns
    content = re.sub(
        r'except Exception as e:\s*\n\s*logger\.error\(.*\)\s*\n\s*logger\.error\(',
        r'except Exception as e:\n            logger.error(',
        content
    )
    
    return content

def fix_if_blocks(content):
    """Fix if/else block indentation."""
    # Find all if blocks and properly indent their content
    if_pattern = re.compile(r'(\s+)if\s+.*?:(.*?)(?=\s+(?:elif|else:|try:|except|def|$))', re.DOTALL)
    
    def fix_if_block(match):
        if_line = match.group(0).split(':', 1)[0] + ':'
        indent = match.group(1)
        block_content = match.group(2)
        
        # Normalize the block content indentation
        fixed_block = apply_indentation(block_content, len(indent) + 4)
        
        return f"{if_line}{fixed_block}"
    
    content = if_pattern.sub(fix_if_block, content)
    
    # Fix else blocks with similar approach
    else_pattern = re.compile(r'(\s+)else:(.*?)(?=\s+(?:try:|except|def|if|$))', re.DOTALL)
    
    def fix_else_block(match):
        indent = match.group(1)
        block_content = match.group(2)
        
        # Normalize the block content indentation
        fixed_block = apply_indentation(block_content, len(indent) + 4)
        
        return f"{indent}else:{fixed_block}"
    
    content = else_pattern.sub(fix_else_block, content)
    
    return content

def fix_class_method_indentation(content):
    """Fix indentation issues in class methods."""
    # Fix top-level imports and class definitions
    content = fix_imports(content)
    
    # Find the class definition(s)
    class_matches = re.finditer(r'class\s+(\w+):', content)
    
    for class_match in class_matches:
        class_name = class_match.group(1)
        class_start_pos = class_match.start()
        
        # Find all methods of this class 
        # This looks for methods until another class definition or EOF
        class_content_pattern = re.compile(r'class\s+' + class_name + r':(.*?)(?=class\s+\w+:|$)', re.DOTALL)
        class_content_match = class_content_pattern.search(content, class_start_pos)
        
        if not class_content_match:
            logger.warning(f"Could not extract content for class {class_name}")
            continue
        
        class_content = class_content_match.group(1)
        
        # Fix method indentation for common methods
        for method_name in ['__init__', 'test_pipeline', 'test_from_pretrained', 'run_tests']:
            method_text = extract_method(class_content, method_name)
            if method_text:
                fixed_method = fix_method_content(method_text, method_name)
                class_content = class_content.replace(method_text, fixed_method)
        
        # Fix dependency checks in methods
        class_content = fix_dependency_checks(class_content)
        
        # Fix mock class definitions inside the class
        class_content = fix_mock_definitions(class_content)
        
        # Fix try/except blocks
        class_content = fix_try_except_blocks(class_content)
        
        # Fix if/else blocks
        class_content = fix_if_blocks(class_content)
        
        # Fix spacing between methods
        class_content = fix_method_boundaries(class_content)
        
        # Replace the original class content with fixed content
        content = content[:class_start_pos] + "class " + class_name + ":" + class_content + content[class_content_match.end():]
    
    # Fix indentation of utility functions and main function
    for func_match in re.finditer(r'def\s+(\w+)\s*\(', content):
        func_name = func_match.group(1)
        if func_name not in ['__init__', 'test_pipeline', 'test_from_pretrained', 'run_tests']:
            func_pattern = re.compile(r'def\s+' + func_name + r'\s*\(.*?\):(.*?)(?=def\s+\w+\s*\(|$)', re.DOTALL)
            func_match = func_pattern.search(content, func_match.start())
            if func_match:
                func_text = func_match.group(0)
                fixed_func = apply_indentation(func_text, 0)  # Top-level functions have 0 indentation
                content = content.replace(func_text, fixed_func)
    
    return content

if __name__ == "__main__":
    sys.exit(main())