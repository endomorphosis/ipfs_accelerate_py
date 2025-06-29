#!/usr/bin/env python3
"""
Deep template fixer for hyphenated model test files.

This script fixes the template files by directly targeting the specific syntax errors
like unterminated string literals, newlines in print statements, and other common issues.
"""

import os
import re
import logging
from pathlib import Path
import argparse
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = CURRENT_DIR / "templates"
FIXED_TEMPLATES_DIR = CURRENT_DIR / "fixed_templates_deep"

def fix_print_statements(content):
    """Fix problematic print statements in template files."""
    # Find all print statements with newlines or unterminated strings
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check for problematic print statements that continue on next line
        if 'print(f"' in line and not line.strip().endswith('")'): 
            # This is a multiline print that's problematic
            print_start = line
            current_line = i
            buffer = [print_start]
            
            # Collect all lines until we find the end of the print statement
            while current_line + 1 < len(lines) and not lines[current_line].strip().endswith('")'):
                current_line += 1
                buffer.append(lines[current_line])
            
            # Fix this print statement
            if current_line + 1 < len(lines):
                # Extract content inside the print statement
                full_statement = ' '.join(buffer)
                match = re.match(r'(.+?print\(f")(.+)', full_statement, re.DOTALL)
                
                if match:
                    prefix = match.group(1)
                    content_part = match.group(2)
                    
                    # Clean up the content part and ensure it has closing quote and parenthesis
                    content_part = content_part.replace('\n', ' ')
                    if not content_part.endswith('")'):
                        content_part = content_part + '")'
                    
                    # Combine and add to fixed lines
                    fixed_lines.append(f"{prefix}{content_part}")
                    i = current_line + 1  # Skip all the lines we've processed
                    continue
            
        # Regular line processing if not a problematic print
        fixed_lines.append(line)
        i += 1
    
    return '\n'.join(fixed_lines)

def fix_specific_print_errors(content):
    """Fix specific print statement errors in template files."""
    # Fix prints with newlines
    content = re.sub(r'print\(f"\n', r'print(f"', content)
    
    # Fix unterminated f-strings with extra closing parentheses
    content = re.sub(r'print\(f"\s*Available [^"]*:")+")', r'print(f"Available MODEL-family models:")', content)
    
    # Fix the specific pattern in decoder_only_template.py
    content = re.sub(r'print\(f"\s*Available GPT-2-family models:")+")', 
                   r'print(f"Available MODEL-family models:")', content)
    
    # Fix the pattern in speech_template.py
    content = re.sub(r'print\(f"\s*Available SPEECH-family models:"\nBERT Models Testing Summary:',
                   r'print(f"Available SPEECH-family models:")\n    # Print summary', content)
    
    # Fix similar issues in other templates
    content = re.sub(r'print\(f"\nAvailable VISION-TEXT-family models:"\nBERT Models Testing Summary:',
                   r'print(f"Available VISION-TEXT-family models:")\n    # Print summary', content)
    
    # Fix the weird string in multimodal_template.py
    content = re.sub(r'print\(f"\s*Available MULTIMODAL-family models:")+")',
                   r'print(f"Available MULTIMODAL-family models:")', content)
    
    return content

def fix_template_file(file_path, output_dir=None):
    """Fix deep syntax errors in a template file."""
    if output_dir is None:
        output_dir = FIXED_TEMPLATES_DIR
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    template_name = os.path.basename(file_path)
    logger.info(f"Deep fixing template file: {template_name}")
    
    try:
        # Read template content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Apply deep fixes
        content = fix_print_statements(content)
        content = fix_specific_print_errors(content)
        
        # Additional deep fixes for specific templates
        if "decoder_only" in template_name:
            # Fix decoder_only_template.py specific issues
            content = content.replace('print(f"\nAvailable GPT2-family models:")', 'print("Available GPT2-family models:")')
            content = content.replace('print(f"\nAvailable GPT-2-family models:")', 'print("Available GPT-2-family models:")')
        
        elif "vision_text" in template_name:
            # Fix vision_text_template.py specific issues
            content = content.replace('print(f"\nAvailable VISION TEXT-family models:")', 'print("Available VISION-TEXT-family models:")')
            
        elif "speech" in template_name:
            # Fix speech_template.py specific issues
            content = content.replace('print(f"\nAvailable SPEECH-family models:")', 'print("Available SPEECH-family models:")')
            
        elif "multimodal" in template_name:
            # Fix multimodal_template.py specific issues
            content = content.replace('print(f"\nAvailable MULTIMODAL-family models:")', 'print("Available MULTIMODAL-family models:")')
        
        # Check if content is valid Python syntax
        try:
            compile(content, template_name, 'exec')
            logger.info(f"  ✅ Fixed syntax is valid")
        except SyntaxError as e:
            logger.warning(f"  ⚠️ Fixed syntax still has errors: {e}")
            
            # Try a more aggressive approach for the issues
            if "unterminated" in str(e) and "print" in str(e):
                logger.info("  🔧 Attempting aggressive fix for unterminated print statement")
                
                # Find the problematic line
                lines = content.split('\n')
                line_no = e.lineno - 1  # 0-indexed
                
                # Simply replace the entire line with a safe alternative
                if line_no < len(lines):
                    if "Available" in lines[line_no]:
                        model_type = re.search(r'Available ([A-Z-]+)-family', lines[line_no])
                        if model_type:
                            model_name = model_type.group(1)
                            lines[line_no] = f'        print("Available {model_name}-family models:")'
                        else:
                            lines[line_no] = '        print("Available models:")'
                    
                    content = '\n'.join(lines)
                    
                    # Verify fix worked
                    try:
                        compile(content, template_name, 'exec')
                        logger.info(f"  ✅ Aggressive fix successful")
                    except SyntaxError as e2:
                        logger.warning(f"  ❌ Aggressive fix failed: {e2}")
        
        # Write fixed template to output directory
        output_path = os.path.join(output_dir, template_name)
        with open(output_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Successfully deep fixed template file: {template_name} -> {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error fixing template file {template_name}: {str(e)}")
        return False

def fix_all_templates(output_dir=None):
    """Fix all template files in the templates directory."""
    if output_dir is None:
        output_dir = FIXED_TEMPLATES_DIR
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all template files
    template_files = [os.path.join(TEMPLATES_DIR, f) for f in os.listdir(TEMPLATES_DIR) if f.endswith('_template.py')]
    
    success_count = 0
    failure_count = 0
    
    for template_path in template_files:
        if fix_template_file(template_path, output_dir):
            success_count += 1
        else:
            failure_count += 1
    
    logger.info(f"Deep fixed {success_count} templates, {failure_count} failed")
    return success_count, failure_count

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Deep fix syntax errors in template files")
    parser.add_argument("--template", type=str, help="Specific template file to fix (relative to templates/)")
    parser.add_argument("--all", action="store_true", help="Fix all template files")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for fixed templates")
    
    args = parser.parse_args()
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = FIXED_TEMPLATES_DIR
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if args.template:
        # Fix specific template
        template_path = os.path.join(TEMPLATES_DIR, args.template)
        if not os.path.exists(template_path):
            logger.error(f"Template file not found: {template_path}")
            return 1
        
        success = fix_template_file(template_path, output_dir)
        return 0 if success else 1
    
    elif args.all:
        # Fix all templates
        success_count, failure_count = fix_all_templates(output_dir)
        return 0 if failure_count == 0 else 1
    
    else:
        logger.error("No action specified. Use --template or --all")
        return 1

if __name__ == "__main__":
    sys.exit(main())