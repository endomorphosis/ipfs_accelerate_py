#!/usr/bin/env python3
"""Comprehensive fix for all template files."""

import os
import sys
import re
import shutil
import tempfile
import subprocess

# Path to templates
templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

# All template files
templates = [
    "encoder_only_template.py",
    "decoder_only_template.py",
    "encoder_decoder_template.py",
    "vision_template.py",
    "vision_text_template.py",
    "speech_template.py",
    "multimodal_template.py",
    "minimal_template.py"
]

def find_problematic_lines(file_path):
    """Find problematic lines in a template file."""
    problematic_lines = []
    
    try:
        # Try to compile the file
        subprocess.run([sys.executable, "-m", "py_compile", file_path], 
                      capture_output=True, check=True)
        print(f"✅ {file_path} has no syntax errors")
        return []
    except subprocess.CalledProcessError as e:
        # Get error output
        error_output = e.stderr.decode('utf-8')
        print(f"❌ {file_path} has syntax errors:")
        print(error_output)
        
        # Extract problematic line numbers from error output
        line_matches = re.finditer(r'line (\d+)', error_output)
        for match in line_matches:
            try:
                line_num = int(match.group(1))
                problematic_lines.append(line_num)
            except ValueError:
                continue
    
    return problematic_lines

def fix_template_file(template_name):
    """Fix a specific template file by targeting problematic lines."""
    file_path = os.path.join(templates_dir, template_name)
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    # Create backup
    backup_path = file_path + ".comprehensive.bak"
    shutil.copy2(file_path, backup_path)
    print(f"Created backup: {backup_path}")
    
    # Find problematic lines
    problematic_lines = find_problematic_lines(file_path)
    if not problematic_lines:
        return True  # No problems found
    
    print(f"Problematic line numbers: {problematic_lines}")
    
    # Read the file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Track if we made any changes
    made_changes = False
    
    # Look for specific patterns and fix them
    for i in range(len(lines)):
        line = lines[i]
        line_num = i + 1
        
        # Skip if this line isn't close to a problematic line
        if not any(abs(line_num - prob) < 3 for prob in problematic_lines):
            continue
        
        # Fix 1: Unterminated print statements
        if 'print(' in line and ('print("' in line or 'print(f"' in line) and line.strip().endswith('print('):
            # This is likely an unterminated print statement
            if i + 1 < len(lines) and not lines[i + 1].strip().startswith('"') and not lines[i + 1].strip().startswith('f"'):
                # The next line is part of the unterminated string
                next_line = lines[i + 1].strip()
                
                # If it looks like a title line, create a properly formatted f-string
                if "Models Testing Summary" in next_line or "Available" in next_line:
                    if "BERT" in next_line and "print(f" in line:
                        lines[i] = '        print(f"\\nBERT Models Testing Summary:")\n'
                    elif "VISION" in next_line and "print(f" in line:
                        lines[i] = '        print(f"\\nVISION Models Testing Summary:")\n'
                    elif "SPEECH" in next_line and "print(f" in line:
                        lines[i] = '        print(f"\\nSPEECH Models Testing Summary:")\n'
                    elif "CLIP" in next_line and "print(f" in line:
                        lines[i] = '        print(f"\\nCLIP Models Testing Summary:")\n'
                    elif "MULTIMODAL" in next_line and "print(f" in line:
                        lines[i] = '        print(f"\\nMULTIMODAL Models Testing Summary:")\n'
                    else:
                        lines[i] = '        print(f"\\nModels Testing Summary:")\n'
                    
                    lines[i + 1] = ''  # Clear the next line
                    made_changes = True
        
        # Fix 2: Missing parenthesis
        elif 'print(' in line and ('"' in line or 'f"' in line) and line.strip().endswith(')'):
            # Check if there's an extra closing parenthesis
            if line.count('(') < line.count(')'):
                lines[i] = line.replace(')"', '")')
                made_changes = True
        
        # Fix 3: General fixes for unterminated strings with newlines
        elif ('print(' in line and '"' in line and '\n' in line and not line.strip().endswith('"')):
            # Handle unterminated string literals with newlines
            if "\\n" not in line:
                line = line.replace('\n', '\\n').replace('\\\\n', '\\n')
                if not line.strip().endswith('"'):
                    line = line.rstrip() + '"\n'
                lines[i] = line
                made_changes = True
    
    # Write the fixed file if changes were made
    if made_changes:
        with open(file_path, 'w') as f:
            f.writelines(lines)
        
        # Check if the file is now valid
        try:
            subprocess.run([sys.executable, "-m", "py_compile", file_path], 
                          capture_output=True, check=True)
            print(f"✅ Successfully fixed {file_path}")
            return True
        except subprocess.CalledProcessError:
            print(f"❌ {file_path} still has syntax errors after fixes")
            # Keep trying with different approach
    
    # If we get here, we need to try more aggressive fixes
    print(f"Trying more aggressive fixes for {file_path}")
    
    # Get the content of the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix 1: Fix all unterminated f-strings
    content = re.sub(r'print\(f"(?!\s*")', 'print(f"\\n', content)
    
    # Fix 2: Fix specific multiline strings
    for line_num in problematic_lines:
        # Extract the context around the problematic line
        start = max(0, line_num - 5)
        end = min(len(lines), line_num + 5)
        context = ''.join(lines[start:end])
        
        # Look for known problematic patterns in the context
        if 'print(' in context and 'Models Testing Summary' in context:
            # This is likely a problematic summary title print
            model_type = "BERT"  # Default
            if "VISION" in context:
                model_type = "VISION" 
            elif "SPEECH" in context:
                model_type = "SPEECH"
            elif "CLIP" in context:
                model_type = "CLIP"
            elif "MULTIMODAL" in context:
                model_type = "MULTIMODAL"
            
            # Create a proper replacement pattern
            replacement = f'        print(f"\\n{model_type} Models Testing Summary:")\n        total = len(results)\n'
            
            # Try to find and replace the pattern in the full content
            content = re.sub(r'print\(".*?\n.*?Models Testing Summary:"\)', replacement, content, flags=re.DOTALL)
    
    # Fix 3: Handle extraneous closing parentheses after print
    content = content.replace(')))\n', '))\n')
    content = content.replace('")))\n', '"))\n')
    
    # Write the aggressive fixes
    with open(file_path, 'w') as f:
        f.write(content)
    
    # Verify the aggressive fixes
    try:
        subprocess.run([sys.executable, "-m", "py_compile", file_path], 
                      capture_output=True, check=True)
        print(f"✅ Successfully fixed {file_path} with aggressive approach")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ {file_path} still has syntax errors after aggressive fixes")
        
        # Try a targeted approach
        if "multimodal_template.py" in file_path:
            # Handle multimodal template specifically
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Direct fixes for known problematic lines in multimodal template
            for i, line in enumerate(lines):
                if i == 777:  # Line 778 (0-indexed)
                    lines[i] = '        print("\\nMULTIMODAL Models Testing Summary:")\n'
                elif i == 778:  # Line 779 (0-indexed)
                    if "Models Testing Summary" in line:
                        lines[i] = '        total = len(results)\n'
            
            with open(file_path, 'w') as f:
                f.writelines(lines)
            
        elif "vision_text_template.py" in file_path:
            # Handle vision_text template specifically
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Direct fixes for known problematic lines in vision_text template
            for i, line in enumerate(lines):
                if i == 775:  # Line 776 (0-indexed)
                    lines[i] = '        print("\\nVISION TEXT Models Testing Summary:")\n'
                elif i == 776:  # Line 777 (0-indexed)
                    if "Models Testing Summary" in line:
                        lines[i] = '        total = len(results)\n'
            
            with open(file_path, 'w') as f:
                f.writelines(lines)
            
        elif "speech_template.py" in file_path:
            # Handle speech template specifically
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Direct fixes for known problematic lines in speech template
            for i, line in enumerate(lines):
                if i == 756:  # Line 757 (0-indexed)
                    lines[i] = '        print("\\nSPEECH Models Testing Summary:")\n'
                elif i == 757:  # Line 758 (0-indexed)
                    if "Models Testing Summary" in line:
                        lines[i] = '        total = len(results)\n'
            
            with open(file_path, 'w') as f:
                f.writelines(lines)
        
        # Check if specific fixes worked
        try:
            subprocess.run([sys.executable, "-m", "py_compile", file_path], 
                          capture_output=True, check=True)
            print(f"✅ Successfully fixed {file_path} with targeted fixes")
            return True
        except subprocess.CalledProcessError:
            print(f"❌ {file_path} still has syntax errors after all attempts")
            
            # Restore backup as a last resort
            shutil.copy2(backup_path, file_path)
            print(f"Restored {file_path} from backup")
            return False

def fix_all_templates():
    """Fix all template files."""
    success_count = 0
    
    for template in templates:
        print(f"\n=== Processing {template} ===")
        if fix_template_file(template):
            success_count += 1
    
    print(f"\nFixed {success_count} of {len(templates)} templates")
    return success_count == len(templates)

if __name__ == "__main__":
    sys.exit(0 if fix_all_templates() else 1)