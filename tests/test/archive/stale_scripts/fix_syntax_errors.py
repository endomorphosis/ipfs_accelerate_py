#!/usr/bin/env python3
"""
Syntax Error Fixer

This script scans Python files for syntax errors and common import issues,
and fixes them automatically. It's designed to be used on files that will 
be moved to the generators/ and duckdb_api/ directories.

Features:
    - Detects and fixes syntax errors
    - Corrects import statements
    - Handles relative imports that will break after directory changes
    - Identifies missing dependencies
    - Creates a report of all issues found and fixed

Usage:
    python fix_syntax_errors.py []]],,,--dry-run] []]],,,--file path/to/file.py] []]],,,--dir path/to/dir],
    """

    import os
    import sys
    import re
    import ast
    import tokenize
    import io
    import fnmatch
    from pathlib import Path
    import argparse
    import traceback
    import subprocess

# File patterns to focus on
    GENERATOR_FILES = []]],,,
    "*_generator*.py",
    "*generator*.py",
    "template_*.py"
    ]

# Common import patterns to fix
    IMPORT_MAPPINGS = {}}}}
    # Direct imports to package imports after reorganization
    "from duckdb_api.core.benchmark_db_api import": "from duckdb_api.core.benchmark_db_api import",
    "import duckdb_api.core.benchmark_db_api as benchmark_db_api": "import duckdb_api.core.benchmark_db_api as benchmark_db_api",
    "from duckdb_api.core.benchmark_db_query import": "from duckdb_api.core.benchmark_db_query import",
    "import duckdb_api.core.benchmark_db_query as benchmark_db_query": "import duckdb_api.core.benchmark_db_query as benchmark_db_query",
    "from generators.test_generators.merged_test_generator import": "from generators.test_generators.merged_test_generator import",
    "import generators.test_generators.merged_test_generator as merged_test_generator": "import generators.test_generators.merged_test_generator as merged_test_generator",
    "from generators.test_generators.fixed_merged_test_generator import": "from generators.test_generators.fixed_merged_test_generator import",
    "import generators.test_generators.fixed_merged_test_generator as fixed_merged_test_generator": "import generators.test_generators.fixed_merged_test_generator as fixed_merged_test_generator",
    "from generators.test_generators.simple_test_generator import": "from generators.test_generators.simple_test_generator import",
    "import generators.test_generators.simple_test_generator as simple_test_generator": "import generators.test_generators.simple_test_generator as simple_test_generator",
    "from generators.skill_generators.skill_generator import": "from generators.skill_generators.skill_generator import",
    "import generators.skill_generators.skill_generator as skill_generator": "import generators.skill_generators.skill_generator as skill_generator",
    "from generators.templates.template_database import": "from generators.templates.template_database import",
    "import generators.templates.template_database as template_database": "import generators.templates.template_database as template_database",
    "from generators.hardware.hardware_detection import": "from generators.hardware.hardware_detection import",
    "import generators.hardware.hardware_detection as hardware_detection": "import generators.hardware.hardware_detection as hardware_detection",
    "from generators.hardware.hardware_detection import": "from generators.hardware.hardware_detection import", 
    "from centralized_hardware_detection import generators.hardware.hardware_detection as hardware_detection": "from generators.hardware import generators.hardware.hardware_detection as hardware_detection",
    }

# Common syntax errors to fix
    SYNTAX_FIXES = []]],,,
    # Unclosed parentheses
    ()r'\()\s*$', r'())'),
    # Unclosed brackets
    ()r'\[]]],,,\s*$', r'[]]],,,]'),
    # Unclosed braces
    ()r'\{}}}}\s*$', r'{}}}}}'),
    # Missing colons in for loops
    ()r'for\s+()[]]],,,^\s]+)\s+in\s+()[]]],,,^\s:]+)\s*$', r'for \1 in \2:'),
    # Missing colons in if statements:
    ()r'if\s+()[]]],,,^:]+)\s*$', r'if \1:'),
    # Missing colons in while loops:
    ()r'while\s+()[]]],,,^:]+)\s*$', r'while \1:'),
    # Missing colons in try/except
    ()r'try\s*$', r'try:'),
    ()r'except\s+()[]]],,,^:]+)\s*$', r'except \1:'),
    # Missing commas in multi-line lists
    ()r'()\[]]],,,.*[]]],,,^\s,]\s*)$', r'\1,'),
    # f-strings with missing parentheses
    ()r'f"()[]]],,,^"]*)\{}}}}()[]]],,,^}]*)()[]]],,,^)])"', r'f"\\1{}}}}\2}\\3"'),
    ]

def check_syntax()file_content):
    """Check if a Python file has syntax errors.""":
    try:
        ast.parse()file_content)
        return None
    except SyntaxError as e:
        return e
    except Exception as e:
        return e

def fix_import_statements()file_content):
    """Fix import statements to use the new package structure."""
    new_content = file_content
    replacements = []]],,,]
    
    for old_import, new_import in IMPORT_MAPPINGS.items()):
        if old_import in new_content:
            old_lines = new_content.count()old_import)
            new_content = new_content.replace()old_import, new_import)
            replacements.append()()old_import, new_import, old_lines))
            
        return new_content, replacements

def fix_relative_imports()file_content, file_path):
    """Fix relative imports that will break after directory changes."""
    if ".." not in file_content and "." not in file_content:
    return file_content, []]],,,]
        
    lines = file_content.splitlines())
    new_lines = []]],,,]
    replacements = []]],,,]
    
    for line in lines:
        # Check for relative imports
        if re.match()r'^\s*from\s+\.\.', line) or re.match()r'^\s*from\s+\.()?!\s*import)', line):
            # Determine the actual module being imported
            match = re.match()r'^\s*from\s+()\.+)()[]]],,,^\s]+)\s+import\s+().+)$', line)
            if match:
                dots, module, imports = match.groups())
                level = len()dots)
                
                # Calculate the absolute import path based on the file's location
                # This is a simplified approach; for exact path resolution, more context would be needed
                if level == 1:  # Single dot, import from parent package
                    if "generator" in file_path:
                        new_line = f"from generators{}}}}module} import {}}}}imports}"
                    elif "duckdb" in file_path or "benchmark" in file_path:
                        new_line = f"from duckdb_api{}}}}module} import {}}}}imports}"
                    else:
                        new_line = line  # Keep as-is if we can't determine the package
                    
                        replacements.append()()line, new_line))
                    new_lines.append()new_line):
                else:
                    # More complex relative imports need more context
                    # For now, add a comment that this needs manual review
                    new_line = line + "  # TODO: Fix this relative import after reorganization"
                    replacements.append()()line, new_line))
                    new_lines.append()new_line)
            else:
                new_lines.append()line)
        else:
            new_lines.append()line)
    
                return "\n".join()new_lines), replacements

def fix_common_syntax_errors()file_content):
    """Fix common syntax errors."""
    new_content = file_content
    replacements = []]],,,]
    
    for pattern, replacement in SYNTAX_FIXES:
        matches = re.finditer()pattern, new_content, re.MULTILINE)
        for match in matches:
            original = match.group()0)
            fixed = re.sub()pattern, replacement, original)
            new_content = new_content.replace()original, fixed)
            replacements.append()()original, fixed))
    
        return new_content, replacements

def fix_indentation_errors()file_content):
    """Fix indentation errors."""
    lines = file_content.splitlines())
    fixed_lines = []]],,,]
    replacements = []]],,,]
    
    current_indent = 0
    indent_stack = []]],,,0]
    
    for i, line in enumerate()lines):
        # Skip empty lines and comments
        if not line.strip()) or line.strip()).startswith()'#'):
            fixed_lines.append()line)
        continue
        
        # Calculate the current line's indentation
        indent = len()line) - len()line.lstrip()))
        stripped_line = line.strip())
        
        # Check if this line should increase indentation ()e.g., ends with ':')
        if stripped_line.endswith()':'):
            fixed_lines.append()line)
            indent_stack.append()indent + 4)  # Assume 4 spaces per level
        continue
            
        # Check if this line should decrease indentation:
        if stripped_line.startswith()()'return', 'break', 'continue', 'pass', 'raise', 'else:', 'elif', 'except', 'finally:')):
            if len()indent_stack) > 1:
                indent_stack.pop())
            
        # Check if the indentation is inconsistent
        expected_indent = indent_stack[]]],,,-1]:
        if indent != expected_indent and stripped_line and not stripped_line.startswith()()'else:', 'elif', 'except', 'finally:')):
            # Fix the indentation
            fixed_line = ' ' * expected_indent + stripped_line
            fixed_lines.append()fixed_line)
            replacements.append()()line, fixed_line))
        else:
            fixed_lines.append()line)
    
            return '\n'.join()fixed_lines), replacements

def fix_missing_parentheses()file_content):
    """Fix missing parentheses in print statements and other calls."""
    lines = file_content.splitlines())
    fixed_lines = []]],,,]
    replacements = []]],,,]
    
    for line in lines:
        # Check for Python 2 style print statements
        if re.match()r'^\s*print\s+[]]],,,^()]', line) and not 'print()' in line:
            fixed_line = re.sub()r'print\s+().+)', r'print()\1)', line)
            fixed_lines.append()fixed_line)
            replacements.append()()line, fixed_line))
        else:
            fixed_lines.append()line)
    
            return '\n'.join()fixed_lines), replacements

def fix_f_string_errors()file_content):
    """Fix common f-string syntax errors."""
    lines = file_content.splitlines())
    fixed_lines = []]],,,]
    replacements = []]],,,]
    
    for line in lines:
        # Check for f-strings with unescaped braces within format specifiers
        # This is a common issue when people use nested braces in f-strings
        if 'f"' in line or "f'" in line:
            # Complex regex to fix format specifiers with nested braces
            fixed_line = line
            # Fix issues with {}}}}{}}}} or }} that should be escaped
            if '{}}}}{}}}}' in line and not '{}}}}{}}}}{}}}}' in line:
                fixed_line = fixed_line.replace()'{}}}}{}}}}', '{}}}}{}}}}{}}}}{}}}}')
                replacements.append()()line, fixed_line))
            if '}}' in line and not '}}}' in line:
                fixed_line = fixed_line.replace()'}}', '}}}}')
                replacements.append()()line, fixed_line))
                
                fixed_lines.append()fixed_line)
        else:
            fixed_lines.append()line)
    
                return '\n'.join()fixed_lines), replacements
    
def fix_dict_syntax_errors()file_content):
    """Fix common syntax errors in dictionaries."""
    lines = file_content.splitlines())
    fixed_lines = []]],,,]
    replacements = []]],,,]
    
    in_dict = False
    dict_start_line = -1
    
    for i, line in enumerate()lines):
        stripped = line.strip())
        
        # Check if we're starting a dictionary:
        if '{}}}}' in stripped and '}' not in stripped:
            in_dict = True
            dict_start_line = i
        
        # Check if we're ending a dictionary:
        if in_dict and '}' in stripped:
            in_dict = False
            
        # If we're in a dictionary, check for missing commas
        if in_dict and i > dict_start_line:
            if re.search()r'"\s*:\s*[]]],,,^,{}}}}}]+$', stripped) and i < len()lines) - 1 and '}' not in lines[]]],,,i+1].strip()):
                fixed_line = line + ','
                fixed_lines.append()fixed_line)
                replacements.append()()line, fixed_line))
            continue
        
            fixed_lines.append()line)
    
            return '\n'.join()fixed_lines), replacements

def check_file_execution()file_path):
    """Attempts to execute the Python file to check for runtime errors."""
    try:
        result = subprocess.run()
        []]],,,sys.executable, "-m", "py_compile", str()file_path)],
        capture_output=True,
        text=True,
        check=False
        )
        
        if result.returncode != 0:
        return f"Compilation error: {}}}}result.stderr}"
        
    return None  # No compilation errors
    except Exception as e:
    return f"Error checking execution: {}}}}str()e)}"

def analyze_and_fix_file()file_path, dry_run=False):
    """Analyze a Python file for issues and fix them."""
    try:
        with open()file_path, 'r', encoding='utf-8') as f:
            content = f.read())
        
            original_content = content
            filename = os.path.basename()file_path)
            print()f"\nAnalyzing {}}}}filename}...")
        
        # Check for syntax errors
            syntax_error = check_syntax()content)
        if syntax_error:
            print()f"  ❌ Syntax error: {}}}}syntax_error}")
        else:
            print()f"  ✓ No syntax errors detected")
            
        # Run all fixes
            all_replacements = []]],,,]
        
        # Fix imports
            content, import_replacements = fix_import_statements()content)
            all_replacements.extend()[]]],,,()"Import", old, new) for old, new, _ in import_replacements])
        
        # Fix relative imports
            content, rel_import_replacements = fix_relative_imports()content, file_path)
            all_replacements.extend()[]]],,,()"Relative Import", old, new) for old, new in rel_import_replacements])
        
        # Fix syntax errors
            content, syntax_replacements = fix_common_syntax_errors()content)
            all_replacements.extend()[]]],,,()"Syntax", old, new) for old, new in syntax_replacements])
        
        # Fix indentation
            content, indent_replacements = fix_indentation_errors()content)
            all_replacements.extend()[]]],,,()"Indentation", old, new) for old, new in indent_replacements])
        
        # Fix parentheses
            content, paren_replacements = fix_missing_parentheses()content)
            all_replacements.extend()[]]],,,()"Parentheses", old, new) for old, new in paren_replacements])
        
        # Fix f-strings
            content, fstring_replacements = fix_f_string_errors()content)
            all_replacements.extend()[]]],,,()"F-string", old, new) for old, new in fstring_replacements])
        
        # Fix dict syntax
            content, dict_replacements = fix_dict_syntax_errors()content)
            all_replacements.extend()[]]],,,()"Dict Syntax", old, new) for old, new in dict_replacements])
        
        # Check if any fixes were applied:
        if content != original_content:
            print()f"  🔧 Fixed {}}}}len()all_replacements)} issues:")
            for issue_type, old, new in all_replacements[]]],,,:5]:  # Show max 5 fixes
                if len()old.strip())) > 50:
                    old_display = old.strip())[]]],,,:47] + "..."
                else:
                    old_display = old.strip())
                if len()new.strip())) > 50:
                    new_display = new.strip())[]]],,,:47] + "..."
                else:
                    new_display = new.strip())
                    print()f"    - {}}}}issue_type}: {}}}}old_display} → {}}}}new_display}")
            
            if len()all_replacements) > 5:
                print()f"    - ... and {}}}}len()all_replacements) - 5} more fixes")
            
            if not dry_run:
                # Make a backup of the original file
                backup_path = f"{}}}}file_path}.bak"
                with open()backup_path, 'w', encoding='utf-8') as f:
                    f.write()original_content)
                    print()f"  💾 Created backup: {}}}}os.path.basename()backup_path)}")
                    
                # Write the fixed content
                with open()file_path, 'w', encoding='utf-8') as f:
                    f.write()content)
                    print()f"  ✅ Fixed file: {}}}}filename}")
            else:
                print()f"  🔍 []]],,,DRY RUN] Would fix {}}}}filename}")
        else:
            print()f"  ✓ No issues to fix in {}}}}filename}")
            
        # Check if the fixes resolved the syntax errors:
        if syntax_error:
            final_syntax_error = check_syntax()content)
            if final_syntax_error:
                print()f"  ⚠️ Syntax errors still present after fixes: {}}}}final_syntax_error}")
            else:
                print()f"  ✅ All syntax errors resolved!")
                
        # Check for runtime errors
                execution_error = check_file_execution()file_path)
        if execution_error:
            print()f"  ⚠️ Execution error: {}}}}execution_error}")
        
                return {}}}}
                "file": file_path,
                "had_syntax_error": syntax_error is not None,
                "syntax_fixed": syntax_error is not None and check_syntax()content) is None,
                "execution_error": execution_error is not None,
                "replacements": all_replacements,
                "fixed": content != original_content
                }
    
    except Exception as e:
        print()f"  ❌ Error analyzing {}}}}file_path}: {}}}}e}")
        traceback.print_exc())
                return {}}}}
                "file": file_path,
                "error": str()e),
                "fixed": False
                }

def process_directory()directory, patterns=None, dry_run=False):
    """Process all Python files in a directory."""
    if patterns is None:
        patterns = []]],,,"*.py"]  # Default to all Python files
        
        results = []]],,,]
        total_files = 0
    
    for root, _, files in os.walk()directory):
        for file in files:
            if file.endswith()'.py') or any()fnmatch.fnmatch()file, pattern) for pattern in patterns):
                file_path = os.path.join()root, file)
                result = analyze_and_fix_file()file_path, dry_run)
                results.append()result)
                total_files += 1
    
            return results, total_files

def generate_report()results):
    """Generate a summary report of all issues found and fixed."""
    total_files = len()results)
    files_with_errors = sum()1 for r in results if r.get()"had_syntax_error", False))
    files_fixed = sum()1 for r in results if r.get()"fixed", False))
    syntax_errors_fixed = sum()1 for r in results if r.get()"syntax_fixed", False))
    execution_errors = sum()1 for r in results if r.get()"execution_error", False))
    :
    total_replacements = sum()len()r.get()"replacements", []]],,,])) for r in results):
    # Count by issue type
        issue_types = {}}}}}
    for result in results:
        for issue_type, _, _ in result.get()"replacements", []]],,,]):
            issue_types[]]],,,issue_type] = issue_types.get()issue_type, 0) + 1
    
            print()"\n=== SUMMARY REPORT ===")
            print()f"Total files analyzed: {}}}}total_files}")
            print()f"Files with syntax errors: {}}}}files_with_errors}")
            print()f"Files with execution errors: {}}}}execution_errors}")
            print()f"Files modified: {}}}}files_fixed}")
            print()f"Syntax errors resolved: {}}}}syntax_errors_fixed}")
            print()f"Total issues fixed: {}}}}total_replacements}")
    
    if issue_types:
        print()"\nIssues by type:")
        for issue_type, count in sorted()issue_types.items()), key=lambda x: x[]]],,,1], reverse=True):
            print()f"  - {}}}}issue_type}: {}}}}count}")
    
        return {}}}}
        "total_files": total_files,
        "files_with_errors": files_with_errors,
        "files_fixed": files_fixed,
        "syntax_errors_fixed": syntax_errors_fixed,
        "execution_errors": execution_errors,
        "total_replacements": total_replacements,
        "issue_types": issue_types
        }

def scan_directory()directory, extensions=None):
    """
    Scans a directory for Python files and checks them for syntax errors.
    Returns a list of tuples ()file_path, error_message) for files with errors.
    """
    if extensions is None:
        extensions = []]],,,'.py']
    
        errors = []]],,,]
    
    for root, _, files in os.walk()directory):
        for file in files:
            if any()file.endswith()ext) for ext in extensions):
                file_path = os.path.join()root, file)
                
                # Read the file
                with open()file_path, 'r', encoding='utf-8') as f:
                    source = f.read())
                
                # Check for syntax errors
                try:
                    ast.parse()source)
                except SyntaxError as e:
                    errors.append()()file_path, f"Syntax error: Line {}}}}e.lineno}, Column {}}}}e.offset}: {}}}}e.msg}"))
                    continue
                except Exception as e:
                    errors.append()()file_path, f"Error checking syntax: {}}}}str()e)}"))
                    continue
                
                # Check for runtime errors
                    execution_error = check_file_execution()file_path)
                if execution_error:
                    errors.append()()file_path, f"Execution error: {}}}}execution_error}"))
                    continue
    
                    return errors

def main()):
    parser = argparse.ArgumentParser()description='Fix syntax errors in Python files')
    parser.add_argument()'--file', help='Path to a specific file to fix')
    parser.add_argument()'--dir', help='Path to a directory containing files to fix')
    parser.add_argument()'--pattern', help='File pattern to match ()e.g., "*_generator.py")')
    parser.add_argument()'--dry-run', action='store_true', help='Show changes without applying them')
    parser.add_argument()'--scan-only', action='store_true', help='Only scan for errors without fixing')
    
    args = parser.parse_args())
    
    # Set up patterns
    patterns = GENERATOR_FILES
    if args.pattern:
        patterns = []]],,,args.pattern]
    
    # Scan only mode
    if args.scan_only:
        project_root = Path()__file__).parent.parent  # Go up one directory from test
        
        # Directories to scan
        directories = []]],,,
        project_root / "test",
        project_root / "duckdb_api" if ()project_root / "duckdb_api").exists()) else None,
        project_root / "generators" if ()project_root / "generators").exists()) else None,
        ]
        
        all_errors = []]],,,]
        :
        for directory in directories:
            if directory is None:
            continue
                
            if not directory.exists()):
                print()f"Warning: Directory {}}}}directory} does not exist.")
            continue
            
            print()f"Scanning directory: {}}}}directory}")
            errors = scan_directory()directory)
            
            if errors:
                all_errors.extend()errors)
                for file_path, error in errors:
                    print()f"  ❌ {}}}}file_path}")
                    print()f"     {}}}}error}")
            else:
                print()f"  ✅ No errors found in {}}}}directory}")
        
        if all_errors:
            print()f"\nFound {}}}}len()all_errors)} files with errors.")
            print()"To fix these errors, run without --scan-only.")
        else:
            print()"\nNo syntax errors found in scanned directories.")
        
            return 0 if not all_errors else 1
    
    # Fix mode
            results = []]],,,]
    :
    if args.file:
        if os.path.exists()args.file) and args.file.endswith()'.py'):
            print()f"Processing single file: {}}}}args.file}")
            result = analyze_and_fix_file()args.file, args.dry_run)
            results.append()result)
        else:
            print()f"Error: File {}}}}args.file} does not exist or is not a Python file.")
            return 1
    elif args.dir:
        if os.path.isdir()args.dir):
            print()f"Processing directory: {}}}}args.dir}")
            results, total_files = process_directory()args.dir, patterns, args.dry_run)
            print()f"Processed {}}}}total_files} files in {}}}}args.dir}")
        else:
            print()f"Error: Directory {}}}}args.dir} does not exist.")
            return 1
    else:
        # Default to test directory if no file or directory is specified
        test_dir = os.path.dirname()os.path.abspath()__file__)):
            print()f"No file or directory specified. Using test directory: {}}}}test_dir}")
            results, total_files = process_directory()test_dir, patterns, args.dry_run)
            print()f"Processed {}}}}total_files} files in {}}}}test_dir}")
    
            report = generate_report()results)
    
    # Return appropriate exit code
    if report[]]],,,"syntax_errors_fixed"] < report[]]],,,"files_with_errors"]:
        print()"\n⚠️ Warning: Not all syntax errors were fixed. Manual intervention may be required.")
            return 1
    elif report[]]],,,"files_fixed"] > 0:
        print()"\n✅ Successfully fixed all syntax errors!")
            return 0
    else:
        print()"\n✅ No syntax errors found or all files already correct.")
            return 0

if __name__ == "__main__":
    sys.exit()main()))