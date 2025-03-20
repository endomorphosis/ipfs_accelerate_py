#!/usr/bin/env python3
"""
Unified script for fixing indentation in HuggingFace test files.

This script combines all our fixes and provides a single entry point
for fixing indentation issues in test files.
"""

import os
import sys
import argparse
import glob
import tempfile
import subprocess
import shutil
from pathlib import Path

# Options
BACKUP_FILES = True
VALIDATE_SYNTAX = True
USE_ENHANCED_FIXES = True

# Function to check if Python file has valid syntax
def verify_python_syntax(file_path):
    """Verify that the Python file has valid syntax."""
    try:
        with open(file_path, 'r') as f:
            code = f.read()
        compile(code, file_path, 'exec')
        return True, None
    except SyntaxError as e:
        return False, str(e)

def fix_single_file(file_path, verbose=False, approach="comprehensive"):
    """
    Fix indentation issues in a single file.
    
    Args:
        file_path: Path to the file to fix
        verbose: Enable verbose output
        approach: Fix approach - "comprehensive", "simple", or "minimal"
    
    Returns:
        Tuple of (success, error_message)
    """
    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"
    
    # Create backup of original file
    if BACKUP_FILES:
        backup_path = f"{file_path}.bak"
        shutil.copy2(file_path, backup_path)
        if verbose:
            print(f"Created backup at {backup_path}")
    
    try:
        if approach == "comprehensive":
            # Use fix_file_indentation.py (most thorough)
            if os.path.exists("fix_file_indentation.py"):
                cmd = [sys.executable, "fix_file_indentation.py", file_path]
                subprocess.run(cmd, check=True)
            else:
                # Fall back to imported function if available
                try:
                    from fix_file_indentation import fix_class_method_indentation
                    fix_class_method_indentation(file_path, backup=False)  # Backup already created
                except ImportError:
                    return False, "fix_file_indentation.py not found and import failed"
        
        elif approach == "simple":
            # Use simple_fixer.py (pattern-based fixes)
            if os.path.exists("simple_fixer.py"):
                cmd = [sys.executable, "simple_fixer.py", file_path]
                subprocess.run(cmd, check=True)
            else:
                return False, "simple_fixer.py not found"
        
        elif approach == "minimal":
            # Use minimal_tests template approach
            family = os.path.basename(file_path).replace("test_hf_", "").replace(".py", "")
            template_path = os.path.join("minimal_tests", f"test_hf_{family}.py")
            
            if os.path.exists(template_path):
                # Copy the minimal template and keep the original imports and constants
                with open(file_path, 'r') as src_f:
                    src_content = src_f.read()
                
                with open(template_path, 'r') as template_f:
                    template_content = template_f.read()
                
                # Extract imports from original file
                import_section = ""
                for line in src_content.split('\n'):
                    if line.startswith('import ') or line.startswith('from '):
                        import_section += line + '\n'
                
                # Replace template imports with original imports
                template_import_section = ""
                for line in template_content.split('\n'):
                    if line.startswith('import ') or line.startswith('from '):
                        template_import_section += line + '\n'
                
                if template_import_section and import_section:
                    fixed_content = template_content.replace(template_import_section.strip(), import_section.strip())
                    
                    with open(file_path, 'w') as f:
                        f.write(fixed_content)
                else:
                    # Just copy the template if import extraction failed
                    shutil.copy2(template_path, file_path)
            else:
                return False, f"Minimal template not found for {family}"
        
        else:
            return False, f"Unknown approach: {approach}"
        
        # Validate syntax
        if VALIDATE_SYNTAX:
            success, error = verify_python_syntax(file_path)
            if not success:
                return False, f"Syntax error after fixing: {error}"
        
        return True, None
    
    except Exception as e:
        return False, f"Error fixing file: {e}"

def fix_directory(directory, pattern="test_hf_*.py", verbose=False, approach="comprehensive"):
    """
    Fix all matching files in a directory.
    
    Args:
        directory: Directory to search for files
        pattern: Glob pattern to match files
        verbose: Enable verbose output
        approach: Fix approach - "comprehensive", "simple", or "minimal"
    
    Returns:
        Tuple of (success_count, failure_count, failed_files)
    """
    if not os.path.isdir(directory):
        print(f"Directory not found: {directory}")
        return 0, 0, []
    
    success_count = 0
    failure_count = 0
    failed_files = []
    
    files = glob.glob(os.path.join(directory, pattern))
    
    if verbose:
        print(f"Found {len(files)} files matching pattern {pattern} in {directory}")
    
    for file_path in files:
        if verbose:
            print(f"Fixing {file_path}...")
        
        success, error = fix_single_file(file_path, verbose, approach)
        
        if success:
            success_count += 1
            if verbose:
                print(f"✅ Successfully fixed {file_path}")
        else:
            failure_count += 1
            failed_files.append((file_path, error))
            print(f"❌ Failed to fix {file_path}: {error}")
    
    return success_count, failure_count, failed_files

def regenerate_test_file(family, output_dir=None, validate_syntax=True):
    """
    Regenerate a test file using the fixed generator.
    
    Args:
        family: Model family name (e.g., 'bert', 'gpt2')
        output_dir: Directory to output the generated file
        validate_syntax: Whether to validate syntax
        
    Returns:
        Tuple of (success, output_path)
    """
    if not os.path.exists("test_generator_fixed.py"):
        print("test_generator_fixed.py not found")
        return False, None
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate command
    cmd = [
        sys.executable,
        "test_generator_fixed.py",
        "--family", family
    ]
    
    if output_dir:
        cmd.extend(["--output", output_dir])
    
    try:
        # Run generator
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Generator failed: {result.stderr}")
            return False, None
        
        # Path to the generated file
        output_path = os.path.join(output_dir or ".", f"test_hf_{family}.py")
        
        if not os.path.exists(output_path):
            print(f"Generated file not found: {output_path}")
            return False, None
        
        # Validate syntax if requested
        if validate_syntax:
            success, error = verify_python_syntax(output_path)
            if not success:
                print(f"Syntax error in generated file: {error}")
                return False, output_path
        
        print(f"Successfully generated {output_path}")
        return True, output_path
    
    except Exception as e:
        print(f"Error running generator: {e}")
        return False, None

def integrate_fixes(dry_run=False):
    """
    Integrate fixes into the main generator.
    
    Args:
        dry_run: If True, only show what would be done
    
    Returns:
        Success status
    """
    if not os.path.exists("execute_integration.py"):
        print("execute_integration.py not found")
        return False
    
    cmd = [sys.executable, "execute_integration.py"]
    
    if dry_run:
        cmd.append("--dry-run")
    
    try:
        result = subprocess.run(cmd)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running integration: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fix indentation in HuggingFace test files")
    
    # Action group
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--fix-file", type=str, help="Fix a single file")
    action_group.add_argument("--fix-dir", type=str, help="Fix all files in directory")
    action_group.add_argument("--regenerate", type=str, help="Regenerate test file for model family")
    action_group.add_argument("--integrate", action="store_true", help="Integrate fixes into main generator")
    
    # Additional options
    parser.add_argument("--output", type=str, help="Output directory for regenerated files")
    parser.add_argument("--approach", type=str, choices=["comprehensive", "simple", "minimal"], 
                      default="comprehensive", help="Fix approach")
    parser.add_argument("--no-backup", action="store_true", help="Don't create backups")
    parser.add_argument("--no-validation", action="store_true", help="Skip syntax validation")
    parser.add_argument("--pattern", type=str, default="test_hf_*.py", 
                      help="File pattern when fixing directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set global options
    global BACKUP_FILES, VALIDATE_SYNTAX
    BACKUP_FILES = not args.no_backup
    VALIDATE_SYNTAX = not args.no_validation
    
    if args.fix_file:
        if args.dry_run:
            print(f"Would fix file: {args.fix_file} using {args.approach} approach")
            return 0
        
        success, error = fix_single_file(args.fix_file, args.verbose, args.approach)
        
        if success:
            print(f"✅ Successfully fixed {args.fix_file}")
            return 0
        else:
            print(f"❌ Failed to fix {args.fix_file}: {error}")
            return 1
    
    elif args.fix_dir:
        if args.dry_run:
            print(f"Would fix files matching {args.pattern} in {args.fix_dir} using {args.approach} approach")
            return 0
        
        success_count, failure_count, failed_files = fix_directory(
            args.fix_dir, args.pattern, args.verbose, args.approach
        )
        
        print("\nFix Summary:")
        print(f"- Successful: {success_count}")
        print(f"- Failed: {failure_count}")
        print(f"- Total: {success_count + failure_count}")
        
        if failed_files:
            print("\nFailed files:")
            for file_path, error in failed_files:
                print(f"- {file_path}: {error}")
        
        return 0 if failure_count == 0 else 1
    
    elif args.regenerate:
        if args.dry_run:
            print(f"Would regenerate test file for {args.regenerate} using test_generator_fixed.py")
            if args.output:
                print(f"Output directory would be: {args.output}")
            return 0
        
        success, output_path = regenerate_test_file(
            args.regenerate, args.output, VALIDATE_SYNTAX
        )
        
        if success:
            print(f"✅ Successfully regenerated {output_path}")
            return 0
        else:
            print(f"❌ Failed to regenerate test file for {args.regenerate}")
            return 1
    
    elif args.integrate:
        if args.dry_run:
            print("Would integrate fixes into main generator using execute_integration.py")
            return 0
        
        success = integrate_fixes(args.dry_run)
        
        if success:
            print("✅ Successfully integrated fixes")
            return 0
        else:
            print("❌ Failed to integrate fixes")
            return 1

if __name__ == "__main__":
    sys.exit(main())