#!/usr/bin/env python3
"""
Comprehensive test fixing script that combines regeneration and indentation fixing.

This script provides a unified interface for:
1. Regenerating test files with the fixed generator
2. Fixing indentation in existing test files
3. Verifying syntax of all test files
4. Integrating fixes into the main generator
5. Reporting comprehensive statistics

Usage:
    python comprehensive_test_fix.py [COMMAND] [OPTIONS]

Commands:
    regenerate   Regenerate test files for specific model families
    fix          Fix indentation in existing test files
    integrate    Integrate fixes into the main generator
    verify       Verify syntax of test files
    all          Run all operations in sequence
"""

import os
import sys
import glob
import re
import shutil
import subprocess
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"comprehensive_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Core supported model families
BASE_MODEL_FAMILIES = ["bert", "gpt2", "t5", "vit"]

def execute_command(cmd, description=None):
    """Execute a shell command and return the result."""
    if description:
        logger.info(f"{description}...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            if description:
                logger.info(f"✅ {description} succeeded")
            return True, result.stdout
        else:
            if description:
                logger.error(f"❌ {description} failed: {result.stderr}")
            return False, result.stderr
    except Exception as e:
        if description:
            logger.error(f"❌ {description} error: {e}")
        return False, str(e)

def find_test_files(directory, pattern):
    """Find test files matching the pattern."""
    search_pattern = os.path.join(directory, pattern)
    files = glob.glob(search_pattern)
    return sorted(files)

def verify_syntax(file_path):
    """Verify Python syntax of a file."""
    cmd = [sys.executable, "-m", "py_compile", file_path]
    success, output = execute_command(cmd)
    if success:
        logger.info(f"✅ {file_path}: Syntax is valid")
    else:
        logger.error(f"❌ {file_path}: Syntax error")
    return success

def regenerate_test_file(family, output_dir, fix_indentation=True, verify=True):
    """
    Regenerate a test file for a specific model family.
    
    Args:
        family: Model family name (bert, gpt2, t5, vit)
        output_dir: Output directory for the generated file
        fix_indentation: Whether to fix indentation after regeneration
        verify: Whether to verify syntax after fixing
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command for regeneration
    regenerate_script = "test_generator_fixed.py"
    cmd = [sys.executable, regenerate_script, "--family", family, "--output", output_dir]
    
    # Run regeneration
    success, output = execute_command(
        cmd, 
        description=f"Regenerating test file for {family}"
    )
    
    if not success:
        return False
    
    # Get path to generated file
    output_path = os.path.join(output_dir, f"test_hf_{family}.py")
    
    # Fix indentation if requested
    if fix_indentation and os.path.exists(output_path):
        fix_script = "complete_indentation_fix.py"
        fix_cmd = [sys.executable, fix_script, output_path]
        
        fix_success, fix_output = execute_command(
            fix_cmd, 
            description=f"Fixing indentation in {output_path}"
        )
        
        if not fix_success:
            return False
    
    # Verify syntax if requested
    if verify and os.path.exists(output_path):
        verify_success = verify_syntax(output_path)
        return verify_success
    
    return True

def fix_test_files(directory, pattern, verify=True, force=False):
    """
    Fix indentation in existing test files.
    
    Args:
        directory: Directory containing test files
        pattern: File pattern to match
        verify: Whether to verify syntax after fixing
        force: Whether to run even if backup already exists
    
    Returns:
        Tuple of (num_fixed, num_failed, total)
    """
    # Find test files
    files = find_test_files(directory, pattern)
    logger.info(f"Found {len(files)} files matching pattern {pattern}")
    
    fixed = []
    failed = []
    skipped = []
    
    for file_path in files:
        # Check if backup already exists
        backup_path = f"{file_path}.bak"
        if os.path.exists(backup_path) and not force:
            logger.info(f"Skipping {file_path} - backup already exists (use --force to override)")
            skipped.append(file_path)
            continue
        
        # Run fix command
        fix_script = "complete_indentation_fix.py"
        fix_cmd = [sys.executable, fix_script, file_path]
        
        if verify:
            fix_cmd.append("--verify")
        
        fix_success, fix_output = execute_command(
            fix_cmd, 
            description=f"Fixing indentation in {file_path}"
        )
        
        if fix_success:
            fixed.append(file_path)
        else:
            failed.append(file_path)
    
    # Print summary
    logger.info("\nIndentation Fix Summary:")
    logger.info(f"- Fixed: {len(fixed)} files")
    logger.info(f"- Failed: {len(failed)} files")
    logger.info(f"- Skipped: {len(skipped)} files")
    logger.info(f"- Total: {len(files)} files")
    
    if failed:
        logger.info("\nFailed files:")
        for f in failed:
            logger.info(f"  - {f}")
    
    return len(fixed), len(failed), len(files)

def verify_all_test_files(directory, pattern):
    """
    Verify syntax of all test files.
    
    Args:
        directory: Directory containing test files
        pattern: File pattern to match
    
    Returns:
        Tuple of (num_valid, num_invalid, total)
    """
    # Find test files
    files = find_test_files(directory, pattern)
    logger.info(f"Found {len(files)} files matching pattern {pattern}")
    
    valid = []
    invalid = []
    
    for file_path in files:
        if verify_syntax(file_path):
            valid.append(file_path)
        else:
            invalid.append(file_path)
    
    # Print summary
    logger.info("\nSyntax Verification Summary:")
    logger.info(f"- Valid: {len(valid)} files")
    logger.info(f"- Invalid: {len(invalid)} files")
    logger.info(f"- Total: {len(files)} files")
    
    if invalid:
        logger.info("\nInvalid files:")
        for f in invalid:
            logger.info(f"  - {f}")
    
    return len(valid), len(invalid), len(files)

def integrate_fixes(dry_run=False):
    """
    Integrate the fixes into the main generator.
    
    Args:
        dry_run: Whether to perform a dry run without making changes
    
    Returns:
        bool: True if successful, False otherwise
    """
    integrate_script = "integrate_generator_fixes.py"
    
    # Check if integration script exists
    if not os.path.exists(integrate_script):
        logger.error(f"Integration script not found: {integrate_script}")
        return False
    
    # Build command
    cmd = [sys.executable, integrate_script]
    if dry_run:
        cmd.append("--dry-run")
    
    # Run integration
    success, output = execute_command(
        cmd, 
        description="Integrating fixes into main generator"
    )
    
    if success:
        logger.info(output)
        logger.info("✅ Successfully integrated fixes into main generator")
    
    return success

def run_comprehensive_fix(args):
    """
    Run the comprehensive fix workflow.
    
    Args:
        args: Command-line arguments
    
    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    command = args.command.lower()
    success = True
    
    if command == "regenerate" or command == "all":
        # Regenerate test files
        logger.info("\n===== REGENERATING TEST FILES =====\n")
        
        families = args.families or BASE_MODEL_FAMILIES
        
        if args.all_families:
            # In the real implementation, this would use all available families
            families = BASE_MODEL_FAMILIES
        
        regenerated = []
        failed = []
        
        for family in families:
            if regenerate_test_file(family, args.output_dir, fix_indentation=not args.no_fix, verify=args.verify):
                regenerated.append(family)
            else:
                failed.append(family)
                success = False
        
        # Print summary
        logger.info("\nRegeneration Summary:")
        logger.info(f"- Regenerated: {len(regenerated)} files")
        logger.info(f"- Failed: {len(failed)} files")
        logger.info(f"- Total: {len(families)} files")
        
        if failed:
            logger.info("\nFailed families:")
            for f in failed:
                logger.info(f"  - {f}")
    
    if command == "fix" or command == "all":
        # Fix existing test files
        logger.info("\n===== FIXING TEST FILES =====\n")
        
        fixed, failed, total = fix_test_files(
            directory=args.directory,
            pattern=args.pattern,
            verify=args.verify,
            force=args.force
        )
        
        if failed > 0:
            success = False
    
    if command == "verify" or command == "all":
        # Verify syntax of all test files
        logger.info("\n===== VERIFYING TEST FILES =====\n")
        
        valid, invalid, total = verify_all_test_files(
            directory=args.directory,
            pattern=args.pattern
        )
        
        if invalid > 0:
            success = False
    
    if command == "integrate" or command == "all":
        # Integrate fixes into main generator
        logger.info("\n===== INTEGRATING FIXES =====\n")
        
        if not integrate_fixes(dry_run=args.dry_run):
            success = False
    
    return 0 if success else 1

def main():
    parser = argparse.ArgumentParser(description="Comprehensive test fixing script")
    
    # Main command argument
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    subparsers.required = True
    
    # Regenerate command
    regen_parser = subparsers.add_parser("regenerate", help="Regenerate test files")
    regen_parser.add_argument("--families", nargs="+", help=f"Model families to regenerate")
    regen_parser.add_argument("--all-families", action="store_true", help="Regenerate all known model families")
    regen_parser.add_argument("--output-dir", type=str, default="fixed_tests", help="Output directory for test files")
    regen_parser.add_argument("--no-fix", action="store_true", help="Skip fixing indentation after regeneration")
    regen_parser.add_argument("--verify", action="store_true", help="Verify syntax after regeneration")
    
    # Fix command
    fix_parser = subparsers.add_parser("fix", help="Fix indentation in existing test files")
    fix_parser.add_argument("--pattern", type=str, default="test_hf_*.py", help="File pattern to match")
    fix_parser.add_argument("--directory", type=str, default=".", help="Directory containing test files")
    fix_parser.add_argument("--verify", action="store_true", help="Verify syntax after fixing")
    fix_parser.add_argument("--force", action="store_true", help="Run even if backup already exists")
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify syntax of test files")
    verify_parser.add_argument("--pattern", type=str, default="test_hf_*.py", help="File pattern to match")
    verify_parser.add_argument("--directory", type=str, default=".", help="Directory containing test files")
    
    # Integrate command
    integrate_parser = subparsers.add_parser("integrate", help="Integrate fixes into main generator")
    integrate_parser.add_argument("--dry-run", action="store_true", help="Show what would be integrated without making changes")
    
    # All command (runs all operations)
    all_parser = subparsers.add_parser("all", help="Run all operations in sequence")
    all_parser.add_argument("--families", nargs="+", help=f"Model families to regenerate")
    all_parser.add_argument("--all-families", action="store_true", help="Regenerate all known model families")
    all_parser.add_argument("--output-dir", type=str, default="fixed_tests", help="Output directory for test files")
    all_parser.add_argument("--no-fix", action="store_true", help="Skip fixing indentation after regeneration")
    all_parser.add_argument("--pattern", type=str, default="test_hf_*.py", help="File pattern to match for fixing")
    all_parser.add_argument("--directory", type=str, default=".", help="Directory containing test files")
    all_parser.add_argument("--verify", action="store_true", help="Verify syntax after operations")
    all_parser.add_argument("--force", action="store_true", help="Run even if backup already exists")
    all_parser.add_argument("--dry-run", action="store_true", help="Show what would be integrated without making changes")
    
    args = parser.parse_args()
    
    return run_comprehensive_fix(args)

if __name__ == "__main__":
    sys.exit(main())