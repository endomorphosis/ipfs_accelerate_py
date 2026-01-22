#!/usr/bin/env python3
"""
Validate and Fix Templates

This script combines template validation and indentation fixing.
It can be used to validate templates and fix indentation issues.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import validator and fixer
from generators.validators.template_validator_integration import (
    validate_template_for_generator,
    validate_template_file_for_generator
)
from generators.validators.fix_template_indentation import (
    fix_file,
    fix_directory,
    identify_template_variables
)

def validate_and_fix_file(file_path: str, generator_type: str = "generic", 
                         fix_indent: bool = True, dry_run: bool = False,
                         strict_validation: bool = False) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate and fix a file.
    
    Args:
        file_path: Path to the file
        generator_type: Type of generator
        fix_indent: Whether to fix indentation
        dry_run: If True, only report changes without modifying file
        strict_validation: Whether to enforce strict validation
        
    Returns:
        Tuple of (success, report)
    """
    report = {
        "file": file_path,
        "validation": {
            "success": False,
            "errors": []
        },
        "indentation": {
            "fixed": 0
        }
    }
    
    # Validate the file
    validation_success, validation_errors = validate_template_file_for_generator(
        file_path,
        generator_type=generator_type,
        strict_indentation=strict_validation
    )
    
    report["validation"]["success"] = validation_success
    report["validation"]["errors"] = validation_errors
    
    if validation_success:
        logger.info(f"Template validation passed for {file_path}")
    else:
        logger.warning(f"Template validation failed for {file_path}")
        for error in validation_errors:
            logger.warning(f"  - {error}")
    
    # Fix indentation if requested
    if fix_indent:
        indent_success, fixed_count = fix_file(file_path, dry_run)
        report["indentation"]["success"] = indent_success
        report["indentation"]["fixed"] = fixed_count
        
        if indent_success:
            if fixed_count > 0:
                if dry_run:
                    logger.info(f"Would fix {fixed_count} templates in {file_path} (dry run)")
                else:
                    logger.info(f"Fixed {fixed_count} templates in {file_path}")
            else:
                logger.info(f"No templates needed fixing in {file_path}")
        else:
            logger.error(f"Failed to fix indentation in {file_path}")
    
    return validation_success and (not fix_indent or report["indentation"].get("success", False)), report

def validate_and_fix_directory(directory: str, pattern: str = "*.py", 
                              generator_type: str = "generic", 
                              fix_indent: bool = True, dry_run: bool = False,
                              strict_validation: bool = False) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate and fix files in a directory.
    
    Args:
        directory: Path to the directory
        pattern: Glob pattern for files
        generator_type: Type of generator
        fix_indent: Whether to fix indentation
        dry_run: If True, only report changes without modifying file
        strict_validation: Whether to enforce strict validation
        
    Returns:
        Tuple of (success, report)
    """
    import glob
    
    report = {
        "directory": directory,
        "pattern": pattern,
        "files": [],
        "validation": {
            "success_count": 0,
            "fail_count": 0
        },
        "indentation": {
            "fixed_count": 0
        }
    }
    
    try:
        # Find matching files
        file_pattern = os.path.join(directory, pattern)
        files = glob.glob(file_pattern, recursive=True)
        
        if not files:
            logger.info(f"No files matching {file_pattern} found")
            return True, report
            
        logger.info(f"Found {len(files)} files matching {file_pattern}")
        
        # Process each file
        success_count = 0
        validation_success_count = 0
        validation_fail_count = 0
        indentation_fixed_count = 0
        
        for file_path in files:
            success, file_report = validate_and_fix_file(
                file_path,
                generator_type=generator_type,
                fix_indent=fix_indent,
                dry_run=dry_run,
                strict_validation=strict_validation
            )
            
            report["files"].append(file_report)
            
            if success:
                success_count += 1
                
            if file_report["validation"]["success"]:
                validation_success_count += 1
            else:
                validation_fail_count += 1
                
            if fix_indent:
                indentation_fixed_count += file_report["indentation"]["fixed"]
        
        report["validation"]["success_count"] = validation_success_count
        report["validation"]["fail_count"] = validation_fail_count
        report["indentation"]["fixed_count"] = indentation_fixed_count
        
        logger.info(f"Validation results: {validation_success_count} passed, {validation_fail_count} failed")
        
        if fix_indent:
            if dry_run:
                logger.info(f"Would fix {indentation_fixed_count} templates in {len(files)} files (dry run)")
            else:
                logger.info(f"Fixed {indentation_fixed_count} templates in {len(files)} files")
        
        return validation_success_count > 0, report
    except Exception as e:
        logger.error(f"Error processing directory {directory}: {str(e)}")
        return False, report

def main():
    """Main function for standalone usage"""
    parser = argparse.ArgumentParser(description="Validate and Fix Templates")
    parser.add_argument("--file", type=str, help="Path to a specific file to process")
    parser.add_argument("--directory", type=str, help="Path to a directory to process")
    parser.add_argument("--pattern", type=str, default="**/*.py", help="Glob pattern for files when using --directory")
    parser.add_argument("--generator-type", type=str, default="generic", help="Type of generator to validate for")
    parser.add_argument("--no-fix", action="store_true", help="Skip indentation fixing")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually modify files, just report what would be changed")
    parser.add_argument("--strict", action="store_true", help="Enforce strict validation")
    parser.add_argument("--json", type=str, help="Output JSON report to file")
    
    args = parser.parse_args()
    
    if args.file:
        # Process a single file
        success, report = validate_and_fix_file(
            args.file,
            generator_type=args.generator_type,
            fix_indent=not args.no_fix,
            dry_run=args.dry_run,
            strict_validation=args.strict
        )
    elif args.directory:
        # Process a directory
        success, report = validate_and_fix_directory(
            args.directory,
            pattern=args.pattern,
            generator_type=args.generator_type,
            fix_indent=not args.no_fix,
            dry_run=args.dry_run,
            strict_validation=args.strict
        )
    else:
        parser.print_help()
        return 0
        
    # Output JSON report if requested
    if args.json:
        import json
        
        with open(args.json, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Report saved to {args.json}")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())