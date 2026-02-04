#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup CI Workflows Utility

This script installs the CI workflow files for mobile testing into the GitHub workflows
directory. It also verifies that the workflows are properly configured.

Usage:
    python setup_ci_workflows.py [--install] [--verify] [--dry-run]

Examples:
    # Verify workflow files
    python setup_ci_workflows.py --verify
    
    # Install workflow files
    python setup_ci_workflows.py --install
    
    # Show what would be installed without making changes
    python setup_ci_workflows.py --install --dry-run

Date: May 2025
"""

import os
import sys
import shutil
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Workflow file locations
SOURCE_FILES = {
    "android": "test/android_test_harness/android_ci_workflow.yml",
    "ios": "test/ios_test_harness/ios_ci_workflow.yml",
    "cross_platform": "test/mobile_cross_platform_workflow.yml",
    "setup_runners": "test/setup_mobile_ci_runners_workflow.yml"
}

# Target workflow directory
WORKFLOW_DIR = ".github/workflows"

# Target file names
TARGET_FILES = {
    "android": "android_mobile_ci.yml",
    "ios": "ios_mobile_ci.yml",
    "cross_platform": "mobile_cross_platform_workflow.yml",
    "setup_runners": "setup_mobile_ci_runners.yml"
}

def verify_workflow_files() -> bool:
    """
    Verify that the workflow files exist and are valid YAML.
    
    Returns:
        Success status
    """
    all_valid = True
    
    for workflow_type, source_path in SOURCE_FILES.items():
        logger.info(f"Verifying {workflow_type} workflow file: {source_path}")
        
        # Check if file exists
        if not os.path.exists(source_path):
            logger.error(f"Workflow file does not exist: {source_path}")
            all_valid = False
            continue
        
        # Check if file is valid YAML
        try:
            with open(source_path, 'r') as f:
                yaml_content = yaml.safe_load(f)
            
            # Check for required sections
            if not isinstance(yaml_content, dict):
                logger.error(f"Invalid YAML format in {source_path}: not a dictionary")
                all_valid = False
                continue
            
            if "name" not in yaml_content:
                logger.warning(f"Missing 'name' in {source_path}")
            
            if "on" not in yaml_content:
                logger.error(f"Missing 'on' trigger in {source_path}")
                all_valid = False
                continue
            
            if "jobs" not in yaml_content:
                logger.error(f"Missing 'jobs' section in {source_path}")
                all_valid = False
                continue
            
            logger.info(f"✅ Workflow file is valid: {source_path}")
        
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in {source_path}: {e}")
            all_valid = False
        
        except Exception as e:
            logger.error(f"Error verifying {source_path}: {e}")
            all_valid = False
    
    return all_valid

def check_workflow_directory() -> Tuple[bool, str]:
    """
    Check if the workflow directory exists, create it if needed.
    
    Returns:
        Tuple of (success status, workflow directory path)
    """
    # Get absolute path to workflow directory
    workflow_dir = os.path.abspath(WORKFLOW_DIR)
    
    # Check if directory exists
    if os.path.exists(workflow_dir):
        if not os.path.isdir(workflow_dir):
            logger.error(f"Workflow path exists but is not a directory: {workflow_dir}")
            return False, workflow_dir
        
        logger.info(f"Workflow directory exists: {workflow_dir}")
        return True, workflow_dir
    
    # Create directory
    try:
        os.makedirs(workflow_dir, exist_ok=True)
        logger.info(f"Created workflow directory: {workflow_dir}")
        return True, workflow_dir
    
    except Exception as e:
        logger.error(f"Error creating workflow directory: {e}")
        return False, workflow_dir

def install_workflow_files(dry_run: bool = False) -> bool:
    """
    Install workflow files to the GitHub workflows directory.
    
    Args:
        dry_run: If True, show what would be done without making changes
        
    Returns:
        Success status
    """
    # Verify workflow files first
    if not verify_workflow_files():
        logger.error("Workflow verification failed. Please fix errors before installing.")
        return False
    
    # Check workflow directory
    dir_ok, workflow_dir = check_workflow_directory()
    if not dir_ok:
        logger.error("Workflow directory check failed")
        return False
    
    # Install each workflow file
    all_installed = True
    
    for workflow_type, source_path in SOURCE_FILES.items():
        target_file = TARGET_FILES[workflow_type]
        target_path = os.path.join(workflow_dir, target_file)
        
        logger.info(f"Installing {workflow_type} workflow: {source_path} -> {target_path}")
        
        if dry_run:
            logger.info(f"[DRY RUN] Would copy {source_path} to {target_path}")
            continue
        
        try:
            shutil.copy2(source_path, target_path)
            logger.info(f"✅ Installed workflow file: {target_path}")
        
        except Exception as e:
            logger.error(f"Error installing {target_path}: {e}")
            all_installed = False
    
    return all_installed

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Setup CI Workflows Utility")
    
    parser.add_argument("--verify", action="store_true", help="Verify workflow files")
    parser.add_argument("--install", action="store_true", help="Install workflow files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Default action is verify if none specified
    if not args.verify and not args.install:
        args.verify = True
    
    # Run requested actions
    if args.verify:
        logger.info("Verifying workflow files")
        if verify_workflow_files():
            logger.info("✅ All workflow files are valid")
        else:
            logger.error("❌ Some workflow files are invalid")
            return 1
    
    if args.install:
        logger.info(f"Installing workflow files (dry run: {args.dry_run})")
        if install_workflow_files(args.dry_run):
            logger.info("✅ Workflow files installed successfully")
        else:
            logger.error("❌ Failed to install workflow files")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())