#!/usr/bin/env python3
"""
Next Batch Migration Script

This script identifies and migrates the next batch of test files to the refactored test suite.
It prioritizes files based on the migration plan and focuses on model tests first.
"""

import os
import sys
import argparse
import glob
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

# Constants
DEFAULT_BATCH_SIZE = 5
REFACTORED_DIR = 'refactored_test_suite'
MIGRATION_PLAN_PATH = os.path.join(REFACTORED_DIR, 'migration_plan.json')
CATEGORIES_PRIORITY = ['models', 'hardware', 'api', 'browser', 'resource_pool', 'integration', 'unit']

def find_candidate_files(limit: int = DEFAULT_BATCH_SIZE) -> List[str]:
    """
    Find the next batch of files to migrate based on priority order.
    
    Args:
        limit: Maximum number of files to return
    
    Returns:
        List of file paths
    """
    # Find all test files
    all_test_files = glob.glob('test_*.py', recursive=False)
    
    # Add files from the test directory
    all_test_files.extend(glob.glob('test/test_*.py', recursive=False))
    
    # Filter out files that are already in the refactored directory
    refactored_files = set()
    for root, _, files in os.walk(REFACTORED_DIR):
        for file in files:
            if file.startswith('test_') and file.endswith('.py'):
                refactored_files.add(file)
    
    # Filter out already refactored files
    candidates = [f for f in all_test_files if os.path.basename(f) not in refactored_files]
    
    # Categorize files by priority
    categorized_files = {category: [] for category in CATEGORIES_PRIORITY}
    
    for file_path in candidates:
        # Determine category based on filename
        file_name = os.path.basename(file_path).lower()
        
        # Check each category's keywords
        assigned = False
        for category in CATEGORIES_PRIORITY:
            if category in file_name:
                categorized_files[category].append(file_path)
                assigned = True
                break
        
        # Look for more specific model keywords if not assigned yet
        if not assigned:
            model_keywords = ['bert', 'vit', 'gpt', 'llama', 't5', 'clip', 'whisper', 'wav2vec2']
            if any(keyword in file_name for keyword in model_keywords):
                categorized_files['models'].append(file_path)
                assigned = True
        
        # Look for hardware keywords
        if not assigned:
            hardware_keywords = ['webgpu', 'webnn', 'cuda', 'cpu', 'mps', 'rocm', 'openvino']
            if any(keyword in file_name for keyword in hardware_keywords):
                categorized_files['hardware'].append(file_path)
                assigned = True
        
        # Look for API keywords
        if not assigned:
            api_keywords = ['api', 'claude', 'groq', 'ollama', 'openai']
            if any(keyword in file_name for keyword in api_keywords):
                categorized_files['api'].append(file_path)
                assigned = True
                
        # Look for browser keywords
        if not assigned:
            browser_keywords = ['browser', 'chrome', 'firefox', 'edge', 'safari']
            if any(keyword in file_name for keyword in browser_keywords):
                categorized_files['browser'].append(file_path)
                assigned = True
        
        # Default to unit tests if not categorized
        if not assigned:
            categorized_files['unit'].append(file_path)
    
    # Prioritize specific model tests (VIT, T5 as mentioned in migration progress)
    priority_models = ['vit', 't5']
    priority_files = []
    for keyword in priority_models:
        for file_path in categorized_files['models']:
            if keyword in os.path.basename(file_path).lower():
                priority_files.append(file_path)
                categorized_files['models'].remove(file_path)
    
    # Place priority files at the beginning of models category
    categorized_files['models'] = priority_files + categorized_files['models']
    
    # Select files by priority until we reach the limit
    selected_files = []
    for category in CATEGORIES_PRIORITY:
        if len(selected_files) >= limit:
            break
        
        # Add files from this category up to the limit
        available_slots = limit - len(selected_files)
        selected_files.extend(categorized_files[category][:available_slots])
    
    return selected_files[:limit]

def run_migration(files: List[str], dry_run: bool = False) -> None:
    """
    Run the migration script for the selected files.
    
    Args:
        files: List of file paths to migrate
        dry_run: If True, don't actually migrate, just show what would be done
    """
    # Convert file paths to absolute paths
    abs_files = [os.path.abspath(f) for f in files]
    
    # Build command
    cmd = [
        sys.executable, 
        'migrate_tests.py',
        '--files', 
        *abs_files
    ]
    
    if dry_run:
        cmd.append('--dry-run')
    
    # Run migration script
    print(f"Running migration for {len(files)} files:")
    for f in files:
        print(f"  {f}")
    
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description='Migrate the next batch of test files')
    parser.add_argument('--limit', type=int, default=DEFAULT_BATCH_SIZE,
                      help=f'Maximum number of files to migrate (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--dry-run', action='store_true',
                      help='Run in dry-run mode without actual migration')
    parser.add_argument('--show-only', action='store_true',
                      help='Just show which files would be migrated, without running migration')
    
    args = parser.parse_args()
    
    # Find candidate files
    candidates = find_candidate_files(args.limit)
    
    if args.show_only:
        print(f"Next {len(candidates)} files to migrate:")
        for file in candidates:
            print(f"  {file}")
        return
    
    # Run migration
    run_migration(candidates, args.dry_run)

if __name__ == '__main__':
    main()