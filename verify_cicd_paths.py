#\!/usr/bin/env python
"""
Script to verify that all file paths referenced in CI/CD workflows exist in the new directory structure.
Usage: python verify_cicd_paths.py [project_root]

If project_root is not provided, the current directory is used.
"""

import os
import re
import glob
import sys
import argparse
from pathlib import Path

# Simple color output functions without dependencies
def print_success(message):
    print(f"✅ {message}")

def print_warning(message):
    print(f"⚠️ {message}")

def print_error(message):
    print(f"❌ {message}")

def print_info(message):
    print(f"ℹ️ {message}")

def find_python_scripts_in_workflows(project_root):
    """Find all Python script references in workflow files."""
    workflow_dir = project_root / '.github' / 'workflows'
    python_script_pattern = re.compile(r'python\s+([^\s]+\.py)')
    script_paths = set()
    
    for workflow_file in workflow_dir.glob('*.yml'):
        print_info(f"Analyzing {workflow_file}")
        content = workflow_file.read_text()
        
        # Find all Python script references
        matches = python_script_pattern.findall(content)
        for match in matches:
            if not match.startswith('$'):  # Skip template variables
                script_paths.add(match)
    
    return script_paths

def verify_paths(script_paths, project_root):
    """Verify that all referenced script paths exist."""
    missing_paths = []
    found_paths = []
    
    for script_path in sorted(script_paths):
        full_path = project_root / script_path
        if os.path.exists(full_path):
            found_paths.append(script_path)
        else:
            missing_paths.append(script_path)
    
    return found_paths, missing_paths

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Verify file paths in CI/CD workflows')
    parser.add_argument('project_root', nargs='?', default='.', 
                        help='Path to the project root directory (default: current directory)')
    args = parser.parse_args()
    
    project_root = Path(args.project_root).resolve()
    print_info(f"Using project root: {project_root}")
    
    # Find all Python script references
    script_paths = find_python_scripts_in_workflows(project_root)
    print_info(f"Found {len(script_paths)} Python script references in workflows")
    
    # Verify paths exist
    found_paths, missing_paths = verify_paths(script_paths, project_root)
    
    # Print results
    if found_paths:
        print_success(f"Found {len(found_paths)} scripts:")
        for path in found_paths:
            print(f"  - {path}")
    
    if missing_paths:
        print_error(f"Missing {len(missing_paths)} scripts:")
        for path in missing_paths:
            print(f"  - {path}")
            
            # Try to suggest where the file might be
            base_name = os.path.basename(path)
            possible_locations = list(glob.glob(str(project_root / "**" / base_name), recursive=True))
            if possible_locations:
                print_warning(f"    Possible locations:")
                for loc in possible_locations[:3]:  # Show only top 3 matches
                    print(f"    - {os.path.relpath(loc, project_root)}")
    
    # Summary
    print("\n" + "="*50)
    if missing_paths:
        print_error(f"Summary: {len(found_paths)} scripts found, {len(missing_paths)} scripts missing")
        return 1
    else:
        print_success(f"All {len(found_paths)} referenced scripts found\!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
