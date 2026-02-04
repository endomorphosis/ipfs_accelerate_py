#!/usr/bin/env python3
"""
Automated test directory refactoring script.
Moves files from test/ root to appropriate subdirectories.
"""

import os
import shutil
from pathlib import Path
import subprocess

def categorize_file(filename):
    """Categorize a file based on its name and purpose."""
    
    # Configuration files that should stay in root
    if filename in ['__init__.py', 'conftest.py', 'pytest.ini', 'requirements.txt']:
        return None  # Don't move
    
    # Test files (actual pytest tests)
    if filename.startswith('test_') and not any(x in filename for x in ['template', 'generator', 'helper']):
        # Further categorize by domain
        if any(x in filename for x in ['hf_', 'huggingface']):
            return 'tests/huggingface'
        elif any(x in filename for x in ['hardware', 'cuda', 'gpu', 'cpu', 'npu', 'qualcomm', 'samsung']):
            return 'tests/hardware'
        elif any(x in filename for x in ['api_', 'groq', 'openai', 'claude']):
            return 'tests/api'
        elif any(x in filename for x in ['webgpu', 'webnn', 'browser', 'web_', 'firefox', 'safari']):
            return 'tests/web'
        elif any(x in filename for x in ['ipfs', 'resource_pool', 'p2p']):
            return 'tests/ipfs'
        elif any(x in filename for x in ['mcp_', 'copilot', 'github']):
            return 'tests/mcp'
        elif any(x in filename for x in ['mobile', 'android', 'ios']):
            return 'tests/mobile'
        elif any(x in filename for x in ['integration', 'e2e', 'comprehensive']):
            return 'tests/integration'
        elif any(x in filename for x in ['unit', 'simple', 'basic', 'minimal']):
            return 'tests/unit'
        else:
            return 'tests/other'
    
    # Template files
    if 'template' in filename:
        return 'templates'
    
    # Generator scripts
    if filename.startswith('generate_') or '_generator' in filename:
        return 'generators'
    
    # Utility/helper scripts
    if any(filename.startswith(x) for x in ['fix_', 'check_', 'validate_', 'verify_', 'update_', 'analyze_']):
        return 'scripts/utilities'
    
    # Migration scripts
    if 'migrate' in filename or 'migration' in filename:
        return 'scripts/migration'
    
    # Demo/example files
    if filename.startswith('demo_') or filename.startswith('example_') or 'demo' in filename:
        return 'examples'
    
    # Run scripts
    if filename.startswith('run_'):
        return 'scripts/runners'
    
    # Setup scripts
    if filename.startswith('setup_') or filename.startswith('install_'):
        return 'scripts/setup'
    
    # Build/compile scripts
    if any(x in filename for x in ['build_', 'compile_', 'convert_']):
        return 'scripts/build'
    
    # Monitoring/dashboard scripts
    if any(x in filename for x in ['monitoring', 'dashboard', 'visualization']):
        return 'tools/monitoring'
    
    # Benchmark scripts
    if 'benchmark' in filename:
        return 'tools/benchmarking'
    
    # Model-related utilities
    if any(x in filename for x in ['model_', 'additional_models', 'random_models']):
        return 'tools/models'
    
    # Implementation files
    if 'impl' in filename or 'implementation' in filename:
        return 'implementations'
    
    # Archive scripts
    if 'archive' in filename:
        return 'scripts/archive'
    
    # Documentation builders
    if 'docs' in filename or 'documentation' in filename:
        return 'scripts/docs'
    
    # Default to scripts if unknown
    return 'scripts/other'

def ensure_directory(path):
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    # Create __init__.py if it's a test directory
    if 'tests/' in str(path):
        init_file = path / '__init__.py'
        if not init_file.exists():
            init_file.write_text('"""Test module."""\n')

def move_file_with_git(source, target):
    """Move file using git mv to preserve history."""
    try:
        subprocess.run(['git', 'mv', str(source), str(target)], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        # Fall back to regular move
        shutil.move(str(source), str(target))
        return False

def main():
    """Main refactoring logic."""
    test_dir = Path('test')
    
    # Find all Python files in test root
    py_files = [f for f in test_dir.iterdir() if f.is_file() and f.suffix == '.py']
    
    # Group files by target directory
    moves = {}
    for file in py_files:
        category = categorize_file(file.name)
        if category is None:
            continue  # Skip files that should stay
        
        target_dir = test_dir / category
        if target_dir not in moves:
            moves[target_dir] = []
        moves[target_dir].append(file)
    
    print("=" * 80)
    print("TEST DIRECTORY REFACTORING")
    print("=" * 80)
    print(f"\nTotal files to move: {sum(len(files) for files in moves.values())}")
    print(f"Target directories: {len(moves)}\n")
    
    # Ask for confirmation
    response = input("Proceed with refactoring? (yes/no): ")
    if response.lower() != 'yes':
        print("Refactoring cancelled.")
        return
    
    # Execute moves
    moved_count = 0
    failed_moves = []
    
    for target_dir, files in moves.items():
        print(f"\nMoving {len(files)} files to {target_dir}/")
        ensure_directory(target_dir)
        
        for file in files:
            target_file = target_dir / file.name
            try:
                if target_file.exists():
                    print(f"  SKIP: {file.name} (already exists in target)")
                    continue
                
                move_file_with_git(file, target_file)
                moved_count += 1
                print(f"  ✓ {file.name}")
            except Exception as e:
                failed_moves.append((file, str(e)))
                print(f"  ✗ {file.name}: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("REFACTORING COMPLETE")
    print("=" * 80)
    print(f"Successfully moved: {moved_count} files")
    print(f"Failed moves: {len(failed_moves)} files")
    
    if failed_moves:
        print("\nFailed moves:")
        for file, error in failed_moves:
            print(f"  - {file}: {error}")
    
    print("\nNext steps:")
    print("1. Update imports in moved files")
    print("2. Update imports in files that reference moved files")
    print("3. Run tests to verify")

if __name__ == '__main__':
    main()
