#!/usr/bin/env python3
"""
Batch 2: Move test files to appropriate subdirectories.
"""

import os
from pathlib import Path
import subprocess

def run_command(cmd):
    """Run a shell command."""
    try:
        subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def ensure_directory(path):
    """Ensure directory exists with __init__.py."""
    path.mkdir(parents=True, exist_ok=True)
    if 'tests/' in str(path):
        init_file = path / '__init__.py'
        if not init_file.exists():
            init_file.write_text('"""Test module."""\n')

def categorize_test_file(filename):
    """Categorize a test file."""
    if not filename.startswith('test_'):
        return None
    
    # HuggingFace tests
    if 'hf_' in filename or 'huggingface' in filename:
        return 'tests/huggingface'
    
    # Hardware tests
    if any(x in filename for x in ['hardware', 'cuda', 'gpu', 'cpu', 'npu', 'qualcomm', 'samsung', 'openvino', 'qnn', 'mediatek']):
        return 'tests/hardware'
    
    # API tests
    if any(x in filename for x in ['api_', 'groq', 'openai', 'claude']):
        return 'tests/api'
    
    # Web tests
    if any(x in filename for x in ['webgpu', 'webnn', 'browser', 'web_', 'firefox', 'safari']):
        return 'tests/web'
    
    # IPFS tests
    if any(x in filename for x in ['ipfs', 'resource_pool', 'p2p']):
        return 'tests/ipfs'
    
    # MCP tests
    if any(x in filename for x in ['mcp_', 'copilot', 'github']):
        return 'tests/mcp'
    
    # Mobile tests
    if any(x in filename for x in ['mobile', 'android', 'ios']):
        return 'tests/mobile'
    
    # Integration tests
    if any(x in filename for x in ['integration', 'e2e', 'comprehensive', 'end_to_end']):
        return 'tests/integration'
    
    # Unit tests
    if any(x in filename for x in ['unit', 'simple', 'basic', 'minimal', 'smoke']):
        return 'tests/unit'
    
    # Dashboard tests
    if 'dashboard' in filename or 'visualization' in filename:
        return 'tests/dashboard'
    
    # Model tests
    if any(x in filename for x in ['bert', 'gpt', 'llama', 't5', 'vit', 'clip', 'whisper', 'model_']):
        return 'tests/models'
    
    return 'tests/other'

def move_test_files():
    """Move all test files."""
    test_dir = Path('test')
    
    # Get all test files
    test_files = [f for f in test_dir.iterdir() 
                  if f.is_file() and f.suffix == '.py' and f.name.startswith('test_')]
    
    print(f"Found {len(test_files)} test files to move\n")
    
    # Group by category
    by_category = {}
    for file in test_files:
        category = categorize_test_file(file.name)
        if category:
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(file)
    
    # Move files
    total_moved = 0
    for category, files in sorted(by_category.items()):
        print(f"\n{'=' * 80}")
        print(f"Moving {len(files)} files to {category}/")
        print(f"{'=' * 80}\n")
        
        target_dir = test_dir / category
        ensure_directory(target_dir)
        
        moved = 0
        for file in files:
            target_file = target_dir / file.name
            if target_file.exists():
                print(f"  SKIP: {file.name}")
                continue
            
            if run_command(f'git mv "{file}" "{target_file}"'):
                moved += 1
                total_moved += 1
                print(f"  ✓ {file.name}")
            else:
                print(f"  ✗ {file.name}")
        
        print(f"\nMoved {moved}/{len(files)} files")
    
    print(f"\n{'=' * 80}")
    print(f"TOTAL: Moved {total_moved} test files")
    print(f"{'=' * 80}\n")

def move_remaining_scripts():
    """Move remaining script files."""
    test_dir = Path('test')
    
    # Get all remaining Python files (excluding config)
    remaining = [f for f in test_dir.iterdir() 
                 if f.is_file() and f.suffix == '.py' 
                 and f.name not in ['__init__.py', 'conftest.py', 'pytest.ini']]
    
    if not remaining:
        print("No remaining files to move")
        return
    
    print(f"\n{'=' * 80}")
    print(f"Moving {len(remaining)} remaining files to scripts/other/")
    print(f"{'=' * 80}\n")
    
    target_dir = test_dir / 'scripts' / 'other'
    ensure_directory(target_dir)
    
    moved = 0
    for file in remaining:
        target_file = target_dir / file.name
        if target_file.exists():
            print(f"  SKIP: {file.name}")
            continue
        
        if run_command(f'git mv "{file}" "{target_file}"'):
            moved += 1
            print(f"  ✓ {file.name}")
        else:
            print(f"  ✗ {file.name}")
    
    print(f"\nMoved {moved}/{len(remaining)} files")

def main():
    """Main execution."""
    print("=" * 80)
    print("BATCH 2: MOVE TEST FILES")
    print("=" * 80)
    
    move_test_files()
    move_remaining_scripts()
    
    print("\n" + "=" * 80)
    print("PHASE 2 COMPLETE")
    print("=" * 80)
    print("\nNext: Run update_imports.py to fix all imports")

if __name__ == '__main__':
    main()
