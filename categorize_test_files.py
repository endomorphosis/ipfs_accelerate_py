#!/usr/bin/env python3
"""
Categorize test/ root files into appropriate subdirectories.
This script analyzes files and creates a refactoring plan.
"""

import os
import re
from pathlib import Path
from collections import defaultdict

def categorize_file(filename):
    """Categorize a file based on its name and purpose."""
    
    # Configuration files that should stay in root
    if filename in ['__init__.py', 'conftest.py', 'pytest.ini', 'requirements.txt']:
        return 'config_root'
    
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

def main():
    """Main categorization logic."""
    test_dir = Path('test')
    
    # Find all Python files in test root
    py_files = [f for f in test_dir.iterdir() if f.is_file() and f.suffix == '.py']
    
    # Categorize files
    categories = defaultdict(list)
    for file in py_files:
        category = categorize_file(file.name)
        categories[category].append(file.name)
    
    # Print categorization
    print("=" * 80)
    print("TEST DIRECTORY FILE CATEGORIZATION")
    print("=" * 80)
    print(f"\nTotal Python files in test/ root: {len(py_files)}\n")
    
    for category in sorted(categories.keys()):
        files = sorted(categories[category])
        print(f"\n{category.upper()} ({len(files)} files)")
        print("-" * 80)
        for file in files[:10]:  # Show first 10
            print(f"  - {file}")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more")
    
    # Create refactoring plan
    print("\n" + "=" * 80)
    print("REFACTORING PLAN")
    print("=" * 80)
    
    for category in sorted(categories.keys()):
        if category == 'config_root':
            continue
        files = categories[category]
        target_dir = f"test/{category}"
        print(f"\n{len(files)} files → {target_dir}/")
    
    # Save detailed plan to file
    with open('/tmp/refactoring_plan.txt', 'w') as f:
        for category in sorted(categories.keys()):
            if category == 'config_root':
                continue
            files = sorted(categories[category])
            target_dir = f"test/{category}"
            f.write(f"\n# {target_dir}/ ({len(files)} files)\n")
            for file in files:
                f.write(f"test/{file} -> {target_dir}/{file}\n")
    
    print(f"\n\nDetailed plan saved to /tmp/refactoring_plan.txt")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    move_count = sum(len(files) for cat, files in categories.items() if cat != 'config_root')
    keep_count = len(categories.get('config_root', []))
    print(f"Files to move: {move_count}")
    print(f"Files to keep in root: {keep_count}")
    print(f"Total: {move_count + keep_count}")

if __name__ == '__main__':
    main()
