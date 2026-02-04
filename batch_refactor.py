#!/usr/bin/env python3
"""
Batch refactoring script - executes refactoring in safe batches.
"""

import os
import shutil
from pathlib import Path
import subprocess
import sys

def run_command(cmd, capture=True):
    """Run a shell command."""
    try:
        if capture:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
            return result.stdout
        else:
            subprocess.run(cmd, shell=True, check=True)
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Error: {e}")
        return None

def ensure_directory(path):
    """Ensure directory exists with __init__.py."""
    path.mkdir(parents=True, exist_ok=True)
    if 'tests/' in str(path) or 'test/' in str(path):
        init_file = path / '__init__.py'
        if not init_file.exists():
            init_file.write_text('"""Test module."""\n')

def move_files_batch(files, target_dir, batch_name):
    """Move a batch of files."""
    print(f"\n{'=' * 80}")
    print(f"BATCH: {batch_name}")
    print(f"{'=' * 80}")
    print(f"Moving {len(files)} files to {target_dir}/\n")
    
    ensure_directory(target_dir)
    
    moved = 0
    skipped = 0
    failed = []
    
    for file in files:
        target_file = target_dir / file.name
        
        if target_file.exists():
            print(f"  SKIP: {file.name} (already exists)")
            skipped += 1
            continue
        
        try:
            # Use git mv to preserve history
            result = run_command(f'git mv "{file}" "{target_file}"')
            if result is not None:
                moved += 1
                print(f"  ✓ {file.name}")
            else:
                failed.append((file.name, "git mv failed"))
                print(f"  ✗ {file.name} (git mv failed)")
        except Exception as e:
            failed.append((file.name, str(e)))
            print(f"  ✗ {file.name}: {e}")
    
    print(f"\nBatch summary: {moved} moved, {skipped} skipped, {len(failed)} failed")
    
    if failed:
        print("\nFailed moves:")
        for name, error in failed:
            print(f"  - {name}: {error}")
    
    return moved, skipped, failed

def main():
    """Execute batch refactoring."""
    test_dir = Path('test')
    
    if not test_dir.exists():
        print(f"Error: {test_dir} does not exist")
        return 1
    
    print("=" * 80)
    print("BATCH REFACTORING - TEST DIRECTORY")
    print("=" * 80)
    
    # Batch 1: Templates (23 files) - Low risk, no dependencies
    print("\n\n### PHASE 1: TEMPLATES AND GENERATORS ###\n")
    
    template_files = [f for f in test_dir.iterdir() 
                     if f.is_file() and f.suffix == '.py' and 'template' in f.name]
    if template_files:
        move_files_batch(template_files, test_dir / 'templates', "Templates")
    
    # Batch 2: Generators (24 files)
    generator_files = [f for f in test_dir.iterdir() 
                      if f.is_file() and f.suffix == '.py' 
                      and (f.name.startswith('generate_') or '_generator' in f.name)]
    if generator_files:
        move_files_batch(generator_files, test_dir / 'generators', "Generators")
    
    # Batch 3: Examples (11 files)
    example_files = [f for f in test_dir.iterdir() 
                    if f.is_file() and f.suffix == '.py'
                    and (f.name.startswith('demo_') or f.name.startswith('example_') or 'demo' in f.name)]
    if example_files:
        move_files_batch(example_files, test_dir / 'examples', "Examples & Demos")
    
    # Batch 4: Tools (17 files)
    print("\n\n### PHASE 2: TOOLS AND UTILITIES ###\n")
    
    # Benchmarking tools
    benchmark_files = [f for f in test_dir.iterdir() 
                      if f.is_file() and f.suffix == '.py' and 'benchmark' in f.name]
    if benchmark_files:
        move_files_batch(benchmark_files, test_dir / 'tools' / 'benchmarking', "Benchmarking Tools")
    
    # Monitoring tools
    monitoring_files = [f for f in test_dir.iterdir() 
                       if f.is_file() and f.suffix == '.py'
                       and any(x in f.name for x in ['monitoring', 'dashboard', 'visualization'])]
    if monitoring_files:
        move_files_batch(monitoring_files, test_dir / 'tools' / 'monitoring', "Monitoring Tools")
    
    # Model tools
    model_tool_files = [f for f in test_dir.iterdir() 
                       if f.is_file() and f.suffix == '.py'
                       and any(x in f.name for x in ['model_', 'additional_models', 'random_models'])]
    if model_tool_files:
        move_files_batch(model_tool_files, test_dir / 'tools' / 'models', "Model Tools")
    
    # Batch 5: Scripts
    print("\n\n### PHASE 3: SCRIPTS ###\n")
    
    # Setup scripts
    setup_files = [f for f in test_dir.iterdir() 
                  if f.is_file() and f.suffix == '.py'
                  and (f.name.startswith('setup_') or f.name.startswith('install_'))]
    if setup_files:
        move_files_batch(setup_files, test_dir / 'scripts' / 'setup', "Setup Scripts")
    
    # Migration scripts
    migration_files = [f for f in test_dir.iterdir() 
                      if f.is_file() and f.suffix == '.py'
                      and ('migrate' in f.name or 'migration' in f.name)]
    if migration_files:
        move_files_batch(migration_files, test_dir / 'scripts' / 'migration', "Migration Scripts")
    
    # Build scripts
    build_files = [f for f in test_dir.iterdir() 
                  if f.is_file() and f.suffix == '.py'
                  and any(x in f.name for x in ['build_', 'compile_', 'convert_'])]
    if build_files:
        move_files_batch(build_files, test_dir / 'scripts' / 'build', "Build Scripts")
    
    # Utility scripts
    utility_files = [f for f in test_dir.iterdir() 
                    if f.is_file() and f.suffix == '.py'
                    and any(f.name.startswith(x) for x in ['fix_', 'check_', 'validate_', 'verify_', 'update_', 'analyze_'])]
    if utility_files:
        move_files_batch(utility_files, test_dir / 'scripts' / 'utilities', "Utility Scripts")
    
    # Runner scripts
    runner_files = [f for f in test_dir.iterdir() 
                   if f.is_file() and f.suffix == '.py' and f.name.startswith('run_')]
    if runner_files:
        move_files_batch(runner_files, test_dir / 'scripts' / 'runners', "Runner Scripts")
    
    print("\n\n### REFACTORING COMPLETE (PHASE 1-3) ###\n")
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nPhases 1-3 completed: Templates, Generators, Examples, Tools, and Scripts")
    print("\nNext: Run update_imports.py to fix imports")
    print("Then: Continue with test file reorganization (Phase 4)")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
