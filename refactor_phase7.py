#!/usr/bin/env python3
"""Phase 7: Refactor remaining test/ subdirectories."""

import os
import subprocess
import shutil
from pathlib import Path
from collections import defaultdict

def safe_git_mv(source, target):
    """Move a file or directory using git mv, with fallback."""
    try:
        # Create target parent directory
        target.parent.mkdir(parents=True, exist_ok=True)
        
        result = subprocess.run(
            ['git', 'mv', str(source), str(target)],
            capture_output=True,
            text=True,
            check=True
        )
        return True, None
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def count_files(directory):
    """Count files in a directory."""
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.rglob('*') if _.is_file())

def main():
    test_dir = Path('test')
    
    # Categories based on analysis
    to_delete = [
        'huggingface_transformers', 'output', 'temp_docs', 
        'template_integration', 'template_system', 'template_verification',
        'test_venv', 'venv', 'venvs', 'web_platform_test_output'
    ]
    
    to_move_docs = [
        'doc-builder', 'doc-builder-test', 'docs', 
        'huggingface_doc_builder', 'transformers_docs_built'
    ]
    
    to_archive = [
        'old_scripts', 'playwright_screenshots_functional_legacy',
        'playwright_screenshots_legacy'
    ]
    
    to_review = [
        'fixes', 'improved', 'improvements',
        'refactored_benchmark_suite', 'refactored_generator_suite',
        'refactored_test_suite'
    ]
    
    # Major directories to organize
    to_organize = {
        'api': 'test/tests/api',
        'api_client': 'test/tools/api',
        'api_server': 'test/tools/api',
        'apis': 'test/tests/api',
        'distributed_testing': 'test/tests/distributed',
        'duckdb_api': 'test/tests/api',
        'fixed_web_platform': 'test/tests/web',
        'fixed_web_tests': 'test/tests/web',
        'web_platform': 'test/tests/web',
        'web_platform_integration': 'test/tests/web',
        'web_platform_tests': 'test/tests/web',
        'ipfs_accelerate_js': 'ipfs_accelerate_js',  # Move to root as SDK
        'ipfs_accelerate_py': 'ipfs_accelerate_py',  # Already exists at root
    }
    
    print("=" * 80)
    print("PHASE 7: REFACTORING REMAINING TEST SUBDIRECTORIES")
    print("=" * 80)
    
    stats = defaultdict(int)
    
    # Step 1: Delete empty/temporary directories
    print("\n1. DELETING temporary/empty directories...")
    print("-" * 80)
    for dirname in to_delete:
        dir_path = test_dir / dirname
        if not dir_path.exists():
            print(f"  [SKIP] {dir_path} - doesn't exist")
            continue
        
        file_count = count_files(dir_path)
        if file_count == 0 or dirname in ['venv', 'venvs', 'test_venv']:
            try:
                # Remove from git and filesystem
                subprocess.run(['git', 'rm', '-rf', str(dir_path)], 
                             capture_output=True, check=False)
                if dir_path.exists():
                    shutil.rmtree(dir_path, ignore_errors=True)
                print(f"  [DEL] {dir_path} ({file_count} files)")
                stats['deleted'] += 1
            except Exception as e:
                print(f"  [ERR] {dir_path}: {e}")
    
    # Step 2: Move documentation directories
    print("\n2. MOVING documentation directories...")
    print("-" * 80)
    docs_root = Path('docs')
    for dirname in to_move_docs:
        source = test_dir / dirname
        if not source.exists():
            print(f"  [SKIP] {source} - doesn't exist")
            continue
        
        # Determine target
        if 'builder' in dirname:
            target = docs_root / 'builders' / dirname
        else:
            target = docs_root / dirname
        
        success, error = safe_git_mv(source, target)
        if success:
            print(f"  [MOVE] {source} -> {target}")
            stats['moved_docs'] += 1
        else:
            print(f"  [ERR] {source}: {error}")
    
    # Step 3: Archive legacy directories
    print("\n3. ARCHIVING legacy directories...")
    print("-" * 80)
    archive_dir = Path('archive')
    for dirname in to_archive:
        source = test_dir / dirname
        if not source.exists():
            print(f"  [SKIP] {source} - doesn't exist")
            continue
        
        target = archive_dir / dirname
        success, error = safe_git_mv(source, target)
        if success:
            print(f"  [ARCH] {source} -> {target}")
            stats['archived'] += 1
        else:
            print(f"  [ERR] {source}: {error}")
    
    # Step 4: Review directories - merge if duplicates
    print("\n4. REVIEWING refactored/improved directories...")
    print("-" * 80)
    for dirname in to_review:
        source = test_dir / dirname
        if not source.exists():
            print(f"  [SKIP] {source} - doesn't exist")
            continue
        
        file_count = count_files(source)
        print(f"  [INFO] {source} has {file_count} files - needs manual review")
        
        # For now, move to archive for manual review
        target = archive_dir / 'review' / dirname
        success, error = safe_git_mv(source, target)
        if success:
            print(f"  [ARCH] {source} -> {target} (for review)")
            stats['review'] += 1
    
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"  Deleted:       {stats['deleted']} directories")
    print(f"  Moved (docs):  {stats['moved_docs']} directories")
    print(f"  Archived:      {stats['archived']} directories")
    print(f"  For review:    {stats['review']} directories")
    print("=" * 80)
    
    print("\nPhase 7a complete!")
    print("Next: Phase 7b will organize the remaining 55 directories with content")

if __name__ == '__main__':
    main()
