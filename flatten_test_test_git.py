#!/usr/bin/env python3
"""
Script to flatten test/test/ using git mv to preserve history
"""

import os
import subprocess
from pathlib import Path

def run_git_command(cmd, cwd=None):
    """Run a git command and return the result"""
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return None

def flatten_with_git_mv():
    """Use git mv to flatten test/test/ directory"""
    
    base_dir = Path('/home/runner/work/ipfs_accelerate_py/ipfs_accelerate_py')
    test_test = base_dir / 'test' / 'test'
    
    if not test_test.exists():
        print("✓ test/test/ directory doesn't exist - already flattened!")
        return
    
    os.chdir(base_dir)
    
    # Mapping of test/test subdirectories to their target locations
    mappings = {
        'test/test/api/llm_providers': 'test/tests/api/llm_providers',
        'test/test/api/local_servers': 'test/tests/api/local_servers',
        'test/test/api/internal': 'test/tests/api/internal',
        'test/test/api/huggingface': 'test/tests/api/huggingface',
        'test/test/api/other': 'test/tests/api/other',
        'test/test/integration/browser': 'test/tests/integration/browser',
        'test/test/integration/database': 'test/tests/integration/database',
        'test/test/integration/distributed': 'test/tests/integration/distributed',
        'test/test/models/vision/vit': 'test/tests/models/vision/vit',
        'test/test/models/vision': 'test/tests/models/vision',
        'test/test/models/text/bert': 'test/tests/models/text/bert',
        'test/test/models/text/t5': 'test/tests/models/text/t5',
        'test/test/models/text/gpt': 'test/tests/models/text/gpt',
        'test/test/models/text': 'test/tests/models/text',
        'test/test/models/audio/whisper': 'test/tests/models/audio/whisper',
        'test/test/models/audio': 'test/tests/models/audio',
        'test/test/hardware': 'test/tests/hardware',
        'test/test/common': 'test/tests/other',
        'test/test/docs': 'test/tests/other',
        'test/test/skillset': 'test/tests/other',
        'test/test/template_system': 'test/tests/other',
    }
    
    moved = 0
    skipped = 0
    errors = []
    
    print("="*80)
    print("FLATTENING test/test/ WITH GIT MV")
    print("="*80)
    
    # Process each mapping
    for source_rel, target_rel in mappings.items():
        source = Path(source_rel)
        target = Path(target_rel)
        
        if not source.exists():
            print(f"\n  Skipping {source_rel} - doesn't exist")
            continue
        
        # Ensure target directory exists
        target.mkdir(parents=True, exist_ok=True)
        
        # Find all .py files in source
        py_files = list(source.glob('*.py'))
        
        if not py_files:
            print(f"\n  No .py files in {source_rel}")
            continue
        
        print(f"\n  Processing {source_rel} → {target_rel}")
        print(f"  Found {len(py_files)} files")
        
        for py_file in py_files:
            target_file = target / py_file.name
            
            # Check if target exists
            if target_file.exists():
                # Compare files
                result = subprocess.run(['diff', '-q', str(py_file), str(target_file)], 
                                      capture_output=True)
                if result.returncode == 0:
                    # Files are identical - just remove source
                    print(f"    - {py_file.name} (identical, removing source)")
                    os.remove(py_file)
                    skipped += 1
                else:
                    # Files differ - skip for manual review
                    print(f"    ! {py_file.name} (differs from target, skipping)")
                    errors.append((str(py_file), str(target_file), "Files differ"))
                    skipped += 1
            else:
                # Move with git mv
                cmd = ['git', 'mv', str(py_file), str(target_file)]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"    ✓ {py_file.name}")
                    moved += 1
                else:
                    print(f"    ✗ {py_file.name}: {result.stderr.strip()}")
                    errors.append((str(py_file), str(target_file), result.stderr.strip()))
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Files moved: {moved}")
    print(f"Files skipped: {skipped}")
    print(f"Errors: {len(errors)}")
    
    if errors:
        print(f"\n{'-'*80}")
        print("ERRORS/CONFLICTS:")
        print(f"{'-'*80}")
        for source, target, error in errors[:10]:
            print(f"  {source}")
            print(f"    → {target}")
            print(f"    Error: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    # Clean up empty directories
    print(f"\n{'-'*80}")
    print("Cleaning up empty directories...")
    print(f"{'-'*80}")
    
    for root, dirs, files in os.walk(test_test, topdown=False):
        root_path = Path(root)
        if root_path.exists() and not any(root_path.iterdir()):
            print(f"  Removing {root_path.relative_to(base_dir)}")
            root_path.rmdir()
    
    # Try to remove test/test itself
    if test_test.exists():
        try:
            contents = list(test_test.iterdir())
            if len(contents) == 0:
                test_test.rmdir()
                print(f"\n✓ Removed empty test/test/ directory")
            elif len(contents) == 1 and contents[0].name == '__init__.py':
                contents[0].unlink()
                test_test.rmdir()
                print(f"\n✓ Removed test/test/ directory")
            else:
                print(f"\n! test/test/ directory not empty:")
                for item in contents[:10]:
                    print(f"    - {item.relative_to(base_dir)}")
        except Exception as e:
            print(f"\n✗ Could not remove test/test/: {e}")
    
    print(f"\n{'='*80}")
    print("✓ FLATTEN COMPLETE")
    print(f"{'='*80}")

if __name__ == '__main__':
    flatten_with_git_mv()
