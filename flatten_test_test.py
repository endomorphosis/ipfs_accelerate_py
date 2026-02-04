#!/usr/bin/env python3
"""
Script to flatten the nested test/test/ directory and merge with test/tests/
"""

import os
import shutil
from pathlib import Path
import hashlib

def get_file_hash(filepath):
    """Get SHA256 hash of a file"""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except:
        return None

def flatten_test_test_directory():
    """Flatten test/test/ directory by merging with appropriate locations"""
    
    base_dir = Path('/home/runner/work/ipfs_accelerate_py/ipfs_accelerate_py')
    test_test = base_dir / 'test' / 'test'
    
    if not test_test.exists():
        print("✓ test/test/ directory doesn't exist - already flattened!")
        return
    
    # Mapping of test/test subdirectories to their target locations
    mappings = {
        'test/test/api': 'test/tests/api',
        'test/test/integration': 'test/tests/integration',
        'test/test/models': 'test/tests/models',
        'test/test/hardware': 'test/tests/hardware',
        'test/test/common': 'test/tests/other',  # Move common to other
        'test/test/docs': 'test/tests/other',     # Move docs to other
        'test/test/skillset': 'test/tests/other', # Move skillset to other
        'test/test/template_system': 'test/tests/other', # Move template_system to other
    }
    
    moves = []
    duplicates = []
    errors = []
    
    for source_rel, target_rel in mappings.items():
        source = base_dir / source_rel
        target = base_dir / target_rel
        
        if not source.exists():
            print(f"  Skipping {source_rel} - doesn't exist")
            continue
        
        # Ensure target directory exists
        target.mkdir(parents=True, exist_ok=True)
        
        # Walk through source directory
        for root, dirs, files in os.walk(source):
            root_path = Path(root)
            rel_path = root_path.relative_to(source)
            
            for file in files:
                if not file.endswith('.py'):
                    continue
                
                source_file = root_path / file
                
                # Determine target path
                if rel_path == Path('.'):
                    target_file = target / file
                else:
                    target_subdir = target / rel_path
                    target_subdir.mkdir(parents=True, exist_ok=True)
                    target_file = target_subdir / file
                
                # Check if target exists
                if target_file.exists():
                    # Compare files
                    source_hash = get_file_hash(source_file)
                    target_hash = get_file_hash(target_file)
                    
                    if source_hash == target_hash:
                        duplicates.append((str(source_file.relative_to(base_dir)), 
                                         str(target_file.relative_to(base_dir)), 
                                         'identical'))
                    else:
                        duplicates.append((str(source_file.relative_to(base_dir)), 
                                         str(target_file.relative_to(base_dir)), 
                                         'different'))
                else:
                    moves.append((str(source_file.relative_to(base_dir)), 
                                str(target_file.relative_to(base_dir))))
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"FLATTEN test/test/ DIRECTORY - ANALYSIS")
    print(f"{'='*80}\n")
    
    print(f"Files to move: {len(moves)}")
    print(f"Duplicate files (identical): {sum(1 for d in duplicates if d[2] == 'identical')}")
    print(f"Duplicate files (different): {sum(1 for d in duplicates if d[2] == 'different')}")
    
    if moves:
        print(f"\n{'-'*80}")
        print("FILES TO MOVE:")
        print(f"{'-'*80}")
        for source, target in moves[:20]:
            print(f"  {source}")
            print(f"    → {target}")
        if len(moves) > 20:
            print(f"  ... and {len(moves) - 20} more files")
    
    if duplicates:
        print(f"\n{'-'*80}")
        print("DUPLICATE FILES (first 10):")
        print(f"{'-'*80}")
        for source, target, status in duplicates[:10]:
            print(f"  {source}")
            print(f"    vs {target} ({status})")
        if len(duplicates) > 10:
            print(f"  ... and {len(duplicates) - 10} more duplicates")
    
    # Ask for confirmation
    print(f"\n{'-'*80}")
    response = input("\nProceed with moving files? (yes/no): ")
    
    if response.lower() != 'yes':
        print("Aborted by user")
        return
    
    # Execute moves
    print("\nExecuting moves...")
    moved_count = 0
    for source_rel, target_rel in moves:
        source = base_dir / source_rel
        target = base_dir / target_rel
        
        try:
            # Ensure target directory exists
            target.parent.mkdir(parents=True, exist_ok=True)
            
            # Move file
            shutil.move(str(source), str(target))
            moved_count += 1
            
            if moved_count % 20 == 0:
                print(f"  Moved {moved_count}/{len(moves)} files...")
        except Exception as e:
            errors.append((source_rel, str(e)))
            print(f"  Error moving {source_rel}: {e}")
    
    print(f"\nMoved {moved_count} files")
    
    # Handle duplicates (delete from source if identical)
    deleted_count = 0
    for source_rel, target_rel, status in duplicates:
        if status == 'identical':
            source = base_dir / source_rel
            try:
                source.unlink()
                deleted_count += 1
            except Exception as e:
                errors.append((source_rel, f"Delete error: {e}"))
    
    print(f"Deleted {deleted_count} identical duplicate files")
    
    # Clean up empty directories
    print("\nCleaning up empty directories...")
    for source_rel, target_rel in reversed(list(mappings.items())):
        source = base_dir / source_rel
        if source.exists():
            try:
                # Remove empty subdirectories
                for root, dirs, files in os.walk(source, topdown=False):
                    for dir in dirs:
                        dir_path = Path(root) / dir
                        if dir_path.exists() and not any(dir_path.iterdir()):
                            dir_path.rmdir()
                            print(f"  Removed empty directory: {dir_path.relative_to(base_dir)}")
                
                # Remove source directory if empty
                if source.exists() and not any(source.iterdir()):
                    source.rmdir()
                    print(f"  Removed empty directory: {source.relative_to(base_dir)}")
            except Exception as e:
                print(f"  Error cleaning {source_rel}: {e}")
    
    # Final cleanup of test/test if empty
    if test_test.exists():
        try:
            # Check if empty (only __init__.py might remain)
            contents = list(test_test.iterdir())
            if len(contents) == 0 or (len(contents) == 1 and contents[0].name == '__init__.py'):
                if test_test.joinpath('__init__.py').exists():
                    test_test.joinpath('__init__.py').unlink()
                test_test.rmdir()
                print(f"\n✓ Removed test/test/ directory")
        except Exception as e:
            print(f"\n✗ Could not remove test/test/: {e}")
    
    if errors:
        print(f"\n{'-'*80}")
        print(f"ERRORS ({len(errors)}):")
        print(f"{'-'*80}")
        for file, error in errors[:10]:
            print(f"  {file}: {error}")
    
    print(f"\n{'='*80}")
    print("✓ FLATTEN COMPLETE")
    print(f"{'='*80}")

if __name__ == '__main__':
    flatten_test_test_directory()
