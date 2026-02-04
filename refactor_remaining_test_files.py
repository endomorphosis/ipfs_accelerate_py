#!/usr/bin/env python3
"""Move all remaining non-test files from test/ to appropriate locations."""

import os
import subprocess
from pathlib import Path
from collections import defaultdict

def categorize_file(filename):
    """Categorize a file and determine its target location."""
    name_lower = filename.lower()
    
    # TypeScript source files - these are library/SDK files
    if filename.startswith('ipfs_accelerate_js') and filename.endswith('.ts'):
        if '.test.ts' in filename:
            return 'test/tests/web'  # TypeScript test files
        else:
            return 'ipfs_accelerate_js/src'  # Source files for JS SDK
    
    # HTML demos and examples
    if filename.endswith('.html'):
        if 'demo' in name_lower:
            return 'examples/web/demos'
        else:
            return 'examples/web'
    
    # CSS and JSX files
    if filename.endswith('.css') or filename.endswith('.jsx'):
        return 'examples/web'
    
    # Shell scripts
    if filename.endswith('.sh'):
        if 'run_' in filename or 'test_' in filename:
            return 'test/scripts/runners'
        elif 'setup_' in filename or 'install_' in filename:
            return 'test/scripts/setup'
        elif 'migrate_' in filename or 'archive_' in filename:
            return 'test/scripts/migration'
        elif 'validate_' in filename or 'update_' in filename:
            return 'test/scripts/utilities'
        else:
            return 'scripts'
    
    # Database files
    if filename.endswith('.db') or filename.endswith('.db.wal'):
        return 'test/data/databases'
    
    # SQL files
    if filename.endswith('.sql'):
        return 'test/data/sql'
    
    # Requirements files
    if filename.startswith('requirements'):
        return 'requirements'  # Root level requirements
    
    # Config files
    if any(x in filename for x in ['config', 'setup', 'rollup', 'pytest.ini', 'Makefile']):
        if filename == 'pytest.ini':
            return 'KEEP'  # Keep in test/
        elif filename == 'Makefile':
            return 'test/scripts'
        else:
            return 'config'
    
    # Image files
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        return 'test/data/images'
    
    # Audio/media files
    if filename.endswith(('.mp3', '.wav')):
        return 'test/data/media'
    
    # CSV files
    if filename.endswith('.csv'):
        return 'test/data'
    
    # Text report files
    if filename.endswith('.txt'):
        if 'summary' in name_lower or 'error' in name_lower or 'files' in name_lower:
            return 'docs/reports'
        elif 'out' in name_lower or 'output' in name_lower or 'log' in name_lower:
            return 'test/data/logs'
        else:
            return 'test/data'
    
    # TypeScript definition files
    if filename.endswith('.d.ts'):
        return 'types'
    
    # WGSL shader files
    if filename.endswith('.wgsl'):
        return 'shaders'
    
    # YAML workflow files
    if filename.endswith('.yml') or filename.endswith('.yaml'):
        return '.github/workflows'
    
    # TOML config files
    if filename.endswith('.toml'):
        return 'config'
    
    # Temporary/updated files
    if filename.endswith('.updated'):
        return 'DELETE'
    
    # Batch files
    if filename.endswith('.bat'):
        return 'test/scripts/windows'
    
    return 'other'

def main():
    test_dir = Path('test')
    
    # Find all non-Python files in test/ root (excluding conftest.py and __init__.py)
    all_files = []
    for f in test_dir.iterdir():
        if f.is_file() and f.name not in ['conftest.py', '__init__.py', 'pytest.ini']:
            if not f.name.endswith('.py'):
                all_files.append(f)
    
    print(f"Found {len(all_files)} non-Python files to organize")
    print()
    
    # Categorize files
    categorized = defaultdict(list)
    for f in all_files:
        target = categorize_file(f.name)
        categorized[target].append(f.name)
    
    # Print summary
    print("File Organization Plan:")
    print("=" * 80)
    for target in sorted(categorized.keys()):
        files = categorized[target]
        print(f"\n{target} ({len(files)} files)")
        if len(files) <= 5:
            for fname in files:
                print(f"  - {fname}")
        else:
            for fname in files[:3]:
                print(f"  - {fname}")
            print(f"  ... and {len(files) - 3} more")
    
    print("\n" + "=" * 80)
    print("\nProceed with moving files? (This will use git mv)")
    print("Press Enter to continue, Ctrl+C to cancel...")
    # input()  # Commented out for automation
    
    # Move files
    moved = 0
    deleted = 0
    kept = 0
    
    for target, files in categorized.items():
        if target == 'KEEP':
            kept += len(files)
            continue
        
        if target == 'DELETE':
            for fname in files:
                source = test_dir / fname
                print(f"[DEL] {source}")
                try:
                    source.unlink()
                    deleted += 1
                except Exception as e:
                    print(f"  Error: {e}")
            continue
        
        # Create target directory
        target_dir = Path(target)
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Move files
        for fname in files:
            source = test_dir / fname
            dest = target_dir / fname
            
            try:
                result = subprocess.run(
                    ['git', 'mv', str(source), str(dest)],
                    capture_output=True,
                    text=True,
                    check=True
                )
                moved += 1
                if moved <= 10 or moved % 20 == 0:
                    print(f"[{moved:3d}] {source} -> {dest}")
            except subprocess.CalledProcessError as e:
                # If git mv fails, try regular move
                try:
                    import shutil
                    shutil.move(str(source), str(dest))
                    moved += 1
                    print(f"[{moved:3d}] {source} -> {dest} (regular move)")
                except Exception as e2:
                    print(f"  [ERR] Failed to move {source}: {e2}")
    
    print()
    print("=" * 80)
    print(f"Summary:")
    print(f"  Moved:   {moved} files")
    print(f"  Deleted: {deleted} files")
    print(f"  Kept:    {kept} files")
    print()
    print("Refactoring complete!")

if __name__ == '__main__':
    main()
