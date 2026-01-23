#!/usr/bin/env python3
"""
Automated AsyncIO to AnyIO Migration Helper

This script helps automate common asyncio to anyio migration patterns.
It performs text replacements on Python files.

Usage:
    python migrate_to_anyio.py <file_or_directory>
    
Examples:
    python migrate_to_anyio.py myfile.py
    python migrate_to_anyio.py ./ipfs_accelerate_py/
"""

import sys
import os
import re
from pathlib import Path

# Define replacement patterns
REPLACEMENTS = [
    # Import statements
    (r'^import asyncio$', 'import anyio', re.MULTILINE),
    (r'^from asyncio import (.+)$', r'# TODO: Migrate asyncio import: from asyncio import \1', re.MULTILINE),
    
    # Basic replacements
    (r'asyncio\.run\(', 'anyio.run(', 0),
    (r'asyncio\.sleep\(', 'anyio.sleep(', 0),
    (r'asyncio\.Event\(\)', 'anyio.Event()', 0),
    (r'asyncio\.Lock\(\)', 'anyio.Lock()', 0),
    
    # More complex patterns that need review
    (r'asyncio\.Queue\(', '# TODO: Replace with anyio.create_memory_object_stream - # TODO: Replace with anyio.create_memory_object_stream - asyncio.Queue(', 0),
    (r'asyncio\.create_task\(', '# TODO: Replace with task group - # TODO: Replace with task group - asyncio.create_task(', 0),
    (r'asyncio\.gather\(', '# TODO: Replace with task group - # TODO: Replace with task group - asyncio.gather(', 0),
    (r'asyncio\.wait_for\(', '# TODO: Replace with anyio.fail_after - # TODO: Replace with anyio.fail_after - asyncio.wait_for(', 0),
    (r'asyncio\.get_event_loop\(\)', '# TODO: Remove event loop management - # TODO: Remove event loop management - asyncio.get_event_loop()', 0),
    (r'asyncio\.new_event_loop\(\)', '# TODO: Remove event loop management - # TODO: Remove event loop management - asyncio.new_event_loop()', 0),
    (r'asyncio\.set_event_loop\(', '# TODO: Remove event loop management - # TODO: Remove event loop management - asyncio.set_event_loop(', 0),
    (r'asyncio\.iscoroutinefunction\(', 'inspect.iscoroutinefunction(  # Added import inspect', 0),
    (r'asyncio\.to_thread\(', 'anyio.to_thread.run_sync(', 0),
]

def should_process_file(filepath: Path) -> bool:
    """Check if file should be processed."""
    if not filepath.is_file():
        return False
    if filepath.suffix != '.py':
        return False
    if '__pycache__' in str(filepath):
        return False
    if 'venv' in str(filepath) or 'env' in str(filepath):
        return False
    return True

def migrate_file(filepath: Path, dry_run: bool = True) -> tuple[bool, list[str]]:
    """
    Migrate a single file from asyncio to anyio.
    
    Returns:
        (changed, warnings): Whether changes were made and list of warnings
    """
    try:
        content = filepath.read_text()
        original_content = content
        warnings = []
        
        # Check if file uses asyncio
        if 'asyncio' not in content:
            return False, []
        
        # Apply replacements
        for pattern, replacement, flags in REPLACEMENTS:
            if flags:
                content = re.sub(pattern, replacement, content, flags=flags)
            else:
                content = re.sub(pattern, replacement, content)
        
        # Check if we need to add inspect import
        if 'inspect.iscoroutinefunction' in content and 'import inspect' not in content:
            # Add import after other imports
            import_pos = content.find('import ')
            if import_pos != -1:
                # Find the end of import section
                lines = content.split('\n')
                insert_idx = 0
                for i, line in enumerate(lines):
                    if line.startswith('import ') or line.startswith('from '):
                        insert_idx = i + 1
                    elif insert_idx > 0 and line.strip() and not line.startswith('#'):
                        break
                lines.insert(insert_idx, 'import inspect')
                content = '\n'.join(lines)
        
        # Check for patterns that need manual review
        if 'TODO: Replace with' in content:
            warnings.append(f"File contains TODO comments requiring manual review")
        
        if 'asyncio.Queue' in original_content:
            warnings.append("Contains asyncio.Queue - needs manual conversion to memory streams")
        
        if 'asyncio.create_task' in original_content or 'asyncio.gather' in original_content:
            warnings.append("Contains task creation/gather - needs manual conversion to task groups")
        
        # Write back if not dry run
        if content != original_content:
            if not dry_run:
                # Write modified content directly (no backup)
                filepath.write_text(content)
            return True, warnings
        
        return False, warnings
        
    except Exception as e:
        return False, [f"Error: {e}"]

def migrate_directory(dirpath: Path, dry_run: bool = True) -> dict:
    """Migrate all Python files in a directory."""
    results = {
        'processed': 0,
        'changed': 0,
        'skipped': 0,
        'warnings': []
    }
    
    for filepath in dirpath.rglob('*.py'):
        if should_process_file(filepath):
            results['processed'] += 1
            changed, warnings = migrate_file(filepath, dry_run)
            
            if changed:
                results['changed'] += 1
                print(f"{'[DRY RUN] ' if dry_run else ''}Modified: {filepath}")
                if warnings:
                    for warning in warnings:
                        print(f"  ⚠ {warning}")
                        results['warnings'].append((str(filepath), warning))
            else:
                results['skipped'] += 1
    
    return results

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    path = Path(sys.argv[1])
    dry_run = '--apply' not in sys.argv
    
    if dry_run:
        print("🔍 DRY RUN MODE - No files will be modified")
        print("   Add --apply flag to actually modify files\n")
    else:
        print("⚠️  APPLYING CHANGES - Files will be modified in place\n")
    
    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        sys.exit(1)
    
    if path.is_file():
        changed, warnings = migrate_file(path, dry_run)
        if changed:
            print(f"{'[DRY RUN] ' if dry_run else ''}Modified: {path}")
            if warnings:
                for warning in warnings:
                    print(f"  ⚠ {warning}")
        else:
            print(f"No changes needed: {path}")
    else:
        print(f"Processing directory: {path}\n")
        results = migrate_directory(path, dry_run)
        
        print(f"\n{'='*60}")
        print(f"Results:")
        print(f"  Processed: {results['processed']} files")
        print(f"  Changed:   {results['changed']} files")
        print(f"  Skipped:   {results['skipped']} files")
        print(f"  Warnings:  {len(results['warnings'])}")
        
        if results['warnings']:
            print(f"\n⚠️  Files with warnings (manual review needed):")
            seen_files = set()
            for filepath, warning in results['warnings']:
                if filepath not in seen_files:
                    print(f"  - {filepath}")
                    seen_files.add(filepath)

if __name__ == '__main__':
    main()
