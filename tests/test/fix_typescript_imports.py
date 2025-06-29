#\!/usr/bin/env python3
# fix_typescript_imports.py
# Script to automatically fix import paths in the converted TypeScript files

import os
import sys
import re
import glob
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('fix_typescript_imports.log')
    ]
)
logger = logging.getLogger(__name__)

class Config:
    TARGET_DIR = None
    DRY_RUN = False
    VERBOSE = False
    STATS = {
        "files_checked": 0,
        "files_fixed": 0,
        "imports_fixed": 0,
        "errors": 0
    }
    MODULE_MAP = {
        # Map from Python module paths to TypeScript module paths
        "hardware.backends.webgpu": "./hardware/backends/webgpu_backend",
        "hardware.backends.webnn": "./hardware/backends/webnn_backend",
        "hardware.detection": "./hardware/detection/hardware_detection",
        "hardware.abstraction": "./hardware/hardware_abstraction",
        "model.loaders": "./model/loaders/model_loader",
        "model.transformers": "./model/transformers",
        "model.vision": "./model/vision",
        "model.audio": "./model/audio",
        "browser.optimizations": "./browser/optimizations",
        "browser.resource_pool": "./browser/resource_pool/resource_pool",
        "quantization.techniques": "./quantization/techniques",
        "storage.indexeddb": "./storage/indexeddb/storage_manager",
        "worker.webgpu": "./worker/webgpu",
        "worker.webnn": "./worker/webnn",
        "tensor": "./tensor",
        "utils": "./utils",
        "types": "./types"
    }

def setup_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Fix TypeScript import paths")
    parser.add_argument("--target-dir", default="../ipfs_accelerate_js", help="Target directory with TypeScript files")
    parser.add_argument("--dry-run", action="store_true", help="Don't make changes, just report issues")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    Config.TARGET_DIR = os.path.abspath(args.target_dir)
    Config.DRY_RUN = args.dry_run
    Config.VERBOSE = args.verbose
    
    if not os.path.isdir(Config.TARGET_DIR):
        logger.error(f"Target directory does not exist: {Config.TARGET_DIR}")
        sys.exit(1)
    
    logger.info(f"Fixing TypeScript import paths in: {Config.TARGET_DIR}")
    logger.info(f"Dry run: {Config.DRY_RUN}")

def find_typescript_files() -> List[str]:
    """Find all TypeScript files in the target directory"""
    ts_files = []
    
    for root, _, files in os.walk(Config.TARGET_DIR):
        for file in files:
            if file.endswith((".ts", ".tsx")) and not file.endswith(".d.ts"):
                ts_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(ts_files)} TypeScript files to check")
    return ts_files

def extract_imports(file_path: str) -> List[Tuple[str, str]]:
    """Extract import statements from a TypeScript file"""
    imports = []
    import_patterns = [
        r'import\s+\{[^}]*\}\s+from\s+[\'"]([^\'"]+)[\'"]',  # import { X } from 'path'
        r'import\s+\*\s+as\s+\w+\s+from\s+[\'"]([^\'"]+)[\'"]',  # import * as X from 'path'
        r'import\s+\w+\s+from\s+[\'"]([^\'"]+)[\'"]',  # import X from 'path'
        r'import\s+[\'"]([^\'"]+)[\'"]',  # import 'path'
    ]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            for pattern in import_patterns:
                for match in re.finditer(pattern, content):
                    import_path = match.group(1)
                    original_statement = match.group(0)
                    imports.append((import_path, original_statement))
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        Config.STATS["errors"] += 1
    
    return imports

def fix_import_path(import_path: str, file_path: str) -> str:
    """Fix import path based on module mapping"""
    # Skip external modules and properly formatted relative imports
    if (not import_path.startswith('.') and not import_path.startswith('/')) or \
       (import_path.startswith('./') and any(import_path.startswith(f'./{module}') for module in ['hardware', 'model', 'browser', 'worker', 'utils', 'types'])):
        return import_path
    
    # Handle absolute paths
    if import_path.startswith('/'):
        import_path = import_path[1:]  # Remove leading slash
    
    # Try direct module mapping
    for py_module, ts_module in Config.MODULE_MAP.items():
        if import_path == py_module or import_path.startswith(f"{py_module}/"):
            # Replace module prefix with TypeScript path
            fixed_path = import_path.replace(py_module, ts_module)
            return fixed_path
    
    # Handle relative paths
    if import_path.startswith('.'):
        # Get the components of the path
        path_parts = import_path.split('/')
        # Normalize the path to avoid ../ chains
        if '..' in path_parts:
            # Get the current file's directory
            current_dir = os.path.dirname(file_path)
            target_path = os.path.normpath(os.path.join(current_dir, import_path))
            
            # Convert to relative path from the file
            rel_path = os.path.relpath(target_path, current_dir)
            
            # Add ./ prefix if needed
            if not rel_path.startswith('.'):
                rel_path = f"./{rel_path}"
            
            return rel_path.replace('\\', '/')
    
    # Try to infer the correct path based on target directory structure
    for key, value in Config.MODULE_MAP.items():
        if key in import_path:
            parts = import_path.split('.')
            if len(parts) > 1:
                inferred_path = '/'.join(parts)
                if not inferred_path.startswith('./'):
                    inferred_path = f"./{inferred_path}"
                return inferred_path
    
    # If all else fails, add ./ prefix if it's a relative import without one
    if not import_path.startswith('.') and not import_path.startswith('/') and \
       not any(import_path.startswith(prefix) for prefix in ['http', 'https', '@']):
        return f"./{import_path}"
    
    return import_path

def fix_file_imports(file_path: str) -> Tuple[bool, int]:
    """Check and fix imports in a file"""
    imports = extract_imports(file_path)
    fixed = False
    fixes_count = 0
    
    # Skip if no imports
    if not imports:
        if Config.VERBOSE:
            logger.debug(f"No imports found in {file_path}")
        return False, 0
    
    # Process each import
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content = content
        
        for import_path, original_statement in imports:
            fixed_path = fix_import_path(import_path, file_path)
            
            if fixed_path \!= import_path:
                if Config.VERBOSE:
                    logger.info(f"Fixing import in {file_path}: {import_path} -> {fixed_path}")
                
                # Replace only the path part, preserve the import syntax
                new_statement = original_statement.replace(f'"{import_path}"', f'"{fixed_path}"')
                new_statement = new_statement.replace(f"'{import_path}'", f"'{fixed_path}'")
                
                new_content = new_content.replace(original_statement, new_statement)
                fixed = True
                fixes_count += 1
        
        # Write changes if needed
        if fixed and not Config.DRY_RUN:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            logger.info(f"Fixed {fixes_count} imports in {file_path}")
    except Exception as e:
        logger.error(f"Error fixing imports in {file_path}: {e}")
        Config.STATS["errors"] += 1
        return False, 0
    
    return fixed, fixes_count

def main():
    """Main function"""
    setup_args()
    
    # Find TypeScript files
    ts_files = find_typescript_files()
    
    # Check and fix imports in each file
    for i, file_path in enumerate(ts_files):
        Config.STATS["files_checked"] += 1
        
        if Config.VERBOSE or (i+1) % 25 == 0:
            logger.info(f"Processing file {i+1}/{len(ts_files)}: {file_path}")
        
        fixed, fixes_count = fix_file_imports(file_path)
        
        if fixed:
            Config.STATS["files_fixed"] += 1
            Config.STATS["imports_fixed"] += fixes_count
    
    # Print summary
    logger.info("\nImport Fix Summary:")
    logger.info(f"Files checked: {Config.STATS['files_checked']}")
    logger.info(f"Files fixed: {Config.STATS['files_fixed']}")
    logger.info(f"Imports fixed: {Config.STATS['imports_fixed']}")
    logger.info(f"Errors: {Config.STATS['errors']}")
    
    if Config.DRY_RUN:
        logger.info("This was a dry run, no changes were made")
    else:
        logger.info("All import paths have been fixed")

if __name__ == "__main__":
    main()
