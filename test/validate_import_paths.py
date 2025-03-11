#!/usr/bin/env python3
# validate_import_paths.py
# Script to validate import paths in the migrated TypeScript files

import os
import re
import sys
import glob
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('validate_import_paths.log')
    ]
)
logger = logging.getLogger(__name__)

class Config:
    TARGET_DIR = None
    FIX_IMPORTS = False
    VERBOSE = False
    STATS = {
        "files_checked": 0,
        "files_with_issues": 0,
        "total_import_issues": 0,
        "fixed_import_issues": 0,
        "circular_dependencies": 0,
        "unresolved_imports": 0,
        "valid_files": 0
    }

def setup_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Validate import paths in TypeScript files")
    parser.add_argument("--target-dir", help="Target directory to check", default="../ipfs_accelerate_js")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix import issues automatically")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    Config.TARGET_DIR = os.path.abspath(args.target_dir)
    Config.FIX_IMPORTS = args.fix
    Config.VERBOSE = args.verbose
    
    if not os.path.isdir(Config.TARGET_DIR):
        logger.error(f"Target directory does not exist: {Config.TARGET_DIR}")
        sys.exit(1)
    
    logger.info(f"Checking import paths in: {Config.TARGET_DIR}")
    logger.info(f"Fix imports: {Config.FIX_IMPORTS}")

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
        r'require\(\s*[\'"]([^\'"]+)[\'"]\s*\)'  # require('path')
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
    
    return imports

def is_valid_import(import_path: str, file_path: str) -> bool:
    """Check if an import path is valid"""
    # Skip external modules (not relative)
    if not import_path.startswith('./') and not import_path.startswith('../'):
        return True
    
    # Convert to absolute path
    current_dir = os.path.dirname(file_path)
    target_path = os.path.normpath(os.path.join(current_dir, import_path))
    
    # Check if path exists with .ts, .tsx, or as directory with index.ts
    extensions = ['.ts', '.tsx', '.js', '.jsx']
    
    for ext in extensions:
        if os.path.exists(target_path + ext):
            return True
    
    # Check for directory with index files
    if os.path.isdir(target_path):
        for ext in extensions:
            if os.path.exists(os.path.join(target_path, 'index' + ext)):
                return True
    
    return False

def fix_import_path(import_path: str, file_path: str) -> str:
    """Attempt to fix invalid import path"""
    if not import_path.startswith('./') and not import_path.startswith('../'):
        return import_path  # Not a relative import
    
    current_dir = os.path.dirname(file_path)
    
    # Try adding file extensions
    extensions = ['.ts', '.tsx', '.js', '.jsx']
    for ext in extensions:
        test_path = os.path.normpath(os.path.join(current_dir, import_path + ext))
        relative_path = os.path.relpath(test_path, current_dir)
        
        if os.path.exists(test_path):
            # Return the fixed path without extension (as per TypeScript convention)
            return os.path.splitext(relative_path)[0].replace('\\', '/')
    
    # Try finding index files in directories
    test_dir = os.path.normpath(os.path.join(current_dir, import_path))
    if os.path.isdir(test_dir):
        for ext in extensions:
            if os.path.exists(os.path.join(test_dir, 'index' + ext)):
                return import_path  # Directory with index file is valid
    
    # If all else fails, try to find a similar file
    base_name = os.path.basename(import_path)
    parent_dir = os.path.dirname(os.path.normpath(os.path.join(current_dir, import_path)))
    
    if os.path.isdir(parent_dir):
        for file in os.listdir(parent_dir):
            file_base, ext = os.path.splitext(file)
            if ext in ['.ts', '.tsx', '.js', '.jsx'] and file_base.lower() == base_name.lower():
                # Found a similar file with different casing
                fixed_path = os.path.join(os.path.dirname(import_path), file_base)
                return fixed_path.replace('\\', '/')
    
    # Could not fix the import
    return import_path

def fix_file_imports(file_path: str) -> Tuple[bool, int, int]:
    """Check and fix imports in a file"""
    imports = extract_imports(file_path)
    has_issues = False
    issue_count = 0
    fixed_count = 0
    
    # Skip if no imports
    if not imports:
        if Config.VERBOSE:
            logger.debug(f"No imports found in {file_path}")
        return False, 0, 0
    
    # Check each import
    invalid_imports = []
    for import_path, original_statement in imports:
        if not is_valid_import(import_path, file_path):
            has_issues = True
            issue_count += 1
            
            if Config.VERBOSE:
                logger.debug(f"Invalid import in {file_path}: {import_path}")
            
            if Config.FIX_IMPORTS:
                fixed_path = fix_import_path(import_path, file_path)
                if fixed_path != import_path:
                    invalid_imports.append((import_path, fixed_path, original_statement))
                    fixed_count += 1
    
    # Fix imports if requested
    if Config.FIX_IMPORTS and invalid_imports:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for old_path, new_path, original_statement in invalid_imports:
                # Replace only the path part, preserve the import syntax
                new_statement = original_statement.replace(f'"{old_path}"', f'"{new_path}"').replace(f"'{old_path}'", f"'{new_path}'")
                content = content.replace(original_statement, new_statement)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Fixed {fixed_count} imports in {file_path}")
        except Exception as e:
            logger.error(f"Error fixing imports in {file_path}: {e}")
    
    return has_issues, issue_count, fixed_count

def detect_circular_dependencies(files: List[str]) -> List[Tuple[str, str]]:
    """Detect potential circular dependencies between files"""
    # Build import graph
    import_graph = {}
    
    for file_path in files:
        imports = extract_imports(file_path)
        import_graph[file_path] = []
        
        for import_path, _ in imports:
            if import_path.startswith('./') or import_path.startswith('../'):
                # Convert to absolute path
                current_dir = os.path.dirname(file_path)
                try:
                    target_path = os.path.normpath(os.path.join(current_dir, import_path))
                    # Check for various extensions
                    for ext in ['.ts', '.tsx', '.js', '.jsx']:
                        if os.path.exists(target_path + ext):
                            import_graph[file_path].append(target_path + ext)
                            break
                    # Check for directory with index
                    if os.path.isdir(target_path):
                        for ext in ['.ts', '.tsx', '.js', '.jsx']:
                            if os.path.exists(os.path.join(target_path, 'index' + ext)):
                                import_graph[file_path].append(os.path.join(target_path, 'index' + ext))
                                break
                except Exception as e:
                    if Config.VERBOSE:
                        logger.debug(f"Error resolving import {import_path} in {file_path}: {e}")
    
    # Detect cycles using DFS
    circular_deps = []
    
    def check_cycle(current, path, visited):
        if current in path:
            # Found a cycle
            cycle_start = path.index(current)
            cycle = path[cycle_start:] + [current]
            circular_deps.append((cycle[0], cycle[-2]))  # Report the first and last file in cycle
            return
        
        if current in visited:
            return
        
        visited.add(current)
        new_path = path + [current]
        
        for imported in import_graph.get(current, []):
            check_cycle(imported, new_path, visited)
    
    for file_path in import_graph:
        check_cycle(file_path, [], set())
    
    return circular_deps

def generate_report():
    """Generate a comprehensive validation report"""
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "import_validation_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# TypeScript Import Path Validation Report\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Files Checked:** {Config.STATS['files_checked']}\n")
        f.write(f"- **Files With Valid Imports:** {Config.STATS['valid_files']}\n")
        f.write(f"- **Files With Import Issues:** {Config.STATS['files_with_issues']}\n")
        f.write(f"- **Total Import Issues:** {Config.STATS['total_import_issues']}\n")
        f.write(f"- **Fixed Import Issues:** {Config.STATS['fixed_import_issues']}\n")
        f.write(f"- **Circular Dependencies:** {Config.STATS['circular_dependencies']}\n")
        
        f.write("\n## Next Steps\n\n")
        f.write("1. **Resolve remaining import issues:**\n")
        f.write("   ```bash\n")
        f.write("   python validate_import_paths.py --fix\n")
        f.write("   ```\n\n")
        
        f.write("2. **Run TypeScript compiler to validate:**\n")
        f.write("   ```bash\n")
        f.write("   cd ../ipfs_accelerate_js\n")
        f.write("   npx tsc --noEmit\n")
        f.write("   ```\n\n")
        
        f.write("3. **Resolve any circular dependencies:**\n")
        f.write("   - Review reported circular dependencies\n")
        f.write("   - Refactor code to eliminate circular imports\n")
        f.write("   - Consider using dependency injection patterns\n\n")
        
        f.write("4. **Test the build process:**\n")
        f.write("   ```bash\n")
        f.write("   npm run build\n")
        f.write("   ```\n\n")
        
    logger.info(f"Report generated: {report_path}")

def main():
    """Main function"""
    setup_args()
    
    # Find TypeScript files
    ts_files = find_typescript_files()
    
    # Check imports in each file
    for file_path in ts_files:
        Config.STATS["files_checked"] += 1
        has_issues, issue_count, fixed_count = fix_file_imports(file_path)
        
        if has_issues:
            Config.STATS["files_with_issues"] += 1
            Config.STATS["total_import_issues"] += issue_count
            Config.STATS["fixed_import_issues"] += fixed_count
            
            if not Config.VERBOSE:
                logger.info(f"Found {issue_count} import issues in {file_path}")
        else:
            Config.STATS["valid_files"] += 1
    
    # Detect circular dependencies
    circular_deps = detect_circular_dependencies(ts_files)
    Config.STATS["circular_dependencies"] = len(circular_deps)
    
    if circular_deps:
        logger.warning(f"Found {len(circular_deps)} potential circular dependencies")
        if Config.VERBOSE:
            for source, target in circular_deps:
                rel_source = os.path.relpath(source, Config.TARGET_DIR)
                rel_target = os.path.relpath(target, Config.TARGET_DIR)
                logger.warning(f"Circular dependency: {rel_source} <-> {rel_target}")
    
    # Log summary
    logger.info("\nValidation Summary:")
    logger.info(f"Files checked: {Config.STATS['files_checked']}")
    logger.info(f"Files with valid imports: {Config.STATS['valid_files']}")
    logger.info(f"Files with import issues: {Config.STATS['files_with_issues']}")
    logger.info(f"Total import issues: {Config.STATS['total_import_issues']}")
    logger.info(f"Fixed import issues: {Config.STATS['fixed_import_issues']}")
    logger.info(f"Potential circular dependencies: {Config.STATS['circular_dependencies']}")
    
    # Generate report
    generate_report()

if __name__ == "__main__":
    main()