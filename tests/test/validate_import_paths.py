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
    
    # Fix common issues with import path before proceeding
    # Remove .ts, .tsx, .js, .jsx, .py extensions
    for ext in ['.ts', '.tsx', '.js', '.jsx', '.py']:
        if import_path.endswith(ext):
            import_path = import_path[:-len(ext)]
            break
    
    # Remove duplicate extensions (e.g., .ts.ts)
    for ext in ['.ts', '.tsx', '.js', '.jsx']:
        double_ext = ext + ext
        if double_ext in import_path:
            import_path = import_path.replace(double_ext, ext)
    
    # Handle $ variables in path (common in auto-generated code)
    import_path = re.sub(r'\$\d+', 'module', import_path)
    
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
                # Directory with index file - append /index to the import
                if not import_path.endswith('/index'):
                    return import_path + '/index'
                return import_path
    
    # Try replacing last path segment with 'index'
    if '/' in import_path:
        base_dir = os.path.dirname(import_path)
        base_with_index = base_dir + '/index'
        test_path_with_index = os.path.normpath(os.path.join(current_dir, base_with_index))
        
        for ext in extensions:
            if os.path.exists(test_path_with_index + ext):
                return base_with_index.replace('\\', '/')
    
    # Try finding a file with similar name but different casing
    base_name = os.path.basename(import_path)
    parent_dir = os.path.dirname(os.path.normpath(os.path.join(current_dir, import_path)))
    
    if os.path.isdir(parent_dir):
        for file in os.listdir(parent_dir):
            file_base, ext = os.path.splitext(file)
            if ext in ['.ts', '.tsx', '.js', '.jsx'] and file_base.lower() == base_name.lower():
                # Found a similar file with different casing
                fixed_path = os.path.join(os.path.dirname(import_path), file_base)
                return fixed_path.replace('\\', '/')
    
    # If we can't find the file, try a more aggressive approach:
    # Look for the module in a different location
    
    # Strategy 1: Check if module can be found in any key directories
    key_dirs = ["src/model/transformers", "src/model/vision", "src/model/audio", 
                "src/hardware/backends", "src/optimization/techniques"]
    module_name = os.path.basename(import_path)
    
    for key_dir in key_dirs:
        key_dir_path = os.path.join(Config.TARGET_DIR, key_dir)
        if not os.path.isdir(key_dir_path):
            continue
            
        for file in os.listdir(key_dir_path):
            file_base, ext = os.path.splitext(file)
            if ext in ['.ts', '.tsx'] and (file_base.lower() == module_name.lower() or 
                                           file_base.lower() == module_name.lower() + '_fixed' or
                                           file_base.lower() == 'test_' + module_name.lower()):
                # Found a potential match in a key directory
                return f'{key_dir}/{file_base}'.replace('\\', '/')
    
    # Strategy 2: Try to use the module through the created index file
    if '/' in import_path:
        dir_part = import_path.split('/')[0]
        for key_dir in key_dirs:
            if key_dir.endswith(dir_part):
                return f'{key_dir}/index'.replace('\\', '/')
    
    # Could not fix the import, keep original
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
            
            # Also fix any syntax issues related to braces and array destructuring 
            if file_path.endswith('.ts') or file_path.endswith('.tsx'):
                # Fix duplicate braces
                content = re.sub(r'}\s*}([^}])', r'}\1', content)
                content = re.sub(r'([^{]){(\s*){', r'\1{\2', content)
                
                # Fix issues with array destructuring in imports
                content = re.sub(r'import\s*\{\s*\[\s*([^}]+)\s*\]\s*\}', r'import { \1 }', content)
                
                # Fix duplicate closing bracket/parentheses
                content = re.sub(r'\}\);', r'});', content)
                content = re.sub(r'\]\);', r']);', content)
                
                # Fix missing semicolons in import statements
                content = re.sub(r'(import\s+[^;]+)$', r'\1;', content, flags=re.MULTILINE)
                
                # Fix repeated parentheses (more than 2)
                content = re.sub(r'\){3,}', r'))', content)
                content = re.sub(r'\({3,}', r'((', content)
                
                # Fix repeated brackets (more than 2)
                content = re.sub(r'\]{3,}', r']]', content)
                content = re.sub(r'\[{3,}', r'[[', content)
                
                # Fix repeated braces (more than 2)
                content = re.sub(r'\}{3,}', r'}}', content)
                content = re.sub(r'\{{3,}', r'{{', content)
                
                # Fix commas in brackets
                content = re.sub(r'\[,+', r'[', content)
                content = re.sub(r',+\]', r']', content)
                
                # Fix bad template literals
                content = re.sub(r'\${(\$\d+)}', r'${$1}', content)
                content = re.sub(r'\$\{(\$\d+)\}', r'${$1}', content)
                
                # Fix incorrect import syntax with $
                content = re.sub(r'import\s+\*\s+as\s+\$\d+', r'import * as module', content)
                content = re.sub(r'import\s+\{\s*\$\d+\s*\}', r'import { module }', content)
                content = re.sub(r'from\s+"\$\d+"', r'from "./module"', content)
                
                # Fix repeated punctuation
                content = re.sub(r':+', r':', content)
                content = re.sub(r';+', r';', content)
                content = re.sub(r',+', r',', content)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Fixed {fixed_count} imports in {file_path}")
        except Exception as e:
            logger.error(f"Error fixing imports in {file_path}: {e}")
    
    return has_issues, issue_count, fixed_count

def check_for_destructuring_issues(file_path: str) -> bool:
    """Check file for array destructuring syntax issues"""
    if not (file_path.endswith('.ts') or file_path.endswith('.tsx')):
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for problematic array destructuring patterns
        problems = [
            r'const\s*\[\s*\[',  # Nested array destructuring
            r'const\s*\{[^}]*\[',  # Mixed object/array destructuring
            r'import\s*\{\s*\[',  # Array destructuring in imports
            r'}\);',  # Extra closing parenthesis
            r'}\);',  # Extra closing bracket
        ]
        
        for pattern in problems:
            if re.search(pattern, content):
                return True
        
        return False
    except Exception as e:
        logger.error(f"Error checking destructuring in {file_path}: {e}")
        return False

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

def create_index_files():
    """Create index.ts files in directories to help resolve imports"""
    if not Config.FIX_IMPORTS:
        return
    
    logger.info("Creating index.ts files in key directories...")
    
    # Common directories where we need index files
    key_dirs = [
        "src/model/transformers",
        "src/model/vision",
        "src/model/audio",
        "src/hardware/backends",
        "src/optimization/techniques",
        "src/quantization/techniques",
        "src/browser/optimizations",
        "src/worker/webgpu",
        "src/worker/webnn"
    ]
    
    for rel_dir in key_dirs:
        dir_path = os.path.join(Config.TARGET_DIR, rel_dir)
        index_path = os.path.join(dir_path, "index.ts")
        
        if not os.path.exists(dir_path):
            continue
            
        if os.path.exists(index_path):
            logger.info(f"Index already exists: {rel_dir}/index.ts")
            continue
            
        # Find all .ts files in the directory
        ts_files = []
        for file in os.listdir(dir_path):
            if file.endswith('.ts') and not file == 'index.ts' and not file.endswith('.d.ts'):
                ts_files.append(file)
        
        if not ts_files:
            continue
            
        # Create an index file that exports everything
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write("// Auto-generated index file\n\n")
            
            for file in ts_files:
                module_name = os.path.splitext(file)[0]
                f.write(f'export * from "./{module_name}";\n')
        
        logger.info(f"Created index file: {rel_dir}/index.ts")

def fix_common_import_patterns():
    """Fix common import patterns that might be causing issues"""
    if not Config.FIX_IMPORTS:
        return
        
    logger.info("Fixing common import patterns across files...")
    
    # Find TypeScript files
    ts_files = []
    for root, _, files in os.walk(Config.TARGET_DIR):
        for file in files:
            if file.endswith((".ts", ".tsx")) and not file.endswith(".d.ts"):
                ts_files.append(os.path.join(root, file))
    
    # Common problematic patterns
    replacements = [
        # Fix index imports
        (r'from\s+[\'"]([^\'"]+)/(\w+)[\'"]', 'from "\\1/\\2/index"'),
        # Fix relative imports without extension
        (r'from\s+[\'"](\.[^\'"]*)/((?!index)[^/\'"]+)[\'"]', 'from "\\1/\\2/index"'),
        # Fix double extensions (.ts.ts)
        (r'from\s+[\'"]([^\'"]+)\.ts\.ts[\'"]', 'from "\\1"'),
        # Remove .ts extensions from imports
        (r'from\s+[\'"]([^\'"]+)\.ts[\'"]', 'from "\\1"'),
        # Fix imports with odd characters
        (r'from\s+[\'"]([^\'"]+)\$(\d+)[\'"]', 'from "\\1"'),
        # Fix imports with $ variables
        (r'import\s+\$\d+\s+from', 'import module from'),
        # Fix imports with .py extension
        (r'from\s+[\'"]([^\'"]+)\.py[\'"]', 'from "\\1"'),
    ]
    
    # Apply fixes
    fixed_files = 0
    for file_path in ts_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            modified = False
            new_content = content
            
            for pattern, replacement in replacements:
                # Check if pattern exists in file
                if re.search(pattern, new_content):
                    # Apply replacement
                    new_content = re.sub(pattern, replacement, new_content)
                    modified = True
            
            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                fixed_files += 1
                if Config.VERBOSE:
                    logger.debug(f"Fixed import patterns in {file_path}")
        except Exception as e:
            logger.error(f"Error fixing common patterns in {file_path}: {e}")
    
    logger.info(f"Fixed import patterns in {fixed_files} files")

def main():
    """Main function"""
    setup_args()
    
    # Find TypeScript files
    ts_files = find_typescript_files()
    
    # Create index files to help resolve imports
    create_index_files()
    
    # Fix common import patterns
    fix_common_import_patterns()
    
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
    
    # Do another pass with different fixing strategies if many issues remain
    if Config.FIX_IMPORTS and Config.STATS["files_with_issues"] > 100:
        logger.info("Performing additional pass to fix stubborn import issues...")
        for file_path in ts_files:
            has_issues, issue_count, fixed_count = fix_file_imports(file_path)
            if has_issues and fixed_count > 0:
                Config.STATS["fixed_import_issues"] += fixed_count
    
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