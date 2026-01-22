#!/usr/bin/env python3
# Script to fix import paths in TypeScript files
# This script handles common TypeScript import path issues

import os
import re
import sys
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SDK_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ipfs_accelerate_js"))

def find_ts_files() -> List[str]:
    """Find all TypeScript files in the SDK directory"""
    ts_files = []
    for root, _, files in os.walk(SDK_DIR):
        for file in files:
            if file.endswith((".ts", ".tsx")) and not file.endswith(".d.ts"):
                ts_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(ts_files)} TypeScript files")
    return ts_files

def fix_imports_in_file(file_path: str) -> int:
    """Fix import paths in a TypeScript file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for common import patterns
        import_patterns = [
            # Relative imports like "../foo" or "./foo"
            r'import\s+(?:\{[^}]*\}|\*\s+as\s+\w+|\w+)\s+from\s+[\'"]([\.\/][^\'"]+)[\'"]',
            # Bare imports like "foo"
            r'import\s+[\'"]([^\'"\.\/][^\'"]+)[\'"]',
            # Require statements
            r'require\(\s*[\'"]([^\'"]+)[\'"]\s*\)'
        ]
        
        changed = False
        for pattern in import_patterns:
            # Find all imports
            for match in re.finditer(pattern, content):
                import_path = match.group(1)
                
                # Fix common issues
                fixed_path = import_path
                
                # 1. If the import path lacks extension but should have one
                if not import_path.endswith(('.ts', '.tsx', '.js', '.jsx')) and not import_path.endswith('/'):
                    # Try fixing by adding extension
                    current_dir = os.path.dirname(file_path)
                    
                    # Try with .ts first, then .js
                    for ext in ['.ts', '.js']:
                        test_path = os.path.normpath(os.path.join(current_dir, import_path + ext))
                        if os.path.exists(test_path):
                            # We found the file with extension, but don't add it to the import
                            # TypeScript convention is to not include extensions
                            fixed_path = import_path
                            break
                    
                    # If it's a directory, make sure it ends with /
                    dir_path = os.path.normpath(os.path.join(current_dir, import_path))
                    if os.path.isdir(dir_path):
                        if not fixed_path.endswith('/'):
                            fixed_path = fixed_path + '/'
                
                # 2. If there's an index.ts file in the target directory
                if not import_path.endswith(('.ts', '.tsx', '.js', '.jsx')):
                    current_dir = os.path.dirname(file_path)
                    dir_path = os.path.normpath(os.path.join(current_dir, import_path))
                    
                    if os.path.isdir(dir_path):
                        for ext in ['.ts', '.js']:
                            if os.path.exists(os.path.join(dir_path, 'index' + ext)):
                                # The directory has an index file, so we're good
                                if not fixed_path.endswith('/'):
                                    fixed_path = fixed_path + '/'
                                break
                
                # 3. Fix relative paths that might be wrong
                if fixed_path.startswith('.') and '/' in fixed_path:
                    # This is a relative path, check if it exists
                    current_dir = os.path.dirname(file_path)
                    target_path = os.path.normpath(os.path.join(current_dir, fixed_path))
                    
                    # Handle missing file case
                    if not os.path.exists(target_path) and not os.path.isdir(target_path):
                        # Try finding a similar file (different case, etc.)
                        parent_dir = os.path.dirname(target_path)
                        base_name = os.path.basename(target_path)
                        
                        if os.path.isdir(parent_dir):
                            # Look for a file with similar name
                            for name in os.listdir(parent_dir):
                                if name.lower() == base_name.lower():
                                    # Found a match with different case
                                    path_parts = fixed_path.split('/')
                                    path_parts[-1] = name  # Replace the last part with the correct case
                                    fixed_path = '/'.join(path_parts)
                                    break
                
                # 4. If it's importing from src/ but should be relative
                if fixed_path.startswith('src/'):
                    # Convert to relative path
                    rel_path = os.path.relpath(
                        os.path.join(SDK_DIR, fixed_path),
                        os.path.dirname(file_path)
                    )
                    
                    # Add ./ prefix if needed
                    if not rel_path.startswith('.'):
                        rel_path = './' + rel_path
                    
                    fixed_path = rel_path
                
                # 5. Clean up paths with double slashes
                fixed_path = re.sub(r'\/+', '/', fixed_path)
                
                # If the path changed, update the content
                if fixed_path != import_path:
                    new_import = match.group(0).replace(f'"{import_path}"', f'"{fixed_path}"').replace(f"'{import_path}'", f"'{fixed_path}'")
                    content = content.replace(match.group(0), new_import)
                    changed = True
                    logger.info(f"Fixed import in {file_path}: {import_path} -> {fixed_path}")
        
        # Also clean up duplicate braces and fix destructuring issues
        if '{[' in content or ']}' in content:
            content = re.sub(r'import\s*\{\s*\[\s*([^}]+)\s*\]\s*\}', r'import { \1 }', content)
            changed = True
            logger.info(f"Fixed destructuring in {file_path}")
        
        # Fix missing semicolons in import statements
        if re.search(r'import\s+[^;]+$', content, re.MULTILINE):
            content = re.sub(r'(import\s+[^;]+)$', r'\1;', content, flags=re.MULTILINE)
            changed = True
            logger.info(f"Fixed missing semicolons in {file_path}")
        
        # Write the fixed content back to the file if changes were made
        if changed:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return 1  # Return count of files fixed
        
        return 0  # No changes made
        
    except Exception as e:
        logger.error(f"Error fixing imports in {file_path}: {e}")
        return 0

def main():
    """Main function"""
    logger.info(f"SDK Directory: {SDK_DIR}")
    
    # Find TypeScript files
    ts_files = find_ts_files()
    
    # Fix imports in each file
    fixed_count = 0
    for file_path in ts_files:
        fixed = fix_imports_in_file(file_path)
        fixed_count += fixed
    
    # Log summary
    logger.info(f"Fixed imports in {fixed_count} of {len(ts_files)} files")
    
    # Generate report
    with open("import_paths_fix_report.md", "w") as f:
        f.write("# Import Paths Fix Report\n\n")
        f.write(f"- **Total Files Processed:** {len(ts_files)}\n")
        f.write(f"- **Files Fixed:** {fixed_count}\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("1. **Validate TypeScript compilation:**\n")
        f.write("   ```bash\n")
        f.write("   cd ../ipfs_accelerate_js\n")
        f.write("   npx tsc --noEmit\n")
        f.write("   ```\n\n")
        
        f.write("2. **Test the build process:**\n")
        f.write("   ```bash\n")
        f.write("   npm run build\n")
        f.write("   ```\n\n")
        
        f.write("3. **Create NPM Package:**\n")
        f.write("   ```bash\n")
        f.write("   npm pack\n")
        f.write("   ```\n")
    
    logger.info("Report generated: import_paths_fix_report.md")

if __name__ == "__main__":
    main()