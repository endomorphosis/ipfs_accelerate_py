#!/usr/bin/env python3
"""
Check all imports in test/ directory for broken references after refactoring.
"""
import os
import ast
import sys
from pathlib import Path
from collections import defaultdict

class ImportChecker(ast.NodeVisitor):
    def __init__(self, filepath):
        self.filepath = filepath
        self.imports = []
        self.from_imports = []
        
    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append({
                'module': alias.name,
                'lineno': node.lineno,
                'type': 'import'
            })
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        module = node.module or ''
        for alias in node.names:
            self.from_imports.append({
                'module': module,
                'name': alias.name,
                'lineno': node.lineno,
                'level': node.level,
                'type': 'from_import'
            })
        self.generic_visit(node)

def check_file_imports(filepath):
    """Parse a Python file and extract all imports."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content, filename=str(filepath))
        checker = ImportChecker(filepath)
        checker.visit(tree)
        
        return checker.imports, checker.from_imports
    except SyntaxError as e:
        print(f"Syntax error in {filepath}: {e}")
        return [], []
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return [], []

def find_python_files(directory):
    """Find all Python files in directory."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip __pycache__ and virtual environments
        dirs[:] = [d for d in dirs if d not in ['__pycache__', 'venv', 'venvs', '.git']]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def check_import_exists(base_path, import_info, file_path):
    """Check if an import can be resolved."""
    issues = []
    
    if import_info['type'] == 'from_import':
        module = import_info['module']
        level = import_info['level']
        
        # Handle relative imports
        if level > 0:
            # Calculate the base directory for relative import
            current_dir = os.path.dirname(file_path)
            for _ in range(level - 1):
                current_dir = os.path.dirname(current_dir)
            
            if module:
                module_path = os.path.join(current_dir, module.replace('.', os.sep))
            else:
                module_path = current_dir
            
            # Check if it's a package (has __init__.py) or a module (.py file)
            if not os.path.exists(module_path):
                module_path_py = module_path + '.py'
                module_path_init = os.path.join(module_path, '__init__.py')
                
                if not os.path.exists(module_path_py) and not os.path.exists(module_path_init):
                    issues.append({
                        'file': file_path,
                        'line': import_info['lineno'],
                        'type': 'from_import',
                        'module': module,
                        'level': level,
                        'issue': f"Relative import module not found: {module_path}"
                    })
        
        # Check test.* imports (common pattern in refactored code)
        elif module.startswith('test.'):
            parts = module.split('.')
            module_path = os.path.join(base_path, 'test', *parts[1:])
            
            # Check if it's a valid module
            if not os.path.exists(module_path):
                module_path_py = module_path + '.py'
                module_path_init = os.path.join(module_path, '__init__.py')
                
                if not os.path.exists(module_path_py) and not os.path.exists(module_path_init):
                    issues.append({
                        'file': file_path,
                        'line': import_info['lineno'],
                        'type': 'from_import',
                        'module': module,
                        'issue': f"Module not found: {module_path}"
                    })
    
    return issues

def main():
    base_path = '/home/runner/work/ipfs_accelerate_py/ipfs_accelerate_py'
    test_dir = os.path.join(base_path, 'test')
    
    print("=" * 80)
    print("Checking imports in test/ directory")
    print("=" * 80)
    
    python_files = find_python_files(test_dir)
    print(f"\nFound {len(python_files)} Python files")
    
    all_issues = []
    files_with_test_imports = []
    
    for filepath in python_files:
        imports, from_imports = check_file_imports(filepath)
        
        # Check for test.* imports
        test_imports = []
        for imp in from_imports:
            if imp['module'].startswith('test.'):
                test_imports.append(imp)
                
        if test_imports:
            files_with_test_imports.append((filepath, test_imports))
            
        # Check if imports can be resolved
        for imp in from_imports:
            issues = check_import_exists(base_path, imp, filepath)
            all_issues.extend(issues)
    
    # Report files with test.* imports
    print(f"\n{'=' * 80}")
    print(f"Files with test.* imports: {len(files_with_test_imports)}")
    print("=" * 80)
    
    if files_with_test_imports:
        for filepath, imports in sorted(files_with_test_imports)[:20]:  # Show first 20
            rel_path = os.path.relpath(filepath, base_path)
            print(f"\n{rel_path}:")
            for imp in imports[:5]:  # Show first 5 imports per file
                print(f"  Line {imp['lineno']}: from {imp['module']} import {imp['name']}")
    
    # Report issues
    print(f"\n{'=' * 80}")
    print(f"Potential import issues found: {len(all_issues)}")
    print("=" * 80)
    
    if all_issues:
        issue_groups = defaultdict(list)
        for issue in all_issues:
            key = (issue['module'], issue['issue'])
            issue_groups[key].append(issue)
        
        for (module, issue_msg), issues_list in sorted(issue_groups.items()):
            print(f"\n{issue_msg}")
            print(f"  Module: {module}")
            print(f"  Affected files: {len(issues_list)}")
            for issue in issues_list[:5]:  # Show first 5 files
                rel_path = os.path.relpath(issue['file'], base_path)
                print(f"    - {rel_path}:{issue['line']}")
    else:
        print("\n✓ No obvious import issues detected!")
    
    return len(all_issues)

if __name__ == '__main__':
    sys.exit(main())
