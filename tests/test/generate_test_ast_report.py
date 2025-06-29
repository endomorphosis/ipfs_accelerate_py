#!/usr/bin/env python3
"""
Generate Abstract Syntax Tree (AST) reports for test files in the project.

This script analyzes Python test files and generates a structured report
containing AST information about classes, methods, and their relationships.
The report is saved as JSON for further analysis and visualization.
"""

import os
import ast
import json
import glob
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple


class TestASTVisitor(ast.NodeVisitor):
    """AST visitor that extracts information about test classes and methods."""
    
    def __init__(self):
        self.classes = []
        self.functions = []
        self.imports = []
        self.current_class = None
        
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Process class definitions in the AST."""
        class_info = {
            'name': node.name,
            'line_number': node.lineno,
            'end_line_number': self._get_end_line(node),
            'methods': [],
            'bases': [self._get_name(base) for base in node.bases],
            'decorators': [self._get_name(d) for d in node.decorator_list],
        }
        
        old_class = self.current_class
        self.current_class = class_info
        self.classes.append(class_info)
        
        # Visit all child nodes
        for child in node.body:
            self.visit(child)
            
        self.current_class = old_class
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Process function/method definitions in the AST."""
        func_info = {
            'name': node.name,
            'line_number': node.lineno,
            'end_line_number': self._get_end_line(node),
            'args': self._get_args(node.args),
            'decorators': [self._get_name(d) for d in node.decorator_list],
            'is_test': node.name.startswith('test_') or any(
                'pytest' in self._get_name(d).lower() for d in node.decorator_list
            ),
        }
        
        # Try to extract docstring if available
        docstring = ast.get_docstring(node)
        if docstring:
            func_info['docstring'] = docstring
        
        # Add function to its class if we're inside a class
        if self.current_class:
            self.current_class['methods'].append(func_info)
        else:
            self.functions.append(func_info)
    
    def visit_Import(self, node: ast.Import) -> None:
        """Process import statements."""
        for name in node.names:
            self.imports.append({
                'module': name.name,
                'alias': name.asname,
                'line_number': node.lineno,
                'type': 'import',
            })
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Process from-import statements."""
        module = node.module or ''
        for name in node.names:
            self.imports.append({
                'module': module,
                'name': name.name,
                'alias': name.asname,
                'line_number': node.lineno,
                'type': 'importfrom',
            })
    
    def _get_name(self, node: ast.expr) -> str:
        """Extract name from different node types."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            return self._get_name(node.func)
        elif isinstance(node, ast.Subscript):
            return f"{self._get_name(node.value)}[...]"
        else:
            return str(node.__class__.__name__)
    
    def _get_args(self, args: ast.arguments) -> List[str]:
        """Extract argument names from a function definition."""
        arg_list = []
        
        # Add positional arguments
        for arg in args.args:
            arg_list.append(arg.arg)
        
        # Add *args if present
        if args.vararg:
            arg_list.append(f"*{args.vararg.arg}")
        
        # Add keyword-only args if present
        for arg in args.kwonlyargs:
            arg_list.append(arg.arg)
        
        # Add **kwargs if present
        if args.kwarg:
            arg_list.append(f"**{args.kwarg.arg}")
        
        return arg_list
    
    def _get_end_line(self, node: ast.AST) -> int:
        """Get the ending line number of a node."""
        if hasattr(node, 'end_lineno') and node.end_lineno is not None:
            return node.end_lineno
        
        # For Python < 3.8 compatibility, find the max line number of all child nodes
        max_line = node.lineno
        for child in ast.iter_child_nodes(node):
            if hasattr(child, 'lineno'):
                end_line = self._get_end_line(child)
                max_line = max(max_line, end_line)
        return max_line


def analyze_file(file_path: str) -> Dict[str, Any]:
    """Analyze a single Python file and return its AST info."""
    with open(file_path, 'r', encoding='utf-8') as f:
        file_content = f.read()
    
    try:
        tree = ast.parse(file_content)
        visitor = TestASTVisitor()
        visitor.visit(tree)
        
        # Get file stats
        stats = os.stat(file_path)
        
        # Check if this is a test file
        is_test_file = os.path.basename(file_path).startswith('test_') or any(
            c['name'].startswith('Test') for c in visitor.classes
        )
        
        return {
            'filename': os.path.basename(file_path),
            'path': file_path,
            'size_bytes': stats.st_size,
            'last_modified': datetime.fromtimestamp(stats.st_mtime).isoformat(),
            'is_test_file': is_test_file,
            'classes': visitor.classes,
            'functions': visitor.functions,
            'imports': visitor.imports,
            'metrics': {
                'num_classes': len(visitor.classes),
                'num_methods': sum(len(c['methods']) for c in visitor.classes),
                'num_functions': len(visitor.functions),
                'num_imports': len(visitor.imports),
                'num_test_methods': sum(
                    1 for c in visitor.classes 
                    for m in c['methods'] 
                    if m.get('is_test', False)
                ) + sum(1 for f in visitor.functions if f.get('is_test', False)),
            }
        }
    except SyntaxError as e:
        return {
            'filename': os.path.basename(file_path),
            'path': file_path,
            'error': f"Syntax error: {str(e)}",
            'error_line': e.lineno,
            'is_test_file': False,
            'classes': [],
            'functions': [],
            'imports': [],
            'metrics': {
                'num_classes': 0,
                'num_methods': 0,
                'num_functions': 0,
                'num_imports': 0,
                'num_test_methods': 0,
            }
        }


def find_test_files(directory: str, pattern: str = "**/*.py") -> List[str]:
    """Find all Python files in the given directory matching the pattern."""
    files = glob.glob(os.path.join(directory, pattern), recursive=True)
    # Filter to only include files that actually exist
    return [f for f in files if os.path.isfile(f)]


def generate_report(files: List[str], output_file: str, test_only: bool = True) -> None:
    """Generate a report for all Python files and save it as JSON."""
    report = {
        'generated_at': datetime.now().isoformat(),
        'num_files_analyzed': len(files),
        'files': []
    }
    
    print(f"Analyzing {len(files)} Python files...")
    
    for file_path in files:
        file_info = analyze_file(file_path)
        
        # Skip non-test files if test_only is True
        if test_only and not file_info['is_test_file']:
            continue
            
        report['files'].append(file_info)
    
    # Add summary metrics
    report['summary'] = {
        'total_files': len(report['files']),
        'total_classes': sum(f['metrics']['num_classes'] for f in report['files']),
        'total_methods': sum(f['metrics']['num_methods'] for f in report['files']),
        'total_functions': sum(f['metrics']['num_functions'] for f in report['files']),
        'total_imports': sum(f['metrics']['num_imports'] for f in report['files']),
        'total_test_methods': sum(f['metrics']['num_test_methods'] for f in report['files']),
    }
    
    # Generate class inheritance relationships
    class_relationships = []
    class_dict = {}
    
    # First pass: build a dictionary of all classes
    for file_info in report['files']:
        for class_info in file_info['classes']:
            full_name = f"{file_info['filename']}:{class_info['name']}"
            class_dict[full_name] = {
                'file': file_info['filename'],
                'name': class_info['name'],
                'bases': class_info['bases'],
                'num_methods': len(class_info['methods']),
                'num_test_methods': sum(1 for m in class_info['methods'] if m.get('is_test', False))
            }
    
    # Second pass: build relationships
    for class_name, class_info in class_dict.items():
        for base in class_info['bases']:
            # Try to find base class in our dictionary
            for potential_base in class_dict.keys():
                if class_dict[potential_base]['name'] == base:
                    class_relationships.append({
                        'child': class_name,
                        'parent': potential_base
                    })
    
    report['class_relationships'] = class_relationships
    
    # Save the report
    print(f"Writing report to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"Report generated successfully!")
    print(f"Total files analyzed: {report['summary']['total_files']}")
    print(f"Total classes: {report['summary']['total_classes']}")
    print(f"Total test methods: {report['summary']['total_test_methods']}")


def main():
    parser = argparse.ArgumentParser(description='Generate AST report for Python test files.')
    parser.add_argument('--directory', '-d', type=str, default='.', 
                      help='Directory to analyze (default: current directory)')
    parser.add_argument('--output', '-o', type=str, default='test_ast_report.json',
                      help='Output JSON file (default: test_ast_report.json)')
    parser.add_argument('--pattern', '-p', type=str, default='**/*.py',
                      help='File pattern to match (default: **/*.py)')
    parser.add_argument('--all', '-a', action='store_true',
                      help='Include all Python files, not just test files')
    
    args = parser.parse_args()
    
    # Find all Python files
    files = find_test_files(args.directory, args.pattern)
    
    # Generate the report
    generate_report(files, args.output, not args.all)


if __name__ == '__main__':
    main()