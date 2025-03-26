#!/usr/bin/env python3
"""
Abstract Syntax Tree Analysis Script

This script analyzes the abstract syntax trees of Python files in the repository,
extracting class hierarchies, method signatures, and dependencies to assist in refactoring.
"""

import os
import ast
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ASTAnalyzer:
    """Analyzer for code ASTs."""
    
    def __init__(self, root_dir: str, output_dir: str):
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.classes = {}
        self.functions = {}
        self.imports = {}
        self.dependencies = {}
        self.complexity = {}
        
    def run_analysis(self):
        """Run the full analysis on the codebase."""
        # Find all Python files
        python_files = list(self.root_dir.glob("**/*.py"))
        
        logger.info(f"Found {len(python_files)} Python files")
        
        # Analyze each file
        for file_path in python_files:
            self.analyze_file(file_path)
            
        # Generate reports
        self.generate_class_hierarchy_report()
        self.generate_function_report()
        self.generate_dependency_report()
        self.generate_complexity_report()
        self.generate_summary_report()
        
        logger.info(f"Analysis complete. Reports saved to {self.output_dir}")
        
    def analyze_file(self, file_path: Path):
        """Analyze a single Python file's AST."""
        rel_path = file_path.relative_to(self.root_dir)
        logger.debug(f"Analyzing {rel_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            
            # Extract imports
            self.imports[str(rel_path)] = self._extract_imports(tree)
            
            # Extract classes
            classes = self._extract_classes(tree)
            if classes:
                self.classes[str(rel_path)] = classes
                
            # Extract functions
            functions = self._extract_functions(tree)
            if functions:
                self.functions[str(rel_path)] = functions
                
            # Calculate complexity
            self.complexity[str(rel_path)] = self._calculate_complexity(tree)
            
        except Exception as e:
            logger.error(f"Error analyzing {rel_path}: {e}")
            
    def _extract_imports(self, tree: ast.Module) -> List[Dict[str, Any]]:
        """Extract import statements from AST."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append({
                        "type": "import",
                        "name": name.name,
                        "alias": name.asname
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for name in node.names:
                    imports.append({
                        "type": "importfrom",
                        "module": module,
                        "name": name.name,
                        "alias": name.asname
                    })
                    
        return imports
        
    def _extract_classes(self, tree: ast.Module) -> List[Dict[str, Any]]:
        """Extract class definitions from AST."""
        classes = []
        
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                methods = []
                
                # Get base classes
                bases = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        bases.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        bases.append(f"{self._get_attribute_full_name(base)}")
                
                # Process class methods
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append(self._process_function(item, is_method=True))
                
                # Create class info
                class_info = {
                    "name": node.name,
                    "bases": bases,
                    "methods": methods,
                    "docstring": ast.get_docstring(node)
                }
                
                classes.append(class_info)
                
        return classes
        
    def _extract_functions(self, tree: ast.Module) -> List[Dict[str, Any]]:
        """Extract function definitions from AST."""
        functions = []
        
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                functions.append(self._process_function(node))
                
        return functions
    
    def _process_function(self, node: ast.FunctionDef, is_method: bool = False) -> Dict[str, Any]:
        """Process a function definition."""
        # Extract arguments
        args = []
        for arg in node.args.args:
            arg_name = arg.arg
            
            # Get argument type annotation if available
            arg_type = None
            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    arg_type = arg.annotation.id
                elif isinstance(arg.annotation, ast.Attribute):
                    arg_type = self._get_attribute_full_name(arg.annotation)
                elif isinstance(arg.annotation, ast.Subscript):
                    arg_type = self._get_subscript_name(arg.annotation)
                    
            args.append({
                "name": arg_name,
                "type": arg_type
            })
            
        # Get return type if available
        return_type = None
        if node.returns:
            if isinstance(node.returns, ast.Name):
                return_type = node.returns.id
            elif isinstance(node.returns, ast.Attribute):
                return_type = self._get_attribute_full_name(node.returns)
            elif isinstance(node.returns, ast.Subscript):
                return_type = self._get_subscript_name(node.returns)
                
        # Skip first arg (self) for methods
        if is_method:
            args = args[1:] if args else []
            
        # Calculate cyclomatic complexity
        complexity = self._calculate_function_complexity(node)
            
        return {
            "name": node.name,
            "args": args,
            "return_type": return_type,
            "docstring": ast.get_docstring(node),
            "complexity": complexity
        }
    
    def _get_attribute_full_name(self, node: ast.Attribute) -> str:
        """Get full name of an attribute (e.g., module.Class)."""
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        elif isinstance(node.value, ast.Attribute):
            return f"{self._get_attribute_full_name(node.value)}.{node.attr}"
        return f"?.{node.attr}"
    
    def _get_subscript_name(self, node: ast.Subscript) -> str:
        """Get name of a subscript type (e.g., List[str])."""
        if isinstance(node.value, ast.Name):
            value_name = node.value.id
        elif isinstance(node.value, ast.Attribute):
            value_name = self._get_attribute_full_name(node.value)
        else:
            value_name = "?"
            
        # Try to get subscription
        slice_name = "?"
        
        # Handle different Python versions
        if hasattr(node, "slice"):
            if isinstance(node.slice, ast.Index):
                # Python 3.8 and below
                if hasattr(node.slice, "value"):
                    if isinstance(node.slice.value, ast.Name):
                        slice_name = node.slice.value.id
            else:
                # Python 3.9+
                if isinstance(node.slice, ast.Name):
                    slice_name = node.slice.id
                elif isinstance(node.slice, ast.Tuple):
                    slice_parts = []
                    for elt in node.slice.elts:
                        if isinstance(elt, ast.Name):
                            slice_parts.append(elt.id)
                        else:
                            slice_parts.append("?")
                    slice_name = ", ".join(slice_parts)
                    
        return f"{value_name}[{slice_name}]"
    
    def _calculate_complexity(self, tree: ast.Module) -> Dict[str, Any]:
        """Calculate code complexity metrics."""
        overall_complexity = 1  # Base complexity
        
        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.complexity = 1
                
            def visit_If(self, node):
                self.complexity += 1
                self.generic_visit(node)
                
            def visit_For(self, node):
                self.complexity += 1
                self.generic_visit(node)
                
            def visit_While(self, node):
                self.complexity += 1
                self.generic_visit(node)
                
            def visit_Try(self, node):
                self.complexity += len(node.handlers)  # Each except block
                self.generic_visit(node)
                
            def visit_BoolOp(self, node):
                if isinstance(node.op, (ast.And, ast.Or)):
                    self.complexity += len(node.values) - 1
                self.generic_visit(node)
        
        visitor = ComplexityVisitor()
        visitor.visit(tree)
        overall_complexity = visitor.complexity
        
        # Count lines with code (ignoring comments and blank lines)
        lines = []
        for node in ast.walk(tree):
            if hasattr(node, 'lineno'):
                lines.append(node.lineno)
                
        unique_lines = len(set(lines))
        
        return {
            "cyclomatic_complexity": overall_complexity,
            "lines_of_code": unique_lines
        }
    
    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for item in ast.walk(node):
            if isinstance(item, ast.If) or isinstance(item, ast.For) or isinstance(item, ast.While):
                complexity += 1
            elif isinstance(item, ast.Try):
                complexity += len(item.handlers)  # Each except block
                
        return complexity
                
    def generate_class_hierarchy_report(self):
        """Generate a report of class hierarchies."""
        # Build class hierarchy
        class_hierarchy = {}
        
        # First, collect all classes by name
        all_classes = {}
        for file_path, classes in self.classes.items():
            for cls in classes:
                class_name = cls["name"]
                all_classes[class_name] = {
                    "file": file_path,
                    "info": cls
                }
                
        # Then, build hierarchy based on base classes
        for class_name, data in all_classes.items():
            cls = data["info"]
            bases = cls["bases"]
            
            # Add to hierarchy
            class_hierarchy[class_name] = {
                "file": data["file"],
                "bases": bases,
                "subclasses": [],
                "methods": len(cls["methods"]),
                "docstring": cls["docstring"] is not None
            }
            
            # Add as subclass to parent classes
            for base in bases:
                if base in class_hierarchy:
                    class_hierarchy[base]["subclasses"].append(class_name)
                    
        # Save report
        with open(self.output_dir / "class_hierarchy.json", 'w', encoding='utf-8') as f:
            json.dump(class_hierarchy, f, indent=2)
            
    def generate_function_report(self):
        """Generate a report of function signatures."""
        function_report = {}
        
        for file_path, functions in self.functions.items():
            function_report[file_path] = []
            
            for func in functions:
                signature = {
                    "name": func["name"],
                    "args": [f"{arg['name']}: {arg['type'] or 'Any'}" for arg in func["args"]],
                    "return_type": func["return_type"] or "None",
                    "has_docstring": func["docstring"] is not None,
                    "complexity": func["complexity"]
                }
                function_report[file_path].append(signature)
                
        # Save report
        with open(self.output_dir / "function_signatures.json", 'w', encoding='utf-8') as f:
            json.dump(function_report, f, indent=2)
            
    def generate_dependency_report(self):
        """Generate a report of module dependencies."""
        dependency_report = {}
        
        for file_path, imports in self.imports.items():
            dependency_report[file_path] = {
                "imports": imports,
                "imported_by": []
            }
            
        # Track files that import each file
        for file_path, imports in self.imports.items():
            for imp in imports:
                if imp["type"] == "importfrom":
                    module_parts = imp["module"].split('.')
                    
                    # Try to find a matching Python file
                    for candidate in dependency_report:
                        candidate_parts = os.path.splitext(candidate)[0].split(os.sep)
                        if len(candidate_parts) >= len(module_parts) and candidate_parts[-len(module_parts):] == module_parts:
                            dependency_report[candidate]["imported_by"].append(file_path)
                            
        # Save report
        with open(self.output_dir / "dependencies.json", 'w', encoding='utf-8') as f:
            json.dump(dependency_report, f, indent=2)
            
    def generate_complexity_report(self):
        """Generate a report of code complexity."""
        # Save report
        with open(self.output_dir / "complexity.json", 'w', encoding='utf-8') as f:
            json.dump(self.complexity, f, indent=2)
            
    def generate_summary_report(self):
        """Generate a summary report with insights for refactoring."""
        # Calculate summary statistics
        total_files = len(self.imports)
        total_classes = sum(len(classes) for classes in self.classes.values())
        total_functions = sum(len(funcs) for funcs in self.functions.values())
        
        # Identify highly complex files
        complex_files = sorted(
            [(path, data["cyclomatic_complexity"]) for path, data in self.complexity.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Identify most imported modules
        import_counts = {}
        for file_path, imports in self.imports.items():
            for imp in imports:
                name = imp["module"] + "." + imp["name"] if imp["type"] == "importfrom" else imp["name"]
                import_counts[name] = import_counts.get(name, 0) + 1
                
        common_imports = sorted(
            [(name, count) for name, count in import_counts.items()],
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        # Generate summary
        summary = {
            "total_files_analyzed": total_files,
            "total_classes": total_classes,
            "total_functions": total_functions,
            "most_complex_files": complex_files,
            "most_common_imports": common_imports,
            "refactoring_recommendations": [
                "Create unified ModelConverter base class for all format converters",
                "Implement centralized hardware detection in HardwareDetector",
                "Standardize model file verification with ModelVerifier",
                "Create registry pattern for converter discoverability",
                "Unify logging and error handling across converters",
                "Implement caching system for converted models"
            ]
        }
        
        # Save summary report
        with open(self.output_dir / "summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
            
        # Generate markdown report for easy reading
        self._generate_markdown_summary(summary)
        
    def _generate_markdown_summary(self, summary: Dict[str, Any]):
        """Generate a markdown version of the summary report."""
        md_lines = [
            "# Code Analysis Summary",
            "",
            "## Overview",
            "",
            f"- **Total Files Analyzed**: {summary['total_files_analyzed']}",
            f"- **Total Classes**: {summary['total_classes']}",
            f"- **Total Functions**: {summary['total_functions']}",
            "",
            "## Most Complex Files",
            "",
            "| File | Complexity |",
            "|------|------------|",
        ]
        
        for file_path, complexity in summary["most_complex_files"]:
            md_lines.append(f"| {file_path} | {complexity} |")
            
        md_lines.extend([
            "",
            "## Most Common Imports",
            "",
            "| Module | Usage Count |",
            "|--------|-------------|",
        ])
        
        for module, count in summary["most_common_imports"]:
            md_lines.append(f"| {module} | {count} |")
            
        md_lines.extend([
            "",
            "## Refactoring Recommendations",
            "",
        ])
        
        for i, recommendation in enumerate(summary["refactoring_recommendations"], 1):
            md_lines.append(f"{i}. {recommendation}")
            
        # Save markdown report
        with open(self.output_dir / "summary.md", 'w', encoding='utf-8') as f:
            f.write("\n".join(md_lines))


def main():
    """Main entry point for code analysis script."""
    parser = argparse.ArgumentParser(description="Analyze code AST")
    parser.add_argument("--root", type=str, default=".", help="Root directory of code")
    parser.add_argument("--output", type=str, default="ast_analysis", help="Output directory for reports")
    
    args = parser.parse_args()
    
    analyzer = ASTAnalyzer(args.root, args.output)
    analyzer.run_analysis()
    
    return 0


if __name__ == "__main__":
    main()