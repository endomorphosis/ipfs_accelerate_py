#!/usr/bin/env python3
"""
Generator AST Analyzer

This script analyzes the abstract syntax tree (AST) of generator-related Python files to understand
their structure, dependencies, and components. It provides insights into classes, methods, functions,
imports, and other code elements to inform the refactoring process.

Usage:
    python generator_ast_analyzer.py --path PATH [--output OUTPUT] [--pattern PATTERN]

Arguments:
    --path PATH       Path to analyze (file or directory)
    --output OUTPUT   Output file for the analysis report (default: stdout)
    --pattern PATTERN Glob pattern for files to analyze (default: "**/*generator*.py")
    --verbose         Enable verbose output
    --summary         Generate only summary information
    --dependencies    Focus on dependency analysis
    --templates       Focus on template analysis
"""

import os
import sys
import ast
import json
import glob
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Set, Optional, Union, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeneratorASTAnalyzer:
    """Analyzer for generator-related Python files."""
    
    def __init__(self, verbose=False):
        """Initialize the analyzer."""
        self.verbose = verbose
        self.stats = {
            "files_analyzed": 0,
            "total_lines": 0,
            "classes": 0,
            "methods": 0,
            "functions": 0,
            "imports": 0,
            "try_except_blocks": 0
        }
        self.dependencies = set()
        self.class_info = {}
        self.function_info = {}
        self.import_info = {}
        self.template_info = {}
        self.hardware_detection_info = {}
        self.model_registry_info = {}
        
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single Python file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            lines = content.split("\n")
            self.stats["total_lines"] += len(lines)
            self.stats["files_analyzed"] += 1
            
            # Parse AST
            try:
                tree = ast.parse(content)
                file_info = self._analyze_ast(tree, file_path)
                if self.verbose:
                    logger.info(f"Analyzed {file_path}: {len(lines)} lines")
                return file_info
            except SyntaxError as e:
                logger.error(f"Syntax error in {file_path}: {e}")
                return {"error": str(e)}
                
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return {"error": str(e)}
    
    def analyze_directory(self, directory_path: str, pattern: str = "**/*generator*.py") -> Dict[str, Any]:
        """Analyze all Python files in a directory matching the pattern."""
        results = {}
        
        # Find all Python files matching the pattern
        glob_pattern = os.path.join(directory_path, pattern)
        files = glob.glob(glob_pattern, recursive=True)
        
        if not files:
            logger.warning(f"No files found matching pattern '{pattern}' in '{directory_path}'")
            return results
            
        logger.info(f"Found {len(files)} files to analyze")
        
        # Analyze each file
        for file_path in files:
            if os.path.isfile(file_path) and file_path.endswith(".py"):
                results[file_path] = self.analyze_file(file_path)
                
        return results
    
    def _analyze_ast(self, tree: ast.AST, file_path: str) -> Dict[str, Any]:
        """Analyze the AST of a Python file."""
        file_info = {
            "classes": {},
            "functions": {},
            "imports": [],
            "try_except_blocks": [],
            "hardware_detection": {},
            "model_registry": {},
            "templates": {}
        }
        
        # Visit all nodes
        for node in ast.walk(tree):
            # Analyze classes
            if isinstance(node, ast.ClassDef):
                self.stats["classes"] += 1
                class_info = self._analyze_class(node)
                file_info["classes"][node.name] = class_info
                self.class_info[f"{file_path}:{node.name}"] = class_info
                
                # Check if this might be a template class
                if "template" in node.name.lower() or any("template" in base.id.lower() for base in node.bases if isinstance(base, ast.Name)):
                    file_info["templates"][node.name] = class_info
                    self.template_info[f"{file_path}:{node.name}"] = class_info
                
            # Analyze functions
            elif isinstance(node, ast.FunctionDef) and not isinstance(node.parent, ast.ClassDef):
                self.stats["functions"] += 1
                function_info = self._analyze_function(node)
                file_info["functions"][node.name] = function_info
                self.function_info[f"{file_path}:{node.name}"] = function_info
                
                # Check if this might be related to hardware detection
                if "hardware" in node.name.lower() or "detect" in node.name.lower():
                    file_info["hardware_detection"][node.name] = function_info
                    self.hardware_detection_info[f"{file_path}:{node.name}"] = function_info
                    
                # Check if this might be related to model registry
                if "model" in node.name.lower() and ("registry" in node.name.lower() or "register" in node.name.lower()):
                    file_info["model_registry"][node.name] = function_info
                    self.model_registry_info[f"{file_path}:{node.name}"] = function_info
            
            # Analyze imports
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                self.stats["imports"] += 1
                import_info = self._analyze_import(node)
                file_info["imports"].extend(import_info)
                
                for imp in import_info:
                    module = imp.get("module", "")
                    name = imp.get("name", "")
                    full_import = f"{module}.{name}" if module and name else module or name
                    self.dependencies.add(full_import)
                    
                    if full_import not in self.import_info:
                        self.import_info[full_import] = 0
                    self.import_info[full_import] += 1
            
            # Analyze try-except blocks
            elif isinstance(node, ast.Try):
                self.stats["try_except_blocks"] += 1
                try_except_info = self._analyze_try_except(node)
                file_info["try_except_blocks"].append(try_except_info)
                
        return file_info
    
    def _analyze_class(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Analyze a class definition."""
        class_info = {
            "name": node.name,
            "bases": [self._get_name(base) for base in node.bases],
            "methods": {},
            "attributes": [],
            "lineno": node.lineno,
            "doc": ast.get_docstring(node)
        }
        
        # Visit all nodes in the class
        for child in node.body:
            # Analyze methods
            if isinstance(child, ast.FunctionDef):
                self.stats["methods"] += 1
                method_info = self._analyze_function(child)
                class_info["methods"][child.name] = method_info
                
            # Analyze attributes
            elif isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        class_info["attributes"].append({
                            "name": target.id,
                            "value": self._get_value(child.value),
                            "lineno": child.lineno
                        })
                        
        return class_info
    
    def _analyze_function(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze a function or method definition."""
        function_info = {
            "name": node.name,
            "args": self._analyze_arguments(node.args),
            "returns": self._get_name(node.returns) if node.returns else None,
            "lineno": node.lineno,
            "doc": ast.get_docstring(node),
            "calls": [],
            "has_hardware_detection": False,
            "has_model_selection": False,
            "has_template_rendering": False
        }
        
        # Visit all nodes in the function
        for child in ast.walk(node):
            # Analyze function calls
            if isinstance(child, ast.Call):
                call_info = self._analyze_call(child)
                function_info["calls"].append(call_info)
                
                # Check for specific functionality
                func_name = call_info.get("name", "").lower()
                if func_name:
                    if any(keyword in func_name for keyword in ["hardware", "cuda", "gpu", "device", "detect"]):
                        function_info["has_hardware_detection"] = True
                    if any(keyword in func_name for keyword in ["model", "select", "registry", "transform"]):
                        function_info["has_model_selection"] = True
                    if any(keyword in func_name for keyword in ["template", "render", "generate"]):
                        function_info["has_template_rendering"] = True
                        
        return function_info
    
    def _analyze_arguments(self, args: ast.arguments) -> Dict[str, Any]:
        """Analyze function arguments."""
        arg_info = {
            "positional": [],
            "keyword": [],
            "vararg": None,
            "kwarg": None
        }
        
        # Analyze positional arguments
        for arg in args.args:
            arg_info["positional"].append({
                "name": arg.arg,
                "annotation": self._get_name(arg.annotation) if hasattr(arg, "annotation") and arg.annotation else None
            })
            
        # Analyze keyword-only arguments
        for arg in args.kwonlyargs:
            arg_info["keyword"].append({
                "name": arg.arg,
                "annotation": self._get_name(arg.annotation) if hasattr(arg, "annotation") and arg.annotation else None
            })
            
        # Analyze *args
        if args.vararg:
            arg_info["vararg"] = {
                "name": args.vararg.arg,
                "annotation": self._get_name(args.vararg.annotation) if hasattr(args.vararg, "annotation") and args.vararg.annotation else None
            }
            
        # Analyze **kwargs
        if args.kwarg:
            arg_info["kwarg"] = {
                "name": args.kwarg.arg,
                "annotation": self._get_name(args.kwarg.annotation) if hasattr(args.kwarg, "annotation") and args.kwarg.annotation else None
            }
            
        return arg_info
    
    def _analyze_call(self, node: ast.Call) -> Dict[str, Any]:
        """Analyze a function call."""
        call_info = {
            "name": self._get_name(node.func),
            "args": [self._get_value(arg) for arg in node.args],
            "keywords": {kw.arg: self._get_value(kw.value) for kw in node.keywords if kw.arg},
            "lineno": node.lineno
        }
        return call_info
    
    def _analyze_import(self, node: Union[ast.Import, ast.ImportFrom]) -> List[Dict[str, Any]]:
        """Analyze an import statement."""
        imports = []
        
        if isinstance(node, ast.Import):
            for name in node.names:
                imports.append({
                    "type": "import",
                    "name": name.name,
                    "asname": name.asname,
                    "lineno": node.lineno
                })
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for name in node.names:
                imports.append({
                    "type": "importfrom",
                    "module": module,
                    "name": name.name,
                    "asname": name.asname,
                    "lineno": node.lineno
                })
                
        return imports
    
    def _analyze_try_except(self, node: ast.Try) -> Dict[str, Any]:
        """Analyze a try-except block."""
        try_except_info = {
            "lineno": node.lineno,
            "has_import": False,
            "handlers": []
        }
        
        # Check if the try block contains imports
        for child in ast.walk(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                try_except_info["has_import"] = True
                break
                
        # Analyze exception handlers
        for handler in node.handlers:
            handler_info = {
                "type": self._get_name(handler.type) if handler.type else None,
                "name": handler.name,
                "lineno": handler.lineno
            }
            try_except_info["handlers"].append(handler_info)
            
        return try_except_info
    
    def _get_name(self, node: Optional[ast.AST]) -> str:
        """Get a string representation of a name."""
        if node is None:
            return ""
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            return f"{self._get_name(node.value)}[{self._get_name(node.slice)}]"
        elif isinstance(node, ast.Call):
            return self._get_name(node.func)
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.Constant):
            return str(node.value)
        else:
            return str(node)
    
    def _get_value(self, node: Optional[ast.AST]) -> Any:
        """Get a Python value from an AST node."""
        if node is None:
            return None
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.NameConstant):
            return node.value
        elif isinstance(node, ast.List):
            return [self._get_value(elt) for elt in node.elts]
        elif isinstance(node, ast.Tuple):
            return tuple(self._get_value(elt) for elt in node.elts)
        elif isinstance(node, ast.Set):
            return set(self._get_value(elt) for elt in node.elts)
        elif isinstance(node, ast.Dict):
            return {self._get_value(k): self._get_value(v) for k, v in zip(node.keys, node.values)}
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            return f"{self._get_name(node.func)}(...)"
        else:
            return str(node)
    
    def generate_report(self, results: Dict[str, Any], summary_only: bool = False) -> Dict[str, Any]:
        """Generate a comprehensive report from analysis results."""
        report = {
            "stats": self.stats,
            "dependencies": list(sorted(self.dependencies)),
            "most_common_imports": sorted([(k, v) for k, v in self.import_info.items()], key=lambda x: x[1], reverse=True)[:20],
            "class_count": len(self.class_info),
            "function_count": len(self.function_info),
            "template_classes": len(self.template_info),
            "hardware_detection_functions": len(self.hardware_detection_info),
            "model_registry_functions": len(self.model_registry_info)
        }
        
        if not summary_only:
            report.update({
                "classes": self.class_info,
                "functions": self.function_info,
                "templates": self.template_info,
                "hardware_detection": self.hardware_detection_info,
                "model_registry": self.model_registry_info,
                "file_details": results
            })
            
        return report
    
    def generate_dependency_report(self) -> Dict[str, Any]:
        """Generate a report focused on dependencies."""
        return {
            "dependencies": list(sorted(self.dependencies)),
            "most_common_imports": sorted([(k, v) for k, v in self.import_info.items()], key=lambda x: x[1], reverse=True),
            "dependency_count": len(self.dependencies),
            "import_count": self.stats["imports"]
        }
    
    def generate_template_report(self) -> Dict[str, Any]:
        """Generate a report focused on templates."""
        return {
            "template_classes": self.template_info,
            "template_count": len(self.template_info),
            "template_methods": sum(len(cls.get("methods", {})) for cls in self.template_info.values())
        }
    
    def generate_hardware_report(self) -> Dict[str, Any]:
        """Generate a report focused on hardware detection."""
        return {
            "hardware_detection": self.hardware_detection_info,
            "hardware_detection_count": len(self.hardware_detection_info),
            "hardware_related_functions": sum(1 for func in self.function_info.values() if func.get("has_hardware_detection", False))
        }
    
    def generate_model_report(self) -> Dict[str, Any]:
        """Generate a report focused on model registries and selection."""
        return {
            "model_registry": self.model_registry_info,
            "model_registry_count": len(self.model_registry_info),
            "model_selection_functions": sum(1 for func in self.function_info.values() if func.get("has_model_selection", False))
        }

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze Python files with a focus on generator components")
    parser.add_argument("--path", required=True, help="Path to analyze (file or directory)")
    parser.add_argument("--output", help="Output file for the analysis report (default: stdout)")
    parser.add_argument("--pattern", default="**/*generator*.py", help="Glob pattern for files to analyze")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--summary", action="store_true", help="Generate only summary information")
    parser.add_argument("--dependencies", action="store_true", help="Focus on dependency analysis")
    parser.add_argument("--templates", action="store_true", help="Focus on template analysis")
    parser.add_argument("--hardware", action="store_true", help="Focus on hardware detection analysis")
    parser.add_argument("--models", action="store_true", help="Focus on model registry analysis")
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = GeneratorASTAnalyzer(verbose=args.verbose)
    
    # Analyze the specified path
    if os.path.isfile(args.path):
        results = {args.path: analyzer.analyze_file(args.path)}
    else:
        results = analyzer.analyze_directory(args.path, args.pattern)
    
    # Generate appropriate report
    if args.dependencies:
        report = analyzer.generate_dependency_report()
    elif args.templates:
        report = analyzer.generate_template_report()
    elif args.hardware:
        report = analyzer.generate_hardware_report()
    elif args.models:
        report = analyzer.generate_model_report()
    else:
        report = analyzer.generate_report(results, summary_only=args.summary)
    
    # Output report
    report_json = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report_json)
        logger.info(f"Report saved to {args.output}")
    else:
        print(report_json)

if __name__ == "__main__":
    main()