#!/usr/bin/env python3
"""
Syntax Validator

This module provides syntax validation for generated Python code.
"""

import ast
import logging
import tokenize
import io
from typing import Dict, Any, Optional, List, Tuple, Union

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SyntaxValidator:
    """
    Validator for checking and analyzing Python code syntax.
    
    This class provides methods to validate that generated code has valid Python syntax.
    """
    
    def __init__(self, config=None):
        """
        Initialize the syntax validator.
        
        Args:
            config: Configuration object or dict
        """
        self.config = config or {}
    
    def validate(self, content: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Validate the syntax of Python code.
        
        Args:
            content: Python code to validate
            
        Returns:
            Tuple of (is_valid, error_info)
        """
        try:
            # Compile the code to check syntax
            ast.parse(content)
            return True, None
        except SyntaxError as e:
            # Get error information
            error_info = {
                "type": "SyntaxError",
                "message": str(e),
                "lineno": e.lineno,
                "offset": e.offset,
                "text": e.text.rstrip('\r\n') if e.text else None,
                "filename": e.filename
            }
            return False, error_info
        except Exception as e:
            # Generic error
            error_info = {
                "type": type(e).__name__,
                "message": str(e)
            }
            return False, error_info
    
    def validate_tokens(self, content: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Validate Python code tokens.
        
        Args:
            content: Python code to validate
            
        Returns:
            Tuple of (is_valid, error_info)
        """
        try:
            # Tokenize the code
            tokens = list(tokenize.tokenize(io.BytesIO(content.encode('utf-8')).readline))
            return True, None
        except tokenize.TokenError as e:
            # Get error information
            message, (lineno, offset) = e.args
            error_info = {
                "type": "TokenError",
                "message": message,
                "lineno": lineno,
                "offset": offset
            }
            return False, error_info
        except Exception as e:
            # Generic error
            error_info = {
                "type": type(e).__name__,
                "message": str(e)
            }
            return False, error_info
    
    def analyze(self, content: str) -> Dict[str, Any]:
        """
        Analyze Python code for various features.
        
        Args:
            content: Python code to analyze
            
        Returns:
            Dictionary with analysis results
        """
        results = {
            "classes": [],
            "functions": [],
            "imports": [],
            "try_except_blocks": [],
            "line_count": len(content.split('\n')),
            "char_count": len(content)
        }
        
        try:
            # Parse the AST
            tree = ast.parse(content)
            
            # Walk the AST to find classes, functions, imports, etc.
            for node in ast.walk(tree):
                # Classes
                if isinstance(node, ast.ClassDef):
                    # Get method names
                    methods = []
                    for child in node.body:
                        if isinstance(child, ast.FunctionDef):
                            methods.append(child.name)
                            
                    # Add class info
                    results["classes"].append({
                        "name": node.name,
                        "lineno": node.lineno,
                        "methods": methods,
                        "method_count": len(methods)
                    })
                
                # Functions
                elif isinstance(node, ast.FunctionDef) and not isinstance(node.parent, ast.ClassDef):
                    results["functions"].append({
                        "name": node.name,
                        "lineno": node.lineno,
                        "args": [arg.arg for arg in node.args.args]
                    })
                
                # Imports
                elif isinstance(node, ast.Import):
                    for name in node.names:
                        results["imports"].append({
                            "module": name.name,
                            "alias": name.asname,
                            "lineno": node.lineno
                        })
                elif isinstance(node, ast.ImportFrom):
                    for name in node.names:
                        results["imports"].append({
                            "module": node.module,
                            "name": name.name,
                            "alias": name.asname,
                            "lineno": node.lineno
                        })
                
                # Try-except blocks
                elif isinstance(node, ast.Try):
                    # Get handler exception types
                    handlers = []
                    for handler in node.handlers:
                        if handler.type:
                            if isinstance(handler.type, ast.Name):
                                handlers.append(handler.type.id)
                            elif isinstance(handler.type, ast.Attribute):
                                handlers.append(f"{handler.type.value.id}.{handler.type.attr}")
                            else:
                                handlers.append("unknown")
                        else:
                            handlers.append("all")
                            
                    results["try_except_blocks"].append({
                        "lineno": node.lineno,
                        "handlers": handlers
                    })
                    
            # Update counts
            results["class_count"] = len(results["classes"])
            results["function_count"] = len(results["functions"])
            results["import_count"] = len(results["imports"])
            results["try_except_count"] = len(results["try_except_blocks"])
            
            return results
        except Exception as e:
            logger.warning(f"Error analyzing code: {str(e)}")
            
            # Return basic results
            return {
                "error": str(e),
                "line_count": len(content.split('\n')),
                "char_count": len(content),
                "class_count": 0,
                "function_count": 0,
                "import_count": 0,
                "try_except_count": 0
            }
    
    def detect_issues(self, content: str) -> List[Dict[str, Any]]:
        """
        Detect common issues in Python code.
        
        Args:
            content: Python code to analyze
            
        Returns:
            List of detected issues
        """
        issues = []
        
        # Check for syntax errors
        is_valid, error_info = self.validate(content)
        if not is_valid:
            issues.append({
                "type": "syntax_error",
                "message": error_info["message"],
                "lineno": error_info.get("lineno"),
                "severity": "error"
            })
            
            # Return issues early if there's a syntax error
            return issues
            
        # Try to detect common issues by analyzing the AST
        try:
            # Parse the AST
            tree = ast.parse(content)
            
            # Check for unused imports
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.add(name.asname or name.name)
                elif isinstance(node, ast.ImportFrom):
                    for name in node.names:
                        imports.add(name.asname or name.name)
                        
            # Check for unused imports
            for imp in imports:
                found = False
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name) and node.id == imp:
                        found = True
                        break
                        
                if not found:
                    issues.append({
                        "type": "unused_import",
                        "message": f"Import '{imp}' is not used",
                        "severity": "warning"
                    })
                    
            # Check for undefined variables
            defined_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    defined_names.add(node.name)
                    for arg in node.args.args:
                        defined_names.add(arg.arg)
                elif isinstance(node, ast.ClassDef):
                    defined_names.add(node.name)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    defined_names.add(node.id)
                    
            # Add imported names
            defined_names.update(imports)
            
            # Add built-in names
            defined_names.update(dir(__builtins__))
            
            # Check for undefined variables
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    if node.id not in defined_names:
                        issues.append({
                            "type": "undefined_variable",
                            "message": f"Variable '{node.id}' is not defined",
                            "lineno": node.lineno,
                            "severity": "error"
                        })
                        
            # Check for duplicate function/class names
            names = {}
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if node.name in names:
                        issues.append({
                            "type": "duplicate_definition",
                            "message": f"Duplicate definition of '{node.name}'",
                            "lineno": node.lineno,
                            "severity": "error"
                        })
                    else:
                        names[node.name] = node.lineno
                        
        except Exception as e:
            logger.warning(f"Error detecting issues: {str(e)}")
            
        return issues