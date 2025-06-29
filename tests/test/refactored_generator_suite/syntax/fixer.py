#!/usr/bin/env python3
"""
Syntax Fixer

This module provides syntax fixing for generated Python code.
"""

import re
import logging
import tokenize
import io
import ast
from typing import Dict, Any, Optional, List, Tuple, Union

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SyntaxFixer:
    """
    Fixer for common syntax issues in Python code.
    
    This class provides methods to fix common syntax issues in generated code.
    """
    
    def __init__(self, config=None):
        """
        Initialize the syntax fixer.
        
        Args:
            config: Configuration object or dict
        """
        self.config = config or {}
        
        # Load configuration
        self.indent_size = self.config.get("indent_size", 4)
        self.max_line_length = self.config.get("max_line_length", 100)
        self.use_ast_formatting = self.config.get("use_ast_formatting", False)
    
    def fix(self, content: str) -> str:
        """
        Fix common syntax issues in Python code.
        
        Args:
            content: Python code to fix
            
        Returns:
            Fixed Python code
        """
        # Apply all fixers in sequence
        fixed_content = content
        
        try:
            # First, try to parse the AST to see if we even need to fix anything
            ast.parse(fixed_content)
            
            # If we get here, the code is syntactically valid, but we might still want to fix formatting
            
            # Fix formatting issues
            fixed_content = self.fix_quotes(fixed_content)
            fixed_content = self.fix_indentation(fixed_content)
            
            logger.info("Fixed formatting issues")
            
        except SyntaxError as e:
            logger.info(f"Fixing syntax error at line {e.lineno}: {e.msg}")
            
            # Apply syntax fixers
            fixed_content = self.fix_quotes(fixed_content)
            fixed_content = self.fix_unterminated_strings(fixed_content)
            fixed_content = self.fix_unterminated_parentheses(fixed_content)
            fixed_content = self.fix_missing_colons(fixed_content)
            fixed_content = self.fix_dangling_commas(fixed_content)
            fixed_content = self.fix_indentation(fixed_content)
            
            # Check if fixed
            try:
                ast.parse(fixed_content)
                logger.info("Successfully fixed syntax errors")
            except SyntaxError as e:
                logger.warning(f"Could not fully fix syntax errors: {e}")
                
        return fixed_content
    
    def fix_quotes(self, content: str) -> str:
        """
        Fix quote-related issues in Python code.
        
        Args:
            content: Python code to fix
            
        Returns:
            Fixed Python code
        """
        # Replace multiple consecutive quotes
        fixed_content = content.replace('""""', '"""')
        fixed_content = fixed_content.replace("''''", "'''")
        
        return fixed_content
    
    def fix_unterminated_strings(self, content: str) -> str:
        """
        Fix unterminated string literals in Python code.
        
        Args:
            content: Python code to fix
            
        Returns:
            Fixed Python code
        """
        lines = content.split('\n')
        fixed_lines = lines.copy()
        
        # Track string state
        in_single_quotes = False
        in_double_quotes = False
        in_triple_single_quotes = False
        in_triple_double_quotes = False
        start_line = -1
        start_indent = 0
        
        for i, line in enumerate(lines):
            # Process each character in the line
            j = 0
            while j < len(line):
                char = line[j]
                
                # Check for triple quotes
                if j + 2 < len(line) and line[j:j+3] == '"""':
                    if not in_single_quotes and not in_double_quotes and not in_triple_single_quotes:
                        in_triple_double_quotes = not in_triple_double_quotes
                        if in_triple_double_quotes:
                            start_line = i
                            start_indent = len(line) - len(line.lstrip())
                    j += 3
                    continue
                    
                elif j + 2 < len(line) and line[j:j+3] == "'''":
                    if not in_single_quotes and not in_double_quotes and not in_triple_double_quotes:
                        in_triple_single_quotes = not in_triple_single_quotes
                        if in_triple_single_quotes:
                            start_line = i
                            start_indent = len(line) - len(line.lstrip())
                    j += 3
                    continue
                    
                # Check for regular quotes
                elif char == '"' and not in_single_quotes and not in_triple_single_quotes and not in_triple_double_quotes:
                    if j > 0 and line[j-1] == '\\':
                        # Escaped quote
                        pass
                    else:
                        in_double_quotes = not in_double_quotes
                        if in_double_quotes:
                            start_line = i
                            
                elif char == "'" and not in_double_quotes and not in_triple_single_quotes and not in_triple_double_quotes:
                    if j > 0 and line[j-1] == '\\':
                        # Escaped quote
                        pass
                    else:
                        in_single_quotes = not in_single_quotes
                        if in_single_quotes:
                            start_line = i
                            
                j += 1
                
        # Fix unterminated strings
        if in_triple_double_quotes:
            # Add closing triple double quotes
            indent_str = ' ' * start_indent
            if fixed_lines[-1].strip():
                fixed_lines.append(f"{indent_str}\"\"\"")
            else:
                fixed_lines[-1] = f"{indent_str}\"\"\""
            logger.info(f"Added missing closing triple double quotes with matching indentation from line {start_line+1}")
            
        elif in_triple_single_quotes:
            # Add closing triple single quotes
            indent_str = ' ' * start_indent
            if fixed_lines[-1].strip():
                fixed_lines.append(f"{indent_str}'''")
            else:
                fixed_lines[-1] = f"{indent_str}'''"
            logger.info(f"Added missing closing triple single quotes with matching indentation from line {start_line+1}")
            
        elif in_double_quotes:
            # Add closing double quote
            fixed_lines[start_line] += '"'
            logger.info(f"Added missing closing double quote to line {start_line+1}")
            
        elif in_single_quotes:
            # Add closing single quote
            fixed_lines[start_line] += "'"
            logger.info(f"Added missing closing single quote to line {start_line+1}")
            
        return '\n'.join(fixed_lines)
    
    def fix_unterminated_parentheses(self, content: str) -> str:
        """
        Fix unterminated parentheses, brackets, and braces in Python code.
        
        Args:
            content: Python code to fix
            
        Returns:
            Fixed Python code
        """
        lines = content.split('\n')
        fixed_lines = lines.copy()
        
        # Process line by line
        for i, line in enumerate(fixed_lines):
            # Track opening and closing delimiters in this line
            stack = []
            
            for j, char in enumerate(line):
                if char in '([{':
                    stack.append(char)
                elif char in ')]}':
                    if stack and ((stack[-1] == '(' and char == ')') or
                                  (stack[-1] == '[' and char == ']') or
                                  (stack[-1] == '{' and char == '}')):
                        stack.pop()
                    else:
                        # Mismatched closing delimiter
                        fixed_lines[i] = line[:j] + line[j+1:]
                        logger.info(f"Removed mismatched closing delimiter at line {i+1}, position {j+1}")
                        line = fixed_lines[i]  # Update line for subsequent checks
                        
            # If we have unclosed delimiters at the end of the line,
            # add matching closing delimiters if the line doesn't end with a backslash, comma, or colon
            if stack and not line.rstrip().endswith(('\\', ',', ':')):
                for opening in reversed(stack):
                    closing = {'(': ')', '[': ']', '{': '}'}[opening]
                    fixed_lines[i] += closing
                    logger.info(f"Added missing {closing} to line {i+1}")
                    
        # Check for unbalanced delimiters across lines
        full_content = '\n'.join(fixed_lines)
        stack = []
        
        for i, char in enumerate(full_content):
            if char in '([{':
                stack.append(char)
            elif char in ')]}':
                if stack and ((stack[-1] == '(' and char == ')') or
                              (stack[-1] == '[' and char == ']') or
                              (stack[-1] == '{' and char == '}')):
                    stack.pop()
                    
        # If we have unclosed delimiters at the end of the file, add matching closing delimiters
        if stack:
            for opening in reversed(stack):
                closing = {'(': ')', '[': ']', '{': '}'}[opening]
                fixed_lines[-1] += closing
                logger.info(f"Added missing {closing} at the end of the file")
                
        return '\n'.join(fixed_lines)
    
    def fix_missing_colons(self, content: str) -> str:
        """
        Fix missing colons in Python code.
        
        Args:
            content: Python code to fix
            
        Returns:
            Fixed Python code
        """
        lines = content.split('\n')
        fixed_lines = lines.copy()
        
        # Look for function and class definitions, if blocks, etc. missing colons
        for i, line in enumerate(fixed_lines):
            # Skip lines that already end with colon
            if line.rstrip().endswith(':'):
                continue
                
            # Skip commented lines
            if line.lstrip().startswith('#'):
                continue
                
            # Check for patterns that should end with a colon
            if (re.match(r'^\s*(?:def|class|if|elif|else|for|while|try|except|finally|with|match|case)\b.*\)?\s*$', line) or
                re.match(r'^\s*(?:else|finally|try)\s*$', line)):
                fixed_lines[i] = line + ':'
                logger.info(f"Added missing colon to line {i+1}")
                
        return '\n'.join(fixed_lines)
    
    def fix_dangling_commas(self, content: str) -> str:
        """
        Fix dangling commas in Python code.
        
        Args:
            content: Python code to fix
            
        Returns:
            Fixed Python code
        """
        lines = content.split('\n')
        fixed_lines = lines.copy()
        
        # Look for dangling commas before closing delimiters
        for i, line in enumerate(fixed_lines):
            # Find commas followed by a closing delimiter
            for match in re.finditer(r',([ \t]*)([\)\]}])', line):
                fixed_lines[i] = line[:match.start()] + match.group(1) + match.group(2) + line[match.end():]
                logger.info(f"Removed dangling comma at line {i+1}")
                line = fixed_lines[i]  # Update line for subsequent checks
                
        return '\n'.join(fixed_lines)
    
    def fix_indentation(self, content: str) -> str:
        """
        Fix indentation issues in Python code.
        
        Args:
            content: Python code to fix
            
        Returns:
            Fixed Python code
        """
        # If use_ast_formatting is enabled and code is syntactically valid, try to use AST-based formatting
        if self.use_ast_formatting:
            try:
                fixed_content = self._fix_indentation_with_ast(content)
                logger.info("Fixed indentation using AST")
                return fixed_content
            except Exception as e:
                logger.warning(f"AST-based indentation fixing failed: {e}")
                
        # Fall back to regex-based indentation fixing
        return self._fix_indentation_with_regex(content)
    
    def _fix_indentation_with_regex(self, content: str) -> str:
        """
        Fix indentation issues in Python code using regex.
        
        Args:
            content: Python code to fix
            
        Returns:
            Fixed Python code
        """
        lines = content.split('\n')
        fixed_lines = []
        
        # Track indentation level
        indent_level = 0
        
        # Process line by line
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                fixed_lines.append(line)
                continue
                
            # Check for outdent
            if (stripped.startswith(('else:', 'elif ', 'except:', 'except ', 'finally:', 'case ', 'case:'))
                    and indent_level > 0):
                indent_level -= 1
                
            # Calculate proper indentation
            proper_indent = ' ' * (self.indent_size * indent_level)
            
            # Fix indentation
            fixed_line = proper_indent + stripped
            fixed_lines.append(fixed_line)
            
            # Check for indent
            if stripped.endswith(':') or (stripped.endswith('\\') and indent_level >= 1):
                indent_level += 1
                
            # Check for dedent (empty suite)
            if i + 1 < len(lines) and lines[i + 1].strip() and not lines[i + 1].strip().startswith(('#', 'else:', 'elif ', 'except:', 'except ', 'finally:')):
                next_indent = len(lines[i + 1]) - len(lines[i + 1].lstrip())
                if next_indent <= self.indent_size * (indent_level - 1) and indent_level > 0:
                    indent_level -= 1
                    
        return '\n'.join(fixed_lines)
    
    def _fix_indentation_with_ast(self, content: str) -> str:
        """
        Fix indentation issues in Python code using AST.
        
        Args:
            content: Python code to fix
            
        Returns:
            Fixed Python code
        """
        # Try to use third-party tools if available
        try:
            import black
            mode = black.Mode()
            fixed_content = black.format_str(content, mode=mode)
            return fixed_content
        except ImportError:
            pass
            
        try:
            import autopep8
            fixed_content = autopep8.fix_code(content)
            return fixed_content
        except ImportError:
            pass
            
        # If no formatting tool is available, fall back to custom AST-based formatting
        tree = ast.parse(content)
        
        # Use custom AST formatter
        return self._format_ast(tree, content)
    
    def _format_ast(self, tree: ast.AST, original_content: str) -> str:
        """
        Format AST tree.
        
        Args:
            tree: AST tree
            original_content: Original Python code
            
        Returns:
            Formatted Python code
        """
        # For now, just return the original content, since custom AST formatting is complex
        # In a real implementation, this would generate properly formatted code from the AST
        return original_content