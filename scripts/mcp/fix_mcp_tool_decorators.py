#!/usr/bin/env python3
"""
Fix Tool Registration in Unified MCP Server

This script verifies and fixes the tool registration in the unified_mcp_server.py
by ensuring all tool functions have proper @register_tool decorators.
"""

import os
import sys
import re
import logging
from typing import List, Dict, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fix_tools")

def find_tool_functions(file_path: str) -> List[Dict]:
    """
    Find all functions that appear to be tool functions in the file.
    
    Returns a list of dictionaries with information about each function:
    {
        'name': function_name,
        'line_number': line_number,
        'has_decorator': boolean,
        'description': function_docstring
    }
    """
    functions = []
    
    # Regular expressions for finding functions and decorators
    func_pattern = re.compile(r'^def\s+(\w+)\s*\((.*?)\):(?:\s*"""(.*?)""")?', re.DOTALL | re.MULTILINE)
    decorator_pattern = re.compile(r'^@register_tool\s*\("?([\w_]+)"?\)', re.MULTILINE)
    docstring_pattern = re.compile(r'"""(.*?)"""', re.DOTALL)
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find all functions
    for match in func_pattern.finditer(content):
        func_name = match.group(1)
        func_start = match.start()
        
        # Get line number
        line_number = content[:func_start].count('\n') + 1
        
        # Check for decorator before function
        decorator_found = False
        func_line = content[:func_start].rfind('\n')
        if func_line >= 0:
            decorator_line = content[func_line:func_start]
            decorator_match = decorator_pattern.search(decorator_line)
            if decorator_match:
                decorator_found = True
        
        # Extract docstring
        docstring = ""
        docstring_match = docstring_pattern.search(match.group(0))
        if docstring_match:
            docstring = docstring_match.group(1).strip()
        
        # Store function info if it looks like a tool function
        if (func_name.startswith(('ipfs_', 'get_', 'list_', 'create_', 'delete_', 'update_', 'health_')) or 
            ('model' in func_name and not func_name.startswith('_')) or
            ('hardware' in func_name and not func_name.startswith('_'))):
            functions.append({
                'name': func_name,
                'line_number': line_number,
                'has_decorator': decorator_found,
                'description': docstring
            })
    
    return functions

def find_decorated_functions(file_path: str) -> Set[str]:
    """Find all functions that are already decorated with @register_tool."""
    decorated_funcs = set()
    
    decorator_pattern = re.compile(r'^@register_tool\s*\("?([\w_]+)"?\)', re.MULTILINE)
    func_pattern = re.compile(r'^def\s+(\w+)\s*\(', re.MULTILINE)
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find all decorator-function pairs
    lines = content.split('\n')
    for i in range(len(lines) - 1):
        decorator_match = decorator_pattern.match(lines[i])
        if decorator_match:
            func_match = func_pattern.match(lines[i + 1])
            if func_match:
                decorated_funcs.add(func_match.group(1))
    
    return decorated_funcs

def fix_tool_registration(file_path: str) -> bool:
    """Fix tool registration in the file."""
    logger.info(f"Analyzing {file_path}...")
    
    # Back up the file
    backup_path = f"{file_path}.bak"
    with open(file_path, 'r') as src, open(backup_path, 'w') as dst:
        dst.write(src.read())
    logger.info(f"Created backup at {backup_path}")
    
    # Find functions that should be tools
    functions = find_tool_functions(file_path)
    logger.info(f"Found {len(functions)} potential tool functions")
    
    # Find functions that are already decorated
    decorated_funcs = find_decorated_functions(file_path)
    logger.info(f"Found {len(decorated_funcs)} already decorated functions")
    
    # Identify functions that need decoration
    needs_decoration = []
    for func in functions:
        if not func['has_decorator'] and func['name'] not in decorated_funcs:
            needs_decoration.append(func)
    
    if not needs_decoration:
        logger.info("All tool functions are properly decorated")
        return True
    
    logger.info(f"Need to add decorators to {len(needs_decoration)} functions")
    
    # Read file content
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Add decorators where needed
    added_count = 0
    for func in needs_decoration:
        line_number = func['line_number'] - 1  # Adjust for 0-based indexing
        tool_name = func['name']
        decorator = f'@register_tool("{tool_name}")\n'
        
        # Make sure we're adding to the right function
        if line_number < len(lines) and f"def {tool_name}" in lines[line_number]:
            lines.insert(line_number, decorator)
            added_count += 1
            logger.info(f"Added decorator to {tool_name} at line {func['line_number']}")
        else:
            logger.warning(f"Could not add decorator to {tool_name} - line mismatch")
    
    if added_count > 0:
        # Write updated file
        with open(file_path, 'w') as f:
            f.writelines(lines)
        logger.info(f"Added {added_count} decorators to {file_path}")
        return True
    else:
        logger.warning("No decorators were added")
        return False

def verify_tool_registration(file_path: str) -> bool:
    """Verify that all tool functions have decorators."""
    functions = find_tool_functions(file_path)
    missing_decorators = [func for func in functions if not func['has_decorator']]
    
    if missing_decorators:
        logger.warning(f"Still missing decorators for: {', '.join(func['name'] for func in missing_decorators)}")
        return False
    else:
        logger.info("All tool functions have decorators")
        return True

def main():
    """Main function."""
    unified_server = "unified_mcp_server.py"
    
    if not os.path.exists(unified_server):
        logger.error(f"{unified_server} not found")
        return 1
    
    logger.info(f"Fixing tool registration in {unified_server}...")
    if fix_tool_registration(unified_server):
        logger.info(f"Successfully fixed tool registration in {unified_server}")
        
        # Verify the fix
        if verify_tool_registration(unified_server):
            logger.info("Verification successful - all tools should be properly registered")
        else:
            logger.warning("Verification failed - some tools may still be missing decorators")
        
        return 0
    else:
        logger.error(f"Failed to fix tool registration in {unified_server}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
