#!/usr/bin/env python3
# fix_typescript_syntax.py
# Script to fix common Python-to-TypeScript syntax issues

import os
import sys
import re
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
        logging.FileHandler('fix_typescript_syntax.log')
    ]
)
logger = logging.getLogger(__name__)

class Config:
    TARGET_DIR = None
    FIX_FILES = True
    VERBOSE = False
    STATS = {
        "files_checked": 0,
        "files_fixed": 0,
        "total_fixes": 0,
        "fix_by_type": {
            "python_import": 0,
            "destructuring": 0,
            "dict_syntax": 0,
            "self_to_this": 0,
            "python_class": 0,
            "docstring": 0,
            "exception": 0,
            "string_literal": 0,
            "type_annotation": 0,
            "function_definition": 0,
            "init_file": 0
        }
    }

def setup_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Fix Python-to-TypeScript syntax issues")
    parser.add_argument("--target-dir", help="Target directory to check", default="../ipfs_accelerate_js")
    parser.add_argument("--no-fix", action="store_true", help="Dry run without applying fixes")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--file", help="Process a specific file")
    args = parser.parse_args()
    
    Config.TARGET_DIR = os.path.abspath(args.target_dir)
    Config.FIX_FILES = not args.no_fix
    Config.VERBOSE = args.verbose
    
    if not os.path.isdir(Config.TARGET_DIR):
        logger.error(f"Target directory does not exist: {Config.TARGET_DIR}")
        sys.exit(1)
    
    logger.info(f"Fixing TypeScript syntax in: {Config.TARGET_DIR}")
    logger.info(f"Apply fixes: {Config.FIX_FILES}")

    if args.file:
        return [os.path.join(Config.TARGET_DIR, args.file)]
    return None

def find_typescript_files() -> List[str]:
    """Find all TypeScript files in the target directory"""
    ts_files = []
    
    for root, _, files in os.walk(Config.TARGET_DIR):
        for file in files:
            if file.endswith((".ts", ".tsx")) and not file.endswith(".d.ts"):
                ts_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(ts_files)} TypeScript files to check")
    return ts_files

def fix_init_files(file_path: str) -> Tuple[str, int]:
    """Fix __init__.ts files"""
    if not os.path.basename(file_path) in ["__init__.ts", "__pycache__.ts"]:
        return file_path, 0
    
    # Create a proper index.ts file instead
    dir_name = os.path.dirname(file_path)
    index_path = os.path.join(dir_name, "index.ts")
    
    if Config.FIX_FILES:
        # Get all .ts files in the directory
        ts_files = []
        for file in os.listdir(dir_name):
            if file.endswith('.ts') and file not in ["__init__.ts", "__pycache__.ts", "index.ts"] and not file.endswith('.d.ts'):
                ts_files.append(os.path.splitext(file)[0])
        
        # Create a proper index file
        content = "// Auto-generated index file\n\n"
        for file in ts_files:
            content += f'export * from "./{file}";\n'
        
        try:
            with open(index_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Remove the old __init__.ts file
            os.remove(file_path)
            
            logger.info(f"Replaced {os.path.basename(file_path)} with index.ts in {dir_name}")
            Config.STATS["fix_by_type"]["init_file"] += 1
            return index_path, 1
        except Exception as e:
            logger.error(f"Error replacing {file_path} with index.ts: {e}")
            return file_path, 0
    
    return file_path, 0

def fix_python_imports(content: str) -> Tuple[str, int]:
    """Fix Python import statements"""
    fixes = 0
    
    # Pattern 1: from X import Y, Z
    pattern = r'from\s+(["\']?)([^"\']+)\1\s+import\s+(.+)'
    
    def import_replacement(match):
        nonlocal fixes
        quote = match.group(1) or '"'
        module = match.group(2)
        imports = match.group(3).strip()
        
        # Handle multiple imports
        if ',' in imports:
            # Split by comma, respect curly braces
            import_items = []
            current = ""
            brace_level = 0
            
            for char in imports:
                if char == '{':
                    brace_level += 1
                elif char == '}':
                    brace_level -= 1
                
                if char == ',' and brace_level == 0:
                    import_items.append(current.strip())
                    current = ""
                else:
                    current += char
            
            if current:
                import_items.append(current.strip())
            
            # Create TypeScript import
            imports_list = ", ".join(import_items)
            fixes += 1
            return f'import {{ {imports_list} }} from {quote}{module}{quote};'
        else:
            # Single import
            fixes += 1
            return f'import {{ {imports} }} from {quote}{module}{quote};'
    
    content = re.sub(pattern, import_replacement, content)
    
    # Pattern 2: import X as Y
    pattern = r'import\s+([^\s]+)\s+as\s+([^\s]+)'
    
    def import_as_replacement(match):
        nonlocal fixes
        module = match.group(1)
        alias = match.group(2)
        fixes += 1
        return f'import * as {alias} from "{module}";'
    
    content = re.sub(pattern, import_as_replacement, content)
    
    # Pattern 3: import X
    pattern = r'import\s+([^\s]+)\s*$'
    
    def simple_import_replacement(match):
        nonlocal fixes
        module = match.group(1)
        fixes += 1
        return f'import "{module}";'
    
    content = re.sub(pattern, simple_import_replacement, content)
    
    Config.STATS["fix_by_type"]["python_import"] += fixes
    return content, fixes

def fix_destructuring(content: str) -> Tuple[str, int]:
    """Fix Python tuple/list destructuring to TypeScript destructuring"""
    fixes = 0
    
    # Pattern 1: const [a, b] = something
    pattern = r'const\s*\[\s*([^=\]]+)\]\s*=\s*([^;]+)'
    
    def destructuring_replacement(match):
        nonlocal fixes
        variables = match.group(1).strip()
        value = match.group(2).strip()
        
        # Check for nested destructuring which TypeScript doesn't handle the same way
        if '[' in variables or '{' in variables:
            fixes += 1
            # Create a temporary variable and then individual assignments
            variable_list = re.split(r',\s*', variables)
            result = f'const _tmp = {value};\n'
            
            for i, var in enumerate(variable_list):
                var = var.strip()
                if var:  # Skip empty entries from trailing commas
                    result += f'const {var} = _tmp[{i}];\n'
            
            return result
        else:
            # Simple array destructuring
            return f'const [{variables}] = {value};'
    
    content = re.sub(pattern, destructuring_replacement, content)
    
    # Pattern 2: let [a, b] = something
    pattern = r'let\s*\[\s*([^=\]]+)\]\s*=\s*([^;]+)'
    
    def let_destructuring_replacement(match):
        nonlocal fixes
        variables = match.group(1).strip()
        value = match.group(2).strip()
        
        # Check for nested destructuring
        if '[' in variables or '{' in variables:
            fixes += 1
            # Create a temporary variable and then individual assignments
            variable_list = re.split(r',\s*', variables)
            result = f'const _tmp = {value};\n'
            
            for i, var in enumerate(variable_list):
                var = var.strip()
                if var:  # Skip empty entries from trailing commas
                    result += f'let {var} = _tmp[{i}];\n'
            
            return result
        else:
            # Simple array destructuring
            return f'let [{variables}] = {value};'
    
    content = re.sub(pattern, let_destructuring_replacement, content)
    
    # Pattern 3: var [a, b] = something
    pattern = r'var\s*\[\s*([^=\]]+)\]\s*=\s*([^;]+)'
    
    def var_destructuring_replacement(match):
        nonlocal fixes
        variables = match.group(1).strip()
        value = match.group(2).strip()
        
        # Check for nested destructuring
        if '[' in variables or '{' in variables:
            fixes += 1
            # Create a temporary variable and then individual assignments
            variable_list = re.split(r',\s*', variables)
            result = f'const _tmp = {value};\n'
            
            for i, var in enumerate(variable_list):
                var = var.strip()
                if var:  # Skip empty entries from trailing commas
                    result += f'var {var} = _tmp[{i}];\n'
            
            return result
        else:
            # Simple array destructuring
            return f'var [{variables}] = {value};'
    
    content = re.sub(pattern, var_destructuring_replacement, content)
    
    Config.STATS["fix_by_type"]["destructuring"] += fixes
    return content, fixes

def fix_dict_syntax(content: str) -> Tuple[str, int]:
    """Fix Python dictionary syntax to JavaScript object syntax"""
    fixes = 0
    
    # Pattern 1: Fix dict creation
    pattern = r'(\w+)\s*=\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    
    def dict_replacement(match):
        nonlocal fixes
        variable = match.group(1)
        dict_content = match.group(2)
        
        # Don't modify if it's already a valid JS object
        if re.search(r'\w+\s*:', dict_content):
            return f'{variable} = {{{dict_content}}}'
        
        # Convert Python dict syntax to JS
        if dict_content.strip():
            new_content = dict_content
            # Replace : with =
            new_content = re.sub(r'(\w+)\s*:', r'\1:', new_content)
            fixes += 1
            return f'{variable} = {{{new_content}}}'
        else:
            return f'{variable} = {{}}'
    
    content = re.sub(pattern, dict_replacement, content)
    
    # Pattern 2: Fix None, True, False
    content = re.sub(r'\bNone\b', 'null', content)
    content = re.sub(r'\bTrue\b', 'true', content)
    content = re.sub(r'\bFalse\b', 'false', content)
    
    # Count those fixes
    fixes += len(re.findall(r'\bNull\b|\bTrue\b|\bFalse\b', content))
    
    Config.STATS["fix_by_type"]["dict_syntax"] += fixes
    return content, fixes

def fix_self_to_this(content: str) -> Tuple[str, int]:
    """Replace Python self with JavaScript this"""
    fixes = 0
    
    # Pattern 1: self. -> this.
    self_count = content.count('self.')
    content = content.replace('self.', 'this.')
    fixes += self_count
    
    # Pattern 2: (self) or (self, ...)
    pattern = r'(\w+)\s*\(\s*self\s*(?:,\s*|\))'
    
    def self_param_replacement(match):
        nonlocal fixes
        func_name = match.group(1)
        fixes += 1
        return f'{func_name}(this,'
    
    content = re.sub(pattern, self_param_replacement, content)
    
    Config.STATS["fix_by_type"]["self_to_this"] += fixes
    return content, fixes

def fix_python_class(content: str) -> Tuple[str, int]:
    """Fix Python class definitions"""
    fixes = 0
    
    # Pattern 1: class X: -> class X {
    pattern = r'class\s+(\w+)([^{:]*?):'
    
    def class_replacement(match):
        nonlocal fixes
        class_name = match.group(1)
        inheritance = match.group(2).strip()
        fixes += 1
        
        if inheritance:
            # Handle Python-style inheritance
            inheritance = re.sub(r'\([^)]*\)', lambda m: m.group(0).replace('(', ' extends ').replace(')', ''), inheritance)
            return f'class {class_name}{inheritance} {{'
        else:
            return f'class {class_name} {{'
    
    content = re.sub(pattern, class_replacement, content)
    
    # Pattern 2: def __init__ -> constructor
    pattern = r'def\s+__init__\s*\(\s*self\s*(?:,\s*([^)]*))?\s*\)\s*(?:->.*?)?:'
    
    def init_replacement(match):
        nonlocal fixes
        params = match.group(1) or ''
        fixes += 1
        return f'constructor({params}) {{'
    
    content = re.sub(pattern, init_replacement, content)
    
    # Pattern 3: def method -> method
    pattern = r'def\s+(\w+)\s*\(\s*self\s*(?:,\s*([^)]*))?\s*\)\s*(?:->.*?)?:'
    
    def method_replacement(match):
        nonlocal fixes
        method_name = match.group(1)
        params = match.group(2) or ''
        fixes += 1
        return f'{method_name}({params}) {{'
    
    content = re.sub(pattern, method_replacement, content)
    
    Config.STATS["fix_by_type"]["python_class"] += fixes
    return content, fixes

def fix_python_docstring(content: str) -> Tuple[str, int]:
    """Convert Python docstrings to JSDoc comments"""
    fixes = 0
    
    # Pattern 1: Simple docstring
    pattern = r'"""([^"]*)"""'
    
    def docstring_replacement(match):
        nonlocal fixes
        doc_content = match.group(1).strip()
        lines = doc_content.split('\n')
        
        if len(lines) == 1:
            fixes += 1
            return f'/** {doc_content} */'
        else:
            result = ['/**']
            for line in lines:
                result.append(f' * {line.strip()}')
            result.append(' */')
            fixes += 1
            return '\n'.join(result)
    
    content = re.sub(pattern, docstring_replacement, content)
    
    # Pattern 2: Single quote docstring
    pattern = r"'''([^']*)'''"
    
    def single_quote_docstring_replacement(match):
        nonlocal fixes
        doc_content = match.group(1).strip()
        lines = doc_content.split('\n')
        
        if len(lines) == 1:
            fixes += 1
            return f'/** {doc_content} */'
        else:
            result = ['/**']
            for line in lines:
                result.append(f' * {line.strip()}')
            result.append(' */')
            fixes += 1
            return '\n'.join(result)
    
    content = re.sub(pattern, single_quote_docstring_replacement, content)
    
    Config.STATS["fix_by_type"]["docstring"] += fixes
    return content, fixes

def fix_exceptions(content: str) -> Tuple[str, int]:
    """Fix Python exception handling"""
    fixes = 0
    
    # Pattern 1: except X: -> catch (error) {
    pattern = r'(\s+)except\s+([^:]+):'
    
    def except_replacement(match):
        nonlocal fixes
        indent = match.group(1)
        exception_type = match.group(2).strip()
        fixes += 1
        
        if exception_type == 'Exception':
            return f'{indent}catch (error) {{'
        else:
            return f'{indent}catch (error) {{ // was: {exception_type}'
    
    content = re.sub(pattern, except_replacement, content)
    
    # Pattern 2: try: -> try {
    pattern = r'(\s+)try\s*:'
    
    def try_replacement(match):
        nonlocal fixes
        indent = match.group(1)
        fixes += 1
        return f'{indent}try {{'
    
    content = re.sub(pattern, try_replacement, content)
    
    # Pattern 3: finally: -> finally {
    pattern = r'(\s+)finally\s*:'
    
    def finally_replacement(match):
        nonlocal fixes
        indent = match.group(1)
        fixes += 1
        return f'{indent}finally {{'
    
    content = re.sub(pattern, finally_replacement, content)
    
    # Pattern 4: raise X -> throw new X
    pattern = r'(\s+)raise\s+(\w+)(\(.*\))?'
    
    def raise_replacement(match):
        nonlocal fixes
        indent = match.group(1)
        exception = match.group(2)
        args = match.group(3) or '()'
        fixes += 1
        return f'{indent}throw new {exception}{args}'
    
    content = re.sub(pattern, raise_replacement, content)
    
    Config.STATS["fix_by_type"]["exception"] += fixes
    return content, fixes

def fix_string_literals(content: str) -> Tuple[str, int]:
    """Fix Python f-strings and string interpolation"""
    fixes = 0
    
    # Pattern 1: f"..." -> `...`
    pattern = r'f(["\'])(.+?)\\1'
    
    def fstring_replacement(match):
        nonlocal fixes
        quote = match.group(1)
        string_content = match.group(2)
        
        # Replace {x} with ${x}
        def replace_vars(m):
            var_content = m.group(1)
            return f'${{{var_content}}}'
        
        string_content = re.sub(r'{([^{}]+)}', replace_vars, string_content)
        fixes += 1
        return f'`{string_content}`'
    
    content = re.sub(pattern, fstring_replacement, content)
    
    # Pattern 2: Fix unterminated string literals
    pattern = r'("[^"\n]*|\'[^\'\n]*)$'
    
    def fix_unterminated(match):
        nonlocal fixes
        string_part = match.group(1)
        fixes += 1
        return f'{string_part}{string_part[0]}'
    
    lines = content.split('\n')
    for i in range(len(lines)):
        lines[i] = re.sub(pattern, fix_unterminated, lines[i])
    
    content = '\n'.join(lines)
    
    Config.STATS["fix_by_type"]["string_literal"] += fixes
    return content, fixes

def fix_type_annotations(content: str) -> Tuple[str, int]:
    """Add TypeScript type annotations"""
    fixes = 0
    
    # Pattern 1: Function parameters without types
    pattern = r'function\s+(\w+)\((.*?)(?<!:)\s*(\w+)(?!\s*:)([,)])'
    
    def param_type_replacement(match):
        nonlocal fixes
        func_name = match.group(1)
        before_param = match.group(2)
        param_name = match.group(3)
        after_param = match.group(4)
        fixes += 1
        return f'function {func_name}({before_param}{param_name}: any{after_param}'
    
    content = re.sub(pattern, param_type_replacement, content)
    
    # Pattern 2: Function without return type
    pattern = r'function\s+(\w+)\(([^)]*)\)(?!\s*:)'
    
    def return_type_replacement(match):
        nonlocal fixes
        func_name = match.group(1)
        params = match.group(2)
        fixes += 1
        return f'function {func_name}({params}): any'
    
    content = re.sub(pattern, return_type_replacement, content)
    
    # Pattern 3: Class properties without types
    pattern = r'(\s+)(\w+)\s*=\s*([^:;]+);'
    
    def property_type_replacement(match):
        nonlocal fixes
        indent = match.group(1)
        prop_name = match.group(2)
        value = match.group(3)
        fixes += 1
        return f'{indent}{prop_name}: any = {value};'
    
    content = re.sub(pattern, property_type_replacement, content)
    
    # Pattern 4: Constructor parameters without types
    pattern = r'constructor\((.*?)(?<!:)\s*(\w+)(?!\s*:)([,)])'
    
    def constructor_param_replacement(match):
        nonlocal fixes
        before_param = match.group(1)
        param_name = match.group(2)
        after_param = match.group(3)
        fixes += 1
        return f'constructor({before_param}{param_name}: any{after_param}'
    
    content = re.sub(pattern, constructor_param_replacement, content)
    
    Config.STATS["fix_by_type"]["type_annotation"] += fixes
    return content, fixes

def fix_function_definitions(content: str) -> Tuple[str, int]:
    """Fix Python function definitions"""
    fixes = 0
    
    # Pattern 1: def x() -> class method
    pattern = r'(\s+)def\s+(\w+)\s*\(\s*([^)]*)\s*\)(?:\s*->\s*([^:]+))?\s*:'
    
    def function_replacement(match):
        nonlocal fixes
        indent = match.group(1)
        func_name = match.group(2)
        params = match.group(3)
        return_type = match.group(4)
        
        # Fix params - add types if missing
        if params:
            new_params = []
            for param in params.split(','):
                param = param.strip()
                if param and not ':' in param and param != 'self':
                    if param == 'self':
                        continue  # Skip self in TypeScript
                    new_params.append(f"{param}: any")
                elif param and param != 'self':
                    new_params.append(param)
            params = ', '.join(new_params)
        
        # Add return type
        if return_type:
            fixes += 1
            return f'{indent}{func_name}({params}): {return_type.strip()} {{'
        else:
            fixes += 1
            return f'{indent}{func_name}({params}): any {{'
    
    content = re.sub(pattern, function_replacement, content)
    
    # Pattern 2: Python for loops
    pattern = r'(\s+)for\s+([^:]+):'
    
    def for_loop_replacement(match):
        nonlocal fixes
        indent = match.group(1)
        for_expr = match.group(2).strip()
        
        # Handle for i in range()
        range_match = re.match(r'(\w+)\s+in\s+range\(([^)]+)\)', for_expr)
        if range_match:
            var = range_match.group(1)
            range_args = range_match.group(2).split(',')
            
            if len(range_args) == 1:
                # range(end)
                fixes += 1
                return f'{indent}for (let {var} = 0; {var} < {range_args[0].strip()}; {var}++) {{'
            elif len(range_args) == 2:
                # range(start, end)
                start = range_args[0].strip()
                end = range_args[1].strip()
                fixes += 1
                return f'{indent}for (let {var} = {start}; {var} < {end}; {var}++) {{'
            elif len(range_args) == 3:
                # range(start, end, step)
                start = range_args[0].strip()
                end = range_args[1].strip()
                step = range_args[2].strip()
                fixes += 1
                return f'{indent}for (let {var} = {start}; {var} < {end}; {var} += {step}) {{'
        
        # Handle for x in y
        in_match = re.match(r'(\w+)\s+in\s+(.+)', for_expr)
        if in_match:
            var = in_match.group(1)
            iterable = in_match.group(2).strip()
            fixes += 1
            return f'{indent}for (const {var} of {iterable}) {{'
        
        # Default - wrap in a typescript for statement
        fixes += 1
        return f'{indent}for ({for_expr}) {{'
    
    content = re.sub(pattern, for_loop_replacement, content)
    
    # Pattern 3: Python if statements
    pattern = r'(\s+)if\s+([^:]+):'
    
    def if_replacement(match):
        nonlocal fixes
        indent = match.group(1)
        condition = match.group(2).strip()
        fixes += 1
        return f'{indent}if ({condition}) {{'
    
    content = re.sub(pattern, if_replacement, content)
    
    # Pattern 4: Python elif
    pattern = r'(\s+)elif\s+([^:]+):'
    
    def elif_replacement(match):
        nonlocal fixes
        indent = match.group(1)
        condition = match.group(2).strip()
        fixes += 1
        return f'{indent}else if ({condition}) {{'
    
    content = re.sub(pattern, elif_replacement, content)
    
    # Pattern 5: Python else
    pattern = r'(\s+)else\s*:'
    
    def else_replacement(match):
        nonlocal fixes
        indent = match.group(1)
        fixes += 1
        return f'{indent}else {{'
    
    content = re.sub(pattern, else_replacement, content)
    
    # Pattern 6: Python while loops
    pattern = r'(\s+)while\s+([^:]+):'
    
    def while_replacement(match):
        nonlocal fixes
        indent = match.group(1)
        condition = match.group(2).strip()
        fixes += 1
        return f'{indent}while ({condition}) {{'
    
    content = re.sub(pattern, while_replacement, content)
    
    Config.STATS["fix_by_type"]["function_definition"] += fixes
    return content, fixes

def fix_typescript_file(file_path: str) -> int:
    """Fix TypeScript syntax issues in a file"""
    # Skip node_modules
    if 'node_modules' in file_path:
        return 0
    
    # First, check if it's an __init__.ts file which needs special handling
    new_file_path, init_fixes = fix_init_files(file_path)
    if new_file_path != file_path:
        return init_fixes
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        total_fixes = 0
        modified = False
        
        # Apply fixes
        content, fixes = fix_python_imports(content)
        total_fixes += fixes
        modified = modified or fixes > 0
        
        content, fixes = fix_destructuring(content)
        total_fixes += fixes
        modified = modified or fixes > 0
        
        content, fixes = fix_dict_syntax(content)
        total_fixes += fixes
        modified = modified or fixes > 0
        
        content, fixes = fix_self_to_this(content)
        total_fixes += fixes
        modified = modified or fixes > 0
        
        content, fixes = fix_python_class(content)
        total_fixes += fixes
        modified = modified or fixes > 0
        
        content, fixes = fix_python_docstring(content)
        total_fixes += fixes
        modified = modified or fixes > 0
        
        content, fixes = fix_exceptions(content)
        total_fixes += fixes
        modified = modified or fixes > 0
        
        content, fixes = fix_string_literals(content)
        total_fixes += fixes
        modified = modified or fixes > 0
        
        content, fixes = fix_type_annotations(content)
        total_fixes += fixes
        modified = modified or fixes > 0
        
        content, fixes = fix_function_definitions(content)
        total_fixes += fixes
        modified = modified or fixes > 0
        
        # Write back the fixed content
        if modified and Config.FIX_FILES:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Fixed {total_fixes} issues in {file_path}")
            return total_fixes
        elif modified:
            logger.info(f"Would fix {total_fixes} issues in {file_path} (dry run)")
            return total_fixes
        else:
            if Config.VERBOSE:
                logger.debug(f"No issues found in {file_path}")
            return 0
    except Exception as e:
        logger.error(f"Error fixing {file_path}: {e}")
        return 0

def generate_report():
    """Generate a comprehensive report of the fixes applied"""
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "typescript_syntax_fixes_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# TypeScript Syntax Fixes Report\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Files Checked:** {Config.STATS['files_checked']}\n")
        f.write(f"- **Files Fixed:** {Config.STATS['files_fixed']}\n")
        f.write(f"- **Total Fixes:** {Config.STATS['total_fixes']}\n\n")
        
        f.write("## Fixes by Type\n\n")
        f.write("| Fix Type | Count |\n")
        f.write("|----------|-------|\n")
        
        for fix_type, count in Config.STATS["fix_by_type"].items():
            pretty_name = fix_type.replace('_', ' ').title()
            f.write(f"| {pretty_name} | {count} |\n")
        
        f.write("\n## Next Steps\n\n")
        f.write("1. Run TypeScript compiler to check for remaining issues:\n")
        f.write("   ```bash\n")
        f.write("   cd ../ipfs_accelerate_js\n")
        f.write("   npx tsc --noEmit\n")
        f.write("   ```\n\n")
        
        f.write("2. Fix any remaining syntax errors manually\n\n")
        
        f.write("3. Run the import validation tool again:\n")
        f.write("   ```bash\n")
        f.write("   python validate_import_paths.py --fix\n")
        f.write("   ```\n\n")
        
        f.write("4. Fix more complex TypeScript type issues:\n")
        f.write("   ```bash\n")
        f.write("   python setup_typescript_test.py --fix-types\n")
        f.write("   ```\n\n")
    
    logger.info(f"Report generated: {report_path}")

def main():
    """Main function"""
    specific_files = setup_args()
    
    if specific_files:
        ts_files = specific_files
    else:
        # Find TypeScript files
        ts_files = find_typescript_files()
    
    # Fix TypeScript syntax in each file
    for file_path in ts_files:
        Config.STATS["files_checked"] += 1
        fixes = fix_typescript_file(file_path)
        
        if fixes > 0:
            Config.STATS["files_fixed"] += 1
            Config.STATS["total_fixes"] += fixes
    
    # Log summary
    logger.info("\nFix Summary:")
    logger.info(f"Files checked: {Config.STATS['files_checked']}")
    logger.info(f"Files fixed: {Config.STATS['files_fixed']}")
    logger.info(f"Total fixes: {Config.STATS['total_fixes']}")
    
    # Generate report
    generate_report()

if __name__ == "__main__":
    main()