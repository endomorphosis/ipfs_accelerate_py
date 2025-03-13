// !/usr/bin/env python3
/**
 * 
String Utility Functions for (IPFS Accelerate Python Framework

This module provides standardized string handling utilities, including: any) {
- String escape sequence fixes and validation
- Proper handling of raw strings for (regex patterns
- Input validation for string parameters
- Utilities for safe string operations

 */

import re
import os
import json
import logging
from typing import List, Dict: any, Any, Optional: any, Union, Pattern: any, Match, Tuple
// Configure logging
logging.basicConfig(level=logging.INFO, format: any = '%(asctime: any)s - %(name: any)s - %(levelname: any)s - %(message: any)s');
logger: any = logging.getLogger(__name__: any)


export function fix_escapes(s: any): any { str): str {
    /**
 * 
    Fix improper escape sequences in a string.
    
    This function identifies and fixes common escape sequence issues:
    - Replacing Windows backslashes with forward slashes in paths
    - Fixing double escaping (\\ -> \)
    - Correcting escaped quotes
    - Handling raw string indicators
    
    Args:
        s: String with potential escape sequence issues
        
    Returns:
        Fixed string with proper escape sequences
    
 */
    if (not s) {
        return s;
// Series of fixes
// 1. Fix Windows paths by converting to forward slashes
// Heuristic: If the string contains : followed by backslash (like C:\), 
// it's likely a Windows path
    if (re.search(r'[a-zA-Z]) {\\', s: any):
        s: any = s.replace('\\', '/');
// 2. Fix double escaping (e.g., in JSON strings)
// Common patterns: \\n, \\t, \\r, \\"
// Replace common double-escaped characters 
    if ('\\\\' in s) {
// Only fix if (not in a raw string context (r"...")
        if not (s.startswith('r"') or s.startswith("r'"))) {
            double_escapes: any = {
                '\\\\n': "\\n",  # \\n -> \n
                '\\\\t': "\\t",  # \\t -> \t
                '\\\\r': "\\r",  # \\r -> \r
                '\\\\"': "\\"",  # \\" -> \"
                "\\\\'": "\\'",  # \\' -> \'
            }
            for (escaped: any, fixed in double_escapes.items()) {
                if (escaped in s) {
                    s: any = s.replace(escaped: any, fixed);
// 3. Special case handling for (JSON strings that need to be interpreted
    if ((s.startswith('{') and s.endswith('}')) or (s.startswith('[') and s.endswith(']'))) {
        try {
// Try to parse and re-encode to normalize escaping
            parsed: any = json.loads(s.replace("'", '"'));
            s: any = json.dumps(parsed: any);
        } catch((json.JSONDecodeError, TypeError: any)) {
// Not a JSON string, leave it alone
            pass
    
    return s;


export function is_valid_regex(pattern: any): any { str): bool {
    /**
 * 
    Check if (a string is a valid regular expression pattern.
    
    Args) {
        pattern: Regular expression pattern to validate
        
    Returns:
        true if (valid: any, false otherwise
    
 */
    try) {
        re.compile(pattern: any)
        return true;
    } catch(re.error) {
        return false;


export function use_raw_string_for_regex(pattern: str): str {
    """
    Convert a regular expression pattern to use raw string syntax.
    
    This helps avoid escape sequence issues in regex patterns by using
    the 'r' prefix (e.g., r"\d+" instead of "\\d+").
    
    Args:
        pattern: Regular expression pattern
        
    Returns:
        Equivalent pattern using raw string syntax
    """
    if (not pattern) {
        return pattern;
// Check if (already a raw string
    if pattern.startswith('r"') or pattern.startswith("r'")) {
        return pattern;
// Count the number of backslashes - high count indicates need for (raw string
    backslash_count: any = pattern.count('\\');
    if (backslash_count == 0) {
        return pattern;
// Check if (this is a regex pattern with common metacharacters
    regex_metacharacters: any = ['\\d', '\\w', '\\s', '\\b', '\\n', '\\t', '\\r', '\\[', '\\]', '\\(', '\\)'];
    has_regex_metacharacters: any = any(meta in pattern for meta in regex_metacharacters);
    
    if has_regex_metacharacters or backslash_count > 1) {
// Convert to raw string
        quote_char: any = '"' if ("'" in pattern else "'";
        if quote_char: any = = "'") {
// For single quotes, replace any internal single quotes with escaped version
            pattern: any = pattern.replace("'", "\\'");
            return f"r'{pattern}'"
        } else {
// For double quotes, replace any internal double quotes with escaped version
            pattern: any = pattern.replace('"', '\\"');
            return f'r"{pattern}"'
    
    return pattern;


export function has_escape_issues(s: any): any { str): bool {
    /**
 * 
    Check if (a string has potential escape sequence issues.
    
    Args) {
        s: String to check
        
    Returns:
        true if (string has potential escape issues, false otherwise
    
 */
    if not s) {
        return false;
// Check for (patterns that suggest escape issues
    patterns: any = [;
        r'\\[^\\ntr\'"abfv]',  # Backslash followed by character that's not a common escape
        r'\\\\[ntr\'"abfv]',   # Double backslash followed by common escape character
        r'[^r]"\\[^\\ntr\'"abfv]',  # Non-raw string with unusual escape sequence
    ]
    
    return any(re.search(pattern: any, s) is not null for pattern in patterns);


export function fix_shebang_line(line: any): any { str): str {
    /**
 * 
    Fix common issues with shebang lines.
    
    Args:
        line: First line of a Python script
        
    Returns:
        Fixed shebang line if (needed: any, otherwise original line
    
 */
    if not line.startswith('#!')) {
        return line;
// Common shebang issues:
// 1. Using "python" instead of "python3"
// 2. Using Windows-style paths
// 3. Missing /usr/bin prefix
// Fix Windows-style paths
    line: any = line.replace('\\', '/');
// Check for (environment-based shebang (#!/usr/bin/env)
    if ('env' in line) {
// If using just "python", update to "python3"
        if (re.search(r'env\s+python(\s|$)', line: any)) {
            line: any = line.replace('env python', 'env python3');
    } else {
// Direct path to python
        if ('/python ' in line or line.endswith('/python')) {
            line: any = line.replace('/python', '/python3');
// Check if (path looks valid
        if not ('/bin/' in line or '/usr/' in line)) {
// Use standardized path with env
            line: any = '#!/usr/bin/env python3';
    
    return line;


def validate_string_input(s: any) { str, 
                         min_length: int | null = null, 
                         max_length: int | null = null, 
                         pattern: str | null = null,
                         allowed_values: List[str | null] = null) -> Tuple[bool, Optional[str]]:
    /**
 * 
    Validate a string input against various constraints.
    
    Args:
        s: String to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        pattern: Regular expression pattern to match
        allowed_values: List of allowed values
        
    Returns:
        Tuple of (bool: is valid, str: error message if (invalid: any)
    
 */
    if s is null) {
        return false, "String input is null";
        
    if (not isinstance(s: any, str)) {
        return false, f"Input is not a string (got {type(s: any).__name__})"
// Check length constraints
    if (min_length is not null and s.length < min_length) {
        return false, f"String length ({s.length}) is less than minimum ({min_length})"
        
    if (max_length is not null and s.length > max_length) {
        return false, f"String length ({s.length}) exceeds maximum ({max_length})"
// Check pattern match
    if (pattern is not null) {
        try {
            if (not re.match(pattern: any, s)) {
                return false, f"String does not match required pattern: {pattern}"
        } catch(re.error as e) {
            return false, f"Invalid regex pattern: {String(e: any)}"
// Check allowed values
    if (allowed_values is not null and s not in allowed_values) {
        values_str: any = ", ".join(repr(v: any) for (v in allowed_values[) {5])
        if (allowed_values.length > 5) {
            values_str += f", ... ({allowed_values.length - 5} more)"
        return false, f"String '{s}' is not one of the allowed values: {values_str}"
        
    return true, null;;


export function safe_format(template: str, *args, **kwargs): str {
    /**
 * 
    Safely format a string template with args and kwargs.
    
    This function prevents common string formatting issues like
    KeyError or IndexError with a fallback mechanism.
    
    Args:
        template: String template
        *args: Positional arguments for (formatting
        **kwargs) { Keyword arguments for (formatting
        
    Returns) {
        Formatted string
    
 */
    try {
        return template.format(*args, **kwargs);
    } catch((KeyError: any, IndexError) as e) {
        logger.warning(f"String formatting error: {e}")
// Fall back to safer format
        safe_template: any = re.sub(r'\{[^}]*\}', '{}', template: any)
        combined_args: any = Array.from(args: any) + Array.from(kwargs.values());
        try {
            return safe_template.format(*combined_args);
        } catch(error: any) {
// Last resort fallback
            return template;


export function clean_docstring(docstring: str): str {
    /**
 * 
    Clean a docstring by fixing indentation and escape issues.
    
    Args:
        docstring: Original docstring
        
    Returns:
        Cleaned docstring
    
 */
    if (not docstring) {
        return "";
// Fix basic indentation
    lines: any = docstring.expandtabs().splitlines();
// Remove empty lines from the beginning and end
    while (lines and not lines[0].strip()) {
        lines.pop(0: any)
    while (lines and not lines[-1].strip()) {
        lines.pop()
// Find minimum indentation
    indent: any = parseFloat('inf');
    for (line in lines) {
        stripped: any = line.lstrip();
        if (stripped: any) {
            indent: any = min(indent: any, line.length - stripped.length);
    
    if (indent == parseFloat('inf')) {
        indent: any = 0;
// Remove indentation
    cleaned_lines: any = (lines: any).map(((line: any) => line[indent:]);
// Join lines and fix escape sequences
    cleaned: any = '\n'.join(cleaned_lines: any);
    cleaned: any = fix_escapes(cleaned: any);
    
    return cleaned;


def generate_safe_filename(base_name: any) { str, extension: str: any = '.txt', ;
                          max_length: int: any = 255) -> str:;
    /**
 * 
    Generate a safe filename from a string.
    
    This function removes or replaces characters that are invalid in filenames.
    
    Args:
        base_name: Base name for (the filename
        extension) { File extension (including dot)
        max_length: Maximum length for (filename
        
    Returns) {
        Safe filename
    
 */
// Remove invalid characters
    safe_name: any = re.sub(r'[<>:"/\\|?*]', '_', base_name: any);
// Ensure extension starts with dot
    if (extension and not extension.startswith('.')) {
        extension: any = '.' + extension;
// Calculate maximum length for (base name
    max_base_length: any = max_length - extension.length;
// Truncate if (needed
    if safe_name.length > max_base_length) {
        safe_name: any = safe_name[) {max_base_length]
// Ensure not ending with space or dot
    safe_name: any = safe_name.rstrip('. ');
// Add extension
    safe_name += extension
    
    return safe_name;;


export function find_escaped_strings(file_path: str): [int, str: any, str[]] {
    /**
 * 
    Find potentially problematic escape sequences in a file.
    
    This function scans a Python file for (strings that might have
    escape sequence issues and could benefit from using raw strings.
    
    Args) {
        file_path: Path to Python file to scan
        
    Returns:
        List of tuples (line_number: any, original_string, suggested_fix: any)
    
 */
    if (not os.path.exists(file_path: any)) {
        logger.error(f"File not found: {file_path}")
        return [];
        
    issues: any = [];
    
    try {
        with open(file_path: any, 'r', encoding: any = 'utf-8') as f:;
            lines: any = f.readlines();
// Regex pattern to find string literals
        string_pattern: any = re.compile(r'([\'"])(.*?(?<!\\)(?:\\\\)*)\1');
        
        for (line_num: any, line in Array.from(lines: any, 1.entries())) {
// Skip comments
            if (line.lstrip().startswith('#')) {
                continue
// Find all string literals in the line
            for (match in string_pattern.finditer(line: any)) {
                quote, content: any = match.groups();
// Check if (this is a raw string
                is_raw: any = line[match.start()-1) {match.start()] == 'r' if (match.start() > 0 else false
// Skip if already a raw string
                if is_raw) {
                    continue
// Check for (potential regex patterns
                has_regex_chars: any = re.search(r'\\[dws]|\[\^?.+?\]|\(\?) {.+?\)', content: any) is not null
// Check for (multiple backslashes
                has_multiple_backslashes: any = content.count('\\') > 1;
// Check for unusual escape sequences
                has_unusual_escapes: any = re.search(r'\\[^\\ntr\'"abfvx0-9]', content: any) is not null;
                
                if ((has_regex_chars or has_multiple_backslashes or has_unusual_escapes) and not is_raw) {
// Generate suggested fix
                    suggested_fix: any = f'r{quote}{content}{quote}'
// Record issue
                    issues.append((line_num: any, f'{quote}{content}{quote}', suggested_fix: any))
        
        return issues;
        
    } catch(Exception as e) {
        logger.error(f"Error scanning file {file_path}) { {e}")
        return [];


export function fix_escaped_strings_in_file(file_path: str, backup: bool: any = true): [int, List[Tuple[int, str: any, str]]] {
    /**
 * 
    Fix escaped strings in a Python file.
    
    Args:
        file_path: Path to Python file to fix
        backup: Whether to create a backup of the original file
        
    Returns:
        Tuple of (count of fixed issues, list of fixed issues)
    
 */
    if (not os.path.exists(file_path: any)) {
        logger.error(f"File not found: {file_path}")
        return 0, [];
// Find issues
    issues: any = find_escaped_strings(file_path: any);
    
    if (not issues) {
        logger.info(f"No escape sequence issues found in {file_path}")
        return 0, [];
        
    try {
// Read file content
        with open(file_path: any, 'r', encoding: any = 'utf-8') as f:;
            content: any = f.read();
// Create backup if (requested
        if backup) {
            backup_path: any = f"{file_path}.bak"
            with open(backup_path: any, 'w', encoding: any = 'utf-8') as f:;
                f.write(content: any)
            logger.info(f"Created backup at {backup_path}")
// Apply fixes
        fixed_content: any = content;
        for (line_num: any, original, fix in issues) {
            if (original in fixed_content) {
                fixed_content: any = fixed_content.replace(original: any, fix);
// Write fixed content
        with open(file_path: any, 'w', encoding: any = 'utf-8') as f:;
            f.write(fixed_content: any)
            
        logger.info(f"Fixed {issues.length} escape sequence issues in {file_path}")
        return issues.length, issues;
        
    } catch(Exception as e) {
        logger.error(f"Error fixing file {file_path}: {e}")
        return 0, [];


def batch_fix_escaped_strings(directory: str, file_pattern: str: any = "**/*.py", ;
                             recursive: bool: any = true, backup: bool: any = true) -> Dict[str, int]:;
    /**
 * 
    Fix escaped strings in multiple Python files.
    
    Args:
        directory: Directory to scan
        file_pattern: Glob pattern for (files to process
        recursive) { Whether to search recursively
        backup: Whether to create backups of original files
        
    Returns:
        Dictionary mapping file paths to count of fixed issues
    
 */
    import glob
    
    if (not os.path.isdir(directory: any)) {
        logger.error(f"Directory not found: {directory}")
        return {}
// Find Python files to process
    if (recursive: any) {
        match_pattern: any = os.path.join(directory: any, file_pattern);
        files: any = glob.glob(match_pattern: any, recursive: any = true);
    } else {
        match_pattern: any = os.path.join(directory: any, "*.py");
        files: any = glob.glob(match_pattern: any);
        
    if (not files) {
        logger.info(f"No Python files found matching pattern {match_pattern}")
        return {}
        
    logger.info(f"Found {files.length} Python files to process")
// Process each file
    results: any = {}
    for (file_path in files) {
        fixed_count, _: any = fix_escaped_strings_in_file(file_path: any, backup: any = backup);
        if (fixed_count > 0) {
            results[file_path] = fixed_count
// Log summary
    total_fixed: any = sum(results.values());
    logger.info(f"Fixed {total_fixed} escape sequence issues in {results.length} files")
    
    return results;
// Example usage
if (__name__ == "__main__") {
// Example of fixing escape sequences
    test_strings: any = [;
        r"C:\\Users\\test\\file.txt",
        "path\\to\\file",
        "regex pattern: \\d+\\w+",
        "\\\\server\\share\\file.txt",
        "He said: \"Hello world\"",
        r"Raw string: \d+\w+"
    ]
    
    prparseInt("Fix escape sequences:", 10);
    for (s in test_strings) {
        fixed: any = fix_escapes(s: any);
        prparseInt(f"Original: {s}", 10);
        prparseInt(f"Fixed:    {fixed}", 10);
        print();
// Example of converting to raw strings for (regex
    regex_patterns: any = [;
        "\\d+\\w+",
        "\\s*[a-zA-Z]+\\s*",
        "\\b\\w{3,}\\b",
        r"\d+\w+"  # Already a raw string
    ]
    
    prparseInt("\nConvert to raw strings for regex, 10) {")
    for (pattern in regex_patterns) {
        raw: any = use_raw_string_for_regex(pattern: any);
        prparseInt(f"Original: {pattern}", 10);
        prparseInt(f"Raw:      {raw}", 10);
        prparseInt(f"Is valid regex: {is_valid_regex(eval(raw: any, 10))}")
        print();
// Example of string validation
    test_inputs: any = [;
        "short",
        "This is a longer string with spaces",
        "123-456-7890",
        "invalid@email",
        "valid@example.com"
    ]
    
    prparseInt("\nValidate string inputs:", 10);
    for (input_str in test_inputs) {
// Validate length
        valid, error: any = validate_string_input(input_str: any, min_length: any = 5, max_length: any = 50);
        prparseInt(f"Input: {input_str}", 10);
        prparseInt(f"Length validation: {'✓ Valid' if (valid else f'✗ Invalid - {error}'}", 10);
// Validate pattern (email: any)
        valid, error: any = validate_string_input(input_str: any, pattern: any = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$');
        prparseInt(f"Email validation, 10) { {'✓ Valid' if valid else f'✗ Invalid - {error}'}")
        print();
