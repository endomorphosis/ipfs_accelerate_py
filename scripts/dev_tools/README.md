# Dev Tools Scripts

This directory contains development tools for code quality and documentation management.

## Available Scripts

### compile_checker.py
Checks Python files for syntax errors and compilation issues.

**Usage:**
```bash
python scripts/dev_tools/compile_checker.py ipfs_accelerate_py --verbose
```

**Options:**
- `--verbose, -v`: Show detailed output for each file
- `--exclude`: Patterns to exclude (default: `__pycache__`, `.venv`, `venv`, `.git`, `build`, `dist`)

### comprehensive_import_checker.py
Validates import statements in Python files.

**Usage:**
```bash
python scripts/dev_tools/comprehensive_import_checker.py --directory ipfs_accelerate_py --verbose
```

**Options:**
- `--directory`: Directory to scan (required)
- `--verbose, -v`: Show detailed output
- `--exclude`: Patterns to exclude

### comprehensive_python_checker.py
Checks Python code quality using multiple metrics.

**Usage:**
```bash
python scripts/dev_tools/comprehensive_python_checker.py --directory ipfs_accelerate_py --verbose
```

**Features:**
- Analyzes file size and line count
- Counts functions and classes
- Identifies complexity issues (long functions, large files)
- Provides code quality summary

### docstring_audit.py
Audits Python files for documentation completeness.

**Usage:**
```bash
python scripts/dev_tools/docstring_audit.py --directory ipfs_accelerate_py --output docstring_report.json
```

**Options:**
- `--directory`: Directory to scan (required)
- `--output`: Save results to JSON file
- `--verbose, -v`: Show detailed output
- `--exclude`: Patterns to exclude (default excludes test files)

**Checks:**
- Module docstrings
- Class docstrings
- Function docstrings (excluding private functions)
- Documentation coverage rates

### find_documentation.py
Finds and catalogs documentation files.

**Usage:**
```bash
python scripts/dev_tools/find_documentation.py --directory . --format json
```

**Options:**
- `--directory`: Directory to scan (required)
- `--format`: Output format (`json` or `text`, default: `text`)
- `--exclude`: Patterns to exclude

**Finds:**
- README.md files
- TODO.md files
- CHANGELOG.md files
- CONTRIBUTING.md files
- LICENSE files
- MANIFEST.in files

## Integration with VSCode

All these scripts are integrated with VSCode tasks. Access them via:
1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Type "Tasks: Run Task"
3. Select "Dev Tools: [tool name]"

## Integration with CI/CD

These scripts can be integrated into CI/CD pipelines for automated quality checks:

```yaml
- name: Check Python Compilation
  run: python scripts/dev_tools/compile_checker.py ipfs_accelerate_py

- name: Audit Docstrings
  run: python scripts/dev_tools/docstring_audit.py --directory ipfs_accelerate_py --output docstring_report.json
```

## Quick Examples

**Check if all Python files compile:**
```bash
python scripts/dev_tools/compile_checker.py ipfs_accelerate_py
```

**Validate all imports:**
```bash
python scripts/dev_tools/comprehensive_import_checker.py --directory ipfs_accelerate_py
```

**Analyze code quality:**
```bash
python scripts/dev_tools/comprehensive_python_checker.py --directory ipfs_accelerate_py
```

**Check documentation coverage:**
```bash
python scripts/dev_tools/docstring_audit.py --directory ipfs_accelerate_py --verbose
```

**Find all documentation files:**
```bash
python scripts/dev_tools/find_documentation.py --directory . --format text
```

## Requirements

All scripts use only Python standard library modules. No additional dependencies required.

## Exit Codes

- `0`: Success (all checks passed)
- `1`: Failure (one or more checks failed)

Use exit codes for CI/CD integration:
```bash
if ! python scripts/dev_tools/compile_checker.py ipfs_accelerate_py; then
    echo "Compilation check failed!"
    exit 1
fi
```

## Output Formats

Scripts support multiple output formats:
- **Text**: Human-readable console output
- **JSON**: Machine-readable for automation
- **Verbose**: Detailed file-by-file output

## Related Documentation

- [VSCode Tasks](.vscode/README.md) - IDE integration
- [CI/CD Workflows](.github/workflows/) - Automation integration
- [docs/AUTOMATION_README.md](../docs/AUTOMATION_README.md) - Complete automation guide

---

**Last Updated:** January 29, 2026  
**Version:** 1.0  
**Status:** Active ✅
