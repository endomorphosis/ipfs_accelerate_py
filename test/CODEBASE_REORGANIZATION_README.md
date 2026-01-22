# Codebase Reorganization Tools

This directory contains scripts to help reorganize the codebase according to the plan in `CLAUDE.md`. The reorganization involves moving files from the `test/` directory to the new `generators/` and `duckdb_api/` directories.

## Overview

According to the organization update in `CLAUDE.md`, the codebase will be reorganized as follows:

1. Generator files will move to the **`generators/`** directory with subdirectories:
   - `benchmark_generators/`: Benchmark generation tools
   - `models/`: Model implementations and skills
   - `runners/`: Test runner scripts
   - `skill_generators/`: Skill generation tools
   - `template_generators/`: Template generation utilities
   - `templates/`: Template files for model generation
   - `test_generators/`: Test generation tools
   - `utils/`: Utility functions
   - `hardware/`: Hardware-specific generator tools

2. Database files will move to the **`duckdb_api/`** directory with subdirectories:
   - `core/`: Core database functionality
   - `migration/`: Migration tools for JSON to database
   - `schema/`: Database schema definitions
   - `utils/`: Utility functions for database operations
   - `visualization/`: Result visualization tools
   - `distributed_testing/`: Distributed testing framework components

## Reorganization Scripts

The following scripts are available to help with the reorganization:

### Master Script

- **`reorganize_codebase.py`**: Master script that guides you through the entire reorganization process. It will run the other scripts in the correct order and provide a user interface for the process.

```bash
# Run the full reorganization process interactively
python reorganize_codebase.py

# Start from a specific step
python reorganize_codebase.py --step 3

# Run in dry-run mode (no actual changes)
python reorganize_codebase.py --dry-run

# Run all steps without prompting
python reorganize_codebase.py --non-interactive
```

### Individual Scripts

These scripts can be run individually if you want to perform specific tasks:

1. **`create_package_structure.py`**: Creates the new directory structure with necessary `__init__.py` files.

```bash
# Create the directory structure
python create_package_structure.py

# Show what would be created without making changes
python create_package_structure.py --dry-run
```

2. **`fix_syntax_errors.py`**: Scans Python files for syntax errors and fixes them automatically.

```bash
# Fix syntax errors in all generator files in the test directory
python fix_syntax_errors.py

# Fix syntax errors in a specific file
python fix_syntax_errors.py --file /path/to/file.py

# Scan for errors without fixing them
python fix_syntax_errors.py --scan-only

# Show what would be fixed without making changes
python fix_syntax_errors.py --dry-run
```

3. **`move_files_to_packages.py`**: Moves files to their new locations based on file type.

```bash
# Move all generator files to the generators/ directory
python move_files_to_packages.py --type generator

# Move all database files to the duckdb_api/ directory
python move_files_to_packages.py --type database

# Move a specific file
python move_files_to_packages.py --file /path/to/file.py

# Show what would be moved without making changes
python move_files_to_packages.py --dry-run
```

4. **`update_imports.py`**: Updates import statements in all files to use the new package structure.

```bash
# Update import statements in all files
python update_imports.py

# Update import statements in a specific directory
python update_imports.py --dir /path/to/dir
```

5. **`update_doc_paths.py`**: Updates file paths in documentation files to reflect the new structure.

```bash
# Update paths in all documentation files
python update_doc_paths.py

# Update paths in a specific file
python update_doc_paths.py --file /path/to/file.md

# Show what would be updated without making changes
python update_doc_paths.py --dry-run
```

## Recommended Approach

1. **Backup your code**: Make sure you have a backup or clean git state before starting.

2. **Run the master script**: Use `reorganize_codebase.py` to guide you through the process.

3. **Review changes after each step**: The master script runs in interactive mode by default, allowing you to review changes after each step.

4. **Use dry-run mode**: If you're unsure, use `--dry-run` to see what would happen without making actual changes.

5. **Run tests**: After reorganization, run tests to make sure everything still works.

## Troubleshooting

If you encounter any issues during the reorganization:

1. Check the error messages for specific details.

2. Look at the backup files created by the scripts (with `.bak` extension).

3. Run the problematic script in `--dry-run` mode to see what changes it would make.

4. Fix any remaining issues manually if needed.

## Important Notes

- These scripts make a best effort to correctly categorize and move files, but some files may need manual intervention.

- Backup files are created with `.bak` extension during the syntax fixing process.

- Import statements are updated to reflect the new package structure, but some complex imports may need manual updates.

- Documentation files are updated to reference the new file locations, but some references may need manual updates.

- Run tests after reorganization to ensure everything works correctly.

## Additional Resources

- `CLAUDE.md`: Contains the detailed plan for the reorganization.
- `DOCUMENTATION_INDEX.md`: Overview of all project documentation.
- Each script has detailed help text accessible via `--help` option.