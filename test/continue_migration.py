#\!/usr/bin/env python3
"""
Continues the migration of files from the test directory to the new package structure.
This script uses move_files_to_packages.py to migrate files and update_imports.py to fix imports.
"""

import os
import sys
import subprocess
from pathlib import Path

# Next set of files to migrate (generator files)
GENERATOR_FILES = [
    # Test generators
    "simple_test_generator.py",
    "merged_test_generator.py",
    "qualified_test_generator.py",
    # Template generators
    "template_generator.py",
    "comprehensive_template_generator.py",
    "template_validator.py",
    # Model files
    "model_family_classifier.py",
    "hardware_model_integration.py",
    # Utils
    "utils.py",
]

# Next set of files to migrate (database files)
DATABASE_FILES = [
    # Core files
    "benchmark_db_api.py",
    "benchmark_db_query.py",
    "benchmark_db_maintenance.py",
    # Migration tools
    "migrate_json_to_db.py",
    "benchmark_db_converter.py",
    # Visualization
    "benchmark_db_visualizer.py",
    "benchmark_timing_report.py",
]

def run_command(cmd):
    """Run a command and return its output"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Error: {result.stderr}", file=sys.stderr)
    return result.returncode == 0

def migrate_files(files, file_type):
    """Migrate a list of files to the new structure"""
    print(f"Migrating {len(files)} {file_type} files...")
    
    success_count = 0
    for file_name in files:
        # Check if the file exists
        file_path = Path(__file__).parent / file_name
        if not file_path.exists():
            print(f"File not found: {file_name}")
            continue
        
        # Run the migration script for this file
        cmd = f"python move_files_to_packages.py --file {file_path} --type {file_type}"
        if run_command(cmd):
            success_count += 1
        else:
            print(f"Failed to migrate {file_name}")
    
    print(f"Successfully migrated {success_count}/{len(files)} {file_type} files.")
    return success_count

def update_imports():
    """Update import statements in migrated files"""
    print("Updating imports in migrated files...")
    return run_command("python update_imports.py")

def main():
    """Main function"""
    # Ensure we're in the test directory
    os.chdir(Path(__file__).parent)
    
    # Check if the migration scripts exist
    if not Path("move_files_to_packages.py").exists():
        print("Error: move_files_to_packages.py not found.")
        return 1
    
    if not Path("update_imports.py").exists():
        print("Error: update_imports.py not found.")
        return 1
    
    # Create required directories
    for directory in [
        "../generators/test_generators", 
        "../generators/templates",
        "../generators/models",
        "../generators/utils",
        "../duckdb_api/core",
        "../duckdb_api/migration",
        "../duckdb_api/visualization"
    ]:
        Path(directory).mkdir(parents=True, exist_ok=True)
        # Create __init__.py in each directory
        init_file = Path(directory) / "__init__.py"
        if not init_file.exists():
            with open(init_file, "w") as f:
                f.write("# Auto-generated __init__.py for package structure\n")
    
    # Migrate generator files
    generator_count = migrate_files(GENERATOR_FILES, "generator")
    
    # Migrate database files
    database_count = migrate_files(DATABASE_FILES, "database")
    
    # Update imports in migrated files
    update_imports()
    
    # Print summary
    print("\n=== Migration Summary ===")
    print(f"Generator files migrated: {generator_count}/{len(GENERATOR_FILES)}")
    print(f"Database files migrated: {database_count}/{len(DATABASE_FILES)}")
    print(f"Total files migrated: {generator_count + database_count}/{len(GENERATOR_FILES) + len(DATABASE_FILES)}")
    
    # Run verification
    print("\nVerifying migration...")
    run_command("python verify_migration.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
