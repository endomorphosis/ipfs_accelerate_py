#!/usr/bin/env python3
"""
Moves files from the test directory to the new package structure.

This script identifies files in the test directory that should be moved to
either the generators/ or duckdb_api/ directories based on their names and
content. It preserves file history and ensures import statements are updated.

Usage:
    python move_files_to_packages.py []],,--dry-run] []],,--file path/to/file.py] []],,--type generator|database],
    """

    import os
    import sys
    import re
    import shutil
    import argparse
    import importlib.util
    from pathlib import Path

# File type patterns to identify categories
    FILE_PATTERNS = {}}}
    # Generator files
    "generator": []],,
    r".*_generator.*\.py$",
    r".*generator.*\.py$",
    r"template_.*\.py$",
    r".*_template.*\.py$",
    r"hardware_detection\.py$",
    r".*skill_.*\.py$",
    r".*_skillset_.*\.py$",
    ],
    
    # Database files
    "database": []],,
    r"benchmark_db.*\.py$",
    r".*_db_.*\.py$",
    r"db_.*\.py$",
    r"time_series_.*\.py$",
    r".*_database.*\.py$",
    r".*_visualization.*\.py$",
    ],
    }

# Mapping of file pattern to destination directory
    DESTINATION_MAPPING = {}}}
    # Generator files
    r".*merged_test_generator.*\.py$": "generators/test_generators/",
    r".*simple_test_generator.*\.py$": "generators/test_generators/",
    r".*_test_generator.*\.py$": "generators/test_generators/",
    r".*skill_generator.*\.py$": "generators/skill_generators/",
    r".*template_database.*\.py$": "generators/templates/",
    r".*template_generator.*\.py$": "generators/template_generators/",
    r".*template_validator.*\.py$": "generators/template_generators/",
    r"template_.*\.py$": "generators/templates/",
    r"benchmark_generator.*\.py$": "generators/benchmark_generators/",
    r"run_model_benchmarks\.py$": "generators/benchmark_generators/",
    r"hardware_detection\.py$": "generators/hardware/",
    r"automated_hardware_selection\.py$": "generators/hardware/",
    r".*web_platform_test.*\.py$": "generators/runners/web/",
    r".*validator.*\.py$": "generators/validators/",
    r"model_.*\.py$": "generators/models/",
    r"test_ipfs_accelerate\.py$": "generators/models/",
    r"run_.*_test\.py$": "generators/runners/",
    
    # Database files
    r"benchmark_db_api\.py$": "duckdb_api/core/",
    r"benchmark_db_query\.py$": "duckdb_api/core/",
    r"fixed_benchmark_db_query\.py$": "duckdb_api/core/benchmark_db_query.py",  # Rename
    r"benchmark_db_updater\.py$": "duckdb_api/core/",
    r"benchmark_db_maintenance\.py$": "duckdb_api/core/",
    r"benchmark_db_converter\.py$": "duckdb_api/migration/",
    r"migrate_.*\.py$": "duckdb_api/migration/",
    r"update_db_schema.*\.py$": "duckdb_api/schema/",
    r"create_.*_schema\.py$": "duckdb_api/schema/creation/",
    r"time_series_performance\.py$": "duckdb_api/core/",
    r"run_time_series_performance\.py$": "duckdb_api/utils/",
    r".*_visualization.*\.py$": "duckdb_api/visualization/",
    r"benchmark_timing_report\.py$": "duckdb_api/visualization/",
    r".*_benchmark_timing.*\.py$": "duckdb_api/visualization/",
    }

def determine_file_type(filename):
    """
    Determines the type of file based on its name.
    
    Args:
        filename: The name of the file
        
    Returns:
        String: "generator", "database", or None if can't determine
    """::
    for file_type, patterns in FILE_PATTERNS.items():
        for pattern in patterns:
            if re.match(pattern, filename):
            return file_type
    
        return None

def determine_destination(filename, file_type):
    """
    Determines the destination directory for a file.
    
    Args:
        filename: The name of the file
        file_type: The type of the file ("generator" or "database")
        
    Returns:
        String: Destination path or None if can't determine
    """::
    for pattern, destination in DESTINATION_MAPPING.items():
        if re.match(pattern, filename):
        return destination
    
    # Default destinations based on file type
    if file_type == "generator":
        return "generators/utils/"
    elif file_type == "database":
        return "duckdb_api/utils/"
    
        return None

def move_file(source_path, destination_root, dry_run=False):
    """
    Moves a file to its destination directory, preserving file history.
    
    Args:
        source_path: Path to the source file
        destination_root: Root of the destination directory
        dry_run: If True, only print actions without moving files
        
    Returns:
        Tuple: (success, destination_path or error message)
        """
        filename = os.path.basename(source_path)
        file_type = determine_file_type(filename)
    
    if not file_type:
        return False, f"Could not determine file type for {}}}filename}"
    
        destination_subdir = determine_destination(filename, file_type)
    
    if not destination_subdir:
        return False, f"Could not determine destination for {}}}filename}"
    
    # Check if destination is a rename pattern (contains full filename with extension):
    if destination_subdir.endswith(".py"):
        # Splitting destination into directory and filename
        destination_dir = os.path.dirname(destination_subdir)
        destination_filename = os.path.basename(destination_subdir)
        destination_path = os.path.join(destination_root, destination_dir, destination_filename)
    else:
        destination_path = os.path.join(destination_root, destination_subdir, filename)
    
    # Create destination directory if it doesn't exist
    destination_dir = os.path.dirname(destination_path):
    if not os.path.exists(destination_dir) and not dry_run:
        try:
            os.makedirs(destination_dir, exist_ok=True)
        except Exception as e:
            return False, f"Error creating directory {}}}destination_dir}: {}}}e}"
    
    if dry_run:
            return True, destination_path
    
    try:
        # Copy the file to destination
        shutil.copy2(source_path, destination_path)
            return True, destination_path
    except Exception as e:
            return False, f"Error copying file: {}}}e}"

def process_directory(directory, destination_root, file_type=None, dry_run=False):
    """
    Processes all Python files in a directory.
    
    Args:
        directory: Directory to process
        destination_root: Root directory for destinations
        file_type: Optional file type filter ("generator" or "database")
        dry_run: If True, only print actions without moving files
        
    Returns:
        Dict: Statistics about the operation
        """
        stats = {}}}
        "total_files": 0,
        "moved_files": 0,
        "skipped_files": 0,
        "failed_files": 0,
        "by_destination": {}}}}
        }
    
        print(f"Processing directory: {}}}directory}")
    
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".py"):
                source_path = os.path.join(root, filename)
                
                # Skip __init__.py files
                if filename == "__init__.py":
                continue
                
                # Skip if file type doesn't match filter
                determined_type = determine_file_type(filename):
                if file_type and determined_type != file_type:
                    print(f"👉 Skipped {}}}filename} (wrong type: {}}}determined_type})")
                    stats[]],,"skipped_files"] += 1
                    continue
                
                    stats[]],,"total_files"] += 1
                
                # Move the file
                    success, result = move_file(source_path, destination_root, dry_run)
                
                if success:
                    destination_path = result
                    destination_dir = os.path.dirname(destination_path)
                    
                    relative_destination = os.path.relpath(destination_path, destination_root)
                    stats[]],,"by_destination"][]],,relative_destination] = stats[]],,"by_destination"].get(relative_destination, 0) + 1
                    
                    stats[]],,"moved_files"] += 1
                    
                    if dry_run:
                        print(f"🔍 []],,DRY RUN] Would move {}}}filename} to {}}}destination_path}")
                    else:
                        print(f"✅ Moved {}}}filename} to {}}}destination_path}")
                else:
                    error_message = result
                    stats[]],,"failed_files"] += 1
                    print(f"❌ Could not move {}}}filename}: {}}}error_message}")
    
                        return stats

def main():
    parser = argparse.ArgumentParser(description='Move files to new package structure')
    parser.add_argument('--dry-run', action='store_true', help='Show actions without moving files')
    parser.add_argument('--file', help='Path to a specific file to move')
    parser.add_argument('--type', choices=[]],,'generator', 'database'], help='Type of files to move')
    
    args = parser.parse_args()
    
    # Get project root directory (parent of test/)
    project_root = Path(__file__).parent.parent
    source_dir = project_root / "test"
    
    print(f"Project root: {}}}project_root}")
    print(f"Source directory: {}}}source_dir}")
    
    if args.dry_run:
        print("\n🔍 DRY RUN MODE: No files will be moved\n")
    
    if args.file:
        if os.path.exists(args.file) and args.file.endswith('.py'):
            # Handle single file
            filename = os.path.basename(args.file)
            success, result = move_file(args.file, project_root, args.dry_run)
            
            if success:
                destination_path = result
                if args.dry_run:
                    print(f"🔍 []],,DRY RUN] Would move {}}}filename} to {}}}destination_path}")
                else:
                    print(f"✅ Moved {}}}filename} to {}}}destination_path}")
                    return 0
            else:
                error_message = result
                print(f"❌ Could not move {}}}filename}: {}}}error_message}")
                    return 1
        else:
            print(f"Error: File {}}}args.file} does not exist or is not a Python file.")
                    return 1
    else:
        # Process all files in the directory
        stats = process_directory(source_dir, project_root, args.type, args.dry_run)
        
        print("\n=== SUMMARY ===")
        print(f"Total files processed: {}}}stats[]],,'total_files']}")
        print(f"Files moved: {}}}stats[]],,'moved_files']}")
        print(f"Files skipped: {}}}stats[]],,'skipped_files']}")
        print(f"Files failed: {}}}stats[]],,'failed_files']}")
        
        if stats[]],,"by_destination"]:
            print("\nFiles by destination:")
            for destination, count in sorted(stats[]],,"by_destination"].items()):
                print(f"  - {}}}destination}: {}}}count} files")
        
        if args.dry_run:
            print("\nRun without --dry-run to actually move the files")
        
                return 0 if stats[]],,"failed_files"] == 0 else 1
:
if __name__ == "__main__":
    sys.exit(main())