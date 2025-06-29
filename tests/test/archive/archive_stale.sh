#!/bin/bash

# Create archive directories if they don't exist
mkdir -p archive/stale_scripts
mkdir -p archive/old_documentation

# List of known deprecated script patterns
DEPRECATED_PATTERNS=(
  "fix_*generator*.py"
  "run_*generator*.py"
  "*_merged_test_generator*.py"
  "*_benchmark_db_*.py"
  "*_stale_*.py"
  "archive_old_*.py"
  "*_simulation_*.py"
  "benchmark_*_results*.py"
  "migrate_*.py"
  "cleanup_*.py"
  "fix_*_files.py"
  "update_*schema*.py"
  "update_*database*.py"
  "verify_*integration*.py"
  "verify_*migration*.py"
  "update_*documentation*.py"
  "fix_*syntax*.py"
  "fix_*import*.py"
  "create_*database*.py"
  "initialize_*database*.py"
)

# Move deprecated Python files
for pattern in "${DEPRECATED_PATTERNS[@]}"; do
  find . -name "$pattern" -not -path "./archive/*" -type f | while read file; do
    # Skip if the file doesn't exist (already moved)
    [ ! -f "$file" ] && continue
    
    # Get the relative directory structure
    dir=$(dirname "$file" | sed 's|^\./||')
    
    # Create the directory in archive/stale_scripts if it doesn't exist
    mkdir -p "archive/stale_scripts/$dir"
    
    # Move the file
    mv "$file" "archive/stale_scripts/$dir/"
  done
done

# Move old Markdown docs
find . -name "*.md" -not -path "./archive/*" -not -path "./CLAUDE.md" -not -path "./README.md" -mtime +60 -type f | while read file; do
  # Skip if the file doesn't exist (already moved)
  [ ! -f "$file" ] && continue
  
  # Get the relative directory structure
  dir=$(dirname "$file" | sed 's|^\./||')
  
  # Create the directory in archive/old_documentation if it doesn't exist
  mkdir -p "archive/old_documentation/$dir"
  
  # Move the file
  mv "$file" "archive/old_documentation/$dir/"
done

echo "Stale files moved to archive directory"