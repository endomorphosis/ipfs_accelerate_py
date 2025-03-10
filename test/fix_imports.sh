#\!/bin/bash
# Function to fix imports in Python files
fix_imports() {
  local directory=$1
  local original_base="test"
  local new_base=""
  
  if [[ $directory == *"generators"* ]]; then
    new_base="generators"
  elif [[ $directory == *"duckdb_api"* ]]; then
    new_base="duckdb_api"
  else
    echo "Unknown directory type: $directory"
    return 1
  fi
  
  # Find all Python files in the directory
  find "$directory" -name "*.py" -type f | while read -r file; do
    echo "Fixing imports in $file"
    
    # Replace direct imports from test directory
    sed -i "s/from test\./from $new_base\./g" "$file"
    sed -i "s/import test\./import $new_base\./g" "$file"
    
    # Add a notice at the top of the file
    sed -i "1s/^/# MIGRATED: This file was migrated from test/ to $new_base/\n/" "$file"
  done
}

# Create an __init__.py in every subdirectory to ensure imports work
ensure_init_files() {
  local directory=$1
  
  find "$directory" -type d | while read -r dir; do
    if [ \! -f "$dir/__init__.py" ]; then
      echo "Creating __init__.py in $dir"
      echo "# Auto-generated __init__.py file for module imports" > "$dir/__init__.py"
    fi
  done
}

# Fix imports in migrated files
fix_imports "../generators"
fix_imports "../duckdb_api"

# Ensure __init__.py files exist in all subdirectories
ensure_init_files "../generators"
ensure_init_files "../duckdb_api"

echo "Fixed imports in migrated files and ensured __init__.py files exist in all subdirectories."
