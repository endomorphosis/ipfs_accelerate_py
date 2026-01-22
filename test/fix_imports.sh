#!/bin/bash
# Enhanced script to fix imports across the codebase to use the new structure

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
  
  # Find all Python files in the directory but skip archives and templates
  find "$directory" -name "*.py" -type f -not -path "*/archive/*" -not -path "*/templates/*" | while read -r file; do
    echo "Fixing imports in $file"
    
    # Replace direct imports from test directory
    sed -i "s/from test\./from $new_base\./g" "$file"
    sed -i "s/import test\./import $new_base\./g" "$file"
    
    # Fix direct imports to use package paths
    if [[ $new_base == "duckdb_api" ]]; then
      # Fix benchmark DB API imports
      sed -i 's/from benchmark_db_api import/from duckdb_api.core.benchmark_db_api import/g' "$file"
      sed -i 's/import benchmark_db_api/import duckdb_api.core.benchmark_db_api/g' "$file"
      
      # Fix database related imports
      sed -i 's/from benchmark_db_/from duckdb_api.core.benchmark_db_/g' "$file"
      
      # Fix time series imports
      sed -i 's/from time_series_performance import/from duckdb_api.time_series_performance import/g' "$file"
      
      # Fix visualization imports
      sed -i 's/from benchmark_visualizer import/from duckdb_api.visualization.benchmark_visualizer import/g' "$file"
      sed -i 's/from view_benchmark_results import/from duckdb_api.visualization.view_benchmark_results import/g' "$file"
      
      # Fix utility imports
      sed -i 's/from qnn_simulation_helper import/from duckdb_api.utils.qnn_simulation_helper import/g' "$file"
    fi
    
    if [[ $new_base == "generators" ]]; then
      # Fix generator imports
      sed -i 's/from template_generator/from generators.template_generators/g' "$file"
      sed -i 's/from merged_test_generator/from generators.test_generators/g' "$file"
      sed -i 's/from test_generator/from generators.test_generators/g' "$file"
      sed -i 's/from simple_test_generator/from generators.test_generators.simple_test_generator/g' "$file"
      
      # Fix template imports
      sed -i 's/from template_validator import/from generators.template_generators.template_validator import/g' "$file"
      sed -i 's/from template_processor import/from generators.skill_generators.template_processor import/g' "$file"
    fi
    
    # Add a notice at the top of the file (if not already there)
    if ! grep -q "MIGRATED:" "$file"; then
      sed -i "1s/^/# MIGRATED: This file was migrated from test/ to $new_base/\n/" "$file"
    fi
  done
}

# Create an __init__.py in every subdirectory to ensure imports work
ensure_init_files() {
  local directory=$1
  
  find "$directory" -type d -not -path "*/archive/*" | while read -r dir; do
    if [ ! -f "$dir/__init__.py" ]; then
      echo "Creating __init__.py in $dir"
      echo "# Auto-generated __init__.py file for module imports" > "$dir/__init__.py"
    fi
  done
}

# Verify there are no syntax errors in important files
check_syntax() {
  local directory=$1
  echo "Checking syntax in $directory..."
  
  find "$directory" -name "*.py" -type f -not -path "*/archive/*" -not -path "*/templates/*" | while read -r file; do
    python3 -m py_compile "$file" 2>/dev/null
    if [ $? -ne 0 ]; then
      echo "⚠️ Syntax error in $file"
    fi
  done
}

# Main execution
echo "Starting enhanced import fixing process..."

# Fix imports in migrated files
fix_imports "../generators"
fix_imports "../duckdb_api"

# Ensure __init__.py files exist in all subdirectories
ensure_init_files "../generators"
ensure_init_files "../duckdb_api"

# Check syntax on core directories (excluding templates and archives)
check_syntax "../duckdb_api/core"
check_syntax "../duckdb_api/migration"
check_syntax "../duckdb_api/utils" 
check_syntax "../duckdb_api/visualization"

echo "✅ Fixed imports in migrated files and ensured __init__.py files exist in all subdirectories."
echo "✅ Remember to run tests after this script to ensure everything works correctly."
