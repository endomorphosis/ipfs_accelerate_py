#\!/bin/bash
# Create necessary directories
mkdir -p ../duckdb_api/core/integration ../duckdb_api/schema/templates ../duckdb_api/schema/creation ../duckdb_api/utils/maintenance

# Check and migrate high-priority database files
priority_db_files=(
  "benchmark_database.py"
  "template_database.py"
  "update_template_database.py"
  "create_template_database.py"
  "benchmark_db_fix.py"
  "benchmark_db_performance.py"
  "benchmark_db_analytics.py"
  "benchmark_db_api_client.py"
  "fixed_benchmark_db_query.py"
)

for file in "${priority_db_files[@]}"; do
  if [ -f "$file" ]; then
    if [[ "$file" == *"template"* ]]; then
      echo "Migrating database template file: $file to ../duckdb_api/schema/templates/"
      cp "$file" ../duckdb_api/schema/templates/
    elif [[ "$file" == *"fix"* || "$file" == *"maintenance"* ]]; then
      echo "Migrating database maintenance file: $file to ../duckdb_api/utils/maintenance/"
      cp "$file" ../duckdb_api/utils/maintenance/
    elif [[ "$file" == *"analytics"* || "$file" == *"performance"* ]]; then
      echo "Migrating database analytics file: $file to ../duckdb_api/utils/"
      cp "$file" ../duckdb_api/utils/
    elif [[ "$file" == *"api"* || "$file" == *"client"* ]]; then
      echo "Migrating database API file: $file to ../duckdb_api/core/api/"
      cp "$file" ../duckdb_api/core/api/
    elif [[ "$file" == "benchmark_database.py" ]]; then
      echo "Migrating core database file: $file to ../duckdb_api/core/"
      cp "$file" ../duckdb_api/core/
    else
      echo "Migrating database file: $file to ../duckdb_api/schema/"
      cp "$file" ../duckdb_api/schema/
    fi
  else
    echo "File not found: $file"
  fi
done

# Check and migrate schema creation files
schema_files=(
  "create_db_models.py"
  "create_benchmark_schema.py"
  "create_new_database.py"
  "create_compatibility_tables.py"
  "create_schema_temp.py"
)

for file in "${schema_files[@]}"; do
  if [ -f "$file" ]; then
    echo "Migrating schema creation file: $file to ../duckdb_api/schema/creation/"
    cp "$file" ../duckdb_api/schema/creation/
  else
    echo "File not found: $file"
  fi
done

# Check and migrate integration files
integration_files=(
  "database_integration.py"
  "implement_db_schema_enhancements.py"
)

for file in "${integration_files[@]}"; do
  if [ -f "$file" ]; then
    echo "Migrating integration file: $file to ../duckdb_api/core/integration/"
    cp "$file" ../duckdb_api/core/integration/
  else
    echo "File not found: $file"
  fi
done

# Update the migration progress
echo -e "\n### Additional Database Files Migration - $(date)" >> migration_progress.md
echo "Migrated high-priority database files to appropriate directories" >> migration_progress.md
