#\!/bin/bash
# Create necessary directories if they don't exist
mkdir -p ../duckdb_api/schema/creation ../duckdb_api/core/api ../duckdb_api/utils/integration

# Copy database files
for db_file in benchmark_database.py template_database.py update_template_database.py create_template_database.py; do
  if [ -f "$db_file" ]; then
    echo "Migrating $db_file to ../duckdb_api/schema/"
    cp "$db_file" ../duckdb_api/schema/
  fi
done

# Copy benchmark schema files
if [ -f "create_benchmark_schema.py" ]; then
  echo "Migrating create_benchmark_schema.py to ../duckdb_api/schema/creation/"
  cp create_benchmark_schema.py ../duckdb_api/schema/creation/
fi

if [ -f "create_db_models.py" ]; then
  echo "Migrating create_db_models.py to ../duckdb_api/schema/creation/"
  cp create_db_models.py ../duckdb_api/schema/creation/
fi

# Copy API client and maintenance files
for api_file in benchmark_db_api_client.py benchmark_db_api_orig.py; do
  if [ -f "$api_file" ]; then
    echo "Migrating $api_file to ../duckdb_api/core/api/"
    cp "$api_file" ../duckdb_api/core/api/
  fi
done

for script_file in benchmark_db_maintenance_script.py benchmark_db_query_script.py; do
  if [ -f "$script_file" ]; then
    echo "Migrating $script_file to ../duckdb_api/utils/"
    cp "$script_file" ../duckdb_api/utils/
  fi
done

# Update migration progress
echo -e "\n### Database Migration Progress - $(date)" >> migration_progress.md
echo "Migrated the following database files:" >> migration_progress.md
ls -la ../duckdb_api/schema/ ../duckdb_api/schema/creation/ ../duckdb_api/core/api/ ../duckdb_api/utils/ 2>/dev/null | grep -v "total" | grep -v "\.\." | grep -v "\." >> migration_progress.md
