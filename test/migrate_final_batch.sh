#\!/bin/bash

# Create necessary directories
mkdir -p ../generators/test_generators/remaining
mkdir -p ../generators/models/remaining
mkdir -p ../generators/templates/remaining
mkdir -p ../generators/runners/remaining
mkdir -p ../generators/utils/remaining
mkdir -p ../duckdb_api/schema/remaining
mkdir -p ../duckdb_api/core/remaining
mkdir -p ../duckdb_api/utils/remaining
mkdir -p ../duckdb_api/migration/remaining

echo "Starting final batch migration..."

# Generator files - test patterns
for pattern in "test_generator_*.py" "fix_test_*.py"; do
  for file in $pattern; do
    if [ -f "$file" ]; then
      echo "Migrating test generator file: $file to ../generators/test_generators/remaining/"
      cp "$file" ../generators/test_generators/remaining/
    fi
  done
done

# Generator files - run patterns
for pattern in "run_*test*.py"; do
  for file in $pattern; do
    if [ -f "$file" ] && [ "$file" \!= "run_test_generator_against_all_skills.py" ]; then
      echo "Migrating runner file: $file to ../generators/runners/remaining/"
      cp "$file" ../generators/runners/remaining/
    fi
  done
done

# Database files - additional schema files
for pattern in "update_*db*.py" "check_*db*.py"; do
  for file in $pattern; do
    if [ -f "$file" ]; then
      echo "Migrating schema file: $file to ../duckdb_api/schema/remaining/"
      cp "$file" ../duckdb_api/schema/remaining/
    fi
  done
done

# Database files - utilities
for pattern in "*db_*script*.py" "*simulation*.py"; do
  for file in $pattern; do
    if [ -f "$file" ] && [ "$file" \!= "migration_helper.py" ]; then
      echo "Migrating db utility file: $file to ../duckdb_api/utils/remaining/"
      cp "$file" ../duckdb_api/utils/remaining/
    fi
  done
done

# Count migrated files
gen_count=$(find ../generators/test_generators/remaining ../generators/models/remaining ../generators/templates/remaining ../generators/runners/remaining ../generators/utils/remaining -type f -name "*.py" | wc -l)
db_count=$(find ../duckdb_api/schema/remaining ../duckdb_api/core/remaining ../duckdb_api/utils/remaining ../duckdb_api/migration/remaining -type f -name "*.py" | wc -l)

# Update migration progress
echo -e "\n### Final Batch Migration - $(date)" >> migration_progress.md
echo "Migrated additional files in final batch:" >> migration_progress.md
echo "- Additional generator files: $gen_count" >> migration_progress.md
echo "- Additional database files: $db_count" >> migration_progress.md
echo "" >> migration_progress.md
echo "Total generator files migrated: $((124 + gen_count)) / 183" >> migration_progress.md
echo "Total database files migrated: $((46 + db_count)) / 64" >> migration_progress.md
echo "Total overall migration progress: $(( (124 + gen_count + 46 + db_count) * 100 / (183 + 64) ))%" >> migration_progress.md
