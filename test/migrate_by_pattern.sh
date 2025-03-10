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

echo "Starting pattern-based migration..."

# Generator files - test patterns
test_patterns=("test_vit_from_template.py" "verify_generator_improvements.py" "validate_merged_generator.py")
for pattern in "${test_patterns[@]}"; do
  if [ -f "$pattern" ]; then
    echo "Migrating test file: $pattern to ../generators/test_generators/remaining/"
    cp "$pattern" ../generators/test_generators/remaining/
  fi
done

# Generator files - model patterns
model_patterns=("skill_hf_*.py")
for pattern in "${model_patterns[@]}"; do
  for file in $pattern; do
    if [ -f "$file" ]; then
      echo "Migrating model file: $file to ../generators/models/remaining/"
      cp "$file" ../generators/models/remaining/
    fi
  done
done

# Generator files - utils patterns
util_patterns=("hardware_template_integration.py")
for pattern in "${util_patterns[@]}"; do
  if [ -f "$pattern" ]; then
    echo "Migrating util file: $pattern to ../generators/utils/remaining/"
    cp "$pattern" ../generators/utils/remaining/
  fi
done

# Database files - core patterns
db_core_patterns=("benchmark_database.py" "*db_api*.py")
for pattern in "${db_core_patterns[@]}"; do
  for file in $pattern; do
    if [ -f "$file" ]; then
      echo "Migrating db core file: $file to ../duckdb_api/core/remaining/"
      cp "$file" ../duckdb_api/core/remaining/
    fi
  done
done

# Database files - schema patterns
db_schema_patterns=("*template_database*.py" "*db_schema*.py" "create_*database*.py")
for pattern in "${db_schema_patterns[@]}"; do
  for file in $pattern; do
    if [ -f "$file" ]; then
      echo "Migrating db schema file: $file to ../duckdb_api/schema/remaining/"
      cp "$file" ../duckdb_api/schema/remaining/
    fi
  done
done

# Database files - migration patterns
db_migration_patterns=("*migration*.py" "*db_updater*.py")
for pattern in "${db_migration_patterns[@]}"; do
  for file in $pattern; do
    if [ -f "$file" ] && [[ "$file" \!= "migration_helper.py" ]] && [[ "$file" \!= "migrate_actual_files.sh" ]] && [[ "$file" \!= "migrate_by_pattern.sh" ]] && [[ "$file" \!= "migrate_remaining_files.sh" ]]; then
      echo "Migrating db migration file: $file to ../duckdb_api/migration/remaining/"
      cp "$file" ../duckdb_api/migration/remaining/
    fi
  done
done

# Count migrated files
gen_count=$(find ../generators/test_generators/remaining ../generators/models/remaining ../generators/templates/remaining ../generators/runners/remaining ../generators/utils/remaining -type f -name "*.py" | wc -l)
db_count=$(find ../duckdb_api/schema/remaining ../duckdb_api/core/remaining ../duckdb_api/utils/remaining ../duckdb_api/migration/remaining -type f -name "*.py" | wc -l)

# Update migration progress
echo -e "\n### Pattern-Based Migration - $(date)" >> migration_progress.md
echo "Migrated remaining files based on patterns:" >> migration_progress.md
echo "- Additional generator files: $gen_count" >> migration_progress.md
echo "- Additional database files: $db_count" >> migration_progress.md
echo "" >> migration_progress.md
echo "Total generator files migrated: $((124 + gen_count)) / 183" >> migration_progress.md
echo "Total database files migrated: $((46 + db_count)) / 64" >> migration_progress.md
echo "Total overall migration progress: $(( (124 + gen_count + 46 + db_count) * 100 / (183 + 64) ))%" >> migration_progress.md
