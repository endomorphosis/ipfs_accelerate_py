#\!/bin/bash

echo "Starting final stage migration..."

# Prioritized generator files
generator_files=(
  "bert_test_generator.py"
  "implementation_generator.py"
  "improved_integrated_generator.py" 
  "minimal_generator.py"
  "simple_model_test_generator.py"
  "verify_generator_fixes.py"
)

for file in "${generator_files[@]}"; do
  if [ -f "$file" ] && [ \! -f "../generators/test_generators/functional/$file" ]; then
    echo "Migrating generator file: $file"
    cp "$file" ../generators/test_generators/functional/
  fi
done

# Model-related files
model_files=(
  "model_compression.py"
  "model_family_classifier.py"
  "model_registry_integration.py"
  "model_benchmark_runner.py"
  "model_performance_predictor.py"
)

for file in "${model_files[@]}"; do
  if [ -f "$file" ] && [ \! -f "../generators/models/advanced/$file" ]; then
    echo "Migrating model file: $file"
    cp "$file" ../generators/models/advanced/
  fi
done

# Template-related files
template_files=(
  "add_template.py"
  "new_template.py"
  "hardware_template_integration.py"
  "demonstrate_template_inheritance.py"
)

for file in "${template_files[@]}"; do
  if [ -f "$file" ] && [ \! -f "../generators/templates/extra/$file" ]; then
    echo "Migrating template file: $file"
    cp "$file" ../generators/templates/extra/
  fi
done

# Database-related files
db_files=(
  "check_database_schema.py"
  "initialize_database.py"
  "update_benchmark_database.py"
  "web_audio_benchmark_db.py"
)

for file in "${db_files[@]}"; do
  if [ -f "$file" ] && [ \! -f "../duckdb_api/schema/extra/$file" ]; then
    echo "Migrating database file: $file"
    cp "$file" ../duckdb_api/schema/extra/
  fi
done

# Database verification files
db_verification_files=(
  "verify_database_integration_fixed.py"
)

for file in "${db_verification_files[@]}"; do
  if [ -f "$file" ] && [ \! -f "../duckdb_api/core/verification/$file" ]; then
    echo "Migrating database verification file: $file"
    cp "$file" ../duckdb_api/core/verification/
  fi
done

# Migration tools
migration_tools=(
  "migrate_json_to_db.py"
)

for file in "${migration_tools[@]}"; do
  if [ -f "$file" ] && [ \! -f "../duckdb_api/migration/tools/$file" ]; then
    echo "Migrating migration tool: $file"
    cp "$file" ../duckdb_api/migration/tools/
  fi
done

echo "Final migration stage completed."

# Count migrated files
gen_count=$(find ../generators/test_generators/functional ../generators/models/advanced ../generators/templates/extra -type f -name "*.py" | wc -l)
db_count=$(find ../duckdb_api/schema/extra ../duckdb_api/core/verification ../duckdb_api/migration/tools -type f -name "*.py" | wc -l)

# Update migration progress
echo -e "\n### Final Migration Stage - $(date)" >> migration_progress.md
echo "Migrated final batch of high-priority files:" >> migration_progress.md
echo "- Additional generator files: $gen_count" >> migration_progress.md
echo "- Additional database files: $db_count" >> migration_progress.md
echo "" >> migration_progress.md
echo "Total generator files migrated: $((124 + gen_count)) / 183" >> migration_progress.md
echo "Total database files migrated: $((55 + db_count)) / 64" >> migration_progress.md
echo "Total overall migration progress: $(( (124 + gen_count + 55 + db_count) * 100 / (183 + 64) ))%" >> migration_progress.md
