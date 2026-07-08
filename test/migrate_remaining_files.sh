#\!/bin/bash

# Create necessary directories
mkdir -p ../generators/test_generators/remaining
mkdir -p ../generators/models/remaining
mkdir -p ../generators/templates/remaining
mkdir -p ../generators/runners/remaining
mkdir -p ../duckdb_api/schema/remaining
mkdir -p ../duckdb_api/core/remaining
mkdir -p ../duckdb_api/utils/remaining

# Get the list of remaining generator and DB files from migration_helper.py
generator_files=$(python migration_helper.py --generators | grep -A 100 "Files that need migration" | grep -v "Files that need migration" | grep "\.py" | sed 's/^[[:space:]]*-[[:space:]]*//g')
db_files=$(python migration_helper.py --db | grep -A 100 "Files that need migration" | grep -v "Files that need migration" | grep "\.py" | sed 's/^[[:space:]]*-[[:space:]]*//g')

# Migrate existing generator files
echo "Migrating generator files that exist..."
for file in $generator_files; do
  if [ -f "$file" ]; then
    # Determine destination based on filename pattern
    if [[ "$file" == *"skill_"* ]]; then
      dest="../generators/models/remaining/"
    elif [[ "$file" == *"template"* ]]; then
      dest="../generators/templates/remaining/"
    elif [[ "$file" == *"run_"* ]]; then
      dest="../generators/runners/remaining/"
    else
      dest="../generators/test_generators/remaining/"
    fi
    
    echo "Migrating: $file to $dest"
    cp "$file" "$dest"
  fi
done

# Migrate existing DB files
echo -e "\nMigrating DB files that exist..."
for file in $db_files; do
  if [ -f "$file" ]; then
    # Determine destination based on filename pattern
    if [[ "$file" == *"api"* || "$file" == *"integration"* ]]; then
      dest="../duckdb_api/core/remaining/"
    elif [[ "$file" == *"schema"* || "$file" == *"db_"* || "$file" == *"database"* ]]; then
      dest="../duckdb_api/schema/remaining/"
    else
      dest="../duckdb_api/utils/remaining/"
    fi
    
    echo "Migrating: $file to $dest"
    cp "$file" "$dest"
  fi
done

# Count how many files were actually migrated
gen_count=$(find ../generators/test_generators/remaining ../generators/models/remaining ../generators/templates/remaining ../generators/runners/remaining -type f -name "*.py" | wc -l)
db_count=$(find ../duckdb_api/schema/remaining ../duckdb_api/core/remaining ../duckdb_api/utils/remaining -type f -name "*.py" | wc -l)

# Update migration progress
echo -e "\n### Final Migration Update - $(date)" >> migration_progress.md
echo "Migrated remaining files:" >> migration_progress.md
echo "- Additional generator files: $gen_count" >> migration_progress.md
echo "- Additional database files: $db_count" >> migration_progress.md
echo "" >> migration_progress.md
echo "Total generator files migrated: $((118 + gen_count)) / 183" >> migration_progress.md
echo "Total database files migrated: $((46 + db_count)) / 64" >> migration_progress.md
echo "Total overall migration progress: $(( (118 + gen_count + 46 + db_count) * 100 / (183 + 64) ))%" >> migration_progress.md
