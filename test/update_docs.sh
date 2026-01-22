#\!/bin/bash

# This script updates documentation files to reflect the new folder structure

# Function to update paths in markdown files
update_markdown_file() {
    local file=$1
    echo "Updating $file..."
    
    # Update references to old test directory paths
    sed -i 's|python test_|python generators/models/test_|g' "$file"
    sed -i 's|python benchmark_|python duckdb_api/core/benchmark_|g' "$file"
    
    # Update references to core database tools
    sed -i 's|benchmark_db_query\.py|duckdb_api/core/benchmark_db_query.py|g' "$file"
    sed -i 's|benchmark_db_api\.py|duckdb_api/core/benchmark_db_api.py|g' "$file"
    sed -i 's|benchmark_db_maintenance\.py|duckdb_api/core/benchmark_db_maintenance.py|g' "$file"
    
    # Update references to migration tools
    sed -i 's|migrate_json_to_db\.py|duckdb_api/migration/migrate_json_to_db.py|g' "$file"
    sed -i 's|cleanup_test_results\.py|duckdb_api/migration/cleanup_test_results.py|g' "$file"
    
    # Update references to generator tools
    sed -i 's|merged_test_generator\.py|generators/test_generators/merged_test_generator.py|g' "$file"
    sed -i 's|simple_test_generator\.py|generators/test_generators/simple_test_generator.py|g' "$file"
    sed -i 's|integrated_skillset_generator\.py|generators/skill_generators/integrated_skillset_generator.py|g' "$file"
    
    # Update references to web platform tests
    sed -i 's|python web_platform_|python fixed_web_platform/web_platform_|g' "$file"
    sed -i 's|python test_web_platform_|python generators/runners/web/test_web_platform_|g' "$file"
    
    # Update references to hardware detection
    sed -i 's|hardware_detection\.py|generators/hardware/hardware_detection.py|g' "$file"
    sed -i 's|automated_hardware_selection\.py|generators/hardware/automated_hardware_selection.py|g' "$file"
    
    echo "Updated $file"
}

# Find and update all markdown files in the test directory
find /home/barberb/ipfs_accelerate_py/test -name "*.md" -not -path "*/\.*" | while read file; do
    # Skip CLAUDE.md since we already updated it manually
    if [ "$file" \!= "/home/barberb/ipfs_accelerate_py/test/CLAUDE.md" ]; then
        update_markdown_file "$file"
    fi
done

echo "Documentation update complete\!"
