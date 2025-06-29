#!/bin/bash

# Create a backup of CLAUDE.md
cp CLAUDE.md CLAUDE.md.bak

# List of files to archive (outdated or superseded)
FILES_TO_ARCHIVE=(
  "API_IMPLEMENTATION_PLAN.md"
  "API_IMPLEMENTATION_PLAN_UPDATED.md"
  "API_IMPLEMENTATION_SUMMARY.md"
  "API_IMPLEMENTATION_SUMMARY_UPDATED.md"
  "README_API_COMPLETION.md"
  "README_API_ENHANCEMENTS.md"
  "API_IMPROVEMENTS_SUMMARY.md"
  "TEMPLATE_GENERATOR_IMPROVEMENTS.md"
  "MERGED_GENERATOR_QUICK_REFERENCE.md"
  "QUEUE_BACKOFF_GUIDE.md"
  "TEMPLATE_SYSTEM_README.md"
  "TEST_GENERATOR_README.md"
  "RECOMMENDATION_FOR_TEST_GENERATOR.md"
  "API_CONFIGURATION_REFERENCE.md" 
  "ADVANCED_API_FEATURES_GUIDE.md"
  "API_TESTING_README.md"
  "API_QUICKSTART.md"
  "API_ERROR_DOCUMENTATION.md"
  "API_ENHANCEMENT_README.md"
)

# Create the archive directory if it doesn't exist
mkdir -p archived_md_files

# Loop through each file and move it to the archive directory
for file in "${FILES_TO_ARCHIVE[@]}"; do
  if [ -f "$file" ]; then
    echo "Archiving $file"
    mv "$file" archived_md_files/
  else
    echo "File $file not found, skipping"
  fi
done

echo "Archiving complete. Outdated files moved to archived_md_files/"