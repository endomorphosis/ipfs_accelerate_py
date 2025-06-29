#!/bin/bash

# Create archive directories if they don't exist
mkdir -p archive/backup_files
mkdir -p archive/old_reports

# Move .bak files with directory structure preserved
find . -name "*.bak" -not -path "./archive/*" -type f | while read file; do
  # Get the relative directory structure
  dir=$(dirname "$file" | sed 's|^\./||')
  
  # Create the directory in archive/backup_files if it doesn't exist
  mkdir -p "archive/backup_files/$dir"
  
  # Move the file
  mv "$file" "archive/backup_files/$dir/"
done

# Move old report files
find . -name "*report_*.md" -not -path "./archive/*" -type f | while read file; do
  # Get the relative directory structure
  dir=$(dirname "$file" | sed 's|^\./||')
  
  # Create the directory in archive/old_reports if it doesn't exist
  mkdir -p "archive/old_reports/$dir"
  
  # Move the file
  mv "$file" "archive/old_reports/$dir/"
done

# Move JSON benchmark results files
find . -name "*results*.json" -not -path "./archive/*" -type f | while read file; do
  # Get the relative directory structure
  dir=$(dirname "$file" | sed 's|^\./||')
  
  # Create the directory in archive/old_reports if it doesn't exist
  mkdir -p "archive/old_reports/$dir"
  
  # Move the file
  mv "$file" "archive/old_reports/$dir/"
done

# Move HTML reports
find . -name "*report*.html" -not -path "./archive/*" -type f | while read file; do
  # Get the relative directory structure
  dir=$(dirname "$file" | sed 's|^\./||')
  
  # Create the directory in archive/old_reports if it doesn't exist
  mkdir -p "archive/old_reports/$dir"
  
  # Move the file
  mv "$file" "archive/old_reports/$dir/"
done

echo "Files moved to archive directory"