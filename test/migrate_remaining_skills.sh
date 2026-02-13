#\!/bin/bash
# Create necessary directories if they don't exist
mkdir -p ../generators/models/skills/

# Copy skill files
for skill_file in skill_hf_*.py; do
  if [ -f "$skill_file" ]; then
    echo "Migrating $skill_file to ../generators/models/skills/"
    cp "$skill_file" ../generators/models/skills/
  fi
done

# Update migration progress
echo -e "\n### Skills Migration Progress - $(date)" >> migration_progress.md
echo "Migrated the following skill files:" >> migration_progress.md
for skill_file in ../generators/models/skills/skill_hf_*.py; do
  if [ -f "$skill_file" ]; then
    echo "- $(basename $skill_file)" >> migration_progress.md
  fi
done
