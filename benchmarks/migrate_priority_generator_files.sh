#\!/bin/bash
# Create necessary directories
mkdir -p ../generators/test_generators/enhanced ../generators/templates/model_registry ../generators/skill_generators/models

# Check and migrate high-priority test files
priority_test_files=(
  "test_vit_from_template.py"
  "verify_generator_improvements.py"
  "hardware_template_integration.py"
  "fixed_merged_test_generator_backup.py"
  "validate_merged_generator.py"
  "fixed_merged_test_generator.py"
  "improved_merged_test_generator.py"
  "improved_template_generator.py"
  "comprehensive_template_generator.py"
  "minimal_test_generator.py"
  "fix_test_generator.py"
  "fix_test_generators.py"
  "generate_tests_for_all_skillset_models.py"
)

for file in "${priority_test_files[@]}"; do
  if [ -f "$file" ]; then
    echo "Migrating high-priority file: $file to ../generators/test_generators/"
    cp "$file" ../generators/test_generators/
  else
    echo "File not found: $file"
  fi
done

# Check and migrate specific model files
model_files=(
  "skill_hf_qwen2_7b.py"
  "skill_hf_detr_resnet_50.py"
  "skill_hf_vit_base_patch16_224.py" 
  "skill_hf_wav2vec2_base.py"
  "skill_hf_whisper_tiny.py"
  "skill_hf_clip_vit_base_patch32.py"
  "skill_hf_t5_small.py"
  "skill_hf_llama_7b.py"
  "skill_hf_clap_htsat_fused.py"
)

for file in "${model_files[@]}"; do
  if [ -f "$file" ]; then
    echo "Migrating model file: $file to ../generators/models/skills/"
    cp "$file" ../generators/models/skills/
  else
    echo "File not found: $file"
  fi
done

# Check and migrate template files
template_files=(
  "model_template_registry.py"
  "hf_embedding_template.py"
  "hf_bert_template.py"
  "hf_template.py"
  "add_skill_template.py"
  "update_template.py"
)

for file in "${template_files[@]}"; do
  if [ -f "$file" ]; then
    echo "Migrating template file: $file to ../generators/templates/model_registry/"
    cp "$file" ../generators/templates/model_registry/
  else
    echo "File not found: $file"
  fi
done

# Check and migrate run files
run_files=(
  "run_test.py"
  "run_whisper_test.py"
  "run_llama_test.py"
  "run_clap_test.py"
  "run_wav2vec2_test.py"
  "run_lm_test.py"
  "run_embed_test.py"
  "run_xclip_test.py"
  "run_single_test.py"
)

for file in "${run_files[@]}"; do
  if [ -f "$file" ]; then
    echo "Migrating run file: $file to ../generators/runners/models/"
    cp "$file" ../generators/runners/models/
  else
    echo "File not found: $file"
  fi
done

# Update the migration progress
echo -e "\n### Additional Generator Files Migration - $(date)" >> migration_progress.md
echo "Migrated high-priority generator files to appropriate directories" >> migration_progress.md
