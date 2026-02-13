#!/bin/bash
# Script to regenerate tests for specific model families
# This helps quickly regenerate tests after changes to the test generator

# Colors
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
BLUE="\033[0;34m"
PURPLE="\033[0;35m"
CYAN="\033[0;36m"
NC="\033[0m" # No Color

# Directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
parent_dir=$(dirname "$SCRIPT_DIR")

# Generator script
GENERATOR_SCRIPT="$parent_dir/test_generator.py"

# Temp directory for generated tests
TEMP_DIR="$SCRIPT_DIR/temp_generated"
mkdir -p "$TEMP_DIR"

# List of model families
MODEL_FAMILIES=(
  "bert"
  "gpt2"
  "t5"
  "vit"
  "llama"
  "mistral"
  "falcon"
  "phi"
  "clip"
  "whisper"
  "wav2vec2"
  "sam"
  "roberta"
  "albert"
  "bart"
)

# Function to regenerate test for a specific family
regenerate_test() {
  local family=$1
  local dest_dir=$2
  local backup=${3:-true}
  
  echo -e "${BLUE}Regenerating test for ${YELLOW}$family${BLUE}...${NC}"
  
  # Generate test
  python "$GENERATOR_SCRIPT" --family "$family" --output "$TEMP_DIR"
  
  # Check if generated successfully
  if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to generate test for $family${NC}"
    return 1
  fi
  
  # Check syntax
  python -m py_compile "$TEMP_DIR/test_hf_$family.py"
  if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Syntax check failed for $family${NC}"
    return 1
  fi
  
  # Backup existing file if requested
  if [ "$backup" = true ] && [ -f "$dest_dir/test_hf_$family.py" ]; then
    backup_file="$dest_dir/test_hf_$family.py.bak.$(date +%Y%m%d_%H%M%S)"
    cp "$dest_dir/test_hf_$family.py" "$backup_file"
    echo -e "${PURPLE}📦 Created backup: $backup_file${NC}"
  fi
  
  # Deploy to destination
  cp "$TEMP_DIR/test_hf_$family.py" "$dest_dir/test_hf_$family.py"
  
  echo -e "${GREEN}✅ Successfully regenerated test for $family${NC}"
  return 0
}

# Function to show usage
show_usage() {
  echo -e "${CYAN}Usage: $0 [options] COMMAND${NC}"
  echo
  echo "Commands:"
  echo "  all                   Regenerate all model family tests"
  echo "  core                  Regenerate tests for core families (bert, gpt2, t5, vit)"
  echo "  model_family          Regenerate test for a specific model family"
  echo "  list                  List available model families"
  echo
  echo "Options:"
  echo "  --no-backup           Don't create backups of existing files"
  echo "  --dest-dir DIR        Destination directory (default: parent directory)"
  echo "  --help                Show this help message"
  echo
  echo "Examples:"
  echo "  $0 bert               # Regenerate test for BERT"
  echo "  $0 core               # Regenerate tests for core families"
  echo "  $0 all                # Regenerate all tests"
  echo "  $0 --no-backup all    # Regenerate all tests without backups"
}

# Parse arguments
BACKUP=true
DEST_DIR="$parent_dir"

while [[ $# -gt 0 ]]; do
  case $1 in
    --no-backup)
      BACKUP=false
      shift
      ;;
    --dest-dir)
      DEST_DIR="$2"
      shift 2
      ;;
    --help)
      show_usage
      exit 0
      ;;
    list)
      echo -e "${CYAN}Available model families:${NC}"
      for family in "${MODEL_FAMILIES[@]}"; do
        echo "  $family"
      done
      exit 0
      ;;
    all)
      echo -e "${CYAN}Regenerating tests for all model families...${NC}"
      
      # Count successful generations
      success_count=0
      fail_count=0
      
      for family in "${MODEL_FAMILIES[@]}"; do
        regenerate_test "$family" "$DEST_DIR" "$BACKUP"
        if [ $? -eq 0 ]; then
          ((success_count++))
        else
          ((fail_count++))
        fi
      done
      
      echo
      echo -e "${CYAN}=== Regeneration Summary ===${NC}"
      echo -e "${GREEN}✅ Successfully regenerated: $success_count${NC}"
      echo -e "${RED}❌ Failed: $fail_count${NC}"
      exit 0
      ;;
    core)
      echo -e "${CYAN}Regenerating tests for core model families...${NC}"
      
      # Core families
      core_families=("bert" "gpt2" "t5" "vit")
      
      # Count successful generations
      success_count=0
      fail_count=0
      
      for family in "${core_families[@]}"; do
        regenerate_test "$family" "$DEST_DIR" "$BACKUP"
        if [ $? -eq 0 ]; then
          ((success_count++))
        else
          ((fail_count++))
        fi
      done
      
      echo
      echo -e "${CYAN}=== Regeneration Summary ===${NC}"
      echo -e "${GREEN}✅ Successfully regenerated: $success_count${NC}"
      echo -e "${RED}❌ Failed: $fail_count${NC}"
      exit 0
      ;;
    *)
      # Check if it's a valid model family
      valid_family=false
      for family in "${MODEL_FAMILIES[@]}"; do
        if [ "$family" = "$1" ]; then
          valid_family=true
          break
        fi
      done
      
      if [ "$valid_family" = true ]; then
        regenerate_test "$1" "$DEST_DIR" "$BACKUP"
        exit $?
      else
        echo -e "${RED}❌ Unknown model family or command: $1${NC}"
        echo "Use '$0 list' to see available families"
        exit 1
      fi
      ;;
  esac
done

# If we get here, no command was provided
show_usage
exit 1