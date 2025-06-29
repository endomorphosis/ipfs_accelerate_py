#!/bin/bash
# Install CI Integration
#
# This script installs all CI integration files to their proper locations,
# including GitHub Actions workflows, setup scripts, and makes them executable
#
# Usage:
#   ./install_ci_integration.sh [--dry-run]
#
# Options:
#   --dry-run  Show what would be done without making any changes
#   --help     Display this help message
#
# Date: May 2025

set -e

# Default values
DRY_RUN=false

# Process arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --help)
      echo "Install CI Integration"
      echo ""
      echo "Usage:"
      echo "  ./install_ci_integration.sh [--dry-run]"
      echo ""
      echo "Options:"
      echo "  --dry-run  Show what would be done without making any changes"
      echo "  --help     Display this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Helper function
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

dry_run_prefix=""
if [ "$DRY_RUN" = true ]; then
  dry_run_prefix="[DRY RUN] "
  log "Running in dry run mode - no changes will be made"
fi

# Find script directory and repository root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( dirname "$SCRIPT_DIR" )"

log "Repository root: $REPO_ROOT"
cd "$REPO_ROOT"

# Ensure GitHub Actions directory exists
GITHUB_DIR="$REPO_ROOT/.github"
WORKFLOWS_DIR="$GITHUB_DIR/workflows"

if [ ! -d "$GITHUB_DIR" ]; then
  log "${dry_run_prefix}Creating .github directory"
  if [ "$DRY_RUN" = false ]; then
    mkdir -p "$GITHUB_DIR"
  fi
fi

if [ ! -d "$WORKFLOWS_DIR" ]; then
  log "${dry_run_prefix}Creating .github/workflows directory"
  if [ "$DRY_RUN" = false ]; then
    mkdir -p "$WORKFLOWS_DIR"
  fi
fi

# Install GitHub Actions workflows
log "Installing GitHub Actions workflows..."

# Define workflows to install
declare -A WORKFLOWS=(
  ["android_mobile_ci.yml"]="test/android_test_harness/android_ci_workflow.yml"
  ["ios_mobile_ci.yml"]="test/ios_test_harness/ios_ci_workflow.yml"
  ["mobile_cross_platform_workflow.yml"]="test/mobile_cross_platform_workflow.yml"
  ["setup_mobile_ci_runners.yml"]="test/setup_mobile_ci_runners_workflow.yml"
)

for target_file in "${!WORKFLOWS[@]}"; do
  source_file="${WORKFLOWS[$target_file]}"
  target_path="$WORKFLOWS_DIR/$target_file"
  
  log "${dry_run_prefix}Installing workflow: $source_file -> $target_path"
  
  if [ "$DRY_RUN" = false ]; then
    if [ -f "$source_file" ]; then
      # Create backup if target exists
      if [ -f "$target_path" ]; then
        backup_file="${target_path}.bak.$(date '+%Y%m%d%H%M%S')"
        log "Creating backup: $target_path -> $backup_file"
        cp "$target_path" "$backup_file"
      fi
      
      # Copy file
      cp "$source_file" "$target_path"
      log "✅ Installed: $target_path"
    else
      log "⚠️ Source file not found: $source_file"
    fi
  fi
done

# Make setup scripts executable
log "Making setup scripts executable..."

SETUP_SCRIPTS=(
  "test/setup_android_ci_runner.sh"
  "test/setup_ios_ci_runner.sh"
  "test/install_ci_integration.sh"
)

for script in "${SETUP_SCRIPTS[@]}"; do
  if [ -f "$script" ]; then
    log "${dry_run_prefix}Making executable: $script"
    if [ "$DRY_RUN" = false ]; then
      chmod +x "$script"
    fi
  else
    log "⚠️ Script not found: $script"
  fi
done

# Use the setup_ci_workflows.py script for additional verification
log "Running CI workflow setup script..."
if [ "$DRY_RUN" = false ]; then
  python3 test/setup_ci_workflows.py --verify --verbose
  python3 test/setup_ci_workflows.py --install --verbose
else
  python3 test/setup_ci_workflows.py --verify --verbose
  python3 test/setup_ci_workflows.py --install --dry-run --verbose
fi

log "CI integration installation ${dry_run_prefix}complete!"
log "Next steps:"
log "1. Push the changes to GitHub"
log "2. Set up self-hosted runners using the setup scripts:"
log "   - For Android: test/setup_android_ci_runner.sh"
log "   - For iOS: test/setup_ios_ci_runner.sh"
log "3. Check the Mobile CI Runner Setup Guide: test/MOBILE_CI_RUNNER_SETUP_GUIDE.md"