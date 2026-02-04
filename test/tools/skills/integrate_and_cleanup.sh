#!/bin/bash
# Complete integration and cleanup script for the fixed HuggingFace tests
# This script:
# 1. Tests the fixed files to ensure they work properly
# 2. Deploys them to the main project
# 3. Tests the deployed files
# 4. Cleans up temporary files if everything is successful

# Configuration
FIXED_TESTS_DIR="fixed_tests"
DEST_DIR=".."
BACKUP_DIR="backups"
LOG_FILE="integration_$(date +%Y%m%d_%H%M%S).log"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Colorized output
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
NC="\033[0m" # No Color

# Helper function for logging
log() {
  echo -e "${GREEN}$(date '+%Y-%m-%d %H:%M:%S') - $1${NC}" | tee -a "$LOG_FILE"
}

# Helper function for error messages
error() {
  echo -e "${RED}$(date '+%Y-%m-%d %H:%M:%S') - ERROR: $1${NC}" | tee -a "$LOG_FILE"
}

# Helper function for warnings
warn() {
  echo -e "${YELLOW}$(date '+%Y-%m-%d %H:%M:%S') - WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

# Check for python
if ! command -v python3 &> /dev/null; then
  error "Python 3 is required but not found. Please install Python 3."
  exit 1
fi

# Display banner
log "==========================================="
log "Fixed HuggingFace Test Files Integration"
log "==========================================="
log "Starting integration process..."

# Parse command-line arguments
DEPLOY=0
CLEANUP=0
TEST_ONLY=0
FORCE=0
CPU_ONLY=1

while [[ $# -gt 0 ]]; do
  case $1 in
    --deploy)
      DEPLOY=1
      shift
      ;;
    --cleanup)
      CLEANUP=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --test-only)
      TEST_ONLY=1
      shift
      ;;
    --with-gpu)
      CPU_ONLY=0
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --deploy       Deploy files to destination directory"
      echo "  --cleanup      Clean up temporary files after successful integration"
      echo "  --force        Force operation even if tests fail"
      echo "  --test-only    Test integration without copying files"
      echo "  --with-gpu     Use GPU for testing (default is CPU only)"
      echo "  --help         Show this help message"
      exit 0
      ;;
    *)
      error "Unknown option: $1"
      echo "Use --help to see available options"
      exit 1
      ;;
  esac
done

# Step 1: Test syntax and functionality of fixed files
log "Step 1: Testing fixed files..."

INTEGRATION_CMD="python run_integration.py --fixed-tests-dir $FIXED_TESTS_DIR --dest-dir $DEST_DIR --backup-dir $BACKUP_DIR --save-results"

if [ $CPU_ONLY -eq 1 ]; then
  INTEGRATION_CMD="$INTEGRATION_CMD --cpu-only"
fi

if [ $TEST_ONLY -eq 1 ]; then
  INTEGRATION_CMD="$INTEGRATION_CMD --test-only"
fi

if [ $FORCE -eq 1 ]; then
  INTEGRATION_CMD="$INTEGRATION_CMD --force"
fi

if [ $DEPLOY -eq 1 ]; then
  INTEGRATION_CMD="$INTEGRATION_CMD --deploy"
fi

log "Running: $INTEGRATION_CMD"
if ! eval "$INTEGRATION_CMD"; then
  error "Integration test failed. See log for details."
  if [ $FORCE -ne 1 ]; then
    exit 1
  else
    warn "Continuing despite errors due to --force flag."
  fi
fi

# If cleanup is requested and we're either not deploying or doing test-only
if [ $CLEANUP -eq 1 ] && ([ $DEPLOY -eq 0 ] || [ $TEST_ONLY -eq 1 ]); then
  error "Cleanup requested but files were not deployed (--deploy was not specified or --test-only was specified)."
  error "Skipping cleanup to prevent potential data loss."
  exit 1
fi

# Step 2: Run cleanup if requested and deployment was successful
if [ $CLEANUP -eq 1 ] && [ $DEPLOY -eq 1 ] && [ $TEST_ONLY -eq 0 ]; then
  log "Step 2: Cleaning up temporary files..."

  CLEANUP_CMD="python cleanup_integration.py --backup-dir $BACKUP_DIR"
  if [ $FORCE -eq 1 ]; then
    CLEANUP_CMD="$CLEANUP_CMD --force"
  fi

  # Ask for confirmation unless force is specified
  if [ $FORCE -ne 1 ]; then
    read -p "This will clean up all temporary files. Are you sure? (y/N): " confirm
    if [[ $confirm != [yY] ]]; then
      log "Cleanup aborted by user."
      exit 0
    fi
  fi

  log "Running: $CLEANUP_CMD"
  if ! eval "$CLEANUP_CMD"; then
    error "Cleanup failed. See log for details."
    exit 1
  fi
fi

log "Integration process completed successfully!"
if [ $DEPLOY -eq 1 ] && [ $TEST_ONLY -eq 0 ]; then
  log "Files have been deployed to $DEST_DIR"
  if [ $CLEANUP -eq 1 ]; then
    log "Temporary files have been cleaned up"
  else
    log "You can run cleanup_integration.py to remove temporary files"
  fi
else
  log "This was a test run. No files were deployed."
  log "Use --deploy to actually deploy the files"
fi

log "==========================================="
exit 0