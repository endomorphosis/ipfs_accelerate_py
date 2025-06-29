#!/usr/bin/env python3
"""
Comprehensive toolkit for managing HuggingFace model tests.
This toolkit provides a unified interface for all test-related operations.
"""

import os
import sys
import json
import glob
import argparse
import subprocess
import logging
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"toolkit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
SKILLS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SKILLS_DIR)
TEMP_DIR = os.path.join(SKILLS_DIR, "temp_generated")
BACKUP_DIR = os.path.join(SKILLS_DIR, "backups")
COVERAGE_DIR = os.path.join(SKILLS_DIR, "coverage_visualizations")
TOOLS = {
    "generator": os.path.join(PARENT_DIR, "test_generator.py"),
    "test_suite": os.path.join(SKILLS_DIR, "test_generator_test_suite.py"),
    "coverage": os.path.join(SKILLS_DIR, "visualize_test_coverage.py"),
    "batch_generation": os.path.join(SKILLS_DIR, "generate_batch_tests.py"),
    "regenerate_script": os.path.join(SKILLS_DIR, "regenerate_model_tests.sh"),
    "pre_commit": os.path.join(SKILLS_DIR, "pre-commit"),
    "install_script": os.path.join(SKILLS_DIR, "install_pre_commit.sh")
}

# Template models for each architecture
ARCHITECTURE_TEMPLATES = {
    "encoder_only": "bert",
    "decoder_only": "gpt2",
    "encoder_decoder": "t5",
    "vision": "vit",
    "multimodal": "clip",
    "audio": "wav2vec2"
}

# Common model families
MODEL_FAMILIES = [
    "bert", "gpt2", "t5", "vit", "roberta", "llama", "bart", "mistral",
    "falcon", "phi", "clip", "blip", "whisper", "wav2vec2"
]

def verify_tools():
    """Verify that all required tools exist."""
    missing_tools = []
    for name, path in TOOLS.items():
        if not os.path.exists(path):
            missing_tools.append((name, path))
    
    if missing_tools:
        logger.error("Some required tools are missing:")
        for name, path in missing_tools:
            logger.error(f"  - {name}: {path}")
        return False
    
    return True

def run_generator(family, output_dir=PARENT_DIR, template=None):
    """Run the test generator for a specific model family."""
    if not os.path.exists(TOOLS["generator"]):
        logger.error(f"Generator not found at {TOOLS['generator']}")
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    
    command = [
        sys.executable,
        TOOLS["generator"],
        "--family", family,
        "--output", output_dir
    ]
    
    if template:
        command.extend(["--template", template])
    
    logger.info(f"Generating test for {family}...")
    
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Generator failed for {family}: {result.stderr}")
            return False
        
        # Verify the file exists
        output_path = os.path.join(output_dir, f"test_hf_{family}.py")
        if not os.path.exists(output_path):
            logger.error(f"Generated file not found: {output_path}")
            return False
        
        # Check syntax
        syntax_check = subprocess.run(
            [sys.executable, "-m", "py_compile", output_path],
            capture_output=True,
            text=True
        )
        
        if syntax_check.returncode != 0:
            logger.error(f"Syntax check failed for {family}: {syntax_check.stderr}")
            return False
        
        logger.info(f"Successfully generated test for {family}: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Exception running generator: {e}")
        return False

def run_test_suite():
    """Run the test generator test suite."""
    if not os.path.exists(TOOLS["test_suite"]):
        logger.error(f"Test suite not found at {TOOLS['test_suite']}")
        return False
    
    try:
        logger.info("Running test generator test suite...")
        result = subprocess.run(
            [sys.executable, TOOLS["test_suite"]],
            capture_output=True,
            text=True
        )
        
        # Log output regardless of success/failure
        for line in result.stdout.splitlines():
            logger.info(line)
        
        if result.returncode != 0:
            logger.error("Test suite failed")
            for line in result.stderr.splitlines():
                logger.error(line)
            return False
        
        logger.info("Test suite passed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Exception running test suite: {e}")
        return False

def generate_coverage_report():
    """Generate the test coverage report."""
    if not os.path.exists(TOOLS["coverage"]):
        logger.error(f"Coverage tool not found at {TOOLS['coverage']}")
        return False
    
    try:
        logger.info("Generating coverage report...")
        os.makedirs(COVERAGE_DIR, exist_ok=True)
        
        result = subprocess.run(
            [sys.executable, TOOLS["coverage"], "--output-dir", COVERAGE_DIR],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Coverage generation failed: {result.stderr}")
            return False
        
        # Log some output from the coverage report
        summary_path = os.path.join(COVERAGE_DIR, "coverage_summary.md")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                lines = f.readlines()
                for line in lines[:15]:  # Show first 15 lines
                    logger.info(line.strip())
        
        logger.info(f"Coverage report generated in {COVERAGE_DIR}")
        return True
    
    except Exception as e:
        logger.error(f"Exception generating coverage report: {e}")
        return False

def batch_generate_tests(batch_size=10, all_models=False):
    """Generate tests for multiple models in a batch."""
    if not os.path.exists(TOOLS["batch_generation"]):
        logger.error(f"Batch generation tool not found at {TOOLS['batch_generation']}")
        return False
    
    try:
        logger.info(f"Running batch generation with batch size {batch_size}...")
        
        command = [
            sys.executable,
            TOOLS["batch_generation"],
            "--batch-size", str(batch_size),
            "--output-dir", PARENT_DIR,
            "--temp-dir", TEMP_DIR
        ]
        
        if all_models:
            command.append("--all")
        
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Log output regardless of success/failure
        for line in result.stdout.splitlines():
            logger.info(line)
        
        if result.returncode != 0:
            logger.error("Batch generation failed")
            for line in result.stderr.splitlines():
                logger.error(line)
            return False
        
        logger.info("Batch generation completed")
        return True
    
    except Exception as e:
        logger.error(f"Exception during batch generation: {e}")
        return False

def test_model(model_family, model_id=None, cpu_only=True):
    """Run a specific model test."""
    test_file = os.path.join(PARENT_DIR, f"test_hf_{model_family}.py")
    
    if not os.path.exists(test_file):
        logger.error(f"Test file not found: {test_file}")
        return False
    
    try:
        logger.info(f"Testing model {model_family}...")
        
        command = [sys.executable, test_file]
        
        if model_id:
            command.extend(["--model", model_id])
        
        if cpu_only:
            command.append("--cpu-only")
        
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Extract success indicator from output
        success = "Successfully tested" in result.stdout
        
        # Log output
        for line in result.stdout.splitlines():
            if success:
                logger.info(line)
            else:
                logger.warning(line)
        
        if not success or result.returncode != 0:
            logger.error(f"Test failed for {model_family}")
            for line in result.stderr.splitlines():
                logger.error(line)
            return False
        
        logger.info(f"Test passed for {model_family}")
        return True
    
    except Exception as e:
        logger.error(f"Exception testing model: {e}")
        return False

def install_pre_commit():
    """Install the pre-commit hook."""
    if not os.path.exists(TOOLS["install_script"]):
        logger.error(f"Install script not found at {TOOLS['install_script']}")
        return False
    
    try:
        logger.info("Installing pre-commit hook...")
        
        result = subprocess.run(
            [TOOLS["install_script"]],
            capture_output=True,
            text=True
        )
        
        # Log output
        for line in result.stdout.splitlines():
            logger.info(line)
        
        if result.returncode != 0:
            logger.error("Pre-commit hook installation failed")
            for line in result.stderr.splitlines():
                logger.error(line)
            return False
        
        logger.info("Pre-commit hook installed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Exception installing pre-commit hook: {e}")
        return False

def regenerate_core_models(backup=True):
    """Regenerate tests for core model families."""
    if not os.path.exists(TOOLS["regenerate_script"]):
        logger.error(f"Regeneration script not found at {TOOLS['regenerate_script']}")
        return False
    
    try:
        logger.info("Regenerating core model tests...")
        
        command = [TOOLS["regenerate_script"], "core"]
        if not backup:
            command.insert(1, "--no-backup")
        
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Log output
        for line in result.stdout.splitlines():
            logger.info(line)
        
        if result.returncode != 0:
            logger.error("Core model regeneration failed")
            for line in result.stderr.splitlines():
                logger.error(line)
            return False
        
        logger.info("Core models regenerated successfully")
        return True
    
    except Exception as e:
        logger.error(f"Exception regenerating core models: {e}")
        return False

def run_all_tests(families=None, cpu_only=True):
    """Run tests for multiple model families."""
    if families is None:
        # Default to core families
        families = ["bert", "gpt2", "t5", "vit"]
    
    logger.info(f"Running tests for {len(families)} model families...")
    
    results = {}
    success_count = 0
    
    for family in families:
        logger.info(f"Testing {family}...")
        success = test_model(family, cpu_only=cpu_only)
        results[family] = success
        
        if success:
            success_count += 1
    
    # Log summary
    logger.info("\nTest Results Summary:")
    logger.info(f"Total: {len(families)}, Successful: {success_count}, Failed: {len(families) - success_count}")
    
    for family, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} - {family}")
    
    return success_count == len(families)

def verify_syntax_all():
    """Verify syntax for all test files."""
    test_files = glob.glob(os.path.join(PARENT_DIR, "test_hf_*.py"))
    
    if not test_files:
        logger.warning("No test files found!")
        return False
    
    logger.info(f"Verifying syntax for {len(test_files)} test files...")
    
    all_valid = True
    
    for test_file in test_files:
        file_name = os.path.basename(test_file)
        logger.info(f"Checking syntax: {file_name}")
        
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", test_file],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"❌ {file_name} - Syntax error: {result.stderr}")
            all_valid = False
        else:
            logger.info(f"✅ {file_name} - Valid syntax")
    
    if all_valid:
        logger.info("All test files have valid syntax")
    else:
        logger.error("Some test files have syntax errors")
    
    return all_valid

def print_help():
    """Print help information."""
    help_text = """
🔧 HuggingFace Test Toolkit 🔧

This toolkit provides a unified interface for managing HuggingFace model tests.

Commands:
  generate FAMILY       Generate a test for a specific model family
  test FAMILY           Run a test for a specific model family
  suite                 Run the test generator test suite
  coverage              Generate test coverage report
  batch BATCH_SIZE      Generate tests for multiple models in a batch
  all-tests             Run tests for all core model families
  verify                Verify syntax for all test files
  install-hook          Install the pre-commit hook
  regenerate            Regenerate tests for core model families
  help                  Show this help message

Options:
  --output DIR          Output directory for generated files
  --template MODEL      Template model for generation
  --model-id ID         Specific model ID for testing
  --cpu-only            Use CPU only for testing (default: true)
  --no-backup           Don't create backups when regenerating
  --all                 Process all models when using batch

Examples:
  ./test_toolkit.py generate bert
  ./test_toolkit.py test bert --model-id bert-base-uncased
  ./test_toolkit.py batch 10
  ./test_toolkit.py verify
  ./test_toolkit.py all-tests
    """
    print(help_text)

def list_models():
    """List available model families."""
    print("\nAvailable model families:")
    for i, family in enumerate(MODEL_FAMILIES, 1):
        print(f"  {i}. {family}")
    
    print("\nArchitecture templates:")
    for arch, template in ARCHITECTURE_TEMPLATES.items():
        print(f"  - {arch}: {template}")

def main():
    """Main entry point."""
    if len(sys.argv) < 2 or sys.argv[1] == "help":
        print_help()
        return 0
    
    # Create necessary directories
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)
    os.makedirs(COVERAGE_DIR, exist_ok=True)
    
    # Verify tools
    if not verify_tools():
        logger.error("Some required tools are missing. Please check your installation.")
        return 1
    
    # Process commands
    command = sys.argv[1]
    
    if command == "list":
        list_models()
        return 0
    
    elif command == "generate":
        if len(sys.argv) < 3:
            logger.error("Please specify a model family")
            print("Usage: ./test_toolkit.py generate FAMILY [--output DIR] [--template MODEL]")
            return 1
        
        family = sys.argv[2]
        output_dir = PARENT_DIR
        template = None
        
        # Parse additional arguments
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == "--output" and i + 1 < len(sys.argv):
                output_dir = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--template" and i + 1 < len(sys.argv):
                template = sys.argv[i + 1]
                i += 2
            else:
                logger.warning(f"Ignoring unknown argument: {sys.argv[i]}")
                i += 1
        
        success = run_generator(family, output_dir, template)
        return 0 if success else 1
    
    elif command == "test":
        if len(sys.argv) < 3:
            logger.error("Please specify a model family")
            print("Usage: ./test_toolkit.py test FAMILY [--model-id ID] [--cpu-only]")
            return 1
        
        family = sys.argv[2]
        model_id = None
        cpu_only = True
        
        # Parse additional arguments
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == "--model-id" and i + 1 < len(sys.argv):
                model_id = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--cpu-only":
                cpu_only = True
                i += 1
            elif sys.argv[i] == "--no-cpu-only":
                cpu_only = False
                i += 1
            else:
                logger.warning(f"Ignoring unknown argument: {sys.argv[i]}")
                i += 1
        
        success = test_model(family, model_id, cpu_only)
        return 0 if success else 1
    
    elif command == "suite":
        success = run_test_suite()
        return 0 if success else 1
    
    elif command == "coverage":
        success = generate_coverage_report()
        return 0 if success else 1
    
    elif command == "batch":
        if len(sys.argv) < 3:
            logger.error("Please specify a batch size")
            print("Usage: ./test_toolkit.py batch BATCH_SIZE [--all]")
            return 1
        
        try:
            batch_size = int(sys.argv[2])
        except ValueError:
            logger.error("Batch size must be an integer")
            return 1
        
        all_models = "--all" in sys.argv
        
        success = batch_generate_tests(batch_size, all_models)
        return 0 if success else 1
    
    elif command == "all-tests":
        families = ["bert", "gpt2", "t5", "vit"]  # Default core families
        
        # Check for additional families
        if len(sys.argv) > 2 and sys.argv[2] != "--cpu-only" and sys.argv[2] != "--no-cpu-only":
            families = sys.argv[2].split(",")
        
        cpu_only = "--no-cpu-only" not in sys.argv
        
        success = run_all_tests(families, cpu_only)
        return 0 if success else 1
    
    elif command == "verify":
        success = verify_syntax_all()
        return 0 if success else 1
    
    elif command == "install-hook":
        success = install_pre_commit()
        return 0 if success else 1
    
    elif command == "regenerate":
        backup = "--no-backup" not in sys.argv
        success = regenerate_core_models(backup)
        return 0 if success else 1
    
    else:
        logger.error(f"Unknown command: {command}")
        print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())