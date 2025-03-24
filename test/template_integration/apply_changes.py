#!/usr/bin/env python3
"""
Apply template-generated tests to the main codebase.

This script:
1. Copies regenerated test files to their final destinations
2. Updates architecture mappings in the test generator
3. Updates file permissions to ensure tests are executable
4. Generates a completion report with applied changes

Usage:
    python apply_changes.py [--dry-run] [--backup] [--force]
"""

import os
import sys
import argparse
import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"apply_changes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Define paths
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = SCRIPT_DIR.parent
SKILLS_DIR = REPO_ROOT / "skills"
TEMPLATES_DIR = SKILLS_DIR / "templates"
FINAL_MODELS_DIR = REPO_ROOT / "final_models"
FIXED_TESTS_DIR = SKILLS_DIR / "fixed_tests"

# Define model mappings for final destinations
MODEL_DESTINATIONS = {
    "layoutlmv2": SKILLS_DIR / "test_hf_layoutlmv2.py",
    "layoutlmv3": SKILLS_DIR / "test_hf_layoutlmv3.py",
    "clvp": SKILLS_DIR / "test_hf_clvp.py",
    "bigbird": SKILLS_DIR / "test_hf_bigbird.py",
    "seamless_m4t_v2": SKILLS_DIR / "test_hf_seamless_m4t_v2.py",
    "xlm_prophetnet": SKILLS_DIR / "test_hf_xlm_prophetnet.py"
}

def create_backup(file_path):
    """Create a backup of a file."""
    if os.path.exists(file_path):
        backup_path = f"{file_path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            shutil.copy2(file_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
            return backup_path
        except Exception as e:
            logger.warning(f"Failed to create backup for {file_path}: {e}")
    return None

def copy_file(source, destination, dry_run=False, backup=True, force=False):
    """Copy a file with backup and error handling."""
    try:
        # Check if source exists
        if not os.path.exists(source):
            logger.error(f"Source file not found: {source}")
            return False, f"Source file not found: {source}"
        
        # Check if destination exists and we're not forcing
        if os.path.exists(destination) and not force:
            logger.warning(f"Destination file already exists (use --force to overwrite): {destination}")
            return False, f"Destination file already exists: {destination}"
        
        # Create backup if requested
        if backup and os.path.exists(destination):
            backup_path = create_backup(destination)
            if not backup_path:
                logger.warning(f"Failed to create backup for {destination}")
        
        # Copy the file
        if not dry_run:
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copy2(source, destination)
            os.chmod(destination, 0o755)  # Make executable
            logger.info(f"Copied {source} to {destination}")
        else:
            logger.info(f"Would copy {source} to {destination}")
        
        return True, destination
    except Exception as e:
        logger.error(f"Error copying file: {e}")
        return False, f"Error copying file: {e}"

def update_architecture_types(model_name, architecture, dry_run=False, backup=True):
    """Update the ARCHITECTURE_TYPES dictionary in test_generator_fixed.py."""
    generator_path = os.path.join(SKILLS_DIR, "test_generator_fixed.py")
    
    # Check if file exists
    if not os.path.exists(generator_path):
        logger.error(f"Generator file not found: {generator_path}")
        return False, f"Generator file not found: {generator_path}"
    
    # Create backup if requested
    if backup:
        backup_path = create_backup(generator_path)
        if not backup_path:
            logger.warning(f"Failed to create backup for {generator_path}")
    
    try:
        # Read the file
        with open(generator_path, 'r') as f:
            content = f.read()
        
        # Find the ARCHITECTURE_TYPES dictionary
        import re
        arch_types_start = content.find("ARCHITECTURE_TYPES = {")
        if arch_types_start == -1:
            logger.error("ARCHITECTURE_TYPES not found in generator file")
            return False, "ARCHITECTURE_TYPES not found in generator file"
        
        # Find the specific architecture type section
        arch_type_quoted = f'"{architecture}"'
        arch_pattern = rf'{arch_type_quoted}:\s*\['
        match = re.search(arch_pattern, content)
        if not match:
            logger.error(f"Architecture type '{architecture}' not found in ARCHITECTURE_TYPES")
            return False, f"Architecture type '{architecture}' not found in ARCHITECTURE_TYPES"
        
        # Get the start and end of the architecture list
        list_start_pos = content.find('[', match.start())
        list_end_pos = content.find(']', list_start_pos)
        if list_start_pos == -1 or list_end_pos == -1:
            logger.error(f"Could not find list bounds for architecture '{architecture}'")
            return False, f"Could not find list bounds for architecture '{architecture}'"
        
        # Check if model is already in the list
        architecture_list = content[list_start_pos:list_end_pos]
        model_pattern = rf'"{model_name}"'
        if re.search(model_pattern, architecture_list):
            logger.info(f"Model '{model_name}' is already in the list for architecture '{architecture}'")
            return True, "Model already in architecture list"
        
        # Add the model to the list
        comma = "," if architecture_list.strip() != "[" else ""
        new_content = content[:list_end_pos] + f'{comma} "{model_name}"' + content[list_end_pos:]
        
        # Write the updated content
        if not dry_run:
            with open(generator_path, 'w') as f:
                f.write(new_content)
            logger.info(f"Updated ARCHITECTURE_TYPES with model '{model_name}' in architecture '{architecture}'")
        else:
            logger.info(f"Would update ARCHITECTURE_TYPES with model '{model_name}' in architecture '{architecture}'")
        
        return True, "Architecture types updated"
    
    except Exception as e:
        logger.error(f"Error updating ARCHITECTURE_TYPES: {e}")
        return False, f"Error updating ARCHITECTURE_TYPES: {e}"

def set_executable_permissions(file_path, dry_run=False):
    """Set executable permissions on a file."""
    try:
        if not os.path.exists(file_path):
            logger.warning(f"File not found for permission setting: {file_path}")
            return False
        
        if not dry_run:
            os.chmod(file_path, 0o755)
            logger.info(f"Set executable permissions on {file_path}")
        else:
            logger.info(f"Would set executable permissions on {file_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error setting permissions: {e}")
        return False

def apply_all_changes(dry_run=False, backup=True, force=False):
    """Apply all changes to the main codebase."""
    results = {
        "copied_files": [],
        "updated_architectures": [],
        "failed_operations": []
    }
    
    # Copy regenerated test files to their destinations
    for model_name, destination in MODEL_DESTINATIONS.items():
        source = FIXED_TESTS_DIR / f"test_hf_{model_name}.py"
        
        if not os.path.exists(source):
            logger.warning(f"Regenerated test file not found for {model_name}: {source}")
            results["failed_operations"].append((f"copy_{model_name}", f"Source file not found: {source}"))
            continue
        
        success, message = copy_file(source, destination, dry_run, backup, force)
        if success:
            results["copied_files"].append((model_name, str(destination)))
        else:
            results["failed_operations"].append((f"copy_{model_name}", message))
    
    # Get architecture mappings
    model_architectures = {
        "layoutlmv2": "vision-encoder-text-decoder",
        "layoutlmv3": "vision-encoder-text-decoder",
        "clvp": "speech",
        "bigbird": "encoder-decoder",
        "seamless_m4t_v2": "speech",
        "xlm_prophetnet": "encoder-decoder"
    }
    
    # Update architecture types
    for model_name, architecture in model_architectures.items():
        success, message = update_architecture_types(model_name, architecture, dry_run, backup)
        if success:
            results["updated_architectures"].append((model_name, architecture))
        else:
            results["failed_operations"].append((f"update_arch_{model_name}", message))
    
    # Set executable permissions on all copied files
    for model_name, destination in results["copied_files"]:
        success = set_executable_permissions(destination, dry_run)
        if not success:
            results["failed_operations"].append((f"permissions_{model_name}", f"Failed to set permissions on {destination}"))
    
    return results

def generate_completion_report(results, dry_run=False):
    """Generate a report of the applied changes."""
    report = []
    
    report.append("# Template Changes Application Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    if dry_run:
        report.append("**DRY RUN MODE** - No changes were actually applied")
        report.append("")
    
    # Overall status
    if results["failed_operations"]:
        report.append("## ⚠️ PARTIAL APPLICATION")
        report.append(f"Some operations failed ({len(results['failed_operations'])} failures)")
    else:
        report.append("## ✅ SUCCESSFUL APPLICATION")
        report.append("All changes were applied successfully")
    
    report.append("")
    
    # Copied files
    report.append("## Files Copied")
    report.append("")
    if results["copied_files"]:
        report.append("| Model | Destination |")
        report.append("|-------|------------|")
        for model_name, destination in results["copied_files"]:
            report.append(f"| {model_name} | {destination} |")
    else:
        report.append("No files were copied")
    
    report.append("")
    
    # Updated architectures
    report.append("## Architecture Mappings Updated")
    report.append("")
    if results["updated_architectures"]:
        report.append("| Model | Architecture |")
        report.append("|-------|-------------|")
        for model_name, architecture in results["updated_architectures"]:
            report.append(f"| {model_name} | {architecture} |")
    else:
        report.append("No architecture mappings were updated")
    
    report.append("")
    
    # Failed operations
    if results["failed_operations"]:
        report.append("## Failed Operations")
        report.append("")
        report.append("| Operation | Error |")
        report.append("|-----------|-------|")
        for operation, error in results["failed_operations"]:
            report.append(f"| {operation} | {error} |")
        report.append("")
    
    # Verification steps
    report.append("## Verification Steps")
    report.append("")
    report.append("After applying these changes, you should:")
    report.append("")
    report.append("1. Run the syntax check on all modified files:")
    report.append("   ```bash")
    for model_name, _ in MODEL_DESTINATIONS.items():
        report.append(f"   python -m py_compile skills/test_hf_{model_name}.py")
    report.append("   ```")
    report.append("")
    report.append("2. Run each test with the --help flag to verify basic functionality:")
    report.append("   ```bash")
    for model_name, _ in MODEL_DESTINATIONS.items():
        report.append(f"   python skills/test_hf_{model_name}.py --help")
    report.append("   ```")
    report.append("")
    report.append("3. Verify architecture mappings in the generator:")
    report.append("   ```bash")
    report.append("   grep -A 3 'ARCHITECTURE_TYPES = {' skills/test_generator_fixed.py")
    report.append("   ```")
    
    # Save the report
    report_path = os.path.join(SCRIPT_DIR, "changes_application_report.md")
    with open(report_path, 'w') as f:
        f.write("\n".join(report))
    
    logger.info(f"Application report saved to {report_path}")
    return report_path

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Apply template-generated tests to the main codebase")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--backup", action="store_true", help="Create backups of modified files")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files without confirmation")
    
    args = parser.parse_args()
    
    logger.info("Applying template changes to main codebase...")
    
    # Apply all changes
    results = apply_all_changes(
        dry_run=args.dry_run,
        backup=args.backup,
        force=args.force
    )
    
    # Generate completion report
    report_path = generate_completion_report(results, args.dry_run)
    
    logger.info(f"Application report saved to {report_path}")
    
    # Return success if no failures or dry run
    if not results["failed_operations"] or args.dry_run:
        return 0
    else:
        logger.error(f"Some operations failed ({len(results['failed_operations'])} failures)")
        return 1

if __name__ == "__main__":
    sys.exit(main())