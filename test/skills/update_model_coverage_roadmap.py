#!/usr/bin/env python3
"""
Script to update the HF_MODEL_COVERAGE_ROADMAP.md file to reflect the current implementation status
of HuggingFace model tests.
"""

import os
import re
import glob
from datetime import datetime

# Path to the roadmap file
ROADMAP_FILE = "/home/barberb/ipfs_accelerate_py/test/skills/HF_MODEL_COVERAGE_ROADMAP.md"
# Directory where test files are located
TEST_DIR = "/home/barberb/ipfs_accelerate_py/test/skills"

def get_implemented_models():
    """Get a list of all implemented models from test files."""
    test_files = glob.glob(os.path.join(TEST_DIR, "test_hf_*.py"))
    
    # Filter out files in subdirectories
    test_files = [f for f in test_files if "/" not in os.path.relpath(f, TEST_DIR)]
    
    # Extract model names from file names
    model_names = []
    for test_file in test_files:
        base_name = os.path.basename(test_file)
        model_name = base_name.replace("test_hf_", "").replace(".py", "")
        model_names.append(model_name)
    
    return model_names

def normalize_model_name(name):
    """Normalize model name to handle variations in naming conventions."""
    # Convert to lowercase
    name = name.lower()
    # Convert hyphens to underscores
    name = name.replace("-", "_")
    # Remove common prefixes or suffixes if present
    name = re.sub(r'^hf_', '', name)
    return name

def update_roadmap_file():
    """Update the roadmap file with current implementation status."""
    implemented_models = get_implemented_models()
    normalized_implemented = [normalize_model_name(m) for m in implemented_models]
    
    with open(ROADMAP_FILE, 'r') as f:
        lines = f.readlines()
    
    updated_lines = []
    total_models = 0
    implemented_count = 0
    
    # Pattern to match model entries in the roadmap
    model_pattern = re.compile(r'- \[([ x])\] ([a-zA-Z0-9_]+) \((.*?)\)')
    
    for line in lines:
        match = model_pattern.search(line)
        if match:
            total_models += 1
            checkbox, model_name, model_type = match.groups()
            
            normalized_name = normalize_model_name(model_name)
            
            # Check if the model is implemented
            is_implemented = normalized_name in normalized_implemented
            
            if is_implemented:
                implemented_count += 1
                updated_line = line.replace("[ ]", "[x]")
            else:
                updated_line = line
            
            updated_lines.append(updated_line)
        else:
            # Update the current status section if present
            if "## Current Status" in line:
                today = datetime.now().strftime('%B %d, %Y')
                updated_lines.append(f"## Current Status ({today})\n")
            elif "- **Total model architectures**:" in line:
                updated_lines.append(f"- **Total model architectures**: {total_models}\n")
            elif "- **Currently implemented**:" in line and total_models > 0:
                percentage = round(implemented_count / total_models * 100, 1)
                updated_lines.append(f"- **Currently implemented**: {implemented_count} ({percentage}%)\n")
            elif "- **Remaining to implement**:" in line and total_models > 0:
                remaining = total_models - implemented_count
                percentage = round(remaining / total_models * 100, 1)
                updated_lines.append(f"- **Remaining to implement**: {remaining} ({percentage}%)\n")
            else:
                updated_lines.append(line)
    
    # Write the updated content back to the file
    with open(ROADMAP_FILE, 'w') as f:
        f.writelines(updated_lines)
    
    # Return statistics for reporting
    return {
        "total_models": total_models,
        "implemented_count": implemented_count,
        "percentage": round(implemented_count / total_models * 100 if total_models > 0 else 0, 1)
    }

def main():
    """Main function to update the roadmap."""
    print("Updating HF_MODEL_COVERAGE_ROADMAP.md...")
    stats = update_roadmap_file()
    print(f"Updated roadmap with current implementation status:")
    print(f"- Total models: {stats['total_models']}")
    print(f"- Implemented: {stats['implemented_count']} ({stats['percentage']}%)")
    print(f"- Remaining: {stats['total_models'] - stats['implemented_count']} ({100 - stats['percentage']}%)")
    print("Done!")

if __name__ == "__main__":
    main()