#!/usr/bin/env python3
"""
Script to update the model coverage report in HF_MODEL_COVERAGE_ROADMAP.md
after implementing missing models.
"""

import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Ensure enhanced_generator is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import from enhanced_generator
try:
    from enhanced_generator import MODEL_REGISTRY, get_model_architecture
except ImportError as e:
    print(f"Error: Failed to import from enhanced_generator: {e}")
    print("Make sure to update enhanced_generator.py first with update_model_registry.py")
    sys.exit(1)

def get_implemented_models() -> Dict[str, Set[str]]:
    """Get lists of implemented models grouped by architecture."""
    architectures = {
        "encoder-only": set(),
        "decoder-only": set(),
        "encoder-decoder": set(),
        "vision": set(),
        "vision-text": set(),
        "speech": set(),
        "multimodal": set(),
        "unknown": set()
    }
    
    for model_type in MODEL_REGISTRY.keys():
        arch = get_model_architecture(model_type)
        if arch in architectures:
            architectures[arch].add(model_type)
        else:
            architectures["unknown"].add(model_type)
    
    return architectures

def get_roadmap_models() -> Dict[str, Dict[str, bool]]:
    """
    Parse the roadmap markdown file to extract all model lists
    and their implementation status.
    
    Returns:
        Dictionary of {model_name: {"implemented": bool, "priority": str}}
    """
    roadmap_file = "skills/HF_MODEL_COVERAGE_ROADMAP.md"
    
    if not os.path.exists(roadmap_file):
        print(f"Error: Roadmap file {roadmap_file} not found!")
        sys.exit(1)
    
    with open(roadmap_file, 'r') as f:
        content = f.read()
    
    # Extract all models from checklist items
    models = {}
    priority = "unknown"
    
    # Look for priority sections
    for line in content.splitlines():
        if "## Phase 2: High-Priority Models" in line:
            priority = "critical"
        elif "## Phase 3: Architecture Expansion" in line:
            priority = "high"
        elif "## Phase 4: Medium-Priority Models" in line:
            priority = "medium"
        elif "## Phase 5: Low-Priority Models" in line:
            priority = "low"
        
        # Extract model names from checklist items
        if line.strip().startswith("- ["):
            implemented = "[x]" in line
            # Extract model name - it's the first word after "[x]" or "[ ]"
            match = re.search(r'- \[[ x]\]\s+(\S+)', line)
            if match:
                model_name = match.group(1).lower()
                # Remove trailing punctuation
                model_name = model_name.rstrip(",:;.")
                models[model_name] = {"implemented": implemented, "priority": priority}
    
    return models

def update_roadmap_status(implemented_models: Dict[str, Set[str]]) -> bool:
    """Update the roadmap file with current implementation status."""
    roadmap_file = "skills/HF_MODEL_COVERAGE_ROADMAP.md"
    
    # Get all implemented models as a flat set
    all_implemented = set()
    for models in implemented_models.values():
        all_implemented.update(models)
    
    # Normalize names (replace underscore with hyphen)
    normalized_implemented = {model.replace('_', '-') for model in all_implemented}
    
    # Also include the original names
    normalized_implemented.update(all_implemented)
    
    # Read the roadmap file
    with open(roadmap_file, 'r') as f:
        lines = f.readlines()
    
    # Update checklist items
    updated_lines = []
    total_models = 0
    implemented_count = 0
    
    for line in lines:
        if line.strip().startswith("- ["):
            total_models += 1
            # Extract model name
            match = re.search(r'- \[[ x]\]\s+(\S+)', line)
            if match:
                model_name = match.group(1).lower().rstrip(",:;.")
                model_name_norm = model_name.replace('_', '-')
                
                # Check if the model is implemented
                is_implemented = (
                    model_name in normalized_implemented or 
                    model_name_norm in normalized_implemented
                )
                
                if is_implemented:
                    implemented_count += 1
                    # Update the checkmark if needed
                    if "[ ]" in line:
                        line = line.replace("[ ]", "[x]")
                        if not line.endswith("\n"):
                            line += "\n"
                        # Add implementation date
                        line = line.replace("\n", f" - Implemented on {datetime.now().strftime('%B %d, %Y')}\n")
        
        updated_lines.append(line)
    
    # Update the current status section
    current_status_updated = False
    for i, line in enumerate(updated_lines):
        if "## Current Status" in line:
            # Find the next few lines containing status info
            for j in range(i+1, min(i+15, len(updated_lines))):
                if "**Total Models Tracked:**" in updated_lines[j]:
                    updated_lines[j] = f"- **Total Models Tracked:** {total_models}\n"
                elif "**Implemented Models:**" in updated_lines[j]:
                    percentage = round(implemented_count / total_models * 100, 1)
                    updated_lines[j] = f"- **Implemented Models:** {implemented_count} ({percentage}%)\n"
                elif "**Missing Models:**" in updated_lines[j]:
                    missing = total_models - implemented_count
                    percentage = round(missing / total_models * 100, 1)
                    updated_lines[j] = f"- **Missing Models:** {missing} ({percentage}%)\n"
                    current_status_updated = True
                    break
    
    # Write the updated content
    with open(roadmap_file, 'w') as f:
        f.writelines(updated_lines)
    
    print(f"Updated roadmap status: {implemented_count}/{total_models} models implemented ({round(implemented_count/total_models*100, 1)}%)")
    return current_status_updated

def generate_coverage_report(implemented_models: Dict[str, Set[str]], roadmap_models: Dict[str, Dict[str, bool]]) -> str:
    """Generate a detailed coverage report."""
    # Get architecture counts
    architecture_counts = {arch: len(models) for arch, models in implemented_models.items()}
    total_implemented = sum(architecture_counts.values())
    
    # Calculate priority counts
    priority_models = {
        "critical": set(),
        "high": set(),
        "medium": set(),
        "low": set(),
        "unknown": set()
    }
    
    for model, info in roadmap_models.items():
        priority = info["priority"]
        if priority in priority_models:
            priority_models[priority].add(model)
    
    # Create flat sets for easy comparison
    all_implemented = set()
    for models in implemented_models.values():
        all_implemented.update(models)
    
    normalized_implemented = {model.replace('_', '-') for model in all_implemented}
    normalized_implemented.update(all_implemented)
    
    implemented_by_priority = {
        priority: {model for model in models if model in normalized_implemented or model.replace('_', '-') in normalized_implemented}
        for priority, models in priority_models.items()
    }
    
    priority_counts = {
        priority: (len(implemented), len(models))
        for priority, models in priority_models.items()
        for implemented in [implemented_by_priority[priority]]
        if len(models) > 0
    }
    
    # Build the report
    report = "# HuggingFace Model Coverage Report\n\n"
    report += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Overall summary
    total_roadmap = len(roadmap_models)
    report += "## Summary\n\n"
    report += f"- **Total models in registry:** {total_implemented}\n"
    report += f"- **Total models tracked in roadmap:** {total_roadmap}\n"
    
    # Calculate overall implementation percentage
    implemented_in_roadmap = sum(1 for model in roadmap_models if model in normalized_implemented or model.replace('_', '-') in normalized_implemented)
    percentage = round(implemented_in_roadmap / total_roadmap * 100, 1) if total_roadmap > 0 else 0
    report += f"- **Roadmap models implemented:** {implemented_in_roadmap} ({percentage}%)\n"
    report += f"- **Roadmap models missing:** {total_roadmap - implemented_in_roadmap} ({round(100 - percentage, 1)}%)\n\n"
    
    # Coverage by architecture
    report += "## Coverage by Architecture\n\n"
    for arch, models in implemented_models.items():
        report += f"### {arch.capitalize()} Models\n\n"
        report += f"- **Models in registry:** {len(models)}\n"
        
        if models:
            report += f"- **Implemented models:** {', '.join(sorted(models))}\n\n"
        else:
            report += "- **No models implemented for this architecture**\n\n"
    
    # Coverage by priority
    report += "## Coverage by Priority\n\n"
    for priority, (implemented, total) in priority_counts.items():
        percentage = round(implemented / total * 100, 1) if total > 0 else 0
        report += f"### {priority.capitalize()} Priority Models\n\n"
        report += f"- **Total {priority} models:** {total}\n"
        report += f"- **Implemented:** {implemented} ({percentage}%)\n"
        report += f"- **Missing:** {total - implemented}\n\n"
        
        if implemented > 0:
            implemented_models = sorted([model for model in priority_models[priority] 
                                       if model in normalized_implemented or model.replace('_', '-') in normalized_implemented])
            report += "**Implemented models:**\n"
            for model in implemented_models:
                report += f"- {model}\n"
            report += "\n"
        
        missing_models = sorted([model for model in priority_models[priority] 
                               if model not in normalized_implemented and model.replace('_', '-') not in normalized_implemented])
        if missing_models:
            report += "**Missing models:**\n"
            for model in missing_models:
                report += f"- {model}\n"
            report += "\n"
    
    return report

def main():
    """Main entry point for the script."""
    # Get implemented models from MODEL_REGISTRY
    implemented_models = get_implemented_models()
    
    # Get roadmap models
    roadmap_models = get_roadmap_models()
    
    # Update the roadmap file
    update_success = update_roadmap_status(implemented_models)
    
    # Generate a detailed coverage report
    report = generate_coverage_report(implemented_models, roadmap_models)
    
    # Save the report
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = f"reports/model_coverage_report_{timestamp}.md"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"Coverage report saved to {report_file}")
    return 0 if update_success else 1

if __name__ == "__main__":
    sys.exit(main())