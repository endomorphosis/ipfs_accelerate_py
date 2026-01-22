#!/usr/bin/env python3
"""
Coverage Report Generator

This script analyzes the HuggingFace model test coverage and generates a report.
"""

import os
import sys
import re
import logging
import argparse
import datetime
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path to allow imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def parse_roadmap_file(roadmap_path):
    """
    Parse the roadmap file to extract model information.
    
    Args:
        roadmap_path: Path to HF_MODEL_COVERAGE_ROADMAP.md
    
    Returns:
        Dictionary containing model information
    """
    models_data = {
        "total_models": 0,
        "implemented_models": 0,
        "missing_models": 0,
        "models_by_type": {},
        "models_by_architecture": {},
        "high_priority_models": [],
        "medium_priority_models": [],
        "low_priority_models": []
    }
    
    architecture_types = {
        "encoder-only": [],
        "decoder-only": [],
        "encoder-decoder": [],
        "vision": [],
        "vision-text": [],
        "multimodal": [],
        "speech": []
    }
    
    current_section = None
    high_priority_section = False
    medium_priority_section = False
    
    try:
        with open(roadmap_path, "r") as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            
            # Extract total models information
            total_match = re.search(r"Total Models Tracked:\s*(\d+)", line)
            if total_match:
                models_data["total_models"] = int(total_match.group(1))
                continue
                
            implemented_match = re.search(r"Implemented Models:\s*(\d+)\s*\((.+)%\)", line)
            if implemented_match:
                models_data["implemented_models"] = int(implemented_match.group(1))
                continue
                
            missing_match = re.search(r"Missing Models:\s*(\d+)\s*\((.+)%\)", line)
            if missing_match:
                models_data["missing_models"] = int(missing_match.group(1))
                continue
            
            # Track sections
            if "## Phase 2: High-Priority Models" in line:
                high_priority_section = True
                medium_priority_section = False
                current_section = None
                continue
                
            if "## Phase 3: Architecture Expansion" in line or "## Phase 4: Medium-Priority Models" in line:
                high_priority_section = False
                medium_priority_section = True
                current_section = None
                continue
                
            # Track subheadings in high priority section
            if high_priority_section and line.startswith("### "):
                current_section = line.replace("### ", "").strip()
                continue
                
            # Process model lines
            model_match = re.search(r"- \[([\sx])\] (.+?)(?: \((.+?)\))?$", line)
            if model_match:
                is_implemented = model_match.group(1) == "x"
                model_name = model_match.group(2).strip()
                model_type = model_match.group(3) if model_match.group(3) else model_name.lower()
                
                # Determine architecture from current section
                architecture = None
                if current_section:
                    if "Text" in current_section or "Text Models" in current_section:
                        if "encoder-only" in current_section.lower():
                            architecture = "encoder-only"
                        elif "decoder-only" in current_section.lower():
                            architecture = "decoder-only"
                        elif "encoder-decoder" in current_section.lower():
                            architecture = "encoder-decoder"
                        else:
                            # Determine from model name
                            if any(x in model_name.lower() for x in ["bert", "albert", "roberta", "electra", "xlm-roberta", "mpnet"]):
                                architecture = "encoder-only"
                            elif any(x in model_name.lower() for x in ["gpt", "llama", "bloom", "mistral", "phi", "falcon"]):
                                architecture = "decoder-only"
                            elif any(x in model_name.lower() for x in ["t5", "bart", "pegasus", "longt5", "led"]):
                                architecture = "encoder-decoder"
                    elif "Vision" in current_section:
                        if "text" in current_section.lower() or "Multimodal" in current_section:
                            architecture = "vision-text"
                        else:
                            architecture = "vision"
                    elif "Audio" in current_section or "Speech" in current_section:
                        architecture = "speech"
                    elif "Multimodal" in current_section:
                        architecture = "multimodal"
                
                # Clean model type
                model_type = model_type.lower().replace("-", "_")
                
                # Track model by type
                if model_type not in models_data["models_by_type"]:
                    models_data["models_by_type"][model_type] = {
                        "implemented": 0,
                        "missing": 0,
                        "total": 0,
                        "architecture": architecture
                    }
                
                models_data["models_by_type"][model_type]["total"] += 1
                if is_implemented:
                    models_data["models_by_type"][model_type]["implemented"] += 1
                else:
                    models_data["models_by_type"][model_type]["missing"] += 1
                
                # Track model by architecture
                if architecture:
                    if architecture not in models_data["models_by_architecture"]:
                        models_data["models_by_architecture"][architecture] = {
                            "implemented": 0,
                            "missing": 0,
                            "total": 0
                        }
                    
                    models_data["models_by_architecture"][architecture]["total"] += 1
                    if is_implemented:
                        models_data["models_by_architecture"][architecture]["implemented"] += 1
                    else:
                        models_data["models_by_architecture"][architecture]["missing"] += 1
                    
                    # Track in architecture types
                    if not is_implemented and architecture in architecture_types:
                        architecture_types[architecture].append({
                            "name": model_name,
                            "type": model_type,
                            "architecture": architecture
                        })
                
                # Track in priority lists
                model_info = {
                    "name": model_name,
                    "type": model_type,
                    "implemented": is_implemented,
                    "architecture": architecture
                }
                
                if high_priority_section:
                    models_data["high_priority_models"].append(model_info)
                elif medium_priority_section:
                    models_data["medium_priority_models"].append(model_info)
                
    except Exception as e:
        logger.error(f"Error parsing roadmap file: {e}")
    
    # Add missing models by architecture type
    models_data["missing_by_architecture"] = architecture_types
    
    return models_data

def check_existing_tests(tests_dir):
    """
    Check existing test files in the given directory.
    
    Args:
        tests_dir: Directory containing test files
        
    Returns:
        Set of model types that already have tests
    """
    existing_tests = set()
    
    try:
        for file in os.listdir(tests_dir):
            if file.startswith("test_hf_") and file.endswith(".py"):
                model_type = file[8:-3]  # Extract model type from filename
                existing_tests.add(model_type)
    except Exception as e:
        logger.error(f"Error checking existing tests: {e}")
    
    return existing_tests

def generate_coverage_report(roadmap_path, tests_dir, output_dir):
    """
    Generate a coverage report for HuggingFace model tests.
    
    Args:
        roadmap_path: Path to HF_MODEL_COVERAGE_ROADMAP.md
        tests_dir: Directory containing test files
        output_dir: Directory to save the report
        
    Returns:
        Path to the generated report
    """
    # Parse roadmap file
    models_data = parse_roadmap_file(roadmap_path)
    
    # Check existing tests
    existing_tests = check_existing_tests(tests_dir)
    
    # Update implementation status based on existing tests
    for architecture, models in models_data["missing_by_architecture"].items():
        updated_models = []
        for model in models:
            if model["type"] in existing_tests:
                # This model actually has a test
                if architecture in models_data["models_by_architecture"]:
                    models_data["models_by_architecture"][architecture]["implemented"] += 1
                    models_data["models_by_architecture"][architecture]["missing"] -= 1
            else:
                updated_models.append(model)
        
        models_data["missing_by_architecture"][architecture] = updated_models
    
    # Update priorities based on existing tests
    for priority_list in ["high_priority_models", "medium_priority_models"]:
        for model in models_data[priority_list]:
            if model["type"] in existing_tests:
                model["implemented"] = True
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate report filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"coverage_report_{timestamp}.md")
    json_file = os.path.join(output_dir, f"coverage_report_{timestamp}.json")
    
    # Generate JSON report
    with open(json_file, "w") as f:
        json.dump(models_data, f, indent=2)
    
    # Generate Markdown report
    with open(report_file, "w") as f:
        f.write("# HuggingFace Model Test Coverage Report\n\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write overall stats
        f.write("## Overall Coverage\n\n")
        total = models_data["total_models"]
        implemented = sum(1 for model in models_data.get("high_priority_models", []) if model["implemented"])
        implemented += sum(1 for model in models_data.get("medium_priority_models", []) if model["implemented"])
        missing = total - implemented
        
        if total > 0:
            percentage = (implemented / total) * 100
        else:
            percentage = 0
            
        f.write(f"- **Total Models Tracked:** {total}\n")
        f.write(f"- **Implemented Models:** {implemented} ({percentage:.1f}%)\n")
        f.write(f"- **Missing Models:** {missing} ({100-percentage:.1f}%)\n\n")
        
        # Write architecture stats
        f.write("## Coverage by Architecture\n\n")
        f.write("| Architecture | Implemented | Missing | Total | Coverage |\n")
        f.write("|--------------|-------------|---------|-------|----------|\n")
        
        for arch, stats in models_data["models_by_architecture"].items():
            if stats["total"] > 0:
                percentage = (stats["implemented"] / stats["total"]) * 100
            else:
                percentage = 0
                
            f.write(f"| {arch} | {stats['implemented']} | {stats['missing']} | {stats['total']} | {percentage:.1f}% |\n")
        
        # Write high priority missing models
        f.write("\n## High Priority Missing Models\n\n")
        
        for arch, models in models_data["missing_by_architecture"].items():
            if not models:
                continue
                
            f.write(f"### {arch.capitalize()}\n\n")
            f.write("| Model Type | Model Name | Architecture |\n")
            f.write("|------------|------------|-------------|\n")
            
            for model in models:
                f.write(f"| {model['type']} | {model['name']} | {model['architecture']} |\n")
            
            f.write("\n")
        
        # Write implementation plan
        f.write("## Implementation Plan\n\n")
        f.write("To complete the model coverage, implement tests for the following models in order of priority:\n\n")
        
        # High priority models
        f.write("### High Priority\n\n")
        high_missing = [model for model in models_data["high_priority_models"] if not model["implemented"]]
        
        for model in high_missing:
            f.write(f"- [ ] **{model['type']}** ({model['name']}) - {model['architecture']}\n")
        
        # Medium priority models
        f.write("\n### Medium Priority\n\n")
        medium_missing = [model for model in models_data["medium_priority_models"] if not model["implemented"]]
        
        # Group by architecture
        by_arch = {}
        for model in medium_missing:
            arch = model["architecture"]
            if arch not in by_arch:
                by_arch[arch] = []
            by_arch[arch].append(model)
        
        for arch, models in by_arch.items():
            f.write(f"#### {arch.capitalize()}\n\n")
            for model in models:
                f.write(f"- [ ] **{model['type']}** ({model['name']})\n")
            f.write("\n")
    
    logger.info(f"Generated coverage report at {report_file}")
    logger.info(f"Generated JSON data at {json_file}")
    
    return report_file

def main():
    parser = argparse.ArgumentParser(description="Generate HuggingFace model test coverage report")
    parser.add_argument("--roadmap", default="/home/barberb/ipfs_accelerate_py/test/skills/HF_MODEL_COVERAGE_ROADMAP.md", 
                        help="Path to HF_MODEL_COVERAGE_ROADMAP.md")
    parser.add_argument("--tests-dir", default="/home/barberb/ipfs_accelerate_py/test/skills/fixed_tests", 
                        help="Directory containing test files")
    parser.add_argument("--output-dir", default="reports", help="Directory to save the report")
    
    args = parser.parse_args()
    
    logger.info(f"Generating coverage report for HuggingFace model tests")
    logger.info(f"Roadmap file: {args.roadmap}")
    logger.info(f"Tests directory: {args.tests_dir}")
    
    report_path = generate_coverage_report(args.roadmap, args.tests_dir, args.output_dir)
    
    logger.info(f"Report generated at {report_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())