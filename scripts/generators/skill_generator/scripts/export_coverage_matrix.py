#!/usr/bin/env python3
"""
Export Coverage Matrix

This script exports the model coverage matrix in various formats (CSV, Markdown, HTML).
"""

import os
import sys
import csv
import json
import logging
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set

# Add parent directory to path to allow imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_models_from_report(report_path: str) -> Dict[str, Any]:
    """
    Load model information from a coverage report.
    
    Args:
        report_path: Path to coverage report JSON file
        
    Returns:
        Dictionary with model information
    """
    try:
        with open(report_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading models from report: {str(e)}")
        return {}

def build_model_matrix(report_data: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Set[str], Dict[str, int]]:
    """
    Build the coverage matrix from report data.
    
    Args:
        report_data: Coverage report data
        
    Returns:
        Tuple of (model_matrix, architectures, counts)
    """
    # Extract models by architecture
    architectures = set(report_data.get("models_by_architecture", {}).keys())
    
    # Combine high and medium priority models
    all_models = []
    all_models.extend(report_data.get("high_priority_models", []))
    all_models.extend(report_data.get("medium_priority_models", []))
    
    # Build matrix
    model_matrix = {}
    for model in all_models:
        model_type = model.get("type", "unknown")
        implemented = model.get("implemented", False)
        architecture = model.get("architecture", "unknown")
        
        if model_type not in model_matrix:
            model_matrix[model_type] = {
                "type": model_type,
                "name": model.get("name", model_type),
                "implemented": implemented,
                "architecture": architecture
            }
    
    # Count models by architecture
    counts = {}
    for arch in architectures:
        implemented = 0
        total = 0
        
        for model in all_models:
            if model.get("architecture") == arch:
                total += 1
                if model.get("implemented", False):
                    implemented += 1
        
        counts[arch] = {
            "implemented": implemented,
            "total": total,
            "percentage": (implemented / total * 100) if total > 0 else 0
        }
    
    return model_matrix, architectures, counts

def export_to_csv(model_matrix: Dict[str, Dict[str, Any]], architectures: Set[str], 
                counts: Dict[str, Dict[str, int]], output_file: str):
    """
    Export the coverage matrix to CSV.
    
    Args:
        model_matrix: Model matrix
        architectures: Set of architectures
        counts: Counts by architecture
        output_file: Output file path
    """
    try:
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(["Model Type", "Model Name", "Architecture", "Implemented"])
            
            # Write model rows
            for model_type, model_info in model_matrix.items():
                writer.writerow([
                    model_type,
                    model_info.get("name", model_type),
                    model_info.get("architecture", "unknown"),
                    "Yes" if model_info.get("implemented", False) else "No"
                ])
            
            # Write blank row
            writer.writerow([])
            
            # Write counts
            writer.writerow(["Architecture", "Implemented", "Total", "Percentage"])
            
            for arch, arch_counts in counts.items():
                writer.writerow([
                    arch,
                    arch_counts["implemented"],
                    arch_counts["total"],
                    f"{arch_counts['percentage']:.1f}%"
                ])
        
        logger.info(f"Exported CSV to {output_file}")
    except Exception as e:
        logger.error(f"Error exporting to CSV: {str(e)}")

def export_to_markdown(model_matrix: Dict[str, Dict[str, Any]], architectures: Set[str], 
                     counts: Dict[str, Dict[str, int]], output_file: str):
    """
    Export the coverage matrix to Markdown.
    
    Args:
        model_matrix: Model matrix
        architectures: Set of architectures
        counts: Counts by architecture
        output_file: Output file path
    """
    try:
        with open(output_file, "w") as f:
            # Write header
            f.write("# HuggingFace Model Coverage Matrix\n\n")
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write architecture summary
            f.write("## Coverage by Architecture\n\n")
            f.write("| Architecture | Implemented | Total | Coverage |\n")
            f.write("|--------------|-------------|-------|----------|\n")
            
            for arch, arch_counts in sorted(counts.items()):
                f.write(f"| {arch} | {arch_counts['implemented']} | {arch_counts['total']} | {arch_counts['percentage']:.1f}% |\n")
            
            f.write("\n")
            
            # Write models by architecture
            for arch in sorted(architectures):
                f.write(f"## {arch.capitalize()} Models\n\n")
                f.write("| Model Type | Model Name | Implemented |\n")
                f.write("|------------|------------|-------------|\n")
                
                # Filter models for this architecture
                arch_models = {k: v for k, v in model_matrix.items() if v.get("architecture") == arch}
                
                for model_type, model_info in sorted(arch_models.items()):
                    implemented = "✅" if model_info.get("implemented", False) else "❌"
                    f.write(f"| {model_type} | {model_info.get('name', model_type)} | {implemented} |\n")
                
                f.write("\n")
        
        logger.info(f"Exported Markdown to {output_file}")
    except Exception as e:
        logger.error(f"Error exporting to Markdown: {str(e)}")

def export_to_html(model_matrix: Dict[str, Dict[str, Any]], architectures: Set[str], 
                 counts: Dict[str, Dict[str, int]], output_file: str):
    """
    Export the coverage matrix to HTML.
    
    Args:
        model_matrix: Model matrix
        architectures: Set of architectures
        counts: Counts by architecture
        output_file: Output file path
    """
    try:
        with open(output_file, "w") as f:
            # Write HTML header
            f.write("<!DOCTYPE html>\n")
            f.write("<html lang=\"en\">\n")
            f.write("<head>\n")
            f.write("    <meta charset=\"UTF-8\">\n")
            f.write("    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n")
            f.write("    <title>HuggingFace Model Coverage Matrix</title>\n")
            f.write("    <style>\n")
            f.write("        body { font-family: Arial, sans-serif; margin: 20px; }\n")
            f.write("        h1 { color: #2c3e50; }\n")
            f.write("        h2 { color: #3498db; margin-top: 30px; }\n")
            f.write("        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }\n")
            f.write("        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }\n")
            f.write("        th { background-color: #f2f2f2; }\n")
            f.write("        tr:hover { background-color: #f5f5f5; }\n")
            f.write("        .implemented { color: green; }\n")
            f.write("        .not-implemented { color: red; }\n")
            f.write("        .progress-container { width: 100%; background-color: #e0e0e0; }\n")
            f.write("        .progress-bar { height: 20px; background-color: #4CAF50; text-align: center; color: white; }\n")
            f.write("    </style>\n")
            f.write("</head>\n")
            f.write("<body>\n")
            
            # Write header
            f.write("    <h1>HuggingFace Model Coverage Matrix</h1>\n")
            f.write(f"    <p>Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
            
            # Write architecture summary
            f.write("    <h2>Coverage by Architecture</h2>\n")
            f.write("    <table>\n")
            f.write("        <tr>\n")
            f.write("            <th>Architecture</th>\n")
            f.write("            <th>Implemented</th>\n")
            f.write("            <th>Total</th>\n")
            f.write("            <th>Coverage</th>\n")
            f.write("            <th>Progress</th>\n")
            f.write("        </tr>\n")
            
            for arch, arch_counts in sorted(counts.items()):
                percentage = arch_counts["percentage"]
                f.write("        <tr>\n")
                f.write(f"            <td>{arch}</td>\n")
                f.write(f"            <td>{arch_counts['implemented']}</td>\n")
                f.write(f"            <td>{arch_counts['total']}</td>\n")
                f.write(f"            <td>{percentage:.1f}%</td>\n")
                f.write(f"            <td><div class=\"progress-container\"><div class=\"progress-bar\" style=\"width: {percentage}%\">{percentage:.1f}%</div></div></td>\n")
                f.write("        </tr>\n")
            
            f.write("    </table>\n")
            
            # Write models by architecture
            for arch in sorted(architectures):
                f.write(f"    <h2>{arch.capitalize()} Models</h2>\n")
                f.write("    <table>\n")
                f.write("        <tr>\n")
                f.write("            <th>Model Type</th>\n")
                f.write("            <th>Model Name</th>\n")
                f.write("            <th>Implemented</th>\n")
                f.write("        </tr>\n")
                
                # Filter models for this architecture
                arch_models = {k: v for k, v in model_matrix.items() if v.get("architecture") == arch}
                
                for model_type, model_info in sorted(arch_models.items()):
                    implemented = model_info.get("implemented", False)
                    status_class = "implemented" if implemented else "not-implemented"
                    status_text = "✓" if implemented else "✗"
                    
                    f.write("        <tr>\n")
                    f.write(f"            <td>{model_type}</td>\n")
                    f.write(f"            <td>{model_info.get('name', model_type)}</td>\n")
                    f.write(f"            <td class=\"{status_class}\">{status_text}</td>\n")
                    f.write("        </tr>\n")
                
                f.write("    </table>\n")
            
            # Write HTML footer
            f.write("</body>\n")
            f.write("</html>\n")
        
        logger.info(f"Exported HTML to {output_file}")
    except Exception as e:
        logger.error(f"Error exporting to HTML: {str(e)}")

def export_to_json(model_matrix: Dict[str, Dict[str, Any]], architectures: Set[str], 
                 counts: Dict[str, Dict[str, int]], output_file: str):
    """
    Export the coverage matrix to JSON.
    
    Args:
        model_matrix: Model matrix
        architectures: Set of architectures
        counts: Counts by architecture
        output_file: Output file path
    """
    try:
        data = {
            "generated_at": datetime.datetime.now().isoformat(),
            "models": model_matrix,
            "architectures": list(architectures),
            "counts": counts
        }
        
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported JSON to {output_file}")
    except Exception as e:
        logger.error(f"Error exporting to JSON: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Export model coverage matrix")
    parser.add_argument("--report", required=True, help="Path to coverage report JSON file")
    parser.add_argument("--output-dir", default="reports", help="Directory to output files")
    parser.add_argument("--formats", nargs="+", choices=["csv", "markdown", "html", "json"], 
                      default=["markdown", "html"], help="Output formats")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load report data
    report_data = load_models_from_report(args.report)
    
    if not report_data:
        logger.error(f"Failed to load report data from {args.report}")
        return 1
    
    # Build model matrix
    model_matrix, architectures, counts = build_model_matrix(report_data)
    
    # Generate timestamp for filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export in requested formats
    for output_format in args.formats:
        if output_format == "csv":
            output_file = os.path.join(args.output_dir, f"coverage_matrix_{timestamp}.csv")
            export_to_csv(model_matrix, architectures, counts, output_file)
        elif output_format == "markdown":
            output_file = os.path.join(args.output_dir, f"coverage_matrix_{timestamp}.md")
            export_to_markdown(model_matrix, architectures, counts, output_file)
        elif output_format == "html":
            output_file = os.path.join(args.output_dir, f"coverage_matrix_{timestamp}.html")
            export_to_html(model_matrix, architectures, counts, output_file)
        elif output_format == "json":
            output_file = os.path.join(args.output_dir, f"coverage_matrix_{timestamp}.json")
            export_to_json(model_matrix, architectures, counts, output_file)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())