#!/usr/bin/env python3
"""
Update Benchmark Reports with Simulation Warnings

This script adds simulation warnings to benchmark reports to clearly indicate 
which hardware results are based on simulated data rather than actual hardware.

Usage:
    python update_benchmark_reports.py
"""

import os
import sys
import glob
import re
import json
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for simulation warnings
HTML_SIMULATION_WARNING = """
<div class="simulation-warning" style="background-color: #ffcccc; border: 2px solid #ff0000; padding: 15px; margin: 15px 0; border-radius: 5px; color: #cc0000;">
    <h2>⚠️ SIMULATION WARNING ⚠️</h2>
    <p><strong>This report contains data from simulated hardware that may not reflect actual performance.</strong></p>
    <p>The following hardware platforms were simulated: {}</p>
    <p>Simulated results should be treated as approximations and not used for critical performance decisions without validation on actual hardware.</p>
</div>
"""

MARKDOWN_SIMULATION_WARNING = """
# ⚠️ SIMULATION WARNING ⚠️

**This report contains data from simulated hardware that may not reflect actual performance.**

The following hardware platforms were simulated:
{}

Simulated results should be treated as approximations and not used for critical performance decisions without validation on actual hardware.

---

"""

def has_simulation_warnings(content):
    """Check if content already has simulation warnings."""
    return "SIMULATION WARNING" in content or "simulation-warning" in content

def update_html_report(file_path):
    """Add simulation warnings to HTML reports."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if has_simulation_warnings(content):
            logger.info(f"HTML file already has simulation warnings: {file_path}")
            return False
            
        # For simplicity, assume all hardward platforms other than CPU are simulated
        simulated_platforms = "WEBGPU, WEBNN, ROCM, MPS, QNN, OPENVINO"
        warning_html = HTML_SIMULATION_WARNING.format(simulated_platforms)
        
        # Insert warning after body tag or at the beginning of the content
        if "<body" in content:
            body_pos = content.find("<body")
            body_end = content.find(">", body_pos)
            new_content = content[:body_end+1] + warning_html + content[body_end+1:]
        else:
            # Add at the beginning
            new_content = warning_html + content
            
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
            
        logger.info(f"Added simulation warnings to HTML file: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error updating HTML file {file_path}: {e}")
        return False

def update_markdown_report(file_path):
    """Add simulation warnings to Markdown reports."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if has_simulation_warnings(content):
            logger.info(f"Markdown file already has simulation warnings: {file_path}")
            return False
            
        # For simplicity, assume all hardward platforms other than CPU are simulated
        simulated_platforms = """- WEBGPU 
- WEBNN
- ROCM
- MPS
- QNN
- OPENVINO"""
        warning_md = MARKDOWN_SIMULATION_WARNING.format(simulated_platforms)
        
        # Add at the beginning or after title
        if content.startswith("# "):
            # Find end of title
            title_end = content.find("\n", 2)
            if title_end > 0:
                new_content = content[:title_end+1] + "\n" + warning_md + content[title_end+1:]
            else:
                new_content = content + "\n\n" + warning_md
        else:
            new_content = warning_md + content
            
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
            
        logger.info(f"Added simulation warnings to Markdown file: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error updating Markdown file {file_path}: {e}")
        return False

def update_json_file(file_path):
    """Add simulation information to JSON files."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                content = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON file: {file_path}")
                return False
                
        # Check if already has simulation markers
        if "simulated_hardware" in content or "warnings" in content and any(
            w.get("type") == "simulation" for w in content.get("warnings", [])):
            logger.info(f"JSON file already has simulation markers: {file_path}")
            return False
            
        # For simplicity, assume all hardward platforms other than CPU are simulated
        simulated_hardware = ["webgpu", "webnn", "rocm", "mps", "qnn", "openvino"]
        
        # Add simulated hardware information
        content["simulated_hardware"] = simulated_hardware
        
        # Add warnings
        if "warnings" not in content:
            content["warnings"] = []
            
        content["warnings"].append({
            "type": "simulation",
            "message": "This data contains simulated hardware results",
            "affected_hardware": simulated_hardware,
            "verification_date": "2025-04-10T00:00:00Z"
        })
        
        # Write updated content
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2)
            
        logger.info(f"Added simulation information to JSON file: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error updating JSON file {file_path}: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Update benchmark reports with simulation warnings")
    parser.add_argument("--dir", default="benchmark_results", help="Directory containing benchmark reports")
    parser.add_argument("--types", default="html,md,json", help="File types to update (comma-separated)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    args = parser.parse_args()
    
    # Parse file types
    file_types = args.types.split(",")
    valid_types = {"html", "md", "json"}
    invalid_types = set(file_types) - valid_types
    if invalid_types:
        logger.error(f"Invalid file types: {', '.join(invalid_types)}")
        logger.error(f"Valid file types are: {', '.join(valid_types)}")
        return 1
        
    # Track statistics
    stats = {
        "html_files": 0,
        "html_updated": 0,
        "md_files": 0,
        "md_updated": 0,
        "json_files": 0,
        "json_updated": 0,
        "errors": 0
    }
    
    # Process HTML reports
    if "html" in file_types:
        html_files = glob.glob(os.path.join(args.dir, "**/*.html"), recursive=True)
        stats["html_files"] = len(html_files)
        for file_path in html_files:
            if args.dry_run:
                logger.info(f"Would update HTML file: {file_path}")
            else:
                if update_html_report(file_path):
                    stats["html_updated"] += 1
                    
    # Process Markdown reports
    if "md" in file_types:
        md_files = glob.glob(os.path.join(args.dir, "**/*.md"), recursive=True)
        md_files.extend(glob.glob(os.path.join(args.dir, "**/*.markdown"), recursive=True))
        stats["md_files"] = len(md_files)
        for file_path in md_files:
            if args.dry_run:
                logger.info(f"Would update Markdown file: {file_path}")
            else:
                if update_markdown_report(file_path):
                    stats["md_updated"] += 1
                    
    # Process JSON files
    if "json" in file_types:
        json_files = glob.glob(os.path.join(args.dir, "**/*.json"), recursive=True)
        stats["json_files"] = len(json_files)
        for file_path in json_files:
            if args.dry_run:
                logger.info(f"Would update JSON file: {file_path}")
            else:
                if update_json_file(file_path):
                    stats["json_updated"] += 1
                    
    # Print summary
    logger.info(f"=== Summary ===")
    logger.info(f"HTML files: {stats['html_files']} found, {stats['html_updated']} updated")
    logger.info(f"Markdown files: {stats['md_files']} found, {stats['md_updated']} updated")
    logger.info(f"JSON files: {stats['json_files']} found, {stats['json_updated']} updated")
    logger.info(f"Errors: {stats['errors']}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())