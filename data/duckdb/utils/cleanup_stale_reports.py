#\!/usr/bin/env python3
"""
Cleanup stale benchmark reports and mark them with appropriate warnings.

This script:
1. Scans for benchmark reports with potentially misleading data
2. Adds clear warnings to reports containing simulated data
3. Checks for outdated simulation methods in code files
4. Provides options to mark, archive, or fix problematic files
"""

import os
import re
import sys
import glob
import shutil
import argparse
import json
import datetime
from pathlib import Path


def add_simulation_warning(file_path, is_html=False):
    """Add a simulation warning to the top of a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Skip if warning already exists
    if "SIMULATION WARNING" in content:
        return False
        
    current_date = datetime.datetime.now().strftime("%B %d, %Y")
    
    if is_html:
        warning = f"""
<div style="background-color: #FFEBEE; border: 2px solid #F44336; padding: 10px; margin: 10px 0; border-radius: 5px;">
  <h3 style="color: #D32F2F; margin-top: 0;">⚠️ SIMULATION WARNING - ADDED {current_date}</h3>
  <p style="margin-bottom: 0;">This report may contain <strong>simulated benchmark data</strong> that does not represent actual hardware measurements. The results shown here should be treated as approximations and should <strong>not</strong> be used for critical decision-making without verification on real hardware.</p>
  <p>This warning has been automatically added to reports generated before the simulation detection improvements were implemented. For accurate benchmark data, please refer to reports generated after April 2025 with explicit simulation status indicators.</p>
</div>

"""
    else:
        warning = f"""
# ⚠️ SIMULATION WARNING - ADDED {current_date}

**This report may contain simulated benchmark data that does not represent actual hardware measurements.**
**The results shown here should be treated as approximations and should NOT be used for critical decision-making without verification on real hardware.**

This warning has been automatically added to reports generated before the simulation detection improvements were implemented.
For accurate benchmark data, please refer to reports generated after April 2025 with explicit simulation status indicators.

"""
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(warning + content)
    
    return True

def scan_for_problematic_reports(directory, check_json=False):
    """
    Scan for benchmark reports that might contain misleading data.
    """
    problematic_files = []
    
    # Patterns to look for in reports
    simulation_patterns = [
        r'hardware_\w+_results',
        r'benchmark_\w+_results\.json',
        r'performance_report_\w+\.md',
        r'model_hardware_report_\d+\.md',
        r'hardware_compatibility_matrix\.json'
    ]
    
    # Get all markdown and HTML files
    md_files = glob.glob(os.path.join(directory, "**/*.md"), recursive=True)
    html_files = glob.glob(os.path.join(directory, "**/*.html"), recursive=True)
    
    # If checking JSON files as well
    json_files = []
    if check_json:
        json_files = glob.glob(os.path.join(directory, "**/*.json"), recursive=True)
    
    # Check each markdown file
    for file_path in md_files:
        # Skip files that are clearly not reports
        if "README" in file_path or "GUIDE" in file_path:
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for simulation indicators without explicit warnings
        if any(re.search(pattern, content) for pattern in simulation_patterns) and "SIMULATION WARNING" not in content:
            if "simulated" in content.lower() and not re.search(r'marked as simulated|explicitly simulated|simulation status', content.lower()):
                problematic_files.append((file_path, "markdown", "Contains simulation results without proper warnings"))
    
    # Check each HTML file
    for file_path in html_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for simulation indicators without explicit warnings
        if any(re.search(pattern, content) for pattern in simulation_patterns) and "SIMULATION WARNING" not in content:
            if "simulated" in content.lower() and not re.search(r'marked as simulated|explicitly simulated|simulation status', content.lower()):
                problematic_files.append((file_path, "html", "Contains simulation results without proper warnings"))
    
    # Check each JSON file
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Check for hardware results without simulation flags
            if isinstance(data, dict) and any(k in data for k in ["results", "benchmarks", "performance", "hardware"]):
                if not any(k in data for k in ["is_simulated", "simulation_status", "simulation_flags"]):
                    problematic_files.append((file_path, "json", "Contains benchmark data without simulation flags"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Skip invalid JSON files
            continue
    
    return problematic_files

def mark_problematic_reports(problematic_files):
    """
    Add warnings to problematic reports.
    """
    marked_count = 0
    for file_path, file_type, reason in problematic_files:
        if file_type == "markdown":
            if add_simulation_warning(file_path, is_html=False):
                marked_count += 1
                print(f"Added warning to {file_path}")
        elif file_type == "html":
            if add_simulation_warning(file_path, is_html=True):
                marked_count += 1
                print(f"Added warning to {file_path}")
        # We don't modify JSON files
    
    return marked_count

def archive_problematic_files(problematic_files, archive_dir):
    """
    Archive problematic files.
    """
    os.makedirs(archive_dir, exist_ok=True)
    
    archived_count = 0
    for file_path, file_type, reason in problematic_files:
        # Create the relative path structure in the archive directory
        rel_path = os.path.relpath(file_path, os.path.dirname(archive_dir))
        archive_path = os.path.join(archive_dir, rel_path)
        
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(archive_path), exist_ok=True)
        
        # Copy the file to the archive
        shutil.copy2(file_path, archive_path)
        archived_count += 1
        print(f"Archived {file_path} to {archive_path}")
    
    return archived_count

def scan_for_simulation_patterns_in_code(directory):
    """
    Scan Python files for outdated simulation methods.
    """
    problematic_code_files = []
    
    # Patterns for problematic simulation code
    simulation_code_patterns = [
        (r'def\s+simulate_\w+\s*\([^)]*\)\s*:', "Contains simulation method without explicit status tracking"),
        (r'_simulated_hardware\s*=\s*(True|False)', "Uses direct simulation flag assignment"),
        (r'hardware_simulation\s*=\s*(True|False)', "Uses direct simulation flag assignment"),
        (r'simulate_performance\([^)]*\)', "Uses simulate_performance without status tracking"),
        (r'simulate_benchmark\([^)]*\)', "Uses simulate_benchmark without status tracking"),
        (r'mock_hardware_platform', "Uses mock hardware platform without explicit marking"),
    ]
    
    # Get all Python files
    py_files = glob.glob(os.path.join(directory, "**/*.py"), recursive=True)
    
    # Skip these directories
    skip_dirs = [
        "archived_", 
        "venv", 
        "__pycache__", 
        "test_simulation_detection.py",
        "update_db_schema_for_simulation.py"
    ]
    
    for file_path in py_files:
        # Skip files in specified directories
        if any(skip_dir in file_path for skip_dir in skip_dirs):
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for problematic patterns
        for pattern, reason in simulation_code_patterns:
            if re.search(pattern, content) and "is_simulated" not in content:
                problematic_code_files.append((file_path, pattern, reason))
                break
    
    return problematic_code_files

def fix_report_py_files(directory):
    """
    Fix Python report generator files to include simulation validation.
    """
    # Patterns to identify report generator files
    report_py_patterns = [
        r'def\s+generate_report',
        r'def\s+create_report',
        r'def\s+build_report',
        r'def\s+make_report',
        r'report_generator',
        r'class\s+\w*Report'
    ]
    
    # Get all Python files
    py_files = glob.glob(os.path.join(directory, "**/*.py"), recursive=True)
    
    fixed_count = 0
    for file_path in py_files:
        # Skip files in certain directories
        if any(skip_dir in file_path for skip_dir in ["archived_", "venv", "__pycache__"]):
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check if this is a report generator file
        if any(re.search(pattern, content) for pattern in report_py_patterns):
            # Check if it already has validation
            if "_validate_data_authenticity" not in content and "check_simulation_status" not in content:
                # Backup the file
                backup_path = file_path + ".bak"
                shutil.copy2(file_path, backup_path)
                
                # Add validation function
                validation_code = """
def _validate_data_authenticity(data):
    \"\"\"Validate that the data is authentic and not simulated.\"\"\"
    is_simulated = False
    simulation_reason = None
    
    # Check for simulation flags in data
    if hasattr(data, 'get') and callable(data.get):
        is_simulated = data.get('is_simulated', False)
        simulation_reason = data.get('simulation_reason', None)
    
    # If the data is simulated, add a warning
    if is_simulated:
        print(f"WARNING: Report contains simulated data. Reason: {simulation_reason}")
        return False, simulation_reason
    
    return True, None
"""
                
                # Add validation call to generate_report function
                modified_content = content
                report_funcs = ["generate_report", "create_report", "build_report", "make_report"]
                
                for func in report_funcs:
                    pattern = fr'def\s+{func}\s*\(([^)]*)\)\s*:'
                    match = re.search(pattern, modified_content)
                    
                    if match:
                        args = match.group(1)
                        func_pos = match.start()
                        
                        # Find the function body (the next line with content after the definition)
                        lines = modified_content.split("\n")
                        func_line = None
                        for i, line in enumerate(lines):
                            if re.search(pattern, line):
                                func_line = i
                                break
                        
                        if func_line is not None:
                            # Find the first non-comment line with content
                            for i in range(func_line + 1, len(lines)):
                                if lines[i].strip() and not lines[i].strip().startswith("#"):
                                    # Insert validation before this line
                                    indent = re.match(r'^\s*', lines[i]).group(0)
                                    validation_line = f"{indent}# Validate data authenticity\n{indent}is_authentic, simulation_reason = _validate_data_authenticity(data)\n{indent}if not is_authentic:\n{indent}    print(f\"WARNING: Adding simulation warning to report. Reason: {{simulation_reason}}\")\n"
                                    lines.insert(i, validation_line)
                                    break
                            
                            modified_content = "\n".join(lines)
                
                # Add validation function to the file
                if "_validate_data_authenticity" not in modified_content:
                    # Find the last import and add after it
                    last_import = 0
                    for i, line in enumerate(modified_content.split("\n")):
                        if line.startswith("import ") or line.startswith("from "):
                            last_import = i
                    
                    lines = modified_content.split("\n")
                    lines.insert(last_import + 1, validation_code)
                    modified_content = "\n".join(lines)
                
                # Write modified content back to the file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
                
                fixed_count += 1
                print(f"Fixed {file_path}")
    
    return fixed_count

def main():
    parser = argparse.ArgumentParser(description='Cleanup stale benchmark reports and mark them with warnings.')
    parser.add_argument('--scan', action='store_true', help='Scan for problematic reports without modifying them')
    parser.add_argument('--mark', action='store_true', help='Add warnings to problematic reports')
    parser.add_argument('--archive', action='store_true', help='Archive problematic files')
    parser.add_argument('--archive-dir', default='/home/barberb/ipfs_accelerate_py/test/archived_stale_reports', help='Directory to archive problematic files')
    parser.add_argument('--directory', default='/home/barberb/ipfs_accelerate_py/test', help='Directory to scan')
    parser.add_argument('--check-json', action='store_true', help='Check JSON files as well')
    parser.add_argument('--check-code', action='store_true', help='Check Python code for outdated simulation methods')
    parser.add_argument('--fix-report-py', action='store_true', help='Fix report generator Python files to include validation')
    
    args = parser.parse_args()
    
    if args.scan or (not args.mark and not args.archive and not args.check_code and not args.fix_report_py):
        print(f"Scanning for problematic reports in {args.directory}...")
        problematic_files = scan_for_problematic_reports(args.directory, args.check_json)
        
        if problematic_files:
            print(f"Found {len(problematic_files)} problematic files:")
            for file_path, file_type, reason in problematic_files:
                print(f"- {file_path} ({file_type}): {reason}")
        else:
            print("No problematic reports found.")
    
    if args.mark:
        print(f"Marking problematic reports in {args.directory}...")
        problematic_files = scan_for_problematic_reports(args.directory, args.check_json)
        marked_count = mark_problematic_reports(problematic_files)
        print(f"Added warnings to {marked_count} files.")
    
    if args.archive:
        print(f"Archiving problematic files to {args.archive_dir}...")
        problematic_files = scan_for_problematic_reports(args.directory, args.check_json)
        archived_count = archive_problematic_files(problematic_files, args.archive_dir)
        print(f"Archived {archived_count} files.")
    
    if args.check_code:
        print(f"Scanning for outdated simulation methods in Python code...")
        problematic_code_files = scan_for_simulation_patterns_in_code(args.directory)
        
        if problematic_code_files:
            print(f"Found {len(problematic_code_files)} Python files with outdated simulation methods:")
            for file_path, pattern, reason in problematic_code_files:
                print(f"- {file_path}: {reason}")
        else:
            print("No Python files with outdated simulation methods found.")
    
    if args.fix_report_py:
        print(f"Fixing report generator Python files to include validation...")
        fixed_count = fix_report_py_files(args.directory)
        print(f"Fixed {fixed_count} report generator Python files.")

if __name__ == "__main__":
    main()
