#!/usr/bin/env python3
"""
Cleanup Stale Reports

This script identifies, marks, or removes stale benchmark reports and JSON files
that may contain misleading information about simulated vs. real hardware results.

Created March 2025 to address the issue of misleading benchmark reports.

Usage:
    python cleanup_stale_reports.py --scan               # Scan for problematic files
    python cleanup_stale_reports.py --mark               # Mark problematic files with warnings
    python cleanup_stale_reports.py --remove             # Remove problematic files
    python cleanup_stale_reports.py --archive            # Archive problematic files
    python cleanup_stale_reports.py --fix-report-py      # Add validation to report generators
"""

import os
import sys
import argparse
import logging
import json
import re
import shutil
import datetime
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("cleanup_stale_reports.log")
    ]
)
logger = logging.getLogger(__name__)

# Define directories to scan
DIRS_TO_SCAN = [
    "benchmark_results",
    "api_check_results",
    "archived_test_results",
    "critical_model_results", 
    "hardware_fix_results"
]

class StaleReportCleaner:
    """Identifies and cleans up stale benchmark reports and JSON files."""
    
    def __init__(self, root_dir: str = "."):
        """
        Initialize the cleaner with a root directory.
        
        Args:
            root_dir: Root directory to scan (default: current directory)
        """
        self.root_dir = Path(root_dir)
        self.problematic_files = []
        self.warning_added_to = set()
        self.archive_dir = self.root_dir / "archived_stale_reports"
        
    def scan_for_problematic_files(self) -> List[Dict[str, Any]]:
        """
        Scan for problematic files that may contain misleading benchmark data.
        
        Returns:
            List of dictionaries with information about problematic files
        """
        logger.info(f"Scanning for problematic files in {self.root_dir}...")
        self.problematic_files = []
        
        # Check each directory
        for dir_name in DIRS_TO_SCAN:
            dir_path = self.root_dir / dir_name
            if not dir_path.exists():
                logger.warning(f"Directory not found: {dir_path}")
                continue
                
            logger.info(f"Scanning directory: {dir_path}")
            
            # Check HTML reports
            html_files = list(dir_path.glob("**/*.html"))
            for html_file in html_files:
                if self._is_problematic_html(html_file):
                    self.problematic_files.append({
                        "path": str(html_file),
                        "type": "html",
                        "issue": "May contain simulation results presented as real data",
                        "last_modified": datetime.datetime.fromtimestamp(html_file.stat().st_mtime).isoformat()
                    })
            
            # Check Markdown reports
            md_files = list(dir_path.glob("**/*.md"))
            for md_file in md_files:
                if self._is_problematic_markdown(md_file):
                    self.problematic_files.append({
                        "path": str(md_file),
                        "type": "markdown",
                        "issue": "May contain simulation results presented as real data",
                        "last_modified": datetime.datetime.fromtimestamp(md_file.stat().st_mtime).isoformat()
                    })
            
            # Check JSON files
            json_files = list(dir_path.glob("**/*.json"))
            for json_file in json_files:
                issue = self._check_json_file(json_file)
                if issue:
                    self.problematic_files.append({
                        "path": str(json_file),
                        "type": "json",
                        "issue": issue,
                        "last_modified": datetime.datetime.fromtimestamp(json_file.stat().st_mtime).isoformat()
                    })
        
        logger.info(f"Found {len(self.problematic_files)} problematic files")
        return self.problematic_files
    
    def _is_problematic_html(self, file_path: Path) -> bool:
        """
        Check if an HTML report contains problematic data.
        
        Args:
            file_path: Path to the HTML file
            
        Returns:
            True if problematic, False otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check if it's a benchmark report
            if "Benchmark" not in content or "Report" not in content:
                return False
                
            # Check for simulation warning
            has_simulation_warning = re.search(r'simulation|simulated', content, re.IGNORECASE) is not None
            
            # Check for hardware performance data
            has_hardware_data = re.search(r'(cuda|rocm|mps|openvino|qnn|webgpu|webnn)', content, re.IGNORECASE) is not None
            
            # If it has hardware data but no simulation warning, it may be problematic
            return has_hardware_data and not has_simulation_warning
            
        except Exception as e:
            logger.warning(f"Error checking HTML file {file_path}: {str(e)}")
            return False
    
    def _is_problematic_markdown(self, file_path: Path) -> bool:
        """
        Check if a Markdown report contains problematic data.
        
        Args:
            file_path: Path to the Markdown file
            
        Returns:
            True if problematic, False otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check if it's a benchmark report
            if "Benchmark" not in content or "Report" not in content:
                return False
                
            # Check for simulation warning
            has_simulation_warning = re.search(r'simulation|simulated', content, re.IGNORECASE) is not None
            
            # Check for hardware performance data
            has_hardware_data = re.search(r'(cuda|rocm|mps|openvino|qnn|webgpu|webnn)', content, re.IGNORECASE) is not None
            
            # If it has hardware data but no simulation warning, it may be problematic
            return has_hardware_data and not has_simulation_warning
            
        except Exception as e:
            logger.warning(f"Error checking Markdown file {file_path}: {str(e)}")
            return False
    
    def _check_json_file(self, file_path: Path) -> Optional[str]:
        """
        Check if a JSON file contains problematic data.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Issue description if problematic, None otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if it's a benchmark results file
            if not isinstance(data, dict):
                return None
                
            # Check for benchmark results
            if "results" in data or "benchmark_results" in data or "performance" in data:
                # Check for simulation flag
                has_simulation_flag = False
                
                # Check for various simulation flag patterns
                if "simulation" in data or "simulated" in data:
                    has_simulation_flag = True
                
                # Check for hardware results
                has_hardware_results = False
                for key in ["cuda", "rocm", "mps", "openvino", "qnn", "webgpu", "webnn"]:
                    if key in str(data).lower():
                        has_hardware_results = True
                        break
                
                if has_hardware_results and not has_simulation_flag:
                    return "Contains hardware results without simulation flags"
            
            # Check for truncated or empty files
            if file_path.stat().st_size < 100:
                return "File appears to be truncated or empty"
                
            # Check for outdated format
            if "generated_at" in data and isinstance(data["generated_at"], str):
                try:
                    generated_date = datetime.datetime.fromisoformat(data["generated_at"].replace("Z", "+00:00"))
                    days_old = (datetime.datetime.now() - generated_date).days
                    if days_old > 30:
                        return f"File is outdated (generated {days_old} days ago)"
                except:
                    pass
            
            return None
            
        except json.JSONDecodeError:
            return "Invalid JSON format"
        except Exception as e:
            logger.warning(f"Error checking JSON file {file_path}: {str(e)}")
            return None
    
    def mark_problematic_files(self) -> int:
        """
        Add warning headers to problematic files.
        
        Returns:
            Number of files marked with warnings
        """
        if not self.problematic_files:
            logger.warning("No problematic files found. Run scan_for_problematic_files() first.")
            return 0
            
        count = 0
        for file_info in self.problematic_files:
            file_path = Path(file_info["path"])
            file_type = file_info["type"]
            
            if str(file_path) in self.warning_added_to:
                continue
                
            try:
                # Add appropriate warning based on file type
                if file_type == "html":
                    self._add_html_warning(file_path, file_info["issue"])
                elif file_type == "markdown":
                    self._add_markdown_warning(file_path, file_info["issue"])
                elif file_type == "json":
                    self._add_json_warning(file_path, file_info["issue"])
                
                self.warning_added_to.add(str(file_path))
                count += 1
                logger.info(f"Added warning to {file_path}")
                
            except Exception as e:
                logger.error(f"Error adding warning to {file_path}: {str(e)}")
        
        logger.info(f"Added warnings to {count} files")
        return count
    
    def _add_html_warning(self, file_path: Path, issue: str) -> None:
        """Add warning to HTML file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        warning_html = f"""
        <div style="background-color: #ffcccc; border: 2px solid #ff0000; padding: 10px; margin: 10px 0; color: #cc0000;">
            <h2>⚠️ WARNING: POTENTIALLY MISLEADING DATA ⚠️</h2>
            <p>This report may contain simulated benchmark results that are presented as real hardware data.</p>
            <p>Issue: {issue}</p>
            <p>Marked as problematic by cleanup_stale_reports.py on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """
        
        # Insert after opening body tag
        if "<body>" in content:
            new_content = content.replace("<body>", f"<body>\n{warning_html}")
        else:
            # If no body tag, insert after opening html tag
            new_content = content.replace("<html>", f"<html>\n{warning_html}")
            
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
    
    def _add_markdown_warning(self, file_path: Path, issue: str) -> None:
        """Add warning to Markdown file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        warning_md = f"""
# ⚠️ WARNING: POTENTIALLY MISLEADING DATA ⚠️

**This report may contain simulated benchmark results that are presented as real hardware data.**

Issue: {issue}

*Marked as problematic by cleanup_stale_reports.py on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

---

"""
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(warning_md + content)
    
    def _add_json_warning(self, file_path: Path, issue: str) -> None:
        """Add warning to JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, dict):
                data["WARNING"] = {
                    "message": "POTENTIALLY MISLEADING DATA",
                    "details": "This file may contain simulated benchmark results that are presented as real hardware data.",
                    "issue": issue,
                    "marked_at": datetime.datetime.now().isoformat()
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            else:
                logger.warning(f"Cannot add warning to non-object JSON: {file_path}")
                
        except json.JSONDecodeError:
            # For invalid JSON, create a backup and override with warning
            backup_path = file_path.with_suffix(file_path.suffix + ".bak")
            shutil.copy2(file_path, backup_path)
            
            warning_data = {
                "WARNING": {
                    "message": "INVALID JSON DATA",
                    "details": "The original file was invalid JSON and has been backed up.",
                    "issue": issue,
                    "original_file_backup": str(backup_path),
                    "marked_at": datetime.datetime.now().isoformat()
                }
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(warning_data, f, indent=2)
    
    def remove_problematic_files(self) -> int:
        """
        Remove problematic files.
        
        Returns:
            Number of files removed
        """
        if not self.problematic_files:
            logger.warning("No problematic files found. Run scan_for_problematic_files() first.")
            return 0
            
        count = 0
        for file_info in self.problematic_files:
            file_path = Path(file_info["path"])
            
            try:
                file_path.unlink()
                count += 1
                logger.info(f"Removed {file_path}")
                
            except Exception as e:
                logger.error(f"Error removing {file_path}: {str(e)}")
        
        logger.info(f"Removed {count} files")
        return count
    
    def archive_problematic_files(self) -> int:
        """
        Archive problematic files to a dedicated directory.
        
        Returns:
            Number of files archived
        """
        if not self.problematic_files:
            logger.warning("No problematic files found. Run scan_for_problematic_files() first.")
            return 0
            
        # Create archive directory
        self.archive_dir.mkdir(exist_ok=True, parents=True)
        
        count = 0
        for file_info in self.problematic_files:
            file_path = Path(file_info["path"])
            
            # Create subdirectories in archive to maintain folder structure
            relative_path = file_path.relative_to(self.root_dir)
            archive_path = self.archive_dir / relative_path
            archive_path.parent.mkdir(exist_ok=True, parents=True)
            
            try:
                shutil.copy2(file_path, archive_path)
                count += 1
                logger.info(f"Archived {file_path} to {archive_path}")
                
            except Exception as e:
                logger.error(f"Error archiving {file_path}: {str(e)}")
        
        logger.info(f"Archived {count} files to {self.archive_dir}")
        return count
    
    def fix_report_generators(self) -> List[str]:
        """
        Add validation to report generator Python files.
        
        Returns:
            List of modified files
        """
        logger.info("Fixing report generator scripts...")
        modified_files = []
        
        # Find report generator scripts
        report_scripts = []
        for pattern in ["*report*.py", "*benchmark*report*.py", "*timing*report*.py"]:
            report_scripts.extend(list(self.root_dir.glob(pattern)))
        
        for script_path in report_scripts:
            if self._fix_report_script(script_path):
                modified_files.append(str(script_path))
        
        logger.info(f"Fixed {len(modified_files)} report generator scripts")
        return modified_files
    
    def _fix_report_script(self, script_path: Path) -> bool:
        """
        Add validation code to a report generator script.
        
        Args:
            script_path: Path to the script
            
        Returns:
            True if modified, False otherwise
        """
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if script already has simulation detection
            if "simulation" in content and "is_simulated" in content:
                logger.info(f"Script already has simulation detection: {script_path}")
                return False
            
            # Find appropriate places to insert validation code
            modified_content = content
            
            # Simple string replacement approach - a more robust implementation would use AST
            validation_function = """
def _validate_data_authenticity(self, df):
    \"\"\"
    Validate that the data is authentic and mark simulated results.
    
    Args:
        df: DataFrame with benchmark results
        
    Returns:
        Tuple of (DataFrame with authenticity flags, bool indicating if any simulation was detected)
    \"\"\"
    logger.info("Validating data authenticity...")
    simulation_detected = False
    
    # Add new column to track simulation status
    if 'is_simulated' not in df.columns:
        df['is_simulated'] = False
    
    # Check database for simulation flags if possible
    if self.conn:
        try:
            # Query simulation status from database
            simulation_query = "SELECT hardware_type, COUNT(*) as count, SUM(CASE WHEN is_simulated THEN 1 ELSE 0 END) as simulated_count FROM hardware_platforms GROUP BY hardware_type"
            sim_result = self.conn.execute(simulation_query).fetchdf()
            
            if not sim_result.empty:
                for _, row in sim_result.iterrows():
                    hw = row['hardware_type']
                    if row['simulated_count'] > 0:
                        # Mark rows with this hardware as simulated
                        df.loc[df['hardware_type'] == hw, 'is_simulated'] = True
                        simulation_detected = True
                        logger.warning(f"Detected simulation data for hardware: {hw}")
        except Exception as e:
            logger.warning(f"Failed to check simulation status in database: {e}")
    
    # Additional checks for simulation indicators in the data
    for hw in ['qnn', 'rocm', 'openvino', 'webgpu', 'webnn']:
        hw_data = df[df['hardware_type'] == hw]
        if not hw_data.empty:
            # Check for simulation patterns in the data
            if hw_data['throughput_items_per_second'].std() < 0.1 and len(hw_data) > 1:
                logger.warning(f"Suspiciously uniform performance for {hw} - possible simulation")
                df.loc[df['hardware_type'] == hw, 'is_simulated'] = True
                simulation_detected = True
    
    return df, simulation_detected
"""
            
            # Add validation function
            if "def __init__" in content:
                # Insert after the class definition
                class_pattern = r'class\s+\w+.*:'
                class_matches = list(re.finditer(class_pattern, content))
                if class_matches:
                    pos = class_matches[-1].end()
                    modified_content = content[:pos] + validation_function + content[pos:]
            else:
                # Append to the end of the file
                modified_content = content + "\n" + validation_function
            
            # Add validation call in appropriate places
            if "_fetch_timing_data" in modified_content:
                # Add validation call after fetching data
                fetch_pattern = r'def\s+_fetch_timing_data.*?return\s+.*?$'
                fetch_matches = list(re.finditer(fetch_pattern, modified_content, re.DOTALL))
                if fetch_matches:
                    match = fetch_matches[-1]
                    return_pos = modified_content.rfind("return", match.start(), match.end())
                    if return_pos > 0:
                        result_var = modified_content[return_pos:match.end()].strip().replace("return", "").strip()
                        validation_call = f"\n        # Validate data authenticity\n        {result_var}, simulation_detected = self._validate_data_authenticity({result_var})\n        if simulation_detected:\n            logger.warning(\"SIMULATION DATA DETECTED - Report may contain simulated results\")\n        \n        return {result_var}"
                        modified_content = modified_content[:return_pos] + validation_call + modified_content[match.end():]
            
            # Add simulation warning to report generation
            if "generate_" in modified_content and ("html" in modified_content or "report" in modified_content):
                # Add simulation warning to report generation
                html_pattern = r'def\s+(?:_generate_html_report|generate_.*?_report).*?{.*?}'
                html_matches = list(re.finditer(html_pattern, modified_content, re.DOTALL | re.IGNORECASE))
                
                if html_matches:
                    for match in html_matches:
                        # Find where HTML content is being generated
                        if "warning_html" not in modified_content[match.start():match.end()]:
                            # Add simulation warning
                            simulation_warning = """
            # Add simulation warning if needed
            simulation_detected = any(getattr(data, 'is_simulated', False) for _, data in df.iterrows()) if not df.empty else False
            
            warning_html = ""
            if simulation_detected:
                warning_html = '''
                <div style="background-color: #ffcccc; border: 2px solid #ff0000; padding: 10px; margin: 10px 0; color: #cc0000;">
                    <h2>⚠️ WARNING: REPORT CONTAINS SIMULATED DATA ⚠️</h2>
                    <p>This report contains results from simulated hardware that may not reflect real-world performance.</p>
                    <p>Simulated hardware data is included for comparison purposes only and should not be used for procurement decisions.</p>
                </div>
                '''
"""
                            write_pos = modified_content.find("with open(", match.start(), match.end())
                            if write_pos > 0:
                                modified_content = modified_content[:write_pos] + simulation_warning + modified_content[write_pos:]
            
            # Check if content was modified
            if modified_content != content:
                # Write modified content back to file
                with open(script_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
                
                logger.info(f"Added simulation validation to {script_path}")
                return True
            else:
                logger.info(f"No changes needed for {script_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error fixing script {script_path}: {str(e)}")
            return False
    
    def generate_report(self, output_file: str = "stale_report_cleanup_report.md") -> str:
        """
        Generate a report of problematic files.
        
        Args:
            output_file: Path to output report file
            
        Returns:
            Path to the generated report
        """
        if not self.problematic_files:
            logger.warning("No problematic files found. Run scan_for_problematic_files() first.")
            return ""
            
        report_path = Path(output_file)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Stale and Problematic Report Cleanup Report\n\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## Summary\n\n")
            f.write(f"- Total problematic files found: {len(self.problematic_files)}\n")
            
            # Count by type
            types = {}
            for file_info in self.problematic_files:
                file_type = file_info["type"]
                types[file_type] = types.get(file_type, 0) + 1
            
            f.write("- File types:\n")
            for file_type, count in types.items():
                f.write(f"  - {file_type}: {count}\n")
            
            f.write("\n## Problematic Files\n\n")
            f.write("| File Path | Type | Issue | Last Modified |\n")
            f.write("|-----------|------|-------|---------------|\n")
            
            for file_info in self.problematic_files:
                f.write(f"| {file_info['path']} | {file_info['type']} | {file_info['issue']} | {file_info['last_modified']} |\n")
            
            f.write("\n## Recommendations\n\n")
            f.write("Based on the scan results, we recommend the following actions:\n\n")
            f.write("1. **Mark files with warnings**: Add clear warnings to files that may contain misleading data\n")
            f.write("   ```bash\n   python cleanup_stale_reports.py --mark\n   ```\n\n")
            f.write("2. **Archive problematic files**: Move problematic files to an archive directory\n")
            f.write("   ```bash\n   python cleanup_stale_reports.py --archive\n   ```\n\n")
            f.write("3. **Fix report generators**: Add validation to report generator scripts\n")
            f.write("   ```bash\n   python cleanup_stale_reports.py --fix-report-py\n   ```\n\n")
            f.write("4. **Update database schema**: Ensure the database schema includes simulation flags\n")
            f.write("   ```bash\n   python update_db_schema_for_simulation.py\n   ```\n\n")
            
            f.write("\n## Next Steps\n\n")
            f.write("After implementing these recommendations, re-run the scan to verify that all issues have been addressed:\n\n")
            f.write("```bash\n")
            f.write("python cleanup_stale_reports.py --scan\n")
            f.write("```\n\n")
            
            f.write("If you need to remove problematic files entirely (use with caution):\n\n")
            f.write("```bash\n")
            f.write("python cleanup_stale_reports.py --remove\n")
            f.write("```\n")
        
        logger.info(f"Generated report: {report_path}")
        return str(report_path)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Cleanup stale benchmark reports and JSON files")
    
    # Action arguments
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--scan", action="store_true", help="Scan for problematic files")
    action_group.add_argument("--mark", action="store_true", help="Mark problematic files with warnings")
    action_group.add_argument("--remove", action="store_true", help="Remove problematic files")
    action_group.add_argument("--archive", action="store_true", help="Archive problematic files")
    action_group.add_argument("--fix-report-py", action="store_true", help="Add validation to report generators")
    
    # Optional arguments
    parser.add_argument("--root-dir", default=".", help="Root directory to scan")
    parser.add_argument("--output", default="stale_report_cleanup_report.md", help="Output report file")
    
    args = parser.parse_args()
    
    # Create cleaner
    cleaner = StaleReportCleaner(root_dir=args.root_dir)
    
    if args.scan:
        # Scan for problematic files
        problematic_files = cleaner.scan_for_problematic_files()
        
        # Generate report
        report_path = cleaner.generate_report(output_file=args.output)
        
        if problematic_files:
            print(f"\nFound {len(problematic_files)} problematic files. Report generated: {report_path}")
        else:
            print("\nNo problematic files found.")
            
    elif args.mark:
        # Scan for problematic files
        cleaner.scan_for_problematic_files()
        
        # Mark problematic files
        count = cleaner.mark_problematic_files()
        
        print(f"\nAdded warnings to {count} problematic files.")
        
    elif args.remove:
        # Scan for problematic files
        cleaner.scan_for_problematic_files()
        
        # Confirm removal
        print(f"\nWARNING: This will remove {len(cleaner.problematic_files)} problematic files.")
        confirmation = input("Are you sure you want to continue? (y/N): ")
        
        if confirmation.lower() == 'y':
            # Remove problematic files
            count = cleaner.remove_problematic_files()
            print(f"\nRemoved {count} problematic files.")
        else:
            print("\nRemoval cancelled.")
            
    elif args.archive:
        # Scan for problematic files
        cleaner.scan_for_problematic_files()
        
        # Archive problematic files
        count = cleaner.archive_problematic_files()
        
        print(f"\nArchived {count} problematic files to {cleaner.archive_dir}.")
        
    elif args.fix_report_py:
        # Fix report generators
        modified_files = cleaner.fix_report_generators()
        
        print(f"\nAdded validation to {len(modified_files)} report generator scripts.")
    
if __name__ == "__main__":
    main()