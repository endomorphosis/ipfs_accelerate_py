#!/usr/bin/env python3
"""
Tool for detecting, marking, and cleaning up stale benchmark reports.

This script scans for benchmark reports that may contain misleading data
due to simulated results or other issues, and adds clear warnings to them.
"""

import os
import re
import sys
import logging
import argparse
import datetime
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent.parent.parent))


class StaleReportCleaner:
    """
    Tool for detecting, marking, and cleaning up stale benchmark reports.
    """
    
    def __init__(self, search_dir: str = ".", archive_dir: str = "./archived_reports", 
                debug: bool = False, dry_run: bool = False):
        """
        Initialize the stale report cleaner.
        
        Args:
            search_dir: Directory to search for reports
            archive_dir: Directory to move archived reports to
            debug: Enable debug logging
            dry_run: Don't actually modify or move files
        """
        self.search_dir = search_dir
        self.archive_dir = archive_dir
        self.dry_run = dry_run
        
        # Set up logging
        if debug:
            logger.setLevel(logging.DEBUG)
        
        # File patterns to look for
        self.report_patterns = {
            "markdown": ["**/*benchmark*report*.md", "**/*performance*report*.md", 
                       "**/*hardware*report*.md", "**/*result*report*.md",
                       "**/*benchmark*result*.md", "**/*test*result*.md"],
            "html": ["**/*benchmark*report*.html", "**/*performance*report*.html", 
                   "**/*hardware*report*.html", "**/*result*report*.html",
                   "**/*benchmark*result*.html", "**/*test*result*.html"],
            "json": ["**/*benchmark*result*.json", "**/*performance*result*.json", 
                   "**/*hardware*result*.json", "**/*test*result*.json"]
        }
        
        # Warning templates for different file types
        self.warning_templates = {
            "markdown": """
> ⚠️ **WARNING: This report may contain misleading data!** ⚠️
> 
> This benchmark report is potentially stale or contains simulated results. For 
> accurate and up-to-date information, please check the benchmark database using:
> 
> ```bash
> python -m duckdb_api.utils.view_benchmark_results --output benchmark_summary.md
> ```
>
> Marked as potentially stale on: {date}
""",
            "html": """
<div style="padding: 15px; margin: 20px 0; background-color: #FFF3CD; border-left: 5px solid #FFC107; color: #856404;">
  <h3 style="margin-top: 0;">⚠️ WARNING: This report may contain misleading data! ⚠️</h3>
  <p>This benchmark report is potentially stale or contains simulated results. For 
  accurate and up-to-date information, please check the benchmark database using:</p>
  <pre style="background-color: #F8F9FA; padding: 10px; border-radius: 4px;">python -m duckdb_api.utils.view_benchmark_results --output benchmark_summary.html</pre>
  <p>Marked as potentially stale on: {date}</p>
</div>
""",
            "json": {
                "warning": "⚠️ WARNING: This JSON file may contain misleading data! ⚠️",
                "description": "This benchmark result file is potentially stale or contains simulated results. For accurate and up-to-date information, please check the benchmark database.",
                "recommendation": "python -m duckdb_api.utils.view_benchmark_results --format json --output benchmark_summary.json",
                "marked_stale_on": "{date}"
            }
        }
        
        logger.info(f"Initialized StaleReportCleaner with search directory: {search_dir}")
        
    def scan_for_reports(self) -> Dict[str, List[Path]]:
        """
        Scan for benchmark reports in the search directory.
        
        Returns:
            Dictionary mapping file types to lists of report file paths
        """
        search_path = Path(self.search_dir)
        
        if not search_path.exists():
            logger.error(f"Search directory does not exist: {self.search_dir}")
            return {}
        
        reports = {
            "markdown": [],
            "html": [],
            "json": []
        }
        
        # Scan for each file type
        for file_type, patterns in self.report_patterns.items():
            for pattern in patterns:
                # Find matching files
                matching_files = list(search_path.glob(pattern))
                
                # Add to list, avoiding duplicates
                for file_path in matching_files:
                    if file_path not in reports[file_type]:
                        reports[file_type].append(file_path)
        
        # Log counts
        for file_type, file_list in reports.items():
            logger.info(f"Found {len(file_list)} {file_type} report files")
            
        return reports
    
    def check_if_marked(self, file_path: Path, file_type: str) -> bool:
        """
        Check if a report file is already marked as potentially stale.
        
        Args:
            file_path: Path to the report file
            file_type: Type of file (markdown, html, json)
            
        Returns:
            True if file is already marked, False otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                if file_type == "markdown":
                    return "⚠️ **WARNING: This report may contain misleading data!** ⚠️" in content
                elif file_type == "html":
                    return "⚠️ WARNING: This report may contain misleading data! ⚠️" in content
                elif file_type == "json":
                    # For JSON, check if the warning field exists
                    import json
                    try:
                        data = json.loads(content)
                        return isinstance(data, dict) and data.get("warning") == self.warning_templates["json"]["warning"]
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON file: {file_path}")
                        return False
                        
        except FileNotFoundError as e:
            logger.error(f"File not found: {file_path}")
            logger.error(f"Error details: {e}")
            return False
        except PermissionError as e:
            logger.error(f"Permission denied when accessing file: {file_path}")
            logger.error(f"Error details: {e}")
            return False
        except UnicodeDecodeError as e:
            logger.error(f"Unicode decoding error for file: {file_path}")
            logger.error(f"Error details: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking if file is marked: {file_path}")
            logger.error(f"Error details: {e}")
            return False
    
    def mark_report(self, file_path: Path, file_type: str) -> bool:
        """
        Mark a report file as potentially stale.
        
        Args:
            file_path: Path to the report file
            file_type: Type of file (markdown, html, json)
            
        Returns:
            True if marking was successful, False otherwise
        """
        if self.dry_run:
            logger.info(f"[DRY RUN] Would mark report as potentially stale: {file_path}")
            return True
        
        try:
            # Get date string
            date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if file_type == "markdown":
                # Add warning at the top of the file
                warning = self.warning_templates["markdown"].format(date=date_str)
                new_content = warning + "\n" + content
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                    
            elif file_type == "html":
                # Add warning after <body> tag
                warning = self.warning_templates["html"].format(date=date_str)
                
                if "<body" in content:
                    # Use raw string for regex pattern to avoid escaping issues
                    body_tag_pattern = r"<body[^>]*>"
                    
                    # Try to import string utils for better regex handling
                    try:
                        sys.path.append(str(Path(__file__).parent.parent.parent))
                        from fixed_web_platform.unified_framework.string_utils import is_valid_regex
                        
                        # Validate regex pattern
                        if not is_valid_regex(body_tag_pattern):
                            logger.error(f"Invalid regex pattern: {body_tag_pattern}")
                            return False
                    except ImportError:
                        # Continue without pattern validation
                        pass
                    
                    # Find where body tag ends
                    body_end_match = re.search(body_tag_pattern, content)
                    if body_end_match:
                        body_end = body_end_match.end()
                        new_content = content[:body_end] + warning + content[body_end:]
                        
                        # Use proper error handling for file operations
                        try:
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(new_content)
                        except (IOError, PermissionError) as e:
                            logger.error(f"Error writing to file {file_path}: {e}")
                            return False
                    else:
                        logger.warning(f"Could not find where body tag ends in HTML file: {file_path}")
                        return False
                else:
                    # Just add at the top
                    new_content = warning + content
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    
            elif file_type == "json":
                import json
                try:
                    # Parse JSON
                    data = json.loads(content)
                    
                    # Add warning fields
                    warnings = dict(self.warning_templates["json"])
                    warnings["marked_stale_on"] = date_str
                    
                    # If the data is a dict, add warnings directly
                    if isinstance(data, dict):
                        # Create new data with warning fields first
                        new_data = {**warnings, **data}
                        
                        # Write back to file
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(new_data, f, indent=2)
                    else:
                        # For non-dict data, wrap in a dict with warnings
                        new_data = {
                            **warnings,
                            "data": data
                        }
                        
                        # Write back to file
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(new_data, f, indent=2)
                    
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON file: {file_path}")
                    return False
            
            logger.info(f"Marked report as potentially stale: {file_path}")
            return True
            
        except FileNotFoundError as e:
            logger.error(f"File not found when marking report: {file_path}")
            logger.error(f"Error details: {e}")
            return False
        except PermissionError as e:
            logger.error(f"Permission denied when marking report: {file_path}")
            logger.error(f"Error details: {e}")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error when marking report: {file_path}")
            logger.error(f"Error details: {e}")
            return False
        except UnicodeDecodeError as e:
            logger.error(f"Unicode decode error when marking report: {file_path}")
            logger.error(f"Error details: {e}")
            return False
        except IOError as e:
            logger.error(f"I/O error when marking report: {file_path}")
            logger.error(f"Error details: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error marking report: {file_path}")
            logger.error(f"Error details: {e}")
            return False
    
    def archive_report(self, file_path: Path) -> bool:
        """
        Archive a report file by moving it to the archive directory.
        
        Args:
            file_path: Path to the report file
            
        Returns:
            True if archiving was successful, False otherwise
        """
        if self.dry_run:
            logger.info(f"[DRY RUN] Would archive report: {file_path}")
            return True
        
        # Make sure archive directory exists
        archive_path = Path(self.archive_dir)
        archive_path.mkdir(parents=True, exist_ok=True)
        
        # Construct destination path with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        dest_path = archive_path / dest_name
        
        try:
            # Move file to archive
            shutil.move(str(file_path), str(dest_path))
            logger.info(f"Archived report to: {dest_path}")
            return True
            
        except FileNotFoundError as e:
            logger.error(f"File not found when archiving report: {file_path}")
            logger.error(f"Error details: {e}")
            return False
        except PermissionError as e:
            logger.error(f"Permission denied when archiving report: {file_path}")
            logger.error(f"Error details: {e}")
            return False
        except shutil.Error as e:
            logger.error(f"Shutil error when archiving report: {file_path}")
            logger.error(f"Error details: {e}")
            return False
        except IOError as e:
            logger.error(f"I/O error when archiving report: {file_path}")
            logger.error(f"Error details: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error archiving report: {file_path}")
            logger.error(f"Error details: {e}")
            return False
    
    def fix_report_generator(self, file_path: Path) -> bool:
        """
        Fix a report generator script to include simulation validation.
        
        Args:
            file_path: Path to the report generator script
            
        Returns:
            True if fixing was successful, False otherwise
        """
        if self.dry_run:
            logger.info(f"[DRY RUN] Would fix report generator: {file_path}")
            return True
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check if already fixed
            if "_validate_data_authenticity" in content:
                logger.info(f"Report generator already includes validation: {file_path}")
                return True
                
            # Function to add to the script
            validation_function = """
    def _validate_data_authenticity(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"
        Validate the authenticity of benchmark data by checking simulation status.
        
        Args:
            data: DataFrame with benchmark data
            
        Returns:
            DataFrame with added validation columns
        \"\"\"
        # Check if 'is_simulated' column exists
        if 'is_simulated' not in data.columns:
            # Add is_simulated column with default value False
            data['is_simulated'] = False
            data['simulation_reason'] = None
            data['data_authenticity'] = "✅ Real"
        else:
            # Add data authenticity column
            data['data_authenticity'] = data['is_simulated'].apply(
                lambda is_sim: "⚠️ Simulated" if is_sim else "✅ Real"
            )
            
        return data
"""
            
            # Find appropriate place to add the function
            if "class " in content:
                # Add after the last method in the class
                class_matches = list(re.finditer(r"class\s+\w+", content))
                if class_matches:
                    last_class_match = class_matches[-1]
                    
                    # Find all 'def ' lines
                    method_matches = list(re.finditer(r"    def\s+\w+", content[last_class_match.start():]))
                    
                    if method_matches:
                        last_method_match = method_matches[-1]
                        
                        # Find the end of the method by looking for the next method or the end of the class
                        last_method_start = last_class_match.start() + last_method_match.start()
                        
                        # Find next method or end of file
                        next_method_match = re.search(r"    def\s+\w+", content[last_method_start+1:])
                        next_class_match = re.search(r"class\s+\w+", content[last_method_start+1:])
                        
                        if next_method_match and (not next_class_match or next_method_match.start() < next_class_match.start()):
                            insert_pos = last_method_start + 1 + next_method_match.start()
                        elif next_class_match:
                            insert_pos = last_method_start + 1 + next_class_match.start()
                        else:
                            # End of file
                            insert_pos = len(content)
                        
                        # Insert validation function
                        new_content = content[:insert_pos] + validation_function + content[insert_pos:]
                        
                        # Add validation call in generate_report or similar method
                        if "generate_report" in content:
                            # Find generate_report method
                            report_match = re.search(r"    def\s+generate_report", content)
                            if report_match:
                                report_method_start = report_match.start()
                                
                                # Find where data is processed
                                data_processing_match = re.search(r"(data|df|results)\s*=", content[report_method_start:])
                                if data_processing_match:
                                    data_processing_line_end = content[report_method_start:].find("\n", data_processing_match.end()) + report_method_start
                                    
                                    # Add validation call after data processing
                                    validation_call = "\n        # Validate data authenticity\n        data = self._validate_data_authenticity(data)\n"
                                    new_content = new_content[:data_processing_line_end] + validation_call + new_content[data_processing_line_end:]
                                    
                        # Write back to file
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        
                        logger.info(f"Fixed report generator: {file_path}")
                        return True
            
            logger.warning(f"Could not find appropriate place to add validation to: {file_path}")
            return False
            
        except FileNotFoundError as e:
            logger.error(f"File not found when fixing report generator: {file_path}")
            logger.error(f"Error details: {e}")
            return False
        except PermissionError as e:
            logger.error(f"Permission denied when fixing report generator: {file_path}")
            logger.error(f"Error details: {e}")
            return False
        except UnicodeDecodeError as e:
            logger.error(f"Unicode decode error when fixing report generator: {file_path}")
            logger.error(f"Error details: {e}")
            return False
        except re.error as e:
            logger.error(f"Regex error when fixing report generator: {file_path}")
            logger.error(f"Error details: {e}")
            return False
        except IOError as e:
            logger.error(f"I/O error when fixing report generator: {file_path}")
            logger.error(f"Error details: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error fixing report generator: {file_path}")
            logger.error(f"Error details: {e}")
            return False
    
    def fix_all_report_generators(self) -> Tuple[int, int]:
        """
        Find and fix all report generator scripts.
        
        Returns:
            Tuple of (total scripts found, successfully fixed count)
        """
        search_path = Path(self.search_dir)
        
        # Find all Python files that might be report generators
        py_patterns = ["**/*report*.py", "**/*benchmark*report*.py", "**/*generate*report*.py",
                      "**/*visualization*/*.py", "**/*visualizer*.py"]
        
        report_scripts = set()
        for pattern in py_patterns:
            for file_path in search_path.glob(pattern):
                report_scripts.add(file_path)
        
        logger.info(f"Found {len(report_scripts)} potential report generator scripts")
        
        # Filter scripts that actually generate reports
        actual_report_scripts = []
        for script_path in report_scripts:
            try:
                with open(script_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Check if script generates reports by looking for key patterns
                    if re.search(r"(generate|create|build)_(report|visualization|chart)", content) or \
                       re.search(r"(to_(html|markdown|md|csv|json)|savefig)", content):
                        actual_report_scripts.append(script_path)
            except Exception:
                pass
        
        logger.info(f"Found {len(actual_report_scripts)} actual report generator scripts")
        
        # Fix each script
        fixed_count = 0
        for script_path in actual_report_scripts:
            if self.fix_report_generator(script_path):
                fixed_count += 1
        
        return len(actual_report_scripts), fixed_count
    
    def clean_reports(self) -> Tuple[int, int, int]:
        """
        Scan for reports, mark them as potentially stale, and optionally archive them.
        
        Returns:
            Tuple of (total reports found, marked count, archived count)
        """
        # Scan for reports
        reports = self.scan_for_reports()
        
        total_count = sum(len(file_list) for file_list in reports.values())
        
        if total_count == 0:
            logger.info("No reports found")
            return 0, 0, 0
        
        marked_count = 0
        archived_count = 0
        
        # Process each file type
        for file_type, file_list in reports.items():
            for file_path in file_list:
                # Check if already marked
                if self.check_if_marked(file_path, file_type):
                    logger.info(f"Report already marked as potentially stale: {file_path}")
                    continue
                
                # Mark as potentially stale
                if self.mark_report(file_path, file_type):
                    marked_count += 1
                    
        return total_count, marked_count, archived_count
    
    def archive_all_reports(self) -> Tuple[int, int]:
        """
        Archive all reports found.
        
        Returns:
            Tuple of (total reports found, archived count)
        """
        # Scan for reports
        reports = self.scan_for_reports()
        
        total_count = sum(len(file_list) for file_list in reports.values())
        
        if total_count == 0:
            logger.info("No reports found")
            return 0, 0
        
        archived_count = 0
        
        # Process each file type
        for file_type, file_list in reports.items():
            for file_path in file_list:
                # Archive file
                if self.archive_report(file_path):
                    archived_count += 1
                    
        return total_count, archived_count


def main():
    """Command-line interface for the stale report cleaner."""
    parser = argparse.ArgumentParser(description="Stale Report Cleaner")
    parser.add_argument("--search-dir", default=".",
                       help="Directory to search for reports")
    parser.add_argument("--archive-dir", default="./archived_reports",
                       help="Directory to move archived reports to")
    parser.add_argument("--scan", action="store_true",
                       help="Scan for reports but don't modify them")
    parser.add_argument("--mark", action="store_true",
                       help="Mark reports as potentially stale")
    parser.add_argument("--archive", action="store_true",
                       help="Archive reports after marking them")
    parser.add_argument("--fix-report-py", action="store_true",
                       help="Fix report generator scripts to include validation")
    parser.add_argument("--dry-run", action="store_true",
                       help="Don't actually modify or move files")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    args = parser.parse_args()
    
    # Create cleaner
    cleaner = StaleReportCleaner(
        search_dir=args.search_dir,
        archive_dir=args.archive_dir,
        debug=args.debug,
        dry_run=args.dry_run
    )
    
    # Process reports
    if args.scan:
        reports = cleaner.scan_for_reports()
        total_count = sum(len(file_list) for file_list in reports.values())
        logger.info(f"Found {total_count} total reports")
        
        for file_type, file_list in reports.items():
            if file_list:
                print(f"\n{file_type.upper()} Reports:")
                for file_path in file_list:
                    marked = cleaner.check_if_marked(file_path, file_type)
                    status = "MARKED" if marked else "UNMARKED"
                    print(f"  {file_path} [{status}]")
                    
    elif args.mark:
        total_count, marked_count, archived_count = cleaner.clean_reports()
        logger.info(f"Found {total_count} total reports, marked {marked_count} as potentially stale")
        
        if args.archive:
            logger.info(f"Archived {archived_count} reports")
            
    elif args.archive:
        total_count, archived_count = cleaner.archive_all_reports()
        logger.info(f"Found {total_count} total reports, archived {archived_count}")
        
    elif args.fix_report_py:
        total_count, fixed_count = cleaner.fix_all_report_generators()
        logger.info(f"Found {total_count} report generator scripts, fixed {fixed_count}")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()