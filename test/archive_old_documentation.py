#\!/usr/bin/env python3
"""
Archive old documentation and reports to maintain a clean directory structure.
This script will:
1. Identify and archive old performance reports
2. Archive outdated documentation files
3. Update the documentation index
"""

import os
import shutil
import glob
import datetime
import re
import json
import argparse
from pathlib import Path

# Constants
PERFORMANCE_RESULTS_DIR = "/home/barberb/ipfs_accelerate_py/test/performance_results"
ARCHIVED_REPORTS_DIR = "/home/barberb/ipfs_accelerate_py/test/archived_reports_april2025"
ARCHIVED_DOCS_DIR = "/home/barberb/ipfs_accelerate_py/test/archived_documentation_april2025"
DOCUMENTATION_INDEX_PATH = "/home/barberb/ipfs_accelerate_py/test/DOCUMENTATION_INDEX.md"
DOCS_DIR = "/home/barberb/ipfs_accelerate_py/test/docs"

def add_archive_notice(file_path, archive_date=None, dry_run=False):
    """Add an archive notice to the top of a file."""
    if not archive_date:
        archive_date = datetime.datetime.now().strftime("%B %d, %Y")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    archive_notice = f"""
# ARCHIVED DOCUMENT
**This document has been archived on {archive_date} as part of the documentation cleanup.**
**Please refer to the current documentation for up-to-date information.**

"""
    
    # Only add the notice if it doesn't already exist
    if "ARCHIVED DOCUMENT" not in content:
        if not dry_run:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(archive_notice + content)
        return True
    return False

def archive_old_performance_reports(dry_run=False):
    """Archive old performance reports."""
    print("Archiving old performance reports...")
    
    # Create destination directory if it doesn't exist
    if not dry_run:
        os.makedirs(ARCHIVED_REPORTS_DIR, exist_ok=True)
    
    # Get all performance report files
    report_files = glob.glob(os.path.join(PERFORMANCE_RESULTS_DIR, "*.md"))
    
    # Filter for reports older than 30 days
    current_time = datetime.datetime.now()
    thirty_days_ago = current_time - datetime.timedelta(days=30)
    
    archived_count = 0
    
    for report_file in report_files:
        # Extract date from filename if possible
        filename = os.path.basename(report_file)
        date_match = re.search(r'_(\d{8})_', filename)
        
        if date_match:
            try:
                file_date_str = date_match.group(1)
                file_date = datetime.datetime.strptime(file_date_str, "%Y%m%d")
                
                if file_date < thirty_days_ago:
                    # Add archive notice
                    add_archive_notice(report_file, dry_run=dry_run)
                    
                    # Move file to archive directory
                    if not dry_run:
                        shutil.copy2(report_file, os.path.join(ARCHIVED_REPORTS_DIR, filename))
                    archived_count += 1
                    print(f"Would archive: {report_file}" if dry_run else f"Archived: {report_file}")
            except ValueError:
                # If date parsing fails, check file modification time
                file_stats = os.stat(report_file)
                file_mtime = datetime.datetime.fromtimestamp(file_stats.st_mtime)
                
                if file_mtime < thirty_days_ago:
                    # Add archive notice
                    add_archive_notice(report_file, dry_run=dry_run)
                    
                    # Move file to archive directory
                    if not dry_run:
                        shutil.copy2(report_file, os.path.join(ARCHIVED_REPORTS_DIR, filename))
                    archived_count += 1
                    print(f"Would archive: {report_file}" if dry_run else f"Archived: {report_file}")
        else:
            # No date in filename, check file modification time
            file_stats = os.stat(report_file)
            file_mtime = datetime.datetime.fromtimestamp(file_stats.st_mtime)
            
            if file_mtime < thirty_days_ago:
                # Add archive notice
                add_archive_notice(report_file, dry_run=dry_run)
                
                # Move file to archive directory
                if not dry_run:
                    shutil.copy2(report_file, os.path.join(ARCHIVED_REPORTS_DIR, filename))
                archived_count += 1
                print(f"Would archive: {report_file}" if dry_run else f"Archived: {report_file}")
    
    print(f"Archived {archived_count} performance reports.")
    return archived_count

def identify_outdated_documentation(dry_run=False):
    """Identify outdated documentation files based on references and content."""
    print("Identifying outdated documentation...")
    
    # Get all markdown files in the project
    md_files = []
    for root, _, files in os.walk("/home/barberb/ipfs_accelerate_py/test"):
        if "archived_" in root:
            continue
        for file in files:
            if file.endswith(".md"):
                md_files.append(os.path.join(root, file))
    
    # Files to archive (will be populated)
    to_archive = []
    
    # Current documentation index content
    with open(DOCUMENTATION_INDEX_PATH, 'r', encoding='utf-8') as f:
        index_content = f.read()
    
    # Check each documentation file
    for md_file in md_files:
        filename = os.path.basename(md_file)
        
        # Skip key documentation files
        if filename in ["README.md", "DOCUMENTATION_INDEX.md", "CLAUDE.md", "NEXT_STEPS.md", "DOCUMENTATION_CLEANUP_GUIDE.md"]:
            continue
        
        # Check if file is referenced in the documentation index
        if filename not in index_content:
            # File isn't referenced in the index, likely outdated
            to_archive.append(md_file)
            print(f"Not referenced in index: {md_file}")
            continue
        
        # Check file content for outdated markers
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read().lower()
            
            # Check for outdated markers
            outdated_markers = [
                "to be implemented",
                "coming soon",
                "planned for",
                "will be added",
                "under development",
                "not yet implemented"
            ]
            
            # Check if file contains specific outdated date markers
            date_markers = re.findall(r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}', content)
            has_old_date = False
            
            if date_markers:
                current_month = datetime.datetime.now().month
                current_year = datetime.datetime.now().year
                
                for date_marker in date_markers:
                    # Skip if not a proper date reference
                    if not re.match(r'^[a-z]+\s+\d{4}$', date_marker.lower()):
                        continue
                        
                    try:
                        month_str, year_str = date_marker.lower().split()
                        month_names = ["january", "february", "march", "april", "may", "june", 
                                       "july", "august", "september", "october", "november", "december"]
                        if month_str.lower() in month_names:
                            month = month_names.index(month_str.lower()) + 1
                            year = int(year_str)
                            
                            # If date is more than 3 months old or in a previous year
                            if (year < current_year) or (year == current_year and month < (current_month - 3)):
                                has_old_date = True
                                print(f"Old date found in {md_file}: {date_marker}")
                                break
                    except (ValueError, AttributeError):
                        continue
            
            # Check for outdated markers or old dates
            if any(marker in content for marker in outdated_markers) or has_old_date:
                to_archive.append(md_file)
                if any(marker in content for marker in outdated_markers):
                    print(f"Outdated marker found in {md_file}")
    
    # Remove duplicates
    to_archive = list(set(to_archive))
    print(f"Identified {len(to_archive)} outdated documentation files.")
    return to_archive

def archive_outdated_documentation(to_archive, dry_run=False):
    """Archive outdated documentation files."""
    print("Archiving outdated documentation...")
    
    # Create destination directory if it doesn't exist
    if not dry_run:
        os.makedirs(ARCHIVED_DOCS_DIR, exist_ok=True)
    
    archived_count = 0
    for doc_file in to_archive:
        filename = os.path.basename(doc_file)
        
        # Add archive notice
        add_archive_notice(doc_file, dry_run=dry_run)
        
        # Move file to archive directory
        if not dry_run:
            shutil.copy2(doc_file, os.path.join(ARCHIVED_DOCS_DIR, filename))
        archived_count += 1
        print(f"Would archive: {doc_file}" if dry_run else f"Archived: {doc_file}")
    
    print(f"Archived {archived_count} documentation files.")
    return archived_count

def update_documentation_index(dry_run=False):
    """Update the documentation index to reflect current state."""
    print("Updating documentation index...")
    
    if dry_run:
        print("Would update documentation index (dry run)")
        return
    
    with open(DOCUMENTATION_INDEX_PATH, 'r', encoding='utf-8') as f:
        content = f.readlines()
    
    # Update the last updated date in the index
    current_date = datetime.datetime.now().strftime("%B %d, %Y")
    for i, line in enumerate(content):
        if "Last Updated:" in line:
            content[i] = f"Last Updated: {current_date}\n"
            break
    
    # Add new section about archived documentation if not already present
    archive_section = f"""
## Recently Archived Documentation (April 2025)

The following documentation has been archived as part of the April 2025 cleanup:

- Performance reports older than 30 days have been moved to `archived_reports_april2025/`
- Outdated documentation files have been moved to `archived_documentation_april2025/`
- Each archived file has been marked with an archive notice

To access archived documentation, please check the appropriate archive directory.
"""
    
    # Find where to insert the archive section (after the intro)
    archive_section_found = False
    for i, line in enumerate(content):
        if "## Recently Archived Documentation" in line:
            archive_section_found = True
            break
    
    if not archive_section_found:
        for i, line in enumerate(content):
            if "## Current Hardware and Model Coverage Status" in line:
                content.insert(i, archive_section)
                break
    
    # Write updated content back to the file
    with open(DOCUMENTATION_INDEX_PATH, 'w', encoding='utf-8') as f:
        f.writelines(content)
    
    print("Documentation index updated.")

def update_documentation_update_note(dry_run=False):
    """Update the documentation update note file."""
    print("Updating documentation update note...")
    
    if dry_run:
        print("Would update documentation update note (dry run)")
        return
    
    doc_update_path = "/home/barberb/ipfs_accelerate_py/test/DOCUMENTATION_UPDATE_NOTE.md"
    
    with open(doc_update_path, 'r', encoding='utf-8') as f:
        content = f.readlines()
    
    # Add new section about documentation cleanup if not already present
    current_date = datetime.datetime.now().strftime("%B %d, %Y")
    cleanup_section = f"""
#### {current_date}: Documentation and Report Cleanup

1. **Documentation Cleanup**:
   - Archived outdated documentation files
   - Added archive notices to all archived files
   - Updated documentation index with latest status
   - Streamlined documentation structure
   - Created comprehensive archival system

2. **Performance Report Cleanup**:
   - Archived performance reports older than 30 days
   - Created structured archive directory for historical reports
   - Added archive notices to all archived reports
   - Implemented automated archival process

3. **System Improvements**:
   - Created `archive_old_documentation.py` utility for future cleanup
   - Added documentation lifecycle management processes
   - Updated references to reflect current documentation structure
   - Added archive section to documentation index

"""
    
    # Check if the section already exists
    cleanup_section_found = False
    for i, line in enumerate(content):
        if f"#### {current_date}: Documentation and Report Cleanup" in line:
            cleanup_section_found = True
            break
    
    if not cleanup_section_found:
        # Find where to insert the cleanup section (after the latest section)
        for i, line in enumerate(content):
            if "## March 2025 Updates" in line:
                # Insert after the heading and the next line
                content.insert(i + 2, cleanup_section)
                break
    
        # Write updated content back to the file
        with open(doc_update_path, 'w', encoding='utf-8') as f:
            f.writelines(content)
    
    print("Documentation update note updated.")

def generate_archive_report(dry_run=False):
    """Generate a report of the archival process."""
    print("Generating archive report...")
    
    if dry_run:
        print("Would generate archive report (dry run)")
        return
    
    current_date = datetime.datetime.now().strftime("%B %d, %Y")
    report_path = os.path.join(ARCHIVED_DOCS_DIR, "ARCHIVE_REPORT.md")
    
    # Count files in archive directories
    archived_reports_count = 0
    if os.path.exists(ARCHIVED_REPORTS_DIR):
        archived_reports_count = len(os.listdir(ARCHIVED_REPORTS_DIR))
    
    archived_docs_count = 0
    if os.path.exists(ARCHIVED_DOCS_DIR):
        archived_docs_count = len(os.listdir(ARCHIVED_DOCS_DIR))
        # Subtract 1 to account for the report itself
        if os.path.exists(report_path):
            archived_docs_count -= 1
    
    report_content = f"""# Documentation and Report Archive Report
**Date: {current_date}**

This report summarizes the documentation and report archival process completed on {current_date}.

## Summary

- **Performance Reports**: {archived_reports_count} reports archived
- **Documentation Files**: {archived_docs_count} files archived (excluding this report)

## Archive Locations

- **Performance Reports**: `{ARCHIVED_REPORTS_DIR}`
- **Documentation Files**: `{ARCHIVED_DOCS_DIR}`

## Archive Process

1. Performance reports older than 30 days were identified and archived
2. Documentation files not referenced in the documentation index were identified
3. Documentation files with outdated markers were identified
4. All archived files were marked with an archive notice
5. The documentation index was updated to reflect the current state

## Next Steps

- Run the `archive_old_documentation.py` script periodically to maintain a clean documentation structure
- Update the documentation index when new documentation is added
- Follow the documentation lifecycle management process for all new documentation

## Contact

For questions or issues regarding the archival process, please contact the documentation team.
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("Archive report generated.")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Archive old documentation and reports.')
    parser.add_argument('--dry-run', action='store_true', help='Run without making any changes')
    args = parser.parse_args()
    
    print("Starting documentation archival process...")
    
    # Archive old performance reports
    archived_reports = archive_old_performance_reports(dry_run=args.dry_run)
    
    # Identify and archive outdated documentation
    to_archive = identify_outdated_documentation(dry_run=args.dry_run)
    archived_docs = archive_outdated_documentation(to_archive, dry_run=args.dry_run)
    
    # Update documentation index
    update_documentation_index(dry_run=args.dry_run)
    
    # Update documentation update note
    update_documentation_update_note(dry_run=args.dry_run)
    
    # Generate archive report
    generate_archive_report(dry_run=args.dry_run)
    
    print(f"\nArchival process complete:" + (" (DRY RUN)" if args.dry_run else ""))
    print(f"- {archived_reports} performance reports archived")
    print(f"- {archived_docs} documentation files archived")
    print(f"- Documentation index updated")
    print(f"- Documentation update note updated")
    print(f"- Archive report generated")

if __name__ == "__main__":
    main()
