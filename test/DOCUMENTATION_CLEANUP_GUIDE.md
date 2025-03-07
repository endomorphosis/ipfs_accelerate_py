# Documentation and Report Cleanup Guide

**Date: April 7, 2025**
**Status: ACTIVE**

This guide describes the process for cleaning up and maintaining the documentation and benchmark reports in the IPFS Accelerate Python Framework.

## Overview

The documentation and report cleanup process involves:

1. Identifying and archiving outdated documentation files
2. Scanning for problematic benchmark reports
3. Adding warnings to reports containing simulated data
4. Archiving problematic files
5. Checking for outdated simulation methods in code
6. Fixing report generator Python files to include simulation validation

## Cleanup Tools

The following tools are available for documentation and report cleanup:

### 1. Documentation Archival Tool

The `archive_old_documentation.py` script automates the process of identifying and archiving outdated documentation files.

```bash
# Archive old documentation files
python archive_old_documentation.py
```

This script:
- Identifies documentation files not referenced in the documentation index
- Identifies documentation files with outdated markers like "to be implemented", "coming soon", etc.
- Archives identified files to the `archived_documentation_april2025/` directory
- Adds archive notices to all archived files
- Updates the documentation index with information about archived documentation
- Generates an archive report with statistics

### 2. Stale Reports Cleanup Tool

The `cleanup_stale_reports.py` script scans for, marks, and archives problematic benchmark reports.

```bash
# Scan for problematic reports without modifying them
python cleanup_stale_reports.py --scan

# Add warnings to problematic reports
python cleanup_stale_reports.py --mark

# Archive problematic files
python cleanup_stale_reports.py --archive

# Check for outdated simulation methods in Python code
python cleanup_stale_reports.py --check-code

# Fix report generator Python files to include validation
python cleanup_stale_reports.py --fix-report-py
```

This script:
- Scans for benchmark reports that might contain misleading data
- Adds simulation warnings to reports containing simulated data
- Archives problematic files to the `archived_stale_reports/` directory
- Checks Python files for outdated simulation methods
- Fixes report generator Python files to include validation

### 3. Combined Cleanup Script

The `run_documentation_cleanup.sh` script runs both tools with appropriate options.

```bash
# Run the complete documentation cleanup process
./run_documentation_cleanup.sh
```

This script:
1. Archives old documentation files
2. Scans for problematic benchmark reports
3. Marks problematic reports with warnings
4. Archives problematic reports
5. Checks for outdated simulation methods in code
6. Fixes report generator Python files

## When to Run Cleanup

The documentation and report cleanup process should be run:

1. **Monthly Cleanup**: Run the combined cleanup script once a month to maintain a clean documentation structure
2. **After Major Releases**: Run after each major release to ensure documentation reflects the current state
3. **Before Documentation Updates**: Run before making major documentation updates to ensure you're working with the latest state
4. **After Adding New Benchmark Reports**: Run after adding new benchmark reports to ensure proper simulation warnings

## Guidelines for Documentation Maintenance

### General Guidelines

1. **Update the Documentation Index**: When adding new documentation, always update the documentation index
2. **Archive Outdated Documentation**: When documentation becomes outdated, use the archival process instead of deleting
3. **Mark Known Simulated Data**: Always clearly mark known simulated data in reports
4. **Fix Report Generators**: When adding new report generators, ensure they include simulation validation
5. **Follow Naming Conventions**: Use descriptive names and follow the established naming conventions

### Archiving Guidelines

1. **Add Archive Notice**: All archived files should have an archive notice at the top
2. **Preserve Directory Structure**: Maintain the original directory structure when archiving
3. **Generate Archive Report**: Document what was archived and why
4. **Update References**: Update any references to the archived files

### Report Generation Guidelines

1. **Include Simulation Validation**: All report generators should validate simulation status
2. **Clear Warnings**: Add clear warnings for reports containing simulated data
3. **Explicit Simulation Markers**: Use explicit markers for simulated data points
4. **Database Integration**: Ensure simulation flags are stored in the database

## Future Improvements

The following improvements are planned for the documentation and report cleanup process:

1. **Automated Monthly Cleanup**: Set up a scheduled job to run the cleanup process monthly
2. **Integration with CI/CD Pipeline**: Automatically check for simulation warnings in new reports
3. **Documentation Versioning**: Implement proper versioning for documentation files
4. **Interactive Dashboard**: Create a dashboard showing documentation and report status
5. **Quality Metrics**: Implement quality metrics for documentation and reports

## Conclusion

The documentation and report cleanup process ensures that users always have access to accurate and up-to-date information. By following this guide, you can maintain a clean and reliable documentation structure, ensuring that all reports clearly distinguish between real hardware results and simulated data.

For questions or issues regarding the documentation cleanup process, please file an issue in the project repository.</content>
</invoke>