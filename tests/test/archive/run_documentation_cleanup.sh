#\!/bin/bash
# Run documentation cleanup and archival process

echo "Starting documentation cleanup and archival process..."
echo "====================================================="

# Step 1: Archive old documentation files
echo "Step 1: Archiving old documentation files..."
python3 archive_old_documentation.py
echo "Documentation archival complete."
echo

# Step 2: Scan for problematic benchmark reports
echo "Step 2: Scanning for problematic benchmark reports..."
python3 cleanup_stale_reports.py --scan
echo

# Step 3: Mark problematic reports with warnings
echo "Step 3: Marking problematic reports with warnings..."
python3 cleanup_stale_reports.py --mark
echo

# Step 4: Archive problematic reports
echo "Step 4: Archiving problematic reports..."
python3 cleanup_stale_reports.py --archive
echo

# Step 5: Check for outdated simulation methods in code
echo "Step 5: Checking for outdated simulation methods in code..."
python3 cleanup_stale_reports.py --check-code
echo

# Step 6: Fix report generator Python files
echo "Step 6: Fixing report generator Python files..."
python3 cleanup_stale_reports.py --fix-report-py
echo

echo "Documentation cleanup and archival process completed successfully."
echo "See archived_documentation_april2025/ and archived_reports_april2025/ for archived files."
echo "See DOCUMENTATION_INDEX.md for updated documentation index."
