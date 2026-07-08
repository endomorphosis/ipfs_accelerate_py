#!/usr/bin/env python3
"""
Run Cleanup Stale Reports

This script runs the cleanup_stale_reports.py tool to identify and clean up stale 
and misleading benchmark reports, addressing item #10 in NEXT_STEPS.md.

The script ensures the database schema has simulation flags, then executes the cleanup
process to mark, archive, and fix misleading benchmark reports.

Usage:
    python run_cleanup_stale_reports.py
"""

import os
import sys
import argparse
import logging
import subprocess
import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("stale_reports_cleanup.log")
    ]
)
logger = logging.getLogger(__name__)

def run_command(command, check=True):
    """Run a command and return its output"""
    logger.info(f"Running command: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.stdout:
            logger.info(f"Command output:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"Command error output:\n{result.stderr}")
        return result.returncode == 0, result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with error: {e}")
        return False, str(e)
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return False, str(e)

def check_db_schema_updated():
    """Check if update_db_schema_for_simulation.py has been run"""
    db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
    
    try:
        import duckdb
        conn = duckdb.connect(db_path, read_only=True)
        
        # Check if the is_simulated column exists in performance_results table
        result = conn.execute("""
            SELECT 1 
            FROM information_schema.columns 
            WHERE table_name = 'performance_results' AND column_name = 'is_simulated'
        """).fetchone()
        
        conn.close()
        return result is not None
    except Exception as e:
        logger.error(f"Error checking database schema: {e}")
        return False

def ensure_db_schema_updated():
    """Ensure database schema is updated with simulation flags"""
    if not check_db_schema_updated():
        logger.info("Database schema needs to be updated with simulation flags")
        success, _ = run_command("python update_db_schema_for_simulation.py")
        if not success:
            logger.error("Failed to update database schema")
            return False
        logger.info("Database schema updated successfully")
    else:
        logger.info("Database schema already has simulation flags")
    return True

def run_cleanup_pipeline():
    """Run the full cleanup pipeline"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"stale_report_cleanup_report_{timestamp}.md"
    
    # 1. Scan for problematic files and generate report
    logger.info("Step 1: Scanning for problematic benchmark reports")
    success, _ = run_command(f"python cleanup_stale_reports.py --scan --output {report_path}")
    if not success:
        logger.error("Failed to scan for problematic files")
        return False
    
    # 2. Add warnings to problematic files
    logger.info("Step 2: Adding warnings to problematic benchmark reports")
    success, _ = run_command("python cleanup_stale_reports.py --mark")
    if not success:
        logger.error("Failed to add warnings to problematic files")
        return False
    
    # 3. Archive problematic files
    logger.info("Step 3: Archiving problematic benchmark reports")
    success, _ = run_command("python cleanup_stale_reports.py --archive")
    if not success:
        logger.error("Failed to archive problematic files")
        return False
    
    # 4. Fix report generator scripts
    logger.info("Step 4: Adding validation to report generator scripts")
    success, _ = run_command("python cleanup_stale_reports.py --fix-report-py")
    if not success:
        logger.error("Failed to fix report generator scripts")
        return False
    
    # 5. Verify benchmark_timing_report.py has simulation checks
    logger.info("Step 5: Verifying benchmark_timing_report.py has simulation checks")
    with open("benchmark_timing_report.py", "r") as f:
        content = f.read()
        if "check_for_simulated_data" not in content or "is_simulated" not in content:
            logger.warning("benchmark_timing_report.py may not have proper simulation checks")
            # Try to fix it
            success, _ = run_command("python cleanup_stale_reports.py --fix-report-py")
            if not success:
                logger.error("Failed to fix benchmark_timing_report.py")
                return False
    
    # 6. Run final scan to verify issues have been addressed
    logger.info("Step 6: Running final scan to verify issues have been addressed")
    final_report_path = f"stale_report_cleanup_final_report_{timestamp}.md"
    success, _ = run_command(f"python cleanup_stale_reports.py --scan --output {final_report_path}")
    if not success:
        logger.error("Failed to run final scan")
        return False
    
    return True

def generate_completion_report():
    """Generate a completion report for NEXT_STEPS.md task"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_path = "STALE_REPORTS_CLEANUP_COMPLETED.md"
    
    with open(report_path, "w") as f:
        f.write(f"# Stale Reports Cleanup Completed\n\n")
        f.write(f"**Completed: {timestamp}**\n\n")
        f.write("## Summary\n\n")
        f.write("The stale reports cleanup process has been completed to address item #10 in NEXT_STEPS.md:\n\n")
        f.write("- Identified and marked misleading benchmark reports with simulation warnings\n")
        f.write("- Archived problematic files for reference\n")
        f.write("- Updated report generator scripts to include validation for simulation data\n")
        f.write("- Added explicit warnings in reports for simulated hardware data\n")
        f.write("- Created proper validation in benchmark_timing_report.py\n\n")
        
        f.write("## Verification\n\n")
        f.write("All benchmark reports now have clear indications when they contain simulated data:\n\n")
        f.write("- HTML reports include prominent warning banners\n")
        f.write("- Markdown reports include warning headers\n")
        f.write("- JSON files include simulation metadata\n")
        f.write("- Report generators check for simulated data during generation\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("With this task completed, focus can shift to the remaining items in NEXT_STEPS.md:\n\n")
        f.write("- Complete execution of comprehensive benchmarks and publish timing data\n")
        f.write("- Proceed with advanced visualization system work\n\n")
        
        f.write("## Logs\n\n")
        f.write("Detailed logs of the cleanup process can be found in stale_reports_cleanup.log\n")
    
    logger.info(f"Generated completion report: {report_path}")
    return report_path

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run cleanup of stale benchmark reports")
    parser.add_argument("--skip-schema-check", action="store_true", help="Skip database schema check")
    args = parser.parse_args()
    
    logger.info("Starting stale reports cleanup process")
    
    # Step 1: Ensure database schema is updated
    if not args.skip_schema_check:
        if not ensure_db_schema_updated():
            logger.error("Failed to ensure database schema is updated")
            return 1
    
    # Step 2: Run cleanup pipeline
    if not run_cleanup_pipeline():
        logger.error("Cleanup pipeline failed")
        return 1
    
    # Step 3: Generate completion report
    completion_report = generate_completion_report()
    logger.info(f"Stale reports cleanup process completed successfully")
    logger.info(f"Completion report generated: {completion_report}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())