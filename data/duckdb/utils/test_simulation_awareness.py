#!/usr/bin/env python3
"""
Test Simulation Awareness in Reports

This script verifies that report generators properly check for simulated data
and include appropriate warnings in reports.

Usage:
  python test_simulation_awareness.py
"""

import os
import sys
import json
import logging
from pathlib import Path
import tempfile
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_report():
    """Create a test report with simulation flags"""
    try:
        # Create a temporary report - use benchmark_results since that's one of the directories that's checked
        temp_dir = Path("./benchmark_results/test_simulation")
        temp_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = temp_dir / f"benchmark_report_{timestamp}.md"
        
        # Create the report with simulated hardware mentions
        with open(report_path, "w") as f:
            f.write(f"""# Benchmark Report

Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary

This is a test report mentioning hardware platforms like cuda, rocm, qnn, and webgpu.

## Results

| Model | Hardware | Latency (ms) | Throughput (items/sec) |
|-------|----------|--------------|------------------------|
| bert  | cpu      | 25.3         | 42.1                   |
| bert  | cuda     | 8.2          | 128.7                  |
| bert  | qnn      | 18.5         | 64.3                   |
| bert  | webgpu   | 15.2         | 93.8                   |

""")
        
        logger.info(f"Created test report: {report_path}")
        return str(report_path)
    except Exception as e:
        logger.error(f"Error creating test report: {e}")
        return None

def verify_cleanup_detection(report_path):
    """Verify that cleanup_stale_reports.py detects the report as problematic"""
    try:
        # Import the cleaner class
        from cleanup_stale_reports import StaleReportCleaner
        
        # Create a cleaner instance
        cleaner = StaleReportCleaner(root_dir=".")
        
        # Scan for problematic files
        problematic_files = cleaner.scan_for_problematic_files()
        
        # Check if our report is in the list
        report_found = False
        for file_info in problematic_files:
            if file_info["path"] == report_path:
                report_found = True
                logger.info(f"✅ Report was correctly identified as problematic")
                logger.info(f"  Issue: {file_info['issue']}")
                break
        
        if not report_found:
            logger.warning(f"❌ Report was NOT identified as problematic")
            return False
        
        # Try marking the file
        cleaner.mark_problematic_files()
        
        # Check if the file was marked
        with open(report_path, "r") as f:
            content = f.read()
            
        if "WARNING" in content and "POTENTIALLY MISLEADING DATA" in content:
            logger.info(f"✅ Report was correctly marked with warnings")
            return True
        else:
            logger.warning(f"❌ Report was NOT properly marked with warnings")
            return False
    except Exception as e:
        logger.error(f"Error verifying cleanup detection: {e}")
        return False

def main():
    """Main function"""
    logger.info("Testing simulation awareness in reports")
    
    # Create a test report
    report_path = create_test_report()
    if not report_path:
        return 1
    
    # Verify cleanup tool detection
    success = verify_cleanup_detection(report_path)
    
    # Cleanup
    try:
        Path(report_path).unlink()
        logger.info(f"Cleaned up test report: {report_path}")
    except Exception as e:
        logger.warning(f"Error cleaning up test report: {e}")
    
    if success:
        logger.info("✅ Simulation awareness testing passed")
        return 0
    else:
        logger.error("❌ Simulation awareness testing failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())