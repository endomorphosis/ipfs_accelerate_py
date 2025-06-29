# Simulation Detection and Reporting Improvements

**Last Updated: April 6, 2025**

This document details the enhancements made to the benchmark system for clear detection, marking, and reporting of simulated hardware results vs. real hardware results.

## Problem Statement

Prior to these improvements, the benchmark system had several issues:

1. **Unclear Simulation Status**: Benchmark results did not clearly indicate when hardware was being simulated
2. **Misleading Reports**: Reports could present simulated data as if it were real hardware measurements
3. **No Database Tracking**: The database schema did not include fields to track simulation status
4. **Inconsistent Detection**: Hardware detection methods varied and did not always indicate simulation
5. **Report Ambiguity**: Users couldn't determine if reports contained real or simulated data

## Solution Overview

We implemented a comprehensive system for tracking and clearly indicating when benchmark results are using simulated hardware:

1. **Database Schema Enhancements**: Added simulation flags and reason fields to all relevant tables
2. **Hardware Detection Improvements**: Enhanced hardware detection to clearly track simulation status
3. **Report Validation**: Added validation to all report generators to verify data authenticity
4. **Stale Report Cleanup**: Created tools to identify and mark existing problematic reports
5. **Verification Tools**: Added utilities to check and fix simulation status in the database

## Database Schema Enhancements

### New Columns Added

```sql
-- Added to performance_results table
ALTER TABLE performance_results
ADD COLUMN is_simulated BOOLEAN DEFAULT FALSE,
ADD COLUMN simulation_reason VARCHAR;

-- Added to test_results table
ALTER TABLE test_results
ADD COLUMN is_simulated BOOLEAN DEFAULT FALSE,
ADD COLUMN simulation_reason VARCHAR;

-- Added to hardware_platforms table
ALTER TABLE hardware_platforms
ADD COLUMN is_simulated BOOLEAN DEFAULT FALSE,
ADD COLUMN simulation_reason VARCHAR;
```

### New Tables Created

```sql
-- New table for tracking hardware availability
CREATE TABLE hardware_availability_log (
    id INTEGER PRIMARY KEY,
    hardware_type VARCHAR,
    is_available BOOLEAN,
    is_simulated BOOLEAN DEFAULT FALSE,
    detection_method VARCHAR,
    detection_details JSON,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Implementation Tools

The schema updates are implemented through:

1. **update_db_schema_for_simulation.py**: Automated tool to update database schema
2. **store_benchmark_in_database()**: Enhanced to include simulation flags
3. **add_hardware_platform()**: Updated to track simulation status

## Hardware Detection Improvements

Key improvements to hardware detection:

1. **Explicit Simulation Tracking**: Added `_simulated_hardware` dictionary to track simulation
2. **Environment Variable Handling**: Added proper parsing of simulation environment variables
3. **Robust Error Handling**: Improved error handling for hardware detection failures
4. **Logging Enhancements**: Added detailed logging of simulation status

### Example Hardware Detection Code

```python
def _detect_hardware(self):
    # Initialize tracking for simulated hardware
    self._simulated_hardware = {}
    
    # Detect QNN (Qualcomm Neural Networks)
    try:
        from hardware_detection.qnn_support import detect_qnn
        self.qnn_available, qnn_simulation = detect_qnn()
        if qnn_simulation:
            self._simulated_hardware["qnn"] = "QNN SDK not available"
    except ImportError:
        self.qnn_available = False
        self._simulated_hardware["qnn"] = "QNN support module not available"
    
    # Similar code for other hardware platforms...
```

## Report Validation

All report generators now include validation to check data authenticity:

1. **Validation Function**: Added `_validate_data_authenticity()` to check for simulated data
2. **Database Queries**: Added queries to check simulation status in database
3. **Visual Indicators**: Added clear visual indicators for simulated results
4. **Warning Headers**: Added warning headers to all reports with simulated data

### Example Validation Code

```python
def _validate_data_authenticity(self, df):
    """
    Validate data authenticity and mark simulated results.
    
    Args:
        df: DataFrame with benchmark results
        
    Returns:
        Tuple of (DataFrame with authenticity flags, bool indicating if any simulation was detected)
    """
    logger.info("Validating data authenticity...")
    simulation_detected = False
    
    # Add new column to track simulation status
    if 'is_simulated' not in df.columns:
        df['is_simulated'] = False
    
    # Check database for simulation flags
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
    
    return df, simulation_detected
```

## Stale Report Cleanup

Created tools to identify and mark problematic reports:

1. **cleanup_stale_reports.py**: Utility to scan for and mark/archive problematic reports
2. **Report Scanning**: Functions to scan HTML, Markdown, and JSON files for issues
3. **Warning Markers**: System to add clear warnings to problematic files
4. **Report Generator Fixes**: Tools to add validation to all report generators

### Report Marking Examples

#### HTML Warning

```html
<div style="background-color: #ffcccc; border: 2px solid #ff0000; padding: 10px; margin: 10px 0; color: #cc0000;">
    <h2>⚠️ WARNING: POTENTIALLY MISLEADING DATA ⚠️</h2>
    <p>This report may contain simulated benchmark results that are presented as real hardware data.</p>
    <p>Issue: May contain simulation results presented as real data</p>
    <p>Marked as problematic by cleanup_stale_reports.py on 2025-04-06 12:34:56</p>
</div>
```

#### Markdown Warning

```markdown
# ⚠️ WARNING: POTENTIALLY MISLEADING DATA ⚠️

**This report may contain simulated benchmark results that are presented as real hardware data.**

Issue: May contain simulation results presented as real data

*Marked as problematic by cleanup_stale_reports.py on 2025-04-06 12:34:56*

---
```

#### JSON Warning

```json
{
  "WARNING": {
    "message": "POTENTIALLY MISLEADING DATA",
    "details": "This file may contain simulated benchmark results that are presented as real hardware data.",
    "issue": "Contains hardware results without simulation flags",
    "marked_at": "2025-04-06T12:34:56.789012"
  },
  "results": [...]
}
```

## Verification Tools

Created utilities to check and fix simulation status:

1. **view_benchmark_results.py**: Tool to query database and view simulation status
2. **Simulation Status Checking**: Queries to check if simulation flags are set correctly
3. **Report Generation**: Functions to generate reports with simulation status clearly indicated
4. **Simulation Flag Updating**: SQL commands to fix simulation flags in database

### Example Commands

```bash
# Check simulation status in database
python view_benchmark_results.py --check-simulation

# Generate a report with simulation status
python view_benchmark_results.py --output benchmark_summary.md

# Update simulation flags in database
UPDATE hardware_platforms SET is_simulated = TRUE, simulation_reason = 'Hardware not available' WHERE hardware_type = 'rocm';
UPDATE performance_results SET is_simulated = TRUE, simulation_reason = 'Simulated ROCm hardware' WHERE hardware_id = 4;
```

## Implementation Status

All enhancements have been successfully implemented and tested:

1. ✅ Database schema updated with simulation flags
2. ✅ Hardware detection improved with explicit simulation tracking
3. ✅ Report generators updated with validation functions
4. ✅ Stale reports identified and marked with warnings
5. ✅ Verification tools implemented and tested
6. ✅ Documentation updated to reflect all changes

## Usage Guide

### Updating Database Schema

```bash
# Update database schema to add simulation flags
python update_db_schema_for_simulation.py

# Use custom database path
python update_db_schema_for_simulation.py --db-path ./custom_benchmark.duckdb

# Skip backup creation (not recommended)
python update_db_schema_for_simulation.py --no-backup
```

### Cleaning Up Stale Reports

```bash
# Scan for problematic files
python cleanup_stale_reports.py --scan

# Mark problematic files with warnings
python cleanup_stale_reports.py --mark

# Archive problematic files
python cleanup_stale_reports.py --archive

# Fix report generator scripts
python cleanup_stale_reports.py --fix-report-py
```

### Checking Simulation Status

```bash
# Check simulation status in database
python view_benchmark_results.py --check-simulation

# Generate a report with simulation status
python view_benchmark_results.py --output benchmark_summary.md
```

### Running Benchmarks with Simulation Flags

```bash
# Run benchmark with explicit simulation for unavailable hardware
python run_benchmark_with_db.py --model bert-base-uncased --hardware rocm --batch-sizes 1,2 --simulate
```

## Benefits

These improvements provide several key benefits:

1. **Clear Data Authenticity**: Users can now clearly tell when results use simulated hardware
2. **Trustworthy Reports**: Reports now clearly indicate simulation status
3. **Database Integrity**: Database now tracks simulation status for all entries
4. **Historical Tracking**: Historical reports are now marked with warnings if potentially misleading
5. **Data Quality**: Overall data quality and transparency significantly improved

## Conclusion

With these enhancements, the benchmark system now provides a clear distinction between real hardware measurements and simulated results. This ensures that users have accurate information when making hardware selection decisions and prevents misleading performance comparisons.

All tools and enhancements have been fully tested and are now part of the standard benchmark workflow. These improvements directly address the issues identified in item #10 (Critical Benchmark System Issues) in the NEXT_STEPS.md document.

For detailed implementation details, see the individual tool documentation and BENCHMARK_DB_FIX.md.