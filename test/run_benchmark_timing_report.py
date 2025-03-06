#!/usr/bin/env python3
"""
Run Benchmark Timing Report

This script provides a simple interface to run the comprehensive benchmark timing report
for all 13 model types across 8 hardware endpoints.

Usage:
    python run_benchmark_timing_report.py --generate
    python run_benchmark_timing_report.py --generate --format markdown
    python run_benchmark_timing_report.py --interactive
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from benchmark_timing_report import BenchmarkTimingReport

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Run Benchmark Timing Report")
    
    # Main command groups
    parser.add_argument("--generate", action="store_true", help="Generate comprehensive timing report")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive dashboard")
    
    # Configuration options
    parser.add_argument("--db-path", help="Path to benchmark database (defaults to BENCHMARK_DB_PATH env variable)")
    parser.add_argument("--output", help="Output file for report (defaults to benchmark_timing_report.<format>)")
    parser.add_argument("--format", choices=["html", "md", "markdown", "json"], default="html", help="Output format")
    parser.add_argument("--days", type=int, default=30, help="Days of historical data to include")
    parser.add_argument("--port", type=int, default=8501, help="Port for interactive dashboard")
    
    args = parser.parse_args()
    
    # Find database path
    db_path = args.db_path or os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
    if not Path(db_path).exists():
        logger.error(f"Database not found at: {db_path}")
        logger.error("Please specify a valid database path with --db-path or set the BENCHMARK_DB_PATH environment variable")
        return 1
    
    # Create report generator
    report_gen = BenchmarkTimingReport(db_path=db_path)
    
    if args.generate:
        # Determine output path
        output_path = args.output
        if not output_path:
            output_path = f"benchmark_timing_report.{args.format}"
        
        # Generate report
        result_path = report_gen.generate_timing_report(
            output_format=args.format,
            output_path=output_path,
            days_lookback=args.days
        )
        
        if result_path:
            logger.info(f"Report generated: {result_path}")
        else:
            logger.error("Failed to generate report")
            return 1
            
    elif args.interactive:
        logger.info(f"Launching interactive dashboard on port {args.port}")
        report_gen.create_interactive_dashboard(port=args.port)
    else:
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())