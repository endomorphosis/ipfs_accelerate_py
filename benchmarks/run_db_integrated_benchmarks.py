#!/usr/bin/env python3
"""
Run Comprehensive Benchmark Timing Report

This script generates detailed benchmark timing reports for all 13 model types
across 8 hardware endpoints, with comparative visualizations and analysis.

Usage:
    python run_comprehensive_benchmark_timing.py --generate
    python run_comprehensive_benchmark_timing.py --interactive
    python run_comprehensive_benchmark_timing.py --db-path ./benchmark_db.duckdb
"""

import os
import sys
import argparse
import logging
import datetime
from pathlib import Path
from benchmark_timing_report import BenchmarkTimingReport

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("benchmark_timing_report.log")
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Run Comprehensive Benchmark Timing Report")
    
    # Main command groups
    parser.add_argument("--generate", action="store_true", help="Generate comprehensive timing report")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive dashboard")
    parser.add_argument("--api-server", action="store_true", help="Start API server for report data")
    
    # Configuration options
    parser.add_argument("--db-path", help="Path to benchmark database (defaults to BENCHMARK_DB_PATH env variable)")
    parser.add_argument("--output-dir", default="./reports", help="Output directory for reports (defaults to ./reports)")
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
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create report generator
    report_gen = BenchmarkTimingReport(db_path=db_path)
    
    if args.generate:
        # Determine output path
        output_path = os.path.join(args.output_dir, f"benchmark_timing_report_{timestamp}.{args.format}")
        
        # Generate report
        logger.info(f"Generating comprehensive benchmark timing report to {output_path}")
        result_path = report_gen.generate_timing_report(
            output_format=args.format,
            output_path=output_path,
            days_lookback=args.days
        )
        
        if result_path:
            logger.info(f"Report generated: {result_path}")
            print(f"\nReport successfully generated: {result_path}")
            
            # Create a symlink to the latest report
            latest_link = os.path.join(args.output_dir, f"benchmark_timing_report_latest.{args.format}")
            try:
                if os.path.exists(latest_link):
                    os.unlink(latest_link)
                os.symlink(os.path.basename(result_path), latest_link)
                logger.info(f"Created symlink to latest report: {latest_link}")
            except Exception as e:
                logger.warning(f"Could not create symlink to latest report: {str(e)}")
        else:
            logger.error("Failed to generate report")
            return 1
            
    elif args.interactive:
        logger.info(f"Launching interactive dashboard on port {args.port}")
        report_gen.create_interactive_dashboard(port=args.port)
    elif args.api_server:
        logger.error("API server not yet implemented")
        return 1
    else:
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())