#!/usr/bin/env python3
"""
Comprehensive Benchmark Runner and Analysis

This script orchestrates comprehensive benchmarks and analysis to complete items from NEXT_STEPS.md:
- Run comprehensive benchmarks for all model types across hardware platforms
- Create performance ranking of hardware platforms based on real data 
- Identify and document performance bottlenecks using real measurements
- Generate detailed timing reports and visualizations

It combines multiple tools in sequence:
1. Run comprehensive benchmarks
2. Generate performance rankings
3. Analyze bottlenecks
4. Create comprehensive reports

Usage:
    python run_comprehensive_benchmarks_and_analysis.py --run-all
    python run_comprehensive_benchmarks_and_analysis.py --models bert,t5 --hardware cuda,cpu
    python run_comprehensive_benchmarks_and_analysis.py --skip-benchmarks --analyze-only
"""

import os
import sys
import argparse
import logging
import subprocess
import json
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("comprehensive_benchmark_analysis.log")
    ]
)
logger = logging.getLogger(__name__)

# Define default models to benchmark
DEFAULT_MODELS = ["bert", "t5", "vit", "whisper", "clip", "llama", "llava"]

# Define default hardware to benchmark
DEFAULT_HARDWARE = ["cpu", "cuda"]


def run_benchmarks(models=None, hardware=None, batch_sizes=None, small_models=True,
                   db_path=None, output_dir="./benchmark_results", timeout=None,
                   report_format="html"):
    """Run comprehensive benchmarks."""
    logger.info("Starting comprehensive benchmarks...")
    
    # Prepare command
    cmd = [
        sys.executable,
        "run_comprehensive_benchmarks.py",
    ]
    
    # Add models if specified
    if models:
        cmd.extend(["--models", models])
    
    # Add hardware if specified
    if hardware:
        cmd.extend(["--hardware", hardware])
    
    # Add batch sizes if specified
    if batch_sizes:
        cmd.extend(["--batch-sizes", batch_sizes])
    
    # Add small models flag if specified
    if small_models:
        cmd.append("--small-models")
    
    # Add database path if specified
    if db_path:
        cmd.extend(["--db-path", db_path])
    
    # Add output directory
    cmd.extend(["--output-dir", output_dir])
    
    # Add timeout if specified
    if timeout:
        cmd.extend(["--timeout", str(timeout)])
    
    # Add report format
    cmd.extend(["--report-format", report_format])
    
    # Run command
    logger.info(f"Executing command: {' '.join(cmd)}")
    
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Benchmarks completed successfully")
        return True, process.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Benchmark command failed: {e.stderr}")
        return False, e.stderr


def generate_performance_ranking(models=None, hardware=None, db_path=None, 
                                output_dir="./benchmark_results", report_format="html"):
    """Generate performance ranking report."""
    logger.info("Generating performance ranking report...")
    
    # Prepare command
    cmd = [
        sys.executable,
        "run_performance_ranking.py",
        "--generate"
    ]
    
    # Add model filter if specified
    if models:
        for model in models.split(','):
            cmd.extend(["--model", model])
    
    # Add hardware filter if specified
    if hardware:
        for hw in hardware.split(','):
            cmd.extend(["--hardware", hw])
    
    # Add database path if specified
    if db_path:
        cmd.extend(["--db-path", db_path])
    
    # Add output format
    cmd.extend(["--format", report_format])
    
    # Add output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"performance_ranking_{timestamp}.{report_format}")
    cmd.extend(["--output", output_path])
    
    # Run command
    logger.info(f"Executing command: {' '.join(cmd)}")
    
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Performance ranking report generated: {output_path}")
        return True, output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Performance ranking command failed: {e.stderr}")
        return False, None


def analyze_bottlenecks(models=None, hardware=None, db_path=None, 
                       output_dir="./benchmark_results", report_format="html"):
    """Analyze performance bottlenecks."""
    logger.info("Analyzing performance bottlenecks...")
    
    # Prepare command
    cmd = [
        sys.executable,
        "identify_performance_bottlenecks.py",
        "--analyze"
    ]
    
    # Add model filter if specified
    if models:
        for model in models.split(','):
            cmd.extend(["--model", model])
    
    # Add hardware filter if specified
    if hardware:
        for hw in hardware.split(','):
            cmd.extend(["--hardware", hw])
    
    # Add database path if specified
    if db_path:
        cmd.extend(["--db-path", db_path])
    
    # Add output format
    cmd.extend(["--format", report_format])
    
    # Add output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"bottleneck_analysis_{timestamp}.{report_format}")
    cmd.extend(["--output", output_path])
    
    # Run command
    logger.info(f"Executing command: {' '.join(cmd)}")
    
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Bottleneck analysis report generated: {output_path}")
        return True, output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Bottleneck analysis command failed: {e.stderr}")
        return False, None


def generate_timing_report(db_path=None, output_dir="./benchmark_results", report_format="html"):
    """Generate comprehensive timing report."""
    logger.info("Generating comprehensive timing report...")
    
    # Prepare command
    cmd = [
        sys.executable,
        "benchmark_timing_report.py",
        "--generate"
    ]
    
    # Add database path if specified
    if db_path:
        cmd.extend(["--db-path", db_path])
    
    # Add output format
    cmd.extend(["--format", report_format])
    
    # Add output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"benchmark_timing_report_{timestamp}.{report_format}")
    cmd.extend(["--output", output_path])
    
    # Run command
    logger.info(f"Executing command: {' '.join(cmd)}")
    
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Comprehensive timing report generated: {output_path}")
        return True, output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Timing report command failed: {e.stderr}")
        return False, None


def run_all_analysis(models=None, hardware=None, batch_sizes=None, 
                    small_models=True, db_path=None, output_dir="./benchmark_results", 
                    timeout=None, report_format="html", skip_benchmarks=False):
    """Run all benchmark and analysis steps."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "models": models,
        "hardware": hardware,
        "reports": [],
        "success": False
    }
    
    # Step 1: Run benchmarks
    if not skip_benchmarks:
        logger.info("Step 1: Running comprehensive benchmarks")
        success, benchmark_output = run_benchmarks(
            models=models,
            hardware=hardware,
            batch_sizes=batch_sizes,
            small_models=small_models,
            db_path=db_path,
            output_dir=output_dir,
            timeout=timeout,
            report_format=report_format
        )
        
        if not success:
            logger.error("Benchmark step failed. Stopping analysis.")
            results["error"] = "Benchmark step failed"
            return results
        
        # Add benchmark report to results
        results["reports"].append({
            "type": "benchmark",
            "output": "See benchmark logs for details"
        })
    else:
        logger.info("Skipping benchmark step as requested")
    
    # Step 2: Generate performance ranking
    logger.info("Step 2: Generating performance ranking report")
    success, ranking_path = generate_performance_ranking(
        models=models,
        hardware=hardware,
        db_path=db_path,
        output_dir=output_dir,
        report_format=report_format
    )
    
    if success:
        # Add ranking report to results
        results["reports"].append({
            "type": "performance_ranking",
            "path": ranking_path
        })
    else:
        logger.warning("Performance ranking step failed, but continuing with analysis")
    
    # Step 3: Analyze bottlenecks
    logger.info("Step 3: Analyzing performance bottlenecks")
    success, bottleneck_path = analyze_bottlenecks(
        models=models,
        hardware=hardware,
        db_path=db_path,
        output_dir=output_dir,
        report_format=report_format
    )
    
    if success:
        # Add bottleneck report to results
        results["reports"].append({
            "type": "bottleneck_analysis",
            "path": bottleneck_path
        })
    else:
        logger.warning("Bottleneck analysis step failed, but continuing with report generation")
    
    # Step 4: Generate comprehensive timing report
    logger.info("Step 4: Generating comprehensive timing report")
    success, timing_path = generate_timing_report(
        db_path=db_path,
        output_dir=output_dir,
        report_format=report_format
    )
    
    if success:
        # Add timing report to results
        results["reports"].append({
            "type": "timing_report",
            "path": timing_path
        })
    else:
        logger.warning("Timing report generation failed")
    
    # Generate summary report
    summary_path = os.path.join(output_dir, f"benchmark_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Analysis summary saved to: {summary_path}")
    results["summary_path"] = summary_path
    results["success"] = True
    
    return results


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Comprehensive Benchmark Runner and Analysis")
    
    # Main command groups
    parser.add_argument("--run-all", action="store_true", help="Run all benchmark and analysis steps")
    parser.add_argument("--skip-benchmarks", action="store_true", help="Skip benchmarking step and only generate analysis reports")
    parser.add_argument("--analyze-only", action="store_true", help="Alias for --skip-benchmarks")
    
    # Benchmark configuration
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), 
                      help=f"Comma-separated list of models to benchmark (default: {','.join(DEFAULT_MODELS)})")
    parser.add_argument("--hardware", default=",".join(DEFAULT_HARDWARE),
                      help=f"Comma-separated list of hardware platforms to benchmark (default: {','.join(DEFAULT_HARDWARE)})")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16",
                      help="Comma-separated list of batch sizes to test (default: 1,2,4,8,16)")
    parser.add_argument("--full-models", action="store_true", 
                      help="Use full models instead of smaller variants (slower but more accurate)")
    
    # Output configuration
    parser.add_argument("--db-path", 
                      help="Path to benchmark database (default: BENCHMARK_DB_PATH env var or ./benchmark_db.duckdb)")
    parser.add_argument("--output-dir", default="./benchmark_results",
                      help="Directory to save benchmark results and reports (default: ./benchmark_results)")
    parser.add_argument("--report-format", choices=["html", "md", "markdown", "json"], default="html",
                      help="Format for generated reports (default: html)")
    parser.add_argument("--timeout", type=int, default=1200,
                      help="Timeout in seconds for benchmark execution (default: 1200)")
    
    args = parser.parse_args()
    
    # Determine database path
    db_path = args.db_path or os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
    
    # Handle analyze-only alias
    skip_benchmarks = args.skip_benchmarks or args.analyze_only
    
    # Run all steps if requested
    if args.run_all or True:  # Default behavior is to run all
        results = run_all_analysis(
            models=args.models,
            hardware=args.hardware,
            batch_sizes=args.batch_sizes,
            small_models=not args.full_models,
            db_path=db_path,
            output_dir=args.output_dir,
            timeout=args.timeout,
            report_format=args.report_format,
            skip_benchmarks=skip_benchmarks
        )
        
        if results["success"]:
            # Print summary of generated reports
            print("\nAnalysis complete! Generated reports:")
            for report in results["reports"]:
                report_type = report["type"].replace("_", " ").title()
                if "path" in report:
                    print(f"- {report_type}: {report['path']}")
                else:
                    print(f"- {report_type}: {report.get('output', 'No output path')}")
            
            print(f"\nSummary saved to: {results['summary_path']}")
            return 0
        else:
            print(f"\nAnalysis failed: {results.get('error', 'Unknown error')}")
            return 1
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())