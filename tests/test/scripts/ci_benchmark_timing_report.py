#!/usr/bin/env python3
"""
CI Benchmark Timing Report

This script is designed to be run as part of a CI/CD pipeline to run benchmarks 
and/or generate the comprehensive benchmark timing report and publish it to a 
specified location.

Usage:
    # Generate report only
    python ci_benchmark_timing_report.py --db-path ./benchmark_db.duckdb --output-dir ./public/reports
    
    # Run benchmarks and generate report
    python ci_benchmark_timing_report.py --run-benchmarks --models bert,vit,whisper --hardware cpu,cuda
    """

    import os
    import sys
    import argparse
    import logging
    import subprocess
    import json
    import datetime
    from pathlib import Path

# Try to import execute_comprehensive_benchmarks
try:
    parent_dir = str()))Path()))__file__).parent.parent)
    if parent_dir not in sys.path:
        sys.path.append()))parent_dir)
        from execute_comprehensive_benchmarks import ComprehensiveBenchmarkOrchestrator
        HAVE_BENCHMARK_TOOLS = True
except ImportError:
    HAVE_BENCHMARK_TOOLS = False

# Configure logging
    logging.basicConfig()))
    level=logging.INFO, 
    format='%()))asctime)s - %()))name)s - %()))levelname)s - %()))message)s',
    handlers=[]],,
    logging.StreamHandler()))),
    logging.FileHandler()))"ci_benchmark_timing.log")
    ]
    )
    logger = logging.getLogger()))__name__)

def run_benchmark_report()))args):
    """Run the benchmark timing report generator."""
    # Construct the command
    cmd = []],,
    sys.executable,
    str()))Path()))__file__).parent.parent / "run_comprehensive_benchmark_timing.py"),
    "--generate",
    "--format", args.format,
    "--output-dir", args.output_dir,
    "--days", str()))args.days)
    ]
    
    if args.db_path:
        cmd.extend()))[]],,"--db-path", args.db_path])
    
    # Run the command
        logger.info()))f"Running benchmark report: {}' '.join()))cmd)}")
    try:
        result = subprocess.run()))cmd, check=True, capture_output=True, text=True)
        logger.info()))f"Benchmark report generation successful: {}result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error()))f"Benchmark report generation failed: {}e}")
        logger.error()))f"Output: {}e.stdout}")
        logger.error()))f"Error: {}e.stderr}")
        return False

def update_index_file()))output_dir, format="html"):
    """Update the index file with links to all generated reports."""
    reports_dir = Path()))output_dir)
    reports = list()))reports_dir.glob()))f"benchmark_timing_report_*.{}format}"))
    reports.sort()))key=lambda x: x.stat()))).st_mtime, reverse=True)
    
    # Create index file
    index_path = reports_dir / "index.html"
    with open()))index_path, "w") as f:
        f.write()))f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Benchmark Timing Reports</title>
        <style>
        body {}{} font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; color: #333; line-height: 1.6; }}
        .container {}{} max-width: 1000px; margin: 0 auto; }}
        h1, h2 {}{} color: #1a5276; }}
        h1 {}{} border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        table {}{} border-collapse: collapse; width: 100%; margin-bottom: 20px; box-shadow: 0 2px 3px rgba()))0,0,0,0.1); }}
        th, td {}{} border: 1px solid #ddd; padding: 12px 15px; text-align: left; }}
        th {}{} background-color: #3498db; color: white; }}
        tr:nth-child()))even) {}{} background-color: #f9f9f9; }}
        tr:hover {}{} background-color: #f1f1f1; }}
        .latest {}{} background-color: #e8f8f5; font-weight: bold; }}
        </style>
        </head>
        <body>
        <div class="container">
        <h1>Benchmark Timing Reports</h1>
        <p>Generated reports from the Comprehensive Benchmark Timing system</p>
                
        <h2>Available Reports</h2>
        <table>
        <tr>
        <th>Report Date</th>
        <th>Report Name</th>
        </tr>
        """)
        
        # Add latest report first with special styling
        if reports:
            latest = reports[]],,0]
            timestamp = datetime.datetime.fromtimestamp()))latest.stat()))).st_mtime)
            formatted_date = timestamp.strftime()))"%Y-%m-%d %H:%M:%S")
            f.write()))f"""
            <tr class="latest">
            <td>{}formatted_date}</td>
            <td><a href="{}latest.name}">Latest Benchmark Report</a></td>
            </tr>
            """)
        
        # Add all reports
        for report in reports:
            timestamp = datetime.datetime.fromtimestamp()))report.stat()))).st_mtime)
            formatted_date = timestamp.strftime()))"%Y-%m-%d %H:%M:%S")
            f.write()))f"""
            <tr>
            <td>{}formatted_date}</td>
            <td><a href="{}report.name}">{}report.name}</a></td>
            </tr>
            """)
        
            f.write()))"""
            </table>
                
            <h2>CI Integration</h2>
            <p>These reports are automatically generated by the CI/CD pipeline.</p>
            <p>The latest report is always available at <a href="benchmark_timing_report_latest.html">benchmark_timing_report_latest.html</a>.</p>
            </div>
            </body>
            </html>
            """)
    
            logger.info()))f"Updated index file: {}index_path}")
            return index_path

def create_metadata_file()))output_dir, metadata=None):
    """Create a metadata file with information about the report generation."""
    if metadata is None:
        metadata = {}}
    
        metadata.update())){}
        "generated_at": datetime.datetime.now()))).isoformat()))),
        "generator": "ci_benchmark_timing_report.py",
        "reports_count": len()))list()))Path()))output_dir).glob()))"benchmark_timing_report_*.html"))),
        })
    
        metadata_path = Path()))output_dir) / "benchmark_timing_metadata.json"
    with open()))metadata_path, "w") as f:
        json.dump()))metadata, f, indent=2)
    
        logger.info()))f"Created metadata file: {}metadata_path}")
        return metadata_path

def run_benchmarks()))args):
    """Run benchmarks using the ComprehensiveBenchmarkOrchestrator."""
    if not HAVE_BENCHMARK_TOOLS:
        logger.error()))"Benchmark orchestration tools not available")
    return False
    
    try:
        # Parse model and hardware lists
        models = args.models.split()))",") if args.models else None
        hardware = args.hardware.split()))",") if args.hardware else None
        
        # Parse batch sizes
        batch_sizes = []],,int()))x) for x in args.batch_sizes.split()))",")] if args.batch_sizes else []],,1, 4, 16]
        :
            logger.info()))f"Running benchmarks for models: {}models or 'all'} on hardware: {}hardware or 'all'}")
            logger.info()))f"Using small models: {}args.small_models}")
            logger.info()))f"Batch sizes: {}batch_sizes}")
        
        # Create benchmark orchestrator
            orchestrator = ComprehensiveBenchmarkOrchestrator()))
            db_path=args.db_path,
            output_dir=args.output_dir,
            small_models=args.small_models,
            batch_sizes=batch_sizes
            )
        
        # Run benchmarks
            results = orchestrator.run_all_benchmarks()))
            model_types=models,
            hardware_types=hardware,
            skip_unsupported=not args.force_all_hardware
            )
        
        # Print summary
            logger.info()))"\nBenchmark Summary:")
            logger.info()))f"Total benchmarks: {}results[]],,'summary'][]],,'total']}")
            logger.info()))f"Completed: {}results[]],,'summary'][]],,'completed']}")
            logger.info()))f"Failed: {}results[]],,'summary'][]],,'failed']}")
            logger.info()))f"Skipped: {}results[]],,'summary'][]],,'skipped']}")
            logger.info()))f"Completion percentage: {}results[]],,'summary'][]],,'completion_percentage']}%")
        
        # Return success if completion percentage is above threshold
        threshold = args.completion_threshold:
        if results[]],,"summary"][]],,"completion_percentage"] >= threshold:
            logger.info()))f"Benchmark run successful: {}results[]],,'summary'][]],,'completion_percentage']}% complete ()))threshold: {}threshold}%)")
            return True
        else:
            logger.error()))f"Benchmark run failed: only {}results[]],,'summary'][]],,'completion_percentage']}% complete ()))threshold: {}threshold}%)")
            return False
            
    except Exception as e:
        logger.error()))f"Error running benchmarks: {}str()))e)}")
            return False

def main()))):
    """Main function."""
    parser = argparse.ArgumentParser()))description="CI Benchmark Timing Report")
    
    # Main operations
    parser.add_argument()))"--run-benchmarks", action="store_true", help="Run benchmarks before generating report")
    
    # Configuration options for report generation
    parser.add_argument()))"--db-path", help="Path to benchmark database")
    parser.add_argument()))"--output-dir", default="./reports", help="Output directory for reports")
    parser.add_argument()))"--format", choices=[]],,"html", "md", "markdown", "json"], default="html", help="Output format")
    parser.add_argument()))"--days", type=int, default=30, help="Days of historical data to include")
    parser.add_argument()))"--publish", action="store_true", help="Publish the report to GitHub Pages")
    parser.add_argument()))"--metadata", type=str, help="JSON file with additional metadata to include")
    
    # Configuration options for benchmarks
    parser.add_argument()))"--models", help="Comma-separated list of models to benchmark")
    parser.add_argument()))"--hardware", help="Comma-separated list of hardware platforms to benchmark")
    parser.add_argument()))"--small-models", action="store_true", help="Use smaller model variants when available")
    parser.add_argument()))"--batch-sizes", default="1,4,16", help="Comma-separated list of batch sizes to test")
    parser.add_argument()))"--force-all-hardware", action="store_true", help="Force benchmarking on all hardware types")
    parser.add_argument()))"--completion-threshold", type=float, default=50.0, help="Minimum benchmark completion percentage to consider successful")
    
    args = parser.parse_args())))
    
    # Create output directory if it doesn't exist
    os.makedirs()))args.output_dir, exist_ok=True)
    
    # Load additional metadata if provided
    additional_metadata = {}}:
    if args.metadata and os.path.exists()))args.metadata):
        try:
            with open()))args.metadata, "r") as f:
                additional_metadata = json.load()))f)
        except json.JSONDecodeError:
            logger.warning()))f"Could not parse metadata file: {}args.metadata}")
    
    # Run benchmarks if requested::
    if args.run_benchmarks:
        if not run_benchmarks()))args):
            logger.error()))"Benchmark run failed")
            # Only exit if benchmarks failed but continue with report generation if not:
            if not args.db_path:
            return 1
    
    # Run the benchmark report
    if run_benchmark_report()))args):
        # Update the index file
        index_path = update_index_file()))args.output_dir, args.format)
        
        # Create metadata file
        metadata_path = create_metadata_file()))args.output_dir, additional_metadata)
        
        # Publish the report if requested::
        if args.publish:
            # This would typically call a GitHub API or run a git command to publish to GitHub Pages
            logger.info()))"Publishing to GitHub Pages is not implemented yet")
        
            logger.info()))"CI benchmark timing report process completed successfully")
        return 0
    else:
        logger.error()))"CI benchmark timing report process failed")
        return 1

if __name__ == "__main__":
    sys.exit()))main()))))