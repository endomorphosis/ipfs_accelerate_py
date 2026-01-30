#!/usr/bin/env python3
"""
Run Skillset Benchmarks

This script is a convenient entry point for running benchmarks on the skillset
implementations found in ipfs_accelerate_py/worker/skillset.
"""

import os
import sys
import time
import logging
import argparse
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"skillset_benchmark_{time.strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Import benchmark modules
try:
    # Import benchmark core and skillset benchmark 
    # Make sure the imports are working
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from benchmark_core import BenchmarkRunner, BenchmarkRegistry
    
    # Import benchmark_skillset module directly to ensure it's registered
    from data.benchmarks.benchmark_skillset import SkillsetInferenceBenchmark, SkillsetThroughputBenchmark
    
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    sys.exit(1)
    
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Skillset Benchmark Runner")
    
    # Benchmark type
    parser.add_argument("--type", type=str, choices=["inference", "throughput"], default="inference",
                       help="Type of benchmark to run")
    
    # Hardware selection
    parser.add_argument("--hardware", type=str, choices=["cpu", "cuda", "rocm", "openvino", "mps", "qnn"], default="cpu",
                       help="Hardware to benchmark on")
    
    # Model selection
    parser.add_argument("--model", type=str, default="bert",
                       help="Model to benchmark (use 'all' for all models)")
    parser.add_argument("--random-sample", action="store_true",
                       help="Use random sample of models when 'all' is specified")
    parser.add_argument("--sample-size", type=int, default=10,
                       help="Number of models to sample when using random sampling")
    
    # Benchmark configuration
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8",
                       help="Comma-separated list of batch sizes")
    parser.add_argument("--concurrent-models", type=int, default=3,
                       help="Number of concurrent models for throughput benchmark")
    parser.add_argument("--warmup-runs", type=int, default=2,
                       help="Number of warmup runs")
    parser.add_argument("--measurement-runs", type=int, default=5,
                       help="Number of measurement runs")
    
    # Output configuration
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                       help="Directory for benchmark results")
    parser.add_argument("--report", action="store_true",
                       help="Generate HTML report of benchmark results")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert batch sizes
    batch_sizes = [int(size) for size in args.batch_sizes.split(",")]
    
    # Print available benchmarks
    logger.info("Available benchmarks:")
    for name, metadata in BenchmarkRegistry.list_benchmarks().items():
        logger.info(f"  - {name}: {metadata}")
    
    # Create benchmark runner
    runner = BenchmarkRunner(config={
        "output_dir": args.output_dir
    })
    
    # Determine benchmark name
    benchmark_name = "skillset_inference_benchmark" if args.type == "inference" else "skillset_throughput_benchmark"
    
    # Create benchmark parameters
    params = {
        "hardware": args.hardware,
        "model": args.model,
        "batch_sizes": batch_sizes,
        "random_sample": args.random_sample,
        "sample_size": args.sample_size,
        "concurrent_models": args.concurrent_models,
        "warmup_runs": args.warmup_runs,
        "measurement_runs": args.measurement_runs
    }
    
    # Log benchmark parameters
    logger.info(f"Running {benchmark_name} with parameters:")
    for key, value in params.items():
        logger.info(f"  - {key}: {value}")
    
    # Run benchmark
    try:
        start_time = time.time()
        results = runner.execute(benchmark_name, params)
        duration = time.time() - start_time
        
        # Log completion
        logger.info(f"Benchmark completed in {duration:.2f} seconds")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(
            args.output_dir, 
            f"skillset_{args.type}_benchmark_{args.hardware}_{timestamp}.json"
        )
        
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Results saved to {result_file}")
        
        # Print summary
        if args.type == "inference":
            summary = results.get("summary", {})
            logger.info(f"Benchmark Summary:")
            logger.info(f"Total models: {summary.get('total_models', 0)}")
            logger.info(f"Successful models: {summary.get('successful_models', 0)}")
            logger.info(f"Failed models: {summary.get('failed_models', 0)}")
            logger.info(f"Fastest model: {summary.get('fastest_model', 'N/A')} "
                      f"({summary.get('fastest_init_time_ms', 0):.2f} ms)")
            logger.info(f"Slowest model: {summary.get('slowest_model', 'N/A')} "
                      f"({summary.get('slowest_init_time_ms', 0):.2f} ms)")
            logger.info(f"Mean initialization time: {summary.get('mean_init_time_ms', 0):.2f} ms")
        else:
            throughput = results.get("throughput", {})
            logger.info(f"Benchmark Summary:")
            logger.info(f"Concurrent models: {throughput.get('concurrent_models', 0)}")
            logger.info(f"Models: {', '.join(throughput.get('selected_models', []))}")
            logger.info(f"Total time: {throughput.get('total_time_ms', 0):.2f} ms")
            logger.info(f"Throughput: {throughput.get('throughput_models_per_second', 0):.2f} models/s")
            if "speedup_over_sequential" in throughput:
                logger.info(f"Speedup over sequential: {throughput.get('speedup_over_sequential', 0):.2f}x")
        
        # Generate HTML report if requested
        if args.report:
            try:
                report_file = generate_html_report(results, args, result_file)
                logger.info(f"HTML report generated: {report_file}")
            except Exception as e:
                logger.error(f"Failed to generate HTML report: {e}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error running benchmark: {e}")
        return 1

def generate_html_report(results, args, result_file):
    """Generate HTML report from benchmark results."""
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    
    # Create report directory
    report_dir = os.path.join(args.output_dir, "reports")
    os.makedirs(report_dir, exist_ok=True)
    
    # Generate timestamp for report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create report file
    report_file = os.path.join(
        report_dir, 
        f"skillset_{args.type}_benchmark_{args.hardware}_{timestamp}.html"
    )
    
    # Create HTML content
    html = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "    <meta charset='UTF-8'>",
        "    <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        f"    <title>Skillset {args.type.capitalize()} Benchmark Report</title>",
        "    <style>",
        "        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }",
        "        h1, h2, h3 { color: #333; }",
        "        .container { max-width: 1200px; margin: 0 auto; }",
        "        .summary { background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }",
        "        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }",
        "        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "        th { background-color: #f2f2f2; }",
        "        tr:nth-child(even) { background-color: #f9f9f9; }",
        "        .chart { margin-bottom: 30px; }",
        "        .success { color: green; }",
        "        .failure { color: red; }",
        "    </style>",
        "</head>",
        "<body>",
        "    <div class='container'>",
        f"        <h1>Skillset {args.type.capitalize()} Benchmark Report</h1>",
        f"        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        "        <div class='summary'>",
        "            <h2>Benchmark Configuration</h2>",
        "            <table>",
        "                <tr><th>Parameter</th><th>Value</th></tr>"
    ]
    
    # Add configuration parameters
    for key, value in {
        "Benchmark Type": args.type,
        "Hardware": args.hardware,
        "Model": args.model,
        "Batch Sizes": args.batch_sizes,
        "Random Sample": args.random_sample,
        "Sample Size": args.sample_size,
        "Concurrent Models": args.concurrent_models,
        "Warmup Runs": args.warmup_runs,
        "Measurement Runs": args.measurement_runs
    }.items():
        html.append(f"                <tr><td>{key}</td><td>{value}</td></tr>")
    
    html.append("            </table>")
    
    # Add benchmark summary
    if args.type == "inference":
        summary = results.get("summary", {})
        html.extend([
            "            <h2>Benchmark Summary</h2>",
            "            <table>",
            "                <tr><th>Metric</th><th>Value</th></tr>",
            f"                <tr><td>Total Models</td><td>{summary.get('total_models', 0)}</td></tr>",
            f"                <tr><td>Successful Models</td><td>{summary.get('successful_models', 0)}</td></tr>",
            f"                <tr><td>Failed Models</td><td>{summary.get('failed_models', 0)}</td></tr>",
            f"                <tr><td>Fastest Model</td><td>{summary.get('fastest_model', 'N/A')} ({summary.get('fastest_init_time_ms', 0):.2f} ms)</td></tr>",
            f"                <tr><td>Slowest Model</td><td>{summary.get('slowest_model', 'N/A')} ({summary.get('slowest_init_time_ms', 0):.2f} ms)</td></tr>",
            f"                <tr><td>Mean Initialization Time</td><td>{summary.get('mean_init_time_ms', 0):.2f} ms</td></tr>",
            "            </table>"
        ])
    else:
        throughput = results.get("throughput", {})
        html.extend([
            "            <h2>Benchmark Summary</h2>",
            "            <table>",
            "                <tr><th>Metric</th><th>Value</th></tr>",
            f"                <tr><td>Concurrent Models</td><td>{throughput.get('concurrent_models', 0)}</td></tr>",
            f"                <tr><td>Models</td><td>{', '.join(throughput.get('selected_models', []))}</td></tr>",
            f"                <tr><td>Total Time</td><td>{throughput.get('total_time_ms', 0):.2f} ms</td></tr>",
            f"                <tr><td>Throughput</td><td>{throughput.get('throughput_models_per_second', 0):.2f} models/s</td></tr>"
        ])
        
        if "speedup_over_sequential" in throughput:
            html.append(f"                <tr><td>Speedup over Sequential</td><td>{throughput.get('speedup_over_sequential', 0):.2f}x</td></tr>")
            
        html.append("            </table>")
    
    html.append("        </div>")
    
    # Generate charts
    if args.type == "inference" and "models" in results:
        # Create model initialization time chart
        model_chart_path = os.path.join(report_dir, f"model_init_times_{timestamp}.png")
        
        # Extract model init times
        model_names = []
        init_times = []
        
        for model_name, model_data in results.get("models", {}).items():
            if model_data.get("success", False):
                model_names.append(model_name)
                init_times.append(model_data.get("fastest_init_time_ms", 0))
        
        if model_names and init_times:
            # Sort by initialization time
            sorted_data = sorted(zip(model_names, init_times), key=lambda x: x[1])
            model_names, init_times = zip(*sorted_data)
            
            # Create chart
            plt.figure(figsize=(12, 8))
            plt.barh(model_names, init_times, color='skyblue')
            plt.xlabel('Initialization Time (ms)')
            plt.ylabel('Model')
            plt.title(f'Model Initialization Times ({args.hardware})')
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(model_chart_path)
            plt.close()
            
            # Add chart to report
            html.extend([
                "        <div class='chart'>",
                "            <h2>Model Initialization Times</h2>",
                f"            <img src='{os.path.basename(model_chart_path)}' alt='Model Initialization Times' width='100%'>",
                "        </div>"
            ])
        
        # Create batch size comparison chart for fastest models
        batch_chart_path = os.path.join(report_dir, f"batch_comparison_{timestamp}.png")
        
        # Take top 5 fastest models
        top_models = model_names[:5] if len(model_names) > 5 else model_names
        
        # Extract batch data
        batch_data = {}
        batch_sizes = []
        
        for model_name in top_models:
            model_data = results.get("models", {}).get(model_name, {})
            batch_results = model_data.get("batch_results", {})
            
            model_batch_times = []
            
            for batch_key, batch_info in batch_results.items():
                if batch_info.get("success", False):
                    batch_size = batch_info.get("batch_size", 0)
                    if batch_size not in batch_sizes:
                        batch_sizes.append(batch_size)
                    
                    mean_time = batch_info.get("mean_init_time_ms", 0)
                    model_batch_times.append((batch_size, mean_time))
            
            if model_batch_times:
                batch_data[model_name] = dict(model_batch_times)
        
        if batch_data and batch_sizes:
            # Sort batch sizes
            batch_sizes.sort()
            
            # Create chart
            plt.figure(figsize=(10, 6))
            
            # Plot each model's batch times
            for model_name, batch_times in batch_data.items():
                model_times = [batch_times.get(batch_size, 0) for batch_size in batch_sizes]
                plt.plot(batch_sizes, model_times, marker='o', label=model_name)
            
            plt.xlabel('Batch Size')
            plt.ylabel('Initialization Time (ms)')
            plt.title(f'Batch Size Comparison for Top Models ({args.hardware})')
            plt.grid(linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(batch_chart_path)
            plt.close()
            
            # Add chart to report
            html.extend([
                "        <div class='chart'>",
                "            <h2>Batch Size Comparison for Top Models</h2>",
                f"            <img src='{os.path.basename(batch_chart_path)}' alt='Batch Size Comparison' width='100%'>",
                "        </div>"
            ])
    
    # Add detailed results table
    if args.type == "inference":
        html.extend([
            "        <h2>Detailed Model Results</h2>",
            "        <table>",
            "            <tr>",
            "                <th>Model</th>",
            "                <th>Status</th>",
            "                <th>Import Time (s)</th>",
            "                <th>Instantiation Time (s)</th>",
            "                <th>Fastest Init Time (ms)</th>",
            "            </tr>"
        ])
        
        for model_name, model_data in results.get("models", {}).items():
            success = model_data.get("success", False)
            status_class = "success" if success else "failure"
            status_text = "Success" if success else "Failed"
            error = model_data.get("error", "")
            
            html.append("            <tr>")
            html.append(f"                <td>{model_name}</td>")
            html.append(f"                <td class='{status_class}'>{status_text}</td>")
            
            if success:
                html.extend([
                    f"                <td>{model_data.get('import_time', 0):.4f}</td>",
                    f"                <td>{model_data.get('instantiation_time', 0):.4f}</td>",
                    f"                <td>{model_data.get('fastest_init_time_ms', 0):.2f}</td>"
                ])
            else:
                html.extend([
                    f"                <td colspan='3'>{error}</td>"
                ])
            
            html.append("            </tr>")
        
        html.append("        </table>")
    else:
        # For throughput benchmark
        html.extend([
            "        <h2>Concurrent Execution Results</h2>",
            "        <table>",
            "            <tr>",
            "                <th>Model</th>",
            "                <th>Status</th>",
            "            </tr>"
        ])
        
        for model_name, model_data in results.get("throughput", {}).get("model_results", {}).items():
            success = model_data.get("success", False)
            status_class = "success" if success else "failure"
            status_text = "Success" if success else "Failed"
            error = model_data.get("error", "")
            
            html.append("            <tr>")
            html.append(f"                <td>{model_name}</td>")
            
            if success:
                html.append(f"                <td class='{status_class}'>{status_text}</td>")
            else:
                html.append(f"                <td class='{status_class}'>{status_text}: {error}</td>")
            
            html.append("            </tr>")
        
        html.append("        </table>")
    
    # Add report footer
    html.extend([
        "        <h2>Raw Results</h2>",
        f"        <p>Raw results are available at: {os.path.basename(result_file)}</p>",
        "    </div>",
        "</body>",
        "</html>"
    ])
    
    # Write HTML to file
    with open(report_file, 'w') as f:
        f.write("\n".join(html))
    
    return report_file

if __name__ == "__main__":
    sys.exit(main())