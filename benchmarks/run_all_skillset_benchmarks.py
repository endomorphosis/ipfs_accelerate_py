#!/usr/bin/env python3
"""
Run benchmarks for all skillset implementations.

This script runs benchmarks for all skillset implementations in the skillset directory.
It first generates the benchmark files using generate_skillset_benchmarks.py and then
executes them using the benchmark runner.
"""

import os
import sys
import glob
import time
import logging
import argparse
import importlib.util
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"skillset_benchmark_{time.strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def get_model_name_from_filename(filename: str) -> str:
    """Extract model name from a skillset filename.
    
    Args:
        filename: The filename to extract from (e.g., 'hf_bert.py')
        
    Returns:
        The extracted model name (e.g., 'bert')
    """
    basename = os.path.basename(filename)
    if basename.startswith('hf_') and basename.endswith('.py'):
        return basename[3:-3]  # Strip hf_ prefix and .py suffix
    return basename


def get_skillset_files(skillset_dir: str) -> List[str]:
    """Get a list of all skillset files in the directory.
    
    Args:
        skillset_dir: Directory containing skillset files
        
    Returns:
        List of skillset filenames
    """
    pattern = os.path.join(skillset_dir, 'hf_*.py')
    return glob.glob(pattern)


def generate_benchmark_files(skillset_dir: str, output_dir: str, models: List[str] = None) -> List[str]:
    """Generate benchmark files for skillset implementations.
    
    Args:
        skillset_dir: Directory containing skillset files
        output_dir: Directory to write benchmark files to
        models: List of specific models to generate benchmarks for, or None for all
        
    Returns:
        List of generated benchmark files
    """
    logger.info(f"Generating benchmark files from {skillset_dir} to {output_dir}")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get skillset files
    if models:
        # Generate for specific models
        skillset_files = []
        for model in models:
            file_path = os.path.join(skillset_dir, f"hf_{model}.py")
            if os.path.exists(file_path):
                skillset_files.append(file_path)
            else:
                logger.warning(f"Skillset file not found for model: {model}")
    else:
        # Get all skillset files
        skillset_files = get_skillset_files(skillset_dir)
        
    if not skillset_files:
        logger.error(f"No skillset files found in {skillset_dir}")
        return []
    
    # Sort skillset files for predictable order
    skillset_files.sort()
    
    logger.info(f"Found {len(skillset_files)} skillset files to generate benchmarks for")
    
    # Import benchmark generator
    try:
        # Try to import the generator module
        from generate_skillset_benchmarks import generate_benchmark_for_skillset
        
        # Generate benchmarks for each file
        generated_files = []
        
        for file_path in skillset_files:
            model_name = get_model_name_from_filename(file_path)
            success, output_file = generate_benchmark_for_skillset(model_name, output_dir)
            
            if success:
                generated_files.append(output_file)
    
        logger.info(f"Generated {len(generated_files)} benchmark files")
        return generated_files
    
    except ImportError:
        # If import fails, run the generator script directly
        logger.info("Could not import generator module, running script directly")
        
        # Build command
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_skillset_benchmarks.py"),
            "--skillset-dir", skillset_dir,
            "--output-dir", output_dir
        ]
        
        if models:
            for model in models:
                subprocess.run(cmd + ["--model", model], check=True)
        else:
            subprocess.run(cmd, check=True)
            
        # Get generated files
        pattern = os.path.join(output_dir, "benchmark_*.py")
        generated_files = glob.glob(pattern)
        logger.info(f"Generated {len(generated_files)} benchmark files")
        return generated_files


def run_benchmarks(benchmark_files: List[str], hardware: str, benchmark_type: str,
                  concurrent_workers: int, batch_sizes: List[int], runs: int,
                  output_dir: str, report: bool = True) -> Dict[str, Any]:
    """Run benchmarks for the generated benchmark files.
    
    Args:
        benchmark_files: List of benchmark files to run
        hardware: Hardware to run benchmarks on (cpu, cuda, rocm, etc.)
        benchmark_type: Type of benchmark to run (inference or throughput)
        concurrent_workers: Number of concurrent workers for throughput benchmarks
        batch_sizes: List of batch sizes to benchmark
        runs: Number of measurement runs
        output_dir: Directory to write results to
        report: Whether to generate HTML reports
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Running {benchmark_type} benchmarks on {hardware} hardware for {len(benchmark_files)} models")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Try to import benchmark runner
    try:
        from benchmark_core import BenchmarkRunner, HardwareManager
        
        # Create hardware manager
        hardware_manager = HardwareManager()
        available_hardware = hardware_manager.detect_available_hardware()
        
        if hardware not in available_hardware or not available_hardware[hardware]:
            logger.error(f"Hardware {hardware} is not available")
            return {"success": False, "error": f"Hardware {hardware} is not available"}
        
        # Create benchmark runner
        runner = BenchmarkRunner(config={"output_dir": output_dir})
        
        # Run benchmarks
        results = {}
        
        for benchmark_file in benchmark_files:
            try:
                # Import benchmark module
                model_name = os.path.basename(benchmark_file).replace("benchmark_", "").replace(".py", "")
                
                # Determine benchmark name
                if benchmark_type == "inference":
                    benchmark_name = f"{model_name}_inference_benchmark"
                else:
                    benchmark_name = f"{model_name}_throughput_benchmark"
                
                # Create parameters
                params = {
                    "hardware": hardware,
                    "batch_sizes": batch_sizes,
                    "concurrent_workers": concurrent_workers,
                    "warmup_runs": max(1, runs // 3),  # Use 1/3 of runs for warmup
                    "measurement_runs": runs
                }
                
                # Import benchmark module dynamically
                module_name = f"benchmarks.skillset.benchmark_{model_name}"
                spec = importlib.util.spec_from_file_location(module_name, benchmark_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Log benchmark execution
                logger.info(f"Running benchmark: {benchmark_name} with params: {params}")
                
                # Execute benchmark
                start_time = time.time()
                result = runner.execute(benchmark_name, params)
                duration = time.time() - start_time
                
                # Log completion
                logger.info(f"Benchmark {benchmark_name} completed in {duration:.2f} seconds")
                
                # Add to results
                results[model_name] = result
                
                # Save individual result
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                result_file = os.path.join(
                    output_dir,
                    f"{model_name}_{benchmark_type}_{hardware}_{timestamp}.json"
                )
                
                # Save to file
                import json
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
                    
                logger.info(f"Results saved to {result_file}")
                
                # Generate HTML report if requested
                if report:
                    try:
                        report_file = os.path.join(
                            output_dir,
                            "reports",
                            f"{model_name}_{benchmark_type}_{hardware}_{timestamp}.html"
                        )
                        
                        # Ensure reports directory exists
                        os.makedirs(os.path.dirname(report_file), exist_ok=True)
                        
                        # Generate report
                        generate_html_report(result, model_name, benchmark_type, hardware, report_file)
                        logger.info(f"HTML report generated: {report_file}")
                    except Exception as e:
                        logger.error(f"Failed to generate HTML report: {e}")
                
            except Exception as e:
                logger.error(f"Error running benchmark {benchmark_file}: {e}")
                results[model_name] = {"success": False, "error": str(e)}
        
        # Generate combined report
        if report and results:
            try:
                combined_report_file = os.path.join(
                    output_dir,
                    "reports",
                    f"combined_{benchmark_type}_{hardware}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                )
                
                generate_combined_report(results, benchmark_type, hardware, combined_report_file)
                logger.info(f"Combined HTML report generated: {combined_report_file}")
            except Exception as e:
                logger.error(f"Failed to generate combined HTML report: {e}")
        
        return {
            "success": True,
            "results": results,
            "hardware": hardware,
            "benchmark_type": benchmark_type
        }
    
    except ImportError as e:
        logger.error(f"Error importing benchmark modules: {e}")
        return {"success": False, "error": str(e)}


def generate_html_report(result: Dict[str, Any], model_name: str, benchmark_type: str,
                        hardware: str, output_file: str) -> None:
    """Generate HTML report for benchmark result.
    
    Args:
        result: Benchmark result
        model_name: Name of the model
        benchmark_type: Type of benchmark (inference or throughput)
        hardware: Hardware used for benchmark
        output_file: Output file path
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    
    # Create report directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create HTML content
    html = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "    <meta charset='UTF-8'>",
        "    <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        f"    <title>{model_name} {benchmark_type.capitalize()} Benchmark Report</title>",
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
        f"        <h1>{model_name} {benchmark_type.capitalize()} Benchmark Report</h1>",
        f"        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        "        <div class='summary'>",
        "            <h2>Benchmark Configuration</h2>",
        "            <table>",
        "                <tr><th>Parameter</th><th>Value</th></tr>",
        f"                <tr><td>Model</td><td>{model_name}</td></tr>",
        f"                <tr><td>Type</td><td>{benchmark_type}</td></tr>",
        f"                <tr><td>Hardware</td><td>{hardware}</td></tr>"
    ]
    
    # Add benchmark-specific information
    if benchmark_type == "inference":
        html.extend([
            f"                <tr><td>Import Time</td><td>{result.get('import_time', 0):.4f} s</td></tr>",
            f"                <tr><td>Instantiation Time</td><td>{result.get('instantiation_time', 0):.4f} s</td></tr>",
            f"                <tr><td>Mean Init Time</td><td>{result.get('mean_init_time_ms', 0):.2f} ms</td></tr>"
        ])
        
        # Create chart directory
        chart_dir = os.path.join(os.path.dirname(output_file), "charts")
        os.makedirs(chart_dir, exist_ok=True)
        
        # Create batch size chart if we have batch results
        if "batch_results" in result:
            batch_sizes = []
            init_times = []
            std_devs = []
            
            for batch_key, batch_data in result.get("batch_results", {}).items():
                if batch_data.get("success", False):
                    batch_sizes.append(batch_data.get("batch_size", 0))
                    init_times.append(batch_data.get("mean_init_time_ms", 0))
                    std_devs.append(batch_data.get("std_init_time_ms", 0))
            
            if batch_sizes and init_times:
                # Sort by batch size
                sorted_data = sorted(zip(batch_sizes, init_times, std_devs))
                batch_sizes, init_times, std_devs = zip(*sorted_data)
                
                # Create chart
                plt.figure(figsize=(10, 6))
                plt.errorbar(batch_sizes, init_times, yerr=std_devs, marker='o', linestyle='-')
                plt.xlabel('Batch Size')
                plt.ylabel('Initialization Time (ms)')
                plt.title(f'{model_name} Initialization Time by Batch Size ({hardware})')
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Add batch size table to HTML
                html.append("            </table>")
                html.append("            <h2>Batch Size Results</h2>")
                html.append("            <table>")
                html.append("                <tr><th>Batch Size</th><th>Mean Init Time (ms)</th><th>Std Dev (ms)</th></tr>")
                
                for i in range(len(batch_sizes)):
                    html.append(f"                <tr><td>{batch_sizes[i]}</td><td>{init_times[i]:.2f}</td><td>{std_devs[i]:.2f}</td></tr>")
                
                # Save chart
                chart_path = os.path.join(chart_dir, f"{model_name}_batch_sizes.png")
                plt.savefig(chart_path)
                plt.close()
                
                # Add chart to HTML
                chart_rel_path = os.path.relpath(chart_path, os.path.dirname(output_file))
                html.extend([
                    "            </table>",
                    "            <div class='chart'>",
                    f"                <img src='{chart_rel_path}' alt='Batch Size Chart' style='max-width:100%;'>",
                    "            </div>"
                ])
            else:
                html.append("            </table>")
        else:
            html.append("            </table>")
    
    elif benchmark_type == "throughput":
        # Get throughput data
        throughput = result.get("throughput", {})
        html.extend([
            f"                <tr><td>Concurrent Workers</td><td>{throughput.get('concurrent_workers', 0)}</td></tr>",
            f"                <tr><td>Total Time</td><td>{throughput.get('total_time_ms', 0):.2f} ms</td></tr>",
            f"                <tr><td>Throughput</td><td>{throughput.get('throughput_models_per_second', 0):.2f} models/s</td></tr>"
        ])
        
        if "speedup_over_sequential" in throughput:
            html.append(f"                <tr><td>Speedup</td><td>{throughput.get('speedup_over_sequential', 0):.2f}x</td></tr>")
            
        html.append("            </table>")
        
        # Create chart directory
        chart_dir = os.path.join(os.path.dirname(output_file), "charts")
        os.makedirs(chart_dir, exist_ok=True)
        
        # Create speedup chart if we have the data
        if "speedup_over_sequential" in throughput and "concurrent_workers" in throughput:
            plt.figure(figsize=(8, 6))
            
            # Basic bar chart showing speedup
            workers = throughput.get("concurrent_workers", 0)
            speedup = throughput.get("speedup_over_sequential", 0)
            
            plt.bar(["Sequential", "Concurrent"], [1.0, speedup], color=['skyblue', 'orange'])
            plt.ylabel('Relative Speed')
            plt.title(f'{model_name} Speedup with {workers} Workers ({hardware})')
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            # Save chart
            chart_path = os.path.join(chart_dir, f"{model_name}_speedup.png")
            plt.savefig(chart_path)
            plt.close()
            
            # Add chart to HTML
            chart_rel_path = os.path.relpath(chart_path, os.path.dirname(output_file))
            html.extend([
                "            <div class='chart'>",
                f"                <img src='{chart_rel_path}' alt='Speedup Chart' style='max-width:100%;'>",
                "            </div>"
            ])
    
    # Finish HTML
    html.extend([
        "        </div>",
        "    </div>",
        "</body>",
        "</html>"
    ])
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write("\n".join(html))


def generate_combined_report(results: Dict[str, Dict[str, Any]], benchmark_type: str,
                           hardware: str, output_file: str) -> None:
    """Generate combined HTML report for all benchmark results.
    
    Args:
        results: Dictionary mapping model names to benchmark results
        benchmark_type: Type of benchmark (inference or throughput)
        hardware: Hardware used for benchmark
        output_file: Output file path
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    
    # Create report directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create HTML content
    html = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "    <meta charset='UTF-8'>",
        "    <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        f"    <title>Combined {benchmark_type.capitalize()} Benchmark Report</title>",
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
        f"        <h1>Combined {benchmark_type.capitalize()} Benchmark Report</h1>",
        f"        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        "        <div class='summary'>",
        "            <h2>Benchmark Summary</h2>",
        "            <table>",
        "                <tr><th>Parameter</th><th>Value</th></tr>",
        f"                <tr><td>Benchmark Type</td><td>{benchmark_type}</td></tr>",
        f"                <tr><td>Hardware</td><td>{hardware}</td></tr>",
        f"                <tr><td>Total Models</td><td>{len(results)}</td></tr>"
    ]
    
    # Add benchmark-specific tables
    if benchmark_type == "inference":
        # Create model init time table
        html.extend([
            "            </table>",
            "            <h2>Model Initialization Times</h2>",
            "            <table>",
            "                <tr><th>Model</th><th>Import Time (s)</th><th>Instantiation Time (s)</th><th>Mean Init Time (ms)</th></tr>"
        ])
        
        # Extract model data for chart
        model_names = []
        init_times = []
        
        for model_name, result in results.items():
            if result.get("success", False):
                model_names.append(model_name)
                init_time = result.get("mean_init_time_ms", 0)
                init_times.append(init_time)
                
                html.append(
                    f"                <tr><td>{model_name}</td><td>{result.get('import_time', 0):.4f}</td>"
                    f"<td>{result.get('instantiation_time', 0):.4f}</td><td>{init_time:.2f}</td></tr>"
                )
        
        html.append("            </table>")
        
        # Create chart directory
        chart_dir = os.path.join(os.path.dirname(output_file), "charts")
        os.makedirs(chart_dir, exist_ok=True)
        
        # Create model comparison chart
        if model_names and init_times:
            # Sort by init time
            sorted_data = sorted(zip(model_names, init_times), key=lambda x: x[1])
            model_names, init_times = zip(*sorted_data)
            
            # Create chart
            plt.figure(figsize=(12, 10))
            plt.barh(model_names, init_times, color='skyblue')
            plt.xlabel('Initialization Time (ms)')
            plt.ylabel('Model')
            plt.title(f'Model Initialization Times ({hardware})')
            plt.grid(True, axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save chart
            chart_path = os.path.join(chart_dir, f"combined_init_times.png")
            plt.savefig(chart_path)
            plt.close()
            
            # Add chart to HTML
            chart_rel_path = os.path.relpath(chart_path, os.path.dirname(output_file))
            html.extend([
                "            <div class='chart'>",
                f"                <img src='{chart_rel_path}' alt='Model Init Times Chart' style='max-width:100%;'>",
                "            </div>"
            ])
    
    elif benchmark_type == "throughput":
        # Create throughput table
        html.extend([
            "            </table>",
            "            <h2>Model Throughput Results</h2>",
            "            <table>",
            "                <tr><th>Model</th><th>Concurrent Workers</th><th>Total Time (ms)</th>"
            "<th>Throughput (models/s)</th><th>Speedup</th></tr>"
        ])
        
        # Extract model data for chart
        model_names = []
        throughputs = []
        speedups = []
        
        for model_name, result in results.items():
            if result.get("success", False):
                throughput_data = result.get("throughput", {})
                
                model_names.append(model_name)
                throughput = throughput_data.get("throughput_models_per_second", 0)
                throughputs.append(throughput)
                
                speedup = throughput_data.get("speedup_over_sequential", 0)
                speedups.append(speedup)
                
                html.append(
                    f"                <tr><td>{model_name}</td>"
                    f"<td>{throughput_data.get('concurrent_workers', 0)}</td>"
                    f"<td>{throughput_data.get('total_time_ms', 0):.2f}</td>"
                    f"<td>{throughput:.2f}</td>"
                    f"<td>{speedup:.2f}x</td></tr>"
                )
        
        html.append("            </table>")
        
        # Create chart directory
        chart_dir = os.path.join(os.path.dirname(output_file), "charts")
        os.makedirs(chart_dir, exist_ok=True)
        
        # Create throughput comparison chart
        if model_names and throughputs:
            # Sort by throughput
            sorted_data = sorted(zip(model_names, throughputs, speedups), key=lambda x: x[1], reverse=True)
            model_names, throughputs, speedups = zip(*sorted_data)
            
            # Create throughput chart
            plt.figure(figsize=(12, 10))
            plt.barh(model_names, throughputs, color='skyblue')
            plt.xlabel('Throughput (models/s)')
            plt.ylabel('Model')
            plt.title(f'Model Throughput Comparison ({hardware})')
            plt.grid(True, axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save throughput chart
            throughput_chart_path = os.path.join(chart_dir, f"combined_throughput.png")
            plt.savefig(throughput_chart_path)
            plt.close()
            
            # Create speedup chart
            plt.figure(figsize=(12, 10))
            plt.barh(model_names, speedups, color='orange')
            plt.xlabel('Speedup over Sequential Execution')
            plt.ylabel('Model')
            plt.title(f'Model Speedup Comparison ({hardware})')
            plt.grid(True, axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save speedup chart
            speedup_chart_path = os.path.join(chart_dir, f"combined_speedup.png")
            plt.savefig(speedup_chart_path)
            plt.close()
            
            # Add charts to HTML
            throughput_rel_path = os.path.relpath(throughput_chart_path, os.path.dirname(output_file))
            speedup_rel_path = os.path.relpath(speedup_chart_path, os.path.dirname(output_file))
            
            html.extend([
                "            <div class='chart'>",
                f"                <img src='{throughput_rel_path}' alt='Throughput Comparison Chart' style='max-width:100%;'>",
                "            </div>",
                "            <div class='chart'>",
                f"                <img src='{speedup_rel_path}' alt='Speedup Comparison Chart' style='max-width:100%;'>",
                "            </div>"
            ])
    
    # Finish HTML
    html.extend([
        "        </div>",
        "    </div>",
        "</body>",
        "</html>"
    ])
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write("\n".join(html))


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run benchmarks for skillset implementations")
    parser.add_argument(
        "--skillset-dir", 
        type=str, 
        default="../ipfs_accelerate_py/worker/skillset",
        help="Directory containing skillset files"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="benchmark_results",
        help="Directory to write results to"
    )
    parser.add_argument(
        "--benchmark-dir", 
        type=str, 
        default="benchmarks/skillset",
        help="Directory to write benchmark files to"
    )
    parser.add_argument(
        "--hardware", 
        type=str, 
        choices=["cpu", "cuda", "rocm", "openvino", "mps", "qnn"],
        default="cpu",
        help="Hardware to run benchmarks on"
    )
    parser.add_argument(
        "--type", 
        type=str, 
        choices=["inference", "throughput", "both"],
        default="both",
        help="Type of benchmark to run"
    )
    parser.add_argument(
        "--concurrent-workers", 
        type=int, 
        default=4,
        help="Number of concurrent workers for throughput benchmarks"
    )
    parser.add_argument(
        "--batch-sizes", 
        type=str, 
        default="1,2,4,8",
        help="Comma-separated list of batch sizes"
    )
    parser.add_argument(
        "--runs", 
        type=int, 
        default=5,
        help="Number of measurement runs"
    )
    parser.add_argument(
        "--model", 
        type=str,
        action="append", 
        help="Specific model to benchmark (can be specified multiple times)"
    )
    parser.add_argument(
        "--generate-only", 
        action="store_true",
        help="Only generate benchmark files, don't run benchmarks"
    )
    parser.add_argument(
        "--report", 
        action="store_true",
        help="Generate HTML reports for benchmark results"
    )
    
    args = parser.parse_args()
    
    # Convert batch sizes
    batch_sizes = [int(size) for size in args.batch_sizes.split(",")]
    
    # Generate benchmark files
    benchmark_files = generate_benchmark_files(
        args.skillset_dir,
        args.benchmark_dir,
        args.model
    )
    
    if not benchmark_files:
        logger.error("No benchmark files generated")
        return 1
    
    if args.generate_only:
        logger.info("Generated benchmark files only, not running benchmarks")
        return 0
    
    # Run benchmarks
    all_results = {}
    
    if args.type in ["inference", "both"]:
        # Run inference benchmarks
        inference_results = run_benchmarks(
            benchmark_files,
            args.hardware,
            "inference",
            args.concurrent_workers,
            batch_sizes,
            args.runs,
            args.output_dir,
            args.report
        )
        
        all_results["inference"] = inference_results
        
        if not inference_results.get("success", False):
            logger.error(f"Inference benchmarks failed: {inference_results.get('error', 'Unknown error')}")
        else:
            logger.info("Inference benchmarks completed successfully")
    
    if args.type in ["throughput", "both"]:
        # Run throughput benchmarks
        throughput_results = run_benchmarks(
            benchmark_files,
            args.hardware,
            "throughput",
            args.concurrent_workers,
            batch_sizes,
            args.runs,
            args.output_dir,
            args.report
        )
        
        all_results["throughput"] = throughput_results
        
        if not throughput_results.get("success", False):
            logger.error(f"Throughput benchmarks failed: {throughput_results.get('error', 'Unknown error')}")
        else:
            logger.info("Throughput benchmarks completed successfully")
    
    # Save combined results
    if all_results:
        import json
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_file = os.path.join(
            args.output_dir,
            f"combined_benchmark_results_{args.hardware}_{timestamp}.json"
        )
        
        with open(combined_file, 'w') as f:
            json.dump(all_results, f, indent=2)
            
        logger.info(f"Combined results saved to {combined_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())