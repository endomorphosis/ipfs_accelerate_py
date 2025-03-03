#!/usr/bin/env python3
"""
Schedule regular model benchmarking runs.

This script can be used to set up regular scheduled benchmarking runs
to monitor model performance and detect regressions over time.
It can be run via cron or a similar scheduling system.
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("benchmark_scheduler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Default schedule settings
DEFAULT_BENCHMARK_DIR = "./scheduled_benchmarks"
DEFAULT_MODELS_SET = "small"  # Use small models by default for scheduled runs
DEFAULT_INTERVAL_DAYS = 7  # Weekly by default

def run_scheduled_benchmark(
    output_dir: str,
    models_set: str = DEFAULT_MODELS_SET,
    hardware: list = None,
    notification_email: str = None,
    compare_with_previous: bool = True
):
    """
    Run a scheduled benchmark.
    
    Args:
        output_dir: Directory to save benchmark results
        models_set: Which model set to use ('key', 'small', or 'custom')
        hardware: Hardware platforms to test (defaults to all available)
        notification_email: Email to send notification when benchmark completes
        compare_with_previous: Whether to compare with previous benchmark runs
    """
    logger.info(f"Starting scheduled benchmark run with model set: {models_set}")
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    output_path = Path(output_dir) / timestamp
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Save run metadata
    metadata = {
        "timestamp": timestamp,
        "models_set": models_set,
        "hardware": hardware,
        "notification_email": notification_email,
        "compare_with_previous": compare_with_previous
    }
    
    metadata_file = output_path / "schedule_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Construct benchmark command
    cmd = [
        sys.executable, "run_model_benchmarks.py",
        "--output-dir", str(output_path),
        "--models-set", models_set
    ]
    
    if hardware:
        cmd.extend(["--hardware"] + hardware)
    
    # Run benchmark
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        duration = time.time() - start_time
        
        # Save command output
        with open(output_path / "benchmark_output.log", 'w') as f:
            f.write(result.stdout)
        
        if result.stderr:
            with open(output_path / "benchmark_error.log", 'w') as f:
                f.write(result.stderr)
        
        # Update metadata with results
        metadata["duration"] = duration
        metadata["status"] = "success" if result.returncode == 0 else "error"
        metadata["returncode"] = result.returncode
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Compare with previous runs if requested
        if compare_with_previous:
            compare_with_previous_runs(output_dir, output_path)
        
        # Send notification if requested
        if notification_email and os.path.exists("/usr/bin/mail"):
            subject = f"Benchmark Complete: {'SUCCESS' if result.returncode == 0 else 'FAILURE'}"
            body = f"Benchmark run completed in {duration:.2f} seconds.\nResults available at: {output_path}"
            
            try:
                subprocess.run(
                    ["/usr/bin/mail", "-s", subject, notification_email],
                    input=body,
                    text=True,
                    check=False
                )
                logger.info(f"Notification email sent to {notification_email}")
            except Exception as e:
                logger.error(f"Failed to send notification email: {e}")
        
        logger.info(f"Benchmark completed in {duration:.2f} seconds with status: {metadata['status']}")
        return metadata["status"] == "success"
    
    except Exception as e:
        logger.error(f"Error running benchmark: {e}")
        
        # Update metadata with error
        metadata["status"] = "error"
        metadata["error"] = str(e)
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return False

def compare_with_previous_runs(output_dir: str, current_run_path: Path):
    """
    Compare current benchmark run with previous runs to detect regressions.
    
    Args:
        output_dir: Directory containing all benchmark runs
        current_run_path: Path to the current benchmark run
    """
    logger.info("Comparing with previous benchmark runs...")
    
    # Find previous runs
    output_path = Path(output_dir)
    all_runs = sorted([d for d in output_path.iterdir() if d.is_dir() and d != current_run_path])
    
    if not all_runs:
        logger.info("No previous runs found for comparison")
        return
    
    # Get most recent run
    previous_run = all_runs[-1]
    logger.info(f"Comparing with previous run: {previous_run}")
    
    # Load benchmark results from both runs
    current_results_file = current_run_path / "benchmark_results.json"
    previous_results_file = previous_run / "benchmark_results.json"
    
    if not current_results_file.exists() or not previous_results_file.exists():
        logger.warning("Missing benchmark results files for comparison")
        return
    
    try:
        with open(current_results_file, 'r') as f:
            current_results = json.load(f)
        
        with open(previous_results_file, 'r') as f:
            previous_results = json.load(f)
        
        # Create comparison report
        comparison_report = {
            "current_timestamp": current_results.get("timestamp"),
            "previous_timestamp": previous_results.get("timestamp"),
            "functionality_changes": {},
            "performance_changes": {}
        }
        
        # Compare functionality verification
        if "functionality_verification" in current_results and "functionality_verification" in previous_results:
            for hw_type in current_results["functionality_verification"]:
                if hw_type not in previous_results["functionality_verification"]:
                    continue
                
                current_hw = current_results["functionality_verification"][hw_type]
                previous_hw = previous_results["functionality_verification"][hw_type]
                
                # Extract model results (handle different formats)
                current_models = {}
                if "models" in current_hw:
                    current_models = current_hw["models"]
                elif "model_results" in current_hw:
                    current_models = current_hw["model_results"]
                
                previous_models = {}
                if "models" in previous_hw:
                    previous_models = previous_hw["models"]
                elif "model_results" in previous_hw:
                    previous_models = previous_hw["model_results"]
                
                # Compare models
                for model in current_models:
                    if model not in previous_models:
                        continue
                    
                    # Extract success status (handle different formats)
                    current_success = False
                    if isinstance(current_models[model], dict) and "success" in current_models[model]:
                        current_success = current_models[model]["success"]
                    elif isinstance(current_models[model], bool):
                        current_success = current_models[model]
                    
                    previous_success = False
                    if isinstance(previous_models[model], dict) and "success" in previous_models[model]:
                        previous_success = previous_models[model]["success"]
                    elif isinstance(previous_models[model], bool):
                        previous_success = previous_models[model]
                    
                    # Check for status changes
                    if current_success != previous_success:
                        if hw_type not in comparison_report["functionality_changes"]:
                            comparison_report["functionality_changes"][hw_type] = []
                        
                        comparison_report["functionality_changes"][hw_type].append({
                            "model": model,
                            "previous_status": "success" if previous_success else "failure",
                            "current_status": "success" if current_success else "failure",
                            "regression": previous_success and not current_success
                        })
        
        # Compare performance benchmarks
        if "performance_benchmarks" in current_results and "performance_benchmarks" in previous_results:
            for family in current_results["performance_benchmarks"]:
                if family not in previous_results["performance_benchmarks"]:
                    continue
                
                current_family = current_results["performance_benchmarks"][family]
                previous_family = previous_results["performance_benchmarks"][family]
                
                if "benchmarks" not in current_family or "benchmarks" not in previous_family:
                    continue
                
                for model in current_family["benchmarks"]:
                    if model not in previous_family["benchmarks"]:
                        continue
                    
                    current_model = current_family["benchmarks"][model]
                    previous_model = previous_family["benchmarks"][model]
                    
                    for hw_type in current_model:
                        if hw_type not in previous_model:
                            continue
                        
                        current_hw = current_model[hw_type]
                        previous_hw = previous_model[hw_type]
                        
                        if "performance_summary" not in current_hw or "performance_summary" not in previous_hw:
                            continue
                        
                        current_perf = current_hw["performance_summary"]
                        previous_perf = previous_hw["performance_summary"]
                        
                        # Compare latency
                        if "latency" in current_perf and "latency" in previous_perf:
                            current_latency = current_perf["latency"].get("mean", 0)
                            previous_latency = previous_perf["latency"].get("mean", 0)
                            
                            if previous_latency > 0:
                                latency_change = (current_latency - previous_latency) / previous_latency * 100
                                
                                # Add to report if significant change (more than 5%)
                                if abs(latency_change) > 5:
                                    if family not in comparison_report["performance_changes"]:
                                        comparison_report["performance_changes"][family] = []
                                    
                                    comparison_report["performance_changes"][family].append({
                                        "model": model,
                                        "hardware": hw_type,
                                        "metric": "latency",
                                        "previous": previous_latency,
                                        "current": current_latency,
                                        "change_percent": latency_change,
                                        "regression": latency_change > 5  # Higher latency is a regression
                                    })
                        
                        # Compare throughput
                        if "throughput" in current_perf and "throughput" in previous_perf:
                            current_throughput = current_perf["throughput"].get("mean", 0)
                            previous_throughput = previous_perf["throughput"].get("mean", 0)
                            
                            if previous_throughput > 0:
                                throughput_change = (current_throughput - previous_throughput) / previous_throughput * 100
                                
                                # Add to report if significant change (more than 5%)
                                if abs(throughput_change) > 5:
                                    if family not in comparison_report["performance_changes"]:
                                        comparison_report["performance_changes"][family] = []
                                    
                                    comparison_report["performance_changes"][family].append({
                                        "model": model,
                                        "hardware": hw_type,
                                        "metric": "throughput",
                                        "previous": previous_throughput,
                                        "current": current_throughput,
                                        "change_percent": throughput_change,
                                        "regression": throughput_change < -5  # Lower throughput is a regression
                                    })
        
        # Save comparison report
        comparison_file = current_run_path / "comparison_report.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison_report, f, indent=2)
        
        # Generate human-readable report
        markdown_report = current_run_path / "comparison_report.md"
        with open(markdown_report, 'w') as f:
            f.write("# Benchmark Comparison Report\n\n")
            f.write(f"Comparing current run ({comparison_report['current_timestamp']}) ")
            f.write(f"with previous run ({comparison_report['previous_timestamp']})\n\n")
            
            # Functionality changes
            f.write("## Functionality Changes\n\n")
            
            if not comparison_report["functionality_changes"]:
                f.write("No functionality changes detected.\n\n")
            else:
                for hw_type, changes in comparison_report["functionality_changes"].items():
                    f.write(f"### {hw_type}\n\n")
                    
                    regressions = [c for c in changes if c.get("regression", False)]
                    improvements = [c for c in changes if not c.get("regression", False)]
                    
                    if regressions:
                        f.write("#### Regressions\n\n")
                        f.write("| Model | Previous | Current |\n")
                        f.write("|-------|----------|--------|\n")
                        
                        for regression in regressions:
                            f.write(f"| {regression['model']} | {regression['previous_status']} | {regression['current_status']} |\n")
                        
                        f.write("\n")
                    
                    if improvements:
                        f.write("#### Improvements\n\n")
                        f.write("| Model | Previous | Current |\n")
                        f.write("|-------|----------|--------|\n")
                        
                        for improvement in improvements:
                            f.write(f"| {improvement['model']} | {improvement['previous_status']} | {improvement['current_status']} |\n")
                        
                        f.write("\n")
            
            # Performance changes
            f.write("## Performance Changes\n\n")
            
            if not comparison_report["performance_changes"]:
                f.write("No significant performance changes detected.\n\n")
            else:
                # Group by family
                for family, changes in comparison_report["performance_changes"].items():
                    f.write(f"### {family}\n\n")
                    
                    regressions = [c for c in changes if c.get("regression", False)]
                    improvements = [c for c in changes if not c.get("regression", False)]
                    
                    if regressions:
                        f.write("#### Regressions\n\n")
                        f.write("| Model | Hardware | Metric | Previous | Current | Change |\n")
                        f.write("|-------|----------|--------|----------|---------|--------|\n")
                        
                        for regression in regressions:
                            f.write(f"| {regression['model']} | {regression['hardware']} | {regression['metric']} | ")
                            f.write(f"{regression['previous']:.4f} | {regression['current']:.4f} | {regression['change_percent']:.2f}% |\n")
                        
                        f.write("\n")
                    
                    if improvements:
                        f.write("#### Improvements\n\n")
                        f.write("| Model | Hardware | Metric | Previous | Current | Change |\n")
                        f.write("|-------|----------|--------|----------|---------|--------|\n")
                        
                        for improvement in improvements:
                            f.write(f"| {improvement['model']} | {improvement['hardware']} | {improvement['metric']} | ")
                            f.write(f"{improvement['previous']:.4f} | {improvement['current']:.4f} | {improvement['change_percent']:.2f}% |\n")
                        
                        f.write("\n")
            
            # Summary
            f.write("## Summary\n\n")
            
            total_regressions = sum(len([c for c in changes if c.get("regression", False)]) 
                                  for changes in comparison_report["functionality_changes"].values())
            total_regressions += sum(len([c for c in changes if c.get("regression", False)]) 
                                   for changes in comparison_report["performance_changes"].values())
            
            total_improvements = sum(len([c for c in changes if not c.get("regression", False)]) 
                                   for changes in comparison_report["functionality_changes"].values())
            total_improvements += sum(len([c for c in changes if not c.get("regression", False)]) 
                                    for changes in comparison_report["performance_changes"].values())
            
            f.write(f"- Total regressions: {total_regressions}\n")
            f.write(f"- Total improvements: {total_improvements}\n")
            
            if total_regressions > 0:
                f.write("\n⚠️ **WARNING**: Regressions detected! Please investigate.\n")
            else:
                f.write("\n✅ No regressions detected.\n")
        
        logger.info(f"Comparison report generated: {markdown_report}")
        
        # Check for regressions
        has_regressions = False
        
        for changes in comparison_report["functionality_changes"].values():
            if any(c.get("regression", False) for c in changes):
                has_regressions = True
                break
        
        if not has_regressions:
            for changes in comparison_report["performance_changes"].values():
                if any(c.get("regression", False) for c in changes):
                    has_regressions = True
                    break
        
        if has_regressions:
            logger.warning("⚠️ REGRESSIONS DETECTED! Please investigate.")
        else:
            logger.info("✅ No regressions detected.")
        
        return has_regressions
    
    except Exception as e:
        logger.error(f"Error comparing benchmark runs: {e}")
        return False

def main():
    """Main function for scheduling model benchmarks"""
    parser = argparse.ArgumentParser(description="Schedule model benchmarking runs")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_BENCHMARK_DIR, help="Output directory for benchmark results")
    parser.add_argument("--models-set", choices=["key", "small", "custom"], default=DEFAULT_MODELS_SET, help="Which model set to use")
    parser.add_argument("--hardware", type=str, nargs="+", help="Hardware platforms to test (defaults to all available)")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_DAYS, help="Interval in days between benchmark runs")
    parser.add_argument("--notification-email", type=str, help="Email to send notification when benchmark completes")
    parser.add_argument("--no-compare", action="store_true", help="Disable comparison with previous benchmark runs")
    parser.add_argument("--run-now", action="store_true", help="Run a benchmark immediately instead of scheduling")
    parser.add_argument("--install-cron", action="store_true", help="Install cron job for scheduled benchmarking")
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Save schedule configuration
    schedule_config = {
        "output_dir": args.output_dir,
        "models_set": args.models_set,
        "hardware": args.hardware,
        "interval_days": args.interval,
        "notification_email": args.notification_email,
        "compare_with_previous": not args.no_compare
    }
    
    config_file = output_path / "schedule_config.json"
    with open(config_file, 'w') as f:
        json.dump(schedule_config, f, indent=2)
    
    logger.info(f"Schedule configuration saved to {config_file}")
    
    # Run benchmark now if requested
    if args.run_now:
        run_scheduled_benchmark(
            output_dir=args.output_dir,
            models_set=args.models_set,
            hardware=args.hardware,
            notification_email=args.notification_email,
            compare_with_previous=not args.no_compare
        )
    
    # Install cron job if requested
    if args.install_cron:
        try:
            script_path = os.path.abspath(__file__)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Create cron command
            cron_cmd = f"0 0 */{args.interval} * * cd {current_dir} && {sys.executable} {script_path} --run-now"
            cron_cmd += f" --output-dir {args.output_dir}"
            cron_cmd += f" --models-set {args.models_set}"
            
            if args.hardware:
                cron_cmd += f" --hardware {' '.join(args.hardware)}"
            
            if args.notification_email:
                cron_cmd += f" --notification-email {args.notification_email}"
            
            if args.no_compare:
                cron_cmd += " --no-compare"
            
            # Check if crontab is available
            if os.path.exists("/usr/bin/crontab"):
                # Get existing crontab
                result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
                existing_crontab = result.stdout if result.returncode == 0 else ""
                
                # Check if our command is already in the crontab
                if cron_cmd in existing_crontab:
                    logger.info("Cron job already installed")
                else:
                    # Add our command to the crontab
                    new_crontab = existing_crontab.strip() + f"\n{cron_cmd}\n"
                    subprocess.run(["crontab", "-"], input=new_crontab, text=True, check=True)
                    logger.info("Cron job installed successfully")
            else:
                logger.error("crontab not found, couldn't install cron job")
                logger.info(f"To manually install the cron job, add this line to your crontab:")
                logger.info(cron_cmd)
        except Exception as e:
            logger.error(f"Error installing cron job: {e}")
            logger.info(f"To manually install the cron job, add this line to your crontab:")
            logger.info(cron_cmd)
    
    # Print instructions if not running now or installing cron
    if not args.run_now and not args.install_cron:
        logger.info("Configuration saved. To run a benchmark now, use --run-now")
        logger.info("To install a cron job for scheduled benchmarking, use --install-cron")
        logger.info(f"Benchmark will run every {args.interval} days when scheduled")

if __name__ == "__main__":
    main()