#!/usr/bin/env python3
"""
Run Week 1 Benchmarks

This script automates the Week 1 benchmark tasks from NEXT_STEPS_BENCHMARKING_PLAN.md.
It runs benchmarks for LLAMA, CLAP, LLaVA, sets up web testing environment,
and runs tests with WebNN and WebGPU optimizations.

Usage:
    python run_week1_benchmarks.py --day monday
    python run_week1_benchmarks.py --day tuesday
    python run_week1_benchmarks.py --day wednesday
    python run_week1_benchmarks.py --day thursday
    python run_week1_benchmarks.py --day friday
    python run_week1_benchmarks.py --day all
"""

import os
import sys
import argparse
import logging
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("week1_benchmarks.log")
    ]
)
logger = logging.getLogger(__name__)

# Get the database path from environment or use default
DB_PATH = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")

def run_command(cmd: List[str], description: str) -> bool:
    """
    Run a command and log output.

    Args:
        cmd: Command list to run
        description: Description of the command for logging

    Returns:
        bool: True if command succeeded, False otherwise
    """
    cmd_str = " ".join(cmd)
    logger.info(f"Running {description}: {cmd_str}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Print and log output in real-time
        for line in process.stdout:
            print(line, end='')
            logger.info(line.strip())
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code == 0:
            logger.info(f"✅ {description} completed successfully")
            return True
        else:
            logger.error(f"❌ {description} failed with return code {return_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error running {description}: {str(e)}")
        return False

def setup_web_testing_environment() -> bool:
    """
    Set up web testing environment for WebNN and WebGPU.
    
    Returns:
        bool: True if setup was successful, False otherwise
    """
    cmd = [
        "python", 
        "run_comprehensive_benchmarks.py", 
        "--setup-web-testing"
    ]
    
    return run_command(cmd, "Web testing environment setup")

def run_llama_benchmarks() -> bool:
    """
    Run LLAMA benchmarks on CPU, CUDA, and OpenVINO.
    
    Returns:
        bool: True if benchmarks were successful, False otherwise
    """
    cmd = [
        "python", 
        "run_comprehensive_benchmarks.py", 
        "--models", "llama", 
        "--hardware", "cpu,cuda,openvino",
        "--db-path", DB_PATH
    ]
    
    return run_command(cmd, "LLAMA benchmarks")

def run_clap_benchmarks() -> bool:
    """
    Run CLAP benchmarks with compute shader optimization.
    
    Returns:
        bool: True if benchmarks were successful, False otherwise
    """
    cmd = [
        "python", 
        "run_comprehensive_benchmarks.py", 
        "--models", "clap", 
        "--hardware", "cpu,cuda,webgpu", 
        "--web-compute-shaders",
        "--db-path", DB_PATH
    ]
    
    return run_command(cmd, "CLAP benchmarks with compute shader optimization")

def run_llava_benchmarks() -> bool:
    """
    Run LLaVA benchmarks on CPU and CUDA.
    
    Returns:
        bool: True if benchmarks were successful, False otherwise
    """
    cmd = [
        "python", 
        "run_comprehensive_benchmarks.py", 
        "--models", "llava", 
        "--hardware", "cpu,cuda",
        "--db-path", DB_PATH
    ]
    
    return run_command(cmd, "LLaVA benchmarks")

def run_webnn_bert_t5_tests() -> bool:
    """
    Run WebNN tests for BERT and T5 models.
    
    Returns:
        bool: True if tests were successful, False otherwise
    """
    cmd = [
        "python", 
        "run_comprehensive_benchmarks.py", 
        "--models", "bert,t5", 
        "--hardware", "webnn",
        "--db-path", DB_PATH
    ]
    
    return run_command(cmd, "WebNN tests for BERT and T5")

def run_webnn_vit_tests() -> bool:
    """
    Run WebNN tests for ViT model.
    
    Returns:
        bool: True if tests were successful, False otherwise
    """
    cmd = [
        "python", 
        "run_comprehensive_benchmarks.py", 
        "--models", "vit", 
        "--hardware", "webnn",
        "--db-path", DB_PATH
    ]
    
    return run_command(cmd, "WebNN tests for ViT")

def run_qwen2_benchmarks() -> bool:
    """
    Run Qwen2 benchmarks on CPU and CUDA.
    
    Returns:
        bool: True if benchmarks were successful, False otherwise
    """
    cmd = [
        "python", 
        "run_comprehensive_benchmarks.py", 
        "--models", "qwen2", 
        "--hardware", "cpu,cuda",
        "--db-path", DB_PATH
    ]
    
    return run_command(cmd, "Qwen2 benchmarks")

def run_parallel_loading_tests() -> bool:
    """
    Run parallel loading tests for CLIP and LLaVA on WebGPU.
    
    Returns:
        bool: True if tests were successful, False otherwise
    """
    cmd = [
        "python", 
        "run_comprehensive_benchmarks.py", 
        "--models", "clip,llava", 
        "--hardware", "webgpu", 
        "--web-parallel-loading",
        "--db-path", DB_PATH
    ]
    
    return run_command(cmd, "Parallel loading tests for CLIP and LLaVA")

def run_monday_tasks() -> Dict[str, bool]:
    """
    Run Monday tasks: LLAMA benchmarks and web testing environment setup.
    
    Returns:
        Dict[str, bool]: Results of each task
    """
    results = {}
    results["llama_benchmarks"] = run_llama_benchmarks()
    results["web_testing_setup"] = setup_web_testing_environment()
    return results

def run_tuesday_tasks() -> Dict[str, bool]:
    """
    Run Tuesday tasks: Continue Monday tasks if needed.
    
    Returns:
        Dict[str, bool]: Results of each task
    """
    results = {}
    # Check if Monday tasks were completed successfully
    monday_results_file = "week1_monday_results.json"
    if os.path.exists(monday_results_file):
        with open(monday_results_file, 'r') as f:
            monday_results = json.load(f)
        
        # Run any failed tasks from Monday
        if not monday_results.get("llama_benchmarks", False):
            results["llama_benchmarks_retry"] = run_llama_benchmarks()
        
        if not monday_results.get("web_testing_setup", False):
            results["web_testing_setup_retry"] = setup_web_testing_environment()
    
    return results

def run_wednesday_tasks() -> Dict[str, bool]:
    """
    Run Wednesday tasks: CLAP and LLaVA benchmarks, WebNN testing.
    
    Returns:
        Dict[str, bool]: Results of each task
    """
    results = {}
    results["clap_benchmarks"] = run_clap_benchmarks()
    results["llava_benchmarks"] = run_llava_benchmarks()
    results["webnn_bert_t5_tests"] = run_webnn_bert_t5_tests()
    return results

def run_thursday_tasks() -> Dict[str, bool]:
    """
    Run Thursday tasks: Continue Wednesday tasks if needed.
    
    Returns:
        Dict[str, bool]: Results of each task
    """
    results = {}
    # Check if Wednesday tasks were completed successfully
    wednesday_results_file = "week1_wednesday_results.json"
    if os.path.exists(wednesday_results_file):
        with open(wednesday_results_file, 'r') as f:
            wednesday_results = json.load(f)
        
        # Run any failed tasks from Wednesday
        if not wednesday_results.get("clap_benchmarks", False):
            results["clap_benchmarks_retry"] = run_clap_benchmarks()
        
        if not wednesday_results.get("llava_benchmarks", False):
            results["llava_benchmarks_retry"] = run_llava_benchmarks()
        
        if not wednesday_results.get("webnn_bert_t5_tests", False):
            results["webnn_bert_t5_tests_retry"] = run_webnn_bert_t5_tests()
    
    return results

def run_friday_tasks() -> Dict[str, bool]:
    """
    Run Friday tasks: Complete WebNN testing, Qwen2 benchmarks, parallel loading tests.
    
    Returns:
        Dict[str, bool]: Results of each task
    """
    results = {}
    results["webnn_vit_tests"] = run_webnn_vit_tests()
    results["qwen2_benchmarks"] = run_qwen2_benchmarks()
    results["parallel_loading_tests"] = run_parallel_loading_tests()
    return results

def update_progress_report(day: str, results: Dict[str, bool]) -> None:
    """
    Update the progress report with results from the day's tasks.
    
    Args:
        day: Day of the week
        results: Results of each task
    """
    progress_report_path = Path("/home/barberb/ipfs_accelerate_py/test/benchmark_results/MARCH_2025_BENCHMARK_PROGRESS.md")
    
    if not progress_report_path.exists():
        logger.warning(f"Progress report not found: {progress_report_path}")
        return
    
    with open(progress_report_path, 'r') as f:
        content = f.read()
    
    # Add daily progress update
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    day_title = day.capitalize()
    
    # Format results for the report
    result_lines = []
    for task, success in results.items():
        status = "✅ Completed" if success else "❌ Failed"
        task_name = task.replace("_", " ").title()
        result_lines.append(f"   - {task_name}: {status}")
    
    daily_update = f"\n## {day_title} Progress Update ({timestamp})\n\n"
    daily_update += "\n".join(result_lines)
    daily_update += "\n"
    
    # Add before the "Next Immediate Actions" section if it exists
    if "## Next Immediate Actions" in content:
        content = content.replace("## Next Immediate Actions", daily_update + "\n## Next Immediate Actions")
    else:
        # Otherwise add at the end
        content += daily_update
    
    with open(progress_report_path, 'w') as f:
        f.write(content)
    
    logger.info(f"Updated progress report with {day} results")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run Week 1 Benchmarks")
    parser.add_argument("--day", choices=["monday", "tuesday", "wednesday", "thursday", "friday", "all"],
                       required=True, help="Day of the week to run tasks for")
    parser.add_argument("--db-path", default=DB_PATH, help="Path to benchmark database")
    parser.add_argument("--skip-report-update", action="store_true", help="Skip updating the progress report")
    
    args = parser.parse_args()
    
    # Update database path
    global DB_PATH
    DB_PATH = args.db_path
    
    # Run tasks for the specified day
    results = {}
    if args.day == "monday" or args.day == "all":
        logger.info("Running Monday tasks")
        day_results = run_monday_tasks()
        results["monday"] = day_results
        
        # Save results for reference on Tuesday
        with open("week1_monday_results.json", 'w') as f:
            json.dump(day_results, f, indent=2)
        
        if not args.skip_report_update:
            update_progress_report("monday", day_results)
    
    if args.day == "tuesday" or args.day == "all":
        logger.info("Running Tuesday tasks")
        day_results = run_tuesday_tasks()
        results["tuesday"] = day_results
        
        if not args.skip_report_update:
            update_progress_report("tuesday", day_results)
    
    if args.day == "wednesday" or args.day == "all":
        logger.info("Running Wednesday tasks")
        day_results = run_wednesday_tasks()
        results["wednesday"] = day_results
        
        # Save results for reference on Thursday
        with open("week1_wednesday_results.json", 'w') as f:
            json.dump(day_results, f, indent=2)
        
        if not args.skip_report_update:
            update_progress_report("wednesday", day_results)
    
    if args.day == "thursday" or args.day == "all":
        logger.info("Running Thursday tasks")
        day_results = run_thursday_tasks()
        results["thursday"] = day_results
        
        if not args.skip_report_update:
            update_progress_report("thursday", day_results)
    
    if args.day == "friday" or args.day == "all":
        logger.info("Running Friday tasks")
        day_results = run_friday_tasks()
        results["friday"] = day_results
        
        if not args.skip_report_update:
            update_progress_report("friday", day_results)
    
    # Print summary
    logger.info("\nWeek 1 Benchmark Summary:")
    for day, day_results in results.items():
        logger.info(f"{day.capitalize()}:")
        total = len(day_results)
        successful = sum(1 for success in day_results.values() if success)
        logger.info(f"  {successful}/{total} tasks completed successfully")
        
        for task, success in day_results.items():
            status = "✅ Success" if success else "❌ Failed"
            logger.info(f"  - {task}: {status}")
    
    # Generate weekly summary report
    if args.day == "friday" or args.day == "all":
        logger.info("\nGenerating Week 1 Summary Report")
        try:
            cmd = [
                "python",
                "benchmark_db_query.py",
                "--report", "week1_summary",
                "--format", "markdown",
                "--output", "benchmark_results/week1_summary_report.md",
                "--db-path", DB_PATH
            ]
            run_command(cmd, "Week 1 Summary Report Generation")
        except Exception as e:
            logger.error(f"Error generating Week 1 Summary Report: {str(e)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())