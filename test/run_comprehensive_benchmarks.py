#!/usr/bin/env python3
"""
Run Comprehensive Benchmarks

This script runs the comprehensive benchmarks for all available hardware platforms,
focusing on CPU and CUDA if available. It executes benchmarks for a subset of models
to make progress on item #9 from NEXT_STEPS.md.

Usage:
    python run_comprehensive_benchmarks.py
    python run_comprehensive_benchmarks.py --models bert,t5,vit
    python run_comprehensive_benchmarks.py --hardware cpu,cuda
    python run_comprehensive_benchmarks.py --batch-sizes 1,4,16
    python run_comprehensive_benchmarks.py --force-hardware rocm,webgpu
    python run_comprehensive_benchmarks.py --report-format markdown
"""

import os
import sys
import subprocess
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("comprehensive_benchmarks_run.log")
    ]
)
logger = logging.getLogger(__name__)

# Define subset of models to benchmark
DEFAULT_MODELS = ["bert", "t5", "vit", "whisper"]

# Define all supported hardware platforms
ALL_HARDWARE = ["cpu", "cuda", "rocm", "mps", "openvino", "qnn", "webnn", "webgpu"]

# Define commonly available hardware platforms
DEFAULT_HARDWARE = ["cpu", "cuda"]

def detect_available_hardware(try_advanced_detection=True):
    """
    Detect available hardware platforms.
    
    Args:
        try_advanced_detection: Whether to try using the advanced hardware detection module
        
    Returns:
        dict: Dictionary mapping hardware platform to availability status
    """
    available_hardware = {"cpu": True}  # CPU is always available
    
    # Try to use the advanced hardware detection if available
    if try_advanced_detection:
        try:
            # First try centralized hardware detection
            from centralized_hardware_detection.hardware_detection import detect_hardware_capabilities
            capabilities = detect_hardware_capabilities()
            logger.info("Using centralized hardware detection system")
            
            # Map capabilities to hardware availability
            available_hardware.update({
                "cuda": capabilities.get("cuda", {}).get("available", False),
                "rocm": capabilities.get("rocm", {}).get("available", False),
                "mps": capabilities.get("mps", {}).get("available", False),
                "openvino": capabilities.get("openvino", {}).get("available", False),
                "qnn": capabilities.get("qnn", {}).get("available", False),
                "webnn": capabilities.get("webnn", {}).get("available", False),
                "webgpu": capabilities.get("webgpu", {}).get("available", False)
            })
            
            # Log detected hardware
            for hw, available in available_hardware.items():
                if available:
                    logger.info(f"Detected {hw.upper()} as available")
            
            return available_hardware
        except ImportError:
            logger.info("Centralized hardware detection not available, falling back to basic detection")
    
    # Fallback to basic detection
    try:
        import torch
        if torch.cuda.is_available():
            available_hardware["cuda"] = True
            logger.info(f"CUDA is available with {torch.cuda.device_count()} devices")
        else:
            available_hardware["cuda"] = False
        
        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            available_hardware["mps"] = True
            logger.info("MPS (Apple Silicon) is available")
        else:
            available_hardware["mps"] = False
    except ImportError:
        logger.warning("PyTorch not available, assuming only CPU is available")
        available_hardware["cuda"] = False
        available_hardware["mps"] = False
    
    # Check for OpenVINO
    try:
        import openvino
        available_hardware["openvino"] = True
        logger.info(f"OpenVINO is available (version {openvino.__version__})")
    except ImportError:
        available_hardware["openvino"] = False
    
    # Check for ROCm via environment variable
    if os.environ.get("ROCM_HOME"):
        available_hardware["rocm"] = True
        logger.info("ROCm is available")
    else:
        available_hardware["rocm"] = False
    
    # Other platforms are less likely to be available by default
    available_hardware["qnn"] = False
    available_hardware["webnn"] = False
    available_hardware["webgpu"] = False
    
    return available_hardware

def run_benchmarks(models=None, hardware=None, batch_sizes=None, small_models=True, 
                  db_path=None, output_dir="./benchmark_results", timeout=None,
                  report_format="html", force_hardware=None):
    """
    Run comprehensive benchmarks for specified models and hardware.
    
    Args:
        models: List of models to benchmark (default: DEFAULT_MODELS)
        hardware: List of hardware platforms to benchmark (default: detect_available_hardware())
        batch_sizes: List of batch sizes to test (default: [1, 2, 4, 8, 16])
        small_models: Use smaller model variants for quicker testing
        db_path: Path to benchmark database
        output_dir: Directory to save results
        timeout: Timeout in seconds for each benchmark
        report_format: Output format for the report (html, markdown, json)
        force_hardware: List of hardware platforms to force even if not detected as available
        
    Returns:
        bool: True if benchmarks completed successfully
    """
    models = models or DEFAULT_MODELS
    batch_sizes = batch_sizes or [1, 2, 4, 8, 16]
    
    # Detect available hardware
    available_hardware_dict = detect_available_hardware()
    available_hardware = [hw for hw, available in available_hardware_dict.items() if available]
    
    # Determine hardware to benchmark
    if hardware:
        # User specified hardware
        hardware_to_benchmark = hardware
    else:
        # Use available hardware by default
        hardware_to_benchmark = available_hardware
    
    # Force specified hardware platforms if requested
    if force_hardware:
        for hw in force_hardware:
            if hw not in hardware_to_benchmark:
                hardware_to_benchmark.append(hw)
                logger.warning(f"Forcing benchmark on {hw} even though it may not be available")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get database path
    if not db_path:
        db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
    
    # Convert batch sizes to string
    batch_sizes_str = ",".join(str(bs) for bs in batch_sizes)
    
    # Prepare command
    cmd = [
        sys.executable,
        "execute_comprehensive_benchmarks.py",
        "--run-all",
        "--small-models" if small_models else "",
        "--db-path", db_path,
        "--output-dir", str(output_dir),
        "--batch-sizes", batch_sizes_str,
        "--report-format", report_format,
    ]
    
    # Add model and hardware arguments
    cmd.append("--model")
    cmd.extend(models)
    
    cmd.append("--hardware")
    cmd.extend(hardware_to_benchmark)
    
    # Add timeout if specified
    if timeout:
        cmd.extend(["--timeout", str(timeout)])
    
    # Remove empty arguments
    cmd = [arg for arg in cmd if arg]
    
    # Convert to string for logging
    cmd_str = " ".join(cmd)
    logger.info(f"Running benchmark command: {cmd_str}")
    
    # Track benchmark status
    benchmark_status = {
        "timestamp": datetime.now().isoformat(),
        "models": models,
        "hardware": hardware_to_benchmark,
        "batch_sizes": batch_sizes,
        "small_models": small_models,
        "db_path": db_path,
        "output_dir": str(output_dir),
        "command": cmd_str,
        "status": "running",
        "start_time": datetime.now().isoformat()
    }
    
    # Save initial status
    status_file = output_dir / f"benchmark_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(status_file, 'w') as f:
        json.dump(benchmark_status, f, indent=2)
    
    # Execute the command
    try:
        start_time = datetime.now()
        logger.info(f"Starting benchmarks at {start_time}")
        
        # Run subprocess with real-time output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Collect output for status file
        output_lines = []
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')
            logger.info(line.strip())
            output_lines.append(line.strip())
        
        # Wait for process to complete
        return_code = process.wait()
        
        end_time = datetime.now()
        duration = end_time - start_time
        duration_seconds = duration.total_seconds()
        
        # Update benchmark status
        benchmark_status.update({
            "status": "completed" if return_code == 0 else "failed",
            "return_code": return_code,
            "end_time": end_time.isoformat(),
            "duration_seconds": duration_seconds,
            "output_summary": output_lines[-20:] if len(output_lines) > 20 else output_lines
        })
        
        # Save updated status
        with open(status_file, 'w') as f:
            json.dump(benchmark_status, f, indent=2)
        
        if return_code == 0:
            logger.info(f"Benchmarks completed successfully in {duration}")
            
            # Generate the report
            report_output = output_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{report_format.lower()}"
            report_cmd = [
                sys.executable,
                "benchmark_timing_report.py",
                "--generate",
                "--db-path", db_path,
                "--format", report_format,
                "--output", str(report_output)
            ]
            
            logger.info(f"Generating report: {' '.join(report_cmd)}")
            try:
                report_process = subprocess.run(report_cmd, check=True, capture_output=True, text=True)
                
                # Update benchmark status with report info
                benchmark_status.update({
                    "report_path": str(report_output),
                    "report_generated": True
                })
                
                # Create a symlink to the latest report
                latest_report = output_dir / f"benchmark_report_latest.{report_format.lower()}"
                if latest_report.exists():
                    try:
                        latest_report.unlink()
                    except:
                        logger.warning(f"Could not remove existing symlink: {latest_report}")
                try:
                    latest_report.symlink_to(report_output.name)
                    logger.info(f"Created symlink to latest report: {latest_report}")
                except Exception as e:
                    logger.warning(f"Could not create symlink to latest report: {str(e)}")
                
                logger.info(f"Report generated successfully: {report_output}")
                print(f"\nBenchmark report generated: {report_output}")
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to generate report: {e.stderr}")
                benchmark_status.update({
                    "report_generated": False,
                    "report_error": e.stderr
                })
            
            # Save final status
            with open(status_file, 'w') as f:
                json.dump(benchmark_status, f, indent=2)
            
            # Also save to a "latest" file for easy access
            latest_status = output_dir / "benchmark_status_latest.json"
            with open(latest_status, 'w') as f:
                json.dump(benchmark_status, f, indent=2)
            
            return True
        else:
            logger.error(f"Benchmarks failed with return code {return_code}")
            
            # Save final status
            with open(status_file, 'w') as f:
                json.dump(benchmark_status, f, indent=2)
            
            return False
        
    except Exception as e:
        logger.error(f"Error running benchmarks: {str(e)}")
        
        # Update benchmark status with error
        benchmark_status.update({
            "status": "error",
            "error": str(e),
            "end_time": datetime.now().isoformat()
        })
        
        # Save error status
        with open(status_file, 'w') as f:
            json.dump(benchmark_status, f, indent=2)
        
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run comprehensive benchmarks")
    
    # Core arguments
    parser.add_argument("--models", help="Comma-separated list of models to benchmark (default: bert,t5,vit,whisper)")
    parser.add_argument("--hardware", help="Comma-separated list of hardware platforms to benchmark (default: auto-detect)")
    parser.add_argument("--force-hardware", help="Comma-separated list of hardware platforms to force benchmarking on, even if not detected")
    parser.add_argument("--no-small-models", action="store_true", help="Use full-sized models instead of smaller variants")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16", help="Comma-separated list of batch sizes to test")
    
    # Configuration options
    parser.add_argument("--db-path", help="Path to benchmark database (default: env var BENCHMARK_DB_PATH or ./benchmark_db.duckdb)")
    parser.add_argument("--output-dir", default="./benchmark_results", help="Directory to save results")
    parser.add_argument("--timeout", type=int, help="Timeout in seconds for each benchmark (default: 600)")
    parser.add_argument("--report-format", choices=["html", "markdown", "json"], default="html", help="Output format for the report")
    
    # Advanced options
    parser.add_argument("--skip-report", action="store_true", help="Skip generating the report after benchmarks complete")
    parser.add_argument("--skip-hardware-detection", action="store_true", help="Skip hardware detection and use specified hardware only")
    parser.add_argument("--list-available-hardware", action="store_true", help="List available hardware platforms and exit")
    parser.add_argument("--all-hardware", action="store_true", help="Run benchmarks on all supported hardware platforms (may use simulation)")
    
    args = parser.parse_args()
    
    # List available hardware if requested
    if args.list_available_hardware:
        available_hardware_dict = detect_available_hardware()
        print("\nAvailable Hardware Platforms:")
        for hw in ALL_HARDWARE:
            status = "✅ AVAILABLE" if available_hardware_dict.get(hw, False) else "❌ NOT AVAILABLE"
            print(f"  - {hw.upper()}: {status}")
        return 0
    
    # Process models argument
    models = args.models.split(",") if args.models else DEFAULT_MODELS
    
    # Process hardware argument
    if args.all_hardware:
        # Use all hardware platforms
        hardware = ALL_HARDWARE
        logger.info("Using all supported hardware platforms (may use simulation for unavailable hardware)")
    elif args.hardware:
        # Use specified hardware
        hardware = args.hardware.split(",")
    elif args.skip_hardware_detection:
        # Skip detection and use default hardware
        hardware = DEFAULT_HARDWARE
        logger.info("Skipping hardware detection and using default hardware")
    else:
        # Auto-detect hardware
        available_hardware_dict = detect_available_hardware()
        hardware = [hw for hw, available in available_hardware_dict.items() if available]
        logger.info(f"Auto-detected hardware: {', '.join(hardware)}")
    
    # Process force hardware argument
    force_hardware = args.force_hardware.split(",") if args.force_hardware else None
    
    # Process batch sizes
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    
    logger.info(f"Running benchmarks for models: {', '.join(models)}")
    logger.info(f"Using hardware platforms: {', '.join(hardware)}")
    logger.info(f"Using batch sizes: {', '.join(map(str, batch_sizes))}")
    logger.info(f"Using {'small' if not args.no_small_models else 'full-sized'} models")
    
    success = run_benchmarks(
        models=models,
        hardware=hardware,
        batch_sizes=batch_sizes,
        small_models=not args.no_small_models,
        db_path=args.db_path,
        output_dir=args.output_dir,
        timeout=args.timeout,
        report_format=args.report_format,
        force_hardware=force_hardware
    )
    
    if success:
        print("\nBenchmarks completed successfully!")
    else:
        print("\nBenchmarks failed. Check logs for details.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())