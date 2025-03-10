#!/usr/bin/env python3
"""
Execute Comprehensive Benchmarks

This script orchestrates the execution of comprehensive benchmarks for all 13 model types
across all 8 hardware endpoints and publishes detailed timing data.

It runs the benchmarks, collects the data in the DuckDB database, and then generates
a comprehensive benchmark timing report.

Usage:
    python execute_comprehensive_benchmarks.py --run-all
    python execute_comprehensive_benchmarks.py --model bert --hardware cuda webgpu
    python execute_comprehensive_benchmarks.py --generate-report
"""

import os
import sys
import argparse
import logging
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("comprehensive_benchmarks.log")
    ]
)
logger = logging.getLogger(__name__)

# Define model types and hardware endpoints
MODEL_TYPES = ["bert", "t5", "llama", "clip", "vit", "clap", 
               "wav2vec2", "whisper", "llava", "llava-next", "xclip", "qwen2", "detr"]

HARDWARE_ENDPOINTS = ["cpu", "cuda", "rocm", "mps", "openvino", "qnn", "webnn", "webgpu"]

# Mapping of model types to their canonical test models
MODEL_MAP = {
    "bert": "bert-base-uncased",
    "t5": "t5-small",
    "llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "clip": "openai/clip-vit-base-patch32",
    "vit": "google/vit-base-patch16-224",
    "clap": "laion/clap-htsat-unfused",
    "wav2vec2": "facebook/wav2vec2-base",
    "whisper": "openai/whisper-tiny",
    "llava": "llava-hf/llava-1.5-7b-hf",
    "llava-next": "llava-hf/llava-v1.6-mistral-7b",
    "xclip": "microsoft/xclip-base-patch32",
    "qwen2": "Qwen/Qwen2-0.5B-Instruct",
    "detr": "facebook/detr-resnet-50"
}

# Mapping of model types to their small variants for quicker testing
SMALL_MODEL_MAP = {
    "bert": "prajjwal1/bert-tiny",
    "t5": "google/t5-efficient-tiny",
    "llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "clip": "openai/clip-vit-base-patch16-224",
    "vit": "facebook/deit-tiny-patch16-224",
    "clap": "laion/clap-htsat-unfused", 
    "wav2vec2": "facebook/wav2vec2-base",
    "whisper": "openai/whisper-tiny",
    "llava": "llava-hf/llava-1.5-7b-hf",
    "llava-next": "llava-hf/llava-v1.6-mistral-7b",
    "xclip": "microsoft/xclip-base-patch32",
    "qwen2": "Qwen/Qwen2-0.5B-Instruct",
    "detr": "facebook/detr-resnet-50"
}


class ComprehensiveBenchmarkOrchestrator:
    """Orchestrates comprehensive benchmarks across all models and hardware endpoints."""
    
    def __init__(self, db_path: Optional[str] = None, output_dir: str = "./benchmark_results",
                 small_models: bool = False, batch_sizes: List[int] = [1, 2, 4, 8, 16]):
        """
        Initialize the benchmark orchestrator.
        
        Args:
            db_path: Path to benchmark database (defaults to BENCHMARK_DB_PATH env variable)
            output_dir: Directory to save benchmark results
            small_models: Use smaller model variants when available
            batch_sizes: Batch sizes to test
        """
        self.db_path = db_path or os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
        self.output_dir = Path(output_dir)
        self.small_models = small_models
        self.batch_sizes = batch_sizes
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Time stamp used for result files
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Results tracking
        self.results = {
            "timestamp": self.timestamp,
            "completed_benchmarks": [],
            "failed_benchmarks": [],
            "skipped_benchmarks": []
        }
    
    def detect_available_hardware(self) -> Dict[str, bool]:
        """
        Detect available hardware platforms.
        
        Returns:
            Dictionary of hardware platform availability
        """
        logger.info("Detecting available hardware platforms...")
        hardware_availability = {}
        
        try:
            # Try to import improved hardware detection
            from integrated_improvements.improved_hardware_detection import detect_available_hardware
            hardware_availability = detect_available_hardware()
            logger.info(f"Detected hardware: {hardware_availability}")
        except ImportError:
            # Fallback to basic detection
            logger.warning("Improved hardware detection not available, using basic detection")
            
            # Basic detection
            try:
                import torch
                hardware_availability["cpu"] = True
                hardware_availability["cuda"] = torch.cuda.is_available()
                
                # Check for MPS (Apple Silicon)
                hardware_availability["mps"] = (
                    hasattr(torch, 'backends') and 
                    hasattr(torch.backends, 'mps') and 
                    torch.backends.mps.is_available()
                )
                
                # Check for ROCm
                hardware_availability["rocm"] = (
                    torch.cuda.is_available() and 
                    hasattr(torch.version, 'hip') and 
                    torch.version.hip is not None
                )
                
                # For OpenVINO, QNN, WebNN, WebGPU - assume not available without specific detection
                hardware_availability["openvino"] = False
                hardware_availability["qnn"] = False
                hardware_availability["webnn"] = False
                hardware_availability["webgpu"] = False
                
                # Try to detect OpenVINO
                try:
                    import openvino
                    hardware_availability["openvino"] = True
                except ImportError:
                    pass
            except ImportError:
                logger.warning("PyTorch not available, assuming only CPU is available")
                hardware_availability = {hw: (hw == "cpu") for hw in HARDWARE_ENDPOINTS}
            
        return hardware_availability
    
    def run_benchmark(self, model_type: str, hardware_type: str) -> Dict[str, Any]:
        """
        Run benchmark for a single model and hardware combination.
        
        Args:
            model_type: Model type to benchmark
            hardware_type: Hardware type to benchmark
            
        Returns:
            Dictionary containing benchmark results
        """
        logger.info(f"Running benchmark for {model_type} on {hardware_type}")
        
        # Get model name
        model_name = SMALL_MODEL_MAP[model_type] if self.small_models else MODEL_MAP[model_type]
        
        # Build command - Note: benchmark_hardware_models.py has mutually exclusive arguments
        # We need to use run_model_benchmarks.py directly since that allows specifying both model and hardware
        # instead of trying to use benchmark_hardware_models.py
        cmd = [
            "python", 
            "run_model_benchmarks.py", 
            "--models-set", "small" if self.small_models else "key",
            "--specific-models", model_type,  # Pass the model type/key, not the full model name
            "--hardware", hardware_type,
            "--db-path", self.db_path,
            "--output-dir", str(self.output_dir),
            "--verbose",  # Enable verbose output for better error tracking
            # Remove db_only parameter which is causing issues
        ]
        
        # Add batch sizes
        batch_sizes_str = ",".join(map(str, self.batch_sizes))
        cmd.extend(["--batch-sizes", batch_sizes_str])
        
        # Add timestamp
        cmd.extend(["--timestamp", self.timestamp])
        
        # Execute the benchmark
        cmd_str = " ".join(cmd)
        logger.debug(f"Executing command: {cmd_str}")
        
        try:
            # Define benchmark timeout (10 minutes)
            benchmark_timeout = 600  # seconds
            
            # Run the benchmark with a timeout
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, 
                                    timeout=benchmark_timeout * 2)  # Double timeout for safety
            
            logger.info(f"Benchmark command completed for {model_type} on {hardware_type}")
            
            # Parse output for results
            try:
                output_file = self.output_dir / f"benchmark_{model_type}_{hardware_type}_{self.timestamp}.json"
                if output_file.exists():
                    # Read the JSON output file
                    with open(output_file, 'r') as f:
                        benchmark_result = json.load(f)
                    
                    # Check for error cases in the benchmark results
                    has_errors = False
                    error_details = []
                    
                    if "batch_sizes" in benchmark_result:
                        for batch_size, batch_data in benchmark_result["batch_sizes"].items():
                            if not batch_data.get("success", True):
                                has_errors = True
                                error_details.append({
                                    "batch_size": batch_size,
                                    "error_type": batch_data.get("error_type", "unknown"),
                                    "error_message": batch_data.get("error_message", "Unknown error")
                                })
                    
                    if has_errors:
                        # Even if the command succeeded, there may be batch-level errors
                        return {
                            "status": "error",
                            "model_type": model_type,
                            "hardware_type": hardware_type,
                            "error_type": "batch_errors",
                            "error": f"Errors in {len(error_details)} batch configurations",
                            "result": benchmark_result,
                            "error_details": error_details
                        }
                    else:
                        # Success case
                        return {
                            "status": "success",
                            "model_type": model_type,
                            "hardware_type": hardware_type,
                            "result": benchmark_result
                        }
                else:
                    # Output file not found but command succeeded
                    logger.warning(f"No output file found for {model_type} on {hardware_type}, but command completed")
                    return {
                        "status": "success",
                        "model_type": model_type,
                        "hardware_type": hardware_type,
                        "result": {"stdout": result.stdout, "stderr": result.stderr},
                        "warning": "No output file was generated"
                    }
            except json.JSONDecodeError as e:
                # JSON parsing error
                logger.warning(f"Failed to parse benchmark results JSON for {model_type} on {hardware_type}: {str(e)}")
                return {
                    "status": "error",
                    "model_type": model_type,
                    "hardware_type": hardware_type,
                    "error_type": "json_parse_error",
                    "error": f"Failed to parse benchmark results: {str(e)}",
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            except Exception as e:
                # Other result processing errors
                logger.warning(f"Error processing benchmark results for {model_type} on {hardware_type}: {str(e)}")
                return {
                    "status": "error",
                    "model_type": model_type,
                    "hardware_type": hardware_type,
                    "error_type": "result_processing_error",
                    "error": f"Error processing benchmark results: {str(e)}",
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
                
        except subprocess.TimeoutExpired as e:
            # Timeout error
            logger.error(f"Benchmark timed out for {model_type} on {hardware_type} after {e.timeout} seconds")
            return {
                "status": "error",
                "model_type": model_type,
                "hardware_type": hardware_type,
                "error_type": "timeout",
                "error": f"Benchmark timed out after {e.timeout} seconds",
                "command": cmd_str,
                "stdout": e.stdout if hasattr(e, 'stdout') and e.stdout else None,
                "stderr": e.stderr if hasattr(e, 'stderr') and e.stderr else None
            }
        except subprocess.CalledProcessError as e:
            # Command execution error
            logger.error(f"Benchmark command failed for {model_type} on {hardware_type} with exit code {e.returncode}")
            logger.error(f"Error output: {e.stderr}")
            
            return {
                "status": "error",
                "model_type": model_type,
                "hardware_type": hardware_type,
                "error_type": "execution_error",
                "error": f"Command failed with exit code {e.returncode}",
                "stdout": e.stdout,
                "stderr": e.stderr,
                "command": cmd_str,
                "returncode": e.returncode
            }
        except Exception as e:
            # Unexpected errors
            logger.error(f"Unexpected error running benchmark for {model_type} on {hardware_type}: {str(e)}")
            
            return {
                "status": "error",
                "model_type": model_type,
                "hardware_type": hardware_type,
                "error_type": "unexpected_error",
                "error": str(e),
                "command": cmd_str
            }
    
    def run_all_benchmarks(self, model_types: Optional[List[str]] = None, 
                          hardware_types: Optional[List[str]] = None,
                          skip_unsupported: bool = True) -> Dict[str, Any]:
        """
        Run benchmarks for all specified models and hardware platforms.
        
        Args:
            model_types: List of model types to benchmark, or None for all
            hardware_types: List of hardware types to benchmark, or None for all
            skip_unsupported: Skip hardware platforms that are not available
            
        Returns:
            Dictionary containing benchmark results
        """
        # Use all model and hardware types if not specified
        model_types = model_types or MODEL_TYPES
        hardware_types = hardware_types or HARDWARE_ENDPOINTS
        
        # Detect available hardware if skipping unsupported
        available_hardware = self.detect_available_hardware() if skip_unsupported else {hw: True for hw in hardware_types}
        
        # Track overall result statistics
        total = len(model_types) * len(hardware_types)
        completed = 0
        failed = 0
        skipped = 0
        
        logger.info(f"Starting benchmarks for {len(model_types)} models on {len(hardware_types)} hardware platforms")
        
        # Run benchmarks
        for model_type in model_types:
            for hardware_type in hardware_types:
                # Skip if hardware is not available
                if skip_unsupported and not available_hardware.get(hardware_type, False):
                    logger.info(f"Skipping benchmark for {model_type} on {hardware_type} (hardware not available)")
                    self.results["skipped_benchmarks"].append({
                        "model_type": model_type,
                        "hardware_type": hardware_type,
                        "reason": "Hardware not available"
                    })
                    skipped += 1
                    continue
                
                # Run benchmark
                result = self.run_benchmark(model_type, hardware_type)
                
                # Track result
                if result["status"] == "success":
                    self.results["completed_benchmarks"].append({
                        "model_type": model_type,
                        "hardware_type": hardware_type,
                        "timestamp": datetime.now().isoformat()
                    })
                    completed += 1
                    logger.info(f"Successfully completed benchmark for {model_type} on {hardware_type}")
                else:
                    # Capture detailed error information
                    error_info = {
                        "model_type": model_type,
                        "hardware_type": hardware_type,
                        "error": result.get("error", "Unknown error"),
                        "error_type": result.get("error_type", "unknown"),
                        "stdout": result.get("stdout", ""),
                        "stderr": result.get("stderr", ""),
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # If there's detailed batch information with errors, extract it
                    if "result" in result and isinstance(result["result"], dict) and "batch_sizes" in result["result"]:
                        batch_errors = []
                        for batch_size, batch_data in result["result"]["batch_sizes"].items():
                            if not batch_data.get("success", True):
                                batch_errors.append({
                                    "batch_size": batch_size,
                                    "error_type": batch_data.get("error_type", "unknown"),
                                    "error_message": batch_data.get("error_message", "Unknown error"),
                                    "error_output": batch_data.get("error_output", "")
                                })
                        
                        if batch_errors:
                            error_info["batch_errors"] = batch_errors
                    
                    self.results["failed_benchmarks"].append(error_info)
                    
                    # Log the error for immediate visibility
                    logger.error(f"Benchmark failed for {model_type} on {hardware_type}: {error_info['error']}")
                    if "batch_errors" in error_info:
                        for i, err in enumerate(error_info["batch_errors"]):
                            logger.error(f"  Batch {err['batch_size']} error ({i+1}/{len(error_info['batch_errors'])}): {err['error_message']}")
                    
                    failed += 1
                
                # Save progress after each benchmark
                self._save_progress()
        
        # Save final results
        self.results["summary"] = {
            "total": total,
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "completion_percentage": round((completed / total) * 100, 2) if total > 0 else 0
        }
        self._save_progress()
        
        # Generate comprehensive report
        self.generate_timing_report()
        
        return self.results
    
    def _save_progress(self):
        """Save current progress to a JSON file."""
        progress_file = self.output_dir / f"benchmark_progress_{self.timestamp}.json"
        with open(progress_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Also save to a "latest" file for easy access
        latest_file = self.output_dir / "benchmark_progress_latest.json"
        with open(latest_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def generate_timing_report(self, output_format: str = "html") -> str:
        """
        Generate comprehensive timing report based on benchmark results.
        
        Args:
            output_format: Output format (html, markdown, json)
            
        Returns:
            Path to generated report
        """
        logger.info(f"Generating comprehensive timing report in {output_format} format")
        
        try:
            # Import the benchmark timing report generator
            from benchmark_timing_report import BenchmarkTimingReport
            
            # Create output path
            output_path = self.output_dir / f"comprehensive_benchmark_report_{self.timestamp}.{output_format}"
            
            # Create report generator
            report_gen = BenchmarkTimingReport(db_path=self.db_path)
            
            # Generate report
            report_path = report_gen.generate_timing_report(
                output_format=output_format,
                output_path=str(output_path),
                days_lookback=30
            )
            
            if report_path:
                logger.info(f"Comprehensive timing report generated: {report_path}")
                
                # Create a symlink to the latest report
                latest_path = self.output_dir / f"comprehensive_benchmark_report_latest.{output_format}"
                try:
                    if latest_path.exists():
                        latest_path.unlink()
                    latest_path.symlink_to(Path(report_path).name)
                    logger.info(f"Created symlink to latest report: {latest_path}")
                except Exception as e:
                    logger.warning(f"Failed to create symlink to latest report: {str(e)}")
                
                return report_path
            else:
                logger.error("Failed to generate timing report")
                return ""
                
        except ImportError:
            logger.error("BenchmarkTimingReport module not available. Please install it to generate reports.")
            return ""
        except Exception as e:
            logger.error(f"Error generating timing report: {str(e)}")
            return ""


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Execute Comprehensive Benchmarks")
    
    # Main command groups
    parser.add_argument("--run-all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--generate-report", action="store_true", help="Generate timing report from existing data")
    
    # Model and hardware selection
    parser.add_argument("--model", action="append", nargs="+", help="Model types to benchmark (can specify multiple)")
    parser.add_argument("--hardware", action="append", nargs="+", help="Hardware types to benchmark (can specify multiple)")
    
    # Configuration options
    parser.add_argument("--db-path", help="Path to benchmark database (defaults to BENCHMARK_DB_PATH env variable)")
    parser.add_argument("--output-dir", default="./benchmark_results", help="Output directory for benchmark results")
    parser.add_argument("--small-models", action="store_true", help="Use smaller model variants when available")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16", help="Comma-separated list of batch sizes to test")
    parser.add_argument("--force-all-hardware", action="store_true", help="Force benchmarking on all hardware types, even if not available")
    parser.add_argument("--report-format", choices=["html", "md", "markdown", "json"], default="html", help="Output format for timing report")
    
    args = parser.parse_args()
    
    # Parse batch sizes
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    
    # Create benchmark orchestrator
    orchestrator = ComprehensiveBenchmarkOrchestrator(
        db_path=args.db_path,
        output_dir=args.output_dir,
        small_models=args.small_models,
        batch_sizes=batch_sizes
    )
    
    if args.run_all:
        # Parse model and hardware types
        if args.model:
            # Flatten the list if it's nested due to append+nargs
            model_types = [m for sublist in args.model for m in (sublist if isinstance(sublist, list) else [sublist])]
        else:
            model_types = MODEL_TYPES
            
        if args.hardware:
            # Flatten the list if it's nested due to append+nargs
            hardware_types = [h for sublist in args.hardware for h in (sublist if isinstance(sublist, list) else [sublist])]
        else:
            hardware_types = HARDWARE_ENDPOINTS
        
        # Run benchmarks
        results = orchestrator.run_all_benchmarks(
            model_types=model_types,
            hardware_types=hardware_types,
            skip_unsupported=not args.force_all_hardware
        )
        
        # Print summary
        print("\nBenchmark Summary:")
        print(f"Total benchmarks: {results['summary']['total']}")
        print(f"Completed: {results['summary']['completed']}")
        print(f"Failed: {results['summary']['failed']}")
        print(f"Skipped: {results['summary']['skipped']}")
        print(f"Completion percentage: {results['summary']['completion_percentage']}%")
        
        # Generate report
        report_path = orchestrator.generate_timing_report(output_format=args.report_format)
        if report_path:
            print(f"\nComprehensive timing report generated: {report_path}")
        
    elif args.generate_report:
        # Generate report from existing data
        report_path = orchestrator.generate_timing_report(output_format=args.report_format)
        if report_path:
            print(f"\nComprehensive timing report generated: {report_path}")
        else:
            logger.error("Failed to generate timing report")
            return 1
    else:
        parser.print_help()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())