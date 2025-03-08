#!/usr/bin/env python3
"""
Benchmark Real WebNN/WebGPU Implementations

This script provides an easy way to run comprehensive benchmarks for WebNN and WebGPU
implementations using real browser hardware acceleration. It ensures clear distinction
between real hardware acceleration and simulation.

Usage:
    # Run WebGPU benchmarks with Chrome
    python benchmark_real_webnn_webgpu.py --webgpu --chrome
    
    # Run WebNN benchmarks with Edge (best for WebNN)
    python benchmark_real_webnn_webgpu.py --webnn --edge
    
    # Run audio model benchmarks with Firefox (best for audio with compute shaders)
    python benchmark_real_webnn_webgpu.py --audio --firefox
    
    # Run quantized model benchmarks
    python benchmark_real_webnn_webgpu.py --bits 8 --mixed-precision
    
    # Run comprehensive benchmarks
    python benchmark_real_webnn_webgpu.py --comprehensive
"""

import os
import sys
import time
import subprocess
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Benchmark Real WebNN/WebGPU Implementations")
    
    # Platform options
    platform_group = parser.add_mutually_exclusive_group()
    platform_group.add_argument("--webgpu", action="store_true", help="Use WebGPU platform")
    platform_group.add_argument("--webnn", action="store_true", help="Use WebNN platform")
    
    # Browser options
    browser_group = parser.add_mutually_exclusive_group()
    browser_group.add_argument("--chrome", action="store_true", help="Use Chrome browser")
    browser_group.add_argument("--firefox", action="store_true", help="Use Firefox browser")
    browser_group.add_argument("--edge", action="store_true", help="Use Edge browser")
    
    # Model type options
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--text", action="store_true", help="Benchmark text models")
    model_group.add_argument("--vision", action="store_true", help="Benchmark vision models")
    model_group.add_argument("--audio", action="store_true", help="Benchmark audio models")
    model_group.add_argument("--multimodal", action="store_true", help="Benchmark multimodal models")
    
    # Model options
    parser.add_argument("--model", type=str, help="Specific model to benchmark")
    
    # Quantization options
    parser.add_argument("--bits", type=int, choices=[2, 4, 8, 16], help="Bit precision for quantization")
    parser.add_argument("--mixed-precision", action="store_true", help="Use mixed precision quantization")
    
    # Benchmark options
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8", help="Comma-separated list of batch sizes")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeats for each configuration")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup runs")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive benchmarks")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="./benchmark_results", help="Output directory")
    parser.add_argument("--output-format", choices=["json", "markdown", "html"], default="markdown", help="Output format")
    parser.add_argument("--db-path", type=str, help="Path to DuckDB database")
    parser.add_argument("--no-db", action="store_true", help="Disable database storage")
    
    # Execution options
    parser.add_argument("--headless", action="store_true", help="Run browsers in headless mode")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Create command for run_real_webnn_webgpu_benchmarks.py
    cmd = ["python", "run_real_webnn_webgpu_benchmarks.py"]
    
    # Platform
    if args.webgpu:
        cmd.extend(["--platform", "webgpu"])
    elif args.webnn:
        cmd.extend(["--platform", "webnn"])
    else:
        # Default to WebGPU
        cmd.extend(["--platform", "webgpu"])
    
    # Browser
    if args.firefox:
        cmd.extend(["--browser", "firefox"])
    elif args.edge:
        cmd.extend(["--browser", "edge"])
    elif args.chrome:
        cmd.extend(["--browser", "chrome"])
    else:
        # Choose browser based on platform and model type
        if args.webnn:
            # Edge has best WebNN support
            cmd.extend(["--browser", "edge"])
        elif args.audio:
            # Firefox has best WebGPU compute shader support for audio models
            cmd.extend(["--browser", "firefox"])
        else:
            # Default to Chrome
            cmd.extend(["--browser", "chrome"])
    
    # Model type and model name
    if args.text:
        cmd.extend(["--model-type", "text"])
        if not args.model:
            cmd.extend(["--model", "bert-base-uncased"])
    elif args.vision:
        cmd.extend(["--model-type", "vision"])
        if not args.model:
            cmd.extend(["--model", "vit-base-patch16-224"])
    elif args.audio:
        cmd.extend(["--model-type", "audio"])
        if not args.model:
            cmd.extend(["--model", "whisper-tiny"])
    elif args.multimodal:
        cmd.extend(["--model-type", "multimodal"])
        if not args.model:
            cmd.extend(["--model", "clip-vit-base-patch16"])
    
    # Specific model if provided
    if args.model:
        cmd.extend(["--model", args.model])
    
    # Batch sizes
    cmd.extend(["--batch-sizes", args.batch_sizes])
    
    # Quantization options
    if args.bits:
        cmd.extend(["--bits", str(args.bits)])
    if args.mixed_precision:
        cmd.append("--mixed-precision")
    
    # Benchmark options
    cmd.extend(["--repeats", str(args.repeats)])
    cmd.extend(["--warmup", str(args.warmup)])
    if args.comprehensive:
        cmd.append("--comprehensive")
    
    # Output options
    cmd.extend(["--output-format", args.output_format])
    cmd.extend(["--output-dir", args.output_dir])
    
    # Database options
    if args.no_db:
        # No database
        pass
    elif args.db_path:
        cmd.extend(["--db-path", args.db_path])
    elif "BENCHMARK_DB_PATH" in os.environ:
        # Use database path from environment
        cmd.extend(["--db-path", os.environ["BENCHMARK_DB_PATH"]])
    
    # Execution options
    if args.headless:
        cmd.append("--headless")
    if args.verbose:
        cmd.append("--verbose")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print command
    logger.info(f"Executing: {' '.join(cmd)}")
    
    # Record start time
    start_time = time.time()
    
    # Run command
    result = subprocess.run(cmd)
    
    # Record end time
    end_time = time.time()
    duration = end_time - start_time
    
    # Print result
    if result.returncode == 0:
        logger.info(f"Benchmark completed successfully in {duration:.2f} seconds")
        logger.info(f"Results saved to {args.output_dir}")
        return 0
    elif result.returncode == 2:
        logger.warning("Benchmark completed with SIMULATION mode (no real hardware acceleration)")
        logger.info(f"Results saved to {args.output_dir}")
        return 0
    else:
        logger.error(f"Benchmark failed with exit code {result.returncode}")
        return result.returncode

if __name__ == "__main__":
    sys.exit(main())