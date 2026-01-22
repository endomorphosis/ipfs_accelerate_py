#!/usr/bin/env python3
"""
Run Real Precision Tests for WebNN and WebGPU

This script runs real browser-based tests for WebNN and WebGPU at different precision levels.
It uses the real implementation modules with browser automation to test various quantization levels.

Usage:
    python run_real_precision_tests.py --platform webgpu --browser chrome --bits 4
    python run_real_precision_tests.py --platform webnn --browser edge --bits 8 --experimental

"""

import os
import sys
import json
import time
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for required dependencies
try:
    import websockets
    logger.info("Successfully imported websockets")
except ImportError:
    logger.error("websockets package is required. Install with: pip install websockets")
    sys.exit(1)
    
try:
    from selenium import webdriver
    logger.info("Successfully imported selenium")
except ImportError:
    logger.error("selenium package is required. Install with: pip install selenium")
    sys.exit(1)

# Results directory
RESULTS_DIR = Path("precision_test_results")
RESULTS_DIR.mkdir(exist_ok=True)

async def run_real_precision_test(
    platform: str,
    browser: str,
    model: str,
    bits: int,
    mixed_precision: bool = False,
    experimental: bool = False,
    verbose: bool = False
):
    """Run a real precision test with browser automation."""
    logger.info(f"Testing {platform} with {bits}-bit precision on {browser}")
    
    # Create timestamp for this test
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = RESULTS_DIR / f"{platform}_{browser}_{model}_{bits}bit_{timestamp}.json"
    
    # Set environment variables to ensure real implementation
    os.environ["WEBNN_SIMULATION"] = "0"
    os.environ["WEBGPU_SIMULATION"] = "0"
    
    # Run the webnn_webgpu_quantization_test.py script with the specified parameters
    cmd = [
        sys.executable,
        "webnn_webgpu_quantization_test.py",
        f"--platform={platform}",
        f"--browser={browser}",
        f"--model={model}",
        f"--bits={bits}"
    ]
    
    if mixed_precision:
        cmd.append("--mixed-precision")
        
    if experimental and platform == "webnn":
        os.environ["WEBNN_EXPERIMENTAL_PRECISION"] = "1"
        logger.info("Enabling experimental precision for WebNN (may cause errors)")
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        # Process output
        stdout_str = stdout.decode() if stdout else ""
        stderr_str = stderr.decode() if stderr else ""
        
        # Extract metrics from output
        metrics = extract_metrics_from_output(stdout_str, platform, bits)
        
        # Print output if verbose
        if verbose:
            logger.info("-" * 50)
            logger.info("STDOUT:")
            logger.info(stdout_str)
            if stderr_str:
                logger.info("-" * 50)
                logger.info("STDERR:")
                logger.info(stderr_str)
        
        # Check process return code
        if process.returncode != 0:
            logger.error(f"Test failed with return code {process.returncode}")
            logger.error(f"Error: {stderr_str}")
            metrics["status"] = "failed"
        else:
            logger.info(f"Test completed successfully")
            metrics["status"] = "success"
        
        # Save results to file
        with open(result_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Results saved to {result_file}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error running test: {e}")
        return {"status": "error", "error": str(e)}

def extract_metrics_from_output(output: str, platform: str, bits: int) -> Dict[str, Any]:
    """Extract metrics from the test output."""
    metrics = {
        "platform": platform,
        "bits": bits,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "memory_reduction_percent": None,
        "inference_time_ms": None,
        "is_simulation": True,  # Default to true until proven otherwise
        "features_detected": {}
    }
    
    # Extract metrics from log lines
    for line in output.splitlines():
        if "Memory reduction:" in line:
            try:
                metrics["memory_reduction_percent"] = float(line.split(":")[-1].strip().rstrip("%"))
            except:
                pass
            
        if "Inference completed in" in line:
            try:
                metrics["inference_time_ms"] = float(line.split("in")[-1].strip().rstrip("ms"))
            except:
                pass
        
        if "Feature support:" in line:
            # Try to extract the JSON that follows
            try:
                json_start = output.index("{", output.index("Feature support:"))
                json_end = output.index("}", json_start) + 1
                feature_json = output[json_start:json_end]
                metrics["features_detected"] = json.loads(feature_json)
            except:
                pass
        
        if "using real hardware acceleration" in line.lower():
            metrics["is_simulation"] = False
            
        if "simulation" in line.lower() and "not" not in line.lower():
            metrics["is_simulation"] = True
    
    return metrics

async def run_all_precision_tests(args):
    """Run all precision tests based on command line arguments."""
    platform = args.platform
    browser = args.browser
    model = args.model
    results = {}
    
    if args.all:
        # Run all supported precision levels
        if platform == "webgpu":
            # WebGPU supports 2-bit, 4-bit, 8-bit, 16-bit
            bits_to_test = [2, 4, 8, 16]
        else:
            # WebNN supports 8-bit natively, 4-bit experimentally
            if args.experimental:
                bits_to_test = [4, 8]
            else:
                bits_to_test = [8]
    else:
        # Run only the specified bit precision
        bits_to_test = [args.bits]
    
    # Run tests for each precision level
    for bits in bits_to_test:
        logger.info(f"Testing {bits}-bit precision")
        
        # For WebNN, only test 4-bit if experimental mode is enabled
        if platform == "webnn" and bits < 8 and not args.experimental:
            logger.warning(f"Skipping {bits}-bit test for WebNN without experimental mode")
            continue
            
        # Run the test
        results[bits] = await run_real_precision_test(
            platform=platform,
            browser=browser,
            model=model,
            bits=bits,
            mixed_precision=args.mixed_precision,
            experimental=args.experimental,
            verbose=args.verbose
        )
    
    # Save summary results
    summary_file = RESULTS_DIR / f"{platform}_{browser}_{model}_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Summary results saved to {summary_file}")
    
    # Generate comparison table
    print("\n" + "=" * 60)
    print(f"PRECISION COMPARISON FOR {platform.upper()} ON {browser.upper()}")
    print("=" * 60)
    print(f"{'Bits':^10} | {'Memory Reduction':^20} | {'Inference Time':^20} | {'Simulation':^15}")
    print("-" * 60)
    
    for bits, result in sorted(results.items()):
        memory_reduction = f"{result.get('memory_reduction_percent', 'N/A')}%" if result.get('memory_reduction_percent') is not None else "N/A"
        inference_time = f"{result.get('inference_time_ms', 'N/A')}ms" if result.get('inference_time_ms') is not None else "N/A"
        is_simulation = "Simulated" if result.get('is_simulation', True) else "Real Hardware"
        
        print(f"{bits:^10} | {memory_reduction:^20} | {inference_time:^20} | {is_simulation:^15}")
    
    print("=" * 60)
    return results

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Real Precision Tests for WebNN and WebGPU")
    
    parser.add_argument("--platform", type=str, choices=["webgpu", "webnn"], required=True,
                        help="Platform to test (webgpu or webnn)")
                        
    parser.add_argument("--browser", type=str, default="chrome",
                        help="Browser to use (chrome, firefox, edge, safari)")
                        
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                        help="Model to test")
    
    parser.add_argument("--bits", type=int, default=4,
                        help="Bit precision for quantization (2, 4, 8, or 16)")
    
    parser.add_argument("--all", action="store_true",
                        help="Test all supported precision levels")
    
    parser.add_argument("--mixed-precision", action="store_true",
                        help="Use mixed precision (higher bits for critical layers)")
    
    parser.add_argument("--experimental", action="store_true",
                        help="Enable experimental precision (especially for WebNN)")
    
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Run all tests
    asyncio.run(run_all_precision_tests(args))

if __name__ == "__main__":
    main()