#!/usr/bin/env python3
"""
Run Real WebNN/WebGPU Benchmarks

This is a convenience script that combines the best practices for running real WebNN/WebGPU 
benchmarks with proper browser selection, model optimization, and error handling.

It's designed to make it easy to run reliable benchmarks on real hardware (not simulations)
and store results in the DuckDB database.

Usage:
    # Run WebGPU benchmark with Chrome
    python test/run_real_web_benchmarks.py --platform webgpu --browser chrome --model bert
    
    # Run WebNN benchmark with Edge
    python test/run_real_web_benchmarks.py --platform webnn --browser edge --model bert
    
    # Run audio model with Firefox + compute shaders
    python test/run_real_web_benchmarks.py --platform webgpu --browser firefox --model whisper --compute-shaders
    
    # Run comprehensive benchmarks across all models and platforms
    python test/run_real_web_benchmarks.py --comprehensive
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
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"real_web_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
SUPPORTED_BROWSERS = ["chrome", "firefox", "edge", "safari"]
SUPPORTED_PLATFORMS = ["webnn", "webgpu", "all"]
SUPPORTED_MODELS = {
    "bert": {
        "name": "bert-base-uncased",
        "type": "text",
        "browser_recommendation": "chrome/edge",
        "platform_recommendation": "both",
        "small_variant": "prajjwal1/bert-tiny"
    },
    "t5": {
        "name": "t5-small",
        "type": "text",
        "browser_recommendation": "chrome/edge",
        "platform_recommendation": "both",
        "small_variant": "t5-efficient-tiny"
    },
    "vit": {
        "name": "google/vit-base-patch16-224",
        "type": "vision",
        "browser_recommendation": "chrome",
        "platform_recommendation": "webgpu",
        "small_variant": "google/vit-base-patch16-224-in21k"
    },
    "clip": {
        "name": "openai/clip-vit-base-patch32",
        "type": "vision",
        "browser_recommendation": "chrome",
        "platform_recommendation": "webgpu",
        "small_variant": "openai/clip-vit-base-patch32"
    },
    "whisper": {
        "name": "openai/whisper-small",
        "type": "audio",
        "browser_recommendation": "firefox",
        "platform_recommendation": "webgpu",
        "small_variant": "openai/whisper-tiny",
        "optimizations": ["compute_shaders"]
    },
    "wav2vec2": {
        "name": "facebook/wav2vec2-base-960h",
        "type": "audio",
        "browser_recommendation": "firefox",
        "platform_recommendation": "webgpu",
        "small_variant": "facebook/wav2vec2-base",
        "optimizations": ["compute_shaders"]
    },
    "clap": {
        "name": "laion/clap-htsat-unfused",
        "type": "audio",
        "browser_recommendation": "firefox", 
        "platform_recommendation": "webgpu",
        "small_variant": "laion/clap-htsat-unfused",
        "optimizations": ["compute_shaders"]
    }
}

def check_requirements() -> bool:
    """Check if all required packages are installed."""
    missing_packages = []
    
    try:
        import selenium
    except ImportError:
        missing_packages.append("selenium")
    
    try:
        import websockets
    except ImportError:
        missing_packages.append("websockets")
    
    try:
        import duckdb
    except ImportError:
        missing_packages.append("duckdb")
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.error(f"Install with: pip install {' '.join(missing_packages)}")
        return False
        
    logger.info("All required packages are installed")
    return True

def fix_test_files() -> bool:
    """Make sure test image and audio files exist."""
    script_dir = Path(__file__).resolve().parent
    test_image = script_dir / "test.jpg"
    test_audio = script_dir / "test.mp3"
    
    missing_files = []
    if not test_image.exists():
        missing_files.append(str(test_image))
    
    if not test_audio.exists():
        missing_files.append(str(test_audio))
    
    if missing_files:
        logger.warning(f"Missing test files: {', '.join(missing_files)}")
        
        # Try to run fix script
        try:
            logger.info("Attempting to fix test files...")
            fix_script = script_dir / "fix_real_webnn_webgpu_benchmarks.py"
            
            if fix_script.exists():
                subprocess.run([sys.executable, str(fix_script), "--fix-test-files"], 
                              check=True, capture_output=True)
                logger.info("Test files fixed successfully")
                return True
            else:
                logger.error(f"Fix script not found: {fix_script}")
                return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to fix test files: {e}")
            logger.error(f"Error output: {e.stderr.decode()}")
            return False
        except Exception as e:
            logger.error(f"Error fixing test files: {e}")
            return False
    
    logger.info("All test files exist")
    return True

def get_optimal_browser_for_model(model: str) -> str:
    """Get the optimal browser for a model."""
    model_info = SUPPORTED_MODELS.get(model, {})
    recommendation = model_info.get("browser_recommendation", "chrome")
    
    # Handle multiple recommendations (e.g., "chrome/edge")
    if "/" in recommendation:
        return recommendation.split("/")[0]
    
    return recommendation

def get_optimal_platform_for_model(model: str) -> str:
    """Get the optimal platform for a model."""
    model_info = SUPPORTED_MODELS.get(model, {})
    return model_info.get("platform_recommendation", "webgpu")

def get_recommended_optimizations(model: str) -> List[str]:
    """Get recommended optimizations for a model."""
    model_info = SUPPORTED_MODELS.get(model, {})
    return model_info.get("optimizations", [])

def determine_optimal_configurations() -> List[Dict[str, Any]]:
    """Determine optimal configurations for all models."""
    configurations = []
    
    for model_key, model_info in SUPPORTED_MODELS.items():
        # Get platform recommendations
        platform = model_info.get("platform_recommendation", "webgpu")
        platforms = [platform]
        if platform == "both":
            platforms = ["webgpu", "webnn"]
        
        # Get browser recommendations
        browser_rec = model_info.get("browser_recommendation", "chrome")
        browsers = [browser_rec]
        if "/" in browser_rec:
            browsers = browser_rec.split("/")
        
        # Create configurations for each combination
        for platform in platforms:
            for browser in browsers:
                # Skip invalid combinations (Firefox doesn't support WebNN)
                if platform == "webnn" and browser == "firefox":
                    continue
                
                config = {
                    "model": model_key,
                    "platform": platform,
                    "browser": browser,
                    "optimizations": model_info.get("optimizations", [])
                }
                
                configurations.append(config)
    
    return configurations

def run_benchmark(args, model: str, platform: str, browser: str,
                  use_small_model: bool = False, optimizations: Dict[str, bool] = None) -> bool:
    """Run a benchmark for a specific configuration."""
    script_dir = Path(__file__).resolve().parent
    benchmark_script = script_dir / "run_real_webnn_webgpu_benchmarks.py"
    
    if not benchmark_script.exists():
        logger.error(f"Benchmark script not found: {benchmark_script}")
        return False
    
    # Get model info
    model_info = SUPPORTED_MODELS.get(model, {})
    model_name = model_info.get("small_variant" if use_small_model else "name", model)
    
    # Set up command
    cmd = [
        sys.executable,
        str(benchmark_script),
        "--platform", platform,
        "--model", model,
        "--browser", browser
    ]
    
    # Add database options
    if args.db_path:
        cmd.extend(["--db-path", args.db_path])
    
    if args.db_only:
        cmd.append("--db-only")
    
    # Add warmup flag
    if args.warmup:
        cmd.append("--warmup")
    
    # Add runs
    if args.runs:
        cmd.extend(["--runs", str(args.runs)])
    
    # Add visualization
    if args.visualize:
        cmd.append("--visualize")
    
    # Add format
    if args.format != "all":
        cmd.extend(["--format", args.format])
    
    # Add optimization flags
    if optimizations is None:
        optimizations = {}
    
    # Add model-specific optimizations
    for opt in model_info.get("optimizations", []):
        optimizations[opt] = True
    
    # Apply compute shader optimization
    if optimizations.get("compute_shaders") or args.compute_shaders:
        cmd.append("--compute-shaders")
    
    # Apply parallel loading optimization
    if optimizations.get("parallel_loading") or args.parallel_loading:
        cmd.append("--parallel-loading")
    
    # Apply shader precompilation
    if optimizations.get("shader_precompile") or args.shader_precompile:
        cmd.append("--shader-precompile")
    
    # Apply all optimizations
    if args.all_optimizations:
        cmd.append("--all-optimizations")
    
    # Add small model flag
    if use_small_model:
        cmd.append("--small-models")
    
    # Add allow simulation flag
    if args.allow_simulation:
        cmd.append("--allow-simulation")
    
    # Add visible flag
    if args.visible:
        cmd.append("--visible")
    
    # Add verbose flag
    if args.verbose:
        cmd.append("--verbose")
    
    # Log the command
    logger.info(f"Running benchmark for {model} ({platform}/{browser})")
    logger.info(f"Command: {' '.join(cmd)}")
    
    # Run the benchmark
    try:
        result = subprocess.run(cmd, check=False, capture_output=True)
        
        # Log output
        if result.stdout:
            logger.info(f"Benchmark output: {result.stdout.decode()}")
        
        if result.stderr:
            logger.error(f"Benchmark error: {result.stderr.decode()}")
        
        # Check return code
        if result.returncode == 0:
            logger.info(f"Benchmark successful (real hardware)")
            return True
        elif result.returncode == 2:
            logger.warning(f"Benchmark completed but used simulation")
            return args.allow_simulation
        else:
            logger.error(f"Benchmark failed with code {result.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"Error running benchmark: {e}")
        return False

def run_all_benchmarks(args) -> bool:
    """Run all benchmarks based on command line arguments."""
    # Determine models to benchmark
    models = []
    if args.model == "all":
        models = list(SUPPORTED_MODELS.keys())
    else:
        models = [args.model]
    
    # Determine platforms to benchmark
    platforms = []
    if args.platform == "all":
        platforms = ["webgpu", "webnn"]
    else:
        platforms = [args.platform]
    
    # Determine browsers to benchmark
    browsers = []
    if args.browser == "auto":
        # Use auto-selection for each model
        browsers = None
    else:
        browsers = [args.browser]
    
    # Set up results
    results = []
    
    if args.comprehensive:
        # Run optimized configurations for all models
        logger.info("Running comprehensive benchmarks with optimal configurations")
        configurations = determine_optimal_configurations()
        
        for config in configurations:
            model = config["model"]
            platform = config["platform"]
            browser = config["browser"]
            optimizations = {opt: True for opt in config.get("optimizations", [])}
            
            model_info = SUPPORTED_MODELS.get(model, {})
            model_type = model_info.get("type", "unknown")
            
            logger.info(f"Running benchmark for {model} ({model_type}) on {platform}/{browser}")
            logger.info(f"Optimizations: {list(optimizations.keys())}")
            
            result = run_benchmark(
                args, 
                model=model, 
                platform=platform, 
                browser=browser,
                use_small_model=args.small_models,
                optimizations=optimizations
            )
            
            results.append({
                "model": model,
                "platform": platform,
                "browser": browser,
                "success": result,
                "optimizations": list(optimizations.keys())
            })
    else:
        # Run specific configurations based on arguments
        for model in models:
            for platform in platforms:
                # Determine optimal browser if auto-selected
                if browsers is None:
                    if platform == "webnn":
                        # Prefer Edge for WebNN
                        browser = "edge"
                    else:
                        # Use model-specific recommendation for WebGPU
                        browser = get_optimal_browser_for_model(model)
                    
                    # Skip Firefox with WebNN (not supported)
                    if platform == "webnn" and browser == "firefox":
                        continue
                else:
                    for browser in browsers:
                        # Skip Firefox with WebNN (not supported)
                        if platform == "webnn" and browser == "firefox":
                            logger.warning(f"Skipping {platform}/{browser} (Firefox doesn't support WebNN)")
                            continue
                        
                        # Get model-specific optimizations
                        model_opts = get_recommended_optimizations(model)
                        optimizations = {opt: True for opt in model_opts}
                        
                        result = run_benchmark(
                            args, 
                            model=model, 
                            platform=platform, 
                            browser=browser,
                            use_small_model=args.small_models,
                            optimizations=optimizations
                        )
                        
                        results.append({
                            "model": model,
                            "platform": platform,
                            "browser": browser,
                            "success": result,
                            "optimizations": model_opts
                        })
    
    # Print summary
    successful = sum(1 for r in results if r["success"])
    
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"Total benchmarks: {len(results)}")
    print(f"Successful benchmarks: {successful}")
    print(f"Failed benchmarks: {len(results) - successful}")
    print("="*80)
    
    for i, result in enumerate(results):
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        model = result["model"]
        platform = result["platform"]
        browser = result["browser"]
        opts = ", ".join(result["optimizations"]) if result["optimizations"] else "none"
        
        print(f"{i+1}. {model} ({platform}/{browser}) - {status} - Optimizations: {opts}")
    
    print("="*80)
    
    return successful == len(results)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run real WebNN/WebGPU benchmarks")
    
    # Model selection
    parser.add_argument("--model", choices=list(SUPPORTED_MODELS.keys()) + ["all"], default="bert",
                        help="Model to benchmark")
    
    # Platform selection
    parser.add_argument("--platform", choices=SUPPORTED_PLATFORMS, default="webgpu",
                        help="Platform to benchmark")
    
    # Browser selection
    parser.add_argument("--browser", choices=SUPPORTED_BROWSERS + ["auto"], default="auto",
                        help="Browser to use ('auto' selects the optimal browser for each model)")
    
    # Benchmark options
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of runs for each benchmark")
    parser.add_argument("--warmup", action="store_true",
                        help="Run warmup inference before benchmarking")
    parser.add_argument("--small-models", action="store_true",
                        help="Use smaller model variants for faster benchmarking")
    parser.add_argument("--allow-simulation", action="store_true",
                        help="Allow fallback to simulation if real hardware isn't available")
    
    # Optimization options
    parser.add_argument("--compute-shaders", action="store_true",
                        help="Enable compute shader optimization (recommended for audio models)")
    parser.add_argument("--parallel-loading", action="store_true",
                        help="Enable parallel model loading (recommended for multimodal models)")
    parser.add_argument("--shader-precompile", action="store_true",
                        help="Enable shader precompilation (faster first inference)")
    parser.add_argument("--all-optimizations", action="store_true",
                        help="Enable all optimizations")
    
    # Output options
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations of benchmark results")
    parser.add_argument("--format", choices=["json", "md", "html", "all"], default="all",
                        help="Output format for benchmark results")
    
    # Database options
    parser.add_argument("--db-path", type=str, default=None,
                        help="Path to DuckDB database for storing results")
    parser.add_argument("--db-only", action="store_true",
                        help="Only store results in database, not in files")
    
    # Comprehensive mode
    parser.add_argument("--comprehensive", action="store_true",
                        help="Run comprehensive benchmarks with optimal configurations for all models")
    
    # Debugging options
    parser.add_argument("--visible", action="store_true",
                        help="Make browsers visible (not headless)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set verbose logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check requirements
    if not check_requirements():
        return 1
    
    # Fix test files if needed
    if not fix_test_files():
        logger.warning("Some test files may be missing - benchmarks may fail")
    
    # Update db_path from environment if not specified
    if not args.db_path and "BENCHMARK_DB_PATH" in os.environ:
        args.db_path = os.environ["BENCHMARK_DB_PATH"]
        logger.info(f"Using database path from environment: {args.db_path}")
    
    # Run benchmarks
    success = run_all_benchmarks(args)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())