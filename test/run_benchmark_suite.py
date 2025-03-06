#!/usr/bin/env python
# Suite runner for hardware benchmarks

import os
import sys
import argparse
import logging
import datetime
import json
from pathlib import Path
import subprocess
import time

# Add DuckDB database support
try:
    from benchmark_db_api import BenchmarkDBAPI
    BENCHMARK_DB_AVAILABLE = True
except ImportError:
    BENCHMARK_DB_AVAILABLE = False
    logger.warning("benchmark_db_api not available. Using deprecated JSON fallback.")


# Always deprecate JSON output in favor of DuckDB
DEPRECATE_JSON_OUTPUT = os.environ.get("DEPRECATE_JSON_OUTPUT", "1").lower() in ("1", "true", "yes")


# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Standard benchmark configurations
DEFAULT_OUTPUT_DIR = "./benchmark_results"
DEFAULT_CONFIG_FILE = "./benchmark_config.json"
DEFAULT_BATCH_SIZES = [1, 4, 8]
DEFAULT_SEQUENCE_LENGTHS = [32, 128, 512]
DEFAULT_WARMUP_ITERATIONS = 5
DEFAULT_BENCHMARK_ITERATIONS = 20
DEFAULT_TIMEOUT = 600  # seconds
DEFAULT_MODEL_FAMILIES = ["embedding", "text_generation", "vision", "audio", "multimodal"]
DEFAULT_HARDWARE_TYPES = ["cpu", "cuda", "rocm", "mps", "openvino", "qnn", "webnn", "webgpu"]

# Example benchmark configuration
EXAMPLE_CONFIG = {
    "batch_sizes": [1, 4, 8],
    "sequence_lengths": [32, 128, 512],
    "warmup_iterations": 5,
    "benchmark_iterations": 20,
    "timeout": 600,
    "model_families": {
        "embedding": ["bert-base-uncased", "distilbert-base-uncased", "roberta-base"],
        "text_generation": ["gpt2", "t5-small", "google/flan-t5-small"],
        "vision": ["google/vit-base-patch16-224", "microsoft/resnet-50", "facebook/convnext-tiny-224"],
        "audio": ["openai/whisper-tiny", "facebook/wav2vec2-base"],
        "multimodal": ["openai/clip-vit-base-patch32"]
    },
    "hardware_types": ["cpu", "cuda", "mps", "openvino"],
    "include_web_platforms": False,
    "parallel": True,
    "use_resource_pool": True,
    "generate_plots": True,
    "update_compatibility_matrix": True,
    "schedule": {
        "enabled": False,
        "frequency": "daily",
        "time": "02:00"
    }
}

def check_prerequisites():
    """Check if required components are available"""
    # Check for hardware_benchmark_runner.py
    if not os.path.exists("hardware_benchmark_runner.py"):
        logger.error("hardware_benchmark_runner.py not found. Please ensure it exists in the current directory.")
        return False
    
    # Check for Python dependencies
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("CUDA not available")
        
        # Check MPS availability
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("MPS (Apple Silicon) available")
            
        # Check other dependencies
        import transformers
        logger.info(f"Transformers version: {transformers.__version__}")
        
    except ImportError as e:
        logger.error(f"Missing required dependencies: {e}")
        logger.error("Please install required packages: pip install torch transformers")
        return False
    
    # Check for optional components
    try:
        import matplotlib
        import pandas
        logger.info("Plotting libraries available")
    except ImportError:
        logger.warning("Plotting libraries (matplotlib, pandas) not available. Plotting will be disabled.")
    
    return True

def create_default_config(config_file=DEFAULT_CONFIG_FILE):
    """Create a default benchmark configuration file"""
# JSON output deprecated in favor of database storage
if not DEPRECATE_JSON_OUTPUT:
        with open(config_file, 'w') as f:
            json.dump(EXAMPLE_CONFIG, f, indent=2)
else:
    logger.info("JSON output is deprecated. Results are stored directly in the database.")

    logger.info(f"Created default configuration file: {config_file}")
    return EXAMPLE_CONFIG

def load_config(config_file=DEFAULT_CONFIG_FILE):
    """Load benchmark configuration from file"""
    if not os.path.exists(config_file):
        logger.warning(f"Configuration file {config_file} not found. Creating default configuration.")
        return create_default_config(config_file)
    
    try:
        with open(config_file, 'r') as f:
# Try database first, fall back to JSON if necessary
try:
    from benchmark_db_api import BenchmarkDBAPI
    db_api = BenchmarkDBAPI(db_path=os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb"))
    config = db_api.get_benchmark_results()
    logger.info("Successfully loaded results from database")
except Exception as e:
    logger.warning(f"Error reading from database, falling back to JSON: {e}")
                config = json.load(f)

        logger.info(f"Loaded configuration from {config_file}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_file}: {e}")
        logger.info("Using default configuration instead.")
        return EXAMPLE_CONFIG

def run_benchmark(config, output_dir=DEFAULT_OUTPUT_DIR, models_only=None, hardware_only=None):
    """Run hardware benchmarks with the given configuration"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save current timestamp for output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(output_dir, timestamp)
    os.makedirs(run_output_dir, exist_ok=True)
    
    # Save configuration used for this run
    config_file = os.path.join(run_output_dir, "benchmark_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Determine which model families to benchmark
    model_families = []
    if models_only:
        # Only benchmark specified model families
        model_families = [family for family in models_only if family in config["model_families"]]
    else:
        # Benchmark all model families in config
        model_families = list(config["model_families"].keys())
    
    # Determine which hardware types to benchmark
    hardware_types = hardware_only or config["hardware_types"]
    
    # Run benchmarks for each model family
    results = {}
    for family in model_families:
        logger.info(f"Running benchmarks for {family} models")
        family_results = run_family_benchmark(
            family=family,
            models=config["model_families"][family],
            hardware_types=hardware_types,
            batch_sizes=config.get("batch_sizes", DEFAULT_BATCH_SIZES),
            sequence_lengths=config.get("sequence_lengths", DEFAULT_SEQUENCE_LENGTHS),
            warmup_iterations=config.get("warmup_iterations", DEFAULT_WARMUP_ITERATIONS),
            benchmark_iterations=config.get("benchmark_iterations", DEFAULT_BENCHMARK_ITERATIONS),
            timeout=config.get("timeout", DEFAULT_TIMEOUT),
            parallel=config.get("parallel", True),
            use_resource_pool=config.get("use_resource_pool", True),
            include_web_platforms=config.get("include_web_platforms", False),
            output_dir=run_output_dir
        )
        results[family] = family_results
    
    # Generate plots if enabled
    if config.get("generate_plots", True):
        try:
            plot_benchmark_results(results, run_output_dir)
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
    
    # Update hardware compatibility matrix if enabled
    if config.get("update_compatibility_matrix", True):
        try:
            update_compatibility_matrix(results, run_output_dir)
        except Exception as e:
            logger.error(f"Error updating compatibility matrix: {e}")
    
    return results

def run_family_benchmark(family, models, hardware_types, batch_sizes, sequence_lengths,
                        warmup_iterations, benchmark_iterations, timeout, parallel,
                        use_resource_pool, include_web_platforms, output_dir):
    """Run benchmarks for a specific model family"""
    # Create benchmark command
    cmd = [
        "python", "hardware_benchmark_runner.py",
        "--output-dir", output_dir,
        "--model-families", family,
        "--hardware"] + hardware_types + [
        "--batch-sizes"] + [str(b) for b in batch_sizes] + [
        "--warmup", str(warmup_iterations),
        "--iterations", str(benchmark_iterations),
        "--timeout", str(timeout)
    ]
    
    # Add optional flags
    if parallel:
        cmd.append("--parallel")
    if use_resource_pool:
        cmd.append("--use-resource-pool")
    if include_web_platforms:
        cmd.append("--include-web-platforms")
    
    # Run benchmark
    logger.info(f"Running command: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
# JSON output deprecated in favor of database storage
if not DEPRECATE_JSON_OUTPUT:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Benchmark completed successfully in {time.time() - start_time:.2f} seconds")
# Try database first, fall back to JSON if necessary
try:
    from benchmark_db_api import BenchmarkDBAPI
    db_api = BenchmarkDBAPI(db_path=os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb"))
    results = db_api.get_benchmark_results()
    logger.info("Successfully loaded results from database")
except Exception as e:
    logger.warning(f"Error reading from database, falling back to JSON: {e}")
                
                # Log output

            with open(os.path.join(output_dir, f"benchmark_{family}_output.log"), 'w') as f:
                f.write(result.stdout)
            
            # Return benchmark results
            results_file = os.path.join(output_dir, "benchmark_results.json")
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    return json.load(f)
            
            return {"status": "completed", "output": "See log file for details"}
        except subprocess.CalledProcessError as e:
            logger.error(f"Benchmark failed with return code {e.returncode}")
            logger.error(f"Error output: {e.stderr}")
            
            # Log output and error
            with open(os.path.join(output_dir, f"benchmark_{family}_output.log"), 'w') as f:
                f.write(e.stdout)
            with open(os.path.join(output_dir, f"benchmark_{family}_error.log"), 'w') as f:
                f.write(e.stderr)
            
            return {"status": "failed", "error": str(e)}
# Try database first, fall back to JSON if necessary
try:
    from benchmark_db_api import BenchmarkDBAPI
    db_api = BenchmarkDBAPI(db_path=os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb"))
    results = db_api.get_benchmark_results()
# Try database first, fall back to JSON if necessary
try:
    from benchmark_db_api import BenchmarkDBAPI
    db_api = BenchmarkDBAPI(db_path=os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb"))
    results = db_api.get_benchmark_results()
    logger.info("Successfully loaded results from database")
except Exception as e:
    logger.warning(f"Error reading from database, falling back to JSON: {e}")
        logger.info("Successfully loaded results from database")
    except Exception as e:

    logger.warning(f"Error reading from database, falling back to JSON: {e}")
            except Exception as e:

# Try database first, fall back to JSON if necessary
try:
    from benchmark_db_api import BenchmarkDBAPI
    db_api = BenchmarkDBAPI(db_path=os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb"))
    results = db_api.get_benchmark_results()
    logger.info("Successfully loaded results from database")
except Exception as e:
    logger.warning(f"Error reading from database, falling back to JSON: {e}")
                logger.error(f"Error running benchmark: {e}")
                return {"status": "error", "error": str(e)}

    
    def plot_benchmark_results(results, output_dir):
        """Generate plots from benchmark results"""
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            import numpy as np
        except ImportError:
            logger.warning("Plotting libraries not available. Skipping plot generation.")
            return
        
        # Create plots directory
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Load results from file if path provided
        if isinstance(results, str) and os.path.exists(results):
            with open(results, 'r') as f:
                results = json.load(f)
        
        # Check if results is a path to benchmark_results.json
        if not results and os.path.exists(os.path.join(output_dir, "benchmark_results.json")):
# Try database first, fall back to JSON if necessary
try:
    from benchmark_db_api import BenchmarkDBAPI
    db_api = BenchmarkDBAPI(db_path=os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb"))
    results = db_api.get_benchmark_results()
    logger.info("Successfully loaded results from database")
except Exception as e:
    logger.warning(f"Error reading from database, falling back to JSON: {e}")
                with open(os.path.join(output_dir, "benchmark_results.json"), 'r') as f:
                    results = json.load(f)

        
        # Generate different types of plots
        try:
            # Load all benchmark result files
            result_files = [f for f in os.listdir(output_dir) if f.startswith("benchmark_results_") and f.endswith(".json")]
            if not result_files:
                if os.path.exists(os.path.join(output_dir, "benchmark_results.json")):
                    result_files = ["benchmark_results.json"]
            
            all_results = []
            for file in result_files:
                with open(os.path.join(output_dir, file), 'r') as f:
                    all_results.append(json.load(f))
            
            if not all_results:
                logger.warning("No benchmark results found. Skipping plot generation.")
                return
            
            # Create dataframes for plotting
            plot_hardware_comparison(all_results, plots_dir)
            plot_model_performance(all_results, plots_dir)
            plot_batch_size_scaling(all_results, plots_dir)
            
            logger.info(f"Plots generated in {plots_dir}")
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
    
    def plot_hardware_comparison(results, plots_dir):
        """Plot hardware comparison across model families"""
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        
        # Extract hardware comparison data
        hardware_data = []
        
        for result in results:
            if "benchmarks" not in result:
                continue
                
            for family, models in result["benchmarks"].items():
                for model_name, hw_results in models.items():
                    for hw_type, hw_metrics in hw_results.items():
                        if hw_metrics.get("status") != "completed" or "performance_summary" not in hw_metrics:
                            continue
                        
                        perf = hw_metrics["performance_summary"]
                        if "latency" in perf and "mean" in perf["latency"]:
                            hardware_data.append({
                                "family": family,
                                "model": model_name,
                                "hardware": hw_type,
                                "latency": perf["latency"]["mean"] * 1000,  # ms
                                "throughput": perf["throughput"]["mean"] if "throughput" in perf and "mean" in perf["throughput"] else 0
                            })
        
        if not hardware_data:
            logger.warning("No hardware comparison data available for plotting")
            return
        
        # Create dataframe
        df = pd.DataFrame(hardware_data)
        
        # Plot latency comparison
        plt.figure(figsize=(12, 8))
        df_pivot = df.pivot_table(index="family", columns="hardware", values="latency", aggfunc="mean")
        ax = df_pivot.plot(kind="bar", rot=0)
        plt.title("Average Latency by Model Family and Hardware Platform")
        plt.ylabel("Latency (ms)")
        plt.xlabel("Model Family")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "hardware_latency_comparison.png"))
        
        # Plot throughput comparison
        plt.figure(figsize=(12, 8))
        df_pivot = df.pivot_table(index="family", columns="hardware", values="throughput", aggfunc="mean")
        ax = df_pivot.plot(kind="bar", rot=0)
        plt.title("Average Throughput by Model Family and Hardware Platform")
        plt.ylabel("Throughput (items/sec)")
        plt.xlabel("Model Family")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "hardware_throughput_comparison.png"))
        
        # Calculate relative speedups
        if "cpu" in df["hardware"].unique():
            # Group by family and model to get CPU baseline
            df_cpu = df[df["hardware"] == "cpu"].copy()
            df_cpu.rename(columns={"latency": "cpu_latency", "throughput": "cpu_throughput"}, inplace=True)
            df_cpu = df_cpu[["family", "model", "cpu_latency", "cpu_throughput"]]
            
            # Merge with original data
            df_merged = pd.merge(df, df_cpu, on=["family", "model"])
            
            # Calculate speedups
            df_merged["latency_speedup"] = df_merged["cpu_latency"] / df_merged["latency"]
            df_merged["throughput_speedup"] = df_merged["throughput"] / df_merged["cpu_throughput"]
            
            # Remove CPU rows (speedup=1.0)
            df_speedup = df_merged[df_merged["hardware"] != "cpu"].copy()
            
            # Plot speedups
            plt.figure(figsize=(12, 8))
            df_pivot = df_speedup.pivot_table(index="family", columns="hardware", values="latency_speedup", aggfunc="mean")
            ax = df_pivot.plot(kind="bar", rot=0)
            plt.title("Relative Speedup vs CPU by Model Family")
            plt.ylabel("Speedup Factor (higher is better)")
            plt.xlabel("Model Family")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "hardware_speedup_comparison.png"))
    
    def plot_model_performance(results, plots_dir):
        """Plot performance comparison across models within families"""
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        
        # Extract model performance data
        model_data = []
        
        for result in results:
            if "benchmarks" not in result:
                continue
                
            for family, models in result["benchmarks"].items():
                for model_name, hw_results in models.items():
                    # Find best hardware for this model
                    best_latency = float('inf')
                    best_hw = None
                    best_metrics = None
                    
                    for hw_type, hw_metrics in hw_results.items():
                        if hw_metrics.get("status") != "completed" or "performance_summary" not in hw_metrics:
                            continue
                        
                        perf = hw_metrics["performance_summary"]
                        if "latency" in perf and "mean" in perf["latency"]:
                            latency = perf["latency"]["mean"]
                            if latency < best_latency:
                                best_latency = latency
                                best_hw = hw_type
                                best_metrics = hw_metrics
                    
                    if best_hw and best_metrics:
                        perf = best_metrics["performance_summary"]
                        model_data.append({
                            "family": family,
                            "model": model_name,
                            "best_hardware": best_hw,
                            "latency": perf["latency"]["mean"] * 1000 if "latency" in perf and "mean" in perf["latency"] else 0,  # ms
                            "throughput": perf["throughput"]["mean"] if "throughput" in perf and "mean" in perf["throughput"] else 0
                        })
        
        if not model_data:
            logger.warning("No model performance data available for plotting")
            return
        
        # Create dataframe
        df = pd.DataFrame(model_data)
        
        # Plot latency comparison by family
        for family in df["family"].unique():
            df_family = df[df["family"] == family].copy()
            if len(df_family) <= 1:
                continue
                
            plt.figure(figsize=(12, 6))
            ax = df_family.plot(kind="bar", x="model", y="latency", rot=45)
            plt.title(f"Latency Comparison for {family.title()} Models")
            plt.ylabel("Latency (ms)")
            plt.xlabel("Model")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{family}_model_latency_comparison.png"))
            
            # Plot throughput comparison
            plt.figure(figsize=(12, 6))
            ax = df_family.plot(kind="bar", x="model", y="throughput", rot=45)
            plt.title(f"Throughput Comparison for {family.title()} Models")
            plt.ylabel("Throughput (items/sec)")
            plt.xlabel("Model")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{family}_model_throughput_comparison.png"))
    
    def plot_batch_size_scaling(results, plots_dir):
        """Plot how performance scales with batch size"""
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        
        # Extract batch size scaling data
        batch_data = []
        
        for result in results:
            if "benchmarks" not in result:
                continue
                
            for family, models in result["benchmarks"].items():
                for model_name, hw_results in models.items():
                    for hw_type, hw_metrics in hw_results.items():
                        if hw_metrics.get("status") != "completed" or "benchmark_results" not in hw_metrics:
                            continue
                        
                        for config, config_results in hw_metrics["benchmark_results"].items():
                            if config_results.get("status") != "completed":
                                continue
                                
                            # Extract batch size from config key (e.g., "batch_1_seq_32")
                            if "batch_" in config:
                                parts = config.split("_")
                                batch_idx = parts.index("batch") + 1
                                if batch_idx < len(parts):
                                    try:
                                        batch_size = int(parts[batch_idx])
                                        
                                        # Add to batch data
                                        batch_data.append({
                                            "family": family,
                                            "model": model_name,
                                            "hardware": hw_type,
                                            "batch_size": batch_size,
                                            "latency": config_results.get("avg_latency", 0) * 1000,  # ms
                                            "throughput": config_results.get("throughput", 0)
                                        })
                                    except ValueError:
                                        pass
        
        if not batch_data:
            logger.warning("No batch size scaling data available for plotting")
            return
        
        # Create dataframe
        df = pd.DataFrame(batch_data)
        
        # Plot for CUDA hardware (most interesting for batch scaling)
        if "cuda" in df["hardware"].unique():
            df_cuda = df[df["hardware"] == "cuda"].copy()
            
            # Group by family, model, and batch size
            df_grouped = df_cuda.groupby(["family", "batch_size"]).mean().reset_index()
            
            # Plot batch size vs throughput for each family
            plt.figure(figsize=(12, 8))
            for family in df_grouped["family"].unique():
                family_data = df_grouped[df_grouped["family"] == family]
                plt.plot(family_data["batch_size"], family_data["throughput"], marker="o", label=family)
                
            plt.title("Throughput Scaling with Batch Size (CUDA)")
            plt.xlabel("Batch Size")
            plt.ylabel("Throughput (items/sec)")
            plt.legend()
            plt.grid(linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "batch_size_throughput_scaling.png"))
            
            # Plot batch size vs latency for each family
            plt.figure(figsize=(12, 8))
            for family in df_grouped["family"].unique():
                family_data = df_grouped[df_grouped["family"] == family]
                plt.plot(family_data["batch_size"], family_data["latency"], marker="o", label=family)
                
            plt.title("Latency Scaling with Batch Size (CUDA)")
            plt.xlabel("Batch Size")
            plt.ylabel("Latency (ms)")
            plt.legend()
# Try database first, fall back to JSON if necessary
try:
    from benchmark_db_api import BenchmarkDBAPI
    db_api = BenchmarkDBAPI(db_path=os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb"))
    results = db_api.get_benchmark_results()
    logger.info("Successfully loaded results from database")
# Try database first, fall back to JSON if necessary
try:
    from benchmark_db_api import BenchmarkDBAPI
    db_api = BenchmarkDBAPI(db_path=os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb"))
    results = db_api.get_benchmark_results()
    logger.info("Successfully loaded results from database")
except Exception as e:
    logger.warning(f"Error reading from database, falling back to JSON: {e}")
    except Exception as e:
        logger.warning(f"Error reading from database, falling back to JSON: {e}")

                plt.grid(linestyle="--", alpha=0.7)
                plt.tight_layout()

            plt.savefig(os.path.join(plots_dir, "batch_size_latency_scaling.png"))
            
            # Calculate throughput efficiency (throughput / batch_size)
            df_grouped["throughput_efficiency"] = df_grouped["throughput"] / df_grouped["batch_size"]
            
            # Plot batch size vs throughput efficiency for each family
            plt.figure(figsize=(12, 8))
            for family in df_grouped["family"].unique():
                family_data = df_grouped[df_grouped["family"] == family]
                plt.plot(family_data["batch_size"], family_data["throughput_efficiency"], marker="o", label=family)
                
            plt.title("Throughput Efficiency vs Batch Size (CUDA)")
            plt.xlabel("Batch Size")
            plt.ylabel("Throughput Efficiency (items/sec/batch)")
            plt.legend()
            plt.grid(linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "batch_size_efficiency_scaling.png"))
    
    def update_compatibility_matrix(results, output_dir):
        """Update hardware compatibility matrix based on benchmark results"""
        # Check if hardware_benchmark_runner.py implements this functionality
        # If so, it's already been done during the benchmark
        compatibility_file = os.path.join(output_dir, "hardware_compatibility_matrix.json")
        
        if os.path.exists(compatibility_file):
            logger.info(f"Hardware compatibility matrix already exists at {compatibility_file}")
            return
        
# Try database first, fall back to JSON if necessary
try:
    from benchmark_db_api import BenchmarkDBAPI
    db_api = BenchmarkDBAPI(db_path=os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb"))
    results = db_api.get_benchmark_results()
# Try database first, fall back to JSON if necessary
try:
    from benchmark_db_api import BenchmarkDBAPI
    db_api = BenchmarkDBAPI(db_path=os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb"))
    results = db_api.get_benchmark_results()
    logger.info("Successfully loaded results from database")
except Exception as e:
    logger.warning(f"Error reading from database, falling back to JSON: {e}")
        logger.info("Successfully loaded results from database")
    except Exception as e:
        logger.warning(f"Error reading from database, falling back to JSON: {e}")

            logger.info("Generating hardware compatibility matrix")
            

        # Initialize compatibility matrix
        compatibility_matrix = {
            "timestamp": datetime.datetime.now().isoformat(),
            "hardware_types": [],
            "model_families": {}
        }
        
        # Load results from file if path provided
        if isinstance(results, str) and os.path.exists(results):
            with open(results, 'r') as f:
                results = json.load(f)
        
        # Check if results is a path to benchmark_results.json
        if not results and os.path.exists(os.path.join(output_dir, "benchmark_results.json")):
            with open(os.path.join(output_dir, "benchmark_results.json"), 'r') as f:
                results = json.load(f)
        
        # Extract hardware types
        hardware_types = set()
        
        for result in results:
            if "benchmarks" not in result:
                continue
                
            for family, models in result["benchmarks"].items():
                for model_name, hw_results in models.items():
                    for hw_type in hw_results.keys():
                        hardware_types.add(hw_type)
        
        compatibility_matrix["hardware_types"] = list(hardware_types)
        
        # Extract model families and hardware compatibility
        for result in results:
            if "benchmarks" not in result:
                continue
                
            for family, models in result["benchmarks"].items():
                if family not in compatibility_matrix["model_families"]:
                    compatibility_matrix["model_families"][family] = {
                        "hardware_compatibility": {}
                    }
                
                # Initialize hardware compatibility for this family
                for hw_type in hardware_types:
                    if hw_type not in compatibility_matrix["model_families"][family]["hardware_compatibility"]:
                        compatibility_matrix["model_families"][family]["hardware_compatibility"][hw_type] = {
                            "compatible": False,
                            "performance_rating": None,
                            "benchmark_results": []
                        }
                
                # Update hardware compatibility based on benchmark results
                for model_name, hw_results in models.items():
                    for hw_type, hw_metrics in hw_results.items():
                        # Update compatibility based on benchmark status
                        if hw_metrics.get("status") == "completed":
                            compatibility_matrix["model_families"][family]["hardware_compatibility"][hw_type]["compatible"] = True
                            
                            # Add benchmark result
                            benchmark_summary = {
                                "timestamp": datetime.datetime.now().isoformat(),
                                "model_name": model_name
                            }
                            
                            # Add performance summary if available
                            if "performance_summary" in hw_metrics:
                                perf = hw_metrics["performance_summary"]
                                
                                if "latency" in perf and "mean" in perf["latency"]:
                                    benchmark_summary["mean_latency"] = perf["latency"]["mean"]
                                
                                if "throughput" in perf and "mean" in perf["throughput"]:
                                    benchmark_summary["mean_throughput"] = perf["throughput"]["mean"]
                                
                                if "memory" in perf and "max_allocated" in perf["memory"]:
                                    benchmark_summary["max_memory"] = perf["memory"]["max_allocated"]
                            
                            compatibility_matrix["model_families"][family]["hardware_compatibility"][hw_type]["benchmark_results"].append(benchmark_summary)
                        elif hw_metrics.get("status") in ["load_failed", "benchmark_failed"]:
                            compatibility_matrix["model_families"][family]["hardware_compatibility"][hw_type]["compatible"] = False
                            
                            # Add error information
                            if "error" in hw_metrics:
                                compatibility_matrix["model_families"][family]["hardware_compatibility"][hw_type]["error"] = hw_metrics["error"]
        
        # Calculate performance ratings
        for family, family_data in compatibility_matrix["model_families"].items():
            for hw_type, hw_data in family_data["hardware_compatibility"].items():
                if not hw_data["compatible"]:
                    continue
                    
                # Calculate performance rating based on benchmark results
                benchmark_results = hw_data["benchmark_results"]
                if benchmark_results:
                    # Calculate average throughput across all benchmark results
                    throughputs = [res.get("mean_throughput", 0) for res in benchmark_results if "mean_throughput" in res]
                    if throughputs:
                        avg_throughput = sum(throughputs) / len(throughputs)
                        
                        # Assign performance rating based on throughput
                        if avg_throughput > 0:
                            if hw_type == "cpu":
                                # CPU is baseline
                                rating = "medium"
                            elif avg_throughput > 5:
                                rating = "high"
                            elif avg_throughput > 1:
                                rating = "medium"
                            else:
                                rating = "low"
                                
else:
    logger.info("JSON output is deprecated. Results are stored directly in the database.")

                        hw_data["performance_rating"] = rating
    
    # Save compatibility matrix
    with open(compatibility_file, 'w') as f:
        json.dump(compatibility_matrix, f, indent=2)
    
    logger.info(f"Hardware compatibility matrix saved to {compatibility_file}")

def main():
    """Main function for running benchmark suite"""
    parser = argparse.ArgumentParser(description="Suite runner for hardware benchmarks")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_FILE, help="Path to benchmark configuration file")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for benchmark results")
    parser.add_argument("--create-config", action="store_true", help="Create default configuration file")
    parser.add_argument("--models", type=str, nargs="+", help="Only test specific model families")
    parser.add_argument("--hardware", type=str, nargs="+", help="Only test specific hardware types")
    parser.add_argument("--check", action="store_true", help="Check prerequisites without running benchmarks")
    parser.add_argument("--plot-only", type=str, help="Only generate plots from existing results directory")
    parser.add_argument("--update-matrix-only", type=str, help="Only update compatibility matrix from existing results directory")
    
    parser.add_argument("--db-path", type=str, default=None,
                      help="Path to the benchmark database")
    parser.add_argument("--db-only", action="store_true",
                      help="Store results only in the database, not in JSON")
args = parser.parse_args()
    
    # Create default configuration if requested
    if args.create_config:
        create_default_config(args.config)
        return
    
    # Check prerequisites
    if args.check:
        check_prerequisites()
        return
    
    # Generate plots from existing results
    if args.plot_only:
        if not os.path.exists(args.plot_only):
            logger.error(f"Results directory {args.plot_only} not found")
            return
        plot_benchmark_results(None, args.plot_only)
        return
    
    # Update compatibility matrix from existing results
    if args.update_matrix_only:
        if not os.path.exists(args.update_matrix_only):
            logger.error(f"Results directory {args.update_matrix_only} not found")
            return
        update_compatibility_matrix(None, args.update_matrix_only)
        return
    
    # Check prerequisites before running benchmarks
    if not check_prerequisites():
        logger.error("Prerequisite check failed. Please address the issues before running benchmarks.")
        return
    
    # Load configuration
    config = load_config(args.config)
    
    # Run benchmarks
    run_benchmark(config, args.output_dir, args.models, args.hardware)

if __name__ == "__main__":
    main()