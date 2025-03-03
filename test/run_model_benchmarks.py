#!/usr/bin/env python3
"""
Comprehensive Model Benchmark Runner for IPFS Accelerate

This script provides a user-friendly interface to run benchmarks and verify functionality
of models across different hardware platforms. It builds on the existing hardware_benchmark_runner.py
and makes it easy to benchmark and verify model functionality across different
hardware platforms.

Features:
- Automatic hardware detection with comprehensive compatibility checking
- Benchmarking of key model types across available hardware
- Functionality verification with standardized tests
- Performance profiling with detailed metrics
- Visualization of results with comparative analysis
- Integration with hardware compatibility matrix
- Support for custom model sets and hardware configurations
- Integration with DuckDB database for storing and analyzing benchmark results
"""

import os
import sys
import json
import time
import logging
import argparse
import datetime
import duckdb
from pathlib import Path
import subprocess
from typing import Dict, List, Any, Optional, Union, Tuple

# Add scripts directory to path for module imports
sys.path.append(str(Path(__file__).parent / "scripts"))

# Try to import key components with graceful degradation
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for key model sets
KEY_MODEL_SET = {
    "bert": {"name": "bert-base-uncased", "family": "embedding", "size": "base", "modality": "text"},
    "clap": {"name": "laion/clap-htsat-unfused", "family": "audio", "size": "base", "modality": "audio"},
    "clip": {"name": "openai/clip-vit-base-patch32", "family": "multimodal", "size": "base", "modality": "multimodal"},
    "detr": {"name": "facebook/detr-resnet-50", "family": "vision", "size": "base", "modality": "vision"},
    "llama": {"name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "family": "text_generation", "size": "small", "modality": "text"},
    "llava": {"name": "llava-hf/llava-1.5-7b-hf", "family": "multimodal", "size": "base", "modality": "multimodal"},
    "llava_next": {"name": "llava-hf/llava-v1.6-mistral-7b", "family": "multimodal", "size": "base", "modality": "multimodal"},
    "qwen2": {"name": "Qwen/Qwen2-0.5B-Instruct", "family": "text_generation", "size": "small", "modality": "text"},
    "t5": {"name": "t5-small", "family": "text_generation", "size": "small", "modality": "text"},
    "vit": {"name": "google/vit-base-patch16-224", "family": "vision", "size": "base", "modality": "vision"},
    "wav2vec2": {"name": "facebook/wav2vec2-base", "family": "audio", "size": "base", "modality": "audio"},
    "whisper": {"name": "openai/whisper-tiny", "family": "audio", "size": "small", "modality": "audio"},
    "xclip": {"name": "microsoft/xclip-base-patch32", "family": "multimodal", "size": "base", "modality": "multimodal"}
}

# Small model set for faster testing (useful for CI/CD)
SMALL_MODEL_SET = {
    "bert": {"name": "prajjwal1/bert-tiny", "family": "embedding", "size": "tiny", "modality": "text"},
    "t5": {"name": "google/t5-efficient-tiny", "family": "text_generation", "size": "tiny", "modality": "text"},
    "vit": {"name": "facebook/deit-tiny-patch16-224", "family": "vision", "size": "tiny", "modality": "vision"},
    "whisper": {"name": "openai/whisper-tiny", "family": "audio", "size": "tiny", "modality": "audio"},
    "clip": {"name": "openai/clip-vit-base-patch32", "family": "multimodal", "size": "base", "modality": "multimodal"}
}

# Hardware platforms to test
DEFAULT_HARDWARE_TYPES = ["cpu", "cuda", "mps", "openvino", "rocm"]

# Benchmark parameters
DEFAULT_BATCH_SIZES = [1, 4, 8]
DEFAULT_WARMUP_ITERATIONS = 5
DEFAULT_BENCHMARK_ITERATIONS = 20

class ModelBenchmarkRunner:
    """
    Comprehensive framework for benchmarking and verifying model functionality
    across different hardware platforms.
    """
    
    def __init__(
        self,
        output_dir: str = "./benchmark_results",
        models_set: str = "key",
        custom_models: Optional[Dict[str, Dict[str, str]]] = None,
        hardware_types: Optional[List[str]] = None,
        batch_sizes: Optional[List[int]] = None,
        verify_functionality: bool = True,
        measure_performance: bool = True,
        generate_plots: bool = True,
        update_compatibility_matrix: bool = True,
        use_resource_pool: bool = True,
        db_path: str = "./benchmark_db.duckdb",
        store_in_db: bool = True
    ):
        """
        Initialize the benchmark runner.
        
        Args:
            output_dir: Directory to save benchmark results
            models_set: Which model set to use ('key', 'small', or 'custom')
            custom_models: Custom models to benchmark (if models_set='custom')
            hardware_types: Hardware platforms to test
            batch_sizes: Batch sizes to test
            verify_functionality: Whether to verify basic model functionality
            measure_performance: Whether to measure detailed performance metrics
            generate_plots: Whether to generate visualization plots
            update_compatibility_matrix: Whether to update hardware compatibility matrix
            use_resource_pool: Whether to use ResourcePool for model caching
            db_path: Path to DuckDB database for storing results
            store_in_db: Whether to store results in the database
        """
        self.output_dir = Path(output_dir)
        self.models_set = models_set
        self.custom_models = custom_models
        self.hardware_types = hardware_types or DEFAULT_HARDWARE_TYPES
        self.batch_sizes = batch_sizes or DEFAULT_BATCH_SIZES
        self.verify_functionality = verify_functionality
        self.measure_performance = measure_performance
        self.generate_plots = generate_plots and HAS_VISUALIZATION
        self.update_compatibility_matrix = update_compatibility_matrix
        self.use_resource_pool = use_resource_pool
        self.db_path = db_path
        self.store_in_db = store_in_db
        self.db_conn = None
        
        # Set up models to benchmark
        if models_set == "key":
            self.models = KEY_MODEL_SET
        elif models_set == "small":
            self.models = SMALL_MODEL_SET
        elif models_set == "custom" and custom_models:
            self.models = custom_models
        else:
            logger.warning("Invalid models_set or missing custom_models. Using key model set.")
            self.models = KEY_MODEL_SET
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup timestamp for this run
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / self.timestamp
        self.run_dir.mkdir(exist_ok=True)
        
        # Detect available hardware
        self.available_hardware = self._detect_hardware()
        
        # Initialize results
        self.results = {
            "timestamp": self.timestamp,
            "models": self.models,
            "functionality_verification": {},
            "performance_benchmarks": {},
            "hardware_detected": self.available_hardware
        }
        
        # Initialize database connection if needed
        if self.store_in_db:
            self._initialize_db()
            
    def _initialize_db(self):
        """Initialize database connection and check schema"""
        try:
            # Check if database file exists
            if not os.path.exists(self.db_path):
                logger.warning(f"Database file not found: {self.db_path}")
                logger.warning("Will create new database file")
                
                # Try to import schema creation module
                try:
                    from create_benchmark_schema import create_schema
                    create_schema(self.db_path)
                    logger.info(f"Created new benchmark database at {self.db_path}")
                except ImportError:
                    logger.error("Failed to import create_benchmark_schema module")
                    logger.error("Please run scripts/create_benchmark_schema.py first")
                    self.store_in_db = False
                    return
            
            # Connect to database
            self.db_conn = duckdb.connect(self.db_path)
            
            # Check if required tables exist
            tables = self.db_conn.execute("SHOW TABLES").fetchall()
            table_names = [t[0].lower() for t in tables]
            
            required_tables = ['hardware_platforms', 'models', 'test_runs', 
                             'performance_results', 'hardware_compatibility']
            
            missing_tables = [t for t in required_tables if t.lower() not in table_names]
            
            if missing_tables:
                logger.error(f"Required tables missing from database: {', '.join(missing_tables)}")
                logger.error("Please run scripts/create_benchmark_schema.py to initialize the schema")
                self.store_in_db = False
                return
            
            logger.info(f"Successfully connected to benchmark database: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            self.store_in_db = False
        
    def _detect_hardware(self) -> Dict[str, bool]:
        """
        Detect available hardware platforms.
        
        Returns:
            Dictionary mapping hardware types to availability
        """
        available_hardware = {"cpu": True}  # CPU is always available
        
        # Check for PyTorch and CUDA
        try:
            import torch
            if torch.cuda.is_available():
                available_hardware["cuda"] = True
                logger.info(f"CUDA available with {torch.cuda.device_count()} devices")
            else:
                available_hardware["cuda"] = False
                
            # Check for MPS (Apple Silicon)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                available_hardware["mps"] = True
                logger.info("MPS (Apple Silicon) available")
            else:
                available_hardware["mps"] = False
        except ImportError:
            logger.warning("PyTorch not available for hardware detection")
            available_hardware["cuda"] = False
            available_hardware["mps"] = False
        
        # Check for OpenVINO
        try:
            import openvino
            available_hardware["openvino"] = True
            logger.info(f"OpenVINO available (version {openvino.__version__})")
        except ImportError:
            available_hardware["openvino"] = False
            
        # Check for AMD ROCm
        if os.environ.get("ROCM_HOME"):
            available_hardware["rocm"] = True
            logger.info("AMD ROCm available")
        else:
            available_hardware["rocm"] = False
            
        # Filter hardware types based on availability
        self.hardware_types = [hw for hw in self.hardware_types if available_hardware.get(hw, False)]
        logger.info(f"Available hardware platforms: {', '.join(self.hardware_types)}")
        
        return available_hardware
    
    def run_benchmarks(self):
        """
        Run benchmarks for all specified models on available hardware platforms.
        
        This is the main entry point that:
        1. Verifies basic model functionality (if enabled)
        2. Measures detailed performance metrics (if enabled)
        3. Generates visualization plots (if enabled)
        4. Updates hardware compatibility matrix (if enabled)
        """
        logger.info(f"Starting model benchmarks for {len(self.models)} models on {len(self.hardware_types)} hardware platforms")
        
        try:
            # Create run config file
            self._save_run_configuration()
            
            # Step 1: Verify basic model functionality
            if self.verify_functionality:
                self._verify_model_functionality()
                
            # Step 2: Measure detailed performance metrics
            if self.measure_performance:
                self._run_performance_benchmarks()
                
            # Step 3: Generate visualization plots
            if self.generate_plots:
                self._generate_plots()
                
            # Step 4: Update hardware compatibility matrix
            if self.update_compatibility_matrix:
                self._update_compatibility_matrix()
                
            # Save final results
            self._save_results()
            
            # Generate final report
            report_path = self._generate_report()
            
            logger.info(f"Benchmark run completed. Results saved to {self.run_dir}")
            logger.info(f"Report available at: {report_path}")
            
            return self.results
        
        finally:
            # Close database connection if it exists
            if self.store_in_db and self.db_conn:
                try:
                    self.db_conn.close()
                    logger.info("Database connection closed")
                except Exception as e:
                    logger.error(f"Error closing database connection: {e}")
    
    def _save_run_configuration(self):
        """Save the configuration for this benchmark run"""
        config = {
            "timestamp": self.timestamp,
            "models_set": self.models_set,
            "models": self.models,
            "hardware_types": self.hardware_types,
            "batch_sizes": self.batch_sizes,
            "verify_functionality": self.verify_functionality,
            "measure_performance": self.measure_performance,
            "generate_plots": self.generate_plots,
            "update_compatibility_matrix": self.update_compatibility_matrix,
            "use_resource_pool": self.use_resource_pool,
            "available_hardware": self.available_hardware
        }
        
        config_path = self.run_dir / "benchmark_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to {config_path}")
    
    def _verify_model_functionality(self):
        """
        Verify basic model functionality across hardware platforms.
        
        This runs verify_model_functionality.py to check if models can be loaded
        and provide basic outputs on different hardware platforms.
        """
        logger.info("Verifying model functionality...")
        
        # Create model list for verification
        verify_models = list(self.models.keys())
        
        # Run for each hardware type
        for hardware in self.hardware_types:
            if not self.available_hardware.get(hardware, False):
                logger.warning(f"Hardware {hardware} not available, skipping")
                continue
                
            logger.info(f"Testing functionality on {hardware}...")
            
            try:
                # Determine command based on available scripts
                if os.path.exists("verify_model_functionality.py"):
                    cmd = [
                        sys.executable, "verify_model_functionality.py",
                        "--models"] + verify_models + [
                        "--hardware", hardware,
                        "--output-dir", str(self.run_dir / "functionality"),
                        "--timeout", "300"
                    ]
                else:
                    # Fallback to using the single model verify function
                    self.results["functionality_verification"][hardware] = self._verify_models_manually(verify_models, hardware)
                    continue
                
                # Run the verification command
                logger.info(f"Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, check=False, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"Verification failed on {hardware}: {result.stderr}")
                    self.results["functionality_verification"][hardware] = {
                        "status": "error",
                        "error": result.stderr,
                        "models": {}
                    }
                else:
                    logger.info(f"Verification completed on {hardware}")
                    
                    # Load results
                    results_file = list(self.run_dir.glob(f"functionality/model_functionality_*.json"))
                    if results_file:
                        with open(results_file[-1], 'r') as f:
                            verification_results = json.load(f)
                            self.results["functionality_verification"][hardware] = verification_results
                    else:
                        logger.warning(f"No verification results found for {hardware}")
                        self.results["functionality_verification"][hardware] = {
                            "status": "completed",
                            "warning": "No result file found",
                            "models": {}
                        }
            
            except Exception as e:
                logger.error(f"Error during verification on {hardware}: {e}")
                self.results["functionality_verification"][hardware] = {
                    "status": "error",
                    "error": str(e),
                    "models": {}
                }
        
        logger.info("Model functionality verification completed")
    
    def _verify_models_manually(self, models: List[str], hardware: str) -> Dict[str, Any]:
        """
        Manually verify model functionality when verify_model_functionality.py is not available.
        
        Args:
            models: List of model keys to verify
            hardware: Hardware platform to test on
            
        Returns:
            Dictionary with verification results
        """
        logger.info(f"Manually verifying {len(models)} models on {hardware}")
        
        results = {
            "status": "completed",
            "hardware": hardware,
            "models": {}
        }
        
        for model_key in models:
            if model_key not in self.models:
                logger.warning(f"Model {model_key} not found in model set")
                continue
                
            model_info = self.models[model_key]
            model_name = model_info.get("name", model_key)
            
            logger.info(f"Verifying {model_key} ({model_name}) on {hardware}")
            
            # Set environment variables for hardware
            env = os.environ.copy()
            if hardware == "cpu":
                env["CUDA_VISIBLE_DEVICES"] = ""
            elif hardware == "mps":
                env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            
            # Run test script if available
            test_script = f"skills/test_hf_{model_key}.py"
            if os.path.exists(test_script):
                cmd = [sys.executable, test_script]
                if hardware == "cpu":
                    cmd.append("--cpu-only")
                
                try:
                    start_time = time.time()
                    proc = subprocess.run(cmd, env=env, check=False, capture_output=True, text=True, timeout=300)
                    duration = time.time() - start_time
                    
                    success = proc.returncode == 0
                    results["models"][model_key] = {
                        "success": success,
                        "duration": duration,
                        "error": proc.stderr if not success else None
                    }
                    
                    status = "✅ PASSED" if success else "❌ FAILED"
                    logger.info(f"{status} - {model_key} on {hardware} in {duration:.2f}s")
                except subprocess.TimeoutExpired:
                    logger.error(f"Timeout verifying {model_key} on {hardware}")
                    results["models"][model_key] = {
                        "success": False,
                        "error": "Verification timed out after 300 seconds"
                    }
                except Exception as e:
                    logger.error(f"Error verifying {model_key} on {hardware}: {e}")
                    results["models"][model_key] = {
                        "success": False,
                        "error": str(e)
                    }
            else:
                logger.warning(f"Test script not found for {model_key}")
                results["models"][model_key] = {
                    "success": False,
                    "error": f"Test script {test_script} not found"
                }
        
        # Calculate summary statistics
        total = len(results["models"])
        successful = sum(1 for model, result in results["models"].items() if result.get("success", False))
        results["summary"] = {
            "total": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": (successful / total * 100) if total > 0 else 0
        }
        
        return results
    
    def _run_performance_benchmarks(self):
        """
        Run detailed performance benchmarks using hardware_benchmark_runner.py
        """
        logger.info("Running performance benchmarks...")
        
        # Check if hardware_benchmark_runner.py exists
        if not os.path.exists("hardware_benchmark_runner.py"):
            logger.error("hardware_benchmark_runner.py not found. Performance benchmarks cannot be run.")
            return
        
        # Group models by family for more efficient benchmarking
        models_by_family = {}
        for model_key, model_info in self.models.items():
            family = model_info.get("family", "unknown")
            if family not in models_by_family:
                models_by_family[family] = []
            models_by_family[family].append(model_info)
        
        # Run benchmarks for each family
        for family, models in models_by_family.items():
            logger.info(f"Benchmarking {len(models)} models in {family} family")
            
            # Create custom config for this family
            family_config = {
                "batch_sizes": self.batch_sizes,
                "warmup_iterations": DEFAULT_WARMUP_ITERATIONS,
                "benchmark_iterations": DEFAULT_BENCHMARK_ITERATIONS,
                "timeout": 600,
                "model_families": {family: [model["name"] for model in models]},
                "hardware_types": self.hardware_types,
                "use_resource_pool": self.use_resource_pool,
                "parallel": True,
                "generate_plots": self.generate_plots
            }
            
            # Save family config
            family_config_path = self.run_dir / f"benchmark_config_{family}.json"
            with open(family_config_path, 'w') as f:
                json.dump(family_config, f, indent=2)
            
            # Run benchmark using run_benchmark_suite.py if available
            if os.path.exists("run_benchmark_suite.py"):
                cmd = [
                    sys.executable, "run_benchmark_suite.py",
                    "--config", str(family_config_path),
                    "--output-dir", str(self.run_dir / "performance" / family)
                ]
            else:
                # Fallback to direct hardware_benchmark_runner.py
                cmd = [
                    sys.executable, "hardware_benchmark_runner.py",
                    "--output-dir", str(self.run_dir / "performance" / family),
                    "--model-families", family,
                    "--hardware"] + self.hardware_types + [
                    "--batch-sizes"] + [str(b) for b in self.batch_sizes] + [
                    "--warmup", str(DEFAULT_WARMUP_ITERATIONS),
                    "--iterations", str(DEFAULT_BENCHMARK_ITERATIONS),
                    "--timeout", "600"
                ]
                
                if self.use_resource_pool:
                    cmd.append("--use-resource-pool")
            
            try:
                logger.info(f"Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, check=False, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"Benchmark failed for {family}: {result.stderr}")
                    self.results["performance_benchmarks"][family] = {
                        "status": "error",
                        "error": result.stderr
                    }
                else:
                    logger.info(f"Benchmark completed for {family}")
                    
                    # Find and load results
                    performance_dir = self.run_dir / "performance" / family
                    results_files = list(performance_dir.glob("**/benchmark_results.json"))
                    
                    if results_files:
                        with open(results_files[-1], 'r') as f:
                            benchmark_results = json.load(f)
                            self.results["performance_benchmarks"][family] = benchmark_results
                    else:
                        logger.warning(f"No benchmark results found for {family}")
                        self.results["performance_benchmarks"][family] = {
                            "status": "completed",
                            "warning": "No result file found"
                        }
            except Exception as e:
                logger.error(f"Error during benchmark for {family}: {e}")
                self.results["performance_benchmarks"][family] = {
                    "status": "error",
                    "error": str(e)
                }
        
        logger.info("Performance benchmarks completed")
    
    def _generate_plots(self):
        """
        Generate visualization plots from benchmark results
        """
        if not HAS_VISUALIZATION:
            logger.warning("Plotting libraries not available. Skipping plot generation.")
            return
            
        logger.info("Generating visualization plots...")
        
        # Create plots directory
        plots_dir = self.run_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Check if results are available
        if not self.results.get("functionality_verification") and not self.results.get("performance_benchmarks"):
            logger.warning("No results available for plotting")
            return
        
        try:
            # Plot 1: Functionality verification summary
            self._plot_functionality_summary(plots_dir)
            
            # Plot 2: Hardware comparison
            self._plot_hardware_comparison(plots_dir)
            
            # Plot 3: Model family performance
            self._plot_model_family_performance(plots_dir)
            
            # Plot 4: Batch size scaling
            self._plot_batch_size_scaling(plots_dir)
            
            logger.info(f"Plots generated in {plots_dir}")
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
    
    def _plot_functionality_summary(self, plots_dir: Path):
        """Generate functionality verification summary plot"""
        if not self.results.get("functionality_verification"):
            return
        
        # Extract data
        hardware_platforms = []
        success_rates = []
        
        for hw, results in self.results["functionality_verification"].items():
            if "summary" in results:
                hardware_platforms.append(hw)
                success_rates.append(results["summary"].get("success_rate", 0))
            elif "stats" in results:
                hardware_platforms.append(hw)
                success_rates.append(results["stats"].get("success_rate", 0))
        
        if not hardware_platforms:
            return
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(hardware_platforms, success_rates, color='skyblue')
        plt.title('Model Functionality Success Rate by Hardware Platform')
        plt.xlabel('Hardware Platform')
        plt.ylabel('Success Rate (%)')
        plt.ylim(0, 100)
        
        # Add value labels on top of bars
        for i, v in enumerate(success_rates):
            plt.text(i, v + 2, f'{v:.1f}%', ha='center')
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(plots_dir / "functionality_summary.png")
        plt.close()
    
    def _plot_hardware_comparison(self, plots_dir: Path):
        """Generate hardware performance comparison plot"""
        if not self.results.get("performance_benchmarks"):
            return
        
        # Extract data
        data = []
        
        for family, results in self.results["performance_benchmarks"].items():
            if "benchmarks" not in results:
                continue
                
            for model_name, hw_results in results["benchmarks"].items():
                for hw_type, hw_metrics in hw_results.items():
                    if "performance_summary" not in hw_metrics:
                        continue
                        
                    perf = hw_metrics["performance_summary"]
                    if "latency" in perf and "mean" in perf["latency"]:
                        data.append({
                            "family": family,
                            "model": model_name,
                            "hardware": hw_type,
                            "latency_ms": perf["latency"]["mean"] * 1000,  # Convert to ms
                            "throughput": perf["throughput"]["mean"] if "throughput" in perf and "mean" in perf["throughput"] else 0
                        })
        
        if not data:
            return
        
        # Create dataframe
        df = pd.DataFrame(data)
        
        # Plot latency comparison
        plt.figure(figsize=(12, 8))
        df_pivot = df.pivot_table(index="family", columns="hardware", values="latency_ms", aggfunc="mean")
        ax = df_pivot.plot(kind="bar", rot=0)
        plt.title("Average Latency by Model Family and Hardware Platform")
        plt.ylabel("Latency (ms)")
        plt.xlabel("Model Family")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(plots_dir / "hardware_latency_comparison.png")
        plt.close()
        
        # Plot throughput comparison
        plt.figure(figsize=(12, 8))
        df_pivot = df.pivot_table(index="family", columns="hardware", values="throughput", aggfunc="mean")
        ax = df_pivot.plot(kind="bar", rot=0)
        plt.title("Average Throughput by Model Family and Hardware Platform")
        plt.ylabel("Throughput (items/sec)")
        plt.xlabel("Model Family")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(plots_dir / "hardware_throughput_comparison.png")
        plt.close()
    
    def _plot_model_family_performance(self, plots_dir: Path):
        """Generate model performance comparison within families plot"""
        if not self.results.get("performance_benchmarks"):
            return
        
        # Extract data
        data = []
        
        for family, results in self.results["performance_benchmarks"].items():
            if "benchmarks" not in results:
                continue
                
            # Process each model in the family
            for model_name, hw_results in results["benchmarks"].items():
                # Find best hardware for this model
                best_latency = float('inf')
                best_hw = None
                best_metrics = None
                
                for hw_type, hw_metrics in hw_results.items():
                    if "performance_summary" not in hw_metrics:
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
                    data.append({
                        "family": family,
                        "model": model_name,
                        "best_hardware": best_hw,
                        "latency_ms": perf["latency"]["mean"] * 1000 if "latency" in perf and "mean" in perf["latency"] else 0,  # ms
                        "throughput": perf["throughput"]["mean"] if "throughput" in perf and "mean" in perf["throughput"] else 0
                    })
        
        if not data:
            return
        
        # Create dataframe
        df = pd.DataFrame(data)
        
        # Plot for each family
        for family in df["family"].unique():
            df_family = df[df["family"] == family].copy()
            if len(df_family) <= 1:
                continue
                
            # Plot latency comparison
            plt.figure(figsize=(12, 6))
            ax = df_family.plot(kind="bar", x="model", y="latency_ms", rot=45)
            plt.title(f"Latency Comparison for {family.title()} Models")
            plt.ylabel("Latency (ms)")
            plt.xlabel("Model")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(plots_dir / f"{family}_model_latency_comparison.png")
            plt.close()
            
            # Plot throughput comparison
            plt.figure(figsize=(12, 6))
            ax = df_family.plot(kind="bar", x="model", y="throughput", rot=45)
            plt.title(f"Throughput Comparison for {family.title()} Models")
            plt.ylabel("Throughput (items/sec)")
            plt.xlabel("Model")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(plots_dir / f"{family}_model_throughput_comparison.png")
            plt.close()
    
    def _plot_batch_size_scaling(self, plots_dir: Path):
        """Generate batch size scaling plot"""
        if not self.results.get("performance_benchmarks"):
            return
        
        # Extract batch size scaling data
        batch_data = []
        
        for family, results in self.results["performance_benchmarks"].items():
            if "benchmarks" not in results:
                continue
                
            for model_name, hw_results in results["benchmarks"].items():
                for hw_type, hw_metrics in hw_results.items():
                    if "benchmark_results" not in hw_metrics:
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
                                        "latency_ms": config_results.get("avg_latency", 0) * 1000,  # ms
                                        "throughput": config_results.get("throughput", 0)
                                    })
                                except ValueError:
                                    pass
        
        if not batch_data:
            return
        
        # Create dataframe
        df = pd.DataFrame(batch_data)
        
        # Plot for CUDA hardware (most interesting for batch scaling)
        if "cuda" in df["hardware"].unique():
            df_cuda = df[df["hardware"] == "cuda"].copy()
            
            # Group by family and batch size
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
            plt.savefig(plots_dir / "batch_size_throughput_scaling.png")
            plt.close()
            
            # Plot batch size vs latency for each family
            plt.figure(figsize=(12, 8))
            for family in df_grouped["family"].unique():
                family_data = df_grouped[df_grouped["family"] == family]
                plt.plot(family_data["batch_size"], family_data["latency_ms"], marker="o", label=family)
                
            plt.title("Latency Scaling with Batch Size (CUDA)")
            plt.xlabel("Batch Size")
            plt.ylabel("Latency (ms)")
            plt.legend()
            plt.grid(linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(plots_dir / "batch_size_latency_scaling.png")
            plt.close()
    
    def _update_compatibility_matrix(self):
        """
        Update hardware compatibility matrix based on benchmark results
        """
        logger.info("Updating hardware compatibility matrix...")
        
        # Check if we have results to update with
        if not self.results.get("functionality_verification") and not self.results.get("performance_benchmarks"):
            logger.warning("No results available for compatibility matrix update")
            return
        
        # Load existing compatibility matrix if available
        compatibility_file = self.output_dir / "hardware_compatibility_matrix.json"
        compatibility_matrix = {
            "timestamp": self.timestamp,
            "hardware_types": self.hardware_types,
            "model_families": {}
        }
        
        if compatibility_file.exists():
            try:
                with open(compatibility_file, 'r') as f:
                    compatibility_matrix = json.load(f)
            except Exception as e:
                logger.error(f"Error loading existing compatibility matrix: {e}")
        
        # Update based on functionality verification results
        if self.results.get("functionality_verification"):
            for hw_type, results in self.results["functionality_verification"].items():
                # Extract model results
                model_results = {}
                
                # Handle different result formats
                if "models" in results:
                    model_results = results["models"]
                elif "model_results" in results:
                    model_results = results["model_results"]
                
                # Process each model
                for model_key, result in model_results.items():
                    if model_key not in self.models:
                        continue
                        
                    model_info = self.models[model_key]
                    family = model_info.get("family", "unknown")
                    
                    # Initialize family entry if needed
                    if family not in compatibility_matrix["model_families"]:
                        compatibility_matrix["model_families"][family] = {
                            "hardware_compatibility": {}
                        }
                    
                    # Initialize hardware entry if needed
                    if hw_type not in compatibility_matrix["model_families"][family]["hardware_compatibility"]:
                        compatibility_matrix["model_families"][family]["hardware_compatibility"][hw_type] = {
                            "compatible": False,
                            "performance_rating": None,
                            "benchmark_results": []
                        }
                    
                    # Extract success status (handle different formats)
                    success = False
                    if isinstance(result, dict) and "success" in result:
                        success = result["success"]
                    elif isinstance(result, bool):
                        success = result
                    
                    # Update compatibility
                    compatibility_matrix["model_families"][family]["hardware_compatibility"][hw_type]["compatible"] = success
                    
                    # Add any error information
                    if isinstance(result, dict) and "error" in result and result["error"]:
                        compatibility_matrix["model_families"][family]["hardware_compatibility"][hw_type]["error"] = result["error"]
        
        # Update based on performance benchmark results
        if self.results.get("performance_benchmarks"):
            for family, results in self.results["performance_benchmarks"].items():
                if "benchmarks" not in results:
                    continue
                    
                # Initialize family entry if needed
                if family not in compatibility_matrix["model_families"]:
                    compatibility_matrix["model_families"][family] = {
                        "hardware_compatibility": {}
                    }
                
                # Process each model in the family
                for model_name, hw_results in results["benchmarks"].items():
                    for hw_type, hw_metrics in hw_results.items():
                        # Initialize hardware entry if needed
                        if hw_type not in compatibility_matrix["model_families"][family]["hardware_compatibility"]:
                            compatibility_matrix["model_families"][family]["hardware_compatibility"][hw_type] = {
                                "compatible": False,
                                "performance_rating": None,
                                "benchmark_results": []
                            }
                        
                        # Update compatibility based on status
                        if hw_metrics.get("status") == "completed":
                            compatibility_matrix["model_families"][family]["hardware_compatibility"][hw_type]["compatible"] = True
                            
                            # Add benchmark result
                            benchmark_summary = {
                                "timestamp": self.timestamp,
                                "model_name": model_name
                            }
                            
                            # Add performance metrics if available
                            if "performance_summary" in hw_metrics:
                                perf = hw_metrics["performance_summary"]
                                
                                if "latency" in perf and "mean" in perf["latency"]:
                                    benchmark_summary["mean_latency"] = perf["latency"]["mean"]
                                
                                if "throughput" in perf and "mean" in perf["throughput"]:
                                    benchmark_summary["mean_throughput"] = perf["throughput"]["mean"]
                                
                                if "memory" in perf and "max_allocated" in perf["memory"]:
                                    benchmark_summary["max_memory"] = perf["memory"]["max_allocated"]
                            
                            # Add benchmark result to the list
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
                                
                            hw_data["performance_rating"] = rating
        
        # Save updated compatibility matrix
        with open(compatibility_file, 'w') as f:
            json.dump(compatibility_matrix, f, indent=2)
        
        # Also save a copy with the current run
        run_compatibility_file = self.run_dir / "hardware_compatibility_matrix.json"
        with open(run_compatibility_file, 'w') as f:
            json.dump(compatibility_matrix, f, indent=2)
        
        logger.info(f"Hardware compatibility matrix updated at {compatibility_file}")
        logger.info(f"Compatibility matrix copy saved at {run_compatibility_file}")
    
    def _save_results(self):
        """Save benchmark results to file and database"""
        # Save to JSON file
        results_file = self.run_dir / "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Benchmark results saved to {results_file}")
        
        # Save to database if enabled
        if self.store_in_db and self.db_conn:
            try:
                self._store_results_in_db()
                logger.info("Benchmark results stored in database")
            except Exception as e:
                logger.error(f"Error storing results in database: {e}")
    
    def _store_results_in_db(self):
        """Store benchmark results in DuckDB database"""
        # Create a test run entry
        run_id = self._create_test_run()
        
        # Process functionality verification results
        if self.verify_functionality and self.results.get("functionality_verification"):
            self._store_functionality_results(run_id)
        
        # Process performance benchmark results
        if self.measure_performance and self.results.get("performance_benchmarks"):
            self._store_performance_results(run_id)
        
        # Commit changes
        self.db_conn.commit()
    
    def _create_test_run(self):
        """Create a test run entry in the database"""
        # Get current git info
        git_commit = None
        git_branch = None
        try:
            git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
        except:
            pass
        
        # Create test run entry
        test_name = f"benchmark_run_{self.timestamp}"
        test_type = "benchmark"
        command_line = " ".join(sys.argv)
        metadata = {
            "models_set": self.models_set,
            "hardware_types": self.hardware_types,
            "batch_sizes": self.batch_sizes,
            "verify_functionality": self.verify_functionality,
            "measure_performance": self.measure_performance
        }
        metadata_json = json.dumps(metadata)
        
        # Insert test run
        self.db_conn.execute("""
        INSERT INTO test_runs (test_name, test_type, started_at, completed_at, 
                             git_commit, git_branch, command_line, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            test_name, test_type, 
            datetime.datetime.strptime(self.timestamp, "%Y%m%d_%H%M%S"),
            datetime.datetime.now(),
            git_commit, git_branch, command_line, metadata_json
        ])
        
        # Get the inserted run ID
        run_id = self.db_conn.execute("""
        SELECT run_id FROM test_runs WHERE test_name = ?
        """, [test_name]).fetchone()[0]
        
        return run_id
    
    def _find_or_create_model(self, model_name, model_family, modality):
        """Find or create a model entry in the database"""
        # Check if model exists
        model_id = self.db_conn.execute("""
        SELECT model_id FROM models WHERE model_name = ?
        """, [model_name]).fetchone()
        
        if model_id:
            return model_id[0]
        
        # Create new model
        self.db_conn.execute("""
        INSERT INTO models (model_name, model_family, modality, source)
        VALUES (?, ?, ?, ?)
        """, [model_name, model_family, modality, None])
        
        # Get the inserted ID
        model_id = self.db_conn.execute("""
        SELECT model_id FROM models WHERE model_name = ?
        """, [model_name]).fetchone()[0]
        
        return model_id
    
    def _find_or_create_hardware(self, hardware_type, device_name=None):
        """Find or create a hardware entry in the database"""
        # Check if hardware exists
        query = "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = ?"
        params = [hardware_type]
        
        if device_name:
            query += " AND device_name = ?"
            params.append(device_name)
        
        hardware_id = self.db_conn.execute(query, params).fetchone()
        
        if hardware_id:
            return hardware_id[0]
        
        # Create new hardware
        self.db_conn.execute("""
        INSERT INTO hardware_platforms (hardware_type, device_name)
        VALUES (?, ?)
        """, [hardware_type, device_name])
        
        # Get the inserted ID
        hardware_id = self.db_conn.execute(query, params).fetchone()[0]
        
        return hardware_id
    
    def _store_functionality_results(self, run_id):
        """Store functionality verification results in the database"""
        for hw_type, hw_results in self.results["functionality_verification"].items():
            # Skip if no status or models
            if not hw_results or not isinstance(hw_results, dict):
                continue
            
            # Get hardware ID
            hardware_id = self._find_or_create_hardware(hw_type)
            
            # Process model results
            model_results = {}
            
            # Extract model results based on format (handle different formats)
            if "models" in hw_results:
                model_results = hw_results["models"]
            elif "model_results" in hw_results:
                model_results = hw_results["model_results"]
            
            # Store each model's compatibility result
            for model_key, result in model_results.items():
                if model_key not in self.models:
                    continue
                
                model_info = self.models[model_key]
                model_name = model_info.get("name", model_key)
                model_family = model_info.get("family", "unknown")
                modality = model_info.get("modality", "unknown")
                
                # Get model ID
                model_id = self._find_or_create_model(model_name, model_family, modality)
                
                # Extract success status
                success = False
                error_message = None
                
                if isinstance(result, dict):
                    success = result.get("success", False)
                    error_message = result.get("error")
                elif isinstance(result, bool):
                    success = result
                
                # Add compatibility result
                self.db_conn.execute("""
                INSERT INTO hardware_compatibility (run_id, model_id, hardware_id, is_compatible, 
                                                 error_message, compatibility_score)
                VALUES (?, ?, ?, ?, ?, ?)
                """, [
                    run_id, model_id, hardware_id, success, 
                    error_message, 1.0 if success else 0.0
                ])
    
    def _store_performance_results(self, run_id):
        """Store performance benchmark results in the database"""
        for family, results in self.results["performance_benchmarks"].items():
            if "benchmarks" not in results:
                continue
            
            for model_name, hw_results in results["benchmarks"].items():
                for hw_type, hw_metrics in hw_results.items():
                    # Skip if no performance summary
                    if "performance_summary" not in hw_metrics:
                        continue
                    
                    # Extract performance summary
                    perf = hw_metrics["performance_summary"]
                    
                    # Extract model metadata
                    model_family = family
                    modality = self._get_modality_from_family(family)
                    
                    # Get model and hardware IDs
                    model_id = self._find_or_create_model(model_name, model_family, modality)
                    hardware_id = self._find_or_create_hardware(hw_type)
                    
                    # Extract batch size information from benchmark results
                    batch_results = []
                    
                    if "benchmark_results" in hw_metrics:
                        for config, config_result in hw_metrics["benchmark_results"].items():
                            if "batch_" in config and config_result.get("status") == "completed":
                                # Extract batch size from config name (e.g., "batch_8_seq_128")
                                parts = config.split("_")
                                batch_idx = parts.index("batch") + 1
                                if batch_idx < len(parts):
                                    try:
                                        batch_size = int(parts[batch_idx])
                                        
                                        # Add to batch results
                                        batch_results.append({
                                            "batch_size": batch_size,
                                            "latency": config_result.get("avg_latency", 0) * 1000,  # ms
                                            "throughput": config_result.get("throughput", 0),
                                            "memory_peak": config_result.get("memory_peak_mb", 0)
                                        })
                                    except ValueError:
                                        pass
                    
                    # Store results for each batch size
                    if batch_results:
                        for batch_result in batch_results:
                            self._store_single_performance_result(
                                run_id, model_id, hardware_id, 
                                family, batch_result["batch_size"],
                                batch_result["latency"], batch_result["throughput"], 
                                batch_result["memory_peak"]
                            )
                    else:
                        # Store aggregate results if no batch-specific results
                        latency_ms = 0
                        if "latency" in perf and "mean" in perf["latency"]:
                            latency_ms = perf["latency"]["mean"] * 1000  # Convert to ms
                        
                        throughput = 0
                        if "throughput" in perf and "mean" in perf["throughput"]:
                            throughput = perf["throughput"]["mean"]
                        
                        memory_mb = 0
                        if "memory" in perf and "max_allocated" in perf["memory"]:
                            memory_mb = perf["memory"]["max_allocated"] / (1024 * 1024)  # Convert to MB
                        
                        # Use default batch size if not specified
                        batch_size = self.batch_sizes[0] if self.batch_sizes else 1
                        
                        self._store_single_performance_result(
                            run_id, model_id, hardware_id, 
                            family, batch_size, 
                            latency_ms, throughput, memory_mb
                        )
    
    def _store_single_performance_result(self, run_id, model_id, hardware_id, 
                                        test_case, batch_size, latency_ms, 
                                        throughput, memory_mb):
        """Store a single performance result in the database"""
        self.db_conn.execute("""
        INSERT INTO performance_results (
            run_id, model_id, hardware_id, test_case, batch_size,
            average_latency_ms, throughput_items_per_second, memory_peak_mb
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            run_id, model_id, hardware_id, test_case, batch_size,
            latency_ms, throughput, memory_mb
        ])
    
    def _get_modality_from_family(self, family):
        """Get modality based on model family"""
        if family in ["embedding", "text_generation"]:
            return "text"
        elif family in ["vision"]:
            return "image"
        elif family in ["audio"]:
            return "audio"
        elif family in ["multimodal"]:
            return "multimodal"
        else:
            return "unknown"
    
    def _generate_report(self) -> Path:
        """
        Generate a comprehensive markdown report of benchmark results
        
        Returns:
            Path to the generated report file
        """
        report_file = self.run_dir / "benchmark_report.md"
        
        with open(report_file, 'w') as f:
            # Header
            f.write(f"# Model Benchmark Report\n\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            f.write("## Summary\n\n")
            
            # Hardware platforms
            f.write("### Hardware Platforms\n\n")
            f.write("| Hardware | Available | Used in Benchmarks |\n")
            f.write("|----------|-----------|-------------------|\n")
            
            for hw_type in sorted(self.available_hardware.keys()):
                available = "✅" if self.available_hardware[hw_type] else "❌"
                used = "✅" if hw_type in self.hardware_types else "❌"
                f.write(f"| {hw_type} | {available} | {used} |\n")
            
            f.write("\n")
            
            # Models tested
            f.write("### Models Tested\n\n")
            f.write("| Model Key | Full Name | Family | Size | Modality |\n")
            f.write("|-----------|-----------|--------|------|----------|\n")
            
            for model_key, model_info in self.models.items():
                name = model_info.get("name", model_key)
                family = model_info.get("family", "unknown")
                size = model_info.get("size", "unknown")
                modality = model_info.get("modality", "unknown")
                f.write(f"| {model_key} | {name} | {family} | {size} | {modality} |\n")
            
            f.write("\n")
            
            # Functionality verification results
            if self.results.get("functionality_verification"):
                f.write("## Functionality Verification Results\n\n")
                
                # Success rates by hardware
                f.write("### Success Rates by Hardware\n\n")
                f.write("| Hardware | Success Rate | Models Tested | Successful | Failed |\n")
                f.write("|----------|--------------|---------------|------------|--------|\n")
                
                for hw_type, results in self.results["functionality_verification"].items():
                    # Extract summary stats (handle different formats)
                    success_rate = 0
                    total = 0
                    successful = 0
                    failed = 0
                    
                    if "summary" in results:
                        success_rate = results["summary"].get("success_rate", 0)
                        total = results["summary"].get("total", 0)
                        successful = results["summary"].get("successful", 0)
                        failed = results["summary"].get("failed", 0)
                    elif "stats" in results:
                        success_rate = results["stats"].get("success_rate", 0)
                        total = results["stats"].get("total_tests", 0)
                        successful = results["stats"].get("successful_tests", 0)
                        failed = results["stats"].get("failed_tests", 0)
                    
                    f.write(f"| {hw_type} | {success_rate:.1f}% | {total} | {successful} | {failed} |\n")
                
                f.write("\n")
                
                # Results by model and hardware
                f.write("### Results by Model and Hardware\n\n")
                f.write("| Model |")
                for hw_type in self.hardware_types:
                    f.write(f" {hw_type} |")
                f.write("\n")
                
                f.write("|-------|")
                for _ in self.hardware_types:
                    f.write("---|")
                f.write("\n")
                
                for model_key in self.models.keys():
                    f.write(f"| {model_key} |")
                    
                    for hw_type in self.hardware_types:
                        # Check if we have results for this hardware
                        if hw_type not in self.results["functionality_verification"]:
                            f.write(" - |")
                            continue
                        
                        hw_results = self.results["functionality_verification"][hw_type]
                        
                        # Extract model results (handle different formats)
                        success = False
                        
                        if "models" in hw_results and model_key in hw_results["models"]:
                            result = hw_results["models"][model_key]
                            if isinstance(result, dict) and "success" in result:
                                success = result["success"]
                            elif isinstance(result, bool):
                                success = result
                        elif "model_results" in hw_results and model_key in hw_results["model_results"]:
                            success = hw_results["model_results"][model_key]
                        
                        status = "✅" if success else "❌"
                        f.write(f" {status} |")
                    
                    f.write("\n")
                
                f.write("\n")
                
                # Failures
                f.write("### Failed Tests\n\n")
                
                has_failures = False
                
                for hw_type, results in self.results["functionality_verification"].items():
                    # Extract model results (handle different formats)
                    model_results = {}
                    
                    if "models" in results:
                        model_results = results["models"]
                    elif "model_results" in results:
                        model_results = results["model_results"]
                    
                    # Check for failures
                    failures = []
                    
                    for model_key, result in model_results.items():
                        success = False
                        error = None
                        
                        if isinstance(result, dict):
                            success = result.get("success", False)
                            error = result.get("error")
                        elif isinstance(result, bool):
                            success = result
                        
                        if not success:
                            failures.append((model_key, error))
                    
                    if failures:
                        has_failures = True
                        f.write(f"#### {hw_type}\n\n")
                        
                        for model_key, error in failures:
                            f.write(f"- **{model_key}**: ")
                            if error:
                                f.write(f"{error}\n")
                            else:
                                f.write("Unknown error\n")
                        
                        f.write("\n")
                
                if not has_failures:
                    f.write("No failures detected! All models passed functionality verification.\n\n")
            
            # Performance benchmark results
            if self.results.get("performance_benchmarks"):
                f.write("## Performance Benchmark Results\n\n")
                
                # Results by family and hardware
                for family, results in self.results["performance_benchmarks"].items():
                    if "benchmarks" not in results:
                        continue
                        
                    f.write(f"### {family.title()} Models\n\n")
                    
                    # Latency comparison
                    f.write("#### Latency Comparison (ms)\n\n")
                    f.write("| Model |")
                    for hw_type in self.hardware_types:
                        f.write(f" {hw_type} |")
                    f.write("\n")
                    
                    f.write("|-------|")
                    for _ in self.hardware_types:
                        f.write("---|")
                    f.write("\n")
                    
                    for model_name, hw_results in results["benchmarks"].items():
                        f.write(f"| {model_name} |")
                        
                        for hw_type in self.hardware_types:
                            # Check if we have results for this hardware
                            if hw_type not in hw_results:
                                f.write(" - |")
                                continue
                            
                            hw_metrics = hw_results[hw_type]
                            
                            # Extract latency
                            latency_ms = None
                            
                            if "performance_summary" in hw_metrics:
                                perf = hw_metrics["performance_summary"]
                                if "latency" in perf and "mean" in perf["latency"]:
                                    latency_ms = perf["latency"]["mean"] * 1000  # Convert to ms
                            
                            if latency_ms is not None:
                                f.write(f" {latency_ms:.2f} |")
                            else:
                                f.write(" - |")
                        
                        f.write("\n")
                    
                    f.write("\n")
                    
                    # Throughput comparison
                    f.write("#### Throughput Comparison (items/sec)\n\n")
                    f.write("| Model |")
                    for hw_type in self.hardware_types:
                        f.write(f" {hw_type} |")
                    f.write("\n")
                    
                    f.write("|-------|")
                    for _ in self.hardware_types:
                        f.write("---|")
                    f.write("\n")
                    
                    for model_name, hw_results in results["benchmarks"].items():
                        f.write(f"| {model_name} |")
                        
                        for hw_type in self.hardware_types:
                            # Check if we have results for this hardware
                            if hw_type not in hw_results:
                                f.write(" - |")
                                continue
                            
                            hw_metrics = hw_results[hw_type]
                            
                            # Extract throughput
                            throughput = None
                            
                            if "performance_summary" in hw_metrics:
                                perf = hw_metrics["performance_summary"]
                                if "throughput" in perf and "mean" in perf["throughput"]:
                                    throughput = perf["throughput"]["mean"]
                            
                            if throughput is not None:
                                f.write(f" {throughput:.2f} |")
                            else:
                                f.write(" - |")
                        
                        f.write("\n")
                    
                    f.write("\n")
                    
                    # Batch size scaling (if available)
                    batch_results = {}
                    
                    # Check if we have batch size results
                    for model_name, hw_results in results["benchmarks"].items():
                        for hw_type, hw_metrics in hw_results.items():
                            if "benchmark_results" not in hw_metrics:
                                continue
                                
                            for config, config_results in hw_metrics["benchmark_results"].items():
                                if "batch_" in config and config_results.get("status") == "completed":
                                    if hw_type not in batch_results:
                                        batch_results[hw_type] = True
                    
                    if batch_results:
                        f.write("#### Batch Size Scaling\n\n")
                        f.write("Performance scaling with different batch sizes:\n\n")
                        
                        for hw_type in batch_results.keys():
                            f.write(f"- See batch size scaling plots for {hw_type} in the plots directory\n")
                        
                        f.write("\n")
            
            # Hardware compatibility matrix update
            f.write("## Hardware Compatibility Matrix\n\n")
            
            # Load the compatibility matrix if available
            compatibility_file = self.run_dir / "hardware_compatibility_matrix.json"
            
            if compatibility_file.exists():
                try:
                    with open(compatibility_file, 'r') as matrix_file:
                        matrix = json.load(matrix_file)
                        
                    f.write("### Model Family Compatibility\n\n")
                    f.write("| Model Family |")
                    for hw_type in matrix["hardware_types"]:
                        f.write(f" {hw_type} |")
                    f.write("\n")
                    
                    f.write("|--------------|")
                    for _ in matrix["hardware_types"]:
                        f.write("---|")
                    f.write("\n")
                    
                    for family, family_data in matrix["model_families"].items():
                        f.write(f"| {family} |")
                        
                        for hw_type in matrix["hardware_types"]:
                            compatibility = False
                            rating = None
                            
                            if "hardware_compatibility" in family_data and hw_type in family_data["hardware_compatibility"]:
                                hw_data = family_data["hardware_compatibility"][hw_type]
                                compatibility = hw_data.get("compatible", False)
                                rating = hw_data.get("performance_rating")
                            
                            if compatibility:
                                if rating:
                                    f.write(f" ✅ {rating.title()} |")
                                else:
                                    f.write(" ✅ |")
                            else:
                                f.write(" ❌ |")
                        
                        f.write("\n")
                    
                    f.write("\n")
                    
                    # Hardware-specific issues
                    f.write("### Hardware-Specific Issues\n\n")
                    
                    has_issues = False
                    
                    for hw_type in matrix["hardware_types"]:
                        issues = []
                        
                        for family, family_data in matrix["model_families"].items():
                            if "hardware_compatibility" in family_data and hw_type in family_data["hardware_compatibility"]:
                                hw_data = family_data["hardware_compatibility"][hw_type]
                                
                                if not hw_data.get("compatible", False) and "error" in hw_data:
                                    issues.append((family, hw_data["error"]))
                        
                        if issues:
                            has_issues = True
                            f.write(f"#### {hw_type}\n\n")
                            
                            for family, error in issues:
                                f.write(f"- **{family}**: {error}\n")
                            
                            f.write("\n")
                    
                    if not has_issues:
                        f.write("No specific issues identified.\n\n")
                        
                except Exception as e:
                    f.write(f"Error loading compatibility matrix: {e}\n\n")
            else:
                f.write("Compatibility matrix not available.\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            # Generate recommendations based on benchmark results
            recommendations = self._generate_recommendations()
            
            if recommendations:
                for category, rec_list in recommendations.items():
                    f.write(f"### {category}\n\n")
                    
                    for rec in rec_list:
                        f.write(f"- {rec}\n")
                    
                    f.write("\n")
            else:
                f.write("No specific recommendations available.\n\n")
            
            # Next steps
            f.write("## Next Steps\n\n")
            
            if self.results.get("functionality_verification"):
                # Check for failures
                has_failures = False
                
                for hw_type, results in self.results["functionality_verification"].items():
                    # Extract summary stats (handle different formats)
                    failed = 0
                    
                    if "summary" in results:
                        failed = results["summary"].get("failed", 0)
                    elif "stats" in results:
                        failed = results["stats"].get("failed_tests", 0)
                    
                    if failed > 0:
                        has_failures = True
                        break
                
                if has_failures:
                    f.write("1. Investigate and fix the failing tests\n")
                    f.write("2. Re-run the verification to confirm fixes\n")
                    f.write("3. Run performance benchmarks on successfully verified models\n")
                else:
                    f.write("1. All functionality tests passed! Proceed with performance optimization\n")
                    f.write("2. Implement model compression techniques for resource-constrained environments\n")
                    f.write("3. Consider expanding to multi-node testing\n")
            else:
                f.write("1. Run functionality verification tests\n")
                f.write("2. Perform detailed performance benchmarks\n")
                f.write("3. Update hardware compatibility matrix\n")
            
            f.write("\n")
            
            # Footer
            f.write("---\n\n")
            f.write(f"Generated by Model Benchmark Runner on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        logger.info(f"Report generated at {report_file}")
        return report_file
    
    def _generate_recommendations(self) -> Dict[str, List[str]]:
        """
        Generate recommendations based on benchmark results
        
        Returns:
            Dictionary of recommendation categories and lists of recommendations
        """
        recommendations = {
            "Hardware Selection": [],
            "Model Selection": [],
            "Performance Optimization": [],
            "Compatibility Issues": []
        }
        
        # Generate hardware selection recommendations
        if self.results.get("performance_benchmarks"):
            best_hardware = {}
            
            for family, results in self.results["performance_benchmarks"].items():
                if "benchmarks" not in results:
                    continue
                    
                family_latencies = {}
                
                for hw_type in self.hardware_types:
                    if not self.available_hardware.get(hw_type, False):
                        continue
                        
                    latencies = []
                    
                    for model_name, hw_results in results["benchmarks"].items():
                        if hw_type not in hw_results:
                            continue
                            
                        hw_metrics = hw_results[hw_type]
                        
                        if "performance_summary" in hw_metrics:
                            perf = hw_metrics["performance_summary"]
                            if "latency" in perf and "mean" in perf["latency"]:
                                latencies.append(perf["latency"]["mean"])
                    
                    if latencies:
                        family_latencies[hw_type] = sum(latencies) / len(latencies)
                
                if family_latencies:
                    best_hw = min(family_latencies, key=family_latencies.get)
                    best_hardware[family] = best_hw
            
            # Add recommendations based on best hardware
            for family, hw_type in best_hardware.items():
                recommendations["Hardware Selection"].append(f"Use {hw_type} for {family} models for best performance")
        
        # Add general performance optimization recommendations
        recommendations["Performance Optimization"].append("Consider using smaller batch sizes for latency-sensitive applications")
        recommendations["Performance Optimization"].append("Increase batch sizes for throughput-oriented workloads")
        recommendations["Performance Optimization"].append("Enable model caching when running multiple inferences with the same model")
        
        # Add compatibility issue recommendations
        compatibility_file = self.run_dir / "hardware_compatibility_matrix.json"
        
        if compatibility_file.exists():
            try:
                with open(compatibility_file, 'r') as f:
                    matrix = json.load(f)
                    
                for family, family_data in matrix["model_families"].items():
                    if "hardware_compatibility" not in family_data:
                        continue
                        
                    incompatible_hw = []
                    
                    for hw_type, hw_data in family_data["hardware_compatibility"].items():
                        if not hw_data.get("compatible", False):
                            incompatible_hw.append(hw_type)
                    
                    if incompatible_hw:
                        recommendations["Compatibility Issues"].append(f"{family} models are not compatible with: {', '.join(incompatible_hw)}")
            except Exception as e:
                logger.error(f"Error loading compatibility matrix for recommendations: {e}")
        
        return recommendations

def main():
    """Main function for running model benchmarks from command line"""
    parser = argparse.ArgumentParser(description="Comprehensive Model Benchmark Runner")
    parser.add_argument("--output-dir", type=str, default="./benchmark_results", help="Output directory for benchmark results")
    parser.add_argument("--models-set", choices=["key", "small", "custom"], default="key", help="Which model set to use")
    parser.add_argument("--custom-models", type=str, help="JSON file with custom models configuration (required if models-set=custom)")
    parser.add_argument("--hardware", type=str, nargs="+", help="Hardware platforms to test (defaults to all available)")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES, help="Batch sizes to test")
    parser.add_argument("--verify-only", action="store_true", help="Only verify functionality without performance benchmarks")
    parser.add_argument("--benchmark-only", action="store_true", help="Only run performance benchmarks without verification")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation")
    parser.add_argument("--no-compatibility-update", action="store_true", help="Disable compatibility matrix update")
    parser.add_argument("--no-resource-pool", action="store_true", help="Disable ResourcePool for model caching")
    parser.add_argument("--specific-models", type=str, nargs="+", help="Only benchmark specific models (by key) from the selected set")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    # Database-related arguments
    parser.add_argument("--db-path", type=str, default="./benchmark_db.duckdb", help="Path to DuckDB database for storing results")
    parser.add_argument("--no-db-store", action="store_true", help="Disable storing results in the database")
    parser.add_argument("--visualize-from-db", action="store_true", help="Generate visualizations from database instead of current run results")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Handle custom models
    custom_models = None
    if args.models_set == "custom":
        if not args.custom_models:
            logger.error("--custom-models is required when using --models-set=custom")
            return
        
        try:
            with open(args.custom_models, 'r') as f:
                custom_models = json.load(f)
        except Exception as e:
            logger.error(f"Error loading custom models: {e}")
            return
    
    # Handle specific models
    if args.specific_models:
        if args.models_set == "key":
            model_set = {k: v for k, v in KEY_MODEL_SET.items() if k in args.specific_models}
        elif args.models_set == "small":
            model_set = {k: v for k, v in SMALL_MODEL_SET.items() if k in args.specific_models}
        elif args.models_set == "custom":
            model_set = {k: v for k, v in custom_models.items() if k in args.specific_models}
        
        # Check if we have any models after filtering
        if not model_set:
            logger.error(f"No models found matching the specified keys: {args.specific_models}")
            return
        
        custom_models = model_set
        args.models_set = "custom"
    
    # Create and run benchmarks
    runner = ModelBenchmarkRunner(
        output_dir=args.output_dir,
        models_set=args.models_set,
        custom_models=custom_models,
        hardware_types=args.hardware,
        batch_sizes=args.batch_sizes,
        verify_functionality=not args.benchmark_only,
        measure_performance=not args.verify_only,
        generate_plots=not args.no_plots,
        update_compatibility_matrix=not args.no_compatibility_update,
        use_resource_pool=not args.no_resource_pool,
        db_path=args.db_path,
        store_in_db=not args.no_db_store
    )
    
    results = runner.run_benchmarks()
    
    # Print short summary
    print("\nBenchmark Summary:")
    
    # Verification summary
    if not args.benchmark_only and results.get("functionality_verification"):
        print("\nFunctionality Verification:")
        for hw_type, hw_results in results["functionality_verification"].items():
            success_rate = 0
            if "summary" in hw_results:
                success_rate = hw_results["summary"].get("success_rate", 0)
            elif "stats" in hw_results:
                success_rate = hw_results["stats"].get("success_rate", 0)
            
            print(f"  {hw_type}: {success_rate:.1f}% success rate")
    
    # Performance summary
    if not args.verify_only and results.get("performance_benchmarks"):
        print("\nPerformance Benchmarks:")
        for family in results["performance_benchmarks"].keys():
            print(f"  {family}: Benchmarks completed")
    
    # Database storage status
    if runner.store_in_db and runner.db_conn:
        print("\nResults stored in database:")
        print(f"  Database path: {runner.db_path}")
    
    print(f"\nFull results saved to: {os.path.join(args.output_dir, runner.timestamp)}")
    
    # Additional database visualization if requested
    if args.visualize_from_db and not args.no_db_store:
        try:
            from scripts.benchmark_db_query import generate_report
            print("\nGenerating database visualizations...")
            report_path = generate_report(
                db_path=args.db_path,
                report_type="performance",
                output_format="html",
                output_file=os.path.join(args.output_dir, runner.timestamp, "db_performance_report.html")
            )
            print(f"Database report generated: {report_path}")
        except ImportError:
            print("Could not generate database visualizations - benchmark_db_query module not found")
        except Exception as e:
            print(f"Error generating database visualizations: {e}")

if __name__ == "__main__":
    main()