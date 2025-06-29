#!/usr/bin/env python3
"""
Complete Benchmark Pipeline for HuggingFace Models

This script provides a complete end-to-end pipeline for:
1. Generating model skillsets
2. Creating benchmark files
3. Running benchmarks across multiple hardware backends
4. Storing results in DuckDB
5. Generating reports and visualizations

It serves as the main entry point for comprehensive benchmark execution.
"""

import os
import sys
import logging
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from datetime import datetime
import concurrent.futures

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import refactored benchmark suite components
from benchmark_core.registry import BenchmarkRegistry
from benchmark_core.runner import BenchmarkRunner
from benchmark_core.db_integration import BenchmarkDBManager, BenchmarkDBContext
from benchmark_core.huggingface_integration import (
    ModelArchitectureRegistry, 
    ModelLoader,
    get_priority_models,
    generate_model_benchmark_configs
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("benchmark_pipeline.log")
    ]
)
logger = logging.getLogger(__name__)

class BenchmarkPipeline:
    """Complete pipeline for HuggingFace model benchmarking."""
    
    def __init__(self, 
               db_path: Optional[str] = None, 
               output_dir: Optional[str] = None,
               skillset_output_dir: Optional[str] = None,
               benchmark_output_dir: Optional[str] = None,
               refactored_generator_path: Optional[str] = None,
               parallel_jobs: int = 4,
               interactive: bool = True):
        """
        Initialize the benchmark pipeline.
        
        Args:
            db_path: Path to the DuckDB database
            output_dir: Root directory for output files
            skillset_output_dir: Directory for generated skillset files
            benchmark_output_dir: Directory for generated benchmark files
            refactored_generator_path: Path to the refactored generator suite
            parallel_jobs: Number of parallel jobs for benchmark execution
            interactive: Whether to show progress information interactively
        """
        self.db_path = db_path or os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
        
        # Set output directories
        self.output_dir = Path(output_dir or "./benchmark_output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Path to refactored generator suite
        self.refactored_generator_path = Path(refactored_generator_path or str(Path(__file__).resolve().parent.parent / "refactored_generator_suite"))
        
        # Set skillset and benchmark output directories
        self.skillset_output_dir = Path(skillset_output_dir or str(self.output_dir / "skillsets"))
        self.benchmark_output_dir = Path(benchmark_output_dir or str(self.output_dir / "benchmarks"))
        
        # Ensure directories exist
        os.makedirs(self.skillset_output_dir, exist_ok=True)
        os.makedirs(self.benchmark_output_dir, exist_ok=True)
        
        # Configure database
        self.db_manager = BenchmarkDBManager(self.db_path, auto_create=True)
        
        # Configure benchmark runner
        self.runner = BenchmarkRunner()
        
        # Parallel execution settings
        self.parallel_jobs = parallel_jobs
        self.interactive = interactive
        
        # Track progress
        self.start_time = datetime.now()
        self.progress = {
            "skillset_generation": {
                "total": 0,
                "completed": 0,
                "failed": 0
            },
            "benchmark_generation": {
                "total": 0,
                "completed": 0,
                "failed": 0
            },
            "benchmark_execution": {
                "total": 0,
                "completed": 0,
                "failed": 0
            },
            "report_generation": {
                "total": 0,
                "completed": 0,
                "failed": 0
            }
        }
    
    def generate_model_skillsets(self, priority: str = "high", model_filter: Optional[List[str]] = None) -> bool:
        """
        Generate model skillset implementations.
        
        Args:
            priority: Priority level (critical, high, medium, low, all)
            model_filter: Optional list of specific models to generate
        
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Generating skillsets for {priority} priority models...")
        
        # Check if refactored generator suite exists
        if not self.refactored_generator_path.exists():
            logger.error(f"Refactored generator suite not found at {self.refactored_generator_path}")
            return False
        
        # Prepare command for skillset generation
        generate_script = self.refactored_generator_path / "generate_all_skillsets.py"
        if not generate_script.exists():
            logger.error(f"Skillset generator script not found at {generate_script}")
            return False
        
        # Build command
        cmd_args = [
            sys.executable,
            str(generate_script),
            f"--priority={priority}",
            f"--output-dir={self.skillset_output_dir}"
        ]
        
        # Add model filter if specified
        if model_filter:
            cmd_args.extend([f"--model={model}" for model in model_filter])
        
        # Execute command
        logger.info(f"Running command: {' '.join(cmd_args)}")
        try:
            import subprocess
            result = subprocess.run(cmd_args, check=True, capture_output=True, text=True)
            
            # Check output for error messages
            if "ERROR" in result.stdout or "ERROR" in result.stderr:
                logger.error(f"Errors during skillset generation: {result.stderr}")
                return False
            
            # Update progress
            # Parse output to get counts of generated files
            lines = result.stdout.split("\n")
            for line in lines:
                if "Generated" in line and "skillset files" in line:
                    try:
                        count = int(line.split("Generated")[1].split("skillset")[0].strip())
                        self.progress["skillset_generation"]["completed"] = count
                    except (ValueError, IndexError):
                        pass
            
            logger.info(f"Successfully generated {self.progress['skillset_generation']['completed']} skillset files")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to generate skillsets: {e}")
            logger.error(f"Output: {e.stdout}")
            logger.error(f"Error: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"An error occurred during skillset generation: {e}")
            return False
    
    def generate_benchmark_files(self, 
                               priority: str = "high", 
                               model_type: Optional[str] = None, 
                               model_filter: Optional[List[str]] = None) -> bool:
        """
        Generate benchmark files for models.
        
        Args:
            priority: Priority level (critical, high, medium, low, all)
            model_type: Optional model type filter (encoder, decoder, etc.)
            model_filter: Optional list of specific models to generate benchmarks for
        
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Generating benchmark files for {priority} priority models...")
        
        # Get models for the specified priority
        if model_filter:
            models = model_filter
        else:
            models = get_priority_models(priority)
        
        # Filter by model type if specified
        if model_type:
            filtered_models = []
            for model in models:
                if ModelArchitectureRegistry.get_model_type(model) == model_type:
                    filtered_models.append(model)
            models = filtered_models
        
        # Update progress tracking
        self.progress["benchmark_generation"]["total"] = len(models)
        self.progress["benchmark_generation"]["completed"] = 0
        self.progress["benchmark_generation"]["failed"] = 0
        
        # Prepare command for benchmark file generation
        benchmark_script = Path(__file__).resolve().parent / "generate_benchmark_files.py"
        
        # Check if script exists, use alternate path if not
        if not benchmark_script.exists():
            benchmark_script = Path(__file__).resolve().parent / "benchmarks" / "generate_skillset_benchmarks.py"
            if not benchmark_script.exists():
                logger.error(f"Benchmark generator script not found")
                return False
        
        # Build command
        cmd_args = [
            sys.executable,
            str(benchmark_script),
            f"--output-dir={self.benchmark_output_dir}"
        ]
        
        # Add models to command
        if model_filter:
            cmd_args.extend([f"--model={model}" for model in model_filter])
        else:
            cmd_args.append(f"--priority={priority}")
        
        # Add model type if specified
        if model_type:
            cmd_args.append(f"--model-type={model_type}")
        
        # Execute command
        logger.info(f"Running command: {' '.join(cmd_args)}")
        try:
            import subprocess
            result = subprocess.run(cmd_args, check=True, capture_output=True, text=True)
            
            # Check output for error messages
            if "ERROR" in result.stdout or "ERROR" in result.stderr:
                logger.error(f"Errors during benchmark file generation: {result.stderr}")
                return False
            
            # Update progress
            # Parse output to get counts of generated files
            lines = result.stdout.split("\n")
            for line in lines:
                if "Generated" in line and "benchmark files" in line:
                    try:
                        count = int(line.split("Generated")[1].split("benchmark")[0].strip())
                        self.progress["benchmark_generation"]["completed"] = count
                    except (ValueError, IndexError):
                        pass
            
            logger.info(f"Successfully generated {self.progress['benchmark_generation']['completed']} benchmark files")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to generate benchmark files: {e}")
            logger.error(f"Output: {e.stdout}")
            logger.error(f"Error: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"An error occurred during benchmark file generation: {e}")
            return False
    
    def run_benchmarks(self, 
                     priority: str = "high", 
                     hardware: Union[str, List[str]] = "cpu",
                     batch_sizes: Union[int, List[int]] = [1, 8],
                     model_type: Optional[str] = None,
                     model_filter: Optional[List[str]] = None,
                     precision: Union[str, List[str]] = "fp32",
                     progressive_mode: bool = False,
                     incremental: bool = True) -> bool:
        """
        Run benchmarks for models.
        
        Args:
            priority: Priority level (critical, high, medium, low, all)
            hardware: Hardware backend(s) to benchmark on
            batch_sizes: Batch size(s) to benchmark
            model_type: Optional model type filter (encoder, decoder, etc.)
            model_filter: Optional list of specific models to benchmark
            precision: Precision format(s) to benchmark
            progressive_mode: Whether to use progressive mode (CPU first, then GPU, etc.)
            incremental: Whether to only run benchmarks for missing or outdated results
        
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Running benchmarks for {priority} priority models...")
        
        # Ensure hardware and batch_sizes are lists
        if isinstance(hardware, str):
            hardware = [hardware]
        if isinstance(batch_sizes, int):
            batch_sizes = [batch_sizes]
        if isinstance(precision, str):
            precision = [precision]
        
        # Generate benchmark configurations
        configs = self._generate_benchmark_configs(
            priority=priority,
            hardware=hardware,
            batch_sizes=batch_sizes,
            model_type=model_type,
            model_filter=model_filter,
            precisions=precision,
            progressive_mode=progressive_mode,
            incremental=incremental
        )
        
        # Update progress tracking
        self.progress["benchmark_execution"]["total"] = len(configs)
        self.progress["benchmark_execution"]["completed"] = 0
        self.progress["benchmark_execution"]["failed"] = 0
        
        # Run benchmarks
        if self.parallel_jobs > 1:
            return self._run_benchmarks_parallel(configs)
        else:
            return self._run_benchmarks_sequential(configs)
    
    def _generate_benchmark_configs(self, 
                                   priority: str, 
                                   hardware: List[str],
                                   batch_sizes: List[int],
                                   model_type: Optional[str],
                                   model_filter: Optional[List[str]],
                                   precisions: List[str],
                                   progressive_mode: bool,
                                   incremental: bool) -> List[Dict[str, Any]]:
        """
        Generate benchmark configurations based on parameters.
        
        Args:
            priority: Priority level
            hardware: Hardware backends
            batch_sizes: Batch sizes
            model_type: Model type filter
            model_filter: Specific models to benchmark
            precisions: Precision formats
            progressive_mode: Whether to use progressive mode
            incremental: Whether to use incremental mode
        
        Returns:
            List of benchmark configurations
        """
        # Get models based on priority or filter
        if model_filter:
            models = model_filter
        else:
            models = get_priority_models(priority)
        
        # Filter by model type if specified
        if model_type:
            filtered_models = []
            for model in models:
                if ModelArchitectureRegistry.get_model_type(model) == model_type:
                    filtered_models.append(model)
            models = filtered_models
        
        # Generate configuration list
        configs = []
        
        # Get existing results from database if using incremental mode
        existing_results = {}
        update_threshold = 30  # days
        
        if incremental and self.db_manager.conn is not None:
            try:
                # Query existing results
                query = """
                SELECT 
                    m.model_name,
                    h.hardware_type,
                    pr.batch_size,
                    pr.precision,
                    MAX(pr.created_at) as last_updated
                FROM 
                    performance_results pr
                JOIN 
                    models m ON pr.model_id = m.model_id
                JOIN 
                    hardware_platforms h ON pr.hardware_id = h.hardware_id
                GROUP BY 
                    m.model_name, h.hardware_type, pr.batch_size, pr.precision
                """
                
                results = self.db_manager.conn.execute(query).fetchall()
                
                # Create lookup dictionary
                for row in results:
                    model_name, hw, batch_size, prec, created_at = row
                    
                    key = f"{model_name}_{hw}_{batch_size}_{prec}"
                    existing_results[key] = created_at
                
                logger.info(f"Found {len(existing_results)} existing benchmark results in database")
            except Exception as e:
                logger.error(f"Failed to query existing results: {e}")
                # Continue without incremental mode
                incremental = False
        
        # Process models
        for model_name in models:
            # Determine model type if not already filtered
            if not model_type:
                model_type_detected = ModelArchitectureRegistry.get_model_type(model_name)
            else:
                model_type_detected = model_type
                
            # For progressive mode, apply different hardware based on model type
            if progressive_mode:
                # Start with CPU for all models
                if "cpu" in hardware:
                    for batch_size in batch_sizes:
                        for prec in precisions:
                            config = {
                                "model_name": model_name,
                                "hardware": "cpu",
                                "batch_size": batch_size,
                                "precision": prec,
                                "test_type": "inference"
                            }
                            
                            # Skip if result exists and is recent in incremental mode
                            if incremental:
                                key = f"{model_name}_cpu_{batch_size}_{prec}"
                                if key in existing_results:
                                    last_updated = existing_results[key]
                                    days_since = (datetime.now() - last_updated).days
                                    
                                    if days_since < update_threshold:
                                        logger.debug(f"Skipping benchmark for {key} (updated {days_since} days ago)")
                                        continue
                            
                            configs.append(config)
                
                # Add CUDA for compute-intensive models
                if "cuda" in hardware:
                    # Add CUDA benchmark for decoder and vision models, which tend to be compute-intensive
                    if model_type_detected in ['decoder', 'vision', 'multimodal']:
                        for batch_size in batch_sizes:
                            for prec in precisions:
                                config = {
                                    "model_name": model_name,
                                    "hardware": "cuda",
                                    "batch_size": batch_size,
                                    "precision": prec,
                                    "test_type": "inference"
                                }
                                
                                # Skip if result exists and is recent in incremental mode
                                if incremental:
                                    key = f"{model_name}_cuda_{batch_size}_{prec}"
                                    if key in existing_results:
                                        last_updated = existing_results[key]
                                        days_since = (datetime.now() - last_updated).days
                                        
                                        if days_since < update_threshold:
                                            logger.debug(f"Skipping benchmark for {key} (updated {days_since} days ago)")
                                            continue
                                
                                configs.append(config)
                
                # Add other hardware for key models only
                remaining_hardware = [hw for hw in hardware if hw not in ["cpu", "cuda"]]
                
                # Only add for high-priority models (smaller subset)
                if model_name in get_priority_models("critical"):
                    for hw in remaining_hardware:
                        for batch_size in batch_sizes:
                            for prec in precisions:
                                config = {
                                    "model_name": model_name,
                                    "hardware": hw,
                                    "batch_size": batch_size,
                                    "precision": prec,
                                    "test_type": "inference"
                                }
                                
                                # Skip if result exists and is recent in incremental mode
                                if incremental:
                                    key = f"{model_name}_{hw}_{batch_size}_{prec}"
                                    if key in existing_results:
                                        last_updated = existing_results[key]
                                        days_since = (datetime.now() - last_updated).days
                                        
                                        if days_since < update_threshold:
                                            logger.debug(f"Skipping benchmark for {key} (updated {days_since} days ago)")
                                            continue
                                
                                configs.append(config)
            else:
                # Standard mode - add all hardware configurations
                for hw in hardware:
                    for batch_size in batch_sizes:
                        for prec in precisions:
                            config = {
                                "model_name": model_name,
                                "hardware": hw,
                                "batch_size": batch_size,
                                "precision": prec,
                                "test_type": "inference"
                            }
                            
                            # Skip if result exists and is recent in incremental mode
                            if incremental:
                                key = f"{model_name}_{hw}_{batch_size}_{prec}"
                                if key in existing_results:
                                    last_updated = existing_results[key]
                                    days_since = (datetime.now() - last_updated).days
                                    
                                    if days_since < update_threshold:
                                        logger.debug(f"Skipping benchmark for {key} (updated {days_since} days ago)")
                                        continue
                            
                            configs.append(config)
        
        logger.info(f"Generated {len(configs)} benchmark configurations")
        return configs
    
    def _run_benchmarks_sequential(self, configs: List[Dict[str, Any]]) -> bool:
        """
        Run benchmarks sequentially.
        
        Args:
            configs: List of benchmark configurations
        
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Running {len(configs)} benchmarks sequentially...")
        
        for i, config in enumerate(configs):
            logger.info(f"Running benchmark {i+1}/{len(configs)}: {config}")
            
            try:
                # Run the benchmark
                result = self.runner.execute_benchmark(config)
                
                # Store result in database
                if result and self.db_manager.conn is not None:
                    self.db_manager.store_performance_result(
                        model_name=config["model_name"],
                        hardware_type=config["hardware"],
                        batch_size=config["batch_size"],
                        precision=config["precision"],
                        throughput=result.get("throughput_items_per_second"),
                        latency_avg=result.get("average_latency_ms"),
                        memory_peak=result.get("memory_peak_mb")
                    )
                    
                # Update progress
                self.progress["benchmark_execution"]["completed"] += 1
                
                # Print progress
                if self.interactive:
                    sys.stdout.write(f"\rCompleted: {self.progress['benchmark_execution']['completed']}/{len(configs)} benchmarks...")
                    sys.stdout.flush()
            except Exception as e:
                logger.error(f"Failed to run benchmark for {config}: {e}")
                self.progress["benchmark_execution"]["failed"] += 1
        
        if self.interactive:
            sys.stdout.write("\n")
            sys.stdout.flush()
        
        success_rate = self.progress["benchmark_execution"]["completed"] / len(configs) if configs else 1.0
        logger.info(f"Completed {self.progress['benchmark_execution']['completed']}/{len(configs)} benchmarks ({success_rate:.2%})")
        
        return success_rate > 0.5  # Success if more than 50% completed
    
    def _run_benchmarks_parallel(self, configs: List[Dict[str, Any]]) -> bool:
        """
        Run benchmarks in parallel.
        
        Args:
            configs: List of benchmark configurations
        
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Running {len(configs)} benchmarks in parallel with {self.parallel_jobs} workers...")
        
        # Group configurations by hardware
        hw_groups = {}
        for config in configs:
            hw = config["hardware"]
            if hw not in hw_groups:
                hw_groups[hw] = []
            hw_groups[hw].append(config)
        
        # Process each hardware group sequentially to avoid resource conflicts
        for hw, hw_configs in hw_groups.items():
            logger.info(f"Processing {len(hw_configs)} configurations for hardware: {hw}")
            
            # Run benchmarks for this hardware in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel_jobs) as executor:
                future_to_config = {
                    executor.submit(self._run_single_benchmark, config): config
                    for config in hw_configs
                }
                
                completed = 0
                for future in concurrent.futures.as_completed(future_to_config):
                    config = future_to_config[future]
                    
                    try:
                        success = future.result()
                        if success:
                            self.progress["benchmark_execution"]["completed"] += 1
                        else:
                            self.progress["benchmark_execution"]["failed"] += 1
                    except Exception as e:
                        logger.error(f"Benchmark for {config} generated an exception: {e}")
                        self.progress["benchmark_execution"]["failed"] += 1
                    
                    completed += 1
                    if self.interactive:
                        sys.stdout.write(f"\rCompleted: {completed}/{len(hw_configs)} benchmarks for {hw}...")
                        sys.stdout.flush()
            
            if self.interactive:
                sys.stdout.write("\n")
                sys.stdout.flush()
        
        success_rate = self.progress["benchmark_execution"]["completed"] / len(configs) if configs else 1.0
        logger.info(f"Completed {self.progress['benchmark_execution']['completed']}/{len(configs)} benchmarks ({success_rate:.2%})")
        
        return success_rate > 0.5  # Success if more than 50% completed
    
    def _run_single_benchmark(self, config: Dict[str, Any]) -> bool:
        """
        Run a single benchmark.
        
        Args:
            config: Benchmark configuration
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Run the benchmark
            result = self.runner.execute_benchmark(config)
            
            # Store result in database
            if result and self.db_manager.conn is not None:
                self.db_manager.store_performance_result(
                    model_name=config["model_name"],
                    hardware_type=config["hardware"],
                    batch_size=config["batch_size"],
                    precision=config["precision"],
                    throughput=result.get("throughput_items_per_second"),
                    latency_avg=result.get("average_latency_ms"),
                    memory_peak=result.get("memory_peak_mb")
                )
            
            return True
        except Exception as e:
            logger.error(f"Failed to run benchmark for {config}: {e}")
            return False
    
    def generate_reports(self, 
                       formats: List[str] = ["html", "markdown"],
                       report_types: List[str] = ["summary", "hardware-comparison", "model-comparison", "compatibility-matrix"]) -> bool:
        """
        Generate reports from benchmark results.
        
        Args:
            formats: Output formats (html, markdown, etc.)
            report_types: Types of reports to generate
        
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Generating reports...")
        
        # Create directory for reports
        reports_dir = self.output_dir / "reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        # Update progress tracking
        self.progress["report_generation"]["total"] = len(formats) * len(report_types)
        self.progress["report_generation"]["completed"] = 0
        self.progress["report_generation"]["failed"] = 0
        
        # Generate reports
        for report_type in report_types:
            for fmt in formats:
                logger.info(f"Generating {report_type} report in {fmt} format...")
                
                try:
                    # Generate report
                    output_path = reports_dir / f"{report_type}.{fmt}"
                    
                    if self.db_manager.conn is None:
                        logger.error("Database connection not available. Cannot generate reports.")
                        self.progress["report_generation"]["failed"] += 1
                        continue
                    
                    # Use appropriate query for each report type
                    if report_type == "summary":
                        self._generate_summary_report(output_path, fmt)
                    elif report_type == "hardware-comparison":
                        self._generate_hardware_comparison_report(output_path, fmt)
                    elif report_type == "model-comparison":
                        self._generate_model_comparison_report(output_path, fmt)
                    elif report_type == "compatibility-matrix":
                        self._generate_compatibility_matrix(output_path, fmt)
                    else:
                        logger.warning(f"Unknown report type: {report_type}")
                        self.progress["report_generation"]["failed"] += 1
                        continue
                    
                    # Update progress
                    self.progress["report_generation"]["completed"] += 1
                except Exception as e:
                    logger.error(f"Failed to generate {report_type} report in {fmt} format: {e}")
                    self.progress["report_generation"]["failed"] += 1
        
        success_rate = self.progress["report_generation"]["completed"] / self.progress["report_generation"]["total"] if self.progress["report_generation"]["total"] else 1.0
        logger.info(f"Completed {self.progress['report_generation']['completed']}/{self.progress['report_generation']['total']} reports ({success_rate:.2%})")
        
        return success_rate > 0.5  # Success if more than 50% completed
    
    def _generate_summary_report(self, output_path: Path, fmt: str) -> bool:
        """
        Generate a summary report.
        
        Args:
            output_path: Output file path
            fmt: Output format
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Execute a summary query
            summary_query = """
            SELECT 
                COUNT(DISTINCT m.model_name) as model_count,
                COUNT(DISTINCT h.hardware_type) as hardware_count,
                COUNT(DISTINCT pr.result_id) as result_count,
                COUNT(DISTINCT CASE WHEN h.hardware_type = 'cpu' THEN m.model_name ELSE NULL END) as cpu_model_count,
                COUNT(DISTINCT CASE WHEN h.hardware_type = 'cuda' THEN m.model_name ELSE NULL END) as cuda_model_count
            FROM 
                performance_results pr
            JOIN 
                models m ON pr.model_id = m.model_id
            JOIN 
                hardware_platforms h ON pr.hardware_id = h.hardware_id
            """
            
            summary_results = self.db_manager.conn.execute(summary_query).fetchone()
            
            # Query for model types
            model_types_query = """
            SELECT 
                m.model_type,
                COUNT(DISTINCT m.model_name) as model_count
            FROM 
                models m
            JOIN
                performance_results pr ON m.model_id = pr.model_id
            GROUP BY 
                m.model_type
            ORDER BY 
                model_count DESC
            """
            
            model_types_results = self.db_manager.conn.execute(model_types_query).fetchall()
            
            # Query for hardware types
            hardware_query = """
            SELECT 
                h.hardware_type,
                COUNT(DISTINCT pr.result_id) as result_count
            FROM 
                performance_results pr
            JOIN 
                hardware_platforms h ON pr.hardware_id = h.hardware_id
            GROUP BY 
                h.hardware_type
            ORDER BY 
                result_count DESC
            """
            
            hardware_results = self.db_manager.conn.execute(hardware_query).fetchall()
            
            # Generate report based on format
            if fmt == "html":
                # HTML report
                html_report = f"""
                <html>
                <head>
                    <title>Benchmark Summary Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1 {{ color: #333; }}
                        h2 {{ color: #555; }}
                        table {{ border-collapse: collapse; margin: 15px 0; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    </style>
                </head>
                <body>
                    <h1>Benchmark Summary Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <h2>Overview</h2>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Total Models</td>
                            <td>{summary_results[0]}</td>
                        </tr>
                        <tr>
                            <td>Total Hardware Types</td>
                            <td>{summary_results[1]}</td>
                        </tr>
                        <tr>
                            <td>Total Benchmark Results</td>
                            <td>{summary_results[2]}</td>
                        </tr>
                        <tr>
                            <td>Models Benchmarked on CPU</td>
                            <td>{summary_results[3]}</td>
                        </tr>
                        <tr>
                            <td>Models Benchmarked on CUDA</td>
                            <td>{summary_results[4]}</td>
                        </tr>
                    </table>
                    
                    <h2>Models by Type</h2>
                    <table>
                        <tr>
                            <th>Model Type</th>
                            <th>Count</th>
                        </tr>
                """
                
                for model_type, count in model_types_results:
                    html_report += f"""
                        <tr>
                            <td>{model_type or 'Unknown'}</td>
                            <td>{count}</td>
                        </tr>
                    """
                
                html_report += """
                    </table>
                    
                    <h2>Results by Hardware</h2>
                    <table>
                        <tr>
                            <th>Hardware Type</th>
                            <th>Result Count</th>
                        </tr>
                """
                
                for hw_type, count in hardware_results:
                    html_report += f"""
                        <tr>
                            <td>{hw_type}</td>
                            <td>{count}</td>
                        </tr>
                    """
                
                html_report += """
                    </table>
                </body>
                </html>
                """
                
                with open(output_path, 'w') as f:
                    f.write(html_report)
            elif fmt == "markdown":
                # Markdown report
                md_report = f"""
                # Benchmark Summary Report
                
                Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                ## Overview
                
                | Metric | Value |
                |--------|-------|
                | Total Models | {summary_results[0]} |
                | Total Hardware Types | {summary_results[1]} |
                | Total Benchmark Results | {summary_results[2]} |
                | Models Benchmarked on CPU | {summary_results[3]} |
                | Models Benchmarked on CUDA | {summary_results[4]} |
                
                ## Models by Type
                
                | Model Type | Count |
                |------------|-------|
                """
                
                for model_type, count in model_types_results:
                    md_report += f"| {model_type or 'Unknown'} | {count} |\n"
                
                md_report += """
                
                ## Results by Hardware
                
                | Hardware Type | Result Count |
                |--------------|--------------|
                """
                
                for hw_type, count in hardware_results:
                    md_report += f"| {hw_type} | {count} |\n"
                
                with open(output_path, 'w') as f:
                    f.write(md_report)
            else:
                logger.warning(f"Unsupported format: {fmt}")
                return False
            
            logger.info(f"Generated summary report: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")
            return False
    
    def _generate_hardware_comparison_report(self, output_path: Path, fmt: str) -> bool:
        """
        Generate a hardware comparison report.
        
        Args:
            output_path: Output file path
            fmt: Output format
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Execute a hardware comparison query
            comparison_query = """
            WITH model_hw_stats AS (
                SELECT 
                    m.model_name,
                    m.model_type,
                    h.hardware_type,
                    AVG(pr.throughput_items_per_second) as avg_throughput,
                    AVG(pr.average_latency_ms) as avg_latency,
                    AVG(pr.memory_peak_mb) as avg_memory,
                    COUNT(*) as result_count
                FROM 
                    performance_results pr
                JOIN 
                    models m ON pr.model_id = m.model_id
                JOIN 
                    hardware_platforms h ON pr.hardware_id = h.hardware_id
                WHERE
                    pr.batch_size = 1
                GROUP BY 
                    m.model_name, m.model_type, h.hardware_type
            ),
            cpu_stats AS (
                SELECT 
                    model_name,
                    model_type,
                    avg_throughput as cpu_throughput,
                    avg_latency as cpu_latency,
                    avg_memory as cpu_memory
                FROM 
                    model_hw_stats
                WHERE 
                    hardware_type = 'cpu'
            )
            SELECT 
                hw.model_name,
                hw.model_type,
                hw.hardware_type,
                hw.avg_throughput,
                hw.avg_latency,
                hw.avg_memory,
                cpu.cpu_throughput,
                cpu.cpu_latency,
                cpu.cpu_memory,
                CASE WHEN cpu.cpu_throughput > 0 THEN hw.avg_throughput / cpu.cpu_throughput ELSE NULL END as speedup
            FROM 
                model_hw_stats hw
            LEFT JOIN 
                cpu_stats cpu ON hw.model_name = cpu.model_name
            ORDER BY 
                hw.model_name, hw.hardware_type
            """
            
            comparison_results = self.db_manager.conn.execute(comparison_query).fetchall()
            
            # Generate report based on format
            if fmt == "html":
                # HTML report
                html_report = f"""
                <html>
                <head>
                    <title>Hardware Comparison Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1 {{ color: #333; }}
                        h2 {{ color: #555; }}
                        table {{ border-collapse: collapse; margin: 15px 0; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    </style>
                </head>
                <body>
                    <h1>Hardware Comparison Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <h2>Performance Comparison</h2>
                    <table>
                        <tr>
                            <th>Model</th>
                            <th>Type</th>
                            <th>Hardware</th>
                            <th>Throughput</th>
                            <th>Latency (ms)</th>
                            <th>Memory (MB)</th>
                            <th>CPU Throughput</th>
                            <th>Speedup vs CPU</th>
                        </tr>
                """
                
                for row in comparison_results:
                    model_name, model_type, hardware_type, throughput, latency, memory, cpu_throughput, cpu_latency, cpu_memory, speedup = row
                    
                    # Skip CPU entries in this report (they're shown as reference)
                    if hardware_type == 'cpu':
                        continue
                    
                    html_report += f"""
                        <tr>
                            <td>{model_name}</td>
                            <td>{model_type or 'Unknown'}</td>
                            <td>{hardware_type}</td>
                            <td>{throughput:.2f if throughput else '-'}</td>
                            <td>{latency:.2f if latency else '-'}</td>
                            <td>{memory:.0f if memory else '-'}</td>
                            <td>{cpu_throughput:.2f if cpu_throughput else '-'}</td>
                            <td>{speedup:.2f if speedup else '-'}</td>
                        </tr>
                    """
                
                html_report += """
                    </table>
                </body>
                </html>
                """
                
                with open(output_path, 'w') as f:
                    f.write(html_report)
            elif fmt == "markdown":
                # Markdown report
                md_report = f"""
                # Hardware Comparison Report
                
                Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                ## Performance Comparison
                
                | Model | Type | Hardware | Throughput | Latency (ms) | Memory (MB) | CPU Throughput | Speedup vs CPU |
                |-------|------|----------|------------|--------------|-------------|----------------|---------------|
                """
                
                for row in comparison_results:
                    model_name, model_type, hardware_type, throughput, latency, memory, cpu_throughput, cpu_latency, cpu_memory, speedup = row
                    
                    # Skip CPU entries in this report (they're shown as reference)
                    if hardware_type == 'cpu':
                        continue
                    
                    md_report += f"| {model_name} | {model_type or 'Unknown'} | {hardware_type} | {throughput:.2f if throughput else '-'} | {latency:.2f if latency else '-'} | {memory:.0f if memory else '-'} | {cpu_throughput:.2f if cpu_throughput else '-'} | {speedup:.2f if speedup else '-'} |\n"
                
                with open(output_path, 'w') as f:
                    f.write(md_report)
            else:
                logger.warning(f"Unsupported format: {fmt}")
                return False
            
            logger.info(f"Generated hardware comparison report: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to generate hardware comparison report: {e}")
            return False
    
    def _generate_model_comparison_report(self, output_path: Path, fmt: str) -> bool:
        """
        Generate a model comparison report.
        
        Args:
            output_path: Output file path
            fmt: Output format
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Execute a model comparison query
            comparison_query = """
            SELECT 
                m.model_name,
                m.model_type,
                m.model_family,
                m.parameters_million,
                h.hardware_type,
                pr.batch_size,
                AVG(pr.throughput_items_per_second) as avg_throughput,
                AVG(pr.average_latency_ms) as avg_latency,
                AVG(pr.memory_peak_mb) as avg_memory
            FROM 
                performance_results pr
            JOIN 
                models m ON pr.model_id = m.model_id
            JOIN 
                hardware_platforms h ON pr.hardware_id = h.hardware_id
            WHERE
                pr.batch_size = 1 AND
                h.hardware_type = 'cpu'  -- Use CPU as baseline for comparisons
            GROUP BY 
                m.model_name, m.model_type, m.model_family, m.parameters_million, h.hardware_type, pr.batch_size
            ORDER BY 
                m.model_type, avg_throughput DESC
            """
            
            comparison_results = self.db_manager.conn.execute(comparison_query).fetchall()
            
            # Generate report based on format
            if fmt == "html":
                # HTML report
                html_report = f"""
                <html>
                <head>
                    <title>Model Comparison Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1 {{ color: #333; }}
                        h2 {{ color: #555; }}
                        table {{ border-collapse: collapse; margin: 15px 0; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    </style>
                </head>
                <body>
                    <h1>Model Comparison Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <h2>Performance Comparison (CPU)</h2>
                    <table>
                        <tr>
                            <th>Model</th>
                            <th>Type</th>
                            <th>Family</th>
                            <th>Parameters (M)</th>
                            <th>Throughput</th>
                            <th>Latency (ms)</th>
                            <th>Memory (MB)</th>
                        </tr>
                """
                
                for row in comparison_results:
                    model_name, model_type, model_family, parameters, hardware_type, batch_size, throughput, latency, memory = row
                    
                    html_report += f"""
                        <tr>
                            <td>{model_name}</td>
                            <td>{model_type or 'Unknown'}</td>
                            <td>{model_family or 'Unknown'}</td>
                            <td>{parameters:.1f if parameters else '-'}</td>
                            <td>{throughput:.2f if throughput else '-'}</td>
                            <td>{latency:.2f if latency else '-'}</td>
                            <td>{memory:.0f if memory else '-'}</td>
                        </tr>
                    """
                
                html_report += """
                    </table>
                </body>
                </html>
                """
                
                with open(output_path, 'w') as f:
                    f.write(html_report)
            elif fmt == "markdown":
                # Markdown report
                md_report = f"""
                # Model Comparison Report
                
                Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                ## Performance Comparison (CPU)
                
                | Model | Type | Family | Parameters (M) | Throughput | Latency (ms) | Memory (MB) |
                |-------|------|--------|---------------|------------|--------------|-------------|
                """
                
                for row in comparison_results:
                    model_name, model_type, model_family, parameters, hardware_type, batch_size, throughput, latency, memory = row
                    
                    md_report += f"| {model_name} | {model_type or 'Unknown'} | {model_family or 'Unknown'} | {parameters:.1f if parameters else '-'} | {throughput:.2f if throughput else '-'} | {latency:.2f if latency else '-'} | {memory:.0f if memory else '-'} |\n"
                
                with open(output_path, 'w') as f:
                    f.write(md_report)
            else:
                logger.warning(f"Unsupported format: {fmt}")
                return False
            
            logger.info(f"Generated model comparison report: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to generate model comparison report: {e}")
            return False
    
    def _generate_compatibility_matrix(self, output_path: Path, fmt: str) -> bool:
        """
        Generate a compatibility matrix report.
        
        Args:
            output_path: Output file path
            fmt: Output format
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get unique hardware types
            hardware_query = """
            SELECT DISTINCT hardware_type 
            FROM hardware_platforms
            ORDER BY hardware_type
            """
            
            hardware_types = [row[0] for row in self.db_manager.conn.execute(hardware_query).fetchall()]
            
            # Get unique model names with their types
            model_query = """
            SELECT DISTINCT m.model_name, m.model_type 
            FROM models m
            JOIN performance_results pr ON m.model_id = pr.model_id
            ORDER BY m.model_type, m.model_name
            """
            
            model_info = [(row[0], row[1]) for row in self.db_manager.conn.execute(model_query).fetchall()]
            
            # Get compatibility data
            compat_query = """
            WITH latest_results AS (
                SELECT 
                    model_id,
                    hardware_id,
                    MAX(created_at) as latest_date
                FROM 
                    performance_results
                GROUP BY 
                    model_id, hardware_id
            )
            SELECT 
                m.model_name,
                h.hardware_type,
                CASE WHEN pr.result_id IS NOT NULL THEN 1 ELSE 0 END as is_compatible
            FROM 
                models m
            CROSS JOIN 
                hardware_platforms h
            LEFT JOIN 
                latest_results lr ON m.model_id = lr.model_id AND h.hardware_id = lr.hardware_id
            LEFT JOIN 
                performance_results pr ON lr.model_id = pr.model_id AND lr.hardware_id = pr.hardware_id AND lr.latest_date = pr.created_at
            WHERE 
                m.model_id IN (SELECT DISTINCT model_id FROM performance_results)
            ORDER BY 
                m.model_name, h.hardware_type
            """
            
            compat_results = self.db_manager.conn.execute(compat_query).fetchall()
            
            # Build compatibility matrix
            compat_matrix = {}
            for model_name, hardware_type, is_compatible in compat_results:
                if model_name not in compat_matrix:
                    compat_matrix[model_name] = {}
                
                compat_matrix[model_name][hardware_type] = is_compatible
            
            # Generate report based on format
            if fmt == "html":
                # HTML report
                html_report = f"""
                <html>
                <head>
                    <title>Compatibility Matrix</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1 {{ color: #333; }}
                        h2 {{ color: #555; }}
                        table {{ border-collapse: collapse; margin: 15px 0; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                        th {{ background-color: #f2f2f2; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                        .yes {{ color: green; font-weight: bold; }}
                        .no {{ color: red; }}
                        .model-type {{ font-weight: bold; background-color: #e6e6e6; text-align: left; }}
                    </style>
                </head>
                <body>
                    <h1>Model-Hardware Compatibility Matrix</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <table>
                        <tr>
                            <th>Model</th>
                """
                
                # Add hardware type headers
                for hw_type in hardware_types:
                    html_report += f"<th>{hw_type}</th>"
                
                html_report += "</tr>"
                
                # Group models by type
                current_type = None
                for model_name, model_type in model_info:
                    if model_name not in compat_matrix:
                        continue
                    
                    # Add type header if type changes
                    if model_type != current_type:
                        current_type = model_type
                        html_report += f"""
                        <tr>
                            <td colspan="{len(hardware_types) + 1}" class="model-type">{model_type or 'Unknown'}</td>
                        </tr>
                        """
                    
                    html_report += f"""
                    <tr>
                        <td>{model_name}</td>
                    """
                    
                    for hw_type in hardware_types:
                        is_compatible = compat_matrix[model_name].get(hw_type, 0)
                        symbol = '✓' if is_compatible else '✗'
                        css_class = 'yes' if is_compatible else 'no'
                        
                        html_report += f"""
                        <td class="{css_class}">{symbol}</td>
                        """
                    
                    html_report += "</tr>"
                
                html_report += """
                    </table>
                </body>
                </html>
                """
                
                with open(output_path, 'w') as f:
                    f.write(html_report)
            elif fmt == "markdown":
                # Markdown report
                md_report = f"""
                # Model-Hardware Compatibility Matrix
                
                Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                | Model | {' | '.join(hardware_types)} |
                |-------|{'-|' * len(hardware_types)}
                """
                
                # Group models by type
                current_type = None
                for model_name, model_type in model_info:
                    if model_name not in compat_matrix:
                        continue
                    
                    # Add type header if type changes
                    if model_type != current_type:
                        current_type = model_type
                        md_report += f"| **{model_type or 'Unknown'}** |{' | ' * len(hardware_types)}\n"
                    
                    md_report += f"| {model_name} | "
                    
                    for hw_type in hardware_types:
                        is_compatible = compat_matrix[model_name].get(hw_type, 0)
                        symbol = '✓' if is_compatible else '✗'
                        
                        md_report += f"{symbol} | "
                    
                    md_report = md_report.rstrip(" |") + "\n"
                
                with open(output_path, 'w') as f:
                    f.write(md_report)
            else:
                logger.warning(f"Unsupported format: {fmt}")
                return False
            
            logger.info(f"Generated compatibility matrix: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to generate compatibility matrix: {e}")
            return False
    
    def run_complete_pipeline(self, 
                           priority: str = "high", 
                           hardware: Union[str, List[str]] = "cpu",
                           batch_sizes: Union[int, List[int]] = [1, 8],
                           model_type: Optional[str] = None,
                           model_filter: Optional[List[str]] = None,
                           precision: Union[str, List[str]] = "fp32",
                           progressive_mode: bool = True,
                           incremental: bool = True,
                           report_formats: List[str] = ["html", "markdown"]) -> bool:
        """
        Run the complete benchmark pipeline.
        
        Args:
            priority: Priority level (critical, high, medium, low, all)
            hardware: Hardware backend(s) to benchmark on
            batch_sizes: Batch size(s) to benchmark
            model_type: Optional model type filter (encoder, decoder, etc.)
            model_filter: Optional list of specific models to benchmark
            precision: Precision format(s) to benchmark
            progressive_mode: Whether to use progressive mode (CPU first, then GPU, etc.)
            incremental: Whether to only run benchmarks for missing or outdated results
            report_formats: Output formats for reports
        
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Starting complete benchmark pipeline...")
        
        # Initialize progress tracking
        self.start_time = datetime.now()
        
        # Step 1: Generate model skillsets
        if not self.generate_model_skillsets(priority, model_filter):
            logger.error("Failed to generate model skillsets")
            return False
        
        # Step 2: Generate benchmark files
        if not self.generate_benchmark_files(priority, model_type, model_filter):
            logger.error("Failed to generate benchmark files")
            return False
        
        # Step 3: Run benchmarks
        if not self.run_benchmarks(priority, hardware, batch_sizes, model_type, model_filter, precision, progressive_mode, incremental):
            logger.error("Failed to run benchmarks")
            return False
        
        # Step 4: Generate reports
        if not self.generate_reports(report_formats):
            logger.error("Failed to generate reports")
            return False
        
        # Calculate elapsed time
        elapsed = datetime.now() - self.start_time
        
        # Generate final summary
        logger.info(f"Pipeline completed in {elapsed}")
        logger.info(f"Skillset generation: {self.progress['skillset_generation']['completed']} completed, {self.progress['skillset_generation']['failed']} failed")
        logger.info(f"Benchmark generation: {self.progress['benchmark_generation']['completed']} completed, {self.progress['benchmark_generation']['failed']} failed")
        logger.info(f"Benchmark execution: {self.progress['benchmark_execution']['completed']} completed, {self.progress['benchmark_execution']['failed']} failed")
        logger.info(f"Report generation: {self.progress['report_generation']['completed']} completed, {self.progress['report_generation']['failed']} failed")
        
        return True
    
    def close(self):
        """Close connections and clean up resources."""
        if self.db_manager is not None:
            self.db_manager.close()

def main():
    """Main entry point for benchmark pipeline."""
    parser = argparse.ArgumentParser(description="Complete Benchmark Pipeline for HuggingFace Models")
    
    # Database configuration
    parser.add_argument("--db-path", type=str, help="Path to DuckDB database (default: $BENCHMARK_DB_PATH or ./benchmark_db.duckdb)")
    parser.add_argument("--output-dir", type=str, help="Root directory for output files (default: ./benchmark_output)")
    
    # Model selection
    parser.add_argument("--priority", type=str, choices=["critical", "high", "medium", "low", "all"], default="high", help="Priority level for models to benchmark")
    parser.add_argument("--model", action="append", dest="models", help="Specific model to benchmark (can be used multiple times)")
    parser.add_argument("--model-type", type=str, choices=["encoder", "decoder", "encoder-decoder", "vision", "audio", "multimodal", "diffusion"], help="Filter by model type")
    
    # Hardware selection
    parser.add_argument("--hardware", type=str, default="cpu", help="Hardware backends to benchmark on (comma-separated)")
    parser.add_argument("--progressive-mode", action="store_true", help="Use progressive mode (CPU first, then GPU, etc.)")
    
    # Benchmark configuration
    parser.add_argument("--batch-sizes", type=str, default="1,8", help="Batch sizes to benchmark (comma-separated)")
    parser.add_argument("--precision", type=str, default="fp32", help="Precision formats to benchmark (comma-separated)")
    parser.add_argument("--incremental", action="store_true", help="Only run benchmarks for missing or outdated results")
    
    # Execution configuration
    parser.add_argument("--parallel-jobs", type=int, default=4, help="Number of parallel jobs for benchmark execution")
    parser.add_argument("--no-interactive", action="store_true", help="Disable interactive progress reporting")
    
    # Pipeline control
    parser.add_argument("--skip-skillset-generation", action="store_true", help="Skip model skillset generation")
    parser.add_argument("--skip-benchmark-generation", action="store_true", help="Skip benchmark file generation")
    parser.add_argument("--skip-benchmark-execution", action="store_true", help="Skip benchmark execution")
    parser.add_argument("--skip-report-generation", action="store_true", help="Skip report generation")
    parser.add_argument("--report-formats", type=str, default="html,markdown", help="Report output formats (comma-separated)")
    
    # CI mode
    parser.add_argument("--ci-mode", action="store_true", help="Run in CI mode (non-interactive, fixed output paths)")
    
    args = parser.parse_args()
    
    # Parse comma-separated values
    hardware_list = args.hardware.split(",")
    batch_sizes_list = [int(bs) for bs in args.batch_sizes.split(",")]
    precision_list = args.precision.split(",")
    report_formats_list = args.report_formats.split(",")
    
    # Adjust for CI mode
    if args.ci_mode:
        args.no_interactive = True
        
        # Set fixed output paths in CI mode
        if not args.output_dir:
            args.output_dir = "./ci_benchmark_output"
        if not args.db_path:
            args.db_path = "./ci_benchmark_db.duckdb"
    
    # Initialize pipeline
    pipeline = BenchmarkPipeline(
        db_path=args.db_path,
        output_dir=args.output_dir,
        parallel_jobs=args.parallel_jobs,
        interactive=not args.no_interactive
    )
    
    try:
        # Step 1: Generate model skillsets
        if not args.skip_skillset_generation:
            if not pipeline.generate_model_skillsets(priority=args.priority, model_filter=args.models):
                logger.error("Failed to generate model skillsets")
                return 1
        
        # Step 2: Generate benchmark files
        if not args.skip_benchmark_generation:
            if not pipeline.generate_benchmark_files(priority=args.priority, model_type=args.model_type, model_filter=args.models):
                logger.error("Failed to generate benchmark files")
                return 1
        
        # Step 3: Run benchmarks
        if not args.skip_benchmark_execution:
            if not pipeline.run_benchmarks(
                priority=args.priority,
                hardware=hardware_list,
                batch_sizes=batch_sizes_list,
                model_type=args.model_type,
                model_filter=args.models,
                precision=precision_list,
                progressive_mode=args.progressive_mode,
                incremental=args.incremental
            ):
                logger.error("Failed to run benchmarks")
                return 1
        
        # Step 4: Generate reports
        if not args.skip_report_generation:
            if not pipeline.generate_reports(formats=report_formats_list):
                logger.error("Failed to generate reports")
                return 1
        
        logger.info("Pipeline completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1
    finally:
        pipeline.close()

if __name__ == "__main__":
    sys.exit(main())