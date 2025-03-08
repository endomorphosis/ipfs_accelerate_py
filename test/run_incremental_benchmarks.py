#!/usr/bin/env python
"""
Incremental Benchmark Runner

This script identifies and runs only benchmarks that haven't been completed or are outdated.
Instead of using a weekly schedule, it dynamically determines which model-hardware combinations
need to be benchmarked based on what's already in the database.

Features:
- Queries the database to find missing or outdated benchmark data
- Prioritizes benchmarks based on importance and staleness
- Supports specific hardware platforms, model types, or batch sizes
- Tracks progress and can be resumed if interrupted
- Handles simulation for unavailable hardware platforms

Usage:
    python run_incremental_benchmarks.py --all-platforms
    python run_incremental_benchmarks.py --hardware cuda --models bert,t5,vit
    python run_incremental_benchmarks.py --priority-only --max-benchmarks 10
    python run_incremental_benchmarks.py --refresh-older-than 30 --models whisper
"""

import os
import sys
import json
import argparse
import logging
import datetime
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("incremental_benchmarks.log")
    ]
)
logger = logging.getLogger("incremental_benchmarks")

# Try to import the benchmark database API
try:
    from benchmark_db_api import BenchmarkDBAPI
    HAS_DB_API = True
except ImportError:
    logger.warning("BenchmarkDBAPI not available. Please install it first.")
    HAS_DB_API = False

# Constants
DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16]
HARDWARE_PLATFORMS = ["cpu", "cuda", "rocm", "mps", "openvino", "qnn", "webnn", "webgpu"]
MODEL_FAMILIES = {
    "text": ["bert", "t5", "gpt2", "llama"],
    "vision": ["vit", "clip", "detr"],
    "audio": ["whisper", "wav2vec2", "clap"],
    "multimodal": ["clip", "llava", "llava_next", "xclip"]
}

# Priority model-hardware combinations
PRIORITY_COMBINATIONS = [
    # Key model-hardware pairs
    ("bert", "cpu"), ("bert", "cuda"), ("bert", "webgpu"), ("bert", "openvino"),
    ("t5", "cpu"), ("t5", "cuda"), ("t5", "openvino"),
    ("vit", "cpu"), ("vit", "cuda"), ("vit", "webgpu"), ("vit", "openvino"),
    ("whisper", "cpu"), ("whisper", "cuda"), ("whisper", "webgpu"),
    ("clip", "cpu"), ("clip", "cuda"), ("clip", "webgpu"), ("clip", "openvino"),
    ("wav2vec2", "cpu"), ("wav2vec2", "cuda"), ("wav2vec2", "openvino")
]

class IncrementalBenchmarkRunner:
    """
    Runner for identifying and executing benchmarks incrementally,
    focusing on what hasn't been done or needs refresh.
    """
    
    def __init__(self, args):
        """Initialize the runner with parsed arguments."""
        self.args = args
        
        # Get database path
        self.db_path = args.db_path
        if self.db_path is None:
            self.db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
            logger.info(f"Using database path from environment: {self.db_path}")
        
        # Create DB API client
        if HAS_DB_API:
            self.db_api = BenchmarkDBAPI(db_path=self.db_path)
        else:
            logger.error("BenchmarkDBAPI is required for this script. Please install it first.")
            sys.exit(1)
        
        # Parse hardware platforms
        self.hardware_platforms = []
        if args.all_platforms:
            self.hardware_platforms = HARDWARE_PLATFORMS
        elif args.hardware:
            self.hardware_platforms = [hw.strip() for hw in args.hardware.split(',')]
        else:
            # Default to CPU if nothing specified
            self.hardware_platforms = ["cpu"]
        
        # Parse models
        self.models = []
        if args.all_models:
            # Get all models from the database
            try:
                model_df = self.db_api.get_model_list()
                self.models = model_df['model_name'].tolist()
            except Exception as e:
                logger.error(f"Error getting model list from database: {e}")
                self.models = self._get_default_models()
        elif args.models:
            self.models = [m.strip() for m in args.models.split(',')]
        elif args.model_families:
            families = [f.strip() for f in args.model_families.split(',')]
            self.models = []
            for family in families:
                if family in MODEL_FAMILIES:
                    self.models.extend(MODEL_FAMILIES[family])
        else:
            # Use priority models if nothing specified
            self.models = list(set([m for m, _ in PRIORITY_COMBINATIONS]))
        
        # Parse batch sizes
        if args.batch_sizes:
            self.batch_sizes = [int(b) for b in args.batch_sizes.split(',')]
        else:
            self.batch_sizes = DEFAULT_BATCH_SIZES
        
        # Other parameters
        self.refresh_days = args.refresh_older_than
        self.priority_only = args.priority_only
        self.max_benchmarks = args.max_benchmarks
        self.missing_only = args.missing_only
        self.simulate = args.simulate
        self.force = args.force
        self.dry_run = args.dry_run
        
        # Progress tracking
        self.progress_file = args.progress_file or "benchmark_progress.json"
        self.completed_benchmarks = self._load_progress()
        self.benchmark_count = 0
        
        logger.info(f"Initialized IncrementalBenchmarkRunner")
        logger.info(f"  - Database: {self.db_path}")
        logger.info(f"  - Hardware platforms: {self.hardware_platforms}")
        logger.info(f"  - Models: {self.models}")
        logger.info(f"  - Batch sizes: {self.batch_sizes}")
        logger.info(f"  - Refresh if older than {self.refresh_days} days")
        logger.info(f"  - Priority only: {self.priority_only}")
        logger.info(f"  - Max benchmarks: {self.max_benchmarks}")
    
    def _get_default_models(self) -> List[str]:
        """Return a default list of models to benchmark."""
        return [
            "bert-base-uncased", "t5-small", "gpt2", "facebook/opt-125m",
            "google/vit-base-patch16-224", "openai/clip-vit-base-patch32",
            "facebook/detr-resnet-50", "openai/whisper-tiny",
            "facebook/wav2vec2-base", "laion/clap-htsat-unfused"
        ]
    
    def _load_progress(self) -> Set[str]:
        """Load previously completed benchmarks from the progress file."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    completed = set(data.get('completed_benchmarks', []))
                    logger.info(f"Loaded {len(completed)} completed benchmarks from progress file")
                    return completed
            except Exception as e:
                logger.error(f"Error loading progress file: {e}")
        
        logger.info("No previous progress file found or error loading it. Starting fresh.")
        return set()
    
    def _save_progress(self):
        """Save completed benchmarks to the progress file."""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump({
                    'completed_benchmarks': list(self.completed_benchmarks),
                    'last_updated': datetime.datetime.now().isoformat()
                }, f, indent=2)
            logger.info(f"Saved progress with {len(self.completed_benchmarks)} completed benchmarks")
        except Exception as e:
            logger.error(f"Error saving progress file: {e}")
    
    def _get_benchmark_key(self, model: str, hardware: str, batch_size: int) -> str:
        """Generate a unique key for a benchmark combination."""
        return f"{model}:{hardware}:{batch_size}"
    
    def get_incomplete_benchmarks(self) -> List[Dict[str, Any]]:
        """
        Query the database to find model-hardware-batch combinations that:
        1. Don't exist in the database
        2. Are older than the refresh threshold
        3. Are priority combinations that need testing
        
        Returns a list of benchmark configurations sorted by priority.
        """
        incomplete_benchmarks = []
        
        # Get current timestamp for age calculations
        now = datetime.datetime.now()
        # Create a naive datetime for comparison
        refresh_date = (now - datetime.timedelta(days=self.refresh_days)).replace(tzinfo=None)
        
        # Query database for existing benchmarks
        try:
            # Prepare placeholders for parameters
            model_placeholders = ','.join(['?'] * len(self.models))
            hardware_placeholders = ','.join(['?'] * len(self.hardware_platforms))
            
            # Get performance metrics for all combinations
            sql = f"""
            WITH latest_results AS (
                SELECT 
                    m.model_name,
                    hp.hardware_type,
                    pr.batch_size,
                    COALESCE(pr.test_timestamp, CURRENT_TIMESTAMP) as created_date,
                    ROW_NUMBER() OVER(PARTITION BY m.model_id, hp.hardware_id, pr.batch_size 
                                    ORDER BY COALESCE(pr.test_timestamp, CURRENT_TIMESTAMP) DESC) as rn
                FROM 
                    performance_results pr
                JOIN 
                    models m ON pr.model_id = m.model_id
                JOIN 
                    hardware_platforms hp ON pr.hardware_id = hp.hardware_id
                WHERE 
                    m.model_name IN ({model_placeholders})
                AND
                    hp.hardware_type IN ({hardware_placeholders})
            )
            SELECT * FROM latest_results WHERE rn = 1
            """
            
            # Prepare parameters
            params = self.models + self.hardware_platforms
            
            # Execute query
            existing_df = self.db_api.query(sql, params)
            logger.info(f"Found {len(existing_df)} existing benchmark results in database")
            
            # Convert to dictionary for easier lookup
            existing_benchmarks = {}
            for _, row in existing_df.iterrows():
                key = self._get_benchmark_key(
                    row['model_name'], 
                    row['hardware_type'], 
                    row['batch_size']
                )
                existing_benchmarks[key] = row
            
            # Check each model-hardware-batch combination
            for model in self.models:
                for hardware in self.hardware_platforms:
                    # Skip if not a priority combination and priority_only is set
                    if self.priority_only and (model, hardware) not in PRIORITY_COMBINATIONS:
                        continue
                    
                    for batch_size in self.batch_sizes:
                        benchmark_key = self._get_benchmark_key(model, hardware, batch_size)
                        
                        # Skip if already completed in this run
                        if benchmark_key in self.completed_benchmarks and not self.force:
                            continue
                        
                        # Check if this combination exists and is recent enough
                        needs_benchmark = False
                        priority = 2  # Default priority (lower is higher)
                        
                        if benchmark_key not in existing_benchmarks:
                            # Doesn't exist in database
                            needs_benchmark = True
                            priority = 1  # Higher priority for missing benchmarks
                        else:
                            # Exists but check if it's old
                            benchmark_date = existing_benchmarks[benchmark_key]['created_date']
                            # Handle timezone-aware vs naive datetime comparison
                            if hasattr(benchmark_date, 'tzinfo') and benchmark_date.tzinfo is not None:
                                benchmark_date = benchmark_date.replace(tzinfo=None)
                            
                            # If missing_only flag is set, skip existing benchmarks
                            if self.missing_only:
                                needs_benchmark = False
                            # Otherwise, check if it's older than the refresh threshold
                            elif benchmark_date < refresh_date:
                                needs_benchmark = True
                                # Age-based priority - older benchmarks get higher priority
                                days_old = max(1, (now - benchmark_date).days)
                                priority = max(3, min(10, 3 + int(days_old / 30)))
                        
                        # Adjust priority for priority combinations
                        if (model, hardware) in PRIORITY_COMBINATIONS:
                            priority = max(0, priority - 1)  # Boost priority
                        
                        if needs_benchmark or self.force:
                            incomplete_benchmarks.append({
                                'model': model,
                                'hardware': hardware,
                                'batch_size': batch_size,
                                'priority': priority,
                                'key': benchmark_key
                            })
            
            # Sort by priority (lower is higher)
            incomplete_benchmarks.sort(key=lambda x: x['priority'])
            
            # Limit number of benchmarks if specified
            if self.max_benchmarks > 0 and len(incomplete_benchmarks) > self.max_benchmarks:
                incomplete_benchmarks = incomplete_benchmarks[:self.max_benchmarks]
            
            logger.info(f"Found {len(incomplete_benchmarks)} benchmarks to run")
            return incomplete_benchmarks
            
        except Exception as e:
            logger.error(f"Error querying database for incomplete benchmarks: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def run_benchmark(self, benchmark: Dict[str, Any]) -> bool:
        """
        Run a single benchmark for the given model-hardware-batch combination.
        
        Args:
            benchmark: Dictionary with benchmark configuration
            
        Returns:
            bool: Whether the benchmark completed successfully
        """
        model = benchmark['model']
        hardware = benchmark['hardware']
        batch_size = benchmark['batch_size']
        benchmark_key = benchmark['key']
        
        logger.info(f"Running benchmark: {model} on {hardware} with batch size {batch_size}")
        
        if self.dry_run:
            logger.info(f"DRY RUN: Would run benchmark for {benchmark_key}")
            return True
        
        # Build the command to run the benchmark
        cmd = [
            sys.executable,  # Python executable
            "run_benchmark_with_db.py",
            "--model", model,
            "--hardware", hardware,
            "--batch-sizes", str(batch_size),
            "--db", self.db_path
        ]
        
        # Add simulation flag if needed
        if self.simulate:
            cmd.append("--simulate")
        
        # Run the benchmark
        try:
            logger.info(f"Executing command: {' '.join(cmd)}")
            
            # Run the benchmark process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Capture output
            stdout, stderr = process.communicate()
            
            # Log output
            if stdout:
                logger.info(f"Benchmark stdout:\n{stdout}")
            if stderr:
                logger.warning(f"Benchmark stderr:\n{stderr}")
            
            # Check if benchmark completed successfully
            if process.returncode == 0:
                logger.info(f"Benchmark completed successfully: {benchmark_key}")
                return True
            else:
                # Special case for OpenVINO hardware
                if hardware == "openvino" and stderr and ("device not found" in stderr or "no such device" in stderr.lower()):
                    logger.warning(f"OpenVINO device issue for {benchmark_key}. Consider using benchmark_openvino.py with a specific device.")
                    logger.info(f"Try: python benchmark_openvino.py --model {model} --device CPU --batch-sizes {batch_size}")
                else:
                    logger.error(f"Benchmark failed with return code {process.returncode}: {benchmark_key}")
                return False
            
        except Exception as e:
            logger.error(f"Error running benchmark {benchmark_key}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_report(self) -> None:
        """Generate a report of completed benchmarks and remaining work."""
        try:
            # Get report timestamp
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Create report file
            report_path = f"benchmark_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            
            with open(report_path, 'w') as f:
                f.write(f"# Incremental Benchmark Report\n\n")
                f.write(f"Generated: {timestamp}\n\n")
                
                f.write(f"## Overview\n\n")
                f.write(f"- Database: `{self.db_path}`\n")
                f.write(f"- Completed benchmarks: {self.benchmark_count}\n")
                f.write(f"- Total completed since tracking: {len(self.completed_benchmarks)}\n\n")
                
                f.write(f"## Tested Models\n\n")
                f.write(f"| Model | Hardware Platforms | Batch Sizes |\n")
                f.write(f"|-------|-------------------|-------------|\n")
                
                # Group completed benchmarks by model
                model_data = {}
                for key in self.completed_benchmarks:
                    parts = key.split(':')
                    if len(parts) == 3:
                        model, hardware, batch_size = parts
                        if model not in model_data:
                            model_data[model] = {'hardware': set(), 'batch_sizes': set()}
                        model_data[model]['hardware'].add(hardware)
                        model_data[model]['batch_sizes'].add(batch_size)
                
                # Write model data to table
                for model, data in model_data.items():
                    hardware_str = ", ".join(sorted(data['hardware']))
                    batch_sizes_str = ", ".join(sorted(data['batch_sizes']))
                    f.write(f"| {model} | {hardware_str} | {batch_sizes_str} |\n")
                
                f.write(f"\n## Database Coverage\n\n")
                
                # Query the database for total coverage
                try:
                    coverage_sql = """
                    SELECT 
                        COUNT(DISTINCT m.model_name) as model_count,
                        COUNT(DISTINCT hp.hardware_type) as hardware_count,
                        COUNT(DISTINCT pr.batch_size) as batch_size_count,
                        COUNT(*) as total_benchmarks
                    FROM 
                        performance_results pr
                    JOIN 
                        models m ON pr.model_id = m.model_id
                    JOIN 
                        hardware_platforms hp ON pr.hardware_id = hp.hardware_id
                    """
                    coverage_df = self.db_api.query(coverage_sql)
                    
                    if len(coverage_df) > 0:
                        row = coverage_df.iloc[0]
                        f.write(f"- Total models in database: {row['model_count']}\n")
                        f.write(f"- Total hardware platforms: {row['hardware_count']}\n")
                        f.write(f"- Total batch sizes tested: {row['batch_size_count']}\n")
                        f.write(f"- Total benchmark results: {row['total_benchmarks']}\n\n")
                    
                    # Add hardware coverage
                    hardware_sql = """
                    SELECT 
                        hp.hardware_type,
                        COUNT(DISTINCT m.model_name) as model_count,
                        COUNT(*) as benchmark_count,
                        MAX(pr.test_timestamp) as last_updated
                    FROM 
                        performance_results pr
                    JOIN 
                        models m ON pr.model_id = m.model_id
                    JOIN 
                        hardware_platforms hp ON pr.hardware_id = hp.hardware_id
                    GROUP BY
                        hp.hardware_type
                    ORDER BY
                        benchmark_count DESC
                    """
                    hardware_df = self.db_api.query(hardware_sql)
                    
                    if len(hardware_df) > 0:
                        f.write(f"## Hardware Coverage\n\n")
                        f.write(f"| Hardware | Models Tested | Benchmark Count | Last Updated |\n")
                        f.write(f"|----------|--------------|-----------------|-------------|\n")
                        
                        for _, row in hardware_df.iterrows():
                            f.write(f"| {row['hardware_type']} | {row['model_count']} | {row['benchmark_count']} | {row['last_updated']} |\n")
                    
                except Exception as e:
                    f.write(f"\nError generating coverage statistics: {e}\n")
            
            logger.info(f"Generated benchmark report: {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
    
    def run(self) -> None:
        """Main execution method for the benchmark runner."""
        # Get incomplete benchmarks
        benchmarks = self.get_incomplete_benchmarks()
        
        if not benchmarks:
            logger.info("No incomplete benchmarks found. All done!")
            return
        
        logger.info(f"Found {len(benchmarks)} benchmarks to run")
        
        try:
            # Run each benchmark
            for i, benchmark in enumerate(benchmarks):
                benchmark_key = benchmark['key']
                
                logger.info(f"\nRunning benchmark {i+1} of {len(benchmarks)}: {benchmark_key}")
                logger.info(f"  - Model: {benchmark['model']}")
                logger.info(f"  - Hardware: {benchmark['hardware']}")
                logger.info(f"  - Batch size: {benchmark['batch_size']}")
                logger.info(f"  - Priority: {benchmark['priority']}")
                
                # Run the benchmark
                success = self.run_benchmark(benchmark)
                
                if success:
                    # Add to completed benchmarks
                    self.completed_benchmarks.add(benchmark_key)
                    self.benchmark_count += 1
                    
                    # Save progress after each successful benchmark
                    self._save_progress()
                else:
                    logger.warning(f"Skipping failed benchmark: {benchmark_key}")
                
                # Add a short delay between benchmarks
                if i < len(benchmarks) - 1:
                    time.sleep(1)
            
            logger.info(f"\nCompleted {self.benchmark_count} benchmarks successfully")
            
            # Generate report
            self.generate_report()
            
        except KeyboardInterrupt:
            logger.info("\nBenchmark run interrupted by user")
            logger.info(f"Completed {self.benchmark_count} benchmarks before interruption")
            # Save progress on interrupt
            self._save_progress()
        except Exception as e:
            logger.error(f"Error running benchmarks: {e}")
            import traceback
            traceback.print_exc()
            # Save progress on error
            self._save_progress()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run incremental benchmarks based on what's missing or outdated")
    
    # Database options
    parser.add_argument("--db-path", type=str,
                      help="Path to the benchmark database (defaults to BENCHMARK_DB_PATH env var)")
    
    # Selection options
    parser.add_argument("--all-platforms", action="store_true",
                      help="Benchmark all hardware platforms")
    parser.add_argument("--hardware", type=str,
                      help="Comma-separated list of hardware platforms to benchmark")
    parser.add_argument("--all-models", action="store_true",
                      help="Benchmark all models in the database")
    parser.add_argument("--models", type=str,
                      help="Comma-separated list of models to benchmark")
    parser.add_argument("--model-families", type=str,
                      help="Comma-separated list of model families to benchmark (text, vision, audio, multimodal)")
    parser.add_argument("--batch-sizes", type=str,
                      help="Comma-separated list of batch sizes to test")
    
    # Filtering options
    parser.add_argument("--refresh-older-than", type=int, default=30,
                      help="Refresh benchmarks older than this many days")
    parser.add_argument("--priority-only", action="store_true",
                      help="Only run benchmarks for priority model-hardware combinations")
    parser.add_argument("--max-benchmarks", type=int, default=0,
                      help="Maximum number of benchmarks to run (0 for unlimited)")
    parser.add_argument("--missing-only", action="store_true",
                      help="Only run benchmarks that don't exist in the database at all")
    
    # Execution options
    parser.add_argument("--simulate", action="store_true",
                      help="Simulate benchmarks instead of running real ones")
    parser.add_argument("--force", action="store_true",
                      help="Force re-running benchmarks even if recently completed")
    parser.add_argument("--dry-run", action="store_true",
                      help="List benchmarks to run without actually running them")
    
    # Tracking options
    parser.add_argument("--progress-file", type=str,
                      help="Path to store progress (defaults to benchmark_progress.json)")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    runner = IncrementalBenchmarkRunner(args)
    runner.run()

if __name__ == "__main__":
    main()