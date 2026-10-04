#!/usr/bin/env python
"""
Benchmark Integration with Database

This module enhances the benchmark_all_key_models.py script to directly
store results in the DuckDB database rather than generating JSON files.
It demonstrates how to integrate database storage with benchmarking.

Usage:
    python benchmark_with_db_integration.py --models bert vit clip --hardware cuda cpu
    python benchmark_with_db_integration.py --all-models --all-hardware
"""

import os
import sys
import json
import logging
import argparse
import datetime
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Add DuckDB database support
try:
    from data.duckdb.core.benchmark_db_api import BenchmarkDBAPI
    BENCHMARK_DB_AVAILABLE = True
except ImportError:
    BENCHMARK_DB_AVAILABLE = False
    logger.warning("benchmark_db_api not available. Using deprecated JSON fallback.")


# Always deprecate JSON output in favor of DuckDB
DEPRECATE_JSON_OUTPUT = os.environ.get("DEPRECATE_JSON_OUTPUT", "1").lower() in ("1", "true", "yes")


# Try to import required packages
try:
    import duckdb
    import pandas as pd
except ImportError:
    print("Error: Required packages not installed. Please install with:")
    print("pip install duckdb pandas")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("benchmark_db_integration")

# Import from benchmark_all_key_models
try:
    from benchmark_all_key_models import HIGH_PRIORITY_MODELS, SMALL_VERSIONS
except ImportError:
    # Fallback if module not importable
    HIGH_PRIORITY_MODELS = {
        "bert": {"name": "bert-base-uncased", "family": "embedding", "modality": "text"},
        "clap": {"name": "laion/clap-htsat-unfused", "family": "audio", "modality": "audio"},
        "clip": {"name": "openai/clip-vit-base-patch32", "family": "multimodal", "modality": "multimodal"},
        "detr": {"name": "facebook/detr-resnet-50", "family": "vision", "modality": "vision"},
        "llama": {"name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "family": "text_generation", "modality": "text"},
        "llava": {"name": "llava-hf/llava-1.5-7b-hf", "family": "multimodal", "modality": "multimodal"},
        "llava_next": {"name": "llava-hf/llava-v1.6-mistral-7b", "family": "multimodal", "modality": "multimodal"},
        "qwen2": {"name": "Qwen/Qwen2-0.5B-Instruct", "family": "text_generation", "modality": "text"},
        "t5": {"name": "t5-small", "family": "text_generation", "modality": "text"},
        "vit": {"name": "google/vit-base-patch16-224", "family": "vision", "modality": "vision"},
        "wav2vec2": {"name": "facebook/wav2vec2-base", "family": "audio", "modality": "audio"},
        "whisper": {"name": "openai/whisper-tiny", "family": "audio", "modality": "audio"},
        "xclip": {"name": "microsoft/xclip-base-patch32", "family": "multimodal", "modality": "multimodal"}
    }
    
    SMALL_VERSIONS = {
        "bert": "prajjwal1/bert-tiny",
        "t5": "google/t5-efficient-tiny",
        "vit": "facebook/deit-tiny-patch16-224",
        "whisper": "openai/whisper-tiny",
        "llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "qwen2": "Qwen/Qwen2-0.5B-Instruct"
    }

# All hardware platforms to test
ALL_HARDWARE_PLATFORMS = ["cpu", "cuda", "rocm", "mps", "openvino", "qnn", "webnn", "webgpu"]

class BenchmarkDatabaseIntegration:
    """
    Integrates benchmark testing with direct database storage of results.
    """
    
    def __init__(self, 
                 db_path: str = "./benchmark_db.duckdb",
                 use_small_models: bool = True,
                 models: Optional[List[str]] = None,
                 hardware: Optional[List[str]] = None,
                 debug: bool = False):
        """
        Initialize the benchmark database integration.
        
        Args:
            db_path: Path to the DuckDB database
            use_small_models: Use smaller model variants when available
            models: List of models to benchmark (keys from HIGH_PRIORITY_MODELS)
            hardware: List of hardware platforms to test
            debug: Enable debug logging
        """
        self.db_path = db_path
        self.use_small_models = use_small_models
        self.debug = debug
        
        # Set debug logging if requested
        if debug:
            logger.setLevel(logging.DEBUG)
        
        # Use all models if None provided
        if models is None:
            self.model_keys = list(HIGH_PRIORITY_MODELS.keys())
        else:
            # Validate model keys
            invalid_models = [m for m in models if m not in HIGH_PRIORITY_MODELS]
            if invalid_models:
                raise ValueError(f"Invalid model keys: {', '.join(invalid_models)}")
            self.model_keys = models
        
        # Use CPU only if no hardware specified
        if hardware is None:
            self.hardware = ["cpu"]
        else:
            # Validate hardware platforms
            invalid_hw = [h for h in hardware if h not in ALL_HARDWARE_PLATFORMS]
            if invalid_hw:
                raise ValueError(f"Invalid hardware platforms: {', '.join(invalid_hw)}")
            self.hardware = hardware
        
        # Get models to test
        self.models = self._get_models()
        
        # Ensure database exists and has required schema
        self._ensure_db_exists()
        
        # Get or create test run ID
        self.run_id = self._create_test_run()
        
        logger.info(f"Initialized benchmark database integration with {len(self.models)} models")
        logger.info(f"Testing on hardware: {', '.join(self.hardware)}")
    
    def _get_models(self) -> Dict[str, Dict[str, str]]:
        """
        Get the models to test, using small variants if requested.
        
        Returns:
            Dictionary of models to test
        """
        models = {}
        
        for key in self.model_keys:
            model_info = HIGH_PRIORITY_MODELS[key]
            model_data = model_info.copy()
            
            # Use small version if available and requested
            if self.use_small_models and key in SMALL_VERSIONS:
                model_data["name"] = SMALL_VERSIONS[key]
                model_data["size"] = "small"
            else:
                model_data["size"] = "base"
                
            models[key] = model_data
            
        return models
    
    def _ensure_db_exists(self) -> None:
        """
        Ensure the database exists and has the required schema.
        If not, create it.
        """
        db_file = Path(self.db_path)
        
        # Create parent directories if they don't exist
        db_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Connect to database (creates it if it doesn't exist)
            conn = duckdb.connect(self.db_path)
            
            # Check if tables exist
            tables = conn.execute("SHOW TABLES").fetchall()
            table_names = [t[0].lower() for t in tables]
            
            # List of required tables
            required_tables = [
                'hardware_platforms', 
                'models', 
                'test_runs', 
                'performance_results',
                'hardware_compatibility'
            ]
            
            missing_tables = [t for t in required_tables if t.lower() not in table_names]
            
            if missing_tables:
                logger.warning(f"Missing tables in database: {', '.join(missing_tables)}")
                
                # Check if we should create schema script
                schema_script = None
                possible_paths = [
                    "scripts/create_benchmark_schema.py",
                    "test/scripts/create_benchmark_schema.py"
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        schema_script = path
                        break
                
                if schema_script:
                    logger.info(f"Creating schema using script: {schema_script}")
                    import subprocess
                    try:
                        subprocess.run([sys.executable, schema_script, "--output", self.db_path])
                    except Exception as e:
                        logger.error(f"Error running schema script: {e}")
                        self._create_minimal_schema(conn)
                else:
                    logger.warning("Schema script not found, creating minimal schema")
                    self._create_minimal_schema(conn)
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error ensuring database exists: {e}")
            raise
    
    def _create_minimal_schema(self, conn) -> None:
        """
        Create a minimal database schema if the full schema script is unavailable.
        
        Args:
            conn: DuckDB connection
        """
        logger.info("Creating minimal database schema")
        
        # Models table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS models (
            model_id INTEGER PRIMARY KEY,
            model_name VARCHAR NOT NULL,
            model_family VARCHAR,
            modality VARCHAR,
            source VARCHAR,
            version VARCHAR,
            parameters_million FLOAT,
            metadata JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Hardware platforms table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS hardware_platforms (
            hardware_id INTEGER PRIMARY KEY,
            hardware_type VARCHAR NOT NULL,
            device_name VARCHAR,
            platform VARCHAR,
            platform_version VARCHAR,
            driver_version VARCHAR,
            memory_gb FLOAT,
            compute_units INTEGER,
            metadata JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Test runs table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS test_runs (
            run_id INTEGER PRIMARY KEY,
            test_name VARCHAR NOT NULL,
            test_type VARCHAR NOT NULL,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            execution_time_seconds FLOAT,
            success BOOLEAN,
            git_commit VARCHAR,
            git_branch VARCHAR,
            command_line VARCHAR,
            metadata JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Performance results table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS performance_results (
            result_id INTEGER PRIMARY KEY,
            run_id INTEGER,
            model_id INTEGER NOT NULL,
            hardware_id INTEGER NOT NULL,
            test_case VARCHAR NOT NULL,
            batch_size INTEGER DEFAULT 1,
            precision VARCHAR,
            total_time_seconds FLOAT,
            average_latency_ms FLOAT,
            throughput_items_per_second FLOAT,
            memory_peak_mb FLOAT,
            iterations INTEGER,
            warmup_iterations INTEGER,
            metrics JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES test_runs(run_id),
            FOREIGN KEY (model_id) REFERENCES models(model_id),
            FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id)
        )
        """)
        
        # Hardware compatibility table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS hardware_compatibility (
            compatibility_id INTEGER PRIMARY KEY,
            run_id INTEGER,
            model_id INTEGER NOT NULL,
            hardware_id INTEGER NOT NULL,
            is_compatible BOOLEAN NOT NULL,
            detection_success BOOLEAN NOT NULL,
            initialization_success BOOLEAN NOT NULL,
            error_message VARCHAR,
            error_type VARCHAR,
            suggested_fix VARCHAR,
            workaround_available BOOLEAN,
            compatibility_score FLOAT,
            metadata JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES test_runs(run_id),
            FOREIGN KEY (model_id) REFERENCES models(model_id),
            FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id)
        )
        """)
        
        logger.info("Minimal schema created successfully")
    
    def _create_test_run(self) -> int:
        """
        Create a new test run entry in the database.
        
        Returns:
            run_id: ID of the test run in the database
        """
        # Generate a test name
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        test_name = f"benchmark_db_integration_{timestamp}"
        
        # Get current time
        now = datetime.datetime.now()
        
        # Get command line
        command_line = f"python {' '.join(sys.argv)}"
        
        # Create metadata JSON
        metadata = {
            'models': list(self.models.keys()),
            'hardware': self.hardware,
            'use_small_models': self.use_small_models
        }
        
        # Try to get git info
        try:
            import subprocess
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], 
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
            
            git_branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
        except:
            git_commit = None
            git_branch = None
        
        conn = duckdb.connect(self.db_path)
        try:
            # Get next run_id
            run_id_result = conn.execute("SELECT COALESCE(MAX(run_id), 0) + 1 FROM test_runs").fetchone()
            run_id = run_id_result[0] if run_id_result else 1
            
            # Insert test run
            conn.execute("""
            INSERT INTO test_runs (
                run_id, test_name, test_type, started_at, completed_at, 
                execution_time_seconds, success, git_commit, git_branch, 
                command_line, metadata
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                run_id, test_name, 'performance', now, None, 
                0, True, git_commit, git_branch, command_line, 
                json.dumps(metadata)
            ])
            
            logger.info(f"Created new test run: {test_name} (ID: {run_id})")
            return run_id
        finally:
            conn.close()
    
    def _ensure_model_exists(self, model_key: str) -> int:
        """
        Ensure a model exists in the database, adding it if it doesn't.
        
        Args:
            model_key: Key of the model in self.models
            
        Returns:
            model_id: ID of the model in the database
        """
        if model_key not in self.models:
            raise ValueError(f"Model key not found: {model_key}")
        
        model_info = self.models[model_key]
        model_name = model_info["name"]
        model_family = model_info.get("family", "")
        modality = model_info.get("modality", "")
        
        conn = duckdb.connect(self.db_path)
        try:
            # Check if model exists
            model_result = conn.execute(
                "SELECT model_id FROM models WHERE model_name = ?", 
                [model_name]
            ).fetchone()
            
            if model_result:
                return model_result[0]
            
            # Get next model_id
            model_id_result = conn.execute("SELECT COALESCE(MAX(model_id), 0) + 1 FROM models").fetchone()
            model_id = model_id_result[0] if model_id_result else 1
            
            # Add model to database
            conn.execute("""
            INSERT INTO models (
                model_id, model_name, model_family, modality, source, version, parameters_million
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                model_id, model_name, model_family, modality, 
                'huggingface', '1.0', None
            ])
            
            logger.info(f"Added model to database: {model_name} (ID: {model_id})")
            return model_id
        finally:
            conn.close()
    
    def _ensure_hardware_exists(self, hardware_type: str, device_name: Optional[str] = None) -> int:
        """
        Ensure a hardware platform exists in the database, adding it if it doesn't.
        
        Args:
            hardware_type: Type of hardware (cpu, cuda, etc.)
            device_name: Name of the device (optional)
            
        Returns:
            hardware_id: ID of the hardware in the database
        """
        conn = duckdb.connect(self.db_path)
        try:
            # Build query
            if device_name:
                hw_query = "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = ? AND device_name = ?"
                params = [hardware_type, device_name]
            else:
                hw_query = "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = ? AND device_name IS NULL"
                params = [hardware_type]
            
            # Check if hardware exists
            hw_result = conn.execute(hw_query, params).fetchone()
            
            if hw_result:
                return hw_result[0]
            
            # If we have no device name, try with any device of that type
            if not device_name:
                any_hw = conn.execute(
                    "SELECT hardware_id, device_name FROM hardware_platforms WHERE hardware_type = ? LIMIT 1",
                    [hardware_type]
                ).fetchone()
                
                if any_hw:
                    logger.info(f"Using existing hardware of type {hardware_type}: {any_hw[1]} (ID: {any_hw[0]})")
                    return any_hw[0]
            
            # Get hardware details through detection when possible
            platform = None
            platform_version = None
            driver_version = None
            memory_gb = None
            compute_units = None
            
            # Try to detect hardware details
            try:
                if hardware_type == 'cpu':
                    import platform as plt
                    import psutil
                    platform = plt.system()
                    platform_version = plt.version()
                    memory_gb = psutil.virtual_memory().total / (1024 ** 3)
                    compute_units = psutil.cpu_count(logical=False)
                    if not device_name:
                        device_name = plt.processor()
                
                elif hardware_type == 'cuda':
                    try:
                        import torch
                        if torch.cuda.is_available():
                            if not device_name:
                                device_name = torch.cuda.get_device_name(0)
                            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                            platform = 'CUDA'
                            platform_version = torch.version.cuda
                    except (ImportError, AttributeError):
                        pass
                
                elif hardware_type == 'rocm':
                    try:
                        import torch
                        if torch.cuda.is_available() and 'rocm' in torch.__version__.lower():
                            if not device_name:
                                device_name = torch.cuda.get_device_name(0)
                            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                            platform = 'ROCm'
                    except (ImportError, AttributeError):
                        pass
            except Exception as e:
                logger.warning(f"Error detecting hardware details: {e}")
            
            # Default values if detection failed
            if not device_name:
                if hardware_type == 'cpu':
                    device_name = 'CPU'
                elif hardware_type == 'cuda':
                    device_name = 'NVIDIA GPU'
                elif hardware_type == 'rocm':
                    device_name = 'AMD GPU'
                elif hardware_type == 'mps':
                    device_name = 'Apple Silicon'
                elif hardware_type == 'openvino':
                    device_name = 'OpenVINO Device'
                elif hardware_type == 'webnn':
                    device_name = 'WebNN Device'
                elif hardware_type == 'webgpu':
                    device_name = 'WebGPU Device'
                else:
                    device_name = f"{hardware_type.upper()} Device"
            
            # Get next hardware_id
            hw_id_result = conn.execute("SELECT COALESCE(MAX(hardware_id), 0) + 1 FROM hardware_platforms").fetchone()
            hardware_id = hw_id_result[0] if hw_id_result else 1
            
            # Add hardware to database
            conn.execute("""
            INSERT INTO hardware_platforms (
                hardware_id, hardware_type, device_name, platform, platform_version,
                driver_version, memory_gb, compute_units
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                hardware_id, hardware_type, device_name, platform, platform_version,
                driver_version, memory_gb, compute_units
            ])
            
            logger.info(f"Added hardware to database: {hardware_type} - {device_name} (ID: {hardware_id})")
            return hardware_id
        finally:
            conn.close()
    
    def run_benchmark_simulation(self, model_key: str, hardware_type: str) -> Dict[str, Any]:
        """
        Run a simulated benchmark for a model on a specific hardware platform.
        In a real implementation, you would call your actual benchmark code here.
        
        Args:
            model_key: Key of the model in self.models
            hardware_type: Type of hardware to benchmark on
            
        Returns:
            Dictionary with benchmark results
        """
        if model_key not in self.models:
            raise ValueError(f"Model key not found: {model_key}")
        
        model_info = self.models[model_key]
        model_name = model_info["name"]
        model_family = model_info.get("family", "")
        
        logger.info(f"Simulating benchmark for {model_name} on {hardware_type}")
        
        # Generate simulated results
        # In a real implementation, you would run actual benchmarks here
        import random
        
        # Base values that vary by model family
        if model_family == "embedding":
            base_latency = random.uniform(5, 20)  # ms
            base_throughput = random.uniform(100, 500)  # items/s
            base_memory = random.uniform(500, 2000)  # MB
        elif model_family == "text_generation":
            base_latency = random.uniform(20, 100)  # ms
            base_throughput = random.uniform(10, 100)  # items/s
            base_memory = random.uniform(1000, 5000)  # MB
        elif model_family in ["vision", "multimodal"]:
            base_latency = random.uniform(10, 50)  # ms
            base_throughput = random.uniform(20, 200)  # items/s
            base_memory = random.uniform(800, 3000)  # MB
        elif model_family == "audio":
            base_latency = random.uniform(15, 80)  # ms
            base_throughput = random.uniform(15, 150)  # items/s
            base_memory = random.uniform(700, 2500)  # MB
        else:
            base_latency = random.uniform(10, 50)  # ms
            base_throughput = random.uniform(20, 200)  # items/s
            base_memory = random.uniform(500, 2000)  # MB
        
        # Hardware modifiers
        hw_latency_factor = 1.0
        hw_throughput_factor = 1.0
        hw_memory_factor = 1.0
        
        if hardware_type == "cpu":
            hw_latency_factor = 2.0  # CPU typically slower
            hw_throughput_factor = 0.5  # Lower throughput on CPU
        elif hardware_type == "cuda":
            hw_latency_factor = 0.5  # CUDA faster
            hw_throughput_factor = 2.0  # Higher throughput on CUDA
        elif hardware_type == "rocm":
            hw_latency_factor = 0.6  # ROCm somewhat faster than CPU
            hw_throughput_factor = 1.8  # Good throughput on ROCm
        
        # Model size modifier
        size_factor = 1.0
        if self.use_small_models and model_key in SMALL_VERSIONS:
            size_factor = 0.5  # Small models are faster and use less memory
        
        # Calculate final metrics
        latency = base_latency * hw_latency_factor * size_factor
        throughput = base_throughput * hw_throughput_factor / size_factor
        memory = base_memory * hw_memory_factor * size_factor
        
        # Determine compatibility based on model and hardware
        is_compatible = True
        compatibility_score = 1.0
        
        # Some models might not be compatible with certain hardware
        if model_key in ["llava", "llava_next"] and hardware_type in ["rocm", "mps", "openvino", "webnn", "webgpu"]:
            is_compatible = False
            compatibility_score = 0.0
            error_message = f"{model_key} is not compatible with {hardware_type}"
            error_type = "unsupported_model"
        elif model_key == "xclip" and hardware_type in ["webnn", "webgpu"]:
            is_compatible = False
            compatibility_score = 0.0
            error_message = f"{model_key} is not compatible with {hardware_type}"
            error_type = "unsupported_model"
        else:
            error_message = None
            error_type = None
        
        # Batch sizes to test
        batch_sizes = [1, 2, 4, 8]
        
        # Iterate through batch sizes
        batch_results = []
        for batch_size in batch_sizes:
            # Batch size affects metrics
            batch_latency = latency * (1 + 0.2 * (batch_size - 1))  # Latency increases with batch size
            batch_throughput = throughput * batch_size * 0.9  # Throughput increases sub-linearly
            batch_memory = memory * (1 + 0.3 * (batch_size - 1))  # Memory increases with batch size
            
            # Create result for this batch size
            result = {
                "model_key": model_key,
                "model_name": model_name,
                "hardware_type": hardware_type,
                "batch_size": batch_size,
                "precision": "fp32",
                "test_case": "default",
                "average_latency_ms": batch_latency,
                "throughput_items_per_second": batch_throughput,
                "memory_peak_mb": batch_memory,
                "total_time_seconds": 10.0,  # Simulated total time
                "iterations": 100,
                "warmup_iterations": 10,
                "is_compatible": is_compatible,
                "compatibility_score": compatibility_score,
                "error_message": error_message,
                "error_type": error_type
            }
            
            batch_results.append(result)
        
        # Sleep to simulate benchmark running
        time.sleep(0.5)
        
        return {
            "model_key": model_key,
            "hardware_type": hardware_type,
            "is_compatible": is_compatible,
            "batch_results": batch_results
        }
    
    def store_benchmark_result(self, result: Dict[str, Any]) -> None:
        """
        Store benchmark result in the database.
        
        Args:
            result: Dictionary with benchmark results
        """
        model_key = result["model_key"]
        hardware_type = result["hardware_type"]
        is_compatible = result["is_compatible"]
        batch_results = result["batch_results"]
        
        # Get database IDs
        model_id = self._ensure_model_exists(model_key)
        hardware_id = self._ensure_hardware_exists(hardware_type)
        
        conn = duckdb.connect(self.db_path)
        try:
            # First store compatibility result
            compatibility_sample = batch_results[0]  # Use first batch result for compatibility data
            
            # Get next compatibility_id
            comp_id_result = conn.execute("SELECT COALESCE(MAX(compatibility_id), 0) + 1 FROM hardware_compatibility").fetchone()
            compatibility_id = comp_id_result[0] if comp_id_result else 1
            
            # Insert compatibility result
            conn.execute("""
            INSERT INTO hardware_compatibility (
                compatibility_id, run_id, model_id, hardware_id, is_compatible,
                detection_success, initialization_success, error_message, error_type,
                compatibility_score, metadata
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                compatibility_id, self.run_id, model_id, hardware_id,
                is_compatible, True, is_compatible,
                compatibility_sample.get("error_message"),
                compatibility_sample.get("error_type"),
                compatibility_sample.get("compatibility_score", 1.0 if is_compatible else 0.0),
                json.dumps({})
            ])
            
            # Now store performance results for each batch size
            for batch_result in batch_results:
                # Skip performance results for incompatible combinations
                if not is_compatible:
                    continue
                
                # Get next result_id
                result_id_result = conn.execute("SELECT COALESCE(MAX(result_id), 0) + 1 FROM performance_results").fetchone()
                result_id = result_id_result[0] if result_id_result else 1
                
                # Insert performance result
                conn.execute("""
                INSERT INTO performance_results (
                    result_id, run_id, model_id, hardware_id, test_case, batch_size,
                    precision, total_time_seconds, average_latency_ms, throughput_items_per_second,
                    memory_peak_mb, iterations, warmup_iterations, metrics
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    result_id, self.run_id, model_id, hardware_id,
                    batch_result["test_case"], batch_result["batch_size"],
                    batch_result["precision"], batch_result["total_time_seconds"],
                    batch_result["average_latency_ms"], batch_result["throughput_items_per_second"],
                    batch_result["memory_peak_mb"], batch_result["iterations"],
                    batch_result["warmup_iterations"], json.dumps({})
                ])
            
            # Commit changes
            conn.commit()
            
            logger.info(f"Stored results for {model_key} on {hardware_type}")
        except Exception as e:
            logger.error(f"Error storing benchmark result: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def run_all_benchmarks(self) -> None:
        """
        Run benchmarks for all model-hardware combinations and store results in the database.
        """
        logger.info("Running benchmarks for all model-hardware combinations")
        
        # Track statistics
        total_combinations = len(self.models) * len(self.hardware)
        completed = 0
        successful = 0
        failed = 0
        
        start_time = time.time()
        
        # Run benchmarks for each model on each hardware platform
        for model_key in self.models:
            for hw_type in self.hardware:
                try:
                    # Run benchmark
                    result = self.run_benchmark_simulation(model_key, hw_type)
                    
                    # Store result in database
                    self.store_benchmark_result(result)
                    
                    successful += 1
                    
                except Exception as e:
                    logger.error(f"Error benchmarking {model_key} on {hw_type}: {e}")
                    failed += 1
                
                completed += 1
                
                # Log progress
                progress = (completed / total_combinations) * 100
                logger.info(f"Progress: {progress:.1f}% ({completed}/{total_combinations})")
        
        # Update test run with completion information
        self._update_test_run_completion(start_time)
        
        # Log summary
        duration = time.time() - start_time
        logger.info(f"Benchmarking completed in {duration:.1f} seconds")
        logger.info(f"Total combinations: {total_combinations}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
    
    def _update_test_run_completion(self, start_time: float) -> None:
        """
        Update the test run with completion information.
        
        Args:
            start_time: Start time of the benchmarking run
        """
        conn = duckdb.connect(self.db_path)
        try:
            # Calculate execution time
            now = datetime.datetime.now()
            execution_time = time.time() - start_time
            
            # Update test run
            conn.execute("""
            UPDATE test_runs
            SET completed_at = ?, execution_time_seconds = ?, success = ?
            WHERE run_id = ?
            """, [now, execution_time, True, self.run_id])
            
            logger.info(f"Updated test run (ID: {self.run_id}) with completion information")
        except Exception as e:
            logger.error(f"Error updating test run: {e}")
        finally:
            conn.close()
    
    def get_benchmark_results_for_model(self, model_key: str) -> pd.DataFrame:
        """
        Retrieve benchmark results for a specific model from the database.
        
        Args:
            model_key: Key of the model in self.models
            
        Returns:
            DataFrame with benchmark results
        """
        if model_key not in self.models:
            raise ValueError(f"Model key not found: {model_key}")
        
        model_info = self.models[model_key]
        model_name = model_info["name"]
        
        conn = duckdb.connect(self.db_path)
        try:
            # Query for performance results
            query = """
            SELECT 
                m.model_name,
                hp.hardware_type,
                hp.device_name,
                pr.batch_size,
                pr.average_latency_ms,
                pr.throughput_items_per_second,
                pr.memory_peak_mb
            FROM 
                performance_results pr
            JOIN 
                models m ON pr.model_id = m.model_id
            JOIN 
                hardware_platforms hp ON pr.hardware_id = hp.hardware_id
            WHERE 
                m.model_name = ?
            ORDER BY
                hp.hardware_type, pr.batch_size
            """
            
            df = conn.execute(query, [model_name]).fetch_df()
            return df
        finally:
            conn.close()
    
    def get_compatibility_matrix(self) -> pd.DataFrame:
        """
        Generate a compatibility matrix for all model-hardware combinations.
        
        Returns:
            DataFrame with compatibility matrix
        """
        conn = duckdb.connect(self.db_path)
        try:
            # Query for compatibility results
            query = """
            SELECT 
                m.model_name,
                m.model_family,
                hp.hardware_type,
                MAX(hc.is_compatible) as is_compatible,
                MAX(hc.compatibility_score) as compatibility_score
            FROM 
                hardware_compatibility hc
            JOIN 
                models m ON hc.model_id = m.model_id
            JOIN 
                hardware_platforms hp ON hc.hardware_id = hp.hardware_id
            WHERE
                hc.run_id = ?
            GROUP BY
                m.model_name, m.model_family, hp.hardware_type
            """
            
            df = conn.execute(query, [self.run_id]).fetch_df()
            
            # Pivot the dataframe to create a matrix
            matrix = df.pivot_table(
                index=["model_name", "model_family"],
                columns="hardware_type",
                values="is_compatible",
                aggfunc="max",
                fill_value=False
            )
            
            return matrix
        finally:
            conn.close()
    
    def generate_report(self) -> str:
        """
        Generate a benchmark report.
        
        Returns:
            Path to the generated report file
        """
        # Generate timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"benchmark_db_report_{timestamp}.md"
        
        # Get compatibility matrix
        compatibility_matrix = self.get_compatibility_matrix()
        
        with open(report_file, "w") as f:
            f.write("# Benchmark Database Integration Report\n\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Models tested
            f.write("## Models Tested\n\n")
            f.write("| Model Key | Model Name | Family | Modality |\n")
            f.write("|-----------|------------|--------|----------|\n")
            
            for key, model_info in self.models.items():
                f.write(f"| {key} | {model_info['name']} | {model_info.get('family', '')} | {model_info.get('modality', '')} |\n")
            
            f.write("\n")
            
            # Hardware platforms
            f.write("## Hardware Platforms Tested\n\n")
            f.write("| Hardware Type |\n")
            f.write("|---------------|\n")
            
            for hw in self.hardware:
                f.write(f"| {hw} |\n")
            
            f.write("\n")
            
            # Compatibility Matrix
            f.write("## Hardware Compatibility Matrix\n\n")
            
            # Convert DataFrame to markdown table
            if not compatibility_matrix.empty:
                # Write header
                f.write("| Model / Hardware |")
                for col in compatibility_matrix.columns:
                    f.write(f" {col} |")
                f.write("\n")
                
                # Write separator
                f.write("|--------------|")
                for _ in compatibility_matrix.columns:
                    f.write("---------|")
                f.write("\n")
                
                # Write data
                for idx, row in compatibility_matrix.iterrows():
                    model_name, model_family = idx
                    f.write(f"| {model_name} |")
                    
                    for col in compatibility_matrix.columns:
                        value = row.get(col, False)
                        if value:
                            f.write(" ✅ |")
                        else:
                            f.write(" ❌ |")
                    
                    f.write("\n")
            else:
                f.write("No compatibility data available.\n")
            
            f.write("\n")
            
            # Performance summary for selected models
            f.write("## Performance Summary\n\n")
            
            for model_key in self.models:
                try:
                    # Get results for this model
                    df = self.get_benchmark_results_for_model(model_key)
                    
                    if not df.empty:
                        f.write(f"### {model_key}\n\n")
                        
                        # Format as markdown table
                        f.write("| Hardware | Batch Size | Latency (ms) | Throughput (items/s) | Memory (MB) |\n")
                        f.write("|----------|------------|--------------|---------------------|------------|\n")
                        
                        for _, row in df.iterrows():
                            f.write(f"| {row['hardware_type']} | {row['batch_size']} | {row['average_latency_ms']:.2f} | {row['throughput_items_per_second']:.2f} | {row['memory_peak_mb']:.2f} |\n")
                        
                        f.write("\n")
                except Exception as e:
                    f.write(f"Error retrieving data for {model_key}: {e}\n\n")
            
            # Summary
            f.write("## Summary\n\n")
            f.write(f"- {len(self.models)} models tested\n")
            f.write(f"- {len(self.hardware)} hardware platforms\n")
            
            try:
                # Get total number of performance results
                conn = duckdb.connect(self.db_path)
                result = conn.execute(
                    "SELECT COUNT(*) FROM performance_results WHERE run_id = ?", 
                    [self.run_id]
                ).fetchone()
                count = result[0] if result else 0
                conn.close()
                
                f.write(f"- {count} benchmark results generated\n")
            except:
                pass
            
            f.write("\n")
            
            # Database information
            f.write("## Database Information\n\n")
            f.write(f"- Database path: {self.db_path}\n")
            f.write(f"- Test run ID: {self.run_id}\n")
            
            # Get database size
            try:
                db_size = os.path.getsize(self.db_path) / (1024 * 1024)  # MB
                f.write(f"- Database size: {db_size:.2f} MB\n")
            except:
                pass
            
            f.write("\n")
            
            # Example queries
            f.write("## Example Queries\n\n")
            f.write("```sql\n")
            f.write("-- Get all performance results for BERT\n")
            f.write("SELECT \n")
            f.write("    m.model_name,\n")
            f.write("    hp.hardware_type,\n")
            f.write("    pr.batch_size,\n")
            f.write("    pr.average_latency_ms,\n")
            f.write("    pr.throughput_items_per_second\n")
            f.write("FROM \n")
            f.write("    performance_results pr\n")
            f.write("JOIN \n")
            f.write("    models m ON pr.model_id = m.model_id\n")
            f.write("JOIN \n")
            f.write("    hardware_platforms hp ON pr.hardware_id = hp.hardware_id\n")
            f.write("WHERE \n")
            f.write("    m.model_family = 'bert'\n")
            f.write("ORDER BY\n")
            f.write("    hp.hardware_type, pr.batch_size;\n")
            f.write("\n")
            f.write("-- Get hardware compatibility matrix\n")
            f.write("SELECT \n")
            f.write("    m.model_family,\n")
            f.write("    hp.hardware_type,\n")
            f.write("    COUNT(*) as total_tests,\n")
            f.write("    SUM(CASE WHEN hc.is_compatible THEN 1 ELSE 0 END) as compatible_count,\n")
            f.write("    AVG(hc.compatibility_score) as avg_score\n")
            f.write("FROM \n")
            f.write("    hardware_compatibility hc\n")
            f.write("JOIN \n")
            f.write("    models m ON hc.model_id = m.model_id\n")
            f.write("JOIN \n")
            f.write("    hardware_platforms hp ON hc.hardware_id = hp.hardware_id\n")
            f.write("GROUP BY\n")
            f.write("    m.model_family, hp.hardware_type;\n")
            f.write("```\n")
        
        logger.info(f"Report generated: {report_file}")
        return report_file

def main():
    """Parse arguments and run the benchmark database integration."""
    parser = argparse.ArgumentParser(description="Benchmark integration with database")
    
    parser.add_argument("--db-path", default="./benchmark_db.duckdb",
                       help="Path to the DuckDB database")
    
    # Model selection
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--models", nargs="+", 
                           help="Models to benchmark (space-separated list of keys)")
    model_group.add_argument("--all-models", action="store_true",
                           help="Benchmark all high priority models")
    
    # Hardware selection
    hw_group = parser.add_mutually_exclusive_group()
    hw_group.add_argument("--hardware", nargs="+",
                        help="Hardware platforms to benchmark on (space-separated list)")
    hw_group.add_argument("--all-hardware", action="store_true",
                        help="Benchmark on all hardware platforms")
    
    # Other options
    parser.add_argument("--small-models", action="store_true",
                       help="Use smaller model variants when available")
    parser.add_argument("--generate-report", action="store_true",
                       help="Generate a benchmark report")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    
    parser.add_argument("--db-path", type=str, default=None,
                      help="Path to the benchmark database")
    parser.add_argument("--db-only", action="store_true",
                      help="Store results only in the database, not in JSON")
    args = parser.parse_args()
    
    # Determine models to benchmark
    models = None
    if args.models:
        models = args.models
    elif args.all_models:
        models = list(HIGH_PRIORITY_MODELS.keys())
    else:
        # Default to a few representative models
        models = ["bert", "vit", "t5", "whisper"]
    
    # Determine hardware to benchmark on
    hardware = None
    if args.hardware:
        hardware = args.hardware
    elif args.all_hardware:
        hardware = ALL_HARDWARE_PLATFORMS
    else:
        # Default to CPU only for safety
        hardware = ["cpu"]
    
    try:
        # Create benchmark database integration
        db_path = args.db_path
        if db_path is None:
            db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
            logger.info(f"Using database path from environment: {db_path}")
            
        benchmarker = BenchmarkDatabaseIntegration(
            db_path=db_path,
            use_small_models=args.small_models,
            models=models,
            hardware=hardware,
            debug=args.debug
        )
        
        # Run benchmarks
        benchmarker.run_all_benchmarks()
        
        # Generate report if requested
        if args.generate_report:
            report_file = benchmarker.generate_report()
            print(f"Report generated: {report_file}")
        
    except Exception as e:
        logger.error(f"Error running benchmarks: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())