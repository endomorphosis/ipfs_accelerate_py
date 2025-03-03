#!/usr/bin/env python3
"""
Model Validation Tracker - Database for tracking model validation results

This module provides a simple database interface for tracking model validation
results across different hardware platforms. It stores validation and benchmark
results, provides query capabilities, and supports import/export functionality.
"""

import os
import json
import sqlite3
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DB_PATH = "./model_validation.db"
DEFAULT_FUNCTIONALITY_DIR = "./functionality_reports"
DEFAULT_BENCHMARK_DIR = "./benchmark_results"
DEFAULT_EXPORT_DIR = "./validation_exports"

class ModelValidationTracker:
    """
    Database for tracking model validation and benchmark results
    """
    
    def __init__(self, db_path: str = DEFAULT_DB_PATH, create_if_missing: bool = True):
        """
        Initialize the validation tracker.
        
        Args:
            db_path: Path to the SQLite database file
            create_if_missing: Whether to create the database if it doesn't exist
        """
        self.db_path = db_path
        
        # Check if database exists
        db_exists = os.path.exists(db_path)
        
        # Create directory if needed
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        
        # Connect to database
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        self.cursor = self.conn.cursor()
        
        # Create tables if needed
        if not db_exists and create_if_missing:
            self._create_tables()
        
        logger.info(f"Connected to validation database: {db_path}")
    
    def _create_tables(self):
        """Create necessary database tables if they don't exist"""
        logger.info("Creating database tables")
        
        # Create models table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            family TEXT NOT NULL,
            first_tested TIMESTAMP,
            last_tested TIMESTAMP,
            UNIQUE(name)
        )
        ''')
        
        # Create hardware platforms table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS hardware_platforms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            UNIQUE(name)
        )
        ''')
        
        # Create functionality tests table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS functionality_tests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER NOT NULL,
            hardware_id INTEGER NOT NULL,
            success BOOLEAN NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            error_message TEXT,
            test_details TEXT,
            FOREIGN KEY (model_id) REFERENCES models (id),
            FOREIGN KEY (hardware_id) REFERENCES hardware_platforms (id)
        )
        ''')
        
        # Create benchmark results table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS benchmark_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER NOT NULL,
            hardware_id INTEGER NOT NULL,
            latency_ms REAL,
            throughput REAL,
            memory_mb REAL,
            batch_size INTEGER,
            timestamp TIMESTAMP NOT NULL,
            benchmark_details TEXT,
            FOREIGN KEY (model_id) REFERENCES models (id),
            FOREIGN KEY (hardware_id) REFERENCES hardware_platforms (id)
        )
        ''')
        
        # Create index for faster queries
        self.cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_functionality_model_hardware 
        ON functionality_tests (model_id, hardware_id)
        ''')
        
        self.cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_benchmark_model_hardware 
        ON benchmark_results (model_id, hardware_id)
        ''')
        
        # Commit changes
        self.conn.commit()
        logger.info("Database tables created successfully")
        
        # Add default hardware platforms
        self._add_default_hardware_platforms()
    
    def _add_default_hardware_platforms(self):
        """Add default hardware platforms to the database"""
        default_platforms = [
            ("cpu", "CPU (Central Processing Unit)"),
            ("cuda", "NVIDIA CUDA GPU"),
            ("mps", "Apple Metal Performance Shaders (M1/M2)"),
            ("rocm", "AMD ROCm (Radeon Open Compute)"),
            ("openvino", "Intel OpenVINO"),
            ("webnn", "WebNN (Web Neural Network API)"),
            ("webgpu", "WebGPU")
        ]
        
        for name, description in default_platforms:
            self.add_hardware_platform(name, description)
        
        self.conn.commit()
    
    def add_model(self, name: str, family: str) -> int:
        """
        Add a model to the database or get its ID if it already exists.
        
        Args:
            name: Model name
            family: Model family (embedding, text_generation, vision, audio, multimodal)
            
        Returns:
            Model ID
        """
        # Check if model already exists
        self.cursor.execute("SELECT id FROM models WHERE name = ?", (name,))
        result = self.cursor.fetchone()
        
        if result:
            # Model exists, return its ID
            return result["id"]
        
        # Add new model
        timestamp = datetime.now().isoformat()
        self.cursor.execute(
            "INSERT INTO models (name, family, first_tested, last_tested) VALUES (?, ?, ?, ?)",
            (name, family, timestamp, timestamp)
        )
        self.conn.commit()
        
        return self.cursor.lastrowid
    
    def add_hardware_platform(self, name: str, description: str = "") -> int:
        """
        Add a hardware platform to the database or get its ID if it already exists.
        
        Args:
            name: Hardware platform name
            description: Hardware platform description
            
        Returns:
            Hardware platform ID
        """
        # Check if platform already exists
        self.cursor.execute("SELECT id FROM hardware_platforms WHERE name = ?", (name,))
        result = self.cursor.fetchone()
        
        if result:
            # Platform exists, return its ID
            return result["id"]
        
        # Add new platform
        self.cursor.execute(
            "INSERT INTO hardware_platforms (name, description) VALUES (?, ?)",
            (name, description)
        )
        self.conn.commit()
        
        return self.cursor.lastrowid
    
    def update_model_last_tested(self, model_id: int):
        """Update the last_tested timestamp for a model"""
        timestamp = datetime.now().isoformat()
        self.cursor.execute(
            "UPDATE models SET last_tested = ? WHERE id = ?",
            (timestamp, model_id)
        )
        self.conn.commit()
    
    def add_functionality_test(self, model_name: str, family: str, hardware_name: str, 
                            success: bool, error_message: str = "", 
                            test_details: Dict = None) -> int:
        """
        Add a functionality test result to the database.
        
        Args:
            model_name: Model name
            family: Model family
            hardware_name: Hardware platform name
            success: Whether the test was successful
            error_message: Error message if the test failed
            test_details: Additional test details as a dictionary
            
        Returns:
            Test ID
        """
        # Get or create model and hardware IDs
        model_id = self.add_model(model_name, family)
        hardware_id = self.add_hardware_platform(hardware_name)
        
        # Update model's last_tested timestamp
        self.update_model_last_tested(model_id)
        
        # Convert test details to JSON string if provided
        details_json = json.dumps(test_details) if test_details else None
        
        # Add test result
        timestamp = datetime.now().isoformat()
        self.cursor.execute(
            "INSERT INTO functionality_tests (model_id, hardware_id, success, timestamp, error_message, test_details) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (model_id, hardware_id, success, timestamp, error_message, details_json)
        )
        self.conn.commit()
        
        return self.cursor.lastrowid
    
    def add_benchmark_result(self, model_name: str, family: str, hardware_name: str,
                           latency_ms: float = None, throughput: float = None, 
                           memory_mb: float = None, batch_size: int = 1,
                           benchmark_details: Dict = None) -> int:
        """
        Add a benchmark result to the database.
        
        Args:
            model_name: Model name
            family: Model family
            hardware_name: Hardware platform name
            latency_ms: Latency in milliseconds
            throughput: Throughput in items per second
            memory_mb: Memory usage in megabytes
            batch_size: Batch size used for testing
            benchmark_details: Additional benchmark details as a dictionary
            
        Returns:
            Benchmark ID
        """
        # Get or create model and hardware IDs
        model_id = self.add_model(model_name, family)
        hardware_id = self.add_hardware_platform(hardware_name)
        
        # Update model's last_tested timestamp
        self.update_model_last_tested(model_id)
        
        # Convert benchmark details to JSON string if provided
        details_json = json.dumps(benchmark_details) if benchmark_details else None
        
        # Add benchmark result
        timestamp = datetime.now().isoformat()
        self.cursor.execute(
            "INSERT INTO benchmark_results (model_id, hardware_id, latency_ms, throughput, "
            "memory_mb, batch_size, timestamp, benchmark_details) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (model_id, hardware_id, latency_ms, throughput, memory_mb, batch_size, timestamp, details_json)
        )
        self.conn.commit()
        
        return self.cursor.lastrowid
    
    def get_model_functionality(self, model_name: str, hardware_name: Optional[str] = None) -> List[Dict]:
        """
        Get functionality test results for a model.
        
        Args:
            model_name: Model name
            hardware_name: Optional hardware platform name to filter by
            
        Returns:
            List of functionality test results
        """
        if hardware_name:
            # Get results for specific model and hardware
            self.cursor.execute(
                "SELECT f.*, m.name as model_name, m.family, h.name as hardware_name "
                "FROM functionality_tests f "
                "JOIN models m ON f.model_id = m.id "
                "JOIN hardware_platforms h ON f.hardware_id = h.id "
                "WHERE m.name = ? AND h.name = ? "
                "ORDER BY f.timestamp DESC",
                (model_name, hardware_name)
            )
        else:
            # Get results for specific model on all hardware
            self.cursor.execute(
                "SELECT f.*, m.name as model_name, m.family, h.name as hardware_name "
                "FROM functionality_tests f "
                "JOIN models m ON f.model_id = m.id "
                "JOIN hardware_platforms h ON f.hardware_id = h.id "
                "WHERE m.name = ? "
                "ORDER BY f.timestamp DESC, h.name",
                (model_name,)
            )
        
        # Parse test details JSON if available
        results = []
        for row in self.cursor.fetchall():
            result = dict(row)
            if result["test_details"]:
                try:
                    result["test_details"] = json.loads(result["test_details"])
                except:
                    # Keep as string if JSON parsing fails
                    pass
            results.append(result)
        
        return results
    
    def get_model_benchmarks(self, model_name: str, hardware_name: Optional[str] = None) -> List[Dict]:
        """
        Get benchmark results for a model.
        
        Args:
            model_name: Model name
            hardware_name: Optional hardware platform name to filter by
            
        Returns:
            List of benchmark results
        """
        if hardware_name:
            # Get results for specific model and hardware
            self.cursor.execute(
                "SELECT b.*, m.name as model_name, m.family, h.name as hardware_name "
                "FROM benchmark_results b "
                "JOIN models m ON b.model_id = m.id "
                "JOIN hardware_platforms h ON b.hardware_id = h.id "
                "WHERE m.name = ? AND h.name = ? "
                "ORDER BY b.timestamp DESC",
                (model_name, hardware_name)
            )
        else:
            # Get results for specific model on all hardware
            self.cursor.execute(
                "SELECT b.*, m.name as model_name, m.family, h.name as hardware_name "
                "FROM benchmark_results b "
                "JOIN models m ON b.model_id = m.id "
                "JOIN hardware_platforms h ON b.hardware_id = h.id "
                "WHERE m.name = ? "
                "ORDER BY b.timestamp DESC, h.name",
                (model_name,)
            )
        
        # Parse benchmark details JSON if available
        results = []
        for row in self.cursor.fetchall():
            result = dict(row)
            if result["benchmark_details"]:
                try:
                    result["benchmark_details"] = json.loads(result["benchmark_details"])
                except:
                    # Keep as string if JSON parsing fails
                    pass
            results.append(result)
        
        return results
    
    def get_family_functionality(self, family: str) -> Dict[str, Dict[str, Any]]:
        """
        Get functionality test results for a model family.
        
        Args:
            family: Model family
            
        Returns:
            Dictionary mapping model names to their functionality results
        """
        self.cursor.execute(
            "SELECT m.name as model_name, h.name as hardware_name, "
            "MAX(f.timestamp) as last_tested, "
            "SUM(CASE WHEN f.success THEN 1 ELSE 0 END) as successful_tests, "
            "COUNT(f.id) as total_tests "
            "FROM functionality_tests f "
            "JOIN models m ON f.model_id = m.id "
            "JOIN hardware_platforms h ON f.hardware_id = h.id "
            "WHERE m.family = ? "
            "GROUP BY m.name, h.name "
            "ORDER BY m.name, h.name",
            (family,)
        )
        
        # Organize results by model and hardware
        results = {}
        for row in self.cursor.fetchall():
            model_name = row["model_name"]
            hardware_name = row["hardware_name"]
            
            if model_name not in results:
                results[model_name] = {"hardware": {}}
            
            success_rate = 0
            if row["total_tests"] > 0:
                success_rate = (row["successful_tests"] / row["total_tests"]) * 100
            
            results[model_name]["hardware"][hardware_name] = {
                "last_tested": row["last_tested"],
                "successful_tests": row["successful_tests"],
                "total_tests": row["total_tests"],
                "success_rate": success_rate
            }
        
        return results
    
    def get_family_benchmarks(self, family: str) -> Dict[str, Dict[str, Any]]:
        """
        Get benchmark results for a model family.
        
        Args:
            family: Model family
            
        Returns:
            Dictionary mapping model names to their benchmark results
        """
        self.cursor.execute(
            "SELECT m.name as model_name, h.name as hardware_name, "
            "MAX(b.timestamp) as last_tested, "
            "AVG(b.latency_ms) as avg_latency, "
            "AVG(b.throughput) as avg_throughput, "
            "AVG(b.memory_mb) as avg_memory, "
            "COUNT(b.id) as total_benchmarks "
            "FROM benchmark_results b "
            "JOIN models m ON b.model_id = m.id "
            "JOIN hardware_platforms h ON b.hardware_id = h.id "
            "WHERE m.family = ? "
            "GROUP BY m.name, h.name "
            "ORDER BY m.name, h.name",
            (family,)
        )
        
        # Organize results by model and hardware
        results = {}
        for row in self.cursor.fetchall():
            model_name = row["model_name"]
            hardware_name = row["hardware_name"]
            
            if model_name not in results:
                results[model_name] = {"hardware": {}}
            
            results[model_name]["hardware"][hardware_name] = {
                "last_tested": row["last_tested"],
                "avg_latency": row["avg_latency"],
                "avg_throughput": row["avg_throughput"],
                "avg_memory": row["avg_memory"],
                "total_benchmarks": row["total_benchmarks"]
            }
        
        return results
    
    def get_hardware_compatibility_matrix(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Get a hardware compatibility matrix showing success rates for all
        model families on all hardware platforms.
        
        Returns:
            Dictionary mapping model families to hardware platforms and their success rates
        """
        self.cursor.execute(
            "SELECT m.family, h.name as hardware_name, "
            "SUM(CASE WHEN f.success THEN 1 ELSE 0 END) as successful_tests, "
            "COUNT(f.id) as total_tests "
            "FROM functionality_tests f "
            "JOIN models m ON f.model_id = m.id "
            "JOIN hardware_platforms h ON f.hardware_id = h.id "
            "GROUP BY m.family, h.name "
            "ORDER BY m.family, h.name"
        )
        
        # Organize results by family and hardware
        matrix = {}
        for row in self.cursor.fetchall():
            family = row["family"]
            hardware_name = row["hardware_name"]
            
            if family not in matrix:
                matrix[family] = {}
            
            success_rate = 0
            if row["total_tests"] > 0:
                success_rate = (row["successful_tests"] / row["total_tests"]) * 100
            
            matrix[family][hardware_name] = {
                "successful_tests": row["successful_tests"],
                "total_tests": row["total_tests"],
                "success_rate": success_rate
            }
        
        return matrix
    
    def get_model_hardware_recommendations(self, model_name: str) -> Dict[str, Any]:
        """
        Get hardware recommendations for a specific model based on
        functionality and benchmark results.
        
        Args:
            model_name: Model name
            
        Returns:
            Dictionary with hardware recommendations
        """
        # Get model family
        self.cursor.execute("SELECT family FROM models WHERE name = ?", (model_name,))
        result = self.cursor.fetchone()
        
        if not result:
            logger.warning(f"Model {model_name} not found in database")
            return {}
        
        family = result["family"]
        
        # Get functionality results
        functionality_results = self.get_model_functionality(model_name)
        
        # Get benchmark results
        benchmark_results = self.get_model_benchmarks(model_name)
        
        # Build recommendation data structure
        recommendation = {
            "model_name": model_name,
            "family": family,
            "hardware_compatibility": {},
            "performance_metrics": {},
            "recommendations": {
                "best_for_reliability": None,
                "best_for_performance": None,
                "best_overall": None
            }
        }
        
        # Process functionality results
        hardware_success_rates = {}
        for result in functionality_results:
            hardware_name = result["hardware_name"]
            
            if hardware_name not in hardware_success_rates:
                hardware_success_rates[hardware_name] = {"success": 0, "total": 0}
            
            hardware_success_rates[hardware_name]["total"] += 1
            if result["success"]:
                hardware_success_rates[hardware_name]["success"] += 1
        
        for hardware_name, counts in hardware_success_rates.items():
            success_rate = 0
            if counts["total"] > 0:
                success_rate = (counts["success"] / counts["total"]) * 100
            
            recommendation["hardware_compatibility"][hardware_name] = {
                "success_rate": success_rate,
                "successful_tests": counts["success"],
                "total_tests": counts["total"]
            }
        
        # Process benchmark results
        hardware_performance = {}
        for result in benchmark_results:
            hardware_name = result["hardware_name"]
            
            if hardware_name not in hardware_performance:
                hardware_performance[hardware_name] = {
                    "latency_ms": [], 
                    "throughput": [], 
                    "memory_mb": []
                }
            
            if result["latency_ms"] is not None:
                hardware_performance[hardware_name]["latency_ms"].append(result["latency_ms"])
            
            if result["throughput"] is not None:
                hardware_performance[hardware_name]["throughput"].append(result["throughput"])
            
            if result["memory_mb"] is not None:
                hardware_performance[hardware_name]["memory_mb"].append(result["memory_mb"])
        
        for hardware_name, metrics in hardware_performance.items():
            avg_latency = None
            avg_throughput = None
            avg_memory = None
            
            if metrics["latency_ms"]:
                avg_latency = sum(metrics["latency_ms"]) / len(metrics["latency_ms"])
            
            if metrics["throughput"]:
                avg_throughput = sum(metrics["throughput"]) / len(metrics["throughput"])
            
            if metrics["memory_mb"]:
                avg_memory = sum(metrics["memory_mb"]) / len(metrics["memory_mb"])
            
            recommendation["performance_metrics"][hardware_name] = {
                "avg_latency_ms": avg_latency,
                "avg_throughput": avg_throughput,
                "avg_memory_mb": avg_memory
            }
        
        # Determine recommendations
        # 1. Best for reliability (highest success rate)
        best_reliability_hw = None
        best_reliability_rate = -1
        
        for hw, data in recommendation["hardware_compatibility"].items():
            if data["success_rate"] > best_reliability_rate and data["total_tests"] > 0:
                best_reliability_rate = data["success_rate"]
                best_reliability_hw = hw
        
        if best_reliability_hw:
            recommendation["recommendations"]["best_for_reliability"] = {
                "hardware": best_reliability_hw,
                "success_rate": best_reliability_rate
            }
        
        # 2. Best for performance (lowest latency with decent throughput)
        best_perf_hw = None
        best_perf_score = float('inf')
        
        for hw, metrics in recommendation["performance_metrics"].items():
            # Skip hardware with no success rate or poor success rate
            if hw not in recommendation["hardware_compatibility"] or \
               recommendation["hardware_compatibility"][hw]["success_rate"] < 50:
                continue
            
            if metrics["avg_latency_ms"] is not None:
                # Lower score is better (lower latency is better)
                perf_score = metrics["avg_latency_ms"]
                
                if perf_score < best_perf_score:
                    best_perf_score = perf_score
                    best_perf_hw = hw
        
        if best_perf_hw:
            recommendation["recommendations"]["best_for_performance"] = {
                "hardware": best_perf_hw,
                "latency_ms": recommendation["performance_metrics"][best_perf_hw]["avg_latency_ms"],
                "throughput": recommendation["performance_metrics"][best_perf_hw]["avg_throughput"]
            }
        
        # 3. Best overall (balanced between reliability and performance)
        best_overall_hw = None
        best_overall_score = -1
        
        for hw in recommendation["hardware_compatibility"].keys():
            if hw not in recommendation["performance_metrics"]:
                continue
            
            # Skip hardware with no metrics
            metrics = recommendation["performance_metrics"][hw]
            if metrics["avg_latency_ms"] is None:
                continue
            
            # Calculate balanced score using both reliability and performance
            # Higher score is better
            reliability = recommendation["hardware_compatibility"][hw]["success_rate"] / 100.0
            
            # Normalize latency to 0-1 range (lower is better), assume max 1000ms latency
            latency_normalized = max(0, 1 - (metrics["avg_latency_ms"] / 1000.0))
            
            # Combined score (70% reliability, 30% performance)
            overall_score = (reliability * 0.7) + (latency_normalized * 0.3)
            
            if overall_score > best_overall_score:
                best_overall_score = overall_score
                best_overall_hw = hw
        
        if best_overall_hw:
            recommendation["recommendations"]["best_overall"] = {
                "hardware": best_overall_hw,
                "success_rate": recommendation["hardware_compatibility"][best_overall_hw]["success_rate"],
                "latency_ms": recommendation["performance_metrics"][best_overall_hw]["avg_latency_ms"],
                "throughput": recommendation["performance_metrics"][best_overall_hw]["avg_throughput"]
            }
        
        return recommendation
    
    def import_functionality_reports(self, reports_dir: str = DEFAULT_FUNCTIONALITY_DIR) -> int:
        """
        Import functionality test results from JSON files in a directory.
        
        Args:
            reports_dir: Directory containing functionality report JSON files
            
        Returns:
            Number of test results imported
        """
        reports_dir = Path(reports_dir)
        if not reports_dir.exists():
            logger.warning(f"Reports directory {reports_dir} does not exist")
            return 0
        
        json_files = list(reports_dir.glob("**/model_functionality_*.json"))
        if not json_files:
            logger.warning(f"No functionality report JSON files found in {reports_dir}")
            return 0
        
        total_imported = 0
        
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if "detailed_results" not in data:
                    logger.warning(f"Invalid functionality report format in {file_path}")
                    continue
                
                for result in data["detailed_results"]:
                    model_name = result.get("model", "unknown")
                    hardware_name = result.get("hardware", "unknown")
                    success = result.get("success", False)
                    error = result.get("error", "")
                    
                    # Determine model family
                    family = "unknown"
                    # Try to infer from model name
                    model_lower = model_name.lower()
                    if "bert" in model_lower or "roberta" in model_lower or "distilbert" in model_lower:
                        family = "embedding"
                    elif "gpt" in model_lower or "llama" in model_lower or "t5" in model_lower:
                        family = "text_generation"
                    elif "vit" in model_lower or "resnet" in model_lower or "convnext" in model_lower:
                        family = "vision"
                    elif "whisper" in model_lower or "wav2vec" in model_lower or "hubert" in model_lower:
                        family = "audio"
                    elif "clip" in model_lower or "llava" in model_lower or "blip" in model_lower:
                        family = "multimodal"
                    
                    # Add to database
                    self.add_functionality_test(
                        model_name=model_name,
                        family=family,
                        hardware_name=hardware_name,
                        success=success,
                        error_message=error,
                        test_details=result
                    )
                    total_imported += 1
                
                logger.info(f"Imported {len(data['detailed_results'])} results from {file_path}")
            
            except Exception as e:
                logger.error(f"Error importing from {file_path}: {e}")
        
        return total_imported
    
    def import_benchmark_reports(self, reports_dir: str = DEFAULT_BENCHMARK_DIR) -> int:
        """
        Import benchmark results from JSON files in a directory.
        
        Args:
            reports_dir: Directory containing benchmark report JSON files
            
        Returns:
            Number of benchmark results imported
        """
        reports_dir = Path(reports_dir)
        if not reports_dir.exists():
            logger.warning(f"Reports directory {reports_dir} does not exist")
            return 0
        
        benchmark_dirs = [d for d in reports_dir.iterdir() if d.is_dir()]
        if not benchmark_dirs:
            logger.warning(f"No benchmark result directories found in {reports_dir}")
            return 0
        
        total_imported = 0
        
        for benchmark_dir in benchmark_dirs:
            json_file = benchmark_dir / "benchmark_results.json"
            if not json_file.exists():
                continue
            
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                if "benchmarks" not in data:
                    logger.warning(f"Invalid benchmark report format in {json_file}")
                    continue
                
                for family, models in data["benchmarks"].items():
                    for model_name, hw_results in models.items():
                        for hw_name, metrics in hw_results.items():
                            if metrics.get("status") != "completed" or "performance_summary" not in metrics:
                                continue
                            
                            perf = metrics["performance_summary"]
                            
                            # Extract key metrics
                            latency = None
                            throughput = None
                            memory = None
                            
                            if "latency" in perf and "mean" in perf["latency"]:
                                latency = perf["latency"]["mean"] * 1000  # convert to ms
                            
                            if "throughput" in perf and "mean" in perf["throughput"]:
                                throughput = perf["throughput"]["mean"]
                            
                            if "memory" in perf and "max_allocated" in perf["memory"]:
                                memory = perf["memory"]["max_allocated"]
                            
                            # Add to database
                            self.add_benchmark_result(
                                model_name=model_name,
                                family=family,
                                hardware_name=hw_name,
                                latency_ms=latency,
                                throughput=throughput,
                                memory_mb=memory,
                                benchmark_details=metrics
                            )
                            total_imported += 1
                
                logger.info(f"Imported benchmark results from {json_file}")
            
            except Exception as e:
                logger.error(f"Error importing from {json_file}: {e}")
        
        return total_imported
    
    def export_data(self, export_dir: str = DEFAULT_EXPORT_DIR) -> str:
        """
        Export all validation data to JSON files.
        
        Args:
            export_dir: Directory to save exported data
            
        Returns:
            Path to export directory
        """
        export_dir = Path(export_dir)
        export_dir.mkdir(exist_ok=True, parents=True)
        
        # Export models
        self.cursor.execute(
            "SELECT * FROM models"
        )
        models = [dict(row) for row in self.cursor.fetchall()]
        
        with open(export_dir / "models.json", 'w') as f:
            json.dump(models, f, indent=2)
        
        # Export hardware platforms
        self.cursor.execute(
            "SELECT * FROM hardware_platforms"
        )
        hardware = [dict(row) for row in self.cursor.fetchall()]
        
        with open(export_dir / "hardware_platforms.json", 'w') as f:
            json.dump(hardware, f, indent=2)
        
        # Export functionality tests
        self.cursor.execute(
            "SELECT f.*, m.name as model_name, m.family, h.name as hardware_name "
            "FROM functionality_tests f "
            "JOIN models m ON f.model_id = m.id "
            "JOIN hardware_platforms h ON f.hardware_id = h.id "
            "ORDER BY f.timestamp DESC"
        )
        functionality = []
        for row in self.cursor.fetchall():
            result = dict(row)
            if result["test_details"]:
                try:
                    result["test_details"] = json.loads(result["test_details"])
                except:
                    # Keep as string if JSON parsing fails
                    pass
            functionality.append(result)
        
        with open(export_dir / "functionality_tests.json", 'w') as f:
            json.dump(functionality, f, indent=2)
        
        # Export benchmark results
        self.cursor.execute(
            "SELECT b.*, m.name as model_name, m.family, h.name as hardware_name "
            "FROM benchmark_results b "
            "JOIN models m ON b.model_id = m.id "
            "JOIN hardware_platforms h ON b.hardware_id = h.id "
            "ORDER BY b.timestamp DESC"
        )
        benchmarks = []
        for row in self.cursor.fetchall():
            result = dict(row)
            if result["benchmark_details"]:
                try:
                    result["benchmark_details"] = json.loads(result["benchmark_details"])
                except:
                    # Keep as string if JSON parsing fails
                    pass
            benchmarks.append(result)
        
        with open(export_dir / "benchmark_results.json", 'w') as f:
            json.dump(benchmarks, f, indent=2)
        
        # Export compatibility matrix
        matrix = self.get_hardware_compatibility_matrix()
        with open(export_dir / "compatibility_matrix.json", 'w') as f:
            json.dump(matrix, f, indent=2)
        
        # Create a summary export
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_models": len(models),
            "total_hardware_platforms": len(hardware),
            "total_functionality_tests": len(functionality),
            "total_benchmark_results": len(benchmarks),
            "model_families": {},
            "hardware_platforms": {},
            "compatibility_matrix": matrix
        }
        
        # Add model family statistics
        for model in models:
            family = model["family"]
            if family not in summary["model_families"]:
                summary["model_families"][family] = 0
            summary["model_families"][family] += 1
        
        # Add hardware platform statistics
        self.cursor.execute(
            "SELECT h.name, "
            "COUNT(DISTINCT f.model_id) as functionality_models, "
            "COUNT(f.id) as functionality_tests, "
            "SUM(CASE WHEN f.success THEN 1 ELSE 0 END) as successful_tests "
            "FROM hardware_platforms h "
            "LEFT JOIN functionality_tests f ON h.id = f.hardware_id "
            "GROUP BY h.name"
        )
        for row in self.cursor.fetchall():
            hw_name = row["name"]
            summary["hardware_platforms"][hw_name] = {
                "functionality_models": row["functionality_models"] or 0,
                "functionality_tests": row["functionality_tests"] or 0,
                "successful_tests": row["successful_tests"] or 0
            }
            
            if row["functionality_tests"]:
                summary["hardware_platforms"][hw_name]["success_rate"] = \
                    (row["successful_tests"] / row["functionality_tests"]) * 100
            else:
                summary["hardware_platforms"][hw_name]["success_rate"] = 0
        
        # Add benchmark statistics
        self.cursor.execute(
            "SELECT h.name, COUNT(DISTINCT b.model_id) as benchmark_models, COUNT(b.id) as benchmark_results "
            "FROM hardware_platforms h "
            "LEFT JOIN benchmark_results b ON h.id = b.hardware_id "
            "GROUP BY h.name"
        )
        for row in self.cursor.fetchall():
            hw_name = row["name"]
            if hw_name in summary["hardware_platforms"]:
                summary["hardware_platforms"][hw_name]["benchmark_models"] = row["benchmark_models"] or 0
                summary["hardware_platforms"][hw_name]["benchmark_results"] = row["benchmark_results"] or 0
        
        with open(export_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Exported validation data to {export_dir}")
        return str(export_dir)
    
    def close(self):
        """Close the database connection"""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            logger.info("Database connection closed")

def main():
    parser = argparse.ArgumentParser(description="Model Validation Tracker")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Path to the database file")
    parser.add_argument("--import-functionality", action="store_true", help="Import functionality test results")
    parser.add_argument("--import-benchmarks", action="store_true", help="Import benchmark results")
    parser.add_argument("--export", action="store_true", help="Export validation data")
    parser.add_argument("--functionality-dir", default=DEFAULT_FUNCTIONALITY_DIR, help="Directory with functionality reports")
    parser.add_argument("--benchmark-dir", default=DEFAULT_BENCHMARK_DIR, help="Directory with benchmark results")
    parser.add_argument("--export-dir", default=DEFAULT_EXPORT_DIR, help="Directory to save exported data")
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = ModelValidationTracker(db_path=args.db_path)
    
    try:
        if args.import_functionality:
            imported = tracker.import_functionality_reports(args.functionality_dir)
            logger.info(f"Imported {imported} functionality test results")
        
        if args.import_benchmarks:
            imported = tracker.import_benchmark_reports(args.benchmark_dir)
            logger.info(f"Imported {imported} benchmark results")
        
        if args.export:
            export_path = tracker.export_data(args.export_dir)
            logger.info(f"Exported validation data to {export_path}")
        
        # If no actions specified, print usage
        if not (args.import_functionality or args.import_benchmarks or args.export):
            parser.print_help()
    
    finally:
        # Close database connection
        tracker.close()

if __name__ == "__main__":
    main()