#!/usr/bin/env python3
"""
Distributed Testing Framework - Worker DuckDB Integration

This module provides integration between worker nodes and the DuckDB database system,
allowing workers to store test results and metrics directly in the database without
going through the coordinator for certain operations. It improves efficiency and
reduces network traffic for result storage.

Core features:
- Direct worker-to-database result storage
- Batched result submission for efficiency
- Automatic retry with exponential backoff
- Local caching for offline operation
- Built-in metrics collection and storage
"""

import os
import sys
import json
import time
import uuid
import logging
import threading
import queue
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("worker_duckdb_integration")

# Conditional import for duckdb
try:
    from duckdb_result_processor import DuckDBResultProcessor
    DUCKDB_AVAILABLE = True
except ImportError:
    logger.warning("DuckDBResultProcessor not available. Using fallback storage.")
    DUCKDB_AVAILABLE = False

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


class WorkerDuckDBIntegration:
    """Provides direct DuckDB integration for worker nodes."""
    
    def __init__(self, worker_id, db_path=None, cache_dir=None, max_retries=3, 
                 batch_size=10, submit_interval_seconds=60, enable_metrics=True):
        """Initialize the worker DuckDB integration.
        
        Args:
            worker_id: Unique ID of the worker
            db_path: Path to the DuckDB database file (if None, will use coordinator for results)
            cache_dir: Directory for local caching (defaults to ./cache)
            max_retries: Maximum number of retries for database operations
            batch_size: Maximum number of results to batch before submission
            submit_interval_seconds: Interval for automatic batch submission
            enable_metrics: Whether to collect and store worker metrics
        """
        self.worker_id = worker_id
        self.db_path = db_path
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "cache")
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.submit_interval = submit_interval_seconds
        self.enable_metrics = enable_metrics
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Set up result queue and local cache
        self.result_queue = queue.Queue()
        self.local_cache_path = os.path.join(self.cache_dir, f"worker_{worker_id}_cache.db")
        self.local_cache = self._initialize_local_cache()
        
        # Initialize DuckDB processor if available
        self.db_processor = None
        if db_path and DUCKDB_AVAILABLE:
            try:
                self.db_processor = DuckDBResultProcessor(db_path)
                logger.info(f"Connected to DuckDB database at {db_path}")
            except Exception as e:
                logger.error(f"Error connecting to DuckDB: {e}")
                self.db_processor = None
        
        # Set up background thread for result processing
        self.processing_thread = None
        self.stop_event = threading.Event()
        self.start_processing_thread()
        
        # Set up metrics collection if enabled
        self.metrics_thread = None
        if enable_metrics:
            self.metrics_thread = threading.Thread(target=self._metrics_collection_loop)
            self.metrics_thread.daemon = True
            self.metrics_thread.start()
        
        logger.info(f"Worker DuckDB Integration initialized for worker {worker_id}")
    
    def _initialize_local_cache(self):
        """Initialize the local SQLite cache for results."""
        try:
            conn = sqlite3.connect(self.local_cache_path)
            cursor = conn.cursor()
            
            # Create tables if they don't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cached_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    result_data TEXT,
                    timestamp TEXT,
                    submitted BOOLEAN DEFAULT 0
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cached_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metrics_data TEXT,
                    timestamp TEXT,
                    submitted BOOLEAN DEFAULT 0
                )
            """)
            
            conn.commit()
            logger.info(f"Initialized local cache at {self.local_cache_path}")
            return conn
        except Exception as e:
            logger.error(f"Error initializing local cache: {e}")
            return None
    
    def start_processing_thread(self):
        """Start the background thread for result processing."""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.stop_event.clear()
            self.processing_thread = threading.Thread(target=self._result_processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            logger.info("Started result processing thread")
    
    def stop_processing_thread(self):
        """Stop the background thread for result processing."""
        if self.processing_thread and self.processing_thread.is_alive():
            self.stop_event.set()
            self.processing_thread.join(timeout=5)
            logger.info("Stopped result processing thread")
    
    def _result_processing_loop(self):
        """Background loop for processing results from the queue."""
        while not self.stop_event.is_set():
            try:
                # Try to submit cached results first
                self._submit_cached_results()
                
                # Process queued results
                batch = []
                
                # Try to get up to batch_size results without blocking
                for _ in range(self.batch_size):
                    try:
                        result = self.result_queue.get(block=False)
                        batch.append(result)
                        self.result_queue.task_done()
                    except queue.Empty:
                        break
                
                # Process the batch if any results were collected
                if batch:
                    self._process_result_batch(batch)
                
                # Wait before next check
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in result processing loop: {e}")
                time.sleep(5)  # Wait longer on error
    
    def _process_result_batch(self, batch):
        """Process a batch of results.
        
        Args:
            batch: List of result dictionaries to process
        """
        if not batch:
            return
            
        if self.db_processor:
            # Direct submission to DuckDB
            try:
                success, failed = self.db_processor.store_batch_results(batch)
                
                # Cache any failed results
                if failed:
                    self._cache_results(failed)
                    
                logger.debug(f"Processed batch of {len(batch)} results, {len(failed)} failed")
            except Exception as e:
                logger.error(f"Error submitting to DuckDB: {e}")
                # Cache all results on error
                self._cache_results(batch)
        else:
            # No DuckDB connection, cache all results
            self._cache_results(batch)
            
    def _cache_results(self, results):
        """Cache results in the local SQLite database.
        
        Args:
            results: List of result dictionaries to cache
        """
        if not self.local_cache:
            logger.error("Local cache not available")
            return
            
        try:
            cursor = self.local_cache.cursor()
            
            for result in results:
                cursor.execute(
                    "INSERT INTO cached_results (result_data, timestamp, submitted) VALUES (?, ?, ?)",
                    (json.dumps(result), datetime.now().isoformat(), False)
                )
                
            self.local_cache.commit()
            logger.debug(f"Cached {len(results)} results")
        except Exception as e:
            logger.error(f"Error caching results: {e}")
    
    def _submit_cached_results(self):
        """Try to submit cached results to the database."""
        if not self.db_processor or not self.local_cache:
            return
            
        try:
            cursor = self.local_cache.cursor()
            
            # Get unsubmitted results (limit to batch_size)
            cursor.execute(
                "SELECT id, result_data FROM cached_results WHERE submitted = 0 LIMIT ?",
                (self.batch_size,)
            )
            
            cached_results = cursor.fetchall()
            
            if not cached_results:
                return
                
            batch = []
            ids = []
            
            for result_id, result_data in cached_results:
                try:
                    result = json.loads(result_data)
                    batch.append(result)
                    ids.append(result_id)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in cached result {result_id}")
                    # Mark as submitted to prevent retrying invalid data
                    cursor.execute(
                        "UPDATE cached_results SET submitted = 1 WHERE id = ?",
                        (result_id,)
                    )
            
            if batch:
                success, failed = self.db_processor.store_batch_results(batch)
                
                if success and not failed:
                    # Mark all as submitted
                    placeholders = ",".join("?" for _ in ids)
                    cursor.execute(
                        f"UPDATE cached_results SET submitted = 1 WHERE id IN ({placeholders})",
                        ids
                    )
                elif failed:
                    # Mark only successful ones as submitted
                    failed_ids = set()
                    for fail_result in failed:
                        for i, result in enumerate(batch):
                            if result.get("test_id") == fail_result.get("test_id"):
                                failed_ids.add(ids[i])
                                break
                    
                    success_ids = [id for id in ids if id not in failed_ids]
                    if success_ids:
                        placeholders = ",".join("?" for _ in success_ids)
                        cursor.execute(
                            f"UPDATE cached_results SET submitted = 1 WHERE id IN ({placeholders})",
                            success_ids
                        )
                
                self.local_cache.commit()
                logger.debug(f"Processed {len(batch)} cached results, {len(failed) if failed else 0} failed")
        except Exception as e:
            logger.error(f"Error submitting cached results: {e}")
    
    def _metrics_collection_loop(self):
        """Background loop for collecting and storing worker metrics."""
        while not self.stop_event.is_set():
            try:
                if self.db_processor:
                    metrics = self._collect_metrics()
                    if metrics:
                        self.db_processor.store_worker_metrics(self.worker_id, metrics)
                        logger.debug("Stored worker metrics")
                
                # Wait before next collection
                time.sleep(self.submit_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                time.sleep(5)  # Wait longer on error
    
    def _collect_metrics(self):
        """Collect worker metrics.
        
        Returns:
            Dict: Collected metrics
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "gpu_usage": 0.0,
            "gpu_memory_usage": 0.0,
            "temperature": 0.0,
            "power_usage": 0.0,
            "active_tasks": self.result_queue.qsize(),
            "details": {}
        }
        
        # Try to get CPU and memory usage with psutil
        try:
            import psutil
            process = psutil.Process(os.getpid())
            
            # CPU usage
            metrics["cpu_usage"] = process.cpu_percent(interval=0.1) / 100.0
            
            # Memory usage (GB)
            memory_info = process.memory_info()
            metrics["memory_usage"] = memory_info.rss / (1024 ** 3)
            
            # System-wide CPU and memory usage
            metrics["details"]["system_cpu"] = psutil.cpu_percent() / 100.0
            metrics["details"]["system_memory"] = psutil.virtual_memory().percent / 100.0
            
        except ImportError:
            pass
        
        # Try to get GPU usage with pynvml
        try:
            import pynvml
            pynvml.nvmlInit()
            
            # Get first GPU
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # GPU utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            metrics["gpu_usage"] = utilization.gpu / 100.0
            
            # GPU memory usage
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            metrics["gpu_memory_usage"] = memory_info.used / memory_info.total
            
            # GPU temperature
            metrics["temperature"] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # GPU power usage (W)
            metrics["power_usage"] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            
            pynvml.nvmlShutdown()
            
        except (ImportError, Exception):
            pass
        
        return metrics
    
    def store_result(self, result):
        """Store a test result.
        
        Args:
            result: Dictionary containing test result data
            
        Returns:
            bool: True if the result was queued successfully
        """
        # Ensure required fields
        if "test_id" not in result:
            result["test_id"] = str(uuid.uuid4())
            
        if "worker_id" not in result:
            result["worker_id"] = self.worker_id
            
        if "timestamp" not in result:
            result["timestamp"] = datetime.now().isoformat()
        
        # Add to queue for batch processing
        try:
            self.result_queue.put(result)
            logger.debug(f"Queued result {result['test_id']} for processing")
            return True
        except Exception as e:
            logger.error(f"Error queuing result: {e}")
            return False
    
    def store_batch_results(self, results):
        """Store multiple test results.
        
        Args:
            results: List of result dictionaries to store
            
        Returns:
            int: Number of results queued successfully
        """
        count = 0
        for result in results:
            if self.store_result(result):
                count += 1
        return count
    
    def flush(self, timeout=30):
        """Wait for all queued results to be processed.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if all results were processed, False if timeout occurred
        """
        try:
            self.result_queue.join()
            return True
        except Exception:
            return False
    
    def close(self):
        """Close the integration and free resources."""
        # Stop background threads
        self.stop_event.set()
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
            
        if self.metrics_thread:
            self.metrics_thread.join(timeout=5)
        
        # Close database connections
        if self.db_processor:
            self.db_processor.close()
            
        if self.local_cache:
            self.local_cache.close()
            
        logger.info("Closed worker DuckDB integration")
    
    def get_queue_size(self):
        """Get the current size of the result queue.
        
        Returns:
            int: Number of results in the queue
        """
        return self.result_queue.qsize()
    
    def get_cached_count(self):
        """Get the number of results in the local cache.
        
        Returns:
            int: Number of cached results
        """
        if not self.local_cache:
            return 0
            
        try:
            cursor = self.local_cache.cursor()
            cursor.execute("SELECT COUNT(*) FROM cached_results WHERE submitted = 0")
            return cursor.fetchone()[0]
        except Exception:
            return 0
    
    def get_status(self):
        """Get the status of the integration.
        
        Returns:
            Dict: Status information
        """
        return {
            "worker_id": self.worker_id,
            "queue_size": self.get_queue_size(),
            "cached_count": self.get_cached_count(),
            "db_connected": self.db_processor is not None,
            "processing_active": self.processing_thread is not None and self.processing_thread.is_alive(),
            "metrics_active": self.metrics_thread is not None and self.metrics_thread.is_alive(),
            "cache_path": self.local_cache_path
        }


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Worker DuckDB Integration")
    parser.add_argument("--worker-id", default=str(uuid.uuid4())[:8], help="Worker ID")
    parser.add_argument("--db-path", help="Path to DuckDB database")
    parser.add_argument("--cache-dir", help="Directory for local caching")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for result submission")
    parser.add_argument("--interval", type=int, default=60, help="Interval for automatic batch submission")
    parser.add_argument("--disable-metrics", action="store_true", help="Disable metrics collection")
    parser.add_argument("--test", action="store_true", help="Run a test")
    
    args = parser.parse_args()
    
    integration = WorkerDuckDBIntegration(
        worker_id=args.worker_id,
        db_path=args.db_path,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        submit_interval_seconds=args.interval,
        enable_metrics=not args.disable_metrics
    )
    
    if args.test:
        logger.info("Running test")
        
        # Store a single test result
        test_result = {
            "test_id": str(uuid.uuid4()),
            "model_name": "test-model",
            "hardware_type": "cpu",
            "execution_time": 10.5,
            "success": True,
            "memory_usage": 256.0,
            "test_type": "unit-test"
        }
        
        integration.store_result(test_result)
        
        # Store batch results
        batch_results = []
        for i in range(5):
            batch_results.append({
                "test_id": str(uuid.uuid4()),
                "model_name": f"test-model-{i}",
                "hardware_type": "cpu",
                "execution_time": 10.5 + i,
                "success": i % 2 == 0,
                "memory_usage": 256.0 + i * 10,
                "test_type": "batch-test"
            })
        
        count = integration.store_batch_results(batch_results)
        logger.info(f"Queued {count} batch results")
        
        # Wait for results to be processed
        logger.info("Waiting for results to be processed...")
        integration.flush(timeout=10)
        
        # Get status
        status = integration.get_status()
        logger.info(f"Status: {status}")
        
        # Close integration
        integration.close()