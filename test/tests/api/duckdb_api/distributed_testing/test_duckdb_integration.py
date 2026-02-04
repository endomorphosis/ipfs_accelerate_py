#!/usr/bin/env python3
"""
Test script for DuckDB integration in the Distributed Testing Framework.

This script tests the integration between worker nodes, coordinator, and the DuckDB database
for efficient test result storage and retrieval. It validates the functionality of:

1. DuckDBResultProcessor for direct database operations
2. WorkerDuckDBIntegration for worker-side result caching and processing
3. CoordinatorDuckDBIntegration for centralized result management
"""

import os
import sys
import json
import uuid
import time
import unittest
import tempfile
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("test_duckdb_integration")

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Conditional imports (these will be properly tested in setUp)
try:
    import duckdb
    from duckdb_result_processor import DuckDBResultProcessor
    from worker_duckdb_integration import WorkerDuckDBIntegration
    from coordinator_duckdb_integration import CoordinatorDuckDBIntegration
    DUCKDB_AVAILABLE = True
except ImportError:
    logger.warning("One or more required modules not available, some tests will be skipped")
    DUCKDB_AVAILABLE = False


class TestDuckDBIntegration(unittest.TestCase):
    """Test the DuckDB integration components for the Distributed Testing Framework."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Skip tests if DuckDB is not available
        if not DUCKDB_AVAILABLE:
            raise unittest.SkipTest("DuckDB or integration modules not available")
            
        # Create a temporary directory for test files
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.db_path = os.path.join(cls.temp_dir.name, "test_db.duckdb")
        
        # Store example result for reuse
        cls.example_result = {
            "test_id": str(uuid.uuid4()),
            "worker_id": "test-worker",
            "model_name": "test-model",
            "hardware_type": "cpu",
            "execution_time": 10.5,
            "success": True,
            "memory_usage": 256.0,
            "test_type": "unit-test"
        }
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Remove temporary directory and files
        cls.temp_dir.cleanup()
    
    def test_01_result_processor_init(self):
        """Test initialization of DuckDBResultProcessor."""
        processor = DuckDBResultProcessor(self.db_path)
        self.assertIsNotNone(processor)
        
        # Check if tables are created
        conn = processor.get_connection()
        try:
            # Check test_results table
            result = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='test_results'
            """).fetchone()
            self.assertIsNotNone(result)
            
            # Check worker_metrics table
            result = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='worker_metrics'
            """).fetchone()
            self.assertIsNotNone(result)
        finally:
            processor.release_connection(conn)
            
        processor.close()
    
    def test_02_store_single_result(self):
        """Test storing a single result in the database."""
        processor = DuckDBResultProcessor(self.db_path)
        
        # Store example result
        result = processor.store_result(self.example_result)
        self.assertTrue(result)
        
        # Query database to verify storage
        stored_result = processor.get_result_by_id(self.example_result["test_id"])
        self.assertIsNotNone(stored_result)
        self.assertEqual(stored_result[1], self.example_result["test_id"])
        self.assertEqual(stored_result[3], self.example_result["model_name"])
        self.assertEqual(stored_result[4], self.example_result["hardware_type"])
        
        processor.close()
    
    def test_03_store_batch_results(self):
        """Test storing multiple results in batch."""
        processor = DuckDBResultProcessor(self.db_path)
        
        # Create batch of results
        batch_results = []
        for i in range(5):
            batch_results.append({
                "test_id": str(uuid.uuid4()),
                "worker_id": "test-worker",
                "model_name": f"test-model-{i}",
                "hardware_type": "cpu",
                "execution_time": 10.5 + i,
                "success": i % 2 == 0,
                "memory_usage": 256.0 + i * 10,
                "test_type": "batch-test"
            })
        
        # Store batch
        success, failed = processor.store_batch_results(batch_results)
        self.assertTrue(success)
        self.assertEqual(len(failed), 0)
        
        # Query database to verify storage
        for result in batch_results:
            stored_result = processor.get_result_by_id(result["test_id"])
            self.assertIsNotNone(stored_result)
            self.assertEqual(stored_result[1], result["test_id"])
            self.assertEqual(stored_result[3], result["model_name"])
            
        processor.close()
    
    def test_04_worker_integration(self):
        """Test WorkerDuckDBIntegration."""
        worker_id = "test-worker-integration"
        integration = WorkerDuckDBIntegration(
            worker_id=worker_id,
            db_path=self.db_path,
            batch_size=2,  # Small batch size for testing
            submit_interval_seconds=1  # Short interval for testing
        )
        
        # Store results
        result1 = {
            "test_id": str(uuid.uuid4()),
            "model_name": "test-model-worker",
            "hardware_type": "cpu",
            "execution_time": 10.5,
            "success": True,
            "memory_usage": 256.0,
            "test_type": "worker-test"
        }
        
        result2 = {
            "test_id": str(uuid.uuid4()),
            "model_name": "test-model-worker",
            "hardware_type": "gpu",
            "execution_time": 5.2,
            "success": True,
            "memory_usage": 512.0,
            "test_type": "worker-test"
        }
        
        # Store single result
        integration.store_result(result1)
        
        # Store second result to trigger batch processing
        integration.store_result(result2)
        
        # Wait for batch processing
        time.sleep(2)
        
        # Check status after processing
        status = integration.get_status()
        self.assertEqual(status["worker_id"], worker_id)
        self.assertEqual(status["queue_size"], 0)
        
        # Query database to verify storage
        processor = DuckDBResultProcessor(self.db_path)
        
        stored_result1 = processor.get_result_by_id(result1["test_id"])
        self.assertIsNotNone(stored_result1)
        self.assertEqual(stored_result1[3], result1["model_name"])
        self.assertEqual(stored_result1[2], worker_id)  # Worker ID should be set
        
        stored_result2 = processor.get_result_by_id(result2["test_id"])
        self.assertIsNotNone(stored_result2)
        self.assertEqual(stored_result2[3], result2["model_name"])
        self.assertEqual(stored_result2[2], worker_id)
        
        processor.close()
        
        # Clean up
        integration.close()
    
    def test_05_coordinator_integration(self):
        """Test CoordinatorDuckDBIntegration."""
        integration = CoordinatorDuckDBIntegration(
            db_path=self.db_path,
            batch_size=2,  # Small batch size for testing
            auto_vacuum_interval_hours=0,  # Disable auto-vacuum for testing
            enable_dashboard=False  # Disable dashboard for testing
        )
        
        # Store results
        result1 = {
            "test_id": str(uuid.uuid4()),
            "worker_id": "test-worker-1",
            "model_name": "test-model-coordinator",
            "hardware_type": "cpu",
            "execution_time": 10.5,
            "success": True,
            "memory_usage": 256.0,
            "test_type": "coordinator-test"
        }
        
        result2 = {
            "test_id": str(uuid.uuid4()),
            "worker_id": "test-worker-2",
            "model_name": "test-model-coordinator",
            "hardware_type": "gpu",
            "execution_time": 5.2,
            "success": True,
            "memory_usage": 512.0,
            "test_type": "coordinator-test"
        }
        
        # Store single result (should be cached)
        integration.store_result(result1)
        
        # Store second result to trigger batch processing
        integration.store_result(result2)
        
        # Force cache flush
        integration._flush_cache()
        
        # Query results
        stored_result1 = integration.get_result(result1["test_id"])
        self.assertIsNotNone(stored_result1)
        self.assertEqual(stored_result1[1], result1["test_id"])
        
        stored_result2 = integration.get_result(result2["test_id"])
        self.assertIsNotNone(stored_result2)
        self.assertEqual(stored_result2[1], result2["test_id"])
        
        # Get results by model
        model_results = integration.get_results_by_model("test-model-coordinator")
        self.assertEqual(len(model_results), 2)
        
        # Get summary
        summary = integration.get_summary_by_model_hardware()
        self.assertGreater(len(summary), 0)
        
        # Test report generation (just make sure it doesn't crash)
        markdown_report = integration.generate_report(format='markdown')
        self.assertIsNotNone(markdown_report)
        self.assertIn("Distributed Testing Framework Report", markdown_report)
        
        # Clean up
        integration.close()
    
    def test_06_end_to_end_integration(self):
        """Test end-to-end integration between worker and coordinator."""
        # Create coordinator
        coordinator = CoordinatorDuckDBIntegration(
            db_path=self.db_path,
            batch_size=3,
            auto_vacuum_interval_hours=0,
            enable_dashboard=False
        )
        
        # Create workers
        worker1 = WorkerDuckDBIntegration(
            worker_id="e2e-worker-1",
            db_path=self.db_path,
            batch_size=2,
            submit_interval_seconds=1
        )
        
        worker2 = WorkerDuckDBIntegration(
            worker_id="e2e-worker-2",
            db_path=self.db_path,
            batch_size=2,
            submit_interval_seconds=1
        )
        
        # Generate results for each worker
        for i in range(3):
            # Worker 1 results
            worker1.store_result({
                "test_id": str(uuid.uuid4()),
                "model_name": "bert-base-uncased",
                "hardware_type": "cpu",
                "execution_time": 10.5 + i,
                "success": True,
                "memory_usage": 256.0 + i * 10,
                "test_type": "e2e-test"
            })
            
            # Worker 2 results
            worker2.store_result({
                "test_id": str(uuid.uuid4()),
                "model_name": "vit-base-patch16-224",
                "hardware_type": "gpu",
                "execution_time": 5.2 + i,
                "success": i < 2,  # Last one fails
                "memory_usage": 512.0 + i * 20,
                "test_type": "e2e-test"
            })
        
        # Wait for processing
        time.sleep(2)
        
        # Verify coordinator sees all results
        bert_results = coordinator.get_results_by_model("bert-base-uncased")
        self.assertEqual(len(bert_results), 3)
        
        vit_results = coordinator.get_results_by_model("vit-base-patch16-224")
        self.assertEqual(len(vit_results), 3)
        
        # Check worker-specific results
        worker1_results = coordinator.get_results_by_worker("e2e-worker-1")
        self.assertEqual(len(worker1_results), 3)
        
        worker2_results = coordinator.get_results_by_worker("e2e-worker-2")
        self.assertEqual(len(worker2_results), 3)
        
        # Generate a report
        report = coordinator.generate_report(format='html')
        self.assertIsNotNone(report)
        self.assertIn("html", report.lower())
        self.assertIn("bert-base-uncased", report)
        self.assertIn("vit-base-patch16-224", report)
        
        # Clean up
        worker1.close()
        worker2.close()
        coordinator.close()


if __name__ == "__main__":
    unittest.main()