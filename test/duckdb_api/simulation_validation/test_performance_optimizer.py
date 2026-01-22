#!/usr/bin/env python3
"""
Simple test script for the DBPerformanceOptimizer's get_performance_metrics function.
"""

import os
import sys
import time
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_db_performance_optimizer")

# Add import directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Define test class
class MockDBAPI:
    def _get_connection(self):
        conn = MagicMock()
        # Mock execute responses
        conn.execute().fetchall.return_value = ["SCAN TABLE validation_results USING INDEX"]
        conn.execute().fetchone.return_value = [10]
        return conn

class MockDBIntegration:
    def __init__(self, db_path=None):
        self.db_path = db_path
        self.db_api = MockDBAPI()

class TestPerformanceOptimizer:
    def __init__(self):
        # Mock the database integration
        self.db_integration = MockDBIntegration(db_path="/tmp/test.duckdb")
        
        # Load the actual implementation
        from duckdb_api.simulation_validation.db_performance_optimizer import DBPerformanceOptimizer
        
        # Initialize the optimizer with our mocks
        with patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=1024000):
            self.optimizer = DBPerformanceOptimizer(
                db_path="/tmp/test.duckdb",
                enable_caching=True,
                cache_size=20,
                cache_ttl=60,
                batch_size=10
            )
        
        # Override the db_integration
        self.optimizer.db_integration = self.db_integration
    
    def test_get_performance_metrics(self):
        # Mock the individual metric methods
        with patch.object(self.optimizer, '_get_query_time_metrics', return_value={
                "value": 123.4,
                "status": "good",
                "unit": "ms",
                "history": [100, 150, 200],
                "timestamp": "2025-03-14T12:00:00"
            }), patch.object(self.optimizer, '_get_storage_metrics', return_value={
                "value": 1024000,
                "status": "good",
                "unit": "bytes",
                "history": [900000, 950000, 1024000],
                "timestamp": "2025-03-14T12:00:00"
            }), patch.object(self.optimizer, '_get_index_efficiency_metrics', return_value={
                "value": 95,
                "status": "good",
                "unit": "percent",
                "timestamp": "2025-03-14T12:00:00"
            }), patch.object(self.optimizer, '_get_read_efficiency_metrics', return_value={
                "value": 500,
                "status": "good",
                "unit": "records/second",
                "timestamp": "2025-03-14T12:00:00"
            }), patch.object(self.optimizer, '_get_write_efficiency_metrics', return_value={
                "value": 300,
                "status": "good",
                "unit": "records/second",
                "timestamp": "2025-03-14T12:00:00"
            }), patch.object(self.optimizer, '_get_vacuum_status_metrics', return_value={
                "value": 95,
                "status": "good",
                "unit": "percent",
                "timestamp": "2025-03-14T12:00:00"
            }), patch.object(self.optimizer, '_get_compression_metrics', return_value={
                "value": 2.5,
                "status": "good",
                "unit": "ratio",
                "timestamp": "2025-03-14T12:00:00"
            }), patch.object(self.optimizer, '_get_cache_performance_metrics', return_value={
                "value": 75,
                "status": "good",
                "unit": "percent",
                "timestamp": "2025-03-14T12:00:00"
            }):
                
            # Call the method being tested
            metrics = self.optimizer.get_performance_metrics()
            
            # Check that we got the metrics
            if not isinstance(metrics, dict):
                print("❌ FAIL: Metrics should be a dictionary")
                return False
            
            if len(metrics) == 0:
                print("❌ FAIL: Metrics should not be empty")
                return False
            
            # Check for expected metrics
            expected_metrics = [
                "query_time",
                "storage_size",
                "index_efficiency",
                "read_efficiency",
                "write_efficiency",
                "vacuum_status",
                "compression_ratio"
            ]
            
            for metric in expected_metrics:
                if metric not in metrics:
                    print(f"❌ FAIL: Missing metric {metric}")
                    return False
                
                # Check metric structure
                metric_data = metrics[metric]
                if "value" not in metric_data:
                    print(f"❌ FAIL: Metric {metric} should have a value")
                    return False
                
                if "status" not in metric_data:
                    print(f"❌ FAIL: Metric {metric} should have a status")
                    return False
                
                if "unit" not in metric_data:
                    print(f"❌ FAIL: Metric {metric} should have a unit")
                    return False
            
            print("✅ SUCCESS: get_performance_metrics works as expected")
            return True
    
    def test_get_overall_status(self):
        # Test different status combinations
        
        # 1. All good metrics
        with patch.object(self.optimizer, 'get_performance_metrics', return_value={
            "query_time": {"status": "good"},
            "storage_size": {"status": "good"},
            "index_efficiency": {"status": "good"},
            "read_efficiency": {"status": "good"}
        }):
            status = self.optimizer.get_overall_status()
            if status != "good":
                print(f"❌ FAIL: Status should be 'good' when all metrics are good, got '{status}'")
                return False
        
        # 2. One warning metric
        with patch.object(self.optimizer, 'get_performance_metrics', return_value={
            "query_time": {"status": "good"},
            "storage_size": {"status": "warning"},
            "index_efficiency": {"status": "good"},
            "read_efficiency": {"status": "good"}
        }):
            status = self.optimizer.get_overall_status()
            if status != "warning":
                print(f"❌ FAIL: Status should be 'warning' when one metric has warning status, got '{status}'")
                return False
        
        # 3. One error metric
        with patch.object(self.optimizer, 'get_performance_metrics', return_value={
            "query_time": {"status": "good"},
            "storage_size": {"status": "good"},
            "index_efficiency": {"status": "error"},
            "read_efficiency": {"status": "good"}
        }):
            status = self.optimizer.get_overall_status()
            if status != "error":
                print(f"❌ FAIL: Status should be 'error' when one metric has error status, got '{status}'")
                return False
        
        print("✅ SUCCESS: get_overall_status works as expected")
        return True

def main():
    """Run tests for the DBPerformanceOptimizer."""
    logger.info("Testing DBPerformanceOptimizer implementation")
    
    try:
        tester = TestPerformanceOptimizer()
        metrics_result = tester.test_get_performance_metrics()
        status_result = tester.test_get_overall_status()
        
        if metrics_result and status_result:
            logger.info("All tests passed! The implementation is working correctly.")
            return 0
        else:
            logger.error("Some tests failed. See above for details.")
            return 1
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())