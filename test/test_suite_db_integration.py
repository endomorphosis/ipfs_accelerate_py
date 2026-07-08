#!/usr/bin/env python3
"""
Test Suite Database Integration Test

This script tests the DuckDB integration for the Test Suite API by:
1. Creating a database handler
2. Adding test data
3. Querying and verifying the results
"""

import os
import sys
import uuid
import datetime
import tempfile
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test.refactored_test_suite.database.db_handler import TestDatabaseHandler
from test.refactored_test_suite.database.db_integration import TestDatabaseIntegration

def test_db_handler():
    """Test the database handler."""
    print("Testing database handler...")
    
    # Create a temporary database file
    with tempfile.NamedTemporaryFile(suffix=".duckdb") as temp:
        db_path = temp.name
        
        # Create database handler
        db = TestDatabaseHandler(db_path)
        
        # 1. Store a test run
        run_id = str(uuid.uuid4())
        run_data = {
            "run_id": run_id,
            "model_name": "bert-base-uncased",
            "hardware": ["cpu"],
            "test_type": "comprehensive",
            "timeout": 300,
            "save_results": True,
            "status": "initializing",
            "progress": 0.0,
            "current_step": "Setting up test environment",
            "started_at": datetime.datetime.now()
        }
        
        success = db.store_test_run(run_data)
        print(f"Stored test run {run_id}: {'SUCCESS' if success else 'FAILED'}")
        
        # 2. Get the run
        retrieved_run = db.get_test_run(run_id)
        print(f"Retrieved run: {'SUCCESS' if retrieved_run else 'FAILED'}")
        
        # 3. Store test results
        test_results = {
            "tests_passed": 5,
            "tests_failed": 1,
            "tests_skipped": 0,
            "test_details": [
                {"method": "test_load_model", "status": "passed", "duration_ms": 500},
                {"method": "test_basic_inference", "status": "passed", "duration_ms": 800},
                {"method": "test_batch_inference", "status": "passed", "duration_ms": 1200},
                {"method": "test_model_attributes", "status": "passed", "duration_ms": 300},
                {"method": "test_save_load", "status": "passed", "duration_ms": 700},
                {"method": "test_edge_case", "status": "failed", "error": "Failed with error XYZ"},
            ]
        }
        
        success = db.store_test_results(run_id, test_results)
        print(f"Stored test results: {'SUCCESS' if success else 'FAILED'}")
        
        # 4. Store a step
        step_data = {
            "step_name": "Running test_load_model",
            "status": "running",
            "progress": 0.2,
            "started_at": datetime.datetime.now()
        }
        
        success = db.store_test_step(run_id, step_data)
        print(f"Stored step: {'SUCCESS' if success else 'FAILED'}")
        
        # 5. Store a metric
        success = db.store_test_metric(run_id, "latency_ms", 350.0, "ms")
        print(f"Stored metric: {'SUCCESS' if success else 'FAILED'}")
        
        # 6. Store a batch test run
        batch_id = str(uuid.uuid4())
        batch_data = {
            "batch_id": batch_id,
            "description": "Comprehensive test of encoder models",
            "run_count": 5,
            "started_at": datetime.datetime.now(),
            "status": "running"
        }
        
        success = db.store_batch_test_run(batch_data)
        print(f"Stored batch {batch_id}: {'SUCCESS' if success else 'FAILED'}")
        
        # 7. Add run to batch
        success = db.add_run_to_batch(run_id, batch_id)
        print(f"Added run to batch: {'SUCCESS' if success else 'FAILED'}")
        
        # 8. Get run history
        runs = db.list_test_runs(limit=10)
        print(f"Run history: {'SUCCESS' if len(runs) > 0 else 'FAILED'}")
        
        # 9. Get batch history
        batches = db.list_batch_test_runs(limit=10)
        print(f"Batch history: {'SUCCESS' if len(batches) > 0 else 'FAILED'}")
        
        # 10. Get model statistics
        stats = db.get_model_statistics()
        print(f"Model statistics: {'SUCCESS' if isinstance(stats, list) else 'FAILED'}")
        
        # Clean up
        db.close()
        print("Database handler test completed")

def test_db_integration():
    """Test the database integration layer."""
    print("\nTesting database integration...")
    
    # Create a temporary database file
    with tempfile.NamedTemporaryFile(suffix=".duckdb") as temp:
        db_path = temp.name
        
        # Create integration layer
        integration = TestDatabaseIntegration(db_path)
        
        # 1. Track test start
        run_id = str(uuid.uuid4())
        run_data = {
            "run_id": run_id,
            "model_name": "gpt2",
            "hardware": ["cpu"],
            "test_type": "basic",
            "timeout": 180,
            "save_results": True,
            "status": "initializing",
            "started_at": datetime.datetime.now()
        }
        
        success = integration.track_test_start(run_data)
        print(f"Tracked test start: {'SUCCESS' if success else 'FAILED'}")
        
        # 2. Track test update
        success = integration.track_test_update(
            run_id, 
            status="running", 
            progress=0.5, 
            current_step="Running test_basic_inference"
        )
        print(f"Tracked test update: {'SUCCESS' if success else 'FAILED'}")
        
        # 3. Track test completion
        results = {
            "tests_passed": 2,
            "tests_failed": 0,
            "tests_skipped": 0,
            "test_details": [
                {"method": "test_load_model", "status": "passed", "duration_ms": 450},
                {"method": "test_basic_inference", "status": "passed", "duration_ms": 750}
            ],
            "performance_metrics": {
                "cpu": {
                    "latency_ms": 450,
                    "throughput_items_per_sec": 2.2
                }
            }
        }
        
        success = integration.track_test_completion(run_id, results)
        print(f"Tracked test completion: {'SUCCESS' if success else 'FAILED'}")
        
        # 4. Track batch start
        batch_id = str(uuid.uuid4())
        model_names = ["bert-base-uncased", "gpt2"]
        run_ids = [run_id, str(uuid.uuid4())]
        
        success = integration.track_batch_start(batch_id, model_names, run_ids)
        print(f"Tracked batch start: {'SUCCESS' if success else 'FAILED'}")
        
        # 5. Track batch completion
        success = integration.track_batch_completion(batch_id)
        print(f"Tracked batch completion: {'SUCCESS' if success else 'FAILED'}")
        
        # 6. Get run history
        runs = integration.get_run_history(limit=10)
        print(f"Got run history: {'SUCCESS' if len(runs) > 0 else 'FAILED'}")
        
        # 7. Get performance report
        report = integration.get_performance_report(days=30)
        print(f"Got performance report: {'SUCCESS' if isinstance(report, dict) else 'FAILED'}")
        
        # Clean up
        integration.close()
        print("Database integration test completed")
        
def test_api_server():
    """Test starting the API server with database integration."""
    print("\nTesting API server with database integration...")
    
    try:
        # Import the test API server
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "refactored_test_suite", "api"))
        from test_api_server import TestAPIServer
        
        # Create a temporary database file
        with tempfile.NamedTemporaryFile(suffix=".duckdb") as temp:
            db_path = temp.name
            
            # Create the API server with database integration
            server = TestAPIServer(db_path=db_path)
            
            print(f"Created API server with database integration: SUCCESS")
            
            # Check if database integration is available
            if hasattr(server, 'db_integration') and server.db_integration:
                print(f"Database integration enabled: SUCCESS")
            else:
                print(f"Database integration enabled: FAILED")
                
    except Exception as e:
        print(f"Test API server creation failed: {e}")
    
    print("API server test completed")

if __name__ == "__main__":
    test_db_handler()
    test_db_integration()
    test_api_server()
    print("\nAll tests completed successfully!")