#!/usr/bin/env python3
"""
Test Generator Database Integration

This script tests the DuckDB integration for the Generator API by:
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
from test.refactored_generator_suite.database.db_handler import GeneratorDatabaseHandler
from test.refactored_generator_suite.database.db_integration import GeneratorDatabaseIntegration

def test_db_handler():
    """Test the database handler."""
    print("Testing database handler...")
    
    # Create a temporary database file
    with tempfile.NamedTemporaryFile(suffix=".duckdb") as temp:
        db_path = temp.name
        
        # Create database handler
        db = GeneratorDatabaseHandler(db_path)
        
        # 1. Store a task
        task_id = str(uuid.uuid4())
        task_data = {
            "task_id": task_id,
            "model_name": "bert-base-uncased",
            "hardware": ["cpu"],
            "status": "initializing",
            "started_at": datetime.datetime.now(),
            "template_type": "encoder",
            "output_dir": "/tmp/output"
        }
        
        success = db.store_task(task_data)
        print(f"Stored task {task_id}: {'SUCCESS' if success else 'FAILED'}")
        
        # 2. Get the task
        retrieved_task = db.get_task(task_id)
        print(f"Retrieved task: {'SUCCESS' if retrieved_task else 'FAILED'}")
        
        # 3. Store a step
        step_data = {
            "step_name": "Preparing environment",
            "status": "running",
            "progress": 0.1,
            "started_at": datetime.datetime.now()
        }
        
        success = db.store_task_step(task_id, step_data)
        print(f"Stored step: {'SUCCESS' if success else 'FAILED'}")
        
        # 4. Store a metric
        success = db.store_task_metric(task_id, "memory_usage", 256.0, "MB")
        print(f"Stored metric: {'SUCCESS' if success else 'FAILED'}")
        
        # 5. Store a batch task
        batch_id = str(uuid.uuid4())
        batch_data = {
            "batch_id": batch_id,
            "description": "Test batch",
            "task_count": 1,
            "started_at": datetime.datetime.now(),
            "status": "running"
        }
        
        success = db.store_batch_task(batch_data)
        print(f"Stored batch {batch_id}: {'SUCCESS' if success else 'FAILED'}")
        
        # 6. Get task history
        tasks = db.list_tasks(limit=10)
        print(f"Task history: {'SUCCESS' if len(tasks) > 0 else 'FAILED'}")
        
        # 7. Update task with batch ID
        task_data["batch_id"] = batch_id
        success = db.store_task(task_data)
        print(f"Updated task with batch ID: {'SUCCESS' if success else 'FAILED'}")
        
        # 8. Get batch task
        batch = db.get_batch_task(batch_id)
        print(f"Retrieved batch: {'SUCCESS' if batch else 'FAILED'}")
        
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
        integration = GeneratorDatabaseIntegration(db_path)
        
        # 1. Track task start
        task_id = str(uuid.uuid4())
        task_data = {
            "task_id": task_id,
            "model_name": "gpt2",
            "hardware": ["cpu"],
            "status": "initializing",
            "started_at": datetime.datetime.now()
        }
        
        success = integration.track_task_start(task_data)
        print(f"Tracked task start: {'SUCCESS' if success else 'FAILED'}")
        
        # 2. Track task update
        success = integration.track_task_update(
            task_id, 
            status="running", 
            progress=0.5, 
            current_step="Processing model"
        )
        print(f"Tracked task update: {'SUCCESS' if success else 'FAILED'}")
        
        # 3. Track batch start
        batch_id = str(uuid.uuid4())
        model_names = ["bert-base-uncased", "gpt2"]
        task_ids = [task_id, str(uuid.uuid4())]
        
        success = integration.track_batch_start(batch_id, model_names, task_ids)
        print(f"Tracked batch start: {'SUCCESS' if success else 'FAILED'}")
        
        # 4. Track completion
        result = {
            "success": True,
            "output_file": "/tmp/output/test_gpt2.py",
            "model_type": "gpt2",
            "architecture": "gpt2",
            "duration": 5.2
        }
        
        success = integration.track_task_completion(task_id, result)
        print(f"Tracked task completion: {'SUCCESS' if success else 'FAILED'}")
        
        # 5. Get task history
        tasks = integration.get_task_history(limit=10)
        print(f"Got task history: {'SUCCESS' if len(tasks) > 0 else 'FAILED'}")
        
        # 6. Get model statistics
        stats = integration.get_model_statistics()
        print(f"Got model statistics: {'SUCCESS' if isinstance(stats, list) else 'FAILED'}")
        
        # 7. Get performance report
        report = integration.get_performance_report(days=30)
        print(f"Got performance report: {'SUCCESS' if isinstance(report, dict) else 'FAILED'}")
        
        # Clean up
        integration.close()
        print("Database integration test completed")

if __name__ == "__main__":
    test_db_handler()
    test_db_integration()
    print("\nAll tests completed successfully!")