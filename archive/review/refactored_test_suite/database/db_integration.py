#!/usr/bin/env python3
"""
Test Suite DuckDB Integration Module

This module provides functions for integrating the Test Suite API with DuckDB,
including test run tracking, history, and performance metrics.
"""

import os
import sys
import logging
import datetime
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path

# Import database handler
try:
    from database.db_handler import TestDatabaseHandler
except ImportError:
    # Add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from database.db_handler import TestDatabaseHandler
    except ImportError:
        raise ImportError("TestDatabaseHandler not found. Make sure db_handler.py is installed.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestDatabaseIntegration:
    """
    Integration layer between the Test Suite API and DuckDB database.
    
    This class provides methods for tracking test runs, historical data,
    performance metrics, and batch operations.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the database integration.
        
        Args:
            db_path: Optional path to the database file
        """
        # Use default path if not provided
        if not db_path:
            # Check if we're in development or production
            if os.environ.get("IPFS_ENV") == "production":
                db_path = "/var/lib/ipfs_accelerate/test_runs.duckdb"
            else:
                db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                       "data", "test_runs.duckdb")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database handler
        self.db = TestDatabaseHandler(db_path)
        logger.info(f"Test database integration initialized with database at {db_path}")
    
    def track_test_start(self, run_data: Dict[str, Any]) -> bool:
        """
        Track the start of a test run.
        
        Args:
            run_data: Dictionary containing test run data
            
        Returns:
            True if successful, False otherwise
        """
        # Convert hardware list to a string for storage if needed
        run_info = run_data.copy()
        
        # Ensure started_at is a datetime object
        if 'started_at' in run_info and isinstance(run_info['started_at'], str):
            try:
                run_info['started_at'] = datetime.datetime.fromisoformat(run_info['started_at'])
            except:
                run_info['started_at'] = datetime.datetime.now()
        elif 'started_at' not in run_info:
            run_info['started_at'] = datetime.datetime.now()
        
        # Set initial status
        run_info['status'] = run_info.get('status', 'initializing')
        
        # Store in database
        return self.db.store_test_run(run_info)
    
    def track_test_update(self, run_id: str, status: str, progress: float, 
                         current_step: str, error: Optional[str] = None) -> bool:
        """
        Track an update to a test run.
        
        Args:
            run_id: The ID of the run
            status: Current status
            progress: Current progress (0.0 - 1.0)
            current_step: Current step name
            error: Optional error message
            
        Returns:
            True if successful, False otherwise
        """
        # Get existing run data
        run_data = self.db.get_test_run(run_id)
        if not run_data:
            logger.warning(f"Run {run_id} not found for update")
            return False
        
        # Update run data
        run_data['status'] = status
        run_data['progress'] = progress
        run_data['current_step'] = current_step
        run_data['error'] = error
        
        # Store the step
        step_data = {
            'step_name': current_step,
            'status': status,
            'progress': progress,
            'started_at': datetime.datetime.now(),
            'error': error
        }
        
        # If the step is complete, add completed_at
        if status in ['completed', 'error', 'cancelled']:
            step_data['completed_at'] = datetime.datetime.now()
            if 'started_at' in step_data:
                step_data['duration'] = (step_data['completed_at'] - step_data['started_at']).total_seconds()
        
        # Store step and updated run
        self.db.store_test_step(run_id, step_data)
        return self.db.store_test_run(run_data)
    
    def track_test_completion(self, run_id: str, results: Dict[str, Any]) -> bool:
        """
        Track the completion of a test run.
        
        Args:
            run_id: The ID of the run
            results: Dictionary containing result data
            
        Returns:
            True if successful, False otherwise
        """
        # Get existing run data
        run_data = self.db.get_test_run(run_id)
        if not run_data:
            logger.warning(f"Run {run_id} not found for completion")
            return False
        
        # Update run data with results
        run_data['status'] = 'completed'
        run_data['completed_at'] = datetime.datetime.now()
        run_data['duration'] = (run_data['completed_at'] - run_data['started_at']).total_seconds()
        
        # Add results
        if 'test_details' in results:
            # Count results by status
            tests_passed = sum(1 for test in results['test_details'] if test['status'] == 'passed')
            tests_failed = sum(1 for test in results['test_details'] if test['status'] == 'failed')
            tests_skipped = sum(1 for test in results['test_details'] if test['status'] == 'skipped')
            
            run_data['results'] = {
                'tests_passed': tests_passed,
                'tests_failed': tests_failed,
                'tests_skipped': tests_skipped,
                'test_details': results['test_details']
            }
        else:
            # Use the provided results
            run_data['results'] = results
        
        # Add performance metrics if available
        if 'performance_metrics' in results:
            for hardware, metrics in results['performance_metrics'].items():
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        self.db.store_test_metric(
                            run_id, 
                            f"{hardware}.{metric_name}", 
                            metric_value, 
                            'ms' if 'latency' in metric_name else ''
                        )
        
        # Record any result file
        if 'result_file' in results:
            run_data['result_file'] = results['result_file']
            
            # Add as artifact
            self.db.store_test_artifact(run_id, {
                'artifact_type': 'result_file',
                'artifact_path': results['result_file'],
                'description': 'Test results JSON file'
            })
        
        # Add duration as a metric
        self.db.store_test_metric(run_id, 'duration', run_data['duration'], 'seconds')
        
        # Add final step
        final_step = {
            'step_name': 'Completed',
            'status': 'completed',
            'progress': 1.0,
            'started_at': datetime.datetime.now(),
            'completed_at': datetime.datetime.now(),
            'duration': 0.0
        }
        self.db.store_test_step(run_id, final_step)
        
        # Store updated run
        return self.db.store_test_run(run_data)
    
    def track_test_error(self, run_id: str, error: str) -> bool:
        """
        Track an error in a test run.
        
        Args:
            run_id: The ID of the run
            error: Error message
            
        Returns:
            True if successful, False otherwise
        """
        # Get existing run data
        run_data = self.db.get_test_run(run_id)
        if not run_data:
            logger.warning(f"Run {run_id} not found for error")
            return False
        
        # Update run data with error
        run_data['status'] = 'error'
        run_data['error'] = error
        run_data['completed_at'] = datetime.datetime.now()
        run_data['duration'] = (run_data['completed_at'] - run_data['started_at']).total_seconds()
        
        # Add error step
        error_step = {
            'step_name': 'Error',
            'status': 'error',
            'progress': 1.0,
            'started_at': datetime.datetime.now(),
            'completed_at': datetime.datetime.now(),
            'duration': 0.0,
            'error': error
        }
        self.db.store_test_step(run_id, error_step)
        
        # Store updated run
        return self.db.store_test_run(run_data)
    
    def track_test_cancellation(self, run_id: str) -> bool:
        """
        Track the cancellation of a test run.
        
        Args:
            run_id: The ID of the run
            
        Returns:
            True if successful, False otherwise
        """
        # Get existing run data
        run_data = self.db.get_test_run(run_id)
        if not run_data:
            logger.warning(f"Run {run_id} not found for cancellation")
            return False
        
        # Update run data
        run_data['status'] = 'cancelled'
        run_data['completed_at'] = datetime.datetime.now()
        run_data['duration'] = (run_data['completed_at'] - run_data['started_at']).total_seconds()
        
        # Add cancellation step
        cancel_step = {
            'step_name': 'Cancelled',
            'status': 'cancelled',
            'progress': 1.0,
            'started_at': datetime.datetime.now(),
            'completed_at': datetime.datetime.now(),
            'duration': 0.0
        }
        self.db.store_test_step(run_id, cancel_step)
        
        # Store updated run
        return self.db.store_test_run(run_data)
    
    def track_batch_start(self, batch_id: str, model_names: List[str], run_ids: List[str]) -> bool:
        """
        Track the start of a batch test run.
        
        Args:
            batch_id: The ID of the batch
            model_names: List of model names in the batch
            run_ids: List of run IDs in the batch
            
        Returns:
            True if successful, False otherwise
        """
        # Create batch data
        batch_data = {
            'batch_id': batch_id,
            'description': f"Batch testing of {len(model_names)} models",
            'run_count': len(model_names),
            'started_at': datetime.datetime.now(),
            'status': 'running'
        }
        
        # Store batch
        success = self.db.store_batch_test_run(batch_data)
        
        # Add batch_id to each run
        for run_id in run_ids:
            self.db.add_run_to_batch(run_id, batch_id)
        
        return success
    
    def track_batch_completion(self, batch_id: str) -> bool:
        """
        Track the completion of a batch test run.
        
        Args:
            batch_id: The ID of the batch
            
        Returns:
            True if successful, False otherwise
        """
        # Get batch data
        batch_data = self.db.get_batch_test_run(batch_id)
        if not batch_data:
            logger.warning(f"Batch {batch_id} not found for completion")
            return False
        
        # Get runs for this batch
        runs = batch_data.get('runs', [])
        
        # Count completed and error runs
        completed_count = sum(1 for r in runs if r['status'] == 'completed')
        error_count = sum(1 for r in runs if r['status'] == 'error')
        
        # Update batch status
        if len(runs) == 0:
            batch_data['status'] = 'error'
        elif len(runs) == completed_count:
            batch_data['status'] = 'completed'
        elif error_count > 0:
            batch_data['status'] = 'partial'
        else:
            batch_data['status'] = 'running'
        
        # Set completion time
        if batch_data['status'] in ['completed', 'partial', 'error']:
            batch_data['completed_at'] = datetime.datetime.now()
            batch_data['duration'] = (batch_data['completed_at'] - batch_data['started_at']).total_seconds()
        
        # Store updated batch
        return self.db.store_batch_test_run(batch_data)
    
    def get_run_history(self, limit: int = 100, model_name: Optional[str] = None,
                      test_type: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get test run history from the database.
        
        Args:
            limit: Maximum number of runs to return
            model_name: Optional filter by model name
            test_type: Optional filter by test type
            status: Optional filter by status
            
        Returns:
            List of dictionaries containing test run history data
        """
        return self.db.list_test_runs(
            limit=limit, 
            model_name=model_name,
            test_type=test_type,
            status=status
        )
    
    def get_batch_history(self, limit: int = 100, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get batch test run history from the database.
        
        Args:
            limit: Maximum number of batch runs to return
            status: Optional filter by status
            
        Returns:
            List of dictionaries containing batch test run history data
        """
        return self.db.list_batch_test_runs(limit=limit, status=status)
    
    def get_model_statistics(self) -> List[Dict[str, Any]]:
        """
        Get statistics for each model.
        
        Returns:
            List of dictionaries containing model statistics
        """
        return self.db.get_model_statistics()
    
    def get_hardware_statistics(self) -> List[Dict[str, Any]]:
        """
        Get statistics for each hardware platform.
        
        Returns:
            List of dictionaries containing hardware statistics
        """
        return self.db.get_hardware_statistics()
    
    def get_run_details(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a test run.
        
        Args:
            run_id: The ID of the run
            
        Returns:
            Dictionary containing test run details or None if not found
        """
        return self.db.get_test_run(run_id)
    
    def get_batch_details(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a batch test run.
        
        Args:
            batch_id: The ID of the batch
            
        Returns:
            Dictionary containing batch test run details or None if not found
        """
        return self.db.get_batch_test_run(batch_id)
    
    def get_performance_report(self, days: int = 30, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a performance report for the specified time period.
        
        Args:
            days: Number of days to include
            model_name: Optional filter by model name
            
        Returns:
            Dictionary containing performance report data
        """
        return {
            'daily_stats': self.db.get_test_history(days),
            'model_stats': self.db.get_model_statistics(),
            'hardware_stats': self.db.get_hardware_statistics(),
            'latest_runs': self.db.list_test_runs(limit=20, model_name=model_name),
            'performance_metrics': self.db.get_test_performance_metrics(model_name=model_name, limit=100),
            'generated_at': datetime.datetime.now().isoformat()
        }
    
    def search_runs(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search test runs by query string.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries containing matching test runs
        """
        return self.db.search_test_runs(query, limit)
    
    def export_database(self, filename: str) -> bool:
        """
        Export the database to a JSON file.
        
        Args:
            filename: Path to the output file
            
        Returns:
            True if successful, False otherwise
        """
        return self.db.export_to_json(filename)
    
    def close(self):
        """Close the database connection."""
        self.db.close()