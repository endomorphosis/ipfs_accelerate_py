#!/usr/bin/env python3
"""
Generator DuckDB Integration Module

This module provides functions for integrating the Generator API with DuckDB,
including task tracking, history, and performance metrics.
"""

import os
import sys
import logging
import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import database handler
try:
    from database.db_handler import GeneratorDatabaseHandler
except ImportError:
    raise ImportError("GeneratorDatabaseHandler not found. Make sure db_handler.py is installed.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GeneratorDatabaseIntegration:
    """
    Integration layer between the Generator API and DuckDB database.
    
    This class provides methods for tracking generation tasks, historical data,
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
                db_path = "/var/lib/ipfs_accelerate/generator_tasks.duckdb"
            else:
                db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                       "data", "generator_tasks.duckdb")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database handler
        self.db = GeneratorDatabaseHandler(db_path)
        logger.info(f"Generator database integration initialized with database at {db_path}")
    
    def track_task_start(self, task_data: Dict[str, Any]) -> bool:
        """
        Track the start of a generation task.
        
        Args:
            task_data: Dictionary containing task data
            
        Returns:
            True if successful, False otherwise
        """
        # Convert hardware list to a string for storage if needed
        task_info = task_data.copy()
        
        # Ensure started_at is a datetime object
        if 'started_at' in task_info and isinstance(task_info['started_at'], str):
            try:
                task_info['started_at'] = datetime.datetime.fromisoformat(task_info['started_at'])
            except:
                task_info['started_at'] = datetime.datetime.now()
        elif 'started_at' not in task_info:
            task_info['started_at'] = datetime.datetime.now()
        
        # Set initial status
        task_info['status'] = task_info.get('status', 'initializing')
        
        # Store in database
        return self.db.store_task(task_info)
    
    def track_task_update(self, task_id: str, status: str, progress: float, 
                         current_step: str, error: Optional[str] = None) -> bool:
        """
        Track an update to a generation task.
        
        Args:
            task_id: The ID of the task
            status: Current status
            progress: Current progress (0.0 - 1.0)
            current_step: Current step name
            error: Optional error message
            
        Returns:
            True if successful, False otherwise
        """
        # Get existing task data
        task_data = self.db.get_task(task_id)
        if not task_data:
            logger.warning(f"Task {task_id} not found for update")
            return False
        
        # Update task data
        task_data['status'] = status
        task_data['error'] = error
        
        # Store the step
        step_data = {
            'step_name': current_step,
            'status': status,
            'progress': progress,
            'started_at': datetime.datetime.now(),
            'error': error
        }
        
        # If the step is complete, add completed_at
        if status in ['completed', 'error']:
            step_data['completed_at'] = datetime.datetime.now()
            if 'started_at' in step_data:
                step_data['duration'] = (step_data['completed_at'] - step_data['started_at']).total_seconds()
        
        # Store step and updated task
        self.db.store_task_step(task_id, step_data)
        return self.db.store_task(task_data)
    
    def track_task_completion(self, task_id: str, result: Dict[str, Any]) -> bool:
        """
        Track the completion of a generation task.
        
        Args:
            task_id: The ID of the task
            result: Dictionary containing result data
            
        Returns:
            True if successful, False otherwise
        """
        # Get existing task data
        task_data = self.db.get_task(task_id)
        if not task_data:
            logger.warning(f"Task {task_id} not found for completion")
            return False
        
        # Update task data with results
        task_data['status'] = result.get('success', False) and 'completed' or 'error'
        task_data['completed_at'] = datetime.datetime.now()
        task_data['duration'] = result.get('duration', 
                                          (task_data['completed_at'] - task_data['started_at']).total_seconds())
        task_data['output_file'] = result.get('output_file')
        task_data['architecture'] = result.get('architecture')
        task_data['error'] = result.get('error')
        
        # Add metrics if available
        if 'metrics' not in task_data:
            task_data['metrics'] = {}
        
        # Add duration as a metric
        self.db.store_task_metric(task_id, 'duration', task_data['duration'], 'seconds')
        
        # Add other metrics from result if available
        if 'model_info' in result and isinstance(result['model_info'], dict):
            for key, value in result['model_info'].items():
                if isinstance(value, (int, float)):
                    self.db.store_task_metric(task_id, f"model_info.{key}", value)
        
        # Add final step
        final_step = {
            'step_name': 'Completed',
            'status': task_data['status'],
            'progress': 1.0,
            'started_at': datetime.datetime.now(),
            'completed_at': datetime.datetime.now(),
            'duration': 0.0,
            'error': task_data['error']
        }
        self.db.store_task_step(task_id, final_step)
        
        # Store updated task
        return self.db.store_task(task_data)
    
    def track_batch_start(self, batch_id: str, model_names: List[str], task_ids: List[str]) -> bool:
        """
        Track the start of a batch generation task.
        
        Args:
            batch_id: The ID of the batch
            model_names: List of model names in the batch
            task_ids: List of task IDs in the batch
            
        Returns:
            True if successful, False otherwise
        """
        # Create batch task data
        batch_data = {
            'batch_id': batch_id,
            'description': f"Batch generation of {len(model_names)} models",
            'task_count': len(model_names),
            'started_at': datetime.datetime.now(),
            'status': 'running'
        }
        
        # Store batch task
        success = self.db.store_batch_task(batch_data)
        
        # Add batch_id to each task
        for task_id in task_ids:
            task_data = self.db.get_task(task_id)
            if task_data:
                task_data['batch_id'] = batch_id
                self.db.store_task(task_data)
        
        return success
    
    def track_batch_completion(self, batch_id: str) -> bool:
        """
        Track the completion of a batch generation task.
        
        Args:
            batch_id: The ID of the batch
            
        Returns:
            True if successful, False otherwise
        """
        # Get batch task data
        batch_data = self.db.get_batch_task(batch_id)
        if not batch_data:
            logger.warning(f"Batch {batch_id} not found for completion")
            return False
        
        # Get tasks for this batch
        tasks = self.db.list_tasks(limit=1000, batch_id=batch_id)
        
        # Count completed and error tasks
        completed_count = sum(1 for t in tasks if t['status'] == 'completed')
        error_count = sum(1 for t in tasks if t['status'] == 'error')
        
        # Update batch status
        if len(tasks) == completed_count:
            batch_data['status'] = 'completed'
        elif error_count > 0:
            batch_data['status'] = 'partial'
        else:
            batch_data['status'] = 'running'
        
        # Set completion time
        if batch_data['status'] in ['completed', 'partial']:
            batch_data['completed_at'] = datetime.datetime.now()
            batch_data['duration'] = (batch_data['completed_at'] - batch_data['started_at']).total_seconds()
        
        # Store updated batch task
        return self.db.store_batch_task(batch_data)
    
    def get_task_history(self, limit: int = 100, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get task history from the database.
        
        Args:
            limit: Maximum number of tasks to return
            model_name: Optional filter by model name
            
        Returns:
            List of dictionaries containing task history data
        """
        return self.db.list_tasks(limit=limit, model_name=model_name)
    
    def get_batch_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get batch task history from the database.
        
        Args:
            limit: Maximum number of batch tasks to return
            
        Returns:
            List of dictionaries containing batch task history data
        """
        return self.db.list_batch_tasks(limit=limit)
    
    def get_model_statistics(self) -> List[Dict[str, Any]]:
        """
        Get statistics for each model.
        
        Returns:
            List of dictionaries containing model statistics
        """
        return self.db.get_model_statistics()
    
    def get_task_details(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a task.
        
        Args:
            task_id: The ID of the task
            
        Returns:
            Dictionary containing task details or None if not found
        """
        return self.db.get_task(task_id)
    
    def get_batch_details(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a batch task.
        
        Args:
            batch_id: The ID of the batch
            
        Returns:
            Dictionary containing batch task details or None if not found
        """
        return self.db.get_batch_task(batch_id)
    
    def get_performance_report(self, days: int = 30) -> Dict[str, Any]:
        """
        Get a performance report for the specified time period.
        
        Args:
            days: Number of days to include
            
        Returns:
            Dictionary containing performance report data
        """
        return {
            'daily_stats': self.db.get_task_history(days),
            'model_stats': self.db.get_model_statistics(),
            'hardware_stats': self.db.get_hardware_statistics(),
            'latest_tasks': self.db.list_tasks(limit=20),
            'generated_at': datetime.datetime.now().isoformat()
        }
    
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