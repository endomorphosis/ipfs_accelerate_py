#!/usr/bin/env python3
"""
Generator Database Handler

This module provides DuckDB integration for storing and retrieving generator task data,
including history, performance metrics, and generated model information.
"""

import os
import sys
import json
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

try:
    import duckdb
except ImportError:
    raise ImportError("DuckDB not installed. Run: pip install duckdb")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GeneratorDatabaseHandler:
    """
    Database handler for the Generator API.
    
    This class provides methods for storing and retrieving generator task data
    using DuckDB, including task history, performance metrics, and generated model information.
    """
    
    def __init__(self, db_path: str = "generator_tasks.duckdb"):
        """
        Initialize the database handler.
        
        Args:
            db_path: Path to the DuckDB database file
        """
        self.db_path = db_path
        self.connection = None
        self.initialize_db()
    
    def initialize_db(self):
        """Initialize the database connection and create tables if they don't exist."""
        try:
            # Create the database directory if it doesn't exist
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
            
            # Connect to the database
            self.connection = duckdb.connect(self.db_path)
            
            # Create tables if they don't exist
            self._create_tables()
            
            logger.info(f"Database initialized at {self.db_path}")
        
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def _create_tables(self):
        """Create the necessary tables in the database if they don't exist."""
        try:
            # Create generator_tasks table
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS generator_tasks (
                    task_id VARCHAR PRIMARY KEY,
                    model_name VARCHAR,
                    hardware VARCHAR,
                    status VARCHAR,
                    template_type VARCHAR,
                    template_context VARCHAR,
                    output_file VARCHAR,
                    output_dir VARCHAR,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    duration DOUBLE,
                    error VARCHAR,
                    architecture VARCHAR,
                    creator VARCHAR,
                    task_type VARCHAR DEFAULT 'single',
                    batch_id VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create generator_task_steps table for detailed progress tracking
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS generator_task_steps (
                    id INTEGER PRIMARY KEY,
                    task_id VARCHAR,
                    step_name VARCHAR,
                    status VARCHAR,
                    progress DOUBLE,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    duration DOUBLE,
                    error VARCHAR,
                    FOREIGN KEY (task_id) REFERENCES generator_tasks(task_id)
                );
            """)
            
            # Create generator_task_metrics table for performance and resource metrics
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS generator_task_metrics (
                    id INTEGER PRIMARY KEY,
                    task_id VARCHAR,
                    metric_name VARCHAR,
                    metric_value DOUBLE,
                    metric_unit VARCHAR,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (task_id) REFERENCES generator_tasks(task_id)
                );
            """)
            
            # Create generator_task_tags table for tagging tasks
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS generator_task_tags (
                    id INTEGER PRIMARY KEY,
                    task_id VARCHAR,
                    tag_name VARCHAR,
                    tag_value VARCHAR,
                    FOREIGN KEY (task_id) REFERENCES generator_tasks(task_id)
                );
            """)
            
            # Create generator_batch_tasks table for batch task information
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS generator_batch_tasks (
                    batch_id VARCHAR PRIMARY KEY,
                    description VARCHAR,
                    task_count INTEGER,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    duration DOUBLE,
                    status VARCHAR,
                    creator VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create a view for task summary information
            self.connection.execute("""
                CREATE OR REPLACE VIEW generator_task_summary AS
                SELECT 
                    t.task_id, 
                    t.model_name, 
                    t.status, 
                    t.started_at, 
                    t.completed_at, 
                    t.duration,
                    t.architecture,
                    t.template_type,
                    t.output_file,
                    COUNT(DISTINCT s.id) AS step_count,
                    COUNT(DISTINCT m.id) AS metric_count,
                    t.batch_id
                FROM generator_tasks t
                LEFT JOIN generator_task_steps s ON t.task_id = s.task_id
                LEFT JOIN generator_task_metrics m ON t.task_id = m.task_id
                GROUP BY t.task_id, t.model_name, t.status, t.started_at, t.completed_at,
                         t.duration, t.architecture, t.template_type, t.output_file, t.batch_id
                ORDER BY t.started_at DESC;
            """)
            
            # Create batch task summary view
            self.connection.execute("""
                CREATE OR REPLACE VIEW generator_batch_summary AS
                SELECT 
                    b.batch_id,
                    b.description,
                    b.task_count,
                    b.started_at,
                    b.completed_at,
                    b.duration,
                    b.status,
                    COUNT(t.task_id) AS actual_task_count,
                    SUM(CASE WHEN t.status = 'completed' THEN 1 ELSE 0 END) AS completed_tasks,
                    SUM(CASE WHEN t.status = 'error' THEN 1 ELSE 0 END) AS failed_tasks
                FROM generator_batch_tasks b
                LEFT JOIN generator_tasks t ON b.batch_id = t.batch_id
                GROUP BY b.batch_id, b.description, b.task_count, b.started_at,
                         b.completed_at, b.duration, b.status
                ORDER BY b.started_at DESC;
            """)
            
            logger.info("Database tables created successfully")
        
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def store_task(self, task_data: Dict[str, Any]) -> bool:
        """
        Store a generator task in the database.
        
        Args:
            task_data: Dictionary containing task data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Verify we have the required fields
            if 'task_id' not in task_data or 'model_name' not in task_data:
                logger.error("Missing required fields in task data")
                return False
            
            # Convert hardware list to string if needed
            if 'hardware' in task_data and isinstance(task_data['hardware'], list):
                task_data['hardware'] = json.dumps(task_data['hardware'])
            
            # Convert template_context to string if needed
            if 'template_context' in task_data and isinstance(task_data['template_context'], dict):
                task_data['template_context'] = json.dumps(task_data['template_context'])
            
            # Prepare the values
            values = {
                'task_id': task_data['task_id'],
                'model_name': task_data['model_name'],
                'hardware': task_data.get('hardware'),
                'status': task_data.get('status', 'initializing'),
                'template_type': task_data.get('template_type'),
                'template_context': task_data.get('template_context'),
                'output_file': task_data.get('output_file'),
                'output_dir': task_data.get('output_dir'),
                'started_at': task_data.get('started_at', datetime.datetime.now()),
                'completed_at': task_data.get('completed_at'),
                'duration': task_data.get('duration'),
                'error': task_data.get('error'),
                'architecture': task_data.get('architecture'),
                'creator': task_data.get('creator'),
                'task_type': task_data.get('task_type', 'single'),
                'batch_id': task_data.get('batch_id')
            }
            
            # Check if the task already exists
            result = self.connection.execute(
                "SELECT task_id FROM generator_tasks WHERE task_id = ?", 
                [values['task_id']]
            ).fetchone()
            
            if result:
                # Update existing task
                self.connection.execute("""
                    UPDATE generator_tasks SET
                        model_name = ?,
                        hardware = ?,
                        status = ?,
                        template_type = ?,
                        template_context = ?,
                        output_file = ?,
                        output_dir = ?,
                        started_at = ?,
                        completed_at = ?,
                        duration = ?,
                        error = ?,
                        architecture = ?,
                        creator = ?,
                        task_type = ?,
                        batch_id = ?
                    WHERE task_id = ?
                """, [
                    values['model_name'],
                    values['hardware'],
                    values['status'],
                    values['template_type'],
                    values['template_context'],
                    values['output_file'],
                    values['output_dir'],
                    values['started_at'],
                    values['completed_at'],
                    values['duration'],
                    values['error'],
                    values['architecture'],
                    values['creator'],
                    values['task_type'],
                    values['batch_id'],
                    values['task_id']
                ])
            else:
                # Insert new task
                self.connection.execute("""
                    INSERT INTO generator_tasks (
                        task_id, model_name, hardware, status, template_type,
                        template_context, output_file, output_dir, started_at,
                        completed_at, duration, error, architecture, creator,
                        task_type, batch_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    values['task_id'],
                    values['model_name'],
                    values['hardware'],
                    values['status'],
                    values['template_type'],
                    values['template_context'],
                    values['output_file'],
                    values['output_dir'],
                    values['started_at'],
                    values['completed_at'],
                    values['duration'],
                    values['error'],
                    values['architecture'],
                    values['creator'],
                    values['task_type'],
                    values['batch_id']
                ])
            
            # Store task steps if provided
            if 'steps' in task_data and isinstance(task_data['steps'], list):
                for step in task_data['steps']:
                    self.store_task_step(task_data['task_id'], step)
            
            # Store task metrics if provided
            if 'metrics' in task_data and isinstance(task_data['metrics'], dict):
                for name, value in task_data['metrics'].items():
                    self.store_task_metric(task_data['task_id'], name, value)
            
            # Store task tags if provided
            if 'tags' in task_data and isinstance(task_data['tags'], dict):
                for name, value in task_data['tags'].items():
                    self.store_task_tag(task_data['task_id'], name, value)
            
            return True
        
        except Exception as e:
            logger.error(f"Error storing task {task_data.get('task_id')}: {e}")
            return False
    
    def store_task_step(self, task_id: str, step_data: Dict[str, Any]) -> bool:
        """
        Store a task step in the database.
        
        Args:
            task_id: The ID of the task
            step_data: Dictionary containing step data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Verify we have the required fields
            if 'step_name' not in step_data:
                logger.error("Missing required fields in step data")
                return False
            
            # Prepare the values
            values = {
                'task_id': task_id,
                'step_name': step_data['step_name'],
                'status': step_data.get('status', 'pending'),
                'progress': step_data.get('progress', 0.0),
                'started_at': step_data.get('started_at'),
                'completed_at': step_data.get('completed_at'),
                'duration': step_data.get('duration'),
                'error': step_data.get('error')
            }
            
            # Check if the step already exists
            result = self.connection.execute("""
                SELECT id FROM generator_task_steps 
                WHERE task_id = ? AND step_name = ?
            """, [task_id, values['step_name']]).fetchone()
            
            if result:
                # Update existing step
                self.connection.execute("""
                    UPDATE generator_task_steps SET
                        status = ?,
                        progress = ?,
                        started_at = ?,
                        completed_at = ?,
                        duration = ?,
                        error = ?
                    WHERE task_id = ? AND step_name = ?
                """, [
                    values['status'],
                    values['progress'],
                    values['started_at'],
                    values['completed_at'],
                    values['duration'],
                    values['error'],
                    task_id,
                    values['step_name']
                ])
            else:
                # Insert new step
                self.connection.execute("""
                    INSERT INTO generator_task_steps (
                        task_id, step_name, status, progress,
                        started_at, completed_at, duration, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    task_id,
                    values['step_name'],
                    values['status'],
                    values['progress'],
                    values['started_at'],
                    values['completed_at'],
                    values['duration'],
                    values['error']
                ])
            
            return True
        
        except Exception as e:
            logger.error(f"Error storing task step for task {task_id}: {e}")
            return False
    
    def store_task_metric(self, task_id: str, metric_name: str, metric_value: float, 
                         metric_unit: str = '') -> bool:
        """
        Store a task metric in the database.
        
        Args:
            task_id: The ID of the task
            metric_name: Name of the metric
            metric_value: Value of the metric
            metric_unit: Unit of the metric (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Insert new metric
            self.connection.execute("""
                INSERT INTO generator_task_metrics (
                    task_id, metric_name, metric_value, metric_unit
                ) VALUES (?, ?, ?, ?)
            """, [
                task_id,
                metric_name,
                float(metric_value),
                metric_unit
            ])
            
            return True
        
        except Exception as e:
            logger.error(f"Error storing task metric for task {task_id}: {e}")
            return False
    
    def store_task_tag(self, task_id: str, tag_name: str, tag_value: str) -> bool:
        """
        Store a task tag in the database.
        
        Args:
            task_id: The ID of the task
            tag_name: Name of the tag
            tag_value: Value of the tag
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if the tag already exists
            result = self.connection.execute("""
                SELECT id FROM generator_task_tags 
                WHERE task_id = ? AND tag_name = ?
            """, [task_id, tag_name]).fetchone()
            
            if result:
                # Update existing tag
                self.connection.execute("""
                    UPDATE generator_task_tags SET
                        tag_value = ?
                    WHERE task_id = ? AND tag_name = ?
                """, [
                    tag_value,
                    task_id,
                    tag_name
                ])
            else:
                # Insert new tag
                self.connection.execute("""
                    INSERT INTO generator_task_tags (
                        task_id, tag_name, tag_value
                    ) VALUES (?, ?, ?)
                """, [
                    task_id,
                    tag_name,
                    tag_value
                ])
            
            return True
        
        except Exception as e:
            logger.error(f"Error storing task tag for task {task_id}: {e}")
            return False
    
    def store_batch_task(self, batch_data: Dict[str, Any]) -> bool:
        """
        Store a batch task in the database.
        
        Args:
            batch_data: Dictionary containing batch task data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Verify we have the required fields
            if 'batch_id' not in batch_data:
                logger.error("Missing required fields in batch data")
                return False
            
            # Prepare the values
            values = {
                'batch_id': batch_data['batch_id'],
                'description': batch_data.get('description', ''),
                'task_count': batch_data.get('task_count', 0),
                'started_at': batch_data.get('started_at', datetime.datetime.now()),
                'completed_at': batch_data.get('completed_at'),
                'duration': batch_data.get('duration'),
                'status': batch_data.get('status', 'initializing'),
                'creator': batch_data.get('creator', '')
            }
            
            # Check if the batch already exists
            result = self.connection.execute(
                "SELECT batch_id FROM generator_batch_tasks WHERE batch_id = ?", 
                [values['batch_id']]
            ).fetchone()
            
            if result:
                # Update existing batch
                self.connection.execute("""
                    UPDATE generator_batch_tasks SET
                        description = ?,
                        task_count = ?,
                        started_at = ?,
                        completed_at = ?,
                        duration = ?,
                        status = ?,
                        creator = ?
                    WHERE batch_id = ?
                """, [
                    values['description'],
                    values['task_count'],
                    values['started_at'],
                    values['completed_at'],
                    values['duration'],
                    values['status'],
                    values['creator'],
                    values['batch_id']
                ])
            else:
                # Insert new batch
                self.connection.execute("""
                    INSERT INTO generator_batch_tasks (
                        batch_id, description, task_count, started_at,
                        completed_at, duration, status, creator
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    values['batch_id'],
                    values['description'],
                    values['task_count'],
                    values['started_at'],
                    values['completed_at'],
                    values['duration'],
                    values['status'],
                    values['creator']
                ])
            
            return True
        
        except Exception as e:
            logger.error(f"Error storing batch task {batch_data.get('batch_id')}: {e}")
            return False
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a task from the database.
        
        Args:
            task_id: The ID of the task
            
        Returns:
            Dictionary containing task data, or None if not found
        """
        try:
            # Get the task
            result = self.connection.execute("""
                SELECT * FROM generator_tasks WHERE task_id = ?
            """, [task_id]).fetchone()
            
            if not result:
                return None
            
            # Convert to dictionary
            task_data = dict(result)
            
            # Convert hardware from string to list if needed
            if task_data.get('hardware') and task_data['hardware'].startswith('['):
                try:
                    task_data['hardware'] = json.loads(task_data['hardware'])
                except:
                    pass
            
            # Convert template_context from string to dict if needed
            if task_data.get('template_context') and task_data['template_context'].startswith('{'):
                try:
                    task_data['template_context'] = json.loads(task_data['template_context'])
                except:
                    pass
            
            # Get steps for this task
            steps = self.connection.execute("""
                SELECT * FROM generator_task_steps 
                WHERE task_id = ? 
                ORDER BY started_at
            """, [task_id]).fetchall()
            
            if steps:
                task_data['steps'] = [dict(step) for step in steps]
            
            # Get metrics for this task
            metrics = self.connection.execute("""
                SELECT metric_name, metric_value, metric_unit 
                FROM generator_task_metrics 
                WHERE task_id = ?
            """, [task_id]).fetchall()
            
            if metrics:
                task_data['metrics'] = {
                    metric['metric_name']: {
                        'value': metric['metric_value'],
                        'unit': metric['metric_unit']
                    }
                    for metric in metrics
                }
            
            # Get tags for this task
            tags = self.connection.execute("""
                SELECT tag_name, tag_value 
                FROM generator_task_tags 
                WHERE task_id = ?
            """, [task_id]).fetchall()
            
            if tags:
                task_data['tags'] = {
                    tag['tag_name']: tag['tag_value']
                    for tag in tags
                }
            
            return task_data
        
        except Exception as e:
            logger.error(f"Error getting task {task_id}: {e}")
            return None
    
    def get_batch_task(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a batch task from the database.
        
        Args:
            batch_id: The ID of the batch
            
        Returns:
            Dictionary containing batch task data, or None if not found
        """
        try:
            # Get the batch
            result = self.connection.execute("""
                SELECT * FROM generator_batch_tasks WHERE batch_id = ?
            """, [batch_id]).fetchone()
            
            if not result:
                return None
            
            # Convert to dictionary
            batch_data = dict(result)
            
            # Get tasks for this batch
            tasks = self.connection.execute("""
                SELECT task_id, model_name, status, started_at, completed_at, duration
                FROM generator_tasks 
                WHERE batch_id = ? 
                ORDER BY started_at
            """, [batch_id]).fetchall()
            
            if tasks:
                batch_data['tasks'] = [dict(task) for task in tasks]
            
            return batch_data
        
        except Exception as e:
            logger.error(f"Error getting batch task {batch_id}: {e}")
            return None
    
    def list_tasks(self, limit: int = 100, status: Optional[str] = None, 
                 model_name: Optional[str] = None, batch_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List tasks from the database.
        
        Args:
            limit: Maximum number of tasks to return
            status: Optional status filter
            model_name: Optional model name filter
            batch_id: Optional batch ID filter
            
        Returns:
            List of dictionaries containing task data
        """
        try:
            # Build the query
            query = "SELECT * FROM generator_task_summary"
            params = []
            
            # Add filters
            where_clauses = []
            
            if status:
                where_clauses.append("status = ?")
                params.append(status)
            
            if model_name:
                where_clauses.append("model_name = ?")
                params.append(model_name)
            
            if batch_id:
                where_clauses.append("batch_id = ?")
                params.append(batch_id)
            
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
            
            # Add order and limit
            query += " ORDER BY started_at DESC LIMIT ?"
            params.append(limit)
            
            # Execute the query
            results = self.connection.execute(query, params).fetchall()
            
            # Convert to list of dictionaries
            return [dict(result) for result in results]
        
        except Exception as e:
            logger.error(f"Error listing tasks: {e}")
            return []
    
    def list_batch_tasks(self, limit: int = 100, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List batch tasks from the database.
        
        Args:
            limit: Maximum number of batch tasks to return
            status: Optional status filter
            
        Returns:
            List of dictionaries containing batch task data
        """
        try:
            # Build the query
            query = "SELECT * FROM generator_batch_summary"
            params = []
            
            # Add filters
            if status:
                query += " WHERE status = ?"
                params.append(status)
            
            # Add order and limit
            query += " ORDER BY started_at DESC LIMIT ?"
            params.append(limit)
            
            # Execute the query
            results = self.connection.execute(query, params).fetchall()
            
            # Convert to list of dictionaries
            return [dict(result) for result in results]
        
        except Exception as e:
            logger.error(f"Error listing batch tasks: {e}")
            return []
    
    def delete_task(self, task_id: str) -> bool:
        """
        Delete a task from the database.
        
        Args:
            task_id: The ID of the task
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete related records first
            self.connection.execute("DELETE FROM generator_task_steps WHERE task_id = ?", [task_id])
            self.connection.execute("DELETE FROM generator_task_metrics WHERE task_id = ?", [task_id])
            self.connection.execute("DELETE FROM generator_task_tags WHERE task_id = ?", [task_id])
            
            # Delete the task
            self.connection.execute("DELETE FROM generator_tasks WHERE task_id = ?", [task_id])
            
            return True
        
        except Exception as e:
            logger.error(f"Error deleting task {task_id}: {e}")
            return False
    
    def delete_batch_task(self, batch_id: str, delete_tasks: bool = False) -> bool:
        """
        Delete a batch task from the database.
        
        Args:
            batch_id: The ID of the batch
            delete_tasks: Whether to delete all tasks in the batch
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if delete_tasks:
                # Get all task IDs in this batch
                task_ids = self.connection.execute("""
                    SELECT task_id FROM generator_tasks WHERE batch_id = ?
                """, [batch_id]).fetchall()
                
                # Delete each task
                for task_id in [t[0] for t in task_ids]:
                    self.delete_task(task_id)
            else:
                # Just remove the batch ID from tasks
                self.connection.execute("""
                    UPDATE generator_tasks 
                    SET batch_id = NULL 
                    WHERE batch_id = ?
                """, [batch_id])
            
            # Delete the batch
            self.connection.execute("DELETE FROM generator_batch_tasks WHERE batch_id = ?", [batch_id])
            
            return True
        
        except Exception as e:
            logger.error(f"Error deleting batch task {batch_id}: {e}")
            return False
    
    def get_task_count(self, status: Optional[str] = None, 
                     model_name: Optional[str] = None) -> int:
        """
        Get the count of tasks in the database.
        
        Args:
            status: Optional status filter
            model_name: Optional model name filter
            
        Returns:
            Count of tasks
        """
        try:
            # Build the query
            query = "SELECT COUNT(*) FROM generator_tasks"
            params = []
            
            # Add filters
            where_clauses = []
            
            if status:
                where_clauses.append("status = ?")
                params.append(status)
            
            if model_name:
                where_clauses.append("model_name = ?")
                params.append(model_name)
            
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
            
            # Execute the query
            result = self.connection.execute(query, params).fetchone()
            
            return result[0] if result else 0
        
        except Exception as e:
            logger.error(f"Error getting task count: {e}")
            return 0
    
    def get_performance_metrics(self, model_name: Optional[str] = None, 
                              limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get performance metrics for completed tasks.
        
        Args:
            model_name: Optional model name filter
            limit: Maximum number of records to return
            
        Returns:
            List of dictionaries containing performance metrics
        """
        try:
            # Build the query
            query = """
                SELECT 
                    t.task_id, 
                    t.model_name, 
                    t.hardware, 
                    t.architecture,
                    t.template_type,
                    t.duration,
                    t.completed_at,
                    m.metric_name,
                    m.metric_value,
                    m.metric_unit
                FROM generator_tasks t
                JOIN generator_task_metrics m ON t.task_id = m.task_id
                WHERE t.status = 'completed'
            """
            params = []
            
            # Add model filter if provided
            if model_name:
                query += " AND t.model_name = ?"
                params.append(model_name)
            
            # Add order and limit
            query += " ORDER BY t.completed_at DESC LIMIT ?"
            params.append(limit)
            
            # Execute the query
            results = self.connection.execute(query, params).fetchall()
            
            # Group by task
            metrics_by_task = {}
            for row in results:
                task_id = row['task_id']
                if task_id not in metrics_by_task:
                    metrics_by_task[task_id] = {
                        'task_id': task_id,
                        'model_name': row['model_name'],
                        'hardware': row['hardware'],
                        'architecture': row['architecture'],
                        'template_type': row['template_type'],
                        'duration': row['duration'],
                        'completed_at': row['completed_at'],
                        'metrics': {}
                    }
                
                metrics_by_task[task_id]['metrics'][row['metric_name']] = {
                    'value': row['metric_value'],
                    'unit': row['metric_unit']
                }
            
            return list(metrics_by_task.values())
        
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return []
    
    def get_model_statistics(self) -> List[Dict[str, Any]]:
        """
        Get statistics for each model.
        
        Returns:
            List of dictionaries containing model statistics
        """
        try:
            # Execute the query
            results = self.connection.execute("""
                SELECT 
                    model_name,
                    COUNT(*) as total_tasks,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_tasks,
                    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as failed_tasks,
                    SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) as running_tasks,
                    MIN(duration) as min_duration,
                    MAX(duration) as max_duration,
                    AVG(duration) as avg_duration
                FROM generator_tasks
                GROUP BY model_name
                ORDER BY total_tasks DESC
            """).fetchall()
            
            # Convert to list of dictionaries
            return [dict(result) for result in results]
        
        except Exception as e:
            logger.error(f"Error getting model statistics: {e}")
            return []
    
    def get_hardware_statistics(self) -> List[Dict[str, Any]]:
        """
        Get statistics for each hardware platform.
        
        Returns:
            List of dictionaries containing hardware statistics
        """
        try:
            # Execute the query
            results = self.connection.execute("""
                SELECT 
                    hardware,
                    COUNT(*) as total_tasks,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_tasks,
                    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as failed_tasks,
                    MIN(duration) as min_duration,
                    MAX(duration) as max_duration,
                    AVG(duration) as avg_duration
                FROM generator_tasks
                WHERE hardware IS NOT NULL
                GROUP BY hardware
                ORDER BY total_tasks DESC
            """).fetchall()
            
            # Convert to list of dictionaries
            return [dict(result) for result in results]
        
        except Exception as e:
            logger.error(f"Error getting hardware statistics: {e}")
            return []
    
    def get_task_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get task history for the specified number of days.
        
        Args:
            days: Number of days to include
            
        Returns:
            List of dictionaries containing daily task statistics
        """
        try:
            # Execute the query
            results = self.connection.execute("""
                SELECT 
                    DATE(started_at) as day,
                    COUNT(*) as total_tasks,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_tasks,
                    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as failed_tasks,
                    AVG(duration) as avg_duration
                FROM generator_tasks
                WHERE started_at >= CURRENT_DATE - ?
                GROUP BY DATE(started_at)
                ORDER BY day
            """, [days]).fetchall()
            
            # Convert to list of dictionaries
            return [dict(result) for result in results]
        
        except Exception as e:
            logger.error(f"Error getting task history: {e}")
            return []
    
    def vacuum_database(self) -> bool:
        """
        Vacuum the database to reclaim space and optimize performance.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.connection.execute("VACUUM")
            return True
        
        except Exception as e:
            logger.error(f"Error vacuuming database: {e}")
            return False
    
    def export_to_json(self, filename: str) -> bool:
        """
        Export database contents to a JSON file.
        
        Args:
            filename: Path to the output file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all tasks
            tasks = self.list_tasks(limit=10000)
            
            # Get all batch tasks
            batches = self.list_batch_tasks(limit=10000)
            
            # Create export data
            export_data = {
                'tasks': tasks,
                'batches': batches,
                'stats': {
                    'task_count': self.get_task_count(),
                    'model_stats': self.get_model_statistics(),
                    'hardware_stats': self.get_hardware_statistics()
                },
                'export_date': datetime.datetime.now().isoformat()
            }
            
            # Write to file
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            return True
        
        except Exception as e:
            logger.error(f"Error exporting database to {filename}: {e}")
            return False
    
    def import_from_json(self, filename: str) -> bool:
        """
        Import database contents from a JSON file.
        
        Args:
            filename: Path to the input file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read the file
            with open(filename, 'r') as f:
                import_data = json.load(f)
            
            # Import tasks
            if 'tasks' in import_data:
                for task in import_data['tasks']:
                    self.store_task(task)
            
            # Import batch tasks
            if 'batches' in import_data:
                for batch in import_data['batches']:
                    self.store_batch_task(batch)
            
            return True
        
        except Exception as e:
            logger.error(f"Error importing database from {filename}: {e}")
            return False