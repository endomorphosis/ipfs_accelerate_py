#!/usr/bin/env python3
"""
Test Suite Database Handler

This module provides DuckDB integration for storing and retrieving test run data,
including history, results, and performance metrics.
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

class TestDatabaseHandler:
    """
    Database handler for the Test Suite API.
    
    This class provides methods for storing and retrieving test data
    using DuckDB, including test run history, results, and performance metrics.
    """
    
    def __init__(self, db_path: str = "test_runs.duckdb"):
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
            # Create test_runs table
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS test_runs (
                    run_id VARCHAR PRIMARY KEY,
                    model_name VARCHAR,
                    hardware VARCHAR,
                    test_type VARCHAR,
                    timeout INTEGER,
                    save_results BOOLEAN,
                    status VARCHAR,
                    progress DOUBLE,
                    current_step VARCHAR,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    duration DOUBLE,
                    error VARCHAR,
                    result_file VARCHAR,
                    creator VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create test_run_results table for detailed test results
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS test_run_results (
                    id INTEGER PRIMARY KEY,
                    run_id VARCHAR,
                    tests_passed INTEGER,
                    tests_failed INTEGER,
                    tests_skipped INTEGER,
                    result_data VARCHAR,
                    FOREIGN KEY (run_id) REFERENCES test_runs(run_id)
                );
            """)
            
            # Create test_run_steps table for detailed progress tracking
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS test_run_steps (
                    id INTEGER PRIMARY KEY,
                    run_id VARCHAR,
                    step_name VARCHAR,
                    status VARCHAR,
                    progress DOUBLE,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    duration DOUBLE,
                    error VARCHAR,
                    FOREIGN KEY (run_id) REFERENCES test_runs(run_id)
                );
            """)
            
            # Create test_run_metrics table for performance and resource metrics
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS test_run_metrics (
                    id INTEGER PRIMARY KEY,
                    run_id VARCHAR,
                    metric_name VARCHAR,
                    metric_value DOUBLE,
                    metric_unit VARCHAR,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES test_runs(run_id)
                );
            """)
            
            # Create test_run_tags table for tagging test runs
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS test_run_tags (
                    id INTEGER PRIMARY KEY,
                    run_id VARCHAR,
                    tag_name VARCHAR,
                    tag_value VARCHAR,
                    FOREIGN KEY (run_id) REFERENCES test_runs(run_id)
                );
            """)
            
            # Create batch_test_runs table for batch test information
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS batch_test_runs (
                    batch_id VARCHAR PRIMARY KEY,
                    description VARCHAR,
                    run_count INTEGER,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    duration DOUBLE,
                    status VARCHAR,
                    creator VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create test_run_batch relation table
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS test_run_batch (
                    id INTEGER PRIMARY KEY,
                    run_id VARCHAR,
                    batch_id VARCHAR,
                    FOREIGN KEY (run_id) REFERENCES test_runs(run_id),
                    FOREIGN KEY (batch_id) REFERENCES batch_test_runs(batch_id)
                );
            """)
            
            # Create test_run_artifacts table for files produced during testing
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS test_run_artifacts (
                    id INTEGER PRIMARY KEY,
                    run_id VARCHAR,
                    artifact_type VARCHAR,
                    artifact_path VARCHAR,
                    description VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES test_runs(run_id)
                );
            """)
            
            # Create a view for test run summary information
            self.connection.execute("""
                CREATE OR REPLACE VIEW test_run_summary AS
                SELECT 
                    t.run_id, 
                    t.model_name, 
                    t.hardware,
                    t.test_type,
                    t.status, 
                    t.progress,
                    t.started_at, 
                    t.completed_at, 
                    t.duration,
                    r.tests_passed,
                    r.tests_failed,
                    r.tests_skipped,
                    COUNT(DISTINCT s.id) AS step_count,
                    COUNT(DISTINCT m.id) AS metric_count,
                    COUNT(DISTINCT a.id) AS artifact_count,
                    b.batch_id
                FROM test_runs t
                LEFT JOIN test_run_results r ON t.run_id = r.run_id
                LEFT JOIN test_run_steps s ON t.run_id = s.run_id
                LEFT JOIN test_run_metrics m ON t.run_id = m.run_id
                LEFT JOIN test_run_artifacts a ON t.run_id = a.run_id
                LEFT JOIN test_run_batch b ON t.run_id = b.run_id
                GROUP BY t.run_id, t.model_name, t.hardware, t.test_type, t.status, t.progress,
                         t.started_at, t.completed_at, t.duration, r.tests_passed, 
                         r.tests_failed, r.tests_skipped, b.batch_id
                ORDER BY t.started_at DESC;
            """)
            
            # Create batch test run summary view
            self.connection.execute("""
                CREATE OR REPLACE VIEW batch_test_summary AS
                SELECT 
                    b.batch_id,
                    b.description,
                    b.run_count,
                    b.started_at,
                    b.completed_at,
                    b.duration,
                    b.status,
                    COUNT(rb.run_id) AS actual_run_count,
                    SUM(CASE WHEN t.status = 'completed' THEN 1 ELSE 0 END) AS completed_runs,
                    SUM(CASE WHEN t.status = 'error' THEN 1 ELSE 0 END) AS failed_runs,
                    SUM(CASE WHEN t.status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled_runs,
                    SUM(r.tests_passed) AS total_tests_passed,
                    SUM(r.tests_failed) AS total_tests_failed
                FROM batch_test_runs b
                LEFT JOIN test_run_batch rb ON b.batch_id = rb.batch_id
                LEFT JOIN test_runs t ON rb.run_id = t.run_id
                LEFT JOIN test_run_results r ON t.run_id = r.run_id
                GROUP BY b.batch_id, b.description, b.run_count, b.started_at,
                         b.completed_at, b.duration, b.status
                ORDER BY b.started_at DESC;
            """)
            
            # Create model summary view
            self.connection.execute("""
                CREATE OR REPLACE VIEW model_test_summary AS
                SELECT 
                    model_name,
                    COUNT(*) as total_runs,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_runs,
                    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as failed_runs,
                    SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) as cancelled_runs,
                    AVG(CASE WHEN status = 'completed' THEN duration ELSE NULL END) as avg_duration,
                    MIN(CASE WHEN status = 'completed' THEN duration ELSE NULL END) as min_duration,
                    MAX(CASE WHEN status = 'completed' THEN duration ELSE NULL END) as max_duration,
                    MAX(started_at) as latest_run
                FROM test_runs
                GROUP BY model_name
                ORDER BY total_runs DESC;
            """)
            
            # Create hardware summary view
            self.connection.execute("""
                CREATE OR REPLACE VIEW hardware_test_summary AS
                SELECT 
                    hardware,
                    COUNT(*) as total_runs,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_runs,
                    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as failed_runs,
                    AVG(CASE WHEN status = 'completed' THEN duration ELSE NULL END) as avg_duration,
                    MIN(CASE WHEN status = 'completed' THEN duration ELSE NULL END) as min_duration,
                    MAX(CASE WHEN status = 'completed' THEN duration ELSE NULL END) as max_duration
                FROM test_runs
                GROUP BY hardware
                ORDER BY total_runs DESC;
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
    
    def store_test_run(self, run_data: Dict[str, Any]) -> bool:
        """
        Store a test run in the database.
        
        Args:
            run_data: Dictionary containing test run data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Verify we have the required fields
            if 'run_id' not in run_data or 'model_name' not in run_data:
                logger.error("Missing required fields in run data")
                return False
            
            # Convert hardware list to string if needed
            if 'hardware' in run_data and isinstance(run_data['hardware'], list):
                run_data['hardware'] = json.dumps(run_data['hardware'])
            
            # Prepare the values
            values = {
                'run_id': run_data['run_id'],
                'model_name': run_data['model_name'],
                'hardware': run_data.get('hardware'),
                'test_type': run_data.get('test_type'),
                'timeout': run_data.get('timeout'),
                'save_results': run_data.get('save_results'),
                'status': run_data.get('status', 'initializing'),
                'progress': run_data.get('progress', 0.0),
                'current_step': run_data.get('current_step'),
                'started_at': run_data.get('started_at', datetime.datetime.now()),
                'completed_at': run_data.get('completed_at'),
                'duration': run_data.get('duration'),
                'error': run_data.get('error'),
                'result_file': run_data.get('result_file'),
                'creator': run_data.get('creator')
            }
            
            # Check if the run already exists
            result = self.connection.execute(
                "SELECT run_id FROM test_runs WHERE run_id = ?", 
                [values['run_id']]
            ).fetchone()
            
            if result:
                # Update existing run
                self.connection.execute("""
                    UPDATE test_runs SET
                        model_name = ?,
                        hardware = ?,
                        test_type = ?,
                        timeout = ?,
                        save_results = ?,
                        status = ?,
                        progress = ?,
                        current_step = ?,
                        started_at = ?,
                        completed_at = ?,
                        duration = ?,
                        error = ?,
                        result_file = ?,
                        creator = ?
                    WHERE run_id = ?
                """, [
                    values['model_name'],
                    values['hardware'],
                    values['test_type'],
                    values['timeout'],
                    values['save_results'],
                    values['status'],
                    values['progress'],
                    values['current_step'],
                    values['started_at'],
                    values['completed_at'],
                    values['duration'],
                    values['error'],
                    values['result_file'],
                    values['creator'],
                    values['run_id']
                ])
            else:
                # Insert new run
                self.connection.execute("""
                    INSERT INTO test_runs (
                        run_id, model_name, hardware, test_type, timeout, 
                        save_results, status, progress, current_step, started_at,
                        completed_at, duration, error, result_file, creator
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    values['run_id'],
                    values['model_name'],
                    values['hardware'],
                    values['test_type'],
                    values['timeout'],
                    values['save_results'],
                    values['status'],
                    values['progress'],
                    values['current_step'],
                    values['started_at'],
                    values['completed_at'],
                    values['duration'],
                    values['error'],
                    values['result_file'],
                    values['creator']
                ])
            
            # Store run results if provided
            if 'results' in run_data and isinstance(run_data['results'], dict):
                self.store_test_results(run_data['run_id'], run_data['results'])
            
            # Store run steps if provided
            if 'steps' in run_data and isinstance(run_data['steps'], list):
                for step in run_data['steps']:
                    self.store_test_step(run_data['run_id'], step)
            
            # Store run metrics if provided
            if 'metrics' in run_data and isinstance(run_data['metrics'], dict):
                for name, value in run_data['metrics'].items():
                    self.store_test_metric(run_data['run_id'], name, value)
            
            # Store run tags if provided
            if 'tags' in run_data and isinstance(run_data['tags'], dict):
                for name, value in run_data['tags'].items():
                    self.store_test_tag(run_data['run_id'], name, value)
            
            # Store run artifacts if provided
            if 'artifacts' in run_data and isinstance(run_data['artifacts'], list):
                for artifact in run_data['artifacts']:
                    self.store_test_artifact(run_data['run_id'], artifact)
            
            # Store batch relation if provided
            if 'batch_id' in run_data and run_data['batch_id']:
                self.add_run_to_batch(run_data['run_id'], run_data['batch_id'])
            
            return True
        
        except Exception as e:
            logger.error(f"Error storing test run {run_data.get('run_id')}: {e}")
            return False
    
    def store_test_results(self, run_id: str, results: Dict[str, Any]) -> bool:
        """
        Store test results in the database.
        
        Args:
            run_id: The ID of the test run
            results: Dictionary containing test results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare values
            tests_passed = results.get('tests_passed', 0)
            tests_failed = results.get('tests_failed', 0)
            tests_skipped = results.get('tests_skipped', 0)
            
            # Convert dict to JSON string
            result_data = json.dumps(results)
            
            # Check if results already exist for this run
            result = self.connection.execute("""
                SELECT id FROM test_run_results WHERE run_id = ?
            """, [run_id]).fetchone()
            
            if result:
                # Update existing results
                self.connection.execute("""
                    UPDATE test_run_results SET
                        tests_passed = ?,
                        tests_failed = ?,
                        tests_skipped = ?,
                        result_data = ?
                    WHERE run_id = ?
                """, [
                    tests_passed,
                    tests_failed,
                    tests_skipped,
                    result_data,
                    run_id
                ])
            else:
                # Insert new results
                self.connection.execute("""
                    INSERT INTO test_run_results (
                        run_id, tests_passed, tests_failed, tests_skipped, result_data
                    ) VALUES (?, ?, ?, ?, ?)
                """, [
                    run_id,
                    tests_passed,
                    tests_failed,
                    tests_skipped,
                    result_data
                ])
            
            return True
        
        except Exception as e:
            logger.error(f"Error storing test results for run {run_id}: {e}")
            return False
    
    def store_test_step(self, run_id: str, step_data: Dict[str, Any]) -> bool:
        """
        Store a test step in the database.
        
        Args:
            run_id: The ID of the test run
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
                'run_id': run_id,
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
                SELECT id FROM test_run_steps 
                WHERE run_id = ? AND step_name = ?
            """, [run_id, values['step_name']]).fetchone()
            
            if result:
                # Update existing step
                self.connection.execute("""
                    UPDATE test_run_steps SET
                        status = ?,
                        progress = ?,
                        started_at = ?,
                        completed_at = ?,
                        duration = ?,
                        error = ?
                    WHERE run_id = ? AND step_name = ?
                """, [
                    values['status'],
                    values['progress'],
                    values['started_at'],
                    values['completed_at'],
                    values['duration'],
                    values['error'],
                    run_id,
                    values['step_name']
                ])
            else:
                # Insert new step
                self.connection.execute("""
                    INSERT INTO test_run_steps (
                        run_id, step_name, status, progress,
                        started_at, completed_at, duration, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    run_id,
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
            logger.error(f"Error storing test step for run {run_id}: {e}")
            return False
    
    def store_test_metric(self, run_id: str, metric_name: str, metric_value: float, 
                         metric_unit: str = '') -> bool:
        """
        Store a test metric in the database.
        
        Args:
            run_id: The ID of the test run
            metric_name: Name of the metric
            metric_value: Value of the metric
            metric_unit: Unit of the metric (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Insert new metric
            self.connection.execute("""
                INSERT INTO test_run_metrics (
                    run_id, metric_name, metric_value, metric_unit
                ) VALUES (?, ?, ?, ?)
            """, [
                run_id,
                metric_name,
                float(metric_value),
                metric_unit
            ])
            
            return True
        
        except Exception as e:
            logger.error(f"Error storing test metric for run {run_id}: {e}")
            return False
    
    def store_test_tag(self, run_id: str, tag_name: str, tag_value: str) -> bool:
        """
        Store a test tag in the database.
        
        Args:
            run_id: The ID of the test run
            tag_name: Name of the tag
            tag_value: Value of the tag
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if the tag already exists
            result = self.connection.execute("""
                SELECT id FROM test_run_tags 
                WHERE run_id = ? AND tag_name = ?
            """, [run_id, tag_name]).fetchone()
            
            if result:
                # Update existing tag
                self.connection.execute("""
                    UPDATE test_run_tags SET
                        tag_value = ?
                    WHERE run_id = ? AND tag_name = ?
                """, [
                    tag_value,
                    run_id,
                    tag_name
                ])
            else:
                # Insert new tag
                self.connection.execute("""
                    INSERT INTO test_run_tags (
                        run_id, tag_name, tag_value
                    ) VALUES (?, ?, ?)
                """, [
                    run_id,
                    tag_name,
                    tag_value
                ])
            
            return True
        
        except Exception as e:
            logger.error(f"Error storing test tag for run {run_id}: {e}")
            return False
    
    def store_test_artifact(self, run_id: str, artifact_data: Dict[str, Any]) -> bool:
        """
        Store a test artifact in the database.
        
        Args:
            run_id: The ID of the test run
            artifact_data: Dictionary containing artifact data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Verify we have the required fields
            if 'artifact_path' not in artifact_data or 'artifact_type' not in artifact_data:
                logger.error("Missing required fields in artifact data")
                return False
            
            # Prepare the values
            values = {
                'run_id': run_id,
                'artifact_type': artifact_data['artifact_type'],
                'artifact_path': artifact_data['artifact_path'],
                'description': artifact_data.get('description', '')
            }
            
            # Check if the artifact already exists
            result = self.connection.execute("""
                SELECT id FROM test_run_artifacts 
                WHERE run_id = ? AND artifact_path = ?
            """, [run_id, values['artifact_path']]).fetchone()
            
            if result:
                # Update existing artifact
                self.connection.execute("""
                    UPDATE test_run_artifacts SET
                        artifact_type = ?,
                        description = ?
                    WHERE run_id = ? AND artifact_path = ?
                """, [
                    values['artifact_type'],
                    values['description'],
                    run_id,
                    values['artifact_path']
                ])
            else:
                # Insert new artifact
                self.connection.execute("""
                    INSERT INTO test_run_artifacts (
                        run_id, artifact_type, artifact_path, description
                    ) VALUES (?, ?, ?, ?)
                """, [
                    run_id,
                    values['artifact_type'],
                    values['artifact_path'],
                    values['description']
                ])
            
            return True
        
        except Exception as e:
            logger.error(f"Error storing test artifact for run {run_id}: {e}")
            return False
    
    def store_batch_test_run(self, batch_data: Dict[str, Any]) -> bool:
        """
        Store a batch test run in the database.
        
        Args:
            batch_data: Dictionary containing batch test run data
            
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
                'run_count': batch_data.get('run_count', 0),
                'started_at': batch_data.get('started_at', datetime.datetime.now()),
                'completed_at': batch_data.get('completed_at'),
                'duration': batch_data.get('duration'),
                'status': batch_data.get('status', 'initializing'),
                'creator': batch_data.get('creator', '')
            }
            
            # Check if the batch already exists
            result = self.connection.execute(
                "SELECT batch_id FROM batch_test_runs WHERE batch_id = ?", 
                [values['batch_id']]
            ).fetchone()
            
            if result:
                # Update existing batch
                self.connection.execute("""
                    UPDATE batch_test_runs SET
                        description = ?,
                        run_count = ?,
                        started_at = ?,
                        completed_at = ?,
                        duration = ?,
                        status = ?,
                        creator = ?
                    WHERE batch_id = ?
                """, [
                    values['description'],
                    values['run_count'],
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
                    INSERT INTO batch_test_runs (
                        batch_id, description, run_count, started_at,
                        completed_at, duration, status, creator
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    values['batch_id'],
                    values['description'],
                    values['run_count'],
                    values['started_at'],
                    values['completed_at'],
                    values['duration'],
                    values['status'],
                    values['creator']
                ])
            
            return True
        
        except Exception as e:
            logger.error(f"Error storing batch test run {batch_data.get('batch_id')}: {e}")
            return False
    
    def add_run_to_batch(self, run_id: str, batch_id: str) -> bool:
        """
        Add a test run to a batch.
        
        Args:
            run_id: The ID of the test run
            batch_id: The ID of the batch
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if the relation already exists
            result = self.connection.execute("""
                SELECT id FROM test_run_batch 
                WHERE run_id = ? AND batch_id = ?
            """, [run_id, batch_id]).fetchone()
            
            if not result:
                # Insert new relation
                self.connection.execute("""
                    INSERT INTO test_run_batch (
                        run_id, batch_id
                    ) VALUES (?, ?)
                """, [
                    run_id,
                    batch_id
                ])
            
            return True
        
        except Exception as e:
            logger.error(f"Error adding run {run_id} to batch {batch_id}: {e}")
            return False
    
    def get_test_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a test run from the database.
        
        Args:
            run_id: The ID of the test run
            
        Returns:
            Dictionary containing test run data, or None if not found
        """
        try:
            # Get the test run
            result = self.connection.execute("""
                SELECT * FROM test_runs WHERE run_id = ?
            """, [run_id]).fetchone()
            
            if not result:
                return None
            
            # Convert to dictionary
            run_data = dict(result)
            
            # Convert hardware from string to list if needed
            if run_data.get('hardware') and run_data['hardware'].startswith('['):
                try:
                    run_data['hardware'] = json.loads(run_data['hardware'])
                except:
                    pass
            
            # Get results for this run
            results = self.connection.execute("""
                SELECT * FROM test_run_results 
                WHERE run_id = ?
            """, [run_id]).fetchone()
            
            if results:
                result_dict = dict(results)
                if result_dict.get('result_data'):
                    try:
                        run_data['results'] = json.loads(result_dict['result_data'])
                    except:
                        run_data['results'] = {
                            'tests_passed': result_dict.get('tests_passed', 0),
                            'tests_failed': result_dict.get('tests_failed', 0),
                            'tests_skipped': result_dict.get('tests_skipped', 0)
                        }
                else:
                    run_data['results'] = {
                        'tests_passed': result_dict.get('tests_passed', 0),
                        'tests_failed': result_dict.get('tests_failed', 0),
                        'tests_skipped': result_dict.get('tests_skipped', 0)
                    }
            
            # Get steps for this run
            steps = self.connection.execute("""
                SELECT * FROM test_run_steps 
                WHERE run_id = ? 
                ORDER BY started_at
            """, [run_id]).fetchall()
            
            if steps:
                run_data['steps'] = [dict(step) for step in steps]
            
            # Get metrics for this run
            metrics = self.connection.execute("""
                SELECT metric_name, metric_value, metric_unit 
                FROM test_run_metrics 
                WHERE run_id = ?
            """, [run_id]).fetchall()
            
            if metrics:
                run_data['metrics'] = {
                    metric['metric_name']: {
                        'value': metric['metric_value'],
                        'unit': metric['metric_unit']
                    }
                    for metric in metrics
                }
            
            # Get tags for this run
            tags = self.connection.execute("""
                SELECT tag_name, tag_value 
                FROM test_run_tags 
                WHERE run_id = ?
            """, [run_id]).fetchall()
            
            if tags:
                run_data['tags'] = {
                    tag['tag_name']: tag['tag_value']
                    for tag in tags
                }
            
            # Get artifacts for this run
            artifacts = self.connection.execute("""
                SELECT artifact_type, artifact_path, description 
                FROM test_run_artifacts 
                WHERE run_id = ?
            """, [run_id]).fetchall()
            
            if artifacts:
                run_data['artifacts'] = [dict(artifact) for artifact in artifacts]
            
            # Get batch ID for this run
            batch = self.connection.execute("""
                SELECT batch_id 
                FROM test_run_batch 
                WHERE run_id = ?
            """, [run_id]).fetchone()
            
            if batch:
                run_data['batch_id'] = batch[0]
            
            return run_data
        
        except Exception as e:
            logger.error(f"Error getting test run {run_id}: {e}")
            return None
    
    def get_batch_test_run(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a batch test run from the database.
        
        Args:
            batch_id: The ID of the batch
            
        Returns:
            Dictionary containing batch test run data, or None if not found
        """
        try:
            # Get the batch
            result = self.connection.execute("""
                SELECT * FROM batch_test_runs WHERE batch_id = ?
            """, [batch_id]).fetchone()
            
            if not result:
                return None
            
            # Convert to dictionary
            batch_data = dict(result)
            
            # Get runs for this batch
            run_ids = self.connection.execute("""
                SELECT run_id
                FROM test_run_batch 
                WHERE batch_id = ?
            """, [batch_id]).fetchall()
            
            if run_ids:
                runs = []
                for run_id in [r[0] for r in run_ids]:
                    run_summary = self.connection.execute("""
                        SELECT t.run_id, t.model_name, t.status, 
                               t.started_at, t.completed_at, t.duration,
                               r.tests_passed, r.tests_failed
                        FROM test_runs t
                        LEFT JOIN test_run_results r ON t.run_id = r.run_id
                        WHERE t.run_id = ?
                    """, [run_id]).fetchone()
                    
                    if run_summary:
                        runs.append(dict(run_summary))
                
                batch_data['runs'] = runs
            
            return batch_data
        
        except Exception as e:
            logger.error(f"Error getting batch test run {batch_id}: {e}")
            return None
    
    def list_test_runs(self, limit: int = 100, status: Optional[str] = None, 
                     model_name: Optional[str] = None, batch_id: Optional[str] = None,
                     test_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List test runs from the database.
        
        Args:
            limit: Maximum number of runs to return
            status: Optional status filter
            model_name: Optional model name filter
            batch_id: Optional batch ID filter
            test_type: Optional test type filter
            
        Returns:
            List of dictionaries containing test run data
        """
        try:
            # Build the query
            query = "SELECT * FROM test_run_summary"
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
            
            if test_type:
                where_clauses.append("test_type = ?")
                params.append(test_type)
            
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
            logger.error(f"Error listing test runs: {e}")
            return []
    
    def list_batch_test_runs(self, limit: int = 100, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List batch test runs from the database.
        
        Args:
            limit: Maximum number of batch runs to return
            status: Optional status filter
            
        Returns:
            List of dictionaries containing batch test run data
        """
        try:
            # Build the query
            query = "SELECT * FROM batch_test_summary"
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
            logger.error(f"Error listing batch test runs: {e}")
            return []
    
    def delete_test_run(self, run_id: str) -> bool:
        """
        Delete a test run from the database.
        
        Args:
            run_id: The ID of the test run
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete related records first
            self.connection.execute("DELETE FROM test_run_steps WHERE run_id = ?", [run_id])
            self.connection.execute("DELETE FROM test_run_metrics WHERE run_id = ?", [run_id])
            self.connection.execute("DELETE FROM test_run_tags WHERE run_id = ?", [run_id])
            self.connection.execute("DELETE FROM test_run_artifacts WHERE run_id = ?", [run_id])
            self.connection.execute("DELETE FROM test_run_results WHERE run_id = ?", [run_id])
            self.connection.execute("DELETE FROM test_run_batch WHERE run_id = ?", [run_id])
            
            # Delete the test run
            self.connection.execute("DELETE FROM test_runs WHERE run_id = ?", [run_id])
            
            return True
        
        except Exception as e:
            logger.error(f"Error deleting test run {run_id}: {e}")
            return False
    
    def delete_batch_test_run(self, batch_id: str, delete_runs: bool = False) -> bool:
        """
        Delete a batch test run from the database.
        
        Args:
            batch_id: The ID of the batch
            delete_runs: Whether to delete all runs in the batch
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if delete_runs:
                # Get all run IDs in this batch
                run_ids = self.connection.execute("""
                    SELECT run_id FROM test_run_batch WHERE batch_id = ?
                """, [batch_id]).fetchall()
                
                # Delete each run
                for run_id in [r[0] for r in run_ids]:
                    self.delete_test_run(run_id)
            
            # Delete the batch relations
            self.connection.execute("DELETE FROM test_run_batch WHERE batch_id = ?", [batch_id])
            
            # Delete the batch
            self.connection.execute("DELETE FROM batch_test_runs WHERE batch_id = ?", [batch_id])
            
            return True
        
        except Exception as e:
            logger.error(f"Error deleting batch test run {batch_id}: {e}")
            return False
    
    def get_test_run_count(self, status: Optional[str] = None, 
                         model_name: Optional[str] = None) -> int:
        """
        Get the count of test runs in the database.
        
        Args:
            status: Optional status filter
            model_name: Optional model name filter
            
        Returns:
            Count of test runs
        """
        try:
            # Build the query
            query = "SELECT COUNT(*) FROM test_runs"
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
            logger.error(f"Error getting test run count: {e}")
            return 0
    
    def get_model_statistics(self) -> List[Dict[str, Any]]:
        """
        Get statistics for each model.
        
        Returns:
            List of dictionaries containing model statistics
        """
        try:
            # Execute the query
            results = self.connection.execute("""
                SELECT * FROM model_test_summary
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
                SELECT * FROM hardware_test_summary
            """).fetchall()
            
            # Convert to list of dictionaries
            return [dict(result) for result in results]
        
        except Exception as e:
            logger.error(f"Error getting hardware statistics: {e}")
            return []
    
    def get_test_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get test history for the specified number of days.
        
        Args:
            days: Number of days to include
            
        Returns:
            List of dictionaries containing daily test statistics
        """
        try:
            # Execute the query
            results = self.connection.execute("""
                SELECT 
                    DATE(started_at) as day,
                    COUNT(*) as total_runs,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_runs,
                    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as failed_runs,
                    AVG(duration) as avg_duration
                FROM test_runs
                WHERE started_at >= CURRENT_DATE - ?
                GROUP BY DATE(started_at)
                ORDER BY day
            """, [days]).fetchall()
            
            # Convert to list of dictionaries
            return [dict(result) for result in results]
        
        except Exception as e:
            logger.error(f"Error getting test history: {e}")
            return []
    
    def search_test_runs(self, query_str: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search test runs by query string.
        
        Args:
            query_str: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries containing matching test runs
        """
        try:
            # Prepare the search pattern
            pattern = f"%{query_str}%"
            
            # Execute the query
            results = self.connection.execute("""
                SELECT * FROM test_run_summary
                WHERE model_name LIKE ? 
                   OR hardware LIKE ?
                   OR test_type LIKE ?
                   OR status LIKE ?
                ORDER BY started_at DESC
                LIMIT ?
            """, [pattern, pattern, pattern, pattern, limit]).fetchall()
            
            # Convert to list of dictionaries
            return [dict(result) for result in results]
        
        except Exception as e:
            logger.error(f"Error searching test runs: {e}")
            return []
    
    def get_test_performance_metrics(self, model_name: Optional[str] = None, 
                                   limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get performance metrics for completed test runs.
        
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
                    t.run_id, 
                    t.model_name, 
                    t.hardware, 
                    t.test_type,
                    t.duration,
                    t.completed_at,
                    m.metric_name,
                    m.metric_value,
                    m.metric_unit
                FROM test_runs t
                JOIN test_run_metrics m ON t.run_id = m.run_id
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
            
            # Group by run
            metrics_by_run = {}
            for row in results:
                run_id = row['run_id']
                if run_id not in metrics_by_run:
                    metrics_by_run[run_id] = {
                        'run_id': run_id,
                        'model_name': row['model_name'],
                        'hardware': row['hardware'],
                        'test_type': row['test_type'],
                        'duration': row['duration'],
                        'completed_at': row['completed_at'],
                        'metrics': {}
                    }
                
                metrics_by_run[run_id]['metrics'][row['metric_name']] = {
                    'value': row['metric_value'],
                    'unit': row['metric_unit']
                }
            
            return list(metrics_by_run.values())
        
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
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
            # Get recent test runs
            test_runs = self.list_test_runs(limit=1000)
            
            # Get all batch runs
            batch_runs = self.list_batch_test_runs(limit=1000)
            
            # Create export data
            export_data = {
                'test_runs': test_runs,
                'batch_runs': batch_runs,
                'stats': {
                    'test_run_count': self.get_test_run_count(),
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
            
            # Import test runs
            if 'test_runs' in import_data:
                for run in import_data['test_runs']:
                    self.store_test_run(run)
            
            # Import batch runs
            if 'batch_runs' in import_data:
                for batch in import_data['batch_runs']:
                    self.store_batch_test_run(batch)
            
            return True
        
        except Exception as e:
            logger.error(f"Error importing database from {filename}: {e}")
            return False