#!/usr/bin/env python3
"""
Distributed Testing Framework - Coordinator Server

This module implements the coordinator server for the distributed testing framework,
responsible for managing worker nodes, distributing tasks, and aggregating results.

Core responsibilities:
- Worker node registration and capability tracking
- Task distribution based on worker capabilities
- Worker health monitoring
- Result aggregation and storage
- Administration API
- Job scheduling and prioritization
- Dynamic resource management and adaptive scaling
- Cloud provider integration

Usage:
    python coordinator.py --host 0.0.0.0 --port 8080 --db-path ./benchmark_db.duckdb
"""

import os
import sys
import json
import time
import uuid
import anyio
import logging
import argparse
import threading
import traceback
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import re

# Import the new components
try:
    from duckdb_api.distributed_testing.auto_recovery import AutoRecovery
    AUTO_RECOVERY_AVAILABLE = True
except ImportError:
    logger.warning("Auto Recovery system not available. High availability features disabled.")
    AUTO_RECOVERY_AVAILABLE = False

try:
    from duckdb_api.distributed_testing.performance_trend_analyzer import PerformanceTrendAnalyzer
    PERFORMANCE_ANALYZER_AVAILABLE = True
except ImportError:
    logger.warning("Performance Trend Analyzer not available. Performance analysis features disabled.")
    PERFORMANCE_ANALYZER_AVAILABLE = False

try:
    from duckdb_api.distributed_testing.result_aggregator.service import ResultAggregatorService
    from duckdb_api.distributed_testing.result_aggregator.aggregator import ResultAggregator as DetailedResultAggregator
    RESULT_AGGREGATOR_AVAILABLE = True
except ImportError:
    logger.warning("Result Aggregator not available. Intelligent result aggregation features disabled.")
    RESULT_AGGREGATOR_AVAILABLE = False

# Dynamic Resource Management components
try:
    from duckdb_api.distributed_testing.dynamic_resource_manager import DynamicResourceManager
    DYNAMIC_RESOURCE_MANAGER_AVAILABLE = True
except ImportError:
    logger.warning("Dynamic Resource Manager not available. Advanced resource management features disabled.")
    DYNAMIC_RESOURCE_MANAGER_AVAILABLE = False

try:
    from duckdb_api.distributed_testing.resource_performance_predictor import ResourcePerformancePredictor
    RESOURCE_PREDICTOR_AVAILABLE = True
except ImportError:
    logger.warning("Resource Performance Predictor not available. Resource prediction features disabled.")
    RESOURCE_PREDICTOR_AVAILABLE = False

try:
    from duckdb_api.distributed_testing.cloud_provider_manager import CloudProviderManager
    from duckdb_api.distributed_testing.cloud_provider_integration import (
        AWSCloudProvider, 
        GCPCloudProvider, 
        DockerLocalProvider
    )
    CLOUD_PROVIDER_AVAILABLE = True
except ImportError:
    logger.warning("Cloud Provider Manager not available. Cloud scaling features disabled.")
    CLOUD_PROVIDER_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("coordinator")

# Try to import optional dependencies
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    logger.warning("DuckDB not available. Results will not be stored in database.")
    DUCKDB_AVAILABLE = False

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    logger.warning("websockets not available. WebSocket server will not be available.")
    WEBSOCKETS_AVAILABLE = False
    
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    logger.warning("PyJWT not available. Using simple authentication.")
    JWT_AVAILABLE = False

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Worker node status constants
WORKER_STATUS_REGISTERED = "registered"
WORKER_STATUS_ACTIVE = "active"
WORKER_STATUS_BUSY = "busy"
WORKER_STATUS_UNAVAILABLE = "unavailable"
WORKER_STATUS_DISCONNECTED = "disconnected"

# Task status constants
TASK_STATUS_QUEUED = "queued"
TASK_STATUS_ASSIGNED = "assigned"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"
TASK_STATUS_TIMED_OUT = "timed_out"
TASK_STATUS_CANCELED = "canceled"

# Security constants
TOKEN_EXPIRY_MINUTES = 60
API_KEY_HEADER = "X-API-Key"
AUTH_TOKEN_HEADER = "Authorization"
DEFAULT_API_KEY = "dev_coordinator_key"  # Only for development


class DatabaseManager:
    """Manages interaction with the DuckDB database for the coordinator."""
    
    def __init__(self, db_path: str):
        """Initialize the database manager.
        
        Args:
            db_path: Path to the DuckDB database
        """
        self.db_path = db_path
        self.conn = None
        
        if not DUCKDB_AVAILABLE:
            logger.error("DuckDB is required for database operations")
            raise ImportError("DuckDB is required for database operations")
        
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize the DuckDB connection and create required tables."""
        try:
            self.conn = duckdb.connect(self.db_path)
            self._create_tables()
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    def _create_tables(self):
        """Create the required tables if they don't exist."""
        try:
            # Worker nodes table
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS worker_nodes (
                worker_id VARCHAR PRIMARY KEY,
                hostname VARCHAR,
                registration_time TIMESTAMP,
                last_heartbeat TIMESTAMP,
                status VARCHAR,
                capabilities JSON,
                hardware_metrics JSON,
                tags JSON
            )
            """)
            
            # Distributed tasks table
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS distributed_tasks (
                task_id VARCHAR PRIMARY KEY,
                type VARCHAR,
                priority INTEGER,
                status VARCHAR,
                create_time TIMESTAMP,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                worker_id VARCHAR,
                attempts INTEGER,
                config JSON,
                requirements JSON
            )
            """)
            
            # Task execution history
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS task_execution_history (
                id INTEGER PRIMARY KEY,
                task_id VARCHAR,
                worker_id VARCHAR,
                attempt INTEGER,
                status VARCHAR,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                execution_time_seconds FLOAT,
                error_message VARCHAR,
                hardware_metrics JSON
            )
            """)
            
            # Task results table
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS task_results (
                id INTEGER PRIMARY KEY,
                task_id VARCHAR,
                worker_id VARCHAR,
                timestamp TIMESTAMP,
                results JSON,
                metadata JSON,
                FOREIGN KEY (task_id) REFERENCES distributed_tasks(task_id)
            )
            """)
            
            # Security table for API keys
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                key_id VARCHAR PRIMARY KEY,
                api_key VARCHAR UNIQUE,
                name VARCHAR,
                role VARCHAR,
                created_at TIMESTAMP,
                expires_at TIMESTAMP,
                is_active BOOLEAN
            )
            """)
            
            logger.info("Database tables initialized")
            
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    def add_worker(self, worker_id: str, hostname: str, capabilities: Dict[str, Any],
                   tags: Dict[str, Any] = None):
        """Add a new worker to the database.
        
        Args:
            worker_id: Unique identifier for the worker
            hostname: Hostname of the worker
            capabilities: Dict containing hardware capabilities
            tags: Optional tags for worker categorization
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.conn.execute("""
            INSERT INTO worker_nodes (
                worker_id, hostname, registration_time, last_heartbeat, 
                status, capabilities, hardware_metrics, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                worker_id,
                hostname,
                datetime.now(),
                datetime.now(),
                WORKER_STATUS_REGISTERED,
                json.dumps(capabilities),
                json.dumps({}),
                json.dumps(tags or {})
            ])
            logger.info(f"Added worker {worker_id} to database")
            return True
        except Exception as e:
            logger.error(f"Error adding worker to database: {e}")
            return False
    
    def update_worker_status(self, worker_id: str, status: str):
        """Update the status of a worker.
        
        Args:
            worker_id: ID of the worker to update
            status: New status for the worker
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.conn.execute("""
            UPDATE worker_nodes 
            SET status = ?, last_heartbeat = ?
            WHERE worker_id = ?
            """, [status, datetime.now(), worker_id])
            logger.debug(f"Updated worker {worker_id} status to {status}")
            return True
        except Exception as e:
            logger.error(f"Error updating worker status: {e}")
            return False
    
    def update_worker_heartbeat(self, worker_id: str):
        """Update the heartbeat timestamp for a worker.
        
        Args:
            worker_id: ID of the worker to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.conn.execute("""
            UPDATE worker_nodes 
            SET last_heartbeat = ?
            WHERE worker_id = ?
            """, [datetime.now(), worker_id])
            return True
        except Exception as e:
            logger.error(f"Error updating worker heartbeat: {e}")
            return False
    
    def get_worker(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """Get worker information from the database.
        
        Args:
            worker_id: ID of the worker to retrieve
            
        Returns:
            Dict containing worker information or None if not found
        """
        try:
            result = self.conn.execute("""
            SELECT worker_id, hostname, registration_time, last_heartbeat, 
                   status, capabilities, hardware_metrics, tags
            FROM worker_nodes
            WHERE worker_id = ?
            """, [worker_id]).fetchone()
            
            if not result:
                return None
                
            # Convert capabilities and tags from JSON
            worker_info = {
                "worker_id": result[0],
                "hostname": result[1],
                "registration_time": result[2],
                "last_heartbeat": result[3],
                "status": result[4],
                "capabilities": json.loads(result[5]),
                "hardware_metrics": json.loads(result[6]),
                "tags": json.loads(result[7])
            }
            
            return worker_info
        except Exception as e:
            logger.error(f"Error retrieving worker: {e}")
            return None
    
    def get_available_workers(self) -> List[Dict[str, Any]]:
        """Get a list of available worker nodes.
        
        Returns:
            List of dicts containing worker information
        """
        try:
            results = self.conn.execute("""
            SELECT worker_id, hostname, registration_time, last_heartbeat, 
                   status, capabilities, hardware_metrics, tags
            FROM worker_nodes
            WHERE status IN (?, ?)
            """, [WORKER_STATUS_ACTIVE, WORKER_STATUS_REGISTERED]).fetchall()
            
            workers = []
            for result in results:
                worker_info = {
                    "worker_id": result[0],
                    "hostname": result[1],
                    "registration_time": result[2],
                    "last_heartbeat": result[3],
                    "status": result[4],
                    "capabilities": json.loads(result[5]),
                    "hardware_metrics": json.loads(result[6]),
                    "tags": json.loads(result[7])
                }
                workers.append(worker_info)
            
            return workers
        except Exception as e:
            logger.error(f"Error retrieving available workers: {e}")
            return []
    
    def add_task(self, task_id: str, task_type: str, priority: int, 
                 config: Dict[str, Any], requirements: Dict[str, Any]) -> bool:
        """Add a new task to the database.
        
        Args:
            task_id: Unique identifier for the task
            task_type: Type of task (benchmark, test, etc.)
            priority: Priority of the task (lower is higher priority)
            config: Configuration for the task
            requirements: Hardware requirements for the task
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.conn.execute("""
            INSERT INTO distributed_tasks (
                task_id, type, priority, status, create_time, 
                attempts, config, requirements
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                task_id,
                task_type,
                priority,
                TASK_STATUS_QUEUED,
                datetime.now(),
                0,
                json.dumps(config),
                json.dumps(requirements)
            ])
            logger.info(f"Added task {task_id} to database")
            return True
        except Exception as e:
            logger.error(f"Error adding task to database: {e}")
            return False
    
    def update_task_status(self, task_id: str, status: str, 
                          worker_id: Optional[str] = None) -> bool:
        """Update the status of a task.
        
        Args:
            task_id: ID of the task to update
            status: New status for the task
            worker_id: ID of the worker assigned to the task (if any)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if status == TASK_STATUS_ASSIGNED and worker_id:
                self.conn.execute("""
                UPDATE distributed_tasks 
                SET status = ?, worker_id = ?, start_time = ?
                WHERE task_id = ?
                """, [status, worker_id, datetime.now(), task_id])
            elif status == TASK_STATUS_COMPLETED or status == TASK_STATUS_FAILED:
                self.conn.execute("""
                UPDATE distributed_tasks 
                SET status = ?, end_time = ?
                WHERE task_id = ?
                """, [status, datetime.now(), task_id])
            else:
                self.conn.execute("""
                UPDATE distributed_tasks 
                SET status = ?
                WHERE task_id = ?
                """, [status, task_id])
                
            logger.debug(f"Updated task {task_id} status to {status}")
            return True
        except Exception as e:
            logger.error(f"Error updating task status: {e}")
            return False
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task information from the database.
        
        Args:
            task_id: ID of the task to retrieve
            
        Returns:
            Dict containing task information or None if not found
        """
        try:
            result = self.conn.execute("""
            SELECT task_id, type, priority, status, create_time, 
                   start_time, end_time, worker_id, attempts, 
                   config, requirements
            FROM distributed_tasks
            WHERE task_id = ?
            """, [task_id]).fetchone()
            
            if not result:
                return None
                
            task_info = {
                "task_id": result[0],
                "type": result[1],
                "priority": result[2],
                "status": result[3],
                "create_time": result[4],
                "start_time": result[5],
                "end_time": result[6],
                "worker_id": result[7],
                "attempts": result[8],
                "config": json.loads(result[9]),
                "requirements": json.loads(result[10])
            }
            
            return task_info
        except Exception as e:
            logger.error(f"Error retrieving task: {e}")
            return None
    
    def get_pending_tasks(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get a list of pending tasks, ordered by priority.
        
        Args:
            limit: Maximum number of tasks to retrieve
            
        Returns:
            List of dicts containing task information
        """
        try:
            results = self.conn.execute("""
            SELECT task_id, type, priority, status, create_time, 
                   start_time, end_time, worker_id, attempts, 
                   config, requirements
            FROM distributed_tasks
            WHERE status = ?
            ORDER BY priority ASC, create_time ASC
            LIMIT ?
            """, [TASK_STATUS_QUEUED, limit]).fetchall()
            
            tasks = []
            for result in results:
                task_info = {
                    "task_id": result[0],
                    "type": result[1],
                    "priority": result[2],
                    "status": result[3],
                    "create_time": result[4],
                    "start_time": result[5],
                    "end_time": result[6],
                    "worker_id": result[7],
                    "attempts": result[8],
                    "config": json.loads(result[9]),
                    "requirements": json.loads(result[10])
                }
                tasks.append(task_info)
            
            return tasks
        except Exception as e:
            logger.error(f"Error retrieving pending tasks: {e}")
            return []
    
    def add_task_result(self, task_id: str, worker_id: str, 
                       results: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
        """Add task results to the database.
        
        Args:
            task_id: ID of the task
            worker_id: ID of the worker that executed the task
            results: Results of the task
            metadata: Metadata about the task execution
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.conn.execute("""
            INSERT INTO task_results (
                task_id, worker_id, timestamp, results, metadata
            ) VALUES (?, ?, ?, ?, ?)
            """, [
                task_id,
                worker_id,
                datetime.now(),
                json.dumps(results),
                json.dumps(metadata)
            ])
            logger.info(f"Added results for task {task_id} from worker {worker_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding task results: {e}")
            return False
    
    def add_execution_history(self, task_id: str, worker_id: str, attempt: int,
                             status: str, start_time: datetime, end_time: datetime,
                             execution_time: float, error_message: str = "",
                             hardware_metrics: Dict[str, Any] = None) -> bool:
        """Add task execution history to the database.
        
        Args:
            task_id: ID of the task
            worker_id: ID of the worker that executed the task
            attempt: Attempt number
            status: Final status of the execution
            start_time: Start time of the execution
            end_time: End time of the execution
            execution_time: Execution time in seconds
            error_message: Error message (if any)
            hardware_metrics: Hardware metrics during execution
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.conn.execute("""
            INSERT INTO task_execution_history (
                task_id, worker_id, attempt, status, start_time, end_time,
                execution_time_seconds, error_message, hardware_metrics
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                task_id,
                worker_id,
                attempt,
                status,
                start_time,
                end_time,
                execution_time,
                error_message,
                json.dumps(hardware_metrics or {})
            ])
            logger.info(f"Added execution history for task {task_id}, attempt {attempt}")
            return True
        except Exception as e:
            logger.error(f"Error adding execution history: {e}")
            return False
    
    def store_api_key(self, key_id: str, api_key: str, name: str, role: str,
                     expires_at: Optional[datetime] = None) -> bool:
        """Store an API key in the database.
        
        Args:
            key_id: Unique identifier for the key
            api_key: The API key value
            name: Name of the key
            role: Role associated with the key
            expires_at: Expiration date (or None for no expiration)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.conn.execute("""
            INSERT INTO api_keys (
                key_id, api_key, name, role, created_at, expires_at, is_active
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                key_id,
                api_key,
                name,
                role,
                datetime.now(),
                expires_at,
                True
            ])
            logger.info(f"Stored API key {key_id} for {name}")
            return True
        except Exception as e:
            logger.error(f"Error storing API key: {e}")
            return False
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate an API key and return associated information.
        
        Args:
            api_key: The API key to validate
            
        Returns:
            Dict with key information if valid, None otherwise
        """
        try:
            result = self.conn.execute("""
            SELECT key_id, name, role, created_at, expires_at, is_active
            FROM api_keys
            WHERE api_key = ?
            """, [api_key]).fetchone()
            
            if not result:
                return None
                
            key_info = {
                "key_id": result[0],
                "name": result[1],
                "role": result[2],
                "created_at": result[3],
                "expires_at": result[4],
                "is_active": result[5]
            }
            
            # Check if key is active and not expired
            if not key_info["is_active"]:
                logger.warning(f"API key {key_info['key_id']} is inactive")
                return None
                
            if key_info["expires_at"] and key_info["expires_at"] < datetime.now():
                logger.warning(f"API key {key_info['key_id']} has expired")
                return None
                
            return key_info
        except Exception as e:
            logger.error(f"Error validating API key: {e}")
            return None
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


class SecurityManager:
    """Manages security for the coordinator server."""
    
    def __init__(self, db_manager: DatabaseManager = None, token_secret: str = None):
        """Initialize the security manager.
        
        Args:
            db_manager: Database manager for storing/retrieving API keys
            token_secret: Secret for signing JWT tokens (auto-generated if None)
        """
        self.db_manager = db_manager
        self.token_secret = token_secret or str(uuid.uuid4())
        self.worker_tokens = {}  # worker_id -> token info
        
        # Create default API key if not in database
        if self.db_manager and not self._has_default_key():
            self._create_default_key()
    
    def _has_default_key(self) -> bool:
        """Check if default API key exists in database."""
        if not self.db_manager:
            return False
        
        key_info = self.db_manager.validate_api_key(DEFAULT_API_KEY)
        return key_info is not None
    
    def _create_default_key(self):
        """Create default API key in database."""
        key_id = "default"
        name = "Default Coordinator Key"
        role = "admin"
        
        # Create key with 1 year expiration
        expires_at = datetime.now() + timedelta(days=365)
        
        self.db_manager.store_api_key(key_id, DEFAULT_API_KEY, name, role, expires_at)
        logger.warning(f"Created default API key: {DEFAULT_API_KEY} (for development only)")
    
    def generate_worker_key(self, name: str = None, role: str = "worker") -> Dict[str, Any]:
        """Generate an API key for a worker.
        
        Args:
            name: Name for the key
            role: Role for the key
            
        Returns:
            Dict containing key information
        """
        key_id = f"worker_{uuid.uuid4()}"
        name = name or f"Worker Key {key_id}"
        api_key = f"wk_{uuid.uuid4().hex}"
        
        # Store key in database if available
        if self.db_manager:
            self.db_manager.store_api_key(key_id, api_key, name, role)
        
        key_info = {
            "key_id": key_id,
            "api_key": api_key,
            "name": name,
            "role": role
        }
        
        return key_info
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate an API key.
        
        Args:
            api_key: The API key to validate
            
        Returns:
            Dict with key information if valid, None otherwise
        """
        # Check against default key
        if api_key == DEFAULT_API_KEY:
            return {
                "key_id": "default",
                "name": "Default Coordinator Key",
                "role": "admin"
            }
        
        # Check database if available
        if self.db_manager:
            return self.db_manager.validate_api_key(api_key)
            
        return None
    
    def generate_token(self, worker_id: str, role: str = "worker") -> Optional[str]:
        """Generate a JWT token for a worker.
        
        Args:
            worker_id: ID of the worker
            role: Role for the token
            
        Returns:
            JWT token string if JWT is available, None otherwise
        """
        if not JWT_AVAILABLE:
            logger.warning("JWT not available, cannot generate token")
            return None
            
        token_data = {
            "sub": worker_id,
            "role": role,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRY_MINUTES)
        }
        
        token = jwt.encode(token_data, self.token_secret, algorithm="HS256")
        
        # Store token info
        self.worker_tokens[worker_id] = {
            "token": token,
            "expires_at": token_data["exp"]
        }
        
        return token
    
    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate a JWT token.
        
        Args:
            token: The JWT token to validate
            
        Returns:
            Dict with token claims if valid, None otherwise
        """
        if not JWT_AVAILABLE:
            logger.warning("JWT not available, cannot validate token")
            return None
            
        try:
            token_data = jwt.decode(token, self.token_secret, algorithms=["HS256"])
            return token_data
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None


class TaskManager:
    """Manages task scheduling and distribution."""
    
    def __init__(self, db_manager: DatabaseManager = None):
        """Initialize the task manager.
        
        Args:
            db_manager: Database manager for storing/retrieving tasks
        """
        self.db_manager = db_manager
        self.task_queue = []  # [(priority, create_time, task_id, task)]
        self.running_tasks = {}  # task_id -> worker_id
        self.task_lock = threading.Lock()
        
        # Load pending tasks from database
        self._load_pending_tasks()
    
    def _load_pending_tasks(self):
        """Load pending tasks from database."""
        if not self.db_manager:
            return
            
        pending_tasks = self.db_manager.get_pending_tasks()
        
        with self.task_lock:
            for task in pending_tasks:
                task_id = task["task_id"]
                priority = task["priority"]
                create_time = task["create_time"]
                
                self.task_queue.append((priority, create_time, task_id, task))
                
            # Sort by priority, then create_time
            self.task_queue.sort()
            
        logger.info(f"Loaded {len(pending_tasks)} pending tasks from database")
    
    def add_task(self, task_id: str, task_type: str, priority: int, 
                config: Dict[str, Any], requirements: Dict[str, Any]) -> str:
        """Add a new task to the queue.
        
        Args:
            task_id: Unique identifier for the task (or None to generate)
            task_type: Type of task (benchmark, test, etc.)
            priority: Priority of the task (lower is higher priority)
            config: Configuration for the task
            requirements: Hardware requirements for the task
            
        Returns:
            Task ID
        """
        # Generate task ID if not provided
        if not task_id:
            task_id = f"task_{uuid.uuid4()}"
            
        task = {
            "task_id": task_id,
            "type": task_type,
            "priority": priority,
            "status": TASK_STATUS_QUEUED,
            "create_time": datetime.now(),
            "config": config,
            "requirements": requirements,
            "attempts": 0
        }
        
        # Add to database if available
        if self.db_manager:
            self.db_manager.add_task(task_id, task_type, priority, config, requirements)
        
        # Add to queue
        with self.task_lock:
            create_time = task["create_time"]
            self.task_queue.append((priority, create_time, task_id, task))
            self.task_queue.sort()  # Sort by priority, then create_time
        
        logger.info(f"Added task {task_id} to queue with priority {priority}")
        return task_id
    
    def get_next_task(self, worker_id: str, 
                     worker_capabilities: Dict[str, Any], worker_resources: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Get the next task for a worker based on capabilities and resources.
        
        Args:
            worker_id: ID of the worker
            worker_capabilities: Capabilities of the worker
            worker_resources: Detailed resource information for dynamic resource management
            
        Returns:
            Task dict if a suitable task is found, None otherwise
        """
        # Get the parent CoordinatorServer instance to access dynamic_resource_manager
        coordinator = self._get_coordinator_server()
        dynamic_resource_mgr = None
        resource_predictor = None
        
        if coordinator:
            dynamic_resource_mgr = getattr(coordinator, 'dynamic_resource_manager', None)
            resource_predictor = getattr(coordinator, 'resource_predictor', None)
        
        with self.task_lock:
            if not self.task_queue:
                return None
            
            # Initialize fitness scores for task-worker matching
            task_fitness_scores = []
            
            # Calculate fitness scores for each task
            for i, (priority, create_time, task_id, task) in enumerate(self.task_queue):
                # First check if the worker meets basic requirements
                if not self._worker_meets_requirements(worker_capabilities, task["requirements"]):
                    continue
                
                # If we have a dynamic resource manager, check resource availability
                reservation_possible = True
                fitness_score = 1.0  # Default baseline score
                
                if dynamic_resource_mgr and worker_resources:
                    task_resources = self._estimate_task_resources(task, resource_predictor)
                    
                    # Check if worker has enough resources
                    try:
                        # Calculate fitness score (0.0-1.0) based on how well the resources match
                        fitness_score = dynamic_resource_mgr.calculate_task_worker_fitness(
                            worker_id, task_resources
                        )
                        
                        # Check if reservation is possible
                        reservation_possible = dynamic_resource_mgr.check_resource_availability(
                            worker_id, task_resources
                        )
                        
                        logger.debug(f"Task {task_id} fitness score for worker {worker_id}: {fitness_score:.2f}")
                    except Exception as e:
                        logger.error(f"Error calculating resource fitness: {e}")
                        reservation_possible = True  # Fall back to basic scheduling
                
                if reservation_possible:
                    # Add to list of potential tasks with fitness score
                    task_fitness_scores.append((i, fitness_score, priority, create_time, task_id, task))
            
            # Sort by fitness score (descending), then priority (ascending), then create_time (ascending)
            if task_fitness_scores:
                task_fitness_scores.sort(key=lambda x: (-x[1], x[2], x[3]))
                
                # Get the best matching task
                task_index, fitness_score, _, _, task_id, matching_task = task_fitness_scores[0]
                
                # Remove from queue
                self.task_queue.pop(task_index)
                
                # Mark as assigned
                matching_task["status"] = TASK_STATUS_ASSIGNED
                matching_task["worker_id"] = worker_id
                matching_task["start_time"] = datetime.now()
                matching_task["attempts"] += 1
                
                # Track in running tasks
                self.running_tasks[matching_task["task_id"]] = worker_id
                
                # Reserve resources if dynamic resource manager is available
                if dynamic_resource_mgr and worker_resources:
                    task_resources = self._estimate_task_resources(matching_task, resource_predictor)
                    try:
                        # Create resource reservation
                        reservation_id = dynamic_resource_mgr.reserve_resources(
                            worker_id=worker_id,
                            task_id=matching_task["task_id"],
                            resource_requirements=task_resources
                        )
                        # Store reservation ID in task for later release
                        matching_task["resource_reservation_id"] = reservation_id
                        logger.debug(f"Reserved resources for task {matching_task['task_id']} on worker {worker_id}")
                    except Exception as e:
                        logger.error(f"Error reserving resources: {e}")
                
                # Update in database if available
                if self.db_manager:
                    self.db_manager.update_task_status(
                        matching_task["task_id"], 
                        TASK_STATUS_ASSIGNED, 
                        worker_id
                    )
                
                logger.info(f"Assigned task {matching_task['task_id']} to worker {worker_id} with fitness {fitness_score:.2f}")
                return matching_task
            
            return None
    
    def _estimate_task_resources(self, task: Dict[str, Any], resource_predictor=None) -> Dict[str, Any]:
        """Estimate resource requirements for a task.
        
        Args:
            task: Task configuration
            resource_predictor: Optional resource predictor for ML-based estimation
            
        Returns:
            Dict with estimated resource requirements
        """
        # Extract relevant task parameters for resource prediction
        task_type = task.get("type", "unknown")
        model = task.get("config", {}).get("model", "unknown")
        batch_size = task.get("config", {}).get("batch_size", 1)
        precision = task.get("config", {}).get("precision", "fp32")
        
        # Default resource requirements (conservative estimates)
        resource_requirements = {
            "cpu_cores": 2,
            "memory_mb": 4096,
            "gpu_memory_mb": 0
        }
        
        # If using a resource predictor, get ML-based prediction
        if resource_predictor:
            try:
                predicted_resources = resource_predictor.predict_resource_requirements(
                    task_type=task_type,
                    model=model,
                    batch_size=batch_size,
                    precision=precision
                )
                
                if predicted_resources:
                    resource_requirements = predicted_resources
                    logger.debug(f"Using ML-predicted resources for task {task['task_id']}: {resource_requirements}")
            except Exception as e:
                logger.error(f"Error predicting resources: {e}")
        
        # Use explicit requirements from task if specified (override predictions)
        explicit_requirements = task.get("requirements", {})
        
        if "cpu_cores" in explicit_requirements:
            resource_requirements["cpu_cores"] = explicit_requirements["cpu_cores"]
            
        if "memory_mb" in explicit_requirements:
            resource_requirements["memory_mb"] = explicit_requirements["memory_mb"]
            
        if "gpu_memory_mb" in explicit_requirements:
            resource_requirements["gpu_memory_mb"] = explicit_requirements["gpu_memory_mb"]
        
        return resource_requirements
    
    def _get_coordinator_server(self):
        """Get the parent CoordinatorServer instance.
        
        Returns:
            CoordinatorServer instance or None
        """
        # This is a helper method to access the parent CoordinatorServer
        # which contains the dynamic_resource_manager
        try:
            frame = sys._getframe(1)
            while frame:
                if 'self' in frame.f_locals:
                    instance = frame.f_locals['self']
                    if isinstance(instance, CoordinatorServer):
                        return instance
                frame = frame.f_back
        except Exception:
            pass
        return None
    
    def _worker_meets_requirements(self, worker_capabilities: Dict[str, Any],
                                 task_requirements: Dict[str, Any]) -> bool:
        """Check if a worker meets the requirements for a task.
        
        Args:
            worker_capabilities: Worker's hardware capabilities
            task_requirements: Task's hardware requirements
            
        Returns:
            True if worker meets requirements, False otherwise
        """
        # Check hardware requirements
        if "hardware" in task_requirements:
            required_hardware = task_requirements["hardware"]
            if isinstance(required_hardware, list):
                # Check if worker has any of the required hardware
                worker_hardware = worker_capabilities.get("hardware_types", [])
                if not any(hw in worker_hardware for hw in required_hardware):
                    return False
            elif isinstance(required_hardware, str):
                # Check if worker has the required hardware
                worker_hardware = worker_capabilities.get("hardware_types", [])
                if required_hardware not in worker_hardware:
                    return False
        
        # Check minimum memory
        if "min_memory_gb" in task_requirements:
            min_memory = task_requirements["min_memory_gb"]
            worker_memory = worker_capabilities.get("memory_gb", 0)
            if worker_memory < min_memory:
                return False
        
        # Check minimum CUDA compute capability
        if "min_cuda_compute" in task_requirements:
            min_cuda = task_requirements["min_cuda_compute"]
            worker_cuda = worker_capabilities.get("cuda_compute", 0)
            if worker_cuda < min_cuda:
                return False
        
        # Check for specific browser requirements
        if "browser" in task_requirements:
            required_browser = task_requirements["browser"]
            available_browsers = worker_capabilities.get("browsers", [])
            if required_browser not in available_browsers:
                return False
        
        # Check for specific device requirements (mobile, etc.)
        if "device_type" in task_requirements:
            required_device = task_requirements["device_type"]
            worker_device = worker_capabilities.get("device_type")
            if worker_device != required_device:
                return False
        
        return True
    
    def complete_task(self, task_id: str, worker_id: str, 
                     results: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
        """Mark a task as completed and store results.
        
        Args:
            task_id: ID of the task
            worker_id: ID of the worker that executed the task
            results: Results of the task
            metadata: Metadata about the task execution
            
        Returns:
            True if successful, False otherwise
        """
        # Verify this task is assigned to this worker
        with self.task_lock:
            if task_id not in self.running_tasks:
                logger.warning(f"Task {task_id} not found in running tasks")
                return False
                
            if self.running_tasks[task_id] != worker_id:
                logger.warning(
                    f"Task {task_id} is assigned to {self.running_tasks[task_id]}, "
                    f"not {worker_id}"
                )
                return False
                
            # Remove from running tasks
            del self.running_tasks[task_id]
            
            # Release resources if there was a reservation
            task = self.db_manager.get_task(task_id) if self.db_manager else None
            resource_reservation_id = metadata.get("resource_reservation_id")
            
            if resource_reservation_id or (task and "resource_reservation_id" in task):
                coordinator = self._get_coordinator_server()
                if coordinator and coordinator.dynamic_resource_manager:
                    try:
                        # Get the reservation ID either from metadata or task
                        reservation_id = resource_reservation_id
                        if not reservation_id and task:
                            reservation_id = task.get("resource_reservation_id")
                            
                        if reservation_id:
                            coordinator.dynamic_resource_manager.release_resources(reservation_id)
                            logger.debug(f"Released resources for completed task {task_id}")
                    except Exception as e:
                        logger.error(f"Error releasing resources for task {task_id}: {e}")
        
        # Update task status in database
        if self.db_manager:
            self.db_manager.update_task_status(task_id, TASK_STATUS_COMPLETED)
            
            # Store results
            self.db_manager.add_task_result(task_id, worker_id, results, metadata)
            
            # Add execution history
            start_time = metadata.get("start_time")
            end_time = metadata.get("end_time")
            
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                
            if isinstance(end_time, str):
                end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                
            if not start_time:
                start_time = datetime.now() - timedelta(seconds=metadata.get("execution_time", 0))
                
            if not end_time:
                end_time = datetime.now()
                
            execution_time = metadata.get("execution_time", 0)
            hardware_metrics = metadata.get("hardware_metrics", {})
            
            self.db_manager.add_execution_history(
                task_id, worker_id, metadata.get("attempt", 1),
                TASK_STATUS_COMPLETED, start_time, end_time,
                execution_time, "", hardware_metrics
            )
        
        # Create test result record for aggregation
        test_result = None
        
        try:
            # Get task and worker information
            task = self.db_manager.get_task(task_id) if self.db_manager else {}
            worker = self.db_manager.get_worker(worker_id) if self.db_manager else {}
            
            # Prepare test result for aggregation
            test_result = {
                "test_id": task_id,
                "worker_id": worker_id,
                "task_type": task.get("type") if task else metadata.get("task_type", "unknown"),
                "hardware_id": worker.get("hardware_id") if worker else metadata.get("hardware_id", "unknown"),
                "model": metadata.get("model", "unknown"),
                "batch_size": metadata.get("batch_size", 1),
                "precision": metadata.get("precision", "fp32"),
                "status": "success",
                "timestamp": end_time if isinstance(end_time, datetime) else datetime.now(),
                "duration": execution_time,
                "hardware_metrics": hardware_metrics
            }
            
            # Add result metrics
            for key, value in results.items():
                if isinstance(value, (int, float, bool, str)):
                    test_result[key] = value
                    
        except Exception as e:
            logger.error(f"Error preparing result data for task {task_id}: {e}")
        
        # Process with the dual-layer result aggregation system
        if test_result:
            # First prepare the result to ensure it has all required fields
            # This avoids duplicating preparation logic between aggregators
            prepared_result = self._prepare_result_for_aggregation(test_result)
            
            # Add to high-level result aggregator service if available
            if hasattr(self, 'result_aggregator') and self.result_aggregator:
                try:
                    # Process in high-level result aggregator
                    self.result_aggregator.process_test_result(prepared_result)
                    logger.debug(f"Added task {task_id} to high-level result aggregator")
                except Exception as e:
                    logger.error(f"Error adding task {task_id} to high-level result aggregator: {e}")
                    logger.debug(f"Result data: {prepared_result}")
            
            # Add to detailed result aggregator if available
            if hasattr(self, 'detailed_result_aggregator') and self.detailed_result_aggregator:
                try:
                    # Process in detailed result aggregator
                    self.detailed_result_aggregator.process_test_result(prepared_result)
                    logger.debug(f"Added task {task_id} to detailed result aggregator")
                except Exception as e:
                    logger.error(f"Error adding task {task_id} to detailed result aggregator: {e}")
                    logger.debug(f"Result data: {prepared_result}")
        
        logger.info(f"Task {task_id} completed by worker {worker_id}")
        return True
    
    def fail_task(self, task_id: str, worker_id: str, 
                 error: str, metadata: Dict[str, Any]) -> bool:
        """Mark a task as failed.
        
        Args:
            task_id: ID of the task
            worker_id: ID of the worker that executed the task
            error: Error message
            metadata: Metadata about the task execution
            
        Returns:
            True if successful, False otherwise
        """
        # Verify this task is assigned to this worker
        with self.task_lock:
            if task_id not in self.running_tasks:
                logger.warning(f"Task {task_id} not found in running tasks")
                return False
                
            if self.running_tasks[task_id] != worker_id:
                logger.warning(
                    f"Task {task_id} is assigned to {self.running_tasks[task_id]}, "
                    f"not {worker_id}"
                )
                return False
                
            # Remove from running tasks
            del self.running_tasks[task_id]
            
            # Release resources if there was a reservation
            task = self.db_manager.get_task(task_id) if self.db_manager else None
            resource_reservation_id = metadata.get("resource_reservation_id")
            
            if resource_reservation_id or (task and "resource_reservation_id" in task):
                coordinator = self._get_coordinator_server()
                if coordinator and coordinator.dynamic_resource_manager:
                    try:
                        # Get the reservation ID either from metadata or task
                        reservation_id = resource_reservation_id
                        if not reservation_id and task:
                            reservation_id = task.get("resource_reservation_id")
                            
                        if reservation_id:
                            coordinator.dynamic_resource_manager.release_resources(reservation_id)
                            logger.debug(f"Released resources for failed task {task_id}")
                    except Exception as e:
                        logger.error(f"Error releasing resources for task {task_id}: {e}")
            
            # Check if we should retry
            task = self.db_manager.get_task(task_id) if self.db_manager else None
            
            if task and task["attempts"] < metadata.get("max_retries", 3):
                # Requeue task
                priority = task["priority"]
                create_time = datetime.now()  # Update create time to avoid priority inversion
                
                # Update attempts count
                task["attempts"] += 1
                
                # Add back to queue
                self.task_queue.append((priority, create_time, task_id, task))
                self.task_queue.sort()
                
                logger.info(f"Requeued task {task_id} after failure (attempt {task['attempts']})")
                
                # Update status in database
                if self.db_manager:
                    self.db_manager.update_task_status(task_id, TASK_STATUS_QUEUED)
            else:
                # Mark as failed
                if self.db_manager:
                    self.db_manager.update_task_status(task_id, TASK_STATUS_FAILED)
                
                logger.info(f"Task {task_id} failed by worker {worker_id}: {error}")
        
        # Add execution history
        if self.db_manager:
            start_time = metadata.get("start_time")
            end_time = metadata.get("end_time")
            
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                
            if isinstance(end_time, str):
                end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                
            if not start_time:
                start_time = datetime.now() - timedelta(seconds=metadata.get("execution_time", 0))
                
            if not end_time:
                end_time = datetime.now()
                
            execution_time = metadata.get("execution_time", 0)
            hardware_metrics = metadata.get("hardware_metrics", {})
            
            self.db_manager.add_execution_history(
                task_id, worker_id, metadata.get("attempt", 1),
                TASK_STATUS_FAILED, start_time, end_time,
                execution_time, error, hardware_metrics
            )
        
        # Create test result record for aggregation
        test_result = None
        
        try:
            # Get task and worker information
            task = self.db_manager.get_task(task_id) if self.db_manager else {}
            worker = self.db_manager.get_worker(worker_id) if self.db_manager else {}
            
            # Prepare test result for aggregation
            test_result = {
                "test_id": task_id,
                "worker_id": worker_id,
                "task_type": task.get("type") if task else metadata.get("task_type", "unknown"),
                "hardware_id": worker.get("hardware_id") if worker else metadata.get("hardware_id", "unknown"),
                "model": metadata.get("model", "unknown"),
                "batch_size": metadata.get("batch_size", 1),
                "precision": metadata.get("precision", "fp32"),
                "status": "failed",
                "failure_reason": error,
                "timestamp": end_time if isinstance(end_time, datetime) else datetime.now(),
                "duration": execution_time,
                "hardware_metrics": hardware_metrics
            }
                    
        except Exception as e:
            logger.error(f"Error preparing result data for failed task {task_id}: {e}")
        
        # Process with the dual-layer result aggregation system
        if test_result:
            # First prepare the result to ensure it has all required fields
            # This avoids duplicating preparation logic between aggregators
            prepared_result = self._prepare_result_for_aggregation(test_result)
            
            # Add to high-level result aggregator service if available
            if hasattr(self, 'result_aggregator') and self.result_aggregator:
                try:
                    # Process in high-level result aggregator
                    self.result_aggregator.process_test_result(prepared_result)
                    logger.debug(f"Added failed task {task_id} to high-level result aggregator")
                except Exception as e:
                    logger.error(f"Error adding failed task {task_id} to high-level result aggregator: {e}")
                    logger.debug(f"Result data: {prepared_result}")
            
            # Add to detailed result aggregator if available
            if hasattr(self, 'detailed_result_aggregator') and self.detailed_result_aggregator:
                try:
                    # Process in detailed result aggregator
                    self.detailed_result_aggregator.process_test_result(prepared_result)
                    logger.debug(f"Added failed task {task_id} to detailed result aggregator")
                except Exception as e:
                    logger.error(f"Error adding failed task {task_id} to detailed result aggregator: {e}")
                    logger.debug(f"Result data: {prepared_result}")
        
        return True
        
    def _prepare_result_for_aggregation(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare test result data for aggregation.
        
        This helper method ensures the result has all required fields and
        normalizes data for consistent processing across both aggregators.
        
        Args:
            result: Raw test result data
            
        Returns:
            Prepared result data for aggregation
        """
        # Make a copy to avoid modifying the original
        prepared = result.copy()
        
        # Ensure all required fields are present
        required_fields = [
            "test_id", "worker_id", "task_type", "hardware_id", 
            "model", "status", "timestamp"
        ]
        
        for field in required_fields:
            if field not in prepared:
                if field == "timestamp":
                    prepared[field] = datetime.now()
                else:
                    prepared[field] = "unknown"
        
        # Normalize metric fields if available
        if "duration" in prepared and isinstance(prepared["duration"], (int, float)):
            # Ensure we have latency and throughput metrics
            if "latency_ms" not in prepared:
                prepared["latency_ms"] = prepared["duration"] * 1000.0
                
            if "throughput_items_per_second" not in prepared and "batch_size" in prepared:
                batch_size = prepared["batch_size"]
                prepared["throughput_items_per_second"] = batch_size / prepared["duration"] if prepared["duration"] > 0 else 0
        
        # Add hardware dimensions if available
        if "hardware_metrics" in prepared and isinstance(prepared["hardware_metrics"], dict):
            metrics = prepared["hardware_metrics"]
            
            # CPU metrics
            if "cpu_percent" in metrics:
                prepared["cpu_utilization"] = metrics["cpu_percent"]
                
            # Memory metrics
            if "memory_used_gb" in metrics:
                prepared["memory_usage_gb"] = metrics["memory_used_gb"]
                
            # GPU metrics if available
            if "gpu_metrics" in metrics and isinstance(metrics["gpu_metrics"], list) and metrics["gpu_metrics"]:
                gpu_metric = metrics["gpu_metrics"][0]  # Use first GPU
                if "load_percent" in gpu_metric:
                    prepared["gpu_utilization"] = gpu_metric["load_percent"]
                if "memory_used_gb" in gpu_metric:
                    prepared["gpu_memory_usage_gb"] = gpu_metric["memory_used_gb"]
        
        # Add context information for multi-dimensional analysis
        prepared["context"] = prepared.get("context", {})
        prepared["context"]["aggregated_at"] = datetime.now()
        
        return prepared
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task.
        
        Args:
            task_id: ID of the task to cancel
            
        Returns:
            True if successful, False otherwise
        """
        with self.task_lock:
            # Check if task is queued
            for i, (_, _, tid, _) in enumerate(self.task_queue):
                if tid == task_id:
                    self.task_queue.pop(i)
                    logger.info(f"Canceled queued task {task_id}")
                    
                    # Update status in database
                    if self.db_manager:
                        self.db_manager.update_task_status(task_id, TASK_STATUS_CANCELED)
                        
                    return True
            
            # Check if task is running
            if task_id in self.running_tasks:
                # Can't actually stop a running task, just mark it as canceled
                # The worker will continue to execute it
                worker_id = self.running_tasks[task_id]
                logger.info(f"Marked running task {task_id} as canceled (worker: {worker_id})")
                
                # Update status in database
                if self.db_manager:
                    self.db_manager.update_task_status(task_id, TASK_STATUS_CANCELED)
                    
                return True
        
        # Task not found
        logger.warning(f"Task {task_id} not found")
        return False
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Dict with task status information if found, None otherwise
        """
        # Check if task is in running tasks
        with self.task_lock:
            if task_id in self.running_tasks:
                worker_id = self.running_tasks[task_id]
                return {
                    "task_id": task_id,
                    "status": TASK_STATUS_RUNNING,
                    "worker_id": worker_id
                }
            
            # Check if task is in queue
            for _, _, tid, task in self.task_queue:
                if tid == task_id:
                    return {
                        "task_id": task_id,
                        "status": TASK_STATUS_QUEUED,
                        "priority": task["priority"]
                    }
        
        # Check database
        if self.db_manager:
            task = self.db_manager.get_task(task_id)
            if task:
                return task
        
        # Task not found
        return None
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get statistics about the task queue.
        
        Returns:
            Dict with queue statistics
        """
        with self.task_lock:
            return {
                "queued_tasks": len(self.task_queue),
                "running_tasks": len(self.running_tasks),
                "total_tasks": len(self.task_queue) + len(self.running_tasks)
            }


class WorkerManager:
    """Manages worker nodes and their capabilities."""
    
    def __init__(self, db_manager: DatabaseManager = None, 
                heartbeat_timeout: int = 60):
        """Initialize the worker manager.
        
        Args:
            db_manager: Database manager for storing/retrieving worker information
            heartbeat_timeout: Timeout in seconds for worker heartbeats
        """
        self.db_manager = db_manager
        self.heartbeat_timeout = heartbeat_timeout
        self.workers = {}  # worker_id -> worker info
        self.active_connections = {}  # worker_id -> websocket
        self.worker_lock = threading.Lock()
        
        # Start heartbeat checker
        self.heartbeat_thread = threading.Thread(
            target=self._check_heartbeats, 
            daemon=True
        )
        self.heartbeat_thread.start()
    
    def register_worker(self, worker_id: str, hostname: str, 
                       capabilities: Dict[str, Any], websocket=None,
                       tags: Dict[str, Any] = None, resources: Dict[str, Any] = None) -> bool:
        """Register a new worker or update an existing one.
        
        Args:
            worker_id: Unique identifier for the worker
            hostname: Hostname of the worker
            capabilities: Dict containing hardware capabilities
            websocket: WebSocket connection for the worker
            tags: Optional tags for worker categorization
            resources: Detailed resource information for dynamic resource management
            
        Returns:
            True if successful, False otherwise
        """
        with self.worker_lock:
            # Check if worker already exists
            worker_exists = worker_id in self.workers
            
            # Create or update worker info
            worker_info = {
                "worker_id": worker_id,
                "hostname": hostname,
                "registration_time": datetime.now(),
                "last_heartbeat": datetime.now(),
                "status": WORKER_STATUS_ACTIVE,
                "capabilities": capabilities,
                "hardware_metrics": {},
                "tags": tags or {},
                "resources": resources or {}
            }
            
            self.workers[worker_id] = worker_info
            
            # Store websocket connection if provided
            if websocket:
                self.active_connections[worker_id] = websocket
            
            # Add to database if available
            if self.db_manager and not worker_exists:
                self.db_manager.add_worker(worker_id, hostname, capabilities, tags)
            
            # Register with Dynamic Resource Manager if available
            coordinator = self._get_coordinator_server()
            if coordinator and coordinator.dynamic_resource_manager and resources:
                try:
                    # Register with dynamic resource manager
                    coordinator.dynamic_resource_manager.register_worker(worker_id, resources)
                    logger.info(f"Registered worker {worker_id} with Dynamic Resource Manager")
                except Exception as e:
                    logger.error(f"Error registering worker with Dynamic Resource Manager: {e}")
            
            action = "Updated" if worker_exists else "Registered"
            logger.info(f"{action} worker {worker_id} ({hostname})")
            return True
    
    def _get_coordinator_server(self):
        """Get the parent CoordinatorServer instance.
        
        Returns:
            CoordinatorServer instance or None
        """
        # This is a helper method to access the parent CoordinatorServer
        # which contains the dynamic_resource_manager
        try:
            frame = sys._getframe(1)
            while frame:
                if 'self' in frame.f_locals:
                    instance = frame.f_locals['self']
                    if isinstance(instance, CoordinatorServer):
                        return instance
                frame = frame.f_back
        except Exception:
            pass
        return None
    
    def update_worker_heartbeat(self, worker_id: str) -> bool:
        """Update the heartbeat timestamp for a worker.
        
        Args:
            worker_id: ID of the worker
            
        Returns:
            True if successful, False otherwise
        """
        with self.worker_lock:
            if worker_id not in self.workers:
                logger.warning(f"Worker {worker_id} not found")
                return False
                
            # Update heartbeat
            self.workers[worker_id]["last_heartbeat"] = datetime.now()
            
            # Set status to active if it was unavailable or disconnected
            current_status = self.workers[worker_id]["status"]
            if current_status in [WORKER_STATUS_UNAVAILABLE, WORKER_STATUS_DISCONNECTED]:
                self.workers[worker_id]["status"] = WORKER_STATUS_ACTIVE
                logger.info(f"Worker {worker_id} is now active")
            
            # Update in database if available
            if self.db_manager:
                self.db_manager.update_worker_heartbeat(worker_id)
                
                # Update status if needed
                if current_status in [WORKER_STATUS_UNAVAILABLE, WORKER_STATUS_DISCONNECTED]:
                    self.db_manager.update_worker_status(worker_id, WORKER_STATUS_ACTIVE)
            
            return True
    
    def update_worker_status(self, worker_id: str, status: str) -> bool:
        """Update the status of a worker.
        
        Args:
            worker_id: ID of the worker
            status: New status for the worker
            
        Returns:
            True if successful, False otherwise
        """
        with self.worker_lock:
            if worker_id not in self.workers:
                logger.warning(f"Worker {worker_id} not found")
                return False
                
            # Update status
            self.workers[worker_id]["status"] = status
            
            # Update in database if available
            if self.db_manager:
                self.db_manager.update_worker_status(worker_id, status)
            
            logger.info(f"Updated worker {worker_id} status to {status}")
            return True
    
    def get_worker(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """Get worker information.
        
        Args:
            worker_id: ID of the worker
            
        Returns:
            Dict with worker information if found, None otherwise
        """
        with self.worker_lock:
            return self.workers.get(worker_id)
    
    def get_available_workers(self) -> List[Dict[str, Any]]:
        """Get a list of available worker nodes.
        
        Returns:
            List of dicts containing worker information
        """
        with self.worker_lock:
            return [
                worker for worker in self.workers.values()
                if worker["status"] in [WORKER_STATUS_ACTIVE, WORKER_STATUS_REGISTERED]
            ]
    
    def set_worker_websocket(self, worker_id: str, websocket) -> bool:
        """Set the WebSocket connection for a worker.
        
        Args:
            worker_id: ID of the worker
            websocket: WebSocket connection
            
        Returns:
            True if successful, False otherwise
        """
        with self.worker_lock:
            if worker_id not in self.workers:
                logger.warning(f"Worker {worker_id} not found")
                return False
                
            self.active_connections[worker_id] = websocket
            
            # Update status to active
            self.workers[worker_id]["status"] = WORKER_STATUS_ACTIVE
            
            # Update in database if available
            if self.db_manager:
                self.db_manager.update_worker_status(worker_id, WORKER_STATUS_ACTIVE)
            
            logger.info(f"Set WebSocket connection for worker {worker_id}")
            return True
    
    async def send_message_to_worker(self, worker_id: str, 
                                   message: Dict[str, Any]) -> bool:
        """Send a message to a worker via WebSocket.
        
        Args:
            worker_id: ID of the worker
            message: Message to send
            
        Returns:
            True if successful, False otherwise
        """
        if not WEBSOCKETS_AVAILABLE:
            logger.warning("WebSockets not available, cannot send message")
            return False
            
        with self.worker_lock:
            if worker_id not in self.active_connections:
                logger.warning(f"No active connection for worker {worker_id}")
                return False
                
            websocket = self.active_connections[worker_id]
            
            try:
                message_json = json.dumps(message)
                await websocket.send(message_json)
                return True
            except Exception as e:
                logger.error(f"Error sending message to worker {worker_id}: {e}")
                # Remove connection
                del self.active_connections[worker_id]
                # Update status
                self.workers[worker_id]["status"] = WORKER_STATUS_DISCONNECTED
                # Update in database if available
                if self.db_manager:
                    self.db_manager.update_worker_status(worker_id, WORKER_STATUS_DISCONNECTED)
                return False
    
    async def broadcast_message(self, message: Dict[str, Any], 
                              filter_status: Optional[List[str]] = None) -> int:
        """Broadcast a message to all workers or filtered by status.
        
        Args:
            message: Message to send
            filter_status: List of statuses to filter by (or None for all)
            
        Returns:
            Number of workers the message was sent to
        """
        if not WEBSOCKETS_AVAILABLE:
            logger.warning("WebSockets not available, cannot broadcast")
            return 0
            
        sent_count = 0
        
        with self.worker_lock:
            for worker_id, worker in self.workers.items():
                # Skip workers that don't match filter
                if filter_status and worker["status"] not in filter_status:
                    continue
                    
                # Skip workers without active connection
                if worker_id not in self.active_connections:
                    continue
                    
                # Send message
                websocket = self.active_connections[worker_id]
                
                try:
                    message_json = json.dumps(message)
                    await websocket.send(message_json)
                    sent_count += 1
                except Exception as e:
                    logger.error(f"Error broadcasting to worker {worker_id}: {e}")
                    # Remove connection
                    del self.active_connections[worker_id]
                    # Update status
                    self.workers[worker_id]["status"] = WORKER_STATUS_DISCONNECTED
                    # Update in database if available
                    if self.db_manager:
                        self.db_manager.update_worker_status(worker_id, WORKER_STATUS_DISCONNECTED)
        
        return sent_count
    
    def disconnect_worker(self, worker_id: str) -> bool:
        """Disconnect a worker.
        
        Args:
            worker_id: ID of the worker
            
        Returns:
            True if successful, False otherwise
        """
        with self.worker_lock:
            if worker_id not in self.workers:
                logger.warning(f"Worker {worker_id} not found")
                return False
                
            # Remove from active connections
            if worker_id in self.active_connections:
                del self.active_connections[worker_id]
            
            # Update status
            self.workers[worker_id]["status"] = WORKER_STATUS_DISCONNECTED
            
            # Update in database if available
            if self.db_manager:
                self.db_manager.update_worker_status(worker_id, WORKER_STATUS_DISCONNECTED)
            
            logger.info(f"Disconnected worker {worker_id}")
            return True
    
    def _check_heartbeats(self):
        """Check worker heartbeats and mark inactive workers as unavailable."""
        while True:
            time.sleep(self.heartbeat_timeout / 2)
            
            with self.worker_lock:
                for worker_id, worker in list(self.workers.items()):
                    last_heartbeat = worker["last_heartbeat"]
                    elapsed = (datetime.now() - last_heartbeat).total_seconds()
                    
                    if elapsed > self.heartbeat_timeout and worker["status"] == WORKER_STATUS_ACTIVE:
                        # Mark as unavailable
                        worker["status"] = WORKER_STATUS_UNAVAILABLE
                        
                        # Update in database if available
                        if self.db_manager:
                            self.db_manager.update_worker_status(worker_id, WORKER_STATUS_UNAVAILABLE)
                        
                        logger.warning(
                            f"Worker {worker_id} heartbeat timeout "
                            f"({elapsed:.1f}s > {self.heartbeat_timeout}s)"
                        )
                        
                        # Remove from active connections
                        if worker_id in self.active_connections:
                            del self.active_connections[worker_id]


class CoordinatorServer:
    """Main coordinator server for the distributed testing framework."""
    
    def __init__(self, host: str = "localhost", port: int = 8080,
                 db_path: str = None, token_secret: str = None,
                 heartbeat_timeout: int = 60, auto_recovery: bool = False,
                 coordinator_id: str = None, coordinator_addresses: List[str] = None,
                 performance_analyzer: bool = False, visualization_path: str = None):
        """Initialize the coordinator server.
        
        Args:
            host: Host to bind the server to
            port: Port to bind the server to
            db_path: Path to the DuckDB database
            token_secret: Secret for signing JWT tokens
            heartbeat_timeout: Timeout in seconds for worker heartbeats
            auto_recovery: Enable auto recovery system for high availability
            coordinator_id: Unique identifier for this coordinator instance
            coordinator_addresses: List of other coordinator addresses for clustering
            performance_analyzer: Enable performance trend analyzer
            visualization_path: Path for performance visualizations
        """
        self.host = host
        self.port = port
        self.db_path = db_path
        self.token_secret = token_secret
        self.heartbeat_timeout = heartbeat_timeout
        self.coordinator_id = coordinator_id or f"coordinator-{uuid.uuid4().hex[:8]}"
        self.coordinator_addresses = coordinator_addresses or []
        
        # Initialize database manager if path is provided
        self.db_manager = None
        if db_path and DUCKDB_AVAILABLE:
            try:
                self.db_manager = DatabaseManager(db_path)
                logger.info(f"Database manager initialized with path: {db_path}")
            except Exception as e:
                logger.error(f"Error initializing database manager: {e}")
        
        # Initialize security manager
        self.security_manager = SecurityManager(self.db_manager, token_secret)
        
        # Initialize task manager
        self.task_manager = TaskManager(self.db_manager)
        
        # Initialize worker manager
        self.worker_manager = WorkerManager(self.db_manager, heartbeat_timeout)
        
        # Initialize dynamic resource manager if available
        self.dynamic_resource_manager = None
        self.scaling_interval = 60  # Default scaling evaluation interval in seconds
        
        if DYNAMIC_RESOURCE_MANAGER_AVAILABLE:
            try:
                self.dynamic_resource_manager = DynamicResourceManager(
                    target_utilization=0.7,
                    scale_up_threshold=0.8,
                    scale_down_threshold=0.3,
                    evaluation_window=300,  # 5 minutes
                    scale_up_cooldown=300,  # 5 minutes
                    scale_down_cooldown=600  # 10 minutes
                )
                logger.info("Dynamic Resource Manager initialized")
            except Exception as e:
                logger.error(f"Error initializing Dynamic Resource Manager: {e}")
                self.dynamic_resource_manager = None
        
        # Initialize resource performance predictor if available
        self.resource_predictor = None
        if RESOURCE_PREDICTOR_AVAILABLE:
            try:
                resource_db_path = None
                if self.db_path:
                    # Use a separate SQLite database for resource predictions
                    resource_db_path = self.db_path.replace(".duckdb", "_resources.sqlite")
                
                self.resource_predictor = ResourcePerformancePredictor(database_path=resource_db_path)
                logger.info("Resource Performance Predictor initialized")
            except Exception as e:
                logger.error(f"Error initializing Resource Performance Predictor: {e}")
                self.resource_predictor = None
        
        # Initialize cloud provider manager if available
        self.cloud_provider_manager = None
        if CLOUD_PROVIDER_AVAILABLE:
            try:
                # Look for cloud provider config file
                cloud_config_path = os.path.join(os.path.dirname(self.db_path) if self.db_path else ".", "cloud_config.json")
                if os.path.exists(cloud_config_path):
                    self.cloud_provider_manager = CloudProviderManager(config_path=cloud_config_path)
                    logger.info(f"Cloud Provider Manager initialized with config from {cloud_config_path}")
                else:
                    # Initialize with default configuration
                    self.cloud_provider_manager = CloudProviderManager()
                    
                    # Add available cloud providers
                    try:
                        # Try to add AWS provider
                        aws_provider = AWSCloudProvider("us-east-1")
                        self.cloud_provider_manager.add_provider("aws", aws_provider)
                        logger.info("AWS provider added")
                    except Exception as aws_e:
                        logger.warning(f"Could not initialize AWS provider: {aws_e}")
                    
                    try:
                        # Try to add GCP provider if project ID is available in environment
                        if "GCP_PROJECT_ID" in os.environ:
                            gcp_provider = GCPCloudProvider(
                                region="us-central1-a", 
                                project_id=os.environ.get("GCP_PROJECT_ID")
                            )
                            self.cloud_provider_manager.add_provider("gcp", gcp_provider)
                            logger.info("GCP provider added")
                    except Exception as gcp_e:
                        logger.warning(f"Could not initialize GCP provider: {gcp_e}")
                    
                    try:
                        # Add Docker local provider
                        docker_provider = DockerLocalProvider()
                        self.cloud_provider_manager.add_provider("docker_local", docker_provider)
                        logger.info("Docker local provider added")
                    except Exception as docker_e:
                        logger.warning(f"Could not initialize Docker provider: {docker_e}")
                
                logger.info("Cloud Provider Manager initialized")
            except Exception as e:
                logger.error(f"Error initializing Cloud Provider Manager: {e}")
                self.cloud_provider_manager = None
        
        # Initialize auto recovery system if enabled
        self.auto_recovery = None
        if auto_recovery and AUTO_RECOVERY_AVAILABLE:
            try:
                self.auto_recovery = AutoRecovery(
                    coordinator_id=self.coordinator_id,
                    db_manager=self.db_manager,
                    coordinator_manager=self.worker_manager,
                    task_scheduler=self.task_manager
                )
                
                # Configure auto recovery system
                self.auto_recovery.configure({
                    "coordinator_port": self.port,
                    "coordinator_addresses": self.coordinator_addresses,
                    "failover_enabled": True,
                    "auto_leader_election": True,
                    "auto_discover_coordinators": True
                })
                
                logger.info(f"Auto recovery system initialized with ID: {self.coordinator_id}")
            except Exception as e:
                logger.error(f"Error initializing auto recovery system: {e}")
                self.auto_recovery = None
        
        # Initialize performance trend analyzer if enabled
        self.performance_analyzer = None
        if performance_analyzer and PERFORMANCE_ANALYZER_AVAILABLE:
            try:
                self.performance_analyzer = PerformanceTrendAnalyzer(
                    db_manager=self.db_manager,
                    task_scheduler=self.task_manager
                )
                
                # Configure performance analyzer
                analyzer_config = {
                    "visualization_enabled": visualization_path is not None,
                    "database_enabled": self.db_manager is not None
                }
                
                if visualization_path:
                    analyzer_config["visualization_path"] = visualization_path
                    
                self.performance_analyzer.configure(analyzer_config)
                
                logger.info("Performance trend analyzer initialized")
            except Exception as e:
                logger.error(f"Error initializing performance trend analyzer: {e}")
                self.performance_analyzer = None
                
        # Initialize result aggregators - dual layer system
        self.result_aggregator = None
        self.detailed_result_aggregator = None
        if RESULT_AGGREGATOR_AVAILABLE:
            try:
                # Initialize the ResultAggregatorService from result_aggregator/service.py (high-level service)
                self.result_aggregator = ResultAggregatorService(
                    db_manager=self.db_manager,
                    trend_analyzer=self.performance_analyzer
                )
                
                # Configure result aggregator service
                aggregator_config = {
                    "visualization_enabled": visualization_path is not None,
                    "database_enabled": self.db_manager is not None,
                    "cache_ttl_seconds": 300,  # 5 minutes
                    "update_interval": 300,  # 5 minutes
                    "anomaly_threshold": 2.5,  # Z-score threshold for anomalies
                    "min_data_points": 5,  # Minimum data points for analysis
                    "aggregate_dimensions": ["hardware", "model", "batch_size", "precision", "task_type"],
                    "correlation_metrics": ["throughput", "latency", "memory_usage", "success_rate"],
                    "comparative_lookback_days": 7,  # Days to look back for comparison
                    "normalize_metrics": True,  # Whether to normalize metrics for comparison
                    "workers_historical_limit": 10,  # Maximum workers to include in historical analysis
                    "deduplication_enabled": True,  # Whether to deduplicate similar results
                    "model_family_grouping": True  # Whether to group results by model family
                }
                
                if visualization_path:
                    aggregator_config["visualization_path"] = os.path.join(visualization_path, "result_aggregation")
                    
                self.result_aggregator.configure(aggregator_config)
                
                # Initialize the DetailedResultAggregator from result_aggregator/aggregator.py (detailed analysis)
                self.detailed_result_aggregator = DetailedResultAggregator(
                    db_manager=self.db_manager,
                    task_scheduler=self.task_manager
                )
                
                # Configure detailed result aggregator
                detailed_aggregator_config = {
                    "visualization_enabled": visualization_path is not None,
                    "database_enabled": self.db_manager is not None,
                    "update_interval": 300,  # 5 minutes
                    "history_days": 30,  # Days of history to keep
                    "aggregate_dimensions": ["hardware", "model", "batch_size", "precision", "task_type"],
                    "comparison_metrics": ["throughput", "latency", "memory_usage", "success_rate"],
                    "significance_level": 0.05,  # p-value threshold for statistical significance
                    "outlier_detection_enabled": True,  # Enable outlier detection
                    "generate_regression_alerts": True,  # Generate alerts for regressions
                    "dimension_mapping": {  # Map dimension names between systems
                        "model_hardware": "model:hardware",
                        "task_type": "task_type",
                        "worker": "worker_id"
                    }
                }
                
                if visualization_path:
                    detailed_aggregator_config["visualization_path"] = os.path.join(visualization_path, "detailed_result_aggregation")
                    
                self.detailed_result_aggregator.configure(detailed_aggregator_config)
                
                logger.info("Dual-layer result aggregation system initialized")
            except Exception as e:
                logger.error(f"Error initializing result aggregators: {e}")
                traceback.print_exc()
                self.result_aggregator = None
                self.detailed_result_aggregator = None
        
        # Set up scaling evaluation thread and parameters
        self.scaling_thread = None
        self.scaling_interval = 300  # 5 minutes
        self.scaling_enabled = True
        
        # WebSocket server
        self.websocket_server = None
        self.running = False
        self.stop_event = anyio.Event()
        
        # HTTP server for API
        self.http_server = None
    
    async def start(self):
        """Start the coordinator server."""
        if not WEBSOCKETS_AVAILABLE:
            logger.error("WebSockets not available, cannot start server")
            return False
            
        try:
            # Start auto recovery system if enabled
            if self.auto_recovery:
                self.auto_recovery.start()
                
                # Register event callbacks
                self.auto_recovery.on_become_leader(self._on_become_leader)
                self.auto_recovery.on_leader_changed(self._on_leader_changed)
                
                logger.info("Auto recovery system started")
            
            # Start performance trend analyzer if enabled
            if self.performance_analyzer:
                self.performance_analyzer.start()
                logger.info("Performance trend analyzer started")
            
            # Start result aggregators if enabled - dual-layer system
            if self.result_aggregator:
                self.result_aggregator.start()
                logger.info("High-level result aggregator service started")
                
            if self.detailed_result_aggregator:
                self.detailed_result_aggregator.start()
                logger.info("Detailed result aggregator service started")
                
            # If both aggregators are running, log dual-layer system status
            if self.result_aggregator and self.detailed_result_aggregator:
                logger.info("Dual-layer intelligent result aggregation system is active")
                
            # Start dynamic resource management and scaling evaluation if available
            # NOTE: The scaling thread is now managed in the main() function when --enable-drm is used
            # This ensures it's not started multiple times or without proper configuration
            
            # Start WebSocket server
            logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
            self.websocket_server = await websockets.serve(
                self._handle_websocket,
                self.host,
                self.port
            )
            
            # Start HTTP server for API
            # Note: In a production environment, you'd use a proper HTTP server
            # with appropriate routing and authentication
            # This simple version is for demonstration only
            
            # Set running flag
            self.running = True
            
            # Register signal handlers for graceful shutdown
            # TODO: Use anyio.open_signal_receiver to trigger self.stop()
            
            logger.info(f"Coordinator server started on {self.host}:{self.port}")
            
            # Keep server running until stop event
            await self.stop_event.wait()
            logger.info("Coordinator server stopping...")
            
            return True
        except Exception as e:
            logger.error(f"Error starting coordinator server: {e}")
            return False
    
    async def stop(self):
        """Stop the coordinator server."""
        if not self.running:
            return
            
        # Set stop event
        self.stop_event.set()
        
        # Stop auto recovery system if enabled
        if self.auto_recovery:
            self.auto_recovery.stop()
            logger.info("Auto recovery system stopped")
        
        # Stop performance trend analyzer if enabled
        if self.performance_analyzer:
            self.performance_analyzer.stop()
            logger.info("Performance trend analyzer stopped")
            
        # Stop result aggregators if enabled - dual-layer system
        if self.result_aggregator:
            self.result_aggregator.stop()
            logger.info("High-level result aggregator service stopped")
            
        if self.detailed_result_aggregator:
            self.detailed_result_aggregator.stop()
            logger.info("Detailed result aggregator service stopped")
            
        # If both aggregators were running, log shutdown
        if self.result_aggregator and self.detailed_result_aggregator:
            logger.info("Dual-layer intelligent result aggregation system shutdown complete")
            
        # Set running to false to stop scaling evaluation thread
        self.running = False
        
        # Wait for scaling thread to complete
        if self.scaling_thread and self.scaling_thread.is_alive():
            logger.info("Waiting for scaling thread to complete...")
            self.scaling_thread.join(timeout=5.0)
            
        # Cleanup dynamic resource manager if available
        if self.dynamic_resource_manager:
            try:
                self.dynamic_resource_manager.cleanup()
                logger.info("Dynamic Resource Manager cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up Dynamic Resource Manager: {e}")
                
        # Cleanup resource predictor if available
        if self.resource_predictor:
            try:
                self.resource_predictor.cleanup()
                logger.info("Resource Performance Predictor cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up Resource Performance Predictor: {e}")
        
        # Wait for scaling thread to terminate if it was started
        if self.scaling_thread and self.scaling_thread.is_alive():
            logger.info("Waiting for scaling evaluation thread to terminate...")
            self.scaling_thread.join(timeout=5.0)  # Wait up to 5 seconds
            if self.scaling_thread.is_alive():
                logger.warning("Scaling evaluation thread did not terminate cleanly")
        
        # Close WebSocket server
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
        
        # Close database connection
        if self.db_manager:
            self.db_manager.close()
        
        # Set running flag
        self.running = False
        
        logger.info("Coordinator server stopped")
        
    def _on_become_leader(self):
        """Callback for when this coordinator becomes the leader."""
        logger.info(f"Coordinator {self.coordinator_id} has become the leader")
        
        # Perform leader-specific initialization
        # For example, you could start additional services or enable certain features
        
    def _on_leader_changed(self, old_leader: str, new_leader: str):
        """Callback for when the leader coordinator changes.
        
        Args:
            old_leader: ID of the previous leader coordinator
            new_leader: ID of the new leader coordinator
        """
        logger.info(f"Leader changed from {old_leader} to {new_leader}")
        
        # Adjust behavior based on new leader
        if new_leader == self.coordinator_id:
            logger.info("This coordinator is now the leader")
        else:
            logger.info(f"Following leader: {new_leader}")
    
    async def _handle_websocket(self, websocket, path):
        """Handle WebSocket connections from workers.
        
        Args:
            websocket: WebSocket connection
            path: Connection path
        """
        worker_id = None
        
        try:
            # Perform authentication
            authenticated = await self._authenticate_websocket(websocket)
            if not authenticated:
                logger.warning("Authentication failed")
                return
            
            # Process messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._process_message(websocket, data)
                    
                    # Extract worker_id from registration message
                    if data.get("type") == "register" and not worker_id:
                        worker_id = data.get("worker_id")
                        
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON message: {message}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    traceback.print_exc()
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed: {worker_id}")
        finally:
            # Clean up worker connection
            if worker_id:
                self.worker_manager.disconnect_worker(worker_id)
    
    async def _authenticate_websocket(self, websocket) -> bool:
        """Authenticate a WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            
        Returns:
            True if authentication is successful, False otherwise
        """
        try:
            # Send authentication challenge
            challenge = {
                "type": "auth_challenge",
                "challenge_id": str(uuid.uuid4())
            }
            await websocket.send(json.dumps(challenge))
            
            # Wait for authentication response
            response = await websocket.recv()
            data = json.loads(response)
            
            if data.get("type") != "auth_response":
                logger.warning("Invalid authentication response type")
                return False
            
            # Validate API key
            api_key = data.get("api_key")
            if not api_key:
                logger.warning("No API key provided")
                return False
            
            key_info = self.security_manager.validate_api_key(api_key)
            if not key_info:
                logger.warning("Invalid API key")
                await websocket.send(json.dumps({
                    "type": "auth_result",
                    "success": False,
                    "error": "Invalid API key"
                }))
                return False
            
            # Generate token
            worker_id = data.get("worker_id") or f"worker_{uuid.uuid4()}"
            token = self.security_manager.generate_token(worker_id, key_info.get("role", "worker"))
            
            # Send authentication result
            await websocket.send(json.dumps({
                "type": "auth_result",
                "success": True,
                "worker_id": worker_id,
                "token": token
            }))
            
            return True
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
    
    def get_aggregated_results(self, result_type: str, aggregation_level: str, 
                            filter_params: Dict[str, Any] = None, 
                            time_range: Tuple[datetime, datetime] = None, 
                            use_detailed: bool = False) -> Dict[str, Any]:
        """Get aggregated results from the result aggregator.
        
        Args:
            result_type: Type of results to aggregate
            aggregation_level: Level of aggregation
            filter_params: Parameters to filter results by
            time_range: Time range to filter results by
            use_detailed: Whether to use the detailed aggregator
            
        Returns:
            Dictionary of aggregated results or empty dict if not available
        """
        # Select which aggregator to use based on requested detail level
        if use_detailed and hasattr(self, 'detailed_result_aggregator') and self.detailed_result_aggregator:
            try:
                # For the detailed aggregator from result_aggregator/aggregator.py
                # Map the dimension names if needed
                dimension_mapping = {
                    "model_hardware": "model:hardware",
                    "task_type": "task_type",
                    "worker": "worker_id"
                }
                
                # Map the dimension name if needed
                dimension = dimension_mapping.get(aggregation_level, aggregation_level)
                
                # Validate the result type
                valid_result_types = ["performance", "compatibility", "integration", "web_platform"]
                if result_type not in valid_result_types:
                    logger.warning(f"Unsupported result type for detailed aggregator: {result_type}")
                    result_type = "performance"  # Default to performance
                
                # Get dimension analysis from the detailed aggregator
                dimension_analysis = self.detailed_result_aggregator.get_dimension_analysis(dimension)
                
                # Get other metrics from the detailed aggregator
                regression_data = self.detailed_result_aggregator.get_regressions()
                anomalies = self.detailed_result_aggregator.get_anomalies()
                
                # Add visualization data if available
                visualization_data = {}
                try:
                    visualization_data = self.detailed_result_aggregator.get_visualizations(dimension)
                except Exception as viz_error:
                    logger.debug(f"Could not get visualizations: {viz_error}")
                
                # Return a comprehensive result with all available data
                return {
                    "aggregation_level": aggregation_level,
                    "result_type": result_type,
                    "results": {
                        "dimension_analysis": dimension_analysis,
                        "overall_status": self.detailed_result_aggregator.get_overall_status(),
                        "regressions": regression_data,
                        "anomalies": anomalies,
                        "visualizations": visualization_data,
                        "metadata": {
                            "aggregator": "detailed",
                            "timestamp": datetime.now().isoformat(),
                            "filter_params": filter_params
                        }
                    }
                }
            except Exception as e:
                logger.error(f"Error getting detailed aggregated results: {e}")
                traceback.print_exc()
                return {"error": str(e), "aggregator": "detailed"}
        
        # Use high-level aggregator service if available
        elif hasattr(self, 'result_aggregator') and self.result_aggregator:
            try:
                # For the high-level aggregator from result_aggregator/service.py
                result = self.result_aggregator.aggregate_results(
                    result_type=result_type,
                    aggregation_level=aggregation_level,
                    filter_params=filter_params,
                    time_range=time_range
                )
                
                # Add metadata about which aggregator was used
                if "metadata" not in result:
                    result["metadata"] = {}
                
                result["metadata"]["aggregator"] = "high_level"
                result["metadata"]["timestamp"] = datetime.now().isoformat()
                
                return result
            except Exception as e:
                logger.error(f"Error getting high-level aggregated results: {e}")
                traceback.print_exc()
                return {"error": str(e), "aggregator": "high_level"}
        
        # No aggregator available
        else:
            logger.warning("No result aggregation system available")
            return {"error": "No result aggregation system available"}
    
    async def _process_message(self, websocket, message: Dict[str, Any]):
        """Process a message from a worker.
        
        Args:
            websocket: WebSocket connection
            message: Message data
        """
        message_type = message.get("type")
        
        if message_type == "register":
            # Register worker
            worker_id = message.get("worker_id")
            hostname = message.get("hostname")
            capabilities = message.get("capabilities", {})
            resources = message.get("resources", {})  # Get resource information
            tags = message.get("tags", {})
            
            if not worker_id or not hostname:
                logger.warning("Invalid registration message")
                await websocket.send(json.dumps({
                    "type": "register_result",
                    "success": False,
                    "error": "worker_id and hostname are required"
                }))
                return
            
            # Register worker with resource information
            success = self.worker_manager.register_worker(
                worker_id, hostname, capabilities, websocket, tags, resources
            )
            
            # Register with dynamic resource manager if available
            if hasattr(self, 'dynamic_resource_manager') and self.dynamic_resource_manager and resources:
                try:
                    # Register worker with dynamic resource manager
                    self.dynamic_resource_manager.register_worker(worker_id, resources)
                    logger.info(f"Registered worker {worker_id} with Dynamic Resource Manager")
                except Exception as e:
                    logger.error(f"Error registering worker with Dynamic Resource Manager: {e}")
            
            # Send response
            await websocket.send(json.dumps({
                "type": "register_result",
                "success": success,
                "worker_id": worker_id
            }))
        
        elif message_type == "heartbeat":
            # Update worker heartbeat
            worker_id = message.get("worker_id")
            resources = message.get("resources", {})  # Get updated resource information
            hardware_metrics = message.get("hardware_metrics", {})  # Get hardware metrics
            
            if not worker_id:
                logger.warning("Invalid heartbeat message")
                await websocket.send(json.dumps({
                    "type": "heartbeat_result",
                    "success": False,
                    "error": "worker_id is required"
                }))
                return
            
            # Update heartbeat
            success = self.worker_manager.update_worker_heartbeat(worker_id)
            
            # Update worker info with latest resource metrics
            worker = self.worker_manager.get_worker(worker_id)
            if worker and hardware_metrics:
                # Update hardware metrics in worker info
                worker["hardware_metrics"] = hardware_metrics
            
            # Update dynamic resource manager if available
            if hasattr(self, 'dynamic_resource_manager') and self.dynamic_resource_manager and resources:
                try:
                    # Update worker resources in dynamic resource manager
                    self.dynamic_resource_manager.update_worker_resources(worker_id, resources)
                except Exception as e:
                    logger.error(f"Error updating worker resources in Dynamic Resource Manager: {e}")
            
            # Send response
            await websocket.send(json.dumps({
                "type": "heartbeat_result",
                "success": success,
                "worker_id": worker_id
            }))
        
        elif message_type == "get_task":
            # Get a task for the worker
            worker_id = message.get("worker_id")
            capabilities = message.get("capabilities", {})
            
            if not worker_id:
                logger.warning("Invalid get_task message")
                await websocket.send(json.dumps({
                    "type": "get_task_result",
                    "success": False,
                    "error": "worker_id is required"
                }))
                return
            
            # Update worker capabilities if provided
            worker = self.worker_manager.get_worker(worker_id)
            if worker and capabilities:
                worker["capabilities"].update(capabilities)
            
            # Get worker resources if available
            resources = message.get("resources", {})
            if resources:
                # Update worker resources before getting a task
                if hasattr(self, 'dynamic_resource_manager') and self.dynamic_resource_manager:
                    try:
                        self.dynamic_resource_manager.update_worker_resources(worker_id, resources)
                    except Exception as e:
                        logger.error(f"Error updating worker resources: {e}")
            
            # Get next task with resource-aware scheduling
            task = self.task_manager.get_next_task(worker_id, worker["capabilities"], resources)
            
            if task:
                # Update worker status to busy
                self.worker_manager.update_worker_status(worker_id, WORKER_STATUS_BUSY)
                
                # Send task to worker
                await websocket.send(json.dumps({
                    "type": "get_task_result",
                    "success": True,
                    "worker_id": worker_id,
                    "task": task
                }))
            else:
                # No task available
                await websocket.send(json.dumps({
                    "type": "get_task_result",
                    "success": True,
                    "worker_id": worker_id,
                    "task": None
                }))
        
        elif message_type == "get_aggregated_results":
            # Get aggregated results
            result_type = message.get("result_type")
            aggregation_level = message.get("aggregation_level")
            filter_params = message.get("filter_params", {})
            time_range = message.get("time_range")
            use_detailed = message.get("use_detailed", False)
            
            if not result_type or not aggregation_level:
                logger.warning("Invalid get_aggregated_results message")
                await websocket.send(json.dumps({
                    "type": "get_aggregated_results_result",
                    "success": False,
                    "error": "result_type and aggregation_level are required"
                }))
                return
            
            # Convert time range if provided
            start_time = None
            end_time = None
            
            if time_range:
                start_time_str = time_range.get("start")
                end_time_str = time_range.get("end")
                
                if start_time_str:
                    try:
                        start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
                    except Exception as e:
                        logger.warning(f"Invalid start_time format: {e}")
                        
                if end_time_str:
                    try:
                        end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
                    except Exception as e:
                        logger.warning(f"Invalid end_time format: {e}")
            
            time_range_tuple = None
            if start_time or end_time:
                time_range_tuple = (start_time, end_time)
            
            # Get aggregated results
            results = self.get_aggregated_results(
                result_type=result_type,
                aggregation_level=aggregation_level,
                filter_params=filter_params,
                time_range=time_range_tuple,
                use_detailed=use_detailed
            )
            
            # Send response
            await websocket.send(json.dumps({
                "type": "get_aggregated_results_result",
                "success": True,
                "results": results
            }))
        
        elif message_type == "task_result":
            # Process task result
            worker_id = message.get("worker_id")
            task_id = message.get("task_id")
            success = message.get("success", False)
            results = message.get("results", {})
            metadata = message.get("metadata", {})
            error = message.get("error", "")
            
            if not worker_id or not task_id:
                logger.warning("Invalid task_result message")
                await websocket.send(json.dumps({
                    "type": "task_result_result",
                    "success": False,
                    "error": "worker_id and task_id are required"
                }))
                return
            
            # Process result
            if success:
                # Mark task as completed
                self.task_manager.complete_task(task_id, worker_id, results, metadata)
            else:
                # Mark task as failed
                self.task_manager.fail_task(task_id, worker_id, error, metadata)
            
            # Update worker status to active
            self.worker_manager.update_worker_status(worker_id, WORKER_STATUS_ACTIVE)
            
            # Send response
            await websocket.send(json.dumps({
                "type": "task_result_result",
                "success": True,
                "worker_id": worker_id,
                "task_id": task_id
            }))
        
        elif message_type == "status_update":
            # Update worker status
            worker_id = message.get("worker_id")
            status = message.get("status")
            
            if not worker_id or not status:
                logger.warning("Invalid status_update message")
                await websocket.send(json.dumps({
                    "type": "status_update_result",
                    "success": False,
                    "error": "worker_id and status are required"
                }))
                return
            
            # Update status
            success = self.worker_manager.update_worker_status(worker_id, status)
            
            # Send response
            await websocket.send(json.dumps({
                "type": "status_update_result",
                "success": success,
                "worker_id": worker_id
            }))
        
        elif message_type == "coordinator_heartbeat" and self.auto_recovery:
            # Process coordinator heartbeat (for auto recovery system)
            response = self.auto_recovery.handle_heartbeat(message)
            await websocket.send(json.dumps({
                "type": "coordinator_heartbeat_response",
                "success": True,
                "term": response["term"],
                "match_index": response.get("match_index", 0),
                "vote_granted": False
            }))
        
        elif message_type == "coordinator_request_vote" and self.auto_recovery:
            # Process coordinator vote request (for auto recovery system)
            response = self.auto_recovery.handle_vote_request(message)
            await websocket.send(json.dumps({
                "type": "coordinator_vote_response",
                "term": response["term"],
                "vote_granted": response["vote_granted"]
            }))
            
        elif message_type == "coordinator_sync" and self.auto_recovery:
            # Process coordinator sync request (for auto recovery system)
            response = self.auto_recovery.handle_sync_request(message)
            await websocket.send(json.dumps({
                "type": "coordinator_sync_response",
                "success": response["success"],
                "snapshot": response.get("snapshot", {}),
                "reason": response.get("reason", "")
            }))
            
        elif message_type == "get_performance_trends" and self.performance_analyzer:
            # Get performance trends
            entity_type = message.get("entity_type", "worker")
            entity_id = message.get("entity_id")
            metric = message.get("metric")
            significant_only = message.get("significant_only", True)
            
            if entity_type == "worker":
                trends = self.performance_analyzer.get_worker_trends(
                    entity_id, metric, significant_only
                )
            else:
                trends = self.performance_analyzer.get_task_trends(
                    entity_id, metric, significant_only
                )
                
            await websocket.send(json.dumps({
                "type": "performance_trends_result",
                "success": True,
                "trends": trends
            }))
            
        elif message_type == "get_performance_anomalies" and self.performance_analyzer:
            # Get performance anomalies
            entity_type = message.get("entity_type", "worker")
            entity_id = message.get("entity_id")
            metric = message.get("metric")
            limit = message.get("limit", 10)
            
            if entity_type == "worker":
                anomalies = self.performance_analyzer.get_worker_anomalies(
                    entity_id, metric, limit
                )
            else:
                anomalies = self.performance_analyzer.get_task_anomalies(
                    entity_id, metric, limit
                )
                
            await websocket.send(json.dumps({
                "type": "performance_anomalies_result",
                "success": True,
                "anomalies": anomalies
            }))
            
        elif message_type == "get_performance_report" and self.performance_analyzer:
            # Get performance report
            entity_type = message.get("entity_type", "worker")
            significant_only = message.get("significant_only", True)
            
            report = self.performance_analyzer.get_performance_report(
                entity_type, significant_only
            )
                
            await websocket.send(json.dumps({
                "type": "performance_report_result",
                "success": True,
                "report": report
            }))
                
        else:
            # Unknown message type
            logger.warning(f"Unknown message type: {message_type}")
            await websocket.send(json.dumps({
                "type": "error",
                "error": f"Unknown message type: {message_type}"
            }))
    
    def _scaling_evaluation_loop(self):
        """Background thread for periodic scaling evaluation."""
        while self.running:
            try:
                # Sleep first to allow system to initialize fully
                time.sleep(self.scaling_interval)
                
                if not self.running:
                    break
                    
                if not self.dynamic_resource_manager or not self.cloud_provider_manager:
                    continue
                
                # Evaluate if scaling is needed
                scaling_decision = self.dynamic_resource_manager.evaluate_scaling()
                
                if scaling_decision.action == "scale_up":
                    # Need to scale up, provision new workers
                    workers_to_add = scaling_decision.count
                    logger.info(f"Scaling up: Adding {workers_to_add} worker(s) due to {scaling_decision.reason}")
                    
                    # Determine resource requirements for new workers
                    resource_requirements = scaling_decision.resource_requirements
                    worker_type = scaling_decision.worker_type
                    
                    # Choose provider based on requirements
                    use_gpu = resource_requirements.get("gpu_memory_mb", 0) > 0
                    provider_name = scaling_decision.provider
                    
                    # If provider not specified, choose based on requirements
                    if not provider_name:
                        provider_requirements = {
                            "gpu": use_gpu,
                            "min_cpu_cores": resource_requirements.get("cpu_cores", 2),
                            "min_memory_gb": resource_requirements.get("memory_mb", 4096) / 1024
                        }
                        
                        provider_name = self.cloud_provider_manager.get_preferred_provider(provider_requirements)
                        
                        # If still no provider, use default
                        if not provider_name:
                            provider_name = "docker_local"  # Default to local for testing
                    
                    # Create workers
                    for i in range(workers_to_add):
                        try:
                            # Prepare API key and coordinator URL for worker
                            api_key = self.security_manager.generate_worker_key().get("api_key")
                            coordinator_url = f"ws://{self.host}:{self.port}"
                            
                            # Create worker with required resources
                            worker_result = self.cloud_provider_manager.create_worker(
                                provider=provider_name,
                                resources=resource_requirements,
                                worker_type=worker_type,
                                coordinator_url=coordinator_url,
                                api_key=api_key
                            )
                            
                            if worker_result:
                                logger.info(f"Created new worker: {worker_result.get('worker_id')} on {provider_name}")
                            else:
                                logger.error(f"Failed to create worker on {provider_name}")
                        except Exception as e:
                            logger.error(f"Error creating worker: {e}")
                            logger.debug(traceback.format_exc())
                
                elif scaling_decision.action == "scale_down":
                    # Need to scale down, terminate excess workers
                    workers_to_remove = scaling_decision.count
                    worker_ids = scaling_decision.worker_ids
                    
                    logger.info(f"Scaling down: Removing {workers_to_remove} worker(s) due to {scaling_decision.reason}")
                    
                    # Terminate each worker
                    for worker_id in worker_ids:
                        try:
                            # Find provider for this worker
                            worker_info = self.worker_manager.get_worker(worker_id)
                            if not worker_info:
                                continue
                                
                            # Get provider name from worker tags
                            provider_name = worker_info.get("tags", {}).get("provider", "unknown")
                            
                            # Terminate worker
                            if provider_name != "unknown":
                                success = self.cloud_provider_manager.terminate_worker(
                                    provider=provider_name,
                                    worker_id=worker_id
                                )
                                
                                if success:
                                    logger.info(f"Terminated worker: {worker_id} on {provider_name}")
                                    
                                    # Deregister worker from worker manager and resource manager
                                    self.worker_manager.deregister_worker(worker_id)
                                    if self.dynamic_resource_manager:
                                        self.dynamic_resource_manager.deregister_worker(worker_id)
                                else:
                                    logger.error(f"Failed to terminate worker: {worker_id} on {provider_name}")
                        except Exception as e:
                            logger.error(f"Error terminating worker {worker_id}: {e}")
                            logger.debug(traceback.format_exc())
                
            except Exception as e:
                logger.error(f"Error in scaling evaluation loop: {e}")
                logger.debug(traceback.format_exc())
        
        logger.info("Scaling evaluation thread stopped")

    def generate_worker_key(self, name: str = None) -> Dict[str, Any]:
        """Generate an API key for a worker.
        
        Args:
            name: Name for the key
            
        Returns:
            Dict containing key information
        """
        return self.security_manager.generate_worker_key(name)
    
    def add_task(self, task_type: str, config: Dict[str, Any], 
                requirements: Dict[str, Any], priority: int = 5) -> str:
        """Add a task to the distribution system.
        
        Args:
            task_type: Type of task (benchmark, test, etc.)
            config: Configuration for the task
            requirements: Hardware requirements for the task
            priority: Priority of the task (lower is higher priority)
            
        Returns:
            Task ID
        """
        return self.task_manager.add_task(None, task_type, priority, config, requirements)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the coordinator.
        
        Returns:
            Dict with statistics
        """
        queue_stats = self.task_manager.get_queue_stats()
        
        worker_count = len(self.worker_manager.workers)
        active_workers = len([
            w for w in self.worker_manager.workers.values()
            if w["status"] == WORKER_STATUS_ACTIVE
        ])
        busy_workers = len([
            w for w in self.worker_manager.workers.values()
            if w["status"] == WORKER_STATUS_BUSY
        ])
        
        return {
            "tasks": queue_stats,
            "workers": {
                "total": worker_count,
                "active": active_workers,
                "busy": busy_workers
            }
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Distributed Testing Framework Coordinator")
    
    parser.add_argument("--host", default="localhost",
                      help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8080,
                      help="Port to bind the server to")
    parser.add_argument("--db-path", default=None,
                      help="Path to the DuckDB database")
    parser.add_argument("--heartbeat-timeout", type=int, default=60,
                      help="Timeout in seconds for worker heartbeats")
    parser.add_argument("--generate-worker-key", action="store_true",
                      help="Generate an API key for a worker")
    parser.add_argument("--key-name", default=None,
                      help="Name for the generated API key")
    parser.add_argument("--security-config", default=None,
                      help="Path to security configuration file")
    
    # Auto recovery system options
    parser.add_argument("--auto-recovery", action="store_true",
                      help="Enable auto recovery system for high availability")
    parser.add_argument("--coordinator-id", default=None,
                      help="Unique identifier for this coordinator instance")
    parser.add_argument("--coordinator-addresses", default=None,
                      help="Comma-separated list of other coordinator addresses (host:port)")
    parser.add_argument("--failover-enabled", action="store_true", default=True,
                      help="Enable automatic failover (default: true)")
    parser.add_argument("--auto-leader-election", action="store_true", default=True,
                      help="Enable automatic leader election (default: true)")
    
    # Performance trend analyzer options
    parser.add_argument("--performance-analyzer", action="store_true",
                      help="Enable performance trend analyzer")
    parser.add_argument("--visualization-path", default=None,
                      help="Path for performance visualizations")
    parser.add_argument("--report", action="store_true",
                      help="Generate a performance report when enabled")
    parser.add_argument("--report-output", default="performance_report.html",
                      help="Output file for the performance report")
                      
    # Dynamic Resource Management options
    parser.add_argument("--enable-drm", action="store_true",
                      help="Enable Dynamic Resource Management with adaptive scaling")
    parser.add_argument("--scaling-interval", type=int, default=60,
                      help="Interval in seconds for scaling evaluation")
    parser.add_argument("--cloud-config", default=None,
                      help="Path to cloud provider configuration file")
    parser.add_argument("--target-utilization", type=float, default=0.7,
                      help="Target resource utilization (0.0-1.0)")
    parser.add_argument("--scale-up-threshold", type=float, default=0.8,
                      help="Threshold to trigger scale up (0.0-1.0)")
    parser.add_argument("--scale-down-threshold", type=float, default=0.3,
                      help="Threshold to trigger scale down (0.0-1.0)")
    
    args = parser.parse_args()
    
    # Load security configuration if provided
    token_secret = None
    if args.security_config:
        try:
            with open(args.security_config, "r") as f:
                config = json.load(f)
                token_secret = config.get("token_secret")
        except Exception as e:
            logger.error(f"Error loading security configuration: {e}")
    
    # Parse coordinator addresses if provided
    coordinator_addresses = []
    if args.coordinator_addresses:
        coordinator_addresses = args.coordinator_addresses.split(",")
    
    # Create coordinator
    # Prepare cloud config path if provided
    cloud_config_path = None
    if args.cloud_config:
        cloud_config_path = args.cloud_config
    
    coordinator = CoordinatorServer(
        host=args.host,
        port=args.port,
        db_path=args.db_path,
        token_secret=token_secret,
        heartbeat_timeout=args.heartbeat_timeout,
        auto_recovery=args.auto_recovery,
        coordinator_id=args.coordinator_id,
        coordinator_addresses=coordinator_addresses,
        performance_analyzer=args.performance_analyzer,
        visualization_path=args.visualization_path
    )
    
    # Configure DRM system if enabled
    if args.enable_drm and coordinator.dynamic_resource_manager:
        # Set scaling interval
        coordinator.scaling_interval = args.scaling_interval
        
        # Update DRM parameters
        coordinator.dynamic_resource_manager.target_utilization = args.target_utilization
        coordinator.dynamic_resource_manager.scale_up_threshold = args.scale_up_threshold
        coordinator.dynamic_resource_manager.scale_down_threshold = args.scale_down_threshold
        
        # Start scaling evaluation thread
        coordinator.scaling_thread = threading.Thread(
            target=coordinator._scaling_evaluation_loop,
            daemon=True
        )
        coordinator.scaling_thread.start()
        logger.info(f"Dynamic Resource Management enabled with scaling interval {args.scaling_interval}s")
    
    # Generate worker key if requested
    if args.generate_worker_key:
        key_info = coordinator.generate_worker_key(args.key_name)
        print("Worker API Key Generated:")
        print(f"Key ID: {key_info['key_id']}")
        print(f"API Key: {key_info['api_key']}")
        print(f"Name: {key_info['name']}")
        print(f"Role: {key_info['role']}")
        print("\nUse this key to authenticate worker nodes with the coordinator.")
        return 0
        
    # Generate performance report if requested
    if args.report and coordinator.performance_analyzer:
        try:
            print(f"Generating performance report to {args.report_output}...")
            
            # Initialize the analyzer to load data from database
            coordinator.performance_analyzer.start()
            
            # Generate the report
            entity_type = "worker"  # Default to worker report
            significant_only = True
            
            report = coordinator.performance_analyzer.get_performance_report(
                entity_type, significant_only
            )
            
            # Export the report
            output_format = "html" if args.report_output.endswith(".html") else "json"
            
            if output_format == "html":
                # Generate HTML report (simplified example)
                with open(args.report_output, "w") as f:
                    f.write("<html><head><title>Performance Report</title></head><body>")
                    f.write(f"<h1>Performance Report - {datetime.now().isoformat()}</h1>")
                    f.write("<h2>Trends</h2>")
                    f.write("<pre>" + json.dumps(report["trends"], indent=2) + "</pre>")
                    f.write("<h2>Anomalies</h2>")
                    f.write("<pre>" + json.dumps(report["anomalies"], indent=2) + "</pre>")
                    f.write("</body></html>")
            else:
                # Export JSON report
                with open(args.report_output, "w") as f:
                    json.dump(report, f, indent=2)
                    
            print(f"Report generated successfully to {args.report_output}")
            
            # Stop the analyzer
            coordinator.performance_analyzer.stop()
            return 0
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            traceback.print_exc()
            return 1
    
    # Start coordinator
    try:
        logger.info("Starting coordinator...")
        anyio.run(coordinator.start())
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())