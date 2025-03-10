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

Usage:
    python coordinator.py --host 0.0.0.0 --port 8080 --db-path ./benchmark_db.duckdb
"""

import os
import sys
import json
import time
import uuid
import asyncio
import logging
import argparse
import threading
import traceback
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path

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
                     worker_capabilities: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get the next task for a worker based on capabilities.
        
        Args:
            worker_id: ID of the worker
            worker_capabilities: Capabilities of the worker
            
        Returns:
            Task dict if a suitable task is found, None otherwise
        """
        with self.task_lock:
            if not self.task_queue:
                return None
                
            # Find first matching task
            matching_index = None
            matching_task = None
            
            for i, (_, _, task_id, task) in enumerate(self.task_queue):
                if self._worker_meets_requirements(worker_capabilities, task["requirements"]):
                    matching_index = i
                    matching_task = task
                    break
                    
            if matching_index is not None:
                # Remove from queue
                self.task_queue.pop(matching_index)
                
                # Mark as assigned
                matching_task["status"] = TASK_STATUS_ASSIGNED
                matching_task["worker_id"] = worker_id
                matching_task["start_time"] = datetime.now()
                matching_task["attempts"] += 1
                
                # Track in running tasks
                self.running_tasks[matching_task["task_id"]] = worker_id
                
                # Update in database if available
                if self.db_manager:
                    self.db_manager.update_task_status(
                        matching_task["task_id"], 
                        TASK_STATUS_ASSIGNED, 
                        worker_id
                    )
                
                logger.info(f"Assigned task {matching_task['task_id']} to worker {worker_id}")
                return matching_task
            
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
        
        return True
    
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
                       tags: Dict[str, Any] = None) -> bool:
        """Register a new worker or update an existing one.
        
        Args:
            worker_id: Unique identifier for the worker
            hostname: Hostname of the worker
            capabilities: Dict containing hardware capabilities
            websocket: WebSocket connection for the worker
            tags: Optional tags for worker categorization
            
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
                "tags": tags or {}
            }
            
            self.workers[worker_id] = worker_info
            
            # Store websocket connection if provided
            if websocket:
                self.active_connections[worker_id] = websocket
            
            # Add to database if available
            if self.db_manager and not worker_exists:
                self.db_manager.add_worker(worker_id, hostname, capabilities, tags)
            
            action = "Updated" if worker_exists else "Registered"
            logger.info(f"{action} worker {worker_id} ({hostname})")
            return True
    
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
                 heartbeat_timeout: int = 60):
        """Initialize the coordinator server.
        
        Args:
            host: Host to bind the server to
            port: Port to bind the server to
            db_path: Path to the DuckDB database
            token_secret: Secret for signing JWT tokens
            heartbeat_timeout: Timeout in seconds for worker heartbeats
        """
        self.host = host
        self.port = port
        self.db_path = db_path
        self.token_secret = token_secret
        self.heartbeat_timeout = heartbeat_timeout
        
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
        
        # WebSocket server
        self.websocket_server = None
        self.running = False
        self.stop_event = asyncio.Event()
        
        # HTTP server for API
        self.http_server = None
    
    async def start(self):
        """Start the coordinator server."""
        if not WEBSOCKETS_AVAILABLE:
            logger.error("WebSockets not available, cannot start server")
            return False
            
        try:
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
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(
                    sig,
                    lambda: asyncio.create_task(self.stop())
                )
            
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
            tags = message.get("tags", {})
            
            if not worker_id or not hostname:
                logger.warning("Invalid registration message")
                await websocket.send(json.dumps({
                    "type": "register_result",
                    "success": False,
                    "error": "worker_id and hostname are required"
                }))
                return
            
            # Register worker
            success = self.worker_manager.register_worker(
                worker_id, hostname, capabilities, websocket, tags
            )
            
            # Send response
            await websocket.send(json.dumps({
                "type": "register_result",
                "success": success,
                "worker_id": worker_id
            }))
        
        elif message_type == "heartbeat":
            # Update worker heartbeat
            worker_id = message.get("worker_id")
            
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
            
            # Get next task
            task = self.task_manager.get_next_task(worker_id, worker["capabilities"])
            
            if task:
                # Send task to worker
                await websocket.send(json.dumps({
                    "type": "get_task_result",
                    "success": True,
                    "worker_id": worker_id,
                    "task": task
                }))
                
                # Update worker status to busy
                self.worker_manager.update_worker_status(worker_id, WORKER_STATUS_BUSY)
            else:
                # No task available
                await websocket.send(json.dumps({
                    "type": "get_task_result",
                    "success": True,
                    "worker_id": worker_id,
                    "task": None
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
        
        else:
            # Unknown message type
            logger.warning(f"Unknown message type: {message_type}")
            await websocket.send(json.dumps({
                "type": "error",
                "error": f"Unknown message type: {message_type}"
            }))
    
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
    
    # Create coordinator
    coordinator = CoordinatorServer(
        host=args.host,
        port=args.port,
        db_path=args.db_path,
        token_secret=token_secret,
        heartbeat_timeout=args.heartbeat_timeout
    )
    
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
    
    # Start coordinator
    try:
        logger.info("Starting coordinator...")
        asyncio.run(coordinator.start())
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())