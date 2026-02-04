#!/usr/bin/env python3
"""
Distributed Testing Framework - Performance Tracker

This module implements the performance tracking system for worker nodes
in the distributed testing framework.
"""

import os
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
import sqlite3
import hashlib
from dataclasses import asdict

from test.tests.api.duckdb_api.distributed_testing.load_balancer.models import WorkerPerformance, TestRequirements, WorkerAssignment

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("performance_tracker")


class PerformanceTracker:
    """Tracks and analyzes performance metrics for worker nodes."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the performance tracker.
        
        Args:
            db_path: Path to the SQLite database file, or None for in-memory DB
        """
        self.db_path = db_path or ":memory:"
        self.lock = threading.RLock()
        
        # In-memory cache of recent performance data
        self.performance_cache: Dict[str, Tuple[WorkerPerformance, datetime]] = {}
        self.cache_expiry = 300  # seconds
        self.max_cache_size = 1000
        
        # Create database tables if they don't exist
        self._init_db()
        
    def _init_db(self):
        """Initialize the database schema."""
        conn = self._get_db_connection()
        try:
            cursor = conn.cursor()
            
            # Create test execution history table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_execution_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                worker_id TEXT NOT NULL,
                test_id TEXT NOT NULL,
                test_type TEXT,
                model_id TEXT,
                model_family TEXT,
                status TEXT NOT NULL,
                execution_time REAL,
                success INTEGER,
                start_time TEXT,
                end_time TEXT,
                result_hash TEXT,
                created_at TEXT NOT NULL
            )
            ''')
            
            # Create worker performance summary table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS worker_performance_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                worker_id TEXT NOT NULL,
                test_type TEXT,
                model_id TEXT,
                model_family TEXT,
                avg_execution_time REAL,
                success_rate REAL,
                sample_count INTEGER,
                min_execution_time REAL,
                max_execution_time REAL,
                std_execution_time REAL,
                last_updated TEXT NOT NULL,
                UNIQUE(worker_id, test_type, model_id, model_family)
            )
            ''')
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            conn.rollback()
        finally:
            conn.close()
            
    def _get_db_connection(self):
        """Get a database connection."""
        return sqlite3.connect(self.db_path)
        
    def record_test_execution(self, assignment: WorkerAssignment) -> None:
        """Record test execution metrics.
        
        Args:
            assignment: Completed assignment data
        """
        with self.lock:
            # Skip if assignment doesn't have required attributes
            if not hasattr(assignment, 'test_id') or not hasattr(assignment, 'worker_id'):
                logger.warning("Invalid assignment data for recording execution")
                return
                
            try:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                # Extract test data
                test_id = assignment.test_id
                worker_id = assignment.worker_id
                status = assignment.status
                execution_time = assignment.execution_time
                success = 1 if assignment.success else 0
                start_time = assignment.started_at.isoformat() if assignment.started_at else None
                end_time = assignment.completed_at.isoformat() if assignment.completed_at else None
                
                # Extract test requirements
                test_type = None
                model_id = None
                model_family = None
                
                if hasattr(assignment, 'test_requirements'):
                    requirements = assignment.test_requirements
                    test_type = requirements.test_type
                    model_id = requirements.model_id
                    model_family = requirements.model_family
                    
                # Generate result hash
                result_hash = None
                if assignment.result:
                    result_str = json.dumps(assignment.result, sort_keys=True)
                    result_hash = hashlib.md5(result_str.encode()).hexdigest()
                    
                # Record execution history
                cursor.execute('''
                INSERT INTO test_execution_history (
                    worker_id, test_id, test_type, model_id, model_family,
                    status, execution_time, success, start_time, end_time,
                    result_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    worker_id, test_id, test_type, model_id, model_family,
                    status, execution_time, success, start_time, end_time,
                    result_hash, datetime.now().isoformat()
                ))
                
                # Only update performance summary for completed tests
                if status in ["completed", "failed"] and execution_time is not None:
                    # Check if summary exists
                    cursor.execute('''
                    SELECT avg_execution_time, success_rate, sample_count,
                           min_execution_time, max_execution_time
                    FROM worker_performance_summary
                    WHERE worker_id = ? AND test_type IS ? AND model_id IS ? AND model_family IS ?
                    ''', (worker_id, test_type, model_id, model_family))
                    
                    row = cursor.fetchone()
                    
                    if row:
                        # Update existing summary
                        avg_time, success_rate, count, min_time, max_time = row
                        
                        # Calculate new values
                        new_count = count + 1
                        new_avg_time = (avg_time * count + execution_time) / new_count
                        new_success_rate = (success_rate * count + success) / new_count
                        new_min_time = min(min_time, execution_time) if min_time is not None else execution_time
                        new_max_time = max(max_time, execution_time) if max_time is not None else execution_time
                        
                        # Update summary
                        cursor.execute('''
                        UPDATE worker_performance_summary
                        SET avg_execution_time = ?,
                            success_rate = ?,
                            sample_count = ?,
                            min_execution_time = ?,
                            max_execution_time = ?,
                            last_updated = ?
                        WHERE worker_id = ? AND test_type IS ? AND model_id IS ? AND model_family IS ?
                        ''', (
                            new_avg_time, new_success_rate, new_count,
                            new_min_time, new_max_time, datetime.now().isoformat(),
                            worker_id, test_type, model_id, model_family
                        ))
                    else:
                        # Insert new summary
                        cursor.execute('''
                        INSERT INTO worker_performance_summary (
                            worker_id, test_type, model_id, model_family,
                            avg_execution_time, success_rate, sample_count,
                            min_execution_time, max_execution_time, std_execution_time,
                            last_updated
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            worker_id, test_type, model_id, model_family,
                            execution_time, success, 1,
                            execution_time, execution_time, 0.0,
                            datetime.now().isoformat()
                        ))
                    
                    # Invalidate cache
                    cache_key = f"{worker_id}:{test_type}:{model_id}"
                    if cache_key in self.performance_cache:
                        del self.performance_cache[cache_key]
                
                conn.commit()
                logger.debug(f"Recorded test execution for {test_id} on {worker_id}")
                
            except Exception as e:
                logger.error(f"Error recording test execution: {e}")
                if conn:
                    conn.rollback()
            finally:
                if conn:
                    conn.close()
    
    def get_worker_performance(self, worker_id: str, test_type: Optional[str] = None,
                             model_id: Optional[str] = None, model_family: Optional[str] = None) -> Optional[WorkerPerformance]:
        """Get performance metrics for a worker.
        
        Args:
            worker_id: Worker ID
            test_type: Test type filter (optional)
            model_id: Model ID filter (optional)
            model_family: Model family filter (optional)
            
        Returns:
            WorkerPerformance object or None if not found
        """
        with self.lock:
            # Check cache first
            cache_key = f"{worker_id}:{test_type}:{model_id}"
            if cache_key in self.performance_cache:
                perf, timestamp = self.performance_cache[cache_key]
                
                # Check if still valid
                if (datetime.now() - timestamp).total_seconds() < self.cache_expiry:
                    return perf
                
                # Remove expired entry
                del self.performance_cache[cache_key]
            
            try:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                # Build query based on parameters
                query = '''
                SELECT worker_id, test_type, model_id, model_family,
                       avg_execution_time, success_rate, sample_count,
                       min_execution_time, max_execution_time, std_execution_time,
                       last_updated
                FROM worker_performance_summary
                WHERE worker_id = ?
                '''
                params = [worker_id]
                
                if test_type is not None:
                    query += " AND test_type IS ?"
                    params.append(test_type)
                    
                if model_id is not None:
                    query += " AND model_id IS ?"
                    params.append(model_id)
                    
                if model_family is not None:
                    query += " AND model_family IS ?"
                    params.append(model_family)
                
                cursor.execute(query, params)
                row = cursor.fetchone()
                
                if not row:
                    return None
                    
                # Convert to WorkerPerformance
                performance = WorkerPerformance(
                    worker_id=row[0],
                    test_type=row[1],
                    model_id=row[2],
                    model_family=row[3],
                    average_execution_time=row[4],
                    success_rate=row[5],
                    sample_count=row[6],
                    min_execution_time=row[7],
                    max_execution_time=row[8],
                    std_execution_time=row[9],
                    last_execution_time=datetime.fromisoformat(row[10]) if row[10] else datetime.now(),
                    total_failures=int(row[6] * (1.0 - row[5])) if row[6] and row[5] is not None else 0
                )
                
                # Cache result
                self.performance_cache[cache_key] = (performance, datetime.now())
                
                # Clean up cache if too large
                if len(self.performance_cache) > self.max_cache_size:
                    self._cleanup_cache()
                    
                return performance
                
            except Exception as e:
                logger.error(f"Error getting worker performance: {e}")
                return None
            finally:
                if conn:
                    conn.close()
    
    def get_performance_history(self, worker_id: Optional[str] = None,
                              test_type: Optional[str] = None,
                              time_range: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get performance history for a worker or test type.
        
        Args:
            worker_id: Worker ID (optional)
            test_type: Test type (optional)
            time_range: Time range in seconds (optional)
            
        Returns:
            List of history entries
        """
        with self.lock:
            try:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                # Build query
                query = '''
                SELECT h.worker_id, h.test_id, h.test_type, h.model_id, h.model_family,
                       h.status, h.execution_time, h.success, h.start_time, h.end_time,
                       h.created_at
                FROM test_execution_history h
                WHERE 1=1
                '''
                params = []
                
                if worker_id:
                    query += " AND h.worker_id = ?"
                    params.append(worker_id)
                    
                if test_type:
                    query += " AND h.test_type = ?"
                    params.append(test_type)
                    
                if time_range:
                    cutoff = (datetime.now() - timedelta(seconds=time_range)).isoformat()
                    query += " AND h.created_at > ?"
                    params.append(cutoff)
                    
                query += " ORDER BY h.created_at DESC LIMIT 1000"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                history = []
                for row in rows:
                    history.append({
                        "worker_id": row[0],
                        "test_id": row[1],
                        "test_type": row[2],
                        "model_id": row[3],
                        "model_family": row[4],
                        "status": row[5],
                        "execution_time": row[6],
                        "success": bool(row[7]) if row[7] is not None else None,
                        "start_time": row[8],
                        "end_time": row[9],
                        "created_at": row[10]
                    })
                    
                return history
                
            except Exception as e:
                logger.error(f"Error getting performance history: {e}")
                return []
            finally:
                if conn:
                    conn.close()
                    
    def _cleanup_cache(self) -> None:
        """Clean up expired cache entries."""
        now = datetime.now()
        # Remove expired entries
        expired_keys = []
        for key, (_, timestamp) in self.performance_cache.items():
            if (now - timestamp).total_seconds() > self.cache_expiry:
                expired_keys.append(key)
                
        for key in expired_keys:
            del self.performance_cache[key]
            
        # If still too many entries, remove oldest
        if len(self.performance_cache) > self.max_cache_size:
            # Sort by timestamp (oldest first)
            sorted_items = sorted(self.performance_cache.items(), 
                                key=lambda item: item[1][1])
            
            # Remove oldest half
            for key, _ in sorted_items[:len(sorted_items)//2]:
                del self.performance_cache[key]