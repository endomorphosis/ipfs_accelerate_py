#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Registry Integration for IPFS Accelerate Python Framework

This module implements the model registry integration system mentioned in NEXT_STEPS.md.
It provides components for linking test results to a model registry, calculating
suitability scores for hardware-model pairs, implementing a hardware recommender,
and adding versioning support for model-hardware compatibility.

Date: March 2025
"""

import os
import sys
import json
import time
import datetime
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent))

# Local imports
try:
    from duckdb_api.core.benchmark_db_api import BenchmarkDBAPI, get_db_connection
    import duckdb_api.core.benchmark_db_query as benchmark_db_query
except ImportError:
    logger.warning("Warning: benchmark_db_api could not be imported. Functionality may be limited.")


class ModelRegistrySchema:
    """Schema extensions for model registry integration with DuckDB database."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize with optional database path."""
        self.db_path = db_path or os.environ.get('BENCHMARK_DB_PATH', './benchmark_db.duckdb')
    
    def _get_connection(self):
        """Get a connection to the database."""
        try:
            return get_db_connection(self.db_path)
        except NameError:
            # Fallback if get_db_connection is not available
            import duckdb
            return duckdb.connect(self.db_path)
    
    def create_schema_extensions(self) -> bool:
        """
        Create schema extensions for model registry integration.
        
        Returns:
            bool: Success status
        """
        logger.info("Creating schema extensions for model registry integration")
        
        conn = self._get_connection()
        
        try:
            # Start a transaction for consistency
            conn.execute("BEGIN TRANSACTION")
            
            # Create model registry versions table if it doesn't exist
            conn.execute("""
            CREATE TABLE IF NOT EXISTS model_registry_versions (
                id INTEGER PRIMARY KEY,
                model_id INTEGER,
                version_tag VARCHAR,
                version_hash VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSON,
                FOREIGN KEY (model_id) REFERENCES models(model_id)
            )
            """)
            
            # Create hardware-model compatibility table if it doesn't exist
            conn.execute("""
            CREATE TABLE IF NOT EXISTS hardware_model_compatibility (
                id INTEGER PRIMARY KEY,
                model_id INTEGER,
                hardware_id INTEGER,
                compatibility_score FLOAT,
                suitability_score FLOAT,
                recommended_batch_size INTEGER,
                recommended_precision VARCHAR,
                memory_requirement FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                FOREIGN KEY (model_id) REFERENCES models(model_id),
                FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id)
            )
            """)
            
            # Create task recommendations table if it doesn't exist
            conn.execute("""
            CREATE TABLE IF NOT EXISTS task_recommendations (
                id INTEGER PRIMARY KEY,
                task_type VARCHAR,
                model_id INTEGER,
                hardware_id INTEGER,
                suitability_score FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES models(model_id),
                FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id)
            )
            """)
            
            # Create hardware compatibility snapshots table if it doesn't exist
            conn.execute("""
            CREATE TABLE IF NOT EXISTS hardware_compatibility_snapshots (
                id INTEGER PRIMARY KEY,
                model_id INTEGER,
                version_id INTEGER,
                hardware_id INTEGER,
                compatibility_score FLOAT,
                suitability_score FLOAT,
                recommended_batch_size INTEGER,
                recommended_precision VARCHAR,
                memory_requirement FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES models(model_id),
                FOREIGN KEY (version_id) REFERENCES model_registry_versions(id),
                FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id)
            )
            """)
            
            # Commit the transaction
            conn.execute("COMMIT")
            
            logger.info("Schema extensions created successfully!")
            return True
            
        except Exception as e:
            # Rollback in case of error
            try:
                conn.execute("ROLLBACK")
            except:
                pass
                
            logger.error(f"Error creating schema extensions: {e}")
            return False
            
        finally:
            conn.close()


class ModelRegistryIntegration:
    """Integrates test results with model registry and calculates suitability scores."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize with optional database path."""
        self.db_path = db_path or os.environ.get('BENCHMARK_DB_PATH', './benchmark_db.duckdb')
        
        # Create schema if needed
        self.schema = ModelRegistrySchema(self.db_path)
    
    def _get_connection(self):
        """Get a connection to the database."""
        try:
            return get_db_connection(self.db_path)
        except NameError:
            # Fallback if get_db_connection is not available
            import duckdb
            return duckdb.connect(self.db_path)
    
    def _get_model_id(self, conn, model_name: str) -> Optional[int]:
        """Get model ID from name."""
        result = conn.execute(
            "SELECT model_id FROM models WHERE model_name = ?",
            [model_name]
        ).fetchone()
        
        if result:
            return result[0]
        
        # Try to add the model if it doesn't exist
        try:
            # Get next model ID
            max_id = conn.execute("SELECT MAX(model_id) FROM models").fetchone()[0]
            model_id = 1 if max_id is None else max_id + 1
            
            # Add model
            conn.execute(
                """
                INSERT INTO models (model_id, model_name)
                VALUES (?, ?)
                """,
                [model_id, model_name]
            )
            
            logger.info(f"Added new model: {model_name} (ID: {model_id})")
            return model_id
            
        except Exception as e:
            logger.error(f"Error adding model: {e}")
            return None
    
    def _get_hardware_id(self, conn, hardware_type: str) -> Optional[int]:
        """Get hardware ID from type."""
        result = conn.execute(
            "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = ?",
            [hardware_type]
        ).fetchone()
        
        if result:
            return result[0]
        
        # Try to add the hardware if it doesn't exist
        try:
            # Get next hardware ID
            max_id = conn.execute("SELECT MAX(hardware_id) FROM hardware_platforms").fetchone()[0]
            hardware_id = 1 if max_id is None else max_id + 1
            
            # Add hardware
            conn.execute(
                """
                INSERT INTO hardware_platforms (hardware_id, hardware_type)
                VALUES (?, ?)
                """,
                [hardware_id, hardware_type]
            )
            
            logger.info(f"Added new hardware: {hardware_type} (ID: {hardware_id})")
            return hardware_id
            
        except Exception as e:
            logger.error(f"Error adding hardware: {e}")
            return None
    
    def link_test_results(self, 
                        model_name: str, 
                        model_version: str, 
                        result_ids: List[int]) -> bool:
        """
        Link test results to a model version in the registry.
        
        Args:
            model_name: Name of the model
            model_version: Version tag of the model
            result_ids: List of test result IDs
            
        Returns:
            Success status
        """
        conn = self._get_connection()
        
        try:
            # Get model ID
            model_id = self._get_model_id(conn, model_name)
            if model_id is None:
                logger.error(f"Could not find or create model: {model_name}")
                return False
            
            # Check if version exists
            result = conn.execute(
                """
                SELECT id FROM model_registry_versions 
                WHERE model_id = ? AND version_tag = ?
                """,
                [model_id, model_version]
            ).fetchone()
            
            version_id = None
            if result:
                version_id = result[0]
                logger.info(f"Using existing version: {model_version} (ID: {version_id})")
            else:
                # Create version
                max_id = conn.execute("SELECT MAX(id) FROM model_registry_versions").fetchone()[0]
                version_id = 1 if max_id is None else max_id + 1
                
                conn.execute(
                    """
                    INSERT INTO model_registry_versions 
                    (id, model_id, version_tag)
                    VALUES (?, ?, ?)
                    """,
                    [version_id, model_id, model_version]
                )
                
                logger.info(f"Created version: {model_version} (ID: {version_id})")
            
            # Link test results
            updated = 0
            for result_id in result_ids:
                try:
                    # Check if result exists
                    result = conn.execute(
                        "SELECT id FROM test_results WHERE id = ?",
                        [result_id]
                    ).fetchone()
                    
                    if not result:
                        logger.warning(f"Test result ID {result_id} not found")
                        continue
                    
                    # Update test result with version ID
                    conn.execute(
                        """
                        UPDATE test_results 
                        SET version_id = ?,
                            metadata = json_insert(coalesce(metadata, '{}'), '$.model_version', ?)
                        WHERE id = ?
                        """,
                        [version_id, model_version, result_id]
                    )
                    
                    updated += 1
                except Exception as e:
                    logger.error(f"Error linking test result {result_id}: {e}")
            
            logger.info(f"Linked {updated}/{len(result_ids)} test results to version {model_version}")
            return updated > 0
            
        except Exception as e:
            logger.error(f"Error linking test results: {e}")
            return False
            
        finally:
            conn.close()
    
    def calculate_suitability_scores(self, 
                                   model_id: Optional[int] = None,
                                   model_name: Optional[str] = None, 
                                   hardware_id: Optional[int] = None,
                                   hardware_type: Optional[str] = None,
                                   update_db: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Calculate compatibility and suitability scores for hardware-model pairs.
        
        Args:
            model_id: Optional model ID to filter by
            model_name: Optional model name to filter by
            hardware_id: Optional hardware ID to filter by
            hardware_type: Optional hardware type to filter by
            update_db: Whether to update the database with calculated scores
            
        Returns:
            Dictionary mapping model_name -> hardware_type -> scores
        """
        conn = self._get_connection()
        
        try:
            # Build query to get performance data
            query = """
            SELECT 
                m.model_id,
                m.model_name,
                hp.hardware_id,
                hp.hardware_type,
                tr.batch_size,
                tr.throughput,
                tr.latency,
                tr.memory_usage
            FROM 
                test_results tr
            JOIN 
                models m ON tr.model_id = m.model_id
            JOIN 
                hardware_platforms hp ON tr.hardware_id = hp.hardware_id
            WHERE
                tr.throughput IS NOT NULL
            """
            
            params = []
            
            # Add filters
            if model_id is not None:
                query += " AND m.model_id = ?"
                params.append(model_id)
            elif model_name is not None:
                query += " AND m.model_name = ?"
                params.append(model_name)
            
            if hardware_id is not None:
                query += " AND hp.hardware_id = ?"
                params.append(hardware_id)
            elif hardware_type is not None:
                query += " AND hp.hardware_type = ?"
                params.append(hardware_type)
            
            # Execute query
            results = conn.execute(query, params).fetchall()
            if not results:
                logger.warning("No performance data found for calculating suitability scores")
                return {}
            
            # Group results by model_name, hardware_type
            grouped_data = {}
            for row in results:
                m_id, m_name, h_id, h_type, batch_size, throughput, latency, memory = row
                
                if m_name not in grouped_data:
                    grouped_data[m_name] = {}
                
                if h_type not in grouped_data[m_name]:
                    grouped_data[m_name][h_type] = {
                        'model_id': m_id,
                        'hardware_id': h_id,
                        'throughput_values': [],
                        'latency_values': [],
                        'memory_values': [],
                        'batch_sizes': []
                    }
                
                data = grouped_data[m_name][h_type]
                data['throughput_values'].append(throughput)
                data['latency_values'].append(latency)
                data['memory_values'].append(memory if memory is not None else 0)
                data['batch_sizes'].append(batch_size if batch_size is not None else 1)
            
            # Calculate scores for each model-hardware pair
            all_throughputs = []
            all_latencies = []
            all_memories = []
            
            # First pass: collect all values for normalization
            for m_name, hardware_data in grouped_data.items():
                for h_type, data in hardware_data.items():
                    all_throughputs.extend(data['throughput_values'])
                    all_latencies.extend(data['latency_values'])
                    all_memories.extend(data['memory_values'])
            
            # Calculate statistics
            max_throughput = max(all_throughputs) if all_throughputs else 1
            min_latency = min(all_latencies) if all_latencies else 1
            min_memory = min(all_memories) if all_memories else 1
            max_memory = max(all_memories) if all_memories else 1
            
            # Second pass: calculate scores
            scores = {}
            for m_name, hardware_data in grouped_data.items():
                scores[m_name] = {}
                
                for h_type, data in hardware_data.items():
                    # Calculate throughput score (higher is better)
                    avg_throughput = np.mean(data['throughput_values'])
                    throughput_score = avg_throughput / max_throughput
                    
                    # Calculate latency score (lower is better)
                    avg_latency = np.mean(data['latency_values'])
                    latency_score = min_latency / avg_latency
                    
                    # Calculate memory score (lower is better)
                    avg_memory = np.mean(data['memory_values'])
                    memory_range = max_memory - min_memory
                    memory_score = 1.0 - ((avg_memory - min_memory) / memory_range) if memory_range > 0 else 0.5
                    
                    # Calculate confidence factor
                    test_count = len(data['throughput_values'])
                    confidence_factor = min(1.0, 0.5 + 0.1 * test_count)
                    
                    # Calculate recommended batch size
                    best_throughput_idx = np.argmax(data['throughput_values'])
                    recommended_batch_size = data['batch_sizes'][best_throughput_idx]
                    
                    # Calculate compatibility score
                    compatibility_score = (throughput_score * 0.5) + (latency_score * 0.3) + (memory_score * 0.2)
                    
                    # Calculate suitability score
                    suitability_score = compatibility_score * confidence_factor
                    
                    # Store scores
                    scores[m_name][h_type] = {
                        'model_id': data['model_id'],
                        'hardware_id': data['hardware_id'],
                        'throughput_score': throughput_score,
                        'latency_score': latency_score,
                        'memory_score': memory_score,
                        'compatibility_score': compatibility_score,
                        'confidence_factor': confidence_factor,
                        'suitability_score': suitability_score,
                        'recommended_batch_size': recommended_batch_size,
                        'test_count': test_count
                    }
                    
                    # Update database if requested
                    if update_db:
                        try:
                            # Check if entry exists
                            result = conn.execute(
                                """
                                SELECT id FROM hardware_model_compatibility
                                WHERE model_id = ? AND hardware_id = ?
                                """,
                                [data['model_id'], data['hardware_id']]
                            ).fetchone()
                            
                            if result:
                                # Update existing entry
                                conn.execute(
                                    """
                                    UPDATE hardware_model_compatibility
                                    SET compatibility_score = ?,
                                        suitability_score = ?,
                                        recommended_batch_size = ?,
                                        memory_requirement = ?,
                                        last_updated = CURRENT_TIMESTAMP
                                    WHERE model_id = ? AND hardware_id = ?
                                    """,
                                    [
                                        compatibility_score,
                                        suitability_score,
                                        recommended_batch_size,
                                        avg_memory,
                                        data['model_id'],
                                        data['hardware_id']
                                    ]
                                )
                            else:
                                # Create new entry
                                max_id = conn.execute("SELECT MAX(id) FROM hardware_model_compatibility").fetchone()[0]
                                entry_id = 1 if max_id is None else max_id + 1
                                
                                conn.execute(
                                    """
                                    INSERT INTO hardware_model_compatibility
                                    (id, model_id, hardware_id, compatibility_score, suitability_score, 
                                     recommended_batch_size, memory_requirement)
                                    VALUES (?, ?, ?, ?, ?, ?, ?)
                                    """,
                                    [
                                        entry_id,
                                        data['model_id'],
                                        data['hardware_id'],
                                        compatibility_score,
                                        suitability_score,
                                        recommended_batch_size,
                                        avg_memory
                                    ]
                                )
                        except Exception as e:
                            logger.error(f"Error updating database for {m_name} on {h_type}: {e}")
            
            return scores
            
        except Exception as e:
            logger.error(f"Error calculating suitability scores: {e}")
            return {}
            
        finally:
            conn.close()


class HardwareRecommender:
    """Recommends hardware based on model and task requirements."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize with optional database path."""
        self.db_path = db_path or os.environ.get('BENCHMARK_DB_PATH', './benchmark_db.duckdb')
        
        # Initialize ModelRegistryIntegration
        self.registry = ModelRegistryIntegration(self.db_path)
    
    def _get_connection(self):
        """Get a connection to the database."""
        try:
            return get_db_connection(self.db_path)
        except NameError:
            # Fallback if get_db_connection is not available
            import duckdb
            return duckdb.connect(self.db_path)
    
    def recommend_hardware(self, 
                         model_name: str, 
                         task_type: str = 'inference',
                         batch_size: Optional[int] = None,
                         latency_sensitive: bool = False,
                         memory_constrained: bool = False,
                         top_k: int = 3,
                         update_db: bool = True) -> List[Dict[str, Any]]:
        """
        Recommend hardware for a model and task.
        
        Args:
            model_name: Name of the model
            task_type: Type of task ('inference' or 'training')
            batch_size: Optional batch size
            latency_sensitive: Whether the task is latency sensitive
            memory_constrained: Whether the environment is memory constrained
            top_k: Number of recommendations to return
            update_db: Whether to update the database with recommendations
            
        Returns:
            List of hardware recommendations, sorted by suitability
        """
        conn = self._get_connection()
        
        try:
            # Get model ID
            result = conn.execute(
                "SELECT model_id FROM models WHERE model_name = ?",
                [model_name]
            ).fetchone()
            
            if not result:
                logger.error(f"Model not found: {model_name}")
                return []
            
            model_id = result[0]
            
            # Get hardware compatibility scores
            query = """
            SELECT 
                hmc.hardware_id,
                hp.hardware_type,
                hmc.compatibility_score,
                hmc.suitability_score,
                hmc.recommended_batch_size,
                hmc.memory_requirement
            FROM 
                hardware_model_compatibility hmc
            JOIN 
                hardware_platforms hp ON hmc.hardware_id = hp.hardware_id
            WHERE 
                hmc.model_id = ?
            """
            
            results = conn.execute(query, [model_id]).fetchall()
            
            if not results:
                # Calculate scores if none found
                logger.info(f"No hardware compatibility scores found for {model_name}, calculating now...")
                self.registry.calculate_suitability_scores(model_name=model_name, update_db=True)
                
                # Try again
                results = conn.execute(query, [model_id]).fetchall()
                
                if not results:
                    logger.error(f"No hardware platforms found for {model_name}")
                    return []
            
            # Process results
            recommendations = []
            for row in results:
                hardware_id, hardware_type, compatibility_score, suitability_score, recommended_batch_size, memory_requirement = row
                
                # Apply task-specific weighting
                if task_type == 'inference':
                    # For inference: prioritize latency and throughput
                    task_weight = 0.7 if latency_sensitive else 0.5
                else:
                    # For training: prioritize throughput and memory efficiency
                    task_weight = 0.7 if memory_constrained else 0.5
                
                # Apply requirement-specific weighting
                if latency_sensitive:
                    # Increase weight of latency score
                    latency_weight = 0.6
                    throughput_weight = 0.3
                    memory_weight = 0.1
                elif memory_constrained:
                    # Increase weight of memory efficiency
                    latency_weight = 0.2
                    throughput_weight = 0.3
                    memory_weight = 0.5
                else:
                    # Default weights
                    latency_weight = 0.3
                    throughput_weight = 0.5
                    memory_weight = 0.2
                
                # Get individual scores (assuming they're stored in compatibility score calculation)
                # If not available, use the compatibility score as a proxy
                throughput_score = compatibility_score * throughput_weight
                latency_score = compatibility_score * latency_weight
                memory_score = compatibility_score * memory_weight
                
                # Calculate weighted score
                weighted_score = (throughput_score + latency_score + memory_score) * task_weight
                
                # Adjust score based on batch size if specified
                if batch_size is not None and recommended_batch_size is not None:
                    batch_factor = min(1.0, batch_size / recommended_batch_size) if batch_size < recommended_batch_size else 1.0
                    weighted_score *= batch_factor
                
                # Store recommendation
                recommendations.append({
                    'hardware_id': hardware_id,
                    'hardware_type': hardware_type,
                    'suitability_score': suitability_score,
                    'weighted_score': weighted_score,
                    'recommended_batch_size': recommended_batch_size,
                    'memory_requirement': memory_requirement
                })
            
            # Sort by weighted score
            recommendations.sort(key=lambda x: x['weighted_score'], reverse=True)
            
            # Limit to top_k
            recommendations = recommendations[:top_k]
            
            # Update task recommendations in database if requested
            if update_db:
                for rank, rec in enumerate(recommendations):
                    try:
                        # Check if recommendation exists
                        result = conn.execute(
                            """
                            SELECT id FROM task_recommendations
                            WHERE task_type = ? AND model_id = ? AND hardware_id = ?
                            """,
                            [task_type, model_id, rec['hardware_id']]
                        ).fetchone()
                        
                        if result:
                            # Update existing recommendation
                            conn.execute(
                                """
                                UPDATE task_recommendations
                                SET suitability_score = ?,
                                    last_updated = CURRENT_TIMESTAMP
                                WHERE id = ?
                                """,
                                [rec['weighted_score'], result[0]]
                            )
                        else:
                            # Create new recommendation
                            max_id = conn.execute("SELECT MAX(id) FROM task_recommendations").fetchone()[0]
                            entry_id = 1 if max_id is None else max_id + 1
                            
                            conn.execute(
                                """
                                INSERT INTO task_recommendations
                                (id, task_type, model_id, hardware_id, suitability_score)
                                VALUES (?, ?, ?, ?, ?)
                                """,
                                [entry_id, task_type, model_id, rec['hardware_id'], rec['weighted_score']]
                            )
                    except Exception as e:
                        logger.error(f"Error updating task recommendation: {e}")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error recommending hardware: {e}")
            return []
            
        finally:
            conn.close()
    
    def update_task_recommendations(self, task_type: str) -> int:
        """
        Update all task recommendations for a specific task type.
        
        Args:
            task_type: Type of task ('inference' or 'training')
            
        Returns:
            Number of updated recommendations
        """
        conn = self._get_connection()
        
        try:
            # Get all models
            models = conn.execute("SELECT model_id, model_name FROM models").fetchall()
            
            updated = 0
            for model_id, model_name in models:
                try:
                    # Get recommendations
                    recommendations = self.recommend_hardware(
                        model_name=model_name,
                        task_type=task_type,
                        update_db=True
                    )
                    
                    updated += len(recommendations)
                except Exception as e:
                    logger.error(f"Error updating recommendations for {model_name}: {e}")
            
            logger.info(f"Updated {updated} task recommendations for {task_type}")
            return updated
            
        except Exception as e:
            logger.error(f"Error updating task recommendations: {e}")
            return 0
            
        finally:
            conn.close()


class VersionControlSystem:
    """Manages model versions and hardware compatibility over time."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize with optional database path."""
        self.db_path = db_path or os.environ.get('BENCHMARK_DB_PATH', './benchmark_db.duckdb')
        
        # Initialize ModelRegistryIntegration
        self.registry = ModelRegistryIntegration(self.db_path)
    
    def _get_connection(self):
        """Get a connection to the database."""
        try:
            return get_db_connection(self.db_path)
        except NameError:
            # Fallback if get_db_connection is not available
            import duckdb
            return duckdb.connect(self.db_path)
    
    def add_model_version(self, 
                        model_name: str, 
                        version_tag: str,
                        version_hash: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> Optional[int]:
        """
        Add a new model version to the registry.
        
        Args:
            model_name: Name of the model
            version_tag: Version tag (e.g., 'v1.0.0')
            version_hash: Optional version hash (e.g., Git commit hash)
            metadata: Optional metadata for the version
            
        Returns:
            Version ID if successful, None otherwise
        """
        conn = self._get_connection()
        
        try:
            # Get model ID
            result = conn.execute(
                "SELECT model_id FROM models WHERE model_name = ?",
                [model_name]
            ).fetchone()
            
            if not result:
                logger.error(f"Model not found: {model_name}")
                return None
            
            model_id = result[0]
            
            # Check if version exists
            result = conn.execute(
                """
                SELECT id FROM model_registry_versions
                WHERE model_id = ? AND version_tag = ?
                """,
                [model_id, version_tag]
            ).fetchone()
            
            if result:
                logger.warning(f"Version {version_tag} already exists for {model_name}")
                return result[0]
            
            # Create version
            max_id = conn.execute("SELECT MAX(id) FROM model_registry_versions").fetchone()[0]
            version_id = 1 if max_id is None else max_id + 1
            
            # Convert metadata to JSON if provided
            metadata_json = json.dumps(metadata) if metadata else None
            
            conn.execute(
                """
                INSERT INTO model_registry_versions
                (id, model_id, version_tag, version_hash, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                [version_id, model_id, version_tag, version_hash, metadata_json]
            )
            
            logger.info(f"Added version {version_tag} for {model_name}")
            return version_id
            
        except Exception as e:
            logger.error(f"Error adding model version: {e}")
            return None
            
        finally:
            conn.close()
    
    def get_version_history(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Get version history for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of version entries
        """
        conn = self._get_connection()
        
        try:
            # Get model ID
            result = conn.execute(
                "SELECT model_id FROM models WHERE model_name = ?",
                [model_name]
            ).fetchone()
            
            if not result:
                logger.error(f"Model not found: {model_name}")
                return []
            
            model_id = result[0]
            
            # Get version history
            results = conn.execute(
                """
                SELECT 
                    id, version_tag, version_hash, created_at, metadata
                FROM 
                    model_registry_versions
                WHERE 
                    model_id = ?
                ORDER BY 
                    created_at DESC
                """,
                [model_id]
            ).fetchall()
            
            # Format results
            versions = []
            for row in results:
                version_id, version_tag, version_hash, created_at, metadata_json = row
                
                # Parse metadata
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                versions.append({
                    'id': version_id,
                    'version_tag': version_tag,
                    'version_hash': version_hash,
                    'created_at': created_at,
                    'metadata': metadata
                })
            
            return versions
            
        except Exception as e:
            logger.error(f"Error getting version history: {e}")
            return []
            
        finally:
            conn.close()
    
    def create_compatibility_snapshot(self, 
                                    model_name: str, 
                                    version_tag: str) -> bool:
        """
        Create a snapshot of hardware compatibility for a model version.
        
        Args:
            model_name: Name of the model
            version_tag: Version tag
            
        Returns:
            Success status
        """
        conn = self._get_connection()
        
        try:
            # Get model ID
            result = conn.execute(
                "SELECT model_id FROM models WHERE model_name = ?",
                [model_name]
            ).fetchone()
            
            if not result:
                logger.error(f"Model not found: {model_name}")
                return False
            
            model_id = result[0]
            
            # Get version ID
            result = conn.execute(
                """
                SELECT id FROM model_registry_versions
                WHERE model_id = ? AND version_tag = ?
                """,
                [model_id, version_tag]
            ).fetchone()
            
            if not result:
                logger.error(f"Version {version_tag} not found for {model_name}")
                return False
            
            version_id = result[0]
            
            # Get hardware compatibility data
            results = conn.execute(
                """
                SELECT 
                    hardware_id, compatibility_score, suitability_score,
                    recommended_batch_size, memory_requirement
                FROM 
                    hardware_model_compatibility
                WHERE 
                    model_id = ?
                """,
                [model_id]
            ).fetchall()
            
            if not results:
                # Try to calculate scores
                self.registry.calculate_suitability_scores(model_id=model_id, update_db=True)
                
                # Try again
                results = conn.execute(
                    """
                    SELECT 
                        hardware_id, compatibility_score, suitability_score,
                        recommended_batch_size, memory_requirement
                    FROM 
                        hardware_model_compatibility
                    WHERE 
                        model_id = ?
                    """,
                    [model_id]
                ).fetchall()
                
                if not results:
                    logger.error(f"No hardware compatibility data found for {model_name}")
                    return False
            
            # Create snapshots
            created = 0
            for row in results:
                hardware_id, compatibility_score, suitability_score, recommended_batch_size, memory_requirement = row
                
                # Check if snapshot exists
                result = conn.execute(
                    """
                    SELECT id FROM hardware_compatibility_snapshots
                    WHERE model_id = ? AND version_id = ? AND hardware_id = ?
                    """,
                    [model_id, version_id, hardware_id]
                ).fetchone()
                
                if result:
                    logger.warning(f"Snapshot already exists for {model_name} version {version_tag} on hardware {hardware_id}")
                    continue
                
                # Create snapshot
                max_id = conn.execute("SELECT MAX(id) FROM hardware_compatibility_snapshots").fetchone()[0]
                snapshot_id = 1 if max_id is None else max_id + 1
                
                conn.execute(
                    """
                    INSERT INTO hardware_compatibility_snapshots
                    (id, model_id, version_id, hardware_id, compatibility_score, suitability_score,
                     recommended_batch_size, memory_requirement)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        snapshot_id, model_id, version_id, hardware_id,
                        compatibility_score, suitability_score,
                        recommended_batch_size, memory_requirement
                    ]
                )
                
                created += 1
            
            logger.info(f"Created {created} compatibility snapshots for {model_name} version {version_tag}")
            return created > 0
            
        except Exception as e:
            logger.error(f"Error creating compatibility snapshot: {e}")
            return False
            
        finally:
            conn.close()
    
    def compare_compatibility_versions(self, 
                                     model_name: str, 
                                     version_tag1: str, 
                                     version_tag2: str) -> List[Dict[str, Any]]:
        """
        Compare hardware compatibility between two model versions.
        
        Args:
            model_name: Name of the model
            version_tag1: First version tag
            version_tag2: Second version tag
            
        Returns:
            List of compatibility changes
        """
        conn = self._get_connection()
        
        try:
            # Get model ID
            result = conn.execute(
                "SELECT model_id FROM models WHERE model_name = ?",
                [model_name]
            ).fetchone()
            
            if not result:
                logger.error(f"Model not found: {model_name}")
                return []
            
            model_id = result[0]
            
            # Get version IDs
            result1 = conn.execute(
                """
                SELECT id FROM model_registry_versions
                WHERE model_id = ? AND version_tag = ?
                """,
                [model_id, version_tag1]
            ).fetchone()
            
            result2 = conn.execute(
                """
                SELECT id FROM model_registry_versions
                WHERE model_id = ? AND version_tag = ?
                """,
                [model_id, version_tag2]
            ).fetchone()
            
            if not result1 or not result2:
                logger.error(f"One or both versions not found: {version_tag1}, {version_tag2}")
                return []
            
            version_id1 = result1[0]
            version_id2 = result2[0]
            
            # Create snapshots if they don't exist
            self.create_compatibility_snapshot(model_name, version_tag1)
            self.create_compatibility_snapshot(model_name, version_tag2)
            
            # Get hardware platforms
            hardware_platforms = conn.execute(
                "SELECT hardware_id, hardware_type FROM hardware_platforms"
            ).fetchall()
            
            # Compare snapshots for each hardware platform
            changes = []
            for hardware_id, hardware_type in hardware_platforms:
                # Get snapshots
                snapshot1 = conn.execute(
                    """
                    SELECT compatibility_score, suitability_score, recommended_batch_size, memory_requirement
                    FROM hardware_compatibility_snapshots
                    WHERE model_id = ? AND version_id = ? AND hardware_id = ?
                    """,
                    [model_id, version_id1, hardware_id]
                ).fetchone()
                
                snapshot2 = conn.execute(
                    """
                    SELECT compatibility_score, suitability_score, recommended_batch_size, memory_requirement
                    FROM hardware_compatibility_snapshots
                    WHERE model_id = ? AND version_id = ? AND hardware_id = ?
                    """,
                    [model_id, version_id2, hardware_id]
                ).fetchone()
                
                if not snapshot1 or not snapshot2:
                    # Skip if either snapshot doesn't exist
                    continue
                
                # Calculate changes
                compatibility1, suitability1, batch_size1, memory1 = snapshot1
                compatibility2, suitability2, batch_size2, memory2 = snapshot2
                
                compatibility_change = ((compatibility2 - compatibility1) / compatibility1) * 100 if compatibility1 else 0
                suitability_change = ((suitability2 - suitability1) / suitability1) * 100 if suitability1 else 0
                batch_size_change = batch_size2 - batch_size1 if batch_size1 and batch_size2 else 0
                memory_change = ((memory2 - memory1) / memory1) * 100 if memory1 else 0
                
                # Add to changes
                changes.append({
                    'hardware_id': hardware_id,
                    'hardware_type': hardware_type,
                    'compatibility_v1': compatibility1,
                    'compatibility_v2': compatibility2,
                    'compatibility_change': compatibility_change,
                    'suitability_v1': suitability1,
                    'suitability_v2': suitability2,
                    'suitability_change': suitability_change,
                    'batch_size_v1': batch_size1,
                    'batch_size_v2': batch_size2,
                    'batch_size_change': batch_size_change,
                    'memory_v1': memory1,
                    'memory_v2': memory2,
                    'memory_change': memory_change
                })
            
            # Sort by compatibility change (largest absolute change first)
            changes.sort(key=lambda x: abs(x['compatibility_change']), reverse=True)
            
            return changes
            
        except Exception as e:
            logger.error(f"Error comparing compatibility versions: {e}")
            return []
            
        finally:
            conn.close()


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Model Registry Integration')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Create schema command
    schema_parser = subparsers.add_parser('create-schema', help='Create schema extensions')
    schema_parser.add_argument('--db-path', help='Database path')
    
    # Link tests command
    link_parser = subparsers.add_parser('link-tests', help='Link test results to a model version')
    link_parser.add_argument('--db-path', help='Database path')
    link_parser.add_argument('--model', required=True, help='Model name')
    link_parser.add_argument('--version', required=True, help='Version tag')
    link_parser.add_argument('--result-ids', required=True, help='Comma-separated list of test result IDs')
    
    # Calculate scores command
    scores_parser = subparsers.add_parser('calculate-scores', help='Calculate suitability scores')
    scores_parser.add_argument('--db-path', help='Database path')
    scores_parser.add_argument('--model', help='Model name filter')
    scores_parser.add_argument('--hardware', help='Hardware type filter')
    scores_parser.add_argument('--no-update', action='store_true', help='Do not update the database')
    
    # Recommend hardware command
    recommend_parser = subparsers.add_parser('recommend', help='Recommend hardware')
    recommend_parser.add_argument('--db-path', help='Database path')
    recommend_parser.add_argument('--model', required=True, help='Model name')
    recommend_parser.add_argument('--task', choices=['inference', 'training'], default='inference', help='Task type')
    recommend_parser.add_argument('--batch-size', type=int, help='Batch size')
    recommend_parser.add_argument('--latency-sensitive', action='store_true', help='Task is latency sensitive')
    recommend_parser.add_argument('--memory-constrained', action='store_true', help='Environment is memory constrained')
    recommend_parser.add_argument('--top-k', type=int, default=3, help='Number of recommendations to return')
    recommend_parser.add_argument('--no-update', action='store_true', help='Do not update the database')
    
    # Update task recommendations command
    update_task_parser = subparsers.add_parser('update-task', help='Update task recommendations')
    update_task_parser.add_argument('--db-path', help='Database path')
    update_task_parser.add_argument('--task', required=True, choices=['inference', 'training'], help='Task type')
    
    # Add version command
    add_version_parser = subparsers.add_parser('add-version', help='Add a model version')
    add_version_parser.add_argument('--db-path', help='Database path')
    add_version_parser.add_argument('--model', required=True, help='Model name')
    add_version_parser.add_argument('--version', required=True, help='Version tag')
    add_version_parser.add_argument('--hash', help='Version hash')
    add_version_parser.add_argument('--metadata', help='JSON metadata')
    
    # Version history command
    version_history_parser = subparsers.add_parser('version-history', help='Get version history')
    version_history_parser.add_argument('--db-path', help='Database path')
    version_history_parser.add_argument('--model', required=True, help='Model name')
    
    # Create snapshot command
    create_snapshot_parser = subparsers.add_parser('create-snapshot', help='Create compatibility snapshot')
    create_snapshot_parser.add_argument('--db-path', help='Database path')
    create_snapshot_parser.add_argument('--model', required=True, help='Model name')
    create_snapshot_parser.add_argument('--version', required=True, help='Version tag')
    
    # Compare versions command
    compare_versions_parser = subparsers.add_parser('compare-versions', help='Compare compatibility versions')
    compare_versions_parser.add_argument('--db-path', help='Database path')
    compare_versions_parser.add_argument('--model', required=True, help='Model name')
    compare_versions_parser.add_argument('--version1', required=True, help='First version tag')
    compare_versions_parser.add_argument('--version2', required=True, help='Second version tag')
    compare_versions_parser.add_argument('--output', help='Output JSON file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'create-schema':
        schema = ModelRegistrySchema(args.db_path)
        success = schema.create_schema_extensions()
        if success:
            print("Schema extensions created successfully!")
        else:
            print("Failed to create schema extensions")
            return 1
    
    elif args.command == 'link-tests':
        integration = ModelRegistryIntegration(args.db_path)
        result_ids = [int(id.strip()) for id in args.result_ids.split(',')]
        success = integration.link_test_results(args.model, args.version, result_ids)
        if success:
            print(f"Linked test results to {args.model} version {args.version}")
        else:
            print("Failed to link test results")
            return 1
    
    elif args.command == 'calculate-scores':
        integration = ModelRegistryIntegration(args.db_path)
        scores = integration.calculate_suitability_scores(
            model_name=args.model,
            hardware_type=args.hardware,
            update_db=not args.no_update
        )
        if scores:
            # Print summary
            print(f"Calculated scores for {len(scores)} models")
            for model_name, hardware_data in scores.items():
                print(f"  {model_name}: {len(hardware_data)} hardware platforms")
                for hardware_type, score_data in hardware_data.items():
                    print(f"    {hardware_type}: compatibility={score_data['compatibility_score']:.4f}, suitability={score_data['suitability_score']:.4f}")
        else:
            print("No scores calculated")
            return 1
    
    elif args.command == 'recommend':
        recommender = HardwareRecommender(args.db_path)
        recommendations = recommender.recommend_hardware(
            model_name=args.model,
            task_type=args.task,
            batch_size=args.batch_size,
            latency_sensitive=args.latency_sensitive,
            memory_constrained=args.memory_constrained,
            top_k=args.top_k,
            update_db=not args.no_update
        )
        if recommendations:
            print(f"Hardware recommendations for {args.model} ({args.task}):")
            for i, rec in enumerate(recommendations):
                print(f"  {i+1}. {rec['hardware_type']} (score: {rec['weighted_score']:.4f}, recommended batch size: {rec['recommended_batch_size']})")
        else:
            print("No recommendations found")
            return 1
    
    elif args.command == 'update-task':
        recommender = HardwareRecommender(args.db_path)
        updated = recommender.update_task_recommendations(args.task)
        print(f"Updated {updated} task recommendations for {args.task}")
    
    elif args.command == 'add-version':
        version_control = VersionControlSystem(args.db_path)
        
        # Parse metadata
        metadata = None
        if args.metadata:
            try:
                metadata = json.loads(args.metadata)
            except json.JSONDecodeError:
                print("Invalid JSON metadata")
                return 1
        
        version_id = version_control.add_model_version(
            model_name=args.model,
            version_tag=args.version,
            version_hash=args.hash,
            metadata=metadata
        )
        if version_id:
            print(f"Added version {args.version} for {args.model} (ID: {version_id})")
        else:
            print("Failed to add version")
            return 1
    
    elif args.command == 'version-history':
        version_control = VersionControlSystem(args.db_path)
        versions = version_control.get_version_history(args.model)
        if versions:
            print(f"Version history for {args.model}:")
            for i, version in enumerate(versions):
                print(f"  {i+1}. {version['version_tag']} (created: {version['created_at']})")
                if version['version_hash']:
                    print(f"     Hash: {version['version_hash']}")
                if version['metadata']:
                    print(f"     Metadata: {json.dumps(version['metadata'])}")
        else:
            print(f"No versions found for {args.model}")
            return 1
    
    elif args.command == 'create-snapshot':
        version_control = VersionControlSystem(args.db_path)
        success = version_control.create_compatibility_snapshot(args.model, args.version)
        if success:
            print(f"Created compatibility snapshot for {args.model} version {args.version}")
        else:
            print("Failed to create snapshot")
            return 1
    
    elif args.command == 'compare-versions':
        version_control = VersionControlSystem(args.db_path)
        changes = version_control.compare_compatibility_versions(args.model, args.version1, args.version2)
        if changes:
            print(f"Compatibility changes for {args.model} between {args.version1} and {args.version2}:")
            for i, change in enumerate(changes):
                print(f"  {i+1}. {change['hardware_type']}:")
                print(f"     Compatibility: {change['compatibility_v1']:.4f} -> {change['compatibility_v2']:.4f} ({change['compatibility_change']:+.2f}%)")
                print(f"     Suitability: {change['suitability_v1']:.4f} -> {change['suitability_v2']:.4f} ({change['suitability_change']:+.2f}%)")
                print(f"     Batch Size: {change['batch_size_v1']} -> {change['batch_size_v2']} ({change['batch_size_change']:+d})")
                print(f"     Memory: {change['memory_v1']:.2f} -> {change['memory_v2']:.2f} ({change['memory_change']:+.2f}%)")
            
            # Save to file if requested
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(changes, f, indent=2)
                print(f"Saved comparison to {args.output}")
        else:
            print("No comparison data available")
            return 1
    
    else:
        parser.print_help()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())