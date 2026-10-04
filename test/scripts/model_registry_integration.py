#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Registry::: Integration for IPFS Accelerate Python Framework

This module implements the enhanced model registry::: integration mentioned in NEXT_STEPS.md.
It provides components for integrating test results with the model registry:::, creating a 
suitability scoring system for hardware-model pairs, implementing an automatic recommender
based on task requirements, and adding versioning for model-hardware compatibility.

Date: March 2025
"""

import os
import sys
import json
import time
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path

# Add parent directory to path
sys.path.append())))))))))))))))str())))))))))))))))Path())))))))))))))))__file__).resolve())))))))))))))))).parent.parent))

# Local imports
try::::
    from data.duckdb.core.benchmark_db_api import get_db_connection
    from data.duckdb.core.benchmark_db_query import query_database
    from hardware_selector import select_hardware
    from model_family_classifier import classify_model
except ImportError:
    print())))))))))))))))"Warning: Some local modules could not be imported.")
    

class ModelRegistry:::Integration:
    """
    Integrates test results with the model registry::: and provides 
    hardware-model suitability scoring.
    """
    
    def __init__())))))))))))))))self, db_path: Optional[str] = None):,,,
    """Initialize with optional database path."""
    self.db_path = db_path or os.environ.get())))))))))))))))'BENCHMARK_DB_PATH', './benchmark_db.duckdb')
    
    def create_schema_extensions())))))))))))))))self) -> None:
        """Create schema extensions for model registry::: integration."""
        conn = get_db_connection())))))))))))))))self.db_path)
        
        # Create model registry::: version table if it doesn't exist
        conn.execute())))))))))))))))"""
        CREATE TABLE IF NOT EXISTS model_registry:::_versions ())))))))))))))))
        id INTEGER PRIMARY KEY,
        model_id INTEGER,
        version_tag VARCHAR,
        version_hash VARCHAR,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        metadata JSON,
        FOREIGN KEY ())))))))))))))))model_id) REFERENCES models())))))))))))))))id)
        )
        """)
        
        # Create hardware-model compatibility table if it doesn't exist
        conn.execute())))))))))))))))"""
        CREATE TABLE IF NOT EXISTS hardware_model_compatibility ())))))))))))))))
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
        FOREIGN KEY ())))))))))))))))model_id) REFERENCES models())))))))))))))))id),
        FOREIGN KEY ())))))))))))))))hardware_id) REFERENCES hardware_platforms())))))))))))))))id)
        )
        """)
        
        # Create task-specific recommendations table if it doesn't exist
        conn.execute())))))))))))))))"""
        CREATE TABLE IF NOT EXISTS task_recommendations ())))))))))))))))
        id INTEGER PRIMARY KEY,
        task_type VARCHAR,
        model_id INTEGER,
        hardware_id INTEGER,
        suitability_score FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY ())))))))))))))))model_id) REFERENCES models())))))))))))))))id),
        FOREIGN KEY ())))))))))))))))hardware_id) REFERENCES hardware_platforms())))))))))))))))id)
        )
        """)
        
        conn.close()))))))))))))))))
        print())))))))))))))))"Schema extensions created successfully!")
        
    def link_test_results())))))))))))))))self, :
        model_name: str,
        model_version: str,
        result_ids: List[int]) -> bool:,
        """
        Link test results to a specific model version in the registry:::.
        
        Args:
            model_name: Name of the model
            model_version: Version tag of the model
            result_ids: List of test result IDs to link
            
        Returns:
            Success status
            """
            conn = get_db_connection())))))))))))))))self.db_path)
        
        # Get model ID
            model_query = "SELECT id FROM models WHERE name = ?"
            model_result = conn.execute())))))))))))))))model_query, ())))))))))))))))model_name,)).fetchone()))))))))))))))))
        
        if not model_result:
            print())))))))))))))))f"Model '{}}}}model_name}' not found in the database.")
            conn.close()))))))))))))))))
            return False
            
            model_id = model_result[0],,
            ,
        # Get or create model version
            version_query = """
            SELECT id FROM model_registry:::_versions
            WHERE model_id = ? AND version_tag = ?
            """
            version_result = conn.execute())))))))))))))))version_query, ())))))))))))))))model_id, model_version)).fetchone()))))))))))))))))
        
        if version_result:
            version_id = version_result[0],,
        ,else:
            # Create new version entry:::
            insert_query = """
            INSERT INTO model_registry:::_versions ())))))))))))))))model_id, version_tag)
            VALUES ())))))))))))))))?, ?)
            """
            conn.execute())))))))))))))))insert_query, ())))))))))))))))model_id, model_version))
            conn.commit()))))))))))))))))
            
            # Get the new version ID
            version_id = conn.execute())))))))))))))))version_query, ())))))))))))))))model_id, model_version)).fetchone()))))))))))))))))[0],,
            ,
        # Link test results to model version
        for result_id in result_ids:
            # Check if result exists
            result_query = "SELECT id FROM performance_results WHERE id = ?"
            result_exists = conn.execute())))))))))))))))result_query, ())))))))))))))))result_id,)).fetchone()))))))))))))))))
            :
            if not result_exists:
                print())))))))))))))))f"Test result ID {}}}}result_id} not found in the database.")
                continue
                
            # Update result to link to model version
                update_query = """
                UPDATE performance_results
                SET model_version_id = ?
                WHERE id = ?
                """
                conn.execute())))))))))))))))update_query, ())))))))))))))))version_id, result_id))
        
                conn.commit()))))))))))))))))
                conn.close()))))))))))))))))
        
                print())))))))))))))))f"Linked {}}}}len())))))))))))))))result_ids)} test result())))))))))))))))s) to model '{}}}}model_name}' version '{}}}}model_version}'.")
            return True
    
            def calculate_suitability_scores())))))))))))))))self,
            model_id: Optional[int] = None,
            hardware_id: Optional[int] = None,
            update_db: bool = True) -> List[Dict[str, Any]]:,,,,
            """
            Calculate suitability scores for hardware-model pairs.
        
        Args:
            model_id: Optional filter by model ID
            hardware_id: Optional filter by hardware ID
            update_db: Whether to update scores in the database
            
        Returns:
            List of hardware-model pairs with suitability scores
            """
            conn = get_db_connection())))))))))))))))self.db_path)
        
        # Build filters for query
            filters = [],,,
            params = {}}}}}
        
        if model_id is not None:
            filters.append())))))))))))))))"pr.model_id = :model_id")
            params['model_id'] = model_id
            ,
        if hardware_id is not None:
            filters.append())))))))))))))))"pr.hardware_id = :hardware_id")
            params['hardware_id'] = hardware_id
            ,
            where_clause = f"WHERE {}}}}' AND '.join())))))))))))))))filters)}" if filters else ""
        
        # Query performance results
            query = f"""
            SELECT
            pr.model_id,
            m.name as model_name,
            pr.hardware_id,
            hp.type as hardware_type,
            AVG())))))))))))))))pr.throughput) as avg_throughput,
            AVG())))))))))))))))pr.latency) as avg_latency,
            AVG())))))))))))))))pr.memory_usage) as avg_memory_usage,
            COUNT())))))))))))))))*) as test_count
            FROM
            performance_results pr
            JOIN
            models m ON pr.model_id = m.id
            JOIN
            hardware_platforms hp ON pr.hardware_id = hp.id
            {}}}}where_clause}
            GROUP BY
            pr.model_id, m.name, pr.hardware_id, hp.type
            """
        
            result = conn.execute())))))))))))))))query, params).fetchall()))))))))))))))))
        :
        if not result:
            print())))))))))))))))"No performance results found for the specified filters.")
            conn.close()))))))))))))))))
            return [],,,
            
        # Calculate suitability scores
            suitability_scores = [],,,
        
        for row in result:
            model_id, model_name, hardware_id, hardware_type, avg_throughput, avg_latency, avg_memory_usage, test_count = row
            
            # Get maximum throughput for this model across all hardware
            max_query = """
            SELECT MAX())))))))))))))))avg_throughput) FROM ())))))))))))))))
            SELECT
            AVG())))))))))))))))throughput) as avg_throughput
            FROM
            performance_results
            WHERE
            model_id = ?
            GROUP BY
            hardware_id
            )
            """
            max_throughput = conn.execute())))))))))))))))max_query, ())))))))))))))))model_id,)).fetchone()))))))))))))))))[0],, or 1.0
            
            # Get minimum latency for this model across all hardware
            min_query = """
            SELECT MIN())))))))))))))))avg_latency) FROM ())))))))))))))))
            SELECT
            AVG())))))))))))))))latency) as avg_latency
            FROM
            performance_results
            WHERE
            model_id = ?
            GROUP BY
            hardware_id
            )
            """
            min_latency = conn.execute())))))))))))))))min_query, ())))))))))))))))model_id,)).fetchone()))))))))))))))))[0],, or 1.0
            
            # Calculate throughput score ())))))))))))))))0-1)
            throughput_score = avg_throughput / max_throughput if max_throughput > 0 else 0
            
            # Calculate latency score ())))))))))))))))0-1)
            latency_score = min_latency / avg_latency if avg_latency > 0 else 0
            
            # Calculate memory efficiency ())))))))))))))))lower is better)
            memory_query = """
            SELECT AVG())))))))))))))))memory_usage) FROM performance_results WHERE model_id = ?
            """
            avg_model_memory = conn.execute())))))))))))))))memory_query, ())))))))))))))))model_id,)).fetchone()))))))))))))))))[0],, or 1.0
            memory_score = avg_model_memory / avg_memory_usage if avg_memory_usage > 0 else 0
            
            # Calculate compatibility score ())))))))))))))))0-1)
            # - Higher throughput is better
            # - Lower latency is better
            # - Lower memory usage is better
            compatibility_score = ())))))))))))))))throughput_score * 0.5) + ())))))))))))))))latency_score * 0.3) + ())))))))))))))))memory_score * 0.2)
            
            # Calculate suitability score based on compatibility and test count
            # More tests means more confidence in the score
            confidence_factor = min())))))))))))))))1.0, test_count / 10)  # Cap at 1.0
            suitability_score = compatibility_score * confidence_factor
            
            # Determine recommended batch size
            batch_query = """
            SELECT
            batch_size,
            AVG())))))))))))))))throughput) as avg_throughput
            FROM
            performance_results
            WHERE
            model_id = ? AND hardware_id = ?
            GROUP BY
            batch_size
            ORDER BY
            avg_throughput DESC
            LIMIT 1
            """
            batch_result = conn.execute())))))))))))))))batch_query, ())))))))))))))))model_id, hardware_id)).fetchone()))))))))))))))))
            recommended_batch_size = batch_result[0],, if batch_result else 1
            
            # Determine recommended precision
            precision_query = """
            SELECT
            precision,
            AVG())))))))))))))))throughput) as avg_throughput
            FROM
            performance_results
            WHERE
            model_id = ? AND hardware_id = ?
            AND precision IS NOT NULL
            GROUP BY
            precision
            ORDER BY
            avg_throughput DESC
            LIMIT 1
            """
            precision_result = conn.execute())))))))))))))))precision_query, ())))))))))))))))model_id, hardware_id)).fetchone()))))))))))))))))
            recommended_precision = precision_result[0],, if precision_result else "fp32"
            
            # Prepare result
            pair = {}}}}:
                'model_id': model_id,
                'model_name': model_name,
                'hardware_id': hardware_id,
                'hardware_type': hardware_type,
                'compatibility_score': compatibility_score,
                'suitability_score': suitability_score,
                'recommended_batch_size': recommended_batch_size,
                'recommended_precision': recommended_precision,
                'avg_memory_usage': avg_memory_usage,
                'avg_throughput': avg_throughput,
                'avg_latency': avg_latency,
                'test_count': test_count
                }
            
                suitability_scores.append())))))))))))))))pair)
            
            # Update database if requested:
            if update_db:
                # Check if entry::: exists
                check_query = """
                SELECT id FROM hardware_model_compatibility 
                WHERE model_id = ? AND hardware_id = ?
                """
                existing = conn.execute())))))))))))))))check_query, ())))))))))))))))model_id, hardware_id)).fetchone()))))))))))))))))
                :
                if existing:
                    # Update existing entry:::
                    update_query = """
                    UPDATE hardware_model_compatibility
                    SET compatibility_score = ?,
                    suitability_score = ?,
                    recommended_batch_size = ?,
                    recommended_precision = ?,
                    memory_requirement = ?,
                    last_updated = CURRENT_TIMESTAMP
                    WHERE model_id = ? AND hardware_id = ?
                    """
                    conn.execute())))))))))))))))update_query, ())))))))))))))))
                    compatibility_score,
                    suitability_score,
                    recommended_batch_size,
                    recommended_precision,
                    avg_memory_usage,
                    model_id,
                    hardware_id
                    ))
                else:
                    # Create new entry:::
                    insert_query = """
                    INSERT INTO hardware_model_compatibility
                    ())))))))))))))))model_id, hardware_id, compatibility_score, suitability_score,
                    recommended_batch_size, recommended_precision, memory_requirement)
                    VALUES ())))))))))))))))?, ?, ?, ?, ?, ?, ?)
                    """
                    conn.execute())))))))))))))))insert_query, ())))))))))))))))
                    model_id,
                    hardware_id,
                    compatibility_score,
                    suitability_score,
                    recommended_batch_size,
                    recommended_precision,
                    avg_memory_usage
                    ))
        
        if update_db:
            conn.commit()))))))))))))))))
            
            conn.close()))))))))))))))))
        
                    return suitability_scores


class HardwareRecommender:
    """
    Automatic recommender for hardware based on model and task requirements.
    """
    
    def __init__())))))))))))))))self, db_path: Optional[str] = None):,,,
    """Initialize with optional database path."""
    self.db_path = db_path or os.environ.get())))))))))))))))'BENCHMARK_DB_PATH', './benchmark_db.duckdb')
        
    def recommend_hardware())))))))))))))))self, 
    model_name: str,
    task_type: Optional[str] = None,
    batch_size: Optional[int] = None,
    latency_sensitive: bool = False,
    memory_constrained: bool = False,
    top_k: int = 3) -> List[Dict[str, Any]]:,,,,
    """
    Recommend hardware for a given model and task requirements.
        
        Args:
            model_name: Name of the model
            task_type: Type of task ())))))))))))))))e.g., 'inference', 'training')
            batch_size: Optional batch size requirement
            latency_sensitive: Whether the task is latency-sensitive
            memory_constrained: Whether there are memory constraints
            top_k: Number of recommendations to return
            
        Returns:
            List of hardware recommendations with scores
            """
            conn = get_db_connection())))))))))))))))self.db_path)
        
        # Get model ID
            model_query = "SELECT id FROM models WHERE name = ?"
            model_result = conn.execute())))))))))))))))model_query, ())))))))))))))))model_name,)).fetchone()))))))))))))))))
        
        if not model_result:
            print())))))))))))))))f"Model '{}}}}model_name}' not found in the database.")
            conn.close()))))))))))))))))
            return [],,,
            
            model_id = model_result[0],,
            ,
        # Get hardware compatibility scores
            query = """
            SELECT
            hmc.hardware_id,
            hp.type as hardware_type,
            hp.name as hardware_name,
            hp.vendor as hardware_vendor,
            hmc.compatibility_score,
            hmc.suitability_score,
            hmc.recommended_batch_size,
            hmc.recommended_precision,
            hmc.memory_requirement
            FROM
            hardware_model_compatibility hmc
            JOIN
            hardware_platforms hp ON hmc.hardware_id = hp.id
            WHERE
            hmc.model_id = ?
            """
        
            params = [model_id]
            ,
        # Add task-specific filtering if provided::
        if task_type:
            query += """
            AND hmc.hardware_id IN ())))))))))))))))
            SELECT hardware_id FROM task_recommendations
            WHERE model_id = ? AND task_type = ?
            )
            """
            params.extend())))))))))))))))[model_id, task_type])
            ,
        # Add batch size filtering if provided::
        if batch_size:
            query += " AND hmc.recommended_batch_size >= ?"
            params.append())))))))))))))))batch_size)
            
            result = conn.execute())))))))))))))))query, params).fetchall()))))))))))))))))
            conn.close()))))))))))))))))
        
        if not result:
            print())))))))))))))))f"No hardware compatibility data found for model '{}}}}model_name}'.")
            return [],,,
            
        # Prepare recommendations
            recommendations = [],,,
        
        for row in result:
            hardware_id, hardware_type, hardware_name, hardware_vendor, compatibility_score, suitability_score, recommended_batch_size, recommended_precision, memory_requirement = row
            
            # Calculate weighted score based on requirements
            if latency_sensitive and memory_constrained:
                # Balance between latency and memory
                weighted_score = compatibility_score * 0.6 + suitability_score * 0.4
            elif latency_sensitive:
                # Prioritize compatibility ())))))))))))))))which includes latency score)
                weighted_score = compatibility_score * 0.8 + suitability_score * 0.2
            elif memory_constrained:
                # Consider memory usage more heavily
                # Lower memory_requirement is better
                memory_factor = 1.0 - min())))))))))))))))1.0, memory_requirement / 16.0)  # Assume 16GB is high
                weighted_score = compatibility_score * 0.4 + suitability_score * 0.3 + memory_factor * 0.3
            else:
                # Default weighting
                weighted_score = compatibility_score * 0.5 + suitability_score * 0.5
                
            # Prepare recommendation
                recommendation = {}}}}
                'hardware_id': hardware_id,
                'hardware_type': hardware_type,
                'hardware_name': hardware_name,
                'hardware_vendor': hardware_vendor,
                'compatibility_score': compatibility_score,
                'suitability_score': suitability_score,
                'weighted_score': weighted_score,
                'recommended_batch_size': recommended_batch_size,
                'recommended_precision': recommended_precision,
                'memory_requirement': memory_requirement
                }
            
                recommendations.append())))))))))))))))recommendation)
            
        # Sort by weighted score and return top_k
                recommendations.sort())))))))))))))))key=lambda x: x['weighted_score'], reverse=True),
                return recommendations[:top_k]
                ,
    def update_task_recommendations())))))))))))))))self, task_type: str) -> None:
        """
        Update task-specific recommendations for all models.
        
        Args:
            task_type: Type of task ())))))))))))))))e.g., 'inference', 'training')
            """
            conn = get_db_connection())))))))))))))))self.db_path)
        
        # Get all models
            models_query = "SELECT id, name FROM models"
            models = conn.execute())))))))))))))))models_query).fetchall()))))))))))))))))
        
        for model_id, model_name in models:
            print())))))))))))))))f"Updating recommendations for model '{}}}}model_name}', task '{}}}}task_type}'...")
            
            # Calculate weighted scores based on task type
            if task_type == 'inference':
                # For inference, prioritize latency and throughput
                # Adjust the query based on the schema
                query = """
                SELECT
                hmc.hardware_id,
                hp.type as hardware_type,
                hmc.compatibility_score * 0.7 + hmc.suitability_score * 0.3 as task_score
                FROM
                hardware_model_compatibility hmc
                JOIN
                hardware_platforms hp ON hmc.hardware_id = hp.id
                WHERE
                hmc.model_id = ?
                """
            elif task_type == 'training':
                # For training, prioritize throughput and memory efficiency
                query = """
                SELECT
                hmc.hardware_id,
                hp.type as hardware_type,
                hmc.compatibility_score * 0.4 + hmc.suitability_score * 0.6 as task_score
                FROM
                hardware_model_compatibility hmc
                JOIN
                hardware_platforms hp ON hmc.hardware_id = hp.id
                WHERE
                hmc.model_id = ?
                """
            else:
                # Default scoring
                query = """
                SELECT
                hmc.hardware_id,
                hp.type as hardware_type,
                hmc.compatibility_score * 0.5 + hmc.suitability_score * 0.5 as task_score
                FROM
                hardware_model_compatibility hmc
                JOIN
                hardware_platforms hp ON hmc.hardware_id = hp.id
                WHERE
                hmc.model_id = ?
                """
                
                hardware_scores = conn.execute())))))))))))))))query, ())))))))))))))))model_id,)).fetchall()))))))))))))))))
            
            for hardware_id, hardware_type, task_score in hardware_scores:
                # Check if recommendation exists
                check_query = """
                SELECT id FROM task_recommendations
                WHERE model_id = ? AND hardware_id = ? AND task_type = ?
                """
                existing = conn.execute())))))))))))))))check_query, ())))))))))))))))model_id, hardware_id, task_type)).fetchone()))))))))))))))))
                :
                if existing:
                    # Update existing recommendation
                    update_query = """
                    UPDATE task_recommendations
                    SET suitability_score = ?,
                    last_updated = CURRENT_TIMESTAMP
                    WHERE model_id = ? AND hardware_id = ? AND task_type = ?
                    """
                    conn.execute())))))))))))))))update_query, ())))))))))))))))task_score, model_id, hardware_id, task_type))
                else:
                    # Create new recommendation
                    insert_query = """
                    INSERT INTO task_recommendations
                    ())))))))))))))))model_id, hardware_id, task_type, suitability_score)
                    VALUES ())))))))))))))))?, ?, ?, ?)
                    """
                    conn.execute())))))))))))))))insert_query, ())))))))))))))))model_id, hardware_id, task_type, task_score))
            
                    conn.commit()))))))))))))))))
                    conn.close()))))))))))))))))
        
                    print())))))))))))))))f"Task recommendations updated for '{}}}}task_type}'.")


class VersionControlSystem:
    """
    Version control system for model-hardware compatibility.
    """
    
    def __init__())))))))))))))))self, db_path: Optional[str] = None):,,,
    """Initialize with optional database path."""
    self.db_path = db_path or os.environ.get())))))))))))))))'BENCHMARK_DB_PATH', './benchmark_db.duckdb')
        
    def add_model_version())))))))))))))))self, 
    model_name: str,
    version_tag: str,
    version_hash: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None) -> Optional[int]:,
    """
    Add a new model version to the registry:::.
        
        Args:
            model_name: Name of the model
            version_tag: Version tag ())))))))))))))))e.g., 'v1.0.0')
            version_hash: Optional hash for the version ())))))))))))))))e.g., git commit)
            metadata: Optional metadata for the version
            
        Returns:
            ID of the created version, or None if failed
            """
            conn = get_db_connection())))))))))))))))self.db_path)
        
        # Get model ID
            model_query = "SELECT id FROM models WHERE name = ?"
            model_result = conn.execute())))))))))))))))model_query, ())))))))))))))))model_name,)).fetchone()))))))))))))))))
        :
        if not model_result:
            print())))))))))))))))f"Model '{}}}}model_name}' not found in the database.")
            conn.close()))))))))))))))))
            return None
            
            model_id = model_result[0],,
            ,
        # Check if version already exists
            version_query = """
            SELECT id FROM model_registry:::_versions
            WHERE model_id = ? AND version_tag = ?
            """
            existing = conn.execute())))))))))))))))version_query, ())))))))))))))))model_id, version_tag)).fetchone()))))))))))))))))
        :
        if existing:
            print())))))))))))))))f"Version '{}}}}version_tag}' already exists for model '{}}}}model_name}'.")
            conn.close()))))))))))))))))
            return existing[0],,
            ,
        # Prepare metadata JSON
            metadata_json = json.dumps())))))))))))))))metadata) if metadata else None
        
        # Create new version
            insert_query = """
            INSERT INTO model_registry:::_versions
            ())))))))))))))))model_id, version_tag, version_hash, metadata)
            VALUES ())))))))))))))))?, ?, ?, ?)
            """
            conn.execute())))))))))))))))insert_query, ())))))))))))))))model_id, version_tag, version_hash, metadata_json))
            conn.commit()))))))))))))))))
        
        # Get the new version ID
            version_id = conn.execute())))))))))))))))version_query, ())))))))))))))))model_id, version_tag)).fetchone()))))))))))))))))[0],,
            ,
            conn.close()))))))))))))))))
        
            print())))))))))))))))f"Added version '{}}}}version_tag}' for model '{}}}}model_name}'.")
            return version_id
    :
        def get_version_history())))))))))))))))self, model_name: str) -> List[Dict[str, Any]]:,,,,
        """
        Get version history for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of versions with metadata
            """
            conn = get_db_connection())))))))))))))))self.db_path)
        
        # Get model ID
            model_query = "SELECT id FROM models WHERE name = ?"
            model_result = conn.execute())))))))))))))))model_query, ())))))))))))))))model_name,)).fetchone()))))))))))))))))
        
        if not model_result:
            print())))))))))))))))f"Model '{}}}}model_name}' not found in the database.")
            conn.close()))))))))))))))))
            return [],,,
            
            model_id = model_result[0],,
            ,
        # Get versions
            query = """
            SELECT
            id,
            version_tag,
            version_hash,
            created_at,
            metadata
            FROM
            model_registry:::_versions
            WHERE
            model_id = ?
            ORDER BY
            created_at DESC
            """
        
            versions = [],,,
        for row in conn.execute())))))))))))))))query, ())))))))))))))))model_id,)).fetchall())))))))))))))))):
            id, version_tag, version_hash, created_at, metadata_json = row
            
            metadata = json.loads())))))))))))))))metadata_json) if metadata_json else {}}}}}
            
            versions.append()))))))))))))))){}}}}:
                'id': id,
                'version_tag': version_tag,
                'version_hash': version_hash,
                'created_at': created_at,
                'metadata': metadata
                })
            
                conn.close()))))))))))))))))
        
            return versions
    
            def create_compatibility_snapshot())))))))))))))))self,
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
            conn = get_db_connection())))))))))))))))self.db_path)
        
        # Get model ID
            model_query = "SELECT id FROM models WHERE name = ?"
            model_result = conn.execute())))))))))))))))model_query, ())))))))))))))))model_name,)).fetchone()))))))))))))))))
        
        if not model_result:
            print())))))))))))))))f"Model '{}}}}model_name}' not found in the database.")
            conn.close()))))))))))))))))
            return False
            
            model_id = model_result[0],,
            ,
        # Get version ID
            version_query = """
            SELECT id FROM model_registry:::_versions
            WHERE model_id = ? AND version_tag = ?
            """
            version_result = conn.execute())))))))))))))))version_query, ())))))))))))))))model_id, version_tag)).fetchone()))))))))))))))))
        
        if not version_result:
            print())))))))))))))))f"Version '{}}}}version_tag}' not found for model '{}}}}model_name}'.")
            conn.close()))))))))))))))))
            return False
            
            version_id = version_result[0],,
            ,
        # Get current compatibility data
            compat_query = """
            SELECT
            hardware_id,
            compatibility_score,
            suitability_score,
            recommended_batch_size,
            recommended_precision,
            memory_requirement
            FROM
            hardware_model_compatibility
            WHERE
            model_id = ?
            """
        
            compatibility_data = conn.execute())))))))))))))))compat_query, ())))))))))))))))model_id,)).fetchall()))))))))))))))))
        
        if not compatibility_data:
            print())))))))))))))))f"No compatibility data found for model '{}}}}model_name}'.")
            conn.close()))))))))))))))))
            return False
        
        # Create snapshot table if it doesn't exist
            conn.execute())))))))))))))))"""
            CREATE TABLE IF NOT EXISTS hardware_compatibility_snapshots ())))))))))))))))
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
            FOREIGN KEY ())))))))))))))))model_id) REFERENCES models())))))))))))))))id),
            FOREIGN KEY ())))))))))))))))version_id) REFERENCES model_registry:::_versions())))))))))))))))id),
            FOREIGN KEY ())))))))))))))))hardware_id) REFERENCES hardware_platforms())))))))))))))))id)
            )
            """)
        
        # Insert snapshot data:
        for row in compatibility_data:
            hardware_id, compatibility_score, suitability_score, recommended_batch_size, recommended_precision, memory_requirement = row
            
            # Check if snapshot already exists
            check_query = """
            SELECT id FROM hardware_compatibility_snapshots
            WHERE model_id = ? AND version_id = ? AND hardware_id = ?
            """
            existing = conn.execute())))))))))))))))check_query, ())))))))))))))))model_id, version_id, hardware_id)).fetchone()))))))))))))))))
            :
            if existing:
                # Update existing snapshot
                update_query = """
                UPDATE hardware_compatibility_snapshots
                SET compatibility_score = ?,
                suitability_score = ?,
                recommended_batch_size = ?,
                recommended_precision = ?,
                memory_requirement = ?,
                created_at = CURRENT_TIMESTAMP
                WHERE model_id = ? AND version_id = ? AND hardware_id = ?
                """
                conn.execute())))))))))))))))update_query, ())))))))))))))))
                compatibility_score,
                suitability_score,
                recommended_batch_size,
                recommended_precision,
                memory_requirement,
                model_id,
                version_id,
                hardware_id
                ))
            else:
                # Create new snapshot
                insert_query = """
                INSERT INTO hardware_compatibility_snapshots
                ())))))))))))))))model_id, version_id, hardware_id, compatibility_score, suitability_score,
                recommended_batch_size, recommended_precision, memory_requirement)
                VALUES ())))))))))))))))?, ?, ?, ?, ?, ?, ?, ?)
                """
                conn.execute())))))))))))))))insert_query, ())))))))))))))))
                model_id,
                version_id,
                hardware_id,
                compatibility_score,
                suitability_score,
                recommended_batch_size,
                recommended_precision,
                memory_requirement
                ))
        
                conn.commit()))))))))))))))))
                conn.close()))))))))))))))))
        
                print())))))))))))))))f"Created compatibility snapshot for model '{}}}}model_name}' version '{}}}}version_tag}'.")
                return True
    
                def compare_compatibility_versions())))))))))))))))self,
                model_name: str,
                version_tag1: str,
                version_tag2: str) -> List[Dict[str, Any]]:,,,,
                """
                Compare hardware compatibility between two model versions.
        
        Args:
            model_name: Name of the model
            version_tag1: First version tag
            version_tag2: Second version tag
            
        Returns:
            List of compatibility changes
            """
            conn = get_db_connection())))))))))))))))self.db_path)
        
        # Get model ID
            model_query = "SELECT id FROM models WHERE name = ?"
            model_result = conn.execute())))))))))))))))model_query, ())))))))))))))))model_name,)).fetchone()))))))))))))))))
        
        if not model_result:
            print())))))))))))))))f"Model '{}}}}model_name}' not found in the database.")
            conn.close()))))))))))))))))
            return [],,,
            
            model_id = model_result[0],,
            ,
        # Get version IDs
            version_query = """
            SELECT id FROM model_registry:::_versions
            WHERE model_id = ? AND version_tag = ?
            """
            version1_result = conn.execute())))))))))))))))version_query, ())))))))))))))))model_id, version_tag1)).fetchone()))))))))))))))))
            version2_result = conn.execute())))))))))))))))version_query, ())))))))))))))))model_id, version_tag2)).fetchone()))))))))))))))))
        
        if not version1_result or not version2_result:
            missing = [],,,
            if not version1_result:
                missing.append())))))))))))))))version_tag1)
            if not version2_result:
                missing.append())))))))))))))))version_tag2)
                
                print())))))))))))))))f"Version())))))))))))))))s) not found for model '{}}}}model_name}': {}}}}', '.join())))))))))))))))missing)}")
                conn.close()))))))))))))))))
                return [],,,
            
                version1_id = version1_result[0],,
                ,version2_id = version2_result[0],,
                ,
        # Get snapshot data for both versions
                query = """
                SELECT
                v1.hardware_id,
                hp.type as hardware_type,
                v1.compatibility_score as score1,
                v2.compatibility_score as score2,
                v1.suitability_score as suitability1,
                v2.suitability_score as suitability2,
                v1.recommended_batch_size as batch1,
                v2.recommended_batch_size as batch2,
                v1.memory_requirement as memory1,
                v2.memory_requirement as memory2
                FROM
                hardware_compatibility_snapshots v1
                JOIN
                hardware_platforms hp ON v1.hardware_id = hp.id
                LEFT JOIN
                hardware_compatibility_snapshots v2 ON
                v1.model_id = v2.model_id AND
                v1.hardware_id = v2.hardware_id AND
                v2.version_id = ?
                WHERE
                v1.model_id = ? AND
                v1.version_id = ?
                """
        
                result = conn.execute())))))))))))))))query, ())))))))))))))))version2_id, model_id, version1_id)).fetchall()))))))))))))))))
        
                conn.close()))))))))))))))))
        
        if not result:
            print())))))))))))))))f"No compatibility snapshots found for model '{}}}}model_name}'.")
                return [],,,
            
        # Calculate changes
                changes = [],,,
        
        for row in result:
            hardware_id, hardware_type, score1, score2, suitability1, suitability2, batch1, batch2, memory1, memory2 = row
            
            # Handle NULL values
            score2 = score2 or 0
            suitability2 = suitability2 or 0
            batch2 = batch2 or 0
            memory2 = memory2 or 0
            
            # Calculate percent changes
            compat_change = ())))))))))))))))())))))))))))))))score2 - score1) / score1 * 100) if score1 > 0 else 0
            suitability_change = ())))))))))))))))())))))))))))))))suitability2 - suitability1) / suitability1 * 100) if suitability1 > 0 else 0
            batch_change = batch2 - batch1
            memory_change = ())))))))))))))))())))))))))))))))memory2 - memory1) / memory1 * 100) if memory1 > 0 else 0
            
            # Determine if this is a significant change
            is_significant = ())))))))))))))))
            abs())))))))))))))))compat_change) > 5 or
            abs())))))))))))))))suitability_change) > 10 or
            batch_change != 0 or
            abs())))))))))))))))memory_change) > 10
            )
            :
            if is_significant:
                changes.append()))))))))))))))){}}}}
                'hardware_id': hardware_id,
                'hardware_type': hardware_type,
                'compatibility_change': compat_change,
                'suitability_change': suitability_change,
                'batch_size_change': batch_change,
                'memory_change': memory_change,
                'version1': version_tag1,
                'version2': version_tag2
                })
                
        # Sort by absolute compatibility change
                changes.sort())))))))))))))))key=lambda x: abs())))))))))))))))x['compatibility_change']), reverse=True)
                ,
                return changes


def main())))))))))))))))):
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser())))))))))))))))description='Model Registry::: Integration')
    subparsers = parser.add_subparsers())))))))))))))))dest='command', help='Command to execute')
    
    # Schema creation command
    schema_parser = subparsers.add_parser())))))))))))))))'create-schema', help='Create schema extensions')
    schema_parser.add_argument())))))))))))))))'--db-path', help='Database path')
    
    # Link test results command
    link_parser = subparsers.add_parser())))))))))))))))'link-tests', help='Link test results to model version')
    link_parser.add_argument())))))))))))))))'--db-path', help='Database path')
    link_parser.add_argument())))))))))))))))'--model', required=True, help='Model name')
    link_parser.add_argument())))))))))))))))'--version', required=True, help='Model version')
    link_parser.add_argument())))))))))))))))'--result-ids', required=True, help='Comma-separated list of result IDs')
    
    # Calculate suitability scores command
    scores_parser = subparsers.add_parser())))))))))))))))'calculate-scores', help='Calculate suitability scores')
    scores_parser.add_argument())))))))))))))))'--db-path', help='Database path')
    scores_parser.add_argument())))))))))))))))'--model', help='Model name filter')
    scores_parser.add_argument())))))))))))))))'--hardware', help='Hardware type filter')
    scores_parser.add_argument())))))))))))))))'--no-update', action='store_true', help="Don't update database")
    
    # Recommend hardware command
    recommend_parser = subparsers.add_parser())))))))))))))))'recommend', help='Recommend hardware for model')
    recommend_parser.add_argument())))))))))))))))'--db-path', help='Database path')
    recommend_parser.add_argument())))))))))))))))'--model', required=True, help='Model name')
    recommend_parser.add_argument())))))))))))))))'--task', help='Task type ())))))))))))))))inference, training)')
    recommend_parser.add_argument())))))))))))))))'--batch-size', type=int, help='Batch size requirement')
    recommend_parser.add_argument())))))))))))))))'--latency-sensitive', action='store_true', help='Task is latency-sensitive')
    recommend_parser.add_argument())))))))))))))))'--memory-constrained', action='store_true', help='Memory constraints exist')
    recommend_parser.add_argument())))))))))))))))'--top', type=int, default=3, help='Number of recommendations')
    
    # Update task recommendations command
    task_parser = subparsers.add_parser())))))))))))))))'update-task', help='Update task recommendations')
    task_parser.add_argument())))))))))))))))'--db-path', help='Database path')
    task_parser.add_argument())))))))))))))))'--task', required=True, help='Task type ())))))))))))))))inference, training)')
    
    # Add model version command
    version_parser = subparsers.add_parser())))))))))))))))'add-version', help='Add model version')
    version_parser.add_argument())))))))))))))))'--db-path', help='Database path')
    version_parser.add_argument())))))))))))))))'--model', required=True, help='Model name')
    version_parser.add_argument())))))))))))))))'--version', required=True, help='Version tag')
    version_parser.add_argument())))))))))))))))'--hash', help='Version hash')
    version_parser.add_argument())))))))))))))))'--metadata', help='JSON metadata')
    
    # Get version history command
    history_parser = subparsers.add_parser())))))))))))))))'version-history', help='Get version history')
    history_parser.add_argument())))))))))))))))'--db-path', help='Database path')
    history_parser.add_argument())))))))))))))))'--model', required=True, help='Model name')
    
    # Create compatibility snapshot command
    snapshot_parser = subparsers.add_parser())))))))))))))))'create-snapshot', help='Create compatibility snapshot')
    snapshot_parser.add_argument())))))))))))))))'--db-path', help='Database path')
    snapshot_parser.add_argument())))))))))))))))'--model', required=True, help='Model name')
    snapshot_parser.add_argument())))))))))))))))'--version', required=True, help='Version tag')
    
    # Compare compatibility versions command
    compare_parser = subparsers.add_parser())))))))))))))))'compare-versions', help='Compare compatibility versions')
    compare_parser.add_argument())))))))))))))))'--db-path', help='Database path')
    compare_parser.add_argument())))))))))))))))'--model', required=True, help='Model name')
    compare_parser.add_argument())))))))))))))))'--version1', required=True, help='First version tag')
    compare_parser.add_argument())))))))))))))))'--version2', required=True, help='Second version tag')
    
    args = parser.parse_args()))))))))))))))))
    
    if args.command == 'create-schema':
        integration = ModelRegistry:::Integration())))))))))))))))args.db_path)
        integration.create_schema_extensions()))))))))))))))))
        
    elif args.command == 'link-tests':
        integration = ModelRegistry:::Integration())))))))))))))))args.db_path)
        result_ids = [int())))))))))))))))id.strip()))))))))))))))))) for id in args.result_ids.split())))))))))))))))',')]:,
        integration.link_test_results())))))))))))))))args.model, args.version, result_ids)
        
    elif args.command == 'calculate-scores':
        integration = ModelRegistry:::Integration())))))))))))))))args.db_path)
        
        # Get model_id if model name provided
        model_id = None:
        if args.model:
            conn = get_db_connection())))))))))))))))args.db_path)
            query = "SELECT id FROM models WHERE name = ?"
            result = conn.execute())))))))))))))))query, ())))))))))))))))args.model,)).fetchone()))))))))))))))))
            conn.close()))))))))))))))))
            
            if result:
                model_id = result[0],,
                ,
        # Get hardware_id if hardware type provided
        hardware_id = None:
        if args.hardware:
            conn = get_db_connection())))))))))))))))args.db_path)
            query = "SELECT id FROM hardware_platforms WHERE type = ?"
            result = conn.execute())))))))))))))))query, ())))))))))))))))args.hardware,)).fetchone()))))))))))))))))
            conn.close()))))))))))))))))
            
            if result:
                hardware_id = result[0],,
                ,
                scores = integration.calculate_suitability_scores())))))))))))))))
                model_id=model_id,
                hardware_id=hardware_id,
                update_db=not args.no_update
                )
        
        # Print scores
                print())))))))))))))))f"Calculated {}}}}len())))))))))))))))scores)} suitability scores:")
        for score in scores:
            print())))))))))))))))f"- {}}}}score['model_name']} on {}}}}score['hardware_type']}: {}}}}score['suitability_score']:.4f} ())))))))))))))))compat: {}}}}score['compatibility_score']:.4f})")
            ,
    elif args.command == 'recommend':
        recommender = HardwareRecommender())))))))))))))))args.db_path)
        recommendations = recommender.recommend_hardware())))))))))))))))
        model_name=args.model,
        task_type=args.task,
        batch_size=args.batch_size,
        latency_sensitive=args.latency_sensitive,
        memory_constrained=args.memory_constrained,
        top_k=args.top
        )
        
        # Print recommendations
        print())))))))))))))))f"Top {}}}}len())))))))))))))))recommendations)} hardware recommendations for model '{}}}}args.model}':")
        for i, rec in enumerate())))))))))))))))recommendations, 1):
            print())))))))))))))))f"{}}}}i}. {}}}}rec['hardware_type']} ()))))))))))))))){}}}}rec['hardware_name']}):"),
            print())))))))))))))))f"   Score: {}}}}rec['weighted_score']:.4f} ())))))))))))))))compat: {}}}}rec['compatibility_score']:.4f}, suit: {}}}}rec['suitability_score']:.4f})"),
            print())))))))))))))))f"   Recommended batch size: {}}}}rec['recommended_batch_size']}"),
            print())))))))))))))))f"   Recommended precision: {}}}}rec['recommended_precision']}"),
            print())))))))))))))))f"   Memory requirement: {}}}}rec['memory_requirement']:.2f} GB"),
            print()))))))))))))))))
            
    elif args.command == 'update-task':
        recommender = HardwareRecommender())))))))))))))))args.db_path)
        recommender.update_task_recommendations())))))))))))))))args.task)
        
    elif args.command == 'add-version':
        version_control = VersionControlSystem())))))))))))))))args.db_path)
        
        # Parse metadata if provided::
        metadata = None
        if args.metadata:
            try::::
                metadata = json.loads())))))))))))))))args.metadata)
            except json.JSONDecodeError:
                print())))))))))))))))"Error: Invalid JSON metadata.")
                return
                
                version_id = version_control.add_model_version())))))))))))))))
                model_name=args.model,
                version_tag=args.version,
                version_hash=args.hash,
                metadata=metadata
                )
        
        if version_id:
            print())))))))))))))))f"Added version '{}}}}args.version}' for model '{}}}}args.model}' with ID {}}}}version_id}.")
            
    elif args.command == 'version-history':
        version_control = VersionControlSystem())))))))))))))))args.db_path)
        versions = version_control.get_version_history())))))))))))))))args.model)
        
        if versions:
            print())))))))))))))))f"Version history for model '{}}}}args.model}':")
            for version in versions:
                print())))))))))))))))f"- {}}}}version['version_tag']} ())))))))))))))))created: {}}}}version['created_at']})"),
                if version['version_hash']:,
                print())))))))))))))))f"  Hash: {}}}}version['version_hash']}"),
                if version['metadata']:,
                print())))))))))))))))f"  Metadata: {}}}}json.dumps())))))))))))))))version['metadata'], indent=2)}")
                ,
    elif args.command == 'create-snapshot':
        version_control = VersionControlSystem())))))))))))))))args.db_path)
        success = version_control.create_compatibility_snapshot())))))))))))))))args.model, args.version)
        
        if success:
            print())))))))))))))))f"Created compatibility snapshot for model '{}}}}args.model}' version '{}}}}args.version}'.")
            
    elif args.command == 'compare-versions':
        version_control = VersionControlSystem())))))))))))))))args.db_path)
        changes = version_control.compare_compatibility_versions())))))))))))))))
        args.model,
        args.version1,
        args.version2
        )
        
        if changes:
            print())))))))))))))))f"Compatibility changes for model '{}}}}args.model}' between versions '{}}}}args.version1}' and '{}}}}args.version2}':")
            for change in changes:
                print())))))))))))))))f"- {}}}}change['hardware_type']}:"),
                print())))))))))))))))f"  Compatibility: {}}}}change['compatibility_change']:.2f}%"),
                print())))))))))))))))f"  Suitability: {}}}}change['suitability_change']:.2f}%"),
                print())))))))))))))))f"  Batch size: {}}}}'+' if change['batch_size_change'] > 0 else ''}{}}}}change['batch_size_change']}"):,
                print())))))))))))))))f"  Memory usage: {}}}}change['memory_change']:.2f}%"),
                print()))))))))))))))))
        else:
            print())))))))))))))))f"No significant compatibility changes found between versions '{}}}}args.version1}' and '{}}}}args.version2}'.")
            
    else:
        parser.print_help()))))))))))))))))


if __name__ == "__main__":
    main()))))))))))))))))