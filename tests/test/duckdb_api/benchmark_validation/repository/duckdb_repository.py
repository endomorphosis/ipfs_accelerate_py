#!/usr/bin/env python3
"""
DuckDB Validation Repository

This module implements a repository for storing and retrieving benchmark validation data
using DuckDB as the storage backend.
"""

import os
import sys
import logging
import json
import datetime
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple, Set

import duckdb
import pandas as pd

from duckdb_api.benchmark_validation.core.base import (
    BenchmarkResult,
    ValidationResult,
    ValidationStatus,
    ValidationLevel,
    BenchmarkType,
    ValidationRepository
)

logger = logging.getLogger("benchmark_validation.duckdb_repository")

class DuckDBValidationRepository(ValidationRepository):
    """
    DuckDB-based repository for benchmark validation data.
    
    This class provides methods for storing and retrieving benchmark validation data
    using DuckDB as the storage backend, enabling efficient data management and querying.
    """
    
    def __init__(
        self, 
        db_path: str = "benchmark_db.duckdb",
        create_if_missing: bool = True,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the DuckDBValidationRepository.
        
        Args:
            db_path: Path to the DuckDB database file
            create_if_missing: Whether to create the database if it doesn't exist
            config: Additional configuration for the repository
        """
        super().__init__(config or {})
        
        self.db_path = db_path
        
        # Create parent directories if needed
        if create_if_missing:
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        
        # Connect to DuckDB
        try:
            self.conn = duckdb.connect(db_path)
            logger.info(f"Connected to DuckDB database at {db_path}")
        except Exception as e:
            logger.error(f"Error connecting to DuckDB database at {db_path}: {e}")
            raise
    
    def initialize_tables(self) -> None:
        """Create tables for benchmark validation data if they don't exist."""
        logger.info("Initializing benchmark validation tables")
        
        try:
            # Create benchmark_results table
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_results (
                result_id VARCHAR PRIMARY KEY,
                benchmark_type VARCHAR NOT NULL,
                model_id INTEGER,
                hardware_id INTEGER,
                run_id INTEGER,
                timestamp TIMESTAMP,
                metrics JSON,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Create validation_results table
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS validation_results (
                id VARCHAR PRIMARY KEY,
                benchmark_result_id VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                validation_level VARCHAR NOT NULL,
                confidence_score FLOAT,
                validation_metrics JSON,
                issues JSON,
                recommendations JSON,
                validation_timestamp TIMESTAMP,
                validator_id VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (benchmark_result_id) REFERENCES benchmark_results(result_id)
            )
            """)
            
            # Create certifications table
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS certifications (
                certification_id VARCHAR PRIMARY KEY,
                benchmark_id VARCHAR NOT NULL,
                certification_level VARCHAR NOT NULL,
                certification_timestamp TIMESTAMP,
                certification_authority VARCHAR,
                certification_version VARCHAR,
                certification_requirements JSON,
                validation_results JSON,
                certification_data JSON,
                certification_hash VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (benchmark_id) REFERENCES benchmark_results(result_id)
            )
            """)
            
            logger.info("Benchmark validation tables initialized")
            
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    def save_validation_result(
        self,
        validation_result: ValidationResult
    ) -> str:
        """
        Save a validation result.
        
        Args:
            validation_result: The validation result to save
            
        Returns:
            Identifier for the saved validation result
        """
        logger.info(f"Saving validation result {validation_result.id}")
        
        try:
            # First make sure the benchmark result is saved
            benchmark_result = validation_result.benchmark_result
            self.save_benchmark_result(benchmark_result)
            
            # Convert validation metrics to JSON
            validation_metrics_json = json.dumps(validation_result.validation_metrics)
            issues_json = json.dumps(validation_result.issues)
            recommendations_json = json.dumps(validation_result.recommendations)
            
            # Check if validation result already exists
            existing = self.conn.execute(
                "SELECT id FROM validation_results WHERE id = ?",
                [validation_result.id]
            ).fetchone()
            
            if existing:
                # Update existing validation result
                self.conn.execute(
                    """
                    UPDATE validation_results
                    SET benchmark_result_id = ?,
                        status = ?,
                        validation_level = ?,
                        confidence_score = ?,
                        validation_metrics = ?,
                        issues = ?,
                        recommendations = ?,
                        validation_timestamp = ?,
                        validator_id = ?
                    WHERE id = ?
                    """,
                    [
                        benchmark_result.result_id,
                        validation_result.status.name,
                        validation_result.validation_level.name,
                        validation_result.confidence_score,
                        validation_metrics_json,
                        issues_json,
                        recommendations_json,
                        validation_result.validation_timestamp,
                        validation_result.validator_id,
                        validation_result.id
                    ]
                )
            else:
                # Insert new validation result
                self.conn.execute(
                    """
                    INSERT INTO validation_results (
                        id, benchmark_result_id, status, validation_level,
                        confidence_score, validation_metrics, issues, recommendations,
                        validation_timestamp, validator_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        validation_result.id,
                        benchmark_result.result_id,
                        validation_result.status.name,
                        validation_result.validation_level.name,
                        validation_result.confidence_score,
                        validation_metrics_json,
                        issues_json,
                        recommendations_json,
                        validation_result.validation_timestamp,
                        validation_result.validator_id
                    ]
                )
            
            return validation_result.id
            
        except Exception as e:
            logger.error(f"Error saving validation result: {e}")
            raise
    
    def load_validation_result(
        self,
        result_id: str
    ) -> Optional[ValidationResult]:
        """
        Load a validation result by ID.
        
        Args:
            result_id: Identifier for the validation result
            
        Returns:
            ValidationResult if found, None otherwise
        """
        logger.info(f"Loading validation result {result_id}")
        
        try:
            result = self.conn.execute(
                """
                SELECT * FROM validation_results WHERE id = ?
                """,
                [result_id]
            ).fetchone()
            
            if not result:
                return None
            
            # Load the associated benchmark result
            benchmark_result = self.load_benchmark_result(result['benchmark_result_id'])
            if not benchmark_result:
                logger.error(f"Associated benchmark result {result['benchmark_result_id']} not found")
                return None
            
            # Parse JSON fields
            validation_metrics = json.loads(result['validation_metrics'])
            issues = json.loads(result['issues'])
            recommendations = json.loads(result['recommendations'])
            
            # Create ValidationResult object
            validation_result = ValidationResult(
                benchmark_result=benchmark_result,
                status=ValidationStatus[result['status']],
                validation_level=ValidationLevel[result['validation_level']],
                confidence_score=result['confidence_score'],
                validation_metrics=validation_metrics,
                issues=issues,
                recommendations=recommendations,
                validation_timestamp=result['validation_timestamp'],
                validator_id=result['validator_id']
            )
            
            # Set ID to match the stored value
            validation_result.id = result_id
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error loading validation result: {e}")
            return None
    
    def query_validation_results(
        self,
        filters: Dict[str, Any] = None,
        limit: int = 100
    ) -> List[ValidationResult]:
        """
        Query validation results using filters.
        
        Args:
            filters: Dictionary of filter criteria
            limit: Maximum number of results to return
            
        Returns:
            List of ValidationResult objects matching filters
        """
        logger.info(f"Querying validation results with filters: {filters}")
        
        try:
            # Build query conditions
            conditions = []
            params = []
            
            if filters:
                for key, value in filters.items():
                    # Handle nested fields (e.g., benchmark_result.model_id)
                    if key.startswith("benchmark_result."):
                        field = key.split(".", 1)[1]
                        conditions.append(f"vr.benchmark_result_id IN (SELECT result_id FROM benchmark_results WHERE {field} = ?)")
                        params.append(value)
                    elif key == "status":
                        conditions.append("vr.status = ?")
                        params.append(value.name if isinstance(value, ValidationStatus) else value)
                    elif key == "validation_level":
                        conditions.append("vr.validation_level = ?")
                        params.append(value.name if isinstance(value, ValidationLevel) else value)
                    elif key == "min_confidence_score":
                        conditions.append("vr.confidence_score >= ?")
                        params.append(value)
                    elif key == "max_confidence_score":
                        conditions.append("vr.confidence_score <= ?")
                        params.append(value)
                    elif key == "since":
                        conditions.append("vr.validation_timestamp >= ?")
                        params.append(value)
                    elif key == "until":
                        conditions.append("vr.validation_timestamp <= ?")
                        params.append(value)
                    elif key == "validator_id":
                        conditions.append("vr.validator_id = ?")
                        params.append(value)
                    else:
                        # General fields
                        conditions.append(f"vr.{key} = ?")
                        params.append(value)
            
            # Build query
            query = """
            SELECT vr.* FROM validation_results vr
            """
            
            if conditions:
                query += f" WHERE {' AND '.join(conditions)}"
            
            query += " ORDER BY vr.validation_timestamp DESC LIMIT ?"
            params.append(limit)
            
            # Execute query
            result = self.conn.execute(query, params)
            rows = result.fetchall()
            
            # Load validation results
            validation_results = []
            for row in rows:
                validation_result = self.load_validation_result(row['id'])
                if validation_result:
                    validation_results.append(validation_result)
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error querying validation results: {e}")
            return []
    
    def save_benchmark_result(
        self,
        benchmark_result: BenchmarkResult
    ) -> str:
        """
        Save a benchmark result.
        
        Args:
            benchmark_result: The benchmark result to save
            
        Returns:
            Identifier for the saved benchmark result
        """
        logger.info(f"Saving benchmark result {benchmark_result.result_id}")
        
        try:
            # Convert metrics and metadata to JSON
            metrics_json = json.dumps(benchmark_result.metrics)
            metadata_json = json.dumps(benchmark_result.metadata)
            
            # Check if benchmark result already exists
            existing = self.conn.execute(
                "SELECT result_id FROM benchmark_results WHERE result_id = ?",
                [benchmark_result.result_id]
            ).fetchone()
            
            if existing:
                # Update existing benchmark result
                self.conn.execute(
                    """
                    UPDATE benchmark_results
                    SET benchmark_type = ?,
                        model_id = ?,
                        hardware_id = ?,
                        run_id = ?,
                        timestamp = ?,
                        metrics = ?,
                        metadata = ?
                    WHERE result_id = ?
                    """,
                    [
                        benchmark_result.benchmark_type.name,
                        benchmark_result.model_id,
                        benchmark_result.hardware_id,
                        benchmark_result.run_id,
                        benchmark_result.timestamp,
                        metrics_json,
                        metadata_json,
                        benchmark_result.result_id
                    ]
                )
            else:
                # Insert new benchmark result
                self.conn.execute(
                    """
                    INSERT INTO benchmark_results (
                        result_id, benchmark_type, model_id, hardware_id,
                        run_id, timestamp, metrics, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        benchmark_result.result_id,
                        benchmark_result.benchmark_type.name,
                        benchmark_result.model_id,
                        benchmark_result.hardware_id,
                        benchmark_result.run_id,
                        benchmark_result.timestamp,
                        metrics_json,
                        metadata_json
                    ]
                )
            
            return benchmark_result.result_id
            
        except Exception as e:
            logger.error(f"Error saving benchmark result: {e}")
            raise
    
    def load_benchmark_result(
        self,
        result_id: str
    ) -> Optional[BenchmarkResult]:
        """
        Load a benchmark result by ID.
        
        Args:
            result_id: Identifier for the benchmark result
            
        Returns:
            BenchmarkResult if found, None otherwise
        """
        logger.info(f"Loading benchmark result {result_id}")
        
        try:
            result = self.conn.execute(
                """
                SELECT * FROM benchmark_results WHERE result_id = ?
                """,
                [result_id]
            ).fetchone()
            
            if not result:
                return None
            
            # Parse JSON fields
            metrics = json.loads(result['metrics'])
            metadata = json.loads(result['metadata'])
            
            # Create BenchmarkResult object
            benchmark_result = BenchmarkResult(
                result_id=result_id,
                benchmark_type=BenchmarkType[result['benchmark_type']],
                model_id=result['model_id'],
                hardware_id=result['hardware_id'],
                metrics=metrics,
                run_id=result['run_id'],
                timestamp=result['timestamp'],
                metadata=metadata
            )
            
            return benchmark_result
            
        except Exception as e:
            logger.error(f"Error loading benchmark result: {e}")
            return None
    
    def save_certification(
        self,
        certification: Dict[str, Any]
    ) -> str:
        """
        Save a benchmark certification.
        
        Args:
            certification: The certification to save
            
        Returns:
            Identifier for the saved certification
        """
        logger.info(f"Saving certification {certification['certification_id']}")
        
        try:
            # Convert JSON fields
            certification_requirements_json = json.dumps(certification['certification_requirements'])
            validation_results_json = json.dumps(certification['validation_results'])
            certification_data_json = json.dumps(certification['certification_data'])
            
            # Check if certification already exists
            existing = self.conn.execute(
                "SELECT certification_id FROM certifications WHERE certification_id = ?",
                [certification['certification_id']]
            ).fetchone()
            
            if existing:
                # Update existing certification
                self.conn.execute(
                    """
                    UPDATE certifications
                    SET benchmark_id = ?,
                        certification_level = ?,
                        certification_timestamp = ?,
                        certification_authority = ?,
                        certification_version = ?,
                        certification_requirements = ?,
                        validation_results = ?,
                        certification_data = ?,
                        certification_hash = ?
                    WHERE certification_id = ?
                    """,
                    [
                        certification['benchmark_id'],
                        certification['certification_level'],
                        certification['certification_timestamp'],
                        certification['certification_authority'],
                        certification['certification_version'],
                        certification_requirements_json,
                        validation_results_json,
                        certification_data_json,
                        certification['certification_hash'],
                        certification['certification_id']
                    ]
                )
            else:
                # Insert new certification
                self.conn.execute(
                    """
                    INSERT INTO certifications (
                        certification_id, benchmark_id, certification_level,
                        certification_timestamp, certification_authority, certification_version,
                        certification_requirements, validation_results, certification_data,
                        certification_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        certification['certification_id'],
                        certification['benchmark_id'],
                        certification['certification_level'],
                        certification['certification_timestamp'],
                        certification['certification_authority'],
                        certification['certification_version'],
                        certification_requirements_json,
                        validation_results_json,
                        certification_data_json,
                        certification['certification_hash']
                    ]
                )
            
            return certification['certification_id']
            
        except Exception as e:
            logger.error(f"Error saving certification: {e}")
            raise
    
    def load_certification(
        self,
        certification_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load a certification by ID.
        
        Args:
            certification_id: Identifier for the certification
            
        Returns:
            Certification dictionary if found, None otherwise
        """
        logger.info(f"Loading certification {certification_id}")
        
        try:
            result = self.conn.execute(
                """
                SELECT * FROM certifications WHERE certification_id = ?
                """,
                [certification_id]
            ).fetchone()
            
            if not result:
                return None
            
            # Parse JSON fields
            certification_requirements = json.loads(result['certification_requirements'])
            validation_results = json.loads(result['validation_results'])
            certification_data = json.loads(result['certification_data'])
            
            # Create certification dictionary
            certification = {
                'certification_id': result['certification_id'],
                'benchmark_id': result['benchmark_id'],
                'certification_level': result['certification_level'],
                'certification_timestamp': result['certification_timestamp'].isoformat() if result['certification_timestamp'] else None,
                'certification_authority': result['certification_authority'],
                'certification_version': result['certification_version'],
                'certification_requirements': certification_requirements,
                'validation_results': validation_results,
                'certification_data': certification_data,
                'certification_hash': result['certification_hash']
            }
            
            return certification
            
        except Exception as e:
            logger.error(f"Error loading certification: {e}")
            return None
    
    def query_certifications(
        self,
        filters: Dict[str, Any] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query certifications using filters.
        
        Args:
            filters: Dictionary of filter criteria
            limit: Maximum number of results to return
            
        Returns:
            List of certification dictionaries matching filters
        """
        logger.info(f"Querying certifications with filters: {filters}")
        
        try:
            # Build query conditions
            conditions = []
            params = []
            
            if filters:
                for key, value in filters.items():
                    # Handle special filters
                    if key == "model_id" or key == "hardware_id":
                        conditions.append(f"c.benchmark_id IN (SELECT result_id FROM benchmark_results WHERE {key} = ?)")
                        params.append(value)
                    elif key == "since":
                        conditions.append("c.certification_timestamp >= ?")
                        params.append(value)
                    elif key == "until":
                        conditions.append("c.certification_timestamp <= ?")
                        params.append(value)
                    else:
                        # General fields
                        conditions.append(f"c.{key} = ?")
                        params.append(value)
            
            # Build query
            query = """
            SELECT c.* FROM certifications c
            """
            
            if conditions:
                query += f" WHERE {' AND '.join(conditions)}"
            
            query += " ORDER BY c.certification_timestamp DESC LIMIT ?"
            params.append(limit)
            
            # Execute query
            result = self.conn.execute(query, params)
            rows = result.fetchall()
            
            # Load certifications
            certifications = []
            for row in rows:
                certification = self.load_certification(row['certification_id'])
                if certification:
                    certifications.append(certification)
            
            return certifications
            
        except Exception as e:
            logger.error(f"Error querying certifications: {e}")
            return []