#!/usr/bin/env python3
"""
Schema for Benchmark Validation Database

This module defines the DuckDB schema for storing benchmark validation data,
including tables for benchmark results, validation results, and certifications.
"""

import os
import sys
import logging
import datetime
import json
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple, Set

import duckdb

logger = logging.getLogger("benchmark_validation.schema")

class BenchmarkValidationSchema:
    """Schema definition for benchmark validation database."""
    
    @staticmethod
    def create_tables(conn: duckdb.DuckDBPyConnection) -> None:
        """
        Create tables for benchmark validation data.
        
        Args:
            conn: DuckDB connection
        """
        logger.info("Creating benchmark validation tables")
        
        try:
            # Create benchmark_results table
            conn.execute("""
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
            conn.execute("""
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
            conn.execute("""
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
            
            # Create reproducibility_results table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS reproducibility_results (
                id VARCHAR PRIMARY KEY,
                model_id INTEGER,
                hardware_id INTEGER,
                status VARCHAR NOT NULL,
                reproducibility_score FLOAT,
                sample_size INTEGER,
                metrics JSON,
                validation_level VARCHAR NOT NULL,
                validation_timestamp TIMESTAMP,
                validator_id VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Create outlier_detection_results table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS outlier_detection_results (
                id VARCHAR PRIMARY KEY,
                model_id INTEGER,
                hardware_id INTEGER,
                metric VARCHAR NOT NULL,
                outlier_score FLOAT,
                threshold FLOAT,
                is_outlier BOOLEAN,
                benchmark_result_id VARCHAR NOT NULL,
                detection_timestamp TIMESTAMP,
                detector_id VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (benchmark_result_id) REFERENCES benchmark_results(result_id)
            )
            """)
            
            # Create data_quality_issues table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS data_quality_issues (
                id VARCHAR PRIMARY KEY,
                issue_type VARCHAR NOT NULL,
                issue_message VARCHAR NOT NULL,
                benchmark_result_id VARCHAR,
                detection_timestamp TIMESTAMP,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (benchmark_result_id) REFERENCES benchmark_results(result_id)
            )
            """)
            
            # Create stability_analysis table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS stability_analysis (
                id VARCHAR PRIMARY KEY,
                model_id INTEGER,
                hardware_id INTEGER,
                metric VARCHAR NOT NULL,
                stability_score FLOAT,
                mean_value FLOAT,
                std_dev FLOAT,
                coefficient_of_variation FLOAT,
                num_results INTEGER,
                time_window_days INTEGER,
                start_date TIMESTAMP,
                end_date TIMESTAMP,
                analysis_timestamp TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            logger.info("Benchmark validation tables created successfully")
            
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    @staticmethod
    def create_views(conn: duckdb.DuckDBPyConnection) -> None:
        """
        Create views for benchmark validation data.
        
        Args:
            conn: DuckDB connection
        """
        logger.info("Creating benchmark validation views")
        
        try:
            # Create validation_summary view
            conn.execute("""
            CREATE OR REPLACE VIEW validation_summary AS
            SELECT
                m.model_id,
                m.model_name,
                hp.hardware_id,
                hp.hardware_type,
                hp.device_name,
                vr.validation_level,
                vr.status,
                COUNT(*) as validation_count,
                AVG(vr.confidence_score) as avg_confidence_score,
                MIN(vr.validation_timestamp) as first_validation,
                MAX(vr.validation_timestamp) as last_validation
            FROM
                validation_results vr
            JOIN
                benchmark_results br ON vr.benchmark_result_id = br.result_id
            JOIN
                models m ON br.model_id = m.model_id
            JOIN
                hardware_platforms hp ON br.hardware_id = hp.hardware_id
            GROUP BY
                m.model_id, m.model_name, hp.hardware_id, hp.hardware_type, hp.device_name,
                vr.validation_level, vr.status
            """)
            
            # Create certification_summary view
            conn.execute("""
            CREATE OR REPLACE VIEW certification_summary AS
            SELECT
                m.model_id,
                m.model_name,
                hp.hardware_id,
                hp.hardware_type,
                hp.device_name,
                c.certification_level,
                COUNT(*) as certification_count,
                MIN(c.certification_timestamp) as first_certification,
                MAX(c.certification_timestamp) as last_certification
            FROM
                certifications c
            JOIN
                benchmark_results br ON c.benchmark_id = br.result_id
            JOIN
                models m ON br.model_id = m.model_id
            JOIN
                hardware_platforms hp ON br.hardware_id = hp.hardware_id
            GROUP BY
                m.model_id, m.model_name, hp.hardware_id, hp.hardware_type, hp.device_name,
                c.certification_level
            """)
            
            # Create reproducibility_summary view
            conn.execute("""
            CREATE OR REPLACE VIEW reproducibility_summary AS
            SELECT
                m.model_id,
                m.model_name,
                hp.hardware_id,
                hp.hardware_type,
                hp.device_name,
                rr.validation_level,
                rr.status,
                AVG(rr.reproducibility_score) as avg_reproducibility_score,
                MIN(rr.validation_timestamp) as first_validation,
                MAX(rr.validation_timestamp) as last_validation
            FROM
                reproducibility_results rr
            JOIN
                models m ON rr.model_id = m.model_id
            JOIN
                hardware_platforms hp ON rr.hardware_id = hp.hardware_id
            GROUP BY
                m.model_id, m.model_name, hp.hardware_id, hp.hardware_type, hp.device_name,
                rr.validation_level, rr.status
            """)
            
            # Create outlier_summary view
            conn.execute("""
            CREATE OR REPLACE VIEW outlier_summary AS
            SELECT
                m.model_id,
                m.model_name,
                hp.hardware_id,
                hp.hardware_type,
                hp.device_name,
                odr.metric,
                COUNT(*) as detection_count,
                COUNT(CASE WHEN odr.is_outlier THEN 1 END) as outlier_count,
                ROUND(COUNT(CASE WHEN odr.is_outlier THEN 1 END) * 100.0 / COUNT(*), 2) as outlier_percentage,
                AVG(odr.outlier_score) as avg_outlier_score,
                MIN(odr.detection_timestamp) as first_detection,
                MAX(odr.detection_timestamp) as last_detection
            FROM
                outlier_detection_results odr
            JOIN
                benchmark_results br ON odr.benchmark_result_id = br.result_id
            JOIN
                models m ON br.model_id = m.model_id
            JOIN
                hardware_platforms hp ON br.hardware_id = hp.hardware_id
            GROUP BY
                m.model_id, m.model_name, hp.hardware_id, hp.hardware_type, hp.device_name,
                odr.metric
            """)
            
            # Create stability_summary view
            conn.execute("""
            CREATE OR REPLACE VIEW stability_summary AS
            SELECT
                m.model_id,
                m.model_name,
                hp.hardware_id,
                hp.hardware_type,
                hp.device_name,
                sa.metric,
                AVG(sa.stability_score) as avg_stability_score,
                AVG(sa.coefficient_of_variation) as avg_cv,
                MIN(sa.start_date) as earliest_data,
                MAX(sa.end_date) as latest_data,
                MIN(sa.analysis_timestamp) as first_analysis,
                MAX(sa.analysis_timestamp) as last_analysis
            FROM
                stability_analysis sa
            JOIN
                models m ON sa.model_id = m.model_id
            JOIN
                hardware_platforms hp ON sa.hardware_id = hp.hardware_id
            GROUP BY
                m.model_id, m.model_name, hp.hardware_id, hp.hardware_type, hp.device_name,
                sa.metric
            """)
            
            logger.info("Benchmark validation views created successfully")
            
        except Exception as e:
            logger.error(f"Error creating views: {e}")
            raise
    
    @staticmethod
    def add_validation_schema_to_db(db_path: str) -> None:
        """
        Add benchmark validation schema to an existing database.
        
        Args:
            db_path: Path to the DuckDB database
        """
        logger.info(f"Adding benchmark validation schema to {db_path}")
        
        try:
            # Connect to database
            conn = duckdb.connect(db_path)
            
            # Create tables
            BenchmarkValidationSchema.create_tables(conn)
            
            # Create views
            try:
                BenchmarkValidationSchema.create_views(conn)
            except Exception as e:
                # Views may fail if models or hardware_platforms tables don't exist
                logger.warning(f"Could not create views: {e}")
            
            logger.info("Benchmark validation schema added successfully")
            
        except Exception as e:
            logger.error(f"Error adding schema to database: {e}")
            raise