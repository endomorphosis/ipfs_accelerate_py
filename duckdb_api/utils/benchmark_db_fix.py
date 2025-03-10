#!/usr/bin/env python
"""
Benchmark Database Fix Tool

This script fixes database schema issues in the DuckDB database, particularly
addressing timestamp type errors and rebuilding problematic tables.
"""

import os
import sys
import logging
import argparse
import datetime
from pathlib import Path

try:
    import duckdb
    import pandas as pd
except ImportError:
    print()))))))"Error: Required packages not installed. Please install with:")
    print()))))))"pip install duckdb pandas")
    sys.exit()))))))1)

# Configure logging
    logging.basicConfig()))))))level=logging.INFO,
    format='%()))))))asctime)s - %()))))))name)s - %()))))))levelname)s - %()))))))message)s')
    logger = logging.getLogger()))))))__name__)

class BenchmarkDBFix:
    """
    Fix tool for the benchmark database.
    """
    
    def __init__()))))))self, db_path: str = "./benchmark_db.duckdb", debug: bool = False):
        """
        Initialize the benchmark database fix tool.
        
        Args:
            db_path: Path to the DuckDB database
            debug: Enable debug logging
            """
            self.db_path = db_path
        
        # Set up logging
        if debug:
            logger.setLevel()))))))logging.DEBUG)
        
        # Verify database exists
        if not os.path.exists()))))))db_path):
            logger.warning()))))))f"\1{db_path}\3")
            logger.info()))))))"Creating a new database file")
        
            logger.info()))))))f"\1{db_path}\3")
    
    def fix_timestamp_issues()))))))self) -> bool:
        """
        Fix timestamp type issues in the database.
        
        Returns:
            True if successful, False otherwise
        """:::
        try:
            # Connect to database
            conn = duckdb.connect()))))))self.db_path)
            
            # Check if we have any tables
            tables = conn.execute()))))))"SHOW TABLES").fetchall()))))))):
            if not tables:
                logger.warning()))))))"No tables found in database")
                conn.close())))))))
                return False
            
            # Create backups of problematic tables
            try:
                conn.execute()))))))"CREATE TABLE IF NOT EXISTS test_runs_backup AS SELECT * FROM test_runs")
                logger.info()))))))"Created backup of test_runs table")
            except Exception as e:
                logger.warning()))))))f"\1{e}\3")
            
            try:
                conn.execute()))))))"CREATE TABLE IF NOT EXISTS performance_results_backup AS SELECT * FROM performance_results")
                logger.info()))))))"Created backup of performance_results table")
            except Exception as e:
                logger.warning()))))))f"\1{e}\3")
            
            # Drop problematic tables
            try:
                conn.execute()))))))"DROP TABLE IF EXISTS test_runs")
                logger.info()))))))"Dropped test_runs table")
            except Exception as e:
                logger.error()))))))f"\1{e}\3")
            
            try:
                conn.execute()))))))"DROP TABLE IF EXISTS performance_results")
                logger.info()))))))"Dropped performance_results table")
            except Exception as e:
                logger.error()))))))f"\1{e}\3")
            
            # Recreate tables with correct schema
                conn.execute()))))))"""
                CREATE TABLE IF NOT EXISTS test_runs ()))))))
                run_id INTEGER PRIMARY KEY,
                test_name VARCHAR NOT NULL,
                test_type VARCHAR NOT NULL,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                execution_time_seconds FLOAT,
                success BOOLEAN,
                git_commit VARCHAR,
                git_branch VARCHAR,
                command_line VARCHAR,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                logger.info()))))))"Recreated test_runs table with correct schema")
            
                conn.execute()))))))"""
                CREATE TABLE IF NOT EXISTS performance_results ()))))))
                result_id INTEGER PRIMARY KEY,
                run_id INTEGER,
                model_id INTEGER NOT NULL,
                hardware_id INTEGER NOT NULL,
                test_case VARCHAR,
                batch_size INTEGER,
                precision VARCHAR,
                total_time_seconds FLOAT,
                average_latency_ms FLOAT,
                throughput_items_per_second FLOAT,
                memory_peak_mb FLOAT,
                iterations INTEGER,
                warmup_iterations INTEGER,
                metrics JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY ()))))))run_id) REFERENCES test_runs()))))))run_id),
                FOREIGN KEY ()))))))model_id) REFERENCES models()))))))model_id),
                FOREIGN KEY ()))))))hardware_id) REFERENCES hardware_platforms()))))))hardware_id)
                )
                """)
                logger.info()))))))"Recreated performance_results table with correct schema")
            
            # Try to restore data from backups
            try:
                # Use a more targeted approach for timestamp fields
                conn.execute()))))))"""
                INSERT INTO test_runs 
                SELECT 
                run_id,
                test_name,
                test_type,
                CAST()))))))started_at AS TIMESTAMP),
                CAST()))))))completed_at AS TIMESTAMP),
                execution_time_seconds,
                success,
                git_commit,
                git_branch,
                command_line,
                metadata,
                CAST()))))))created_at AS TIMESTAMP)
                FROM test_runs_backup
                """)
                logger.info()))))))"Restored data to test_runs table")
            except Exception as e:
                logger.warning()))))))f"\1{e}\3")
            
            try:
                conn.execute()))))))"""
                INSERT INTO performance_results
                SELECT 
                result_id,
                run_id,
                model_id,
                hardware_id,
                test_case,
                batch_size,
                precision,
                total_time_seconds,
                average_latency_ms,
                throughput_items_per_second,
                memory_peak_mb,
                iterations,
                warmup_iterations,
                metrics,
                CAST()))))))created_at AS TIMESTAMP)
                FROM performance_results_backup
                """)
                logger.info()))))))"Restored data to performance_results table")
            except Exception as e:
                logger.warning()))))))f"\1{e}\3")
            
            # Close connection
                conn.close())))))))
            
                logger.info()))))))"Fixed timestamp issues in database")
                return True
            
        except Exception as e:
            logger.error()))))))f"\1{e}\3")
                return False
    
    def fix_web_platform_tables()))))))self) -> bool:
        """
        Fix web platform tables in the database.
        
        Returns:
            True if successful, False otherwise
        """:::
        try:
            # Connect to database
            conn = duckdb.connect()))))))self.db_path)
            
            # Check if we have any web platform tables
            tables = conn.execute()))))))"SHOW TABLES").fetchall()))))))):
                table_names = [],t[],0].lower()))))))) for t in tables]:,
                web_platform_tables = [],
                'web_platform_results',
                'webgpu_advanced_features'
                ]
            
            # Backup existing tables if they exist:
            for table in web_platform_tables:
                if table.lower()))))))) in table_names:
                    try:
                        conn.execute()))))))f"\1{table}\3")
                        logger.info()))))))f"Created backup of {table} table")
                    except Exception as e:
                        logger.warning()))))))f"\1{e}\3")
                    
                    try:
                        conn.execute()))))))f"\1{table}\3")
                        logger.info()))))))f"Dropped {table} table")
                    except Exception as e:
                        logger.error()))))))f"\1{e}\3")
            
            # Create web platform tables with correct schema
                        conn.execute()))))))"""
                        CREATE TABLE IF NOT EXISTS web_platform_results ()))))))
                        result_id INTEGER PRIMARY KEY,
                        run_id INTEGER,
                        model_id INTEGER NOT NULL,
                        hardware_id INTEGER NOT NULL,
                        platform VARCHAR NOT NULL,
                        browser VARCHAR,
                        browser_version VARCHAR,
                        test_file VARCHAR,
                        success BOOLEAN,
                        load_time_ms FLOAT,
                        initialization_time_ms FLOAT,
                        inference_time_ms FLOAT,
                        total_time_ms FLOAT,
                        shader_compilation_time_ms FLOAT,
                        memory_usage_mb FLOAT,
                        error_message VARCHAR,
                        metrics JSON,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY ()))))))run_id) REFERENCES test_runs()))))))run_id),
                        FOREIGN KEY ()))))))model_id) REFERENCES models()))))))model_id),
                        FOREIGN KEY ()))))))hardware_id) REFERENCES hardware_platforms()))))))hardware_id)
                        )
                        """)
                        logger.info()))))))"Recreated web_platform_results table with correct schema")
            
                        conn.execute()))))))"""
                        CREATE TABLE IF NOT EXISTS webgpu_advanced_features ()))))))
                        feature_id INTEGER PRIMARY KEY,
                        result_id INTEGER NOT NULL,
                        compute_shader_support BOOLEAN,
                        parallel_compilation BOOLEAN,
                        shader_cache_hit BOOLEAN,
                        workgroup_size INTEGER,
                        compute_pipeline_time_ms FLOAT,
                        pre_compiled_pipeline BOOLEAN,
                        memory_optimization_level VARCHAR,
                        audio_acceleration BOOLEAN,
                        video_acceleration BOOLEAN,
                        parallel_loading BOOLEAN,
                        parallel_loading_speedup FLOAT,
                        components_loaded INTEGER,
                        component_loading_time_ms FLOAT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY ()))))))result_id) REFERENCES web_platform_results()))))))result_id)
                        )
                        """)
                        logger.info()))))))"Recreated webgpu_advanced_features table with correct schema")
            
            # Try to restore data from backups
            if 'web_platform_results_backup'.lower()))))))) in table_names:
                try:
                    conn.execute()))))))"""
                    INSERT INTO web_platform_results
                    SELECT 
                    result_id,
                    run_id,
                    model_id,
                    hardware_id,
                    platform,
                    browser,
                    browser_version,
                    test_file,
                    success,
                    load_time_ms,
                    initialization_time_ms,
                    inference_time_ms,
                    total_time_ms,
                    shader_compilation_time_ms,
                    memory_usage_mb,
                    error_message,
                    metrics,
                    CAST()))))))created_at AS TIMESTAMP)
                    FROM web_platform_results_backup
                    """)
                    logger.info()))))))"Restored data to web_platform_results table")
                except Exception as e:
                    logger.warning()))))))f"\1{e}\3")
            
            if 'webgpu_advanced_features_backup'.lower()))))))) in table_names:
                try:
                    conn.execute()))))))"""
                    INSERT INTO webgpu_advanced_features
                    SELECT 
                    feature_id,
                    result_id,
                    compute_shader_support,
                    parallel_compilation,
                    shader_cache_hit,
                    workgroup_size,
                    compute_pipeline_time_ms,
                    pre_compiled_pipeline,
                    memory_optimization_level,
                    audio_acceleration,
                    video_acceleration,
                    parallel_loading,
                    parallel_loading_speedup,
                    components_loaded,
                    component_loading_time_ms,
                    CAST()))))))created_at AS TIMESTAMP)
                    FROM webgpu_advanced_features_backup
                    """)
                    logger.info()))))))"Restored data to webgpu_advanced_features table")
                except Exception as e:
                    logger.warning()))))))f"\1{e}\3")
            
            # Close connection
                    conn.close())))))))
            
                    logger.info()))))))"Fixed web platform tables in database")
                    return True
            
        except Exception as e:
            logger.error()))))))f"\1{e}\3")
                    return False
    
    def recreate_core_tables()))))))self) -> bool:
        """
        Recreate core tables in the database.
        
        Returns:
            True if successful, False otherwise
        """:::
        try:
            # Connect to database
            conn = duckdb.connect()))))))self.db_path)
            
            # Check existing tables
            tables = conn.execute()))))))"SHOW TABLES").fetchall())))))))
            table_names = [],t[],0].lower()))))))) for t in tables]:,
            # Create core tables if they don't exist:
            if 'models' not in table_names:
                conn.execute()))))))"""
                CREATE TABLE models ()))))))
                model_id INTEGER PRIMARY KEY,
                model_name VARCHAR NOT NULL,
                model_family VARCHAR,
                modality VARCHAR,
                source VARCHAR,
                version VARCHAR,
                parameters_million FLOAT,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                logger.info()))))))"Created models table")
            
            if 'hardware_platforms' not in table_names:
                conn.execute()))))))"""
                CREATE TABLE hardware_platforms ()))))))
                hardware_id INTEGER PRIMARY KEY,
                hardware_type VARCHAR NOT NULL,
                device_name VARCHAR,
                platform VARCHAR,
                platform_version VARCHAR,
                driver_version VARCHAR,
                memory_gb FLOAT,
                compute_units INTEGER,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                logger.info()))))))"Created hardware_platforms table")
            
            if 'hardware_compatibility' not in table_names:
                conn.execute()))))))"""
                CREATE TABLE hardware_compatibility ()))))))
                compatibility_id INTEGER PRIMARY KEY,
                run_id INTEGER,
                model_id INTEGER NOT NULL,
                hardware_id INTEGER NOT NULL,
                is_compatible BOOLEAN NOT NULL,
                detection_success BOOLEAN NOT NULL,
                initialization_success BOOLEAN NOT NULL,
                error_message VARCHAR,
                error_type VARCHAR,
                suggested_fix VARCHAR,
                workaround_available BOOLEAN,
                compatibility_score FLOAT,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY ()))))))run_id) REFERENCES test_runs()))))))run_id),
                FOREIGN KEY ()))))))model_id) REFERENCES models()))))))model_id),
                FOREIGN KEY ()))))))hardware_id) REFERENCES hardware_platforms()))))))hardware_id)
                )
                """)
                logger.info()))))))"Created hardware_compatibility table")
            
            # Close connection
                conn.close())))))))
            
                logger.info()))))))"Recreated core tables in database")
                return True
            
        except Exception as e:
            logger.error()))))))f"\1{e}\3")
                return False
    
    def fix_database()))))))self) -> bool:
        """
        Fix various issues in the database.
        
        Returns:
            True if successful, False otherwise
        """:::
            success = True
        
        # Recreate core tables first
        if not self.recreate_core_tables()))))))):
            logger.warning()))))))"Could not recreate core tables")
            success = False
        
        # Fix timestamp issues
        if not self.fix_timestamp_issues()))))))):
            logger.warning()))))))"Could not fix timestamp issues")
            success = False
        
        # Fix web platform tables
        if not self.fix_web_platform_tables()))))))):
            logger.warning()))))))"Could not fix web platform tables")
            success = False
        
            return success

def main()))))))):
    """Command-line interface for the benchmark database fix tool."""
    parser = argparse.ArgumentParser()))))))description="Benchmark Database Fix Tool")
    parser.add_argument()))))))"--db", default="./benchmark_db.duckdb",
    help="Path to the DuckDB database")
    parser.add_argument()))))))"--fix-all", action="store_true",
    help="Fix all issues in the database")
    parser.add_argument()))))))"--fix-timestamps", action="store_true",
    help="Fix timestamp issues in the database")
    parser.add_argument()))))))"--fix-web-platform", action="store_true",
    help="Fix web platform tables in the database")
    parser.add_argument()))))))"--recreate-core-tables", action="store_true",
    help="Recreate core tables in the database")
    parser.add_argument()))))))"--debug", action="store_true",
    help="Enable debug logging")
    args = parser.parse_args())))))))
    
    # Create fix tool
    fix_tool = BenchmarkDBFix()))))))db_path=args.db, debug=args.debug)
    
    # Perform requested actions
    if args.fix_all:
        logger.info()))))))"Fixing all database issues...")
        success = fix_tool.fix_database())))))))
        
        if success:
            logger.info()))))))"Successfully fixed all database issues")
        else:
            logger.error()))))))"Failed to fix all database issues")
            
    elif args.fix_timestamps:
        logger.info()))))))"Fixing timestamp issues...")
        success = fix_tool.fix_timestamp_issues())))))))
        
        if success:
            logger.info()))))))"Successfully fixed timestamp issues")
        else:
            logger.error()))))))"Failed to fix timestamp issues")
            
    elif args.fix_web_platform:
        logger.info()))))))"Fixing web platform tables...")
        success = fix_tool.fix_web_platform_tables())))))))
        
        if success:
            logger.info()))))))"Successfully fixed web platform tables")
        else:
            logger.error()))))))"Failed to fix web platform tables")
            
    elif args.recreate_core_tables:
        logger.info()))))))"Recreating core tables...")
        success = fix_tool.recreate_core_tables())))))))
        
        if success:
            logger.info()))))))"Successfully recreated core tables")
        else:
            logger.error()))))))"Failed to recreate core tables")
            
    else:
        # No specific action requested, print help
        parser.print_help())))))))

if __name__ == "__main__":
    main())))))))