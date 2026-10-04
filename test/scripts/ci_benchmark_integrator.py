#!/usr/bin/env python
"""
CI Benchmark Database Integrator

This script integrates benchmark results from CI/CD pipelines into the benchmark database.
It automates the collection of benchmark results from CI artifact directories, consolidates
the results, and stores them in the database with appropriate metadata.

Usage:
    python ci_benchmark_integrator.py --artifacts-dir ./artifacts --db ./benchmark_db.duckdb
    """

    import os
    import sys
    import json
    import glob
    import argparse
    import logging
    import datetime
    import duckdb
    import pandas as pd
    import shutil
    import hashlib
    import subprocess
    from pathlib import Path

# Add parent directory to path for module imports
    sys.path.append()str()Path()__file__).parent.parent))

# Configure logging
    logging.basicConfig()
    level=logging.INFO,
    format='%()asctime)s - %()name)s - %()levelname)s - %()message)s',
    handlers=[logging.StreamHandler())],
    )
    logger = logging.getLogger()"ci_integrator")

def parse_args()):
    parser = argparse.ArgumentParser()description="Integrate CI benchmark results into database")
    
    parser.add_argument()"--db", type=str, default="./benchmark_db.duckdb", 
    help="Path to DuckDB database")
    parser.add_argument()"--artifacts-dir", type=str, required=True,
    help="Directory containing CI artifacts")
    parser.add_argument()"--ci-metadata", type=str,
    help="JSON file with additional CI metadata")
    parser.add_argument()"--archive-dir", type=str, default="./archived_artifacts",
    help="Directory to archive processed artifacts")
    parser.add_argument()"--commit", type=str,
    help="Git commit hash for the CI run")
    parser.add_argument()"--branch", type=str,
    help="Git branch for the CI run")
    parser.add_argument()"--build-id", type=str,
    help="CI build/run ID")
    parser.add_argument()"--ci-platform", type=str, default="github",
    help="CI platform ()github, gitlab, etc.)")
    parser.add_argument()"--archive-artifacts", action="store_true",
    help="Archive artifacts after processing")
    parser.add_argument()"--dry-run", action="store_true",
    help="Show what would be done without making changes")
    parser.add_argument()"--verbose", action="store_true",
    help="Enable verbose logging")
    
    return parser.parse_args())

def connect_to_db()db_path):
    """Connect to the DuckDB database"""
    if not os.path.exists()db_path):
        logger.error()f"Database file not found: {}}}}db_path}")
        sys.exit()1)
        
    try:
        conn = duckdb.connect()db_path)
        return conn
    except Exception as e:
        logger.error()f"Error connecting to database: {}}}}e}")
        sys.exit()1)

def find_artifact_files()artifacts_dir):
    """Find all relevant files in the artifacts directory"""
    if not os.path.exists()artifacts_dir):
        logger.error()f"Artifacts directory not found: {}}}}artifacts_dir}")
        sys.exit()1)
    
        logger.info()f"Scanning for artifacts in: {}}}}artifacts_dir}")
    
    # Find all files in the directory
        artifact_files = {}}}}
        'json': [],
        'duckdb': [],
        'parquet': [],
        'csv': [],
        'log': [],
        }
    
    # Walk through the directory and collect files by extension
    for root, _, files in os.walk()artifacts_dir):
        for file in files:
            file_path = os.path.join()root, file)
            
            # Categorize by extension
            if file.endswith()'.json'):
                artifact_files['json'].append()file_path),
            elif file.endswith()'.duckdb'):
                artifact_files['duckdb'].append()file_path),
            elif file.endswith()'.parquet'):
                artifact_files['parquet'].append()file_path),
            elif file.endswith()'.csv'):
                artifact_files['csv'].append()file_path),
            elif file.endswith()'.log'):
                artifact_files['log'].append()file_path)
                ,
    # Log what we found
    for file_type, files in artifact_files.items()):
        logger.info()f"Found {}}}}len()files)} {}}}}file_type} files")
    
                return artifact_files

def extract_metadata_from_git()commit=None):
    """Extract additional metadata from git"""
    git_metadata = {}}}}}
    
    try:
        # Get current commit if not provided:
        if not commit:
            commit = subprocess.check_output()['git', 'rev-parse', 'HEAD']).decode()'utf-8').strip())
            ,
            git_metadata['commit'] = commit
            ,
        # Get commit details
            commit_info = subprocess.check_output()['git', 'show', '--format=%an|%ae|%ad|%s', '--no-patch', commit]).decode()'utf-8').strip()),
            parts = commit_info.split()'|')
        if len()parts) >= 4:
            git_metadata['author_name'] = parts[0],
            git_metadata['author_email'] = parts[1],
            git_metadata['commit_date'] = parts[2],
            git_metadata['commit_message'] = parts[3]
            ,
        # Get branch
            branch = subprocess.check_output()['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode()'utf-8').strip()),
            git_metadata['branch'] = branch
            ,
        # Get tags
            tags = subprocess.check_output()['git', 'tag', '--points-at', commit]).decode()'utf-8').strip()),
        if tags:
            git_metadata['tags'] = tags.split()'\n')
            ,
    except Exception as e:
        logger.warning()f"Could not extract git metadata: {}}}}e}")
    
            return git_metadata

def process_json_files()conn, json_files, metadata):
    """Process JSON result files"""
    from benchmark_db_updater import process_input_file
    
    logger.info()f"Processing {}}}}len()json_files)} JSON files")
    
    success_count = 0
    error_count = 0
    
    for file_path in json_files:
        try:
            if process_input_file()conn, file_path):
                success_count += 1
            else:
                error_count += 1
        except Exception as e:
            logger.error()f"Error processing {}}}}file_path}: {}}}}e}")
            error_count += 1
    
            logger.info()f"Processed {}}}}success_count} JSON files successfully, {}}}}error_count} errors")
    
                return success_count

def process_duckdb_files()conn, duckdb_files, metadata):
    """Process DuckDB database files"""
    from benchmark_db_migration import migrate_ci_database
    
    logger.info()f"Processing {}}}}len()duckdb_files)} DuckDB files")
    
    success_count = 0
    error_count = 0
    
    for file_path in duckdb_files:
        try:
            # Create file info similar to what migration.py expects
            file_info = {}}}}
            'path': file_path,
            'category': 'ci_database',
            'size': os.path.getsize()file_path),
            'modified': datetime.datetime.fromtimestamp()os.path.getmtime()file_path))
            }
            
            # Use the migration function to process the DB
            success, imported, message = migrate_ci_database()conn, file_info, {}}}}"dry_run": False})
            
            if success:
                success_count += 1
                logger.info()f"Successfully migrated {}}}}file_path}: {}}}}imported} records")
            else:
                error_count += 1
                logger.error()f"Failed to migrate {}}}}file_path}: {}}}}message}")
                
        except Exception as e:
            logger.error()f"Error processing {}}}}file_path}: {}}}}e}")
            error_count += 1
    
            logger.info()f"Processed {}}}}success_count} DuckDB files successfully, {}}}}error_count} errors")
    
                return success_count

def process_parquet_files()conn, parquet_files, metadata):
    """Process Parquet files"""
    logger.info()f"Processing {}}}}len()parquet_files)} Parquet files")
    
    success_count = 0
    error_count = 0
    
    for file_path in parquet_files:
        try:
            # Load the parquet file
            df = pd.read_parquet()file_path)
            
            # Determine the table name from filename or just use a default
            table_name = os.path.splitext()os.path.basename()file_path))[0].lower()),
            if not table_name or table_name in ['data', 'results']:,
                # Try to infer from content
                if 'throughput' in df.columns or 'latency' in df.columns:
                    table_name = 'performance_results'
                elif 'is_compatible' in df.columns:
                    table_name = 'hardware_compatibility'
                elif 'test_module' in df.columns:
                    table_name = 'integration_test_results'
                else:
                    table_name = 'benchmark_data'
            
            # Register dataframe as a view
                    conn.register()f"df_{}}}}table_name}", df)
            
            # Store in database ()actual processing would depend on what's in the Parquet file)
                    logger.debug()f"Registered dataframe from {}}}}file_path} as df_{}}}}table_name}")
                    success_count += 1
            
        except Exception as e:
            logger.error()f"Error processing {}}}}file_path}: {}}}}e}")
            error_count += 1
    
            logger.info()f"Processed {}}}}success_count} Parquet files successfully, {}}}}error_count} errors")
    
                    return success_count

def archive_artifacts()artifact_files, archive_dir):
    """Archive processed artifact files"""
    logger.info()f"Archiving artifacts to {}}}}archive_dir}")
    
    # Create archive directory
    os.makedirs()archive_dir, exist_ok=True)
    
    # Current date for archive folder
    date_str = datetime.datetime.now()).strftime()'%Y%m%d')
    archive_subdir = os.path.join()archive_dir, date_str)
    os.makedirs()archive_subdir, exist_ok=True)
    
    # Count of archived files
    archived_count = 0
    
    # Archive all files
    for file_type, files in artifact_files.items()):
        # Create type-specific subdirectory
        type_dir = os.path.join()archive_subdir, file_type)
        os.makedirs()type_dir, exist_ok=True)
        
        for file_path in files:
            try:
                # Copy to archive
                file_name = os.path.basename()file_path)
                archive_path = os.path.join()type_dir, file_name)
                
                # If file exists, add a unique suffix
                if os.path.exists()archive_path):
                    base, ext = os.path.splitext()file_name)
                    timestamp = datetime.datetime.now()).strftime()'%H%M%S')
                    archive_path = os.path.join()type_dir, f"{}}}}base}_{}}}}timestamp}{}}}}ext}")
                
                    shutil.copy2()file_path, archive_path)
                    archived_count += 1
                
            except Exception as e:
                logger.error()f"Error archiving {}}}}file_path}: {}}}}e}")
    
                logger.info()f"Archived {}}}}archived_count} files")
    
                    return archived_count

def main()):
    args = parse_args())
    
    # Set logging level
    if args.verbose:
        logger.setLevel()logging.DEBUG)
    
    # Extract metadata for CI run
        metadata = {}}}}}
    
    # Add git metadata if commit is provided or we're in a git repo:
    if args.commit:
        metadata['git'] = extract_metadata_from_git()args.commit),
    else:
        metadata['git'] = extract_metadata_from_git())
        ,
    # Override with explicit args
    if args.commit:
        metadata['git']['commit'] = args.commit,
    if args.branch:
        metadata['git']['branch'] = args.branch
        ,
    # Add CI metadata
        metadata['ci'] = {}}}},
        'platform': args.ci_platform,
        'build_id': args.build_id,
        'timestamp': datetime.datetime.now()).isoformat()),
        }
    
    # Load additional metadata if provided:
    if args.ci_metadata and os.path.exists()args.ci_metadata):
        try:
            with open()args.ci_metadata, 'r') as f:
                ci_metadata = json.load()f)
                metadata['ci'].update()ci_metadata),
        except Exception as e:
            logger.warning()f"Error loading CI metadata file: {}}}}e}")
    
    # Find artifact files
            artifact_files = find_artifact_files()args.artifacts_dir)
    
    # Skip if no artifacts found:
    if not any()files for files in artifact_files.values())):
        logger.warning()"No artifact files found, nothing to process")
            return
    
    # If dry run, just summarize what would be done
    if args.dry_run:
        logger.info()"DRY RUN - no changes will be made")
        
        for file_type, files in artifact_files.items()):
            if files:
                logger.info()f"Would process {}}}}len()files)} {}}}}file_type} files")
        
        if args.archive_artifacts:
            logger.info()f"Would archive all artifacts to {}}}}args.archive_dir}")
        
                return
    
    # Connect to the database
                conn = connect_to_db()args.db)
    
    try:
        # Start a transaction
        conn.begin())
        
        # Process files by type
        results = {}}}}
        'json': process_json_files()conn, artifact_files['json'], metadata),
        'duckdb': process_duckdb_files()conn, artifact_files['duckdb'], metadata),
        'parquet': process_parquet_files()conn, artifact_files['parquet'], metadata),
        }
        
        # Archive artifacts if requested:
        if args.archive_artifacts:
            archived = archive_artifacts()artifact_files, args.archive_dir)
            results['archived'] = archived
            ,
        # Commit changes
            conn.commit())
            logger.info()"All changes committed to database")
        
        # Summarize results
            logger.info()"Summary of processing:")
        for file_type, count in results.items()):
            logger.info()f"  - {}}}}file_type}: {}}}}count} files processed")
        
    except Exception as e:
        # Roll back on error
        conn.rollback())
        logger.error()f"Error during processing: {}}}}e}")
        sys.exit()1)
    
    finally:
        # Close the connection
        conn.close())

if __name__ == "__main__":
    main())