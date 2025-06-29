#!/usr/bin/env python
"""
Database maintenance utilities for the benchmark database.

This module provides tools for maintaining the benchmark database, including
optimization, backup, and integrity checks.
"""

import os
import sys
import logging
import argparse
import datetime
import shutil
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    import duckdb
    import pandas as pd
except ImportError:
    print("Error: Required packages not installed. Please install with:")
    print("pip install duckdb pandas")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BenchmarkDBMaintenance:
    """
    Utilities for maintaining the benchmark database.
    """
    
    def __init__(self, db_path: str = "./benchmark_db.duckdb", debug: bool = False):
        """
        Initialize the benchmark database maintenance utilities.
        
        Args:
            db_path: Path to the DuckDB database
            debug: Enable debug logging
        """
        self.db_path = db_path
        self.db_file = Path(db_path)
        
        # Set up logging
        if debug:
            logger.setLevel(logging.DEBUG)
        
        logger.info(f"Initialized BenchmarkDBMaintenance for database: {db_path}")
    
    def check_integrity(self) -> bool:
        """
        Check the integrity of the database.
        
        Returns:
            True if database is valid, False otherwise
        """
        # First, check if the file exists
        if not self.db_file.exists():
            logger.error(f"Database file not found: {self.db_path}")
            return False
        
        # Check if the file is a valid DuckDB database
        try:
            con = duckdb.connect(self.db_path)
            
            # Check if expected tables exist
            expected_tables = [
                'hardware_platforms',
                'models',
                'test_runs',
                'performance_results',
                'hardware_compatibility'
            ]
            
            tables = con.execute("SHOW TABLES").fetchall()
            table_names = [t[0].lower() for t in tables]
            
            # Check if all expected tables exist
            missing_tables = [t for t in expected_tables if t.lower() not in table_names]
            if missing_tables:
                logger.error(f"Missing expected tables: {', '.join(missing_tables)}")
                return False
            
            # Run pragma check on the database
            try:
                con.execute("PRAGMA database_check")
                logger.info("Database integrity check passed successfully")
            except Exception as e:
                logger.error(f"Database integrity check failed: {e}")
                return False
            
            # Close connection
            con.close()
            return True
            
        except Exception as e:
            logger.error(f"Error checking database integrity: {e}")
            return False
    
    def optimize_db(self) -> bool:
        """
        Optimize the database by vacuuming and analyzing.
        
        Returns:
            True if optimization was successful, False otherwise
        """
        try:
            con = duckdb.connect(self.db_path)
            
            # Vacuum the database to reclaim space
            logger.info("Vacuuming database...")
            con.execute("VACUUM")
            
            # Analyze tables to improve query performance
            logger.info("Analyzing database tables...")
            con.execute("ANALYZE")
            
            # Close connection
            con.close()
            
            logger.info("Database optimization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error optimizing database: {e}")
            return False
    
    def backup_db(self, backup_dir: str = "./db_backups", compress: bool = True) -> Optional[str]:
        """
        Create a backup of the database.
        
        Args:
            backup_dir: Directory to store backups
            compress: Whether to compress the backup
            
        Returns:
            Path to the backup file, or None if backup failed
        """
        # Create backup directory if it doesn't exist
        os.makedirs(backup_dir, exist_ok=True)
        
        # Create backup filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"benchmark_db_backup_{timestamp}.duckdb"
        backup_path = os.path.join(backup_dir, backup_filename)
        
        try:
            # Copy the database file
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backup created: {backup_path}")
            
            # Compress the backup if requested
            if compress:
                compressed_path = f"{backup_path}.tar.gz"
                with tarfile.open(compressed_path, "w:gz") as tar:
                    tar.add(backup_path, arcname=os.path.basename(backup_path))
                
                # Remove the uncompressed backup
                os.remove(backup_path)
                backup_path = compressed_path
                logger.info(f"Database backup compressed: {backup_path}")
            
            return backup_path
            
        except Exception as e:
            logger.error(f"Error creating database backup: {e}")
            return None
    
    def purge_old_backups(self, backup_dir: str = "./db_backups", retention_days: int = 30) -> int:
        """
        Purge old database backups beyond the retention period.
        
        Args:
            backup_dir: Directory containing backups
            retention_days: Number of days to retain backups
            
        Returns:
            Number of backups deleted
        """
        if not os.path.exists(backup_dir):
            logger.warning(f"Backup directory not found: {backup_dir}")
            return 0
        
        # Calculate cutoff date
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=retention_days)
        
        # Find all backup files
        backup_pattern = "benchmark_db_backup_*.duckdb*"
        backup_files = list(Path(backup_dir).glob(backup_pattern))
        
        # Count deleted files
        deleted_count = 0
        
        for backup_file in backup_files:
            try:
                # Extract date from filename
                date_str = str(backup_file.stem).split('_')[-2]
                time_str = str(backup_file.stem).split('_')[-1]
                
                if date_str.isdigit() and len(date_str) == 8 and time_str.isdigit():
                    # Parse date string
                    file_date = datetime.datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
                    
                    # Check if file is older than cutoff date
                    if file_date < cutoff_date:
                        # Delete the file
                        backup_file.unlink()
                        logger.info(f"Deleted old backup: {backup_file}")
                        deleted_count += 1
            except Exception as e:
                logger.warning(f"Error processing backup file {backup_file}: {e}")
        
        logger.info(f"Purged {deleted_count} old backups (retention: {retention_days} days)")
        return deleted_count
    
    def clean_json_files(self, json_dir: str = "./benchmark_results", older_than_days: int = 30) -> int:
        """
        Clean up old JSON benchmark files that have been migrated to the database.
        
        Args:
            json_dir: Directory containing JSON files
            older_than_days: Only clean files older than this many days
            
        Returns:
            Number of files cleaned up
        """
        if not os.path.exists(json_dir):
            logger.warning(f"JSON directory not found: {json_dir}")
            return 0
        
        # Calculate cutoff date
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=older_than_days)
        
        # Find all JSON files
        json_files = list(Path(json_dir).rglob("*.json"))
        
        # Count deleted files
        deleted_count = 0
        
        for json_file in json_files:
            try:
                # Get file modification time
                mod_time = datetime.datetime.fromtimestamp(json_file.stat().st_mtime)
                
                # Check if file is older than cutoff date
                if mod_time < cutoff_date:
                    # Delete the file
                    json_file.unlink()
                    logger.info(f"Deleted old JSON file: {json_file}")
                    deleted_count += 1
            except Exception as e:
                logger.warning(f"Error processing JSON file {json_file}: {e}")
        
        logger.info(f"Cleaned up {deleted_count} old JSON files (older than {older_than_days} days)")
        return deleted_count
    
    def get_database_stats(self) -> Dict:
        """
        Get statistics about the database.
        
        Returns:
            Dictionary of database statistics
        """
        stats = {
            'file_size_mb': 0,
            'table_counts': {},
            'total_rows': 0,
            'last_modified': None
        }
        
        # Check if the file exists
        if not self.db_file.exists():
            logger.error(f"Database file not found: {self.db_path}")
            return stats
        
        # Get file size
        stats['file_size_mb'] = self.db_file.stat().st_size / (1024 * 1024)
        
        # Get last modified time
        stats['last_modified'] = datetime.datetime.fromtimestamp(
            self.db_file.stat().st_mtime
        ).strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            con = duckdb.connect(self.db_path)
            
            # Get list of tables
            tables = con.execute("SHOW TABLES").fetchall()
            
            # Get row count for each table
            total_rows = 0
            for table in tables:
                table_name = table[0]
                count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                stats['table_counts'][table_name] = count
                total_rows += count
            
            stats['total_rows'] = total_rows
            
            # Close connection
            con.close()
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
        
        return stats
    
    def generate_migration_stats(self, output_file: Optional[str] = None) -> Dict:
        """
        Generate migration statistics report.
        
        Args:
            output_file: Path to output JSON file, or None to not write to file
            
        Returns:
            Dictionary of migration statistics
        """
        stats = self.get_database_stats()
        
        # Add migration-specific stats
        migration_stats = {
            'database_stats': stats,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'database_path': str(self.db_file.absolute()),
        }
        
        # Write to file if requested
        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump(migration_stats, f, indent=2)
            logger.info(f"Migration statistics written to: {output_file}")
        
        return migration_stats

def main():
    """Command-line interface for the benchmark database maintenance utilities."""
    parser = argparse.ArgumentParser(description="Benchmark Database Maintenance Utilities")
    parser.add_argument("--db-path", default="./benchmark_db.duckdb",
                       help="Path to the DuckDB database")
    parser.add_argument("--check-integrity", action="store_true",
                       help="Check database integrity")
    parser.add_argument("--optimize-db", action="store_true",
                       help="Optimize the database (vacuum and analyze)")
    parser.add_argument("--backup", action="store_true",
                       help="Create a backup of the database")
    parser.add_argument("--backup-dir", default="./db_backups",
                       help="Directory to store backups")
    parser.add_argument("--backup-compress", action="store_true",
                       help="Compress database backups")
    parser.add_argument("--purge-backups", action="store_true",
                       help="Purge old database backups")
    parser.add_argument("--backup-retention", type=int, default=30,
                       help="Number of days to retain backups")
    parser.add_argument("--clean-json", action="store_true",
                       help="Clean up old JSON benchmark files")
    parser.add_argument("--json-dir", default="./benchmark_results",
                       help="Directory containing JSON files")
    parser.add_argument("--older-than", type=int, default=30,
                       help="Only clean files older than this many days")
    parser.add_argument("--migration-stats", action="store_true",
                       help="Generate migration statistics report")
    parser.add_argument("--output", default=None,
                       help="Output file for migration statistics")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    args = parser.parse_args()
    
    # Create maintenance tool
    maintenance = BenchmarkDBMaintenance(db_path=args.db_path, debug=args.debug)
    
    # Perform requested actions
    if args.check_integrity:
        success = maintenance.check_integrity()
        if not success:
            sys.exit(1)
    
    if args.optimize_db:
        success = maintenance.optimize_db()
        if not success:
            sys.exit(1)
    
    if args.backup:
        backup_path = maintenance.backup_db(
            backup_dir=args.backup_dir, 
            compress=args.backup_compress
        )
        if not backup_path:
            sys.exit(1)
    
    if args.purge_backups:
        maintenance.purge_old_backups(
            backup_dir=args.backup_dir,
            retention_days=args.backup_retention
        )
    
    if args.clean_json:
        maintenance.clean_json_files(
            json_dir=args.json_dir,
            older_than_days=args.older_than
        )
    
    if args.migration_stats:
        maintenance.generate_migration_stats(output_file=args.output)
    
    # If no action requested, print statistics
    if not any([
        args.check_integrity, args.optimize_db, args.backup,
        args.purge_backups, args.clean_json, args.migration_stats
    ]):
        stats = maintenance.get_database_stats()
        print("\nDatabase Statistics:")
        print(f"File Size: {stats['file_size_mb']:.2f} MB")
        print(f"Last Modified: {stats['last_modified']}")
        print(f"Total Rows: {stats['total_rows']}")
        print("\nTable Row Counts:")
        for table, count in stats['table_counts'].items():
            print(f"  - {table}: {count}")

if __name__ == "__main__":
    main()