#!/usr/bin/env python
"""
Migration utility to migrate existing DuckDB databases to IPFS storage.

This script helps migrate benchmark databases to IPFS-backed storage,
creating backups and tracking CIDs for version control.

Usage:
    # Migrate a single database
    python migrate_to_ipfs.py --db benchmark_db.duckdb
    
    # Migrate multiple databases
    python migrate_to_ipfs.py --db db1.duckdb --db db2.duckdb
    
    # Migrate without backups (not recommended)
    python migrate_to_ipfs.py --db benchmark_db.duckdb --no-backup
    
    # Use custom IPFS configuration
    python migrate_to_ipfs.py --db benchmark_db.duckdb --config ipfs_config.json
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

try:
    from data.duckdb.ipfs_integration import (
        IPFSConfig,
        IPFSDBMigration,
        load_config_from_file
    )
except ImportError as e:
    print(f"Error: Could not import IPFS integration modules: {e}")
    print("Make sure ipfs_datasets_py and ipfs_kit_py submodules are initialized.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def migrate_databases(
    db_paths: List[str],
    config_path: Optional[str] = None,
    create_backups: bool = True,
    verify: bool = True
) -> bool:
    """
    Migrate databases to IPFS storage.
    
    Args:
        db_paths: List of database paths to migrate
        config_path: Path to IPFS configuration file
        create_backups: Whether to create backups before migration
        verify: Whether to verify migration success
        
    Returns:
        True if all migrations successful, False otherwise
    """
    logger.info(f"Starting migration of {len(db_paths)} database(s)")
    
    # Load configuration
    if config_path:
        logger.info(f"Loading IPFS configuration from {config_path}")
        config = load_config_from_file(config_path)
    else:
        logger.info("Using default IPFS configuration")
        config = IPFSConfig(enable_ipfs_storage=True)
    
    # Check if IPFS is available
    from data.duckdb.ipfs_integration import IPFSStorage
    storage = IPFSStorage(config)
    if not storage.is_available():
        logger.error("IPFS storage is not available. Please ensure:")
        logger.error("1. ipfs_datasets_py and ipfs_kit_py submodules are initialized")
        logger.error("2. IPFS daemon is running (if using local IPFS)")
        logger.error("3. Configuration is correct")
        return False
    
    logger.info("IPFS storage is available")
    
    # Create migration utility
    migration = IPFSDBMigration(config=config)
    
    # Migrate each database
    results = migration.batch_migrate(db_paths, create_backups=create_backups)
    
    # Report results
    successful = []
    failed = []
    
    for db_path, cid in results.items():
        if cid:
            successful.append((db_path, cid))
            logger.info(f"✓ {db_path} -> IPFS CID: {cid}")
        else:
            failed.append(db_path)
            logger.error(f"✗ {db_path} -> Migration failed")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("MIGRATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total databases: {len(db_paths)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    
    if successful:
        logger.info("\nSuccessfully migrated:")
        for db_path, cid in successful:
            logger.info(f"  {db_path}")
            logger.info(f"    IPFS CID: {cid}")
    
    if failed:
        logger.warning("\nFailed to migrate:")
        for db_path in failed:
            logger.warning(f"  {db_path}")
    
    # Verification
    if verify and successful:
        logger.info("\nVerifying migrations...")
        from data.duckdb.ipfs_integration import IPFSDBBackend
        
        for db_path, cid in successful:
            try:
                backend = IPFSDBBackend(db_path, config=config)
                current_cid = backend.get_current_cid()
                if current_cid == cid:
                    logger.info(f"✓ Verified: {db_path}")
                else:
                    logger.warning(f"! CID mismatch for {db_path}: {current_cid} != {cid}")
            except Exception as e:
                logger.error(f"✗ Verification failed for {db_path}: {e}")
    
    logger.info("="*60)
    
    return len(failed) == 0


def main():
    """Main entry point for migration script."""
    parser = argparse.ArgumentParser(
        description="Migrate DuckDB databases to IPFS storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate single database
  python migrate_to_ipfs.py --db benchmark_db.duckdb
  
  # Migrate multiple databases
  python migrate_to_ipfs.py --db db1.duckdb --db db2.duckdb
  
  # Use custom configuration
  python migrate_to_ipfs.py --db benchmark_db.duckdb --config ipfs.json
  
  # Skip backups (not recommended)
  python migrate_to_ipfs.py --db benchmark_db.duckdb --no-backup
        """
    )
    
    parser.add_argument(
        '--db',
        dest='databases',
        action='append',
        required=True,
        help='Database file to migrate (can be specified multiple times)'
    )
    
    parser.add_argument(
        '--config',
        help='Path to IPFS configuration file (JSON)'
    )
    
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backups before migration'
    )
    
    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='Skip verification after migration'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate database files
    db_paths = []
    for db in args.databases:
        db_path = Path(db)
        if not db_path.exists():
            logger.error(f"Database file not found: {db}")
            sys.exit(1)
        db_paths.append(db)
    
    # Run migration
    try:
        success = migrate_databases(
            db_paths=db_paths,
            config_path=args.config,
            create_backups=not args.no_backup,
            verify=not args.no_verify
        )
        
        if success:
            logger.info("All migrations completed successfully!")
            sys.exit(0)
        else:
            logger.error("Some migrations failed. Check logs for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("\nMigration interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Migration failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
