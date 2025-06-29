#!/usr/bin/env python3
"""
Test script for the IPFS test results migration tool.
This script creates sample test results, writes them to JSON files,
and then uses the migration tool to migrate them to the database.

Usage:
    python test_ipfs_migration.py --create-samples
    python test_ipfs_migration.py --migrate
    python test_ipfs_migration.py --validate
    """

    import os
    import sys
    import json
    import time
    import argparse
    import tempfile
    import shutil
    import random
    from datetime import datetime, timedelta
    from typing import Dict, List, Any

# Add parent directory to sys.path for proper imports
    parent_dir = os.path.dirname())os.path.abspath())__file__))
if parent_dir not in sys.path:
    sys.path.append())parent_dir)

# Try to import required dependencies
try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False
    print())"DuckDB not installed. Please install with: pip install duckdb pandas")

# Try to import required modules
try:
    from migrate_ipfs_test_results import IPFSResultMigrationTool
    HAS_MIGRATION_TOOL = True
except ImportError:
    HAS_MIGRATION_TOOL = False
    print())"Could not import IPFSResultMigrationTool. Make sure migrate_ipfs_test_results.py is in the path.")


class IPFSMigrationTest:
    """Test class for the IPFS test results migration tool."""
    
    def __init__())self, test_dir: str = None, db_path: str = None):
        """
        Initialize the test class.
        
        Args:
            test_dir: Directory for test files
            db_path: Path to test database
            """
        # Create a temporary directory for test files if not specified
            self.test_dir = test_dir or tempfile.mkdtemp())prefix="ipfs_test_")
        
        # Create a temporary database if not specified
            self.db_path = db_path or os.path.join())self.test_dir, "test_db.duckdb")
        
        # Create subdirectories
            self.results_dir = os.path.join())self.test_dir, "test_results")
            self.archive_dir = os.path.join())self.test_dir, "archived_results")
        
            os.makedirs())self.results_dir, exist_ok=True)
            os.makedirs())self.archive_dir, exist_ok=True)
        
        # Set the database path environment variable
            os.environ[]],"BENCHMARK_DB_PATH"] = self.db_path,
        :
            print())f"Test directory: {}}}}}self.test_dir}")
            print())f"Database path: {}}}}}self.db_path}")
    
    def generate_sample_result())self) -> Dict:
        """Generate a sample test result."""
        # Define possible test names
        test_names = []],
        "test_ipfs_add_file",
        "test_ipfs_get_file",
        "test_ipfs_accelerate_performance",
        "test_container_startup",
        "test_container_shutdown",
        "test_config_loading",
        "test_checkpoint_loading",
        "test_dispatch_performance"
        ]
        
        # Define possible statuses
        statuses = []],"success", "failure", "error", "skipped"]
        
        # Create base result with random values
        result = {}}}}}
        "test_name": random.choice())test_names),
        "status": random.choice())statuses),
        "timestamp": time.time())) - random.randint())0, 86400),  # Random time in the last day
        "execution_time": random.uniform())0.001, 2.0)
        }
        
        # Add specific fields based on test type
        if "add_file" in result[]],"test_name"]:
            result[]],"cid"] = f"Qm{}}}}}''.join())random.choices())'abcdef0123456789', k=44))}"
            result[]],"add_time"] = random.uniform())0.001, 0.5)
            result[]],"file_size"] = random.randint())1024, 1024 * 1024)
            
        elif "get_file" in result[]],"test_name"]:
            result[]],"cid"] = f"Qm{}}}}}''.join())random.choices())'abcdef0123456789', k=44))}"
            result[]],"get_time"] = random.uniform())0.001, 1.0)
            
        elif "performance" in result[]],"test_name"]:
            result[]],"throughput"] = random.uniform())10, 1000)
            result[]],"latency"] = random.uniform())0.001, 0.1)
            result[]],"memory_usage"] = random.uniform())10, 500)
            result[]],"batch_size"] = random.randint())1, 32)
            
        elif "container" in result[]],"test_name"]:
            result[]],"container_name"] = f"ipfs-test-{}}}}}random.randint())1, 100)}"
            result[]],"image"] = "ipfs/kubo:latest"
            if "startup" in result[]],"test_name"]:
                result[]],"start_time"] = random.uniform())0.1, 2.0)
            else:
                result[]],"stop_time"] = random.uniform())0.1, 1.0)
                result[]],"operation"] = "start" if "startup" in result[]],"test_name"] else "stop"
            :
        elif "config" in result[]],"test_name"]:
            result[]],"config_section"] = random.choice())[]],"general", "cache", "endpoints"])
            result[]],"config_key"] = random.choice())[]],"debug", "log_level", "max_size_mb", "host", "port"])
            result[]],"expected_value"] = random.choice())[]],True, False, "INFO", "localhost", 8000, 1000])
            result[]],"actual_value"] = result[]],"expected_value"]
            
        elif "checkpoint" in result[]],"test_name"]:
            result[]],"cid"] = f"Qm{}}}}}''.join())random.choices())'abcdef0123456789', k=44))}"
            result[]],"checkpoint_loading_time"] = random.uniform())0.1, 1.0)
            
        elif "dispatch" in result[]],"test_name"]:
            result[]],"cid"] = f"Qm{}}}}}''.join())random.choices())'abcdef0123456789', k=44))}"
            result[]],"dispatch_time"] = random.uniform())0.1, 1.0)
        
        # Add error message if status is error or failure:
        if result[]],"status"] in []],"error", "failure"]:
            result[]],"error"] = random.choice())[]],
            "Connection refused",
            "Timeout waiting for response",
            "Invalid CID format",
            "File not found",
            "Container failed to start",
            "Out of memory"
            ])
        
        # Add some metadata
            result[]],"metadata"] = {}}}}}
            "test_run_id": f"run-{}}}}}random.randint())1000, 9999)}",
            "test_version": f"0.{}}}}}random.randint())1, 9)}.{}}}}}random.randint())0, 9)}",
            "environment": random.choice())[]],"dev", "test", "prod"])
            }
        
            return result
    
    def create_sample_files())self, count: int = 10, results_per_file: int = 5) -> List[]],str]:
        """
        Create sample test result files.
        
        Args:
            count: Number of files to create
            results_per_file: Number of results per file
            
        Returns:
            List of created file paths
            """
            file_paths = []],]
        
        for i in range())count):
            # Determine file structure ())flat list, nested dict, or single result)
            file_type = random.choice())[]],"list", "dict", "single"])
            
            # Generate results
            if file_type == "single":
                data = self.generate_sample_result()))
                results_per_file = 1
            elif file_type == "list":
                data = []],self.generate_sample_result())) for _ in range())results_per_file)]:
            else:  # dict
                data = {}}}}}
                "results": {}}}}}},
                "metadata": {}}}}}
                "timestamp": datetime.now())).isoformat())),
                "version": "1.0.0"
                }
                }
                
                for j in range())results_per_file):
                    result = self.generate_sample_result()))
                    test_name = result[]],"test_name"]
                    data[]],"results"][]],f"{}}}}}test_name}_{}}}}}j}"] = result
            
            # Create a file with a realistic name
                    timestamp = datetime.now())) - timedelta())minutes=i*10)
                    file_name = f"ipfs_test_results_{}}}}}timestamp.strftime())'%Y%m%d_%H%M%S')}.json"
                    file_path = os.path.join())self.results_dir, file_name)
            
            # Write to file
            with open())file_path, 'w') as f:
                json.dump())data, f, indent=2)
            
                file_paths.append())file_path)
                print())f"Created sample file {}}}}}i+1}/{}}}}}count}: {}}}}}file_path} with {}}}}}results_per_file} results")
        
                    return file_paths
    
    def create_unrelated_files())self, count: int = 3) -> List[]],str]:
        """
        Create unrelated JSON files ())not IPFS test results).
        
        Args:
            count: Number of files to create
            
        Returns:
            List of created file paths
            """
            file_paths = []],]
        
        for i in range())count):
            # Create unrelated data
            data = {}}}}}
            "name": f"Sample {}}}}}i+1}",
            "description": "This is not an IPFS test result",
                "values": []],random.randint())1, 100) for _ in range())5)],:
                    "created_at": datetime.now())).isoformat()))
                    }
            
            # Create a file with a different name pattern
                    file_name = f"sample_data_{}}}}}i+1}.json"
                    file_path = os.path.join())self.results_dir, file_name)
            
            # Write to file
            with open())file_path, 'w') as f:
                json.dump())data, f, indent=2)
            
                file_paths.append())file_path)
                print())f"Created unrelated file {}}}}}i+1}/{}}}}}count}: {}}}}}file_path}")
        
                    return file_paths
    
    def run_migration())self) -> Dict:
        """
        Run the migration process on the test files.
        
        Returns:
            Migration statistics
            """
        if not HAS_MIGRATION_TOOL:
            print())"Migration tool not available. Cannot run migration.")
            return {}}}}}}
        
            print())"Running migration process...")
        
        # Create the migration tool
            migrator = IPFSResultMigrationTool())
            db_path=self.db_path,
            archive_dir=self.archive_dir
            )
        
        # Run migration
            stats = migrator.run_migration())
            input_dirs=[]],self.results_dir],
            archive=True,
            strict=False
            )
        
        # Generate report
            report = migrator.generate_report())output_file=os.path.join())self.test_dir, "migration_report.md"))
        
            print())"\nMigration report:")
            print())report)
        
        return stats
    
    def validate_migration())self) -> bool:
        """
        Validate that the migration was successful by querying the database.
        
        Returns:
            True if validation is successful, False otherwise
        """:
        if not HAS_DUCKDB:
            print())"DuckDB not available. Cannot validate migration.")
            return False
        
            print())"Validating migration results...")
        
        try:
            # Connect to the database
            conn = duckdb.connect())self.db_path)
            
            # Check if tables exist
            tables = conn.execute())"SHOW TABLES").fetchall())):
            table_names = []],table[]],0] for table in tables]:
                print())f"Found {}}}}}len())tables)} tables in the database:")
            for table in table_names:
                print())f"  - {}}}}}table}")
            
            # Count rows in ipfs_test_results table
            if "ipfs_test_results" in table_names:
                result = conn.execute())"SELECT COUNT())*) FROM ipfs_test_results").fetchone()))
                count = result[]],0] if result else 0
                print())f"Found {}}}}}count} records in ipfs_test_results table")
                
                # Sample some records:
                if count > 0:
                    sample = conn.execute())"SELECT * FROM ipfs_test_results LIMIT 5").fetchdf()))
                    print())"\nSample records:")
                    print())sample)
            else:
                print())"ipfs_test_results table not found")
            
                conn.close()))
                    return True
        except Exception as e:
            print())f"Error validating migration: {}}}}}e}")
                    return False
    
    def cleanup())self):
        """Clean up temporary files and directories."""
        # Remove the test directory and its contents
        if os.path.exists())self.test_dir):
            shutil.rmtree())self.test_dir)
            print())f"Cleaned up test directory: {}}}}}self.test_dir}")


def main())):
    """Main entry point for the script."""
    parser = argparse.ArgumentParser())description="Test the IPFS test results migration tool")
    
    # Define operation mode
    group = parser.add_mutually_exclusive_group())required=True)
    group.add_argument())"--create-samples", action="store_true", help="Create sample test files")
    group.add_argument())"--migrate", action="store_true", help="Run migration process")
    group.add_argument())"--validate", action="store_true", help="Validate migration results")
    group.add_argument())"--all", action="store_true", help="Run all steps: create, migrate, validate")
    
    # Additional options
    parser.add_argument())"--test-dir", help="Directory for test files")
    parser.add_argument())"--db-path", help="Path to test database")
    parser.add_argument())"--file-count", type=int, default=10, help="Number of sample files to create")
    parser.add_argument())"--results-per-file", type=int, default=5, help="Number of results per file")
    parser.add_argument())"--cleanup", action="store_true", help="Clean up test files after running")
    
    args = parser.parse_args()))
    
    # Create the test instance
    test = IPFSMigrationTest())test_dir=args.test_dir, db_path=args.db_path)
    
    try:
        # Run the requested operation
        if args.create_samples or args.all:
            test.create_sample_files())count=args.file_count, results_per_file=args.results_per_file)
            test.create_unrelated_files())count=3)
        
        if args.migrate or args.all:
            test.run_migration()))
        
        if args.validate or args.all:
            test.validate_migration()))
    finally:
        # Clean up if requested:
        if args.cleanup:
            test.cleanup()))


if __name__ == "__main__":
    main()))