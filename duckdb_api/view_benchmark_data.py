#!/usr/bin/env python3
"""
View benchmark data in database to debug reports
"""

import sys
import duckdb
import logging
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def view_tables(db_path):
    """View table structure and sample data in benchmark database"""
    try:
        conn = duckdb.connect(db_path, read_only=True)
        
        # Get table list
        tables = conn.execute("SHOW TABLES").fetchall()
        logger.info(f"Found {len(tables)} tables in database {db_path}")
        
        for table in tables:
            table_name = table[0]
            logger.info(f"\n=== TABLE: {table_name} ===")
            
            # Get column info
            columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            logger.info(f"Columns: {', '.join([col[1] for col in columns])}")
            
            # Get row count
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            logger.info(f"Row count: {count}")
            
            # Get sample data
            if count > 0:
                sample = conn.execute(f"SELECT * FROM {table_name} LIMIT 5").fetchall()
                logger.info("Sample data (first 5 rows):")
                for row in sample:
                    logger.info(f"  {row}")
        
        # Specifically check for benchmark data
        if "performance_results" in [t[0] for t in tables]:
            # Query for benchmark data
            logger.info("\n=== QUERY BENCHMARK DATA ===")
            try:
                query = """
                SELECT 
                    m.model_name,
                    m.model_family,
                    hp.hardware_type,
                    pr.batch_size,
                    pr.average_latency_ms,
                    pr.throughput_items_per_second,
                    pr.memory_peak_mb
                FROM 
                    performance_results pr
                JOIN 
                    models m ON pr.model_id = m.model_id
                JOIN 
                    hardware_platforms hp ON pr.hardware_id = hp.hardware_id
                ORDER BY
                    m.model_family, hp.hardware_type
                """
                
                result = conn.execute(query).fetchdf()
                if len(result) > 0:
                    logger.info(f"Found {len(result)} benchmark results")
                    print(result.head(10))
                else:
                    logger.info("No benchmark results found")
            except Exception as e:
                logger.error(f"Error querying benchmark data: {e}")
        
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error viewing database: {e}")
        return False

if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "./benchmark_db.duckdb"
    view_tables(db_path)