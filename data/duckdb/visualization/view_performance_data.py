#!/usr/bin/env python3
"""
View performance data in database
"""

import sys
import duckdb
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def view_performance_data(db_path):
    """View performance data in benchmark database"""
    try:
        conn = duckdb.connect(db_path, read_only=True)
        
        # List updated performance data
        query = """
        SELECT 
            pr.result_id,
            m.model_name,
            m.model_family,
            hp.hardware_type,
            pr.batch_size,
            pr.average_latency_ms,
            pr.throughput_items_per_second,
            pr.memory_peak_mb,
            pr.test_timestamp
        FROM 
            performance_results pr
        JOIN 
            models m ON pr.model_id = m.model_id
        JOIN 
            hardware_platforms hp ON pr.hardware_id = hp.hardware_id
        WHERE
            pr.average_latency_ms > 0  -- Filter for real data with non-zero values
        ORDER BY
            pr.result_id
        """
        
        try:
            result = conn.execute(query).fetchdf()
            if len(result) > 0:
                logger.info(f"Found {len(result)} real benchmark results")
                print(result)
            else:
                logger.info("No real benchmark results found")
        except Exception as e:
            logger.error(f"Error querying performance data: {e}")
        
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error viewing database: {e}")
        return False

if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "./benchmark_db.duckdb"
    view_performance_data(db_path)