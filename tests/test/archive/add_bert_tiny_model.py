#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add bert-tiny to the models table in the database

This script ensures bert-tiny is available in the models table
for benchmarking and result tracking.
"""

import os
import sys
import logging
import duckdb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def add_model():
    """Add bert-tiny to the models table."""
    # Connect to the database
    db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
    try:
        conn = duckdb.connect(db_path)
        logger.info(f"Connected to database: {db_path}")
        
        # Check if bert-tiny is already in the models table
        model_check_query = "SELECT COUNT(*) FROM models WHERE model_name = 'bert-tiny'"
        model_count = conn.execute(model_check_query).fetchone()[0]
        
        if model_count == 0:
            # Find the max model_id to use for the new model
            max_id_query = "SELECT MAX(model_id) FROM models"
            max_id_result = conn.execute(max_id_query).fetchone()[0]
            new_id = max_id_result + 1 if max_id_result is not None else 1
            
            # Add bert-tiny to the models table
            conn.execute("""
            INSERT INTO models (model_id, model_name, model_family, model_type, model_size, parameters_million, added_at)
            VALUES (?, 'bert-tiny', 'bert', 'text', 'tiny', 4.4, NOW())
            """, [new_id])
            conn.commit()
            logger.info(f"Added bert-tiny to models table with model_id={new_id}")
        else:
            model_id_query = "SELECT model_id FROM models WHERE model_name = 'bert-tiny'"
            model_id_result = conn.execute(model_id_query).fetchone()[0]
            logger.info(f"bert-tiny already exists in models table with model_id={model_id_result}")
        
        # Check if we need to add hardware CPU entry
        hw_check_query = "SELECT COUNT(*) FROM hardware_platforms WHERE hardware_type = 'cpu'"
        hw_count = conn.execute(hw_check_query).fetchone()[0]
        
        if hw_count == 0:
            # Find the max hardware_id to use for the new entry
            max_id_query = "SELECT MAX(hardware_id) FROM hardware_platforms"
            max_id_result = conn.execute(max_id_query).fetchone()[0]
            new_id = max_id_result + 1 if max_id_result is not None else 1
            
            # Add CPU to the hardware_platforms table
            conn.execute("""
            INSERT INTO hardware_platforms (hardware_id, hardware_type, device_name, compute_units, memory_capacity, 
                                           driver_version, supported_precisions, max_batch_size, detected_at)
            VALUES (?, 'cpu', 'Intel Xeon', 8, 16.0, 'N/A', 'fp32,fp16', 64, NOW())
            """, [new_id])
            conn.commit()
            logger.info(f"Added CPU to hardware_platforms table with hardware_id={new_id}")
        else:
            hw_id_query = "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = 'cpu'"
            hw_id_result = conn.execute(hw_id_query).fetchone()[0]
            logger.info(f"CPU already exists in hardware_platforms table with hardware_id={hw_id_result}")
        
        # Close the connection
        conn.close()
        logger.info("Database update completed")
        return True
    except Exception as e:
        logger.error(f"Error updating database: {e}")
        return False

def main():
    """Main function to add bert-tiny to the models table."""
    success = add_model()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())