#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Mobile/Edge Database Schema

This script creates the database schema for storing mobile and edge device metrics
in the benchmark database. It adds tables for thermal metrics, battery impact,
mobile device information, and hardware-specific metrics.

Date: April 2025
"""

import os
import sys
import logging
import argparse
import datetime
from pathlib import Path
from typing import Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent))

def get_db_connection(db_path: str):
    """
    Get a connection to the benchmark database.
    
    Args:
        db_path: Path to the database
        
    Returns:
        Database connection
    """
    try:
        import duckdb
        return duckdb.connect(db_path)
    except ImportError:
        logger.error("DuckDB not installed. Please install it with 'pip install duckdb'")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        sys.exit(1)

def create_mobile_edge_schema(db_path: str, overwrite: bool = False) -> bool:
    """
    Create mobile/edge database schema.
    
    Args:
        db_path: Path to the database
        overwrite: Whether to overwrite existing tables
        
    Returns:
        Success status
    """
    logger.info(f"Creating mobile/edge database schema in {db_path}")
    
    conn = get_db_connection(db_path)
    
    try:
        # Create mobile device metrics table
        if overwrite:
            conn.execute("DROP TABLE IF EXISTS mobile_device_metrics")
        
        conn.execute("""
        CREATE TABLE IF NOT EXISTS mobile_device_metrics (
            id INTEGER PRIMARY KEY,
            result_id INTEGER,
            device_model VARCHAR,
            os_version VARCHAR,
            processor_type VARCHAR,
            battery_capacity_mah INTEGER,
            battery_temperature_celsius FLOAT,
            cpu_temperature_celsius FLOAT,
            gpu_temperature_celsius FLOAT,
            cpu_utilization_percent FLOAT,
            gpu_utilization_percent FLOAT,
            memory_utilization_percent FLOAT,
            network_utilization_percent FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (result_id) REFERENCES performance_results(id)
        )
        """)
        logger.info("Created mobile_device_metrics table")
        
        # Create thermal throttling events table
        if overwrite:
            conn.execute("DROP TABLE IF EXISTS thermal_throttling_events")
        
        conn.execute("""
        CREATE TABLE IF NOT EXISTS thermal_throttling_events (
            id INTEGER PRIMARY KEY,
            result_id INTEGER,
            start_time FLOAT,
            end_time FLOAT,
            duration_seconds FLOAT,
            max_temperature_celsius FLOAT,
            performance_impact_percent FLOAT,
            throttling_level INTEGER,
            zone_name VARCHAR,
            event_type VARCHAR,
            actions_taken JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (result_id) REFERENCES performance_results(id)
        )
        """)
        logger.info("Created thermal_throttling_events table")
        
        # Create battery impact results table
        if overwrite:
            conn.execute("DROP TABLE IF EXISTS battery_impact_results")
        
        conn.execute("""
        CREATE TABLE IF NOT EXISTS battery_impact_results (
            id INTEGER PRIMARY KEY,
            model_id INTEGER,
            hardware_id INTEGER,
            test_procedure VARCHAR,
            batch_size INTEGER,
            quantization_method VARCHAR,
            power_consumption_avg FLOAT,
            power_consumption_peak FLOAT,
            energy_per_inference FLOAT,
            battery_impact_percent_hour FLOAT,
            temperature_increase FLOAT,
            performance_per_watt FLOAT,
            battery_life_impact FLOAT,
            device_state VARCHAR,
            test_config JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (model_id) REFERENCES models(id),
            FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(id)
        )
        """)
        logger.info("Created battery_impact_results table")
        
        # Create battery impact time series table
        if overwrite:
            conn.execute("DROP TABLE IF EXISTS battery_impact_time_series")
        
        conn.execute("""
        CREATE TABLE IF NOT EXISTS battery_impact_time_series (
            id INTEGER PRIMARY KEY,
            result_id INTEGER,
            timestamp FLOAT,
            power_consumption FLOAT,
            temperature FLOAT,
            throughput FLOAT,
            memory_usage FLOAT,
            FOREIGN KEY (result_id) REFERENCES battery_impact_results(id)
        )
        """)
        logger.info("Created battery_impact_time_series table")
        
        # Create battery discharge rate table
        if overwrite:
            conn.execute("DROP TABLE IF EXISTS battery_discharge_rates")
        
        conn.execute("""
        CREATE TABLE IF NOT EXISTS battery_discharge_rates (
            id INTEGER PRIMARY KEY,
            result_id INTEGER,
            timestamp FLOAT,
            battery_level_percent FLOAT,
            discharge_rate_percent_per_hour FLOAT,
            estimated_remaining_time_minutes FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (result_id) REFERENCES battery_impact_results(id)
        )
        """)
        logger.info("Created battery_discharge_rates table")
        
        # Create MediaTek AI Engine metrics table
        if overwrite:
            conn.execute("DROP TABLE IF EXISTS mediatek_ai_metrics")
        
        conn.execute("""
        CREATE TABLE IF NOT EXISTS mediatek_ai_metrics (
            id INTEGER PRIMARY KEY,
            result_id INTEGER,
            device_model VARCHAR,
            chipset VARCHAR,
            apu_version VARCHAR,
            apu_utilization_percent FLOAT,
            npu_utilization_percent FLOAT,
            dsp_utilization_percent FLOAT,
            gpu_utilization_percent FLOAT,
            cpu_utilization_percent FLOAT,
            memory_bandwidth_gbps FLOAT,
            power_efficiency_inferences_per_watt FLOAT,
            thermal_zone_apu_temp_celsius FLOAT,
            thermal_zone_npu_temp_celsius FLOAT,
            supported_precisions JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (result_id) REFERENCES performance_results(id)
        )
        """)
        logger.info("Created mediatek_ai_metrics table")
        
        # Create Qualcomm AI Engine metrics table
        if overwrite:
            conn.execute("DROP TABLE IF EXISTS qualcomm_ai_metrics")
        
        conn.execute("""
        CREATE TABLE IF NOT EXISTS qualcomm_ai_metrics (
            id INTEGER PRIMARY KEY,
            result_id INTEGER,
            device_model VARCHAR,
            qnn_version VARCHAR,
            hexagon_dsp_version VARCHAR,
            npu_utilization_percent FLOAT,
            dsp_utilization_percent FLOAT,
            gpu_utilization_percent FLOAT,
            cpu_utilization_percent FLOAT,
            memory_bandwidth_gbps FLOAT,
            power_efficiency_inferences_per_watt FLOAT,
            thermal_zone_dsp_temp_celsius FLOAT,
            thermal_zone_npu_temp_celsius FLOAT,
            supported_precisions JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (result_id) REFERENCES performance_results(id)
        )
        """)
        logger.info("Created qualcomm_ai_metrics table")
        
        # Create Samsung NPU metrics table
        if overwrite:
            conn.execute("DROP TABLE IF EXISTS samsung_npu_metrics")
        
        conn.execute("""
        CREATE TABLE IF NOT EXISTS samsung_npu_metrics (
            id INTEGER PRIMARY KEY,
            result_id INTEGER,
            device_model VARCHAR,
            exynos_version VARCHAR,
            npu_utilization_percent FLOAT,
            dsp_utilization_percent FLOAT,
            gpu_utilization_percent FLOAT,
            cpu_utilization_percent FLOAT,
            memory_bandwidth_gbps FLOAT,
            power_efficiency_inferences_per_watt FLOAT,
            thermal_zone_npu_temp_celsius FLOAT,
            supported_precisions JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (result_id) REFERENCES performance_results(id)
        )
        """)
        logger.info("Created samsung_npu_metrics table")
        
        # Add additional columns to hardware_platforms table
        try:
            conn.execute("""
            ALTER TABLE hardware_platforms 
            ADD COLUMN IF NOT EXISTS mobile_device BOOLEAN DEFAULT FALSE,
            ADD COLUMN IF NOT EXISTS edge_device BOOLEAN DEFAULT FALSE,
            ADD COLUMN IF NOT EXISTS chipset_name VARCHAR,
            ADD COLUMN IF NOT EXISTS chipset_details JSON,
            ADD COLUMN IF NOT EXISTS mobile_optimized BOOLEAN DEFAULT FALSE,
            ADD COLUMN IF NOT EXISTS power_efficient BOOLEAN DEFAULT FALSE
            """)
            logger.info("Added mobile/edge columns to hardware_platforms table")
        except Exception as e:
            logger.warning(f"Error adding columns to hardware_platforms table: {e}")
        
        # Add supported vendors
        try:
            conn.execute("""
            INSERT OR IGNORE INTO hardware_vendors (vendor_name, vendor_type, website, api_integration)
            VALUES 
                ('Qualcomm', 'Mobile NPU', 'https://www.qualcomm.com', TRUE),
                ('MediaTek', 'Mobile NPU', 'https://www.mediatek.com', TRUE),
                ('Samsung Exynos', 'Mobile NPU', 'https://www.samsung.com/exynos/', TRUE)
            """)
            logger.info("Added mobile/edge vendors")
        except Exception as e:
            logger.warning(f"Error adding mobile/edge vendors: {e}")
        
        # Create view for mobile performance comparison
        if overwrite:
            conn.execute("DROP VIEW IF EXISTS mobile_performance_comparison")
        
        conn.execute("""
        CREATE VIEW IF NOT EXISTS mobile_performance_comparison AS
        SELECT
            m.model_name,
            m.model_family,
            h.hardware_type,
            h.hardware_vendor,
            p.batch_size,
            p.sequence_length,
            p.precision,
            p.throughput_items_per_second,
            p.latency_ms,
            b.power_consumption_avg,
            b.energy_per_inference,
            b.battery_impact_percent_hour,
            b.performance_per_watt,
            b.temperature_increase
        FROM
            performance_results p
        JOIN
            models m ON p.model_id = m.id
        JOIN
            hardware_platforms h ON p.hardware_id = h.id
        LEFT JOIN
            battery_impact_results b ON p.id = b.result_id
        WHERE
            h.mobile_device = TRUE OR h.edge_device = TRUE
        """)
        logger.info("Created mobile_performance_comparison view")
        
        logger.info("Mobile/edge database schema created successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error creating mobile/edge database schema: {e}")
        return False
    
    finally:
        conn.close()

def add_sample_data(db_path: str) -> bool:
    """
    Add sample data to mobile/edge tables.
    
    Args:
        db_path: Path to the database
        
    Returns:
        Success status
    """
    logger.info(f"Adding sample data to mobile/edge tables in {db_path}")
    
    conn = get_db_connection(db_path)
    
    try:
        # Add sample hardware platforms
        conn.execute("""
        INSERT OR IGNORE INTO hardware_platforms 
            (hardware_type, hardware_vendor, hardware_name, mobile_device, edge_device, chipset_name, mobile_optimized, power_efficient)
        VALUES 
            ('qualcomm', 'Qualcomm', 'Snapdragon 8 Gen 3', TRUE, FALSE, 'Snapdragon 8 Gen 3', TRUE, TRUE),
            ('mediatek', 'MediaTek', 'Dimensity 9300', TRUE, FALSE, 'Dimensity 9300', TRUE, TRUE),
            ('samsung', 'Samsung Exynos', 'Exynos 2400', TRUE, FALSE, 'Exynos 2400', TRUE, TRUE)
        """)
        logger.info("Added sample hardware platforms")
        
        # Get the IDs of the hardware platforms and a model
        result = conn.execute("""
        SELECT id FROM hardware_platforms WHERE hardware_name = 'Dimensity 9300'
        """).fetchone()
        if not result:
            logger.warning("Could not find Dimensity 9300 hardware platform")
            return False
        mediatek_id = result[0]
        
        result = conn.execute("""
        SELECT id FROM hardware_platforms WHERE hardware_name = 'Snapdragon 8 Gen 3'
        """).fetchone()
        if not result:
            logger.warning("Could not find Snapdragon 8 Gen 3 hardware platform")
            return False
        qualcomm_id = result[0]
        
        result = conn.execute("""
        SELECT id FROM models WHERE model_name = 'bert-base-uncased' LIMIT 1
        """).fetchone()
        if not result:
            logger.warning("Could not find bert-base-uncased model")
            return False
        model_id = result[0]
        
        # Add sample battery impact results
        mediatek_result_id = conn.execute("""
        INSERT INTO battery_impact_results 
            (model_id, hardware_id, test_procedure, batch_size, quantization_method, 
             power_consumption_avg, power_consumption_peak, energy_per_inference, 
             battery_impact_percent_hour, temperature_increase, performance_per_watt, 
             battery_life_impact, device_state, test_config, created_at)
        VALUES 
            (?, ?, 'continuous_inference', 1, 'INT8', 
             450.0, 650.0, 4.5, 
             3.2, 4.5, 85.0, 
             4.5, 'screen_on_active', '{"duration_seconds": 300}', ?)
        RETURNING id
        """, [model_id, mediatek_id, datetime.datetime.now().isoformat()]).fetchone()[0]
        
        qualcomm_result_id = conn.execute("""
        INSERT INTO battery_impact_results 
            (model_id, hardware_id, test_procedure, batch_size, quantization_method, 
             power_consumption_avg, power_consumption_peak, energy_per_inference, 
             battery_impact_percent_hour, temperature_increase, performance_per_watt, 
             battery_life_impact, device_state, test_config, created_at)
        VALUES 
            (?, ?, 'continuous_inference', 1, 'INT8', 
             420.0, 600.0, 4.2, 
             3.0, 4.2, 90.0, 
             4.2, 'screen_on_active', '{"duration_seconds": 300}', ?)
        RETURNING id
        """, [model_id, qualcomm_id, datetime.datetime.now().isoformat()]).fetchone()[0]
        
        # Add sample time series data
        for i in range(10):
            timestamp = time.time() + i * 30
            power = 450.0 + i * 10
            temp = 35.0 + i * 0.5
            
            conn.execute("""
            INSERT INTO battery_impact_time_series 
                (result_id, timestamp, power_consumption, temperature, throughput, memory_usage)
            VALUES 
                (?, ?, ?, ?, ?, ?)
            """, [mediatek_result_id, timestamp, power, temp, 100.0, 250.0])
            
            conn.execute("""
            INSERT INTO battery_impact_time_series 
                (result_id, timestamp, power_consumption, temperature, throughput, memory_usage)
            VALUES 
                (?, ?, ?, ?, ?, ?)
            """, [qualcomm_result_id, timestamp, power - 30, temp - 0.3, 105.0, 240.0])
        
        # Add sample MediaTek AI metrics
        conn.execute("""
        INSERT INTO mediatek_ai_metrics 
            (result_id, device_model, chipset, apu_version, 
             apu_utilization_percent, npu_utilization_percent, dsp_utilization_percent, 
             gpu_utilization_percent, cpu_utilization_percent, memory_bandwidth_gbps, 
             power_efficiency_inferences_per_watt, thermal_zone_apu_temp_celsius, 
             thermal_zone_npu_temp_celsius, supported_precisions, created_at)
        VALUES 
            (?, 'MediaTek Test Device', 'Dimensity 9300', '2.0', 
             65.0, 70.0, 40.0, 
             30.0, 25.0, 12.5, 
             22.0, 65.0, 
             62.0, '["FP32", "FP16", "INT8", "INT4"]', ?)
        """, [mediatek_result_id, datetime.datetime.now().isoformat()])
        
        # Add sample Qualcomm AI metrics
        conn.execute("""
        INSERT INTO qualcomm_ai_metrics 
            (result_id, device_model, qnn_version, hexagon_dsp_version, 
             npu_utilization_percent, dsp_utilization_percent, 
             gpu_utilization_percent, cpu_utilization_percent, memory_bandwidth_gbps, 
             power_efficiency_inferences_per_watt, thermal_zone_dsp_temp_celsius, 
             thermal_zone_npu_temp_celsius, supported_precisions, created_at)
        VALUES 
            (?, 'Qualcomm Test Device', '2.11', 'V68', 
             68.0, 45.0, 
             28.0, 22.0, 14.0, 
             24.0, 62.0, 
             60.0, '["FP32", "FP16", "INT8", "INT4"]', ?)
        """, [qualcomm_result_id, datetime.datetime.now().isoformat()])
        
        # Add sample thermal throttling events
        conn.execute("""
        INSERT INTO thermal_throttling_events 
            (result_id, start_time, end_time, duration_seconds, 
             max_temperature_celsius, performance_impact_percent, throttling_level, 
             zone_name, event_type, actions_taken, created_at)
        VALUES 
            (?, ?, ?, ?, 
             ?, ?, ?, 
             ?, ?, ?, ?)
        """, [
            mediatek_result_id, 
            time.time(), 
            time.time() + 120,
            120.0,
            68.0,
            25.0,
            2,
            "apu",
            "WARNING",
            json.dumps(["Apply moderate throttling (25% performance reduction)"]),
            datetime.datetime.now().isoformat()
        ])
        
        logger.info("Added sample data to mobile/edge tables")
        return True
    
    except Exception as e:
        logger.error(f"Error adding sample data to mobile/edge tables: {e}")
        return False
    
    finally:
        conn.close()

def verify_schema(db_path: str) -> bool:
    """
    Verify that mobile/edge schema was created correctly.
    
    Args:
        db_path: Path to the database
        
    Returns:
        Success status
    """
    logger.info(f"Verifying mobile/edge database schema in {db_path}")
    
    conn = get_db_connection(db_path)
    
    try:
        # Check if tables exist
        tables = [
            "mobile_device_metrics",
            "thermal_throttling_events",
            "battery_impact_results",
            "battery_impact_time_series",
            "battery_discharge_rates",
            "mediatek_ai_metrics",
            "qualcomm_ai_metrics",
            "samsung_npu_metrics"
        ]
        
        for table in tables:
            result = conn.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'").fetchone()
            if not result:
                logger.error(f"Table {table} does not exist")
                return False
            logger.info(f"Verified table: {table}")
        
        # Check if view exists
        result = conn.execute("SELECT name FROM sqlite_master WHERE type='view' AND name='mobile_performance_comparison'").fetchone()
        if not result:
            logger.error("View mobile_performance_comparison does not exist")
            return False
        logger.info("Verified view: mobile_performance_comparison")
        
        # Check if hardware_platforms table has the new columns
        result = conn.execute("PRAGMA table_info(hardware_platforms)").fetchall()
        columns = [r[1] for r in result]
        
        required_columns = [
            "mobile_device",
            "edge_device",
            "chipset_name",
            "chipset_details",
            "mobile_optimized",
            "power_efficient"
        ]
        
        for column in required_columns:
            if column not in columns:
                logger.error(f"Column {column} does not exist in hardware_platforms table")
                return False
            logger.info(f"Verified column: hardware_platforms.{column}")
        
        logger.info("Mobile/edge database schema verification successful")
        return True
    
    except Exception as e:
        logger.error(f"Error verifying mobile/edge database schema: {e}")
        return False
    
    finally:
        conn.close()

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Create Mobile/Edge Database Schema')
    parser.add_argument('--db-path', help='Path to the benchmark database', default='./benchmark_db.duckdb')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing tables')
    parser.add_argument('--add-sample-data', action='store_true', help='Add sample data to tables')
    parser.add_argument('--verify', action='store_true', help='Verify schema after creation')
    
    args = parser.parse_args()
    
    # Create schema
    success = create_mobile_edge_schema(args.db_path, args.overwrite)
    
    if not success:
        logger.error("Failed to create mobile/edge database schema")
        return 1
    
    # Add sample data if requested
    if args.add_sample_data and success:
        import time
        import json
        success = add_sample_data(args.db_path)
        
        if not success:
            logger.error("Failed to add sample data to mobile/edge tables")
            return 1
    
    # Verify schema if requested
    if args.verify and success:
        success = verify_schema(args.db_path)
        
        if not success:
            logger.error("Mobile/edge database schema verification failed")
            return 1
    
    logger.info("Mobile/edge database schema creation completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())