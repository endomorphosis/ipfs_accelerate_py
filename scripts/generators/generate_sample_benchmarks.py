#!/usr/bin/env python3
"""
Generate Sample Benchmark Data

This script populates the benchmark database with sample data for all 13 key models
across all 8 hardware platforms to test database integration.
"""

import os
import sys
import logging
import random
import datetime
import duckdb
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define model types and hardware endpoints
MODEL_TYPES = {
    "bert": {"family": "bert", "modality": "text", "name": "bert-base-uncased"},
    "t5": {"family": "t5", "modality": "text", "name": "t5-small"},
    "llama": {"family": "llama", "modality": "text", "name": "llama-7b"},
    "clip": {"family": "clip", "modality": "multimodal", "name": "clip-vit-base-patch32"},
    "vit": {"family": "vit", "modality": "vision", "name": "vit-base-patch16-224"},
    "clap": {"family": "clap", "modality": "audio", "name": "clap-htsat-base"},
    "wav2vec2": {"family": "wav2vec2", "modality": "audio", "name": "wav2vec2-base"},
    "whisper": {"family": "whisper", "modality": "audio", "name": "whisper-tiny"},
    "llava": {"family": "llava", "modality": "multimodal", "name": "llava-7b"},
    "llava-next": {"family": "llava-next", "modality": "multimodal", "name": "llava-next-7b"},
    "xclip": {"family": "xclip", "modality": "vision", "name": "xclip-base"},
    "qwen2": {"family": "qwen2", "modality": "text", "name": "qwen2-7b"},
    "detr": {"family": "detr", "modality": "vision", "name": "detr-resnet-50"}
}

HARDWARE_ENDPOINTS = [
    "cpu", "cuda", "rocm", "mps", "openvino", "qnn", "webnn", "webgpu"
]

# Predefined performance characteristics to make the data realistic
MODEL_CHARACTERISTICS = {
    # Text models
    "bert": {"latency": {"cpu": 25.5, "cuda": 8.2, "rocm": 9.4, "mps": 15.6, "openvino": 12.8, "qnn": 18.3, "webnn": 19.7, "webgpu": 14.5},
             "throughput": {"cpu": 45.8, "cuda": 215.3, "rocm": 187.9, "mps": 112.4, "openvino": 154.2, "qnn": 98.6, "webnn": 87.3, "webgpu": 128.9},
             "memory": 850},
    "t5": {"latency": {"cpu": 32.7, "cuda": 10.3, "rocm": 11.9, "mps": 19.2, "openvino": 16.5, "qnn": 22.8, "webnn": 24.5, "webgpu": 18.2},
           "throughput": {"cpu": 35.4, "cuda": 175.8, "rocm": 158.6, "mps": 98.3, "openvino": 124.5, "qnn": 82.1, "webnn": 76.4, "webgpu": 102.3},
           "memory": 950},
    "llama": {"latency": {"cpu": 85.3, "cuda": 18.7, "rocm": 21.3, "mps": 41.8, "openvino": 37.2, "qnn": 76.5, "webnn": 98.7, "webgpu": 78.3},
             "throughput": {"cpu": 12.6, "cuda": 87.5, "rocm": 74.3, "mps": 32.7, "openvino": 42.9, "qnn": 19.4, "webnn": 15.6, "webgpu": 21.8},
             "memory": 4250},
    "qwen2": {"latency": {"cpu": 92.4, "cuda": 22.1, "rocm": 24.3, "mps": 47.6, "openvino": 39.8, "qnn": 81.7, "webnn": 104.3, "webgpu": 83.6},
             "throughput": {"cpu": 9.8, "cuda": 72.5, "rocm": 63.2, "mps": 27.9, "openvino": 38.4, "qnn": 16.2, "webnn": 12.1, "webgpu": 18.5},
             "memory": 4750},
    
    # Vision models 
    "vit": {"latency": {"cpu": 28.6, "cuda": 7.8, "rocm": 8.9, "mps": 17.3, "openvino": 14.2, "qnn": 21.6, "webnn": 22.8, "webgpu": 16.4},
            "throughput": {"cpu": 42.5, "cuda": 198.3, "rocm": 175.6, "mps": 104.8, "openvino": 142.7, "qnn": 89.4, "webnn": 82.6, "webgpu": 115.2},
            "memory": 920},
    "xclip": {"latency": {"cpu": 38.2, "cuda": 9.5, "rocm": 10.8, "mps": 21.6, "openvino": 19.3, "qnn": 32.8, "webnn": 36.5, "webgpu": 24.7},
             "throughput": {"cpu": 28.4, "cuda": 154.3, "rocm": 137.8, "mps": 87.2, "openvino": 112.5, "qnn": 53.6, "webnn": 48.9, "webgpu": 84.3},
             "memory": 1250},
    "detr": {"latency": {"cpu": 42.5, "cuda": 11.2, "rocm": 12.9, "mps": 24.8, "openvino": 21.4, "qnn": 34.6, "webnn": 39.8, "webgpu": 27.5},
            "throughput": {"cpu": 24.6, "cuda": 132.8, "rocm": 118.4, "mps": 74.9, "openvino": 98.5, "qnn": 47.3, "webnn": 42.1, "webgpu": 68.7},
            "memory": 1180},
    
    # Audio models
    "whisper": {"latency": {"cpu": 35.4, "cuda": 9.2, "rocm": 10.5, "mps": 19.8, "openvino": 16.9, "qnn": 28.3, "webnn": 36.7, "webgpu": 19.2},
                "throughput": {"cpu": 35.8, "cuda": 187.5, "rocm": 162.3, "mps": 94.7, "openvino": 118.3, "qnn": 68.2, "webnn": 58.9, "webgpu": 102.8},
                "memory": 1050},
    "wav2vec2": {"latency": {"cpu": 31.2, "cuda": 8.4, "rocm": 9.6, "mps": 18.1, "openvino": 15.4, "qnn": 25.7, "webnn": 32.5, "webgpu": 17.8},
                 "throughput": {"cpu": 38.7, "cuda": 198.6, "rocm": 176.4, "mps": 98.5, "openvino": 124.6, "qnn": 71.8, "webnn": 63.2, "webgpu": 108.5},
                 "memory": 980},
    "clap": {"latency": {"cpu": 38.6, "cuda": 9.8, "rocm": 11.2, "mps": 21.3, "openvino": 18.1, "qnn": 30.4, "webnn": 38.2, "webgpu": 20.5},
             "throughput": {"cpu": 32.4, "cuda": 178.3, "rocm": 154.2, "mps": 88.7, "openvino": 112.3, "qnn": 64.5, "webnn": 54.8, "webgpu": 96.7},
             "memory": 1120},
    
    # Multimodal models
    "clip": {"latency": {"cpu": 45.2, "cuda": 12.5, "rocm": 14.2, "mps": 24.7, "openvino": 21.3, "qnn": 36.8, "webnn": 42.5, "webgpu": 28.4},
             "throughput": {"cpu": 26.8, "cuda": 148.5, "rocm": 129.3, "mps": 78.6, "openvino": 96.4, "qnn": 52.1, "webnn": 45.3, "webgpu": 78.9},
             "memory": 1350},
    "llava": {"latency": {"cpu": 124.5, "cuda": 32.8, "rocm": 37.5, "mps": 79.3, "openvino": 65.8, "qnn": 112.3, "webnn": 145.2, "webgpu": 98.5},
              "throughput": {"cpu": 8.2, "cuda": 42.7, "rocm": 36.5, "mps": 16.3, "openvino": 22.4, "qnn": 10.2, "webnn": 7.8, "webgpu": 12.5},
              "memory": 7850},
    "llava-next": {"latency": {"cpu": 132.7, "cuda": 35.4, "rocm": 40.2, "mps": 84.5, "openvino": 70.3, "qnn": 118.6, "webnn": 154.3, "webgpu": 104.2},
                   "throughput": {"cpu": 7.4, "cuda": 38.5, "rocm": 32.8, "mps": 14.7, "openvino": 19.8, "qnn": 9.1, "webnn": 6.8, "webgpu": 11.2},
                   "memory": 8250},
}

def connect_to_db(db_path):
    """Connect to the DuckDB database"""
    try:
        # Try different connection methods for compatibility
        try:
            conn = duckdb.connect(db_path)
        except TypeError:
            conn = duckdb.connect(database=db_path)
        
        logger.info(f"Connected to database: {db_path}")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        return None

def ensure_model_exists(conn, model_name, model_family, modality):
    """Ensure a model exists in the database"""
    try:
        # Check if model exists
        result = conn.execute(
            "SELECT model_id FROM models WHERE model_name = ?", 
            [model_name]
        ).fetchone()
        
        if result:
            logger.info(f"Model exists: {model_name} (ID: {result[0]})")
            return result[0]
        
        # Get next ID
        max_id_result = conn.execute("SELECT COALESCE(MAX(model_id), 0) + 1 FROM models").fetchone()
        next_id = max_id_result[0] if max_id_result else 1
        
        # Insert new model matching the actual schema
        conn.execute(
            """
            INSERT INTO models (model_id, model_name, model_family, model_type, model_size, 
                             parameters_million, added_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [next_id, model_name, model_family, modality, "base", 
             random.uniform(0.5, 20.0), datetime.datetime.now()]
        )
        
        # Get inserted ID
        result = conn.execute(
            "SELECT model_id FROM models WHERE model_name = ?", 
            [model_name]
        ).fetchone()
        
        logger.info(f"Created model: {model_name} (ID: {result[0]})")
        return result[0]
    except Exception as e:
        logger.error(f"Error ensuring model exists ({model_name}): {e}")
        return None

def ensure_hardware_exists(conn, hardware_type):
    """Ensure hardware platform exists in the database"""
    try:
        # Check if hardware exists
        result = conn.execute(
            "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = ?", 
            [hardware_type]
        ).fetchone()
        
        if result:
            logger.info(f"Hardware exists: {hardware_type} (ID: {result[0]})")
            return result[0]
        
        # Get next ID
        max_id_result = conn.execute("SELECT COALESCE(MAX(hardware_id), 0) + 1 FROM hardware_platforms").fetchone()
        next_id = max_id_result[0] if max_id_result else 1
        
        # Insert new hardware matching the actual schema
        conn.execute(
            """
            INSERT INTO hardware_platforms (hardware_id, hardware_type, device_name, compute_units, 
                                          memory_capacity, driver_version, supported_precisions, 
                                          max_batch_size, detected_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [next_id, hardware_type, f"{hardware_type.upper()} Device", 
            16, 16.0, "latest", "fp32,fp16,int8", 32, datetime.datetime.now()]
        )
        
        # Get inserted ID
        result = conn.execute(
            "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = ?", 
            [hardware_type]
        ).fetchone()
        
        logger.info(f"Created hardware: {hardware_type} (ID: {result[0]})")
        return result[0]
    except Exception as e:
        logger.error(f"Error ensuring hardware exists ({hardware_type}): {e}")
        return None

def add_performance_result(conn, model_id, hardware_id, batch_size, model_type, hardware_type):
    """Add a performance result to the database"""
    try:
        # Get performance characteristics for this model and hardware
        if model_type not in MODEL_CHARACTERISTICS:
            model_type = random.choice(list(MODEL_CHARACTERISTICS.keys()))
            
        model_char = MODEL_CHARACTERISTICS[model_type]
        
        # Randomize a bit to make it interesting
        latency_multiplier = random.uniform(0.95, 1.05)
        throughput_multiplier = random.uniform(0.95, 1.05)
        memory_multiplier = random.uniform(0.98, 1.02)
        
        # Get base metrics with fallbacks
        latency_base = model_char["latency"].get(hardware_type, 30.0)
        throughput_base = model_char["throughput"].get(hardware_type, 100.0)
        memory_base = model_char.get("memory", 1000.0)
        
        # Adjust for batch size
        latency = latency_base * (1 + 0.1 * batch_size) * latency_multiplier
        throughput = throughput_base * batch_size * throughput_multiplier
        memory = memory_base * (1 + 0.05 * batch_size) * memory_multiplier
        
        # Create random timestamp in last 30 days
        days_ago = random.randint(0, 30)
        timestamp = datetime.datetime.now() - datetime.timedelta(days=days_ago)
        
        # Get max ID for auto-increment
        max_id_result = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM performance_results").fetchone()
        next_id = max_id_result[0] if max_id_result else 1
        
        # Insert result
        conn.execute(
            """
            INSERT INTO performance_results (
                id, model_id, hardware_id, batch_size, sequence_length, 
                average_latency_ms, throughput_items_per_second, memory_peak_mb, test_timestamp
            ) VALUES (?, ?, ?, ?, 128, ?, ?, ?, ?)
            """,
            [next_id, model_id, hardware_id, batch_size, latency, throughput, memory, timestamp]
        )
        
        logger.info(f"Added performance result: {model_type} on {hardware_type}, batch_size={batch_size}")
        return True
    except Exception as e:
        logger.error(f"Error adding performance result: {e}")
        return False

def main():
    """Main function to generate sample benchmark data"""
    # Set up argument parser
    import argparse
    parser = argparse.ArgumentParser(description="Generate sample benchmark data for testing")
    parser.add_argument("--db", type=str, default="./benchmark_db.duckdb", help="Path to DuckDB database")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8", help="Batch sizes to simulate")
    args = parser.parse_args()
    
    # Parse batch sizes
    batch_sizes = [int(size) for size in args.batch_sizes.split(",")]
    
    # Connect to database
    conn = connect_to_db(args.db)
    if not conn:
        logger.error("Failed to connect to database. Exiting.")
        return 1
    
    try:
        # Each operation should be in its own transaction to avoid cascading errors
        # We'll commit after each successful operation
        
        # Process all models and hardware platforms
        for model_key, model_info in MODEL_TYPES.items():
            # Ensure model exists
            model_id = ensure_model_exists(
                conn, 
                model_info["name"], 
                model_info["family"], 
                model_info["modality"]
            )
            
            if not model_id:
                logger.warning(f"Skipping model {model_key}")
                continue
            
            # Process each hardware platform
            for hardware_type in HARDWARE_ENDPOINTS:
                # Ensure hardware exists
                hardware_id = ensure_hardware_exists(conn, hardware_type)
                
                if not hardware_id:
                    logger.warning(f"Skipping hardware {hardware_type}")
                    continue
                
                # Add performance results for different batch sizes
                for batch_size in batch_sizes:
                    add_performance_result(
                        conn, 
                        model_id, 
                        hardware_id, 
                        batch_size, 
                        model_key, 
                        hardware_type
                    )
        
        logger.info("Successfully generated sample benchmark data")
        
    except Exception as e:
        logger.error(f"Error generating sample data: {e}")
        return 1
    
    finally:
        # Close connection
        conn.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())