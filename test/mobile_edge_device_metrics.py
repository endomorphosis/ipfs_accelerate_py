#!/usr/bin/env python
"""
Mobile/Edge Device Metrics Implementation

This script implements the Mobile/Edge Device Metrics database table as outlined in NEXT_STEPS.md.
It provides:
1. Schema creation for Mobile/Edge Device Metrics
2. Data collection utilities for QNN devices
3. Metrics integration with existing benchmark systems
4. Database query and reporting tools

Usage:
    python mobile_edge_device_metrics.py --create-schema --db-path ./benchmark_db.duckdb
    python mobile_edge_device_metrics.py --collect --model bert-base --device snapdragon8gen3
    python mobile_edge_device_metrics.py --report --format markdown --output mobile_edge_report.md
"""

import os
import sys
import time
import json
import logging
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent))

# Try to import QNN support modules
try:
    from hardware_detection.qnn_support import QNNCapabilityDetector, QNNPowerMonitor, QNNModelOptimizer
    QNN_AVAILABLE = True
except ImportError:
    logger.warning("QNN support modules not available")
    QNN_AVAILABLE = False

# Try to import database modules
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    logger.warning("DuckDB not available. Install with pip install duckdb")
    DUCKDB_AVAILABLE = False


def get_db_connection(db_path: str):
    """
    Get a connection to the benchmark database.
    
    Args:
        db_path: Path to the database
        
    Returns:
        Database connection
    """
    if not DUCKDB_AVAILABLE:
        raise ImportError("DuckDB not installed. Please install it with 'pip install duckdb'")
    
    try:
        return duckdb.connect(db_path)
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        raise


class MobileEdgeMetricsSchema:
    """Implements the Mobile/Edge Device Metrics database schema"""
    
    def __init__(self, db_path: str):
        """
        Initialize with database path.
        
        Args:
            db_path: Path to the DuckDB database
        """
        self.db_path = db_path
    
    def create_schema(self, overwrite: bool = False) -> bool:
        """
        Create the mobile/edge device metrics schema in the database.
        
        Args:
            overwrite: Whether to drop and recreate existing tables
            
        Returns:
            Success status
        """
        logger.info(f"Creating mobile/edge device metrics schema in {self.db_path}")
        
        conn = get_db_connection(self.db_path)
        
        try:
            # Create mobile edge metrics table
            if overwrite:
                conn.execute("DROP TABLE IF EXISTS mobile_edge_metrics")
            
            conn.execute("""
            CREATE TABLE IF NOT EXISTS mobile_edge_metrics (
                id INTEGER PRIMARY KEY,
                performance_id INTEGER,
                device_model VARCHAR,
                battery_impact_percent FLOAT,
                thermal_throttling_detected BOOLEAN,
                thermal_throttling_duration_seconds INTEGER,
                battery_temperature_celsius FLOAT,
                soc_temperature_celsius FLOAT,
                power_efficiency_score FLOAT,
                startup_time_ms FLOAT,
                runtime_memory_profile JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (performance_id) REFERENCES performance_results(id)
            )
            """)
            logger.info("Created mobile_edge_metrics table")
            
            # Create detailed thermal metrics table
            if overwrite:
                conn.execute("DROP TABLE IF EXISTS thermal_metrics")
            
            conn.execute("""
            CREATE TABLE IF NOT EXISTS thermal_metrics (
                id INTEGER PRIMARY KEY,
                mobile_edge_id INTEGER,
                timestamp FLOAT,
                soc_temperature_celsius FLOAT,
                battery_temperature_celsius FLOAT,
                cpu_temperature_celsius FLOAT,
                gpu_temperature_celsius FLOAT,
                ambient_temperature_celsius FLOAT,
                throttling_active BOOLEAN,
                throttling_level INTEGER,
                FOREIGN KEY (mobile_edge_id) REFERENCES mobile_edge_metrics(id)
            )
            """)
            logger.info("Created thermal_metrics table")
            
            # Create power consumption metrics table
            if overwrite:
                conn.execute("DROP TABLE IF EXISTS power_consumption_metrics")
            
            conn.execute("""
            CREATE TABLE IF NOT EXISTS power_consumption_metrics (
                id INTEGER PRIMARY KEY,
                mobile_edge_id INTEGER,
                timestamp FLOAT,
                total_power_mw FLOAT,
                cpu_power_mw FLOAT,
                gpu_power_mw FLOAT,
                dsp_power_mw FLOAT,
                npu_power_mw FLOAT,
                memory_power_mw FLOAT,
                FOREIGN KEY (mobile_edge_id) REFERENCES mobile_edge_metrics(id)
            )
            """)
            logger.info("Created power_consumption_metrics table")
            
            # Create device capability metrics table
            if overwrite:
                conn.execute("DROP TABLE IF EXISTS device_capabilities")
            
            conn.execute("""
            CREATE TABLE IF NOT EXISTS device_capabilities (
                id INTEGER PRIMARY KEY,
                device_model VARCHAR,
                chipset VARCHAR,
                ai_engine_version VARCHAR,
                compute_units INTEGER,
                total_memory_mb INTEGER,
                cpu_cores INTEGER,
                gpu_cores INTEGER,
                dsp_cores INTEGER,
                npu_cores INTEGER,
                max_cpu_freq_mhz INTEGER,
                max_gpu_freq_mhz INTEGER,
                supported_precisions JSON,
                driver_version VARCHAR,
                os_version VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            logger.info("Created device_capabilities table")
            
            # Create mobile application metrics table
            if overwrite:
                conn.execute("DROP TABLE IF EXISTS app_metrics")
            
            conn.execute("""
            CREATE TABLE IF NOT EXISTS app_metrics (
                id INTEGER PRIMARY KEY,
                mobile_edge_id INTEGER,
                app_memory_usage_mb FLOAT,
                system_memory_available_mb FLOAT,
                app_cpu_usage_percent FLOAT,
                system_cpu_usage_percent FLOAT,
                ui_responsiveness_ms FLOAT,
                battery_drain_percent_hour FLOAT,
                background_mode BOOLEAN,
                screen_on BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (mobile_edge_id) REFERENCES mobile_edge_metrics(id)
            )
            """)
            logger.info("Created app_metrics table")
            
            # Create mobile optimization settings table
            if overwrite:
                conn.execute("DROP TABLE IF EXISTS optimization_settings")
            
            conn.execute("""
            CREATE TABLE IF NOT EXISTS optimization_settings (
                id INTEGER PRIMARY KEY,
                mobile_edge_id INTEGER,
                quantization_method VARCHAR,
                precision VARCHAR,
                thread_count INTEGER,
                batch_size INTEGER,
                power_mode VARCHAR,
                memory_optimization VARCHAR,
                delegate VARCHAR,
                cache_enabled BOOLEAN,
                optimization_level INTEGER,
                additional_settings JSON,
                FOREIGN KEY (mobile_edge_id) REFERENCES mobile_edge_metrics(id)
            )
            """)
            logger.info("Created optimization_settings table")
            
            # Create view for comprehensive mobile metrics
            if overwrite:
                conn.execute("DROP VIEW IF EXISTS mobile_device_performance_view")
            
            conn.execute("""
            CREATE VIEW IF NOT EXISTS mobile_device_performance_view AS
            SELECT
                m.id AS metrics_id,
                pr.id AS performance_id,
                mod.model_name,
                mod.model_family,
                m.device_model,
                m.battery_impact_percent,
                m.thermal_throttling_detected,
                m.thermal_throttling_duration_seconds,
                m.battery_temperature_celsius,
                m.soc_temperature_celsius,
                m.power_efficiency_score,
                m.startup_time_ms,
                pr.average_latency_ms,
                pr.throughput_items_per_second,
                pr.memory_peak_mb,
                o.quantization_method,
                o.precision,
                o.power_mode,
                o.thread_count
            FROM
                mobile_edge_metrics m
            JOIN
                performance_results pr ON m.performance_id = pr.id
            JOIN
                models mod ON pr.model_id = mod.id
            LEFT JOIN
                optimization_settings o ON o.mobile_edge_id = m.id
            """)
            logger.info("Created mobile_device_performance_view")
            
            logger.info("Mobile/edge device metrics schema created successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error creating mobile/edge device metrics schema: {e}")
            return False
        
        finally:
            conn.close()
    
    def verify_schema(self) -> bool:
        """
        Verify that the mobile/edge device metrics schema exists.
        
        Returns:
            True if all tables and views exist
        """
        logger.info(f"Verifying mobile/edge device metrics schema in {self.db_path}")
        
        conn = get_db_connection(self.db_path)
        
        try:
            # Check if tables exist
            tables = [
                "mobile_edge_metrics",
                "thermal_metrics",
                "power_consumption_metrics",
                "device_capabilities",
                "app_metrics",
                "optimization_settings"
            ]
            
            missing_tables = []
            for table in tables:
                result = conn.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'").fetchone()
                if not result:
                    missing_tables.append(table)
                else:
                    logger.info(f"Table exists: {table}")
            
            # Check if view exists
            result = conn.execute("SELECT name FROM sqlite_master WHERE type='view' AND name='mobile_device_performance_view'").fetchone()
            if not result:
                missing_tables.append("mobile_device_performance_view (view)")
            else:
                logger.info("View exists: mobile_device_performance_view")
            
            if missing_tables:
                logger.error(f"Missing tables/views: {', '.join(missing_tables)}")
                return False
            
            logger.info("All mobile/edge device metrics tables and views exist")
            return True
        
        except Exception as e:
            logger.error(f"Error verifying mobile/edge device metrics schema: {e}")
            return False
        
        finally:
            conn.close()


class MobileEdgeMetricsCollector:
    """Collects mobile/edge device metrics from QNN devices"""
    
    def __init__(self, db_path: str):
        """
        Initialize with database path.
        
        Args:
            db_path: Path to the DuckDB database
        """
        self.db_path = db_path
        
        # Initialize QNN support if available
        if QNN_AVAILABLE:
            self.capability_detector = QNNCapabilityDetector()
            self.power_monitor = None
            self.model_optimizer = None
        else:
            logger.warning("QNN support not available, using simulated data")
            self.capability_detector = None
            self.power_monitor = None
            self.model_optimizer = None
    
    def collect_metrics(self, model_name: str, device_name: Optional[str] = None, 
                       duration_seconds: int = 60, use_simulation: bool = False) -> Dict[str, Any]:
        """
        Collect mobile/edge device metrics for a model on a device.
        
        Args:
            model_name: Name of the model
            device_name: Name of the device (optional)
            duration_seconds: Duration of monitoring in seconds
            use_simulation: Use simulated data even if QNN is available
            
        Returns:
            Dict with collected metrics
        """
        # Initialize metrics dictionary
        metrics = {
            "model_name": model_name,
            "device_name": device_name,
            "collection_time": datetime.datetime.now().isoformat(),
            "duration_seconds": duration_seconds,
            "simulated": use_simulation or not QNN_AVAILABLE
        }
        
        # Collect real metrics if QNN is available and simulation not requested
        if QNN_AVAILABLE and not use_simulation:
            logger.info(f"Collecting real QNN metrics for {model_name} on device {device_name}")
            metrics.update(self._collect_real_metrics(model_name, device_name, duration_seconds))
        else:
            logger.info(f"Collecting simulated metrics for {model_name} on device {device_name}")
            metrics.update(self._collect_simulated_metrics(model_name, device_name, duration_seconds))
        
        return metrics
    
    def _collect_real_metrics(self, model_name: str, device_name: Optional[str], 
                             duration_seconds: int) -> Dict[str, Any]:
        """
        Collect real metrics using QNN support.
        
        Args:
            model_name: Name of the model
            device_name: Name of the device
            duration_seconds: Duration of monitoring in seconds
            
        Returns:
            Dict with collected metrics
        """
        # Select device
        if device_name:
            self.capability_detector.select_device(device_name)
        else:
            self.capability_detector.select_device()
        
        # Get device capabilities
        capabilities = self.capability_detector.get_capability_summary()
        device_name = capabilities["device_name"]
        
        # Initialize power monitor for the device
        self.power_monitor = QNNPowerMonitor(device_name)
        
        # Start power monitoring
        logger.info(f"Starting power monitoring for {duration_seconds} seconds")
        self.power_monitor.start_monitoring()
        
        # Simulate model running for the duration
        # In a real implementation, this would run actual inference
        time.sleep(duration_seconds)
        
        # Stop monitoring and get results
        power_results = self.power_monitor.stop_monitoring()
        
        # Initialize model optimizer for optimization recommendations
        self.model_optimizer = QNNModelOptimizer(device_name)
        model_optimizations = self.model_optimizer.recommend_optimizations(model_name)
        
        # Calculate battery life impact
        battery_life = self.power_monitor.estimate_battery_life(
            power_results["average_power_watts"], 
            battery_capacity_mah=5000  # Typical flagship battery
        )
        
        # Combine all metrics
        metrics = {
            "device_model": device_name,
            "battery_impact_percent": power_results["estimated_battery_impact_percent"],
            "thermal_throttling_detected": power_results["thermal_throttling_detected"],
            "thermal_throttling_duration_seconds": power_results.get("thermal_throttling_duration_seconds", 0),
            "battery_temperature_celsius": power_results.get("average_battery_temp_celsius", 35.0),
            "soc_temperature_celsius": power_results["average_soc_temp_celsius"],
            "power_efficiency_score": power_results["power_efficiency_score"],
            "startup_time_ms": 150.0,  # Placeholder - would be measured in real implementation
            "runtime_memory_profile": {
                "peak_memory_mb": 512,  # Placeholder
                "average_memory_mb": 350  # Placeholder
            },
            "power_metrics": {
                "average_power_watts": power_results["average_power_watts"],
                "peak_power_watts": power_results["peak_power_watts"]
            },
            "battery_metrics": {
                "battery_capacity_mah": 5000,
                "battery_energy_wh": battery_life["battery_energy_wh"],
                "estimated_runtime_hours": battery_life["estimated_runtime_hours"],
                "battery_percent_per_hour": battery_life["battery_percent_per_hour"]
            },
            "optimization_recommendations": {
                "recommended_optimizations": model_optimizations.get("recommended_optimizations", []),
                "estimated_power_efficiency_score": model_optimizations.get("estimated_power_efficiency_score", 0)
            },
            "device_capabilities": capabilities
        }
        
        return metrics
    
    def _collect_simulated_metrics(self, model_name: str, device_name: Optional[str], 
                                  duration_seconds: int) -> Dict[str, Any]:
        """
        Collect simulated metrics for testing.
        
        Args:
            model_name: Name of the model
            device_name: Name of the device
            duration_seconds: Duration of monitoring in seconds
            
        Returns:
            Dict with simulated metrics
        """
        import random
        
        # Set default device name if not provided
        if not device_name:
            device_name = "Snapdragon 8 Gen 3"
        
        # Determine model complexity for metrics scaling
        if "tiny" in model_name or "mini" in model_name:
            complexity_factor = 0.3
        elif "small" in model_name:
            complexity_factor = 0.5
        elif "base" in model_name:
            complexity_factor = 0.8
        elif "large" in model_name:
            complexity_factor = 1.2
        else:
            complexity_factor = 1.0
        
        # Determine device capabilities for metrics scaling
        if "8 Gen 3" in device_name:
            device_factor = 1.2
            device_memory = 8192
            compute_units = 16
        elif "8 Gen 2" in device_name:
            device_factor = 1.0
            device_memory = 6144
            compute_units = 12
        elif "7+" in device_name:
            device_factor = 0.8
            device_memory = 4096
            compute_units = 8
        else:
            device_factor = 0.6
            device_memory = 3072
            compute_units = 6
        
        # Generate base metrics
        base_power_watts = 0.8 * complexity_factor / device_factor
        base_temp_celsius = 35.0 + (10.0 * complexity_factor / device_factor)
        
        # Add some randomness
        power_watts = max(0.1, base_power_watts + random.uniform(-0.2, 0.2))
        temp_celsius = max(30.0, base_temp_celsius + random.uniform(-2.0, 2.0))
        
        # Thermal throttling detection
        thermal_throttling = temp_celsius > 45.0
        if thermal_throttling:
            throttling_duration = duration_seconds * random.uniform(0.1, 0.3)
        else:
            throttling_duration = 0
        
        # Battery impact
        battery_impact = (power_watts / 3.5) * 100  # As percentage of typical full device power
        
        # Generate time series data for thermal and power metrics
        thermal_data = []
        power_data = []
        sample_count = min(int(duration_seconds / 2), 30)  # Sample every 2 seconds, max 30 samples
        
        for i in range(sample_count):
            rel_time = i / max(1, sample_count - 1)  # 0 to 1
            timestamp = time.time() + (rel_time * duration_seconds)
            
            # Temperature rises over time
            temp_rise = base_temp_celsius + (rel_time * 5.0 * complexity_factor / device_factor)
            temp_with_noise = max(30.0, temp_rise + random.uniform(-1.0, 1.0))
            
            thermal_data.append({
                "timestamp": timestamp,
                "soc_temperature_celsius": temp_with_noise,
                "battery_temperature_celsius": max(30.0, temp_with_noise - 3.0 + random.uniform(-0.5, 0.5)),
                "cpu_temperature_celsius": max(30.0, temp_with_noise + 2.0 + random.uniform(-1.0, 1.0)),
                "gpu_temperature_celsius": max(30.0, temp_with_noise + 5.0 + random.uniform(-1.0, 1.0)),
                "ambient_temperature_celsius": 25.0 + random.uniform(-2.0, 2.0),
                "throttling_active": temp_with_noise > 45.0,
                "throttling_level": 1 if temp_with_noise > 45.0 else 0
            })
            
            # Power varies based on workload
            power_with_noise = max(0.1, base_power_watts * (0.8 + 0.4 * random.random()))
            
            power_data.append({
                "timestamp": timestamp,
                "total_power_mw": power_with_noise * 1000,  # Convert to mW
                "cpu_power_mw": power_with_noise * 400,  # 40% of total
                "gpu_power_mw": power_with_noise * 300,  # 30% of total
                "dsp_power_mw": power_with_noise * 100,  # 10% of total
                "npu_power_mw": power_with_noise * 150,  # 15% of total
                "memory_power_mw": power_with_noise * 50   # 5% of total
            })
        
        # Generate simulated device capabilities
        device_capabilities = {
            "device_name": device_name,
            "compute_units": compute_units,
            "memory_mb": device_memory,
            "precision_support": ["fp32", "fp16", "int8"] + (["int4"] if "8 Gen" in device_name else []),
            "sdk_version": "2.10",
            "chipset": device_name,
            "cpu_cores": 8,
            "gpu_cores": compute_units,
            "dsp_cores": 2,
            "npu_cores": 4,
            "max_cpu_freq_mhz": 3000,
            "max_gpu_freq_mhz": 1000
        }
        
        # Generate optimization recommendations
        recommended_optimizations = ["quantization:fp16"]
        if "int8" in device_capabilities["precision_support"]:
            recommended_optimizations.append("quantization:int8")
        if "int4" in device_capabilities["precision_support"] and complexity_factor > 0.8:
            recommended_optimizations.append("quantization:int4")
        
        recommended_optimizations.append("memory:kv_cache_optimization" if complexity_factor > 0.8 else "pruning:magnitude")
        
        # Combine all metrics
        metrics = {
            "device_model": device_name,
            "battery_impact_percent": round(battery_impact, 2),
            "thermal_throttling_detected": thermal_throttling,
            "thermal_throttling_duration_seconds": round(throttling_duration, 2),
            "battery_temperature_celsius": round(temp_celsius - 3.0, 2),
            "soc_temperature_celsius": round(temp_celsius, 2),
            "power_efficiency_score": round(100 - min(100, battery_impact), 2),
            "startup_time_ms": round(100 + 200 * complexity_factor / device_factor, 2),
            "runtime_memory_profile": {
                "peak_memory_mb": round(256 * complexity_factor, 2),
                "average_memory_mb": round(180 * complexity_factor, 2)
            },
            "power_metrics": {
                "average_power_watts": round(power_watts, 2),
                "peak_power_watts": round(power_watts * 1.3, 2)
            },
            "battery_metrics": {
                "battery_capacity_mah": 5000,
                "battery_energy_wh": round(5000 * 3.85 / 1000, 2),
                "estimated_runtime_hours": round(19.25 / power_watts, 2),
                "battery_percent_per_hour": round(battery_impact, 2)
            },
            "thermal_data": thermal_data,
            "power_data": power_data,
            "optimization_recommendations": {
                "recommended_optimizations": recommended_optimizations,
                "estimated_power_efficiency_score": round(80 - 20 * complexity_factor / device_factor, 2)
            },
            "device_capabilities": device_capabilities,
            "app_metrics": {
                "app_memory_usage_mb": round(200 * complexity_factor, 2),
                "system_memory_available_mb": round(device_memory - 500 - 200 * complexity_factor, 2),
                "app_cpu_usage_percent": round(30 * complexity_factor, 2),
                "system_cpu_usage_percent": round(50 * complexity_factor, 2),
                "ui_responsiveness_ms": round(16 + 50 * complexity_factor / device_factor, 2),
                "battery_drain_percent_hour": round(battery_impact, 2)
            },
            "optimization_settings": {
                "quantization_method": "int8" if "int8" in device_capabilities["precision_support"] else "fp16",
                "precision": "int8" if "int8" in device_capabilities["precision_support"] else "fp16",
                "thread_count": 4,
                "batch_size": 1,
                "power_mode": "balanced",
                "memory_optimization": "kv_cache" if complexity_factor > 0.8 else "none",
                "delegate": "hexagon" if "8 Gen" in device_name else "gpu",
                "cache_enabled": True,
                "optimization_level": 3
            }
        }
        
        return metrics
    
    def store_metrics(self, metrics: Dict[str, Any], performance_id: Optional[int] = None) -> Optional[int]:
        """
        Store mobile/edge device metrics in the database.
        
        Args:
            metrics: The metrics to store
            performance_id: Optional ID linking to performance_results table
            
        Returns:
            ID of the inserted mobile_edge_metrics record, or None on failure
        """
        logger.info(f"Storing mobile/edge device metrics in {self.db_path}")
        
        conn = get_db_connection(self.db_path)
        
        try:
            # Check if the schema exists
            schema = MobileEdgeMetricsSchema(self.db_path)
            if not schema.verify_schema():
                logger.error("Mobile/edge device metrics schema not found")
                schema.create_schema()
            
            # Check if device_capabilities table exists for the device
            device_model = metrics["device_model"]
            device_exists = conn.execute("SELECT id FROM device_capabilities WHERE device_model = ?", [device_model]).fetchone()
            
            if not device_exists and "device_capabilities" in metrics:
                # Insert device capabilities
                device_caps = metrics["device_capabilities"]
                conn.execute("""
                INSERT INTO device_capabilities (
                    device_model, chipset, ai_engine_version, compute_units, total_memory_mb,
                    cpu_cores, gpu_cores, dsp_cores, npu_cores, max_cpu_freq_mhz,
                    max_gpu_freq_mhz, supported_precisions, driver_version, os_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    device_model, 
                    device_caps.get("chipset", device_model),
                    device_caps.get("sdk_version", "unknown"),
                    device_caps.get("compute_units", 0),
                    device_caps.get("memory_mb", 0),
                    device_caps.get("cpu_cores", 0),
                    device_caps.get("gpu_cores", 0),
                    device_caps.get("dsp_cores", 0),
                    device_caps.get("npu_cores", 0),
                    device_caps.get("max_cpu_freq_mhz", 0),
                    device_caps.get("max_gpu_freq_mhz", 0),
                    json.dumps(device_caps.get("precision_support", [])),
                    device_caps.get("driver_version", "unknown"),
                    device_caps.get("os_version", "unknown")
                ])
                logger.info(f"Added device capabilities for {device_model}")
            
            # Insert mobile_edge_metrics
            mobile_edge_id = conn.execute("""
            INSERT INTO mobile_edge_metrics (
                performance_id, device_model, battery_impact_percent, thermal_throttling_detected,
                thermal_throttling_duration_seconds, battery_temperature_celsius,
                soc_temperature_celsius, power_efficiency_score, startup_time_ms, runtime_memory_profile
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id
            """, [
                performance_id,
                metrics["device_model"],
                metrics["battery_impact_percent"],
                metrics["thermal_throttling_detected"],
                metrics["thermal_throttling_duration_seconds"],
                metrics["battery_temperature_celsius"],
                metrics["soc_temperature_celsius"],
                metrics["power_efficiency_score"],
                metrics["startup_time_ms"],
                json.dumps(metrics["runtime_memory_profile"])
            ]).fetchone()[0]
            logger.info(f"Added mobile edge metrics with ID {mobile_edge_id}")
            
            # Insert thermal metrics if available
            if "thermal_data" in metrics and metrics["thermal_data"]:
                thermal_data = metrics["thermal_data"]
                for data_point in thermal_data:
                    conn.execute("""
                    INSERT INTO thermal_metrics (
                        mobile_edge_id, timestamp, soc_temperature_celsius, battery_temperature_celsius,
                        cpu_temperature_celsius, gpu_temperature_celsius, ambient_temperature_celsius,
                        throttling_active, throttling_level
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        mobile_edge_id,
                        data_point["timestamp"],
                        data_point["soc_temperature_celsius"],
                        data_point["battery_temperature_celsius"],
                        data_point["cpu_temperature_celsius"],
                        data_point["gpu_temperature_celsius"],
                        data_point["ambient_temperature_celsius"],
                        data_point["throttling_active"],
                        data_point["throttling_level"]
                    ])
                logger.info(f"Added {len(thermal_data)} thermal data points")
            
            # Insert power consumption metrics if available
            if "power_data" in metrics and metrics["power_data"]:
                power_data = metrics["power_data"]
                for data_point in power_data:
                    conn.execute("""
                    INSERT INTO power_consumption_metrics (
                        mobile_edge_id, timestamp, total_power_mw, cpu_power_mw,
                        gpu_power_mw, dsp_power_mw, npu_power_mw, memory_power_mw
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        mobile_edge_id,
                        data_point["timestamp"],
                        data_point["total_power_mw"],
                        data_point["cpu_power_mw"],
                        data_point["gpu_power_mw"],
                        data_point["dsp_power_mw"],
                        data_point["npu_power_mw"],
                        data_point["memory_power_mw"]
                    ])
                logger.info(f"Added {len(power_data)} power consumption data points")
            
            # Insert app metrics if available
            if "app_metrics" in metrics:
                app_metrics = metrics["app_metrics"]
                conn.execute("""
                INSERT INTO app_metrics (
                    mobile_edge_id, app_memory_usage_mb, system_memory_available_mb,
                    app_cpu_usage_percent, system_cpu_usage_percent, ui_responsiveness_ms,
                    battery_drain_percent_hour, background_mode, screen_on
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    mobile_edge_id,
                    app_metrics["app_memory_usage_mb"],
                    app_metrics["system_memory_available_mb"],
                    app_metrics["app_cpu_usage_percent"],
                    app_metrics["system_cpu_usage_percent"],
                    app_metrics["ui_responsiveness_ms"],
                    app_metrics["battery_drain_percent_hour"],
                    app_metrics.get("background_mode", False),
                    app_metrics.get("screen_on", True)
                ])
                logger.info("Added app metrics")
            
            # Insert optimization settings if available
            if "optimization_settings" in metrics:
                opt_settings = metrics["optimization_settings"]
                conn.execute("""
                INSERT INTO optimization_settings (
                    mobile_edge_id, quantization_method, precision, thread_count,
                    batch_size, power_mode, memory_optimization, delegate,
                    cache_enabled, optimization_level, additional_settings
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    mobile_edge_id,
                    opt_settings["quantization_method"],
                    opt_settings["precision"],
                    opt_settings["thread_count"],
                    opt_settings["batch_size"],
                    opt_settings["power_mode"],
                    opt_settings["memory_optimization"],
                    opt_settings["delegate"],
                    opt_settings["cache_enabled"],
                    opt_settings["optimization_level"],
                    json.dumps(opt_settings.get("additional_settings", {}))
                ])
                logger.info("Added optimization settings")
            
            logger.info(f"Successfully stored all mobile/edge device metrics with ID {mobile_edge_id}")
            return mobile_edge_id
        
        except Exception as e:
            logger.error(f"Error storing mobile/edge device metrics: {e}")
            return None
        
        finally:
            conn.close()


class MobileEdgeMetricsReporter:
    """Generates reports from mobile/edge device metrics"""
    
    def __init__(self, db_path: str):
        """
        Initialize with database path.
        
        Args:
            db_path: Path to the DuckDB database
        """
        self.db_path = db_path
    
    def generate_report(self, format: str = "markdown", output_path: Optional[str] = None,
                       device_model: Optional[str] = None, model_name: Optional[str] = None) -> str:
        """
        Generate a report of mobile/edge device metrics.
        
        Args:
            format: Output format ("markdown", "json", "html")
            output_path: Path to save the report
            device_model: Filter by device model
            model_name: Filter by model name
            
        Returns:
            Report content or path to saved report
        """
        logger.info(f"Generating mobile/edge device metrics report in {format} format")
        
        conn = get_db_connection(self.db_path)
        
        try:
            # Build query with filters
            query = """
            SELECT
                m.id AS metrics_id,
                mdpv.model_name,
                mdpv.model_family,
                mdpv.device_model,
                mdpv.battery_impact_percent,
                mdpv.thermal_throttling_detected,
                mdpv.thermal_throttling_duration_seconds,
                mdpv.battery_temperature_celsius,
                mdpv.soc_temperature_celsius,
                mdpv.power_efficiency_score,
                mdpv.startup_time_ms,
                mdpv.average_latency_ms,
                mdpv.throughput_items_per_second,
                mdpv.memory_peak_mb,
                mdpv.quantization_method,
                mdpv.precision,
                mdpv.power_mode,
                mdpv.thread_count,
                m.created_at
            FROM
                mobile_edge_metrics m
            JOIN
                mobile_device_performance_view mdpv ON m.id = mdpv.metrics_id
            WHERE 1=1
            """
            
            params = []
            
            if device_model:
                query += " AND mdpv.device_model = ?"
                params.append(device_model)
            
            if model_name:
                query += " AND mdpv.model_name = ?"
                params.append(model_name)
            
            query += " ORDER BY m.created_at DESC"
            
            # Execute query
            results = conn.execute(query, params).fetchall()
            column_names = [desc[0] for desc in conn.description]
            
            # Convert to list of dicts
            results_list = []
            for row in results:
                result_dict = {column_names[i]: row[i] for i in range(len(column_names))}
                results_list.append(result_dict)
            
            # Generate report based on format
            if format == "json":
                report_content = json.dumps(results_list, indent=2, default=str)
            elif format == "html":
                report_content = self._generate_html_report(results_list)
            else:  # markdown is default
                report_content = self._generate_markdown_report(results_list)
            
            # Save report if output path provided
            if output_path:
                with open(output_path, "w") as f:
                    f.write(report_content)
                logger.info(f"Saved report to {output_path}")
                return output_path
            
            return report_content
        
        except Exception as e:
            logger.error(f"Error generating mobile/edge device metrics report: {e}")
            return f"Error generating report: {e}"
        
        finally:
            conn.close()
    
    def _generate_markdown_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate a markdown report from results"""
        if not results:
            return "# Mobile/Edge Device Metrics Report\n\nNo metrics found matching the criteria."
        
        report = f"# Mobile/Edge Device Metrics Report\n\nGenerated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += f"Found {len(results)} metrics records\n\n"
        
        # Summary table
        report += "## Summary\n\n"
        report += "| Model | Device | Power Efficiency | Battery Impact | Thermal Throttling | Latency (ms) | Throughput |\n"
        report += "|-------|--------|-----------------|----------------|-------------------|--------------|------------|\n"
        
        for result in results:
            thermal = "Yes" if result["thermal_throttling_detected"] else "No"
            report += f"| {result['model_name']} | {result['device_model']} | {result['power_efficiency_score']} | "
            report += f"{result['battery_impact_percent']}% | {thermal} | {result['average_latency_ms']} | "
            report += f"{result['throughput_items_per_second']} |\n"
        
        # Group by device
        report += "\n## Performance by Device\n\n"
        devices = {}
        
        for result in results:
            device = result["device_model"]
            if device not in devices:
                devices[device] = []
            devices[device].append(result)
        
        for device, device_results in devices.items():
            report += f"### {device}\n\n"
            report += "| Model | Power Efficiency | Battery Impact | Thermal Throttling | Latency (ms) | Throughput |\n"
            report += "|-------|-----------------|----------------|-------------------|--------------|------------|\n"
            
            for result in device_results:
                thermal = "Yes" if result["thermal_throttling_detected"] else "No"
                report += f"| {result['model_name']} | {result['power_efficiency_score']} | "
                report += f"{result['battery_impact_percent']}% | {thermal} | {result['average_latency_ms']} | "
                report += f"{result['throughput_items_per_second']} |\n"
            
            report += "\n"
        
        # Group by model
        report += "\n## Performance by Model\n\n"
        models = {}
        
        for result in results:
            model = result["model_name"]
            if model not in models:
                models[model] = []
            models[model].append(result)
        
        for model, model_results in models.items():
            report += f"### {model}\n\n"
            report += "| Device | Power Efficiency | Battery Impact | Thermal Throttling | Latency (ms) | Throughput |\n"
            report += "|--------|-----------------|----------------|-------------------|--------------|------------|\n"
            
            for result in model_results:
                thermal = "Yes" if result["thermal_throttling_detected"] else "No"
                report += f"| {result['device_model']} | {result['power_efficiency_score']} | "
                report += f"{result['battery_impact_percent']}% | {thermal} | {result['average_latency_ms']} | "
                report += f"{result['throughput_items_per_second']} |\n"
            
            report += "\n"
        
        # Optimization insights
        report += "\n## Optimization Insights\n\n"
        
        # Find best device for each model by power efficiency
        best_devices = {}
        for result in results:
            model = result["model_name"]
            if model not in best_devices or result["power_efficiency_score"] > best_devices[model]["score"]:
                best_devices[model] = {
                    "device": result["device_model"],
                    "score": result["power_efficiency_score"],
                    "quantization": result["quantization_method"],
                    "precision": result["precision"],
                    "power_mode": result["power_mode"]
                }
        
        report += "### Best Configurations\n\n"
        report += "| Model | Best Device | Power Efficiency | Quantization | Precision | Power Mode |\n"
        report += "|-------|-------------|-----------------|--------------|-----------|------------|\n"
        
        for model, best in best_devices.items():
            report += f"| {model} | {best['device']} | {best['score']} | {best['quantization']} | "
            report += f"{best['precision']} | {best['power_mode']} |\n"
        
        report += "\n"
        
        # Temperature analysis
        report += "### Temperature Analysis\n\n"
        report += "| Model | Device | Avg SOC Temp (°C) | Avg Battery Temp (°C) | Thermal Throttling |\n"
        report += "|-------|--------|------------------|----------------------|-------------------|\n"
        
        for result in results:
            thermal = f"Yes ({result['thermal_throttling_duration_seconds']}s)" if result["thermal_throttling_detected"] else "No"
            report += f"| {result['model_name']} | {result['device_model']} | {result['soc_temperature_celsius']} | "
            report += f"{result['battery_temperature_celsius']} | {thermal} |\n"
        
        report += "\n### Recommendations\n\n"
        
        # Generate recommendations based on results
        recommendations = [
            "Use INT8 quantization for best power efficiency on supported devices",
            "Monitor thermal throttling which can significantly impact performance",
            f"Use {list(best_devices.values())[0]['device'] if best_devices else 'latest devices'} for best mobile performance",
            "Balance between performance and battery impact based on application needs"
        ]
        
        for recommendation in recommendations:
            report += f"- {recommendation}\n"
        
        return report
    
    def _generate_html_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate an HTML report from results"""
        if not results:
            return "<html><body><h1>Mobile/Edge Device Metrics Report</h1><p>No metrics found matching the criteria.</p></body></html>"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Mobile/Edge Device Metrics Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .throttling-yes {{ color: red; }}
                .throttling-no {{ color: green; }}
                .efficiency-high {{ color: green; }}
                .efficiency-medium {{ color: orange; }}
                .efficiency-low {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Mobile/Edge Device Metrics Report</h1>
            <p>Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Found {len(results)} metrics records</p>
            
            <h2>Summary</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Device</th>
                    <th>Power Efficiency</th>
                    <th>Battery Impact</th>
                    <th>Thermal Throttling</th>
                    <th>Latency (ms)</th>
                    <th>Throughput</th>
                </tr>
        """
        
        for result in results:
            thermal_class = "throttling-yes" if result["thermal_throttling_detected"] else "throttling-no"
            thermal = "Yes" if result["thermal_throttling_detected"] else "No"
            
            efficiency_class = "efficiency-high"
            if result["power_efficiency_score"] < 60:
                efficiency_class = "efficiency-low"
            elif result["power_efficiency_score"] < 80:
                efficiency_class = "efficiency-medium"
            
            html += f"""
                <tr>
                    <td>{result['model_name']}</td>
                    <td>{result['device_model']}</td>
                    <td class="{efficiency_class}">{result['power_efficiency_score']}</td>
                    <td>{result['battery_impact_percent']}%</td>
                    <td class="{thermal_class}">{thermal}</td>
                    <td>{result['average_latency_ms']}</td>
                    <td>{result['throughput_items_per_second']}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>Performance by Device</h2>
        """
        
        # Group by device
        devices = {}
        for result in results:
            device = result["device_model"]
            if device not in devices:
                devices[device] = []
            devices[device].append(result)
        
        for device, device_results in devices.items():
            html += f"""
            <h3>{device}</h3>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Power Efficiency</th>
                    <th>Battery Impact</th>
                    <th>Thermal Throttling</th>
                    <th>Latency (ms)</th>
                    <th>Throughput</th>
                </tr>
            """
            
            for result in device_results:
                thermal_class = "throttling-yes" if result["thermal_throttling_detected"] else "throttling-no"
                thermal = "Yes" if result["thermal_throttling_detected"] else "No"
                
                efficiency_class = "efficiency-high"
                if result["power_efficiency_score"] < 60:
                    efficiency_class = "efficiency-low"
                elif result["power_efficiency_score"] < 80:
                    efficiency_class = "efficiency-medium"
                
                html += f"""
                    <tr>
                        <td>{result['model_name']}</td>
                        <td class="{efficiency_class}">{result['power_efficiency_score']}</td>
                        <td>{result['battery_impact_percent']}%</td>
                        <td class="{thermal_class}">{thermal}</td>
                        <td>{result['average_latency_ms']}</td>
                        <td>{result['throughput_items_per_second']}</td>
                    </tr>
                """
            
            html += """
            </table>
            """
        
        # Group by model
        html += """
            <h2>Performance by Model</h2>
        """
        
        models = {}
        for result in results:
            model = result["model_name"]
            if model not in models:
                models[model] = []
            models[model].append(result)
        
        for model, model_results in models.items():
            html += f"""
            <h3>{model}</h3>
            <table>
                <tr>
                    <th>Device</th>
                    <th>Power Efficiency</th>
                    <th>Battery Impact</th>
                    <th>Thermal Throttling</th>
                    <th>Latency (ms)</th>
                    <th>Throughput</th>
                </tr>
            """
            
            for result in model_results:
                thermal_class = "throttling-yes" if result["thermal_throttling_detected"] else "throttling-no"
                thermal = "Yes" if result["thermal_throttling_detected"] else "No"
                
                efficiency_class = "efficiency-high"
                if result["power_efficiency_score"] < 60:
                    efficiency_class = "efficiency-low"
                elif result["power_efficiency_score"] < 80:
                    efficiency_class = "efficiency-medium"
                
                html += f"""
                    <tr>
                        <td>{result['device_model']}</td>
                        <td class="{efficiency_class}">{result['power_efficiency_score']}</td>
                        <td>{result['battery_impact_percent']}%</td>
                        <td class="{thermal_class}">{thermal}</td>
                        <td>{result['average_latency_ms']}</td>
                        <td>{result['throughput_items_per_second']}</td>
                    </tr>
                """
            
            html += """
            </table>
            """
        
        # Optimization insights
        html += """
            <h2>Optimization Insights</h2>
            <h3>Best Configurations</h3>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Best Device</th>
                    <th>Power Efficiency</th>
                    <th>Quantization</th>
                    <th>Precision</th>
                    <th>Power Mode</th>
                </tr>
        """
        
        # Find best device for each model by power efficiency
        best_devices = {}
        for result in results:
            model = result["model_name"]
            if model not in best_devices or result["power_efficiency_score"] > best_devices[model]["score"]:
                best_devices[model] = {
                    "device": result["device_model"],
                    "score": result["power_efficiency_score"],
                    "quantization": result["quantization_method"],
                    "precision": result["precision"],
                    "power_mode": result["power_mode"]
                }
        
        for model, best in best_devices.items():
            efficiency_class = "efficiency-high"
            if best["score"] < 60:
                efficiency_class = "efficiency-low"
            elif best["score"] < 80:
                efficiency_class = "efficiency-medium"
            
            html += f"""
                <tr>
                    <td>{model}</td>
                    <td>{best['device']}</td>
                    <td class="{efficiency_class}">{best['score']}</td>
                    <td>{best['quantization']}</td>
                    <td>{best['precision']}</td>
                    <td>{best['power_mode']}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h3>Temperature Analysis</h3>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Device</th>
                    <th>Avg SOC Temp (°C)</th>
                    <th>Avg Battery Temp (°C)</th>
                    <th>Thermal Throttling</th>
                </tr>
        """
        
        for result in results:
            thermal_class = "throttling-yes" if result["thermal_throttling_detected"] else "throttling-no"
            thermal = f"Yes ({result['thermal_throttling_duration_seconds']}s)" if result["thermal_throttling_detected"] else "No"
            
            html += f"""
                <tr>
                    <td>{result['model_name']}</td>
                    <td>{result['device_model']}</td>
                    <td>{result['soc_temperature_celsius']}</td>
                    <td>{result['battery_temperature_celsius']}</td>
                    <td class="{thermal_class}">{thermal}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h3>Recommendations</h3>
            <ul>
        """
        
        # Generate recommendations based on results
        recommendations = [
            "Use INT8 quantization for best power efficiency on supported devices",
            "Monitor thermal throttling which can significantly impact performance",
            f"Use {list(best_devices.values())[0]['device'] if best_devices else 'latest devices'} for best mobile performance",
            "Balance between performance and battery impact based on application needs"
        ]
        
        for recommendation in recommendations:
            html += f"<li>{recommendation}</li>\n"
        
        html += """
            </ul>
        </body>
        </html>
        """
        
        return html


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Mobile/Edge Device Metrics Implementation")
    
    # General arguments
    parser.add_argument("--db-path", default="./benchmark_db.duckdb", help="Path to the benchmark database")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # create-schema command
    create_parser = subparsers.add_parser("create-schema", help="Create mobile/edge device metrics schema")
    create_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing tables")
    
    # verify-schema command
    verify_parser = subparsers.add_parser("verify-schema", help="Verify mobile/edge device metrics schema")
    
    # collect command
    collect_parser = subparsers.add_parser("collect", help="Collect mobile/edge device metrics")
    collect_parser.add_argument("--model", required=True, help="Model name")
    collect_parser.add_argument("--device", help="Device name")
    collect_parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    collect_parser.add_argument("--simulate", action="store_true", help="Use simulated data")
    collect_parser.add_argument("--performance-id", type=int, help="Performance ID to link with")
    collect_parser.add_argument("--output-json", help="Save metrics to JSON file")
    
    # report command
    report_parser = subparsers.add_parser("report", help="Generate mobile/edge device metrics report")
    report_parser.add_argument("--format", choices=["markdown", "json", "html"], default="markdown", help="Report format")
    report_parser.add_argument("--output", help="Output file path")
    report_parser.add_argument("--device", help="Filter by device model")
    report_parser.add_argument("--model", help="Filter by model name")
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Handle command
    if args.command == "create-schema":
        schema = MobileEdgeMetricsSchema(args.db_path)
        success = schema.create_schema(args.overwrite)
        if success:
            print("Mobile/edge device metrics schema created successfully")
        else:
            print("Failed to create mobile/edge device metrics schema")
            return 1
    
    elif args.command == "verify-schema":
        schema = MobileEdgeMetricsSchema(args.db_path)
        success = schema.verify_schema()
        if success:
            print("Mobile/edge device metrics schema verification successful")
        else:
            print("Mobile/edge device metrics schema verification failed")
            return 1
    
    elif args.command == "collect":
        collector = MobileEdgeMetricsCollector(args.db_path)
        metrics = collector.collect_metrics(args.model, args.device, args.duration, args.simulate)
        
        # Save metrics to JSON if requested
        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(metrics, f, indent=2, default=str)
            print(f"Metrics saved to {args.output_json}")
        
        # Store metrics in database
        mobile_edge_id = collector.store_metrics(metrics, args.performance_id)
        if mobile_edge_id:
            print(f"Metrics stored in database with ID {mobile_edge_id}")
        else:
            print("Failed to store metrics in database")
            return 1
    
    elif args.command == "report":
        reporter = MobileEdgeMetricsReporter(args.db_path)
        report = reporter.generate_report(args.format, args.output, args.device, args.model)
        
        if args.output:
            print(f"Report saved to {report}")
        else:
            print(report)
    
    else:
        parser.print_help()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())