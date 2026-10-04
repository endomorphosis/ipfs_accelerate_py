#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Time-Series Performance Tracking

This script tests the time-series performance tracking functionality by:
    1. Creating sample performance data
    2. Setting baselines
    3. Detecting regressions
    4. Analyzing trends
    5. Generating reports

    Author: IPFS Accelerate Python Framework Team
    Date: March 15, 2025
    Version: 1.0
    """

    import os
    import sys
    import tempfile
    import logging
    import datetime
    import argparse
    import random
    from pathlib import Path

    import pandas as pd
    import numpy as np
    import duckdb

# Add parent directory to path to import time_series_performance_tracker
    sys.path.insert())))))))))))))))0, os.path.abspath())))))))))))))))os.path.dirname())))))))))))))))__file__)))
    from time_series_performance_tracker import ())))))))))))))))
    TimeSeriesSchema,
    VersionManager,
    TimeSeriesManager,
    RegressionDetector,
    TrendVisualizer,
    NotificationSystem
    )

# Configure logging
    logging.basicConfig())))))))))))))))
    level=logging.INFO,
    format='%())))))))))))))))asctime)s - %())))))))))))))))name)s - %())))))))))))))))levelname)s - %())))))))))))))))message)s'
    )
    logger = logging.getLogger())))))))))))))))__name__)

def create_test_database())))))))))))))))):
    """Create a test database with sample performance data"""
    
    # Create temporary database file
    db_file = tempfile.mktemp())))))))))))))))suffix='.duckdb')
    logger.info())))))))))))))))f"Creating test database at {}db_file}")
    
    # Connect to database
    conn = duckdb.connect())))))))))))))))db_file)
    
    # Create tables
    conn.execute())))))))))))))))"""
    CREATE TABLE models ())))))))))))))))
    model_id INTEGER PRIMARY KEY,
    model_name VARCHAR,
    model_type VARCHAR,
    model_family VARCHAR,
    model_size VARCHAR
    )
    """)
    
    conn.execute())))))))))))))))"""
    CREATE TABLE hardware_platforms ())))))))))))))))
    hardware_id INTEGER PRIMARY KEY,
    hardware_type VARCHAR,
    device_name VARCHAR,
    compute_units INTEGER,
    memory_capacity FLOAT
    )
    """)
    
    conn.execute())))))))))))))))"""
    CREATE TABLE performance_results ())))))))))))))))
    id INTEGER PRIMARY KEY,
    model_id INTEGER,
    hardware_id INTEGER,
    batch_size INTEGER,
    sequence_length INTEGER,
    precision VARCHAR,
    throughput_items_per_second FLOAT,
    latency_ms FLOAT,
    memory_mb FLOAT,
    power_watts FLOAT,
    timestamp TIMESTAMP,
    version_tag VARCHAR,
    git_commit_hash VARCHAR,
    environment_hash VARCHAR,
    run_group_id VARCHAR,
    FOREIGN KEY ())))))))))))))))model_id) REFERENCES models())))))))))))))))model_id),
    FOREIGN KEY ())))))))))))))))hardware_id) REFERENCES hardware_platforms())))))))))))))))hardware_id)
    )
    """)
    
    # Insert sample data
    
    # Insert models
    conn.execute())))))))))))))))"""
    INSERT INTO models ())))))))))))))))model_id, model_name, model_type, model_family, model_size)
    VALUES
    ())))))))))))))))1, 'bert-base-uncased', 'text', 'bert', 'base'),
    ())))))))))))))))2, 't5-small', 'text', 't5', 'small'),
    ())))))))))))))))3, 'vit-base', 'vision', 'vit', 'base')
    """)
    
    # Insert hardware platforms
    conn.execute())))))))))))))))"""
    INSERT INTO hardware_platforms ())))))))))))))))hardware_id, hardware_type, device_name, compute_units, memory_capacity)
    VALUES
    ())))))))))))))))1, 'cuda', 'NVIDIA A100', 108, 40960),
    ())))))))))))))))2, 'rocm', 'AMD MI250', 220, 65536),
    ())))))))))))))))3, 'webgpu', 'Chrome WebGPU', 0, 0)
    """)
    
    # Close connection
    conn.close()))))))))))))))))
    
    return db_file

def generate_sample_performance_data())))))))))))))))ts_perf, days=60, samples_per_day=2):
    """
    Generate sample performance data for testing
    
    Args:
        ts_perf: TimeSeriesPerformance instance
        days: Number of days to generate data for
        samples_per_day: Number of samples per day
        """
        logger.info())))))))))))))))f"Generating sample performance data for {}days} days with {}samples_per_day} samples per day")
    
    # Define models and hardware to use
        models = [],())))))))))))))))1, "bert-base-uncased"), ())))))))))))))))2, "t5-small"), ())))))))))))))))3, "vit-base")],
        hardware = [],())))))))))))))))1, "cuda"), ())))))))))))))))2, "rocm"), ())))))))))))))))3, "webgpu")],
        batch_sizes = [],1, 4, 16],
        precisions = [],"fp32", "fp16"]
        ,
    # Define base performance metrics with realistic values
        base_metrics = {}
        # ())))))))))))))))model_id, hardware_id, batch_size, precision): ())))))))))))))))throughput, latency, memory, power)
        ())))))))))))))))1, 1, 1, "fp32"): ())))))))))))))))100.0, 10.0, 2048.0, 150.0),    # BERT on CUDA batch=1 fp32
        ())))))))))))))))1, 1, 4, "fp32"): ())))))))))))))))380.0, 11.0, 2048.0, 180.0),    # BERT on CUDA batch=4 fp32
        ())))))))))))))))1, 1, 16, "fp32"): ())))))))))))))))1500.0, 12.0, 2200.0, 220.0),  # BERT on CUDA batch=16 fp32
        ())))))))))))))))1, 1, 1, "fp16"): ())))))))))))))))120.0, 8.0, 1024.0, 120.0),     # BERT on CUDA batch=1 fp16
        ())))))))))))))))1, 1, 4, "fp16"): ())))))))))))))))460.0, 9.0, 1024.0, 140.0),     # BERT on CUDA batch=4 fp16
        ())))))))))))))))1, 1, 16, "fp16"): ())))))))))))))))1800.0, 10.0, 1100.0, 180.0),  # BERT on CUDA batch=16 fp16
        
        ())))))))))))))))1, 2, 1, "fp32"): ())))))))))))))))90.0, 12.0, 2048.0, 170.0),     # BERT on ROCm batch=1 fp32
        ())))))))))))))))1, 2, 4, "fp32"): ())))))))))))))))340.0, 13.0, 2048.0, 200.0),    # BERT on ROCm batch=4 fp32
        ())))))))))))))))1, 2, 16, "fp32"): ())))))))))))))))1350.0, 14.0, 2200.0, 240.0),  # BERT on ROCm batch=16 fp32
        ())))))))))))))))1, 2, 1, "fp16"): ())))))))))))))))110.0, 10.0, 1024.0, 140.0),    # BERT on ROCm batch=1 fp16
        ())))))))))))))))1, 2, 4, "fp16"): ())))))))))))))))420.0, 11.0, 1024.0, 160.0),    # BERT on ROCm batch=4 fp16
        ())))))))))))))))1, 2, 16, "fp16"): ())))))))))))))))1650.0, 12.0, 1100.0, 200.0),  # BERT on ROCm batch=16 fp16
        
        ())))))))))))))))1, 3, 1, "fp32"): ())))))))))))))))30.0, 35.0, 1800.0, 20.0),      # BERT on WebGPU batch=1 fp32
        ())))))))))))))))1, 3, 4, "fp32"): ())))))))))))))))110.0, 38.0, 1800.0, 22.0),     # BERT on WebGPU batch=4 fp32
        ())))))))))))))))1, 3, 16, "fp32"): ())))))))))))))))400.0, 42.0, 2000.0, 25.0),    # BERT on WebGPU batch=16 fp32
        ())))))))))))))))1, 3, 1, "fp16"): ())))))))))))))))35.0, 30.0, 900.0, 18.0),       # BERT on WebGPU batch=1 fp16
        ())))))))))))))))1, 3, 4, "fp16"): ())))))))))))))))130.0, 33.0, 900.0, 20.0),      # BERT on WebGPU batch=4 fp16
        ())))))))))))))))1, 3, 16, "fp16"): ())))))))))))))))480.0, 36.0, 1000.0, 22.0),    # BERT on WebGPU batch=16 fp16
        
        # Add similar patterns for T5 and ViT models
        ())))))))))))))))2, 1, 1, "fp32"): ())))))))))))))))80.0, 15.0, 1800.0, 140.0),     # T5 on CUDA batch=1 fp32
        ())))))))))))))))2, 2, 1, "fp32"): ())))))))))))))))70.0, 18.0, 1800.0, 160.0),     # T5 on ROCm batch=1 fp32
        ())))))))))))))))2, 3, 1, "fp32"): ())))))))))))))))25.0, 40.0, 1600.0, 18.0),      # T5 on WebGPU batch=1 fp32
        
        ())))))))))))))))3, 1, 1, "fp32"): ())))))))))))))))200.0, 5.0, 1500.0, 120.0),     # ViT on CUDA batch=1 fp32
        ())))))))))))))))3, 2, 1, "fp32"): ())))))))))))))))180.0, 6.0, 1500.0, 140.0),     # ViT on ROCm batch=1 fp32
        ())))))))))))))))3, 3, 1, "fp32"): ())))))))))))))))60.0, 18.0, 1300.0, 15.0),      # ViT on WebGPU batch=1 fp32
        }
    
    # Configuration to create regressions for certain combinations
        create_regressions = [],
        # ())))))))))))))))model_id, hardware_id, batch_size, precision, day_start)
        ())))))))))))))))1, 1, 1, "fp32", 40),  # BERT on CUDA batch=1 fp32, regression starts at day 40
        ())))))))))))))))2, 2, 1, "fp32", 45),  # T5 on ROCm batch=1 fp32, regression starts at day 45
        ())))))))))))))))3, 3, 1, "fp32", 50),  # ViT on WebGPU batch=1 fp32, regression starts at day 50
        ]
    
    # Generate data
        end_date = datetime.datetime.now()))))))))))))))))
        start_date = end_date - datetime.timedelta())))))))))))))))days=days)
        date_range = pd.date_range())))))))))))))))start=start_date, end=end_date, periods=days*samples_per_day)
    
    # Lists to store results for batch insertion
        result_ids = [],]
    
    # Create some realistic trends and variations in the data
    for day_idx, date in enumerate())))))))))))))))date_range):
        for model_id, model_name in models:
            for hardware_id, hardware_type in hardware:
                # Don't generate data for all combinations to keep test manageable
                if ())))))))))))))))model_id, hardware_id) not in [],())))))))))))))))1, 1), ())))))))))))))))1, 2), ())))))))))))))))1, 3), ())))))))))))))))2, 1), ())))))))))))))))2, 2), ())))))))))))))))2, 3), ())))))))))))))))3, 1), ())))))))))))))))3, 2), ())))))))))))))))3, 3)]:
                continue
                
                for batch_size in batch_sizes[],:2]:  # Just use batch sizes 1 and 4 to keep it simpler
                    for precision in precisions:
                        # Skip some combinations to keep test manageable
                        if model_id > 1 and batch_size > 1:
                        continue
                            
                        # Check if we have base metrics for this combination
                        key = ())))))))))))))))model_id, hardware_id, batch_size, precision):
                        if key not in base_metrics:
                            continue
                        
                        # Get base metrics
                            base_throughput, base_latency, base_memory, base_power = base_metrics[],key]
                        
                        # Add some random variation ())))))))))))))))1-2%)
                            random_factor = 1.0 + random.uniform())))))))))))))))-0.01, 0.01)
                        
                        # Add a slight improvement trend over time ())))))))))))))))0.1% per day)
                            trend_factor = 1.0 + ())))))))))))))))day_idx * 0.001)
                        
                        # Check if we should create a regression
                        regression = False:
                        for reg_model, reg_hw, reg_batch, reg_prec, reg_day in create_regressions:
                            if ())))))))))))))))model_id, hardware_id, batch_size, precision) == ())))))))))))))))reg_model, reg_hw, reg_batch, reg_prec) and day_idx >= reg_day:
                                regression = True
                            break
                        
                        # Apply regression if needed ())))))))))))))))5-10% worse)
                            regression_factor = 0.9 if regression else 1.0
                        
                        # Calculate final metrics with all factors
                            throughput = base_throughput * trend_factor * random_factor * regression_factor
                        
                        # For latency, higher is worse, so invert the trend and regression factors
                            latency = base_latency * ())))))))))))))))2.0 - trend_factor) * random_factor * ())))))))))))))))2.0 - regression_factor)
                        
                        # Memory and power typically don't improve as quickly with optimizations
                            memory = base_memory * ())))))))))))))))1.0 + ())))))))))))))))day_idx * 0.0002)) * random_factor * ())))))))))))))))regression_factor if regression else 1.0)
                            power = base_power * ())))))))))))))))1.0 + ())))))))))))))))day_idx * 0.0001)) * random_factor * ())))))))))))))))regression_factor if regression else 1.0)
                        
                        # Record performance result with a timestamp for this day
                            sequence_length = 128 if model_id in [],1, 2] else None
                            version_tag = f"v1.0.{}day_idx // 10}"  # Change version every 10 days
                        
                            result_id = ts_perf.record_performance_result())))))))))))))))
                            model_id=model_id,
                            hardware_id=hardware_id,
                            batch_size=batch_size,
                            sequence_length=sequence_length,
                            precision=precision,
                            throughput=throughput,
                            latency=latency,
                            memory=memory,
                            power=power,
                            version_tag=version_tag,
                            run_group_id=f"test_run_{}day_idx}"
                            )
                        
                            result_ids.append())))))))))))))))result_id)
    
                            logger.info())))))))))))))))f"Generated {}len())))))))))))))))result_ids)} sample performance results")
                            return result_ids
:
def test_time_series_performance())))))))))))))))):
    """Test the time-series performance tracking system"""
    
    # Create test database
    db_file = create_test_database()))))))))))))))))
    
    try:
        # Initialize schema
        schema = TimeSeriesSchema())))))))))))))))db_path=db_file)
        schema.create_schema_extensions()))))))))))))))))
        
        # Initialize managers
        version_manager = VersionManager())))))))))))))))db_path=db_file)
        ts_manager = TimeSeriesManager())))))))))))))))db_path=db_file)
        detector = RegressionDetector())))))))))))))))db_path=db_file, threshold=0.1)
        visualizer = TrendVisualizer())))))))))))))))db_path=db_file)
        
        # Add a version
        logger.info())))))))))))))))"Creating version entry")
        version_id = version_manager.create_version())))))))))))))))
        version_tag="v1.0.0-test",
        description="Test version",
        commit_hash="abcdef123456",
        git_branch="test"
        )
        logger.info())))))))))))))))f"Created version with ID {}version_id}")
        
        # Generate sample performance data
        logger.info())))))))))))))))"Generating sample performance data")
        
        # Record sample data for regression testing
        base_throughput = 100.0
        base_latency = 10.0
        base_memory = 1024.0
        
        # Record data points with small variations to establish baseline
        for i in range())))))))))))))))10):
            # Gradual improvement trend
            throughput = base_throughput + ())))))))))))))))i * 0.5)  # Small improvement
            latency = base_latency - ())))))))))))))))i * 0.1)  # Small improvement
            memory = base_memory
            
            for model_id, model_name in [],())))))))))))))))1, "bert-base-uncased"), ())))))))))))))))2, "t5-small"), ())))))))))))))))3, "vit-base")],:
                for hardware_id, hardware_type in [],())))))))))))))))1, "cuda"), ())))))))))))))))2, "rocm"), ())))))))))))))))3, "webgpu")],:
                    ts_manager.record_performance())))))))))))))))
                    model_name=model_name,
                    hardware_type=hardware_type,
                    batch_size=1,
                    test_type="inference",
                    throughput=throughput,
                    latency=latency,
                    memory_usage=memory,
                    version_tag="v1.0.0-test"
                    )
        
        # Add regression points
                    logger.info())))))))))))))))"Adding regression data points")
        
        # Add regression for each model-hardware combination
        for model_id, model_name in [],())))))))))))))))1, "bert-base-uncased"), ())))))))))))))))2, "t5-small")]:
            for hardware_id, hardware_type in [],())))))))))))))))1, "cuda"), ())))))))))))))))2, "rocm")]:
                ts_manager.record_performance())))))))))))))))
                model_name=model_name,
                hardware_type=hardware_type,
                batch_size=1,
                test_type="inference",
                throughput=base_throughput * 0.75,  # 25% regression
                latency=base_latency * 1.3,  # 30% regression
                memory_usage=base_memory,
                version_tag="v1.0.0-test"
                )
        
        # Detect regressions
                logger.info())))))))))))))))"Detecting regressions")
        
        for metric in [],'throughput', 'latency']:
            regressions = detector.detect_regressions())))))))))))))))
            model_name=None,  # All models
            hardware_type=None,  # All hardware
            metric=metric,
            window_size=5
            )
            
            logger.info())))))))))))))))f"Detected {}len())))))))))))))))regressions)} {}metric} regressions")
            
            if regressions:
                logger.info())))))))))))))))f"{}metric.capitalize()))))))))))))))))} regression details:")
                for reg in regressions[],:3]:  # Show only first 3 for brevity
                logger.info())))))))))))))))f"  {}reg[],'model_name']} on {}reg[],'hardware_type']}: "
                f"{}reg[],'percent_change']:.2f}% change ())))))))))))))))severity: {}reg[],'severity']})")
                
                # Record regressions
                detector.record_regressions())))))))))))))))regressions)
        
        # Generate visualizations
                logger.info())))))))))))))))"Generating visualizations")
                visualizations_dir = tempfile.mkdtemp()))))))))))))))))
        
        # Create trend visualization
        for model_name in [],"bert-base-uncased", "t5-small"]:
            for hardware_type in [],"cuda", "rocm"]:
                for metric in [],"throughput", "latency"]:
                    output_file = os.path.join())))))))))))))))visualizations_dir, f"{}model_name}_{}hardware_type}_{}metric}.png")
                    viz_path = visualizer.visualize_metric_trend())))))))))))))))
                    model_name=model_name,
                    hardware_type=hardware_type,
                    metric=metric,
                    days=30,
                    output_file=output_file
                    )
                    logger.info())))))))))))))))f"Generated {}metric} visualization for {}model_name} on {}hardware_type} at {}viz_path}")
        
        # Create regression dashboard
                    dashboard_path = visualizer.create_regression_dashboard())))))))))))))))
                    days=30,
                    limit=5,
                    output_file=os.path.join())))))))))))))))visualizations_dir, "regression_dashboard.png")
                    )
                    logger.info())))))))))))))))f"Generated regression dashboard at {}dashboard_path}")
        
        # Create comparison dashboard
                    comparison_path = visualizer.create_comparative_dashboard())))))))))))))))
                    model_names=[],"bert-base-uncased", "t5-small"],
                    hardware_types=[],"cuda", "rocm"],
                    metric="throughput",
                    days=30,
                    output_file=os.path.join())))))))))))))))visualizations_dir, "comparison_dashboard.png")
                    )
                    logger.info())))))))))))))))f"Generated comparison dashboard at {}comparison_path}")
        
        # Test notification system
                    notifier = NotificationSystem())))))))))))))))db_path=db_file)
                    pending = notifier.get_pending_notifications()))))))))))))))))
                    logger.info())))))))))))))))f"Found {}len())))))))))))))))pending)} pending notifications")
        
        # Mark notifications as sent
        for notification in pending:
            notifier.mark_notification_sent())))))))))))))))notification[],'id'])
            
            logger.info())))))))))))))))"All tests completed successfully")
        
    finally:
        # Clean up
        try:
            os.remove())))))))))))))))db_file)
            logger.info())))))))))))))))f"Removed test database at {}db_file}")
        except:
            pass

def main())))))))))))))))):
    """Main function for running tests"""
    
    parser = argparse.ArgumentParser())))))))))))))))description="Test Time-Series Performance Tracking")
    parser.add_argument())))))))))))))))"--keep-db", action="store_true", help="Keep the test database after running")
    parser.add_argument())))))))))))))))"--db-path", help="Path to database to use ())))))))))))))))created if it doesn't exist)")
    args = parser.parse_args()))))))))))))))))
    :
    if args.keep_db:
        # Create test database in current directory
        db_file = args.db_path or "test_time_series_performance.duckdb"
        
        # Initialize components
        schema = TimeSeriesSchema())))))))))))))))db_path=db_file)
        schema.create_schema_extensions()))))))))))))))))
        
        version_manager = VersionManager())))))))))))))))db_path=db_file)
        ts_manager = TimeSeriesManager())))))))))))))))db_path=db_file)
        
        # Create test version
        version_id = version_manager.create_version())))))))))))))))
        version_tag="v1.0.0-test",
        description="Test version for time series performance tracking",
        commit_hash="abcdef123456",
        git_branch="main"
        )
        
        # Generate sample performance data
        base_throughput = 100.0
        base_latency = 10.0
        base_memory = 1024.0
        
        logger.info())))))))))))))))"Generating sample data...")
        
        # Add baseline data
        for i in range())))))))))))))))10):
            # Small variations in metrics
            throughput = base_throughput + ())))))))))))))))i * 0.5)
            latency = base_latency - ())))))))))))))))i * 0.1)
            memory = base_memory
            
            for model_name in [],"bert-base-uncased", "t5-small", "vit-base"]:
                for hardware_type in [],"cuda", "rocm", "webgpu"]:
                    ts_manager.record_performance())))))))))))))))
                    model_name=model_name,
                    hardware_type=hardware_type,
                    batch_size=1,
                    test_type="inference",
                    throughput=throughput,
                    latency=latency,
                    memory_usage=memory,
                    version_tag="v1.0.0-test"
                    )
        
        # Add regression data points
        for model_name in [],"bert-base-uncased", "t5-small"]:
            for hardware_type in [],"cuda", "rocm"]:
                ts_manager.record_performance())))))))))))))))
                model_name=model_name,
                hardware_type=hardware_type,
                batch_size=1,
                test_type="inference",
                throughput=base_throughput * 0.7,  # 30% regression
                latency=base_latency * 1.4,  # 40% regression
                memory_usage=base_memory,
                version_tag="v1.0.0-test"
                )
        
                logger.info())))))))))))))))f"Generated sample performance data in {}db_file}")
                logger.info())))))))))))))))f"You can use this database for testing with:")
                logger.info())))))))))))))))f"  python test/time_series_performance_tracker.py detect --db-path {}db_file}")
                logger.info())))))))))))))))f"  python test/time_series_performance_tracker.py visualize --model bert-base-uncased --hardware cuda --db-path {}db_file}")
    else:
        # Run all tests
        test_time_series_performance()))))))))))))))))

if __name__ == "__main__":
    main()))))))))))))))))