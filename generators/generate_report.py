#!/usr/bin/env python3
"""
Generate a performance report from the benchmark database.
"""

import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

def generate_markdown_report(db_path):
    """Generate a markdown report from the benchmark database."""
    
    try:
        con = duckdb.connect(db_path)
        
        # Get hardware platform data
        hardware_df = con.execute("""
            SELECT hardware_id, hardware_type, device_name, driver_version, compute_units, memory_gb
            FROM hardware_platforms
        """).fetchdf()
        
        # Get model data
        model_df = con.execute("""
            SELECT model_id, model_name, model_family, modality
            FROM models
        """).fetchdf()
        
        # Get performance results
        performance_df = con.execute("""
            SELECT 
                m.model_name, 
                h.hardware_type, 
                p.batch_size, 
                p.precision,
                p.average_latency_ms, 
                p.throughput_items_per_second, 
                p.memory_peak_mb,
                p.created_at
            FROM performance_results p 
            JOIN models m ON p.model_id = m.model_id 
            JOIN hardware_platforms h ON p.hardware_id = h.hardware_id
            ORDER BY m.model_name, h.hardware_type, p.batch_size
        """).fetchdf()
        
        # Get hardware compatibility matrix
        compatibility_df = con.execute("""
            SELECT 
                m.model_name,
                m.model_family,
                SUM(CASE WHEN h.hardware_type = 'cpu' THEN 1 ELSE 0 END) as cpu_support,
                SUM(CASE WHEN h.hardware_type = 'cuda' THEN 1 ELSE 0 END) as cuda_support,
                SUM(CASE WHEN h.hardware_type = 'rocm' THEN 1 ELSE 0 END) as rocm_support,
                SUM(CASE WHEN h.hardware_type = 'mps' THEN 1 ELSE 0 END) as mps_support,
                SUM(CASE WHEN h.hardware_type = 'openvino' THEN 1 ELSE 0 END) as openvino_support,
                SUM(CASE WHEN h.hardware_type = 'webnn' THEN 1 ELSE 0 END) as webnn_support,
                SUM(CASE WHEN h.hardware_type = 'webgpu' THEN 1 ELSE 0 END) as webgpu_support
            FROM models m
            LEFT JOIN performance_results p ON m.model_id = p.model_id
            LEFT JOIN hardware_platforms h ON p.hardware_id = h.hardware_id
            GROUP BY m.model_name, m.model_family
        """).fetchdf()
        
        # Start building the markdown report
        markdown = f"""# IPFS Accelerate Python Framework - Benchmark Report

## Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

This report provides an analysis of benchmark results from the IPFS Accelerate Python Framework database.

### Hardware Platforms

The following hardware platforms are included in the benchmark results:

| Hardware Type | Device Name | Driver Version | Compute Units | Memory (GB) |
|---------------|-------------|----------------|---------------|-------------|
"""
        
        # Add hardware platform data
        for _, row in hardware_df.iterrows():
            markdown += f"| {row['hardware_type']} | {row['device_name'] or 'N/A'} | {row['driver_version'] or 'N/A'} | {row['compute_units'] or 'N/A'} | {row['memory_gb'] or 'N/A'} |\n"
        
        # Add performance results
        markdown += """
### Performance Results

The following table shows the performance results for each model on different hardware platforms:

| Model | Hardware | Batch Size | Precision | Latency (ms) | Throughput (items/sec) | Memory (MB) |
|-------|----------|------------|-----------|--------------|------------------------|------------|
"""
        
        # Add performance data
        for _, row in performance_df.iterrows():
            markdown += f"| {row['model_name']} | {row['hardware_type']} | {row['batch_size']} | {row['precision'] or 'N/A'} | {row['average_latency_ms']:.2f} | {row['throughput_items_per_second']:.2f} | {row['memory_peak_mb']:.2f} |\n"
        
        # Add compatibility matrix
        markdown += """
### Hardware Compatibility Matrix

The following table shows which models have been tested on different hardware platforms:

| Model | Model Family | CPU | CUDA | ROCm | MPS | OpenVINO | WebNN | WebGPU |
|-------|-------------|-----|------|------|-----|----------|-------|--------|
"""
        
        # Add compatibility data
        for _, row in compatibility_df.iterrows():
            # Handle potential NA/None values
            cpu_support = row['cpu_support'] if not pd.isna(row['cpu_support']) else 0
            cuda_support = row['cuda_support'] if not pd.isna(row['cuda_support']) else 0
            rocm_support = row['rocm_support'] if not pd.isna(row['rocm_support']) else 0
            mps_support = row['mps_support'] if not pd.isna(row['mps_support']) else 0
            openvino_support = row['openvino_support'] if not pd.isna(row['openvino_support']) else 0
            webnn_support = row['webnn_support'] if not pd.isna(row['webnn_support']) else 0
            webgpu_support = row['webgpu_support'] if not pd.isna(row['webgpu_support']) else 0
            
            markdown += f"| {row['model_name']} | {row['model_family'] if not pd.isna(row['model_family']) else 'N/A'} "
            markdown += f"| {'✅' if cpu_support > 0 else '❌'} "
            markdown += f"| {'✅' if cuda_support > 0 else '❌'} "
            markdown += f"| {'✅' if rocm_support > 0 else '❌'} "
            markdown += f"| {'✅' if mps_support > 0 else '❌'} "
            markdown += f"| {'✅' if openvino_support > 0 else '❌'} "
            markdown += f"| {'✅' if webnn_support > 0 else '❌'} "
            markdown += f"| {'✅' if webgpu_support > 0 else '❌'} |\n"
        
        # Add model performance comparisons
        markdown += """
## Performance Comparisons

### Throughput Comparison

The following chart compares the throughput (items/second) for different models on various hardware platforms:

```
"""
        
        # Create a simple ASCII chart for throughput comparison
        # Try to create pivot table, but handle the case where there might be no data
        try:
            pivot_df = performance_df.pivot_table(
                index='model_name', 
                columns='hardware_type', 
                values='throughput_items_per_second',
                aggfunc='mean'
            )
            
            # Add the ASCII chart to markdown
            markdown += pivot_df.to_string()
        except Exception as e:
            markdown += f"No throughput data available. Error: {str(e)}"
            
        markdown += """
```

### Latency Comparison

The following chart compares the average latency (ms) for different models on various hardware platforms:

```
"""
        
        # Create a simple ASCII chart for latency comparison
        try:
            pivot_df = performance_df.pivot_table(
                index='model_name', 
                columns='hardware_type', 
                values='average_latency_ms',
                aggfunc='mean'
            )
            
            # Add the ASCII chart to markdown
            markdown += pivot_df.to_string()
        except Exception as e:
            markdown += f"No latency data available. Error: {str(e)}"
        markdown += """
```

## Conclusion

This report provides a snapshot of the current benchmark results. For more detailed analysis, please use the benchmark_db_query.py tool.
"""
        
        # Return the markdown report
        return markdown
        
    except Exception as e:
        return f"Error generating report: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        db_path = "./benchmark_db.duckdb"
    
    report = generate_markdown_report(db_path)
    
    # Write to file
    output_file = "benchmark_report.md"
    with open(output_file, "w") as f:
        f.write(report)
    
    print(f"Report generated and saved to {output_file}")

def _validate_data_authenticity(self, df):
    """
    Validate that the data is authentic and mark simulated results.
    
    Args:
        df: DataFrame with benchmark results
        
    Returns:
        Tuple of (DataFrame with authenticity flags, bool indicating if any simulation was detected)
    """
    logger.info("Validating data authenticity...")
    simulation_detected = False
    
    # Add new column to track simulation status
    if 'is_simulated' not in df.columns:
        df['is_simulated'] = False
    
    # Check database for simulation flags if possible
    if self.conn:
        try:
            # Query simulation status from database
            simulation_query = "SELECT hardware_type, COUNT(*) as count, SUM(CASE WHEN is_simulated THEN 1 ELSE 0 END) as simulated_count FROM hardware_platforms GROUP BY hardware_type"
            sim_result = self.conn.execute(simulation_query).fetchdf()
            
            if not sim_result.empty:
                for _, row in sim_result.iterrows():
                    hw = row['hardware_type']
                    if row['simulated_count'] > 0:
                        # Mark rows with this hardware as simulated
                        df.loc[df['hardware_type'] == hw, 'is_simulated'] = True
                        simulation_detected = True
                        logger.warning(f"Detected simulation data for hardware: {hw}")
        except Exception as e:
            logger.warning(f"Failed to check simulation status in database: {e}")
    
    # Additional checks for simulation indicators in the data
    for hw in ['qnn', 'rocm', 'openvino', 'webgpu', 'webnn']:
        hw_data = df[df['hardware_type'] == hw]
        if not hw_data.empty:
            # Check for simulation patterns in the data
            if hw_data['throughput_items_per_second'].std() < 0.1 and len(hw_data) > 1:
                logger.warning(f"Suspiciously uniform performance for {hw} - possible simulation")
                df.loc[df['hardware_type'] == hw, 'is_simulated'] = True
                simulation_detected = True
    
    return df, simulation_detected
