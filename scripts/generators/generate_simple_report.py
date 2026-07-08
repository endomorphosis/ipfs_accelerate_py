#!/usr/bin/env python3
"""
Generate a simplified performance report from the benchmark database.
"""

import duckdb
import os
import sys
from datetime import datetime

def generate_markdown_report(db_path):
    """Generate a markdown report from the benchmark database."""
    
    try:
        con = duckdb.connect(db_path)
        
        # Get hardware platform data
        hardware_results = con.execute("""
            SELECT hardware_type, device_name, driver_version
            FROM hardware_platforms
        """).fetchall()
        
        # Get model data
        model_results = con.execute("""
            SELECT model_name, model_family
            FROM models
            LIMIT 10
        """).fetchall()
        
        # Get performance results
        performance_results = con.execute("""
            SELECT 
                m.model_name, 
                h.hardware_type, 
                p.batch_size, 
                p.precision,
                p.average_latency_ms, 
                p.throughput_items_per_second, 
                p.memory_peak_mb
            FROM performance_results p 
            JOIN models m ON p.model_id = m.model_id 
            JOIN hardware_platforms h ON p.hardware_id = h.hardware_id
            ORDER BY m.model_name, h.hardware_type, p.batch_size
        """).fetchall()
        
        # Start building the markdown report
        markdown = f"""# IPFS Accelerate Python Framework - Benchmark Report

## Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

This report provides an analysis of benchmark results from the IPFS Accelerate Python Framework database.

### Hardware Platforms

The following hardware platforms are included in the benchmark results:

| Hardware Type | Device Name | Driver Version |
|---------------|-------------|----------------|
"""
        
        # Add hardware platform data
        for row in hardware_results:
            hardware_type, device_name, driver_version = row
            device_name = device_name if device_name is not None else "N/A"
            driver_version = driver_version if driver_version is not None else "N/A"
            markdown += f"| {hardware_type} | {device_name} | {driver_version} |\n"
        
        # Add models section
        markdown += """
### Models

The following models are included in the benchmark results:

| Model | Model Family |
|-------|-------------|
"""

        # Add model data
        for row in model_results:
            model_name, model_family = row
            model_family = model_family if model_family is not None else "N/A"
            markdown += f"| {model_name} | {model_family} |\n"
        
        # Add performance results
        markdown += """
### Performance Results

The following table shows the performance results for each model on different hardware platforms:

| Model | Hardware | Batch Size | Precision | Latency (ms) | Throughput (items/sec) | Memory (MB) |
|-------|----------|------------|-----------|--------------|------------------------|------------|
"""
        
        # Add performance data
        for row in performance_results:
            model_name, hardware_type, batch_size, precision, latency, throughput, memory = row
            precision = precision if precision is not None else "N/A"
            markdown += f"| {model_name} | {hardware_type} | {batch_size} | {precision} | {latency:.2f} | {throughput:.2f} | {memory:.2f} |\n"
        
        # Add conclusion
        markdown += """
## Conclusion

This report provides a snapshot of the current benchmark results. For more detailed analysis, please use the benchmark_db_query.py tool.
"""
        
        # Return the markdown report
        return markdown
        
    except Exception as e:
        return f"Error generating report: {str(e)}\n\nTraceback:\n{sys.exc_info()}"

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
