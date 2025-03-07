#\!/usr/bin/env python
import sys
import duckdb
import pandas as pd

def main():
    """Query the benchmark database and output performance tables"""
    try:
        # Connect to the database
        conn = duckdb.connect('benchmark_db.duckdb')
        
        # List tables to confirm schema
        tables = conn.execute("SHOW TABLES").fetchall()
        print(f"Tables in database: {[table[0] for table in tables]}")
        
        # Get models
        models_df = conn.execute("""
            SELECT model_id, model_name, model_family
            FROM models
            ORDER BY model_id
        """).fetch_df()
        print("\nModels in database:")
        print(models_df)
        
        # Get hardware platforms  
        hardware_df = conn.execute("""
            SELECT hardware_id, hardware_type, device_name
            FROM hardware_platforms
            ORDER BY hardware_id
        """).fetch_df()
        print("\nHardware platforms in database:")
        print(hardware_df)
        
        # Count performance results
        result_count = conn.execute("""
            SELECT COUNT(*) AS count
            FROM performance_results
        """).fetchone()[0]
        print(f"\nTotal performance results: {result_count}")
        
        # Get performance metrics for high-priority models
        # Use the models listed in CLAUDE.md
        key_models = [
            'bert-base-uncased', 'bert-tiny',
            't5-small', 't5-efficient-tiny',
            'llama', 'opt-125m',
            'clip', 'vit-base',
            'clap', 'whisper-tiny',
            'wav2vec2', 'llava',
            'llava-next', 'xclip',
            'qwen2', 'qwen3',
            'detr'
        ]
        
        # Create a markdown table of results
        print("\n## Benchmark Results for High-Priority Models Across Hardware Backends\n")
        print("| Model | Hardware | Latency (ms) | Throughput (items/s) | Memory (MB) |")
        print("|-------|----------|--------------|----------------------|-------------|")
        
        # Match models using LIKE for partial matches
        for model_pattern in key_models:
            model_name = f"%{model_pattern}%"
            
            # Get performance results
            results = conn.execute("""
                SELECT 
                    m.model_name,
                    h.hardware_type,
                    p.average_latency_ms,
                    p.throughput_items_per_second,
                    p.memory_peak_mb
                FROM 
                    performance_results p
                JOIN 
                    models m ON p.model_id = m.model_id
                JOIN 
                    hardware_platforms h ON p.hardware_id = h.hardware_id
                WHERE 
                    m.model_name LIKE ?
                ORDER BY 
                    m.model_name, h.hardware_type
            """, [model_name]).fetchall()
            
            for row in results:
                model = row[0]
                hardware = row[1]
                latency = f"{row[2]:.2f}" if row[2] is not None else "N/A"
                throughput = f"{row[3]:.2f}" if row[3] is not None else "N/A"
                memory = f"{row[4]:.2f}" if row[4] is not None else "N/A"
                
                print(f"| {model} | {hardware} | {latency} | {throughput} | {memory} |")
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
