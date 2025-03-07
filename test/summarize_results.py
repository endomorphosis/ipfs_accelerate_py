#\!/usr/bin/env python
import sys
import duckdb
import pandas as pd
import numpy as np

def main():
    """Create a summarized table of benchmark results for the 13 high-priority models"""
    try:
        # Connect to the database
        conn = duckdb.connect('benchmark_db.duckdb')
        
        # Define the 13 high-priority models based on CLAUDE.md
        key_models = [
            ('bert', 'bert-base-uncased'),
            ('t5', 't5-small'),
            ('llama', 'llama-7b'),
            ('clip', 'clip-vit-base-patch32'),
            ('vit', 'vit-base-patch16-224'),
            ('clap', 'clap-htsat-base'),
            ('whisper', 'whisper-tiny'),
            ('wav2vec2', 'wav2vec2-base'),
            ('llava', 'llava-7b'),
            ('llava_next', 'llava-next-7b'),
            ('xclip', 'xclip-base'),
            ('qwen2', 'qwen2-7b'),
            ('detr', 'detr-resnet-50')
        ]
        
        hardware_types = ['cpu', 'cuda', 'rocm', 'mps', 'openvino', 'qnn', 'webnn', 'webgpu']
        
        # Create summary table with average metrics for each model-hardware combination
        print("| Model | Hardware | Avg Latency (ms) | Avg Throughput (items/s) | Avg Memory (MB) |")
        print("|-------|----------|------------------|--------------------------|-----------------|")
        
        for model_family, model_name in key_models:
            # Get performance summary for this model across all hardware
            for hw in hardware_types:
                # Get average metrics for this model-hardware combination
                results = conn.execute("""
                    SELECT 
                        m.model_name,
                        h.hardware_type,
                        AVG(p.average_latency_ms) as avg_latency, 
                        AVG(p.throughput_items_per_second) as avg_throughput,
                        AVG(p.memory_peak_mb) as avg_memory
                    FROM 
                        performance_results p
                    JOIN 
                        models m ON p.model_id = m.model_id
                    JOIN 
                        hardware_platforms h ON p.hardware_id = h.hardware_id
                    WHERE 
                        m.model_name = ? AND h.hardware_type = ?
                    GROUP BY 
                        m.model_name, h.hardware_type
                """, [model_name, hw]).fetchall()
                
                if results and len(results) > 0:
                    row = results[0]
                    latency = f"{row[2]:.2f}" if row[2] is not None else "N/A"
                    throughput = f"{row[3]:.2f}" if row[3] is not None else "N/A"
                    memory = f"{row[4]:.2f}" if row[4] is not None else "N/A"
                    
                    print(f"| {model_name} | {hw} | {latency} | {throughput} | {memory} |")
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
