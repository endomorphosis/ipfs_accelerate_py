#\!/usr/bin/env python
import os
import sys
import subprocess
import duckdb

def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package])
        print(f"Successfully installed {package}")
        return True
    except:
        print(f"Failed to install {package}")
        return False

try:
    import duckdb
except ImportError:
    print("Trying to install duckdb...")
    success = install_package("duckdb")
    if not success:
        print("Unable to install duckdb. Using simplified fallback mode.")
        duckdb = None

def get_benchmark_data():
    if duckdb is None:
        print("DuckDB not available, cannot query database")
        return None
    
    try:
        # Connect to database
        conn = duckdb.connect('benchmark_db.duckdb')
        
        # Query to get table names
        tables = conn.execute("SHOW TABLES").fetchall()
        print(f"Tables in database: {[t[0] for t in tables]}")
        
        # Check for performance_results table
        result = conn.execute("""
        SELECT 
            COUNT(*) as count,
            COUNT(DISTINCT model_id) as model_count,
            COUNT(DISTINCT hardware_id) as hardware_count,
            MIN(created_at) as first_record,
            MAX(created_at) as last_record
        FROM 
            performance_results
        """).fetchone()
        
        print(f"Performance results: {result}")
        
        # Get model info
        models = conn.execute("""
        SELECT model_id, model_name, model_family 
        FROM models 
        LIMIT 20
        """).fetchall()
        
        print(f"Models in database: {models}")
        
        # Get hardware info
        hardware = conn.execute("""
        SELECT hardware_id, hardware_type, device_name 
        FROM hardware_platforms 
        LIMIT 20
        """).fetchall()
        
        print(f"Hardware in database: {hardware}")
        
        # Get performance metrics for 13 high-priority models
        key_models = [
            'bert-base-uncased', 'bert-tiny',
            't5-small', 't5-efficient-tiny',
            'llama', 'opt-125m',
            'clip', 'vit-base',
            'clap', 'whisper-tiny',
            'wav2vec2', 'llava',
            'xclip', 'qwen2',
            'detr'
        ]
        
        print("\nPerformance data for high-priority models:")
        print("| Model | Hardware | Latency (ms) | Throughput (items/s) | Memory (MB) |")
        print("|-------|----------|--------------|----------------------|-------------|")
        
        for model in key_models:
            # Use LIKE to handle model name variations
            results = conn.execute("""
            SELECT 
                m.model_name, 
                h.hardware_type, 
                AVG(pr.average_latency_ms) as avg_latency, 
                AVG(pr.throughput_items_per_second) as avg_throughput,
                AVG(pr.memory_peak_mb) as avg_memory
            FROM 
                performance_results pr
            JOIN 
                models m ON pr.model_id = m.model_id
            JOIN 
                hardware_platforms h ON pr.hardware_id = h.hardware_id
            WHERE 
                m.model_name LIKE ?
            GROUP BY 
                m.model_name, h.hardware_type
            ORDER BY 
                m.model_name, h.hardware_type
            """, [f"%{model}%"]).fetchall()
            
            for row in results:
                latency = row[2] if row[2] is not None else "N/A"
                throughput = row[3] if row[3] is not None else "N/A"
                memory = row[4] if row[4] is not None else "N/A"
                
                if isinstance(latency, (int, float)):
                    latency = f"{latency:.2f}"
                if isinstance(throughput, (int, float)):
                    throughput = f"{throughput:.2f}"
                if isinstance(memory, (int, float)):
                    memory = f"{memory:.2f}"
                
                print(f"| {row[0]} | {row[1]} | {latency} | {throughput} | {memory} |")
        
        # Get hardware compatibility matrix
        print("\nHardware compatibility matrix:")
        print("| Model Family | CPU | CUDA | ROCm | MPS | OpenVINO | QNN | WebNN | WebGPU |")
        print("|--------------|-----|------|------|-----|----------|-----|-------|--------|")
        
        model_families = conn.execute("""
        SELECT DISTINCT model_family 
        FROM models 
        WHERE model_family IS NOT NULL
        ORDER BY model_family
        """).fetchall()
        
        for family in model_families:
            family_name = family[0]
            if not family_name:
                continue
                
            # Start row with model family
            row = f"| {family_name} |"
            
            # Get compatibility for each hardware type
            hardware_types = ['cpu', 'cuda', 'rocm', 'mps', 'openvino', 'qnn', 'webnn', 'webgpu']
            
            for hw_type in hardware_types:
                # Get compatibility data
                comp_data = conn.execute("""
                SELECT 
                    COUNT(CASE WHEN hc.is_compatible THEN 1 END) as compatible_count,
                    COUNT(*) as total_count
                FROM 
                    hardware_compatibility hc
                JOIN 
                    models m ON hc.model_id = m.model_id
                JOIN 
                    hardware_platforms hp ON hc.hardware_id = hp.hardware_id
                WHERE 
                    m.model_family = ? AND hp.hardware_type = ?
                """, [family_name, hw_type]).fetchone()
                
                if comp_data and comp_data[1] > 0:
                    compatibility_rate = comp_data[0] / comp_data[1]
                    
                    if compatibility_rate > 0.8:
                        status = "✅ High"
                    elif compatibility_rate > 0.5:
                        status = "✅ Medium"
                    else:
                        status = "⚠️ Limited"
                else:
                    status = "❓"
                
                row += f" {status} |"
            
            print(row)
        
        conn.close()
        return True
    
    except Exception as e:
        print(f"Error querying database: {e}")
        return False

if __name__ == "__main__":
    get_benchmark_data()
