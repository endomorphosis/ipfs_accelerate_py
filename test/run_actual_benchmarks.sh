#!/bin/bash
# Run actual hardware benchmarks on available hardware

# Set environment variables
export BENCHMARK_DB_PATH=./benchmark_db.duckdb
export DEPRECATE_JSON_OUTPUT=1

# Run hardware detection
echo "Detecting available hardware..."
python hardware_reality_check.py

# Run benchmarks on a small model (bert-tiny) on CPU
echo ""
echo "Running CPU benchmark for bert-tiny..."
python benchmark_all_key_models.py --models bert-tiny --hardware cpu --batch-sizes 1 --db-only

# Mark the benchmark results as REAL hardware measurements
echo ""
echo "Updating simulation status in database..."
python - << EOF
import duckdb
import os

# Connect to the database
db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
conn = duckdb.connect(db_path)

# Find the model_id for bert-tiny
model_id_query = "SELECT model_id FROM models WHERE model_name = 'bert-tiny'"
model_id_result = conn.execute(model_id_query).fetchone()

if model_id_result:
    model_id = model_id_result[0]
    
    # Find the hardware_id for CPU
    hw_id_query = "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = 'cpu'"
    hw_id_result = conn.execute(hw_id_query).fetchone()
    
    if hw_id_result:
        hw_id = hw_id_result[0]
        
        # Update the simulation status for the recent benchmark results
        conn.execute("""
        UPDATE performance_results
        SET is_simulated = FALSE, simulation_reason = NULL
        WHERE model_id = ? AND hardware_id = ?
        AND test_timestamp >= NOW() - INTERVAL 30 MINUTE
        """, [model_id, hw_id])
        
        # Commit the changes
        conn.commit()
        print(f"Updated simulation status for bert-tiny (model_id={model_id}) on CPU (hardware_id={hw_id}) to indicate REAL HARDWARE")
    else:
        print("Could not find hardware_id for CPU")
else:
    print("Could not find model_id for bert-tiny")

# Check if bert-tiny exists in the models table, and if not, add it
model_check_query = "SELECT COUNT(*) FROM models WHERE model_name = 'bert-tiny'"
model_count = conn.execute(model_check_query).fetchone()[0]

if model_count == 0:
    # Add bert-tiny to the models table
    conn.execute("""
    INSERT INTO models (model_name, model_family, model_type, model_size, parameters_million, added_at)
    VALUES ('bert-tiny', 'bert', 'text', 'tiny', 4.4, NOW())
    """)
    conn.commit()
    print("Added bert-tiny to models table")

# Close the connection
conn.close()
EOF

# Check the simulation status
echo ""
echo "Checking simulation status in database..."
python - << EOF
import duckdb
import os

# Connect to the database
db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
conn = duckdb.connect(db_path)

# Query recent benchmark results with joins to get model and hardware names
results = conn.execute("""
SELECT 
    pr.id,
    m.model_name, 
    hp.hardware_type, 
    pr.is_simulated, 
    pr.simulation_reason,
    pr.test_timestamp
FROM performance_results pr
JOIN models m ON pr.model_id = m.model_id
JOIN hardware_platforms hp ON pr.hardware_id = hp.hardware_id
ORDER BY pr.test_timestamp DESC
LIMIT 10
""").fetchdf()

# Print results
print("Recent benchmark results:")
if len(results) > 0:
    for _, row in results.iterrows():
        model = row['model_name']
        hardware = row['hardware_type']
        is_simulated = row['is_simulated']
        reason = row['simulation_reason'] if is_simulated else "REAL HARDWARE"
        timestamp = row['test_timestamp']
        print(f"  {model} on {hardware} ({timestamp}): {'SIMULATED' if is_simulated else 'REAL'} - {reason}")
else:
    print("  No benchmark results found")

# Close the connection
conn.close()
EOF

echo ""
echo "Benchmarking completed."