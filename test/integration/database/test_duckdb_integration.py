"""
Test for DuckDB integration in IPFS Accelerate.

This test verifies the integration between IPFS Accelerate and DuckDB for
storing and retrieving benchmark results and model compatibility information.
"""

import os
import pytest
import logging
import time
import json
import tempfile
from typing import Dict, List, Any, Optional, Tuple

# Database-specific imports
try:
    import duckdb
    import pandas as pd
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

# Test fixtures
@pytest.fixture
def temp_db_path():
    """Create a temporary DuckDB database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.duckdb', delete=False) as tmp:
        db_path = tmp.name
    
    yield db_path
    
    # Clean up
    if os.path.exists(db_path):
        os.unlink(db_path)

@pytest.fixture
def db_connection(temp_db_path):
    """Create a connection to the test database."""
    if not DUCKDB_AVAILABLE:
        pytest.skip("DuckDB not available")
    
    connection = duckdb.connect(temp_db_path)
    
    # Create schema
    connection.execute("""
    CREATE TABLE model_info (
        model_id VARCHAR,
        model_type VARCHAR,
        model_family VARCHAR,
        parameters BIGINT,
        added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (model_id)
    )
    """)
    
    connection.execute("""
    CREATE TABLE hardware_info (
        hardware_id VARCHAR,
        hardware_type VARCHAR,
        description VARCHAR,
        added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (hardware_id)
    )
    """)
    
    connection.execute("""
    CREATE TABLE benchmark_results (
        id BIGINT AUTO_INCREMENT,
        model_id VARCHAR,
        hardware_id VARCHAR,
        test_name VARCHAR,
        batch_size INTEGER,
        duration_ms DOUBLE,
        memory_mb DOUBLE,
        throughput DOUBLE,
        run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (id),
        FOREIGN KEY (model_id) REFERENCES model_info(model_id),
        FOREIGN KEY (hardware_id) REFERENCES hardware_info(hardware_id)
    )
    """)
    
    # Add sample data
    connection.execute("""
    INSERT INTO model_info (model_id, model_type, model_family, parameters)
    VALUES 
        ('bert-base-uncased', 'text', 'bert', 110000000),
        ('t5-small', 'text', 't5', 60000000),
        ('vit-base-patch16-224', 'vision', 'vit', 86000000)
    """)
    
    connection.execute("""
    INSERT INTO hardware_info (hardware_id, hardware_type, description)
    VALUES 
        ('cpu', 'cpu', 'Generic CPU'),
        ('cuda', 'gpu', 'NVIDIA GPU'),
        ('webgpu', 'browser', 'WebGPU in Chrome')
    """)
    
    # No initial benchmark results
    
    yield connection
    
    # Close connection
    connection.close()

@pytest.fixture
def sample_benchmark_results():
    """Generate sample benchmark results for testing."""
    return [
        {
            "model_id": "bert-base-uncased",
            "hardware_id": "cpu",
            "test_name": "inference",
            "batch_size": 1,
            "duration_ms": 150.5,
            "memory_mb": 1200.0,
            "throughput": 6.6
        },
        {
            "model_id": "bert-base-uncased",
            "hardware_id": "cuda",
            "test_name": "inference",
            "batch_size": 1,
            "duration_ms": 10.2,
            "memory_mb": 1500.0,
            "throughput": 98.0
        },
        {
            "model_id": "t5-small",
            "hardware_id": "cpu",
            "test_name": "inference",
            "batch_size": 1,
            "duration_ms": 200.3,
            "memory_mb": 950.0,
            "throughput": 5.0
        },
        {
            "model_id": "vit-base-patch16-224",
            "hardware_id": "webgpu",
            "test_name": "inference",
            "batch_size": 4,
            "duration_ms": 85.7,
            "memory_mb": 1800.0,
            "throughput": 46.7
        }
    ]

@pytest.mark.integration
@pytest.mark.database
class TestDuckDBIntegration:
    """
    Tests for DuckDB database integration.
    
    These tests verify that IPFS Accelerate can store and retrieve
    benchmark results and model compatibility information from DuckDB.
    """
    
    def test_db_connection(self, db_connection):
        """Test database connection and initial setup."""
        # Check that tables exist
        result = db_connection.execute("SHOW TABLES").fetchall()
        table_names = [row[0] for row in result]
        
        assert "model_info" in table_names
        assert "hardware_info" in table_names
        assert "benchmark_results" in table_names
        
        # Check that initial data was inserted correctly
        model_count = db_connection.execute("SELECT COUNT(*) FROM model_info").fetchone()[0]
        assert model_count == 3
        
        hardware_count = db_connection.execute("SELECT COUNT(*) FROM hardware_info").fetchone()[0]
        assert hardware_count == 3
    
    def test_store_benchmark_results(self, db_connection, sample_benchmark_results):
        """Test storing benchmark results in the database."""
        # Insert sample benchmark results
        for result in sample_benchmark_results:
            db_connection.execute("""
            INSERT INTO benchmark_results 
                (model_id, hardware_id, test_name, batch_size, duration_ms, memory_mb, throughput)
            VALUES 
                (?, ?, ?, ?, ?, ?, ?)
            """, 
            (
                result["model_id"], 
                result["hardware_id"], 
                result["test_name"], 
                result["batch_size"], 
                result["duration_ms"], 
                result["memory_mb"], 
                result["throughput"]
            ))
        
        # Verify that results were inserted
        count = db_connection.execute("SELECT COUNT(*) FROM benchmark_results").fetchone()[0]
        assert count == len(sample_benchmark_results)
    
    def test_query_benchmark_results(self, db_connection, sample_benchmark_results):
        """Test querying benchmark results from the database."""
        # Insert sample benchmark results first
        for result in sample_benchmark_results:
            db_connection.execute("""
            INSERT INTO benchmark_results 
                (model_id, hardware_id, test_name, batch_size, duration_ms, memory_mb, throughput)
            VALUES 
                (?, ?, ?, ?, ?, ?, ?)
            """, 
            (
                result["model_id"], 
                result["hardware_id"], 
                result["test_name"], 
                result["batch_size"], 
                result["duration_ms"], 
                result["memory_mb"], 
                result["throughput"]
            ))
        
        # Query for BERT results
        bert_results = db_connection.execute("""
        SELECT br.*, mi.model_type, mi.model_family, hi.hardware_type
        FROM benchmark_results br
        JOIN model_info mi ON br.model_id = mi.model_id
        JOIN hardware_info hi ON br.hardware_id = hi.hardware_id
        WHERE br.model_id = 'bert-base-uncased'
        """).fetchall()
        
        assert len(bert_results) == 2
        
        # Query for CPU results
        cpu_results = db_connection.execute("""
        SELECT br.*, mi.model_type, mi.model_family, hi.hardware_type
        FROM benchmark_results br
        JOIN model_info mi ON br.model_id = mi.model_id
        JOIN hardware_info hi ON br.hardware_id = hi.hardware_id
        WHERE br.hardware_id = 'cpu'
        """).fetchall()
        
        assert len(cpu_results) == 2
        
        # Query for fastest inference time
        fastest_result = db_connection.execute("""
        SELECT br.*, mi.model_type, mi.model_family, hi.hardware_type
        FROM benchmark_results br
        JOIN model_info mi ON br.model_id = mi.model_id
        JOIN hardware_info hi ON br.hardware_id = hi.hardware_id
        ORDER BY br.duration_ms ASC
        LIMIT 1
        """).fetchone()
        
        assert fastest_result is not None
        assert fastest_result[2] == "cuda"  # hardware_id
    
    def test_compatibility_matrix(self, db_connection, sample_benchmark_results):
        """Test generating a model compatibility matrix."""
        # Insert sample benchmark results first
        for result in sample_benchmark_results:
            db_connection.execute("""
            INSERT INTO benchmark_results 
                (model_id, hardware_id, test_name, batch_size, duration_ms, memory_mb, throughput)
            VALUES 
                (?, ?, ?, ?, ?, ?, ?)
            """, 
            (
                result["model_id"], 
                result["hardware_id"], 
                result["test_name"], 
                result["batch_size"], 
                result["duration_ms"], 
                result["memory_mb"], 
                result["throughput"]
            ))
        
        # Generate compatibility matrix
        matrix_query = """
        SELECT 
            mi.model_id,
            mi.model_type,
            mi.model_family,
            hi.hardware_id,
            hi.hardware_type,
            CASE WHEN br.id IS NOT NULL THEN 'compatible' ELSE 'unknown' END as compatibility,
            COALESCE(br.throughput, 0) as throughput
        FROM 
            model_info mi
        CROSS JOIN 
            hardware_info hi
        LEFT JOIN 
            benchmark_results br ON mi.model_id = br.model_id AND hi.hardware_id = br.hardware_id
        ORDER BY 
            mi.model_id, hi.hardware_id
        """
        
        matrix_results = db_connection.execute(matrix_query).fetchall()
        
        # We should have rows for every model-hardware combination
        assert len(matrix_results) == 9  # 3 models x 3 hardware types
        
        # Count compatible combinations
        compatible_count = sum(1 for row in matrix_results if row[5] == 'compatible')
        assert compatible_count == 4  # Same as the number of benchmark results
        
        # Verify specific compatibility
        bert_cuda_row = next((row for row in matrix_results 
                             if row[0] == 'bert-base-uncased' and row[3] == 'cuda'), None)
        assert bert_cuda_row is not None
        assert bert_cuda_row[5] == 'compatible'
        assert bert_cuda_row[6] == 98.0  # throughput
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_batch_size_scaling(self, db_connection, batch_size):
        """Test analyzing batch size scaling in benchmark results."""
        # Insert batch size scaling data
        for i, size in enumerate([1, 2, 4, 8]):
            # Simulate increasing throughput but diminishing returns
            throughput = 100 * size * 0.9**i
            
            db_connection.execute("""
            INSERT INTO benchmark_results 
                (model_id, hardware_id, test_name, batch_size, duration_ms, memory_mb, throughput)
            VALUES 
                ('bert-base-uncased', 'cuda', 'batch_scaling', ?, ?, ?, ?)
            """, 
            (
                size,
                1000 / throughput,  # duration increases with batch size
                1000 + size * 200,   # memory increases with batch size
                throughput
            ))
        
        # Query for specific batch size
        result = db_connection.execute("""
        SELECT batch_size, duration_ms, memory_mb, throughput
        FROM benchmark_results
        WHERE model_id = 'bert-base-uncased' 
          AND hardware_id = 'cuda'
          AND test_name = 'batch_scaling'
          AND batch_size = ?
        """, (batch_size,)).fetchone()
        
        assert result is not None
        assert result[0] == batch_size
        
        # Verify scaling analysis
        scaling_query = """
        WITH base AS (
            SELECT throughput
            FROM benchmark_results
            WHERE model_id = 'bert-base-uncased' 
              AND hardware_id = 'cuda'
              AND test_name = 'batch_scaling'
              AND batch_size = 1
        )
        SELECT 
            br.batch_size,
            br.throughput,
            br.throughput / (br.batch_size * base.throughput) as efficiency
        FROM 
            benchmark_results br,
            base
        WHERE 
            br.model_id = 'bert-base-uncased' 
            AND br.hardware_id = 'cuda'
            AND br.test_name = 'batch_scaling'
        ORDER BY 
            br.batch_size
        """
        
        scaling_results = db_connection.execute(scaling_query).fetchall()
        
        # Verify we have efficiency metrics
        assert len(scaling_results) == 4
        
        # Efficiency should decrease with batch size due to diminishing returns
        efficiencies = [row[2] for row in scaling_results]
        for i in range(1, len(efficiencies)):
            assert efficiencies[i] <= efficiencies[i-1]
    
    def test_export_results_to_pandas(self, db_connection, sample_benchmark_results):
        """Test exporting benchmark results to pandas for analysis."""
        if not hasattr(pd, 'DataFrame'):
            pytest.skip("pandas not available")
        
        # Insert sample benchmark results first
        for result in sample_benchmark_results:
            db_connection.execute("""
            INSERT INTO benchmark_results 
                (model_id, hardware_id, test_name, batch_size, duration_ms, memory_mb, throughput)
            VALUES 
                (?, ?, ?, ?, ?, ?, ?)
            """, 
            (
                result["model_id"], 
                result["hardware_id"], 
                result["test_name"], 
                result["batch_size"], 
                result["duration_ms"], 
                result["memory_mb"], 
                result["throughput"]
            ))
        
        # Export to pandas DataFrame
        df = db_connection.execute("""
        SELECT 
            br.model_id,
            mi.model_type,
            mi.model_family,
            br.hardware_id,
            hi.hardware_type,
            br.batch_size,
            br.duration_ms,
            br.memory_mb,
            br.throughput
        FROM 
            benchmark_results br
        JOIN 
            model_info mi ON br.model_id = mi.model_id
        JOIN 
            hardware_info hi ON br.hardware_id = hi.hardware_id
        """).df()
        
        # Verify DataFrame shape
        assert df.shape[0] == len(sample_benchmark_results)
        assert df.shape[1] == 9
        
        # Test pandas operations
        # Calculate average throughput by hardware type
        avg_throughput = df.groupby('hardware_type')['throughput'].mean()
        assert len(avg_throughput) == 3
        assert 'gpu' in avg_throughput.index
        assert 'cpu' in avg_throughput.index
        assert 'browser' in avg_throughput.index
        
        # GPU should be faster than CPU
        assert avg_throughput['gpu'] > avg_throughput['cpu']