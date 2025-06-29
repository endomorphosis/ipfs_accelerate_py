"""
Test for DuckDB integration.

This test verifies the integration with DuckDB for storing test results,
benchmark data, and model compatibility information.
"""

import pytest
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add the root directory to the Python path
test_dir = Path(__file__).resolve().parent.parent.parent.parent
if str(test_dir) not in sys.path:
    sys.path.insert(0, str(test_dir))

# Conditionally import DuckDB
try:
    import duckdb
    has_duckdb = True
except ImportError:
    has_duckdb = False


@pytest.fixture
def db_path():
    """Create a temporary DuckDB database for testing."""
    return ":memory:"  # Use in-memory database for tests


@pytest.fixture
def db_connection(db_path):
    """Create a DuckDB connection."""
    if not has_duckdb:
        pytest.skip("DuckDB not installed")
    return duckdb.connect(db_path)


@pytest.fixture
def benchmark_schema(db_connection):
    """Create the benchmark schema for testing."""
    db_connection.execute("""
        CREATE TABLE IF NOT EXISTS benchmark_results (
            id INTEGER PRIMARY KEY,
            timestamp TIMESTAMP,
            model_name VARCHAR,
            model_type VARCHAR,
            hardware_platform VARCHAR,
            batch_size INTEGER,
            sequence_length INTEGER,
            avg_latency_ms DOUBLE,
            throughput DOUBLE,
            memory_usage_mb DOUBLE,
            precision VARCHAR,
            notes VARCHAR
        )
    """)
    return db_connection


@pytest.fixture
def compatibility_schema(db_connection):
    """Create the compatibility schema for testing."""
    db_connection.execute("""
        CREATE TABLE IF NOT EXISTS model_compatibility (
            id INTEGER PRIMARY KEY,
            model_name VARCHAR,
            model_type VARCHAR,
            hardware_platform VARCHAR,
            supported BOOLEAN,
            min_batch_size INTEGER,
            max_batch_size INTEGER,
            min_sequence_length INTEGER,
            max_sequence_length INTEGER,
            precision VARCHAR,
            notes VARCHAR
        )
    """)
    return db_connection


@pytest.mark.integration
@pytest.mark.database
@pytest.mark.duckdb
class TestDuckDBIntegration:
    """Test suite for DuckDB integration."""
    
    @pytest.mark.skipif(not has_duckdb, reason="DuckDB not installed")
    def test_connection(self, db_connection):
        """Test basic DuckDB connection."""
        result = db_connection.execute("SELECT 1 AS test").fetchall()
        assert result == [(1,)]
    
    @pytest.mark.skipif(not has_duckdb, reason="DuckDB not installed")
    def test_benchmark_schema(self, benchmark_schema):
        """Test benchmark schema creation."""
        result = benchmark_schema.execute("PRAGMA table_info(benchmark_results)").fetchall()
        assert len(result) == 12  # Number of columns in the table
        
        # Verify column names
        columns = [row[1] for row in result]
        assert "model_name" in columns
        assert "hardware_platform" in columns
        assert "avg_latency_ms" in columns
        assert "throughput" in columns
    
    @pytest.mark.skipif(not has_duckdb, reason="DuckDB not installed")
    def test_compatibility_schema(self, compatibility_schema):
        """Test compatibility schema creation."""
        result = compatibility_schema.execute("PRAGMA table_info(model_compatibility)").fetchall()
        assert len(result) == 11  # Number of columns in the table
        
        # Verify column names
        columns = [row[1] for row in result]
        assert "model_name" in columns
        assert "hardware_platform" in columns
        assert "supported" in columns
        assert "precision" in columns
    
    @pytest.mark.skipif(not has_duckdb, reason="DuckDB not installed")
    def test_insert_benchmark_data(self, benchmark_schema):
        """Test inserting benchmark data."""
        benchmark_schema.execute("""
            INSERT INTO benchmark_results (
                timestamp, model_name, model_type, hardware_platform, 
                batch_size, sequence_length, avg_latency_ms, throughput,
                memory_usage_mb, precision, notes
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, [
            datetime.now(),
            "bert-base-uncased",
            "text",
            "webgpu",
            1,
            128,
            15.5,
            64.5,
            1024.0,
            "float16",
            "Test benchmark entry"
        ])
        
        result = benchmark_schema.execute("SELECT COUNT(*) FROM benchmark_results").fetchone()[0]
        assert result == 1
    
    @pytest.mark.skipif(not has_duckdb, reason="DuckDB not installed")
    def test_insert_compatibility_data(self, compatibility_schema):
        """Test inserting compatibility data."""
        compatibility_schema.execute("""
            INSERT INTO model_compatibility (
                model_name, model_type, hardware_platform, supported,
                min_batch_size, max_batch_size, min_sequence_length,
                max_sequence_length, precision, notes
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, [
            "bert-base-uncased",
            "text",
            "webgpu",
            True,
            1,
            8,
            8,
            512,
            "float16",
            "Fully supported on WebGPU"
        ])
        
        result = compatibility_schema.execute("SELECT COUNT(*) FROM model_compatibility").fetchone()[0]
        assert result == 1
    
    @pytest.mark.skipif(not has_duckdb, reason="DuckDB not installed")
    def test_query_benchmark_data(self, benchmark_schema):
        """Test querying benchmark data."""
        # Insert test data
        for i in range(5):
            benchmark_schema.execute("""
                INSERT INTO benchmark_results (
                    timestamp, model_name, model_type, hardware_platform, 
                    batch_size, sequence_length, avg_latency_ms, throughput,
                    memory_usage_mb, precision, notes
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, [
                datetime.now(),
                f"model-{i}",
                "text",
                "webgpu" if i % 2 == 0 else "cuda",
                2 ** i,
                128,
                10.0 + i * 5,
                50.0 - i * 5,
                512.0 + i * 256,
                "float16",
                f"Test entry {i}"
            ])
        
        # Query for WebGPU models
        result = benchmark_schema.execute("""
            SELECT model_name, hardware_platform, avg_latency_ms, throughput
            FROM benchmark_results
            WHERE hardware_platform = 'webgpu'
            ORDER BY avg_latency_ms
        """).fetchall()
        
        assert len(result) == 3  # Number of WebGPU models (even indices)
        assert result[0][0] == "model-0"  # First model
        assert result[0][1] == "webgpu"   # Platform
        assert result[0][2] == 10.0       # Latency for first model
    
    @pytest.mark.skipif(not has_duckdb, reason="DuckDB not installed")
    def test_query_compatibility_data(self, compatibility_schema):
        """Test querying compatibility data."""
        # Insert test data
        hardware_platforms = ["webgpu", "webnn", "cuda", "rocm", "cpu"]
        for i, platform in enumerate(hardware_platforms):
            compatibility_schema.execute("""
                INSERT INTO model_compatibility (
                    model_name, model_type, hardware_platform, supported,
                    min_batch_size, max_batch_size, min_sequence_length,
                    max_sequence_length, precision, notes
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, [
                "bert-base-uncased",
                "text",
                platform,
                i < 3,  # Only supported on webgpu, webnn, cuda
                1,
                2 ** (i + 1),
                8,
                512 - i * 32,
                "float16" if i < 2 else "float32",
                f"Compatibility notes for {platform}"
            ])
        
        # Query supported platforms
        result = compatibility_schema.execute("""
            SELECT hardware_platform, max_batch_size, precision
            FROM model_compatibility
            WHERE model_name = 'bert-base-uncased' AND supported = TRUE
            ORDER BY max_batch_size DESC
        """).fetchall()
        
        assert len(result) == 3  # Number of supported platforms
        assert result[0][0] == "cuda"     # Platform with highest max_batch_size
        assert result[0][1] == 8          # max_batch_size for cuda
        assert result[0][2] == "float32"  # Precision for cuda
    
    @pytest.mark.skipif(not has_duckdb, reason="DuckDB not installed")
    def test_pandas_integration(self, benchmark_schema):
        """Test integrating with pandas for analysis."""
        # Insert test data
        batch_sizes = [1, 2, 4, 8, 16]
        platforms = ["webgpu", "cuda", "webgpu", "cuda", "webgpu"]
        latencies = [10.0, 8.0, 15.0, 12.0, 20.0]
        
        for i in range(5):
            benchmark_schema.execute("""
                INSERT INTO benchmark_results (
                    timestamp, model_name, model_type, hardware_platform, 
                    batch_size, sequence_length, avg_latency_ms, throughput,
                    memory_usage_mb, precision, notes
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, [
                datetime.now(),
                "bert-base-uncased",
                "text",
                platforms[i],
                batch_sizes[i],
                128,
                latencies[i],
                batch_sizes[i] * 128 / latencies[i] * 1000,  # calculated throughput
                512.0,
                "float16",
                f"Test entry batch={batch_sizes[i]}"
            ])
        
        # Query as pandas DataFrame
        df = benchmark_schema.execute("""
            SELECT hardware_platform, batch_size, avg_latency_ms, throughput
            FROM benchmark_results
            WHERE model_name = 'bert-base-uncased'
            ORDER BY batch_size
        """).fetchdf()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert list(df.columns) == ["hardware_platform", "batch_size", "avg_latency_ms", "throughput"]
        
        # Check some pandas operations
        webgpu_df = df[df["hardware_platform"] == "webgpu"]
        assert len(webgpu_df) == 3
        
        avg_latency_by_platform = df.groupby("hardware_platform")["avg_latency_ms"].mean()
        assert len(avg_latency_by_platform) == 2
        assert avg_latency_by_platform["webgpu"] > avg_latency_by_platform["cuda"]