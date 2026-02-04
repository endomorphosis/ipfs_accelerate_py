#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the merge_benchmark_databases.py utility.
"""

import os
import sys
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Local imports
from test.merge_benchmark_databases import BenchmarkDatabaseMerger


@pytest.fixture
def mock_db_api():
    """Create a mock for the BenchmarkDBAPI."""
    mock = MagicMock()
    mock.get_all_benchmark_runs.return_value = [
        {
            "id": "run1",
            "device_info": {"platform": "android", "model": "Pixel 4"},
            "model_name": "bert-base-uncased",
            "timestamp": "2025-04-01T12:00:00Z"
        },
        {
            "id": "run2",
            "device_info": {"platform": "android", "model": "Pixel 4"},
            "model_name": "mobilenet-v2",
            "timestamp": "2025-04-01T12:30:00Z"
        }
    ]
    mock.get_benchmark_configurations.return_value = [
        {"run_id": "run1", "id": "config1", "configuration": {"batch_size": 1}}
    ]
    mock.get_benchmark_results.return_value = [
        {"run_id": "run1", "config_id": "config1", "latency_ms": 10.5}
    ]
    return mock


@pytest.fixture
def temp_db_files():
    """Create temporary database files for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create android database file
        android_db = os.path.join(temp_dir, "android_results.duckdb")
        with open(android_db, 'w') as f:
            f.write("mock duckdb file")
        
        # Create ios database file
        ios_db = os.path.join(temp_dir, "ios_results.duckdb")
        with open(ios_db, 'w') as f:
            f.write("mock duckdb file")
        
        # Create output database file path
        output_db = os.path.join(temp_dir, "merged_results.duckdb")
        
        yield temp_dir, android_db, ios_db, output_db


class TestBenchmarkDatabaseMerger:
    """Test cases for the BenchmarkDatabaseMerger class."""
    
    def test_find_input_files(self, temp_db_files):
        """Test finding input files."""
        temp_dir, android_db, ios_db, output_db = temp_db_files
        
        # Test with individual files
        merger = BenchmarkDatabaseMerger(
            output_db=output_db,
            input_dbs=[android_db, ios_db]
        )
        input_files = merger.find_input_files()
        assert len(input_files) == 2
        assert android_db in input_files
        assert ios_db in input_files
        
        # Test with directory
        merger = BenchmarkDatabaseMerger(
            output_db=output_db,
            input_dir=temp_dir,
            pattern="*.duckdb"
        )
        input_files = merger.find_input_files()
        assert len(input_files) == 2
        
        # Test with both
        merger = BenchmarkDatabaseMerger(
            output_db=output_db,
            input_dbs=[android_db],
            input_dir=temp_dir,
            pattern="ios_*.duckdb"
        )
        input_files = merger.find_input_files()
        assert len(input_files) == 2
        
        # Test with non-existent file
        merger = BenchmarkDatabaseMerger(
            output_db=output_db,
            input_dbs=[android_db, "non_existent.duckdb"]
        )
        input_files = merger.find_input_files()
        assert len(input_files) == 1
        assert android_db in input_files
    
    @patch('test.merge_benchmark_databases.BenchmarkDBAPI')
    def test_connect_to_output_db(self, mock_benchmark_db_api_class, temp_db_files):
        """Test connecting to output database."""
        temp_dir, android_db, ios_db, output_db = temp_db_files
        
        # Test successful connection
        mock_benchmark_db_api_class.return_value = MagicMock()
        merger = BenchmarkDatabaseMerger(output_db=output_db)
        result = merger.connect_to_output_db()
        assert result is True
        mock_benchmark_db_api_class.assert_called_once_with(output_db)
        
        # Test with existing file
        with open(output_db, 'w') as f:
            f.write("existing db")
        
        mock_benchmark_db_api_class.reset_mock()
        result = merger.connect_to_output_db()
        assert result is True
        mock_benchmark_db_api_class.assert_called_once_with(output_db)
        
        # Test connection error
        mock_benchmark_db_api_class.reset_mock()
        mock_benchmark_db_api_class.side_effect = Exception("Connection error")
        result = merger.connect_to_output_db()
        assert result is False
    
    @patch('test.merge_benchmark_databases.BenchmarkDBAPI')
    def test_merge_database(self, mock_benchmark_db_api_class, mock_db_api, temp_db_files):
        """Test merging a single database."""
        temp_dir, android_db, ios_db, output_db = temp_db_files
        
        # Setup mock
        mock_benchmark_db_api_class.return_value = mock_db_api
        
        # Create merger instance with mock output DB
        merger = BenchmarkDatabaseMerger(output_db=output_db)
        merger.db_api = mock_db_api
        
        # Test successful merge
        result = merger.merge_database(android_db)
        assert result is True
        
        # Verify calls to get data
        mock_db_api.get_all_benchmark_runs.assert_called_once()
        assert mock_db_api.get_benchmark_configurations.call_count == 2
        assert mock_db_api.get_benchmark_results.call_count == 2
        
        # Verify calls to insert data
        assert mock_db_api.insert_benchmark_run.call_count == 2
        assert mock_db_api.insert_benchmark_configuration.call_count >= 1
        assert mock_db_api.insert_benchmark_result.call_count >= 1
        
        # Check statistics
        assert merger.stats["benchmark_runs"] == 2
        assert "android" in merger.stats["device_platforms"]
        assert "bert-base-uncased" in merger.stats["model_names"]
        assert "mobilenet-v2" in merger.stats["model_names"]
        
        # Test merge with error
        mock_db_api.get_all_benchmark_runs.side_effect = Exception("Database error")
        result = merger.merge_database(ios_db)
        assert result is False
        assert merger.stats["errors"] == 1
    
    @patch('test.merge_benchmark_databases.BenchmarkDBAPI')
    def test_merge_all_databases(self, mock_benchmark_db_api_class, mock_db_api, temp_db_files):
        """Test merging all databases."""
        temp_dir, android_db, ios_db, output_db = temp_db_files
        
        # Setup mock
        mock_benchmark_db_api_class.return_value = mock_db_api
        
        # Test with no input files
        merger = BenchmarkDatabaseMerger(
            output_db=output_db,
            input_dbs=[]
        )
        with patch.object(merger, 'find_input_files', return_value=[]):
            result = merger.merge_all_databases()
            assert result is False
        
        # Test with input files
        merger = BenchmarkDatabaseMerger(
            output_db=output_db,
            input_dbs=[android_db, ios_db]
        )
        with patch.object(merger, 'merge_database', return_value=True):
            result = merger.merge_all_databases()
            assert result is True
            assert merger.merge_database.call_count == 2
    
    def test_generate_summary(self, temp_db_files):
        """Test generating summary information."""
        temp_dir, android_db, ios_db, output_db = temp_db_files
        
        merger = BenchmarkDatabaseMerger(output_db=output_db)
        merger.stats = {
            "input_files": 2,
            "benchmark_runs": 5,
            "device_platforms": {"android", "ios"},
            "model_names": {"bert-base-uncased", "mobilenet-v2", "whisper-tiny"},
            "errors": 0
        }
        
        summary = merger.generate_summary()
        assert summary["output_database"] == output_db
        assert summary["input_files"] == 2
        assert summary["successful_merges"] == 2
        assert summary["errors"] == 0
        assert summary["benchmark_runs"] == 5
        assert set(summary["device_platforms"]) == {"android", "ios"}
        assert len(summary["model_names"]) == 3


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])