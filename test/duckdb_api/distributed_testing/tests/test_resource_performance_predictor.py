#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the Resource Performance Predictor component of the Dynamic Resource Management system.
"""

import unittest
import os
import sys
import tempfile
import json
import sqlite3
import time
from unittest.mock import MagicMock, patch, mock_open
import numpy as np

# Add parent directory to path to import modules correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from resource_performance_predictor import ResourcePerformancePredictor
from constants import (
    DEFAULT_PREDICTION_CONFIDENCE_THRESHOLD,
    DEFAULT_PREDICTION_UPDATE_INTERVAL,
    DEFAULT_MIN_SAMPLES_FOR_ML,
    DEFAULT_MIN_SAMPLES_FOR_STATS
)


class TestResourcePerformancePredictor(unittest.TestCase):
    """Test suite for ResourcePerformancePredictor class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test database
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_predictor.db")
        
        # Initialize predictor with test database
        self.predictor = ResourcePerformancePredictor(
            db_path=self.db_path,
            confidence_threshold=DEFAULT_PREDICTION_CONFIDENCE_THRESHOLD,
            update_interval=DEFAULT_PREDICTION_UPDATE_INTERVAL,
            min_samples_for_ml=DEFAULT_MIN_SAMPLES_FOR_ML,
            min_samples_for_stats=DEFAULT_MIN_SAMPLES_FOR_STATS
        )
        
        # Sample task data
        self.task_data = {
            "type": "benchmark",
            "config": {
                "model": "bert-base-uncased",
                "batch_size": 4,
                "precision": "fp16",
                "sequence_length": 128
            },
            "requirements": {
                "hardware": ["cuda"]
            }
        }
        
        # Sample resource usage data
        self.resource_usage = {
            "cpu_cores": 4.2,
            "memory_mb": 8192,
            "gpu_memory_mb": 4096,
            "execution_time_seconds": 120
        }

    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()

    def test_init_default_values(self):
        """Test initialization with default values."""
        predictor = ResourcePerformancePredictor()
        self.assertIsNotNone(predictor.db_path)
        self.assertEqual(predictor.confidence_threshold, DEFAULT_PREDICTION_CONFIDENCE_THRESHOLD)
        self.assertEqual(predictor.update_interval, DEFAULT_PREDICTION_UPDATE_INTERVAL)
        self.assertEqual(predictor.min_samples_for_ml, DEFAULT_MIN_SAMPLES_FOR_ML)
        self.assertEqual(predictor.min_samples_for_stats, DEFAULT_MIN_SAMPLES_FOR_STATS)

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        custom_predictor = ResourcePerformancePredictor(
            db_path=":memory:",
            confidence_threshold=0.8,
            update_interval=1800,
            min_samples_for_ml=50,
            min_samples_for_stats=10
        )
        
        self.assertEqual(custom_predictor.db_path, ":memory:")
        self.assertEqual(custom_predictor.confidence_threshold, 0.8)
        self.assertEqual(custom_predictor.update_interval, 1800)
        self.assertEqual(custom_predictor.min_samples_for_ml, 50)
        self.assertEqual(custom_predictor.min_samples_for_stats, 10)

    def test_create_database(self):
        """Test database creation."""
        # Verify the database and tables are created
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if task_executions table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='task_executions'")
        self.assertIsNotNone(cursor.fetchone())
        
        # Check if models table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='prediction_models'")
        self.assertIsNotNone(cursor.fetchone())
        
        # Check if task_execution_features table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='task_execution_features'")
        self.assertIsNotNone(cursor.fetchone())
        
        conn.close()

    def test_record_task_execution(self):
        """Test recording task execution data."""
        # Record task execution
        self.predictor.record_task_execution(
            task_data=self.task_data,
            resource_usage=self.resource_usage,
            success=True
        )
        
        # Verify data was recorded
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM task_executions")
        count = cursor.fetchone()[0]
        self.assertEqual(count, 1)
        
        # Check task data
        cursor.execute("SELECT task_type, task_config, hardware, success FROM task_executions")
        row = cursor.fetchone()
        self.assertEqual(row[0], "benchmark")
        
        # Check task config was stored as JSON
        task_config = json.loads(row[1])
        self.assertEqual(task_config["model"], "bert-base-uncased")
        self.assertEqual(task_config["batch_size"], 4)
        
        # Check hardware requirements
        hardware = json.loads(row[2])
        self.assertEqual(hardware[0], "cuda")
        
        # Check success flag
        self.assertTrue(row[3])
        
        # Check resource usage was recorded
        cursor.execute("SELECT cpu_cores, memory_mb, gpu_memory_mb, execution_time_seconds FROM task_executions")
        row = cursor.fetchone()
        self.assertEqual(row[0], 4.2)
        self.assertEqual(row[1], 8192)
        self.assertEqual(row[2], 4096)
        self.assertEqual(row[3], 120)
        
        conn.close()

    def test_record_failed_task_execution(self):
        """Test recording failed task execution."""
        # Record failed task execution
        self.predictor.record_task_execution(
            task_data=self.task_data,
            resource_usage=self.resource_usage,
            success=False,
            error_message="Out of memory"
        )
        
        # Verify data was recorded
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT success, error_message FROM task_executions")
        row = cursor.fetchone()
        self.assertFalse(row[0])
        self.assertEqual(row[1], "Out of memory")
        
        conn.close()

    def test_predict_resource_requirements_no_data(self):
        """Test resource prediction with no historical data."""
        # No data recorded yet, should use default values
        prediction = self.predictor.predict_resource_requirements(self.task_data)
        
        self.assertIsNotNone(prediction)
        self.assertIn("cpu_cores", prediction)
        self.assertIn("memory_mb", prediction)
        self.assertIn("gpu_memory_mb", prediction)
        self.assertIn("prediction_method", prediction)
        self.assertEqual(prediction["prediction_method"], "default")

    @patch('resource_performance_predictor.ResourcePerformancePredictor._get_statistical_prediction')
    def test_predict_resource_requirements_statistical(self, mock_statistical):
        """Test resource prediction using statistical method."""
        # Mock statistical prediction method
        mock_statistical.return_value = {
            "cpu_cores": 4.5,
            "memory_mb": 9000,
            "gpu_memory_mb": 4500,
            "confidence": 0.6
        }
        
        # Mock few samples (less than min_samples_for_ml)
        with patch.object(self.predictor, '_get_sample_count') as mock_count:
            mock_count.return_value = DEFAULT_MIN_SAMPLES_FOR_STATS + 2
            
            prediction = self.predictor.predict_resource_requirements(self.task_data)
            
            self.assertEqual(prediction["prediction_method"], "statistical")
            self.assertEqual(prediction["cpu_cores"], 4.5)
            self.assertEqual(prediction["memory_mb"], 9000)
            self.assertEqual(prediction["gpu_memory_mb"], 4500)
            self.assertEqual(prediction["confidence"], 0.6)

    @patch('resource_performance_predictor.ResourcePerformancePredictor._get_ml_prediction')
    def test_predict_resource_requirements_ml(self, mock_ml):
        """Test resource prediction using ML method."""
        # Mock ML prediction method
        mock_ml.return_value = {
            "cpu_cores": 4.2,
            "memory_mb": 8500,
            "gpu_memory_mb": 4200,
            "confidence": 0.85
        }
        
        # Mock many samples (more than min_samples_for_ml)
        with patch.object(self.predictor, '_get_sample_count') as mock_count:
            mock_count.return_value = DEFAULT_MIN_SAMPLES_FOR_ML + 10
            
            prediction = self.predictor.predict_resource_requirements(self.task_data)
            
            self.assertEqual(prediction["prediction_method"], "ml")
            self.assertEqual(prediction["cpu_cores"], 4.2)
            self.assertEqual(prediction["memory_mb"], 8500)
            self.assertEqual(prediction["gpu_memory_mb"], 4200)
            self.assertEqual(prediction["confidence"], 0.85)

    def test_calculate_batch_scaling_factor(self):
        """Test calculation of batch scaling factor."""
        # Record several task executions with different batch sizes
        for batch_size in [1, 2, 4, 8, 16]:
            task_data = self.task_data.copy()
            task_data["config"] = self.task_data["config"].copy()
            task_data["config"]["batch_size"] = batch_size
            
            # Resource usage scales roughly linearly with batch size (with some variance)
            resource_usage = {
                "cpu_cores": 2 + (batch_size * 0.5) + np.random.normal(0, 0.2),
                "memory_mb": 4096 + (batch_size * 512) + np.random.normal(0, 100),
                "gpu_memory_mb": 1024 + (batch_size * 256) + np.random.normal(0, 50),
                "execution_time_seconds": 30 + (batch_size * 10) + np.random.normal(0, 2)
            }
            
            self.predictor.record_task_execution(
                task_data=task_data,
                resource_usage=resource_usage,
                success=True
            )
        
        # Calculate batch scaling for same model but different batch size
        scaling_factor = self.predictor.calculate_batch_scaling_factor(
            task_data=self.task_data,  # Original task with batch_size=4
            target_batch_size=8
        )
        
        # Should return reasonable scaling factors for each resource type
        self.assertIn("cpu_cores", scaling_factor)
        self.assertIn("memory_mb", scaling_factor)
        self.assertIn("gpu_memory_mb", scaling_factor)
        
        # Scaling factors should be roughly 2.0 (8/4) but with some variance
        self.assertGreater(scaling_factor["cpu_cores"], 1.5)
        self.assertLess(scaling_factor["cpu_cores"], 2.5)
        self.assertGreater(scaling_factor["memory_mb"], 1.5)
        self.assertLess(scaling_factor["memory_mb"], 2.5)
        self.assertGreater(scaling_factor["gpu_memory_mb"], 1.5)
        self.assertLess(scaling_factor["gpu_memory_mb"], 2.5)

    @patch('resource_performance_predictor.ResourcePerformancePredictor._check_model_update_needed')
    @patch('resource_performance_predictor.ResourcePerformancePredictor._train_model')
    def test_train_models(self, mock_train, mock_check_update):
        """Test training models."""
        # Mock check_model_update_needed to return True
        mock_check_update.return_value = True
        
        # Call train_models
        self.predictor.train_models()
        
        # Verify _train_model was called
        mock_train.assert_called()

    def test_extract_task_features(self):
        """Test extraction of task features."""
        # Basic task
        features = self.predictor._extract_task_features(self.task_data)
        
        # Should extract features from task config
        self.assertIn("model_bert-base-uncased", features)
        self.assertTrue(features["model_bert-base-uncased"])
        self.assertIn("batch_size", features)
        self.assertEqual(features["batch_size"], 4)
        self.assertIn("precision_fp16", features)
        self.assertTrue(features["precision_fp16"])
        self.assertIn("sequence_length", features)
        self.assertEqual(features["sequence_length"], 128)
        
        # Should include hardware requirements
        self.assertIn("hardware_cuda", features)
        self.assertTrue(features["hardware_cuda"])
        
        # Task with different features
        different_task = {
            "type": "test",
            "config": {
                "model": "t5-small",
                "batch_size": 8,
                "precision": "fp32",
                "test_args": ["--verbose"]
            },
            "requirements": {
                "hardware": ["cpu"],
                "min_memory_gb": 16
            }
        }
        
        features = self.predictor._extract_task_features(different_task)
        
        # Should extract different features
        self.assertIn("model_t5-small", features)
        self.assertTrue(features["model_t5-small"])
        self.assertIn("batch_size", features)
        self.assertEqual(features["batch_size"], 8)
        self.assertIn("precision_fp32", features)
        self.assertTrue(features["precision_fp32"])
        self.assertIn("hardware_cpu", features)
        self.assertTrue(features["hardware_cpu"])
        self.assertIn("min_memory_gb", features)
        self.assertEqual(features["min_memory_gb"], 16)


if __name__ == '__main__':
    unittest.main()