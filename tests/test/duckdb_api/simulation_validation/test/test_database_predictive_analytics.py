#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for the Database Predictive Analytics module.
Focuses on testing parameter persistence functionality.
"""

import os
import sys
import unittest
import tempfile
import shutil
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hashlib

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database_predictive_analytics import DatabasePredictiveAnalytics
from test_data_generator import TestDataGenerator

class TestDatabasePredictiveAnalytics(unittest.TestCase):
    """Test cases for the database predictive analytics module."""
    
    def setUp(self):
        """Set up test environment before each test method."""
        # Create a temporary directory for parameter storage
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a data generator with a fixed seed for reproducibility
        self.data_generator = TestDataGenerator(seed=42)
        
        # Create test data
        self.test_df = self._create_test_dataframe()
        
        # Initialize the analyzer with custom configuration
        self.config = {
            'parameter_persistence': {
                'enabled': True,
                'storage_path': self.temp_dir,
                'format': 'json',
                'max_age_days': 30,
                'revalidate_after_days': 7,
                'cache_in_memory': True
            }
        }
        self.analyzer = DatabasePredictiveAnalytics(config=self.config)
    
    def tearDown(self):
        """Clean up test environment after each test method."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def _create_test_dataframe(self):
        """Create a test dataframe with time series data for forecasting."""
        # Create a dataframe with a time series
        dates = pd.date_range(start='2025-01-01', periods=100, freq='D')
        
        # Create a time series with trend, seasonality, and noise
        trend = np.linspace(0, 10, 100)  # Upward trend
        seasonality = 5 * np.sin(np.linspace(0, 10 * np.pi, 100))  # Seasonal component
        noise = np.random.normal(0, 1, 100)  # Random noise
        
        values = trend + seasonality + noise
        
        # Create the dataframe
        df = pd.DataFrame({
            'date': dates,
            'value': values,
            'metric_name': 'test_metric',
            'model_name': 'bert-base-uncased',
            'hardware_name': 'gpu_rtx3080'
        })
        
        return df
    
    def test_parameter_storage_path(self):
        """Test the parameter storage path creation."""
        # Test that the path gets created correctly
        path = self.analyzer._get_parameter_storage_path()
        self.assertEqual(path, self.temp_dir)
        self.assertTrue(os.path.exists(path))
    
    def test_parameter_key_generation(self):
        """Test the generation of parameter keys."""
        # Test key generation with different inputs
        key1 = self.analyzer._get_parameter_key('arima', 'test_metric', {})
        key2 = self.analyzer._get_parameter_key('arima', 'another_metric', {})
        key3 = self.analyzer._get_parameter_key('exponential_smoothing', 'test_metric', {})
        
        # Keys should be different for different inputs
        self.assertNotEqual(key1, key2)
        self.assertNotEqual(key1, key3)
        self.assertNotEqual(key2, key3)
        
        # Same inputs should produce the same key
        key1_duplicate = self.analyzer._get_parameter_key('arima', 'test_metric', {})
        self.assertEqual(key1, key1_duplicate)
    
    def test_data_signature_creation(self):
        """Test the creation of data signatures."""
        # Test signature for a dataframe
        signature = self.analyzer._create_data_signature(self.test_df)
        
        # Signature should be a string
        self.assertIsInstance(signature, str)
        self.assertTrue(len(signature) > 0)
        
        # Same dataframe should produce the same signature
        signature_duplicate = self.analyzer._create_data_signature(self.test_df)
        self.assertEqual(signature, signature_duplicate)
        
        # Modified dataframe should produce a different signature
        df_modified = self.test_df.copy()
        df_modified.loc[0, 'value'] = 9999.0
        signature_modified = self.analyzer._create_data_signature(df_modified)
        self.assertNotEqual(signature, signature_modified)
    
    def test_parameter_save_load(self):
        """Test saving and loading parameters."""
        # Create test parameters
        model_type = 'arima'
        metric_name = 'test_metric'
        params = {
            'p': 2,
            'd': 1,
            'q': 2,
            'performance': 0.95,
            'metric_name': metric_name
        }
        
        # Save parameters
        data_signature = self.analyzer._create_data_signature(self.test_df)
        save_result = self.analyzer._save_parameters(
            model_type, 
            metric_name, 
            params,
            data_signature
        )
        
        # Test save result
        self.assertTrue(save_result)
        
        # Load parameters
        loaded_params = self.analyzer._load_parameters(
            model_type,
            metric_name,
            data_signature
        )
        
        # Test loaded parameters
        self.assertIsNotNone(loaded_params)
        self.assertEqual(loaded_params['p'], params['p'])
        self.assertEqual(loaded_params['d'], params['d'])
        self.assertEqual(loaded_params['q'], params['q'])
        self.assertEqual(loaded_params['performance'], params['performance'])
        self.assertEqual(loaded_params['metric_name'], params['metric_name'])
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Create test parameters
        model_type = 'arima'
        metric_name = 'test_metric'
        data_signature = self.analyzer._create_data_signature(self.test_df)
        
        # Create a valid parameter set (recent)
        valid_params = {
            'p': 2,
            'd': 1,
            'q': 2,
            'performance': 0.95,
            'metric_name': metric_name,
            'data_signature': data_signature,
            'timestamp': datetime.now().isoformat()
        }
        
        # Test validation of valid parameters
        is_valid = self.analyzer._validate_parameters(
            valid_params,
            data_signature,
            force_revalidation=False
        )
        self.assertTrue(is_valid)
        
        # Create an old parameter set
        old_timestamp = (datetime.now() - timedelta(days=60)).isoformat()
        old_params = valid_params.copy()
        old_params['timestamp'] = old_timestamp
        
        # Test validation of old parameters
        is_valid = self.analyzer._validate_parameters(
            old_params,
            data_signature,
            force_revalidation=False
        )
        self.assertFalse(is_valid)
        
        # Create a parameter set with different signature
        diff_signature_params = valid_params.copy()
        diff_signature_params['data_signature'] = 'different_signature'
        
        # Test validation of parameters with different signature
        is_valid = self.analyzer._validate_parameters(
            diff_signature_params,
            data_signature,
            force_revalidation=False
        )
        self.assertFalse(is_valid)
        
        # Test force revalidation
        is_valid = self.analyzer._validate_parameters(
            valid_params,
            data_signature,
            force_revalidation=True
        )
        self.assertFalse(is_valid)
    
    def test_parameter_persistence_end_to_end_arima(self):
        """Test end-to-end parameter persistence for ARIMA forecasting."""
        # Create a test dataframe
        test_df = self.test_df.copy()
        metric_name = 'test_metric'
        model_type = 'arima'
        
        # First run - should tune parameters
        result1 = self.analyzer.forecast_time_series(
            test_df,
            metric_name=metric_name,
            model_type=model_type,
            forecast_days=7
        )
        
        # Check if forecast was generated
        self.assertIsNotNone(result1)
        self.assertTrue('forecast' in result1)
        self.assertTrue('model_params' in result1)
        
        # Run again - should use saved parameters
        result2 = self.analyzer.forecast_time_series(
            test_df,
            metric_name=metric_name,
            model_type=model_type,
            forecast_days=7
        )
        
        # Check if forecast was generated
        self.assertIsNotNone(result2)
        self.assertTrue('forecast' in result2)
        self.assertTrue('model_params' in result2)
        
        # Parameters should be the same
        self.assertEqual(result1['model_params'], result2['model_params'])
        
        # Force revalidation - should tune parameters again
        result3 = self.analyzer.forecast_time_series(
            test_df,
            metric_name=metric_name,
            model_type=model_type,
            forecast_days=7,
            force_parameter_revalidation=True
        )
        
        # Check if forecast was generated
        self.assertIsNotNone(result3)
        self.assertTrue('forecast' in result3)
        self.assertTrue('model_params' in result3)
    
    def test_parameter_persistence_end_to_end_exponential_smoothing(self):
        """Test end-to-end parameter persistence for Exponential Smoothing forecasting."""
        # Create a test dataframe
        test_df = self.test_df.copy()
        metric_name = 'test_metric_exp'
        model_type = 'exponential_smoothing'
        
        # First run - should tune parameters
        result1 = self.analyzer.forecast_time_series(
            test_df,
            metric_name=metric_name,
            model_type=model_type,
            forecast_days=7
        )
        
        # Check if forecast was generated
        self.assertIsNotNone(result1)
        self.assertTrue('forecast' in result1)
        self.assertTrue('model_params' in result1)
        
        # Run again - should use saved parameters
        result2 = self.analyzer.forecast_time_series(
            test_df,
            metric_name=metric_name,
            model_type=model_type,
            forecast_days=7
        )
        
        # Check if forecast was generated
        self.assertIsNotNone(result2)
        self.assertTrue('forecast' in result2)
        self.assertTrue('model_params' in result2)
        
        # Parameters should be the same
        self.assertEqual(result1['model_params'], result2['model_params'])
    
    def test_parameter_persistence_end_to_end_linear_regression(self):
        """Test end-to-end parameter persistence for Linear Regression forecasting."""
        # Create a test dataframe
        test_df = self.test_df.copy()
        metric_name = 'test_metric_linear'
        model_type = 'linear_regression'
        
        # First run - should tune parameters
        result1 = self.analyzer.forecast_time_series(
            test_df,
            metric_name=metric_name,
            model_type=model_type,
            forecast_days=7
        )
        
        # Check if forecast was generated
        self.assertIsNotNone(result1)
        self.assertTrue('forecast' in result1)
        self.assertTrue('model_params' in result1)
        
        # Run again - should use saved parameters
        result2 = self.analyzer.forecast_time_series(
            test_df,
            metric_name=metric_name,
            model_type=model_type,
            forecast_days=7
        )
        
        # Check if forecast was generated
        self.assertIsNotNone(result2)
        self.assertTrue('forecast' in result2)
        self.assertTrue('model_params' in result2)
        
        # Parameters should be the same
        self.assertEqual(result1['model_params'], result2['model_params'])
    
    def test_storage_format_json(self):
        """Test JSON storage format for parameters."""
        # Set JSON format
        self.analyzer.config['parameter_persistence']['format'] = 'json'
        
        # Create test parameters
        model_type = 'arima'
        metric_name = 'test_metric_json'
        params = {
            'p': 2,
            'd': 1,
            'q': 2,
            'performance': 0.95,
            'metric_name': metric_name
        }
        
        # Save parameters
        data_signature = self.analyzer._create_data_signature(self.test_df)
        save_result = self.analyzer._save_parameters(
            model_type, 
            metric_name, 
            params,
            data_signature
        )
        
        # Test save result
        self.assertTrue(save_result)
        
        # Get the key
        key = self.analyzer._get_parameter_key(model_type, metric_name, {})
        
        # Check that the file was created with JSON extension
        file_path = os.path.join(self.temp_dir, f"{key}.json")
        self.assertTrue(os.path.exists(file_path))
        
        # Load the file directly
        with open(file_path, 'r') as f:
            loaded_data = json.load(f)
        
        # Check the loaded data
        self.assertEqual(loaded_data['p'], params['p'])
        self.assertEqual(loaded_data['d'], params['d'])
        self.assertEqual(loaded_data['q'], params['q'])
    
    def test_storage_format_pickle(self):
        """Test pickle storage format for parameters."""
        # Set pickle format
        self.analyzer.config['parameter_persistence']['format'] = 'pickle'
        
        # Create test parameters
        model_type = 'arima'
        metric_name = 'test_metric_pickle'
        params = {
            'p': 2,
            'd': 1,
            'q': 2,
            'performance': 0.95,
            'metric_name': metric_name
        }
        
        # Save parameters
        data_signature = self.analyzer._create_data_signature(self.test_df)
        save_result = self.analyzer._save_parameters(
            model_type, 
            metric_name, 
            params,
            data_signature
        )
        
        # Test save result
        self.assertTrue(save_result)
        
        # Get the key
        key = self.analyzer._get_parameter_key(model_type, metric_name, {})
        
        # Check that the file was created with pickle extension
        file_path = os.path.join(self.temp_dir, f"{key}.pkl")
        self.assertTrue(os.path.exists(file_path))
        
        # Load the file directly
        with open(file_path, 'rb') as f:
            loaded_data = pickle.load(f)
        
        # Check the loaded data
        self.assertEqual(loaded_data['p'], params['p'])
        self.assertEqual(loaded_data['d'], params['d'])
        self.assertEqual(loaded_data['q'], params['q'])
    
    def test_clean_parameter_storage(self):
        """Test cleaning parameter storage."""
        # Create test parameters
        model_type = 'arima'
        metrics = ['metric1', 'metric2', 'metric3']
        data_signature = self.analyzer._create_data_signature(self.test_df)
        
        # Save parameters for multiple metrics
        for metric in metrics:
            params = {
                'p': 2,
                'd': 1,
                'q': 2,
                'performance': 0.95,
                'metric_name': metric
            }
            self.analyzer._save_parameters(model_type, metric, params, data_signature)
        
        # Verify files were created
        for metric in metrics:
            key = self.analyzer._get_parameter_key(model_type, metric, {})
            file_path = os.path.join(self.temp_dir, f"{key}.json")
            self.assertTrue(os.path.exists(file_path))
        
        # Clean parameters
        self.analyzer.clean_parameter_storage()
        
        # Verify files were deleted
        for metric in metrics:
            key = self.analyzer._get_parameter_key(model_type, metric, {})
            file_path = os.path.join(self.temp_dir, f"{key}.json")
            self.assertFalse(os.path.exists(file_path))
    
    def test_memory_cache(self):
        """Test in-memory caching of parameters."""
        # Ensure memory cache is enabled
        self.analyzer.config['parameter_persistence']['cache_in_memory'] = True
        
        # Create test parameters
        model_type = 'arima'
        metric_name = 'test_metric_cache'
        params = {
            'p': 2,
            'd': 1,
            'q': 2,
            'performance': 0.95,
            'metric_name': metric_name
        }
        
        # Save parameters
        data_signature = self.analyzer._create_data_signature(self.test_df)
        self.analyzer._save_parameters(model_type, metric_name, params, data_signature)
        
        # Get the key
        key = self.analyzer._get_parameter_key(model_type, metric_name, {})
        
        # Check that parameters are in memory cache
        self.assertTrue(key in self.analyzer.hyperparameter_cache)
        
        # Load parameters from cache
        loaded_params = self.analyzer._load_parameters(model_type, metric_name, data_signature)
        
        # Check loaded parameters
        self.assertEqual(loaded_params['p'], params['p'])
        
        # Disable memory cache
        self.analyzer.config['parameter_persistence']['cache_in_memory'] = False
        
        # Clear the cache
        self.analyzer.hyperparameter_cache = {}
        
        # Load parameters from disk
        loaded_params = self.analyzer._load_parameters(model_type, metric_name, data_signature)
        
        # Check loaded parameters
        self.assertEqual(loaded_params['p'], params['p'])

if __name__ == '__main__':
    unittest.main()