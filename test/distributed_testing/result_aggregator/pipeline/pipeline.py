#!/usr/bin/env python3
"""
Data Processing Pipeline Module for Result Aggregator

This module provides a flexible pipeline framework for processing and transforming
test results data. It allows creating customizable processing pipelines with
composable transformation stages.

Usage:
    from result_aggregator.pipeline.pipeline import ProcessingPipeline, DataSource
    from result_aggregator.pipeline.transforms import FilterTransform, AggregateTransform
    
    # Create a pipeline for processing test results
    pipeline = ProcessingPipeline(
        name="Performance Analysis Pipeline",
        data_source=DataSource(db_path='path/to/benchmark_db.duckdb'),
        transforms=[
            FilterTransform(model_name='bert-base-uncased'),
            AggregateTransform(group_by='hardware_type', metrics=['latency', 'throughput'])
        ]
    )
    
    # Execute the pipeline
    results = pipeline.execute()
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
import json
import inspect
from abc import ABC, abstractmethod
from pathlib import Path

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

logger = logging.getLogger(__name__)


class DataSource:
    """
    Data source for the processing pipeline.
    
    Provides an interface to retrieve data from various sources such as
    DuckDB database, CSV files, JSON files, or other data sources.
    """
    
    def __init__(self, 
                db_path: Optional[str] = None, 
                connection: Optional[Any] = None,
                file_path: Optional[str] = None,
                data: Optional[pd.DataFrame] = None,
                query: Optional[str] = None,
                query_params: Optional[List[Any]] = None):
        """
        Initialize the data source.
        
        Args:
            db_path: Path to the DuckDB database file
            connection: Existing database connection
            file_path: Path to a data file (CSV, JSON, etc.)
            data: A pandas DataFrame to use directly
            query: Custom SQL query to execute
            query_params: Parameters for the custom query
        
        Raises:
            ValueError: If no data source is provided
        """
        self.db_path = db_path
        self._conn = connection
        self.file_path = file_path
        self.data = data
        self.query = query
        self.query_params = query_params or []
        
        if not any([db_path, connection, file_path, data is not None]):
            raise ValueError("At least one data source must be provided")
        
        # Initialize connection if needed
        if db_path and not connection and DUCKDB_AVAILABLE:
            self._conn = duckdb.connect(db_path)
    
    def get_data(self, custom_query: Optional[str] = None, 
                query_params: Optional[List[Any]] = None) -> pd.DataFrame:
        """
        Retrieve data from the configured source.
        
        Args:
            custom_query: Optional custom query to override the default query
            query_params: Parameters for the custom query
        
        Returns:
            A pandas DataFrame with the data
        
        Raises:
            RuntimeError: If data retrieval fails
        """
        try:
            # Handle direct data first
            if self.data is not None:
                return self.data.copy()
            
            # Handle database sources
            if self._conn:
                query = custom_query or self.query or """
                    SELECT r.*, m.metric_name, m.metric_value
                    FROM test_results r
                    JOIN performance_metrics m ON r.result_id = m.result_id
                """
                params = query_params or self.query_params
                
                return self._conn.execute(query, params).fetchdf()
                
            # Handle file sources
            if self.file_path:
                file_ext = os.path.splitext(self.file_path)[1].lower()
                
                if file_ext == '.csv':
                    return pd.read_csv(self.file_path)
                elif file_ext in ['.json', '.jsonl']:
                    return pd.read_json(self.file_path)
                elif file_ext == '.parquet':
                    return pd.read_parquet(self.file_path)
                elif file_ext in ['.xlsx', '.xls']:
                    return pd.read_excel(self.file_path)
                else:
                    raise ValueError(f"Unsupported file format: {file_ext}")
            
            raise RuntimeError("No valid data source configured")
            
        except Exception as e:
            logger.error(f"Error retrieving data: {str(e)}")
            raise RuntimeError(f"Failed to retrieve data: {str(e)}")
    
    def close(self):
        """Close the database connection if it exists."""
        if hasattr(self, '_conn') and self._conn and not self._conn.closed:
            self._conn.close()
    
    def __del__(self):
        """Ensure the database connection is closed when the object is deleted."""
        self.close()


class Transform(ABC):
    """
    Abstract base class for data transforms in the processing pipeline.
    
    A transform takes a DataFrame as input, applies some transformation,
    and returns a transformed DataFrame.
    """
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the transformation to the input data.
        
        Args:
            data: Input DataFrame
        
        Returns:
            Transformed DataFrame
        """
        pass
    
    def get_name(self) -> str:
        """Get the name of the transform for logging and reporting."""
        return self.__class__.__name__


class ProcessingPipeline:
    """
    A pipeline for processing test results data.
    
    The pipeline applies a sequence of transforms to the data
    from the specified data source.
    """
    
    def __init__(self, 
                name: str,
                data_source: DataSource,
                transforms: List[Transform],
                error_handler: Optional[Callable[[Exception, str], None]] = None):
        """
        Initialize the processing pipeline.
        
        Args:
            name: Name of the pipeline
            data_source: Data source for the pipeline
            transforms: List of transforms to apply
            error_handler: Optional function to handle errors
        """
        self.name = name
        self.data_source = data_source
        self.transforms = transforms
        self.error_handler = error_handler or self._default_error_handler
        self.start_time = None
        self.end_time = None
        self.processed_rows = 0
        self.success = False
        self.stage_metrics = {}
    
    def _default_error_handler(self, exception: Exception, stage: str):
        """
        Default error handler for pipeline errors.
        
        Args:
            exception: The exception that occurred
            stage: The pipeline stage where the error occurred
        """
        logger.error(f"Error in pipeline '{self.name}' at stage '{stage}': {str(exception)}")
    
    def execute(self, 
               custom_query: Optional[str] = None, 
               query_params: Optional[List[Any]] = None) -> pd.DataFrame:
        """
        Execute the pipeline by retrieving data and applying all transforms.
        
        Args:
            custom_query: Optional custom query to override the default query
            query_params: Parameters for the custom query
        
        Returns:
            The final transformed DataFrame
        
        Raises:
            RuntimeError: If the pipeline execution fails
        """
        self.start_time = datetime.now()
        self.success = False
        self.stage_metrics = {}
        
        try:
            logger.info(f"Starting pipeline '{self.name}'")
            
            # Retrieve data from the source
            logger.info(f"Retrieving data from source")
            stage_start = datetime.now()
            data = self.data_source.get_data(custom_query, query_params)
            stage_end = datetime.now()
            
            self.processed_rows = len(data)
            self.stage_metrics['data_retrieval'] = {
                'duration_seconds': (stage_end - stage_start).total_seconds(),
                'rows': len(data)
            }
            
            logger.info(f"Retrieved {len(data)} rows from data source")
            
            # Apply each transform
            for i, transform in enumerate(self.transforms):
                transform_name = transform.get_name()
                stage_name = f"transform_{i}_{transform_name}"
                
                try:
                    logger.info(f"Applying transform: {transform_name}")
                    stage_start = datetime.now()
                    data = transform.transform(data)
                    stage_end = datetime.now()
                    
                    self.stage_metrics[stage_name] = {
                        'duration_seconds': (stage_end - stage_start).total_seconds(),
                        'rows': len(data),
                        'transform': transform_name
                    }
                    
                    logger.info(f"Transform {transform_name} completed: {len(data)} rows")
                    
                except Exception as e:
                    self.error_handler(e, stage_name)
                    raise RuntimeError(f"Transform {transform_name} failed: {str(e)}")
            
            self.success = True
            self.end_time = datetime.now()
            
            logger.info(f"Pipeline '{self.name}' completed successfully")
            
            return data
            
        except Exception as e:
            self.end_time = datetime.now()
            logger.error(f"Pipeline '{self.name}' failed: {str(e)}")
            raise RuntimeError(f"Pipeline execution failed: {str(e)}")
    
    def get_metrics(self) -> Dict:
        """
        Get metrics about the pipeline execution.
        
        Returns:
            Dictionary with pipeline execution metrics
        """
        total_duration = None
        if self.start_time and self.end_time:
            total_duration = (self.end_time - self.start_time).total_seconds()
        
        return {
            'name': self.name,
            'success': self.success,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_duration_seconds': total_duration,
            'processed_rows': self.processed_rows,
            'stages': self.stage_metrics
        }


class PipelineManager:
    """
    Manager for creating, executing, and maintaining processing pipelines.
    
    Provides functionality to create pipelines from configurations,
    execute multiple pipelines, and manage pipeline results.
    """
    
    def __init__(self, db_path: Optional[str] = None, connection: Optional[Any] = None):
        """
        Initialize the pipeline manager.
        
        Args:
            db_path: Path to the DuckDB database file
            connection: Existing database connection
        """
        self.db_path = db_path
        self._conn = connection
        self.pipelines = {}
        self.results = {}
        
        # Initialize connection if needed
        if db_path and not connection and DUCKDB_AVAILABLE:
            self._conn = duckdb.connect(db_path)
    
    def create_pipeline(self, 
                      name: str,
                      transforms: List[Transform],
                      custom_data_source: Optional[DataSource] = None) -> ProcessingPipeline:
        """
        Create a new processing pipeline.
        
        Args:
            name: Name of the pipeline
            transforms: List of transforms to apply
            custom_data_source: Optional custom data source
        
        Returns:
            The created processing pipeline
        """
        data_source = custom_data_source or DataSource(db_path=self.db_path, connection=self._conn)
        
        pipeline = ProcessingPipeline(
            name=name,
            data_source=data_source,
            transforms=transforms
        )
        
        self.pipelines[name] = pipeline
        
        return pipeline
    
    def create_pipeline_from_config(self, config: Dict) -> ProcessingPipeline:
        """
        Create a pipeline from a configuration dictionary.
        
        Args:
            config: Pipeline configuration dictionary
        
        Returns:
            The created processing pipeline
        
        Raises:
            ValueError: If the configuration is invalid
        """
        if 'name' not in config:
            raise ValueError("Pipeline configuration must include a name")
        
        name = config['name']
        
        # Create transforms from config
        transforms = []
        for transform_config in config.get('transforms', []):
            transform_type = transform_config.get('type')
            if not transform_type:
                raise ValueError(f"Transform configuration must include a type")
            
            # Import the transform class
            try:
                # Assume transforms are in the result_aggregator.pipeline.transforms module
                from result_aggregator.pipeline.transforms import get_transform_by_name
                transform = get_transform_by_name(transform_type, **transform_config.get('params', {}))
                transforms.append(transform)
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Failed to import transform {transform_type}: {str(e)}")
        
        # Create data source from config
        data_source_config = config.get('data_source', {})
        data_source = None
        
        if data_source_config:
            data_source = DataSource(
                db_path=data_source_config.get('db_path', self.db_path),
                connection=self._conn,
                file_path=data_source_config.get('file_path'),
                query=data_source_config.get('query'),
                query_params=data_source_config.get('query_params')
            )
        else:
            data_source = DataSource(db_path=self.db_path, connection=self._conn)
        
        # Create and store the pipeline
        pipeline = ProcessingPipeline(
            name=name,
            data_source=data_source,
            transforms=transforms
        )
        
        self.pipelines[name] = pipeline
        
        return pipeline
    
    def execute_pipeline(self, 
                       name: str,
                       custom_query: Optional[str] = None, 
                       query_params: Optional[List[Any]] = None) -> pd.DataFrame:
        """
        Execute a pipeline by name.
        
        Args:
            name: Name of the pipeline to execute
            custom_query: Optional custom query to override the default query
            query_params: Parameters for the custom query
        
        Returns:
            The final transformed DataFrame
        
        Raises:
            ValueError: If the pipeline does not exist
        """
        if name not in self.pipelines:
            raise ValueError(f"Pipeline '{name}' does not exist")
        
        pipeline = self.pipelines[name]
        
        try:
            result = pipeline.execute(custom_query, query_params)
            self.results[name] = {
                'data': result,
                'metrics': pipeline.get_metrics(),
                'timestamp': datetime.now().isoformat()
            }
            return result
        except Exception as e:
            logger.error(f"Error executing pipeline '{name}': {str(e)}")
            self.results[name] = {
                'error': str(e),
                'metrics': pipeline.get_metrics(),
                'timestamp': datetime.now().isoformat()
            }
            raise
    
    def execute_all_pipelines(self) -> Dict[str, pd.DataFrame]:
        """
        Execute all registered pipelines.
        
        Returns:
            Dictionary mapping pipeline names to their results
        """
        results = {}
        
        for name in self.pipelines:
            try:
                results[name] = self.execute_pipeline(name)
            except Exception as e:
                logger.error(f"Error executing pipeline '{name}': {str(e)}")
                results[name] = None
        
        return results
    
    def get_pipeline_metrics(self, name: Optional[str] = None) -> Dict:
        """
        Get metrics about pipeline executions.
        
        Args:
            name: Optional name of a specific pipeline
        
        Returns:
            Dictionary with pipeline execution metrics
        """
        if name:
            if name not in self.results:
                return {'error': f"No results found for pipeline '{name}'"}
            
            return self.results[name].get('metrics', {})
        else:
            return {name: result.get('metrics', {}) for name, result in self.results.items()}
    
    def save_pipeline_config(self, name: str, file_path: str) -> bool:
        """
        Save a pipeline configuration to a file.
        
        Args:
            name: Name of the pipeline
            file_path: Path to save the configuration
        
        Returns:
            True if successful, False otherwise
        """
        if name not in self.pipelines:
            logger.error(f"Pipeline '{name}' does not exist")
            return False
        
        pipeline = self.pipelines[name]
        
        # Create a configuration dictionary
        config = {
            'name': pipeline.name,
            'transforms': []
        }
        
        # Add transforms
        for transform in pipeline.transforms:
            transform_config = {
                'type': transform.__class__.__name__
            }
            
            # Try to extract parameters using inspection
            try:
                params = {}
                signature = inspect.signature(transform.__class__.__init__)
                for param_name, param in signature.parameters.items():
                    if param_name not in ['self', 'args', 'kwargs'] and hasattr(transform, param_name):
                        params[param_name] = getattr(transform, param_name)
                
                transform_config['params'] = params
            except Exception as e:
                logger.warning(f"Could not extract parameters for transform {transform.__class__.__name__}: {str(e)}")
            
            config['transforms'].append(transform_config)
        
        # Add data source configuration if possible
        data_source = pipeline.data_source
        if data_source:
            data_source_config = {}
            
            if data_source.db_path:
                data_source_config['db_path'] = data_source.db_path
            
            if data_source.file_path:
                data_source_config['file_path'] = data_source.file_path
            
            if data_source.query:
                data_source_config['query'] = data_source.query
            
            if data_source.query_params:
                data_source_config['query_params'] = data_source.query_params
            
            config['data_source'] = data_source_config
        
        # Save the configuration
        try:
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Pipeline configuration saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving pipeline configuration: {str(e)}")
            return False
    
    def load_pipeline_config(self, file_path: str) -> Optional[ProcessingPipeline]:
        """
        Load a pipeline configuration from a file.
        
        Args:
            file_path: Path to the configuration file
        
        Returns:
            The loaded pipeline, or None if loading failed
        """
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
            
            return self.create_pipeline_from_config(config)
        except Exception as e:
            logger.error(f"Error loading pipeline configuration: {str(e)}")
            return None
    
    def save_pipeline_result(self, name: str, file_path: str, format: str = 'csv') -> bool:
        """
        Save a pipeline result to a file.
        
        Args:
            name: Name of the pipeline
            file_path: Path to save the result
            format: Format to save the result ('csv', 'json', 'parquet', 'excel')
        
        Returns:
            True if successful, False otherwise
        """
        if name not in self.results or 'data' not in self.results[name]:
            logger.error(f"No data results found for pipeline '{name}'")
            return False
        
        data = self.results[name]['data']
        
        try:
            format = format.lower()
            
            if format == 'csv':
                data.to_csv(file_path, index=False)
            elif format == 'json':
                data.to_json(file_path, orient='records', lines=True)
            elif format == 'parquet':
                data.to_parquet(file_path, index=False)
            elif format in ['excel', 'xlsx']:
                data.to_excel(file_path, index=False)
            else:
                logger.error(f"Unsupported format: {format}")
                return False
            
            logger.info(f"Pipeline result saved to {file_path} in {format} format")
            return True
        except Exception as e:
            logger.error(f"Error saving pipeline result: {str(e)}")
            return False
    
    def close(self):
        """Close database connections."""
        if hasattr(self, '_conn') and self._conn and not self._conn.closed:
            self._conn.close()
    
    def __del__(self):
        """Ensure database connections are closed when the object is deleted."""
        self.close()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Example usage
    try:
        # Import transforms
        from result_aggregator.pipeline.transforms import (
            FilterTransform, 
            AggregateTransform, 
            TimeWindowTransform
        )
        
        # Create a data source
        data_source = DataSource(db_path="benchmark_db.duckdb")
        
        # Create a pipeline
        pipeline = ProcessingPipeline(
            name="Performance Analysis Pipeline",
            data_source=data_source,
            transforms=[
                FilterTransform(model_name='bert-base-uncased'),
                TimeWindowTransform(days=30),
                AggregateTransform(group_by='hardware_type', metrics=['latency', 'throughput'])
            ]
        )
        
        # Execute the pipeline
        results = pipeline.execute()
        print(f"Pipeline executed successfully, processed {len(results)} rows")
        
        # Get pipeline metrics
        metrics = pipeline.get_metrics()
        print(f"Pipeline metrics: {json.dumps(metrics, indent=2)}")
        
        # Create a pipeline manager
        manager = PipelineManager(db_path="benchmark_db.duckdb")
        
        # Create a pipeline using the manager
        manager.create_pipeline(
            name="Hardware Comparison",
            transforms=[
                TimeWindowTransform(days=30),
                AggregateTransform(group_by=['model_name', 'hardware_type'], 
                                  metrics=['latency', 'throughput'])
            ]
        )
        
        # Execute the pipeline
        results = manager.execute_pipeline("Hardware Comparison")
        print(f"Pipeline executed successfully, processed {len(results)} rows")
        
        # Save the pipeline configuration
        manager.save_pipeline_config("Hardware Comparison", "hardware_comparison_pipeline.json")
        
        # Save the pipeline results
        manager.save_pipeline_result("Hardware Comparison", "hardware_comparison_results.csv")
        
    except Exception as e:
        logging.error(f"Error in example: {str(e)}")
    finally:
        # Clean up connections
        if 'data_source' in locals():
            data_source.close()
        if 'manager' in locals():
            manager.close()