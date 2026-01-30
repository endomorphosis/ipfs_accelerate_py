"""
Repository adapter for integrating the predictive performance system with DuckDB.

This module provides adapter classes that connect the existing predictive performance components
with the DuckDB repository for persistent storage and retrieval.
"""

import logging
import uuid
import time
import joblib
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime
import json

from .predictor_repository import DuckDBPredictorRepository

# Try to import from the predictive_performance package
try:
    from predictive_performance.hardware_model_predictor import HardwareModelPredictor
    HARDWARE_MODEL_PREDICTOR_AVAILABLE = True
    from predictive_performance.model_performance_predictor import (
        load_prediction_models, predict_performance, generate_prediction_matrix
    )
    MODEL_PERFORMANCE_PREDICTOR_AVAILABLE = True
except ImportError:
    HARDWARE_MODEL_PREDICTOR_AVAILABLE = False
    MODEL_PERFORMANCE_PREDICTOR_AVAILABLE = False

# Setup logger
logger = logging.getLogger(__name__)

class HardwareModelPredictorDuckDBAdapter:
    """
    Adapter for integrating the HardwareModelPredictor with DuckDB repository.
    
    This class wraps the HardwareModelPredictor and connects it to the DuckDB 
    repository for persisting predictions, hardware-model mappings, and recommendations.
    """
    
    def __init__(
        self,
        predictor: Optional[HardwareModelPredictor] = None,
        repository: DuckDBPredictorRepository = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the adapter.
        
        Args:
            predictor: The HardwareModelPredictor instance to adapt
            repository: The DuckDB repository for storing results
            user_id: Optional user ID for tracking recommendations
            metadata: Optional metadata for this instance
        """
        self.predictor = predictor
        self.repository = repository
        self.user_id = user_id or "anonymous"
        self.metadata = metadata or {}
        
        # Initialize predictor if not provided and available
        if self.predictor is None and HARDWARE_MODEL_PREDICTOR_AVAILABLE:
            try:
                self.predictor = HardwareModelPredictor()
                logger.info("Initialized HardwareModelPredictor")
            except Exception as e:
                logger.warning(f"Failed to initialize HardwareModelPredictor: {e}")
        
        # Initialize repository if not provided
        if self.repository is None:
            try:
                self.repository = DuckDBPredictorRepository()
                logger.info("Initialized DuckDBPredictorRepository")
            except Exception as e:
                logger.error(f"Failed to initialize DuckDBPredictorRepository: {e}")
                raise
    
    def predict_optimal_hardware(self,
                               model_name: str,
                               model_family: Optional[str] = None,
                               batch_size: int = 1,
                               sequence_length: int = 128,
                               mode: str = "inference",
                               precision: str = "fp32",
                               available_hardware: Optional[List[str]] = None,
                               store_recommendation: bool = True) -> Dict[str, Any]:
        """
        Predict the optimal hardware and store the recommendation in the repository.
        
        Args:
            model_name: Name of the model
            model_family: Optional model family/category
            batch_size: Batch size
            sequence_length: Sequence length
            mode: "inference" or "training"
            precision: Precision to use (fp32, fp16, int8)
            available_hardware: Optional list of available hardware types
            store_recommendation: Whether to store the recommendation in the repository
            
        Returns:
            Dictionary with hardware recommendation and performance predictions
        """
        if self.predictor is None:
            raise RuntimeError("HardwareModelPredictor is not available")
        
        # Call the predictor
        recommendation = self.predictor.predict_optimal_hardware(
            model_name=model_name,
            model_family=model_family,
            batch_size=batch_size,
            sequence_length=sequence_length,
            mode=mode,
            precision=precision,
            available_hardware=available_hardware
        )
        
        # Store the recommendation if requested
        if store_recommendation and self.repository is not None:
            try:
                # Generate a unique recommendation ID
                recommendation_id = f"rec-{int(time.time())}-{hash(model_name)}"
                
                # Create repository record
                repo_recommendation = {
                    'timestamp': datetime.now(),
                    'user_id': self.user_id,
                    'model_name': model_name,
                    'model_family': recommendation.get('model_family', model_family),
                    'batch_size': batch_size,
                    'sequence_length': sequence_length,
                    'precision': precision,
                    'mode': mode,
                    'primary_recommendation': recommendation.get('primary_recommendation'),
                    'fallback_options': recommendation.get('fallback_options', []),
                    'compatible_hardware': recommendation.get('compatible_hardware', []),
                    'reason': recommendation.get('explanation'),
                    'recommendation_id': recommendation_id,
                    'was_accepted': None,  # Will be updated later if feedback is provided
                    'user_feedback': None,
                    'metadata': {
                        **self.metadata,
                        'model_size': recommendation.get('model_size'),
                        'model_size_category': recommendation.get('model_size_category'),
                        'prediction_source': recommendation.get('prediction_source')
                    }
                }
                
                self.repository.store_recommendation(repo_recommendation)
                
                # Store hardware-model mappings
                primary_hw = recommendation.get('primary_recommendation')
                compatible_hw = recommendation.get('compatible_hardware', [])
                
                # Store primary recommendation as top mapping
                if primary_hw:
                    primary_mapping = {
                        'timestamp': datetime.now(),
                        'model_name': model_name,
                        'model_family': recommendation.get('model_family', model_family),
                        'hardware_platform': primary_hw,
                        'compatibility_score': 0.95,  # Assumed high score for primary
                        'recommendation_rank': 1,
                        'is_primary_recommendation': True,
                        'reason': recommendation.get('explanation'),
                        'mapping_id': f"map-{recommendation_id}-{hash(primary_hw)}",
                        'metadata': {
                            **self.metadata,
                            'recommendation_id': recommendation_id
                        }
                    }
                    
                    self.repository.store_hardware_model_mapping(primary_mapping)
                
                # Store additional compatible hardware as lower-ranked mappings
                for i, hw in enumerate([h for h in compatible_hw if h != primary_hw]):
                    # Calculate a lower compatibility score for fallback options
                    compatibility_score = max(0.9 - (i * 0.1), 0.5)
                    
                    fallback_mapping = {
                        'timestamp': datetime.now(),
                        'model_name': model_name,
                        'model_family': recommendation.get('model_family', model_family),
                        'hardware_platform': hw,
                        'compatibility_score': compatibility_score,
                        'recommendation_rank': i + 2,  # Start at rank 2 (after primary)
                        'is_primary_recommendation': False,
                        'reason': f"Fallback option {i+1} for {model_name}",
                        'mapping_id': f"map-{recommendation_id}-{hash(hw)}",
                        'metadata': {
                            **self.metadata,
                            'recommendation_id': recommendation_id
                        }
                    }
                    
                    self.repository.store_hardware_model_mapping(fallback_mapping)
                
                # Add recommendation ID to the result
                recommendation['recommendation_id'] = recommendation_id
                
            except Exception as e:
                logger.error(f"Error storing recommendation: {e}")
        
        return recommendation
    
    def record_recommendation_feedback(self,
                                     recommendation_id: str,
                                     was_accepted: bool,
                                     user_feedback: Optional[str] = None) -> bool:
        """
        Record feedback for a hardware recommendation.
        
        Args:
            recommendation_id: ID of the recommendation
            was_accepted: Whether the recommendation was accepted
            user_feedback: Optional feedback text
            
        Returns:
            True if feedback was recorded successfully, False otherwise
        """
        if self.repository is None:
            return False
        
        try:
            # Get the existing recommendation
            recommendations = self.repository.get_recommendations(
                recommendation_id=recommendation_id,
                limit=1
            )
            
            if not recommendations:
                logger.warning(f"Recommendation {recommendation_id} not found")
                return False
            
            # Get the first (and only) recommendation
            recommendation = recommendations[0]
            
            # Update the recommendation with feedback
            recommendation['was_accepted'] = was_accepted
            recommendation['user_feedback'] = user_feedback
            
            # Store the updated recommendation
            self.repository.store_recommendation(recommendation)
            
            logger.info(f"Recorded feedback for recommendation {recommendation_id}")
            return True
        except Exception as e:
            logger.error(f"Error recording recommendation feedback: {e}")
            return False
    
    def predict_performance(self,
                          model_name: str,
                          model_family: str,
                          hardware: Union[str, List[str]],
                          batch_size: int = 1,
                          sequence_length: int = 128,
                          mode: str = "inference",
                          precision: str = "fp32",
                          store_prediction: bool = True) -> Dict[str, Any]:
        """
        Predict performance for a model on specified hardware and store the prediction.
        
        Args:
            model_name: Name of the model
            model_family: Model family/category
            hardware: Hardware type or list of hardware types
            batch_size: Batch size
            sequence_length: Sequence length
            mode: "inference" or "training"
            precision: Precision to use (fp32, fp16, int8)
            store_prediction: Whether to store the prediction in the repository
            
        Returns:
            Dictionary with performance predictions
        """
        if self.predictor is None:
            raise RuntimeError("HardwareModelPredictor is not available")
        
        # Call the predictor
        performance = self.predictor.predict_performance(
            model_name=model_name,
            model_family=model_family,
            hardware=hardware,
            batch_size=batch_size,
            sequence_length=sequence_length,
            mode=mode,
            precision=precision
        )
        
        # Store the predictions if requested
        if store_prediction and self.repository is not None:
            try:
                # Convert single hardware to list if needed
                if isinstance(hardware, str):
                    hardware_list = [hardware]
                else:
                    hardware_list = hardware
                
                # Store prediction for each hardware platform
                for hw in hardware_list:
                    if hw in performance.get('predictions', {}):
                        pred = performance['predictions'][hw]
                        
                        # Generate a unique prediction ID
                        prediction_id = f"pred-{int(time.time())}-{hash(model_name)}-{hash(hw)}"
                        
                        # Create repository record
                        repo_prediction = {
                            'timestamp': datetime.now(),
                            'model_name': model_name,
                            'model_family': model_family,
                            'hardware_platform': hw,
                            'batch_size': batch_size,
                            'sequence_length': sequence_length,
                            'precision': precision,
                            'mode': mode,
                            'throughput': pred.get('throughput'),
                            'latency': pred.get('latency'),
                            'memory_usage': pred.get('memory_usage'),
                            'confidence_score': 0.8,  # Default confidence, can be refined later
                            'prediction_source': pred.get('source', 'hardware_model_predictor'),
                            'prediction_id': prediction_id,
                            'metadata': self.metadata
                        }
                        
                        self.repository.store_prediction(repo_prediction)
                        
                        # Store the prediction ID in the result
                        performance['predictions'][hw]['prediction_id'] = prediction_id
                
            except Exception as e:
                logger.error(f"Error storing performance prediction: {e}")
        
        return performance
    
    def record_actual_performance(self,
                                model_name: str,
                                model_family: str,
                                hardware_platform: str,
                                batch_size: int,
                                sequence_length: int,
                                precision: str,
                                mode: str,
                                throughput: Optional[float] = None,
                                latency: Optional[float] = None,
                                memory_usage: Optional[float] = None,
                                prediction_id: Optional[str] = None,
                                measurement_source: str = "benchmark") -> Dict[str, Any]:
        """
        Record actual performance measurements and compare with predictions.
        
        Args:
            model_name: Name of the model
            model_family: Model family/category
            hardware_platform: Hardware platform
            batch_size: Batch size
            sequence_length: Sequence length
            precision: Precision used (fp32, fp16, int8)
            mode: "inference" or "training"
            throughput: Optional throughput measurement
            latency: Optional latency measurement
            memory_usage: Optional memory usage measurement
            prediction_id: Optional ID of a previous prediction to compare with
            measurement_source: Source of the measurement
            
        Returns:
            Dictionary with the recorded measurement and comparison with prediction
        """
        if self.repository is None:
            raise RuntimeError("Repository is not available")
        
        try:
            # Generate a unique measurement ID
            measurement_id = f"meas-{int(time.time())}-{hash(model_name)}-{hash(hardware_platform)}"
            
            # Create repository record
            measurement = {
                'timestamp': datetime.now(),
                'model_name': model_name,
                'model_family': model_family,
                'hardware_platform': hardware_platform,
                'batch_size': batch_size,
                'sequence_length': sequence_length,
                'precision': precision,
                'mode': mode,
                'throughput': throughput,
                'latency': latency,
                'memory_usage': memory_usage,
                'measurement_source': measurement_source,
                'measurement_id': measurement_id,
                'metadata': self.metadata
            }
            
            self.repository.store_measurement(measurement)
            
            # Look up associated prediction if prediction_id provided
            prediction = None
            if prediction_id:
                predictions = self.repository.get_predictions(
                    prediction_id=prediction_id,
                    limit=1
                )
                if predictions:
                    prediction = predictions[0]
            
            # If no specific prediction_id provided, look for most recent prediction
            # for this model and hardware configuration
            if not prediction:
                predictions = self.repository.get_predictions(
                    model_name=model_name,
                    hardware_platform=hardware_platform,
                    batch_size=batch_size,
                    limit=1
                )
                if predictions:
                    prediction = predictions[0]
                    prediction_id = prediction.get('prediction_id')
            
            # If a prediction is found, compute and store error metrics
            result = {
                'measurement_id': measurement_id,
                'prediction_id': prediction_id,
                'measurement': measurement
            }
            
            if prediction:
                errors = []
                
                # Compare metrics if both predicted and actual values are available
                for metric, actual_value, predicted_key in [
                    ('throughput', throughput, 'throughput'),
                    ('latency', latency, 'latency'),
                    ('memory_usage', memory_usage, 'memory_usage')
                ]:
                    if actual_value is not None and prediction.get(predicted_key) is not None:
                        predicted_value = prediction.get(predicted_key)
                        absolute_error = abs(predicted_value - actual_value)
                        relative_error = absolute_error / abs(actual_value) if abs(actual_value) > 1e-10 else 0
                        
                        error = {
                            'prediction_id': prediction_id,
                            'measurement_id': measurement_id,
                            'model_name': model_name,
                            'hardware_platform': hardware_platform,
                            'metric': metric,
                            'predicted_value': predicted_value,
                            'actual_value': actual_value,
                            'absolute_error': absolute_error,
                            'relative_error': relative_error,
                            'metadata': {
                                'batch_size': batch_size,
                                'precision': precision,
                                'mode': mode
                            }
                        }
                        
                        self.repository.store_prediction_error(error)
                        errors.append(error)
                
                result['errors'] = errors
                result['prediction'] = prediction
            
            return result
        except Exception as e:
            logger.error(f"Error recording actual performance: {e}")
            raise

class ModelPerformancePredictorDuckDBAdapter:
    """
    Adapter for integrating model performance prediction with DuckDB repository.
    
    This class provides methods for storing and retrieving ML models for 
    performance prediction and their associated metadata in the DuckDB repository.
    """
    
    def __init__(
        self,
        repository: DuckDBPredictorRepository = None,
        models: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the adapter.
        
        Args:
            repository: The DuckDB repository for storing results
            models: Optional dictionary of trained prediction models
            metadata: Optional metadata for this instance
        """
        self.repository = repository
        self.models = models
        self.metadata = metadata or {}
        
        # Initialize repository if not provided
        if self.repository is None:
            try:
                self.repository = DuckDBPredictorRepository()
                logger.info("Initialized DuckDBPredictorRepository")
            except Exception as e:
                logger.error(f"Failed to initialize DuckDBPredictorRepository: {e}")
                raise
        
        # Load models if not provided and available
        if self.models is None and MODEL_PERFORMANCE_PREDICTOR_AVAILABLE:
            try:
                self.models = load_prediction_models()
                logger.info("Loaded prediction models")
            except Exception as e:
                logger.warning(f"Failed to load prediction models: {e}")
    
    def store_model(self,
                  model: Any,
                  model_type: str,
                  target_metric: str,
                  hardware_platform: str,
                  model_family: Optional[str] = None,
                  features: Optional[List[str]] = None,
                  training_score: Optional[float] = None,
                  validation_score: Optional[float] = None,
                  test_score: Optional[float] = None,
                  additional_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a machine learning model used for performance prediction.
        
        Args:
            model: The trained ML model object
            model_type: Type of model (e.g., RandomForest, GradientBoosting)
            target_metric: Target metric for prediction (e.g., throughput, latency)
            hardware_platform: Hardware platform the model is trained for
            model_family: Optional model family filter
            features: Optional list of feature names used by the model
            training_score: Optional training score (e.g., R²)
            validation_score: Optional validation score
            test_score: Optional test score
            additional_metadata: Optional additional metadata
            
        Returns:
            The ID of the stored model
        """
        if self.repository is None:
            raise RuntimeError("Repository is not available")
        
        try:
            # Serialize the model
            serialized_model = joblib.dumps(model)
            
            # Generate a unique model ID
            model_id = f"model-{int(time.time())}-{target_metric}-{hardware_platform}"
            
            # Combine metadata
            combined_metadata = {
                **self.metadata,
                **(additional_metadata or {})
            }
            
            # Create repository record
            model_data = {
                'timestamp': datetime.now(),
                'model_type': model_type,
                'target_metric': target_metric,
                'hardware_platform': hardware_platform,
                'model_family': model_family,
                'serialized_model': serialized_model,
                'features_list': features,
                'training_score': training_score,
                'validation_score': validation_score,
                'test_score': test_score,
                'model_id': model_id,
                'metadata': combined_metadata
            }
            
            self.repository.store_prediction_model(model_data)
            
            # If feature importances are available, store them
            if hasattr(model, 'feature_importances_') and features:
                self._store_feature_importances(
                    model, model_id, features, method='native'
                )
            
            return model_id
        except Exception as e:
            logger.error(f"Error storing prediction model: {e}")
            raise
    
    def _store_feature_importances(self,
                               model: Any,
                               model_id: str,
                               feature_names: List[str],
                               method: str = 'native') -> None:
        """
        Store feature importances for a model.
        
        Args:
            model: The trained ML model object
            model_id: ID of the stored model
            feature_names: List of feature names
            method: Method used to calculate importances
        """
        try:
            # Get feature importances from the model
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            else:
                logger.warning(f"Model {model_id} does not have feature_importances_ attribute")
                return
            
            # Sanity check on lengths
            if len(importances) != len(feature_names):
                logger.warning(f"Feature importances length {len(importances)} doesn't match feature names length {len(feature_names)}")
                return
            
            # Pair importances with feature names and sort by importance
            paired = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
            
            # Store each feature importance
            for i, (feature, importance) in enumerate(paired):
                importance_data = {
                    'timestamp': datetime.now(),
                    'model_id': model_id,
                    'feature_name': feature,
                    'importance_score': float(importance),
                    'rank': i + 1,
                    'method': method,
                    'metadata': {
                        'relative_importance': float(importance) / max(importances) if max(importances) > 0 else 0
                    }
                }
                
                self.repository.store_feature_importance(importance_data)
        except Exception as e:
            logger.error(f"Error storing feature importances: {e}")
    
    def load_model(self,
                 target_metric: str,
                 hardware_platform: Optional[str] = None,
                 model_family: Optional[str] = None,
                 model_id: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a prediction model from the repository.
        
        Args:
            target_metric: Target metric (e.g., throughput, latency)
            hardware_platform: Optional hardware platform filter
            model_family: Optional model family filter
            model_id: Optional specific model ID
            
        Returns:
            Tuple of (loaded model object, model metadata)
        """
        if self.repository is None:
            raise RuntimeError("Repository is not available")
        
        try:
            # Query for the model
            filters = {
                'target_metric': target_metric
            }
            
            if hardware_platform:
                filters['hardware_platform'] = hardware_platform
            
            if model_family:
                filters['model_family'] = model_family
            
            if model_id:
                filters['model_id'] = model_id
            
            models = self.repository.get_prediction_models(**filters, limit=1)
            
            if not models:
                raise ValueError(f"No model found for {filters}")
            
            # Get the first (and most recent) model
            model_data = models[0]
            
            # Deserialize the model
            model = joblib.loads(model_data['serialized_model'])
            
            # Get feature importances
            feature_importances = self.repository.get_feature_importance(
                model_id=model_data['model_id']
            )
            
            # Add feature importances to metadata
            metadata = model_data.copy()
            metadata['feature_importances'] = feature_importances
            
            return model, metadata
        except Exception as e:
            logger.error(f"Error loading prediction model: {e}")
            raise
    
    def predict(self,
              model_name: str,
              hardware_platform: str,
              batch_size: int = 1,
              sequence_length: int = 128,
              precision: str = "fp32",
              mode: str = "inference",
              model_family: Optional[str] = None,
              store_prediction: bool = True) -> Dict[str, Any]:
        """
        Make a performance prediction using stored ML models.
        
        Args:
            model_name: Name of the model
            hardware_platform: Hardware platform
            batch_size: Batch size
            sequence_length: Sequence length
            precision: Precision (fp32, fp16, int8)
            mode: "inference" or "training"
            model_family: Optional model family
            store_prediction: Whether to store the prediction
            
        Returns:
            Dictionary with performance predictions
        """
        if not MODEL_PERFORMANCE_PREDICTOR_AVAILABLE:
            raise RuntimeError("ModelPerformancePredictor is not available")
        
        try:
            # Make prediction using the external function
            prediction_result = predict_performance(
                models=self.models,
                model_name=model_name,
                model_category=model_family,
                hardware=hardware_platform,
                batch_size=batch_size,
                sequence_length=sequence_length,
                precision=precision,
                mode=mode
            )
            
            if not prediction_result:
                logger.warning(f"No prediction result returned for {model_name} on {hardware_platform}")
                return {
                    'model_name': model_name,
                    'hardware_platform': hardware_platform,
                    'error': 'No prediction result returned'
                }
            
            # Store the prediction if requested
            if store_prediction and self.repository is not None:
                # Generate prediction ID
                prediction_id = f"pred-{int(time.time())}-{hash(model_name)}-{hash(hardware_platform)}"
                
                # Create repository record
                repo_prediction = {
                    'timestamp': datetime.now(),
                    'model_name': model_name,
                    'model_family': model_family,
                    'hardware_platform': hardware_platform,
                    'batch_size': batch_size,
                    'sequence_length': sequence_length,
                    'precision': precision,
                    'mode': mode,
                    'throughput': prediction_result.get('throughput'),
                    'latency': prediction_result.get('latency_mean'),
                    'memory_usage': prediction_result.get('memory_usage'),
                    'confidence_score': prediction_result.get('confidence', 0.8),
                    'prediction_source': 'model_performance_predictor',
                    'prediction_id': prediction_id,
                    'metadata': {
                        **self.metadata,
                        'model_features': prediction_result.get('features_used'),
                        'model_type': prediction_result.get('model_type')
                    }
                }
                
                self.repository.store_prediction(repo_prediction)
                
                # Add prediction ID to result
                prediction_result['prediction_id'] = prediction_id
            
            return prediction_result
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def generate_prediction_matrix(self,
                                model_configs: List[Dict[str, str]],
                                hardware_platforms: List[str],
                                batch_sizes: List[int],
                                precision_options: List[str] = ["fp32"],
                                mode: str = "inference",
                                store_predictions: bool = True) -> Dict[str, Any]:
        """
        Generate a prediction matrix for various models and hardware configurations.
        
        Args:
            model_configs: List of model configs with 'name' and 'category' keys
            hardware_platforms: List of hardware platforms
            batch_sizes: List of batch sizes
            precision_options: List of precision options
            mode: "inference" or "training"
            store_predictions: Whether to store the predictions
            
        Returns:
            Dictionary with prediction matrix
        """
        if not MODEL_PERFORMANCE_PREDICTOR_AVAILABLE:
            raise RuntimeError("ModelPerformancePredictor is not available")
        
        try:
            # Generate matrix using the external function
            matrix = generate_prediction_matrix(
                models=self.models,
                model_configs=model_configs,
                hardware_platforms=hardware_platforms,
                batch_sizes=batch_sizes,
                precision_options=precision_options,
                mode=mode
            )
            
            # Store predictions if requested
            if store_predictions and self.repository is not None and matrix:
                stored_predictions = []
                
                # Iterate through all models in the matrix
                for model_name, model_data in matrix.get('models', {}).items():
                    model_category = model_data.get('category')
                    
                    # Iterate through hardware platforms
                    for hw in hardware_platforms:
                        hw_predictions = model_data.get('predictions', {}).get(hw, {})
                        
                        # Iterate through batch sizes
                        for batch_size_str, batch_data in hw_predictions.items():
                            try:
                                batch_size = int(batch_size_str)
                            except ValueError:
                                continue
                            
                            # Iterate through precision options
                            for precision in precision_options:
                                precision_data = batch_data.get(precision, {})
                                
                                # Generate prediction ID
                                prediction_id = f"pred-{int(time.time())}-{hash(model_name)}-{hash(hw)}-{batch_size}-{hash(precision)}"
                                
                                # Create repository record
                                repo_prediction = {
                                    'timestamp': datetime.now(),
                                    'model_name': model_name,
                                    'model_family': model_category,
                                    'hardware_platform': hw,
                                    'batch_size': batch_size,
                                    'sequence_length': 128,  # Default
                                    'precision': precision,
                                    'mode': mode,
                                    'throughput': precision_data.get('throughput'),
                                    'latency': precision_data.get('latency_mean'),
                                    'memory_usage': precision_data.get('memory_usage'),
                                    'confidence_score': precision_data.get('confidence', 0.8),
                                    'prediction_source': 'matrix_prediction',
                                    'prediction_id': prediction_id,
                                    'metadata': {
                                        **self.metadata,
                                        'matrix_id': matrix.get('matrix_id', 'unknown'),
                                        'timestamp': matrix.get('timestamp', datetime.now().isoformat())
                                    }
                                }
                                
                                # Only store if we have actual prediction values
                                if any(repo_prediction.get(k) is not None for k in ['throughput', 'latency', 'memory_usage']):
                                    try:
                                        self.repository.store_prediction(repo_prediction)
                                        stored_predictions.append(prediction_id)
                                    except Exception as e:
                                        logger.warning(f"Error storing matrix prediction: {e}")
                
                logger.info(f"Stored {len(stored_predictions)} predictions from prediction matrix")
                matrix['stored_predictions'] = len(stored_predictions)
            
            return matrix
        except Exception as e:
            logger.error(f"Error generating prediction matrix: {e}")
            raise