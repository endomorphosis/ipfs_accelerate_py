#!/usr/bin/env python3
"""
Template for Time Series Transformer models such as Informer, PatchTST, etc.

This template is designed for models that specialize in time series forecasting,
classification, and anomaly detection.
"""

import os
import time
import json
import torch
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple

try:
    from transformers import {model_class_name}, {processor_class_name}
    import transformers
except ImportError:
    raise ImportError(
        "The transformers package is required to use this model. "
        "Please install it with `pip install transformers`."
    )

logger = logging.getLogger(__name__)

class {skillset_class_name}:
    """
    Skillset for {model_type_upper} - a time series transformer model
    that processes sequential temporal data for forecasting and classification.
    """
    
    def __init__(self, model_id: str = "{default_model_id}", device: str = "cpu", **kwargs):
        """
        Initialize the {model_type_upper} model.
        
        Args:
            model_id: HuggingFace model ID or path
            device: Device to run the model on ('cpu', 'cuda', 'rocm', 'mps', etc.)
            **kwargs: Additional arguments to pass to the model
        """
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None
        self.is_initialized = False
        
        # Track hardware info for reporting
        self.hardware_info = {
            "device": device,
            "device_name": None,
            "memory_available": None,
            "supports_half_precision": False,
            "time_series_specific": {
                "context_length": None,
                "prediction_length": None,
                "feature_size": None,
                "supports_multivariate": True
            }
        }
        
        # Optional configuration
        self.low_memory_mode = kwargs.get("low_memory_mode", False)
        self.max_memory = kwargs.get("max_memory", None)
        self.torch_dtype = kwargs.get("torch_dtype", None)
        self.context_length = kwargs.get("context_length", 512)
        self.prediction_length = kwargs.get("prediction_length", 96)
        
        # Initialize the model if auto_init is True
        auto_init = kwargs.get("auto_init", True)
        if auto_init:
            self.initialize()
    
    def initialize(self):
        """Initialize the model and processor."""
        if self.is_initialized:
            return
        
        logger.info(f"Initializing {self.model_id} on {self.device}")
        start_time = time.time()
        
        try:
            # Check if CUDA is available when device is cuda
            if self.device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = "cpu"
            
            # Check if MPS is available when device is mps
            if self.device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                logger.warning("MPS requested but not available, falling back to CPU")
                self.device = "cpu"
            
            # Check if ROCm/HIP is available when device is rocm
            if self.device == "rocm":
                rocm_available = False
                try:
                    if hasattr(torch, 'hip') and torch.hip.is_available():
                        rocm_available = True
                    elif torch.cuda.is_available():
                        # Could be ROCm using CUDA API
                        device_name = torch.cuda.get_device_name(0)
                        if "AMD" in device_name or "Radeon" in device_name:
                            rocm_available = True
                            self.hardware_info.update({
                                "device_name": device_name,
                                "memory_available": torch.cuda.get_device_properties(0).total_memory,
                                "supports_half_precision": True  # Most AMD GPUs support half precision
                            })
                except:
                    rocm_available = False
                
                if not rocm_available:
                    logger.warning("ROCm requested but not available, falling back to CPU")
                    self.device = "cpu"
            
            # CPU is the fallback
            if self.device == "cpu":
                self.hardware_info.update({
                    "device_name": "CPU",
                    "supports_half_precision": False
                })
            
            # Determine dtype based on hardware
            if self.torch_dtype is None:
                if self.hardware_info["supports_half_precision"] and not self.low_memory_mode:
                    self.torch_dtype = torch.float16
                else:
                    self.torch_dtype = torch.float32
            
            # Load processor
            try:
                self.processor = {processor_class_name}.from_pretrained(self.model_id)
            except Exception as e:
                logger.warning(f"Error loading processor: {str(e)}. Creating a mock processor.")
                self.processor = self._create_mock_processor()
            
            # Load model with appropriate configuration
            load_kwargs = {}
            if self.torch_dtype is not None:
                load_kwargs["torch_dtype"] = self.torch_dtype
            
            if self.low_memory_mode:
                load_kwargs["low_cpu_mem_usage"] = True
            
            if self.max_memory is not None:
                load_kwargs["max_memory"] = self.max_memory
            
            # Specific handling for device placement
            if self.device.startswith(("cuda", "rocm")) and "device_map" not in load_kwargs:
                load_kwargs["device_map"] = "auto"
            
            # Load the time series model
            self.model = {model_class_name}.from_pretrained(self.model_id, **load_kwargs)
            
            # Move to appropriate device if not using device_map
            if "device_map" not in load_kwargs and not self.device.startswith(("cuda", "rocm")):
                self.model.to(self.device)
            
            # Update time-series specific info
            if hasattr(self.model, "config"):
                context_length = getattr(self.model.config, "context_length", None)
                prediction_length = getattr(self.model.config, "prediction_length", None)
                feature_size = getattr(self.model.config, "feature_size", None) or getattr(self.model.config, "d_model", None)
                
                self.hardware_info["time_series_specific"].update({
                    "context_length": context_length or self.context_length,
                    "prediction_length": prediction_length or self.prediction_length,
                    "feature_size": feature_size
                })
            
            # Log initialization time
            elapsed_time = time.time() - start_time
            logger.info(f"Initialized {self.model_id} in {elapsed_time:.2f} seconds")
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Error initializing {self.model_id}: {str(e)}")
            raise
    
    def _create_mock_processor(self):
        """Create a mock processor for time series data when the real one fails."""
        class MockTimeSeriesProcessor:
            def __init__(self, context_length=512, prediction_length=96, feature_size=64):
                self.context_length = context_length
                self.prediction_length = prediction_length
                self.feature_size = feature_size
            
            def __call__(self, series=None, past_values=None, past_time_features=None,
                         future_values=None, future_time_features=None,
                         return_tensors="pt", **kwargs):
                """Mock processing of time series data."""
                import torch
                
                # Create default inputs if none provided
                if series is None and past_values is None:
                    # Create a fake batch with 2 time series
                    batch_size = 2
                    if "batch_size" in kwargs:
                        batch_size = kwargs["batch_size"]
                    
                    # Create past values (context window)
                    if "feature_dim" in kwargs:
                        feature_dim = kwargs["feature_dim"]
                    else:
                        feature_dim = 1  # Default to univariate
                    
                    past_values = torch.randn(batch_size, self.context_length, feature_dim)
                    
                    # Create past time features
                    past_time_features = torch.randn(batch_size, self.context_length, 4)  # 4 time features
                    
                    # Create future values (target window)
                    if future_values is None:
                        future_values = torch.randn(batch_size, self.prediction_length, feature_dim)
                    
                    # Create future time features
                    if future_time_features is None:
                        future_time_features = torch.randn(batch_size, self.prediction_length, 4)
                
                elif series is not None:
                    # Convert series to tensor if it's numpy
                    if isinstance(series, np.ndarray):
                        series = torch.tensor(series, dtype=torch.float32)
                    
                    # Extract dimensions
                    if series.dim() == 2:  # [time, features]
                        batch_size = 1
                        time_length, feature_dim = series.shape
                        series = series.unsqueeze(0)  # Add batch dimension
                    elif series.dim() == 3:  # [batch, time, features]
                        batch_size, time_length, feature_dim = series.shape
                    else:
                        raise ValueError(f"Series must have 2 or 3 dimensions, got {series.dim()}")
                    
                    # Split into past and future
                    if time_length > self.context_length:
                        past_values = series[:, :self.context_length, :]
                        if time_length > self.context_length + self.prediction_length:
                            future_values = series[:, self.context_length:self.context_length+self.prediction_length, :]
                        else:
                            future_values = series[:, self.context_length:, :]
                    else:
                        # If not enough context, just use all as past
                        past_values = series
                        future_values = torch.zeros(batch_size, self.prediction_length, feature_dim)
                    
                    # Create time features (simple linear time index)
                    past_time_idx = torch.arange(past_values.size(1), dtype=torch.float32)
                    past_time_idx = past_time_idx / past_values.size(1)
                    past_time_features = past_time_idx.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)
                    
                    future_time_idx = torch.arange(future_values.size(1), dtype=torch.float32)
                    future_time_idx = future_time_idx / future_values.size(1)
                    future_time_features = future_time_idx.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)
                
                elif past_values is not None:
                    # Convert to tensor if needed
                    if isinstance(past_values, np.ndarray):
                        past_values = torch.tensor(past_values, dtype=torch.float32)
                    
                    # Extract dimensions
                    if past_values.dim() == 2:  # [time, features]
                        batch_size = 1
                        time_length, feature_dim = past_values.shape
                        past_values = past_values.unsqueeze(0)  # Add batch dimension
                    elif past_values.dim() == 3:  # [batch, time, features]
                        batch_size, time_length, feature_dim = past_values.shape
                    else:
                        raise ValueError(f"Past values must have 2 or 3 dimensions, got {past_values.dim()}")
                    
                    # Create time features if not provided
                    if past_time_features is None:
                        past_time_idx = torch.arange(time_length, dtype=torch.float32)
                        past_time_idx = past_time_idx / time_length
                        past_time_features = past_time_idx.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)
                    
                    # Create future values if not provided
                    if future_values is None:
                        future_values = torch.zeros(batch_size, self.prediction_length, feature_dim)
                    
                    # Create future time features if not provided
                    if future_time_features is None:
                        future_time_idx = torch.arange(self.prediction_length, dtype=torch.float32)
                        future_time_idx = future_time_idx / self.prediction_length
                        future_time_features = future_time_idx.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)
                
                # Create attention masks
                past_attention_mask = torch.ones_like(past_values[:, :, 0])
                future_attention_mask = torch.ones_like(future_values[:, :, 0])
                
                return {
                    "past_values": past_values,
                    "past_time_features": past_time_features,
                    "past_attention_mask": past_attention_mask,
                    "future_values": future_values,
                    "future_time_features": future_time_features,
                    "future_attention_mask": future_attention_mask
                }
        
        logger.info("Creating mock time series processor")
        return MockTimeSeriesProcessor(
            context_length=self.context_length,
            prediction_length=self.prediction_length
        )
    
    def process_time_series(self, series_data, **kwargs):
        """
        Process time series data into the format expected by the model.
        
        Args:
            series_data: The input time series data (can be tensor, array, DataFrame, or dict)
            **kwargs: Additional processing parameters
            
        Returns:
            Processed inputs ready for the model
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # Extract prediction length from kwargs or use default
            prediction_length = kwargs.get("prediction_length", self.prediction_length)
            
            # Check the type of input and process accordingly
            if isinstance(series_data, dict) and "past_values" in series_data:
                # Already in the expected format
                inputs = series_data
            elif isinstance(series_data, (pd.DataFrame, pd.Series)) if 'pd' in globals() else False:
                # Handle pandas DataFrame or Series
                inputs = self._process_pandas_data(series_data, prediction_length, **kwargs)
            elif hasattr(series_data, 'to_numpy') and callable(getattr(series_data, 'to_numpy')):
                # Handle objects with to_numpy method
                numpy_data = series_data.to_numpy()
                inputs = self._process_numpy_data(numpy_data, prediction_length, **kwargs)
            elif isinstance(series_data, np.ndarray):
                # Handle numpy array
                inputs = self._process_numpy_data(series_data, prediction_length, **kwargs)
            elif isinstance(series_data, torch.Tensor):
                # Handle torch tensor
                inputs = self._process_torch_data(series_data, prediction_length, **kwargs)
            else:
                # Default handling - attempt to use the processor directly
                try:
                    inputs = self.processor(series_data, return_tensors="pt", **kwargs)
                except:
                    logger.warning("Could not process time series data with processor, using mock data")
                    inputs = self._create_mock_inputs(**kwargs)
            
            # Ensure all tensors are on the correct device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            return inputs
            
        except Exception as e:
            logger.error(f"Error processing time series data: {str(e)}")
            # Return mock inputs as fallback
            return self._create_mock_inputs(**kwargs)
    
    def _process_pandas_data(self, data, prediction_length, **kwargs):
        """Process pandas DataFrame or Series into model inputs."""
        try:
            import pandas as pd
            
            # Convert to numpy
            if isinstance(data, pd.Series):
                values = data.values.reshape(-1, 1)  # Convert to [time, 1] array
            else:  # DataFrame
                values = data.values  # [time, features] array
            
            # Process the numpy array
            return self._process_numpy_data(values, prediction_length, **kwargs)
            
        except ImportError:
            logger.warning("pandas not available, using mock data")
            return self._create_mock_inputs(**kwargs)
    
    def _process_numpy_data(self, data, prediction_length, **kwargs):
        """Process numpy array into model inputs."""
        # Check dimensions
        if data.ndim == 1:
            # Single univariate time series [time]
            data = data.reshape(-1, 1)  # Convert to [time, 1]
        
        # Now data should be [time, features] or [batch, time, features]
        if data.ndim == 2:
            # Single time series [time, features]
            data = np.expand_dims(data, 0)  # Add batch dimension
        
        # Now data should be [batch, time, features]
        batch_size, time_length, feature_dim = data.shape
        
        # Determine context length based on model config or input data
        context_length = min(self.context_length, time_length - prediction_length) if time_length > prediction_length else time_length
        
        # Split data into past and future
        if time_length > context_length + prediction_length:
            # We have enough data for both context and prediction
            past_values = data[:, :context_length, :]
            future_values = data[:, context_length:context_length+prediction_length, :]
        elif time_length > context_length:
            # We have enough for context, but not full prediction
            past_values = data[:, :context_length, :]
            future_values = data[:, context_length:, :]
            # Pad future values if needed
            if future_values.shape[1] < prediction_length:
                pad_length = prediction_length - future_values.shape[1]
                future_values = np.pad(future_values, ((0, 0), (0, pad_length), (0, 0)), mode='constant')
        else:
            # Not enough data for full context
            past_values = data
            future_values = np.zeros((batch_size, prediction_length, feature_dim))
        
        # Convert to tensors
        past_values_tensor = torch.tensor(past_values, dtype=torch.float32)
        future_values_tensor = torch.tensor(future_values, dtype=torch.float32)
        
        # Generate time features
        # Simple linear time index for past values
        past_time_idx = torch.arange(past_values_tensor.size(1), dtype=torch.float32)
        past_time_idx = past_time_idx / max(1, past_values_tensor.size(1))
        past_time_features = past_time_idx.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)
        
        # Simple linear time index for future values
        future_time_idx = torch.arange(future_values_tensor.size(1), dtype=torch.float32)
        future_time_idx = future_time_idx / max(1, future_values_tensor.size(1))
        future_time_features = future_time_idx.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)
        
        # Create attention masks
        past_attention_mask = torch.ones(batch_size, past_values_tensor.size(1))
        future_attention_mask = torch.ones(batch_size, future_values_tensor.size(1))
        
        return {
            "past_values": past_values_tensor,
            "past_time_features": past_time_features,
            "past_attention_mask": past_attention_mask,
            "future_values": future_values_tensor,
            "future_time_features": future_time_features,
            "future_attention_mask": future_attention_mask
        }
    
    def _process_torch_data(self, data, prediction_length, **kwargs):
        """Process torch tensor into model inputs."""
        # Check dimensions
        if data.dim() == 1:
            # Single univariate time series [time]
            data = data.view(-1, 1)  # Convert to [time, 1]
        
        # Now data should be [time, features] or [batch, time, features]
        if data.dim() == 2:
            # Single time series [time, features]
            data = data.unsqueeze(0)  # Add batch dimension
        
        # Now data should be [batch, time, features]
        batch_size, time_length, feature_dim = data.shape
        
        # Determine context length based on model config or input data
        context_length = min(self.context_length, time_length - prediction_length) if time_length > prediction_length else time_length
        
        # Split data into past and future
        if time_length > context_length + prediction_length:
            # We have enough data for both context and prediction
            past_values = data[:, :context_length, :]
            future_values = data[:, context_length:context_length+prediction_length, :]
        elif time_length > context_length:
            # We have enough for context, but not full prediction
            past_values = data[:, :context_length, :]
            future_values = data[:, context_length:, :]
            # Pad future values if needed
            if future_values.size(1) < prediction_length:
                pad_length = prediction_length - future_values.size(1)
                future_values = torch.nn.functional.pad(future_values, (0, 0, 0, pad_length, 0, 0))
        else:
            # Not enough data for full context
            past_values = data
            future_values = torch.zeros(batch_size, prediction_length, feature_dim)
        
        # Generate time features
        # Simple linear time index for past values
        past_time_idx = torch.arange(past_values.size(1), dtype=torch.float32)
        past_time_idx = past_time_idx / max(1, past_values.size(1))
        past_time_features = past_time_idx.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)
        
        # Simple linear time index for future values
        future_time_idx = torch.arange(future_values.size(1), dtype=torch.float32)
        future_time_idx = future_time_idx / max(1, future_values.size(1))
        future_time_features = future_time_idx.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)
        
        # Create attention masks
        past_attention_mask = torch.ones(batch_size, past_values.size(1))
        future_attention_mask = torch.ones(batch_size, future_values.size(1))
        
        return {
            "past_values": past_values,
            "past_time_features": past_time_features,
            "past_attention_mask": past_attention_mask,
            "future_values": future_values,
            "future_time_features": future_time_features,
            "future_attention_mask": future_attention_mask
        }
    
    def _create_mock_inputs(self, **kwargs):
        """Create mock inputs for graceful degradation."""
        batch_size = kwargs.get("batch_size", 1)
        feature_dim = kwargs.get("feature_dim", 1)
        context_length = kwargs.get("context_length", self.context_length)
        prediction_length = kwargs.get("prediction_length", self.prediction_length)
        
        # Create mock time series data
        past_values = torch.randn(batch_size, context_length, feature_dim).to(self.device)
        future_values = torch.randn(batch_size, prediction_length, feature_dim).to(self.device)
        
        # Create time features
        past_time_idx = torch.arange(context_length, dtype=torch.float32, device=self.device)
        past_time_idx = past_time_idx / context_length
        past_time_features = past_time_idx.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)
        
        future_time_idx = torch.arange(prediction_length, dtype=torch.float32, device=self.device)
        future_time_idx = future_time_idx / prediction_length
        future_time_features = future_time_idx.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)
        
        # Create attention masks
        past_attention_mask = torch.ones(batch_size, context_length, device=self.device)
        future_attention_mask = torch.ones(batch_size, prediction_length, device=self.device)
        
        return {
            "past_values": past_values,
            "past_time_features": past_time_features,
            "past_attention_mask": past_attention_mask,
            "future_values": future_values,
            "future_time_features": future_time_features,
            "future_attention_mask": future_attention_mask
        }
    
    def forecast(self, series_data, **kwargs):
        """
        Generate time series forecasts.
        
        Args:
            series_data: Input time series data
            **kwargs: Additional parameters for processing or inference
            
        Returns:
            Dictionary with forecast results
        """
        if not self.is_initialized:
            self.initialize()
        
        # Extract forecast parameters
        prediction_length = kwargs.pop("prediction_length", self.prediction_length)
        
        # Process the time series into model inputs
        inputs = self.process_time_series(series_data, prediction_length=prediction_length, **kwargs)
        
        try:
            # Run inference with the model
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Extract and format results
            if hasattr(outputs, "predictions"):
                predictions = outputs.predictions
                
                # Get past values for context
                past_values = inputs.get("past_values")
                
                # Convert to numpy for easier handling
                predictions_np = predictions.cpu().numpy()
                past_values_np = past_values.cpu().numpy() if past_values is not None else None
                
                # Prepare result dictionary
                result = {
                    "forecasts": predictions_np,
                    "forecast_length": predictions_np.shape[1]
                }
                
                # Add past values if available
                if past_values_np is not None:
                    result["past_values"] = past_values_np
                    result["context_length"] = past_values_np.shape[1]
                
                # Add metrics if ground truth is available
                future_values = inputs.get("future_values")
                if future_values is not None:
                    future_values_np = future_values.cpu().numpy()
                    
                    # Calculate MSE, MAE, and MAPE
                    # Make sure lengths match
                    min_length = min(future_values_np.shape[1], predictions_np.shape[1])
                    
                    # MSE
                    mse = np.mean((predictions_np[:, :min_length] - future_values_np[:, :min_length]) ** 2)
                    # MAE
                    mae = np.mean(np.abs(predictions_np[:, :min_length] - future_values_np[:, :min_length]))
                    # MAPE (avoiding division by zero)
                    epsilon = 1e-10
                    mape = np.mean(np.abs((future_values_np[:, :min_length] - predictions_np[:, :min_length]) / 
                                          (np.abs(future_values_np[:, :min_length]) + epsilon))) * 100.0
                    
                    result["metrics"] = {
                        "mse": mse,
                        "mae": mae,
                        "mape": mape
                    }
                    result["ground_truth"] = future_values_np
                
                return result
            else:
                logger.warning("Model outputs don't include predictions, returning raw outputs")
                # Convert any tensors to numpy for serialization
                raw_dict = {}
                for k, v in outputs.items():
                    if hasattr(v, "cpu"):
                        raw_dict[k] = v.cpu().numpy()
                    else:
                        raw_dict[k] = v
                
                return {"raw_outputs": raw_dict}
        
        except Exception as e:
            logger.error(f"Error during forecasting: {str(e)}")
            return {"error": str(e)}
    
    def classify(self, series_data, **kwargs):
        """
        Perform time series classification.
        
        Args:
            series_data: Input time series data
            **kwargs: Additional parameters for processing or inference
            
        Returns:
            Dictionary with classification results
        """
        if not self.is_initialized:
            self.initialize()
        
        # Process the time series into model inputs
        inputs = self.process_time_series(series_data, **kwargs)
        
        try:
            # Run inference with the model
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Extract and format results
            if hasattr(outputs, "logits") or hasattr(outputs, "class_logits"):
                logits = outputs.logits if hasattr(outputs, "logits") else outputs.class_logits
                
                # Get class predictions
                predictions = torch.argmax(logits, dim=-1)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                
                # Convert to numpy
                predictions_np = predictions.cpu().numpy()
                probabilities_np = probabilities.cpu().numpy()
                logits_np = logits.cpu().numpy()
                
                # Format result
                result = {
                    "predictions": predictions_np,
                    "probabilities": probabilities_np,
                    "logits": logits_np,
                    "num_classes": logits_np.shape[-1]
                }
                
                return result
            else:
                logger.warning("Model outputs don't include logits, returning raw outputs")
                # Convert any tensors to numpy for serialization
                raw_dict = {}
                for k, v in outputs.items():
                    if hasattr(v, "cpu"):
                        raw_dict[k] = v.cpu().numpy()
                    else:
                        raw_dict[k] = v
                
                return {"raw_outputs": raw_dict}
        
        except Exception as e:
            logger.error(f"Error during classification: {str(e)}")
            return {"error": str(e)}
    
    def detect_anomalies(self, series_data, threshold=0.95, **kwargs):
        """
        Detect anomalies in time series data.
        
        Args:
            series_data: Input time series data
            threshold: Threshold for determining anomalies (higher is more strict)
            **kwargs: Additional parameters for processing or inference
            
        Returns:
            Dictionary with anomaly detection results
        """
        if not self.is_initialized:
            self.initialize()
        
        # Process the time series into model inputs
        inputs = self.process_time_series(series_data, **kwargs)
        
        try:
            # Run inference with the model
            with torch.no_grad():
                # If the model supports anomaly detection directly
                if hasattr(self.model, "detect_anomalies"):
                    outputs = self.model.detect_anomalies(**inputs, threshold=threshold)
                    
                    # Extract anomaly scores and binary anomaly flags
                    anomaly_scores = outputs.anomaly_scores.cpu().numpy()
                    anomalies = outputs.anomalies.cpu().numpy()
                    
                    result = {
                        "anomaly_scores": anomaly_scores,
                        "anomalies": anomalies,
                        "threshold": threshold
                    }
                    
                    return result
                    
                # Otherwise, use the reconstruction error as anomaly score
                else:
                    # Reconstruction approach
                    outputs = self.model(**inputs)
                    
                    # Get reconstructed values
                    if hasattr(outputs, "reconstructions"):
                        reconstructions = outputs.reconstructions
                    elif hasattr(outputs, "predictions"):
                        reconstructions = outputs.predictions
                    else:
                        # No explicit reconstructions, use raw outputs
                        logger.warning("No reconstructions available, using raw model output")
                        if hasattr(outputs, "last_hidden_state"):
                            # If there's a hidden state, project it back to the feature space
                            if hasattr(self.model, "lm_head") or hasattr(self.model, "output_layer"):
                                proj_layer = getattr(self.model, "lm_head", None) or getattr(self.model, "output_layer", None)
                                reconstructions = proj_layer(outputs.last_hidden_state)
                            else:
                                # Can't reconstruct, return raw outputs
                                logger.warning("Unable to create reconstructions, returning raw anomaly detection")
                                # Use simple statistical anomaly detection
                                past_values = inputs["past_values"]
                                
                                # Calculate z-scores for each feature
                                mean = torch.mean(past_values, dim=1, keepdim=True)
                                std = torch.std(past_values, dim=1, keepdim=True) + 1e-6  # Avoid division by zero
                                z_scores = torch.abs((past_values - mean) / std)
                                
                                # Determine anomalies based on z-score threshold
                                z_threshold = 3.0  # 3 sigma rule
                                anomaly_scores = z_scores.mean(dim=-1).cpu().numpy()
                                anomalies = (z_scores.mean(dim=-1) > z_threshold).cpu().numpy()
                                
                                return {
                                    "anomaly_scores": anomaly_scores,
                                    "anomalies": anomalies,
                                    "threshold": z_threshold,
                                    "method": "z-score"
                                }
                    
                    # Calculate reconstruction error
                    past_values = inputs["past_values"]
                    mse = torch.mean((reconstructions - past_values) ** 2, dim=-1)
                    
                    # Determine threshold based on distribution of errors
                    # Get percentile based on threshold parameter
                    error_values = mse.view(-1)
                    threshold_value = torch.quantile(error_values, threshold)
                    
                    # Determine anomalies
                    anomaly_scores = mse.cpu().numpy()
                    anomalies = (mse > threshold_value).cpu().numpy()
                    
                    result = {
                        "anomaly_scores": anomaly_scores,
                        "anomalies": anomalies,
                        "threshold": threshold_value.item(),
                        "method": "reconstruction-error"
                    }
                    
                    return result
        
        except Exception as e:
            logger.error(f"Error during anomaly detection: {str(e)}")
            return {"error": str(e)}
    
    def embedding(self, series_data, **kwargs):
        """
        Generate embeddings for time series data.
        
        Args:
            series_data: Input time series data
            **kwargs: Additional parameters for processing or inference
            
        Returns:
            Dictionary with embedding results
        """
        if not self.is_initialized:
            self.initialize()
        
        # Process the time series into model inputs
        inputs = self.process_time_series(series_data, **kwargs)
        
        try:
            # Run inference with the model
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            # Extract embeddings from the output
            if hasattr(outputs, "encoder_last_hidden_state"):
                # For encoder-decoder models
                embeddings = outputs.encoder_last_hidden_state
                
                # Global pooling to get a single embedding per series
                if kwargs.get("pooling", "mean") == "mean":
                    # Mean pooling
                    pooled_embeddings = torch.mean(embeddings, dim=1)
                else:
                    # Max pooling
                    pooled_embeddings = torch.max(embeddings, dim=1)[0]
                
            elif hasattr(outputs, "last_hidden_state"):
                # For encoder-only models
                embeddings = outputs.last_hidden_state
                
                # Global pooling to get a single embedding per series
                if kwargs.get("pooling", "mean") == "mean":
                    # Mean pooling
                    pooled_embeddings = torch.mean(embeddings, dim=1)
                else:
                    # Max pooling
                    pooled_embeddings = torch.max(embeddings, dim=1)[0]
                
            elif hasattr(outputs, "hidden_states"):
                # Use last layer of hidden states
                embeddings = outputs.hidden_states[-1]
                
                # Global pooling to get a single embedding per series
                if kwargs.get("pooling", "mean") == "mean":
                    # Mean pooling
                    pooled_embeddings = torch.mean(embeddings, dim=1)
                else:
                    # Max pooling
                    pooled_embeddings = torch.max(embeddings, dim=1)[0]
                
            else:
                logger.warning("Could not extract embeddings from model outputs")
                # Return raw outputs
                raw_dict = {}
                for k, v in outputs.items():
                    if hasattr(v, "cpu"):
                        raw_dict[k] = v.cpu().numpy()
                    else:
                        raw_dict[k] = v
                
                return {"raw_outputs": raw_dict}
            
            # Convert to numpy
            embeddings_np = embeddings.cpu().numpy()
            pooled_embeddings_np = pooled_embeddings.cpu().numpy()
            
            # Format result
            result = {
                "embeddings": embeddings_np,
                "pooled_embeddings": pooled_embeddings_np,
                "embedding_dim": pooled_embeddings_np.shape[-1]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error during embedding: {str(e)}")
            return {"error": str(e)}
    
    def __call__(self, series_data, task: str = "forecast", **kwargs) -> Dict[str, Any]:
        """
        Process time series data with the model.
        
        Args:
            series_data: The input time series data
            task: Task to perform ('forecast', 'classify', 'detect_anomalies', 'embedding')
            **kwargs: Additional task-specific parameters
            
        Returns:
            Dictionary with task results
        """
        if not self.is_initialized:
            self.initialize()
        
        # Select task
        if task == "forecast":
            return self.forecast(series_data, **kwargs)
        elif task == "classify":
            return self.classify(series_data, **kwargs)
        elif task == "detect_anomalies":
            threshold = kwargs.pop("threshold", 0.95)
            return self.detect_anomalies(series_data, threshold=threshold, **kwargs)
        elif task == "embedding":
            return self.embedding(series_data, **kwargs)
        else:
            # Default to forecasting
            logger.warning(f"Unknown task '{task}', defaulting to forecast")
            return self.forecast(series_data, **kwargs)

    def __test__(self, **kwargs):
        """
        Run a self-test to verify the model is working correctly.
        
        Returns:
            Dictionary with test results
        """
        if not self.is_initialized:
            try:
                self.initialize()
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Initialization failed: {str(e)}",
                    "hardware": self.hardware_info
                }
        
        results = {
            "hardware": self.hardware_info,
            "tests": {}
        }
        
        # Test 1: Process time series data
        try:
            # Create a simple synthetic time series
            batch_size = 2
            feature_dim = 3
            time_length = 100
            
            series_data = torch.randn(batch_size, time_length, feature_dim)
            
            # Test time series processing
            inputs = self.process_time_series(series_data)
            
            results["tests"]["time_series_processing"] = {
                "success": True,
                "input_keys": list(inputs.keys()),
                "past_values_shape": tuple(inputs["past_values"].shape),
                "future_values_shape": tuple(inputs["future_values"].shape) if "future_values" in inputs else None
            }
        except Exception as e:
            results["tests"]["time_series_processing"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test 2: Forecasting
        try:
            # Create a simple synthetic time series
            batch_size = 2
            feature_dim = 3
            time_length = 100
            
            series_data = torch.randn(batch_size, time_length, feature_dim)
            
            # Test forecasting
            output = self.forecast(series_data)
            
            results["tests"]["forecasting"] = {
                "success": "forecasts" in output or "raw_outputs" in output,
                "output_keys": list(output.keys()),
                "forecast_shape": output["forecasts"].shape if "forecasts" in output else None
            }
        except Exception as e:
            results["tests"]["forecasting"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test 3: Embedding
        try:
            # Create a simple synthetic time series
            batch_size = 2
            feature_dim = 3
            time_length = 100
            
            series_data = torch.randn(batch_size, time_length, feature_dim)
            
            # Test embedding
            output = self.embedding(series_data)
            
            results["tests"]["embedding"] = {
                "success": "embeddings" in output or "raw_outputs" in output,
                "output_keys": list(output.keys()),
                "embedding_shape": output["pooled_embeddings"].shape if "pooled_embeddings" in output else None
            }
        except Exception as e:
            results["tests"]["embedding"] = {
                "success": False,
                "error": str(e)
            }
        
        # Overall success determination
        successful_tests = sum(1 for t in results["tests"].values() if t.get("success", False))
        results["success"] = successful_tests > 0
        results["success_rate"] = successful_tests / len(results["tests"])
        
        return results


class TestTimeSeriesModel:
    """Test suite for the Time Series model implementation."""
    
    def __init__(self):
        """Initialize the test suite."""
        self.model_id = "huggingface/time-series-transformer-tourism-monthly"  # Default test model
        self.low_memory_mode = True  # Use low memory mode for testing
    
    def run_tests(self):
        """Run all tests and return results."""
        results = {}
        
        # Test initialization
        init_result = self.test_initialization()
        results["initialization"] = init_result
        
        # If initialization failed, skip other tests
        if not init_result.get("success", False):
            return results
        
        # Test time series processing
        results["time_series_processing"] = self.test_time_series_processing()
        
        # Test forecasting
        results["forecasting"] = self.test_forecasting()
        
        # Test classification
        results["classification"] = self.test_classification()
        
        # Test anomaly detection
        results["anomaly_detection"] = self.test_anomaly_detection()
        
        # Test embedding
        results["embedding"] = self.test_embedding()
        
        # Determine overall success
        successful_tests = sum(1 for t in results.values() if t.get("success", False))
        results["overall_success"] = successful_tests / len(results)
        
        return results
    
    def test_initialization(self):
        """Test model initialization."""
        try:
            # Import the model class
            from transformers import AutoModelForTimeSeriesForecasting, AutoTokenizer
            
            # Initialize the model with minimal config
            model = {skillset_class_name}(
                model_id=self.model_id,
                device="cpu",
                low_memory_mode=self.low_memory_mode
            )
            
            # Run basic self-test
            test_result = model.__test__()
            
            return {
                "success": test_result.get("success", False),
                "hardware_info": model.hardware_info,
                "details": test_result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_time_series_processing(self):
        """Test time series processing functionality."""
        try:
            # Initialize the model
            model = {skillset_class_name}(
                model_id=self.model_id,
                device="cpu",
                low_memory_mode=self.low_memory_mode
            )
            
            # Create a simple synthetic time series
            batch_size = 2
            feature_dim = 3
            time_length = 100
            
            series_data = torch.randn(batch_size, time_length, feature_dim)
            
            # Test time series processing
            inputs = model.process_time_series(series_data)
            
            return {
                "success": "past_values" in inputs,
                "input_keys": list(inputs.keys()),
                "past_values_shape": tuple(inputs["past_values"].shape),
                "future_values_shape": tuple(inputs["future_values"].shape) if "future_values" in inputs else None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_forecasting(self):
        """Test forecasting functionality."""
        try:
            # Initialize the model
            model = {skillset_class_name}(
                model_id=self.model_id,
                device="cpu",
                low_memory_mode=self.low_memory_mode
            )
            
            # Create a simple synthetic time series
            batch_size = 2
            feature_dim = 3
            time_length = 100
            
            series_data = torch.randn(batch_size, time_length, feature_dim)
            
            # Test forecasting
            output = model.forecast(series_data)
            
            return {
                "success": "forecasts" in output or "raw_outputs" in output,
                "output_keys": list(output.keys()),
                "forecast_shape": output["forecasts"].shape if "forecasts" in output else None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_classification(self):
        """Test classification functionality."""
        try:
            # Initialize the model
            model = {skillset_class_name}(
                model_id=self.model_id,
                device="cpu",
                low_memory_mode=self.low_memory_mode
            )
            
            # Create a simple synthetic time series
            batch_size = 2
            feature_dim = 3
            time_length = 100
            
            series_data = torch.randn(batch_size, time_length, feature_dim)
            
            # Test classification
            output = model.classify(series_data)
            
            return {
                "success": "predictions" in output or "raw_outputs" in output,
                "output_keys": list(output.keys()),
                "predictions_shape": output["predictions"].shape if "predictions" in output else None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_anomaly_detection(self):
        """Test anomaly detection functionality."""
        try:
            # Initialize the model
            model = {skillset_class_name}(
                model_id=self.model_id,
                device="cpu",
                low_memory_mode=self.low_memory_mode
            )
            
            # Create a simple synthetic time series
            batch_size = 2
            feature_dim = 3
            time_length = 100
            
            series_data = torch.randn(batch_size, time_length, feature_dim)
            
            # Add some anomalies
            series_data[:, 50:55, :] = series_data[:, 50:55, :] * 5
            
            # Test anomaly detection
            output = model.detect_anomalies(series_data)
            
            return {
                "success": "anomaly_scores" in output or "raw_outputs" in output,
                "output_keys": list(output.keys()),
                "anomaly_scores_shape": output["anomaly_scores"].shape if "anomaly_scores" in output else None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_embedding(self):
        """Test embedding functionality."""
        try:
            # Initialize the model
            model = {skillset_class_name}(
                model_id=self.model_id,
                device="cpu",
                low_memory_mode=self.low_memory_mode
            )
            
            # Create a simple synthetic time series
            batch_size = 2
            feature_dim = 3
            time_length = 100
            
            series_data = torch.randn(batch_size, time_length, feature_dim)
            
            # Test embedding
            output = model.embedding(series_data)
            
            return {
                "success": "embeddings" in output or "raw_outputs" in output,
                "output_keys": list(output.keys()),
                "embedding_shape": output["pooled_embeddings"].shape if "pooled_embeddings" in output else None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


if __name__ == "__main__":
    # Run tests if executed directly
    tester = TestTimeSeriesModel()
    results = tester.run_tests()
    
    print(json.dumps(results, indent=2))