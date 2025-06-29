#!/usr/bin/env python3
"""
Enhanced Model Performance Predictor for the IPFS Accelerate framework.

This script implements advanced ML-based performance prediction as part of Phase 16 of the
IPFS Accelerate Python framework project. The predictor uses ensemble learning, custom feature
engineering, and uncertainty quantification to provide accurate predictions with confidence
scores for untested model-hardware combinations.

Key capabilities:
1. Predicts throughput, latency, and memory usage for unseen configurations with uncertainty estimates
2. Supports all hardware platforms: CPU, CUDA, ROCm, MPS, OpenVINO, QNN, WebNN, WebGPU
3. Works with both inference and training mode
4. Handles both single-node and distributed training
5. Provides confidence scoring for all predictions
6. Uses ensemble modeling for improved accuracy

Usage:
  python model_performance_predictor.py --train --database hardware_model_benchmark_db.parquet
  python model_performance_predictor.py --predict --model bert --hardware cuda --batch-size 32
  python model_performance_predictor.py --generate-matrix --output hardware_prediction_matrix.json
  python model_performance_predictor.py --evaluate --database hardware_model_benchmark_db.parquet
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Machine learning imports
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    HistGradientBoostingRegressor, VotingRegressor,
    StackingRegressor, BaggingRegressor, AdaBoostRegressor
)
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, PolynomialFeatures, 
    RobustScaler, PowerTransformer, QuantileTransformer
)
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    RandomizedSearchCV, KFold, StratifiedKFold
)
from sklearn.metrics import (
    mean_absolute_percentage_error, r2_score, mean_squared_error,
    mean_absolute_error, median_absolute_error, explained_variance_score
)
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import (
    ElasticNet, Ridge, Lasso, HuberRegressor,
    RANSACRegressor, TheilSenRegressor
)
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    SelectFromModel, RFE
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence warnings
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("model_performance_predictor")

# Global constants
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_DIR = PROJECT_ROOT / "test"
BENCHMARK_DIR = TEST_DIR / "benchmark_results"
PREDICTOR_DIR = BENCHMARK_DIR / "predictors"
MODEL_DIR = PREDICTOR_DIR / "models"

# Ensure directories exist
BENCHMARK_DIR.mkdir(exist_ok=True, parents=True)
PREDICTOR_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# Default database path
DEFAULT_DB_PATH = BENCHMARK_DIR / "hardware_model_benchmark_db.parquet"

# Prediction metrics
PREDICTION_METRICS = [
    "throughput",       # samples/sec
    "latency_mean",     # ms
    "memory_usage",     # MB
]

# Hardware categories for prediction
HARDWARE_CATEGORIES = [
    "cpu", 
    "cuda", 
    "mps", 
    "rocm", 
    "openvino", 
    "qnn",
    "webnn", 
    "webgpu",
    "distributed_ddp",
    "distributed_deepspeed",
    "distributed_fsdp"
]

# Model categories for prediction
MODEL_CATEGORIES = [
    "text_embedding",
    "text_generation",
    "vision",
    "audio",
    "multimodal",
    "video"
]

# Hardware capability mapping (for feature engineering)
HARDWARE_CAPABILITIES = {
    "cpu": {
        "compute_score": 1.0,
        "memory_bandwidth": 50.0,  # GB/s
        "supports_fp16": False,
        "supports_int8": True,
        "supports_int4": False,
        "parallel_cores": 1.0,     # Relative score
        "memory_hierarchy_efficiency": 0.8,
        "power_efficiency": 0.9,
        "memory_capacity": 64.0,   # GB
        "cache_efficiency": 0.75,
        "memory_latency": 100.0,   # ns
        "compute_precision": 1.0   # Relative score
    },
    "cuda": {
        "compute_score": 5.0,
        "memory_bandwidth": 900.0,  # GB/s
        "supports_fp16": True,
        "supports_int8": True,
        "supports_int4": True,
        "parallel_cores": 5.0,      # Relative score
        "memory_hierarchy_efficiency": 0.9,
        "power_efficiency": 0.6,
        "memory_capacity": 24.0,    # GB (typical)
        "cache_efficiency": 0.85,
        "memory_latency": 400.0,    # ns
        "compute_precision": 0.95   # Relative score
    },
    "rocm": {
        "compute_score": 4.5,
        "memory_bandwidth": 800.0,  # GB/s
        "supports_fp16": True,
        "supports_int8": True,
        "supports_int4": True,
        "parallel_cores": 4.5,      # Relative score
        "memory_hierarchy_efficiency": 0.85,
        "power_efficiency": 0.65,
        "memory_capacity": 16.0,    # GB (typical)
        "cache_efficiency": 0.82,
        "memory_latency": 450.0,    # ns
        "compute_precision": 0.93   # Relative score
    },
    "mps": {
        "compute_score": 3.0,
        "memory_bandwidth": 400.0,  # GB/s
        "supports_fp16": True,
        "supports_int8": True,
        "supports_int4": False,
        "parallel_cores": 3.0,      # Relative score
        "memory_hierarchy_efficiency": 0.95,
        "power_efficiency": 0.8,
        "memory_capacity": 32.0,    # GB (unified memory)
        "cache_efficiency": 0.9,
        "memory_latency": 200.0,    # ns
        "compute_precision": 0.98   # Relative score
    },
    "openvino": {
        "compute_score": 2.5,
        "memory_bandwidth": 100.0,  # GB/s
        "supports_fp16": True,
        "supports_int8": True,
        "supports_int4": False,
        "parallel_cores": 2.0,      # Relative score
        "memory_hierarchy_efficiency": 0.85,
        "power_efficiency": 0.85,
        "memory_capacity": 32.0,    # GB (system memory)
        "cache_efficiency": 0.8,
        "memory_latency": 120.0,    # ns
        "compute_precision": 0.9    # Relative score
    },
    "qnn": {
        "compute_score": 3.0,
        "memory_bandwidth": 80.0,   # GB/s
        "supports_fp16": True,
        "supports_int8": True,
        "supports_int4": True,
        "parallel_cores": 2.5,      # Relative score
        "memory_hierarchy_efficiency": 0.9,
        "power_efficiency": 0.95,   # Best power efficiency
        "memory_capacity": 8.0,     # GB (typical mobile)
        "cache_efficiency": 0.92,
        "memory_latency": 150.0,    # ns
        "compute_precision": 0.85   # Relative score
    },
    "webnn": {
        "compute_score": 1.5,
        "memory_bandwidth": 50.0,   # GB/s
        "supports_fp16": True,
        "supports_int8": False,
        "supports_int4": False,
        "parallel_cores": 1.5,      # Relative score
        "memory_hierarchy_efficiency": 0.7,
        "power_efficiency": 0.75,
        "memory_capacity": 4.0,     # GB (limited in browser)
        "cache_efficiency": 0.6,
        "memory_latency": 500.0,    # ns
        "compute_precision": 0.85   # Relative score
    },
    "webgpu": {
        "compute_score": 2.0,
        "memory_bandwidth": 100.0,  # GB/s
        "supports_fp16": True,
        "supports_int8": False,
        "supports_int4": False,
        "parallel_cores": 2.0,      # Relative score
        "memory_hierarchy_efficiency": 0.7,
        "power_efficiency": 0.7,
        "memory_capacity": 2.0,     # GB (limited in browser)
        "cache_efficiency": 0.5,
        "memory_latency": 600.0,    # ns
        "compute_precision": 0.8    # Relative score
    }
}

# Model family characteristics (for feature engineering)
MODEL_FAMILY_CHARACTERISTICS = {
    "text_embedding": {
        "compute_intensity": 0.7,
        "memory_intensity": 0.5,
        "parallelism_benefit": 0.8,
        "precision_sensitivity": 0.5,
        "batch_scalability": 0.9,
        "io_intensity": 0.3,
        "attention_mechanism_complexity": 0.6,
        "sequence_length_sensitivity": 0.7,
        "tokenization_overhead": 0.2,
        "cache_locality": 0.8,
        "quantization_efficiency": 0.9,  # High quantization efficiency
        "parameter_efficiency": 0.8,     # Good parameter-to-performance ratio
        "convergence_speed": 0.9,        # Fast convergence for training
        "gradient_communication_overhead": 0.3,
        "inter_layer_dependency": 0.5    # Moderate layer dependencies
    },
    "text_generation": {
        "compute_intensity": 0.8,
        "memory_intensity": 0.9,
        "parallelism_benefit": 0.7,
        "precision_sensitivity": 0.6,
        "batch_scalability": 0.7,
        "io_intensity": 0.4,
        "attention_mechanism_complexity": 0.9,
        "sequence_length_sensitivity": 0.9,
        "tokenization_overhead": 0.3,
        "cache_locality": 0.6,
        "quantization_efficiency": 0.7,  # Moderate quantization efficiency
        "parameter_efficiency": 0.6,     # Moderate parameter-to-performance ratio
        "convergence_speed": 0.5,        # Moderate convergence speed
        "gradient_communication_overhead": 0.7,
        "inter_layer_dependency": 0.8    # High layer dependencies due to attention
    },
    "vision": {
        "compute_intensity": 0.9,
        "memory_intensity": 0.6,
        "parallelism_benefit": 0.9,
        "precision_sensitivity": 0.4,
        "batch_scalability": 0.8,
        "io_intensity": 0.7,
        "attention_mechanism_complexity": 0.7,
        "sequence_length_sensitivity": 0.1,
        "tokenization_overhead": 0.0,
        "cache_locality": 0.7,
        "quantization_efficiency": 0.8,  # Good quantization efficiency
        "parameter_efficiency": 0.7,     # Good parameter-to-performance ratio
        "convergence_speed": 0.7,        # Good convergence speed
        "gradient_communication_overhead": 0.5,
        "inter_layer_dependency": 0.6    # Moderate layer dependencies
    },
    "audio": {
        "compute_intensity": 0.7,
        "memory_intensity": 0.7,
        "parallelism_benefit": 0.7,
        "precision_sensitivity": 0.7,
        "batch_scalability": 0.7,
        "io_intensity": 0.8,
        "attention_mechanism_complexity": 0.6,
        "sequence_length_sensitivity": 0.8,
        "tokenization_overhead": 0.1,
        "cache_locality": 0.5,
        "quantization_efficiency": 0.6,  # Moderate quantization efficiency
        "parameter_efficiency": 0.6,     # Moderate parameter-to-performance ratio
        "convergence_speed": 0.6,        # Moderate convergence speed
        "gradient_communication_overhead": 0.6,
        "inter_layer_dependency": 0.7    # Moderate-high layer dependencies
    },
    "multimodal": {
        "compute_intensity": 0.9,
        "memory_intensity": 0.9,
        "parallelism_benefit": 0.8,
        "precision_sensitivity": 0.6,
        "batch_scalability": 0.6,
        "io_intensity": 0.9,
        "attention_mechanism_complexity": 0.9,
        "sequence_length_sensitivity": 0.7,
        "tokenization_overhead": 0.4,
        "cache_locality": 0.4,
        "quantization_efficiency": 0.5,  # Lower quantization efficiency due to complexity
        "parameter_efficiency": 0.5,     # Lower parameter-to-performance ratio
        "convergence_speed": 0.4,        # Slower convergence due to complexity
        "gradient_communication_overhead": 0.8,
        "inter_layer_dependency": 0.9    # High layer dependencies across modalities
    },
    "video": {
        "compute_intensity": 1.0,
        "memory_intensity": 0.8,
        "parallelism_benefit": 1.0,
        "precision_sensitivity": 0.5,
        "batch_scalability": 0.5,
        "io_intensity": 1.0,
        "attention_mechanism_complexity": 0.8,
        "sequence_length_sensitivity": 0.8,
        "tokenization_overhead": 0.0,
        "cache_locality": 0.3,
        "quantization_efficiency": 0.6,  # Moderate quantization efficiency
        "parameter_efficiency": 0.4,     # Low parameter-to-performance ratio due to complexity
        "convergence_speed": 0.3,        # Slow convergence speed
        "gradient_communication_overhead": 0.9,
        "inter_layer_dependency": 0.9    # High layer dependencies
    }
}

class HardwareModelFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Advanced transformer that creates domain-specific features based on hardware and model characteristics.
    
    This transformer enriches the feature space with specialized domain knowledge about 
    hardware capabilities and model characteristics to significantly improve prediction accuracy.
    It captures complex interactions between model architectures and hardware capabilities
    that are critical for performance prediction.
    """
    
    def __init__(self, use_extended_features=True):
        """
        Initialize the transformer.
        
        Args:
            use_extended_features (bool): Whether to include extended (more complex) features
        """
        self.hardware_capabilities = HARDWARE_CAPABILITIES
        self.model_characteristics = MODEL_FAMILY_CHARACTERISTICS
        self.use_extended_features = use_extended_features
    
    def fit(self, X, y=None):
        """Fit transformer (stateless, returns self)."""
        return self
    
    def transform(self, X):
        """
        Transform the data by adding domain-specific features.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Transformed features with additional columns
        """
        # Create a copy to avoid modifying the original
        X_transformed = X.copy()
        
        # Track log warnings for unknown hardware or categories
        unknown_hardware = set()
        unknown_categories = set()
        
        # Iterate through each row
        for idx, row in X_transformed.iterrows():
            # Get hardware platform and model category
            hardware = row.get("hardware_platform")
            category = row.get("category")
            precision = row.get("precision_numeric", 32)
            batch_size = row.get("batch_size", 1)
            is_distributed = row.get("is_distributed", False)
            gpu_count = row.get("gpu_count", 1)
            model_size = row.get("model_size_estimate", 100000000)
            sequence_length = row.get("sequence_length", 128)
            mode = row.get("mode", "inference")
            
            # Skip if hardware or category not found
            if not hardware:
                continue
                
            if not category:
                continue
                
            # Normalize hardware name (handle variants)
            if isinstance(hardware, str):
                if hardware.startswith("cuda"):
                    hardware = "cuda"
                elif hardware.startswith("rocm"):
                    hardware = "rocm"
                elif hardware.startswith("mps"):
                    hardware = "mps"
                elif hardware.startswith("openvino"):
                    hardware = "openvino"
                elif hardware.startswith("qualcomm") or hardware.startswith("qnn"):
                    hardware = "qnn"
                elif hardware.startswith("distributed"):
                    # Handle distributed setups more granularly
                    if "ddp" in hardware.lower():
                        is_distributed = True
                        X_transformed.at[idx, "is_distributed"] = True
                        X_transformed.at[idx, "dist_strategy"] = "ddp"
                    elif "deepspeed" in hardware.lower():
                        is_distributed = True
                        X_transformed.at[idx, "is_distributed"] = True
                        X_transformed.at[idx, "dist_strategy"] = "deepspeed"
                    elif "fsdp" in hardware.lower():
                        is_distributed = True
                        X_transformed.at[idx, "is_distributed"] = True
                        X_transformed.at[idx, "dist_strategy"] = "fsdp"
            
            # Get hardware capabilities and model characteristics
            hw_caps = self.hardware_capabilities.get(hardware, {})
            model_chars = self.model_characteristics.get(category, {})
            
            if not hw_caps:
                if hardware not in unknown_hardware:
                    unknown_hardware.add(hardware)
                continue
                
            if not model_chars:
                if category not in unknown_categories:
                    unknown_categories.add(category)
                continue
            
            # Estimate scaling parameters
            log_batch_size = np.log2(max(batch_size, 1))
            log_model_size = np.log10(max(model_size, 1000))
            log_seq_length = np.log2(max(sequence_length, 1))
            
            # ===== Basic Features =====
            
            # 1. Compute-memory balance
            compute_memory_ratio = (
                hw_caps.get("compute_score", 1.0) / 
                model_chars.get("compute_intensity", 0.5)
            ) / (
                hw_caps.get("memory_bandwidth", 50.0) / 
                model_chars.get("memory_intensity", 0.5)
            )
            X_transformed.at[idx, "compute_memory_ratio"] = compute_memory_ratio
            
            # 2. Parallelism utilization potential
            parallelism_score = (
                hw_caps.get("parallel_cores", 1.0) * 
                model_chars.get("parallelism_benefit", 0.5)
            )
            X_transformed.at[idx, "parallelism_score"] = parallelism_score
            
            # 3. Precision efficiency score
            precision_factor = 1.0
            if precision == 16 and hw_caps.get("supports_fp16", False):
                precision_factor = 1.5 * (1.0 - model_chars.get("precision_sensitivity", 0.5) * 0.5)
            elif precision == 8 and hw_caps.get("supports_int8", False):
                precision_factor = 2.0 * (1.0 - model_chars.get("precision_sensitivity", 0.5) * 0.7)
            elif precision == 4 and hw_caps.get("supports_int4", False):
                precision_factor = 2.5 * (1.0 - model_chars.get("precision_sensitivity", 0.5) * 0.9)
            
            # Apply quantization efficiency factor from model characteristics
            precision_factor *= model_chars.get("quantization_efficiency", 0.8)
            
            X_transformed.at[idx, "precision_efficiency"] = precision_factor
            
            # 4. Batch efficiency score (how efficiently can this hardware scale with batch size)
            batch_efficiency = (
                1.0 + 
                log_batch_size * 
                model_chars.get("batch_scalability", 0.5) *
                (hw_caps.get("memory_hierarchy_efficiency", 0.8) ** 0.5)
            )
            X_transformed.at[idx, "batch_efficiency"] = batch_efficiency
            
            # 5. Distributed scaling factor
            if is_distributed and gpu_count > 1:
                # Advanced distributed scaling model based on hardware and model
                # Incorporate communication overhead from model characteristics
                comm_overhead = model_chars.get("gradient_communication_overhead", 0.5)
                # Logarithmic scaling efficiency with diminishing returns adjusted by communication overhead
                scaling_efficiency = (0.9 - (comm_overhead * 0.2)) ** np.log2(gpu_count)
                distributed_factor = gpu_count * scaling_efficiency
                
                # Additional factors for distributed training strategies
                dist_strategy = row.get("dist_strategy", "ddp")
                if dist_strategy == "deepspeed":
                    # DeepSpeed typically has better memory efficiency
                    distributed_factor *= 1.1
                elif dist_strategy == "fsdp":
                    # FSDP may have higher communication overhead for some models
                    distributed_factor *= (0.9 + (0.1 * (1 - comm_overhead)))
            else:
                distributed_factor = 1.0
            X_transformed.at[idx, "distributed_factor"] = distributed_factor
            
            # 6. Memory pressure estimate
            memory_pressure = (
                model_chars.get("memory_intensity", 0.5) * 
                batch_size *
                (model_size / 1e8) ** 0.5 /  # Scale with square root of model size
                hw_caps.get("memory_bandwidth", 50.0)
            )
            X_transformed.at[idx, "memory_pressure"] = memory_pressure
            
            # 7. Hardware-model compatibility score (higher is better)
            compatibility_score = (
                hw_caps.get("compute_score", 1.0) * model_chars.get("compute_intensity", 0.5) +
                hw_caps.get("memory_bandwidth", 50.0) * model_chars.get("memory_intensity", 0.5) +
                hw_caps.get("memory_hierarchy_efficiency", 0.8) * 0.3 +
                hw_caps.get("power_efficiency", 0.7) * 0.2
            ) / 2.0
            X_transformed.at[idx, "compatibility_score"] = compatibility_score
            
            # ===== Extended Features =====
            if self.use_extended_features:
                # 8. Cache efficiency based on model and hardware
                cache_efficiency = (
                    hw_caps.get("cache_efficiency", 0.8) * 
                    model_chars.get("cache_locality", 0.7)
                )
                X_transformed.at[idx, "cache_efficiency"] = cache_efficiency
                
                # 9. Memory capacity pressure - how much of available memory is used
                memory_capacity_pressure = (
                    (model_size / 1e6) *  # Convert to MB
                    batch_size * 
                    (1 + (0.2 * is_distributed)) /  # Small overhead for distributed
                    (hw_caps.get("memory_capacity", 16.0) * 1024)  # Convert to MB
                )
                X_transformed.at[idx, "memory_capacity_pressure"] = memory_capacity_pressure
                
                # 10. Sequence length scaling factor (especially important for transformers)
                sequence_scaling = (
                    1.0 + 
                    (log_seq_length / 5.0) * 
                    model_chars.get("sequence_length_sensitivity", 0.5)
                )
                X_transformed.at[idx, "sequence_scaling"] = sequence_scaling
                
                # 11. Compute-precision balance
                compute_precision_balance = (
                    hw_caps.get("compute_score", 1.0) * 
                    hw_caps.get("compute_precision", 0.9) / 
                    model_chars.get("precision_sensitivity", 0.5)
                )
                X_transformed.at[idx, "compute_precision_balance"] = compute_precision_balance
                
                # 12. Model scale efficiency (how efficiently hardware handles larger models)
                model_scale_efficiency = (
                    1.0 / (1.0 + 
                    0.1 * log_model_size * 
                    (1.0 - hw_caps.get("memory_hierarchy_efficiency", 0.8)))
                )
                X_transformed.at[idx, "model_scale_efficiency"] = model_scale_efficiency
                
                # 13. Attention mechanism efficiency
                # (Important for transformer models, affects both compute and memory)
                attention_efficiency = (
                    hw_caps.get("compute_score", 1.0) / 
                    model_chars.get("attention_mechanism_complexity", 0.5) *
                    hw_caps.get("memory_bandwidth", 50.0) / 400.0  # Normalize to typical bandwidth
                )
                X_transformed.at[idx, "attention_efficiency"] = attention_efficiency
                
                # 14. Power efficiency for mobile/edge devices
                power_efficiency = hw_caps.get("power_efficiency", 0.7)
                X_transformed.at[idx, "power_efficiency"] = power_efficiency
                
                # 15. Parameter efficiency score - how efficiently the model uses its parameters
                parameter_efficiency = model_chars.get("parameter_efficiency", 0.7)
                X_transformed.at[idx, "parameter_efficiency"] = parameter_efficiency
                
                # 16. Device memory throughput utilization (different from bandwidth)
                memory_throughput_score = (
                    hw_caps.get("memory_bandwidth", 50.0) * 
                    (1.0 - 0.2 * hw_caps.get("memory_latency", 200.0) / 500.0) *
                    model_chars.get("memory_intensity", 0.5)
                )
                X_transformed.at[idx, "memory_throughput_score"] = memory_throughput_score / 100.0  # Normalize
                
                # 17. Training vs Inference specific features
                if mode == "training":
                    # Training efficiency score
                    training_efficiency = (
                        model_chars.get("convergence_speed", 0.7) *
                        hw_caps.get("compute_score", 1.0) /
                        (1.0 + 0.5 * model_chars.get("gradient_communication_overhead", 0.5))
                    )
                    X_transformed.at[idx, "training_efficiency"] = training_efficiency
                    
                    # Backward pass overhead estimate
                    backward_overhead = 1.0 + 0.8 * model_chars.get("inter_layer_dependency", 0.5)
                    X_transformed.at[idx, "backward_overhead"] = backward_overhead
                else:
                    # Inference-specific throughput potential
                    inference_throughput_potential = (
                        hw_caps.get("compute_score", 1.0) * 
                        hw_caps.get("memory_bandwidth", 50.0) / 100.0 * 
                        precision_factor *
                        batch_efficiency
                    )
                    X_transformed.at[idx, "inference_throughput_potential"] = inference_throughput_potential
                
                # 18. IO bottleneck risk
                io_bottleneck_risk = (
                    model_chars.get("io_intensity", 0.5) /
                    (hw_caps.get("memory_bandwidth", 50.0) / 200.0)  # Normalize
                )
                X_transformed.at[idx, "io_bottleneck_risk"] = io_bottleneck_risk
                
                # 19. Latency sensitivity score
                latency_sensitivity = (
                    1.0 / (hw_caps.get("memory_latency", 200.0) / 200.0) *  # Normalize and invert
                    (0.7 + 0.3 * (1.0 - model_chars.get("sequence_length_sensitivity", 0.5)))
                )
                X_transformed.at[idx, "latency_sensitivity"] = latency_sensitivity
                
                # 20. Combined hardware-model efficiency
                combined_efficiency = (
                    (compute_memory_ratio * 0.3) + 
                    (parallelism_score * 0.2) + 
                    (precision_factor * 0.15) + 
                    (batch_efficiency * 0.15) + 
                    (cache_efficiency * 0.1) + 
                    (parameter_efficiency * 0.1)
                )
                X_transformed.at[idx, "combined_efficiency"] = combined_efficiency
                
                # 21. Hardware-model interaction terms
                X_transformed.at[idx, "compute_memory_interaction"] = (
                    hw_caps.get("compute_score", 1.0) * 
                    hw_caps.get("memory_bandwidth", 50.0) / 
                    200.0  # Normalize
                )
                
                # 22. Sequence-batch interaction (how sequence length affects batch processing)
                X_transformed.at[idx, "seq_batch_interaction"] = (
                    sequence_scaling * batch_efficiency
                )
                
        # Log warnings about unknown hardware or categories
        if unknown_hardware:
            logging.warning(f"Unknown hardware platforms encountered: {unknown_hardware}")
        if unknown_categories:
            logging.warning(f"Unknown model categories encountered: {unknown_categories}")
            
        return X_transformed

def load_benchmark_data(db_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load benchmark data from the database.
    
    Args:
        db_path (str): Path to the benchmark database
        
    Returns:
        pd.DataFrame: Benchmark data
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    
    try:
        if not os.path.exists(db_path):
            logger.error(f"Benchmark database not found at {db_path}")
            return pd.DataFrame()
        
        df = pd.read_parquet(db_path)
        
        # Filter for completed benchmarks only
        df = df[df["status"] == "success"]
        
        # Fill missing values
        for metric in PREDICTION_METRICS:
            if metric in df.columns:
                df[metric] = df[metric].fillna(0)
        
        logger.info(f"Loaded {len(df)} benchmark results from {db_path}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading benchmark data: {e}")
        return pd.DataFrame()

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Advanced preprocessing pipeline for benchmark data.
    
    This function implements a comprehensive preprocessing pipeline for benchmark data,
    including feature extraction, feature engineering, outlier detection, and handling
    of missing values. It uses domain-specific knowledge to create informative features
    for hardware-aware performance prediction.
    
    Args:
        df (pd.DataFrame): Benchmark data
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Preprocessed data and preprocessing info
    """
    if df.empty:
        logger.error("Empty dataframe provided for preprocessing")
        return df, {}
    
    try:
        logger.info(f"Starting preprocessing of {len(df)} benchmark records")
        original_count = len(df)
        
        # ===== Basic Processing =====
        
        # Extract hardware platform from hardware column
        df["hardware_platform"] = df["hardware"].apply(
            lambda x: x.split("_")[0] if isinstance(x, str) and "_" in x else x
        )
        
        # Normalize hardware platform names
        df["hardware_platform"] = df["hardware_platform"].apply(
            lambda x: "cuda" if isinstance(x, str) and x.startswith("cuda") else 
                      "rocm" if isinstance(x, str) and x.startswith("rocm") else
                      "mps" if isinstance(x, str) and x.startswith("mps") else
                      "openvino" if isinstance(x, str) and x.startswith("openvino") else
                      "qnn" if isinstance(x, str) and (x.startswith("qualcomm") or x.startswith("qnn")) else
                      x
        )
        
        # Add distributed training flag and strategy
        df["is_distributed"] = df["hardware"].apply(
            lambda x: "distributed" in str(x)
        )
        
        # Extract distributed strategy
        def extract_dist_strategy(hw_str):
            if not isinstance(hw_str, str) or "distributed" not in hw_str.lower():
                return "none"
            if "ddp" in hw_str.lower():
                return "ddp"
            elif "deepspeed" in hw_str.lower():
                return "deepspeed"
            elif "fsdp" in hw_str.lower():
                return "fsdp"
            elif "zero" in hw_str.lower():
                return "zero"
            else:
                return "other"
                
        df["dist_strategy"] = df["hardware"].apply(extract_dist_strategy)
        
        # Add GPU count for distributed training
        df["gpu_count"] = df.apply(
            lambda row: row.get("total_gpus", 1) if row.get("is_distributed", False) else 1, 
            axis=1
        )
        
        # Convert precision to numeric (fp32=32, fp16=16, int8=8, int4=4)
        df["precision_numeric"] = df["precision"].apply(
            lambda x: 32 if x == "fp32" else 
                    16 if x == "fp16" else 
                    8 if x == "int8" else 
                    4 if x == "int4" else 32
        )
        
        # ===== Model Size Estimation =====
        
        # Add model size estimate based on name
        df["model_size_estimate"] = df["model_name"].apply(_estimate_model_size)
        df["model_size_log"] = np.log10(df["model_size_estimate"] + 1)
        
        # Create model size category
        df["model_size_category"] = pd.cut(
            df["model_size_estimate"],
            bins=[0, 10_000_000, 100_000_000, 1_000_000_000, float('inf')],
            labels=["tiny", "small", "medium", "large"]
        )
        
        # ===== Handle Missing Values =====
        
        # Add sequence length default if not present
        if "sequence_length" not in df.columns:
            df["sequence_length"] = 128
        else:
            # Fill missing sequence lengths with defaults by category
            category_seq_defaults = {
                "text_embedding": 128,
                "text_generation": 512,
                "vision": 224,  # Common image size
                "audio": 16000,  # 1 second of audio at 16kHz
                "multimodal": 256,
                "video": 30  # 30 frames
            }
            
            for category, default_len in category_seq_defaults.items():
                mask = (df["category"] == category) & (df["sequence_length"].isna())
                df.loc[mask, "sequence_length"] = default_len
                
            # Fill any remaining NAs with global default
            df["sequence_length"] = df["sequence_length"].fillna(128)
        
        # Fill missing mode with 'inference'
        if "mode" in df.columns:
            df["mode"] = df["mode"].fillna("inference")
        else:
            df["mode"] = "inference"
            
        # ===== Create Basic Interaction Features =====
        
        df["batch_x_precision"] = df["batch_size"] * (32 / df["precision_numeric"])
        df["batch_x_gpus"] = df["batch_size"] * df["gpu_count"]
        df["model_size_x_batch"] = df["model_size_log"] * df["batch_size"]
        df["seq_length_log"] = np.log2(df["sequence_length"].clip(lower=1))
        
        # Create model complexity score (combination of size and sequence length)
        df["model_complexity_score"] = df["model_size_log"] * (1 + 0.1 * df["seq_length_log"])
        
        # ===== Model Category Encoding with Domain Knowledge =====
        
        # One-hot encode model category with domain knowledge
        if "category" in df.columns:
            # Fill missing categories based on model name
            if df["category"].isna().any():
                def infer_category(model_name):
                    if not isinstance(model_name, str):
                        return "text_embedding"  # Default
                    
                    model_lower = model_name.lower()
                    if any(kw in model_lower for kw in ['clip', 'vit', 'resnet', 'swin']):
                        return "vision"
                    elif any(kw in model_lower for kw in ['gpt', 't5', 'llama', 'baichuan', 'falcon']):
                        return "text_generation"
                    elif any(kw in model_lower for kw in ['bert', 'roberta', 'electra', 'distilbert']):
                        return "text_embedding"
                    elif any(kw in model_lower for kw in ['whisper', 'wav2vec', 'clap']):
                        return "audio"
                    elif any(kw in model_lower for kw in ['llava', 'blip']):
                        return "multimodal"
                    else:
                        return "text_embedding"  # Default fallback
                
                df.loc[df["category"].isna(), "category"] = df.loc[df["category"].isna(), "model_name"].apply(infer_category)
        
        # ===== Advanced Feature Engineering =====
        
        # Apply advanced feature transformer with extended features
        feature_transformer = HardwareModelFeatureTransformer(use_extended_features=True)
        df_transformed = feature_transformer.transform(df)
        
        # Merge back any new columns
        for col in df_transformed.columns:
            if col not in df.columns:
                df[col] = df_transformed[col]
                
        # ===== Detect and Handle Outliers =====
        
        # Detect outliers in performance metrics and create filtered version
        for metric in PREDICTION_METRICS:
            if metric in df.columns:
                # Skip if column is all NaN
                if df[metric].isna().all():
                    continue
                    
                # Fill NaN values with median to allow outlier calculation
                temp_values = df[metric].fillna(df[metric].median())
                
                # Calculate outlier bounds using IQR method
                Q1 = temp_values.quantile(0.05)
                Q3 = temp_values.quantile(0.95)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Create outlier flag for this metric
                df[f"{metric}_outlier"] = (temp_values < lower_bound) | (temp_values > upper_bound)
                
                # Log outlier statistics
                outlier_count = df[f"{metric}_outlier"].sum()
                if outlier_count > 0:
                    logger.info(f"Detected {outlier_count} outliers in {metric} ({outlier_count/len(df):.1%})")
        
        # Create overall outlier flag
        outlier_cols = [col for col in df.columns if col.endswith('_outlier')]
        if outlier_cols:
            df["is_outlier"] = df[outlier_cols].any(axis=1)
            logger.info(f"Detected {df['is_outlier'].sum()} total outliers ({df['is_outlier'].sum()/len(df):.1%})")
        else:
            df["is_outlier"] = False
        
        # Create a filtered dataset without outliers for training
        df_filtered = df[~df["is_outlier"]].copy()
        
        # Check if we filtered too many rows
        if len(df_filtered) < 0.7 * original_count:
            logger.warning(f"Outlier removal filtered too many rows ({len(df_filtered)}/{original_count}). Using original dataset.")
            df_filtered = df.copy()
        
        # ===== Define Feature Sets =====
        
        # Base feature columns (always included)
        base_feature_cols = [
            "batch_size",
            "precision_numeric",
            "gpu_count",
            "is_distributed",
            "mode",
            "category",
            "hardware_platform",
            "model_size_log",
            "sequence_length",
            "batch_x_precision", 
            "batch_x_gpus",
            "model_size_x_batch",
            "seq_length_log",
            "model_complexity_score"
        ]
        
        # Derived feature columns (added by HardwareModelFeatureTransformer)
        basic_derived_cols = [
            "compute_memory_ratio",
            "parallelism_score",
            "precision_efficiency",
            "batch_efficiency",
            "distributed_factor",
            "memory_pressure",
            "compatibility_score"
        ]
        
        # Extended feature columns (added by HardwareModelFeatureTransformer with extended features)
        extended_derived_cols = [
            "cache_efficiency",
            "memory_capacity_pressure",
            "sequence_scaling",
            "compute_precision_balance",
            "model_scale_efficiency",
            "attention_efficiency",
            "power_efficiency",
            "parameter_efficiency",
            "memory_throughput_score",
            "training_efficiency",
            "backward_overhead",
            "inference_throughput_potential",
            "io_bottleneck_risk",
            "latency_sensitivity",
            "combined_efficiency",
            "compute_memory_interaction",
            "seq_batch_interaction"
        ]
        
        # Get actual available columns from basic and extended features
        available_basic_derived = [col for col in basic_derived_cols if col in df_filtered.columns]
        available_extended_derived = [col for col in extended_derived_cols if col in df_filtered.columns]
        
        # Combine all feature columns
        feature_cols = base_feature_cols + available_basic_derived + available_extended_derived
        
        # Remove columns that don't exist
        feature_cols = [col for col in feature_cols if col in df_filtered.columns]
        
        # ===== Handle Missing Values in Features =====
        
        # For numeric features, fill NaN with median
        numeric_cols = df_filtered[feature_cols].select_dtypes(include=['number']).columns.tolist()
        for col in numeric_cols:
            median_val = df_filtered[col].median()
            df_filtered[col] = df_filtered[col].fillna(median_val)
            
        # For categorical features, fill NaN with most frequent value
        cat_cols = [c for c in feature_cols if c not in numeric_cols]
        for col in cat_cols:
            mode_val = df_filtered[col].mode().iloc[0] if not df_filtered[col].mode().empty else None
            df_filtered[col] = df_filtered[col].fillna(mode_val)
        
        # ===== Data Quality Checks =====
        
        # Check for remaining NaN values in feature columns
        remaining_nans = df_filtered[feature_cols].isna().sum().sum()
        if remaining_nans > 0:
            logger.warning(f"There are still {remaining_nans} NaN values in feature columns after preprocessing")
            # Drop rows with NaN values in feature columns as a last resort
            df_filtered = df_filtered.dropna(subset=feature_cols)
            logger.info(f"Dropped rows with NaN values. Remaining rows: {len(df_filtered)}")
        
        # Check for NaN values in target columns
        for target in PREDICTION_METRICS:
            if target in df_filtered.columns and df_filtered[target].isna().any():
                logger.warning(f"Target column {target} contains {df_filtered[target].isna().sum()} NaN values")
                # Drop rows with NaN values in this target
                df_filtered = df_filtered.dropna(subset=[target])
                logger.info(f"Dropped rows with NaN in target {target}. Remaining rows: {len(df_filtered)}")
        
        # Final sanity check - if too many rows were removed, use original data
        if len(df_filtered) < 20 or len(df_filtered) < 0.2 * original_count:
            logger.warning(f"Too many rows were removed during preprocessing ({len(df_filtered)}/{original_count}). Using original dataset with basic NaN handling.")
            df_filtered = df.copy()
            # Basic NaN handling
            for col in feature_cols:
                if col in df_filtered.columns:
                    if df_filtered[col].dtype.kind in 'fc':  # float or complex
                        df_filtered[col] = df_filtered[col].fillna(df_filtered[col].median())
                    else:
                        df_filtered[col] = df_filtered[col].fillna(df_filtered[col].mode().iloc[0] if not df_filtered[col].mode().empty else None)
            
            # Handle NaN in target columns
            for target in PREDICTION_METRICS:
                if target in df_filtered.columns:
                    df_filtered = df_filtered.dropna(subset=[target])
        
        # ===== Create Preprocessing Info Dictionary =====
        
        # Create numeric and categorical column lists
        numeric_columns = df_filtered[feature_cols].select_dtypes(include=['number']).columns.tolist()
        categorical_columns = [c for c in feature_cols if c not in numeric_columns]
        
        # Create preprocessing info dictionary
        preprocessing_info = {
            "feature_columns": feature_cols,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "target_columns": PREDICTION_METRICS,
            "timestamp": datetime.now().isoformat(),
            "hardware_capabilities": HARDWARE_CAPABILITIES,
            "model_characteristics": MODEL_FAMILY_CHARACTERISTICS,
            "original_count": original_count,
            "preprocessed_count": len(df_filtered),
            "outlier_count": df["is_outlier"].sum() if "is_outlier" in df else 0,
            "has_extended_features": len(available_extended_derived) > 0,
            "preprocessing_version": "2.0.0"
        }
        
        logger.info(f"Preprocessed data: {len(df_filtered)} rows (from {original_count}), {len(feature_cols)} features")
        return df_filtered, preprocessing_info
    
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return df, {}
        
def _estimate_model_size(model_name: str) -> int:
    """
    Estimate model size in parameters based on model name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Estimated number of parameters
    """
    if not isinstance(model_name, str):
        return 100000000  # Default to 100M if not a string
        
    model_name_lower = model_name.lower()
    
    # Look for size indicators in the model name
    if "tiny" in model_name_lower:
        return 10000000  # 10M parameters
    elif "small" in model_name_lower:
        return 50000000  # 50M parameters
    elif "base" in model_name_lower:
        return 100000000  # 100M parameters
    elif "large" in model_name_lower:
        return 300000000  # 300M parameters
    elif "xl" in model_name_lower or "huge" in model_name_lower:
        return 1000000000  # 1B parameters
    
    # Check for specific models
    if "bert" in model_name_lower:
        if "tiny" in model_name_lower:
            return 4000000  # 4M parameters
        elif "mini" in model_name_lower:
            return 11000000  # 11M parameters
        elif "small" in model_name_lower:
            return 29000000  # 29M parameters
        elif "base" in model_name_lower:
            return 110000000  # 110M parameters
        elif "large" in model_name_lower:
            return 340000000  # 340M parameters
        else:
            return 110000000  # Default to base size
    elif "t5" in model_name_lower:
        if "small" in model_name_lower:
            return 60000000  # 60M parameters
        elif "base" in model_name_lower:
            return 220000000  # 220M parameters
        elif "large" in model_name_lower:
            return 770000000  # 770M parameters
        elif "3b" in model_name_lower:
            return 3000000000  # 3B parameters
        elif "11b" in model_name_lower:
            return 11000000000  # 11B parameters
        else:
            return 220000000  # Default to base size
    elif "gpt2" in model_name_lower:
        if "small" in model_name_lower or "sm" in model_name_lower:
            return 124000000  # 124M parameters
        elif "medium" in model_name_lower or "med" in model_name_lower:
            return 355000000  # 355M parameters
        elif "large" in model_name_lower or "lg" in model_name_lower:
            return 774000000  # 774M parameters
        elif "xl" in model_name_lower:
            return 1500000000  # 1.5B parameters
        else:
            return 124000000  # Default to small size
    
    # Default size if not recognized
    return 100000000  # 100M parameters

class ModelComplexityTransformer(BaseEstimator, TransformerMixin):
    """
    Advanced feature transformer that creates interaction terms and polynomial features
    based on model complexity and hardware characteristics.
    
    This transformer intelligently creates polynomial features and interaction terms
    focused on the most important relationships between hardware capabilities and
    model characteristics. It implements feature selection to avoid combinatorial
    explosion while capturing key interactions.
    """
    
    def __init__(self, degree=2, interaction_only=True, feature_selection='auto', 
                 n_features_to_select=20, importance_threshold=0.01, 
                 include_hardware_model_interactions=True):
        """
        Initialize the transformer.
        
        Args:
            degree (int): Maximum degree of polynomial features
            interaction_only (bool): Whether to include only interaction terms
            feature_selection (str): Method for feature selection ('auto', 'k_best', 'model_based', 'none')
            n_features_to_select (int): Number of features to select if using k_best
            importance_threshold (float): Threshold for feature importance if using model_based
            include_hardware_model_interactions (bool): Whether to add special hardware-model interaction terms
        """
        self.degree = degree
        self.interaction_only = interaction_only
        self.feature_selection = feature_selection
        self.n_features_to_select = n_features_to_select
        self.importance_threshold = importance_threshold
        self.include_hardware_model_interactions = include_hardware_model_interactions
        
        # Base polynomial features
        self.poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
        
        # Feature selector (initialized in fit)
        self.selector = None
        
        # Hardware and model feature names (set in fit)
        self.hardware_features = []
        self.model_features = []
        self.performance_features = []
        self.original_feature_names = None
        self.poly_feature_names = None
        
    def _get_feature_names(self, X):
        """Get feature names by category."""
        feature_categories = {
            'hardware': ['compute_score', 'memory_bandwidth', 'parallel_cores', 'memory_hierarchy_efficiency',
                         'power_efficiency', 'cache_efficiency', 'memory_latency', 'compute_precision',
                         'hardware_platform'],
            'model': ['model_size', 'category', 'model_name', 'model_size_log', 'model_size_estimate',
                      'model_size_category', 'attention_mechanism'],
            'performance': ['batch_size', 'precision_numeric', 'sequence_length', 'gpu_count',
                           'is_distributed', 'mode']
        }
        
        self.hardware_features = [col for col in X.columns if any(hw_feat in col for hw_feat in feature_categories['hardware'])]
        self.model_features = [col for col in X.columns if any(model_feat in col for model_feat in feature_categories['model'])]
        self.performance_features = [col for col in X.columns if any(perf_feat in col for perf_feat in feature_categories['performance'])]
        
    def fit(self, X, y=None):
        """
        Fit transformer.
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series, optional): Target values for feature selection
            
        Returns:
            self: The fitted transformer
        """
        # Check if feature selection requires target values
        if self.feature_selection in ['k_best', 'model_based'] and y is None:
            raise ValueError(f"Feature selection method '{self.feature_selection}' requires target values")
        
        # Get feature categories for intelligent interactions
        self._get_feature_names(X)
        self.original_feature_names = X.columns.tolist()
        
        # Fit polynomial features to input data
        numeric_X = X.select_dtypes(include=['number']).copy()
        self.poly.fit(numeric_X)
        
        # Get polynomial feature names
        try:
            if hasattr(self.poly, 'get_feature_names_out'):
                self.poly_feature_names = self.poly.get_feature_names_out(numeric_X.columns.tolist())
            else:
                self.poly_feature_names = [f"poly_{i}" for i in range(self.poly.n_output_features_)]
        except:
            self.poly_feature_names = [f"poly_{i}" for i in range(self.poly.n_output_features_)]
        
        # Set up feature selection if enabled
        if self.feature_selection != 'none' and y is not None:
            # Transform data with polynomial features
            poly_X = self.poly.transform(numeric_X)
            
            if self.feature_selection == 'k_best':
                # Use SelectKBest for feature selection
                self.selector = SelectKBest(f_regression, k=min(self.n_features_to_select, poly_X.shape[1]))
                self.selector.fit(poly_X, y)
            
            elif self.feature_selection == 'model_based':
                # Use model-based feature selection
                base_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                self.selector = SelectFromModel(base_model, threshold=self.importance_threshold)
                self.selector.fit(poly_X, y)
            
            elif self.feature_selection == 'auto':
                # Automatically choose based on data size
                if poly_X.shape[1] > 100:  # Many features
                    # Start with model-based selection
                    base_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                    self.selector = SelectFromModel(base_model, threshold=self.importance_threshold)
                    try:
                        self.selector.fit(poly_X, y)
                        # If too few features selected, fall back to k-best
                        if sum(self.selector.get_support()) < 5:
                            self.selector = SelectKBest(f_regression, k=min(self.n_features_to_select, poly_X.shape[1]))
                            self.selector.fit(poly_X, y)
                    except:
                        # Fall back to k-best if model-based fails
                        self.selector = SelectKBest(f_regression, k=min(self.n_features_to_select, poly_X.shape[1]))
                        self.selector.fit(poly_X, y)
                else:
                    # Use k-best for smaller feature sets
                    self.selector = SelectKBest(f_regression, k=min(self.n_features_to_select, poly_X.shape[1]))
                    self.selector.fit(poly_X, y)
        
        return self
    
    def transform(self, X):
        """
        Transform the data with polynomial features and intelligent feature selection.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            np.ndarray: Transformed features
        """
        # Transform only numeric columns
        numeric_X = X.select_dtypes(include=['number']).copy()
        
        # Check if we have data to transform
        if numeric_X.shape[1] == 0:
            return X.values
        
        # Apply polynomial features
        poly_X = self.poly.transform(numeric_X)
        
        # Apply feature selection if enabled
        if self.selector is not None:
            poly_X = self.selector.transform(poly_X)
        
        # Create custom hardware-model interactions if enabled
        if self.include_hardware_model_interactions:
            # Get hardware and model numeric features
            hw_features = [col for col in numeric_X.columns if col in self.hardware_features]
            model_features = [col for col in numeric_X.columns if col in self.model_features]
            perf_features = [col for col in numeric_X.columns if col in self.performance_features]
            
            # Create interactions manually for specific combinations
            hw_model_interactions = np.zeros((numeric_X.shape[0], 0))
            
            if hw_features and model_features:
                for hw_col in hw_features:
                    for model_col in model_features:
                        # Create interaction term
                        interaction = numeric_X[hw_col].values.reshape(-1, 1) * numeric_X[model_col].values.reshape(-1, 1)
                        hw_model_interactions = np.hstack((hw_model_interactions, interaction))
            
            # Add performance interactions
            if hw_features and perf_features:
                for hw_col in hw_features:
                    for perf_col in perf_features:
                        # Create interaction term
                        interaction = numeric_X[hw_col].values.reshape(-1, 1) * numeric_X[perf_col].values.reshape(-1, 1)
                        hw_model_interactions = np.hstack((hw_model_interactions, interaction))
            
            # Combine with polynomial features
            if hw_model_interactions.shape[1] > 0:
                poly_X = np.hstack((poly_X, hw_model_interactions))
        
        return poly_X
    
    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names.
        
        Args:
            input_features: Input feature names
            
        Returns:
            list: Output feature names
        """
        if input_features is None:
            input_features = self.original_feature_names or []
            
        # Get polynomial feature names
        if self.poly_feature_names is not None:
            poly_names = list(self.poly_feature_names)
        else:
            poly_names = [f"poly_{i}" for i in range(self.poly.n_output_features_)]
        
        # Apply feature selection if enabled
        if self.selector is not None:
            mask = self.selector.get_support()
            poly_names = [name for name, selected in zip(poly_names, mask) if selected]
        
        # Add hardware-model interaction names if enabled
        if self.include_hardware_model_interactions:
            hw_features = [col for col in input_features if col in self.hardware_features]
            model_features = [col for col in input_features if col in self.model_features]
            perf_features = [col for col in input_features if col in self.performance_features]
            
            for hw_col in hw_features:
                for model_col in model_features:
                    poly_names.append(f"{hw_col}_{model_col}_interaction")
            
            for hw_col in hw_features:
                for perf_col in perf_features:
                    poly_names.append(f"{hw_col}_{perf_col}_interaction")
        
        return poly_names

def train_prediction_models(
    df: pd.DataFrame, 
    preprocessing_info: Dict[str, Any],
    test_size: float = 0.2,
    random_state: int = 42,
    hyperparameter_tuning: bool = True,
    use_ensemble: bool = True,
    model_complexity: str = 'auto',  # 'simple', 'standard', 'complex', or 'auto'
    use_cross_validation: bool = True,
    cv_folds: int = 5,
    feature_selection: str = 'auto',  # 'none', 'k_best', 'model_based', or 'auto'
    uncertainty_estimation: bool = True,
    n_jobs: int = -1,  # Use all available cores
    progress_callback: Optional[Callable] = None  # Optional callback for progress tracking
) -> Dict[str, Any]:
    """
    Advanced training pipeline for performance prediction models with ensemble learning,
    hyperparameter optimization, and uncertainty estimation.
    
    This function implements a comprehensive ML training pipeline with advanced features:
    - Ensemble learning with multiple model types
    - Automated hyperparameter tuning
    - Cross-validation for robust evaluation
    - Advanced feature engineering and selection
    - Uncertainty estimation for predictions
    
    Args:
        df (pd.DataFrame): Preprocessed benchmark data
        preprocessing_info (Dict[str, Any]): Preprocessing information
        test_size (float): Fraction of data to use for testing
        random_state (int): Random seed for reproducibility
        hyperparameter_tuning (bool): Whether to perform hyperparameter tuning
        use_ensemble (bool): Whether to use ensemble models
        model_complexity (str): Complexity level of models ('simple', 'standard', 'complex', or 'auto')
        use_cross_validation (bool): Whether to use cross-validation for evaluation
        cv_folds (int): Number of cross-validation folds
        feature_selection (str): Feature selection method ('none', 'k_best', 'model_based', or 'auto')
        uncertainty_estimation (bool): Whether to estimate prediction uncertainty
        n_jobs (int): Number of parallel jobs for training and tuning
        progress_callback (Callable, optional): Callback function for progress updates
        
    Returns:
        Dict[str, Any]: Trained models and evaluation metrics
    """
    if df.empty:
        logger.error("Empty dataframe provided for training")
        return {}
    
    try:
        start_time = time.time()
        logger.info(f"Starting model training with {len(df)} samples")
        
        # Unpack preprocessing info
        feature_cols = preprocessing_info["feature_columns"]
        numeric_cols = preprocessing_info["numeric_columns"]
        categorical_cols = preprocessing_info["categorical_columns"]
        target_cols = preprocessing_info["target_columns"]
        
        # Report on available data dimensions
        logger.info(f"Training data has {len(feature_cols)} features "
                    f"({len(numeric_cols)} numeric, {len(categorical_cols)} categorical)")
        
        # Initialize models dictionary
        models = {}
        
        # Auto-select model complexity based on data size
        if model_complexity == 'auto':
            if len(df) < 100:
                model_complexity = 'simple'
                logger.info("Auto-selected 'simple' model complexity due to small dataset size")
            elif len(df) < 500:
                model_complexity = 'standard'
                logger.info("Auto-selected 'standard' model complexity")
            else:
                model_complexity = 'complex'
                logger.info("Auto-selected 'complex' model complexity for large dataset")
        
        # Auto-adjust cross-validation settings based on data size
        if use_cross_validation:
            if len(df) < 100:
                # For small datasets, use fewer folds
                cv_folds = min(3, cv_folds)
                logger.info(f"Adjusted CV folds to {cv_folds} due to small dataset size")
            elif len(df) < 1000 and cv_folds > 5:
                # For medium datasets, cap at 5 folds for efficiency
                cv_folds = 5
                logger.info(f"Adjusted CV folds to {cv_folds} for efficiency")
        
        # ===== Create Preprocessing and Feature Engineering Pipeline =====
        
        # 1. Numerical features pipeline with advanced preprocessing
        if model_complexity == 'simple':
            # Simple preprocessing for small datasets
            numeric_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
        else:
            # More sophisticated preprocessing for larger datasets
            numeric_transformer = Pipeline([
                ('imputer', KNNImputer(n_neighbors=5)),
                ('scaler', RobustScaler()),
                ('power', PowerTransformer(method='yeo-johnson', standardize=True))
            ])
        
        # 2. Categorical features pipeline with advanced encoding
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # 3. Combined preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='drop'
        )
        
        # 4. Feature selection strategy
        if feature_selection == 'auto':
            # Choose strategy based on number of features
            if len(feature_cols) < 20:
                feature_selector = SelectKBest(f_regression, k='all')
                logger.info("Using all features (auto feature selection)")
            else:
                feature_selector = SelectKBest(f_regression, k=min(40, len(feature_cols) - 5))
                logger.info(f"Using top {min(40, len(feature_cols) - 5)} features (auto feature selection)")
        elif feature_selection == 'k_best':
            feature_selector = SelectKBest(f_regression, k=min(40, len(feature_cols) - 5))
        elif feature_selection == 'model_based':
            feature_selector = SelectFromModel(
                GradientBoostingRegressor(n_estimators=100, random_state=random_state),
                threshold='median'
            )
        else:  # 'none'
            feature_selector = SelectKBest(f_regression, k='all')
        
        # 5. Feature engineering setup (complexity dependent)
        if model_complexity == 'simple':
            # Simple polynomial features for small datasets
            complexity_transformer = ModelComplexityTransformer(
                degree=1, 
                interaction_only=True,
                feature_selection='none',
                include_hardware_model_interactions=True
            )
        elif model_complexity == 'standard':
            # Standard complexity with selected interactions
            complexity_transformer = ModelComplexityTransformer(
                degree=2, 
                interaction_only=True,
                feature_selection=feature_selection,
                n_features_to_select=min(30, len(feature_cols)),
                include_hardware_model_interactions=True
            )
        else:  # 'complex'
            # High complexity with more interactions and polynomial features
            complexity_transformer = ModelComplexityTransformer(
                degree=2, 
                interaction_only=False,
                feature_selection=feature_selection,
                n_features_to_select=min(50, len(feature_cols)),
                include_hardware_model_interactions=True
            )
        
        # 6. Complete feature engineering pipeline
        feature_engineering = Pipeline([
            ('preprocessor', preprocessor),
            ('selector', feature_selector),
            ('poly', complexity_transformer)
        ])
        
        # Setup cross-validation strategy if enabled
        if use_cross_validation:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            logger.info(f"Using {cv_folds}-fold cross-validation")
        else:
            cv = None
            logger.info("Cross-validation disabled")
            
        # ===== Process Each Target Metric =====
        
        # If progress callback provided, estimate total steps
        total_steps = len(target_cols) * (1 + hyperparameter_tuning)
        current_step = 0
        
        for target in target_cols:
            if progress_callback:
                current_step += 1
                progress_callback(current_step / total_steps, f"Processing {target}")
                
            if target not in df.columns:
                logger.warning(f"Target column {target} not found in data, skipping")
                continue
            
            logger.info(f"Training model for {target}...")
            
            # Prepare data
            X = df[feature_cols]
            y = df[target]
            
            # Skip if target has no variance
            if y.std() == 0:
                logger.warning(f"Target {target} has no variance, skipping")
                continue
            
            # Check for outlier information from preprocessing
            if f"{target}_outlier" in df.columns:
                # Use preprocessing outlier detection
                outlier_mask = ~df[f"{target}_outlier"]
                X_filtered = X[outlier_mask]
                y_filtered = y[outlier_mask]
                logger.info(f"Using preprocessing outlier detection: {len(X_filtered)}/{len(X)} samples")
            else:
                # Traditional outlier filtering if not already done in preprocessing
                if target == "throughput" or target == "latency_mean" or target == "memory_usage":
                    # Identify outliers using IQR method
                    Q1 = y.quantile(0.05)
                    Q3 = y.quantile(0.95)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Filter outliers
                    outlier_filter = (y >= lower_bound) & (y <= upper_bound)
                    X_filtered = X[outlier_filter]
                    y_filtered = y[outlier_filter]
                    
                    if len(X_filtered) < len(X) * 0.7:  # If removing too many points, revert to full dataset
                        X_filtered = X
                        y_filtered = y
                        logger.warning(f"Too many outliers detected for {target}, using full dataset.")
                    else:
                        logger.info(f"Filtered {len(X) - len(X_filtered)} outliers for {target}.")
                else:
                    X_filtered = X
                    y_filtered = y
            
            # Apply log transform for targets with large ranges (e.g., throughput)
            log_transform = False
            if target == "throughput" and y_filtered.max() / (y_filtered.min() + 1e-8) > 100:
                # Check if log transform would help (large range of values)
                log_transform = True
                logger.info(f"Applying log transform to {target} due to wide range")
            
            # Split data for final evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y_filtered, test_size=test_size, random_state=random_state
            )
            
            # ===== Create Model Based on Complexity and Target =====
            
            # Adjust model configuration based on target and complexity
            if use_ensemble:
                # Create specialized ensemble based on target and complexity
                if model_complexity == 'simple':
                    # Simpler ensemble models with fewer components
                    if target == "throughput":
                        base_model = VotingRegressor([
                            ('gbm', HistGradientBoostingRegressor(
                                random_state=random_state,
                                max_iter=100,
                                learning_rate=0.1
                            )),
                            ('rf', RandomForestRegressor(
                                random_state=random_state,
                                n_estimators=50,
                                max_depth=None
                            ))
                        ])
                    elif target == "latency_mean":
                        base_model = VotingRegressor([
                            ('gbm', HistGradientBoostingRegressor(
                                random_state=random_state,
                                max_iter=100,
                                learning_rate=0.1
                            )),
                            ('elastic', ElasticNet(
                                random_state=random_state,
                                alpha=0.1,
                                l1_ratio=0.5
                            ))
                        ])
                    else:  # memory_usage
                        base_model = VotingRegressor([
                            ('gbm', HistGradientBoostingRegressor(
                                random_state=random_state,
                                max_iter=100
                            )),
                            ('elastic', ElasticNet(
                                random_state=random_state,
                                alpha=0.1
                            ))
                        ])
                        
                elif model_complexity == 'standard':
                    # Standard complexity ensemble with balanced components
                    if target == "throughput":
                        base_model = VotingRegressor([
                            ('gbm', HistGradientBoostingRegressor(
                                random_state=random_state,
                                max_iter=150,
                                learning_rate=0.1,
                                max_depth=7
                            )),
                            ('rf', RandomForestRegressor(
                                random_state=random_state,
                                n_estimators=100,
                                max_depth=None
                            )),
                            ('mlp', MLPRegressor(
                                random_state=random_state,
                                hidden_layer_sizes=(100, 50),
                                early_stopping=True,
                                max_iter=500
                            ))
                        ])
                    elif target == "latency_mean":
                        base_model = VotingRegressor([
                            ('gbm', HistGradientBoostingRegressor(
                                random_state=random_state,
                                max_iter=150,
                                learning_rate=0.05,
                                max_depth=7
                            )),
                            ('rf', RandomForestRegressor(
                                random_state=random_state,
                                n_estimators=100,
                                min_samples_leaf=5
                            )),
                            ('elastic', ElasticNet(
                                random_state=random_state,
                                alpha=0.1,
                                l1_ratio=0.5
                            ))
                        ])
                    else:  # memory_usage
                        base_model = VotingRegressor([
                            ('gbm', HistGradientBoostingRegressor(
                                random_state=random_state,
                                max_iter=150,
                                learning_rate=0.1
                            )),
                            ('rf', RandomForestRegressor(
                                random_state=random_state,
                                n_estimators=100
                            )),
                            ('elastic', ElasticNet(
                                random_state=random_state,
                                alpha=0.01,
                                l1_ratio=0.2
                            ))
                        ])
                        
                else:  # complex
                    # Advanced ensemble with more sophisticated components
                    if target == "throughput":
                        # Use stacking regressor for more complex model
                        estimators = [
                            ('gbm', HistGradientBoostingRegressor(
                                random_state=random_state,
                                max_iter=200,
                                learning_rate=0.05,
                                max_depth=10
                            )),
                            ('rf', RandomForestRegressor(
                                random_state=random_state,
                                n_estimators=150,
                                max_depth=None,
                                min_samples_leaf=2
                            )),
                            ('mlp', MLPRegressor(
                                random_state=random_state,
                                hidden_layer_sizes=(200, 100, 50),
                                max_iter=1000,
                                early_stopping=True
                            )),
                            ('knn', KNeighborsRegressor(
                                n_neighbors=7,
                                weights='distance'
                            )),
                            ('svr', SVR(
                                kernel='rbf',
                                C=10.0,
                                epsilon=0.1
                            ))
                        ]
                        
                        # Use AdaBoost with GBM as meta-estimator
                        base_model = StackingRegressor(
                            estimators=estimators,
                            final_estimator=AdaBoostRegressor(
                                base_estimator=HistGradientBoostingRegressor(
                                    random_state=random_state,
                                    max_iter=100
                                ),
                                random_state=random_state,
                                n_estimators=10
                            ),
                            cv=3,
                            n_jobs=n_jobs
                        )
                        
                    elif target == "latency_mean":
                        # Custom stacking regressor for latency prediction
                        estimators = [
                            ('gbm', HistGradientBoostingRegressor(
                                random_state=random_state,
                                max_iter=200,
                                learning_rate=0.05,
                                max_depth=8
                            )),
                            ('rf', RandomForestRegressor(
                                random_state=random_state,
                                n_estimators=150,
                                min_samples_leaf=3
                            )),
                            ('huber', HuberRegressor(
                                epsilon=1.5,
                                alpha=0.01
                            )),
                            ('knn', KNeighborsRegressor(
                                n_neighbors=5,
                                weights='distance'
                            ))
                        ]
                        
                        base_model = StackingRegressor(
                            estimators=estimators,
                            final_estimator=ElasticNet(
                                random_state=random_state,
                                alpha=0.01,
                                l1_ratio=0.1
                            ),
                            cv=3,
                            n_jobs=n_jobs
                        )
                    else:  # memory_usage
                        # Memory usage typically has more linear relationships
                        estimators = [
                            ('gbm', HistGradientBoostingRegressor(
                                random_state=random_state,
                                max_iter=200,
                                learning_rate=0.05
                            )),
                            ('rf', RandomForestRegressor(
                                random_state=random_state,
                                n_estimators=100
                            )),
                            ('ridge', Ridge(
                                random_state=random_state,
                                alpha=0.5
                            )),
                            ('elastic', ElasticNet(
                                random_state=random_state,
                                alpha=0.01,
                                l1_ratio=0.2
                            ))
                        ]
                        
                        base_model = StackingRegressor(
                            estimators=estimators,
                            final_estimator=TheilSenRegressor(
                                random_state=random_state
                            ),
                            cv=3,
                            n_jobs=n_jobs
                        )
            else:
                # Use single model with appropriate complexity
                if model_complexity == 'simple':
                    base_model = HistGradientBoostingRegressor(
                        random_state=random_state,
                        max_iter=100,
                        learning_rate=0.1,
                        max_depth=5
                    )
                elif model_complexity == 'standard':
                    base_model = HistGradientBoostingRegressor(
                        random_state=random_state,
                        max_iter=200,
                        learning_rate=0.05,
                        max_depth=7
                    )
                else:  # complex
                    base_model = HistGradientBoostingRegressor(
                        random_state=random_state,
                        max_iter=300,
                        learning_rate=0.02,
                        max_depth=10,
                        l2_regularization=0.01
                    )
            
            # Apply log transform to target if needed
            if log_transform:
                final_model = TransformedTargetRegressor(
                    regressor=base_model,
                    func=np.log1p,  # log(1+x) to handle zeros
                    inverse_func=lambda x: np.expm1(x)  # exp(x)-1
                )
            else:
                final_model = base_model
            
            # Create full pipeline
            pipeline = Pipeline([
                ('features', feature_engineering),
                ('model', final_model)
            ])
            
            # ===== Hyperparameter Tuning =====
            
            if hyperparameter_tuning:
                if progress_callback:
                    current_step += 1
                    progress_callback(current_step / total_steps, f"Tuning hyperparameters for {target}")
                    
                logger.info(f"Performing hyperparameter tuning for {target}...")
                
                # Define parameter grid based on model complexity and type
                if model_complexity == 'simple':
                    # Simpler parameter grid for quicker tuning
                    if use_ensemble:
                        if isinstance(base_model, VotingRegressor):
                            if 'gbm' in base_model.named_estimators_:
                                param_grid = {
                                    'model__gbm__learning_rate': [0.05, 0.1],
                                    'model__gbm__max_depth': [5, None]
                                }
                            else:
                                param_grid = {}  # Fallback empty grid
                        else:
                            param_grid = {}  # Fallback for other ensemble types
                    else:
                        param_grid = {
                            'model__learning_rate': [0.05, 0.1],
                            'model__max_depth': [5, None],
                            'model__max_iter': [100, 200]
                        }
                elif model_complexity == 'standard':
                    # Standard parameter grid with moderate coverage
                    if use_ensemble:
                        if isinstance(base_model, VotingRegressor):
                            if 'gbm' in base_model.named_estimators_:
                                param_grid = {
                                    'features__selector__k': [10, 20, 'all'] if hasattr(feature_selector, 'k') else [],
                                    'model__gbm__learning_rate': [0.01, 0.05, 0.1],
                                    'model__gbm__max_depth': [5, 7, None]
                                }
                            else:
                                param_grid = {
                                    'features__selector__k': [10, 20, 'all'] if hasattr(feature_selector, 'k') else []
                                }
                        else:
                            param_grid = {
                                'features__selector__k': [10, 20, 'all'] if hasattr(feature_selector, 'k') else []
                            }
                    else:
                        param_grid = {
                            'features__selector__k': [10, 20, 'all'] if hasattr(feature_selector, 'k') else [],
                            'model__learning_rate': [0.01, 0.05, 0.1],
                            'model__max_depth': [5, 7, None],
                            'model__max_iter': [150, 200, 300],
                            'model__l2_regularization': [0.0, 0.01]
                        }
                else:  # complex
                    # More extensive parameter grid for complex models
                    if use_ensemble:
                        if isinstance(base_model, StackingRegressor):
                            # For stacking, we focus on tuning the final estimator and key base estimators
                            if isinstance(base_model.final_estimator_, AdaBoostRegressor):
                                param_grid = {
                                    'features__selector__k': [20, 30, 'all'] if hasattr(feature_selector, 'k') else [],
                                    'model__final_estimator__n_estimators': [5, 10, 20],
                                    'model__final_estimator__learning_rate': [0.5, 1.0, 1.5]
                                }
                            elif isinstance(base_model.final_estimator_, ElasticNet):
                                param_grid = {
                                    'features__selector__k': [20, 30, 'all'] if hasattr(feature_selector, 'k') else [],
                                    'model__final_estimator__alpha': [0.001, 0.01, 0.1],
                                    'model__final_estimator__l1_ratio': [0.1, 0.5, 0.9]
                                }
                            else:
                                param_grid = {
                                    'features__selector__k': [20, 30, 'all'] if hasattr(feature_selector, 'k') else []
                                }
                        else:
                            param_grid = {
                                'features__selector__k': [20, 30, 'all'] if hasattr(feature_selector, 'k') else []
                            }
                    else:
                        param_grid = {
                            'features__selector__k': [20, 30, 'all'] if hasattr(feature_selector, 'k') else [],
                            'model__learning_rate': [0.01, 0.02, 0.05],
                            'model__max_depth': [7, 10, 15, None],
                            'model__max_iter': [200, 300, 400],
                            'model__l2_regularization': [0.0, 0.01, 0.1],
                            'model__max_bins': [255]
                        }
                    
                # Adjust for log-transformed target
                if log_transform:
                    # Convert model__X params to model__regressor__X
                    new_param_grid = {}
                    for param, values in param_grid.items():
                        if param.startswith('model__') and not param.startswith('model__regressor__'):
                            # Replace model__ with model__regressor__
                            new_param = param.replace('model__', 'model__regressor__')
                            new_param_grid[new_param] = values
                        else:
                            new_param_grid[param] = values
                    param_grid = new_param_grid
                
                # Choose search method based on grid size
                if model_complexity == 'complex' or len(param_grid) > 5:
                    # Use randomized search for large parameter spaces
                    logger.info("Using RandomizedSearchCV due to large parameter space")
                    n_iter = min(20, 3 ** len(param_grid))  # Adjust based on grid size
                    search = RandomizedSearchCV(
                        pipeline,
                        param_distributions=param_grid,
                        n_iter=n_iter,
                        cv=cv if cv else min(5, len(X_train) // 10) if len(X_train) >= 40 else 3,
                        scoring='neg_root_mean_squared_error',
                        n_jobs=n_jobs,
                        random_state=random_state,
                        verbose=1 if logger.level <= logging.INFO else 0
                    )
                else:
                    # Use grid search for smaller parameter spaces
                    logger.info("Using GridSearchCV for parameter tuning")
                    search = GridSearchCV(
                        pipeline,
                        param_grid=param_grid,
                        cv=cv if cv else min(5, len(X_train) // 10) if len(X_train) >= 40 else 3,
                        scoring='neg_root_mean_squared_error',
                        n_jobs=n_jobs,
                        verbose=1 if logger.level <= logging.INFO else 0
                    )
                
                # Fit the search if we have parameters to tune
                if param_grid:
                    try:
                        search.fit(X_train, y_train)
                        pipeline = search.best_estimator_
                        logger.info(f"Best parameters for {target}: {search.best_params_}")
                    except Exception as e:
                        logger.warning(f"Hyperparameter tuning failed for {target}: {e}. Using default model.")
                        # Fit with default parameters
                        pipeline.fit(X_train, y_train)
                else:
                    logger.info(f"No parameters to tune for {target}, using default configuration")
                    pipeline.fit(X_train, y_train)
            else:
                # Train model without hyperparameter tuning
                logger.info(f"Training model for {target} without hyperparameter tuning")
                pipeline.fit(X_train, y_train)
            
            # ===== Model Evaluation =====
            
            # Calculate core evaluation metrics
            train_r2 = pipeline.score(X_train, y_train)
            test_r2 = pipeline.score(X_test, y_test)
            
            # Make predictions on test set
            y_pred = pipeline.predict(X_test)
            
            # Calculate various metrics for comprehensive evaluation
            metrics = {}
            
            # Base metrics
            metrics["train_r2"] = float(train_r2)
            metrics["test_r2"] = float(test_r2)
            
            # Only calculate MAPE for positive values
            if (y_test > 0).any():
                mape = mean_absolute_percentage_error(y_test[y_test > 0], y_pred[y_test > 0])
                metrics["mape"] = float(mape)
            else:
                metrics["mape"] = float('inf')
                
            # Error metrics
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            metrics["mae"] = float(mean_absolute_error(y_test, y_pred))
            metrics["median_ae"] = float(median_absolute_error(y_test, y_pred))
            metrics["explained_var"] = float(explained_variance_score(y_test, y_pred))
            
            # Data statistics
            metrics["n_samples"] = len(df)
            metrics["n_train_samples"] = len(X_train)
            metrics["n_test_samples"] = len(X_test)
            metrics["target_mean"] = float(np.mean(y_test))
            metrics["target_std"] = float(np.std(y_test))
            metrics["target_min"] = float(np.min(y_test))
            metrics["target_max"] = float(np.max(y_test))
            metrics["target_median"] = float(np.median(y_test))
            
            # Model complexity metrics
            metrics["hyperparameter_tuned"] = hyperparameter_tuning
            metrics["ensemble_model"] = use_ensemble
            metrics["model_complexity"] = model_complexity
            metrics["log_transform_applied"] = log_transform
            metrics["training_timestamp"] = datetime.now().isoformat()
            
            # ===== Uncertainty Estimation =====
            
            confidence_intervals = {}
            
            if uncertainty_estimation:
                logger.info(f"Calculating uncertainty estimates for {target}")
                
                # Method 1: Random Forest prediction intervals if available
                rf_model = None
                
                # Extract Random Forest model if it exists
                if isinstance(pipeline.named_steps['model'], RandomForestRegressor):
                    rf_model = pipeline.named_steps['model']
                elif isinstance(pipeline.named_steps['model'], VotingRegressor) and 'rf' in pipeline.named_steps['model'].named_estimators_:
                    rf_model = pipeline.named_steps['model'].named_estimators_['rf']
                elif isinstance(pipeline.named_steps['model'], TransformedTargetRegressor) and isinstance(pipeline.named_steps['model'].regressor, RandomForestRegressor):
                    rf_model = pipeline.named_steps['model'].regressor
                elif isinstance(pipeline.named_steps['model'], TransformedTargetRegressor) and isinstance(pipeline.named_steps['model'].regressor, VotingRegressor) and 'rf' in pipeline.named_steps['model'].regressor.named_estimators_:
                    rf_model = pipeline.named_steps['model'].regressor.named_estimators_['rf']
                
                if rf_model is not None:
                    try:
                        # Get feature matrix
                        X_test_transformed = pipeline.named_steps['features'].transform(X_test)
                        
                        # Get all tree predictions
                        tree_preds = np.array([tree.predict(X_test_transformed) for tree in rf_model.estimators_])
                        
                        # Calculate prediction intervals
                        pred_std = np.std(tree_preds, axis=0)
                        
                        confidence_intervals["rf_mean_std"] = float(np.mean(pred_std))
                        confidence_intervals["rf_min_std"] = float(np.min(pred_std))
                        confidence_intervals["rf_max_std"] = float(np.max(pred_std))
                        confidence_intervals["rf_rel_uncertainty"] = float(np.mean(pred_std) / np.mean(y_test) if np.mean(y_test) > 0 else 1.0)
                        confidence_intervals["rf_prediction_intervals"] = {
                            "lower_bound": list(y_pred - 1.96 * pred_std),
                            "upper_bound": list(y_pred + 1.96 * pred_std)
                        }
                    except Exception as e:
                        logger.warning(f"Failed to calculate RF prediction intervals: {e}")
                
                # Method 2: Bootstrap uncertainty estimation for any model
                try:
                    # Create bootstrap samples and calculate prediction variance
                    n_bootstrap = 30
                    bootstrap_predictions = []
                    
                    with ThreadPoolExecutor(max_workers=min(n_jobs if n_jobs > 0 else os.cpu_count(), 8)) as executor:
                        futures = []
                        
                        for i in range(n_bootstrap):
                            # Create bootstrap sample
                            indices = np.random.choice(len(X_train), len(X_train), replace=True)
                            X_boot = X_train.iloc[indices].copy()
                            y_boot = y_train.iloc[indices].copy()
                            
                            # Clone the pipeline for parallel training
                            boot_pipeline = clone(pipeline)
                            
                            # Submit training task
                            future = executor.submit(
                                lambda p, X, y, X_test: p.fit(X, y).predict(X_test),
                                boot_pipeline, X_boot, y_boot, X_test
                            )
                            futures.append(future)
                        
                        # Collect results
                        for future in as_completed(futures):
                            try:
                                bootstrap_predictions.append(future.result())
                            except Exception as e:
                                logger.warning(f"Bootstrap sample failed: {e}")
                    
                    # Convert to array for calculations
                    bootstrap_predictions = np.array(bootstrap_predictions)
                    
                    # Calculate statistics
                    bootstrap_mean = np.mean(bootstrap_predictions, axis=0)
                    bootstrap_std = np.std(bootstrap_predictions, axis=0)
                    
                    # Store in confidence intervals
                    confidence_intervals["bootstrap_mean_std"] = float(np.mean(bootstrap_std))
                    confidence_intervals["bootstrap_min_std"] = float(np.min(bootstrap_std))
                    confidence_intervals["bootstrap_max_std"] = float(np.max(bootstrap_std))
                    confidence_intervals["bootstrap_rel_uncertainty"] = float(np.mean(bootstrap_std) / np.mean(y_test) if np.mean(y_test) > 0 else 1.0)
                    
                    # Calculate prediction intervals (assuming normal distribution, 95% CI)
                    lower_bounds = bootstrap_mean - 1.96 * bootstrap_std
                    upper_bounds = bootstrap_mean + 1.96 * bootstrap_std
                    
                    confidence_intervals["bootstrap_prediction_intervals"] = {
                        "lower_bound": lower_bounds.tolist(),
                        "upper_bound": upper_bounds.tolist()
                    }
                    
                    # Estimate overall uncertainty as mean of bootstrap and RF (if available)
                    if "rf_mean_std" in confidence_intervals:
                        confidence_intervals["combined_uncertainty"] = (
                            confidence_intervals["bootstrap_rel_uncertainty"] + 
                            confidence_intervals["rf_rel_uncertainty"]
                        ) / 2.0
                    else:
                        confidence_intervals["combined_uncertainty"] = confidence_intervals["bootstrap_rel_uncertainty"]
                    
                except Exception as e:
                    logger.warning(f"Failed to calculate bootstrap uncertainty: {e}")
                    # Fallback uncertainty estimation based on error metrics
                    confidence_intervals["fallback_uncertainty"] = float(metrics["rmse"] / metrics["target_mean"] if metrics["target_mean"] > 0 else 0.3)
            
            # If no uncertainty estimation was performed or it failed, add fallback
            if not confidence_intervals:
                confidence_intervals["fallback_uncertainty"] = float(metrics["rmse"] / metrics["target_mean"] if metrics["target_mean"] > 0 else 0.3)
            
            # Add confidence information to metrics
            metrics["confidence_intervals"] = confidence_intervals
            
            # ===== Feature Importance Analysis =====
            
            # Extract feature importance when available
            feature_importance = {}
            
            # Check if model has direct feature_importances_ attribute
            if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
                importances = pipeline.named_steps['model'].feature_importances_
                try:
                    # Get feature names after preprocessing
                    if hasattr(pipeline.named_steps['features'], 'get_feature_names_out'):
                        feature_names = pipeline.named_steps['features'].get_feature_names_out()
                    else:
                        feature_names = [f"feature_{i}" for i in range(len(importances))]
                    
                    feature_importance["direct"] = dict(zip(feature_names, importances.tolist() if hasattr(importances, 'tolist') else importances))
                except Exception as e:
                    logger.warning(f"Failed to get feature names for importance: {e}")
                    feature_importance["direct"] = dict(zip([f"feature_{i}" for i in range(len(importances))], importances.tolist() if hasattr(importances, 'tolist') else importances))
            
            # For TransformedTargetRegressor, check the underlying regressor
            elif isinstance(pipeline.named_steps['model'], TransformedTargetRegressor) and hasattr(pipeline.named_steps['model'].regressor, 'feature_importances_'):
                importances = pipeline.named_steps['model'].regressor.feature_importances_
                try:
                    if hasattr(pipeline.named_steps['features'], 'get_feature_names_out'):
                        feature_names = pipeline.named_steps['features'].get_feature_names_out()
                    else:
                        feature_names = [f"feature_{i}" for i in range(len(importances))]
                    
                    feature_importance["regressor"] = dict(zip(feature_names, importances.tolist() if hasattr(importances, 'tolist') else importances))
                except Exception as e:
                    logger.warning(f"Failed to get feature names for importance: {e}")
                    feature_importance["regressor"] = dict(zip([f"feature_{i}" for i in range(len(importances))], importances.tolist() if hasattr(importances, 'tolist') else importances))
            
            # For VotingRegressor, get importance from component models
            elif isinstance(pipeline.named_steps['model'], VotingRegressor):
                for name, estimator in pipeline.named_steps['model'].named_estimators_.items():
                    if hasattr(estimator, 'feature_importances_'):
                        importances = estimator.feature_importances_
                        try:
                            if hasattr(pipeline.named_steps['features'], 'get_feature_names_out'):
                                feature_names = pipeline.named_steps['features'].get_feature_names_out()
                            else:
                                feature_names = [f"feature_{i}" for i in range(len(importances))]
                            
                            feature_importance[f"{name}"] = dict(zip(feature_names, importances.tolist() if hasattr(importances, 'tolist') else importances))
                        except Exception as e:
                            logger.warning(f"Failed to get feature names for {name}: {e}")
                            feature_importance[f"{name}"] = dict(zip([f"feature_{i}" for i in range(len(importances))], importances.tolist() if hasattr(importances, 'tolist') else importances))
            
            # For StackingRegressor, get importance from first-level estimators
            elif isinstance(pipeline.named_steps['model'], StackingRegressor):
                for name, estimator in pipeline.named_steps['model'].estimators_:
                    if hasattr(estimator, 'feature_importances_'):
                        importances = estimator.feature_importances_
                        try:
                            if hasattr(pipeline.named_steps['features'], 'get_feature_names_out'):
                                feature_names = pipeline.named_steps['features'].get_feature_names_out()
                            else:
                                feature_names = [f"feature_{i}" for i in range(len(importances))]
                            
                            feature_importance[f"{name}"] = dict(zip(feature_names, importances.tolist() if hasattr(importances, 'tolist') else importances))
                        except Exception as e:
                            logger.warning(f"Failed to get feature names for {name}: {e}")
                            feature_importance[f"{name}"] = dict(zip([f"feature_{i}" for i in range(len(importances))], importances.tolist() if hasattr(importances, 'tolist') else importances))
            
            # For TransformedTargetRegressor with VotingRegressor, get from components
            elif isinstance(pipeline.named_steps['model'], TransformedTargetRegressor) and isinstance(pipeline.named_steps['model'].regressor, VotingRegressor):
                voting_regressor = pipeline.named_steps['model'].regressor
                for name, estimator in voting_regressor.named_estimators_.items():
                    if hasattr(estimator, 'feature_importances_'):
                        importances = estimator.feature_importances_
                        try:
                            if hasattr(pipeline.named_steps['features'], 'get_feature_names_out'):
                                feature_names = pipeline.named_steps['features'].get_feature_names_out()
                            else:
                                feature_names = [f"feature_{i}" for i in range(len(importances))]
                            
                            feature_importance[f"{name}"] = dict(zip(feature_names, importances.tolist() if hasattr(importances, 'tolist') else importances))
                        except Exception as e:
                            logger.warning(f"Failed to get feature names for {name}: {e}")
                            feature_importance[f"{name}"] = dict(zip([f"feature_{i}" for i in range(len(importances))], importances.tolist() if hasattr(importances, 'tolist') else importances))
            
            # If no direct feature importance is available, use permutation importance
            if not feature_importance and len(X_test) > 10:
                try:
                    from sklearn.inspection import permutation_importance
                    
                    # Calculate permutation importance on test set (faster, avoids overfitting)
                    perm_importance = permutation_importance(
                        pipeline, X_test, y_test, 
                        n_repeats=10, 
                        random_state=random_state,
                        n_jobs=min(n_jobs if n_jobs > 0 else 1, 4)  # Limit jobs for permutation
                    )
                    
                    # Store mean importance scores
                    importances = perm_importance.importances_mean
                    
                    # Get feature names if possible
                    if hasattr(X_test, 'columns'):
                        feature_names = X_test.columns.tolist()
                    else:
                        feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
                    
                    feature_importance["permutation"] = dict(zip(feature_names, importances.tolist() if hasattr(importances, 'tolist') else importances))
                    
                    # Add standard deviations for more complete information
                    importance_std = perm_importance.importances_std
                    feature_importance["permutation_std"] = dict(zip(feature_names, importance_std.tolist() if hasattr(importance_std, 'tolist') else importance_std))
                
                except Exception as e:
                    logger.warning(f"Failed to calculate permutation importance: {e}")
            
            # Add feature importance to metrics
            metrics["feature_importance"] = feature_importance
            
            # ===== Add Model Parameters and Hyperparameter Tuning Results =====
            
            # Add hyperparameter tuning results if available
            if hyperparameter_tuning and 'search' in locals() and hasattr(search, 'best_params_'):
                metrics["best_params"] = search.best_params_
                metrics["cv_results_summary"] = {
                    "best_score": float(search.best_score_) if hasattr(search, 'best_score_') else None,
                    "mean_fit_time": float(np.mean(search.cv_results_['mean_fit_time'])) if 'mean_fit_time' in search.cv_results_ else None,
                    "n_evaluated_params": len(search.cv_results_['params']) if 'params' in search.cv_results_ else 0
                }
                
                # Add more detailed CV results summary
                if 'cv_results_' in dir(search):
                    # Extract key metrics from CV results
                    best_idx = search.best_index_ if hasattr(search, 'best_index_') else 0
                    metrics["cv_results_summary"]["detailed"] = {
                        "best_params": search.best_params_,
                        "best_score_mean": float(search.cv_results_['mean_test_score'][best_idx]) if 'mean_test_score' in search.cv_results_ else None,
                        "best_score_std": float(search.cv_results_['std_test_score'][best_idx]) if 'std_test_score' in search.cv_results_ else None,
                        "best_params_fit_time": float(search.cv_results_['mean_fit_time'][best_idx]) if 'mean_fit_time' in search.cv_results_ else None,
                        "best_params_score_time": float(search.cv_results_['mean_score_time'][best_idx]) if 'mean_score_time' in search.cv_results_ else None
                    }
            
            # Add cross-validation results if performed
            if use_cross_validation:
                try:
                    # Perform cross-validation on the final pipeline
                    cv_scores = cross_val_score(pipeline, X_filtered, y_filtered, cv=cv_folds, scoring='r2')
                    metrics["cv_r2_mean"] = float(np.mean(cv_scores))
                    metrics["cv_r2_std"] = float(np.std(cv_scores))
                    metrics["cv_r2_min"] = float(np.min(cv_scores))
                    metrics["cv_r2_max"] = float(np.max(cv_scores))
                    
                    # Add more detailed cross-validation metrics if time permits
                    try:
                        from sklearn.model_selection import cross_validate
                        
                        cv_results = cross_validate(
                            pipeline, X_filtered, y_filtered, 
                            cv=cv_folds, 
                            scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'],
                            return_train_score=True
                        )
                        
                        metrics["cv_detailed"] = {
                            "test_r2_mean": float(np.mean(cv_results['test_r2'])),
                            "test_r2_std": float(np.std(cv_results['test_r2'])),
                            "test_rmse_mean": float(np.mean(np.sqrt(-cv_results['test_neg_mean_squared_error']))),
                            "test_rmse_std": float(np.std(np.sqrt(-cv_results['test_neg_mean_squared_error']))),
                            "test_mae_mean": float(np.mean(-cv_results['test_neg_mean_absolute_error'])),
                            "test_mae_std": float(np.std(-cv_results['test_neg_mean_absolute_error'])),
                            "train_r2_mean": float(np.mean(cv_results['train_r2'])) if 'train_r2' in cv_results else None,
                            "train_r2_std": float(np.std(cv_results['train_r2'])) if 'train_r2' in cv_results else None,
                            "fit_time_mean": float(np.mean(cv_results['fit_time'])),
                            "score_time_mean": float(np.mean(cv_results['score_time']))
                        }
                    except Exception as e:
                        logger.warning(f"Failed to compute detailed CV metrics: {e}")
                except Exception as e:
                    logger.warning(f"Failed to compute cross-validation scores: {e}")
            
            # ===== Store Model and Metrics =====
            
            # Store model and metrics
            models[target] = {
                "pipeline": pipeline,
                "metrics": metrics
            }
            
            # Log completion and metrics
            logger.info(f"Model for {target}: R² = {test_r2:.4f}, MAPE = {metrics.get('mape', 'N/A'):.2%}, RMSE = {metrics['rmse']:.4f}")
        
        # ===== Complete Model Dictionary =====
        
        # Store preprocessing info
        models["preprocessing_info"] = preprocessing_info
        
        # Add model metadata
        models["metadata"] = {
            "version": "3.0.0",
            "description": "Advanced prediction models with ensemble learning, uncertainty estimation, and comprehensive feature engineering",
            "created_at": datetime.now().isoformat(),
            "training_time_seconds": time.time() - start_time,
            "hyperparameter_tuning": hyperparameter_tuning,
            "use_ensemble": use_ensemble,
            "model_complexity": model_complexity,
            "feature_engineering": True,
            "uncertainty_estimation": uncertainty_estimation,
            "use_cross_validation": use_cross_validation,
            "cv_folds": cv_folds if use_cross_validation else None,
            "n_models": len(models) - 1  # Subtract 1 for preprocessing_info
        }
        
        logger.info(f"Completed training {len(models) - 1} prediction models in {time.time() - start_time:.2f} seconds")
        return models
    
    except Exception as e:
        logger.error(f"Error training prediction models: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}

def save_prediction_models(models: Dict[str, Any], output_dir: Optional[str] = None) -> str:
    """
    Save trained prediction models to disk.
    
    Args:
        models (Dict[str, Any]): Trained models and evaluation metrics
        output_dir (str): Directory to save models
        
    Returns:
        str: Path to saved models
    """
    if not models:
        logger.error("No models provided for saving")
        return ""
    
    try:
        if output_dir is None:
            output_dir = MODEL_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamp for model version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(output_dir, f"models_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save preprocessing info
        preproc_info = models.get("preprocessing_info", {})
        with open(os.path.join(model_dir, "preprocessing_info.json"), "w") as f:
            json.dump(preproc_info, f, indent=2)
        
        # Save metrics
        metrics = {
            target: model_info["metrics"] 
            for target, model_info in models.items() 
            if target != "preprocessing_info"
        }
        with open(os.path.join(model_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Save pipelines
        for target, model_info in models.items():
            if target == "preprocessing_info":
                continue
                
            pipeline = model_info["pipeline"]
            pipeline_path = os.path.join(model_dir, f"{target}_pipeline.joblib")
            joblib.dump(pipeline, pipeline_path)
        
        # Create latest symlink
        latest_path = os.path.join(output_dir, "latest")
        if os.path.exists(latest_path) or os.path.islink(latest_path):
            os.remove(latest_path)
        
        # Create relative symlink
        os.symlink(os.path.basename(model_dir), latest_path)
        
        logger.info(f"Saved models to {model_dir}")
        return model_dir
    
    except Exception as e:
        logger.error(f"Error saving prediction models: {e}")
        return ""

def load_prediction_models(model_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Load trained prediction models from disk.
    
    Args:
        model_dir (str): Directory containing models
        
    Returns:
        Dict[str, Any]: Loaded models and preprocessing info
    """
    try:
        if model_dir is None:
            model_dir = os.path.join(MODEL_DIR, "latest")
        
        if not os.path.exists(model_dir):
            logger.error(f"Model directory {model_dir} not found")
            return {}
        
        # Load preprocessing info
        preproc_path = os.path.join(model_dir, "preprocessing_info.json")
        if not os.path.exists(preproc_path):
            logger.error(f"Preprocessing info not found at {preproc_path}")
            return {}
            
        with open(preproc_path, "r") as f:
            preprocessing_info = json.load(f)
        
        # Initialize models dictionary
        models = {"preprocessing_info": preprocessing_info}
        
        # Load pipelines
        for target in PREDICTION_METRICS:
            pipeline_path = os.path.join(model_dir, f"{target}_pipeline.joblib")
            if not os.path.exists(pipeline_path):
                logger.warning(f"Pipeline for {target} not found at {pipeline_path}")
                continue
                
            pipeline = joblib.load(pipeline_path)
            
            # Load metrics
            metrics_path = os.path.join(model_dir, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    all_metrics = json.load(f)
                    metrics = all_metrics.get(target, {})
            else:
                metrics = {}
            
            models[target] = {
                "pipeline": pipeline,
                "metrics": metrics
            }
        
        logger.info(f"Loaded {len(models) - 1} models from {model_dir}")
        return models
    
    except Exception as e:
        logger.error(f"Error loading prediction models: {e}")
        return {}

def predict_performance(
    models: Dict[str, Any],
    model_name: str,
    model_category: str,
    hardware: str,
    batch_size: int,
    precision: str = "fp32",
    mode: str = "inference",
    gpu_count: int = 1,
    is_distributed: bool = False,
    sequence_length: int = 128,
    calculate_uncertainty: bool = True
) -> Dict[str, Any]:
    """
    Predict performance for a model-hardware configuration with uncertainty estimates.
    
    Args:
        models (Dict[str, Any]): Trained prediction models
        model_name (str): Name of the model
        model_category (str): Category of the model
        hardware (str): Hardware platform
        batch_size (int): Batch size
        precision (str): Precision (fp32, fp16, int8)
        mode (str): Mode (inference, training)
        gpu_count (int): Number of GPUs (for distributed training)
        is_distributed (bool): Whether this is a distributed configuration
        sequence_length (int): Sequence length for text models
        calculate_uncertainty (bool): Whether to calculate uncertainty estimates
        
    Returns:
        Dict[str, Any]: Predicted performance metrics with uncertainty estimates
    """
    if not models:
        logger.error("No models provided for prediction")
        return {}
    
    try:
        # Get preprocessing info
        preprocessing_info = models.get("preprocessing_info", {})
        if not preprocessing_info:
            logger.error("No preprocessing info found in models")
            return {}
        
        # Create input features
        precision_numeric = 32 if precision == "fp32" else 16 if precision == "fp16" else 8 if precision == "int8" else 32
        hardware_platform = hardware.split("_")[0] if "_" in hardware else hardware
        
        # Get model size estimate
        model_size_estimate = _estimate_model_size(model_name)
        model_size_log = np.log10(model_size_estimate + 1)
        
        # Create model size category
        size_categories = ["tiny", "small", "medium", "large"]
        if model_size_estimate < 10_000_000:
            model_size_category = "tiny"
        elif model_size_estimate < 100_000_000:
            model_size_category = "small"
        elif model_size_estimate < 1_000_000_000:
            model_size_category = "medium"
        else:
            model_size_category = "large"
        
        # Create interaction features
        batch_x_precision = batch_size * (32 / precision_numeric)
        batch_x_gpus = batch_size * gpu_count
        
        # Create base input dataframe
        input_data = pd.DataFrame({
            "batch_size": [batch_size],
            "precision_numeric": [precision_numeric],
            "gpu_count": [gpu_count],
            "is_distributed": [is_distributed],
            "mode": [mode],
            "category": [model_category],
            "hardware_platform": [hardware_platform],
            "model_size_log": [model_size_log],
            "model_size_category": [model_size_category],
            "sequence_length": [sequence_length],
            "batch_x_precision": [batch_x_precision],
            "batch_x_gpus": [batch_x_gpus],
            "model_name": [model_name]
        })
        
        # Apply hardware feature transformer if it exists in preprocessing info
        hw_caps = HARDWARE_CAPABILITIES.get(hardware_platform, {})
        model_chars = MODEL_FAMILY_CHARACTERISTICS.get(model_category, {})
        
        # Add derived features if hardware and model info available
        if hw_caps and model_chars:
            # Add derived features directly
            feature_transformer = HardwareModelFeatureTransformer()
            input_data = feature_transformer.transform(input_data)
        
        # Make predictions for each metric
        predictions = {}
        uncertainties = {}
        
        # Get metadata from models
        metadata = models.get("metadata", {})
        model_version = metadata.get("version", "1.0.0")
        
        # Calculate overall confidence score based on how similar this config is to training data
        confidence_score = 1.0  # Default high confidence
        
        # Lower confidence if using advanced features not present in the input data
        required_features = set()
        for target in PREDICTION_METRICS:
            if target in models:
                target_model = models[target]
                # Check if the model has metrics
                if "metrics" in target_model:
                    # Check if the model has feature importance
                    if "feature_importance" in target_model["metrics"]:
                        feature_importance = target_model["metrics"]["feature_importance"]
                        # Add important features to the set
                        if isinstance(feature_importance, dict):
                            for feature in feature_importance:
                                if isinstance(feature, str) and not feature.startswith('feature_'):
                                    required_features.add(feature)
        
        # Calculate missing features ratio
        if required_features:
            missing_features = [f for f in required_features if f not in input_data.columns]
            missing_ratio = len(missing_features) / len(required_features)
            # Adjust confidence score based on missing features
            confidence_score *= (1.0 - 0.3 * missing_ratio)  # Reduce confidence by up to 30%
        
        # Make predictions for each target metric
        for target in PREDICTION_METRICS:
            if target not in models:
                logger.warning(f"No model found for {target}, skipping prediction")
                continue
                
            pipeline = models[target]["pipeline"]
            
            # Make prediction
            try:
                # Get model metrics for confidence calculation
                model_metrics = models[target].get("metrics", {})
                
                # Make the prediction
                pred_value = float(pipeline.predict(input_data)[0])
                
                # Ensure predictions make sense with domain constraints
                if target == "throughput" and pred_value < 0:
                    pred_value = 0
                elif target == "latency_mean" and pred_value < 0:
                    pred_value = 1
                elif target == "memory_usage" and pred_value < 0:
                    pred_value = 1
                
                # Store prediction
                predictions[target] = pred_value
                
                # Calculate uncertainty if requested and possible
                uncertainty = None
                if calculate_uncertainty:
                    # Method 1: Use confidence intervals from model metrics if available
                    if "confidence_intervals" in model_metrics:
                        confidence_intervals = model_metrics["confidence_intervals"]
                        if "mean_pred_std" in confidence_intervals:
                            # Scale uncertainty based on prediction size
                            relative_scale = pred_value / model_metrics.get("target_mean", 1.0)
                            uncertainty = confidence_intervals["mean_pred_std"] * max(1.0, relative_scale)
                    
                    # Method 2: Use test error metrics to estimate uncertainty
                    if uncertainty is None and "rmse" in model_metrics:
                        # Base uncertainty on RMSE and relative prediction size
                        relative_scale = pred_value / model_metrics.get("target_mean", 1.0)
                        uncertainty = model_metrics["rmse"] * max(0.5, relative_scale)
                    
                    # Method 3: Use fixed percentage if nothing else available
                    if uncertainty is None:
                        # Fallback constant uncertainty percentage based on target
                        if target == "throughput":
                            uncertainty = pred_value * 0.15  # 15% uncertainty
                        elif target == "latency_mean":
                            uncertainty = pred_value * 0.20  # 20% uncertainty
                        elif target == "memory_usage":
                            uncertainty = pred_value * 0.10  # 10% uncertainty
                
                # Store uncertainty
                if uncertainty is not None:
                    uncertainties[target] = float(uncertainty)
                
                # Calculate metric-specific confidence
                metric_confidence = 1.0
                
                # Adjust confidence based on model R-squared
                if "test_r2" in model_metrics:
                    metric_confidence *= max(0.5, min(1.0, model_metrics["test_r2"]))
                
                # Adjust confidence based on whether model was hyperparameter tuned
                if model_metrics.get("hyperparameter_tuned", False):
                    metric_confidence *= 1.05  # 5% bonus for hyperparameter tuned models
                
                # Adjust confidence based on number of training samples
                n_train_samples = model_metrics.get("n_train_samples", 100)
                sample_factor = min(1.0, n_train_samples / 500)  # Cap at 500 samples
                metric_confidence *= 0.7 + (0.3 * sample_factor)  # Scale between 0.7-1.0
                
                # Adjust confidence for extreme predictions
                if "target_mean" in model_metrics and "target_std" in model_metrics:
                    z_score = abs(pred_value - model_metrics["target_mean"]) / max(0.1, model_metrics["target_std"])
                    if z_score > 2.0:  # More than 2 standard deviations from mean
                        outlier_penalty = min(0.5, 0.1 * (z_score - 2.0))  # Cap at 50% reduction
                        metric_confidence *= (1.0 - outlier_penalty)
                
                # Store metric confidence
                predictions[f"{target}_confidence"] = float(min(1.0, max(0.1, metric_confidence)))
                
            except Exception as e:
                logger.error(f"Error predicting {target}: {e}")
                predictions[target] = None
        
        # Calculate overall prediction uncertainty
        prediction_uncertainty = {}
        for target, value in predictions.items():
            if target in PREDICTION_METRICS and value is not None:
                # Get uncertainty from direct calculation or estimate
                uncertainty_value = uncertainties.get(target, value * 0.15)  # Default to 15% if not calculated
                
                # Create prediction interval
                prediction_uncertainty[target] = {
                    "value": float(value),
                    "uncertainty": float(uncertainty_value),
                    "confidence": float(predictions.get(f"{target}_confidence", 0.7)),
                    "lower_bound": float(max(0, value - uncertainty_value)),
                    "upper_bound": float(value + uncertainty_value)
                }
        
        # Add metadata
        result = {
            "model": model_name,
            "model_size_estimate": model_size_estimate,
            "model_category": model_category,
            "hardware": hardware,
            "batch_size": batch_size,
            "precision": precision,
            "mode": mode,
            "gpu_count": gpu_count,
            "is_distributed": is_distributed,
            "sequence_length": sequence_length,
            "predictions": {k: v for k, v in predictions.items() if k in PREDICTION_METRICS},
            "uncertainties": prediction_uncertainty,
            "confidence_score": float(min(1.0, max(0.1, confidence_score))),
            "is_predicted": True,
            "prediction_model_version": model_version,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add explanation for any anomalous predictions
        explanation = []
        for target in PREDICTION_METRICS:
            if target in predictions and predictions[target] is not None:
                target_conf = predictions.get(f"{target}_confidence", 1.0)
                if target_conf < 0.5:
                    explanation.append(f"Low confidence in {target} prediction due to limited training data or extrapolation.")
        
        if explanation:
            result["explanation"] = explanation
        
        return result
    
    except Exception as e:
        logger.error(f"Error predicting performance: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}

def generate_prediction_matrix(
    models: Dict[str, Any],
    model_configs: Optional[List[Dict[str, Any]]] = None,
    hardware_platforms: Optional[List[str]] = None,
    batch_sizes: Optional[List[int]] = None,
    precision_options: Optional[List[str]] = None,
    mode: str = "inference",
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a prediction matrix for various model-hardware configurations.
    
    Args:
        models (Dict[str, Any]): Trained prediction models
        model_configs (List[Dict[str, Any]]): List of model configurations
        hardware_platforms (List[str]): List of hardware platforms
        batch_sizes (List[int]): List of batch sizes
        precision_options (List[str]): List of precision options
        mode (str): Mode (inference, training)
        output_file (str): Path to output file
        
    Returns:
        Dict[str, Any]: Prediction matrix
    """
    if not models:
        logger.error("No models provided for prediction matrix")
        return {}
    
    try:
        # Set default values
        if model_configs is None:
            model_configs = [
                {"name": "bert-base-uncased", "category": "text_embedding"},
                {"name": "t5-small", "category": "text_generation"},
                {"name": "facebook/opt-125m", "category": "text_generation"},
                {"name": "openai/whisper-tiny", "category": "audio"},
                {"name": "google/vit-base-patch16-224", "category": "vision"},
                {"name": "openai/clip-vit-base-patch32", "category": "multimodal"}
            ]
        
        if hardware_platforms is None:
            hardware_platforms = ["cpu", "cuda", "mps", "openvino"]
        
        if batch_sizes is None:
            batch_sizes = [1, 8, 32]
        
        if precision_options is None:
            precision_options = ["fp32", "fp16"]
        
        # Initialize prediction matrix
        matrix = {
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "models": {},
            "hardware_platforms": hardware_platforms,
            "batch_sizes": batch_sizes,
            "precision_options": precision_options
        }
        
        # Generate predictions for each configuration
        for model_config in model_configs:
            model_name = model_config["name"]
            model_category = model_config["category"]
            
            logger.info(f"Generating predictions for {model_name}")
            
            model_results = {
                "name": model_name,
                "category": model_category,
                "predictions": {}
            }
            
            for hardware in hardware_platforms:
                hardware_results = {}
                
                for batch_size in batch_sizes:
                    batch_results = {}
                    
                    for precision in precision_options:
                        # Skip incompatible configurations
                        if precision == "fp16" and hardware == "cpu":
                            continue
                        
                        # Make prediction
                        pred = predict_performance(
                            models=models,
                            model_name=model_name,
                            model_category=model_category,
                            hardware=hardware,
                            batch_size=batch_size,
                            precision=precision,
                            mode=mode
                        )
                        
                        if pred:
                            batch_results[precision] = {
                                "throughput": pred.get("throughput"),
                                "latency_mean": pred.get("latency_mean"),
                                "memory_usage": pred.get("memory_usage")
                            }
                    
                    if batch_results:
                        hardware_results[str(batch_size)] = batch_results
                
                if hardware_results:
                    model_results["predictions"][hardware] = hardware_results
            
            matrix["models"][model_name] = model_results
        
        # Save prediction matrix
        if output_file:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(matrix, f, indent=2)
            logger.info(f"Saved prediction matrix to {output_file}")
        
        return matrix
    
    except Exception as e:
        logger.error(f"Error generating prediction matrix: {e}")
        return {}

def visualize_predictions(
    matrix: Dict[str, Any],
    metric: str = "throughput",
    output_dir: Optional[str] = None
) -> List[str]:
    """
    Visualize predictions from the prediction matrix.
    
    Args:
        matrix (Dict[str, Any]): Prediction matrix
        metric (str): Metric to visualize (throughput, latency_mean, memory_usage)
        output_dir (str): Directory to save visualizations
        
    Returns:
        List[str]: Paths to visualization files
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set output directory
        if output_dir is None:
            output_dir = PREDICTOR_DIR / "visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        visualization_files = []
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)
        
        # Get models, hardware platforms, and batch sizes
        models = list(matrix["models"].keys())
        hardware_platforms = matrix["hardware_platforms"]
        batch_sizes = [int(bs) for bs in matrix["batch_sizes"]]
        precision = matrix["precision_options"][0]  # Use first precision option
        
        # 1. Hardware comparison for each model at a fixed batch size
        batch_size = batch_sizes[1] if len(batch_sizes) > 1 else batch_sizes[0]
        
        plt.figure()
        
        # Extract data for plotting
        data = []
        for model_name in models:
            model_data = matrix["models"][model_name]
            for hw in hardware_platforms:
                if hw in model_data["predictions"]:
                    if str(batch_size) in model_data["predictions"][hw]:
                        if precision in model_data["predictions"][hw][str(batch_size)]:
                            value = model_data["predictions"][hw][str(batch_size)][precision].get(metric)
                            if value is not None:
                                data.append({
                                    "model": model_name.split("/")[-1],
                                    "hardware": hw,
                                    "value": value
                                })
        
        # Create DataFrame for plotting
        if data:
            df = pd.DataFrame(data)
            
            # Create bar plot
            ax = sns.barplot(x="model", y="value", hue="hardware", data=df)
            
            # Set plot title and labels
            metric_label = "Throughput (samples/sec)" if metric == "throughput" else "Latency (ms)" if metric == "latency_mean" else "Memory Usage (MB)"
            plt.title(f"{metric_label} by Hardware Platform (Batch Size = {batch_size})")
            plt.xlabel("Model")
            plt.ylabel(metric_label)
            plt.xticks(rotation=45, ha="right")
            plt.legend(title="Hardware")
            plt.tight_layout()
            
            # Save figure
            output_file = os.path.join(output_dir, f"{metric}_hardware_comparison.png")
            plt.savefig(output_file)
            plt.close()
            visualization_files.append(output_file)
        
        # 2. Batch size scaling for each model on a fixed hardware
        hardware = "cuda" if "cuda" in hardware_platforms else hardware_platforms[0]
        
        plt.figure()
        
        # Extract data for plotting
        data = []
        for model_name in models:
            model_data = matrix["models"][model_name]
            if hardware in model_data["predictions"]:
                for bs_str in model_data["predictions"][hardware]:
                    if precision in model_data["predictions"][hardware][bs_str]:
                        value = model_data["predictions"][hardware][bs_str][precision].get(metric)
                        if value is not None:
                            data.append({
                                "model": model_name.split("/")[-1],
                                "batch_size": int(bs_str),
                                "value": value
                            })
        
        # Create DataFrame for plotting
        if data:
            df = pd.DataFrame(data)
            
            # Create line plot
            plt.figure(figsize=(10, 6))
            for model in df["model"].unique():
                model_df = df[df["model"] == model]
                model_df = model_df.sort_values("batch_size")
                plt.plot(model_df["batch_size"], model_df["value"], marker='o', label=model)
            
            # Set plot title and labels
            metric_label = "Throughput (samples/sec)" if metric == "throughput" else "Latency (ms)" if metric == "latency_mean" else "Memory Usage (MB)"
            plt.title(f"{metric_label} vs Batch Size on {hardware.upper()}")
            plt.xlabel("Batch Size")
            plt.ylabel(metric_label)
            plt.legend()
            plt.grid(True)
            
            # Save figure
            output_file = os.path.join(output_dir, f"{metric}_batch_scaling_{hardware}.png")
            plt.savefig(output_file)
            plt.close()
            visualization_files.append(output_file)
        
        # 3. Hardware efficiency comparison (normalized to CPU)
        if "cpu" in hardware_platforms and len(hardware_platforms) > 1:
            plt.figure()
            
            # Extract data for plotting
            data = []
            for model_name in models:
                model_data = matrix["models"][model_name]
                
                # Skip if CPU data is missing
                if "cpu" not in model_data["predictions"]:
                    continue
                    
                # Get CPU baseline
                for bs_str in model_data["predictions"]["cpu"]:
                    if precision in model_data["predictions"]["cpu"][bs_str]:
                        cpu_value = model_data["predictions"]["cpu"][bs_str][precision].get(metric)
                        
                        if cpu_value is not None and cpu_value > 0:
                            # Get values for other hardware
                            for hw in hardware_platforms:
                                if hw == "cpu":
                                    continue
                                    
                                if hw in model_data["predictions"] and bs_str in model_data["predictions"][hw]:
                                    if precision in model_data["predictions"][hw][bs_str]:
                                        hw_value = model_data["predictions"][hw][bs_str][precision].get(metric)
                                        
                                        if hw_value is not None:
                                            # Calculate speedup
                                            if metric == "throughput":
                                                speedup = hw_value / cpu_value
                                            else:
                                                speedup = cpu_value / hw_value
                                                
                                            data.append({
                                                "model": model_name.split("/")[-1],
                                                "hardware": hw,
                                                "batch_size": int(bs_str),
                                                "speedup": speedup
                                            })
            
            # Create DataFrame for plotting
            if data:
                df = pd.DataFrame(data)
                
                # Filter to specific batch size
                df_filtered = df[df["batch_size"] == batch_size]
                
                if not df_filtered.empty:
                    # Create bar plot
                    ax = sns.barplot(x="model", y="speedup", hue="hardware", data=df_filtered)
                    
                    # Set plot title and labels
                    plt.title(f"Speedup vs. CPU for {metric} (Batch Size = {batch_size})")
                    plt.xlabel("Model")
                    plt.ylabel("Speedup Factor (higher is better)")
                    plt.xticks(rotation=45, ha="right")
                    plt.legend(title="Hardware")
                    plt.axhline(y=1.0, color='r', linestyle='--')
                    plt.tight_layout()
                    
                    # Save figure
                    output_file = os.path.join(output_dir, f"{metric}_speedup_comparison.png")
                    plt.savefig(output_file)
                    plt.close()
                    visualization_files.append(output_file)
        
        return visualization_files
    
    except ImportError:
        logger.warning("Matplotlib or seaborn not available, skipping visualization")
        return []
    except Exception as e:
        logger.error(f"Error visualizing predictions: {e}")
        return []

def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description="Advanced ML-based Performance Predictor")
    
    # Main options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Train prediction models")
    group.add_argument("--predict", action="store_true", help="Predict performance for a configuration")
    group.add_argument("--generate-matrix", action="store_true", help="Generate prediction matrix")
    group.add_argument("--visualize", action="store_true", help="Visualize predictions")
    group.add_argument("--version", action="store_true", help="Show predictor version")
    group.add_argument("--evaluate", action="store_true", help="Evaluate model accuracy on test data")
    
    # Training options
    parser.add_argument("--database", help="Path to benchmark database")
    parser.add_argument("--output-dir", help="Directory to save models")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data to use for testing (0.0-1.0)")
    parser.add_argument("--ensemble", action="store_true", help="Use ensemble models for better accuracy")
    parser.add_argument("--no-hyperparameter-tuning", action="store_true", help="Skip hyperparameter tuning (faster training)")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel jobs (-1 for all cores)")
    
    # Prediction options
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--category", help="Model category (text_embedding, text_generation, vision, audio, multimodal, video)")
    parser.add_argument("--hardware", help="Hardware platform")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp32", help="Precision")
    parser.add_argument("--mode", choices=["inference", "training"], default="inference", help="Mode")
    parser.add_argument("--seq-length", type=int, default=128, help="Sequence length for text models")
    parser.add_argument("--gpu-count", type=int, default=1, help="Number of GPUs for distributed training")
    parser.add_argument("--distributed", action="store_true", help="Use distributed training")
    parser.add_argument("--no-uncertainty", action="store_true", help="Disable uncertainty estimation")
    parser.add_argument("--model-dir", help="Directory containing prediction models")
    
    # Matrix options
    parser.add_argument("--output", help="Output file for prediction matrix or visualization")
    parser.add_argument("--metric", choices=["throughput", "latency_mean", "memory_usage"], default="throughput", help="Metric to visualize")
    parser.add_argument("--batch-sizes", type=str, default="1,8,32", help="Comma-separated list of batch sizes for matrix")
    parser.add_argument("--precisions", type=str, default="fp32,fp16", help="Comma-separated list of precisions for matrix")
    parser.add_argument("--hardware-platforms", type=str, help="Comma-separated list of hardware platforms for matrix")
    parser.add_argument("--custom-models", type=str, help="Path to JSON file with custom model configs")
    
    # Visualization options
    parser.add_argument("--theme", choices=["light", "dark"], default="light", help="Theme for visualizations")
    parser.add_argument("--output-format", choices=["png", "svg", "pdf"], default="png", help="Output format for visualizations")
    parser.add_argument("--interactive", action="store_true", help="Generate interactive HTML visualizations")
    
    # Logging options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--quiet", action="store_true", help="Suppress all non-error output")
    parser.add_argument("--log-file", help="Path to log file")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    # Create directories
    os.makedirs(PREDICTOR_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if args.version:
        print("IPFS Accelerate Performance Predictor v2.0.0")
        print("Part of Phase 16 of the IPFS Accelerate Python Framework")
        sys.exit(0)
    
    if args.train:
        # Train prediction models
        print("Training advanced prediction models...")
        
        # Load benchmark data
        df = load_benchmark_data(args.database)
        if df.empty:
            print("No benchmark data available, exiting")
            sys.exit(1)
        
        # Preprocess data
        df, preprocessing_info = preprocess_data(df)
        if df.empty:
            print("Error preprocessing data, exiting")
            sys.exit(1)
        
        # Train models
        models = train_prediction_models(
            df, 
            preprocessing_info,
            test_size=args.test_size,
            random_state=args.random_seed,
            hyperparameter_tuning=not args.no_hyperparameter_tuning,
            use_ensemble=args.ensemble,
            n_jobs=args.n_jobs
        )
        
        if not models:
            print("Error training models, exiting")
            sys.exit(1)
        
        # Print model metrics
        for target in PREDICTION_METRICS:
            if target in models:
                metrics = models[target].get("metrics", {})
                print(f"\nModel metrics for {target}:")
                print(f"  Test R²: {metrics.get('test_r2', 'N/A'):.4f}")
                print(f"  MAPE: {metrics.get('mape', 'N/A'):.2%}")
                print(f"  RMSE: {metrics.get('rmse', 'N/A'):.4f}")
                
                if "best_params" in metrics:
                    print(f"  Best parameters: {metrics['best_params']}")
                
                # Print top feature importances if available
                if "feature_importance" in metrics and isinstance(metrics["feature_importance"], dict):
                    importances = metrics["feature_importance"]
                    print("  Top feature importances:")
                    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
                    for feature, importance in sorted_features:
                        print(f"    {feature}: {importance:.4f}")
        
        # Save models
        model_dir = save_prediction_models(models, args.output_dir)
        if not model_dir:
            print("Error saving models, exiting")
            sys.exit(1)
        
        print(f"Trained prediction models saved to {model_dir}")
    
    elif args.predict:
        # Predict performance for a configuration
        if not args.model:
            print("No model name provided, exiting")
            sys.exit(1)
        
        if not args.hardware:
            print("No hardware platform provided, exiting")
            sys.exit(1)
        
        if not args.batch_size:
            print("No batch size provided, exiting")
            sys.exit(1)
        
        if not args.category:
            print("No model category provided, exiting")
            sys.exit(1)
        
        # Load models
        models = load_prediction_models(args.model_dir)
        if not models:
            print("No prediction models available, exiting")
            sys.exit(1)
        
        # Make prediction
        prediction = predict_performance(
            models=models,
            model_name=args.model,
            model_category=args.category,
            hardware=args.hardware,
            batch_size=args.batch_size,
            precision=args.precision,
            mode=args.mode,
            sequence_length=args.seq_length,
            gpu_count=args.gpu_count,
            is_distributed=args.distributed,
            calculate_uncertainty=not args.no_uncertainty
        )
        
        if not prediction:
            print("Error making prediction, exiting")
            sys.exit(1)
        
        # Print prediction
        print("\nPerformance Prediction:")
        print(f"Model: {args.model}")
        print(f"Hardware: {args.hardware}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Precision: {args.precision}")
        print(f"Mode: {args.mode}")
        
        # Print predictions with uncertainty
        for target in PREDICTION_METRICS:
            if target in prediction.get("predictions", {}):
                value = prediction["predictions"][target]
                if target in prediction.get("uncertainties", {}):
                    uncertainty = prediction["uncertainties"][target]
                    confidence = uncertainty.get("confidence", 0.0) * 100
                    lower = uncertainty.get("lower_bound", value * 0.85)
                    upper = uncertainty.get("upper_bound", value * 1.15)
                    
                    if target == "throughput":
                        print(f"Throughput: {value:.2f} samples/sec (confidence: {confidence:.1f}%)")
                        print(f"  Range: {lower:.2f} - {upper:.2f} samples/sec")
                    elif target == "latency_mean":
                        print(f"Latency: {value:.2f} ms (confidence: {confidence:.1f}%)")
                        print(f"  Range: {lower:.2f} - {upper:.2f} ms")
                    elif target == "memory_usage":
                        print(f"Memory Usage: {value:.2f} MB (confidence: {confidence:.1f}%)")
                        print(f"  Range: {lower:.2f} - {upper:.2f} MB")
                else:
                    if target == "throughput":
                        print(f"Throughput: {value:.2f} samples/sec")
                    elif target == "latency_mean":
                        print(f"Latency: {value:.2f} ms")
                    elif target == "memory_usage":
                        print(f"Memory Usage: {value:.2f} MB")
        
        # Print overall confidence
        print(f"Overall Confidence: {prediction.get('confidence_score', 0.0) * 100:.1f}%")
        
        # Print explanations if any
        if "explanation" in prediction and prediction["explanation"]:
            print("\nExplanations:")
            for explanation in prediction["explanation"]:
                print(f"- {explanation}")
        
        # Save prediction
        if args.output:
            with open(args.output, "w") as f:
                json.dump(prediction, f, indent=2)
            print(f"Prediction saved to {args.output}")
    
    elif args.generate_matrix:
        # Generate prediction matrix
        # Load models
        models = load_prediction_models(args.model_dir)
        if not models:
            print("No prediction models available, exiting")
            sys.exit(1)
        
        # Parse batch sizes and precisions
        batch_sizes = [int(bs) for bs in args.batch_sizes.split(",")]
        precisions = args.precisions.split(",")
        
        # Parse hardware platforms if provided
        hardware_platforms = None
        if args.hardware_platforms:
            hardware_platforms = args.hardware_platforms.split(",")
        
        # Load custom models if provided
        model_configs = None
        if args.custom_models:
            try:
                with open(args.custom_models, "r") as f:
                    model_configs = json.load(f)
            except Exception as e:
                print(f"Error loading custom models: {e}")
                sys.exit(1)
        
        # Generate matrix
        matrix = generate_prediction_matrix(
            models=models,
            model_configs=model_configs,
            hardware_platforms=hardware_platforms,
            batch_sizes=batch_sizes,
            precision_options=precisions,
            mode=args.mode,
            output_file=args.output
        )
        
        if not matrix:
            print("Error generating prediction matrix, exiting")
            sys.exit(1)
        
        print(f"Generated prediction matrix with {len(matrix.get('models', {}))} models")
        
        if args.output:
            print(f"Prediction matrix saved to {args.output}")
    
    elif args.visualize:
        # Visualize predictions
        if not args.output:
            print("No output file provided, exiting")
            sys.exit(1)
        
        # Load prediction matrix
        try:
            with open(args.output, "r") as f:
                matrix = json.load(f)
        except Exception as e:
            print(f"Error loading prediction matrix: {e}")
            sys.exit(1)
        
        # Create visualizations
        visualization_files = visualize_predictions(
            matrix=matrix,
            metric=args.metric,
            output_dir=args.output_dir
        )
        
        if not visualization_files:
            print("Error creating visualizations, exiting")
            sys.exit(1)
        
        print(f"Created {len(visualization_files)} visualizations:")
        for vis_file in visualization_files:
            print(f"- {vis_file}")

    elif args.evaluate:
        # Load models
        models = load_prediction_models(args.model_dir)
        if not models:
            print("No prediction models available, exiting")
            sys.exit(1)
        
        # Load benchmark data
        df = load_benchmark_data(args.database)
        if df.empty:
            print("No benchmark data available, exiting")
            sys.exit(1)
        
        # Preprocess data
        df, preprocessing_info = preprocess_data(df)
        if df.empty:
            print("Error preprocessing data, exiting")
            sys.exit(1)
        
        # Evaluate models on test data
        print("Evaluating model accuracy on test data...")
        
        # Prepare metrics lists
        metrics = {target: {"actual": [], "predicted": []} for target in PREDICTION_METRICS}
        
        # Create evaluation set
        eval_size = min(100, len(df))  # Use up to 100 samples for evaluation
        eval_indices = np.random.choice(len(df), size=eval_size, replace=False)
        eval_df = df.iloc[eval_indices]
        
        # Make predictions for each sample
        for idx, row in eval_df.iterrows():
            model_name = row.get("model_name", "unknown")
            model_category = row.get("category", "unknown")
            hardware = row.get("hardware", "unknown")
            batch_size = row.get("batch_size", 1)
            precision_numeric = row.get("precision_numeric", 32)
            precision = "fp32" if precision_numeric == 32 else "fp16" if precision_numeric == 16 else "int8"
            mode = row.get("mode", "inference")
            
            # Make prediction
            prediction = predict_performance(
                models=models,
                model_name=model_name,
                model_category=model_category,
                hardware=hardware,
                batch_size=batch_size,
                precision=precision,
                mode=mode,
                calculate_uncertainty=True
            )
            
            # Collect actual and predicted values
            for target in PREDICTION_METRICS:
                if target in row and target in prediction.get("predictions", {}):
                    actual = row[target]
                    predicted = prediction["predictions"][target]
                    
                    metrics[target]["actual"].append(actual)
                    metrics[target]["predicted"].append(predicted)
        
        # Calculate evaluation metrics
        print("\nEvaluation Metrics:")
        for target in PREDICTION_METRICS:
            if not metrics[target]["actual"]:
                print(f"No evaluation data available for {target}")
                continue
            
            actual = np.array(metrics[target]["actual"])
            predicted = np.array(metrics[target]["predicted"])
            
            # Calculate metrics
            r2 = r2_score(actual, predicted)
            mape_val = mean_absolute_percentage_error(actual[actual > 0], predicted[actual > 0]) if (actual > 0).any() else float('inf')
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            
            print(f"\n{target.capitalize()} Evaluation:")
            print(f"  R²: {r2:.4f}")
            print(f"  MAPE: {mape_val:.2%}")
            print(f"  RMSE: {rmse:.4f}")
            
            # Calculate percentage of predictions within uncertainty bounds
            if target in prediction.get("uncertainties", {}):
                uncertainty = prediction["uncertainties"][target]
                lower_bounds = predicted - (predicted * 0.15)  # Use 15% as default uncertainty
                upper_bounds = predicted + (predicted * 0.15)
                
                within_bounds = np.logical_and(actual >= lower_bounds, actual <= upper_bounds)
                percentage_within = np.mean(within_bounds) * 100
                
                print(f"  Predictions within uncertainty bounds: {percentage_within:.1f}%")
        
        # Save evaluation results if output specified
        if args.output:
            eval_results = {
                "metrics": {},
                "timestamp": datetime.now().isoformat(),
                "n_samples": eval_size,
                "eval_method": "random_sampling"
            }
            
            for target in PREDICTION_METRICS:
                if not metrics[target]["actual"]:
                    continue
                    
                actual = np.array(metrics[target]["actual"])
                predicted = np.array(metrics[target]["predicted"])
                
                eval_results["metrics"][target] = {
                    "r2": float(r2_score(actual, predicted)),
                    "mape": float(mean_absolute_percentage_error(actual[actual > 0], predicted[actual > 0]) if (actual > 0).any() else float('inf')),
                    "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
                    "mean_actual": float(np.mean(actual)),
                    "mean_predicted": float(np.mean(predicted)),
                    "n_samples": len(actual)
                }
            
            with open(args.output, "w") as f:
                json.dump(eval_results, f, indent=2)
            print(f"\nEvaluation results saved to {args.output}")

if __name__ == "__main__":
    main()