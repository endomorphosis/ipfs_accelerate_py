"""

# Import hardware detection capabilities if available
try:
    from hardware_detection import (
        HAS_CUDA, HAS_ROCM, HAS_OPENVINO, HAS_MPS, HAS_WEBNN, HAS_WEBGPU,
        detect_all_hardware
    )
    HAS_HARDWARE_DETECTION = True
except ImportError:
    HAS_HARDWARE_DETECTION = False
    # We'll detect hardware manually as fallback
Hardware Selector for the IPFS Accelerate Python Framework.

This module implements an automated system for selecting optimal hardware
for a given model and task based on benchmarking data.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Optional, Any, Tuple
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("hardware_selector")


class HardwareSelector:
    """A system for selecting optimal hardware based on model characteristics and benchmarking data."""

    def __init__(self, 
                database_path: str = "./benchmark_results",
                config_path: Optional[str] = None):
        """
        Initialize the hardware selector.
        
        Args:
            database_path (str): Path to the benchmark results database.
            config_path (Optional[str]): Path to configuration file.
        """
        self.database_path = Path(database_path)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Load benchmark data
        self.benchmarks = self._load_benchmark_data()
        
        # Load hardware compatibility matrix
        self.compatibility_matrix = self._load_compatibility_matrix()
        
        # Initialize prediction models
        self.prediction_models = {}
        self._initialize_prediction_models()
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """
        Load configuration from file.
        
        Args:
            config_path (Optional[str]): Path to the configuration file.
            
        Returns:
            Dict: Configuration dictionary.
        """
        default_config = {
            "selection_criteria": {
                "inference": {
                    "latency_weight": 0.4,
                    "throughput_weight": 0.3,
                    "memory_weight": 0.2,
                    "compatibility_weight": 0.1
                },
                "training": {
                    "throughput_weight": 0.4,
                    "convergence_weight": 0.3,
                    "memory_weight": 0.2,
                    "compatibility_weight": 0.1
                }
            },
            "model_families": {
                "embedding": {
                    "batch_size_importance": "medium",
                    "model_size_importance": "low"
                },
                "text_generation": {
                    "batch_size_importance": "low",
                    "model_size_importance": "high"
                },
                "vision": {
                    "batch_size_importance": "medium",
                    "model_size_importance": "medium"
                },
                "audio": {
                    "batch_size_importance": "medium",
                    "model_size_importance": "medium"
                },
                "multimodal": {
                    "batch_size_importance": "high",
                    "model_size_importance": "high"
                }
            },
            "hardware_preferences": {
                "cpu": {
                    "cost_factor": 1.0,
                    "availability_factor": 1.0,
                    "power_factor": 0.8
                },
                "cuda": {
                    "cost_factor": 0.7,
                    "availability_factor": 0.8,
                    "power_factor": 0.4
                },
                "rocm": {
                    "cost_factor": 0.8,
                    "availability_factor": 0.6,
                    "power_factor": 0.5
                },
                "mps": {
                    "cost_factor": 0.9,
                    "availability_factor": 0.7,
                    "power_factor": 0.7
                },
                "openvino": {
                    "cost_factor": 0.9,
                    "availability_factor": 0.6,
                    "power_factor": 0.7
                },
                "webnn": {
                    "cost_factor": 1.0,
                    "availability_factor": 0.5,
                    "power_factor": 0.9
                },
                "webgpu": {
                    "cost_factor": 0.9,
                    "availability_factor": 0.4,
                    "power_factor": 0.6
                }
            },
            "batch_size_thresholds": {
                "small": 1,
                "medium": 8,
                "large": 32
            },
            "model_size_categories": {
                "small": 100000000,  # ~100M parameters
                "medium": 1000000000,  # ~1B parameters
                "large": 10000000000  # ~10B parameters
            },
            "prediction_features": [
                "model_family",
                "model_size",
                "batch_size",
                "hardware_type",
                "sequence_length"
            ],
            "fallback_order": [
                "cuda",
                "rocm",
                "mps",
                "openvino",
                "cpu",
                "webgpu",
                "webnn"
            ]
        }
        
        if config_path is None:
            return default_config
        
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Configuration file {config_path} not found, using default configuration")
            return default_config
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return config
    
    def _load_benchmark_data(self) -> Dict:
        """
        Load benchmark data from database.
        
        Returns:
            Dict: Benchmark data.
        """
        # Check for aggregated benchmark data
        aggregated_file = self.database_path / "processed_results" / "aggregated_benchmarks.json"
        if aggregated_file.exists():
            with open(aggregated_file, 'r') as f:
                return json.load(f)
        
        # If no aggregated data, build from raw results
        logger.info("No aggregated benchmark data found, building from raw results")
        
        benchmarks = {
            "inference": {},
            "training": {},
            "hardware_compatibility": {}
        }
        
        # Load inference benchmarks
        inference_results_dir = self.database_path / "raw_results"
        if inference_results_dir.exists():
            inference_files = list(inference_results_dir.glob("*.json"))
            for file in inference_files:
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                    
                    # Skip training benchmark files
                    if "training_benchmark" in file.name:
                        continue
                    
                    # Process benchmark data
                    if "benchmarks" in data:
                        self._process_inference_benchmark(data, benchmarks["inference"])
                except Exception as e:
                    logger.warning(f"Error processing inference benchmark file {file}: {e}")
        
        # Load training benchmarks
        training_results_dir = self.database_path / "raw_results"
        if training_results_dir.exists():
            training_files = list(training_results_dir.glob("training_benchmark_*.json"))
            for file in training_files:
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                    
                    # Process training benchmark data
                    if "benchmarks" in data:
                        self._process_training_benchmark(data, benchmarks["training"])
                except Exception as e:
                    logger.warning(f"Error processing training benchmark file {file}: {e}")
        
        # Save aggregated data
        os.makedirs(os.path.dirname(aggregated_file), exist_ok=True)
        with open(aggregated_file, 'w') as f:
            json.dump(benchmarks, f, indent=2)
        
        return benchmarks
    
    def _process_inference_benchmark(self, benchmark_data: Dict, benchmarks: Dict):
        """
        Process inference benchmark data.
        
        Args:
            benchmark_data (Dict): Raw benchmark data.
            benchmarks (Dict): Dictionary to store processed benchmark data.
        """
        timestamp = benchmark_data.get("timestamp", "")
        
        for model_family, models in benchmark_data.get("benchmarks", {}).items():
            if model_family not in benchmarks:
                benchmarks[model_family] = {}
                
            for model_name, hardware_results in models.items():
                model_key = f"{model_family}/{model_name}"
                if model_key not in benchmarks[model_family]:
                    benchmarks[model_family][model_key] = {}
                
                for hardware_type, hw_data in hardware_results.items():
                    if hw_data.get("status") != "completed":
                        continue
                    
                    if hardware_type not in benchmarks[model_family][model_key]:
                        benchmarks[model_family][model_key][hardware_type] = []
                    
                    # Extract metrics from benchmark
                    perf_summary = hw_data.get("performance_summary", {})
                    metrics = {
                        "timestamp": timestamp,
                        "latency": perf_summary.get("latency", {}).get("mean", 0),
                        "throughput": perf_summary.get("throughput", {}).get("mean", 0),
                        "memory_usage": perf_summary.get("memory_usage", {}).get("mean", 0)
                    }
                    
                    # Add batch size data if available
                    batch_results = hw_data.get("benchmark_results", {})
                    for batch_key, batch_data in batch_results.items():
                        # Extract batch size from key (e.g., "batch_1_seq_32")
                        parts = batch_key.split("_")
                        if len(parts) >= 2 and parts[0] == "batch":
                            try:
                                batch_size = int(parts[1])
                                metrics["batch_size"] = batch_size
                            except ValueError:
                                pass
                        
                        # Extract sequence length if available
                        if len(parts) >= 4 and parts[2] == "seq":
                            try:
                                seq_length = int(parts[3])
                                metrics["sequence_length"] = seq_length
                            except ValueError:
                                pass
                        
                        # Add batch metrics
                        metrics[f"latency_{batch_key}"] = batch_data.get("avg_latency", 0)
                        metrics[f"throughput_{batch_key}"] = batch_data.get("throughput", 0)
                    
                    benchmarks[model_family][model_key][hardware_type].append(metrics)
    
    def _process_training_benchmark(self, benchmark_data: Dict, benchmarks: Dict):
        """
        Process training benchmark data.
        
        Args:
            benchmark_data (Dict): Raw benchmark data.
            benchmarks (Dict): Dictionary to store processed benchmark data.
        """
        timestamp = benchmark_data.get("timestamp", "")
        
        for model_family, models in benchmark_data.get("benchmarks", {}).items():
            if model_family not in benchmarks:
                benchmarks[model_family] = {}
                
            for model_name, hardware_results in models.items():
                model_key = f"{model_family}/{model_name}"
                if model_key not in benchmarks[model_family]:
                    benchmarks[model_family][model_key] = {}
                
                for hardware_type, hw_data in hardware_results.items():
                    if hw_data.get("status") != "completed":
                        continue
                    
                    if hardware_type not in benchmarks[model_family][model_key]:
                        benchmarks[model_family][model_key][hardware_type] = []
                    
                    # Extract performance summary metrics
                    perf_summary = hw_data.get("performance_summary", {})
                    
                    # Create base metrics dictionary
                    metrics = {
                        "timestamp": timestamp,
                        "training_time": perf_summary.get("training_time", {}).get("mean", 0),
                        "memory_usage": perf_summary.get("memory_usage", {}).get("mean", 0),
                        "throughput": perf_summary.get("throughput", {}).get("mean", 0),
                        "loss_convergence": perf_summary.get("loss_convergence", {}).get("mean", 0)
                    }
                    
                    # Process batch size data
                    batch_size_data = perf_summary.get("training_time", {}).get("by_batch_size", {})
                    for batch_size_str, time_value in batch_size_data.items():
                        try:
                            batch_size = int(batch_size_str)
                            metrics[f"training_time_batch_{batch_size}"] = time_value
                            
                            # Also add throughput data if available
                            throughput_data = perf_summary.get("throughput", {}).get("by_batch_size", {})
                            if batch_size_str in throughput_data:
                                metrics[f"throughput_batch_{batch_size}"] = throughput_data[batch_size_str]
                        except ValueError:
                            pass
                    
                    # Add learning rate data
                    lr_data = perf_summary.get("loss_convergence", {}).get("by_learning_rate", {})
                    for lr_str, convergence_value in lr_data.items():
                        metrics[f"convergence_lr_{lr_str}"] = convergence_value
                    
                    # Add mixed precision data
                    mixed_precision_data = perf_summary.get("training_time", {}).get("by_mixed_precision", {})
                    if "mixed_precision" in mixed_precision_data:
                        metrics["mixed_precision_time"] = mixed_precision_data["mixed_precision"]
                        metrics["mixed_precision_speedup"] = mixed_precision_data.get("full_precision", 0) / mixed_precision_data["mixed_precision"] if mixed_precision_data.get("full_precision", 0) > 0 else 1.0
                    
                    benchmarks[model_family][model_key][hardware_type].append(metrics)
    
    def _load_compatibility_matrix(self) -> Dict:
        """
        Load hardware compatibility matrix.
        
        Returns:
            Dict: Hardware compatibility matrix.
        """
        # Check for compatibility matrix file
        matrix_file = self.database_path / "hardware_compatibility_matrix.json"
        if matrix_file.exists():
            with open(matrix_file, 'r') as f:
                return json.load(f)
        
        # Load compatibility matrix from Claude.md if available
        claude_md_file = Path("./CLAUDE.md")
        if claude_md_file.exists():
            logger.info("Loading hardware compatibility information from CLAUDE.md")
            return self._parse_compatibility_from_claude_md(claude_md_file)
        
        # Create a default compatibility matrix
        logger.warning("No hardware compatibility matrix found, creating default")
        
        # Default matrix with basic compatibility info
        matrix = {
            "timestamp": "2025-03-01T00:00:00Z",
            "hardware_types": ["cpu", "cuda", "rocm", "mps", "openvino", "webnn", "webgpu"],
            "model_families": {
                "embedding": {
                    "hardware_compatibility": {
                        "cpu": {"compatible": True, "performance_rating": "medium"},
                        "cuda": {"compatible": True, "performance_rating": "high"},
                        "rocm": {"compatible": True, "performance_rating": "high"},
                        "mps": {"compatible": True, "performance_rating": "high"},
                        "openvino": {"compatible": True, "performance_rating": "medium"},
                        "webnn": {"compatible": True, "performance_rating": "high"},
                        "webgpu": {"compatible": True, "performance_rating": "medium"}
                    }
                },
                "text_generation": {
                    "hardware_compatibility": {
                        "cpu": {"compatible": True, "performance_rating": "low"},
                        "cuda": {"compatible": True, "performance_rating": "high"},
                        "rocm": {"compatible": True, "performance_rating": "medium"},
                        "mps": {"compatible": True, "performance_rating": "medium"},
                        "openvino": {"compatible": True, "performance_rating": "low"},
                        "webnn": {"compatible": False, "performance_rating": "unknown"},
                        "webgpu": {"compatible": True, "performance_rating": "low"}
                    }
                },
                "vision": {
                    "hardware_compatibility": {
                        "cpu": {"compatible": True, "performance_rating": "medium"},
                        "cuda": {"compatible": True, "performance_rating": "high"},
                        "rocm": {"compatible": True, "performance_rating": "medium"},
                        "mps": {"compatible": True, "performance_rating": "high"},
                        "openvino": {"compatible": True, "performance_rating": "high"},
                        "webnn": {"compatible": True, "performance_rating": "medium"},
                        "webgpu": {"compatible": True, "performance_rating": "medium"}
                    }
                },
                "audio": {
                    "hardware_compatibility": {
                        "cpu": {"compatible": True, "performance_rating": "medium"},
                        "cuda": {"compatible": True, "performance_rating": "high"},
                        "rocm": {"compatible": True, "performance_rating": "medium"},
                        "mps": {"compatible": True, "performance_rating": "medium"},
                        "openvino": {"compatible": True, "performance_rating": "medium"},
                        "webnn": {"compatible": False, "performance_rating": "unknown"},
                        "webgpu": {"compatible": False, "performance_rating": "unknown"}
                    }
                },
                "multimodal": {
                    "hardware_compatibility": {
                        "cpu": {"compatible": True, "performance_rating": "low"},
                        "cuda": {"compatible": True, "performance_rating": "high"},
                        "rocm": {"compatible": False, "performance_rating": "unknown"},
                        "mps": {"compatible": False, "performance_rating": "unknown"},
                        "openvino": {"compatible": False, "performance_rating": "unknown"},
                        "webnn": {"compatible": False, "performance_rating": "unknown"},
                        "webgpu": {"compatible": False, "performance_rating": "unknown"}
                    }
                }
            }
        }
        
        # Save default matrix
        with open(matrix_file, 'w') as f:
            json.dump(matrix, f, indent=2)
        
        return matrix
    
    def _parse_compatibility_from_claude_md(self, claude_md_file: Path) -> Dict:
        """
        Parse hardware compatibility information from CLAUDE.md file.
        
        Args:
            claude_md_file (Path): Path to CLAUDE.md file.
            
        Returns:
            Dict: Hardware compatibility matrix.
        """
        import re
        
        # Initialize compatibility matrix
        matrix = {
            "timestamp": "2025-03-01T00:00:00Z",
            "hardware_types": ["cpu", "cuda", "rocm", "mps", "openvino", "webnn", "webgpu"],
            "model_families": {}
        }
        
        # Read the file content
        with open(claude_md_file, 'r') as f:
            content = f.read()
        
        # Look for model family-based compatibility chart
        compatibility_section = re.search(r'### Model Family-Based Compatibility Chart\s+\|\s+Model Family\s+\|.*?\|(.*?)(?=\n\n|\Z)', content, re.DOTALL)
        
        if compatibility_section:
            rows = compatibility_section.group(1).strip().split('\n')
            
            # Process each row
            for row in rows:
                # Skip the header separator row
                if '---' in row:
                    continue
                
                # Parse the row
                cells = [cell.strip() for cell in row.split('|')]
                if len(cells) < 9:  # Make sure we have enough cells
                    continue
                
                model_family = cells[1].lower()
                
                # Remove "Models" from model family name if present
                model_family = model_family.replace('models', '').strip()
                
                # Map to standard model family names
                if 'embed' in model_family:
                    model_family = 'embedding'
                elif 'text' in model_family and 'gen' in model_family:
                    model_family = 'text_generation'
                elif 'vision' in model_family or 'clip' in model_family or 'vit' in model_family:
                    model_family = 'vision'
                elif 'audio' in model_family or 'whisper' in model_family or 'wav2vec' in model_family:
                    model_family = 'audio'
                elif 'multi' in model_family or 'llava' in model_family:
                    model_family = 'multimodal'
                
                # Initialize model family in matrix
                if model_family not in matrix["model_families"]:
                    matrix["model_families"][model_family] = {
                        "hardware_compatibility": {}
                    }
                
                # Add compatibility info for each hardware type
                hardware_types = ["cuda", "rocm", "mps", "openvino", "webnn", "webgpu"]
                for i, hw_type in enumerate(hardware_types):
                    cell_value = cells[i + 2]  # +2 to skip the first two columns
                    
                    # Parse compatibility and performance rating
                    compatible = "✅" in cell_value
                    
                    performance_rating = "unknown"
                    if "High" in cell_value:
                        performance_rating = "high"
                    elif "Medium" in cell_value:
                        performance_rating = "medium"
                    elif "Low" in cell_value:
                        performance_rating = "low"
                    
                    matrix["model_families"][model_family]["hardware_compatibility"][hw_type] = {
                        "compatible": compatible,
                        "performance_rating": performance_rating
                    }
                
                # Always add CPU compatibility (assuming it's always compatible with at least low performance)
                matrix["model_families"][model_family]["hardware_compatibility"]["cpu"] = {
                    "compatible": True,
                    "performance_rating": "medium"
                }
        
        return matrix
    
    def _initialize_prediction_models(self):
        """Initialize prediction models for performance prediction."""
        # Initialize flags
        self.sklearn_available = False
        self.using_external_models = False
        
        # Create empty model dictionary
        self.prediction_models = {
            "inference": {},
            "training": {}
        }
        
        # Create a configuration file path for model hyperparameters
        model_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_hyperparams.json")
        model_hyperparams = self._load_model_hyperparams(model_config_path)
        
        # Check if scikit-learn is available
        try:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.preprocessing import StandardScaler
            self.sklearn_available = True
            logger.info("scikit-learn is available, enabling advanced prediction features")
        except ImportError:
            logger.warning("scikit-learn not available, performance prediction will be limited")
            return
        
        # Check if external prediction models are available
        external_models_available = False
        try:
            # Try to load external model
            from model_performance_predictor import load_prediction_models
            external_models = load_prediction_models()
            if external_models and len(external_models) > 1:  # Check if we have more than just preprocessing_info
                logger.info("Loading external performance prediction models")
                self.prediction_models["external"] = external_models
                self.using_external_models = True
                external_models_available = True
        except (ImportError, Exception) as e:
            logger.debug(f"External performance models not available: {e}")
            self.using_external_models = False
        
        # Skip internal model training if external models are available
        if external_models_available:
            logger.info("Using external prediction models, skipping internal model training")
            return
        
        # Create prediction models for inference and training
        for mode in ["inference", "training"]:
            # Prepare training data
            try:
                X, y_latency, y_throughput, y_memory = self._prepare_prediction_data(mode)
            except Exception as e:
                logger.warning(f"Failed to prepare prediction data for {mode}: {e}")
                continue
            
            if len(X) == 0:
                logger.warning(f"No data available for {mode} performance prediction")
                # Create fallback models for each metric
                self._initialize_fallback_models(mode)
                continue
            
            # Ensure metrics dictionaries exist
            if mode not in self.prediction_models:
                self.prediction_models[mode] = {}
            
            # Set thresholds for model quality
            min_samples_for_training = 10  # Minimum number of samples required for training
            
            # Latency prediction model
            self._train_prediction_model(
                mode=mode,
                metric="latency",
                X=X,
                y=y_latency,
                min_samples=min_samples_for_training,
                model_type="RandomForest",
                hyperparams=model_hyperparams.get("latency", {})
            )
            
            # Throughput prediction model
            self._train_prediction_model(
                mode=mode,
                metric="throughput",
                X=X,
                y=y_throughput,
                min_samples=min_samples_for_training,
                model_type="GradientBoosting",
                hyperparams=model_hyperparams.get("throughput", {})
            )
            
            # Memory usage prediction model
            self._train_prediction_model(
                mode=mode,
                metric="memory_usage",
                X=X,
                y=y_memory,
                min_samples=min_samples_for_training,
                model_type="GradientBoosting",
                hyperparams=model_hyperparams.get("memory_usage", {})
            )
    
    def _train_prediction_model(self, mode, metric, X, y, min_samples, model_type, hyperparams):
        """
        Train a prediction model for a specific metric.
        
        Args:
            mode (str): "inference" or "training"
            metric (str): The metric to predict (latency, throughput, memory_usage)
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target values
            min_samples (int): Minimum number of samples required for training
            model_type (str): Type of model to train ("RandomForest" or "GradientBoosting")
            hyperparams (dict): Hyperparameters for the model
        """
        if y is None or len(y) < min_samples:
            logger.warning(f"Insufficient data for {mode} {metric} prediction (samples: {0 if y is None else len(y)})")
            self._initialize_fallback_model(mode, metric)
            return
        
        try:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.preprocessing import StandardScaler
            
            # Create scaler and scale features
            scaler_X = StandardScaler()
            X_scaled = scaler_X.fit_transform(X)
            
            # Create model with hyperparameters
            default_rf_params = {
                "n_estimators": 100, 
                "max_depth": 10,
                "min_samples_split": 5,
                "random_state": 42
            }
            
            default_gb_params = {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 5,
                "random_state": 42
            }
            
            # Merge default params with custom hyperparams
            if model_type == "RandomForest":
                model_params = {**default_rf_params, **hyperparams}
                model = RandomForestRegressor(**model_params)
            else:  # GradientBoosting
                model_params = {**default_gb_params, **hyperparams}
                model = GradientBoostingRegressor(**model_params)
            
            # Fit model
            model.fit(X_scaled, y)
            
            # Store model
            self.prediction_models[mode][metric] = {
                "model": model,
                "scaler": scaler_X,
                "importance": dict(zip(
                    self.config["prediction_features"], 
                    model.feature_importances_
                )),
                "training_samples": len(y),
                "model_type": model_type,
                "params": model_params
            }
            
            logger.info(f"Trained {mode} {metric} prediction model with {len(y)} samples")
            
        except Exception as e:
            logger.warning(f"Failed to train {mode} {metric} prediction model: {e}")
            self._initialize_fallback_model(mode, metric)
    
    def _initialize_fallback_models(self, mode):
        """
        Initialize fallback models for a mode when no training data is available.
        
        Args:
            mode (str): "inference" or "training"
        """
        for metric in ["latency", "throughput", "memory_usage"]:
            self._initialize_fallback_model(mode, metric)
    
    def _initialize_fallback_model(self, mode, metric):
        """
        Initialize a fallback model for a specific metric.
        
        Args:
            mode (str): "inference" or "training"
            metric (str): The metric to predict (latency, throughput, memory_usage)
        """
        logger.warning(f"Initializing fallback model for {mode} {metric}")
        
        # Create a simple rule-based fallback model
        self.prediction_models.setdefault(mode, {})[metric] = {
            "model": None,
            "scaler": None,
            "fallback": True,
            "fallback_rules": {
                "embedding": {"cpu": 0.5, "cuda": 0.8, "rocm": 0.7, "mps": 0.7, "openvino": 0.6, "webnn": 0.5, "webgpu": 0.5},
                "text_generation": {"cpu": 0.3, "cuda": 0.9, "rocm": 0.8, "mps": 0.6, "openvino": 0.4, "webnn": 0.2, "webgpu": 0.3},
                "vision": {"cpu": 0.4, "cuda": 0.9, "rocm": 0.8, "mps": 0.7, "openvino": 0.7, "webnn": 0.6, "webgpu": 0.6},
                "audio": {"cpu": 0.4, "cuda": 0.9, "rocm": 0.7, "mps": 0.6, "openvino": 0.5, "webnn": 0.3, "webgpu": 0.4},
                "multimodal": {"cpu": 0.3, "cuda": 0.9, "rocm": 0.7, "mps": 0.5, "openvino": 0.4, "webnn": 0.2, "webgpu": 0.3}
            }
        }
    
    def _load_model_hyperparams(self, config_path):
        """
        Load model hyperparameters from a configuration file.
        
        Args:
            config_path (str): Path to configuration file.
            
        Returns:
            dict: Model hyperparameters.
        """
        default_hyperparams = {
            "latency": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5
            },
            "throughput": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 5
            },
            "memory_usage": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 5
            }
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    custom_hyperparams = json.load(f)
                # Merge default with custom
                merged_hyperparams = {}
                for metric in default_hyperparams:
                    merged_hyperparams[metric] = {**default_hyperparams[metric], **(custom_hyperparams.get(metric, {}))}
                return merged_hyperparams
            except Exception as e:
                logger.warning(f"Failed to load model hyperparameters from {config_path}: {e}")
                return default_hyperparams
        else:
            # Create the default config file for future customization
            try:
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                with open(config_path, 'w') as f:
                    json.dump(default_hyperparams, f, indent=2)
                logger.info(f"Created default model hyperparameters file at {config_path}")
            except Exception as e:
                logger.warning(f"Failed to create default hyperparameters file: {e}")
            return default_hyperparams
    
    def _prepare_prediction_data(self, mode: str) -> Tuple:
        """
        Prepare data for training prediction models.
        
        Args:
            mode (str): "inference" or "training".
            
        Returns:
            Tuple: (X, y_latency, y_throughput, y_memory) where X is the feature matrix,
                  y_latency is the latency target, y_throughput is the throughput target,
                  and y_memory is the memory usage target.
        """
        # Collect data from benchmarks
        data = []
        
        benchmarks = self.benchmarks.get(mode, {})
        for model_family, models in benchmarks.items():
            for model_key, hw_results in models.items():
                model_name = model_key.split("/")[1]
                
                # Estimate model size based on model name
                model_size = self._estimate_model_size(model_name)
                
                for hw_type, metrics_list in hw_results.items():
                    for metrics in metrics_list:
                        # Skip invalid metrics
                        if not metrics or not isinstance(metrics, dict):
                            continue
                            
                        # Get batch size if available, default to 1
                        batch_size = metrics.get("batch_size", 1)
                        
                        # Get sequence length if available, default to 128
                        seq_length = metrics.get("sequence_length", 128)
                        
                        # Create feature vector
                        feature = {
                            "model_family": self._encode_model_family(model_family),
                            "model_size": model_size,
                            "batch_size": batch_size,
                            "sequence_length": seq_length,
                            "hardware_type": self._encode_hardware_type(hw_type)
                        }
                        
                        # Add metrics based on mode
                        if mode == "inference":
                            feature["latency"] = metrics.get("latency", 0)
                            feature["throughput"] = metrics.get("throughput", 0)
                            feature["memory_usage"] = metrics.get("memory_usage", 0)
                        else:  # training
                            feature["latency"] = metrics.get("training_time", 0)
                            feature["throughput"] = metrics.get("throughput", 0)
                            feature["memory_usage"] = metrics.get("memory_usage", 0)
                        
                        # Additional metadata features for more accurate prediction
                        if "precision" in metrics:
                            precision = metrics["precision"]
                            # Convert precision to numeric value
                            if precision == "fp16":
                                feature["precision_numeric"] = 16
                            elif precision == "int8":
                                feature["precision_numeric"] = 8
                            else:  # default to fp32
                                feature["precision_numeric"] = 32
                        else:
                            feature["precision_numeric"] = 32  # default to fp32
                            
                        # Add feature only if we have valid metric values
                        if feature["latency"] > 0 or feature["throughput"] > 0 or feature["memory_usage"] > 0:
                            data.append(feature)
        
        # Convert to pandas DataFrame
        if not data:
            return [], None, None, None
        
        try:
            import pandas as pd
            df = pd.DataFrame(data)
            
            # Define feature columns based on config
            feature_cols = [col for col in self.config["prediction_features"] if col in df.columns]
            
            # Extract features and targets
            X = df[feature_cols].values
            y_latency = df["latency"].values if "latency" in df.columns else None
            y_throughput = df["throughput"].values if "throughput" in df.columns else None
            y_memory = df["memory_usage"].values if "memory_usage" in df.columns else None
            
            # Log the feature importance (feature names and their counts)
            logger.info(f"Training on {len(data)} samples with features: {feature_cols}")
            logger.info(f"Target stats - Latency: {len(y_latency) if y_latency is not None else 0} samples, " +
                      f"Throughput: {len(y_throughput) if y_throughput is not None else 0} samples, " +
                      f"Memory: {len(y_memory) if y_memory is not None else 0} samples")
            
            return X, y_latency, y_throughput, y_memory
        except Exception as e:
            logger.warning(f"Failed to prepare prediction data: {e}")
            return [], None, None, None
    
    def _encode_model_family(self, family: str) -> float:
        """
        Encode model family as a numerical value.
        
        Args:
            family (str): Model family name.
            
        Returns:
            float: Numerical encoding of model family.
        """
        families = {
            "embedding": 0.0,
            "text_generation": 1.0,
            "vision": 2.0,
            "audio": 3.0,
            "multimodal": 4.0
        }
        return families.get(family, 0.0)
    
    def _encode_hardware_type(self, hw_type: str) -> float:
        """
        Encode hardware type as a numerical value.
        
        Args:
            hw_type (str): Hardware type.
            
        Returns:
            float: Numerical encoding of hardware type.
        """
        types = {
            "cpu": 0.0,
            "cuda": 1.0,
            "rocm": 2.0,
            "mps": 3.0,
            "openvino": 4.0,
            "webnn": 5.0,
            "webgpu": 6.0
        }
        return types.get(hw_type, 0.0)
        
    def _get_fallback_score(self, model_info: Dict, model_family: str, hw_type: str, metric: str) -> float:
        """
        Get a fallback score from the fallback rules.
        
        Args:
            model_info (Dict): Model information dictionary.
            model_family (str): Model family.
            hw_type (str): Hardware type.
            metric (str): Metric type (latency, throughput, memory_usage).
            
        Returns:
            float: Fallback score.
        """
        # Get fallback rules
        fallback_rules = model_info.get("fallback_rules", {})
        
        # Get score for model family and hardware type
        if model_family in fallback_rules and hw_type in fallback_rules[model_family]:
            return fallback_rules[model_family][hw_type]
        
        # Fall back to reasonable defaults
        default_scores = {
            "latency": {
                "cpu": 0.4,
                "cuda": 0.8,
                "rocm": 0.7,
                "mps": 0.6,
                "openvino": 0.6,
                "webnn": 0.5,
                "webgpu": 0.5
            },
            "throughput": {
                "cpu": 0.3,
                "cuda": 0.9,
                "rocm": 0.8,
                "mps": 0.7,
                "openvino": 0.6,
                "webnn": 0.4,
                "webgpu": 0.5
            },
            "memory_usage": {
                "cpu": 0.5,
                "cuda": 0.6,
                "rocm": 0.6,
                "mps": 0.7,
                "openvino": 0.7,
                "webnn": 0.8,
                "webgpu": 0.7
            }
        }
        
        # Get default score for metric and hardware type
        if metric in default_scores and hw_type in default_scores[metric]:
            return default_scores[metric][hw_type]
        
        # Last resort default
        return 0.5  # Middle value as safest default
    
    def _estimate_model_size(self, model_name: str) -> int:
        """
        Estimate model size based on model name.
        
        Args:
            model_name (str): Model name.
            
        Returns:
            int: Estimated model size in parameters.
        """
        model_name = model_name.lower()
        
        # Look for size indicators in the model name
        if "tiny" in model_name:
            return 10000000  # 10M parameters
        elif "small" in model_name:
            return 50000000  # 50M parameters
        elif "base" in model_name:
            return 100000000  # 100M parameters
        elif "large" in model_name:
            return 300000000  # 300M parameters
        elif "xl" in model_name or "huge" in model_name:
            return 1000000000  # 1B parameters
        
        # Check for specific models
        if "bert" in model_name:
            if "tiny" in model_name:
                return 4000000  # 4M parameters
            elif "mini" in model_name:
                return 11000000  # 11M parameters
            elif "small" in model_name:
                return 29000000  # 29M parameters
            elif "base" in model_name:
                return 110000000  # 110M parameters
            elif "large" in model_name:
                return 340000000  # 340M parameters
            else:
                return 110000000  # Default to base size
        elif "t5" in model_name:
            if "small" in model_name:
                return 60000000  # 60M parameters
            elif "base" in model_name:
                return 220000000  # 220M parameters
            elif "large" in model_name:
                return 770000000  # 770M parameters
            elif "3b" in model_name:
                return 3000000000  # 3B parameters
            elif "11b" in model_name:
                return 11000000000  # 11B parameters
            else:
                return 220000000  # Default to base size
        elif "gpt2" in model_name:
            if "small" in model_name or "sm" in model_name:
                return 124000000  # 124M parameters
            elif "medium" in model_name or "med" in model_name:
                return 355000000  # 355M parameters
            elif "large" in model_name or "lg" in model_name:
                return 774000000  # 774M parameters
            elif "xl" in model_name:
                return 1500000000  # 1.5B parameters
            else:
                return 124000000  # Default to small size
        
        # Default size if not recognized
        return 100000000  # 100M parameters
    
    def select_hardware(self, 
                       model_family: str,
                       model_name: str,
                       batch_size: int = 1,
                       sequence_length: int = 128,
                       mode: str = "inference",
                       available_hardware: Optional[List[str]] = None,
                       precision: str = "fp32") -> Dict:
        """
        Select optimal hardware for a given model and task.
        
        Args:
            model_family (str): Model family (embedding, text_generation, vision, audio, multimodal).
            model_name (str): Model name.
            batch_size (int): Batch size.
            sequence_length (int): Sequence length.
            mode (str): "inference" or "training".
            available_hardware (Optional[List[str]]): List of available hardware types.
                                                    If None, uses all hardware.
            precision (str): Precision to use (fp32, fp16, int8).
            
        Returns:
            Dict: Hardware selection results with scores and recommendations.
        """
        # Validate input
        if model_family not in ["embedding", "text_generation", "vision", "audio", "multimodal"]:
            logger.warning(f"Unknown model family: {model_family}, defaulting to embedding")
            model_family = "embedding"
        
        if mode not in ["inference", "training"]:
            logger.warning(f"Unknown mode: {mode}, defaulting to inference")
            mode = "inference"
        
        # Use all hardware if not specified
        if available_hardware is None:
            available_hardware = ["cpu", "cuda", "rocm", "mps", "openvino", "webnn", "webgpu"]
        
        # Check hardware availability in compatibility matrix
        compatible_hardware = []
        for hw_type in available_hardware:
            try:
                hw_compat = self.compatibility_matrix["model_families"][model_family]["hardware_compatibility"].get(hw_type, {})
                if hw_compat.get("compatible", False):
                    compatible_hardware.append(hw_type)
            except KeyError:
                pass
        
        if not compatible_hardware:
            logger.warning(f"No compatible hardware found for {model_family}, using all available hardware")
            compatible_hardware = available_hardware
        
        # Get selection criteria based on mode
        selection_criteria = self.config["selection_criteria"].get(mode, {})
        latency_weight = selection_criteria.get("latency_weight", 0.4)
        throughput_weight = selection_criteria.get("throughput_weight", 0.3)
        memory_weight = selection_criteria.get("memory_weight", 0.2)
        compatibility_weight = selection_criteria.get("compatibility_weight", 0.1)
        
        # Calculate category for batch size
        batch_category = "small"
        if batch_size >= self.config["batch_size_thresholds"]["large"]:
            batch_category = "large"
        elif batch_size >= self.config["batch_size_thresholds"]["medium"]:
            batch_category = "medium"
        
        # Calculate model size and category
        model_size = self._estimate_model_size(model_name)
        model_size_category = "small"
        if model_size >= self.config["model_size_categories"]["large"]:
            model_size_category = "large"
        elif model_size >= self.config["model_size_categories"]["medium"]:
            model_size_category = "medium"
        
        # Get model family preferences
        family_preferences = self.config["model_families"].get(model_family, {})
        batch_size_importance = family_preferences.get("batch_size_importance", "medium")
        model_size_importance = family_preferences.get("model_size_importance", "medium")
        
        # Calculate scores for each hardware type
        hardware_scores = {}
        predictions = {}
        
        # Define precision numeric value
        precision_numeric = 32  # default to fp32
        if precision == "fp16":
            precision_numeric = 16
        elif precision == "int8":
            precision_numeric = 8
        
        for hw_type in compatible_hardware:
            # Skip incompatible precision-hardware combinations
            if precision != "fp32" and hw_type == "cpu":
                continue  # CPU doesn't support reduced precision
                
            # Get hardware preferences
            hw_preferences = self.config["hardware_preferences"].get(hw_type, {})
            cost_factor = hw_preferences.get("cost_factor", 1.0)
            availability_factor = hw_preferences.get("availability_factor", 1.0)
            power_factor = hw_preferences.get("power_factor", 1.0)
            
            # Get compatibility rating
            hw_compat = self.compatibility_matrix["model_families"][model_family]["hardware_compatibility"].get(hw_type, {})
            compatibility_rating = 0.0
            if hw_compat.get("performance_rating") == "high":
                compatibility_rating = 1.0
            elif hw_compat.get("performance_rating") == "medium":
                compatibility_rating = 0.7
            elif hw_compat.get("performance_rating") == "low":
                compatibility_rating = 0.4
            
            # Initialize prediction values
            latency_score = 0.0
            throughput_score = 0.0
            memory_usage = 0.0
            
            # Try external model first if available
            if hasattr(self, 'using_external_models') and self.using_external_models and self.prediction_models.get("external"):
                try:
                    # Use model_performance_predictor.py functionality
                    from model_performance_predictor import predict_performance
                    
                    # Make prediction
                    external_pred = predict_performance(
                        models=self.prediction_models["external"],
                        model_name=model_name,
                        model_category=model_family,
                        hardware=hw_type,
                        batch_size=batch_size,
                        precision=precision,
                        mode=mode
                    )
                    
                    if external_pred:
                        # Store predictions
                        predictions[hw_type] = {
                            "latency": external_pred.get("latency_mean", 0),
                            "throughput": external_pred.get("throughput", 0),
                            "memory_usage": external_pred.get("memory_usage", 0)
                        }
                        
                        # Calculate scores
                        latency_pred = external_pred.get("latency_mean", 0)
                        throughput_pred = external_pred.get("throughput", 0)
                        memory_usage = external_pred.get("memory_usage", 0)
                        
                        if latency_pred > 0:
                            latency_score = 1.0 / max(0.001, latency_pred)
                        if throughput_pred > 0:
                            throughput_score = throughput_pred
                        
                        logger.info(f"Using external prediction model for {hw_type} with {model_name}")
                    else:
                        logger.debug(f"External prediction failed for {hw_type} with {model_name}")
                except Exception as e:
                    logger.debug(f"Failed to use external prediction: {e}")
            
            # Use internal prediction models if external failed or not available
            if (latency_score == 0.0 or throughput_score == 0.0) and mode in self.prediction_models:
                # Try using trained models if available
                if self.sklearn_available:
                    # Prepare feature vector for prediction
                    feature = [
                        self._encode_model_family(model_family),
                        model_size,
                        batch_size,
                        sequence_length,
                        self._encode_hardware_type(hw_type)
                    ]
                    
                    # Add precision if available in feature list
                    if "precision_numeric" in self.config["prediction_features"]:
                        feature.append(precision_numeric)
                    
                    # Predict latency
                    if "latency" in self.prediction_models[mode] and latency_score == 0.0:
                        model_info = self.prediction_models[mode]["latency"]
                        if "fallback" in model_info and model_info["fallback"]:
                            # Use fallback rules for latency
                            latency_score = self._get_fallback_score(model_info, model_family, hw_type, "latency")
                            predictions.setdefault(hw_type, {})["latency"] = 1.0 / max(0.001, latency_score) if latency_score > 0 else 0
                            latency_score = max(0.001, latency_score)
                            logger.debug(f"Using fallback model for latency prediction: {hw_type}={latency_score}")
                        elif "model" in model_info and model_info["model"] is not None:
                            try:
                                scaler = model_info["scaler"]
                                model = model_info["model"]
                                
                                # Scale feature
                                feature_scaled = scaler.transform([feature])
                                
                                # Predict
                                latency_pred = max(0.001, model.predict(feature_scaled)[0])
                                
                                # Store prediction
                                predictions.setdefault(hw_type, {})["latency"] = latency_pred
                                
                                # Convert to score (lower latency is better)
                                latency_score = 1.0 / latency_pred
                                logger.debug(f"Predicted latency for {hw_type}: {latency_pred}")
                            except Exception as e:
                                logger.warning(f"Failed to predict latency for {hw_type}: {e}")
                                # Fall back to compatibility rating on error
                                latency_score = compatibility_rating
                    
                    # Predict throughput
                    if "throughput" in self.prediction_models[mode] and throughput_score == 0.0:
                        model_info = self.prediction_models[mode]["throughput"]
                        if "fallback" in model_info and model_info["fallback"]:
                            # Use fallback rules for throughput
                            throughput_score = self._get_fallback_score(model_info, model_family, hw_type, "throughput")
                            predictions.setdefault(hw_type, {})["throughput"] = throughput_score
                            logger.debug(f"Using fallback model for throughput prediction: {hw_type}={throughput_score}")
                        elif "model" in model_info and model_info["model"] is not None:
                            try:
                                scaler = model_info["scaler"]
                                model = model_info["model"]
                                
                                # Scale feature
                                feature_scaled = scaler.transform([feature])
                                
                                # Predict
                                throughput_pred = max(0.001, model.predict(feature_scaled)[0])
                                
                                # Store prediction
                                predictions.setdefault(hw_type, {})["throughput"] = throughput_pred
                                
                                # Convert to score (higher throughput is better)
                                throughput_score = throughput_pred
                                logger.debug(f"Predicted throughput for {hw_type}: {throughput_pred}")
                            except Exception as e:
                                logger.warning(f"Failed to predict throughput for {hw_type}: {e}")
                                # Fall back to compatibility rating on error
                                throughput_score = compatibility_rating
                    
                    # Predict memory usage
                    if "memory_usage" in self.prediction_models[mode] and memory_usage == 0.0:
                        model_info = self.prediction_models[mode]["memory_usage"]
                        if "fallback" in model_info and model_info["fallback"]:
                            # Use fallback rules for memory usage - higher values mean MORE memory usage, so invert for scoring
                            memory_score = self._get_fallback_score(model_info, model_family, hw_type, "memory_usage")
                            # Convert to reasonable memory usage estimate (5-20% of model size)
                            memory_usage = model_size * (0.05 + (1.0 - memory_score) * 0.15)
                            predictions.setdefault(hw_type, {})["memory_usage"] = memory_usage
                            logger.debug(f"Using fallback model for memory prediction: {hw_type}={memory_usage}")
                        elif "model" in model_info and model_info["model"] is not None:
                            try:
                                scaler = model_info["scaler"]
                                model = model_info["model"]
                                
                                # Scale feature
                                feature_scaled = scaler.transform([feature])
                                
                                # Predict
                                memory_pred = max(0.001, model.predict(feature_scaled)[0])
                                
                                # Store prediction
                                predictions.setdefault(hw_type, {})["memory_usage"] = memory_pred
                                memory_usage = memory_pred
                                logger.debug(f"Predicted memory usage for {hw_type}: {memory_pred}")
                            except Exception as e:
                                logger.warning(f"Failed to predict memory usage for {hw_type}: {e}")
                                # Fall back to model size-based estimate
                                memory_usage = model_size * 0.1  # 10% of model size as a fallback
                else:
                    # Use fallback predictions if scikit-learn is not available
                    for metric in ["latency", "throughput", "memory_usage"]:
                        if metric in self.prediction_models[mode]:
                            model_info = self.prediction_models[mode][metric]
                            if "fallback" in model_info and model_info["fallback"]:
                                score = self._get_fallback_score(model_info, model_family, hw_type, metric)
                                if metric == "latency" and latency_score == 0.0:
                                    latency_score = score
                                    predictions.setdefault(hw_type, {})["latency"] = 1.0 / max(0.001, score)
                                elif metric == "throughput" and throughput_score == 0.0:
                                    throughput_score = score
                                    predictions.setdefault(hw_type, {})["throughput"] = score
                                elif metric == "memory_usage" and memory_usage == 0.0:
                                    memory_usage = model_size * (0.05 + (1.0 - score) * 0.15)
                                    predictions.setdefault(hw_type, {})["memory_usage"] = memory_usage
            
            # If we couldn't predict, use compatibility rating as an approximation
            if latency_score == 0.0:
                latency_score = compatibility_rating
            if throughput_score == 0.0:
                throughput_score = compatibility_rating
            
            # Normalize scores
            max_latency_score = 100.0  # Arbitrary normalization constant
            max_throughput_score = 1000.0  # Arbitrary normalization constant
            
            latency_score = min(1.0, latency_score / max_latency_score)
            throughput_score = min(1.0, throughput_score / max_throughput_score)
            
            # Calculate memory factor
            if memory_usage > 0:
                # Invert memory score (lower is better)
                memory_factor = 1.0 - min(1.0, memory_usage / (model_size * 0.01))  # Assuming approx 1% of model size in memory
            else:
                # Use model size category as heuristic
                if model_size_category == "large":
                    memory_factor = 0.3
                elif model_size_category == "medium":
                    memory_factor = 0.6
                else:
                    memory_factor = 0.9
            
            # Adjust based on batch size and model size importance
            batch_importance_factor = 0.7  # Medium importance
            if batch_size_importance == "high":
                batch_importance_factor = 1.0
            elif batch_size_importance == "low":
                batch_importance_factor = 0.4
            
            model_importance_factor = 0.7  # Medium importance
            if model_size_importance == "high":
                model_importance_factor = 1.0
            elif model_size_importance == "low":
                model_importance_factor = 0.4
            
            # Apply importance factors to scores
            if batch_category == "large":
                # For large batch sizes, throughput is more important
                throughput_score *= batch_importance_factor
            elif batch_category == "small":
                # For small batch sizes, latency is more important
                latency_score *= batch_importance_factor
            
            # Apply model size importance to memory factor
            memory_factor *= model_importance_factor
            
            # Calculate final score
            score = (
                latency_weight * latency_score +
                throughput_weight * throughput_score +
                compatibility_weight * compatibility_rating +
                memory_weight * memory_factor
            )
            
            # Apply hardware preference factors
            preference_factor = (cost_factor + availability_factor + power_factor) / 3.0
            score *= preference_factor
            
            hardware_scores[hw_type] = {
                "score": score,
                "latency_score": latency_score,
                "throughput_score": throughput_score,
                "compatibility_score": compatibility_rating,
                "memory_factor": memory_factor,
                "preference_factor": preference_factor
            }
            
            # Add predictions if available
            if hw_type in predictions:
                hardware_scores[hw_type]["predictions"] = predictions[hw_type]
        
        # Sort hardware types by score
        ranked_hardware = sorted(hardware_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        
        # Determine primary and fallback recommendations
        primary_recommendation = ranked_hardware[0][0] if ranked_hardware else "cpu"
        
        # Find fallback options
        fallback_options = []
        for hw_type, _ in ranked_hardware[1:]:
            fallback_options.append(hw_type)
        
        # Add additional fallbacks from configured fallback order if needed
        for hw_type in self.config["fallback_order"]:
            if hw_type not in fallback_options and hw_type != primary_recommendation and hw_type in compatible_hardware:
                fallback_options.append(hw_type)
        
        # Create result
        result = {
            "model_family": model_family,
            "model_name": model_name,
            "model_size": model_size,
            "model_size_category": model_size_category,
            "batch_size": batch_size,
            "batch_category": batch_category,
            "sequence_length": sequence_length,
            "precision": precision,
            "mode": mode,
            "primary_recommendation": primary_recommendation,
            "fallback_options": fallback_options[:2],  # Limit to top 2 fallbacks
            "all_scores": hardware_scores,
            "compatible_hardware": compatible_hardware,
            "prediction_source": "external" if (hasattr(self, 'using_external_models') and self.using_external_models) else "internal"
        }
        
        # Generate a human-readable explanation of the recommendation
        result["explanation"] = self._generate_recommendation_explanation(result)
        
        return result
        
    def _generate_recommendation_explanation(self, result: Dict) -> str:
        """
        Generate a human-readable explanation of the hardware recommendation.
        
        Args:
            result (Dict): Hardware selection result.
            
        Returns:
            str: Human-readable explanation.
        """
        # Extract key information from the result
        model_name = result["model_name"]
        model_family = result["model_family"]
        batch_size = result["batch_size"]
        recommendation = result["primary_recommendation"]
        fallbacks = result["fallback_options"]
        scores = result["all_scores"]
        precision = result.get("precision", "fp32")
        
        # Get scores for primary recommendation
        rec_scores = scores.get(recommendation, {})
        
        # Determine the key factors that led to this recommendation
        key_factors = []
        if rec_scores.get("latency_score", 0) > 0.7:
            key_factors.append("low latency")
        if rec_scores.get("throughput_score", 0) > 0.7:
            key_factors.append("high throughput")
        if rec_scores.get("memory_factor", 0) > 0.7:
            key_factors.append("efficient memory usage")
        if rec_scores.get("compatibility_score", 0) > 0.7:
            key_factors.append("strong compatibility")
        
        # If no strong factors, include whatever factors we have
        if not key_factors:
            if rec_scores.get("latency_score", 0) > 0.3:
                key_factors.append("acceptable latency")
            if rec_scores.get("throughput_score", 0) > 0.3:
                key_factors.append("reasonable throughput")
            if rec_scores.get("memory_factor", 0) > 0.3:
                key_factors.append("adequate memory handling")
        
        # Combine factors into a readable string
        if len(key_factors) > 1:
            factors_str = ", ".join(key_factors[:-1]) + f" and {key_factors[-1]}"
        elif key_factors:
            factors_str = key_factors[0]
        else:
            factors_str = "overall balanced performance"
        
        # Generate explanation
        explanation = f"{recommendation.upper()} is recommended for {model_family} model '{model_name}' "
        explanation += f"with batch size {batch_size} at {precision} precision "
        explanation += f"due to {factors_str}."
        
        # Add fallback information
        if fallbacks:
            explanation += f" If unavailable, consider {' or '.join([fb.upper() for fb in fallbacks])} as alternatives."
        
        # Add prediction source if based on performance models
        if result.get("prediction_source"):
            source = "advanced performance prediction models" if result["prediction_source"] == "external" else "internal prediction models"
            explanation += f" This recommendation is based on {source}."
        
        return explanation
    
    def create_hardware_selection_map(self, model_families: Optional[List[str]] = None) -> Dict:
        """
        Create a hardware selection map for different model families, sizes, and batch sizes.
        
        Args:
            model_families (Optional[List[str]]): List of model families to include.
                                                If None, includes all families.
            
        Returns:
            Dict: Hardware selection map.
        """
        # Use all model families if not specified
        if model_families is None:
            model_families = ["embedding", "text_generation", "vision", "audio", "multimodal"]
        
        # Define model sizes and batch sizes to test
        model_sizes = {
            "small": "small",  # Example model name suffix
            "medium": "base",
            "large": "large"
        }
        
        batch_sizes = [1, 4, 16, 32, 64]
        
        # Create selection map
        selection_map = {
            "timestamp": "2025-03-01T00:00:00Z",
            "model_families": {}
        }
        
        for model_family in model_families:
            selection_map["model_families"][model_family] = {
                "model_sizes": {},
                "inference": {
                    "batch_sizes": {}
                },
                "training": {
                    "batch_sizes": {}
                }
            }
            
            # Test different model sizes with default batch size
            for size_category, size_suffix in model_sizes.items():
                model_name = f"{model_family}-{size_suffix}"
                
                # Select hardware for inference and training
                inference_result = self.select_hardware(
                    model_family=model_family,
                    model_name=model_name,
                    batch_size=1,
                    mode="inference"
                )
                
                training_result = self.select_hardware(
                    model_family=model_family,
                    model_name=model_name,
                    batch_size=16,
                    mode="training"
                )
                
                # Store results
                selection_map["model_families"][model_family]["model_sizes"][size_category] = {
                    "inference": {
                        "primary": inference_result["primary_recommendation"],
                        "fallbacks": inference_result["fallback_options"]
                    },
                    "training": {
                        "primary": training_result["primary_recommendation"],
                        "fallbacks": training_result["fallback_options"]
                    }
                }
            
            # Test different batch sizes with medium-sized model
            model_name = f"{model_family}-base"
            
            for batch_size in batch_sizes:
                # Select hardware for inference and training
                inference_result = self.select_hardware(
                    model_family=model_family,
                    model_name=model_name,
                    batch_size=batch_size,
                    mode="inference"
                )
                
                training_result = self.select_hardware(
                    model_family=model_family,
                    model_name=model_name,
                    batch_size=batch_size,
                    mode="training"
                )
                
                # Store results
                selection_map["model_families"][model_family]["inference"]["batch_sizes"][str(batch_size)] = {
                    "primary": inference_result["primary_recommendation"],
                    "fallbacks": inference_result["fallback_options"]
                }
                
                selection_map["model_families"][model_family]["training"]["batch_sizes"][str(batch_size)] = {
                    "primary": training_result["primary_recommendation"],
                    "fallbacks": training_result["fallback_options"]
                }
        
        return selection_map
    
    def select_hardware_for_task(self,
                               model_family: str,
                               model_name: str,
                               task_type: str,
                               batch_size: int = 1,
                               sequence_length: int = 128,
                               available_hardware: Optional[List[str]] = None,
                               distributed: bool = False,
                               gpu_count: int = 1,
                               training_config: Optional[Dict[str, Any]] = None) -> Dict:
        """
        Select hardware for a specific task.
        
        Args:
            model_family (str): Model family.
            model_name (str): Model name.
            task_type (str): Task type (e.g., "classification", "generation", "embedding", etc.).
            batch_size (int): Batch size.
            sequence_length (int): Sequence length.
            available_hardware (Optional[List[str]]): List of available hardware types.
            distributed (bool): Whether to consider distributed training configurations.
            gpu_count (int): Number of GPUs for distributed training.
            training_config (Optional[Dict[str, Any]]): Additional training configuration parameters.
            
        Returns:
            Dict: Hardware selection results.
        """
        # Map task type to mode
        mode = "inference"
        if task_type in ["training", "fine-tuning", "fine_tuning"]:
            mode = "training"
        
        # Set appropriate sequence length based on task
        if sequence_length is None:
            if task_type in ["summarization", "translation"]:
                sequence_length = 512
            elif task_type in ["generation", "chat"]:
                sequence_length = 1024
            elif task_type in ["classification", "sentiment"]:
                sequence_length = 128
            else:
                sequence_length = 256
        
        # Set appropriate batch size based on task if not specified
        if batch_size is None:
            if task_type in ["generation", "chat"]:
                batch_size = 1
            elif task_type in ["training", "fine-tuning", "fine_tuning"]:
                batch_size = 16
            elif task_type in ["classification", "embedding"]:
                batch_size = 32
            else:
                batch_size = 8
        
        # If distributed training is requested, modify available hardware
        if distributed and mode == "training":
            distributed_hardware = []
            
            # Only include hardware that supports distributed training
            for hw in available_hardware or ["cuda", "rocm"]:
                if hw in ["cuda", "rocm"]:
                    distributed_hardware.append(f"distributed_{hw}_ddp")
                    if gpu_count >= 4:  # Only suggest FSDP for larger GPU counts
                        distributed_hardware.append(f"distributed_{hw}_fsdp")
                    if hw == "cuda":  # DeepSpeed is primarily for CUDA
                        distributed_hardware.append("distributed_cuda_deepspeed")
            
            # Set available hardware to distributed options
            if distributed_hardware:
                available_hardware = distributed_hardware
                logger.info(f"Using distributed hardware options: {available_hardware}")
            else:
                logger.warning("Distributed training requested but no suitable hardware found")
        
        # Add training specific configurations
        training_params = {}
        if mode == "training" and training_config:
            # Extract relevant training parameters that affect hardware selection
            if "optimizer" in training_config:
                training_params["optimizer"] = training_config["optimizer"]
            if "mixed_precision" in training_config:
                training_params["mixed_precision"] = training_config["mixed_precision"]
            if "gradient_accumulation_steps" in training_config:
                training_params["gradient_accumulation_steps"] = training_config["gradient_accumulation_steps"]
            if "gradient_checkpointing" in training_config:
                training_params["gradient_checkpointing"] = training_config["gradient_checkpointing"]
            if "sharded_ddp" in training_config:
                training_params["sharded_ddp"] = training_config["sharded_ddp"]
            
            # Adjust batch size based on gradient accumulation
            if "gradient_accumulation_steps" in training_params:
                effective_batch_size = batch_size * training_params["gradient_accumulation_steps"]
                logger.info(f"Effective batch size with gradient accumulation: {effective_batch_size}")
        
        # Select hardware
        result = self.select_hardware(
            model_family=model_family,
            model_name=model_name,
            batch_size=batch_size,
            sequence_length=sequence_length,
            mode=mode,
            available_hardware=available_hardware
        )
        
        # Add distributed training info if applicable
        if distributed and mode == "training":
            result["distributed"] = True
            result["gpu_count"] = gpu_count
            result["training_params"] = training_params
            
            # Add distributed-specific recommendations
            if gpu_count <= 2:
                result["distributed_strategy"] = "DDP"
            elif gpu_count <= 8:
                if "large" in model_name or "xl" in model_name.lower():
                    result["distributed_strategy"] = "FSDP"
                else:
                    result["distributed_strategy"] = "DDP"
            else:  # More than 8 GPUs
                if "large" in model_name or "xl" in model_name.lower() or "billion" in model_name.lower():
                    result["distributed_strategy"] = "DeepSpeed+ZeRO3" if "cuda" in result["primary_recommendation"] else "FSDP"
                else:
                    result["distributed_strategy"] = "DDP"
        
        return result
    
    def save_selection_map(self, output_file: str = "hardware_selection_map.json"):
        """
        Create and save a hardware selection map.
        
        Args:
            output_file (str): Path to save the selection map.
        """
        selection_map = self.create_hardware_selection_map()
        
        with open(output_file, 'w') as f:
            json.dump(selection_map, f, indent=2)
        
        logger.info(f"Hardware selection map saved to {output_file}")
    
    def get_hardware_recommendations(self, model_family: str, model_size_category: str, batch_size: int, mode: str) -> Dict:
        """
        Get hardware recommendations from the selection map.
        
        Args:
            model_family (str): Model family.
            model_size_category (str): Model size category ("small", "medium", "large").
            batch_size (int): Batch size.
            mode (str): "inference" or "training".
            
        Returns:
            Dict: Hardware recommendations.
        """
        # Create a temporary selection map if needed
        selection_map_file = Path("hardware_selection_map.json")
        if not selection_map_file.exists():
            selection_map = self.create_hardware_selection_map([model_family])
        else:
            with open(selection_map_file, 'r') as f:
                selection_map = json.load(f)
        
        # Find the closest batch size in the map
        batch_sizes = [int(bs) for bs in selection_map["model_families"].get(model_family, {}).get(mode, {}).get("batch_sizes", {})]
        if not batch_sizes:
            return {
                "primary": "cuda",
                "fallbacks": ["cpu"]
            }
        
        closest_batch_size = min(batch_sizes, key=lambda x: abs(x - batch_size))
        
        # Get recommendations
        recommendations = selection_map["model_families"].get(model_family, {}).get(mode, {}).get("batch_sizes", {}).get(str(closest_batch_size), {})
        
        if not recommendations:
            # Fall back to model size-based recommendation
            recommendations = selection_map["model_families"].get(model_family, {}).get("model_sizes", {}).get(model_size_category, {}).get(mode, {})
        
        if not recommendations:
            return {
                "primary": "cuda",
                "fallbacks": ["cpu"]
            }
        
        return recommendations
        
    def get_distributed_training_config(self, 
                                   model_family: str,
                                   model_name: str,
                                   gpu_count: int,
                                   batch_size: int = 8,
                                   strategy: Optional[str] = None,
                                   use_mixed_precision: bool = True,
                                   max_memory_gb: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate an optimal distributed training configuration based on model and available hardware.
        
        Args:
            model_family (str): Model family.
            model_name (str): Model name.
            gpu_count (int): Number of GPUs available.
            batch_size (int): Per-GPU batch size.
            strategy (Optional[str]): Distributed strategy ("DDP", "FSDP", "DeepSpeed").
            use_mixed_precision (bool): Whether to use mixed precision training.
            max_memory_gb (Optional[int]): Maximum GPU memory in GB.
            
        Returns:
            Dict[str, Any]: Distributed training configuration.
        """
        # Determine model size
        model_size = self._estimate_model_size(model_name)
        model_size_gb = model_size * 4 / (1024 * 1024 * 1024)  # Approximate size in GB (4 bytes per parameter)
        
        # Determine appropriate strategy if not specified
        if strategy is None:
            if gpu_count <= 2:
                strategy = "DDP"
            elif gpu_count <= 8:
                if model_size_gb > 10:  # For models larger than 10GB memory footprint
                    strategy = "FSDP"
                else:
                    strategy = "DDP"
            else:  # More than 8 GPUs
                if model_size_gb > 20:  # For very large models
                    strategy = "DeepSpeed"
                elif model_size_gb > 10:
                    strategy = "FSDP"
                else:
                    strategy = "DDP"
        
        # Base configuration
        config = {
            "model_family": model_family,
            "model_name": model_name,
            "distributed_strategy": strategy,
            "gpu_count": gpu_count,
            "per_gpu_batch_size": batch_size,
            "global_batch_size": batch_size * gpu_count,
            "mixed_precision": use_mixed_precision,
            "gradient_accumulation_steps": 1
        }
        
        # Calculate memory requirements
        # This is approximate and depends on many factors
        params_memory_gb = model_size_gb
        activations_memory_gb = model_size_gb * 0.5 * batch_size  # Rough estimate for activations
        optimizer_memory_gb = model_size_gb * 2  # Adam optimizer states

        total_memory_gb = params_memory_gb + activations_memory_gb + optimizer_memory_gb
        memory_per_gpu_gb = total_memory_gb / gpu_count

        # Add memory estimates
        config["estimated_memory"] = {
            "parameters_gb": params_memory_gb,
            "activations_gb": activations_memory_gb,
            "optimizer_gb": optimizer_memory_gb,
            "total_gb": total_memory_gb,
            "per_gpu_gb": memory_per_gpu_gb
        }
        
        # Check if we need memory optimization techniques
        if max_memory_gb and memory_per_gpu_gb > max_memory_gb:
            logger.info(f"Memory requirements ({memory_per_gpu_gb:.2f} GB) exceed available GPU memory ({max_memory_gb} GB)")
            
            # Calculate needed memory reduction
            reduction_factor = memory_per_gpu_gb / max_memory_gb
            
            # Apply memory optimization techniques based on reduction factor
            optimizations = []
            
            # 1. First, try gradient accumulation - increases effective batch size
            grad_accum_steps = max(1, int(np.ceil(reduction_factor)))
            if grad_accum_steps > 1:
                config["gradient_accumulation_steps"] = grad_accum_steps
                config["global_batch_size"] = batch_size * gpu_count * grad_accum_steps
                optimizations.append(f"Gradient accumulation (x{grad_accum_steps})")
                memory_per_gpu_gb = (params_memory_gb + (activations_memory_gb / grad_accum_steps) + optimizer_memory_gb) / gpu_count
            
            # 2. If still not enough, enable gradient checkpointing
            if memory_per_gpu_gb > max_memory_gb:
                config["gradient_checkpointing"] = True
                # Gradient checkpointing trades compute for memory by not storing intermediate activations
                # Can reduce activation memory by 3-5x at cost of ~20% slower training
                memory_per_gpu_gb = (params_memory_gb + (activations_memory_gb / (grad_accum_steps * 3)) + optimizer_memory_gb) / gpu_count
                optimizations.append("Gradient checkpointing")
            
            # 3. For DeepSpeed or FSDP, we can further optimize with special techniques
            if strategy == "DeepSpeed":
                config["deepspeed_config"] = {
                    "zero_stage": 2  # Stage 2 shards optimizer states
                }
                # ZeRO stage 2 reduces optimizer memory by ~8x
                memory_per_gpu_gb = (params_memory_gb + (activations_memory_gb / (grad_accum_steps * 3)) + (optimizer_memory_gb / 8)) / gpu_count
                optimizations.append("DeepSpeed ZeRO Stage 2")
                
                # If still not enough, use ZeRO stage 3
                if memory_per_gpu_gb > max_memory_gb:
                    config["deepspeed_config"]["zero_stage"] = 3  # Stage 3 also shards parameters and gradients
                    # ZeRO stage 3 reduces parameter memory by ~N times (where N is GPU count)
                    memory_per_gpu_gb = (params_memory_gb / gpu_count + (activations_memory_gb / (grad_accum_steps * 3)) + (optimizer_memory_gb / 8)) / gpu_count
                    optimizations.append("DeepSpeed ZeRO Stage 3")
            
            elif strategy == "FSDP":
                config["fsdp_config"] = {
                    "sharding_strategy": "FULL_SHARD"  # Shard params, grads, and optimizer states
                }
                # FSDP reduces memory by ~N times (N = GPU count) for parameters and optimizer states
                memory_per_gpu_gb = (params_memory_gb / gpu_count + (activations_memory_gb / (grad_accum_steps * 3)) + (optimizer_memory_gb / gpu_count)) / gpu_count
                optimizations.append("FSDP Full Sharding")
                
                # If still not enough, add activation checkpointing and CPU offloading
                if memory_per_gpu_gb > max_memory_gb:
                    config["fsdp_config"]["activation_checkpointing"] = True
                    config["fsdp_config"]["cpu_offload"] = True
                    # CPU offloading can further reduce memory by moving optimizer states to CPU
                    memory_per_gpu_gb = (params_memory_gb / gpu_count + (activations_memory_gb / (grad_accum_steps * 5)) + (optimizer_memory_gb / gpu_count / 2)) / gpu_count
                    optimizations.append("FSDP CPU Offloading")
            
            # 4. For very large models, suggest 8-bit optimizers if still needed
            if memory_per_gpu_gb > max_memory_gb:
                config["use_8bit_optimizer"] = True  # 8-bit Adam
                memory_per_gpu_gb = memory_per_gpu_gb * 0.75  # 8-bit optimizers save about 25% more memory
                optimizations.append("8-bit Optimizer")
            
            # Update memory estimates after optimizations
            config["estimated_memory"]["optimized_per_gpu_gb"] = memory_per_gpu_gb
            config["memory_optimizations"] = optimizations
            
            if memory_per_gpu_gb > max_memory_gb:
                logger.warning(f"Even with optimizations, memory requirements ({memory_per_gpu_gb:.2f} GB) exceed available GPU memory ({max_memory_gb} GB)")
                config["memory_warning"] = f"Training may fail due to insufficient GPU memory. Consider further reducing batch size or using more GPUs."
            else:
                logger.info(f"Memory optimizations applied: {', '.join(optimizations)}")
                logger.info(f"Estimated memory per GPU after optimizations: {memory_per_gpu_gb:.2f} GB")
        
        return config

    def select_hardware_for_training_benchmark(self,
                                              model_name: str,
                                              mode: str = "training",
                                              available_hardware: Optional[List[str]] = None,
                                              batch_sizes: Optional[List[int]] = None,
                                              include_distributed: bool = True,
                                              max_gpus: int = 8) -> Dict[str, Any]:
        """
        Select hardware configurations for a training benchmark suite.
        
        Args:
            model_name (str): Model name to benchmark.
            mode (str): "training" or "inference".
            available_hardware (Optional[List[str]]): Available hardware platforms.
            batch_sizes (Optional[List[int]]): Batch sizes to test.
            include_distributed (bool): Whether to include distributed configurations.
            max_gpus (int): Maximum number of GPUs to consider for distributed training.
            
        Returns:
            Dict[str, Any]: Benchmark configuration suggestions.
        """
        # Determine model family from model name
        model_family = self._determine_model_family(model_name)
        
        # Set default batch sizes based on model family if not provided
        if batch_sizes is None:
            if model_family == "text_generation":
                batch_sizes = [1, 4, 8]
            elif model_family == "embedding":
                batch_sizes = [8, 32, 128]
            elif model_family == "vision":
                batch_sizes = [8, 32, 64]
            elif model_family == "audio":
                batch_sizes = [4, 16, 32]
            elif model_family == "multimodal":
                batch_sizes = [2, 4, 8]
            else:
                batch_sizes = [1, 8, 32]
        
        # Set default hardware if not provided
        if available_hardware is None:
            available_hardware = ["cpu", "cuda", "rocm", "mps", "openvino"]
        
        # Build benchmark configurations
        config = {
            "model_name": model_name,
            "model_family": model_family,
            "mode": mode,
            "single_device": {},
            "distributed": {}
        }
        
        # Single device configurations
        for hardware in available_hardware:
            hardware_configs = []
            
            for batch_size in batch_sizes:
                # Select hardware for this batch size
                result = self.select_hardware(
                    model_family=model_family,
                    model_name=model_name,
                    batch_size=batch_size,
                    mode=mode,
                    available_hardware=[hardware]
                )
                
                # Skip if not compatible
                if hardware not in result.get("compatible_hardware", []):
                    continue
                
                # Create configuration for this batch size
                hardware_configs.append({
                    "batch_size": batch_size,
                    "hardware": hardware,
                    "estimated_score": result.get("all_scores", {}).get(hardware, {}).get("score", 0),
                    "mixed_precision": hardware != "cpu"  # Enable mixed precision for non-CPU hardware
                })
            
            if hardware_configs:
                config["single_device"][hardware] = hardware_configs
        
        # Distributed configurations
        if include_distributed and mode == "training":
            # Only consider CUDA and ROCm for distributed training
            for hardware in [hw for hw in available_hardware if hw in ["cuda", "rocm"]]:
                distributed_configs = []
                
                # Test with different numbers of GPUs
                for gpu_count in [2, 4, 8][:max_gpus//2]:  # Up to max_gpus/2
                    for batch_size in batch_sizes:
                        # Use a smaller batch size for multi-GPU to account for memory constraints
                        adjusted_batch_size = max(1, batch_size // 2) if gpu_count > 2 else batch_size
                        
                        # Create distributed config using helper method
                        distributed_config = self.get_distributed_training_config(
                            model_family=model_family,
                            model_name=model_name,
                            gpu_count=gpu_count,
                            batch_size=adjusted_batch_size
                        )
                        
                        # Add to configurations
                        distributed_configs.append({
                            "gpu_count": gpu_count,
                            "per_gpu_batch_size": adjusted_batch_size,
                            "global_batch_size": adjusted_batch_size * gpu_count,
                            "strategy": distributed_config["distributed_strategy"],
                            "hardware": hardware,
                            "mixed_precision": True,
                            "optimizations": distributed_config.get("memory_optimizations", []),
                            "estimated_memory_gb": distributed_config.get("estimated_memory", {}).get("per_gpu_gb", 0)
                        })
                
                if distributed_configs:
                    config["distributed"][hardware] = distributed_configs
        
        return config

    def _determine_model_family(self, model_name: str) -> str:
        """
        Determine model family based on model name.
        
        Args:
            model_name (str): Model name.
            
        Returns:
            str: Model family.
        """
        model_name = model_name.lower()
        
        if any(term in model_name for term in ["bert", "roberta", "distilbert", "electra", "albert", "mpnet"]):
            return "embedding"
        elif any(term in model_name for term in ["gpt", "llama", "llm", "opt", "bloom", "falcon", "mistral", "phi", "t5", "mt5"]):
            return "text_generation"
        elif any(term in model_name for term in ["vit", "resnet", "convnext", "deit", "beit", "swin"]):
            return "vision"
        elif any(term in model_name for term in ["whisper", "wav2vec", "hubert", "audio"]):
            return "audio"
        elif any(term in model_name for term in ["clip", "llava", "blip", "flava", "multimodal"]):
            return "multimodal"
        else:
            return "embedding"  # Default to embedding for unknown models


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hardware Selector")
    parser.add_argument("--database", type=str, default="./benchmark_results", help="Path to benchmark results database")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--model-family", type=str, help="Model family")
    parser.add_argument("--model-name", type=str, help="Model name")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--sequence-length", type=int, default=128, help="Sequence length")
    parser.add_argument("--mode", type=str, default="inference", choices=["inference", "training"], help="Mode (inference or training)")
    parser.add_argument("--hardware", type=str, nargs="+", help="Available hardware types")
    parser.add_argument("--task", type=str, help="Task type")
    parser.add_argument("--create-map", action="store_true", help="Create hardware selection map")
    parser.add_argument("--output", type=str, default="hardware_selection_map.json", help="Output file for selection map")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--distributed", action="store_true", help="Consider distributed training configurations")
    parser.add_argument("--gpu-count", type=int, default=1, help="Number of GPUs for distributed training")
    parser.add_argument("--training-benchmark", action="store_true", help="Generate training benchmark configurations")
    parser.add_argument("--max-memory-gb", type=int, help="Maximum GPU memory in GB")
    parser.add_argument("--max-gpus", type=int, default=8, help="Maximum number of GPUs to consider for distributed training")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.debug:
        logging.getLogger("hardware_selector").setLevel(logging.DEBUG)
    
    # Create hardware selector
    selector = HardwareSelector(
        database_path=args.database,
        config_path=args.config
    )
    
    if args.create_map:
        # Create and save hardware selection map
        selector.save_selection_map(args.output)
    elif args.training_benchmark and args.model_name:
        # Generate training benchmark configurations
        result = selector.select_hardware_for_training_benchmark(
            model_name=args.model_name,
            mode=args.mode,
            available_hardware=args.hardware,
            include_distributed=args.distributed,
            max_gpus=args.max_gpus
        )
        
        # Save output if specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Training benchmark configurations saved to {args.output}")
        
        # Print summary
        print(f"Training benchmark configuration for {args.model_name}:")
        print(f"Model family: {result['model_family']}")
        print(f"Mode: {result['mode']}")
        
        print("\nSingle device configurations:")
        for hw, configs in result['single_device'].items():
            print(f"  {hw}: {len(configs)} configurations")
            for i, config in enumerate(configs):
                print(f"    - Batch size: {config['batch_size']}, Mixed precision: {config['mixed_precision']}")
        
        if args.distributed and 'distributed' in result and result['distributed']:
            print("\nDistributed configurations:")
            for hw, configs in result['distributed'].items():
                print(f"  {hw}: {len(configs)} configurations")
                for i, config in enumerate(configs):
                    print(f"    - {config['gpu_count']} GPUs, Strategy: {config['strategy']}, Batch size: {config['per_gpu_batch_size']} (global: {config['global_batch_size']}), Est. memory: {config['estimated_memory_gb']:.2f} GB per GPU")
                    if 'optimizations' in config and config['optimizations']:
                        print(f"      Optimizations: {', '.join(config['optimizations'])}")
    elif args.model_family and args.model_name:
        # Select hardware for specific model
        # Check if distributed training mode
        if args.distributed and args.mode == "training":
            # Create a simple training config if needed
            training_config = None
            if args.max_memory_gb:
                training_config = {
                    "mixed_precision": True
                }
            
            result = selector.select_hardware_for_task(
                model_family=args.model_family,
                model_name=args.model_name,
                task_type=args.task or args.mode,
                batch_size=args.batch_size,
                sequence_length=args.sequence_length,
                available_hardware=args.hardware,
                distributed=True,
                gpu_count=args.gpu_count,
                training_config=training_config
            )
            
            # Print results
            print(f"Distributed hardware selection for {args.model_family}/{args.model_name}:")
            print(f"Primary recommendation: {result['primary_recommendation']}")
            print(f"Distributed strategy: {result.get('distributed_strategy', 'DDP')}")
            print(f"GPU count: {args.gpu_count}")
            print(f"Fallback options: {', '.join(result['fallback_options'])}")
            
            # If max memory was specified, also generate a detailed configuration
            if args.max_memory_gb:
                training_config = selector.get_distributed_training_config(
                    model_family=args.model_family,
                    model_name=args.model_name,
                    gpu_count=args.gpu_count,
                    batch_size=args.batch_size,
                    max_memory_gb=args.max_memory_gb
                )
                
                print("\nDetailed distributed training configuration:")
                print(f"Strategy: {training_config['distributed_strategy']}")
                print(f"Per-GPU batch size: {training_config['per_gpu_batch_size']}")
                print(f"Global batch size: {training_config['global_batch_size']}")
                print(f"Mixed precision: {training_config['mixed_precision']}")
                
                if "gradient_accumulation_steps" in training_config and training_config["gradient_accumulation_steps"] > 1:
                    print(f"Gradient accumulation steps: {training_config['gradient_accumulation_steps']}")
                
                if "gradient_checkpointing" in training_config and training_config["gradient_checkpointing"]:
                    print("Gradient checkpointing: Enabled")
                
                if "memory_optimizations" in training_config and training_config["memory_optimizations"]:
                    print(f"Memory optimizations: {', '.join(training_config['memory_optimizations'])}")
                
                print("\nMemory estimates:")
                memory_info = training_config.get("estimated_memory", {})
                print(f"Parameters: {memory_info.get('parameters_gb', 0):.2f} GB")
                print(f"Activations: {memory_info.get('activations_gb', 0):.2f} GB")
                print(f"Optimizer states: {memory_info.get('optimizer_gb', 0):.2f} GB")
                print(f"Total memory: {memory_info.get('total_gb', 0):.2f} GB")
                print(f"Per-GPU memory: {memory_info.get('per_gpu_gb', 0):.2f} GB")
                
                if "optimized_per_gpu_gb" in memory_info:
                    print(f"Optimized per-GPU memory: {memory_info['optimized_per_gpu_gb']:.2f} GB")
                
                if "memory_warning" in training_config:
                    print(f"\nWARNING: {training_config['memory_warning']}")
        else:
            # Normal (non-distributed) selection
            if args.task:
                result = selector.select_hardware_for_task(
                    model_family=args.model_family,
                    model_name=args.model_name,
                    task_type=args.task,
                    batch_size=args.batch_size,
                    sequence_length=args.sequence_length,
                    available_hardware=args.hardware
                )
            else:
                result = selector.select_hardware(
                    model_family=args.model_family,
                    model_name=args.model_name,
                    batch_size=args.batch_size,
                    sequence_length=args.sequence_length,
                    mode=args.mode,
                    available_hardware=args.hardware
                )
            
            # Print results
            print(f"Hardware selection for {args.model_family}/{args.model_name}:")
            print(f"Primary recommendation: {result['primary_recommendation']}")
            print(f"Fallback options: {', '.join(result['fallback_options'])}")
            print(f"Compatible hardware: {', '.join(result['compatible_hardware'])}")
            print("\nScores:")
            for hw_type, scores in result['all_scores'].items():
                print(f"  {hw_type}: {scores['score']:.4f} (latency: {scores['latency_score']:.4f}, throughput: {scores['throughput_score']:.4f})")
    else:
        parser.print_help()