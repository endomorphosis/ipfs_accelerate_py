#!/usr/bin/env python3
"""
Model Performance Predictor for the IPFS Accelerate framework.

This script implements ML-based performance prediction as part of Phase 16 of the
IPFS Accelerate Python framework project. The predictor is trained on collected
benchmark data and can predict performance for untested model-hardware combinations.

Key capabilities:
1. Predicts throughput, latency, and memory usage for unseen configurations
2. Supports all hardware platforms: CPU, CUDA, ROCm, MPS, OpenVINO, WebNN, WebGPU
3. Works with both inference and training mode
4. Handles both single-node and distributed training

Usage:
  python model_performance_predictor.py --train --database hardware_model_benchmark_db.parquet
  python model_performance_predictor.py --predict --model bert --hardware cuda --batch-size 32
  python model_performance_predictor.py --generate-matrix --output hardware_prediction_matrix.json
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
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_percentage_error, r2_score

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
    Preprocess benchmark data for training prediction models.
    
    Args:
        df (pd.DataFrame): Benchmark data
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Preprocessed data and preprocessing info
    """
    if df.empty:
        logger.error("Empty dataframe provided for preprocessing")
        return df, {}
    
    try:
        # Extract hardware platform from hardware column
        df["hardware_platform"] = df["hardware"].apply(
            lambda x: x.split("_")[0] if isinstance(x, str) and "_" in x else x
        )
        
        # Add distributed training flag
        df["is_distributed"] = df["hardware"].apply(
            lambda x: "distributed" in str(x)
        )
        
        # Add GPU count for distributed training
        df["gpu_count"] = df.apply(
            lambda row: row.get("total_gpus", 1) if row.get("is_distributed", False) else 1, 
            axis=1
        )
        
        # Convert precision to numeric (fp32=32, fp16=16, int8=8)
        df["precision_numeric"] = df["precision"].apply(
            lambda x: 32 if x == "fp32" else 16 if x == "fp16" else 8 if x == "int8" else 32
        )
        
        # Create feature matrix
        feature_cols = [
            "batch_size",
            "precision_numeric",
            "gpu_count",
            "is_distributed",
            "mode",
            "category",
            "hardware_platform"
        ]
        
        # Remove rows with missing values in feature columns
        for col in feature_cols:
            if col in df.columns:
                df = df.dropna(subset=[col])
        
        preprocessing_info = {
            "feature_columns": feature_cols,
            "numeric_columns": ["batch_size", "precision_numeric", "gpu_count"],
            "categorical_columns": ["is_distributed", "mode", "category", "hardware_platform"],
            "target_columns": PREDICTION_METRICS,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Preprocessed data with {len(df)} rows and {len(feature_cols)} features")
        return df, preprocessing_info
    
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        return df, {}

def train_prediction_models(
    df: pd.DataFrame, 
    preprocessing_info: Dict[str, Any],
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train prediction models for each performance metric.
    
    Args:
        df (pd.DataFrame): Preprocessed benchmark data
        preprocessing_info (Dict[str, Any]): Preprocessing information
        test_size (float): Fraction of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        Dict[str, Any]: Trained models and evaluation metrics
    """
    if df.empty:
        logger.error("Empty dataframe provided for training")
        return {}
    
    try:
        feature_cols = preprocessing_info["feature_columns"]
        numeric_cols = preprocessing_info["numeric_columns"]
        categorical_cols = preprocessing_info["categorical_columns"]
        target_cols = preprocessing_info["target_columns"]
        
        # Initialize models dictionary
        models = {}
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ],
            remainder='drop'
        )
        
        # Train model for each target metric
        for target in target_cols:
            if target not in df.columns:
                logger.warning(f"Target column {target} not found in data, skipping")
                continue
            
            # Prepare data
            X = df[feature_cols]
            y = df[target]
            
            # Skip if target has no variance
            if y.std() == 0:
                logger.warning(f"Target {target} has no variance, skipping")
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Create model pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=random_state
                ))
            ])
            
            # Train model
            logger.info(f"Training model for {target}...")
            pipeline.fit(X_train, y_train)
            
            # Evaluate model
            train_r2 = pipeline.score(X_train, y_train)
            test_r2 = pipeline.score(X_test, y_test)
            
            # Make predictions
            y_pred = pipeline.predict(X_test)
            
            # Calculate MAPE
            mape = mean_absolute_percentage_error(
                y_test[y_test > 0], 
                y_pred[y_test > 0]
            ) if (y_test > 0).any() else float('inf')
            
            # Store model and metrics
            models[target] = {
                "pipeline": pipeline,
                "metrics": {
                    "train_r2": float(train_r2),
                    "test_r2": float(test_r2),
                    "mape": float(mape),
                    "n_samples": len(df),
                    "feature_importance": dict(zip(
                        feature_cols, 
                        pipeline.named_steps['regressor'].feature_importances_
                    )) if hasattr(pipeline.named_steps['regressor'], 'feature_importances_') else {}
                }
            }
            
            logger.info(f"Model for {target}: R² = {test_r2:.4f}, MAPE = {mape:.2%}")
        
        models["preprocessing_info"] = preprocessing_info
        
        return models
    
    except Exception as e:
        logger.error(f"Error training prediction models: {e}")
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
    is_distributed: bool = False
) -> Dict[str, float]:
    """
    Predict performance for a model-hardware configuration.
    
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
        
    Returns:
        Dict[str, float]: Predicted performance metrics
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
        
        # Create input dataframe
        input_data = pd.DataFrame({
            "batch_size": [batch_size],
            "precision_numeric": [precision_numeric],
            "gpu_count": [gpu_count],
            "is_distributed": [is_distributed],
            "mode": [mode],
            "category": [model_category],
            "hardware_platform": [hardware_platform]
        })
        
        # Make predictions for each metric
        predictions = {}
        
        for target in PREDICTION_METRICS:
            if target not in models:
                logger.warning(f"No model found for {target}, skipping prediction")
                continue
                
            pipeline = models[target]["pipeline"]
            
            # Make prediction
            try:
                pred_value = float(pipeline.predict(input_data)[0])
                
                # Ensure predictions make sense
                if target == "throughput" and pred_value < 0:
                    pred_value = 0
                elif target == "latency_mean" and pred_value < 0:
                    pred_value = 1
                elif target == "memory_usage" and pred_value < 0:
                    pred_value = 1
                    
                predictions[target] = pred_value
            except Exception as e:
                logger.error(f"Error predicting {target}: {e}")
                predictions[target] = None
        
        # Add metadata
        predictions["model"] = model_name
        predictions["hardware"] = hardware
        predictions["batch_size"] = batch_size
        predictions["precision"] = precision
        predictions["mode"] = mode
        predictions["is_predicted"] = True
        predictions["timestamp"] = datetime.now().isoformat()
        
        return predictions
    
    except Exception as e:
        logger.error(f"Error predicting performance: {e}")
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
    parser = argparse.ArgumentParser(description="ML-based Performance Predictor")
    
    # Main options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Train prediction models")
    group.add_argument("--predict", action="store_true", help="Predict performance for a configuration")
    group.add_argument("--generate-matrix", action="store_true", help="Generate prediction matrix")
    group.add_argument("--visualize", action="store_true", help="Visualize predictions")
    
    # Training options
    parser.add_argument("--database", help="Path to benchmark database")
    parser.add_argument("--output-dir", help="Directory to save models")
    
    # Prediction options
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--category", help="Model category")
    parser.add_argument("--hardware", help="Hardware platform")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp32", help="Precision")
    parser.add_argument("--mode", choices=["inference", "training"], default="inference", help="Mode")
    
    # Matrix options
    parser.add_argument("--output", help="Output file for prediction matrix or visualization")
    parser.add_argument("--metric", choices=["throughput", "latency_mean", "memory_usage"], default="throughput", help="Metric to visualize")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(PREDICTOR_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if args.train:
        # Train prediction models
        print("Training prediction models...")
        
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
        models = train_prediction_models(df, preprocessing_info)
        if not models:
            print("Error training models, exiting")
            sys.exit(1)
        
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
        models = load_prediction_models()
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
            mode=args.mode
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
        print(f"Throughput: {prediction.get('throughput', 'N/A'):.2f} samples/sec")
        print(f"Latency: {prediction.get('latency_mean', 'N/A'):.2f} ms")
        print(f"Memory Usage: {prediction.get('memory_usage', 'N/A'):.2f} MB")
        
        # Save prediction
        if args.output:
            with open(args.output, "w") as f:
                json.dump(prediction, f, indent=2)
            print(f"Prediction saved to {args.output}")
    
    elif args.generate_matrix:
        # Generate prediction matrix
        # Load models
        models = load_prediction_models()
        if not models:
            print("No prediction models available, exiting")
            sys.exit(1)
        
        # Generate matrix
        matrix = generate_prediction_matrix(
            models=models,
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
            metric=args.metric
        )
        
        if not visualization_files:
            print("Error creating visualizations, exiting")
            sys.exit(1)
        
        print(f"Created {len(visualization_files)} visualizations:")
        for vis_file in visualization_files:
            print(f"- {vis_file}")

if __name__ == "__main__":
    main()