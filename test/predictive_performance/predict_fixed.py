#!/usr/bin/env python3
"""
Predictive Performance System Prediction Module

This module makes predictions for model-hardware configurations using the trained 
ML models. It can make individual predictions, generate prediction matrices for 
multiple configurations, and visualize prediction results.

Usage:
    # Make a single prediction
    python predict.py --model-dir ./models --model bert-base-uncased --hardware cuda --batch-size 8
    
    # Generate prediction matrix for multiple configurations
    python predict.py --model-dir ./models --generate-matrix --output matrix.json
    
    # Visualize predictions from a matrix
    python predict.py --model-dir ./models --visualize --matrix-file matrix.json --output-dir ./visualizations
    """

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List, Union

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model_performance_predictor module
try:
    from model_performance_predictor import (
        load_prediction_models,
        predict_performance,
        generate_prediction_matrix,
        visualize_predictions,
        PREDICTION_METRICS,
        MODEL_CATEGORIES,
        HARDWARE_CATEGORIES
    )
    MODELS_AVAILABLE = True
except ImportError:
    logger = logging.getLogger("predictive_performance.predict")
    logger.warning("model_performance_predictor module not available, using simulation mode")
    MODELS_AVAILABLE = False
    
    # Define constants for simulation mode
    PREDICTION_METRICS = ["throughput", "latency", "memory", "power"]
    MODEL_CATEGORIES = ["text_embedding", "text_generation", "vision", "audio", "multimodal"]
    HARDWARE_CATEGORIES = ["cpu", "cuda", "rocm", "mps", "openvino", "qnn", "webnn", "webgpu"]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("predictive_performance.predict")

# Default paths
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_DIR = PROJECT_ROOT
PREDICTIVE_DIR = TEST_DIR / "predictive_performance"
MODELS_DIR = PREDICTIVE_DIR / "models" / "trained_models" / "latest"
OUTPUT_DIR = PREDICTIVE_DIR / "predictions"
VISUALIZATIONS_DIR = PREDICTIVE_DIR / "visualizations"

def make_prediction(
    model_dir: Optional[str] = None,
    model_name: str = "",
    model_category: str = "",
    hardware: str = "",
    batch_size: int = 1,
    precision: str = "fp32",
    mode: str = "inference",
    gpu_count: int = 1,
    is_distributed: bool = False,
    sequence_length: int = 128,
    output_file: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Make a performance prediction for a specific configuration.
    
    Args:
        model_dir (str): Directory containing trained models
        model_name (str): Name of the model
        model_category (str): Category of the model
        hardware (str): Hardware platform
        batch_size (int): Batch size
        precision (str): Precision (fp32, fp16, int8)
        mode (str): Mode (inference, training)
        gpu_count (int): Number of GPUs (for distributed setups)
        is_distributed (bool): Whether this is a distributed setup
        sequence_length (int): Sequence length for text models
        output_file (str): Path to output file
        
    Returns:
        Tuple[bool, Dict[str, Any]]: Success flag and prediction result
    """
    try:
        # Set default model directory if not provided
        if model_dir is None:
            model_dir = MODELS_DIR
        
        # Load prediction models
        logger.info(f"Loading prediction models from {model_dir}")
        models = load_prediction_models(model_dir)
        
        if not models:
            logger.error(f"Failed to load models from {model_dir}")
            return False, {}
        
        # Check if required parameters are provided
        if not model_name:
            logger.error("Model name is required")
            return False, {}
        
        # Infer model category if not provided
        if not model_category:
            model_category = _infer_model_category(model_name)
            logger.info(f"Inferred model category: {model_category}")
        
        if not hardware:
            logger.error("Hardware platform is required")
            return False, {}
        
        if not batch_size:
            logger.error("Batch size is required")
            return False, {}
        
        # Make prediction
        logger.info(f"Making prediction for {model_name} on {hardware} with batch size {batch_size}")
        
        prediction = predict_performance(
            models=models,
            model_name=model_name,
            model_category=model_category,
            hardware=hardware,
            batch_size=batch_size,
            precision=precision,
            mode=mode,
            gpu_count=gpu_count,
            is_distributed=is_distributed,
            sequence_length=sequence_length,
            calculate_uncertainty=True
        )
        
        if not prediction:
            logger.error("Failed to make prediction")
            return False, {}
        
        # Add timestamp and request info
        prediction["request_timestamp"] = datetime.now().isoformat()
        prediction["request_info"] = {
            "model_dir": str(model_dir),
            "model_name": model_name,
            "model_category": model_category,
            "hardware": hardware,
            "batch_size": batch_size,
            "precision": precision,
            "mode": mode,
            "gpu_count": gpu_count,
            "is_distributed": is_distributed,
            "sequence_length": sequence_length
        }
        
        # Save prediction to file if output_file is provided
        if output_file:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
            # Save prediction
            with open(output_file, 'w') as f:
                json.dump(prediction, f, indent=2)
            
            logger.info(f"Prediction saved to {output_file}")
        
        return True, prediction
    
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, {}

def generate_matrix(
    model_dir: Optional[str] = None,
    model_configs: Optional[List[Dict[str, Any]]] = None,
    hardware_platforms: Optional[List[str]] = None,
    batch_sizes: Optional[List[int]] = None,
    precision_options: Optional[List[str]] = None,
    mode: str = "inference",
    output_file: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Generate a prediction matrix for multiple configurations.
    
    Args:
        model_dir (str): Directory containing trained models
        model_configs (List[Dict[str, Any]]): List of model configurations
        hardware_platforms (List[str]): List of hardware platforms
        batch_sizes (List[int]): List of batch sizes
        precision_options (List[str]): List of precision options
        mode (str): Mode (inference, training)
        output_file (str): Path to output file
        
    Returns:
        Tuple[bool, Dict[str, Any]]: Success flag and prediction matrix
    """
    try:
        # Set default model directory if not provided
        if model_dir is None:
            model_dir = MODELS_DIR
        
        # Load prediction models
        logger.info(f"Loading prediction models from {model_dir}")
        models = load_prediction_models(model_dir)
        
        if not models:
            logger.error(f"Failed to load models from {model_dir}")
            return False, {}
        
        # Set default model configs if not provided
        if model_configs is None:
            model_configs = [
                {"name": "bert-base-uncased", "category": "text_embedding"},
                {"name": "t5-small", "category": "text_generation"},
                {"name": "facebook/opt-125m", "category": "text_generation"},
                {"name": "openai/whisper-tiny", "category": "audio"},
                {"name": "google/vit-base-patch16-224", "category": "vision"},
                {"name": "openai/clip-vit-base-patch32", "category": "multimodal"}
            ]
        
        # Set default hardware platforms if not provided
        if hardware_platforms is None:
            hardware_platforms = ["cpu", "cuda", "mps", "openvino", "webnn", "webgpu"]
        
        # Set default batch sizes if not provided
        if batch_sizes is None:
            batch_sizes = [1, 8, 32]
        
        # Set default precision options if not provided
        if precision_options is None:
            precision_options = ["fp32", "fp16"]
        
        # Generate matrix
        logger.info("Generating prediction matrix")
        logger.info(f"Models: {[m['name'] for m in model_configs]}")
        logger.info(f"Hardware platforms: {hardware_platforms}")
        logger.info(f"Batch sizes: {batch_sizes}")
        logger.info(f"Precision options: {precision_options}")
        
        matrix = generate_prediction_matrix(
            models=models,
            model_configs=model_configs,
            hardware_platforms=hardware_platforms,
            batch_sizes=batch_sizes,
            precision_options=precision_options,
            mode=mode,
            output_file=output_file
        )
        
        if not matrix:
            logger.error("Failed to generate prediction matrix")
            return False, {}
        
        # Add generation info
        matrix["generation_info"] = {
            "model_dir": str(model_dir),
            "timestamp": datetime.now().isoformat(),
            "n_models": len(model_configs),
            "n_hardware": len(hardware_platforms),
            "n_batch_sizes": len(batch_sizes),
            "n_precisions": len(precision_options),
            "mode": mode
        }
        
        # Save matrix to file if output_file is provided
        if output_file:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
            # Save matrix
            with open(output_file, 'w') as f:
                json.dump(matrix, f, indent=2)
            
            logger.info(f"Prediction matrix saved to {output_file}")
        
        return True, matrix
    
    except Exception as e:
        logger.error(f"Error generating prediction matrix: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, {}

def visualize_matrix(
    matrix_file: str,
    metric: str = "throughput",
    output_dir: Optional[str] = None,
    format: str = "png"
) -> Tuple[bool, List[str]]:
    """
    Visualize predictions from a matrix.
    
    Args:
        matrix_file (str): Path to matrix file
        metric (str): Metric to visualize
        output_dir (str): Directory to save visualizations
        format (str): Output format
        
    Returns:
        Tuple[bool, List[str]]: Success flag and list of visualization files
    """
    try:
        # Check if matrix file exists
        if not os.path.exists(matrix_file):
            logger.error(f"Matrix file not found: {matrix_file}")
            return False, []
        
        # Load matrix
        with open(matrix_file, 'r') as f:
            matrix = json.load(f)
        
        # Set default output directory if not provided
        if output_dir is None:
            output_dir = VISUALIZATIONS_DIR
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Visualize predictions
        logger.info(f"Visualizing {metric} from matrix")
        
        visualization_files = visualize_predictions(
            matrix=matrix,
            metric=metric,
            output_dir=output_dir
        )
        
        if not visualization_files:
            logger.error("Failed to create visualizations")
            return False, []
        
        logger.info(f"Generated {len(visualization_files)} visualization files")
        
        return True, visualization_files
    
    except Exception as e:
        logger.error(f"Error visualizing matrix: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, []

def _infer_model_category(model_name: str) -> str:
    """
    Infer model category from model name.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        str: Inferred model category
    """
    model_lower = model_name.lower()
    
    # Check for vision models
    if any(kw in model_lower for kw in ['vit', 'resnet', 'swin', 'deit', 'convnext']):
        return "vision"
    
    # Check for text generation models
    if any(kw in model_lower for kw in ['gpt', 't5', 'llama', 'opt', 'falcon', 'bloom']):
        return "text_generation"
    
    # Check for text embedding models
    if any(kw in model_lower for kw in ['bert', 'roberta', 'electra', 'deberta', 'albert']):
        return "text_embedding"
    
    # Check for audio models
    if any(kw in model_lower for kw in ['whisper', 'wav2vec', 'clap', 'hubert']):
        return "audio"
    
    # Check for multimodal models
    if any(kw in model_lower for kw in ['clip', 'flava', 'blip', 'llava']):
        return "multimodal"
    
    # Default to text embedding if unknown
    return "text_embedding"

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Predictive Performance System Prediction Module")
    
    # Model directory
    parser.add_argument("--model-dir", help="Directory containing trained models")
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    
    # Single prediction parser
    predict_parser = subparsers.add_parser("predict", help="Make a single prediction")
    predict_parser.add_argument("--model", required=True, help="Model name")
    predict_parser.add_argument("--category", help="Model category")
    predict_parser.add_argument("--hardware", required=True, help="Hardware platform")
    predict_parser.add_argument("--batch-size", type=int, required=True, help="Batch size")
    predict_parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp32", help="Precision")
    predict_parser.add_argument("--mode", choices=["inference", "training"], default="inference", help="Mode")
    predict_parser.add_argument("--gpu-count", type=int, default=1, help="Number of GPUs")
    predict_parser.add_argument("--distributed", action="store_true", help="Distributed setup")
    predict_parser.add_argument("--sequence-length", type=int, default=128, help="Sequence length")
    predict_parser.add_argument("--output", help="Output file")
    
    # Matrix generation parser
    matrix_parser = subparsers.add_parser("matrix", help="Generate prediction matrix")
    matrix_parser.add_argument("--models", help="Comma-separated list of models")
    matrix_parser.add_argument("--categories", help="Comma-separated list of model categories")
    matrix_parser.add_argument("--hardware", help="Comma-separated list of hardware platforms")
    matrix_parser.add_argument("--batch-sizes", help="Comma-separated list of batch sizes")
    matrix_parser.add_argument("--precisions", help="Comma-separated list of precision options")
    matrix_parser.add_argument("--inference-mode", choices=["inference", "training"], default="inference", help="Mode")
    matrix_parser.add_argument("--output", required=True, help="Output file")
    
    # Visualization parser
    vis_parser = subparsers.add_parser("visualize", help="Visualize predictions")
    vis_parser.add_argument("--matrix-file", required=True, help="Matrix file")
    vis_parser.add_argument("--metric", choices=["throughput", "latency_mean", "memory_usage"], default="throughput", help="Metric to visualize")
    vis_parser.add_argument("--output-dir", help="Output directory")
    vis_parser.add_argument("--format", choices=["png", "svg", "pdf"], default="png", help="Output format")
    
    # Add common arguments
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    # For backwards compatibility with simple command line interface
    parser.add_argument("--model", help="Model name (for predict mode)")
    parser.add_argument("--hardware", help="Hardware platform (for predict mode)")
    parser.add_argument("--batch-size", type=int, help="Batch size (for predict mode)")
    parser.add_argument("--generate-matrix", action="store_true", help="Generate prediction matrix")
    parser.add_argument("--visualize", action="store_true", help="Visualize predictions")
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--matrix-file", help="Matrix file (for visualize mode)")
    parser.add_argument("--metric", choices=["throughput", "latency_mean", "memory_usage"], help="Metric to visualize (for visualize mode)")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine mode of operation based on args
    if args.mode == "predict":
        # Use subparser arguments
        success, prediction = make_prediction(
            model_dir=args.model_dir,
            model_name=args.model,
            model_category=args.category,
            hardware=args.hardware,
            batch_size=args.batch_size,
            precision=args.precision,
            mode=args.mode,
            gpu_count=args.gpu_count,
            is_distributed=args.distributed,
            sequence_length=args.sequence_length,
            output_file=args.output
        )
        
        if not success:
            sys.exit(1)
        
        # Print prediction
        print("\nPerformance Prediction:")
        print(f"Model: {args.model}")
        print(f"Hardware: {args.hardware}")
        print(f"Batch Size: {args.batch_size}")
        
        # Print metrics with confidence
        for metric in PREDICTION_METRICS:
            if metric in prediction.get("predictions", {}):
                value = prediction["predictions"][metric]
                
                if metric in prediction.get("uncertainties", {}):
                    uncertainty = prediction["uncertainties"][metric]
                    confidence = uncertainty.get("confidence", 0.0) * 100
                    lower = uncertainty.get("lower_bound", 0.0)
                    upper = uncertainty.get("upper_bound", 0.0)
                    
                    if metric == "throughput":
                        print(f"Throughput: {value:.2f} samples/sec (confidence: {confidence:.1f}%)")
                        print(f"  Range: {lower:.2f} - {upper:.2f} samples/sec")
                    elif metric == "latency_mean":
                        print(f"Latency: {value:.2f} ms (confidence: {confidence:.1f}%)")
                        print(f"  Range: {lower:.2f} - {upper:.2f} ms")
                    elif metric == "memory_usage":
                        print(f"Memory Usage: {value:.2f} MB (confidence: {confidence:.1f}%)")
                        print(f"  Range: {lower:.2f} - {upper:.2f} MB")
                else:
                    if metric == "throughput":
                        print(f"Throughput: {value:.2f} samples/sec")
                    elif metric == "latency_mean":
                        print(f"Latency: {value:.2f} ms")
                    elif metric == "memory_usage":
                        print(f"Memory Usage: {value:.2f} MB")
        
        # Print overall confidence
        print(f"Overall Confidence: {prediction.get('confidence_score', 0.0) * 100:.1f}%")
        
        # Print explanations if any
        if "explanation" in prediction and prediction["explanation"]:
            print("\nExplanations:")
            for explanation in prediction["explanation"]:
                print(f"- {explanation}")
    
    elif args.mode == "matrix":
        # Parse lists of models, categories, hardware, batch sizes, and precisions
        models = []
        
        if args.models:
            model_names = [m.strip() for m in args.models.split(",")]
            categories = [c.strip() for c in args.categories.split(",")] if args.categories else []
            
            # If categories provided, ensure same length as models
            if categories and len(categories) != len(model_names):
                if len(categories) == 1:
                    # Use same category for all models
                    categories = [categories[0]] * len(model_names)
                else:
                    logger.error("Number of categories must match number of models")
                    sys.exit(1)
            
            # If categories not provided, infer them
            if not categories:
                categories = [_infer_model_category(m) for m in model_names]
            
            # Create model configs
            for i, model_name in enumerate(model_names):
                models.append({
                    "name": model_name,
                    "category": categories[i]
                })
        
        hardware_platforms = [h.strip() for h in args.hardware.split(",")] if args.hardware else None
        batch_sizes = [int(bs.strip()) for bs in args.batch_sizes.split(",")] if args.batch_sizes else None
        precision_options = [p.strip() for p in args.precisions.split(",")] if args.precisions else None
        
        # Generate matrix
        success, matrix = generate_matrix(
            model_dir=args.model_dir,
            model_configs=models if models else None,
            hardware_platforms=hardware_platforms,
            batch_sizes=batch_sizes,
            precision_options=precision_options,
            mode=args.inference_mode,
            output_file=args.output
        )
        
        if not success:
            sys.exit(1)
        
        # Print summary
        print("\nPrediction Matrix Summary:")
        print(f"Models: {len(matrix.get('models', {}))}")
        print(f"Hardware Platforms: {len(matrix.get('hardware_platforms', []))}")
        print(f"Batch Sizes: {matrix.get('batch_sizes', [])}")
        print(f"Precision Options: {matrix.get('precision_options', [])}")
        print(f"Mode: {matrix.get('mode', 'inference')}")
        
        if args.output:
            print(f"\nMatrix saved to: {args.output}")
            print("\nTo visualize the matrix, run:")
            print(f"python predict.py visualize --matrix-file {args.output} --metric throughput --output-dir ./visualizations")
    
    elif args.mode == "visualize":
        # Visualize predictions
        success, visualization_files = visualize_matrix(
            matrix_file=args.matrix_file,
            metric=args.metric,
            output_dir=args.output_dir,
            format=args.format
        )
        
        if not success:
            sys.exit(1)
        
        # Print summary
        print("\nVisualization Summary:")
        print(f"Matrix File: {args.matrix_file}")
        print(f"Metric: {args.metric}")
        print(f"Generated {len(visualization_files)} visualization files:")
        
        for vis_file in visualization_files:
            print(f"- {vis_file}")
    
    else:
        # Backwards compatibility mode
        if args.model and args.hardware and args.batch_size:
            # Make prediction
            success, prediction = make_prediction(
                model_dir=args.model_dir,
                model_name=args.model,
                hardware=args.hardware,
                batch_size=args.batch_size,
                output_file=args.output
            )
            
            if not success:
                sys.exit(1)
            
            # Print prediction
            print("\nPerformance Prediction:")
            print(f"Model: {args.model}")
            print(f"Hardware: {args.hardware}")
            print(f"Batch Size: {args.batch_size}")
            
            # Print metrics
            for metric in PREDICTION_METRICS:
                if metric in prediction.get("predictions", {}):
                    value = prediction["predictions"][metric]
                    
                    if metric == "throughput":
                        print(f"Throughput: {value:.2f} samples/sec")
                    elif metric == "latency_mean":
                        print(f"Latency: {value:.2f} ms")
                    elif metric == "memory_usage":
                        print(f"Memory Usage: {value:.2f} MB")
            
            # Print overall confidence
            print(f"Overall Confidence: {prediction.get('confidence_score', 0.0) * 100:.1f}%")
        
        elif args.generate_matrix:
            # Generate matrix
            success, matrix = generate_matrix(
                model_dir=args.model_dir,
                output_file=args.output
            )
            
            if not success:
                sys.exit(1)
            
            # Print summary
            print("\nPrediction Matrix Summary:")
            print(f"Models: {len(matrix.get('models', {}))}")
            print(f"Hardware Platforms: {len(matrix.get('hardware_platforms', []))}")
            print(f"Batch Sizes: {matrix.get('batch_sizes', [])}")
            print(f"Precision Options: {matrix.get('precision_options', [])}")
            
            if args.output:
                print(f"\nMatrix saved to: {args.output}")
        
        elif args.visualize:
            if not args.matrix_file:
                logger.error("Matrix file required for visualization")
                sys.exit(1)
            
            # Visualize predictions
            metric = args.metric or "throughput"
            success, visualization_files = visualize_matrix(
                matrix_file=args.matrix_file,
                metric=metric,
                output_dir=args.output_dir
            )
            
            if not success:
                sys.exit(1)
            
            # Print summary
            print("\nVisualization Summary:")
            print(f"Matrix File: {args.matrix_file}")
            print(f"Metric: {metric}")
            print(f"Generated {len(visualization_files)} visualization files:")
            
            for vis_file in visualization_files:
                print(f"- {vis_file}")
        
        else:
            # Print help
            parser.print_help()
            sys.exit(1)

class PerformancePredictor:
    """
    Performance Predictor for predicting model-hardware performance metrics.
    
    This class provides an easy-to-use interface for making predictions about
    model performance on various hardware platforms using ML-based prediction models.
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize the performance predictor.
        
        Args:
            model_dir: Directory containing trained prediction models
        """
        self.model_dir = model_dir or MODELS_DIR
        self.models = {}
        
        # Try to load models if available
        if MODELS_AVAILABLE:
            try:
                self.models = load_prediction_models(self.model_dir)
                if self.models:
                    logger.info(f"Loaded {len(self.models)} prediction models")
                else:
                    logger.warning("No prediction models found, using simulation mode")
            except Exception as e:
                logger.warning(f"Failed to load prediction models: {e}")
        
        # Hardware performance characteristics (for simulation mode)
        self.hardware_performance = {
            # Relative performance values for simulation mode
            "cpu": {"throughput_factor": 1.0, "latency_factor": 1.0, "memory_factor": 1.0, "power_factor": 1.0},
            "cuda": {"throughput_factor": 8.0, "latency_factor": 0.2, "memory_factor": 1.2, "power_factor": 3.0},
            "rocm": {"throughput_factor": 7.5, "latency_factor": 0.25, "memory_factor": 1.2, "power_factor": 2.8},
            "mps": {"throughput_factor": 5.0, "latency_factor": 0.3, "memory_factor": 1.1, "power_factor": 2.2},
            "openvino": {"throughput_factor": 3.5, "latency_factor": 0.4, "memory_factor": 0.8, "power_factor": 1.5},
            "qnn": {"throughput_factor": 2.5, "latency_factor": 0.5, "memory_factor": 0.7, "power_factor": 0.8},
            "webnn": {"throughput_factor": 2.0, "latency_factor": 0.6, "memory_factor": 0.9, "power_factor": 1.0},
            "webgpu": {"throughput_factor": 3.0, "latency_factor": 0.5, "memory_factor": 1.0, "power_factor": 1.2},
        }
        
        # Model type characteristics (for simulation mode)
        self.model_type_factors = {
            "text_embedding": {"base_throughput": 200, "base_latency": 10, "base_memory": 1024, "base_power": 50},
            "text_generation": {"base_throughput": 20, "base_latency": 100, "base_memory": 4096, "base_power": 150},
            "vision": {"base_throughput": 50, "base_latency": 30, "base_memory": 2048, "base_power": 100},
            "audio": {"base_throughput": 10, "base_latency": 200, "base_memory": 3072, "base_power": 120},
            "multimodal": {"base_throughput": 5, "base_latency": 300, "base_memory": 6144, "base_power": 180},
        }
        
        # Model size lookup (for simulation mode)
        self.model_sizes = {
            "bert-base-uncased": {"size_factor": 1.0, "type": "text_embedding"},
            "bert-tiny": {"size_factor": 0.2, "type": "text_embedding"},
            "prajjwal1/bert-tiny": {"size_factor": 0.2, "type": "text_embedding"},
            "t5-small": {"size_factor": 0.8, "type": "text_generation"},
            "t5-efficient-tiny": {"size_factor": 0.3, "type": "text_generation"},
            "whisper-tiny": {"size_factor": 0.5, "type": "audio"},
            "llama-7b": {"size_factor": 3.0, "type": "text_generation"},
            "vit-base": {"size_factor": 1.0, "type": "vision"},
            "clip-vit-base": {"size_factor": 1.2, "type": "multimodal"},
        }
    
    def predict(self, model_name: str, model_type: str, hardware_platform: str, batch_size: int = 1,
               precision: str = "fp32", sequence_length: int = 128) -> Dict[str, Any]:
        """
        Predict performance metrics for a model on a specific hardware platform.
        
        Args:
            model_name: Name of the model
            model_type: Type/category of the model
            hardware_platform: Hardware platform
            batch_size: Batch size
            precision: Precision format (fp32, fp16, int8)
            sequence_length: Sequence length for text models
            
        Returns:
            Dictionary containing predicted metrics and confidence scores
        """
        # Use real prediction model if available
        if MODELS_AVAILABLE and self.models:
            success, prediction = make_prediction(
                model_dir=self.model_dir,
                model_name=model_name,
                model_category=model_type,
                hardware=hardware_platform,
                batch_size=batch_size,
                precision=precision,
                sequence_length=sequence_length
            )
            
            if success:
                return prediction
            
            logger.warning("Real prediction failed, falling back to simulation mode")
        
        # Simulation mode - generate reasonable predictions based on hardware and model characteristics
        return self._simulate_prediction(model_name, model_type, hardware_platform, batch_size, precision)
    
    def _simulate_prediction(self, model_name: str, model_type: str, hardware: str,
                           batch_size: int, precision: str) -> Dict[str, Any]:
        """Simulate a prediction when real models aren't available."""
        # Get model info, with fallbacks
        model_info = self.model_sizes.get(model_name, {"size_factor": 1.0, "type": model_type})
        if not model_type:
            model_type = model_info["type"]
            
        # Get base metrics for this type of model
        model_base = self.model_type_factors.get(model_type, self.model_type_factors["text_embedding"])
        
        # Get hardware performance factors
        hw_factors = self.hardware_performance.get(hardware, self.hardware_performance["cpu"])
        
        # Calculate size factor based on model
        size_factor = model_info["size_factor"]
        
        # Calculate precision factor
        precision_factors = {"fp32": 1.0, "fp16": 1.5, "int8": 2.0, "int4": 2.5}
        precision_factor = precision_factors.get(precision, 1.0)
        
        # Calculate batch factor (non-linear scaling with diminishing returns)
        batch_factor = batch_size ** 0.7
        
        # Calculate metrics
        throughput = (model_base["base_throughput"] * hw_factors["throughput_factor"] *
                    precision_factor / size_factor * batch_factor)
        
        latency = (model_base["base_latency"] * hw_factors["latency_factor"] *
                 size_factor / precision_factor * (1 + 0.1 * batch_size))
        
        memory = (model_base["base_memory"] * hw_factors["memory_factor"] *
                size_factor / (precision_factors[precision] ** 0.5) *
                (1 + 0.2 * (batch_size - 1)))
        
        power = model_base["base_power"] * hw_factors["power_factor"] * (1 + 0.1 * batch_size)
        
        # Add random variation to make it more realistic
        import random
        random.seed(hash(f"{model_name}_{model_type}_{hardware}_{batch_size}_{precision}"))
        
        variation = 0.15  # 15% random variation
        throughput *= random.uniform(1 - variation, 1 + variation)
        latency *= random.uniform(1 - variation, 1 + variation)
        memory *= random.uniform(1 - variation, 1 + variation)
        power *= random.uniform(1 - variation, 1 + variation)
        
        # Calculate confidence scores (simulated)
        base_confidence = 0.92  # Base confidence value
        confidence_variation = 0.05
        confidence = base_confidence * random.uniform(1 - confidence_variation, 1 + confidence_variation)
        confidence_latency = base_confidence * random.uniform(1 - confidence_variation, 1 + confidence_variation)
        confidence_memory = base_confidence * random.uniform(1 - confidence_variation, 1 + confidence_variation)
        confidence_power = base_confidence * random.uniform(1 - confidence_variation, 1 + confidence_variation)
        
        # Create prediction result
        result = {
            "predictions": {
                "throughput": throughput,
                "latency_mean": latency,
                "memory_usage": memory,
                "power_consumption": power
            },
            "confidence_score": confidence,
            "uncertainties": {
                "throughput": {
                    "confidence": confidence,
                    "lower_bound": throughput * 0.9,
                    "upper_bound": throughput * 1.1
                },
                "latency_mean": {
                    "confidence": confidence_latency,
                    "lower_bound": latency * 0.9,
                    "upper_bound": latency * 1.1
                },
                "memory_usage": {
                    "confidence": confidence_memory,
                    "lower_bound": memory * 0.9,
                    "upper_bound": memory * 1.1
                },
                "power_consumption": {
                    "confidence": confidence_power,
                    "lower_bound": power * 0.9,
                    "upper_bound": power * 1.1
                }
            },
            "request_timestamp": datetime.now().isoformat(),
            "request_info": {
                "model_name": model_name,
                "model_type": model_type,
                "hardware": hardware,
                "batch_size": batch_size,
                "precision": precision,
                "simulation_mode": True
            }
        }
        
        return result
    
    def visualize_hardware_comparison(self, model_name: str, model_type: str, batch_size: int,
                                     output_file: str = "hardware_comparison.png"):
        """Generate a comparison chart of hardware platforms for a specific model."""
        # Get predictions for all hardware platforms
        hardware_platforms = ["cpu", "cuda", "rocm", "mps", "openvino", "qnn", "webnn", "webgpu"]
        results = {}
        
        for hw in hardware_platforms:
            prediction = self.predict(model_name, model_type, hw, batch_size)
            results[hw] = prediction
        
        # Prepare data for visualization
        throughputs = [results[hw]["predictions"]["throughput"] for hw in hardware_platforms]
        latencies = [results[hw]["predictions"]["latency_mean"] for hw in hardware_platforms]
        
        # Create visualization
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Throughput chart
        ax1.bar(hardware_platforms, throughputs, color='skyblue')
        ax1.set_title(f"Throughput Comparison - {model_name}")
        ax1.set_xlabel("Hardware Platform")
        ax1.set_ylabel("Throughput (items/second)")
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.set_ylim(bottom=0)
        
        # Latency chart
        ax2.bar(hardware_platforms, latencies, color='salmon')
        ax2.set_title(f"Latency Comparison - {model_name}")
        ax2.set_xlabel("Hardware Platform")
        ax2.set_ylabel("Latency (ms)")
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        ax2.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        
        return output_file

if __name__ == "__main__":
    main()