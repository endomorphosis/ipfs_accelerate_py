#!/usr/bin/env python
"""
Hardware Model Predictor for the IPFS Accelerate framework.

This module provides integration between hardware detection, model performance prediction,
and hardware selection systems to create a comprehensive decision-making system for
model deployment. It combines the functionalities of model_performance_predictor.py
and hardware_selector.py to provide accurate hardware recommendations based on
benchmarking data.

Part of Phase 16 of the IPFS Accelerate project.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Attempt to import advanced components
try:
    from hardware_selector import HardwareSelector
    HARDWARE_SELECTOR_AVAILABLE = True
    logger.info("Hardware selector module available")
except ImportError:
    HARDWARE_SELECTOR_AVAILABLE = False
    logger.warning("Hardware selector module not available, falling back to basic hardware selection")

try:
    from model_performance_predictor import predict_performance, load_prediction_models, generate_prediction_matrix, visualize_predictions
    PREDICTOR_AVAILABLE = True
    logger.info("Model performance predictor module available")
except ImportError:
    PREDICTOR_AVAILABLE = False
    logger.warning("Model performance predictor module not available, falling back to heuristic predictions")

# Try to import hardware-model integration
try:
    from hardware_model_integration import (
        MODEL_FAMILY_DEVICE_PREFERENCES,
        MODEL_FAMILY_MEMORY_REQUIREMENTS,
        MODEL_FAMILY_HARDWARE_COMPATIBILITY,
        get_model_size_tier,
        check_hardware_compatibility,
        estimate_memory_requirements,
        detect_hardware_availability,
        classify_model_family,
        select_optimal_device,
        integrate_hardware_and_model
    )
    INTEGRATION_AVAILABLE = True
    logger.info("Hardware-model integration module available")
except ImportError:
    INTEGRATION_AVAILABLE = False
    logger.warning("Hardware-model integration module not available")

class HardwareModelPredictor:
    """Main class for hardware model prediction."""
    
    def __init__(self, 
                 benchmark_dir: str = "./benchmark_results",
                 database_path: Optional[str] = None,
                 config_path: Optional[str] = None):
        """
        Initialize the hardware model predictor.
        
        Args:
            benchmark_dir: Path to benchmark results directory
            database_path: Optional path to benchmark database
            config_path: Optional path to configuration file
        """
        self.benchmark_dir = Path(benchmark_dir)
        self.database_path = database_path or str(self.benchmark_dir / "benchmark_db.duckdb")
        self.config_path = config_path
        
        # Initialize components
        self.hardware_selector = None
        self.prediction_models = None
        self.available_hardware = self._detect_hardware()
        
        # Initialize hardware selector if available
        if HARDWARE_SELECTOR_AVAILABLE:
            try:
                self.hardware_selector = HardwareSelector(
                    database_path=str(self.benchmark_dir),
                    config_path=self.config_path
                )
                logger.info("Hardware selector initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize hardware selector: {e}")
        
        # Load prediction models if available
        if PREDICTOR_AVAILABLE:
            try:
                self.prediction_models = load_prediction_models()
                if self.prediction_models:
                    logger.info("Performance prediction models loaded")
                else:
                    logger.warning("No prediction models found")
            except Exception as e:
                logger.warning(f"Failed to load prediction models: {e}")
    
    def _detect_hardware(self) -> Dict[str, bool]:
        """
        Detect available hardware.
        
        Returns:
            Dictionary of available hardware
        """
        if INTEGRATION_AVAILABLE:
            return detect_hardware_availability()
        
        # Basic detection if integration not available
        available_hw = {
            "cpu": True,  # CPU is always available
            "cuda": False,
            "rocm": False,
            "mps": False,
            "openvino": False,
            "webnn": False,
            "webgpu": False
        }
        
        # Try to detect CUDA
        try:
            import torch
            available_hw["cuda"] = torch.cuda.is_available()
            
            # Check for MPS (Apple Silicon)
            if hasattr(torch.backends, "mps"):
                available_hw["mps"] = torch.backends.mps.is_available()
        except ImportError:
            pass
        
        # Try to detect ROCm through PyTorch
        try:
            import torch
            if torch.cuda.is_available() and "rocm" in torch.__version__.lower():
                available_hw["rocm"] = True
        except (ImportError, AttributeError):
            pass
        
        # Try to detect OpenVINO
        try:
            import openvino
            available_hw["openvino"] = True
        except ImportError:
            pass
        
        return available_hw
    
    def predict_optimal_hardware(self,
                                model_name: str,
                                model_family: Optional[str] = None,
                                batch_size: int = 1,
                                sequence_length: int = 128,
                                mode: str = "inference",
                                precision: str = "fp32",
                                available_hardware: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Predict the optimal hardware for a given model and configuration.
        
        Args:
            model_name: Name of the model
            model_family: Optional model family/category
            batch_size: Batch size
            sequence_length: Sequence length
            mode: "inference" or "training"
            precision: Precision to use (fp32, fp16, int8)
            available_hardware: Optional list of available hardware types
            
        Returns:
            Dictionary with hardware recommendation and performance predictions
        """
        # Use specified available hardware or detected hardware
        if available_hardware is None:
            available_hardware = [hw for hw, available in self.available_hardware.items() if available]
        
        # Determine model family if not provided
        if model_family is None and INTEGRATION_AVAILABLE:
            model_family, _ = classify_model_family(model_name)
        elif model_family is None:
            # Simple fallback heuristic
            model_name_lower = model_name.lower()
            if "bert" in model_name_lower or "roberta" in model_name_lower:
                model_family = "embedding"
            elif "t5" in model_name_lower or "gpt" in model_name_lower or "llama" in model_name_lower:
                model_family = "text_generation"
            elif "vit" in model_name_lower or "clip" in model_name_lower:
                model_family = "vision"
            elif "whisper" in model_name_lower or "wav2vec" in model_name_lower:
                model_family = "audio"
            elif "llava" in model_name_lower or "blip" in model_name_lower:
                model_family = "multimodal"
            else:
                model_family = "text_generation"  # Default to text generation
        
        # Step 1: Use advanced hardware selector if available
        if self.hardware_selector is not None:
            try:
                recommendation = self.hardware_selector.select_hardware(
                    model_family=model_family,
                    model_name=model_name,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    mode=mode,
                    precision=precision,
                    available_hardware=available_hardware
                )
                
                logger.info(f"Used advanced hardware selector: {recommendation['primary_recommendation']} for {model_name}")
                return recommendation
            except Exception as e:
                logger.warning(f"Advanced hardware selection failed: {e}, falling back to basic selection")
        
        # Step 2: Use hardware-model integration if available
        if INTEGRATION_AVAILABLE:
            try:
                # Filter to only available hardware
                filtered_hw = {}
                for hw in available_hardware:
                    filtered_hw[hw] = True
                
                integration_result = integrate_hardware_and_model(
                    model_name=model_name,
                    model_family=model_family,
                    hardware_info=filtered_hw
                )
                
                # Convert to similar format as hardware selector
                result = {
                    "model_family": model_family,
                    "model_name": model_name,
                    "model_size": integration_result["model_size"],
                    "model_size_category": integration_result["model_size"],
                    "batch_size": batch_size,
                    "sequence_length": sequence_length,
                    "precision": precision,
                    "mode": mode,
                    "primary_recommendation": integration_result["device"],
                    "fallback_options": [hw for hw in integration_result["compatible_devices"] if hw != integration_result["device"]],
                    "compatible_hardware": integration_result["compatible_devices"],
                    "predicted_memory_mb": integration_result["estimated_memory_mb"],
                    "explanation": integration_result["reason"],
                    "prediction_source": "hardware_model_integration",
                    "available_hardware": integration_result["available_hardware"]
                }
                
                logger.info(f"Used hardware-model integration: {result['primary_recommendation']} for {model_name}")
                return result
            except Exception as e:
                logger.warning(f"Hardware-model integration failed: {e}, falling back to basic selection")
        
        # Step 3: Use fallback method if nothing else worked
        logger.info(f"Using simple fallback method for hardware selection")
        
        # Determine model size
        model_size = self._estimate_model_size(model_name)
        model_size_category = "small" if model_size < 100000000 else "medium" if model_size < 1000000000 else "large"
        
        # Simple hardware preference lists by model family
        preferences = {
            "embedding": ["cuda", "mps", "rocm", "openvino", "cpu"],
            "text_generation": ["cuda", "rocm", "mps", "cpu"],
            "vision": ["cuda", "openvino", "rocm", "mps", "cpu"],
            "audio": ["cuda", "cpu", "mps", "rocm"],
            "multimodal": ["cuda", "cpu"]
        }
        
        # Get preferences for this family
        family_preferences = preferences.get(model_family, ["cuda", "cpu"])
        
        # Filter by available hardware
        compatible_hw = [hw for hw in family_preferences if hw in available_hardware]
        
        # Default to CPU if nothing else is available
        if not compatible_hw:
            compatible_hw = ["cpu"]
        
        # Create recommendation
        result = {
            "model_family": model_family,
            "model_name": model_name,
            "model_size": model_size,
            "model_size_category": model_size_category,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "precision": precision,
            "mode": mode,
            "primary_recommendation": compatible_hw[0],
            "fallback_options": compatible_hw[1:],
            "compatible_hardware": compatible_hw,
            "explanation": f"Simple fallback selection based on model family preferences",
            "prediction_source": "fallback"
        }
        
        return result
    
    def predict_performance(self,
                          model_name: str,
                          model_family: str,
                          hardware: Union[str, List[str]],
                          batch_size: int = 1,
                          sequence_length: int = 128,
                          mode: str = "inference",
                          precision: str = "fp32") -> Dict[str, Any]:
        """
        Predict performance for a model on specified hardware.
        
        Args:
            model_name: Name of the model
            model_family: Model family/category
            hardware: Hardware type or list of hardware types
            batch_size: Batch size
            sequence_length: Sequence length
            mode: "inference" or "training"
            precision: Precision to use (fp32, fp16, int8)
            
        Returns:
            Dictionary with performance predictions
        """
        # Convert single hardware to list
        if isinstance(hardware, str):
            hardware_list = [hardware]
        else:
            hardware_list = hardware
        
        result = {
            "model_name": model_name,
            "model_family": model_family,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "mode": mode,
            "precision": precision,
            "predictions": {}
        }
        
        # Step 1: Use advanced performance predictor if available
        if PREDICTOR_AVAILABLE and self.prediction_models:
            try:
                for hw in hardware_list:
                    prediction = predict_performance(
                        models=self.prediction_models,
                        model_name=model_name,
                        model_category=model_family,
                        hardware=hw,
                        batch_size=batch_size,
                        precision=precision,
                        mode=mode
                    )
                    
                    if prediction:
                        result["predictions"][hw] = {
                            "throughput": prediction.get("throughput"),
                            "latency": prediction.get("latency_mean"),
                            "memory_usage": prediction.get("memory_usage"),
                            "source": "model_performance_predictor"
                        }
                
                if result["predictions"]:
                    logger.info(f"Used model performance predictor for {model_name}")
                    return result
            except Exception as e:
                logger.warning(f"Advanced performance prediction failed: {e}, falling back to heuristics")
        
        # Step 2: Use hardware-model integration if available
        if INTEGRATION_AVAILABLE:
            try:
                # Estimate memory requirements
                model_size_tier = get_model_size_tier(model_name, model_family)
                memory_mb = estimate_memory_requirements(model_family, model_size_tier)
                
                for hw in hardware_list:
                    # Check compatibility
                    is_compatible, reason = check_hardware_compatibility(model_family, model_size_tier, hw)
                    
                    if is_compatible:
                        # Use heuristic performance estimates
                        if hw == "cuda":
                            relative_speed = 1.0
                        elif hw == "rocm":
                            relative_speed = 0.9
                        elif hw == "mps":
                            relative_speed = 0.8
                        elif hw == "openvino":
                            relative_speed = 0.7
                        else:
                            relative_speed = 0.5  # CPU and others
                        
                        # Adjust for precision
                        if precision == "fp16":
                            relative_speed *= 1.3  # 30% faster
                        elif precision == "int8":
                            relative_speed *= 1.6  # 60% faster
                        
                        # Base values adjusted by relative speed
                        throughput = 100 * relative_speed * batch_size / (1 + (batch_size / 32))
                        latency = 20 / relative_speed * (1 + (batch_size / 16))
                        
                        result["predictions"][hw] = {
                            "throughput": throughput,
                            "latency": latency,
                            "memory_usage": memory_mb * (1 + (batch_size / 8)),
                            "source": "hardware_model_integration"
                        }
                    else:
                        result["predictions"][hw] = {
                            "compatible": False,
                            "reason": reason,
                            "source": "hardware_model_integration"
                        }
                
                if result["predictions"]:
                    logger.info(f"Used hardware-model integration for performance prediction")
                    return result
            except Exception as e:
                logger.warning(f"Hardware-model integration for performance failed: {e}, falling back to basic heuristics")
        
        # Step 3: Use basic heuristics as fallback
        model_size = self._estimate_model_size(model_name)
        
        for hw in hardware_list:
            # Base values depend on hardware type
            if hw == "cuda":
                base_throughput = 100
                base_latency = 10
            elif hw == "rocm":
                base_throughput = 80
                base_latency = 12
            elif hw == "mps":
                base_throughput = 60
                base_latency = 15
            elif hw == "openvino":
                base_throughput = 50
                base_latency = 18
            else:
                base_throughput = 20
                base_latency = 30
            
            # Adjust for batch size
            throughput = base_throughput * (batch_size / (1 + (batch_size / 32)))
            latency = base_latency * (1 + (batch_size / 16))
            
            # Adjust for model size
            size_factor = 1.0
            if model_size > 1000000000:  # > 1B params
                size_factor = 5.0
            elif model_size > 100000000:  # > 100M params
                size_factor = 2.0
            
            throughput /= size_factor
            latency *= size_factor
            
            # Adjust for precision
            if precision == "fp16":
                throughput *= 1.3
                latency /= 1.3
            elif precision == "int8":
                throughput *= 1.6
                latency /= 1.6
            
            result["predictions"][hw] = {
                "throughput": throughput,
                "latency": latency,
                "memory_usage": model_size * 0.004 * batch_size,  # Rough estimate based on model size
                "source": "basic_heuristic"
            }
        
        logger.info(f"Used basic heuristics for performance prediction")
        return result
    
    def _estimate_model_size(self, model_name: str) -> int:
        """
        Estimate model size in parameters based on model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Estimated number of parameters
        """
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
    
    def create_hardware_prediction_matrix(self,
                                        models: Optional[List[Dict[str, str]]] = None,
                                        batch_sizes: Optional[List[int]] = None,
                                        hardware_platforms: Optional[List[str]] = None,
                                        precision: str = "fp32",
                                        mode: str = "inference") -> Dict[str, Any]:
        """
        Create a comprehensive hardware prediction matrix.
        
        Args:
            models: List of models with 'name' and 'family' keys
            batch_sizes: List of batch sizes to test
            hardware_platforms: List of hardware platforms to test
            precision: Precision to use
            mode: "inference" or "training"
            
        Returns:
            Dictionary with prediction matrix
        """
        # Use default models if not specified
        if models is None:
            models = [
                {"name": "bert-base-uncased", "family": "embedding"},
                {"name": "t5-small", "family": "text_generation"},
                {"name": "gpt2", "family": "text_generation"},
                {"name": "facebook/dino-vitb16", "family": "vision"},
                {"name": "openai/clip-vit-base-patch32", "family": "vision"},
                {"name": "openai/whisper-tiny", "family": "audio"}
            ]
        
        # Use default batch sizes if not specified
        if batch_sizes is None:
            batch_sizes = [1, 8, 32]
        
        # Use available hardware if not specified
        if hardware_platforms is None:
            hardware_platforms = [hw for hw, available in self.available_hardware.items() if available]
        
        # Use generate_prediction_matrix from model_performance_predictor if available
        if PREDICTOR_AVAILABLE and self.prediction_models and hasattr(sys.modules['model_performance_predictor'], 'generate_prediction_matrix'):
            try:
                # Convert models to expected format
                model_configs = []
                for model in models:
                    model_configs.append({
                        "name": model["name"],
                        "category": model["family"]
                    })
                
                matrix = generate_prediction_matrix(
                    models=self.prediction_models,
                    model_configs=model_configs,
                    hardware_platforms=hardware_platforms,
                    batch_sizes=batch_sizes,
                    precision_options=[precision],
                    mode=mode
                )
                
                if matrix:
                    logger.info(f"Used generate_prediction_matrix from model_performance_predictor")
                    return matrix
            except Exception as e:
                logger.warning(f"Failed to use generate_prediction_matrix: {e}, falling back to manual creation")
        
        # Create matrix manually
        matrix = {
            "timestamp": "2025-03-01T00:00:00Z",
            "mode": mode,
            "models": {},
            "hardware_platforms": hardware_platforms,
            "batch_sizes": batch_sizes,
            "precision_options": [precision]
        }
        
        # Generate predictions for each model and configuration
        for model in models:
            model_name = model["name"]
            model_family = model["family"]
            
            # Get hardware recommendation
            recommendation = self.predict_optimal_hardware(
                model_name=model_name,
                model_family=model_family,
                batch_size=batch_sizes[0],
                mode=mode,
                precision=precision,
                available_hardware=hardware_platforms
            )
            
            # Initialize model entry
            matrix["models"][model_name] = {
                "name": model_name,
                "category": model_family,
                "predictions": {}
            }
            
            # Generate predictions for each batch size and hardware
            for hw in hardware_platforms:
                matrix["models"][model_name]["predictions"][hw] = {}
                
                for batch_size in batch_sizes:
                    # Get performance prediction
                    perf_pred = self.predict_performance(
                        model_name=model_name,
                        model_family=model_family,
                        hardware=hw,
                        batch_size=batch_size,
                        mode=mode,
                        precision=precision
                    )
                    
                    if hw in perf_pred["predictions"]:
                        pred = perf_pred["predictions"][hw]
                        
                        # Add to matrix
                        matrix["models"][model_name]["predictions"][hw][str(batch_size)] = {
                            precision: {
                                "throughput": pred.get("throughput"),
                                "latency_mean": pred.get("latency"),
                                "memory_usage": pred.get("memory_usage")
                            }
                        }
        
        return matrix
    
    def visualize_matrix(self, matrix: Dict[str, Any], output_dir: str = "./visualizations"):
        """
        Generate visualizations from the prediction matrix.
        
        Args:
            matrix: Prediction matrix
            output_dir: Directory to save visualizations
            
        Returns:
            List of generated visualization files
        """
        if PREDICTOR_AVAILABLE and hasattr(sys.modules['model_performance_predictor'], 'visualize_predictions'):
            try:
                # Create output directory
                os.makedirs(output_dir, exist_ok=True)
                
                # Generate visualizations
                visualization_files = []
                
                for metric in ["throughput", "latency_mean", "memory_usage"]:
                    files = visualize_predictions(
                        matrix=matrix,
                        metric=metric,
                        output_dir=output_dir
                    )
                    visualization_files.extend(files)
                
                return visualization_files
            except Exception as e:
                logger.warning(f"Failed to generate visualizations: {e}")
                return []
        else:
            logger.warning("visualize_predictions not available in model_performance_predictor")
            return []

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Hardware Model Predictor")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--family", type=str, help="Model family/category")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-length", type=int, default=128, help="Sequence length")
    parser.add_argument("--mode", type=str, choices=["inference", "training"], default="inference", help="Mode")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "int8"], default="fp32", help="Precision")
    parser.add_argument("--hardware", type=str, nargs="+", help="Hardware platforms to consider")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_results", help="Benchmark results directory")
    parser.add_argument("--database", type=str, help="Path to benchmark database")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--generate-matrix", action="store_true", help="Generate prediction matrix")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--output-dir", type=str, default="./visualizations", help="Output directory for visualizations")
    parser.add_argument("--output-file", type=str, help="Output file for prediction matrix")
    parser.add_argument("--detect-hardware", action="store_true", help="Detect available hardware")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Create predictor
    predictor = HardwareModelPredictor(
        benchmark_dir=args.benchmark_dir,
        database_path=args.database,
        config_path=args.config
    )
    
    # Show detected hardware
    if args.detect_hardware:
        print("Detected hardware:")
        for hw_type, available in predictor.available_hardware.items():
            status = "✅ Available" if available else "❌ Not available"
            print(f"  - {hw_type}: {status}")
        return
    
    # Generate prediction matrix if requested
    if args.generate_matrix:
        matrix = predictor.create_hardware_prediction_matrix(
            hardware_platforms=args.hardware,
            precision=args.precision,
            mode=args.mode
        )
        
        # Save matrix to file if specified
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(matrix, f, indent=2)
            print(f"Prediction matrix saved to {args.output_file}")
        else:
            print(json.dumps(matrix, indent=2))
        
        # Generate visualizations if requested
        if args.visualize:
            visualization_files = predictor.visualize_matrix(matrix, args.output_dir)
            if visualization_files:
                print("Generated visualizations:")
                for file in visualization_files:
                    print(f"  - {file}")
        
        return
    
    # Predict optimal hardware if model specified
    if args.model:
        # Get hardware recommendation
        recommendation = predictor.predict_optimal_hardware(
            model_name=args.model,
            model_family=args.family,
            batch_size=args.batch_size,
            sequence_length=args.seq_length,
            mode=args.mode,
            precision=args.precision,
            available_hardware=args.hardware
        )
        
        # Get performance prediction for recommended hardware
        performance = predictor.predict_performance(
            model_name=args.model,
            model_family=recommendation["model_family"],
            hardware=recommendation["primary_recommendation"],
            batch_size=args.batch_size,
            sequence_length=args.seq_length,
            mode=args.mode,
            precision=args.precision
        )
        
        # Print results
        print(f"\nHardware Recommendation for {args.model}:")
        print(f"  Primary Recommendation: {recommendation['primary_recommendation']}")
        print(f"  Fallback Options: {', '.join(recommendation['fallback_options'])}")
        print(f"  Compatible Hardware: {', '.join(recommendation['compatible_hardware'])}")
        print(f"  Model Family: {recommendation['model_family']}")
        print(f"  Model Size: {recommendation['model_size_category']} ({recommendation['model_size']} parameters)")
        print(f"  Explanation: {recommendation['explanation']}")
        print("\nPerformance Prediction:")
        
        hw = recommendation["primary_recommendation"]
        if hw in performance["predictions"]:
            pred = performance["predictions"][hw]
            print(f"  Throughput: {pred.get('throughput', 'N/A'):.2f} items/sec")
            print(f"  Latency: {pred.get('latency', 'N/A'):.2f} ms")
            print(f"  Memory Usage: {pred.get('memory_usage', 'N/A'):.2f} MB")
            print(f"  Prediction Source: {pred.get('source', 'N/A')}")
        else:
            print("  No performance prediction available")
        
        return
    
    # If no specific action, print help
    parser.print_help()

if __name__ == "__main__":
    main()