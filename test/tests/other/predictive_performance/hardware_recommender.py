#!/usr/bin/env python3
"""
Hardware Recommendation System for the Predictive Performance System.

This module provides a specialized system for recommending optimal hardware
platforms for specific models and configurations, based on performance predictions
and user-defined constraints.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional, Union
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Configure logging
logging.basicConfig())))))))))))))))))))))))
level=logging.INFO,
format='%())))))))))))))))))))))))asctime)s - %())))))))))))))))))))))))name)s - %())))))))))))))))))))))))levelname)s - %())))))))))))))))))))))))message)s'
)
logger = logging.getLogger())))))))))))))))))))))))"hardware_recommender")

# Import the prediction module
try:
    from test.tests.other.predictive_performance.predict import PerformancePredictor
except ImportError:
    # When running as standalone script
    sys.path.append())))))))))))))))))))))))str())))))))))))))))))))))))Path())))))))))))))))))))))))__file__).parent))
    try:
        from predict import PerformancePredictor
    except ImportError:
        logger.error())))))))))))))))))))))))"Failed to import PerformancePredictor. Make sure the predict.py module is in the same directory.")
        PerformancePredictor = None


class HardwareRecommender:
    """
    Hardware Recommendation System based on performance predictions.
    
    This class provides methods to recommend optimal hardware platforms
    for specific models and configurations, generate comparative visualizations,
    and export detailed recommendation reports.
    """
    
    def __init__())))))))))))))))))))))))
    self,
    predictor: Optional[]]]]]]]]],,,,,,,,,PerformancePredictor] = None,
    predictor_params: Optional[]]]]]]]]],,,,,,,,,Dict[]]]]]]]]],,,,,,,,,str, Any]] = None,
    available_hardware: Optional[]]]]]]]]],,,,,,,,,List[]]]]]]]]],,,,,,,,,str]] = None,
    power_constraints: Optional[]]]]]]]]],,,,,,,,,Dict[]]]]]]]]],,,,,,,,,str, float]] = None,
    cost_weights: Optional[]]]]]]]]],,,,,,,,,Dict[]]]]]]]]],,,,,,,,,str, float]] = None,
    confidence_threshold: float = 0.7,
    verbose: bool = False
    ):
        """
        Initialize the Hardware Recommender.
        
        Args:
            predictor: PerformancePredictor instance to use for predictions
            predictor_params: Parameters to pass to PerformancePredictor if not provided:
                available_hardware: List of available hardware platforms to consider
                power_constraints: Dictionary mapping hardware platforms to power constraints
                cost_weights: Dictionary of cost factor weights for recommendations
                confidence_threshold: Minimum confidence threshold for recommendations
                verbose: Whether to print detailed logs
                """
        # Set up the predictor
        if predictor is not None:
            self.predictor = predictor
        elif PerformancePredictor is not None:
            predictor_args = predictor_params or {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            self.predictor = PerformancePredictor())))))))))))))))))))))))**predictor_args)
        else:
            raise ImportError())))))))))))))))))))))))"PerformancePredictor class not available and no predictor instance provided")
        
        # Available hardware platforms
            self.available_hardware = available_hardware or []]]]]]]]],,,,,,,,,
            "cpu", "cuda", "rocm", "mps", "openvino", "qnn", "webnn", "webgpu"
            ]
        
        # Power constraints ())))))))))))))))))))))))watts)
            self.power_constraints = power_constraints or {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        
        # Cost weights
            self.cost_weights = cost_weights or {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "performance": 0.6,  # Higher performance is better
            "power_efficiency": 0.2,  # Higher power efficiency is better
            "memory_usage": 0.1,  # Lower memory usage is better
            "availability": 0.1,  # Higher availability is better
            }
        
        # Confidence threshold
            self.confidence_threshold = confidence_threshold
        
        # Verbose mode
            self.verbose = verbose
        
        # Default hardware properties
            self.hardware_properties = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "cpu": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "availability": 1.0,  # Always available
            "relative_cost": 1.0,
            "power_rating": 65.0,  # Watts ())))))))))))))))))))))))typical)
            "memory_capacity": 32.0,  # GB ())))))))))))))))))))))))typical)
            "parallel_capabilities": 0.5,
            "quantization_support": 0.7,
            "development_complexity": 0.2,  # Low complexity
            "deployment_complexity": 0.1,  # Very low complexity
            },
            "cuda": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "availability": 0.8,  # Usually available
            "relative_cost": 2.0,
            "power_rating": 250.0,  # Watts ())))))))))))))))))))))))typical)
            "memory_capacity": 16.0,  # GB ())))))))))))))))))))))))typical)
            "parallel_capabilities": 1.0,
            "quantization_support": 0.9,
            "development_complexity": 0.4,
            "deployment_complexity": 0.5,
            },
            "rocm": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "availability": 0.5,  # Somewhat available
            "relative_cost": 1.8,
            "power_rating": 230.0,  # Watts ())))))))))))))))))))))))typical)
            "memory_capacity": 12.0,  # GB ())))))))))))))))))))))))typical)
            "parallel_capabilities": 0.95,
            "quantization_support": 0.8,
            "development_complexity": 0.5,
            "deployment_complexity": 0.6,
            },
            "mps": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "availability": 0.4,  # Less common
            "relative_cost": 2.5,
            "power_rating": 60.0,  # Watts ())))))))))))))))))))))))typical)
            "memory_capacity": 16.0,  # GB ())))))))))))))))))))))))typical)
            "parallel_capabilities": 0.9,
            "quantization_support": 0.7,
            "development_complexity": 0.4,
            "deployment_complexity": 0.5,
            },
            "openvino": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "availability": 0.7,  # Fairly available
            "relative_cost": 1.2,
            "power_rating": 80.0,  # Watts ())))))))))))))))))))))))typical)
            "memory_capacity": 32.0,  # GB ())))))))))))))))))))))))shared with system)
            "parallel_capabilities": 0.7,
            "quantization_support": 1.0,  # Excellent quantization
            "development_complexity": 0.5,
            "deployment_complexity": 0.4,
            },
            "qnn": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "availability": 0.3,  # Less common
            "relative_cost": 1.5,
            "power_rating": 15.0,  # Watts ())))))))))))))))))))))))mobile-optimized)
            "memory_capacity": 4.0,  # GB ())))))))))))))))))))))))typical for mobile)
            "parallel_capabilities": 0.7,
            "quantization_support": 0.9,
            "development_complexity": 0.7,
            "deployment_complexity": 0.7,
            },
            "webnn": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "availability": 0.4,  # Browser-dependent
            "relative_cost": 0.8,
            "power_rating": 40.0,  # Watts ())))))))))))))))))))))))varies widely)
            "memory_capacity": 2.0,  # GB ())))))))))))))))))))))))limited by browser)
            "parallel_capabilities": 0.4,
            "quantization_support": 0.5,
            "development_complexity": 0.6,
            "deployment_complexity": 0.3,
            },
            "webgpu": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "availability": 0.5,  # Browser-dependent
            "relative_cost": 0.9,
            "power_rating": 50.0,  # Watts ())))))))))))))))))))))))varies widely)
            "memory_capacity": 1.5,  # GB ())))))))))))))))))))))))limited by browser)
            "parallel_capabilities": 0.6,
            "quantization_support": 0.6,
            "development_complexity": 0.7,
            "deployment_complexity": 0.4,
            },
            }
        
        # Update hardware properties with power constraints
        if self.power_constraints:
            for hw, power in self.power_constraints.items())))))))))))))))))))))))):
                if hw in self.hardware_properties:
                    self.hardware_properties[]]]]]]]]],,,,,,,,,hw][]]]]]]]]],,,,,,,,,"power_rating"] = power
        
                    logger.info())))))))))))))))))))))))f"Initialized HardwareRecommender with {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}len())))))))))))))))))))))))self.available_hardware)} available hardware platforms")
    
                    def recommend_hardware())))))))))))))))))))))))
                    self,
                    model_name: str,
                    model_type: str,
                    batch_size: int,
                    optimization_metric: str = "throughput",
                    precision_format: str = "FP32",
                    power_constrained: bool = False,
                    memory_constrained: bool = False,
                    deployment_constrained: bool = False,
                    custom_constraints: Optional[]]]]]]]]],,,,,,,,,Dict[]]]]]]]]],,,,,,,,,str, Any]] = None,
                    consideration_threshold: float = 0.8,
                return_all_candidates: bool = False
    ) -> Dict[]]]]]]]]],,,,,,,,,str, Any]:
        """
        Recommend optimal hardware platform for a specific model and configuration.
        
        Args:
            model_name: Name of the model
            model_type: Type of the model
            batch_size: Batch size for inference
            optimization_metric: Performance metric to optimize for
            ())))))))))))))))))))))))throughput, latency, memory, power)
            precision_format: Precision format ())))))))))))))))))))))))FP32, FP16, INT8, etc.)
            power_constrained: Whether power consumption is a primary constraint
            memory_constrained: Whether memory usage is a primary constraint
            deployment_constrained: Whether deployment complexity is a constraint
            custom_constraints: Dictionary of custom constraints to apply
            consideration_threshold: Relative performance threshold for consideration ())))))))))))))))))))))))0-1)
        return_all_candidates: Whether to return all candidates in the response
            
        Returns:
            Dictionary with recommended hardware platform and performance metrics
            """
            logger.info())))))))))))))))))))))))f"Recommending hardware for model '{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}model_name}' with batch size {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}batch_size}")
        
        # Filter available hardware based on constraints
            available_hardware = self._filter_hardware_by_constraints())))))))))))))))))))))))
            power_constrained=power_constrained,
            memory_constrained=memory_constrained,
            deployment_constrained=deployment_constrained,
            custom_constraints=custom_constraints
            )
        
        if not available_hardware:
            logger.warning())))))))))))))))))))))))"No hardware platforms available after applying constraints")
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "error": "No hardware platforms available after applying constraints",
            "model_name": model_name,
            "model_type": model_type,
            "batch_size": batch_size,
            "constraints_applied": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "power_constrained": power_constrained,
            "memory_constrained": memory_constrained,
            "deployment_constrained": deployment_constrained,
            "custom_constraints": custom_constraints
            }
            }
        
        # Get predictions for all available hardware platforms
            predictions = self._get_predictions_for_hardware())))))))))))))))))))))))
            model_name=model_name,
            model_type=model_type,
            batch_size=batch_size,
            hardware_platforms=available_hardware,
            precision_format=precision_format,
            metrics=[]]]]]]]]],,,,,,,,,optimization_metric, "memory", "power"]  # Always include memory and power
            )
        
        # Check if predictions were successful:
        if "error" in predictions:
            return predictions
        
        # Calculate combined scores based on all factors
            scored_predictions = self._calculate_combined_scores())))))))))))))))))))))))
            predictions=predictions,
            optimization_metric=optimization_metric,
            power_constrained=power_constrained,
            memory_constrained=memory_constrained,
            deployment_constrained=deployment_constrained
            )
        
        # Sort by combined score ())))))))))))))))))))))))descending)
            sorted_predictions = sorted())))))))))))))))))))))))
            scored_predictions, 
            key=lambda x: x[]]]]]]]]],,,,,,,,,"combined_score"], 
            reverse=True
            )
        
        # Get the best recommendation
            best_recommendation = sorted_predictions[]]]]]]]]],,,,,,,,,0]
        
        # Find viable alternatives
            alternatives = []]]]]]]]],,,,,,,,,]
        for pred in sorted_predictions[]]]]]]]]],,,,,,,,,1:]:
            # Check if the prediction is viable ())))))))))))))))))))))))within threshold of best):
            if pred[]]]]]]]]],,,,,,,,,"combined_score"] >= best_recommendation[]]]]]]]]],,,,,,,,,"combined_score"] * consideration_threshold:
                alternatives.append())))))))))))))))))))))))pred)
        
        # Create response
                response = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "model_name": model_name,
                "model_type": model_type,
                "batch_size": batch_size,
                "metric": optimization_metric,
                "recommendation": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "hardware_platform": best_recommendation[]]]]]]]]],,,,,,,,,"hardware_platform"],
                "estimated_value": best_recommendation[]]]]]]]]],,,,,,,,,"predicted_value"],
                "uncertainty": best_recommendation[]]]]]]]]],,,,,,,,,"uncertainty"],
                "confidence": best_recommendation[]]]]]]]]],,,,,,,,,"confidence"],
                "combined_score": best_recommendation[]]]]]]]]],,,,,,,,,"combined_score"],
                "power_estimate": best_recommendation[]]]]]]]]],,,,,,,,,"power_estimate"],
                "memory_estimate": best_recommendation[]]]]]]]]],,,,,,,,,"memory_estimate"],
                "power_efficiency": best_recommendation[]]]]]]]]],,,,,,,,,"power_efficiency"]
                },
                "alternatives": []]]]]]]]],,,,,,,,,
                {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "hardware_platform": alt[]]]]]]]]],,,,,,,,,"hardware_platform"],
                "estimated_value": alt[]]]]]]]]],,,,,,,,,"predicted_value"],
                "combined_score": alt[]]]]]]]]],,,,,,,,,"combined_score"],
                "power_estimate": alt[]]]]]]]]],,,,,,,,,"power_estimate"],
                "memory_estimate": alt[]]]]]]]]],,,,,,,,,"memory_estimate"]
                }
                for alt in alternatives[]]]]]]]]],,,,,,,,,:3]  # Limit to top 3 alternatives
                ],
                "constraint_weights": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "performance": self.cost_weights[]]]]]]]]],,,,,,,,,"performance"],
                "power_efficiency": self.cost_weights[]]]]]]]]],,,,,,,,,"power_efficiency"] * ())))))))))))))))))))))))2 if power_constrained else 1),:
                "memory_usage": self.cost_weights[]]]]]]]]],,,,,,,,,"memory_usage"] * ())))))))))))))))))))))))2 if memory_constrained else 1),:
                    "availability": self.cost_weights[]]]]]]]]],,,,,,,,,"availability"]
                    }
                    }
        
        # Include all candidates if requested::
        if return_all_candidates:
            response[]]]]]]]]],,,,,,,,,"all_candidates"] = sorted_predictions
        
            logger.info())))))))))))))))))))))))f"Recommended {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}best_recommendation[]]]]]]]]],,,,,,,,,'hardware_platform']} for model '{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}model_name}' "
            f"with estimated {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}optimization_metric} of {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}best_recommendation[]]]]]]]]],,,,,,,,,'predicted_value']:.2f}")
        
                    return response
    
                    def _filter_hardware_by_constraints())))))))))))))))))))))))
                    self,
                    power_constrained: bool = False,
                    memory_constrained: bool = False,
                    deployment_constrained: bool = False,
                    custom_constraints: Optional[]]]]]]]]],,,,,,,,,Dict[]]]]]]]]],,,,,,,,,str, Any]] = None
    ) -> List[]]]]]]]]],,,,,,,,,str]:
        """
        Filter available hardware platforms based on constraints.
        
        Args:
            power_constrained: Whether power consumption is a primary constraint
            memory_constrained: Whether memory usage is a primary constraint
            deployment_constrained: Whether deployment complexity is a constraint
            custom_constraints: Dictionary of custom constraints to apply
            
        Returns:
            List of hardware platforms that meet the constraints
            """
        # Start with all available hardware
            filtered_hardware = list())))))))))))))))))))))))self.available_hardware)
        
        # Apply power constraints
        if power_constrained:
            power_threshold = 100.0  # Watts ())))))))))))))))))))))))default)
            if custom_constraints and "max_power" in custom_constraints:
                power_threshold = custom_constraints[]]]]]]]]],,,,,,,,,"max_power"]
            
                filtered_hardware = []]]]]]]]],,,,,,,,,
                hw for hw in filtered_hardware:::::::
                    if hw in self.hardware_properties and
                    self.hardware_properties[]]]]]]]]],,,,,,,,,hw][]]]]]]]]],,,,,,,,,"power_rating"] <= power_threshold
                    ]
        
        # Apply memory constraints:
        if memory_constrained:
            memory_threshold = 4.0  # GB ())))))))))))))))))))))))default)
            if custom_constraints and "min_memory" in custom_constraints:
                memory_threshold = custom_constraints[]]]]]]]]],,,,,,,,,"min_memory"]
            
                filtered_hardware = []]]]]]]]],,,,,,,,,
                hw for hw in filtered_hardware:::::::
                    if hw in self.hardware_properties and
                    self.hardware_properties[]]]]]]]]],,,,,,,,,hw][]]]]]]]]],,,,,,,,,"memory_capacity"] >= memory_threshold
                    ]
        
        # Apply deployment constraints:
        if deployment_constrained:
            deployment_threshold = 0.5  # Moderate complexity ())))))))))))))))))))))))default)
            if custom_constraints and "max_deployment_complexity" in custom_constraints:
                deployment_threshold = custom_constraints[]]]]]]]]],,,,,,,,,"max_deployment_complexity"]
            
                filtered_hardware = []]]]]]]]],,,,,,,,,
                hw for hw in filtered_hardware:::::::
                    if hw in self.hardware_properties and
                    self.hardware_properties[]]]]]]]]],,,,,,,,,hw][]]]]]]]]],,,,,,,,,"deployment_complexity"] <= deployment_threshold
                    ]
        
        # Apply custom constraints:
        if custom_constraints:
            # Filter by availability
            if "min_availability" in custom_constraints:
                min_availability = custom_constraints[]]]]]]]]],,,,,,,,,"min_availability"]
                filtered_hardware = []]]]]]]]],,,,,,,,,
                    hw for hw in filtered_hardware:::::::
                        if hw in self.hardware_properties and
                        self.hardware_properties[]]]]]]]]],,,,,,,,,hw][]]]]]]]]],,,,,,,,,"availability"] >= min_availability
                        ]
            
            # Filter by quantization support:
            if "min_quantization_support" in custom_constraints:
                min_quant = custom_constraints[]]]]]]]]],,,,,,,,,"min_quantization_support"]
                filtered_hardware = []]]]]]]]],,,,,,,,,
                    hw for hw in filtered_hardware:::::::
                        if hw in self.hardware_properties and
                        self.hardware_properties[]]]]]]]]],,,,,,,,,hw][]]]]]]]]],,,,,,,,,"quantization_support"] >= min_quant
                        ]
            
            # Filter by development complexity:
            if "max_development_complexity" in custom_constraints:
                max_dev = custom_constraints[]]]]]]]]],,,,,,,,,"max_development_complexity"]
                filtered_hardware = []]]]]]]]],,,,,,,,,
                    hw for hw in filtered_hardware:::::::
                        if hw in self.hardware_properties and
                        self.hardware_properties[]]]]]]]]],,,,,,,,,hw][]]]]]]]]],,,,,,,,,"development_complexity"] <= max_dev
                        ]
            
            # Filter by parallel capabilities:
            if "min_parallel_capabilities" in custom_constraints:
                min_parallel = custom_constraints[]]]]]]]]],,,,,,,,,"min_parallel_capabilities"]
                filtered_hardware = []]]]]]]]],,,,,,,,,
                    hw for hw in filtered_hardware:::::::
                        if hw in self.hardware_properties and
                        self.hardware_properties[]]]]]]]]],,,,,,,,,hw][]]]]]]]]],,,,,,,,,"parallel_capabilities"] >= min_parallel
                        ]
        
        # If no hardware passes the constraints, return all available hardware with a warning:
        if not filtered_hardware:
            logger.warning())))))))))))))))))))))))"No hardware platforms meet all constraints, returning all available hardware")
                        return list())))))))))))))))))))))))self.available_hardware)
        
                return filtered_hardware
    
                def _get_predictions_for_hardware())))))))))))))))))))))))
                self,
                model_name: str,
                model_type: str,
                batch_size: int,
                hardware_platforms: List[]]]]]]]]],,,,,,,,,str],
                precision_format: str,
                metrics: List[]]]]]]]]],,,,,,,,,str]
    ) -> List[]]]]]]]]],,,,,,,,,Dict[]]]]]]]]],,,,,,,,,str, Any]]:
        """
        Get predictions for all specified hardware platforms and metrics.
        
        Args:
            model_name: Name of the model
            model_type: Type of the model
            batch_size: Batch size for inference
            hardware_platforms: List of hardware platforms to get predictions for
            precision_format: Precision format to use
            metrics: List of metrics to predict
            
        Returns:
            List of predictions for each hardware platform
            """
        # Check if predictor is available:
        if not hasattr())))))))))))))))))))))))self, 'predictor'):
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "error": "Predictor not available",
            "model_name": model_name,
            "model_type": model_type,
            "batch_size": batch_size
            }
        
        # Get predictions for all metrics and hardware platforms
            predictions = []]]]]]]]],,,,,,,,,]
        
        for hardware in hardware_platforms:
            prediction = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "hardware_platform": hardware,
            "model_name": model_name,
            "model_type": model_type,
            "batch_size": batch_size,
            "precision_format": precision_format
            }
            
            # Add hardware properties
            if hardware in self.hardware_properties:
                for key, value in self.hardware_properties[]]]]]]]]],,,,,,,,,hardware].items())))))))))))))))))))))))):
                    prediction[]]]]]]]]],,,,,,,,,f"hw_{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}key}"] = value
            
            # Get predictions for each metric
            for metric in metrics:
                try:
                    metric_pred = self.predictor.predict())))))))))))))))))))))))
                    model_name=model_name,
                    model_type=model_type,
                    hardware_platform=hardware,
                    batch_size=batch_size,
                    metric=metric
                    )
                    
                    # Add prediction and confidence
                    prediction[]]]]]]]]],,,,,,,,,f"{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}metric}_prediction"] = metric_pred[]]]]]]]]],,,,,,,,,metric]
                    prediction[]]]]]]]]],,,,,,,,,f"{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}metric}_uncertainty"] = metric_pred[]]]]]]]]],,,,,,,,,"uncertainty"]
                    prediction[]]]]]]]]],,,,,,,,,f"{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}metric}_confidence"] = metric_pred[]]]]]]]]],,,,,,,,,"confidence"]
                    
                    # For the main metric, add these directly
                    if metric == metrics[]]]]]]]]],,,,,,,,,0]:
                        prediction[]]]]]]]]],,,,,,,,,"predicted_value"] = metric_pred[]]]]]]]]],,,,,,,,,metric]
                        prediction[]]]]]]]]],,,,,,,,,"uncertainty"] = metric_pred[]]]]]]]]],,,,,,,,,"uncertainty"]
                        prediction[]]]]]]]]],,,,,,,,,"confidence"] = metric_pred[]]]]]]]]],,,,,,,,,"confidence"]
                except Exception as e:
                    logger.error())))))))))))))))))))))))f"Error predicting {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}metric} for {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}hardware}: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))))))))))))))))))e)}")
                    prediction[]]]]]]]]],,,,,,,,,f"{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}metric}_prediction"] = None
                    prediction[]]]]]]]]],,,,,,,,,f"{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}metric}_uncertainty"] = None
                    prediction[]]]]]]]]],,,,,,,,,f"{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}metric}_confidence"] = None
                    
                    if metric == metrics[]]]]]]]]],,,,,,,,,0]:
                        prediction[]]]]]]]]],,,,,,,,,"predicted_value"] = None
                        prediction[]]]]]]]]],,,,,,,,,"uncertainty"] = None
                        prediction[]]]]]]]]],,,,,,,,,"confidence"] = 0.0
            
            # If we have power and throughput/latency, calculate power efficiency
                        main_metric = metrics[]]]]]]]]],,,,,,,,,0]
            if main_metric in []]]]]]]]],,,,,,,,,"throughput", "latency"] and "power_prediction" in prediction and prediction[]]]]]]]]],,,,,,,,,"power_prediction"]:
                if main_metric == "throughput" and prediction[]]]]]]]]],,,,,,,,,"throughput_prediction"]:
                    # Higher throughput per watt is better
                    prediction[]]]]]]]]],,,,,,,,,"power_efficiency"] = prediction[]]]]]]]]],,,,,,,,,"throughput_prediction"] / max())))))))))))))))))))))))1.0, prediction[]]]]]]]]],,,,,,,,,"power_prediction"])
                elif main_metric == "latency" and prediction[]]]]]]]]],,,,,,,,,"latency_prediction"]:
                    # Lower latency per watt is better ())))))))))))))))))))))))inverted)
                    prediction[]]]]]]]]],,,,,,,,,"power_efficiency"] = 1.0 / ())))))))))))))))))))))))max())))))))))))))))))))))))0.1, prediction[]]]]]]]]],,,,,,,,,"latency_prediction"]) * max())))))))))))))))))))))))1.0, prediction[]]]]]]]]],,,,,,,,,"power_prediction"]))
            else:
                prediction[]]]]]]]]],,,,,,,,,"power_efficiency"] = None
            
            # Add estimates for power and memory
                prediction[]]]]]]]]],,,,,,,,,"power_estimate"] = prediction.get())))))))))))))))))))))))"power_prediction") or self.hardware_properties.get())))))))))))))))))))))))hardware, {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}).get())))))))))))))))))))))))"power_rating", 100.0)
                prediction[]]]]]]]]],,,,,,,,,"memory_estimate"] = prediction.get())))))))))))))))))))))))"memory_prediction") or self.hardware_properties.get())))))))))))))))))))))))hardware, {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}).get())))))))))))))))))))))))"memory_capacity", 8.0)
            
            # Only add predictions with valid main metric values
            if prediction[]]]]]]]]],,,,,,,,,"predicted_value"] is not None:
                predictions.append())))))))))))))))))))))))prediction)
        
                return predictions
    
                def _calculate_combined_scores())))))))))))))))))))))))
                self,
                predictions: List[]]]]]]]]],,,,,,,,,Dict[]]]]]]]]],,,,,,,,,str, Any]],
                optimization_metric: str,
                power_constrained: bool = False,
                memory_constrained: bool = False,
                deployment_constrained: bool = False
    ) -> List[]]]]]]]]],,,,,,,,,Dict[]]]]]]]]],,,,,,,,,str, Any]]:
        """
        Calculate combined scores for all predictions based on multiple factors.
        
        Args:
            predictions: List of predictions for each hardware platform
            optimization_metric: Main metric to optimize for
            power_constrained: Whether power consumption is a primary constraint
            memory_constrained: Whether memory usage is a primary constraint
            deployment_constrained: Whether deployment complexity is a constraint
            
        Returns:
            List of predictions with combined scores
            """
        # Create a copy to avoid modifying the original
            scored_predictions = []]]]]]]]],,,,,,,,,]
        
        # Get min/max values for normalization
            metric_values = []]]]]]]]],,,,,,,,,p[]]]]]]]]],,,,,,,,,"predicted_value"] for p in predictions if p[]]]]]]]]],,,,,,,,,"predicted_value"] is not None]
            power_values = []]]]]]]]],,,,,,,,,p[]]]]]]]]],,,,,,,,,"power_estimate"] for p in predictions if p[]]]]]]]]],,,,,,,,,"power_estimate"] is not None]
            memory_values = []]]]]]]]],,,,,,,,,p[]]]]]]]]],,,,,,,,,"memory_estimate"] for p in predictions if p[]]]]]]]]],,,,,,,,,"memory_estimate"] is not None]
            efficiency_values = []]]]]]]]],,,,,,,,,p[]]]]]]]]],,,,,,,,,"power_efficiency"] for p in predictions if p[]]]]]]]]],,,,,,,,,"power_efficiency"] is not None]
        :
        if not metric_values:
            logger.warning())))))))))))))))))))))))"No valid metric values found")
            return predictions
        
        # Get min/max values with safety checks
            min_metric = min())))))))))))))))))))))))metric_values) if metric_values else 0.0
            max_metric = max())))))))))))))))))))))))metric_values) if metric_values else 1.0
            min_power = min())))))))))))))))))))))))power_values) if power_values else 0.0
            max_power = max())))))))))))))))))))))))power_values) if power_values else 1.0
            min_memory = min())))))))))))))))))))))))memory_values) if memory_values else 0.0
            max_memory = max())))))))))))))))))))))))memory_values) if memory_values else 1.0
            min_efficiency = min())))))))))))))))))))))))efficiency_values) if efficiency_values else 0.0
            max_efficiency = max())))))))))))))))))))))))efficiency_values) if efficiency_values else 1.0
        
        # Prevent division by zero:
        if max_metric == min_metric:
            max_metric = min_metric + 1.0
        if max_power == min_power:
            max_power = min_power + 1.0
        if max_memory == min_memory:
            max_memory = min_memory + 1.0
        if max_efficiency == min_efficiency:
            max_efficiency = min_efficiency + 1.0
        
        # Calculate scores for each prediction
        for pred in predictions:
            pred_copy = pred.copy()))))))))))))))))))))))))
            
            # Skip predictions with invalid values
            if pred[]]]]]]]]],,,,,,,,,"predicted_value"] is None:
            continue
            
            # Initialize component scores
            performance_score = 0.0
            power_score = 0.0
            memory_score = 0.0
            efficiency_score = 0.0
            availability_score = 0.0
            deployment_score = 0.0
            
            # Calculate normalized performance score
            # For throughput, higher is better
            # For latency and memory, lower is better
            if optimization_metric == "throughput":
                performance_score = ())))))))))))))))))))))))pred[]]]]]]]]],,,,,,,,,"predicted_value"] - min_metric) / ())))))))))))))))))))))))max_metric - min_metric)
            elif optimization_metric in []]]]]]]]],,,,,,,,,"latency", "memory"]:
                performance_score = 1.0 - ())))))))))))))))))))))))pred[]]]]]]]]],,,,,,,,,"predicted_value"] - min_metric) / ())))))))))))))))))))))))max_metric - min_metric)
            
            # Apply confidence weighting to performance score
                performance_score *= min())))))))))))))))))))))))1.0, max())))))))))))))))))))))))0.5, pred[]]]]]]]]],,,,,,,,,"confidence"]))
            
            # Calculate power score ())))))))))))))))))))))))lower is better)
            if pred[]]]]]]]]],,,,,,,,,"power_estimate"] is not None:
                power_score = 1.0 - ())))))))))))))))))))))))pred[]]]]]]]]],,,,,,,,,"power_estimate"] - min_power) / ())))))))))))))))))))))))max_power - min_power)
            
            # Calculate memory score ())))))))))))))))))))))))lower is better)
            if pred[]]]]]]]]],,,,,,,,,"memory_estimate"] is not None:
                memory_score = 1.0 - ())))))))))))))))))))))))pred[]]]]]]]]],,,,,,,,,"memory_estimate"] - min_memory) / ())))))))))))))))))))))))max_memory - min_memory)
            
            # Calculate efficiency score ())))))))))))))))))))))))higher is better)
            if pred[]]]]]]]]],,,,,,,,,"power_efficiency"] is not None:
                efficiency_score = ())))))))))))))))))))))))pred[]]]]]]]]],,,,,,,,,"power_efficiency"] - min_efficiency) / ())))))))))))))))))))))))max_efficiency - min_efficiency)
            
            # Calculate availability score
                availability_score = pred.get())))))))))))))))))))))))"hw_availability", 0.5)
            
            # Calculate deployment score ())))))))))))))))))))))))lower complexity is better)
                deployment_score = 1.0 - pred.get())))))))))))))))))))))))"hw_deployment_complexity", 0.5)
            
            # Calculate combined score with weighted components
            # Adjust weights based on constraints
                perf_weight = self.cost_weights[]]]]]]]]],,,,,,,,,"performance"]
                power_weight = self.cost_weights[]]]]]]]]],,,,,,,,,"power_efficiency"] * ())))))))))))))))))))))))2.0 if power_constrained else 1.0)
                memory_weight = self.cost_weights[]]]]]]]]],,,,,,,,,"memory_usage"] * ())))))))))))))))))))))))2.0 if memory_constrained else 1.0)
                avail_weight = self.cost_weights[]]]]]]]]],,,,,,,,,"availability"]
                deploy_weight = 0.1 * ())))))))))))))))))))))))2.0 if deployment_constrained else 1.0)
            
            # Normalize weights
                weight_sum = perf_weight + power_weight + memory_weight + avail_weight + deploy_weight
                perf_weight /= weight_sum
                power_weight /= weight_sum
                memory_weight /= weight_sum
                avail_weight /= weight_sum
                deploy_weight /= weight_sum
            
            # Calculate final score
                combined_score = ())))))))))))))))))))))))
                perf_weight * performance_score +
                power_weight * ())))))))))))))))))))))))0.5 * power_score + 0.5 * efficiency_score) +
                memory_weight * memory_score +
                avail_weight * availability_score +
                deploy_weight * deployment_score
                )
            
            # Add scores to prediction
                pred_copy[]]]]]]]]],,,,,,,,,"performance_score"] = performance_score
                pred_copy[]]]]]]]]],,,,,,,,,"power_score"] = power_score
                pred_copy[]]]]]]]]],,,,,,,,,"memory_score"] = memory_score
                pred_copy[]]]]]]]]],,,,,,,,,"efficiency_score"] = efficiency_score
                pred_copy[]]]]]]]]],,,,,,,,,"availability_score"] = availability_score
                pred_copy[]]]]]]]]],,,,,,,,,"deployment_score"] = deployment_score
                pred_copy[]]]]]]]]],,,,,,,,,"combined_score"] = combined_score
            
            # Calculate score components for explanation
            pred_copy[]]]]]]]]],,,,,,,,,"score_components"] = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}:
                "performance": perf_weight * performance_score,
                "power_efficiency": power_weight * ())))))))))))))))))))))))0.5 * power_score + 0.5 * efficiency_score),
                "memory_usage": memory_weight * memory_score,
                "availability": avail_weight * availability_score,
                "deployment": deploy_weight * deployment_score
                }
            
                scored_predictions.append())))))))))))))))))))))))pred_copy)
        
                return scored_predictions
    
                def batch_recommend())))))))))))))))))))))))
                self,
                models: List[]]]]]]]]],,,,,,,,,Dict[]]]]]]]]],,,,,,,,,str, Any]],
                optimization_metric: str = "throughput",
                precision_format: str = "FP32",
                power_constrained: bool = False,
                memory_constrained: bool = False,
                deployment_constrained: bool = False,
                custom_constraints: Optional[]]]]]]]]],,,,,,,,,Dict[]]]]]]]]],,,,,,,,,str, Any]] = None
    ) -> Dict[]]]]]]]]],,,,,,,,,str, Any]:
        """
        Recommend hardware platforms for multiple models in a batch.
        
        Args:
            models: List of dictionaries with model_name, model_type, and batch_size
            optimization_metric: Performance metric to optimize for
            precision_format: Precision format to use
            power_constrained: Whether power consumption is a primary constraint
            memory_constrained: Whether memory usage is a primary constraint
            deployment_constrained: Whether deployment complexity is a constraint
            custom_constraints: Dictionary of custom constraints to apply
            
        Returns:
            Dictionary with recommendations for each model
            """
            logger.info())))))))))))))))))))))))f"Batch recommending hardware for {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}len())))))))))))))))))))))))models)} models")
        
        # Get recommendations for each model
            recommendations = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        for model_config in models:
            model_name = model_config[]]]]]]]]],,,,,,,,,"model_name"]
            model_type = model_config.get())))))))))))))))))))))))"model_type", "unknown")
            batch_size = model_config.get())))))))))))))))))))))))"batch_size", 1)
            
            try:
                recommendation = self.recommend_hardware())))))))))))))))))))))))
                model_name=model_name,
                model_type=model_type,
                batch_size=batch_size,
                optimization_metric=optimization_metric,
                precision_format=precision_format,
                power_constrained=power_constrained,
                memory_constrained=memory_constrained,
                deployment_constrained=deployment_constrained,
                custom_constraints=custom_constraints,
            return_all_candidates=False
            )
            recommendations[]]]]]]]]],,,,,,,,,model_name] = recommendation
            except Exception as e:
                logger.error())))))))))))))))))))))))f"Error recommending hardware for {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}model_name}: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))))))))))))))))))e)}")
                recommendations[]]]]]]]]],,,,,,,,,model_name] = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "error": str())))))))))))))))))))))))e),
                "model_name": model_name,
                "model_type": model_type,
                "batch_size": batch_size
                }
        
        # Create summary of recommendations
                summary = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "total_models": len())))))))))))))))))))))))models),
            "successful_recommendations": sum())))))))))))))))))))))))1 for r in recommendations.values())))))))))))))))))))))))) if "error" not in r),:
            "failed_recommendations": sum())))))))))))))))))))))))1 for r in recommendations.values())))))))))))))))))))))))) if "error" in r),:
                "hardware_distribution": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},
                "average_scores": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},
                "constraints_applied": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "power_constrained": power_constrained,
                "memory_constrained": memory_constrained,
                "deployment_constrained": deployment_constrained,
                "custom_constraints": custom_constraints
                }
                }
        
        # Calculate hardware distribution
        for r in recommendations.values())))))))))))))))))))))))):
            if "error" not in r and "recommendation" in r:
                hw = r[]]]]]]]]],,,,,,,,,"recommendation"][]]]]]]]]],,,,,,,,,"hardware_platform"]
                summary[]]]]]]]]],,,,,,,,,"hardware_distribution"][]]]]]]]]],,,,,,,,,hw] = summary[]]]]]]]]],,,,,,,,,"hardware_distribution"].get())))))))))))))))))))))))hw, 0) + 1
        
        # Calculate average scores
                score_sums = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                score_counts = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        for r in recommendations.values())))))))))))))))))))))))):
            if "error" not in r and "recommendation" in r:
                for key in []]]]]]]]],,,,,,,,,"combined_score", "estimated_value", "power_efficiency"]:
                    if key in r[]]]]]]]]],,,,,,,,,"recommendation"]:
                        score_sums[]]]]]]]]],,,,,,,,,key] = score_sums.get())))))))))))))))))))))))key, 0.0) + r[]]]]]]]]],,,,,,,,,"recommendation"][]]]]]]]]],,,,,,,,,key]
                        score_counts[]]]]]]]]],,,,,,,,,key] = score_counts.get())))))))))))))))))))))))key, 0) + 1
        
        for key, total in score_sums.items())))))))))))))))))))))))):
            count = score_counts[]]]]]]]]],,,,,,,,,key]
            if count > 0:
                summary[]]]]]]]]],,,,,,,,,"average_scores"][]]]]]]]]],,,,,,,,,key] = total / count
        
        # Return results
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "recommendations": recommendations,
            "summary": summary
            }
    
            def generate_hardware_comparison_chart())))))))))))))))))))))))
            self,
            model_name: str,
            model_type: str,
            batch_sizes: List[]]]]]]]]],,,,,,,,,int],
            metric: str = "throughput",
            precision_format: str = "FP32",
            hardware_platforms: Optional[]]]]]]]]],,,,,,,,,List[]]]]]]]]],,,,,,,,,str]] = None,
            output_path: Optional[]]]]]]]]],,,,,,,,,str] = None,
            chart_title: Optional[]]]]]]]]],,,,,,,,,str] = None,
            include_power_efficiency: bool = True,
            style: str = "darkgrid"
    ) -> Dict[]]]]]]]]],,,,,,,,,str, Any]:
        """
        Generate a comparison chart for hardware platforms across different batch sizes.
        
        Args:
            model_name: Name of the model
            model_type: Type of the model
            batch_sizes: List of batch sizes to compare
            metric: Performance metric to compare
            precision_format: Precision format to use
            hardware_platforms: List of hardware platforms to compare ())))))))))))))))))))))))None for all available)
            output_path: Path to save the chart ())))))))))))))))))))))))None for no saving)
            chart_title: Custom title for the chart ())))))))))))))))))))))))None for auto-generated)
            include_power_efficiency: Whether to include power efficiency in the comparison
            style: Visual style for the chart ())))))))))))))))))))))))darkgrid, whitegrid, dark, white, ticks)
            
        Returns:
            Dictionary with chart data and metadata
            """
            logger.info())))))))))))))))))))))))f"Generating hardware comparison chart for model '{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}model_name}' "
            f"with {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}len())))))))))))))))))))))))batch_sizes)} batch sizes")
        
        # Use specified hardware platforms or all available
            hardware_platforms = hardware_platforms or self.available_hardware
        
        # Collect predictions for all combinations
            predictions = []]]]]]]]],,,,,,,,,]
        for batch_size in batch_sizes:
            batch_predictions = self._get_predictions_for_hardware())))))))))))))))))))))))
            model_name=model_name,
            model_type=model_type,
            batch_size=batch_size,
            hardware_platforms=hardware_platforms,
            precision_format=precision_format,
            metrics=[]]]]]]]]],,,,,,,,,metric, "power"]  # Always include power for efficiency
            )
            
            # Skip batch if error:
            if isinstance())))))))))))))))))))))))batch_predictions, dict) and "error" in batch_predictions:
                logger.error())))))))))))))))))))))))f"Error getting predictions for batch size {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}batch_size}: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}batch_predictions[]]]]]]]]],,,,,,,,,'error']}")
            continue
            
            # Add batch size to predictions
            for pred in batch_predictions:
                pred[]]]]]]]]],,,,,,,,,"batch_size"] = batch_size
            
                predictions.extend())))))))))))))))))))))))batch_predictions)
        
        # Convert to DataFrame for easier plotting
                data = []]]]]]]]],,,,,,,,,]
        for pred in predictions:
            if pred[]]]]]]]]],,,,,,,,,"predicted_value"] is not None:
                row = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "Hardware": pred[]]]]]]]]],,,,,,,,,"hardware_platform"],
                "Batch Size": pred[]]]]]]]]],,,,,,,,,"batch_size"],
                metric.title())))))))))))))))))))))))): pred[]]]]]]]]],,,,,,,,,"predicted_value"],
                "Confidence": pred[]]]]]]]]],,,,,,,,,"confidence"]
                }
                
                # Add power efficiency if available: and requested:
                if include_power_efficiency and pred[]]]]]]]]],,,,,,,,,"power_efficiency"] is not None:
                    row[]]]]]]]]],,,,,,,,,"Power Efficiency"] = pred[]]]]]]]]],,,,,,,,,"power_efficiency"]
                
                    data.append())))))))))))))))))))))))row)
        
                    df = pd.DataFrame())))))))))))))))))))))))data)
        
        # Check if we have data:
        if len())))))))))))))))))))))))df) == 0:
            logger.error())))))))))))))))))))))))"No valid prediction data available for chart")
                    return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                    "error": "No valid prediction data available for chart",
                    "model_name": model_name,
                    "model_type": model_type,
                    "batch_sizes": batch_sizes,
                    "metric": metric
                    }
        
        # Set seaborn style
                    sns.set_style())))))))))))))))))))))))style)
        
        # Create figure
                    fig, axes = plt.subplots())))))))))))))))))))))))nrows=1, ncols=2 if include_power_efficiency else 1,
                    figsize=())))))))))))))))))))))))15 if include_power_efficiency else 10, 6))
        
        # Plot metric comparison:
        if include_power_efficiency:
            metric_ax = axes[]]]]]]]]],,,,,,,,,0]
        else:
            metric_ax = axes
            
        # Plot metric by batch size for each hardware
            sns.lineplot())))))))))))))))))))))))
            data=df,
            x="Batch Size",
            y=metric.title())))))))))))))))))))))))),
            hue="Hardware",
            style="Hardware",
            markers=True,
            dashes=False,
            ax=metric_ax
            )
        
        # Set metric plot labels and title
            metric_title = f"{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}metric.title()))))))))))))))))))))))))} by Batch Size"
            metric_ax.set_title())))))))))))))))))))))))metric_title)
            metric_ax.set_xlabel())))))))))))))))))))))))"Batch Size")
            metric_ax.set_ylabel())))))))))))))))))))))))metric.title())))))))))))))))))))))))))
            metric_ax.grid())))))))))))))))))))))))True, linestyle="--", alpha=0.7)
        
        # Plot power efficiency if requested::
        if include_power_efficiency and "Power Efficiency" in df.columns:
            efficiency_ax = axes[]]]]]]]]],,,,,,,,,1]
            
            sns.lineplot())))))))))))))))))))))))
            data=df,
            x="Batch Size",
            y="Power Efficiency",
            hue="Hardware",
            style="Hardware",
            markers=True,
            dashes=False,
            ax=efficiency_ax
            )
            
            # Set efficiency plot labels and title
            efficiency_title = "Power Efficiency by Batch Size"
            efficiency_ax.set_title())))))))))))))))))))))))efficiency_title)
            efficiency_ax.set_xlabel())))))))))))))))))))))))"Batch Size")
            efficiency_ax.set_ylabel())))))))))))))))))))))))"Performance per Watt")
            efficiency_ax.grid())))))))))))))))))))))))True, linestyle="--", alpha=0.7)
        
        # Set overall figure title
        if chart_title:
            fig.suptitle())))))))))))))))))))))))chart_title, fontsize=16)
        else:
            fig.suptitle())))))))))))))))))))))))f"Hardware Comparison for {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}model_name} ()))))))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}precision_format})", fontsize=16)
        
        # Adjust layout
            fig.tight_layout()))))))))))))))))))))))))
        if chart_title:
            plt.subplots_adjust())))))))))))))))))))))))top=0.9)
        
        # Save chart if output path provided:
        if output_path:
            plt.savefig())))))))))))))))))))))))output_path, dpi=300, bbox_inches="tight")
            logger.info())))))))))))))))))))))))f"Saved hardware comparison chart to {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}output_path}")
        
        # Create result dictionary
            result = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "model_name": model_name,
            "model_type": model_type,
            "batch_sizes": batch_sizes,
            "metric": metric,
            "hardware_platforms": hardware_platforms,
            "data": df.to_dict())))))))))))))))))))))))orient="records"),
            "chart_saved": output_path is not None,
            "output_path": output_path
            }
        
            return result
    
            def generate_recommendation_report())))))))))))))))))))))))
            self,
            models: List[]]]]]]]]],,,,,,,,,Dict[]]]]]]]]],,,,,,,,,str, Any]],
            optimization_metric: str = "throughput",
            include_charts: bool = True,
            output_dir: Optional[]]]]]]]]],,,,,,,,,str] = None,
            output_format: str = "html",
            power_constrained: bool = False,
            memory_constrained: bool = False
    ) -> Dict[]]]]]]]]],,,,,,,,,str, Any]:
        """
        Generate a comprehensive hardware recommendation report for multiple models.
        
        Args:
            models: List of dictionaries with model_name, model_type, and batch_size
            optimization_metric: Performance metric to optimize for
            include_charts: Whether to include charts in the report
            output_dir: Directory to save the report and charts ())))))))))))))))))))))))None for no saving)
            output_format: Format for the report ())))))))))))))))))))))))html, markdown, json)
            power_constrained: Whether power consumption is a primary constraint
            memory_constrained: Whether memory usage is a primary constraint
            
        Returns:
            Dictionary with report data and metadata
            """
            logger.info())))))))))))))))))))))))f"Generating hardware recommendation report for {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}len())))))))))))))))))))))))models)} models")
        
        # Get batch recommendations for all models
            recommendations = self.batch_recommend())))))))))))))))))))))))
            models=models,
            optimization_metric=optimization_metric,
            power_constrained=power_constrained,
            memory_constrained=memory_constrained
            )
        
        # Generate charts if requested::
            charts = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        if include_charts and output_dir:
            os.makedirs())))))))))))))))))))))))output_dir, exist_ok=True)
            
            for model_config in models:
                model_name = model_config[]]]]]]]]],,,,,,,,,"model_name"]
                model_type = model_config.get())))))))))))))))))))))))"model_type", "unknown")
                batch_size = model_config.get())))))))))))))))))))))))"batch_size", 1)
                
                # Generate batch sizes around the specified one
                batch_sizes = []]]]]]]]],,,,,,,,,max())))))))))))))))))))))))1, batch_size // 2), batch_size, batch_size * 2]
                
                try:
                    chart_path = os.path.join())))))))))))))))))))))))output_dir, f"{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}model_name.replace())))))))))))))))))))))))'/', '_')}_comparison.png")
                    chart_data = self.generate_hardware_comparison_chart())))))))))))))))))))))))
                    model_name=model_name,
                    model_type=model_type,
                    batch_sizes=batch_sizes,
                    metric=optimization_metric,
                    output_path=chart_path
                    )
                    charts[]]]]]]]]],,,,,,,,,model_name] = chart_data
                except Exception as e:
                    logger.error())))))))))))))))))))))))f"Error generating chart for {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}model_name}: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))))))))))))))))))e)}")
        
        # Create report data
                    report_data = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                    "title": "Hardware Recommendation Report",
                    "date": pd.Timestamp.now())))))))))))))))))))))))).strftime())))))))))))))))))))))))"%Y-%m-%d %H:%M:%S"),
                    "optimization_metric": optimization_metric,
                    "constraints": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                    "power_constrained": power_constrained,
                    "memory_constrained": memory_constrained
                    },
                    "models_analyzed": len())))))))))))))))))))))))models),
                    "recommendations": recommendations,
                    "charts": charts if include_charts else {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                    }
        
        # Save report if output directory provided:
        if output_dir:
            os.makedirs())))))))))))))))))))))))output_dir, exist_ok=True)
            
            if output_format == "json":
                # Save JSON report
                json_path = os.path.join())))))))))))))))))))))))output_dir, "hardware_recommendations.json")
                with open())))))))))))))))))))))))json_path, "w") as f:
                    # Convert non-serializable objects to strings
                    json_report = self._prepare_for_serialization())))))))))))))))))))))))report_data)
                    json.dump())))))))))))))))))))))))json_report, f, indent=2)
                    logger.info())))))))))))))))))))))))f"Saved JSON report to {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}json_path}")
                    report_data[]]]]]]]]],,,,,,,,,"report_path"] = json_path
            
            elif output_format == "html":
                # Generate HTML report
                html_path = os.path.join())))))))))))))))))))))))output_dir, "hardware_recommendations.html")
                self._generate_html_report())))))))))))))))))))))))report_data, html_path)
                logger.info())))))))))))))))))))))))f"Saved HTML report to {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}html_path}")
                report_data[]]]]]]]]],,,,,,,,,"report_path"] = html_path
            
            elif output_format == "markdown":
                # Generate Markdown report
                md_path = os.path.join())))))))))))))))))))))))output_dir, "hardware_recommendations.md")
                self._generate_markdown_report())))))))))))))))))))))))report_data, md_path)
                logger.info())))))))))))))))))))))))f"Saved Markdown report to {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}md_path}")
                report_data[]]]]]]]]],,,,,,,,,"report_path"] = md_path
        
                return report_data
    
    def _prepare_for_serialization())))))))))))))))))))))))self, data):
        """Prepare data for JSON serialization by converting non-serializable objects."""
        if isinstance())))))))))))))))))))))))data, dict):
        return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}k: self._prepare_for_serialization())))))))))))))))))))))))v) for k, v in data.items()))))))))))))))))))))))))}
        elif isinstance())))))))))))))))))))))))data, list):
            return []]]]]]]]],,,,,,,,,self._prepare_for_serialization())))))))))))))))))))))))item) for item in data]:
        elif isinstance())))))))))))))))))))))))data, ())))))))))))))))))))))))np.int64, np.int32, np.int16, np.int8)):
                return int())))))))))))))))))))))))data)
        elif isinstance())))))))))))))))))))))))data, ())))))))))))))))))))))))np.float64, np.float32, np.float16)):
                return float())))))))))))))))))))))))data)
        elif isinstance())))))))))))))))))))))))data, ())))))))))))))))))))))))pd.DataFrame, pd.Series)):
                return data.to_dict()))))))))))))))))))))))))
        elif hasattr())))))))))))))))))))))))data, 'tolist'):
                return data.tolist()))))))))))))))))))))))))
        else:
                return data
    
    def _generate_html_report())))))))))))))))))))))))self, report_data, output_path):
        """Generate HTML report and save to output path."""
        # Create HTML content
        html_content = """<!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Hardware Recommendation Report</title>
        <style>
        body {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        font-family: Arial, sans-serif;
        line-height: 1.6;
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
        color: #333;
        }
        h1, h2, h3 {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        color: #2c3e50;
        }
        .report-header {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        margin-bottom: 30px;
        border-bottom: 1px solid #eee;
        padding-bottom: 20px;
        }
        .summary-section {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        }
        .recommendation-card {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 20px;
        background-color: #fff;
        }
        .recommendation-header {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        border-bottom: 1px solid #eee;
        padding-bottom: 10px;
        margin-bottom: 15px;
        display: flex;
        justify-content: space-between;
        }
        .recommendation-body {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        display: flex;
        flex-wrap: wrap;
        }
        .recommendation-details {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        flex: 1;
        min-width: 300px;
        }
        .recommendation-chart {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        flex: 2;
        min-width: 400px;
        text-align: center;
        }
        .alternatives-section {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        margin-top: 15px;
        padding-top: 15px;
        border-top: 1px solid #eee;
        }
        .alternative-item {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        padding: 8px;
        margin-bottom: 5px;
        background-color: #f5f5f5;
        border-radius: 3px;
        }
        table {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 15px;
        }
        table, th, td {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        border: 1px solid #ddd;
        }
        th, td {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        padding: 8px;
        text-align: left;
        }
        th {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        background-color: #f2f2f2;
        }
        .metric-good {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        color: #27ae60;
        font-weight: bold;
        }
        .metric-average {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        color: #f39c12;
        font-weight: bold;
        }
        .metric-poor {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        color: #e74c3c;
        font-weight: bold;
        }
        .confidence-indicator {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 5px;
        }
        .confidence-high {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        background-color: #27ae60;
        }
        .confidence-medium {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        background-color: #f39c12;
        }
        .confidence-low {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        background-color: #e74c3c;
        }
        </style>
        </head>
        <body>
        <div class="report-header">
        <h1>Hardware Recommendation Report</h1>
        <p><strong>Generated:</strong> {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}date}</p>
        <p><strong>Optimization Metric:</strong> {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}metric}</p>
        <p><strong>Constraints:</strong> 
        Power-constrained: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}power_constrained},
        Memory-constrained: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}memory_constrained}
        </p>
        </div>

        <div class="summary-section">
        <h2>Summary</h2>
        <p><strong>Models Analyzed:</strong> {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}models_analyzed}</p>
        <p><strong>Successful Recommendations:</strong> {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}successful_recommendations}</p>
        <p><strong>Failed Recommendations:</strong> {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}failed_recommendations}</p>
        
        <h3>Hardware Distribution</h3>
        <table>
        <tr>
        <th>Hardware Platform</th>
        <th>Count</th>
        <th>Percentage</th>
        </tr>
        {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}hardware_distribution_rows}
        </table>
        </div>

        <h2>Recommendations by Model</h2>
        {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}recommendation_cards}
        </body>
        </html>
        """
        
        # Format summary information
        recommendations = report_data[]]]]]]]]],,,,,,,,,"recommendations"]
        summary = recommendations[]]]]]]]]],,,,,,,,,"summary"]
        
        # Format hardware distribution rows
        hardware_distribution = summary[]]]]]]]]],,,,,,,,,"hardware_distribution"]
        total_recommendations = summary[]]]]]]]]],,,,,,,,,"successful_recommendations"]
        
        hardware_distribution_rows = ""
        for hw, count in hardware_distribution.items())))))))))))))))))))))))):
            percentage = ())))))))))))))))))))))))count / total_recommendations) * 100 if total_recommendations > 0 else 0:
                hardware_distribution_rows += f"""
                <tr>
                <td>{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}hw}</td>
                <td>{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}count}</td>:
                    <td>{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}percentage:.1f}%</td>
                    </tr>
                    """
        
        # Format recommendation cards
                    recommendation_cards = ""
        for model_name, rec in recommendations[]]]]]]]]],,,,,,,,,"recommendations"].items())))))))))))))))))))))))):
            if "error" in rec:
                # Skip failed recommendations
            continue
                
            model_type = rec[]]]]]]]]],,,,,,,,,"model_type"]
            batch_size = rec[]]]]]]]]],,,,,,,,,"batch_size"]
            main_recommendation = rec[]]]]]]]]],,,,,,,,,"recommendation"]
            alternatives = rec.get())))))))))))))))))))))))"alternatives", []]]]]]]]],,,,,,,,,])
            
            # Determine confidence class
            confidence = main_recommendation[]]]]]]]]],,,,,,,,,"confidence"]
            if confidence >= 0.8:
                confidence_class = "confidence-high"
            elif confidence >= 0.5:
                confidence_class = "confidence-medium"
            else:
                confidence_class = "confidence-low"
            
            # Format alternatives
                alternatives_html = ""
            for alt in alternatives:
                alternatives_html += f"""
                <div class="alternative-item">
                <strong>{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}alt[]]]]]]]]],,,,,,,,,"hardware_platform"]}</strong> -
                {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}rec[]]]]]]]]],,,,,,,,,"metric"]}: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}alt[]]]]]]]]],,,,,,,,,"estimated_value"]:.2f},
                Score: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}alt[]]]]]]]]],,,,,,,,,"combined_score"]:.2f},
                Power: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}alt[]]]]]]]]],,,,,,,,,"power_estimate"]:.1f}W,
                Memory: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}alt[]]]]]]]]],,,,,,,,,"memory_estimate"]:.1f}GB
                </div>
                """
            
            # Check if chart is available
            chart_html = "":
            if model_name in report_data[]]]]]]]]],,,,,,,,,"charts"] and "output_path" in report_data[]]]]]]]]],,,,,,,,,"charts"][]]]]]]]]],,,,,,,,,model_name]:
                chart_path = report_data[]]]]]]]]],,,,,,,,,"charts"][]]]]]]]]],,,,,,,,,model_name][]]]]]]]]],,,,,,,,,"output_path"]
                # Get just the filename for the HTML
                chart_filename = os.path.basename())))))))))))))))))))))))chart_path)
                chart_html = f"""
                <div class="recommendation-chart">
                <img src="{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}chart_filename}" alt="Hardware comparison for {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}model_name}" style="max-width:100%;">
                </div>
                """
            
            # Format the recommendation card
                recommendation_cards += f"""
                <div class="recommendation-card">
                <div class="recommendation-header">
                <h3>{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}model_name}</h3>
                <div>Type: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}model_type}, Batch Size: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}batch_size}</div>
                </div>
                
                <div class="recommendation-body">
                <div class="recommendation-details">
                <h4>
                <span class="confidence-indicator {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}confidence_class}"></span>
                Recommended Hardware: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}main_recommendation[]]]]]]]]],,,,,,,,,"hardware_platform"]}
                </h4>
                        
                <table>
                <tr>
                <th>Metric</th>
                <th>Value</th>
                </tr>
                <tr>
                <td>{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}rec[]]]]]]]]],,,,,,,,,"metric"].title()))))))))))))))))))))))))}</td>
                <td>{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}main_recommendation[]]]]]]]]],,,,,,,,,"estimated_value"]:.2f}</td>
                </tr>
                <tr>
                <td>Power Consumption</td>
                <td>{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}main_recommendation[]]]]]]]]],,,,,,,,,"power_estimate"]:.1f}W</td>
                </tr>
                <tr>
                <td>Memory Usage</td>
                <td>{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}main_recommendation[]]]]]]]]],,,,,,,,,"memory_estimate"]:.1f}GB</td>
                </tr>
                <tr>
                <td>Power Efficiency</td>
                <td>{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}main_recommendation[]]]]]]]]],,,,,,,,,"power_efficiency"]:.2f}</td>
                </tr>
                <tr>
                <td>Confidence</td>
                <td>{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}main_recommendation[]]]]]]]]],,,,,,,,,"confidence"]:.2f}</td>
                </tr>
                <tr>
                <td>Combined Score</td>
                <td>{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}main_recommendation[]]]]]]]]],,,,,,,,,"combined_score"]:.2f}</td>
                </tr>
                </table>
                        
                <div class="alternatives-section">
                <h4>Alternative Hardware Options:</h4>
                {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}alternatives_html}
                </div>
                </div>
                    
                {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}chart_html}
                </div>
                </div>
                """
        
        # Replace placeholders in the HTML template
                formatted_html = html_content.format())))))))))))))))))))))))
                date=report_data[]]]]]]]]],,,,,,,,,"date"],
                metric=report_data[]]]]]]]]],,,,,,,,,"optimization_metric"],
                power_constrained="Yes" if report_data[]]]]]]]]],,,,,,,,,"constraints"][]]]]]]]]],,,,,,,,,"power_constrained"] else "No",
                memory_constrained="Yes" if report_data[]]]]]]]]],,,,,,,,,"constraints"][]]]]]]]]],,,,,,,,,"memory_constrained"] else "No",
                models_analyzed=report_data[]]]]]]]]],,,,,,,,,"models_analyzed"],
                successful_recommendations=summary[]]]]]]]]],,,,,,,,,"successful_recommendations"],
                failed_recommendations=summary[]]]]]]]]],,,,,,,,,"failed_recommendations"],
                hardware_distribution_rows=hardware_distribution_rows,
                recommendation_cards=recommendation_cards
                )
        
        # Write to file:
        with open())))))))))))))))))))))))output_path, "w") as f:
            f.write())))))))))))))))))))))))formatted_html)
    
    def _generate_markdown_report())))))))))))))))))))))))self, report_data, output_path):
        """Generate Markdown report and save to output path."""
        # Create Markdown content
        md_content = f"""# Hardware Recommendation Report

        **Generated:** {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}report_data[]]]]]]]]],,,,,,,,,"date"]}
        **Optimization Metric:** {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}report_data[]]]]]]]]],,,,,,,,,"optimization_metric"]}
        **Constraints:**
- Power-constrained: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"Yes" if report_data[]]]]]]]]],,,,,,,,,"constraints"][]]]]]]]]],,,,,,,,,"power_constrained"] else "No"}  :
    - Memory-constrained: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"Yes" if report_data[]]]]]]]]],,,,,,,,,"constraints"][]]]]]]]]],,,,,,,,,"memory_constrained"] else "No"}

## Summary
:
    **Models Analyzed:** {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}report_data[]]]]]]]]],,,,,,,,,"models_analyzed"]}
    **Successful Recommendations:** {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}report_data[]]]]]]]]],,,,,,,,,"recommendations"][]]]]]]]]],,,,,,,,,"summary"][]]]]]]]]],,,,,,,,,"successful_recommendations"]}
    **Failed Recommendations:** {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}report_data[]]]]]]]]],,,,,,,,,"recommendations"][]]]]]]]]],,,,,,,,,"summary"][]]]]]]]]],,,,,,,,,"failed_recommendations"]}

### Hardware Distribution

    | Hardware Platform | Count | Percentage |
    |------------------|-------|------------|
    """
        
        # Add hardware distribution
    hardware_distribution = report_data[]]]]]]]]],,,,,,,,,"recommendations"][]]]]]]]]],,,,,,,,,"summary"][]]]]]]]]],,,,,,,,,"hardware_distribution"]
    total_recommendations = report_data[]]]]]]]]],,,,,,,,,"recommendations"][]]]]]]]]],,,,,,,,,"summary"][]]]]]]]]],,,,,,,,,"successful_recommendations"]
        
        for hw, count in hardware_distribution.items())))))))))))))))))))))))):
            percentage = ())))))))))))))))))))))))count / total_recommendations) * 100 if total_recommendations > 0 else 0:
                md_content += f"| {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}hw} | {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}count} | {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}percentage:.1f}% |\n"
        
                md_content += "\n## Recommendations by Model\n\n"
        
        # Add recommendations for each model
        for model_name, rec in report_data[]]]]]]]]],,,,,,,,,"recommendations"][]]]]]]]]],,,,,,,,,"recommendations"].items())))))))))))))))))))))))):
            if "error" in rec:
                # Skip failed recommendations
            continue
                
            model_type = rec[]]]]]]]]],,,,,,,,,"model_type"]
            batch_size = rec[]]]]]]]]],,,,,,,,,"batch_size"]
            main_recommendation = rec[]]]]]]]]],,,,,,,,,"recommendation"]
            alternatives = rec.get())))))))))))))))))))))))"alternatives", []]]]]]]]],,,,,,,,,])
            
            md_content += f"""### {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}model_name}

            **Type:** {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}model_type}
            **Batch Size:** {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}batch_size}

#### Recommended Hardware: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}main_recommendation[]]]]]]]]],,,,,,,,,"hardware_platform"]}

            | Metric | Value |
            |--------|-------|
            | {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}rec[]]]]]]]]],,,,,,,,,"metric"].title()))))))))))))))))))))))))} | {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}main_recommendation[]]]]]]]]],,,,,,,,,"estimated_value"]:.2f} |
            | Power Consumption | {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}main_recommendation[]]]]]]]]],,,,,,,,,"power_estimate"]:.1f}W |
            | Memory Usage | {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}main_recommendation[]]]]]]]]],,,,,,,,,"memory_estimate"]:.1f}GB |
            | Power Efficiency | {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}main_recommendation[]]]]]]]]],,,,,,,,,"power_efficiency"]:.2f} |
            | Confidence | {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}main_recommendation[]]]]]]]]],,,,,,,,,"confidence"]:.2f} |
            | Combined Score | {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}main_recommendation[]]]]]]]]],,,,,,,,,"combined_score"]:.2f} |

#### Alternative Hardware Options:

            """
            
            # Add alternatives
            for alt in alternatives:
                md_content += f"- **{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}alt[]]]]]]]]],,,,,,,,,'hardware_platform']}** - {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}rec[]]]]]]]]],,,,,,,,,'metric']}: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}alt[]]]]]]]]],,,,,,,,,'estimated_value']:.2f}, Score: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}alt[]]]]]]]]],,,,,,,,,'combined_score']:.2f}, Power: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}alt[]]]]]]]]],,,,,,,,,'power_estimate']:.1f}W, Memory: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}alt[]]]]]]]]],,,,,,,,,'memory_estimate']:.1f}GB\n"
            
                md_content += "\n"
            
            # Add chart reference if available:
            if model_name in report_data[]]]]]]]]],,,,,,,,,"charts"] and "output_path" in report_data[]]]]]]]]],,,,,,,,,"charts"][]]]]]]]]],,,,,,,,,model_name]:
                chart_path = report_data[]]]]]]]]],,,,,,,,,"charts"][]]]]]]]]],,,,,,,,,model_name][]]]]]]]]],,,,,,,,,"output_path"]
                chart_filename = os.path.basename())))))))))))))))))))))))chart_path)
                md_content += f"![]]]]]]]]],,,,,,,,,Hardware comparison for {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}model_name}]()))))))))))))))))))))))){}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}chart_filename})\n\n"
        
        # Write to file
        with open())))))))))))))))))))))))output_path, "w") as f:
            f.write())))))))))))))))))))))))md_content)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser())))))))))))))))))))))))description="Hardware Recommendation System")
    parser.add_argument())))))))))))))))))))))))"--model", required=True, help="Model name to recommend hardware for")
    parser.add_argument())))))))))))))))))))))))"--type", default="unknown", help="Model type")
    parser.add_argument())))))))))))))))))))))))"--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument())))))))))))))))))))))))"--metric", default="throughput", choices=[]]]]]]]]],,,,,,,,,"throughput", "latency", "memory", "power"],
    help="Performance metric to optimize for")
    parser.add_argument())))))))))))))))))))))))"--precision", default="FP32", help="Precision format")
    parser.add_argument())))))))))))))))))))))))"--power-constrained", action="store_true", help="Apply power constraints")
    parser.add_argument())))))))))))))))))))))))"--memory-constrained", action="store_true", help="Apply memory constraints")
    parser.add_argument())))))))))))))))))))))))"--output", help="Output path for chart or report")
    parser.add_argument())))))))))))))))))))))))"--chart", action="store_true", help="Generate a comparison chart")
    
    args = parser.parse_args()))))))))))))))))))))))))
    
    # Create hardware recommender
    try:
        from predict import PerformancePredictor
        predictor = PerformancePredictor()))))))))))))))))))))))))
        recommender = HardwareRecommender())))))))))))))))))))))))predictor=predictor)
        
        # Recommend hardware
        if args.chart:
            # Generate comparison chart
            batch_sizes = []]]]]]]]],,,,,,,,,max())))))))))))))))))))))))1, args.batch_size // 2), args.batch_size, args.batch_size * 2]
            result = recommender.generate_hardware_comparison_chart())))))))))))))))))))))))
            model_name=args.model,
            model_type=args.type,
            batch_sizes=batch_sizes,
            metric=args.metric,
            precision_format=args.precision,
            output_path=args.output
            )
            
            print())))))))))))))))))))))))f"Generated hardware comparison chart for {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}args.model}")
            if args.output:
                print())))))))))))))))))))))))f"Chart saved to: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}args.output}")
        else:
            # Get hardware recommendation
            recommendation = recommender.recommend_hardware())))))))))))))))))))))))
            model_name=args.model,
            model_type=args.type,
            batch_size=args.batch_size,
            optimization_metric=args.metric,
            precision_format=args.precision,
            power_constrained=args.power_constrained,
            memory_constrained=args.memory_constrained
            )
            
            # Print recommendation
            print())))))))))))))))))))))))f"\nRecommended hardware for {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}args.model} ())))))))))))))))))))))))batch size {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}args.batch_size}):")
            print())))))))))))))))))))))))f"  Hardware: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}recommendation[]]]]]]]]],,,,,,,,,'recommendation'][]]]]]]]]],,,,,,,,,'hardware_platform']}")
            print())))))))))))))))))))))))f"  {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}args.metric}: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}recommendation[]]]]]]]]],,,,,,,,,'recommendation'][]]]]]]]]],,,,,,,,,'estimated_value']:.2f}")
            print())))))))))))))))))))))))f"  Power: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}recommendation[]]]]]]]]],,,,,,,,,'recommendation'][]]]]]]]]],,,,,,,,,'power_estimate']:.1f}W")
            print())))))))))))))))))))))))f"  Memory: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}recommendation[]]]]]]]]],,,,,,,,,'recommendation'][]]]]]]]]],,,,,,,,,'memory_estimate']:.1f}GB")
            print())))))))))))))))))))))))f"  Confidence: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}recommendation[]]]]]]]]],,,,,,,,,'recommendation'][]]]]]]]]],,,,,,,,,'confidence']:.2f}")
            
            # Print alternatives
            if recommendation[]]]]]]]]],,,,,,,,,'alternatives']:
                print())))))))))))))))))))))))"\nAlternative options:")
                for i, alt in enumerate())))))))))))))))))))))))recommendation[]]]]]]]]],,,,,,,,,'alternatives'], 1):
                    print())))))))))))))))))))))))f"  {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}i}. {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}alt[]]]]]]]]],,,,,,,,,'hardware_platform']} - {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}args.metric}: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}alt[]]]]]]]]],,,,,,,,,'estimated_value']:.2f}, "
                    f"Power: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}alt[]]]]]]]]],,,,,,,,,'power_estimate']:.1f}W, Memory: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}alt[]]]]]]]]],,,,,,,,,'memory_estimate']:.1f}GB")
            
            # Save to output if specified:
            if args.output:
                import json
                with open())))))))))))))))))))))))args.output, "w") as f:
                    json.dump())))))))))))))))))))))))recommendation, f, indent=2)
                    print())))))))))))))))))))))))f"\nRecommendation saved to: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}args.output}")
    
    except ImportError as e:
        print())))))))))))))))))))))))f"Error: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}str())))))))))))))))))))))))e)}")
        print())))))))))))))))))))))))"Please ensure you have the required modules installed.")
        sys.exit())))))))))))))))))))))))1)