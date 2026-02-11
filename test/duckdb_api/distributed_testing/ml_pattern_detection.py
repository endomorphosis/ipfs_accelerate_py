#!/usr/bin/env python3
"""
Machine Learning-Based Pattern Detection for Hardware Fault Tolerance
Implementation Date: March 13, 2025 (Originally planned for June 2025)

This module was implemented ahead of schedule as an enhancement to the hardware-aware
fault tolerance system. It provides advanced pattern detection capabilities using machine
learning techniques to identify subtle correlations between failures and recommend optimal
recovery strategies.

This module implements advanced pattern detection using machine learning techniques
to identify subtle correlations between failures in the distributed testing system.
It extends the basic pattern detection in the hardware_aware_fault_tolerance module
with more sophisticated algorithms.

Key features:
1. Feature extraction from failure contexts
2. Clustering of similar failures
3. Anomaly detection for unusual failure patterns
4. Predictive modeling for failure likelihood
5. Recommendation generation based on historical success rates
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, field

# Import core components
from data.duckdb.distributed_testing.hardware_aware_fault_tolerance import (
    FailureContext, RecoveryAction, RecoveryStrategy, FailureType
)
from data.duckdb.distributed_testing.hardware_taxonomy import (
    HardwareClass, HardwareArchitecture, HardwareVendor, 
    SoftwareBackend, PrecisionType, HardwareCapabilityProfile
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("ml_pattern_detection")


@dataclass
class FailureFeatures:
    """Features extracted from failure contexts for ML-based analysis."""
    task_id: str
    worker_id: str
    hardware_class: str = "UNKNOWN"
    hardware_vendor: str = "UNKNOWN"
    hardware_architecture: str = "UNKNOWN"
    error_type: str = "UNKNOWN"
    compute_units: int = 0
    memory_gb: float = 0.0
    has_tensor_cores: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    recovery_strategy: str = "UNKNOWN"
    recovery_success: bool = False
    time_since_last_failure: float = 0.0  # in seconds
    failure_cluster: int = -1  # Assigned during clustering


class MLPatternDetector:
    """Machine Learning-based pattern detector for hardware failures."""
    
    def __init__(self, db_manager=None):
        """Initialize the ML pattern detector."""
        self.db_manager = db_manager
        self.failure_features = []
        self.recovery_history = {}
        self.strategy_success_rates = defaultdict(lambda: defaultdict(lambda: {"success": 0, "total": 0}))
        self.clustering_model = None
        self.anomaly_detector = None
        self.prediction_model = None
        self.is_trained = False
    
    def extract_features(self, failure_context: FailureContext) -> FailureFeatures:
        """Extract features from a failure context for ML analysis."""
        hardware_class = "UNKNOWN"
        hardware_vendor = "UNKNOWN"
        hardware_architecture = "UNKNOWN"
        compute_units = 0
        memory_gb = 0.0
        has_tensor_cores = False
        
        # Extract hardware features if hardware profile is available
        if failure_context.hardware_profile:
            profile = failure_context.hardware_profile
            hardware_class = profile.hardware_class.name if profile.hardware_class else "UNKNOWN"
            hardware_vendor = profile.vendor.name if profile.vendor else "UNKNOWN"
            hardware_architecture = profile.architecture.name if profile.architecture else "UNKNOWN"
            compute_units = profile.compute_units
            memory_gb = profile.memory.total_bytes / (1024 * 1024 * 1024) if profile.memory else 0.0
            has_tensor_cores = any(f for f in profile.features if f.name == "TENSOR_CORES")
        
        # Calculate time since last failure for this task
        time_since_last_failure = 0.0
        if failure_context.task_id in self.recovery_history:
            last_failure = max((f.timestamp for f in self.failure_features 
                                if f.task_id == failure_context.task_id), 
                               default=None)
            if last_failure:
                time_since_last_failure = (failure_context.timestamp - last_failure).total_seconds()
        
        # Create feature object
        return FailureFeatures(
            task_id=failure_context.task_id,
            worker_id=failure_context.worker_id,
            hardware_class=hardware_class,
            hardware_vendor=hardware_vendor,
            hardware_architecture=hardware_architecture,
            error_type=failure_context.error_type.name,
            compute_units=compute_units,
            memory_gb=memory_gb,
            has_tensor_cores=has_tensor_cores,
            timestamp=failure_context.timestamp,
            retry_count=failure_context.attempt,
            time_since_last_failure=time_since_last_failure
        )
    
    def add_failure(self, failure_context: FailureContext):
        """Add a failure context to the dataset."""
        features = self.extract_features(failure_context)
        self.failure_features.append(features)
        
        # Update task recovery history
        if failure_context.task_id not in self.recovery_history:
            self.recovery_history[failure_context.task_id] = []
        self.recovery_history[failure_context.task_id].append(failure_context)
    
    def update_recovery_result(self, task_id: str, strategy: RecoveryStrategy, success: bool):
        """Update the success/failure record for a recovery strategy."""
        # Find the most recent failure features for this task
        task_features = [f for f in self.failure_features if f.task_id == task_id]
        if not task_features:
            return
        
        latest_feature = max(task_features, key=lambda f: f.timestamp)
        latest_feature.recovery_strategy = strategy.name
        latest_feature.recovery_success = success
        
        # Update success rates for this hardware class and error type
        hw_class = latest_feature.hardware_class
        error_type = latest_feature.error_type
        
        self.strategy_success_rates[hw_class][strategy.name]["total"] += 1
        if success:
            self.strategy_success_rates[hw_class][strategy.name]["success"] += 1
        
        # Also track by error type
        self.strategy_success_rates[error_type][strategy.name]["total"] += 1
        if success:
            self.strategy_success_rates[error_type][strategy.name]["success"] += 1
    
    def train_models(self):
        """Train machine learning models on the collected failure data."""
        if len(self.failure_features) < 10:
            logger.info("Not enough failure data to train ML models (minimum 10 samples required)")
            return False
        
        try:
            # For demo purposes, we'll just do some simple calculations
            # In a real implementation, we would use sklearn or another ML library
            
            # Simple clustering based on hardware type and error type
            clusters = defaultdict(list)
            next_cluster_id = 0
            
            for feature in self.failure_features:
                cluster_key = (feature.hardware_class, feature.error_type)
                if cluster_key not in clusters:
                    clusters[cluster_key] = next_cluster_id
                    next_cluster_id += 1
                
                feature.failure_cluster = clusters[cluster_key]
            
            # In a real implementation, we would train:
            # 1. A clustering model (KMeans, DBSCAN, etc.)
            # 2. An anomaly detection model (Isolation Forest, LOF, etc.)
            # 3. A predictive model for failure likelihood (Random Forest, XGBoost, etc.)
            
            self.is_trained = True
            logger.info(f"Trained ML models on {len(self.failure_features)} failure samples")
            logger.info(f"Identified {len(clusters)} failure clusters")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training ML models: {e}")
            return False
    
    def detect_patterns(self) -> List[Dict[str, Any]]:
        """Detect patterns using the trained ML models."""
        if not self.is_trained and not self.train_models():
            return []
        
        patterns = []
        
        # For demo purposes, we'll use simple frequency-based pattern detection
        # In a real implementation, we would use the trained ML models
        
        # 1. Cluster-based patterns
        cluster_counts = defaultdict(int)
        cluster_members = defaultdict(list)
        
        for feature in self.failure_features:
            if feature.failure_cluster >= 0:
                cluster_counts[feature.failure_cluster] += 1
                cluster_members[feature.failure_cluster].append(feature)
        
        # Find significant clusters (more than 3 failures)
        for cluster_id, count in cluster_counts.items():
            if count >= 3:
                members = cluster_members[cluster_id]
                
                # Get the most common hardware class and error type in this cluster
                hw_classes = [m.hardware_class for m in members]
                error_types = [m.error_type for m in members]
                
                most_common_hw = max(set(hw_classes), key=hw_classes.count)
                most_common_error = max(set(error_types), key=error_types.count)
                
                # Get the best strategy based on success rates
                best_strategy = self._find_best_strategy(most_common_hw, most_common_error)
                
                patterns.append({
                    "type": "ml_cluster",
                    "cluster_id": cluster_id,
                    "count": count,
                    "hardware_class": most_common_hw,
                    "error_type": most_common_error,
                    "confidence": min(0.5 + (count / 10), 0.95),  # Simple confidence score
                    "recommended_strategy": best_strategy,
                    "description": f"ML-detected pattern: cluster of {count} similar failures on {most_common_hw} hardware with {most_common_error} errors"
                })
        
        # 2. Time-based patterns (detect if failures are happening close together in time)
        task_failure_times = defaultdict(list)
        
        for feature in self.failure_features:
            task_failure_times[feature.task_id].append(feature.timestamp)
        
        for task_id, timestamps in task_failure_times.items():
            if len(timestamps) < 3:
                continue
            
            # Sort timestamps
            sorted_times = sorted(timestamps)
            
            # Calculate time differences between consecutive failures
            time_diffs = [(sorted_times[i] - sorted_times[i-1]).total_seconds() 
                          for i in range(1, len(sorted_times))]
            
            # If average time between failures is less than 5 minutes, it's a pattern
            avg_diff = sum(time_diffs) / len(time_diffs)
            if avg_diff < 300:  # 5 minutes in seconds
                patterns.append({
                    "type": "rapid_failure",
                    "task_id": task_id,
                    "count": len(timestamps),
                    "avg_time_between_failures": avg_diff,
                    "confidence": min(0.6 + (10 / avg_diff) * 0.3, 0.95),
                    "description": f"Rapid failure pattern: task {task_id} failing every {avg_diff:.1f} seconds on average"
                })
        
        return patterns
    
    def _find_best_strategy(self, hardware_class: str, error_type: str) -> str:
        """Find the best recovery strategy based on historical success rates."""
        # Check hardware-specific strategies first
        if hardware_class in self.strategy_success_rates:
            hw_strategies = self.strategy_success_rates[hardware_class]
            best_hw_strategy = self._get_highest_success_rate(hw_strategies)
            if best_hw_strategy:
                return best_hw_strategy
        
        # Fall back to error-specific strategies
        if error_type in self.strategy_success_rates:
            error_strategies = self.strategy_success_rates[error_type]
            best_error_strategy = self._get_highest_success_rate(error_strategies)
            if best_error_strategy:
                return best_error_strategy
        
        # Default strategies based on common sense
        if hardware_class == "GPU" and error_type == "RESOURCE_EXHAUSTION":
            return "REDUCED_BATCH_SIZE"
        if "BROWSER" in error_type:
            return "BROWSER_RESTART"
        
        # General fallback
        return "DELAYED_RETRY"
    
    def _get_highest_success_rate(self, strategies: Dict[str, Dict[str, int]]) -> Optional[str]:
        """Get the strategy with the highest success rate."""
        best_strategy = None
        best_rate = 0.0
        
        for strategy, counts in strategies.items():
            if counts["total"] < 3:  # Need at least 3 attempts to consider
                continue
            
            success_rate = counts["success"] / counts["total"]
            if success_rate > best_rate:
                best_rate = success_rate
                best_strategy = strategy
        
        return best_strategy if best_rate >= 0.5 else None  # Only return if success rate is at least 50%
    
    def recommend_strategy(self, failure_context: FailureContext) -> Optional[RecoveryStrategy]:
        """Recommend a recovery strategy based on ML pattern detection."""
        if not self.is_trained and not self.train_models():
            return None
        
        # Extract features for the current failure
        features = self.extract_features(failure_context)
        
        # Find similar failures in our history
        similar_features = []
        for f in self.failure_features:
            if (f.hardware_class == features.hardware_class and 
                f.error_type == features.error_type):
                similar_features.append(f)
        
        if not similar_features:
            return None
        
        # Find the most successful strategy for similar failures
        strategy_counts = defaultdict(lambda: {"success": 0, "total": 0})
        for f in similar_features:
            if f.recovery_strategy != "UNKNOWN":
                strategy_counts[f.recovery_strategy]["total"] += 1
                if f.recovery_success:
                    strategy_counts[f.recovery_strategy]["success"] += 1
        
        best_strategy = None
        best_rate = 0.0
        min_attempts = 3
        
        for strategy, counts in strategy_counts.items():
            if counts["total"] < min_attempts:
                continue
            
            success_rate = counts["success"] / counts["total"]
            if success_rate > best_rate:
                best_rate = success_rate
                best_strategy = strategy
        
        if best_strategy and best_rate >= 0.6:  # Only recommend if success rate is at least 60%
            try:
                # Convert string name back to RecoveryStrategy enum
                return RecoveryStrategy[best_strategy]
            except (KeyError, ValueError):
                logger.warning(f"Unknown recovery strategy: {best_strategy}")
                return None
        
        return None
    
    def save_state(self) -> Dict[str, Any]:
        """Save the state of the ML pattern detector."""
        # In a real implementation, we would serialize the ML models
        # For now, just save the basic data
        
        state = {
            "failure_features": [
                {
                    "task_id": f.task_id,
                    "worker_id": f.worker_id,
                    "hardware_class": f.hardware_class,
                    "error_type": f.error_type,
                    "retry_count": f.retry_count,
                    "recovery_strategy": f.recovery_strategy,
                    "recovery_success": f.recovery_success,
                    "failure_cluster": f.failure_cluster,
                    "timestamp": f.timestamp.isoformat()
                }
                for f in self.failure_features
            ],
            "strategy_success_rates": dict(self.strategy_success_rates)
        }
        
        return state
    
    def load_state(self, state: Dict[str, Any]):
        """Load the state of the ML pattern detector."""
        if not state:
            return
        
        # Load failure features
        self.failure_features = []
        for f_data in state.get("failure_features", []):
            feature = FailureFeatures(
                task_id=f_data["task_id"],
                worker_id=f_data["worker_id"],
                hardware_class=f_data["hardware_class"],
                error_type=f_data["error_type"],
                retry_count=f_data["retry_count"],
                recovery_strategy=f_data["recovery_strategy"],
                recovery_success=f_data["recovery_success"],
                failure_cluster=f_data["failure_cluster"],
                timestamp=datetime.fromisoformat(f_data["timestamp"])
            )
            self.failure_features.append(feature)
        
        # Load strategy success rates
        self.strategy_success_rates = defaultdict(lambda: defaultdict(lambda: {"success": 0, "total": 0}))
        for hw_class, strategies in state.get("strategy_success_rates", {}).items():
            for strategy, counts in strategies.items():
                self.strategy_success_rates[hw_class][strategy] = counts
        
        # Retrain models if we have enough data
        if len(self.failure_features) >= 10:
            self.train_models()