#!/usr/bin/env python3
"""
Performance baseline manager for regression detection.

This module manages performance baselines for model tests, enabling
automated regression detection across test runs.
"""

import json
import os
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


class PerformanceBaselineManager:
    """Manages performance baselines for regression detection."""
    
    def __init__(self, baseline_file: str = "test/.performance_baselines.json"):
        """Initialize the baseline manager.
        
        Args:
            baseline_file: Path to the baseline storage file
        """
        self.baseline_file = Path(baseline_file)
        self.baselines = self._load_baselines()
    
    def _load_baselines(self) -> Dict[str, Any]:
        """Load baselines from file."""
        if self.baseline_file.exists():
            try:
                with open(self.baseline_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                warnings.warn(f"Failed to load baselines: {e}. Starting with empty baselines.")
                return {}
        return {}
    
    def _save_baselines(self):
        """Save baselines to file."""
        # Ensure directory exists
        self.baseline_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(self.baseline_file, 'w') as f:
                json.dump(self.baselines, f, indent=2)
        except IOError as e:
            warnings.warn(f"Failed to save baselines: {e}")
    
    def get_baseline(self, model_name: str, device: str = "cpu") -> Optional[Dict[str, float]]:
        """Get baseline for a model on a specific device.
        
        Args:
            model_name: Name of the model
            device: Device string (e.g., "cpu", "cuda")
            
        Returns:
            Dictionary with baseline metrics or None if not found
        """
        key = f"{model_name}_{device}"
        return self.baselines.get(key)
    
    def set_baseline(self, model_name: str, metrics: Dict[str, float], device: str = "cpu"):
        """Set baseline for a model on a specific device.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of performance metrics
            device: Device string
        """
        key = f"{model_name}_{device}"
        self.baselines[key] = {
            **metrics,
            "timestamp": datetime.now().isoformat(),
            "device": device
        }
        self._save_baselines()
    
    def check_regression(self, model_name: str, current_metrics: Dict[str, float],
                        device: str = "cpu", tolerance: float = 0.20) -> Dict[str, Any]:
        """Check for performance regressions.
        
        Args:
            model_name: Name of the model
            current_metrics: Current performance metrics
            device: Device string
            tolerance: Allowed deviation (e.g., 0.20 = 20%)
            
        Returns:
            Dictionary with regression analysis
        """
        baseline = self.get_baseline(model_name, device)
        
        if baseline is None:
            return {
                "has_baseline": False,
                "regressions": [],
                "message": f"No baseline found for {model_name} on {device}"
            }
        
        regressions = []
        improvements = []
        
        for metric_name, current_value in current_metrics.items():
            if metric_name not in baseline:
                continue
            
            baseline_value = baseline[metric_name]
            
            # Skip non-numeric values
            if not isinstance(baseline_value, (int, float)) or not isinstance(current_value, (int, float)):
                continue
            
            # Calculate percentage change
            if baseline_value == 0:
                continue
            
            change = (current_value - baseline_value) / baseline_value
            
            # For latency/time metrics, higher is worse
            # For throughput metrics, lower is worse
            # We'll treat all metrics as "lower is better" for simplicity
            # unless metric name contains "throughput" or "rate"
            
            is_throughput = "throughput" in metric_name.lower() or "rate" in metric_name.lower()
            
            if is_throughput:
                # For throughput, decrease is bad
                if change < -tolerance:
                    regressions.append({
                        "metric": metric_name,
                        "baseline": baseline_value,
                        "current": current_value,
                        "change_percent": change * 100,
                        "threshold": -tolerance * 100
                    })
                elif change > tolerance:
                    improvements.append({
                        "metric": metric_name,
                        "baseline": baseline_value,
                        "current": current_value,
                        "change_percent": change * 100
                    })
            else:
                # For latency/memory, increase is bad
                if change > tolerance:
                    regressions.append({
                        "metric": metric_name,
                        "baseline": baseline_value,
                        "current": current_value,
                        "change_percent": change * 100,
                        "threshold": tolerance * 100
                    })
                elif change < -tolerance:
                    improvements.append({
                        "metric": metric_name,
                        "baseline": baseline_value,
                        "current": current_value,
                        "change_percent": change * 100
                    })
        
        return {
            "has_baseline": True,
            "regressions": regressions,
            "improvements": improvements,
            "baseline_timestamp": baseline.get("timestamp", "unknown"),
            "message": self._format_regression_message(regressions, improvements)
        }
    
    def _format_regression_message(self, regressions: List[Dict], improvements: List[Dict]) -> str:
        """Format regression check message.
        
        Args:
            regressions: List of regression details
            improvements: List of improvement details
            
        Returns:
            Formatted message string
        """
        parts = []
        
        if regressions:
            parts.append(f"⚠️  {len(regressions)} performance regression(s) detected:")
            for reg in regressions:
                parts.append(
                    f"  - {reg['metric']}: {reg['baseline']:.4f} → {reg['current']:.4f} "
                    f"({reg['change_percent']:+.1f}%, threshold: {reg['threshold']:+.1f}%)"
                )
        
        if improvements:
            parts.append(f"✅ {len(improvements)} performance improvement(s):")
            for imp in improvements:
                parts.append(
                    f"  - {imp['metric']}: {imp['baseline']:.4f} → {imp['current']:.4f} "
                    f"({imp['change_percent']:+.1f}%)"
                )
        
        if not regressions and not improvements:
            parts.append("✅ Performance within acceptable range")
        
        return "\n".join(parts)
    
    def list_baselines(self, model_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all baselines, optionally filtered by model name.
        
        Args:
            model_filter: Optional model name filter
            
        Returns:
            List of baseline information
        """
        results = []
        for key, baseline in self.baselines.items():
            if model_filter and model_filter not in key:
                continue
            
            model_device = key.split('_', 1)
            results.append({
                "key": key,
                "model": model_device[0] if len(model_device) > 0 else "unknown",
                "device": baseline.get("device", "unknown"),
                "timestamp": baseline.get("timestamp", "unknown"),
                "metrics": {k: v for k, v in baseline.items() 
                           if k not in ["timestamp", "device"]}
            })
        
        return results
    
    def clear_baselines(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """Clear baselines, optionally filtered by model and/or device.
        
        Args:
            model_name: Optional model name filter
            device: Optional device filter
        """
        if model_name is None and device is None:
            # Clear all
            self.baselines = {}
        else:
            # Filter and remove
            keys_to_remove = []
            for key in self.baselines.keys():
                if model_name and not key.startswith(f"{model_name}_"):
                    continue
                if device and not key.endswith(f"_{device}"):
                    continue
                keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.baselines[key]
        
        self._save_baselines()


# Global instance
_baseline_manager = None

def get_baseline_manager() -> PerformanceBaselineManager:
    """Get or create the global baseline manager instance."""
    global _baseline_manager
    if _baseline_manager is None:
        _baseline_manager = PerformanceBaselineManager()
    return _baseline_manager
