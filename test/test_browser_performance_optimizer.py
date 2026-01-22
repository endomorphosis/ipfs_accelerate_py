#!/usr/bin/env python3
"""
Test script for the browser performance optimizer.

This script tests the browser performance optimizer module by simulating
browser history data and verifying optimization recommendations.
"""

import os
import sys
import unittest
import logging
import time
from unittest import mock
from typing import Dict, List, Any, Optional
import json
from enum import Enum

# Add parent directory to path to import browser_performance_optimizer
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the module to test
try:
    from fixed_web_platform.browser_performance_optimizer import (
        BrowserPerformanceOptimizer,
        OptimizationPriority,
        BrowserCapabilityScore,
        OptimizationRecommendation
    )
except ImportError:
    print("Could not import browser_performance_optimizer module")
    # Create mock classes for testing
    class OptimizationPriority(Enum):
        LATENCY = "latency"
        THROUGHPUT = "throughput"
        MEMORY_EFFICIENCY = "memory_efficiency"
        RELIABILITY = "reliability"
        BALANCED = "balanced"
    
    class BrowserCapabilityScore:
        def __init__(self, browser_type, model_type, score, confidence, sample_count, strengths, weaknesses, last_updated):
            self.browser_type = browser_type
            self.model_type = model_type
            self.score = score
            self.confidence = confidence
            self.sample_count = sample_count
            self.strengths = strengths
            self.weaknesses = weaknesses
            self.last_updated = last_updated
    
    class OptimizationRecommendation:
        def __init__(self, browser_type, platform, confidence, parameters, reason, metrics):
            self.browser_type = browser_type
            self.platform = platform
            self.confidence = confidence
            self.parameters = parameters
            self.reason = reason
            self.metrics = metrics
        
        def to_dict(self):
            return {
                "browser": self.browser_type,
                "platform": self.platform,
                "confidence": self.confidence,
                "parameters": self.parameters,
                "reason": self.reason,
                "metrics": self.metrics
            }
    
    class BrowserPerformanceOptimizer:
        def __init__(self, browser_history=None, model_types_config=None, confidence_threshold=0.6, 
                    min_samples_required=5, adaptation_rate=0.25, logger=None):
            self.browser_history = browser_history
            self.model_types_config = model_types_config or {}
            self.confidence_threshold = confidence_threshold
            self.min_samples_required = min_samples_required
            self.adaptation_rate = adaptation_rate
            self.logger = logger or logging.getLogger(__name__)
        
        def get_optimized_configuration(self, model_type, model_name=None, available_browsers=None, user_preferences=None):
            return OptimizationRecommendation(
                browser_type="chrome",
                platform="webgpu",
                confidence=0.7,
                parameters={"batch_size": 1},
                reason="Default recommendation",
                metrics={}
            )
        
        def apply_runtime_optimizations(self, model, browser_type, execution_context):
            return execution_context.copy()

class MockBrowserHistory:
    """Mock browser history for testing."""
    
    def __init__(self, capability_scores=None, recommendations=None, performance_recommendations=None):
        self.capability_scores_data = capability_scores or {}
        self.recommendations_data = recommendations or {}
        self.performance_recommendations_data = performance_recommendations or {}
    
    def get_capability_scores(self, browser=None, model_type=None):
        """Get capability scores for browser/model type."""
        if browser and model_type:
            return self.capability_scores_data
        return self.capability_scores_data
    
    def get_browser_recommendations(self, model_type, model_name=None):
        """Get browser recommendations for model type."""
        if model_type in self.recommendations_data:
            return self.recommendations_data[model_type]
        return {"recommended_browser": "chrome", "recommended_platform": "webgpu", "confidence": 0.5}
    
    def get_performance_recommendations(self):
        """Get performance recommendations."""
        return self.performance_recommendations_data

class MockModel:
    """Mock model for testing."""
    
    def __init__(self, model_type, model_name):
        self.model_type = model_type
        self.model_name = model_name
        
class TestBrowserPerformanceOptimizer(unittest.TestCase):
    """Test cases for the BrowserPerformanceOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger("test_browser_performance_optimizer")
        
        # Create mock browser history
        self.mock_capability_scores = {
            "firefox": {
                "audio": {
                    "score": 90.0,
                    "confidence": 0.8,
                    "sample_size": 20
                },
                "vision": {
                    "score": 70.0,
                    "confidence": 0.6,
                    "sample_size": 15
                },
                "text_embedding": {
                    "score": 60.0,
                    "confidence": 0.5,
                    "sample_size": 10
                }
            },
            "chrome": {
                "audio": {
                    "score": 70.0,
                    "confidence": 0.7,
                    "sample_size": 18
                },
                "vision": {
                    "score": 85.0,
                    "confidence": 0.8,
                    "sample_size": 25
                },
                "text_embedding": {
                    "score": 75.0,
                    "confidence": 0.7,
                    "sample_size": 15
                }
            },
            "edge": {
                "audio": {
                    "score": 65.0,
                    "confidence": 0.6,
                    "sample_size": 12
                },
                "vision": {
                    "score": 70.0,
                    "confidence": 0.7,
                    "sample_size": 14
                },
                "text_embedding": {
                    "score": 85.0,
                    "confidence": 0.9,
                    "sample_size": 30
                }
            }
        }
        
        self.mock_recommendations = {
            "audio": {
                "recommended_browser": "firefox",
                "recommended_platform": "webgpu",
                "confidence": 0.8,
                "reason": "Based on historical performance"
            },
            "vision": {
                "recommended_browser": "chrome",
                "recommended_platform": "webgpu",
                "confidence": 0.9,
                "reason": "Based on historical performance"
            },
            "text_embedding": {
                "recommended_browser": "edge",
                "recommended_platform": "webnn",
                "confidence": 0.85,
                "reason": "Based on historical performance"
            }
        }
        
        self.mock_performance_recommendations = {
            "recommendations": {
                "browser_firefox": {
                    "issue": "high_failure_rate",
                    "description": "Firefox has a high failure rate (15%)",
                    "failure_rate": 0.15,
                    "recommendation": "Consider using a different browser type"
                },
                "model_bert-base": {
                    "issue": "degrading_performance",
                    "description": "Performance is degrading for bert-base",
                    "trend_magnitude": 0.8,
                    "recommendation": "Consider browser type change or hardware upgrade"
                }
            },
            "recommendation_count": 2
        }
        
        self.browser_history = MockBrowserHistory(
            capability_scores=self.mock_capability_scores,
            recommendations=self.mock_recommendations,
            performance_recommendations=self.mock_performance_recommendations
        )
        
        # Create optimizer with mock browser history
        self.optimizer = BrowserPerformanceOptimizer(
            browser_history=self.browser_history,
            model_types_config={
                "text_embedding": {"priority": "latency"},
                "vision": {"priority": "throughput"},
                "audio": {"priority": "memory_efficiency"}
            },
            logger=self.logger
        )
    
    def test_get_optimization_priority(self):
        """Test getting optimization priority."""
        # Test configured priority
        priority = self.optimizer.get_optimization_priority("text_embedding")
        self.assertEqual(priority, OptimizationPriority.LATENCY)
        
        # Test default priority
        priority = self.optimizer.get_optimization_priority("text")
        self.assertEqual(priority, OptimizationPriority.LATENCY)
        
        # Test unknown model type
        priority = self.optimizer.get_optimization_priority("unknown")
        self.assertEqual(priority, OptimizationPriority.BALANCED)
        
        # Test with invalid configuration
        self.optimizer.model_types_config["test"] = {"priority": "invalid"}
        priority = self.optimizer.get_optimization_priority("test")
        self.assertEqual(priority, OptimizationPriority.BALANCED)
    
    def test_get_browser_capability_score(self):
        """Test getting browser capability score."""
        # Test with history data
        score = self.optimizer.get_browser_capability_score("firefox", "audio")
        self.assertEqual(score.browser_type, "firefox")
        self.assertEqual(score.model_type, "audio")
        self.assertGreaterEqual(score.score, 80)  # Should be high for firefox/audio
        
        # Test with predefined capabilities
        score = self.optimizer.get_browser_capability_score("safari", "audio")
        self.assertEqual(score.browser_type, "safari")
        self.assertEqual(score.model_type, "audio")
        self.assertTrue(len(score.strengths) > 0)  # Should have predefined strengths
        
        # Test with unknown browser/model
        score = self.optimizer.get_browser_capability_score("unknown", "unknown")
        self.assertEqual(score.browser_type, "unknown")
        self.assertEqual(score.model_type, "unknown")
        self.assertEqual(score.score, 50.0)  # Default neutral score
    
    def test_get_best_browser_for_model(self):
        """Test getting the best browser for a model."""
        # Test with history data
        browser, confidence, reason = self.optimizer.get_best_browser_for_model(
            "audio", ["firefox", "chrome", "edge"]
        )
        self.assertEqual(browser, "firefox")  # Firefox should be best for audio
        self.assertGreaterEqual(confidence, 0.7)
        
        # Test with single browser
        browser, confidence, reason = self.optimizer.get_best_browser_for_model(
            "audio", ["chrome"]
        )
        self.assertEqual(browser, "chrome")  # Only option
        
        # Test with empty list
        browser, confidence, reason = self.optimizer.get_best_browser_for_model(
            "audio", []
        )
        self.assertEqual(browser, "chrome")  # Default
        self.assertEqual(confidence, 0.0)
    
    def test_get_best_platform_for_browser_model(self):
        """Test getting the best platform for browser/model."""
        # Test with history data
        platform, confidence, reason = self.optimizer.get_best_platform_for_browser_model(
            "edge", "text_embedding"
        )
        self.assertEqual(platform, "webnn")  # Edge should use WebNN for text
        self.assertGreaterEqual(confidence, 0.7)
        
        # Test with default preferences
        platform, confidence, reason = self.optimizer.get_best_platform_for_browser_model(
            "firefox", "vision"
        )
        self.assertEqual(platform, "webgpu")  # Default for Firefox
        
        # Test with unknown browser
        platform, confidence, reason = self.optimizer.get_best_platform_for_browser_model(
            "unknown", "vision"
        )
        self.assertEqual(platform, "webgpu")  # Generic default
    
    def test_get_optimization_parameters(self):
        """Test getting optimization parameters."""
        # Test latency focused
        params = self.optimizer.get_optimization_parameters(
            "text_embedding", OptimizationPriority.LATENCY
        )
        self.assertEqual(params["batch_size"], 1)  # Latency focused uses batch size 1
        
        # Test throughput focused
        params = self.optimizer.get_optimization_parameters(
            "vision", OptimizationPriority.THROUGHPUT
        )
        self.assertGreater(params["batch_size"], 1)  # Throughput uses larger batches
        
        # Test memory focused
        params = self.optimizer.get_optimization_parameters(
            "audio", OptimizationPriority.MEMORY_EFFICIENCY
        )
        self.assertEqual(params["batch_size"], 1)  # Memory focused uses smaller batches
        
        # Test unknown model type
        params = self.optimizer.get_optimization_parameters(
            "unknown", OptimizationPriority.LATENCY
        )
        self.assertTrue("batch_size" in params)  # Should have default params
    
    def test_get_optimized_configuration(self):
        """Test getting optimized configuration."""
        # Test audio model
        config = self.optimizer.get_optimized_configuration(
            model_type="audio",
            model_name="whisper-tiny",
            available_browsers=["firefox", "chrome", "edge"]
        )
        self.assertEqual(config.browser_type, "firefox")  # Firefox is best for audio
        self.assertEqual(config.platform, "webgpu")  # WebGPU is recommended for audio models
        self.assertTrue("audio_thread_priority" in config.parameters)  # Should have audio optimizations
        
        # Test vision model
        config = self.optimizer.get_optimized_configuration(
            model_type="vision",
            model_name="vit-base",
            available_browsers=["firefox", "chrome", "edge"]
        )
        self.assertEqual(config.browser_type, "chrome")  # Chrome is best for vision
        self.assertEqual(config.platform, "webgpu")  # WebGPU is recommended for vision models
        
        # Test text model with user preferences
        config = self.optimizer.get_optimized_configuration(
            model_type="text_embedding",
            model_name="bert-base",
            available_browsers=["firefox", "chrome", "edge"],
            user_preferences={"batch_size": 4, "custom_param": "value"}
        )
        self.assertEqual(config.browser_type, "edge")  # Edge is best for text
        self.assertEqual(config.platform, "webnn")  # WebNN is recommended for text models
        self.assertEqual(config.parameters["batch_size"], 4)  # User preference should override
        self.assertEqual(config.parameters["custom_param"], "value")  # Custom param should be included
    
    def test_apply_runtime_optimizations(self):
        """Test applying runtime optimizations."""
        # Create mock models
        audio_model = MockModel("audio", "whisper-tiny")
        vision_model = MockModel("vision", "vit-base")
        text_model = MockModel("text_embedding", "bert-base")
        
        # Test Firefox audio optimizations
        context = {"batch_size": 2}
        optimized = self.optimizer.apply_runtime_optimizations(
            audio_model, "firefox", context
        )
        self.assertEqual(optimized["batch_size"], 2)  # Should keep user setting
        self.assertTrue(optimized["compute_shader_optimization"])  # Should add Firefox audio optimization
        
        # Test Chrome vision optimizations
        context = {}
        optimized = self.optimizer.apply_runtime_optimizations(
            vision_model, "chrome", context
        )
        self.assertTrue(optimized["parallel_compute_pipelines"])  # Should add Chrome vision optimization
        self.assertTrue(optimized["vision_optimized_shaders"])  # Should add Chrome vision optimization
        
        # Test Edge text optimizations
        context = {"priority_list": ["webnn", "cpu"]}
        optimized = self.optimizer.apply_runtime_optimizations(
            text_model, "edge", context
        )
        self.assertEqual(optimized["priority_list"], ["webnn", "cpu"])  # Should keep user setting
        self.assertTrue(optimized["webnn_optimization"])  # Should add Edge text optimization
    
    def test_cache_usage(self):
        """Test cache usage."""
        # First call should not hit cache
        config1 = self.optimizer.get_optimized_configuration(
            model_type="audio",
            model_name="whisper-tiny",
            available_browsers=["firefox", "chrome", "edge"]
        )
        
        # Second call should hit cache
        config2 = self.optimizer.get_optimized_configuration(
            model_type="audio",
            model_name="whisper-tiny",
            available_browsers=["firefox", "chrome", "edge"]
        )
        
        # Both should be identical
        self.assertEqual(config1.browser_type, config2.browser_type)
        self.assertEqual(config1.platform, config2.platform)
        
        # Check cache hit count
        self.assertEqual(self.optimizer.cache_hit_count, 1)
        
        # Clear cache
        self.optimizer.clear_caches()
        
        # Third call should not hit cache
        config3 = self.optimizer.get_optimized_configuration(
            model_type="audio",
            model_name="whisper-tiny",
            available_browsers=["firefox", "chrome", "edge"]
        )
        
        # Should still have same result
        self.assertEqual(config1.browser_type, config3.browser_type)
        self.assertEqual(config1.platform, config3.platform)
        
        # Check cache hit count
        self.assertEqual(self.optimizer.cache_hit_count, 1)  # Should not have increased
    
    def test_adaptation(self):
        """Test adaptation to performance changes."""
        # Get optimization statistics before adaptation
        stats_before = self.optimizer.get_optimization_statistics()
        
        # Force adaptation
        self.optimizer.last_adaptation_time = 0
        self.optimizer._adapt_to_performance_changes()
        
        # Get optimization statistics after adaptation
        stats_after = self.optimizer.get_optimization_statistics()
        
        # Adaptation count should have increased
        self.assertEqual(stats_after["adaptation_count"], stats_before["adaptation_count"] + 1)
        
        # Caches should be empty
        self.assertEqual(stats_after["capability_scores_cache_size"], 0)
        self.assertEqual(stats_after["recommendation_cache_size"], 0)

def run_tests():
    """Run the test suite."""
    unittest.main()

if __name__ == "__main__":
    run_tests()