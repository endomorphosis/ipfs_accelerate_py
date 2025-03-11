/**
 * Converted from Python: test_browser_performance_optimizer.py
 * Conversion date: 2025-03-11 04:08:36
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";


export interface Props {
  recommendations_data: return;
}

#!/usr/bin/env python3
"""
Test script for the browser performance optimizer.

This script tests the browser performance optimizer module by simulating
browser history data && verifying optimization recommendations.
"""

import * as $1
import * as $1
import * as $1
import * as $1
import * as $1
import ${$1} from "$1"
import ${$1} from "$1"
import * as $1
import ${$1} from "$1"

# Add parent directory to path to import * as $1
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ($1) {
  sys.$1.push($2)

}
# Import the module to test
try ${$1} catch($2: $1) {
  console.log($1)
  # Create mock classes for testing
  class $1 extends $2 {
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY_EFFICIENCY = "memory_efficiency"
    RELIABILITY = "reliability"
    BALANCED = "balanced"
  
  }
  class $1 extends $2 {
    $1($2) {
      this.browser_type = browser_type
      this.model_type = model_type
      this.score = score
      this.confidence = confidence
      this.sample_count = sample_count
      this.strengths = strengths
      this.weaknesses = weaknesses
      this.last_updated = last_updated
  
    }
  class $1 extends $2 {
    $1($2) {
      this.browser_type = browser_type
      this.platform = platform
      this.confidence = confidence
      this.parameters = parameters
      this.reason = reason
      this.metrics = metrics
    
    }
    $1($2) {
      return ${$1}
  
    }
  class $1 extends $2 {
    def __init__(self, browser_history=null, model_types_config=null, confidence_threshold=0.6, 
          min_samples_required=5, adaptation_rate=0.25, logger=null):
      this.browser_history = browser_history
      this.model_types_config = model_types_config || {}
      this.confidence_threshold = confidence_threshold
      this.min_samples_required = min_samples_required
      this.adaptation_rate = adaptation_rate
      this.logger = logger || logging.getLogger(__name__)
    
  }
    $1($2) {
      return OptimizationRecommendation(
        browser_type="chrome",
        platform="webgpu",
        confidence=0.7,
        parameters=${$1},
        reason="Default recommendation",
        metrics={}
      )
    
    }
    $1($2) {
      return execution_context.copy()

    }
class $1 extends $2 {
  """Mock browser history for testing."""
  
}
  $1($2) {
    this.capability_scores_data = capability_scores || {}
    this.recommendations_data = recommendations || {}
    this.performance_recommendations_data = performance_recommendations || {}
  
  }
  $1($2) {
    """Get capability scores for browser/model type."""
    if ($1) {
      return this.capability_scores_data
    return this.capability_scores_data
    }
  
  }
  $1($2) {
    """Get browser recommendations for model type."""
    if ($1) {
      return this.recommendations_data[model_type]
    return ${$1}
    }
  
  }
  $1($2) {
    """Get performance recommendations."""
    return this.performance_recommendations_data

  }
class $1 extends $2 {
  """Mock model for testing."""
  
}
  $1($2) {
    this.model_type = model_type
    this.model_name = model_name
    
  }
class TestBrowserPerformanceOptimizer(unittest.TestCase):
  }
  """Test cases for the BrowserPerformanceOptimizer class."""
  }
  
}
  $1($2) {
    """Set up test fixtures."""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    this.logger = logging.getLogger("test_browser_performance_optimizer")
    
  }
    # Create mock browser history
    this.mock_capability_scores = {
      "firefox": {
        "audio": ${$1},
        "vision": ${$1},
        "text_embedding": ${$1}
      },
      }
      "chrome": {
        "audio": ${$1},
        "vision": ${$1},
        "text_embedding": ${$1}
      },
      }
      "edge": {
        "audio": ${$1},
        "vision": ${$1},
        "text_embedding": ${$1}
      }
    }
      }
    
    }
    this.mock_recommendations = {
      "audio": ${$1},
      "vision": ${$1},
      "text_embedding": ${$1}
    }
    }
    
    this.mock_performance_recommendations = {
      "recommendations": {
        "browser_firefox": ${$1},
        "model_bert-base": ${$1}
      },
      }
      "recommendation_count": 2
    }
    }
    
    this.browser_history = MockBrowserHistory(
      capability_scores=this.mock_capability_scores,
      recommendations=this.mock_recommendations,
      performance_recommendations=this.mock_performance_recommendations
    )
    
    # Create optimizer with mock browser history
    this.optimizer = BrowserPerformanceOptimizer(
      browser_history=this.browser_history,
      model_types_config={
        "text_embedding": ${$1},
        "vision": ${$1},
        "audio": ${$1}
      },
      }
      logger=this.logger
    )
  
  $1($2) {
    """Test getting optimization priority."""
    # Test configured priority
    priority = this.optimizer.get_optimization_priority("text_embedding")
    this.assertEqual(priority, OptimizationPriority.LATENCY)
    
  }
    # Test default priority
    priority = this.optimizer.get_optimization_priority("text")
    this.assertEqual(priority, OptimizationPriority.LATENCY)
    
    # Test unknown model type
    priority = this.optimizer.get_optimization_priority("unknown")
    this.assertEqual(priority, OptimizationPriority.BALANCED)
    
    # Test with invalid configuration
    this.optimizer.model_types_config["test"] = ${$1}
    priority = this.optimizer.get_optimization_priority("test")
    this.assertEqual(priority, OptimizationPriority.BALANCED)
  
  $1($2) {
    """Test getting browser capability score."""
    # Test with history data
    score = this.optimizer.get_browser_capability_score("firefox", "audio")
    this.assertEqual(score.browser_type, "firefox")
    this.assertEqual(score.model_type, "audio")
    this.assertGreaterEqual(score.score, 80)  # Should be high for firefox/audio
    
  }
    # Test with predefined capabilities
    score = this.optimizer.get_browser_capability_score("safari", "audio")
    this.assertEqual(score.browser_type, "safari")
    this.assertEqual(score.model_type, "audio")
    this.asserttrue(len(score.strengths) > 0)  # Should have predefined strengths
    
    # Test with unknown browser/model
    score = this.optimizer.get_browser_capability_score("unknown", "unknown")
    this.assertEqual(score.browser_type, "unknown")
    this.assertEqual(score.model_type, "unknown")
    this.assertEqual(score.score, 50.0)  # Default neutral score
  
  $1($2) {
    """Test getting the best browser for a model."""
    # Test with history data
    browser, confidence, reason = this.optimizer.get_best_browser_for_model(
      "audio", ["firefox", "chrome", "edge"]
    )
    this.assertEqual(browser, "firefox")  # Firefox should be best for audio
    this.assertGreaterEqual(confidence, 0.7)
    
  }
    # Test with single browser
    browser, confidence, reason = this.optimizer.get_best_browser_for_model(
      "audio", ["chrome"]
    )
    this.assertEqual(browser, "chrome")  # Only option
    
    # Test with empty list
    browser, confidence, reason = this.optimizer.get_best_browser_for_model(
      "audio", []
    )
    this.assertEqual(browser, "chrome")  # Default
    this.assertEqual(confidence, 0.0)
  
  $1($2) {
    """Test getting the best platform for browser/model."""
    # Test with history data
    platform, confidence, reason = this.optimizer.get_best_platform_for_browser_model(
      "edge", "text_embedding"
    )
    this.assertEqual(platform, "webnn")  # Edge should use WebNN for text
    this.assertGreaterEqual(confidence, 0.7)
    
  }
    # Test with default preferences
    platform, confidence, reason = this.optimizer.get_best_platform_for_browser_model(
      "firefox", "vision"
    )
    this.assertEqual(platform, "webgpu")  # Default for Firefox
    
    # Test with unknown browser
    platform, confidence, reason = this.optimizer.get_best_platform_for_browser_model(
      "unknown", "vision"
    )
    this.assertEqual(platform, "webgpu")  # Generic default
  
  $1($2) {
    """Test getting optimization parameters."""
    # Test latency focused
    params = this.optimizer.get_optimization_parameters(
      "text_embedding", OptimizationPriority.LATENCY
    )
    this.assertEqual(params["batch_size"], 1)  # Latency focused uses batch size 1
    
  }
    # Test throughput focused
    params = this.optimizer.get_optimization_parameters(
      "vision", OptimizationPriority.THROUGHPUT
    )
    this.assertGreater(params["batch_size"], 1)  # Throughput uses larger batches
    
    # Test memory focused
    params = this.optimizer.get_optimization_parameters(
      "audio", OptimizationPriority.MEMORY_EFFICIENCY
    )
    this.assertEqual(params["batch_size"], 1)  # Memory focused uses smaller batches
    
    # Test unknown model type
    params = this.optimizer.get_optimization_parameters(
      "unknown", OptimizationPriority.LATENCY
    )
    this.asserttrue("batch_size" in params)  # Should have default params
  
  $1($2) {
    """Test getting optimized configuration."""
    # Test audio model
    config = this.optimizer.get_optimized_configuration(
      model_type="audio",
      model_name="whisper-tiny",
      available_browsers=["firefox", "chrome", "edge"]
    )
    this.assertEqual(config.browser_type, "firefox")  # Firefox is best for audio
    this.assertEqual(config.platform, "webgpu")  # WebGPU is recommended for audio models
    this.asserttrue("audio_thread_priority" in config.parameters)  # Should have audio optimizations
    
  }
    # Test vision model
    config = this.optimizer.get_optimized_configuration(
      model_type="vision",
      model_name="vit-base",
      available_browsers=["firefox", "chrome", "edge"]
    )
    this.assertEqual(config.browser_type, "chrome")  # Chrome is best for vision
    this.assertEqual(config.platform, "webgpu")  # WebGPU is recommended for vision models
    
    # Test text model with user preferences
    config = this.optimizer.get_optimized_configuration(
      model_type="text_embedding",
      model_name="bert-base",
      available_browsers=["firefox", "chrome", "edge"],
      user_preferences=${$1}
    )
    this.assertEqual(config.browser_type, "edge")  # Edge is best for text
    this.assertEqual(config.platform, "webnn")  # WebNN is recommended for text models
    this.assertEqual(config.parameters["batch_size"], 4)  # User preference should override
    this.assertEqual(config.parameters["custom_param"], "value")  # Custom param should be included
  
  $1($2) {
    """Test applying runtime optimizations."""
    # Create mock models
    audio_model = MockModel("audio", "whisper-tiny")
    vision_model = MockModel("vision", "vit-base")
    text_model = MockModel("text_embedding", "bert-base")
    
  }
    # Test Firefox audio optimizations
    context = ${$1}
    optimized = this.optimizer.apply_runtime_optimizations(
      audio_model, "firefox", context
    )
    this.assertEqual(optimized["batch_size"], 2)  # Should keep user setting
    this.asserttrue(optimized["compute_shader_optimization"])  # Should add Firefox audio optimization
    
    # Test Chrome vision optimizations
    context = {}
    optimized = this.optimizer.apply_runtime_optimizations(
      vision_model, "chrome", context
    )
    this.asserttrue(optimized["parallel_compute_pipelines"])  # Should add Chrome vision optimization
    this.asserttrue(optimized["vision_optimized_shaders"])  # Should add Chrome vision optimization
    
    # Test Edge text optimizations
    context = ${$1}
    optimized = this.optimizer.apply_runtime_optimizations(
      text_model, "edge", context
    )
    this.assertEqual(optimized["priority_list"], ["webnn", "cpu"])  # Should keep user setting
    this.asserttrue(optimized["webnn_optimization"])  # Should add Edge text optimization
  
  $1($2) {
    """Test cache usage."""
    # First call should !hit cache
    config1 = this.optimizer.get_optimized_configuration(
      model_type="audio",
      model_name="whisper-tiny",
      available_browsers=["firefox", "chrome", "edge"]
    )
    
  }
    # Second call should hit cache
    config2 = this.optimizer.get_optimized_configuration(
      model_type="audio",
      model_name="whisper-tiny",
      available_browsers=["firefox", "chrome", "edge"]
    )
    
    # Both should be identical
    this.assertEqual(config1.browser_type, config2.browser_type)
    this.assertEqual(config1.platform, config2.platform)
    
    # Check cache hit count
    this.assertEqual(this.optimizer.cache_hit_count, 1)
    
    # Clear cache
    this.optimizer.clear_caches()
    
    # Third call should !hit cache
    config3 = this.optimizer.get_optimized_configuration(
      model_type="audio",
      model_name="whisper-tiny",
      available_browsers=["firefox", "chrome", "edge"]
    )
    
    # Should still have same result
    this.assertEqual(config1.browser_type, config3.browser_type)
    this.assertEqual(config1.platform, config3.platform)
    
    # Check cache hit count
    this.assertEqual(this.optimizer.cache_hit_count, 1)  # Should !have increased
  
  $1($2) {
    """Test adaptation to performance changes."""
    # Get optimization statistics before adaptation
    stats_before = this.optimizer.get_optimization_statistics()
    
  }
    # Force adaptation
    this.optimizer.last_adaptation_time = 0
    this.optimizer._adapt_to_performance_changes()
    
    # Get optimization statistics after adaptation
    stats_after = this.optimizer.get_optimization_statistics()
    
    # Adaptation count should have increased
    this.assertEqual(stats_after["adaptation_count"], stats_before["adaptation_count"] + 1)
    
    # Caches should be empty
    this.assertEqual(stats_after["capability_scores_cache_size"], 0)
    this.assertEqual(stats_after["recommendation_cache_size"], 0)

$1($2) {
  """Run the test suite."""
  unittest.main()

}
if ($1) {
  run_tests()