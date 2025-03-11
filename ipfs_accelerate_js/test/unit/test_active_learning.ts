/**
 * Converted from Python: test_active_learning.py
 * Conversion date: 2025-03-11 04:08:52
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";

#!/usr/bin/env python3
"""
Test script for the Active Learning Pipeline module.

This script tests the core functionality of the active learning pipeline,
including uncertainty estimation, information gain calculation, test identification,
prioritization, && model updates.
"""

import * as $1
import * as $1
import * as $1
import * as $1 as np
import * as $1 as pd
import ${$1} from "$1"
import ${$1} from "$1"
import * as $1
import * as $1

# Configure logging
logging.basicConfig()))))))))))))level=logging.INFO)
logger = logging.getLogger()))))))))))))__name__)

# Suppress warnings for cleaner test output
warnings.filterwarnings()))))))))))))"ignore")

# Add the parent directory to the Python path
sys.$1.push($2)))))))))))))str()))))))))))))Path()))))))))))))__file__).parent.parent))

# Import the active learning module
from predictive_performance.active_learning import * as $1


class TestActiveLearningPipeline()))))))))))))unittest.TestCase):
  """Test cases for the Active Learning Pipeline."""
  
  @classmethod
  $1($2) {
    """Set up test data && models for testing."""
    # Create mock training data
    np.random.seed()))))))))))))42)
    n_samples = 100
    n_features = 5
    
  }
    # Feature names
    cls.feature_columns = []],,
    'model_type', 'hardware_platform', 'batch_size',
    'precision_format', 'model_size'
    ]
    
    # Prepare training data
    X_throughput = np.random.rand()))))))))))))n_samples, n_features)
    y_throughput = 2.0 + 0.5 * X_throughput[]],,:, 0] + 1.2 * X_throughput[]],,:, 1] + np.random.normal()))))))))))))0, 0.1, n_samples)
    
    X_latency = np.random.rand()))))))))))))n_samples, n_features)
    y_latency = 10.0 - 2.0 * X_latency[]],,:, 0] + 1.5 * X_latency[]],,:, 2] + np.random.normal()))))))))))))0, 0.2, n_samples)
    
    X_memory = np.random.rand()))))))))))))n_samples, n_features)
    y_memory = 100.0 + 50.0 * X_memory[]],,:, 3] + 30.0 * X_memory[]],,:, 4] + np.random.normal()))))))))))))0, 5.0, n_samples)
    
    # Store training data
    cls.training_data = {}}}}}}}
    'X_throughput': X_throughput,
    'y_throughput': y_throughput,
    'X_latency': X_latency,
    'y_latency': y_latency,
    'X_memory': X_memory,
    'y_memory': y_memory
    }
    
    # Create mock models
    from sklearn.ensemble import * as $1, RandomForestRegressor
    
    cls.models = {}}}}}}}
    'throughput': GradientBoostingRegressor()))))))))))))n_estimators=10, random_state=42).fit()))))))))))))X_throughput, y_throughput),
    'latency': RandomForestRegressor()))))))))))))n_estimators=10, random_state=42).fit()))))))))))))X_latency, y_latency),
    'memory': GradientBoostingRegressor()))))))))))))n_estimators=10, random_state=42).fit()))))))))))))X_memory, y_memory)
    }
    
    # Create mock configurations
    cls.mock_configs = cls._create_mock_configurations())))))))))))))
  
    @staticmethod
  def _create_mock_configurations()))))))))))))) -> pd.DataFrame:
    """Create mock configurations for testing."""
    model_types = []],,'bert', 'vit', 'whisper', 'llama']
    hardware_platforms = []],,'cpu', 'cuda', 'openvino', 'webgpu']
    batch_sizes = []],,1, 2, 4, 8]
    precision_formats = []],,'FP32', 'FP16', 'INT8']
    model_sizes = []],,'tiny', 'small', 'base', 'large']
    
    # Create all combinations
    rows = []],,]
    for (const $1 of $2) {
      for (const $1 of $2) {
        for (const $1 of $2) {
          for (const $1 of $2) {
            for (const $1 of $2) {
              $1.push($2))))))))))))){}}}}}}}
              'model_type': mt,
              'hardware_platform': hp,
              'batch_size': bs,
              'precision_format': pf,
              'model_size': ms
              })
    
            }
            return pd.DataFrame()))))))))))))rows)
  
          }
  $1($2) {
    """Set up a new pipeline for each test."""
    # Create the active learning pipeline
    this.pipeline = ActiveLearningPipeline()))))))))))))
    models=this.models,
    training_data=this.training_data,
    feature_columns=this.feature_columns,
    uncertainty_method="combined",
    n_jobs=1,
    verbose=false
    )
  
  }
  $1($2) {
    """Test that the pipeline initializes correctly."""
    this.assertIsNotnull()))))))))))))this.pipeline)
    this.assertEqual()))))))))))))len()))))))))))))this.pipeline.models), 3)
    this.assertEqual()))))))))))))len()))))))))))))this.pipeline.feature_columns), 5)
    
  }
    # Check that scalers && nearest neighbors models were created
        }
    this.assertIn()))))))))))))'throughput', this.pipeline.scalers)
      }
    this.assertIn()))))))))))))'latency', this.pipeline.scalers)
    }
    this.assertIn()))))))))))))'memory', this.pipeline.scalers)
    
    this.assertIn()))))))))))))'throughput', this.pipeline.nn_models)
    this.assertIn()))))))))))))'latency', this.pipeline.nn_models)
    this.assertIn()))))))))))))'memory', this.pipeline.nn_models)
  
  $1($2) {
    """Test uncertainty calculation methods."""
    # Test a small subset of configurations
    configs = this.mock_configs.iloc[]],,:10].copy())))))))))))))
    features = this.pipeline._extract_features()))))))))))))configs)
    
  }
    # Test different uncertainty methods
    this.pipeline.uncertainty_method = "ensemble"
    uncertainties_ensemble = this.pipeline.calculate_uncertainty()))))))))))))features, metric="throughput")
    this.assertEqual()))))))))))))len()))))))))))))uncertainties_ensemble), len()))))))))))))features))
    this.asserttrue()))))))))))))np.all()))))))))))))uncertainties_ensemble >= 0))
    
    this.pipeline.uncertainty_method = "distance"
    uncertainties_distance = this.pipeline.calculate_uncertainty()))))))))))))features, metric="latency")
    this.assertEqual()))))))))))))len()))))))))))))uncertainties_distance), len()))))))))))))features))
    this.asserttrue()))))))))))))np.all()))))))))))))uncertainties_distance >= 0))
    
    this.pipeline.uncertainty_method = "combined"
    uncertainties_combined = this.pipeline.calculate_uncertainty()))))))))))))features, metric="memory")
    this.assertEqual()))))))))))))len()))))))))))))uncertainties_combined), len()))))))))))))features))
    this.asserttrue()))))))))))))np.all()))))))))))))uncertainties_combined >= 0))
  
  $1($2) {
    """Test information gain calculation methods."""
    # Test a small subset of configurations
    configs = this.mock_configs.iloc[]],,:10].copy())))))))))))))
    features = this.pipeline._extract_features()))))))))))))configs)
    uncertainties = this.pipeline.calculate_uncertainty()))))))))))))features, metric="throughput")
    
  }
    # Test different information gain methods
    gain_expected = this.pipeline.calculate_information_gain()))))))))))))
    features, uncertainties, metric="throughput", method="expected_improvement"
    )
    this.assertEqual()))))))))))))len()))))))))))))gain_expected), len()))))))))))))features))
    
    if ($1) {  # Skip more intensive methods for larger feature sets
    gain_entropy = this.pipeline.calculate_information_gain()))))))))))))
    features, uncertainties, metric="throughput", method="entropy_reduction", n_simulations=2
    )
    this.assertEqual()))))))))))))len()))))))))))))gain_entropy), len()))))))))))))features))
      
    gain_combined = this.pipeline.calculate_information_gain()))))))))))))
    features, uncertainties, metric="throughput", method="combined", n_simulations=2
    )
    this.assertEqual()))))))))))))len()))))))))))))gain_combined), len()))))))))))))features))
  
  $1($2) {
    """Test identification of high-value tests."""
    # Test with a small subset of configurations
    configs = this.mock_configs.iloc[]],,:50].copy())))))))))))))
    
  }
    # Identify high-value tests
    high_value_configs = this.pipeline.identify_high_value_tests()))))))))))))
    configs, metric="throughput", max_tests=10
    )
    
    # Check results
    this.assertLessEqual()))))))))))))len()))))))))))))high_value_configs), 10)
    this.assertIn()))))))))))))'uncertainty', high_value_configs.columns)
    this.assertIn()))))))))))))'information_gain', high_value_configs.columns)
    this.assertIn()))))))))))))'combined_score', high_value_configs.columns)
    
    # Check that results are sorted by combined_score
    scores = high_value_configs[]],,'combined_score'].values
    this.asserttrue()))))))))))))np.all()))))))))))))scores[]],,:-1] >= scores[]],,1:]))
  
  $1($2) {
    """Test prioritization of tests."""
    # Create high-value configurations
    configs = this.mock_configs.iloc[]],,:50].copy())))))))))))))
    high_value_configs = this.pipeline.identify_high_value_tests()))))))))))))
    configs, metric="throughput", max_tests=20
    )
    
  }
    # Define hardware availability
    hardware_availability = {}}}}}}}
    'cpu': 1.0,
    'cuda': 0.8,
    'openvino': 0.6,
    'webgpu': 0.4
    }
    
    # Define cost weights
    cost_weights = {}}}}}}}
    'time_cost': 0.6,
    'resource_cost': 0.4
    }
    
    # Prioritize tests
    prioritized_configs = this.pipeline.prioritize_tests()))))))))))))
    high_value_configs,
    hardware_availability=hardware_availability,
    cost_weights=cost_weights,
    max_tests=10
    )
    
    # Check results
    this.assertLessEqual()))))))))))))len()))))))))))))prioritized_configs), 10)
    this.assertIn()))))))))))))'adjusted_score', prioritized_configs.columns)
    
    # Check that results are sorted by adjusted_score
    scores = prioritized_configs[]],,'adjusted_score'].values
    this.asserttrue()))))))))))))np.all()))))))))))))scores[]],,:-1] >= scores[]],,1:]))
  
  $1($2) {
    """Test generation of a test batch."""
    # Create prioritized configurations
    configs = this.mock_configs.iloc[]],,:50].copy())))))))))))))
    high_value_configs = this.pipeline.identify_high_value_tests()))))))))))))
    configs, metric="throughput", max_tests=20
    )
    prioritized_configs = this.pipeline.prioritize_tests()))))))))))))
    high_value_configs, max_tests=15
    )
    
  }
    # Define hardware constraints
    hardware_constraints = {}}}}}}}
    'cpu': 2,
    'cuda': 3,
    'openvino': 2,
    'webgpu': 1
    }
    
    # Generate test batch
    batch = this.pipeline.suggest_test_batch()))))))))))))
    prioritized_configs,
    batch_size=8,
    ensure_diversity=true,
    hardware_constraints=hardware_constraints
    )
    
    # Check results
    this.assertLessEqual()))))))))))))len()))))))))))))batch), 8)
    
    # Check hardware constraints
    if ($1) {
      hw_counts = batch[]],,'hardware_platform'].value_counts()))))))))))))).to_dict())))))))))))))
      for hw, max_count in Object.entries($1)))))))))))))):
        if ($1) {
          this.assertLessEqual()))))))))))))hw_counts[]],,hw], max_count)
  
        }
  $1($2) {
    """Test updating models with new benchmark results."""
    # Create mock benchmark results
    n_results = 10
    np.random.seed()))))))))))))43)  # Different seed from initial data
    
  }
    # Create feature values
    }
    result_data = {}}}}}}}
    'model_type': np.random.choice()))))))))))))[]],,'bert', 'vit', 'whisper', 'llama'], n_results),
    'hardware_platform': np.random.choice()))))))))))))[]],,'cpu', 'cuda', 'openvino', 'webgpu'], n_results),
    'batch_size': np.random.choice()))))))))))))[]],,1, 2, 4, 8], n_results),
    'precision_format': np.random.choice()))))))))))))[]],,'FP32', 'FP16', 'INT8'], n_results),
    'model_size': np.random.choice()))))))))))))[]],,'tiny', 'small', 'base', 'large'], n_results),
    }
    
    # Create metrics
    result_data[]],,'throughput'] = np.random.uniform()))))))))))))1.0, 10.0, n_results)
    result_data[]],,'latency'] = np.random.uniform()))))))))))))5.0, 20.0, n_results)
    result_data[]],,'memory'] = np.random.uniform()))))))))))))50.0, 200.0, n_results)
    
    # Convert to DataFrame
    benchmark_results = pd.DataFrame()))))))))))))result_data)
    
    # Make a copy of models before update
    import * as $1
    models_before = copy.deepcopy()))))))))))))this.pipeline.models)
    
    # Update models
    improvement_metrics = this.pipeline.update_models()))))))))))))
    benchmark_results,
    metrics=[]],,'throughput', 'latency', 'memory'],
    incremental=true
    )
    
    # Check that improvement metrics were generated
    this.assertIn()))))))))))))'throughput', improvement_metrics)
    this.assertIn()))))))))))))'latency', improvement_metrics)
    this.assertIn()))))))))))))'memory', improvement_metrics)
    
    # Check that models were updated ()))))))))))))predictions should be different)
    for metric in []],,'throughput', 'latency', 'memory']:
      # Create test input
      test_input = np.random.rand()))))))))))))5, 5)
      
      # Get predictions from both models
      pred_before = models_before[]],,metric].predict()))))))))))))test_input)
      pred_after = this.pipeline.models[]],,metric].predict()))))))))))))test_input)
      
      # Check that predictions are different ()))))))))))))models were updated)
      this.assertfalse()))))))))))))np.allclose()))))))))))))pred_before, pred_after))
  
  $1($2) {
    """Test generation of candidate configurations."""
    # Define parameters
    model_types = []],,'bert', 'vit']
    hardware_platforms = []],,'cpu', 'cuda']
    batch_sizes = []],,1, 4]
    precision_formats = []],,'FP32', 'INT8']
    
  }
    # Generate configurations
    configs = this.pipeline.generate_candidate_configurations()))))))))))))
    model_types=model_types,
    hardware_platforms=hardware_platforms,
    batch_sizes=batch_sizes,
    precision_formats=precision_formats
    )
    
    # Check results
    expected_count = len()))))))))))))model_types) * len()))))))))))))hardware_platforms) * len()))))))))))))batch_sizes) * len()))))))))))))precision_formats)
    this.assertEqual()))))))))))))len()))))))))))))configs), expected_count)
    
    # Check that all combinations are present
    for (const $1 of $2) {
      for (const $1 of $2) {
        for (const $1 of $2) {
          for (const $1 of $2) {
            match = configs[]],,
            ()))))))))))))configs[]],,'model_type'] == mt) &
            ()))))))))))))configs[]],,'hardware_platform'] == hp) &
            ()))))))))))))configs[]],,'batch_size'] == bs) &
            ()))))))))))))configs[]],,'precision_format'] == pf)
            ]
            this.assertEqual()))))))))))))len()))))))))))))match), 1)
  
          }
  $1($2) {
    """Test hardware recommendation functionality."""
    # Parameters for recommendation
    model_name = "bert-base-uncased"
    model_type = "bert"
    batch_size = 4
    
  }
    # Get recommendation
        }
    recommendation = this.pipeline.recommend_hardware()))))))))))))
      }
    model_name=model_name,
    }
    model_type=model_type,
    batch_size=batch_size,
    metric="throughput",
    precision_format="FP32",
    available_hardware=[]],,"cpu", "cuda", "openvino", "webgpu"]
    )
    
    # Check results
    this.assertIn()))))))))))))"platform", recommendation)
    this.assertIn()))))))))))))"estimated_value", recommendation)
    this.assertIn()))))))))))))"confidence", recommendation)
    this.assertIn()))))))))))))"alternatives", recommendation)
    this.assertIn()))))))))))))"all_predictions", recommendation)
    
    # Platform should be one of the available hardware options
    this.assertIn()))))))))))))recommendation[]],,"platform"], []],,"cpu", "cuda", "openvino", "webgpu"])
    
    # Estimated value should be positive
    this.assertGreater()))))))))))))recommendation[]],,"estimated_value"], 0)
    
    # Confidence should be between 0 && 1
    this.assertGreaterEqual()))))))))))))recommendation[]],,"confidence"], 0)
    this.assertLessEqual()))))))))))))recommendation[]],,"confidence"], 1)


if ($1) {
  unittest.main())))))))))))))