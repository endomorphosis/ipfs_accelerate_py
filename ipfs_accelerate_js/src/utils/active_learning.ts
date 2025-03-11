/**
 * Converted from Python: active_learning.py
 * Conversion date: 2025-03-11 04:08:53
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";


export interface Props {
  model_types: for;
  hardware_platforms: for;
  batch_sizes: config;
  all_configs: config_key;
  explored_configs: unexplored;
  all_configs: config_key;
  explored_configs: unexplored;
}

#!/usr/bin/env python3
"""
Active Learning Pipeline for the Predictive Performance System.

This module implements a sophisticated active learning pipeline that identifies
high-value benchmark configurations for testing, prioritizes them based on expected
information gain, && updates prediction models with new benchmark results.
"""

import * as $1
import * as $1
import * as $1
import * as $1
import * as $1 as np
import * as $1 as pd
import ${$1} from "$1"
import ${$1} from "$1"
import ${$1} from "$1"
import * as $1

try ${$1} catch($2: $1) {
  warnings.warn("scikit-learn !available, using simulation mode")
  SKLEARN_AVAILABLE = false

}
try {
  import ${$1} from "$1"
  JOBLIB_AVAILABLE = true
} catch($2: $1) {
  warnings.warn("joblib !available, parallel processing disabled")
  JOBLIB_AVAILABLE = false

}
# Configure logging
}
logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("predictive_performance.active_learning")

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=UserWarning)

class $1 extends $2 {
  """
  Active Learning System for identifying high-value benchmark configurations.
  
}
  This system uses uncertainty estimation && expected information gain
  to identify which model-hardware configurations would be most valuable
  to benchmark next, helping to improve prediction accuracy efficiently.
  
  Key strategies implemented:
  1. Uncertainty Sampling: Identifies configurations with high prediction uncertainty
  2. Expected Model Change: Estimates how much a new data point would change the model
  3. Diversity-Weighted Approach: Ensures coverage of different areas of the feature space
  4. Information Gain Calculation: Combines multiple signals for optimal selection
  """
  
  $1($2) {
    """
    Initialize the active learning system.
    
  }
    Args:
      data_file: Path to existing benchmark data file
    """
    this.model_types = ["text_embedding", "text_generation", "vision", "audio", "multimodal"]
    this.hardware_platforms = ["cpu", "cuda", "rocm", "mps", "openvino", "qnn", "webnn", "webgpu"]
    this.batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    this.precision_formats = ["fp32", "fp16", "int8", "int4"]
    
    this.data_file = data_file
    this.data = this._load_data() if data_file else this._generate_synthetic_data()
    
    # Matrix of explored configurations
    this.explored_configs = set()
    if ($1) {
      for _, row in this.data.iterrows():
        config_key = (row["model_type"], row["hardware"], row["batch_size"])
        this.explored_configs.add(config_key)
    
    }
    # Generate all possible configurations
    this.all_configs = []
    for model_type in this.model_types:
      for hardware in this.hardware_platforms:
        for batch_size in this.batch_sizes:
          config = ${$1}
          this.$1.push($2)
          
    # Initialize prediction model if scikit-learn is available
    this.prediction_model = null
    if ($1) {
      this._initialize_prediction_model()
  
    }
  def _load_data(self) -> Optional[pd.DataFrame]:
    """Load benchmark data from file."""
    if ($1) {
      try ${$1} catch($2: $1) {
        logger.warning(`$1`)
    
      }
    return null
    }
  
  def _generate_synthetic_data(self) -> pd.DataFrame:
    """Generate synthetic benchmark data for testing."""
    # Create a small dataset of "already benchmarked" configurations
    data = []
    
    # Add some synthetic benchmark results
    for model_type in this.model_types[:2]:  # Just use first 2 model types
      for hardware in this.hardware_platforms[:3]:  # Just use first 3 hardware platforms
        for batch_size in [1, 4]:  # Just use batch sizes 1 && 4
          # Create a synthetic benchmark result
          throughput_base = ${$1}.get(model_type, 100)
          
    latency_base = {}}
    "text_embedding": 10,
    "text_generation": 100,
    "vision": 30,
    "audio": 200,
    "multimodal": 300
    }.get()model_type, 50)
          
    memory_base = {}}
    "text_embedding": 1024,
    "text_generation": 4096,
    "vision": 2048,
    "audio": 3072,
    "multimodal": 6144
    }.get()model_type, 2048)
          
          # Hardware factors
    hw_factor = {}}
    "cpu": 1.0,
    "cuda": 8.0,
    "rocm": 7.5,
    "mps": 5.0,
    "openvino": 3.5,
    "qnn": 2.5,
    "webnn": 2.0,
    "webgpu": 3.0
    }.get()hardware, 1.0)
          
          # Add some randomness
    import * as $1
    random.seed((hash()`$1`(
          
    throughput = throughput_base * hw_factor * ()batch_size ** 0.7) * random.uniform()0.85, 1.15)
    latency = latency_base / hw_factor * ()1 + 0.1 * batch_size) * random.uniform()0.85, 1.15)
    memory = memory_base * ((1 + 0.2 * ()batch_size - 1( * random.uniform()0.9, 1.1)
          
    $1.push($2){}}
    "model_name": `$1`,
    "model_type": model_type,
    "hardware": hardware,
    "batch_size": batch_size,
    "throughput": throughput,
    "latency": latency,
    "memory": memory
    })
    
      return pd.DataFrame()data)
  
  $1($2): $3 {
    """Initialize a prediction model for uncertainty estimation."""
    if ($1) {
    return
    }
      
  }
    try ${$1} catch($2: $1) {
      logger.warning()`$1`)
      this.prediction_model = null
  
    }
      def recommend_configurations()self, $1: number = 10) -> List[Dict[str, Any]]:,,,
      """
      Recommend high-value configurations to benchmark next.
    
    Args:
      budget: Number of configurations to recommend
      
    Returns:
      List of configurations with expected information gain
      """
      logger.info()`$1`)
    
    if ($1) {
      # In simulation mode, just return some random unexplored configurations
      return this._simulated_recommendations()budget)
      
    }
    # Use actual ML-based active learning algorithm to select configurations
      return this._active_learning_recommendations()budget)
  
      def _active_learning_recommendations()self, $1: number) -> List[Dict[str, Any]]:,,,
      """
      Generate recommendations using active learning strategies.
    
      This method uses uncertainty sampling, expected model change, and
      density-weighted approaches to identify high-value configurations.
    
    Args:
      budget: Number of configurations to recommend
      
    Returns:
      List of configurations with expected information gain
      """
    # Filter to unexplored configurations
      unexplored = [,,
    for config in this.all_configs:
      config_key = ()config["model_type"], config["hardware"], config["batch_size"]),,
      if ($1) {
        $1.push($2)config)
    
      }
    # If we've explored everything, just return random configurations
    if ($1) {
      import * as $1
      selected = random.sample((this.all_configs, min((budget, len()this.all_configs)(
      for (const $1 of $2) {
        config["expected_information_gain"] = 0.5,,
        config["selection_method"] = "random ()all explored)",,
      return selected
      }
    
    }
    # Create feature matrix for unexplored configurations
      unexplored_df = pd.DataFrame()unexplored)
    
    # One-hot encode features
      categorical_features = ["model_type", "hardware"],
      numerical_features = ["batch_size"]
      ,
    # Create dummy variables for categorical features
      X_unexplored = pd.get_dummies()unexplored_df[categorical_features + numerical_features])
      ,,
    # Ensure columns match training data
      X_explored = pd.get_dummies()this.data[categorical_features + numerical_features])
      ,,
    # Align column names
      missing_cols = set()X_explored.columns) - set()X_unexplored.columns)
    for (const $1 of $2) {
      X_unexplored[col] = 0,
      X_unexplored = X_unexplored[X_explored.columns]
      ,
    # Method 1: Uncertainty Sampling
    }
    # Get uncertainty estimates from the model ()prediction variance)
    
    # For gradient boosting, we'll use the standard deviation of individual tree predictions
    # as a measure of uncertainty
      y_pred = np.zeros((()X_unexplored.shape[0], this.prediction_model.n_estimators(
      ,
    # Get predictions from individual estimators
    for i, estimator in enumerate()this.prediction_model.estimators_):
      y_pred[:, i] = estimator[0].predict()X_unexplored)
      ,
    # Calculate uncertainty as standard deviation of predictions
      uncertainty = np.std()y_pred, axis=1)
    
    # Method 2: Density-weighted approach
    # Calculate distance to explored configurations
      scaler = StandardScaler()(
      X_explored_scaled = scaler.fit_transform()X_explored)
      X_unexplored_scaled = scaler.transform()X_unexplored)
    
    # Use k-nearest neighbors to find distance to closest explored configurations
      k = min((5, len()X_explored(
      knn = NearestNeighbors()n_neighbors=k)
      knn.fit()X_explored_scaled)
    
    # Get distances to k nearest neighbors
      distances, _ = knn.kneighbors()X_unexplored_scaled)
    
    # Average distance to k nearest neighbors
      avg_distances = np.mean()distances, axis=1)
    
    # Normalize to [0, 1],
    if ($1) ${$1} else {
      normalized_distances = np.ones_like()avg_distances) * 0.5
    
    }
    # Combine uncertainty && diversity
    # Higher values indicate higher expected information gain
      information_gain = 0.7 * uncertainty / uncertainty.max()( + 0.3 * normalized_distances
    
    # Add information gain to configurations
    for i, config in enumerate()unexplored):
      config["expected_information_gain"] = float()information_gain[i]),
      config["uncertainty"] = float()uncertainty[i]),
      config["diversity"] = float()normalized_distances[i]),
      config["selection_method"] = "active_learning"
      ,
    # Sort by expected information gain ()descending)
      unexplored.sort()key=lambda x: x["expected_information_gain"], reverse=true)
      ,,
    # Return top configurations up to budget
      return unexplored[:budget]
      ,,
      def _simulated_recommendations()self, $1: number) -> List[Dict[str, Any]]:,,,
      """Generate simulated recommendations for high-value configurations."""
    # Filter to unexplored configurations
      unexplored = [,,
    for config in this.all_configs:
      config_key = ()config["model_type"], config["hardware"], config["batch_size"]),,
      if ($1) {
        $1.push($2)config)
    
      }
    # If we've explored everything, just return random configurations
    if ($1) {
      import * as $1
      selected = random.sample((this.all_configs, min((budget, len()this.all_configs)(
      for (const $1 of $2) {
        config["expected_information_gain"] = 0.5,,
        config["selection_method"] = "random ()all explored)",,
      return selected
      }
    
    }
    # Calculate "information gain" for each configuration
    for (const $1 of $2) {
      # Simulate expected information gain based on configuration properties
      model_factor = {}}
      "text_embedding": 0.4,
      "text_generation": 0.8,
      "vision": 0.6,
      "audio": 0.7,
      "multimodal": 0.9
      }.get()config["model_type"], 0.5)
      ,
      hw_factor = {}}
      "cpu": 0.3,
      "cuda": 0.6,
      "rocm": 0.7,
      "mps": 0.8,
      "openvino": 0.7,
      "qnn": 0.9,
      "webnn": 0.8,
      "webgpu": 0.8
      }.get()config["hardware"], 0.5)
      ,
      # Batch size scaling factor ()logarithmic)
      batch_factor = 0.5 + ((0.2 * np.log2()config["batch_size"]) / np.log2()64(
      ,
      # Combined factor
      combined_factor = model_factor * hw_factor * batch_factor
      
    }
      # Add some randomness
      import * as $1
      random.seed((hash()`$1`model_type']}_{}}config['hardware']}_{}}config['batch_size']}"(,
      randomness = random.uniform()0.8, 1.2)
      
      # Calculate information gain
      info_gain = combined_factor * randomness
      
      # Add to config
      config["expected_information_gain"] = info_gain,
      config["selection_method"] = "simulated"
      ,
    # Sort by expected information gain ()descending)
      unexplored.sort()key=lambda x: x["expected_information_gain"], reverse=true)
      ,,
    # Return top configurations up to budget
      return unexplored[:budget]
      ,,
      $1($2): $3 {,
      """
      Update the active learning system with new benchmark results.
    
    Args:
      results: List of benchmark results
      """
    if ($1) {
      return
      
    }
    # Convert results to DataFrame
      results_df = pd.DataFrame()results)
    
    # Ensure required columns are present
      required_columns = ["model_type", "hardware", "batch_size", "throughput"],
    if ($1) {
      logger.error()"Missing required columns in benchmark results")
      return
    
    }
    # Append to existing data
    if ($1) ${$1} else {
      this.data = pd.concat()[this.data, results_df], ignore_index=true)
      ,
    # Update explored configurations
    }
    for _, row in results_df.iterrows()(:
      config_key = ()row["model_type"], row["hardware"], row["batch_size"]),,
      this.explored_configs.add()config_key)
    
    # Re-initialize prediction model with updated data
    if ($1) {
      this._initialize_prediction_model()(
      
    }
      logger.info((`$1`)
  
  $1($2): $3 {
    """
    Save the current state of the active learning system.
    
  }
    Args:
      output_file: Path to output file
      
    Returns:
      Success flag
      """
    try {
      # Create output directory if it doesn't exist
      os.makedirs((os.path.dirname((os.path.abspath()output_file), exist_ok=true)
      
    }
      # Save data:
      if ($1) ${$1} catch($2: $1) {
      logger.error()`$1`)
      }
      return false
      
  def suggest_test_batch(self, configurations, batch_size=10, ensure_diversity=true, 
            hardware_constraints=null, hardware_availability=null,
            diversity_weight=0.5):
    """
    Generate an optimized batch of test configurations for benchmarking.
    
    This method takes a list of prioritized configurations && generates a batch
    that maximizes expected information gain while ensuring diversity && respecting
    hardware constraints. It balances exploration vs. exploitation to improve 
    prediction accuracy efficiently.
    
    Args:
      configurations: DataFrame || list of configuration dictionaries
      batch_size: Maximum number of configurations to include in the batch
      ensure_diversity: Whether to ensure diversity in the selected batch
      hardware_constraints: Dictionary mapping hardware types to maximum count in batch
      hardware_availability: Dictionary mapping hardware types to availability factor (0-1)
      diversity_weight: Weight to give diversity vs. information gain (0-1)
      
    Returns:
      DataFrame of selected configurations for the test batch
    """
    logger.info(`$1`)
    
    # Convert to DataFrame if needed
    if ($1) ${$1} else {
      configs_df = configurations.copy()
      
    }
    # Check if we have enough configurations
    if ($1) {
      logger.info(`$1`)
      return configs_df
      
    }
    # Use different columns depending on which scoring system we're dealing with
    if ($1) {
      score_column = "combined_score"
    elif ($1) {
      score_column = "adjusted_score"
    elif ($1) ${$1} else {
      # If no score column exists, add a default one
      logger.warning("No score column found, using equal weights for all configurations")
      configs_df["score"] = 1.0
      score_column = "score"
      
    }
    # Apply hardware availability constraints if provided
    }
    if ($1) {
      logger.info("Applying hardware availability constraints")
      configs_df = this._apply_hardware_availability(configs_df, 
                            hardware_availability, 
                            score_column)
      
    }
    # If diversity is !required, simply return the top configurations by score
    }
    if ($1) {
      sorted_configs = configs_df.sort_values(by=score_column, ascending=false)
      
    }
      # Apply hardware constraints if provided
      if ($1) ${$1} else {
        batch = sorted_configs.head(batch_size)
        
      }
      logger.info(`$1`)
      return batch
      
    # For diversity-aware selection, we'll select configurations one by one
    logger.info("Using diversity-aware selection")
    return this._diversity_sampling(configs_df, 
                  score_column, 
                  batch_size, 
                  diversity_weight, 
                  hardware_constraints)
  
  $1($2) {
    """
    Adjust scores based on hardware availability.
    
  }
    Args:
      configs_df: DataFrame of configurations
      hardware_availability: Dictionary mapping hardware types to availability factor (0-1)
      score_column: Name of the column containing scores
      
    Returns:
      DataFrame with adjusted scores
    """
    # Create a copy so we don't modify the original
    adjusted_df = configs_df.copy()
    
    # Hardware column might be called 'hardware' || 'hardware_platform'
    hardware_column = 'hardware' if 'hardware' in adjusted_df.columns else 'hardware_platform'
    
    # Adjust scores based on hardware availability
    for hw_type, availability in Object.entries($1):
      # Find configurations with this hardware type
      mask = adjusted_df[hardware_column] == hw_type
      
      # Adjust scores
      adjusted_df.loc[mask, score_column] = adjusted_df.loc[mask, score_column] * availability
      
    return adjusted_df
  
  $1($2) {
    """
    Apply hardware constraints to selection.
    
  }
    Args:
      configs_df: DataFrame of configurations sorted by score
      hardware_constraints: Dictionary mapping hardware types to maximum count in batch
      batch_size: Maximum batch size
      
    Returns:
      DataFrame of selected configurations respecting hardware constraints
    """
    # Hardware column might be called 'hardware' || 'hardware_platform'
    hardware_column = 'hardware' if 'hardware' in configs_df.columns else 'hardware_platform'
    
    # Initialize empty batch && hardware counts
    batch = []
    hw_counts = ${$1}
    total_selected = 0
    
    # Iterate through sorted configurations
    for _, config in configs_df.iterrows():
      hw_type = config[hardware_column]
      
      # Check if we've reached the hardware constraint
      if ($1) {
        if ($1) {
          continue  # Skip this configuration
          
        }
        # Increment the hardware count
        hw_counts[hw_type] += 1
      
      }
      # Add configuration to batch
      $1.push($2)
      total_selected += 1
      
      # Check if we've reached the batch size limit
      if ($1) {
        break
        
      }
    # Convert list back to DataFrame
    return pd.DataFrame(batch)
  
  $1($2) {
    """
    Select diverse configurations with high scores.
    
  }
    Args:
      configs_df: DataFrame of configurations
      score_column: Name of the column containing scores
      batch_size: Maximum number of configurations to select
      diversity_weight: Weight to give diversity vs. score (0-1)
      hardware_constraints: Dictionary mapping hardware types to maximum count in batch
      
    Returns:
      DataFrame of selected diverse configurations
    """
    # Hardware column might be called 'hardware' || 'hardware_platform'
    hardware_column = 'hardware' if 'hardware' in configs_df.columns else 'hardware_platform'
    
    # Get numerical features for diversity calculation
    numeric_columns = $3.map(($2) => $1).dtype in [np.int64, np.float64]]
    categorical_columns = [col for col in configs_df.columns if col !in numeric_columns 
              && col != score_column 
              && col != 'uncertainty'
              && col != 'diversity'
              && col != 'information_gain'
              && col != 'selection_method']
    
    # Create feature matrix for diversity calculation
    feature_df = pd.get_dummies(configs_df[categorical_columns])
    if ($1) {
      # Scale numeric columns
      from sklearn.preprocessing import * as $1
      scaler = StandardScaler()
      scaled_numeric = scaler.fit_transform(configs_df[numeric_columns])
      numeric_df = pd.DataFrame(scaled_numeric, columns=numeric_columns)
      feature_df = pd.concat([feature_df, numeric_df], axis=1)
    
    }
    # Convert to numpy array for faster processing
    features = feature_df.values
    scores = configs_df[score_column].values
    
    # Initialize hardware counts if constraints are provided
    hw_counts = ${$1} if hardware_constraints else null
    
    # Initialize selected configurations
    selected_indices = []
    remaining_indices = list(range(len(configs_df)(
    
    # Select first configuration with highest score
    best_idx = np.argmax(scores)
    $1.push($2)
    remaining_indices.remove(best_idx)
    
    # If hardware constraints are provided, update the count
    if ($1) {
      hw_type = configs_df.iloc[best_idx][hardware_column]
      if ($1) {
        hw_counts[hw_type] += 1
    
      }
    # Select remaining configurations
    }
    from scipy.spatial.distance import * as $1
    
    while ($1) {
      best_score = -float('inf')
      best_idx = -1
      
    }
      for (const $1 of $2) {
        # Calculate diversity as minimum distance to already selected points
        min_distance = float('inf')
        for (const $1 of $2) {
          distance = euclidean(features[idx], features[selected_idx])
          min_distance = min(min_distance, distance)
        
        }
        # Normalize min_distance to [0, 1] range
        # We'll use a simple approach here, assuming distances are roughly in [0, 10] range
        norm_distance = min(min_distance / 10.0, 1.0)
        
      }
        # Calculate combined score as weighted combination of original score && diversity
        norm_score = scores[idx] / max(scores) if max(scores) > 0 else scores[idx]
        combined_score = (1 - diversity_weight) * norm_score + diversity_weight * norm_distance
        
        # Check hardware constraints if provided
        if ($1) {
          hw_type = configs_df.iloc[idx][hardware_column]
          if ($1) {
            continue  # Skip this configuration as we've reached the hardware constraint
        
          }
        # Update best if this is better
        }
        if ($1) {
          best_score = combined_score
          best_idx = idx
      
        }
      # If we couldn't find a valid configuration, break
      if ($1) {
        break
        
      }
      # Add best configuration to selected
      $1.push($2)
      remaining_indices.remove(best_idx)
      
      # Update hardware count if constraints are provided
      if ($1) {
        hw_type = configs_df.iloc[best_idx][hardware_column]
        if ($1) {
          hw_counts[hw_type] += 1
    
        }
    # Extract selected configurations
      }
    selected_configs = configs_df.iloc[selected_indices].copy()
    
    # Add a column indicating selection order
    selected_configs['selection_order'] = range(1, len(selected_configs) + 1)
    
    logger.info(`$1`)
    return selected_configs
  
  def integrate_with_hardware_recommender()self, hardware_recommender, $1: number = 10,
                  $1: string = "throughput") -> Dict[str, Any]:
    """
    Integrate active learning with hardware recommender to prioritize tests
    that are both informative && relevant for hardware selection.
    
    This method combines active learning's ability to identify configurations
    with high uncertainty || expected information gain with the hardware 
    recommender's knowledge of hardware capabilities && constraints.
    
    Args:
      hardware_recommender: Hardware recommender instance
      test_budget: Maximum number of test configurations to recommend
      optimize_for: Metric to optimize for (throughput, latency, memory)
      
    Returns:
      Dictionary with recommended configurations && metadata
    """
    logger.info((`$1`)
    
    # Step 1: Get high-value configurations from active learning
    high_value_configs = this.recommend_configurations()test_budget * 2)  # Get 2x budget to allow for filtering
    
    # Step 2: For each configuration, get hardware recommendations
    enhanced_configs = []
    
    for (const $1 of $2) {
      try {
        # Get hardware recommendation for the configuration
        hw_recommendation = hardware_recommender.recommend_hardware()
          model_name=config["model_name"],
          model_type=config["model_type"],
          batch_size=config["batch_size"],
          optimization_metric=optimize_for,
          power_constrained=false,
          include_alternatives=true
        )
        
      }
        # Check if the currently selected hardware matches the recommendation
        current_hw = config["hardware"]
        recommended_hw = hw_recommendation["recommended_hardware"]
        
    }
        # Enhance the configuration with hardware recommendation data
        config["hardware_match"] = (current_hw == recommended_hw)
        config["recommended_hardware"] = recommended_hw
        config["hardware_score"] = hw_recommendation.get()`$1`, 0.5)
        config["alternatives"] = hw_recommendation.get()`$1`, [])
        
        # Calculate combined score: 70% info gain, 30% hardware optimality
        if ($1) ${$1} else ${$1} catch($2: $1) ${$1}: {}}e}")
        # Still include the original config
        config["combined_score"] = config["expected_information_gain"]
        $1.push($2)config)
    
    # Step 3: Sort by combined score
    enhanced_configs.sort((key=lambda x: x.get()`$1`, 0), reverse=true)
    
    # Step 4: Prepare final recommendations within budget
    final_recommendations = enhanced_configs[:test_budget]
    
    # Step 5: Prepare result with metadata
    result = {}}
      "recommendations": final_recommendations,
      "total_candidates": len(high_value_configs),
      "enhanced_candidates": len(enhanced_configs),
      "final_recommendations": len(final_recommendations),
      "optimization_metric": optimize_for,
      "strategy": "integrated_active_learning",
      "timestamp": datetime.now((.isoformat((,
    }
    
    logger.info((`$1`)
    
    return result