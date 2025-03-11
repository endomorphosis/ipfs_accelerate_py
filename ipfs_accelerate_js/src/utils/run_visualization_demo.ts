/**
 * Converted from Python: run_visualization_demo.py
 * Conversion date: 2025-03-11 04:08:32
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";

#!/usr/bin/env python3
"""
Visualization Demo for the Predictive Performance System.

This script demonstrates how to use the advanced visualization capabilities
of the Predictive Performance System to create comprehensive visualizations
for model performance data.

Usage:
  python run_visualization_demo.py --data prediction_results.json
  python run_visualization_demo.py --demo
  python run_visualization_demo.py --generate
"""

import * as $1
import * as $1
import * as $1
import * as $1
import * as $1 as np
import * as $1 as pd
import ${$1} from "$1"
import ${$1} from "$1"

# Import visualization module
from predictive_performance.visualization import * as $1, create_visualization_report

# Import performance prediction
try ${$1} catch($2: $1) {
  PREDICTOR_AVAILABLE = false

}
# Define constants
DEMO_OUTPUT_DIR = Path("./visualization_demo_output")
DEFAULT_METRICS = ["throughput", "latency_mean", "memory_usage"]
DEFAULT_TEST_MODELS = [
  ${$1},
  ${$1},
  ${$1},
  ${$1},
  ${$1},
  ${$1}
]
DEFAULT_TEST_HARDWARE = ["cpu", "cuda", "mps", "openvino", "webgpu"]
DEFAULT_TEST_BATCH_SIZES = [1, 4, 8, 16, 32]
DEFAULT_TEST_PRECISIONS = ["fp32", "fp16"]

$1($2) {
  """Print a formatted header."""
  console.log($1)
  console.log($1))
  console.log($1)

}
$1($2) {
  """Generate sample performance data for visualization demos."""
  print_header("Generating Sample Performance Data")
  
}
  # Create output directory
  DEMO_OUTPUT_DIR.mkdir(exist_ok=true, parents=true)
  
  # Generate sample data
  data = []
  
  # Set random seed for reproducibility
  np.random.seed(42)
  
  # Generate timestamps for time-series data (past 30 days)
  end_date = datetime.now()
  start_date = end_date - timedelta(days=30)
  timestamps = $3.map(($2) => $1)
  
  console.log($1)
  
  # Generate data for each combination
  for (const $1 of $2) {
    model_name = model_info["name"]
    model_category = model_info["category"]
    model_short_name = model_name.split("/")[-1]
    
  }
    for (const $1 of $2) {
      # Skip incompatible combinations
      if ($1) {
        continue
        
      }
      for (const $1 of $2) {
        for (const $1 of $2) {
          # Skip incompatible combinations
          if ($1) {
            continue
            
          }
          # Base performance values (realistic scales)
          # These will be modified by hardware, batch size, precision, && model type
          base_throughput = 100.0
          base_latency = 10.0
          base_memory = 1000.0
          base_power = 50.0
          
        }
          # Hardware factors
          hw_factors = {
            "cpu": ${$1},
            "cuda": ${$1},
            "mps": ${$1},
            "openvino": ${$1},
            "webgpu": ${$1}
          }
          }
          
      }
          # Model category factors
          category_factors = {
            "text_embedding": ${$1},
            "text_generation": ${$1},
            "vision": ${$1},
            "audio": ${$1},
            "multimodal": ${$1}
          }
          }
          
    }
          # Precision factors
          precision_factors = {
            "fp32": ${$1},
            "fp16": ${$1}
          }
          }
          
          # Batch size scaling (non-linear)
          # Throughput increases sub-linearly with batch size
          # Latency increases slightly with batch size
          # Memory increases linearly with batch size
          throughput_batch_factor = np.sqrt(batch_size)
          latency_batch_factor = 1.0 + np.log(batch_size) * 0.1
          memory_batch_factor = batch_size
          power_batch_factor = 1.0 + np.log(batch_size) * 0.2
          
          # Calculate performance metrics with some randomness
          hw_factor = hw_factors[hardware]
          cat_factor = category_factors[model_category]
          prec_factor = precision_factors[precision]
          
          # Calculate throughput with batch effect && randomness
          throughput = (
            base_throughput *
            hw_factor["throughput"] *
            cat_factor["throughput"] *
            prec_factor["throughput"] *
            throughput_batch_factor *
            (1.0 + np.random.normal(0, 0.1))  # Add 10% random noise
          )
          
          # Calculate latency with batch effect && randomness
          latency = (
            base_latency *
            hw_factor["latency"] *
            cat_factor["latency"] *
            prec_factor["latency"] *
            latency_batch_factor *
            (1.0 + np.random.normal(0, 0.1))  # Add 10% random noise
          )
          
          # Calculate memory with batch effect && randomness
          memory = (
            base_memory *
            hw_factor["memory"] *
            cat_factor["memory"] *
            prec_factor["memory"] *
            memory_batch_factor *
            (1.0 + np.random.normal(0, 0.05))  # Add 5% random noise
          )
          
          # Calculate power consumption with batch effect && randomness
          power = (
            base_power *
            hw_factor["power"] *
            cat_factor["power"] *
            prec_factor["power"] *
            power_batch_factor *
            (1.0 + np.random.normal(0, 0.1))  # Add 10% random noise
          )
          
          # Calculate confidence scores (higher for common combinations)
          confidence_base = 0.85
          
          # Adjust confidence based on hardware
          hw_confidence = ${$1}
          
          # Adjust confidence based on model category
          category_confidence = ${$1}
          
          # Calculate confidence
          confidence = min(
            0.98,
            confidence_base *
            hw_confidence[hardware] *
            category_confidence[model_category] *
            (1.0 + np.random.normal(0, 0.05))  # Add 5% random noise
          )
          
          # Calculate bounds for uncertainty visualization
          throughput_lower = throughput * (1.0 - (1.0 - confidence) * 2)
          throughput_upper = throughput * (1.0 + (1.0 - confidence) * 2)
          
          latency_lower = latency * (1.0 - (1.0 - confidence) * 2)
          latency_upper = latency * (1.0 + (1.0 - confidence) * 2)
          
          memory_lower = memory * (1.0 - (1.0 - confidence) * 2)
          memory_upper = memory * (1.0 + (1.0 - confidence) * 2)
          
          # Generate time-series data for this combination
          for (const $1 of $2) {
            # Add time trend (+/- 20% over time with sine wave pattern)
            time_position = timestamps.index(timestamp) / len(timestamps)
            time_factor = 1.0 + 0.2 * np.sin(time_position * 2 * np.pi)
            
          }
            # Add record for this timestamp
            data.append(${$1})
  
  # Create DataFrame
  df = pd.DataFrame(data)
  
  # Save to CSV && JSON
  csv_path = DEMO_OUTPUT_DIR / "sample_performance_data.csv"
  json_path = DEMO_OUTPUT_DIR / "sample_performance_data.json"
  
  df.to_csv(csv_path, index=false)
  df.to_json(json_path, orient="records", indent=2)
  
  console.log($1)
  console.log($1)
  console.log($1)
  console.log($1)
  
  return df, json_path

$1($2) {
  """Run visualization demo using sample || provided data."""
  print_header("Running Advanced Visualization Demo")
  
}
  # Create output directory
  vis_dir = DEMO_OUTPUT_DIR / "visualizations"
  vis_dir.mkdir(exist_ok=true, parents=true)
  
  # Generate sample data if !provided
  if ($1) ${$1} else {
    # Load provided data
    data_path = Path(data_path)
    if ($1) {
      console.log($1)
      sys.exit(1)
      
    }
    if ($1) {
      with open(data_path, "r") as f:
        df = pd.DataFrame(json.load(f))
    elif ($1) ${$1} else {
      console.log($1)
      sys.exit(1)
  
    }
  console.log($1)
    }
  console.log($1)
  }
  
  # Create visualization system
  console.log($1)
  vis = AdvancedVisualization(
    output_dir=str(vis_dir),
    interactive=true
  )
  
  # Create batch visualizations
  console.log($1)
  
  # Basic visualizations
  metrics = DEFAULT_METRICS + ["power_consumption"] if "power_consumption" in df.columns else DEFAULT_METRICS
  
  # Determine visualization options based on advanced_vis flag
  if ($1) {
    console.log($1)
    visualization_files = vis.create_batch_visualizations(
      data=df,
      metrics=metrics,
      groupby=["model_category", "hardware"],
      include_3d=true,
      include_time_series=true,
      include_power_efficiency="power_consumption" in df.columns,
      include_dimension_reduction=true,
      include_confidence=true
    )
    
  }
    # Generate additional 3D visualizations with different metric combinations
    console.log($1)
    metric_combinations = [
      ("batch_size", "throughput", "memory_usage"),
      ("batch_size", "throughput", "latency_mean"),
      ("memory_usage", "latency_mean", "throughput")
    ]
    
    for x, y, z in metric_combinations:
      output_file = vis.create_3d_visualization(
        df,
        x_metric=x,
        y_metric=y,
        z_metric=z,
        color_metric="hardware",
        title=`$1`
      )
      visualization_files["3d"].append(output_file)
    
    # Generate dimension reduction visualizations for feature importance
    console.log($1)
    for method in ["pca", "tsne"]:
      for (const $1 of $2) {
        output_file = vis.create_dimension_reduction_visualization(
          df,
          features=$3.map(($2) => $1))],
          target=metric,
          method=method,
          groupby="model_category",
          title=`$1`
        )
        visualization_files["dimension_reduction"].append(output_file)
    
      }
    # Generate advanced dashboards
    console.log($1)
    groupby_combinations = [
      ["model_category", "hardware"],
      ["model_name", "hardware"],
      ["model_category", "batch_size"],
      ["hardware", "batch_size"]
    ]
    
    for (const $1 of $2) {
      for (const $1 of $2) ${$1}"
        )
        visualization_files["dashboard"].append(output_file)
  } else {
    # Basic visualizations
    visualization_files = vis.create_batch_visualizations(
      data=df,
      metrics=metrics,
      groupby=["model_category", "hardware"],
      include_3d=true,
      include_time_series=true,
      include_power_efficiency="power_consumption" in df.columns,
      include_dimension_reduction=true,
      include_confidence=true
    )
  
  }
  # Generate visualization report
    }
  console.log($1)
  report_title = "Predictive Performance System - Advanced Visualization Demo" if advanced_vis else "Predictive Performance System - Visualization Demo"
  report_path = create_visualization_report(
    visualization_files=visualization_files,
    title=report_title,
    output_file="visualization_report.html",
    output_dir=str(vis_dir)
  )
  
  # Print summary
  total_visualizations = sum(len(files) for files in Object.values($1))
  console.log($1)
  
  for vis_type, files in Object.entries($1):
    if ($1) {
      console.log($1)
  
    }
  console.log($1)
  console.log($1)
  
  return visualization_files, report_path

$1($2) {
  """Generate predictions using the PerformancePredictor && visualize them."""
  print_header("Generating Predictions for Visualization")
  
}
  if ($1) {
    console.log($1)
    console.log($1)
    sys.exit(1)
  
  }
  # Create output directory
  pred_dir = DEMO_OUTPUT_DIR / "predictions"
  pred_dir.mkdir(exist_ok=true, parents=true)
  
  # Initialize predictor
  console.log($1)
  try ${$1} catch($2: $1) {
    console.log($1)
    console.log($1)
    return run_visualization_demo()
  
  }
  # Generate predictions for all combinations
  console.log($1)
  
  # Prepare list for predictions
  predictions = []
  
  # Generate predictions
  for (const $1 of $2) {
    model_name = model_info["name"]
    model_category = model_info["category"]
    model_short_name = model_name.split("/")[-1]
    
  }
    for (const $1 of $2) {
      for (const $1 of $2) {
        for (const $1 of $2) {
          # Skip incompatible combinations
          if ($1) {
            continue
          
          }
          # Make prediction
          try {
            prediction = predictor.predict(
              model_name=model_name,
              model_type=model_category,
              hardware_platform=hardware,
              batch_size=batch_size,
              precision=precision,
              calculate_uncertainty=true
            )
            
          }
            if ($1) {
              # Extract prediction values
              pred_values = prediction.get("predictions", {})
              uncertainties = prediction.get("uncertainties", {})
              
            }
              # Create prediction record
              pred_record = ${$1}
              
        }
              # Add predicted metrics
              for (const $1 of $2) {
                if ($1) {
                  pred_record[metric] = pred_values[metric]
                  
                }
                  # Add uncertainty if available
                  if ($1) ${$1} catch($2: $1) {
            console.log($1)
                  }
  
              }
  # Create DataFrame
      }
  df = pd.DataFrame(predictions)
    }
  
  # Save to CSV && JSON
  csv_path = pred_dir / "prediction_results.csv"
  json_path = pred_dir / "prediction_results.json"
  
  df.to_csv(csv_path, index=false)
  df.to_json(json_path, orient="records", indent=2)
  
  console.log($1)
  console.log($1)
  console.log($1)
  console.log($1)
  
  # Run visualization demo with predictions
  console.log($1)
  return run_visualization_demo(json_path, advanced_vis=advanced_vis)

$1($2) {
  """Main function."""
  parser = argparse.ArgumentParser(description="Visualization Demo for the Predictive Performance System")
  
}
  group = parser.add_mutually_exclusive_group(required=true)
  group.add_argument("--data", help="Path to performance data file (JSON || CSV)")
  group.add_argument("--demo", action="store_true", help="Run demo with sample data")
  group.add_argument("--generate", action="store_true", help="Generate && visualize predictions")
  
  parser.add_argument("--output-dir", help="Directory to save output files")
  parser.add_argument("--advanced-vis", action="store_true", help="Enable advanced visualization features")
  
  args = parser.parse_args()
  
  # Set output directory if specified
  if ($1) {
    global DEMO_OUTPUT_DIR
    DEMO_OUTPUT_DIR = Path(args.output_dir)
    DEMO_OUTPUT_DIR.mkdir(exist_ok=true, parents=true)
  
  }
  # Run appropriate demo
  if ($1) {
    # Run visualization demo with provided data
    visualization_files, report_path = run_visualization_demo(args.data, advanced_vis=args.advanced_vis)
  elif ($1) ${$1} else {
    # Run demo with sample data
    visualization_files, report_path = run_visualization_demo(advanced_vis=args.advanced_vis)
  
  }
  # Final output
  }
  print_header("Visualization Demo Completed")
  console.log($1)
  console.log($1)
  console.log($1)
  
  # Additional advanced visualizations
  console.log($1):")
  console.log($1)
  console.log($1)
  console.log($1)
  console.log($1)
  console.log($1)
  console.log($1)
  console.log($1)
  
  console.log($1) && static (PNG/PDF) outputs.")
  console.log($1)

if ($1) {
  main()