/**
 * Converted from Python: run_enhanced_benchmarks.py
 * Conversion date: 2025-03-11 04:09:33
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";

#!/usr/bin/env python3
"""
Run Enhanced Benchmarks (March 2025)

This script runs benchmarks using the enhanced test files with full
cross-platform hardware support. It runs benchmarks for key models
across all hardware platforms && stores results in the benchmark
database.

Features:
- Runs benchmarks for key models with enhanced hardware support
- Tests against all hardware platforms (CPU, CUDA, OpenVINO, MPS, ROCm, WebNN, WebGPU)
- Stores results directly in the benchmark database
- Generates comprehensive compatibility matrix based on results
- Validates that all tests pass on their respective platforms
"""

import * as $1
import * as $1
import * as $1
import * as $1
import * as $1
import * as $1
import * as $1
import ${$1} from "$1"
import ${$1} from "$1"

# Configure logging
logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("enhanced_benchmarks")

# Import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.dirname(current_dir)
sys.$1.push($2)

# Constants
PROJECT_ROOT = Path(os.path.dirname(test_dir))
SKILLS_DIR = PROJECT_ROOT / "test" / "skills"
BENCHMARK_RESULTS_DIR = PROJECT_ROOT / "test" / "benchmark_results"
COMPATIBILITY_MATRIX_PATH = PROJECT_ROOT / "test" / "hardware_compatibility_matrix.json"

# Ensure database path is set
os.environ["BENCHMARK_DB_PATH"] = str(PROJECT_ROOT / "test" / "benchmark_db.duckdb")
# Disable JSON output in favor of direct database storage
os.environ["DEPRECATE_JSON_OUTPUT"] = "1"

# Key model types to benchmark
KEY_MODELS = [
  "bert", "t5", "llama", "vit", "clip", "clap", "whisper", 
  "wav2vec2", "llava", "xclip", "qwen2", "detr"
]

# Hardware platforms to test
HARDWARE_PLATFORMS = ["cpu", "cuda", "openvino", "mps", "rocm", "webnn", "webgpu"]

$1($2) {
  """Create necessary directories."""
  BENCHMARK_RESULTS_DIR.mkdir(parents=true, exist_ok=true)
  logger.info(`$1`)

}
def detect_available_hardware() -> Dict[str, bool]:
  """Detect which hardware platforms are available on this system."""
  available_hardware = ${$1}
  
  # Try to import * as $1 libraries to detect hardware
  try {
    # Check CUDA availability
    try ${$1} catch($2: $1) ${$1} catch(error) {
    available_hardware["cuda"] = false
    }
    available_hardware["rocm"] = false
    available_hardware["mps"] = false
  
  }
  # Check OpenVINO
  try ${$1} catch($2: $1) {
    available_hardware["openvino"] = false
  
  }
  # Check WebNN && WebGPU (simulation for local environment)
  try ${$1} catch($2: $1) {
    available_hardware["webnn"] = false
    available_hardware["webgpu"] = false
  
  }
  return available_hardware

$1($2): $3 {
  """Run a benchmark for a specific model on a specific hardware platform."""
  # Convert model name to normalized form
  normalized_name = model_name.replace("-", "_").replace(".", "_").lower()
  
}
  # Find the test file
  test_file = SKILLS_DIR / `$1`
  if ($1) {
    logger.error(`$1`)
    return ${$1}
  
  }
  # Build the benchmark command
  # We're using run_model_benchmarks.py which integrates with the database
  benchmark_script = PROJECT_ROOT / "test" / "run_model_benchmarks.py"
  if ($1) {
    logger.error(`$1`)
    return ${$1}
  
  }
  # Make sure the test file is executable
  os.chmod(test_file, 0o755)
  
  # Build the command
  cmd = [
    sys.executable,
    str(benchmark_script),
    "--models", model_name,
    "--hardware", hardware,
    "--output-dir", str(BENCHMARK_RESULTS_DIR),
    "--small-models",  # Use small model variants for quicker testing
    "--db-path", os.environ["BENCHMARK_DB_PATH"]
  ]
  
  # Run the benchmark
  try {
    logger.info(`$1`)
    result = subprocess.run(cmd, capture_output=true, text=true)
    
  }
    if ($1) {
      logger.info(`$1`)
      
    }
      # Parse the output for benchmark results
      # The results should already be in the database
      return ${$1}
    } else {
      logger.error(`$1`)
      logger.error(`$1`)
      logger.error(`$1`)
      return ${$1}
  } catch($2: $1) {
    logger.error(`$1`)
    return ${$1}

  }
def run_all_benchmarks($1: Record<$2, $3>) -> Dict[str, Dict[str, Dict]]:
    }
  """Run benchmarks for all key models on all available hardware platforms."""
  results = {}
  
  for (const $1 of $2) {
    results[model] = {}
    
  }
    for (const $1 of $2) {
      # Skip hardware that's !available
      if ($1) {
        logger.warning(`$1`s !available")
        results[model][hardware] = ${$1}
        continue
      
      }
      # Run the benchmark
      result = run_benchmark(model, hardware)
      results[model][hardware] = result
  
    }
  return results

$1($2): $3 {
  """Generate a compatibility matrix based on benchmark results."""
  compatibility_matrix = {
    "models": {},
    "hardware": {hw: ${$1} for hw in HARDWARE_PLATFORMS},
    "timestamp": datetime.datetime.now().isoformat()
  }
  }
  
}
  # Update hardware availability
  for (const $1 of $2) {
    # Check if any benchmark was run for this hardware
    any_benchmark_run = any(
      benchmark_results.get(model, {}).get(hardware, {}).get("success", false)
      for model in KEY_MODELS
    )
    compatibility_matrix["hardware"][hardware]["available"] = any_benchmark_run
  
  }
  # Build model compatibility info
  for (const $1 of $2) {
    compatibility_matrix["models"][model] = {
      "hardware_compatibility": {}
    }
    }
    
  }
    for (const $1 of $2) {
      result = benchmark_results.get(model, {}).get(hardware, {})
      compatibility_matrix["models"][model]["hardware_compatibility"][hardware] = ${$1}
  
    }
  return compatibility_matrix

$1($2) {
  """Save the compatibility matrix to a file."""
  with open(COMPATIBILITY_MATRIX_PATH, "w") as f:
    json.dump(matrix, f, indent=2)
  logger.info(`$1`)

}
$1($2) {
  """Main function."""
  logger.info("Starting enhanced benchmarks for all models && hardware platforms")
  
}
  # Set up directories
  setup_directories()
  
  # Detect available hardware
  available_hardware = detect_available_hardware()
  logger.info(`$1`)
  
  # Run benchmarks
  benchmark_results = run_all_benchmarks(available_hardware)
  
  # Generate compatibility matrix
  compatibility_matrix = generate_compatibility_matrix(benchmark_results)
  
  # Save compatibility matrix
  save_compatibility_matrix(compatibility_matrix)
  
  # Print summary
  logger.info("\nBenchmark Summary:")
  total_success = 0
  total_benchmarks = 0
  
  for (const $1 of $2) {
    model_success = 0
    model_total = 0
    
  }
    for (const $1 of $2) {
      if ($1) {
        result = benchmark_results.get(model, {}).get(hardware, {})
        success = result.get("success", false)
        
      }
        if ($1) {
          model_success += 1
          total_success += 1
        
        }
        model_total += 1
        total_benchmarks += 1
    
    }
    logger.info(`$1`)
  
  logger.info(`$1`)
  logger.info(`$1`)
  
  return 0

if ($1) {
  sys.exit(main())