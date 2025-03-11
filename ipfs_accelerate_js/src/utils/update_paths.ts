/**
 * Converted from Python: update_paths.py
 * Conversion date: 2025-03-11 04:08:37
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";

#!/usr/bin/env python3
"""
Update file paths in CLAUDE.md to reflect the new directory structure.
"""

import * as $1
import * as $1

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("update_paths")

# Define migration mapping
GENERATOR_FILES = []],,
  # Test generators
"merged_test_generator.py",
"simple_test_generator.py",
"qualified_test_generator.py",
"test_generator_functions.py",
"fix_test_generator.py",
"fix_test_generators.py",
"fix_template_generator.py",
  # Runners && execution scripts
"run_fixed_test_generator.py",
"run_template_system_check.py",
"run_key_model_fixes.sh",
"run_generators_phase16.sh",
"run_improved_generators.sh",
"run_improved_model_tests.py",
"run_all_model_hardware_tests.py",
"run_all_skill_tests.py",
  # Template system
"create_simple_template_db.py",
"simple_template_validator.py",
"template_validator.py",
"template_verifier.py",
"template_inheritance_system.py",
"template_hardware_detection.py",
"template_selection.py",
"update_test_generator_with_hardware_templates.py",
"enhance_key_models_hardware_coverage.py",
  # Skill generation
"integrated_skillset_generator.py",
"create_generator.py",
"skillset_generator.py",
  # Hardware
"hardware_test_generator.py",
"fix_hardware_integration.py",
"hardware_model_integration.py",
"hardware_template_integration.py",
"enhance_hardware_test_coverage.py",
"enhance_priority_hardware_coverage.py",
"automated_hardware_selection.py",
  # Models
"test_hf_bert.py",
"test_hf_llama.py",
"test_hf_vit.py",
"test_hf_clap.py",
"test_bert_qualcomm.py",
"test_bert_with_qualcomm.py",
"skill_hf_bert.py",
"skill_hf_llama.py",
"skill_hf_clap.py",
]

DUCKDB_API_FILES = []],,
  # Core DB files
"benchmark_db_api.py",
"benchmark_db_query.py",
"benchmark_db_updater.py",
"benchmark_db_maintenance.py",
"benchmark_db_converter.py",
"benchmark_db_visualizer.py",
"benchmark_db_migration.py",
"verify_database_integration.py",
"verify_database_integration_fixed.py",
  # Schema
"check_database_schema.py",
"check_db_schema.py",
"create_hardware_model_benchmark_database.py",
"fix_benchmark_db_schema.py",
"initialize_database.py",
"update_benchmark_database.py",
"update_benchmark_db_schema.py",
"update_db_schema_for_simulation.py",
"onnx_db_schema_update.py",
  # Migration
"migrate_all_json_files.py",
"migrate_json_to_db.py",
"benchmark_db_migration.py",
  # Utility
"view_benchmark_results.py",
"view_benchmark_data.py",
"cleanup_db_issues.py",
"cleanup_stale_reports.py",
"run_cleanup_stale_reports.py",
"apply_simulation_detection_fixes.py",
"qnn_simulation_helper.py",
"test_qnn_simulation_fix.py",
"test_simulation_detection.py",
"test_simple_simulation_detection.py",
"simulation_analysis.py",
  # Benchmarking tools
"benchmark_with_db_integration.py",
"run_benchmark_with_db.py",
"run_db_integrated_benchmarks.py",
"run_benchmark_with_simulation_detection.py",
"run_incremental_benchmarks.py",
"run_comprehensive_benchmarks.py",
"benchmark_timing_report.py",
"run_time_series_performance.py",
"time_series_performance.py",
]

# Other file mappings
OTHER_MAPPINGS = {}
"benchmark_all_key_models.py": "duckdb_api/core/benchmark_all_key_models.py",
"run_model_benchmarks.py": "generators/benchmark_generators/run_model_benchmarks.py",
"visualize_qnn_performance.py": "duckdb_api/visualization/visualize_qnn_performance.py",
"predictive_performance_demo.py": "predictive_performance/run_predictive_performance_demo.py",
"schedule_model_benchmarks.py": "duckdb_api/scheduling/schedule_model_benchmarks.py",
"test_web_platform_optimizations.py": "generators/runners/web/test_web_platform_optimizations.py",
"run_web_platform_tests.sh": "generators/runners/web/run_web_platform_tests.sh",
"run_web_platform_integration_tests.sh": "generators/runners/web/run_web_platform_integration_tests.sh",
"web_platform_test_runner.py": "generators/runners/web/web_platform_test_runner.py",
"benchmark_real_webnn_webgpu.py": "generators/runners/web/benchmark_real_webnn_webgpu.py",
"test_ipfs_accelerate.py": "generators/models/test_ipfs_accelerate.py",
"hardware_selection.py": "generators/hardware/hardware_selection.py",
"hardware_detection.py": "generators/hardware/hardware_detection.py",
}

$1($2) {
  """Map old file path to new file path."""
  # Split the path to extract the file name
  if ($1) ${$1} else {
    file_name = old_path.split("/")[]],,-1]
  
  }
  # Check if ($1) {
  if ($1) {
    return OTHER_MAPPINGS[]],,file_name]
  
  }
  # Check if ($1) {
  if ($1) {
    # Determine the appropriate generator subdirectory
    if ($1) {
    return `$1`
    }
    elif ($1) {
    return `$1`
    }
    elif ($1) {
    return `$1`
    }
    elif ($1) {
    return `$1`
    }
    elif ($1) ${$1} else {
    return `$1`
    }
  
  }
  # Check if ($1) {
  if ($1) {
    # Determine the appropriate duckdb_api subdirectory
    if ($1) {
    return `$1`
    }
    elif ($1) {
    return `$1`
    }
    elif ($1) {
    return `$1`
    }
    elif ($1) {
    return `$1`
    }
    elif ($1) ${$1} else {
    return `$1`
    }
  
  }
  # Default case: keep the original path
  }
    logger.warning(`$1`)
    return old_path

  }
$1($2) {
  """Update paths in CLAUDE.md."""
  try {
    with open("CLAUDE.md", "r") as f:
      content = f.read()
    
  }
    # Pattern to match "python test/file.py" commands
      pattern = r"python\s+test/([]],,^\s]+)"
    
}
    # Function to replace matched command with new path
    $1($2) ${$1} catch($2: $1) {
    logger.error(`$1`)
    }

  }
if ($1) {
  update_claude_md()
}