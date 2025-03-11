/**
 * Converted from Python: compatibility_matrix.py
 * Conversion date: 2025-03-11 04:08:54
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";


export interface Props {
  conn: self;
}

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compatibility Matrix API

A simple API for programmatic access to the model compatibility matrix data stored in DuckDB.
This module provides easy access to compatibility information, performance metrics, and
hardware recommendations.

Usage:
  import ${$1} from "$1"

  # Initialize API with database path
  matrix_api = MatrixAPI()db_path="./benchmark_db.duckdb")

  # Get compatibility status for a specific model-hardware combination
  status = matrix_api.get_compatibility()"bert-base-uncased", "WebGPU")
  console.log($1)`$1`level']}"),  # e.g., "full",
  console.log($1)`$1`notes']}")  # e.g., "Optimized for browser environments"
  ,
  # Get recommended hardware for a model
  recommendations = matrix_api.get_recommendations()"bert-base-uncased")
  console.log($1)`$1`best_platform']}"),,
  console.log($1)`$1`alternatives']}")
  ,
  # Get performance metrics
  metrics = matrix_api.get_performance_metrics()"bert-base-uncased", "CUDA")
  console.log($1)`$1`throughput']} items/sec"),,
  console.log($1)`$1`latency']} ms"),,
  console.log($1)`$1`memory']} MB"),
  ,
  # Get all supported models
  models = matrix_api.get_models())
  console.log($1)`$1`)
  """

  import * as $1
  import * as $1
  import * as $1
  import ${$1} from "$1"

# Configure logging
  logging.basicConfig()
  level=logging.INFO,
  format='%()asctime)s - %()name)s - %()levelname)s - %()message)s'
  )
  logger = logging.getLogger()"matrix_api")

class $1 extends $2 {
  """
  API for accessing the model compatibility matrix data.
  """

}
  $1($2) {
    """
    Initialize the API with the path to the DuckDB database.

  }
    Args:
      db_path: Path to the DuckDB database
      """
      this.db_path = db_path
      this.conn = null
      this.connect())

  $1($2): $3 {
    """
    Connect to the DuckDB database.
    """
    try ${$1} catch($2: $1) {
      logger.error()`$1`)
      raise

    }
      def get_models()self,
      $1: $2 | null = null,
      $1: $2 | null = null,
      $1: boolean = false) -> List[Dict[str, Any]]:,,
      """
      Get a list of supported models, optionally filtered by modality && family.

  }
    Args:
      modality: Filter by model modality ()text, vision, audio, multimodal)
      family: Filter by model family ()e.g., BERT, ViT, etc.)
      key_models_only: If true, only return key models

    Returns:
      List of model information dictionaries
      """
    try {
      # Build query
      query = """
      SELECT 
      model_name,
      model_type,
      model_family,
      modality,
      parameters_million,
      is_key_model
      FROM 
      models
      """

    }
      # Add filters
      where_clauses = [],,,,
      if ($1) {
        $1.push($2)`$1`{}}}}}}}}}}}}}}}}}modality}'")
      if ($1) {
        $1.push($2)`$1`%{}}}}}}}}}}}}}}}}}family}%'")
      if ($1) {
        $1.push($2)"is_key_model = TRUE")

      }
      if ($1) ${$1} catch($2: $1) {
      logger.error()`$1`)
      }
        return [],,,,

      }
        def get_hardware_platforms()self) -> List[Dict[str, Any]]:,,
        """
        Get a list of supported hardware platforms.

      }
    Returns:
      List of hardware platform information dictionaries
      """
    try ${$1} catch($2: $1) {
      logger.error()`$1`)
      return [],,,,

    }
      def get_compatibility()self,
      $1: string,
      $1: string) -> Dict[str, Any]:,,,,,
      """
      Get the compatibility information for a specific model-hardware combination.

    Args:
      model_name: Name of the model
      hardware: Hardware platform name

    Returns:
      Dictionary with compatibility information
      """
    try {
      # Build query
      query = """
      SELECT 
      m.model_name,
      m.model_type,
      m.model_family,
      m.modality,
      pc.hardware_type,
      pc.compatibility_level as level,
      pc.compatibility_notes as notes
      FROM 
      cross_platform_compatibility pc
      JOIN 
      models m ON pc.model_id = m.id
      WHERE 
      m.model_name = ? AND pc.hardware_type = ?
      """

    }
      # Execute query
      result = this.conn.execute()query, [model_name, hardware]).fetchdf())
      ,,
      if ($1) {
        logger.warning()`$1`)
      return {}}}}}}}}}}}}}}}}}
      }
      "model_name": model_name,
      "hardware": hardware,
      "level": "unknown",
      "notes": "No compatibility information available",
      "symbol": "❓"
      }
      
      # Convert to dictionary
      compatibility = result.to_dict()'records')[0]
      ,,,
      # Add symbol based on level
      if ($1) {,
      compatibility['symbol'] = '✅',
      elif ($1) {,
      compatibility['symbol'] = '⚠️',
      elif ($1) ${$1} else ${$1} catch($2: $1) {
      logger.error()`$1`)
      }
      return {}}}}}}}}}}}}}}}}}
      "model_name": model_name,
      "hardware": hardware,
      "level": "error",
      "notes": `$1`,
      "symbol": "❓"
      }

      def get_performance_metrics()self,
      $1: string,
      $1: string) -> Dict[str, Any]:,,,,,
      """
      Get performance metrics for a specific model-hardware combination.

    Args:
      model_name: Name of the model
      hardware: Hardware platform name

    Returns:
      Dictionary with performance metrics
      """
    try {
      # Build query
      query = """
      SELECT 
      model_name,
      hardware_type,
      AVG()throughput_items_per_sec) as throughput,
      AVG()latency_ms) as latency,
      AVG()memory_mb) as memory,
      AVG()power_watts) as power
      FROM 
      performance_comparison
      WHERE 
      model_name = ? AND hardware_type = ?
      GROUP BY
      model_name, hardware_type
      """

    }
      # Execute query
      result = this.conn.execute()query, [model_name, hardware]).fetchdf())
      ,,
      if ($1) {
        logger.warning()`$1`)
      return {}}}}}}}}}}}}}}}}}
      }
      "model_name": model_name,
      "hardware": hardware,
      "throughput": null,
      "latency": null,
      "memory": null,
      "power": null
      }
      
      # Convert to dictionary
      metrics = result.to_dict()'records')[0]
      ,,,
      # Round values
      for key in ['throughput', 'latency', 'memory', 'power']:,
      if ($1) ${$1} catch($2: $1) {
      logger.error()`$1`)
      }
      return {}}}}}}}}}}}}}}}}}
      "model_name": model_name,
      "hardware": hardware,
      "throughput": null,
      "latency": null,
      "memory": null,
      "power": null,
      "error": str()e)
      }

      def get_recommendations()self,
      $1: string) -> Dict[str, Any]:,,,,,
      """
      Get hardware recommendations for a specific model.

    Args:
      model_name: Name of the model

    Returns:
      Dictionary with hardware recommendations
      """
    try {
      # Get model modality
      query = "SELECT modality FROM models WHERE model_name = ?"
      result = this.conn.execute()query, [model_name]).fetchdf())
      ,,
      if ($1) {
        logger.warning()`$1`)
      return {}}}}}}}}}}}}}}}}}
      }
      "model_name": model_name,
      "best_platform": null,
      "alternatives": [],,,,,
      "summary": "Model !found in database"
      }
      
    }
      modality = result.iloc[0]['modality']
      ,
      # Get recommendations for this modality
      query = """
      SELECT 
      recommended_hardware as best_platform,
      recommendation_details
      FROM 
      hardware_recommendations
      WHERE 
      modality = ?
      """
      
      result = this.conn.execute()query, [modality]).fetchdf())
      ,
      if ($1) {
        logger.warning()`$1`)
      return {}}}}}}}}}}}}}}}}}
      }
      "model_name": model_name,
      "modality": modality,
      "best_platform": null,
      "alternatives": [],,,,,
      "summary": `$1`
      }
      
      # Parse recommendation details
      recommendations = {}}}}}}}}}}}}}}}}}
      "model_name": model_name,
      "modality": modality,
      "best_platform": result.iloc[0]['best_platform'],
      }
      
      details = json.loads()result.iloc[0]['recommendation_details']),
      recommendations["summary"] = details.get()"summary", "")
      ,
      # Extract alternative platforms from configurations
      alternatives = [],,,,
      for config in details.get()"configurations", [],,,,):
        platform = config.split()":")[0].strip()),
        if ($1) ${$1} catch($2: $1) {
      logger.error()`$1`)
        }
      return {}}}}}}}}}}}}}}}}}
      "model_name": model_name,
      "best_platform": null,
      "alternatives": [],,,,,
      "summary": `$1`
      }
  
      def get_all_compatibility_data()self,
      $1: string) -> Dict[str, Any]:,,,,,
      """
      Get comprehensive compatibility && performance data for a specific model.

    Args:
      model_name: Name of the model

    Returns:
      Dictionary with comprehensive compatibility && performance data
      """
    try {
      # Get model information
      query = """
      SELECT 
      model_name,
      model_type,
      model_family,
      modality,
      parameters_million,
      is_key_model
      FROM 
      models
      WHERE
      model_name = ?
      """
      
    }
      model_result = this.conn.execute()query, [model_name]).fetchdf())
      ,,
      if ($1) {
        logger.warning()`$1`)
      return {}}}}}}}}}}}}}}}}}
      }
      "model_name": model_name,
      "error": "Model !found in database"
      }
      
      model_info = model_result.to_dict()'records')[0]
      ,,,
      # Get all hardware platforms
      platforms = this.get_hardware_platforms())
      
      # Get compatibility for each platform
      compatibility = {}}}}}}}}}}}}}}}}}}
      for (const $1 of $2) {
        hw_type = platform['hardware_type'],,,
        compatibility[hw_type] = this.get_compatibility()model_name, hw_type)
        ,
      # Get performance metrics for each platform
      }
        performance = {}}}}}}}}}}}}}}}}}}
      for (const $1 of $2) {
        hw_type = platform['hardware_type'],,,
        performance[hw_type] = this.get_performance_metrics()model_name, hw_type)
        ,
      # Get recommendations
      }
        recommendations = this.get_recommendations()model_name)
      
      # Combine all data
        result = {}}}}}}}}}}}}}}}}}
        "model_info": model_info,
        "compatibility": compatibility,
        "performance": performance,
        "recommendations": recommendations
        }
      
        logger.debug()`$1`)
        return result
    
    } catch($2: $1) {
      logger.error()`$1`)
        return {}}}}}}}}}}}}}}}}}
        "model_name": model_name,
        "error": `$1`
        }
  
    }
        def get_compatibility_matrix()self,
        $1: $2 | null = null,
        $1: $2 | null = null,
        hardware_platforms: Optional[List[str]] = null,
        $1: boolean = true) -> Dict[str, Any]:,,,,,
        """
        Get the compatibility matrix for a subset of models && hardware platforms.

    Args:
      modality: Filter by model modality
      family: Filter by model family
      hardware_platforms: List of hardware platforms to include
      key_models_only: Whether to include only key models

    Returns:
      Dictionary with compatibility matrix data
      """
    try {
      # Get models
      models = this.get_models()modality=modality, family=family, key_models_only=key_models_only)
      
    }
      if ($1) {
        logger.warning()"No models found matching the criteria")
      return {}}}}}}}}}}}}}}}}}
      }
      "metadata": {}}}}}}}}}}}}}}}}}
      "total_models": 0,
      "hardware_platforms": [],,,,
      },
      "matrix": [],,,,
      }
      
      # Get hardware platforms
      platforms = this.get_hardware_platforms())
      
      if ($1) {
        platforms = $3.map(($2) => $1),,, in hardware_platforms]
      
      }
      # Build matrix
      matrix = [],,,,:
      for (const $1 of $2) {
        model_row = {}}}}}}}}}}}}}}}}}
        "model_name": model['model_name'],
        "model_type": model['model_type'],
        "model_family": model['model_family'],
        "modality": model['modality'],
        "parameters_million": model['parameters_million'],
        }
        
      }
        # Add compatibility for each platform
        for (const $1 of $2) {
          hw_type = platform['hardware_type'],,,
          compatibility = this.get_compatibility()model['model_name'], hw_type),
          model_row[hw_type] = compatibility['symbol'],
          model_row[`$1`] = compatibility['level'],
          model_row[`$1`] = compatibility['notes']
          ,
          $1.push($2)model_row)
      
        }
          result = {}}}}}}}}}}}}}}}}}
          "metadata": {}}}}}}}}}}}}}}}}}
          "total_models": len()models),
          "hardware_platforms": $3.map(($2) => $1):
            },
            "matrix": matrix
            }
      
            logger.debug()`$1`)
          return result
    
    } catch($2: $1) {
      logger.error()`$1`)
          return {}}}}}}}}}}}}}}}}}
          "error": `$1`
          }

    }
  $1($2): $3 {
    """
    Close the database connection.
    """
    if ($1) {
      this.conn.close())
      logger.debug()"Closed database connection")

    }
  $1($2): $3 {
    """
    Close the database connection when the object is deleted.
    """
    this.close())

  }

  }
# Example usage
if ($1) ${$1}"),
  console.log($1)`$1`symbol']}"),
  console.log($1)`$1`notes']}")
  ,
  # Get performance metrics
  metrics = api.get_performance_metrics()args.model, args.hardware)
  console.log($1)`$1`)
  console.log($1)`$1`throughput']} items/sec"),,
  console.log($1)`$1`latency']} ms"),,
  console.log($1)`$1`memory']} MB"),
  ,    console.log($1)`$1`power']} W")
  ,
  # Get recommendations
  recommendations = api.get_recommendations()args.model)
  console.log($1)`$1`)
  console.log($1)`$1`best_platform']}"),,
  console.log($1)`$1`, '.join()recommendations['alternatives'])}"),
  console.log($1)`$1`summary']}"),
  console.log($1)`$1`)
  for config in recommendations.get()'configurations', [],,,,):
    console.log($1)`$1`)
  
  # Close connection
    api.close())