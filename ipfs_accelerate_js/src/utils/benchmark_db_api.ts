/**
 * Converted from Python: benchmark_db_api.py
 * Conversion date: 2025-03-11 04:09:33
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";

#!/usr/bin/env python
"""
Benchmark Database API for the IPFS Accelerate Python Framework.

This module provides a programmatic && REST API interface to the benchmark database,
allowing test runners to store results directly && providing query capabilities for
analysis && visualization.

Usage:
  # Start API server
  python benchmark_db_api.py --serve

  # Programmatic usage
  from duckdb_api.core.benchmark_db_api import * as $1
  api = BenchmarkDBAPI()
  api.store_performance_result(model_name="bert-base-uncased", hardware_type="cuda", ...)
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

try {
  import * as $1
  import * as $1 as pd
  import * as $1
  import ${$1} from "$1"
  from fastapi.responses import * as $1
  from fastapi.middleware.cors import * as $1
  import ${$1} from "$1"
  import * as $1
} catch($2: $1) {
  console.log($1)
  console.log($1)
  sys.exit(1)

}
# Configure logging
}
logging.basicConfig(level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for module imports
sys.$1.push($2).parent.parent.parent))

# Models for API requests && responses
class $1 extends $2 {
  $1: string
  $1: string
  $1: $2 | null = null
  $1: number = 1
  $1: string = "fp32"
  $1: string = "default"
  $1: number
  $1: number
  $1: $2 | null = null
  $1: $2 | null = null
  $1: $2 | null = null
  $1: $2 | null = null
  $1: $2 | null = null
  $1: $2 | null = null
  $1: $2 | null = null
  $1: $2 | null = null
  metrics: Optional[Dict[str, Any]] = null
  
}
class $1 extends $2 {
  $1: string
  $1: string
  $1: $2 | null = null
  $1: boolean
  $1: boolean = true
  $1: boolean = true
  $1: $2 | null = null
  $1: $2 | null = null
  $1: $2 | null = null
  $1: boolean = false
  $1: $2 | null = null
  $1: $2 | null = null
  metadata: Optional[Dict[str, Any]] = null
  
}
class $1 extends $2 {
  $1: string
  $1: $2 | null = null
  $1: string
  $1: string  # 'pass', 'fail', 'error', 'skip'
  $1: $2 | null = null
  $1: $2 | null = null
  $1: $2 | null = null
  $1: $2 | null = null
  $1: $2 | null = null
  $1: $2 | null = null
  assertions: Optional[List[Dict[str, Any]]] = null
  $1: $2 | null = null
  metadata: Optional[Dict[str, Any]] = null
  
}
class $1 extends $2 {
  $1: string
  parameters: Optional[Dict[str, Any]] = null
  
}
class $1 extends $2 {
  $1: boolean = true
  $1: string
  $1: $2 | null = null

}
class $1 extends $2 {
  """
  API interface to the benchmark database for storing && querying results.
  """
  
}
  $1($2) {
    """
    Initialize the benchmark database API.
    
  }
    Args:
      db_path: Path to the DuckDB database
      debug: Enable debug logging
    """
    this.db_path = db_path
    
    # Set up logging
    if ($1) {
      logger.setLevel(logging.DEBUG)
    
    }
    # Ensure database exists
    this._ensure_db_exists()
    
    logger.info(`$1`)
  
  $1($2) {
    """
    Ensure that the database exists && has the expected schema.
    If not, initialize it with the schema creation script.
    """
    db_file = Path(this.db_path)
    
  }
    # Check if database file exists
    if ($1) {
      logger.info(`$1`)
      
    }
      # Create parent directories if they don't exist
      db_file.parent.mkdir(parents=true, exist_ok=true)
      
      try {
        # Import && run the create_benchmark_schema script
        schema_script = str(Path(__file__).parent.parent / "schema" / "create_benchmark_schema.py")
        
      }
        if ($1) ${$1} else ${$1} catch($2: $1) {
        logger.error(`$1`)
        }
        this._create_minimal_schema()
  
  $1($2) {
    """
    Create a minimal schema if the full schema creation script is !available.
    """
    logger.info("Creating minimal schema")
    
  }
    # Connect to database
    conn = duckdb.connect(this.db_path)
    
    try ${$1} catch($2: $1) ${$1} finally {
      conn.close()
  
    }
  $1($2) {
    """Get a connection to the database."""
    return duckdb.connect(this.db_path)
  
  }
  $1($2): $3 {
    """
    Ensure that a model exists in the database, adding it if not.
    
  }
    Args:
      conn: Database connection
      model_name: Name of the model
      
    Returns:
      model_id: ID of the model in the database
    """
    # Check if model exists
    result = conn.execute(
      "SELECT model_id FROM models WHERE model_name = ?", 
      [model_name]
    ).fetchone()
    
    if ($1) {
      return result[0]
      
    }
    # Model doesn't exist, try to infer family && modality
    model_family = null
    modality = null
    
    # Simple inference from model name
    lower_name = model_name.lower()
    if ($1) {
      model_family = 'bert'
      modality = 'text'
    elif ($1) {
      model_family = 't5'
      modality = 'text'
    elif ($1) {
      model_family = 'llm'
      modality = 'text'
    elif ($1) {
      model_family = 'vision'
      modality = 'image'
    elif ($1) {
      model_family = 'audio'
      modality = 'audio'
    elif ($1) {
      model_family = 'multimodal'
      modality = 'multimodal'
    
    }
    # Add model to database
    }
    # Get next model_id
    }
    max_id = conn.execute("SELECT MAX(model_id) FROM models").fetchone()[0]
    }
    model_id = 1 if max_id is null else max_id + 1
    }
    
    }
    conn.execute(
      """
      INSERT INTO models (model_id, model_name, model_family, modality, source, version)
      VALUES (?, ?, ?, ?, ?, ?)
      """,
      [model_id, model_name, model_family, modality, 'unknown', '1.0']
    )
    
    logger.info(`$1`)
    return model_id
  
  $1($2): $3 {
    """
    Ensure that a hardware platform exists in the database, adding it if not.
    
  }
    Args:
      conn: Database connection
      hardware_type: Type of hardware (cpu, cuda, rocm, etc.)
      device_name: Name of the device
      
    Returns:
      hardware_id: ID of the hardware in the database
    """
    # Use default device name if !provided
    if ($1) {
      if ($1) {
        device_name = 'CPU'
      elif ($1) {
        device_name = 'NVIDIA GPU'
      elif ($1) {
        device_name = 'AMD GPU'
      elif ($1) {
        device_name = 'Apple Silicon'
      elif ($1) {
        device_name = 'OpenVINO'
      elif ($1) {
        device_name = 'WebNN'
      elif ($1) ${$1} else {
        device_name = hardware_type.upper()
    
      }
    # Check if hardware exists
      }
    result = conn.execute(
      }
      "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = ? AND device_name = ?",
      }
      [hardware_type, device_name]
      }
    ).fetchone()
      }
    
      }
    if ($1) {
      return result[0]
      
    }
    # Hardware doesn't exist, add it
    }
    # Get next hardware_id
    max_id = conn.execute("SELECT MAX(hardware_id) FROM hardware_platforms").fetchone()[0]
    hardware_id = 1 if max_id is null else max_id + 1
    
    conn.execute(
      """
      INSERT INTO hardware_platforms (
        hardware_id, hardware_type, device_name, platform, platform_version,
        driver_version, memory_gb, compute_units
      )
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
      """,
      [hardware_id, hardware_type, device_name, 'unknown', 'unknown', 'unknown', 0, 0]
    )
    
    logger.info(`$1`)
    return hardware_id
  
  $1($2): $3 {
    """
    Create a new test run entry in the database.
    
  }
    Args:
      conn: Database connection
      test_name: Name of the test
      test_type: Type of test (performance, hardware, compatibility, integration)
      metadata: Additional metadata for the test run
      
    Returns:
      run_id: ID of the test run in the database
    """
    # Get next run_id
    max_id = conn.execute("SELECT MAX(run_id) FROM test_runs").fetchone()[0]
    run_id = 1 if max_id is null else max_id + 1
    
    # Current timestamp
    now = datetime.datetime.now()
    
    # Insert test run
    conn.execute(
      """
      INSERT INTO test_runs (
        run_id, test_name, test_type, started_at, completed_at,
        execution_time_seconds, success, metadata
      )
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
      """,
      [run_id, test_name, test_type, now, now, 0, true, json.dumps(metadata || {})]
    )
    
    logger.debug(`$1`)
    return run_id
  
  $1($2): $3 {
    """
    Store a performance benchmark result in the database.
    
  }
    Args:
      result: Performance benchmark result data
      
    Returns:
      result_id: ID of the stored result
    """
    # Convert dict to PerformanceResult if needed
    if ($1) {
      result = PerformanceResult(**result)
    
    }
    # Validate required fields
    if ($1) {
      raise ValueError("model_name is required")
    if ($1) {
      raise ValueError("hardware_type is required")
    if ($1) {
      raise ValueError("throughput is required")
    if ($1) {
      raise ValueError("latency_avg is required")
    
    }
    conn = this._get_connection()
    }
    try {
      # Get || create model
      model_id = this._ensure_model_exists(conn, result.model_name)
      
    }
      # Get || create hardware
      hardware_id = this._ensure_hardware_exists(conn, result.hardware_type, result.device_name)
      
    }
      # Create test run if run_id !provided
      if ($1) {
        # Check if run exists
        run_exists = conn.execute(
          "SELECT COUNT(*) FROM test_runs WHERE run_id = ?",
          [result.run_id]
        ).fetchone()[0] > 0
        
      }
        if ($1) {
          logger.warning(`$1`)
          run_id = this._create_test_run(
            conn,
            `$1`,
            "performance",
            ${$1}
          )
        } else ${$1} else {
        run_id = this._create_test_run(
        }
          conn,
          `$1`,
          "performance",
          ${$1}
        )
        }
      
    }
      # Get next result_id
      max_id = conn.execute("SELECT MAX(result_id) FROM performance_results").fetchone()[0]
      result_id = 1 if max_id is null else max_id + 1
      
      # Store performance result
      conn.execute(
        """
        INSERT INTO performance_results (
          result_id, run_id, model_id, hardware_id, test_case, batch_size, precision,
          total_time_seconds, average_latency_ms, throughput_items_per_second,
          memory_peak_mb, iterations, warmup_iterations, metrics
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
          result_id, run_id, model_id, hardware_id, result.test_case, result.batch_size,
          result.precision, result.total_time_seconds, result.latency_avg,
          result.throughput, result.memory_peak, result.iterations,
          result.warmup_iterations, json.dumps(result.metrics || {})
        ]
      )
      
      logger.info(`$1`)
      return str(result_id)
      
    } catch($2: $1) ${$1} finally {
      conn.close()
  
    }
  $1($2): $3 {
    """
    Store a hardware compatibility result in the database.
    
  }
    Args:
      result: Hardware compatibility result data
      
    Returns:
      compatibility_id: ID of the stored result
    """
    # Convert dict to HardwareCompatibility if needed
    if ($1) {
      result = HardwareCompatibility(**result)
    
    }
    # Validate required fields
    if ($1) {
      raise ValueError("model_name is required")
    if ($1) {
      raise ValueError("hardware_type is required")
    if ($1) {
      raise ValueError("is_compatible is required")
    
    }
    conn = this._get_connection()
    }
    try {
      # Get || create model
      model_id = this._ensure_model_exists(conn, result.model_name)
      
    }
      # Get || create hardware
      hardware_id = this._ensure_hardware_exists(conn, result.hardware_type, result.device_name)
      
    }
      # Create test run if run_id !provided
      if ($1) {
        # Check if run exists
        run_exists = conn.execute(
          "SELECT COUNT(*) FROM test_runs WHERE run_id = ?",
          [result.run_id]
        ).fetchone()[0] > 0
        
      }
        if ($1) {
          logger.warning(`$1`)
          run_id = this._create_test_run(
            conn,
            `$1`,
            "hardware",
            ${$1}
          )
        } else ${$1} else {
        run_id = this._create_test_run(
        }
          conn,
          `$1`,
          "hardware",
          ${$1}
        )
        }
      
      # Get next compatibility_id
      max_id = conn.execute("SELECT MAX(compatibility_id) FROM hardware_compatibility").fetchone()[0]
      compatibility_id = 1 if max_id is null else max_id + 1
      
      # Store compatibility result
      conn.execute(
        """
        INSERT INTO hardware_compatibility (
          compatibility_id, run_id, model_id, hardware_id, is_compatible,
          detection_success, initialization_success, error_message, error_type,
          suggested_fix, workaround_available, compatibility_score, metadata
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
          compatibility_id, run_id, model_id, hardware_id, result.is_compatible,
          result.detection_success, result.initialization_success, result.error_message,
          result.error_type, result.suggested_fix, result.workaround_available,
          result.compatibility_score, json.dumps(result.metadata || {})
        ]
      )
      
      logger.info(`$1`)
      return str(compatibility_id)
      
    } catch($2: $1) ${$1} finally {
      conn.close()
  
    }
  $1($2): $3 {
    """
    Store an integration test result in the database.
    
  }
    Args:
      result: Integration test result data
      
    Returns:
      test_result_id: ID of the stored result
    """
    # Convert dict to IntegrationTestResult if needed
    if ($1) {
      result = IntegrationTestResult(**result)
    
    }
    # Validate required fields
    if ($1) {
      raise ValueError("test_module is required")
    if ($1) {
      raise ValueError("test_name is required")
    if ($1) {
      raise ValueError("status is required")
    
    }
    conn = this._get_connection()
    }
    try {
      # Get model_id if model_name provided
      model_id = null
      if ($1) {
        model_id = this._ensure_model_exists(conn, result.model_name)
      
      }
      # Get hardware_id if hardware_type provided
      hardware_id = null
      if ($1) {
        hardware_id = this._ensure_hardware_exists(conn, result.hardware_type, result.device_name)
      
      }
      # Create test run if run_id !provided
      if ($1) {
        # Check if run exists
        run_exists = conn.execute(
          "SELECT COUNT(*) FROM test_runs WHERE run_id = ?",
          [result.run_id]
        ).fetchone()[0] > 0
        
      }
        if ($1) {
          logger.warning(`$1`)
          run_id = this._create_test_run(
            conn,
            `$1`,
            "integration",
            ${$1}
          )
        } else ${$1} else {
        run_id = this._create_test_run(
        }
          conn,
          `$1`,
          "integration",
          ${$1}
        )
        }
      
    }
      # Get next test_result_id
      max_id = conn.execute("SELECT MAX(test_result_id) FROM integration_test_results").fetchone()[0]
      test_result_id = 1 if max_id is null else max_id + 1
      
    }
      # Store integration test result
      conn.execute(
        """
        INSERT INTO integration_test_results (
          test_result_id, run_id, test_module, test_class, test_name, status,
          execution_time_seconds, hardware_id, model_id, error_message, error_traceback, metadata
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
          test_result_id, run_id, result.test_module, result.test_class, result.test_name,
          result.status, result.execution_time_seconds, hardware_id, model_id,
          result.error_message, result.error_traceback, json.dumps(result.metadata || {})
        ]
      )
      
      # Store assertions if provided
      if ($1) ${$1} catch($2: $1) ${$1} finally {
      conn.close()
      }
  
  def query(self, $1: string, parameters: Dict = null) -> pd.DataFrame:
    """
    Execute a SQL query against the database.
    
    Args:
      sql: SQL query string
      parameters: Parameters for the query
      
    Returns:
      DataFrame with the query results
    """
    conn = this._get_connection()
    try {
      # Execute query
      if ($1) ${$1} else ${$1} catch($2: $1) ${$1} finally {
      conn.close()
      }
  
    }
  def get_model_hardware_compatibility(self, $1: $2 | null = null,
                  $1: $2 | null = null) -> pd.DataFrame:
    """
    Get model-hardware compatibility data.
    
    Args:
      model_name: Filter by model name (optional)
      hardware_type: Filter by hardware type (optional)
      
    Returns:
      DataFrame with compatibility data
    """
    sql = """
    SELECT
      m.model_name,
      m.model_family,
      hp.hardware_type,
      hp.device_name,
      COUNT(CASE WHEN hc.is_compatible THEN 1 END) AS compatible_count,
      COUNT(CASE WHEN NOT hc.is_compatible THEN 1 END) AS incompatible_count,
      AVG(CASE WHEN hc.compatibility_score IS NOT NULL THEN hc.compatibility_score ELSE 
      CASE WHEN hc.is_compatible THEN 1.0 ELSE 0.0 END END) AS avg_compatibility_score,
      MAX(hc.created_at) AS last_tested
    FROM
      hardware_compatibility hc
    JOIN
      models m ON hc.model_id = m.model_id
    JOIN
      hardware_platforms hp ON hc.hardware_id = hp.hardware_id
    """
    
    conditions = []
    parameters = {}
    
    if ($1) {
      $1.push($2)
      parameters['model_name'] = model_name
    
    }
    if ($1) {
      $1.push($2)
      parameters['hardware_type'] = hardware_type
    
    }
    if ($1) {
      sql += " WHERE " + " AND ".join(conditions)
    
    }
    sql += """
    GROUP BY
      m.model_name, m.model_family, hp.hardware_type, hp.device_name
    """
    
    return this.query(sql, parameters)
  
  def get_performance_metrics(self, $1: $2 | null = null,
              $1: $2 | null = null,
              $1: $2 | null = null,
              $1: $2 | null = null,
              $1: boolean = true) -> pd.DataFrame:
    """
    Get performance metrics data.
    
    Args:
      model_name: Filter by model name (optional)
      hardware_type: Filter by hardware type (optional)
      batch_size: Filter by batch size (optional)
      precision: Filter by precision (optional)
      latest_only: Return only the latest results for each model-hardware combination
      
    Returns:
      DataFrame with performance metrics
    """
    sql = """
    SELECT
      m.model_name,
      m.model_family,
      hp.hardware_type,
      hp.device_name,
      pr.batch_size,
      pr.precision,
      pr.test_case,
      pr.average_latency_ms,
      pr.throughput_items_per_second,
      pr.memory_peak_mb,
      pr.created_at
    FROM
      performance_results pr
    JOIN
      models m ON pr.model_id = m.model_id
    JOIN
      hardware_platforms hp ON pr.hardware_id = hp.hardware_id
    """
    
    conditions = []
    parameters = {}
    
    if ($1) {
      $1.push($2)
      parameters['model_name'] = model_name
    
    }
    if ($1) {
      $1.push($2)
      parameters['hardware_type'] = hardware_type
    
    }
    if ($1) {
      $1.push($2)
      parameters['batch_size'] = batch_size
      
    }
    if ($1) {
      $1.push($2)
      parameters['precision'] = precision
    
    }
    if ($1) {
      sql += " WHERE " + " AND ".join(conditions)
    
    }
    if ($1) {
      sql = `$1`
      WITH ranked_results AS (
        SELECT
          *,
          ROW_NUMBER() OVER(PARTITION BY m.model_id, hp.hardware_id, pr.batch_size, pr.precision
          ORDER BY pr.created_at DESC) as rn
        FROM (${$1}) as base
      )
      SELECT * FROM ranked_results WHERE rn = 1
      """
    
    }
    return this.query(sql, parameters)
  
  def get_integration_test_summary(self, $1: $2 | null = null) -> pd.DataFrame:
    """
    Get integration test summary.
    
    Args:
      test_module: Filter by test module (optional)
      
    Returns:
      DataFrame with integration test summary
    """
    sql = """
    SELECT
      test_module,
      COUNT(*) as total_tests,
      COUNT(CASE WHEN status = 'pass' THEN 1 END) as passed,
      COUNT(CASE WHEN status = 'fail' THEN 1 END) as failed,
      COUNT(CASE WHEN status = 'error' THEN 1 END) as errors,
      COUNT(CASE WHEN status = 'skip' THEN 1 END) as skipped,
      MAX(created_at) as last_run
    FROM
      integration_test_results
    """
    
    parameters = {}
    if ($1) {
      sql += " WHERE test_module = :test_module"
      parameters['test_module'] = test_module
    
    }
    sql += " GROUP BY test_module"
    
    return this.query(sql, parameters)
  
  def get_hardware_list(self) -> pd.DataFrame:
    """Get a list of available hardware platforms."""
    sql = """
    SELECT 
      hardware_type,
      device_name,
      COUNT(*) as usage_count
    FROM 
      hardware_platforms
    GROUP BY 
      hardware_type, device_name
    ORDER BY 
      usage_count DESC
    """
    return this.query(sql)
  
  def get_model_list(self) -> pd.DataFrame:
    """Get a list of available models."""
    sql = """
    SELECT 
      model_name,
      model_family,
      modality,
      COUNT(*) as usage_count
    FROM 
      models
    GROUP BY 
      model_name, model_family, modality
    ORDER BY 
      usage_count DESC
    """
    return this.query(sql)
  
  def get_performance_comparison(self, $1: string, $1: string = "throughput") -> pd.DataFrame:
    """
    Get performance comparison across hardware platforms for a specific model.
    
    Args:
      model_name: Model name to compare
      metric: Metric to compare ("throughput", "latency", "memory")
      
    Returns:
      DataFrame with performance comparison
    """
    metric_column = "throughput_items_per_second"
    if ($1) {
      metric_column = "average_latency_ms"
    elif ($1) {
      metric_column = "memory_peak_mb"
    
    }
    sql = `$1`
    }
    WITH latest_results AS (
      SELECT 
        m.model_name,
        hp.hardware_type,
        hp.device_name,
        pr.batch_size,
        pr.precision,
        pr.${$1} as metric_value,
        ROW_NUMBER() OVER(PARTITION BY m.model_id, hp.hardware_id, pr.batch_size, pr.precision
        ORDER BY pr.created_at DESC) as rn
      FROM 
        performance_results pr
      JOIN 
        models m ON pr.model_id = m.model_id
      JOIN 
        hardware_platforms hp ON pr.hardware_id = hp.hardware_id
      WHERE 
        m.model_name = :model_name
    )
    SELECT
      model_name,
      hardware_type,
      device_name,
      batch_size,
      precision,
      metric_value
    FROM
      latest_results
    WHERE
      rn = 1
    ORDER BY
      metric_value ${$1}
    """
    
    return this.query(sql, ${$1})

# Create FastAPI app if module is run directly
$1($2) {
  """Create FastAPI app for the benchmark database API."""
  app = FastAPI(
    title="Benchmark Database API",
    description="API for storing && querying benchmark results",
    version="0.1.0"
  )
  
}
  # Add CORS middleware
  app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=true,
    allow_methods=["*"],
    allow_headers=["*"],
  )
  
  # Create API instance
  api = BenchmarkDBAPI()
  
  # Root endpoint
  @app.get("/")
  $1($2) {
    return ${$1}
  
  }
  # Health check
  @app.get("/health")
  $1($2) {
    return ${$1}
  
  }
  # Performance endpoints
  @app.post("/performance", response_model=SuccessResponse)
  $1($2) {
    result_id = api.store_performance_result(result)
    return SuccessResponse(success=true, message="Performance result stored successfully", result_id=result_id)
  
  }
  @app.get("/performance")
  def get_performance(
    $1: $2 | null = null,
    $1: $2 | null = null,
    $1: $2 | null = null,
    $1: $2 | null = null,
    $1: boolean = true
  ):
    df = api.get_performance_metrics(model_name, hardware_type, batch_size, precision, latest_only)
    return df.to_dict(orient="records")
  
  @app.get("/performance/comparison/${$1}")
  def get_performance_comparison(
    $1: string,
    $1: string = Query("throughput", enum=["throughput", "latency", "memory"])
  ):
    df = api.get_performance_comparison(model_name, metric)
    return df.to_dict(orient="records")
  
  # Compatibility endpoints
  @app.post("/compatibility", response_model=SuccessResponse)
  $1($2) {
    result_id = api.store_compatibility_result(result)
    return SuccessResponse(success=true, message="Compatibility result stored successfully", result_id=result_id)
  
  }
  @app.get("/compatibility")
  def get_compatibility(
    $1: $2 | null = null,
    $1: $2 | null = null
  ):
    df = api.get_model_hardware_compatibility(model_name, hardware_type)
    return df.to_dict(orient="records")
  
  # Integration test endpoints
  @app.post("/integration", response_model=SuccessResponse)
  $1($2) {
    result_id = api.store_integration_test_result(result)
    return SuccessResponse(success=true, message="Integration test result stored successfully", result_id=result_id)
  
  }
  @app.get("/integration")
  def get_integration(
    $1: $2 | null = null
  ):
    df = api.get_integration_test_summary(test_module)
    return df.to_dict(orient="records")
  
  # Utility endpoints
  @app.post("/query")
  $1($2) {
    df = api.query(query_request.sql, query_request.parameters)
    return df.to_dict(orient="records")
  
  }
  @app.get("/hardware")
  $1($2) {
    df = api.get_hardware_list()
    return df.to_dict(orient="records")
  
  }
  @app.get("/models")
  $1($2) {
    df = api.get_model_list()
    return df.to_dict(orient="records")
  
  }
  return app

$1($2) {
  """Command-line interface for the benchmark database API."""
  parser = argparse.ArgumentParser(description="Benchmark Database API")
  parser.add_argument("--db-path", default="./benchmark_db.duckdb",
          help="Path to the DuckDB database")
  parser.add_argument("--serve", action="store_true",
          help="Start the API server")
  parser.add_argument("--host", default="0.0.0.0",
          help="Host to bind the API server to")
  parser.add_argument("--port", type=int, default=8000,
          help="Port to bind the API server to")
  parser.add_argument("--debug", action="store_true",
          help="Enable debug logging")
  args = parser.parse_args()
  
}
  if ($1) ${$1} else {
    parser.print_help()

  }
if ($1) {
  main()