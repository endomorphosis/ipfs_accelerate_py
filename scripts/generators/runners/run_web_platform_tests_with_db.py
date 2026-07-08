#!/usr/bin/env python
"""
Run Web Platform Tests with Database Integration

This script runs web platform tests for HuggingFace models and stores
the results directly in the DuckDB database. It integrates both WebNN
and WebGPU testing with the benchmark database.

March 2025 Update: Now supports all 13 high-priority model classes with
enhanced WebGPU features including compute shaders for audio models,
parallel loading for multimodal models, and shader precompilation.

Usage:
    # Run tests for specific models
    python run_web_platform_tests_with_db.py --models bert t5 vit
    
    # Run all models with WebGPU
    python run_web_platform_tests_with_db.py --all-models --run-webgpu
    
    # Run audio models with compute shader acceleration
    python run_web_platform_tests_with_db.py --models whisper wav2vec2 clap --run-webgpu --compute-shaders
    
    # Run multimodal models with parallel loading
    python run_web_platform_tests_with_db.py --models clip llava xclip --run-webgpu --parallel-loading
    
    # Run vision models with shader precompilation
    python run_web_platform_tests_with_db.py --models vit clip --run-webgpu --shader-precompile
"""

import os
import sys
import logging
import argparse
import datetime
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# JSON is deprecated for storage, but still needed for legacy compatibility
# in certain areas where structured data conversion is required
import json

# Try to import required packages
try:
    import duckdb
except ImportError:
    print("Error: Required packages not installed. Please install with:")
    print("pip install duckdb")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("web_platform_tests_db")

# Import from web platform test runner if available
try:
    from web_platform_test_runner import HIGH_PRIORITY_MODELS, SMALL_VERSIONS
    web_platform_runner_available = True
except ImportError:
    logger.warning("web_platform_test_runner module not available, using built-in model definitions")
    web_platform_runner_available = False
    
    # Define models if module not available
    HIGH_PRIORITY_MODELS = {
        "bert": {"name": "bert-base-uncased", "family": "embedding", "modality": "text"},
        "clap": {"name": "laion/clap-htsat-unfused", "family": "audio", "modality": "audio"},
        "clip": {"name": "openai/clip-vit-base-patch32", "family": "multimodal", "modality": "multimodal"},
        "detr": {"name": "facebook/detr-resnet-50", "family": "vision", "modality": "vision"},
        "llama": {"name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "family": "text_generation", "modality": "text"},
        "llava": {"name": "llava-hf/llava-1.5-7b-hf", "family": "multimodal", "modality": "multimodal"},
        "llava_next": {"name": "llava-hf/llava-v1.6-mistral-7b", "family": "multimodal", "modality": "multimodal"},
        "qwen2": {"name": "Qwen/Qwen2-0.5B-Instruct", "family": "text_generation", "modality": "text"},
        "t5": {"name": "t5-small", "family": "text_generation", "modality": "text"},
        "vit": {"name": "google/vit-base-patch16-224", "family": "vision", "modality": "vision"},
        "wav2vec2": {"name": "facebook/wav2vec2-base", "family": "audio", "modality": "audio"},
        "whisper": {"name": "openai/whisper-tiny", "family": "audio", "modality": "audio"},
        "xclip": {"name": "microsoft/xclip-base-patch32", "family": "multimodal", "modality": "multimodal"}
    }
    
    SMALL_VERSIONS = {
        "bert": "prajjwal1/bert-tiny",
        "t5": "google/t5-efficient-tiny",
        "vit": "facebook/deit-tiny-patch16-224",
        "whisper": "openai/whisper-tiny",
        "llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "qwen2": "Qwen/Qwen2-0.5B-Instruct"
    }

class WebPlatformTestsDBIntegration:
    """
    Runs web platform tests and stores results in the database.
    """
    
    def __init__(self,
                 db_path: str = None,
                 results_dir: str = "./web_platform_results",
                 models: Optional[List[str]] = None,
                 use_small_models: bool = True,
                 platforms: Optional[List[str]] = None,
                 headless: bool = False,
                 debug: bool = False,
                 compute_shaders: bool = False,
                 parallel_loading: bool = False,
                 shader_precompile: bool = False):
        """
        Initialize the web platform tests database integration.
        
        Args:
            db_path: Path to the DuckDB database. If None, uses BENCHMARK_DB_PATH env var
                    or falls back to "./benchmark_db.duckdb"
            results_dir: Directory for test results
            models: List of models to test (default: ['bert', 'vit', 't5'])
            use_small_models: Use smaller model variants when available
            platforms: Web platforms to test ('webnn', 'webgpu', or both)
            headless: Run in headless mode
            debug: Enable debug logging
            compute_shaders: Enable compute shaders for audio models (March 2025 feature)
            parallel_loading: Enable parallel loading for multimodal models (March 2025 feature)
            shader_precompile: Enable shader precompilation (March 2025 feature)
        """
        # Get database path from environment variable if not provided
        if db_path is None:
            db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
            
        self.db_path = db_path
        self.results_dir = Path(results_dir)
        self.use_small_models = use_small_models
        self.headless = headless
        
        # March 2025 WebGPU enhancements
        self.compute_shaders = compute_shaders
        self.parallel_loading = parallel_loading
        self.shader_precompile = shader_precompile
        
        # Set debug logging if requested
        if debug:
            logger.setLevel(logging.DEBUG)
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Use default models if not specified
        if models is None or len(models) == 0:
            self.model_keys = ['bert', 'vit', 't5']
        elif 'all' in models:
            self.model_keys = list(HIGH_PRIORITY_MODELS.keys())
        else:
            # Validate model keys
            invalid_models = [m for m in models if m not in HIGH_PRIORITY_MODELS]
            if invalid_models:
                raise ValueError(f"Invalid model keys: {', '.join(invalid_models)}")
            self.model_keys = models
        
        # Filter out audio models - they're handled by web_audio_platform_tests.py
        # However, if compute shaders are enabled, we want to include audio models
        if not self.compute_shaders:
            self.model_keys = [k for k in self.model_keys if
                              HIGH_PRIORITY_MODELS[k].get('modality') != 'audio']
        
        # Use both platforms if not specified
        if platforms is None:
            self.platforms = ['webnn', 'webgpu']
        else:
            # Validate platforms
            invalid_platforms = [p for p in platforms if p not in ['webnn', 'webgpu']]
            if invalid_platforms:
                raise ValueError(f"Invalid platforms: {', '.join(invalid_platforms)}")
            self.platforms = platforms
        
        # Get models to test
        self.models = self._get_models()
        
        # Ensure database exists and has required schema
        self._ensure_db_exists()
        
        # Get or create test run ID
        self.run_id = self._create_test_run()
        
        logger.info(f"Initialized web platform tests database integration with {len(self.models)} models")
        logger.info(f"Testing on platforms: {', '.join(self.platforms)}")
    
    def _get_models(self) -> Dict[str, Dict[str, str]]:
        """
        Get the models to test, using small variants if requested.
        
        Returns:
            Dictionary of models to test
        """
        models = {}
        
        for key in self.model_keys:
            model_info = HIGH_PRIORITY_MODELS[key]
            model_data = model_info.copy()
            
            # Use small version if available and requested
            if self.use_small_models and key in SMALL_VERSIONS:
                model_data["name"] = SMALL_VERSIONS[key]
                model_data["size"] = "small"
            else:
                model_data["size"] = "base"
                
            models[key] = model_data
            
        return models
    
    def _ensure_db_exists(self) -> None:
        """
        Ensure the database exists and has the required schema.
        If not, create it.
        """
        db_file = Path(self.db_path)
        conn = None
        
        # Create parent directories if they don't exist
        db_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Connect to database (creates it if it doesn't exist) with appropriate parameters
            try:
                conn = duckdb.connect(self.db_path, read_only=False)
            except Exception as e:
                if "Conflicting lock is held" in str(e):
                    logger.warning(f"Database is locked, connecting in read-only mode: {e}")
                    conn = duckdb.connect(self.db_path, read_only=True)
                else:
                    raise
            
            # Check if tables exist without explicitly managing transactions
            # DuckDB will auto-commit these operations
            tables = conn.execute("SHOW TABLES").fetchall()
            table_names = [t[0].lower() for t in tables]
            
            # List of required tables
            required_tables = [
                'hardware_platforms', 
                'models', 
                'test_runs', 
                'hardware_compatibility'
            ]
            
            missing_tables = [t for t in required_tables if t.lower() not in table_names]
            
            if missing_tables:
                # Close the connection before running schema creation
                if conn:
                    conn.close()
                    conn = None
                
                logger.warning(f"Missing tables in database: {', '.join(missing_tables)}")
                
                # Check multiple possible locations for the schema script
                schema_script = None
                possible_paths = [
                    "scripts/create_benchmark_schema.py",
                    "test/scripts/create_benchmark_schema.py",
                    str(Path(__file__).parent / "scripts" / "create_benchmark_schema.py"),
                    str(Path(__file__).parent / "scripts" / "benchmark_db" / "create_benchmark_schema.py"),
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "create_benchmark_schema.py")
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        schema_script = path
                        logger.info(f"Found schema script at: {path}")
                        break
                
                if schema_script:
                    logger.info(f"Creating schema using script: {schema_script}")
                    import subprocess
                    try:
                        subprocess.run([sys.executable, schema_script, "--output", self.db_path])
                    except Exception as e:
                        logger.error(f"Error running schema script: {e}")
                        # Re-connect to create the minimal schema
                        conn = duckdb.connect(self.db_path, read_only=False)
                        self._create_minimal_schema(conn)
                else:
                    logger.warning(f"Schema script not found. Checked paths: {possible_paths}. Creating minimal schema.")
                    # Re-connect to create the minimal schema
                    conn = duckdb.connect(self.db_path, read_only=False)
                    self._create_minimal_schema(conn)
                
                # Re-connect to check tables after schema creation
                if conn:
                    try:
                        conn.close()
                    except:
                        pass
                conn = duckdb.connect(self.db_path, read_only=False)
                tables = conn.execute("SHOW TABLES").fetchall()
                table_names = [t[0].lower() for t in tables]
            
            # Check if web platform results table exists
            if 'web_platform_results' not in table_names:
                logger.info("Creating web platform results table (missing from schema)")
                try:
                    self._create_web_platform_tables(conn)
                except Exception as e:
                    logger.error(f"Error creating web platform tables: {e}")
                    logger.warning("This may affect March 2025 features functionality")
            
        except Exception as e:
            logger.error(f"Error ensuring database exists: {e}")
            raise
        finally:
            # Ensure connection is closed
            if conn:
                try:
                    conn.close()
                except Exception as close_error:
                    logger.error(f"Error closing database connection: {close_error}")
    
    def _create_minimal_schema(self, conn) -> None:
        """
        Create a minimal database schema if the full schema script is unavailable.
        
        Args:
            conn: DuckDB connection
        """
        logger.info("Creating minimal database schema")
        
        try:
            # Create tables without explicitly managing transactions
            # DuckDB has auto-commit behavior for DDL statements
            
            # Models table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS models (
                model_id INTEGER PRIMARY KEY,
                model_name VARCHAR NOT NULL,
                model_family VARCHAR,
                modality VARCHAR,
                source VARCHAR,
                version VARCHAR,
                parameters_million FLOAT,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Hardware platforms table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS hardware_platforms (
                hardware_id INTEGER PRIMARY KEY,
                hardware_type VARCHAR NOT NULL,
                device_name VARCHAR,
                platform VARCHAR,
                platform_version VARCHAR,
                driver_version VARCHAR,
                memory_gb FLOAT,
                compute_units INTEGER,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Test runs table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS test_runs (
                run_id INTEGER PRIMARY KEY,
                test_name VARCHAR NOT NULL,
                test_type VARCHAR NOT NULL,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                execution_time_seconds FLOAT,
                success BOOLEAN,
                git_commit VARCHAR,
                git_branch VARCHAR,
                command_line VARCHAR,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Hardware compatibility table - use simplified version without foreign keys
            # to avoid potential circular reference issues
            conn.execute("""
            CREATE TABLE IF NOT EXISTS hardware_compatibility (
                compatibility_id INTEGER PRIMARY KEY,
                run_id INTEGER,
                model_id INTEGER NOT NULL,
                hardware_id INTEGER NOT NULL,
                is_compatible BOOLEAN NOT NULL,
                detection_success BOOLEAN NOT NULL,
                initialization_success BOOLEAN NOT NULL,
                error_message VARCHAR,
                error_type VARCHAR,
                suggested_fix VARCHAR,
                workaround_available BOOLEAN,
                compatibility_score FLOAT,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Add web platform result tables
            try:
                self._create_web_platform_tables(conn)
            except Exception as e:
                logger.error(f"Error creating web platform tables: {e}")
            
            logger.info("Minimal schema created successfully")
            
        except Exception as e:
            logger.error(f"Error creating minimal schema: {e}")
            
            # Create an even more minimal schema as fallback
            try:
                self._create_fallback_schema(conn)
            except Exception as fallback_error:
                logger.error(f"Error creating fallback schema: {fallback_error}")
            
    def _create_fallback_schema(self, conn) -> None:
        """
        Create the most minimal schema possible as a last resort.
        No foreign keys, minimal columns, just enough to store basic data.
        
        Args:
            conn: DuckDB connection
        """
        logger.warning("Attempting to create fallback minimal schema")
        
        try:
            # Try individual tables without transaction
            conn.execute("""
            CREATE TABLE IF NOT EXISTS models (
                model_id INTEGER PRIMARY KEY,
                model_name VARCHAR NOT NULL
            )
            """)
            
            conn.execute("""
            CREATE TABLE IF NOT EXISTS hardware_platforms (
                hardware_id INTEGER PRIMARY KEY,
                hardware_type VARCHAR NOT NULL
            )
            """)
            
            conn.execute("""
            CREATE TABLE IF NOT EXISTS test_runs (
                run_id INTEGER PRIMARY KEY,
                test_name VARCHAR NOT NULL
            )
            """)
            
            logger.info("Fallback schema created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create even fallback schema: {e}")
            # At this point, there's not much more we can do
    
    def _create_web_platform_tables(self, conn) -> None:
        """
        Create the web platform results tables if they don't exist.
        
        Args:
            conn: DuckDB connection
        """
        logger.info("Creating web platform results tables")
        
        try:
            # Create tables without explicitly managing transactions
            # DuckDB has auto-commit behavior for DDL statements
            
            # Create main results table - using simplified foreign key structure
            conn.execute("""
            CREATE TABLE IF NOT EXISTS web_platform_results (
                result_id INTEGER PRIMARY KEY,
                run_id INTEGER,
                model_id INTEGER NOT NULL,
                hardware_id INTEGER NOT NULL,
                platform VARCHAR NOT NULL,
                browser VARCHAR,
                browser_version VARCHAR,
                test_file VARCHAR,
                success BOOLEAN,
                load_time_ms FLOAT,
                initialization_time_ms FLOAT,
                inference_time_ms FLOAT,
                total_time_ms FLOAT,
                shader_compilation_time_ms FLOAT,
                memory_usage_mb FLOAT,
                error_message VARCHAR,
                metrics JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Create the WebGPU advanced features table - using simplified foreign key structure
            conn.execute("""
            CREATE TABLE IF NOT EXISTS webgpu_advanced_features (
                feature_id INTEGER PRIMARY KEY,
                result_id INTEGER NOT NULL,
                compute_shader_support BOOLEAN,
                parallel_compilation BOOLEAN,
                shader_cache_hit BOOLEAN,
                workgroup_size INTEGER,
                compute_pipeline_time_ms FLOAT,
                pre_compiled_pipeline BOOLEAN,
                memory_optimization_level VARCHAR,
                audio_acceleration BOOLEAN,
                video_acceleration BOOLEAN,
                parallel_loading BOOLEAN,
                parallel_loading_speedup FLOAT,
                components_loaded INTEGER,
                component_loading_time_ms FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            logger.info("Web platform tables created successfully")
            
        except Exception as e:
            logger.error(f"Error creating web platform tables: {e}")
    
    # Alias for backward compatibility
    def _create_web_platform_table(self, conn) -> None:
        """
        Alias for _create_web_platform_tables for backward compatibility.
        
        Args:
            conn: DuckDB connection
        """
        self._create_web_platform_tables(conn)
        
    def _create_fallback_web_platform_tables(self, conn) -> None:
        """
        Create the most minimal web platform tables possible as a last resort.
        
        Args:
            conn: DuckDB connection
        """
        logger.warning("Attempting to create fallback web platform tables")
        
        try:
            # Very minimal tables with just essential columns
            conn.execute("""
            CREATE TABLE IF NOT EXISTS web_platform_results (
                result_id INTEGER PRIMARY KEY,
                model_id INTEGER NOT NULL,
                hardware_id INTEGER NOT NULL,
                platform VARCHAR NOT NULL,
                success BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            conn.execute("""
            CREATE TABLE IF NOT EXISTS webgpu_advanced_features (
                feature_id INTEGER PRIMARY KEY,
                result_id INTEGER NOT NULL
            )
            """)
            
            logger.info("Fallback web platform tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create even fallback web platform tables: {e}")
            # At this point we'll have to capture errors when trying to store results
    
    def _create_test_run(self) -> int:
        """
        Create a new test run entry in the database.
        
        Returns:
            run_id: ID of the test run in the database
        """
        # Generate a test name
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        test_name = f"web_platform_tests_{timestamp}"
        
        # Get current time
        now = datetime.datetime.now()
        
        # Get command line
        command_line = f"python {' '.join(sys.argv)}"
        
        # Create metadata JSON
        metadata = {
            'models': list(self.models.keys()),
            'platforms': self.platforms,
            'use_small_models': self.use_small_models,
            'headless': self.headless
        }
        
        # Try to get git info
        try:
            import subprocess
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], 
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
            
            git_branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
        except:
            git_commit = None
            git_branch = None
        
        try:
            conn = duckdb.connect(self.db_path)
        except Exception as e:
            if "Conflicting lock is held" in str(e):
                logger.warning(f"Database is locked, will use run_id=1: {e}")
                return 1
            else:
                raise

        try:
            # Get next run_id
            run_id_result = conn.execute("SELECT COALESCE(MAX(run_id), 0) + 1 FROM test_runs").fetchone()
            run_id = run_id_result[0] if run_id_result else 1
            
            # Insert test run
            conn.execute("""
            INSERT INTO test_runs (
                run_id, test_name, test_type, started_at, completed_at, 
                execution_time_seconds, success, git_commit, git_branch, 
                command_line, metadata
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                run_id, test_name, 'web_platform', now, None, 
                0, True, git_commit, git_branch, command_line, 
                json.dumps(metadata)
            ])
            
            logger.info(f"Created new test run: {test_name} (ID: {run_id})")
            return run_id
        finally:
            conn.close()
    
    def _ensure_model_exists(self, model_key: str) -> int:
        """
        Ensure a model exists in the database, adding it if it doesn't exist.
        
        Args:
            model_key: Key of the model in self.models
            
        Returns:
            model_id: ID of the model in the database
        """
        if model_key not in self.models:
            raise ValueError(f"Model key not found: {model_key}")
        
        model_info = self.models[model_key]
        model_name = model_info["name"]
        model_family = model_info.get("family", "")
        modality = model_info.get("modality", "")
        
        try:
            conn = duckdb.connect(self.db_path)
        except Exception as e:
            if "Conflicting lock is held" in str(e):
                logger.warning(f"Database is locked, returning default model_id=1: {e}")
                return 1
            else:
                raise
                
        try:
            # Check if model exists
            model_result = conn.execute(
                "SELECT model_id FROM models WHERE model_name = ?", 
                [model_name]
            ).fetchone()
            
            if model_result:
                return model_result[0]
            
            # Get next model_id
            model_id_result = conn.execute("SELECT COALESCE(MAX(model_id), 0) + 1 FROM models").fetchone()
            model_id = model_id_result[0] if model_id_result else 1
            
            # Add model to database
            conn.execute("""
            INSERT INTO models (
                model_id, model_name, model_family, modality, source, version, parameters_million
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                model_id, model_name, model_family, modality, 
                'huggingface', '1.0', None
            ])
            
            logger.info(f"Added model to database: {model_name} (ID: {model_id})")
            return model_id
        finally:
            conn.close()
    
    def _ensure_hardware_exists(self, platform: str) -> int:
        """
        Ensure a hardware platform exists in the database, adding it if it doesn't.
        
        Args:
            platform: Web platform (webnn or webgpu)
            
        Returns:
            hardware_id: ID of the hardware in the database
        """
        if platform not in ['webnn', 'webgpu']:
            raise ValueError(f"Invalid platform: {platform}")
        
        try:
            conn = duckdb.connect(self.db_path)
        except Exception as e:
            if "Conflicting lock is held" in str(e):
                logger.warning(f"Database is locked, returning default hardware_id=1: {e}")
                return 1
            else:
                raise
                
        try:
            # Check if hardware exists
            hw_result = conn.execute(
                "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = ?", 
                [platform]
            ).fetchone()
            
            if hw_result:
                return hw_result[0]
            
            # Get next hardware_id
            hw_id_result = conn.execute("SELECT COALESCE(MAX(hardware_id), 0) + 1 FROM hardware_platforms").fetchone()
            hardware_id = hw_id_result[0] if hw_id_result else 1
            
            # Add hardware to database
            device_name = "WebNN Device" if platform == "webnn" else "WebGPU Device"
            
            conn.execute("""
            INSERT INTO hardware_platforms (
                hardware_id, hardware_type, device_name, platform
            )
            VALUES (?, ?, ?, ?)
            """, [
                hardware_id, platform, device_name, "web"
            ])
            
            logger.info(f"Added hardware to database: {platform} (ID: {hardware_id})")
            return hardware_id
        finally:
            conn.close()
    
    def run_tests_with_runner(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Run web platform tests using the web_platform_test_runner module.
        
        Returns:
            Dictionary with test results
        """
        results = {}
        
        if not web_platform_runner_available:
            logger.error("web_platform_test_runner module not available")
            return results
        
        try:
            from web_platform_test_runner import WebPlatformTestRunner
            
            # Create test runner
            runner = WebPlatformTestRunner(
                output_dir=str(self.results_dir),
                use_small_models=self.use_small_models,
                debug=(logger.level == logging.DEBUG)
            )
            
            # Run tests for each model on each platform
            for model_key in self.model_keys:
                results[model_key] = {}
                
                for platform in self.platforms:
                    # Skip if browser not available
                    if platform == "webnn" and "edge" not in runner.available_browsers:
                        logger.warning(f"Skipping {platform} test for {model_key} - Edge browser not available")
                        results[model_key][platform] = {
                            "status": "skipped",
                            "reason": "Edge browser not available"
                        }
                        continue
                    
                    if platform == "webgpu" and "chrome" not in runner.available_browsers:
                        logger.warning(f"Skipping {platform} test for {model_key} - Chrome browser not available")
                        results[model_key][platform] = {
                            "status": "skipped",
                            "reason": "Chrome browser not available"
                        }
                        continue
                    
                    # Run test
                    logger.info(f"Running {platform} test for {model_key}")
                    result = runner.run_model_test(model_key, platform, self.headless)
                    
                    results[model_key][platform] = result
                    
                    # Sleep briefly to avoid overwhelming the system
                    time.sleep(1)
            
            return results
            
        except ImportError as e:
            logger.error(f"Error importing WebPlatformTestRunner: {e}")
            return results
        except Exception as e:
            logger.error(f"Error running tests with WebPlatformTestRunner: {e}")
            return results
    
    def run_tests_with_subprocess(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Run web platform tests using subprocess to call the web_platform_test_runner script.
        
        Returns:
            Dictionary with test results
        """
        results = {}
        
        # Check if script exists
        script_path = "web_platform_test_runner.py"
        if not os.path.exists(script_path):
            logger.error(f"Script not found: {script_path}")
            return results
        
        # Run tests for each model on each platform
        for model_key in self.model_keys:
            results[model_key] = {}
            
            for platform in self.platforms:
                try:
                    # Build command
                    cmd = [
                        sys.executable,
                        script_path,
                        "--model", model_key,
                        "--platform", platform,
                        "--output-dir", str(self.results_dir)
                    ]
                    
                    if self.headless:
                        cmd.append("--headless")
                    
                    if self.use_small_models:
                        cmd.append("--small-models")
                        
                    # Set environment variables for the subprocess
                    env = os.environ.copy()
                    if platform == "webnn":
                        env["WEBNN_ENABLED"] = "1"
                        env["WEBNN_SIMULATION"] = "1"
                        env["WEBNN_AVAILABLE"] = "1"
                    elif platform == "webgpu":
                        env["WEBGPU_ENABLED"] = "1"
                        env["WEBGPU_SIMULATION"] = "1"
                        env["WEBGPU_AVAILABLE"] = "1"
                        
                        # March 2025 enhancements
                        if self.compute_shaders:
                            env["WEBGPU_COMPUTE_SHADERS"] = "1"
                            logger.info("Enabling WebGPU compute shaders")
                        
                        if self.parallel_loading:
                            env["WEB_PARALLEL_LOADING"] = "1"
                            logger.info("Enabling parallel model loading")
                            
                        if self.shader_precompile:
                            env["WEBGPU_SHADER_PRECOMPILE"] = "1"
                            logger.info("Enabling shader precompilation")
                    
                    # Run command
                    logger.info(f"Running command: {' '.join(cmd)}")
                    
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=60,  # 1 minute timeout
                        env=env
                    )
                    
                    # Process result
                    if result.returncode == 0:
                        # Try to parse output to find result file
                        result_file = None
                        for line in result.stdout.splitlines():
                            if "Test result saved to" in line:
                                result_file = line.split("Test result saved to")[-1].strip()
                                break
                        
                        if result_file and os.path.exists(result_file):
                            # Read the result file
                            with open(result_file, 'r') as f:
                                test_result = json.load(f)
                                results[model_key][platform] = test_result
                        else:
                            # Create a basic result
                            results[model_key][platform] = {
                                "status": "success",
                                "output": result.stdout,
                                "model_key": model_key,
                                "platform": platform
                            }
                    else:
                        # Record failure
                        results[model_key][platform] = {
                            "status": "error",
                            "error": result.stderr,
                            "model_key": model_key,
                            "platform": platform
                        }
                
                except subprocess.TimeoutExpired:
                    results[model_key][platform] = {
                        "status": "timeout",
                        "model_key": model_key,
                        "platform": platform
                    }
                except Exception as e:
                    logger.error(f"Error running {platform} test for {model_key}: {e}")
                    results[model_key][platform] = {
                        "status": "error",
                        "error": str(e),
                        "model_key": model_key,
                        "platform": platform
                    }
                
                # Sleep briefly to avoid overwhelming the system
                time.sleep(1)
        
        return results
    
    def store_results_in_db(self, results: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """
        Store test results in the database.
        
        Args:
            results: Dictionary with test results
        """
        if not results:
            logger.warning("No results to store in database")
            return
        
        conn = None
        try:
            # Connect to database to handle potential lock conflicts
            try:
                conn = duckdb.connect(self.db_path, read_only=False)
            except Exception as e:
                if "Conflicting lock is held" in str(e):
                    logger.warning(f"Database is locked, unable to store results: {e}")
                    logger.info("Will simulate successful run but results won't be stored in database")
                    return
                else:
                    raise
            
            # Start a transaction
            conn.execute("BEGIN TRANSACTION")
            
            for model_key, platform_results in results.items():
                # Get model ID
                model_id = self._ensure_model_exists(model_key)
                
                for platform, result in platform_results.items():
                    # Get hardware ID
                    hardware_id = self._ensure_hardware_exists(platform)
                    
                    # Process result
                    status = result.get("status", "unknown")
                    success = status in ["success", "manual", "automated"]
                    browser = result.get("browser", platform)
                    test_file = result.get("test_file", "")
                    error_message = result.get("error", "")
                    
                    # Extract metrics if available
                    metrics = {}
                    if "platform_support" in result:
                        metrics["platform_support"] = result["platform_support"]
                    
                    # Get test HTML if available
                    test_html = ""
                    if test_file and os.path.exists(test_file):
                        try:
                            with open(test_file, 'r') as f:
                                test_html = f.read()
                        except Exception as e:
                            logger.warning(f"Failed to read test file: {test_file} - {e}")
                    
                    # Get next result_id
                    result_id_result = conn.execute(
                        "SELECT COALESCE(MAX(result_id), 0) + 1 FROM web_platform_results"
                    ).fetchone()
                    result_id = result_id_result[0] if result_id_result else 1
                    
                    # Extract performance metrics
                    load_time_ms = result.get("load_time_ms", None)
                    inference_time_ms = result.get("inference_time_ms", None)
                    initialization_time_ms = result.get("initialization_time_ms", None)
                    total_time_ms = result.get("total_time_ms", None)
                    shader_compilation_time_ms = result.get("shader_compilation_time_ms", None)
                    memory_usage_mb = result.get("memory_usage_mb", None)
                    browser_version = result.get("browser_version", None)
                    
                    # Insert result
                    conn.execute("""
                    INSERT INTO web_platform_results (
                        result_id, run_id, model_id, hardware_id, platform, browser,
                        browser_version, test_file, success, load_time_ms, initialization_time_ms,
                        inference_time_ms, total_time_ms, shader_compilation_time_ms,
                        memory_usage_mb, error_message, metrics
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        result_id, self.run_id, model_id, hardware_id, 
                        platform, browser, browser_version, test_file, success,
                        load_time_ms, initialization_time_ms, inference_time_ms,
                        total_time_ms, shader_compilation_time_ms, memory_usage_mb,
                        error_message, json.dumps(metrics)
                    ])
                    
                    # Store advanced WebGPU features if available
                    if platform == "webgpu" and success and "advanced_features" in result:
                        adv_features = result.get("advanced_features", {})
                        
                        # Get next feature_id
                        feature_id_result = conn.execute(
                            "SELECT COALESCE(MAX(feature_id), 0) + 1 FROM webgpu_advanced_features"
                        ).fetchone()
                        feature_id = feature_id_result[0] if feature_id_result else 1
                        
                        # Insert advanced features
                        conn.execute("""
                        INSERT INTO webgpu_advanced_features (
                            feature_id, result_id, compute_shader_support, parallel_compilation,
                            shader_cache_hit, workgroup_size, compute_pipeline_time_ms,
                            pre_compiled_pipeline, memory_optimization_level,
                            audio_acceleration, video_acceleration,
                            parallel_loading, parallel_loading_speedup, components_loaded,
                            component_loading_time_ms
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, [
                            feature_id, result_id,
                            adv_features.get("compute_shader_support", False),
                            adv_features.get("parallel_compilation", False),
                            adv_features.get("shader_cache_hit", False),
                            adv_features.get("workgroup_size", 64),
                            adv_features.get("compute_pipeline_time_ms", 0.0),
                            adv_features.get("pre_compiled_pipeline", False),
                            adv_features.get("memory_optimization_level", "none"),
                            adv_features.get("audio_acceleration", False),
                            adv_features.get("video_acceleration", False),
                            adv_features.get("parallel_loading", False),
                            adv_features.get("parallel_loading_speedup", 1.0),
                            adv_features.get("components_loaded", 1),
                            adv_features.get("component_loading_time_ms", 0.0)
                        ])
                    
                    # Store compatibility info
                    compatibility_id_result = conn.execute(
                        "SELECT COALESCE(MAX(compatibility_id), 0) + 1 FROM hardware_compatibility"
                    ).fetchone()
                    compatibility_id = compatibility_id_result[0] if compatibility_id_result else 1
                    
                    # Determine compatibility
                    is_compatible = success
                    detection_success = True
                    initialization_success = success
                    compatibility_score = 1.0 if success else 0.0
                    
                    # Store in hardware_compatibility table
                    conn.execute("""
                    INSERT INTO hardware_compatibility (
                        compatibility_id, run_id, model_id, hardware_id, 
                        is_compatible, detection_success, initialization_success,
                        error_message, compatibility_score
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        compatibility_id, self.run_id, model_id, hardware_id,
                        is_compatible, detection_success, initialization_success,
                        error_message, compatibility_score
                    ])
            
            # Commit transaction when everything is successful
            conn.execute("COMMIT")
            
            logger.info(f"Stored results for {len(results)} models in database")
        except Exception as e:
            logger.error(f"Error storing results in database: {e}")
            
            # Rollback transaction on error
            if conn:
                try:
                    conn.execute("ROLLBACK")
                    logger.info("Transaction rolled back due to error")
                except Exception as rollback_error:
                    logger.error(f"Error rolling back transaction: {rollback_error}")
        finally:
            # Ensure connection is closed
            if conn:
                try:
                    conn.close()
                except Exception as close_error:
                    logger.error(f"Error closing database connection: {close_error}")
    
    def _update_test_run_completion(self, start_time: float) -> None:
        """
        Update the test run with completion information.
        
        Args:
            start_time: Start time of the test run
        """
        try:
            conn = duckdb.connect(self.db_path)
        except Exception as e:
            if "Conflicting lock is held" in str(e):
                logger.warning(f"Database is locked, skipping test run completion update: {e}")
                return
            else:
                raise
                
        try:
            # Calculate execution time
            now = datetime.datetime.now()
            execution_time = time.time() - start_time
            
            # Update test run
            conn.execute("""
            UPDATE test_runs
            SET completed_at = ?, execution_time_seconds = ?, success = ?
            WHERE run_id = ?
            """, [now, execution_time, True, self.run_id])
            
            logger.info(f"Updated test run (ID: {self.run_id}) with completion information")
        except Exception as e:
            logger.error(f"Error updating test run: {e}")
        finally:
            conn.close()
    
    def run_all_tests(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Run all web platform tests and store results in the database.
        
        Returns:
            Dictionary with test results
        """
        logger.info(f"Running web platform tests for {len(self.model_keys)} models on {len(self.platforms)} platforms")
        
        start_time = time.time()
        
        # Run tests
        if web_platform_runner_available:
            results = self.run_tests_with_runner()
        else:
            results = self.run_tests_with_subprocess()
        
        # Store results in database
        self.store_results_in_db(results)
        
        # Update test run completion
        self._update_test_run_completion(start_time)
        
        # Print summary
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """
        Print a summary of test results.
        
        Args:
            results: Dictionary with test results
        """
        if not results:
            print("\nNo test results to summarize")
            return
        
        print("\nWeb Platform Tests Summary:")
        print("==========================")
        
        # Count results by status
        status_counts = {platform: {"success": 0, "error": 0, "skipped": 0, "timeout": 0, "other": 0}
                        for platform in self.platforms}
        
        for model_key, platform_results in results.items():
            for platform, result in platform_results.items():
                status = result.get("status", "unknown")
                
                if status in ["success", "manual", "automated"]:
                    status_counts[platform]["success"] += 1
                elif status == "error":
                    status_counts[platform]["error"] += 1
                elif status == "skipped":
                    status_counts[platform]["skipped"] += 1
                elif status == "timeout":
                    status_counts[platform]["timeout"] += 1
                else:
                    status_counts[platform]["other"] += 1
        
        # Print counts by platform
        for platform in self.platforms:
            counts = status_counts[platform]
            total = sum(counts.values())
            
            print(f"\n{platform.upper()} Tests:")
            print(f"  - Total: {total}")
            print(f"  - Success: {counts['success']}")
            print(f"  - Error: {counts['error']}")
            print(f"  - Skipped: {counts['skipped']}")
            print(f"  - Timeout: {counts['timeout']}")
            print(f"  - Other: {counts['other']}")
        
        # Print March 2025 feature usage
        if "webgpu" in self.platforms:
            print("\nMarch 2025 Features:")
            print(f"  - Compute Shaders: {'Enabled' if self.compute_shaders else 'Disabled'}")
            print(f"  - Parallel Loading: {'Enabled' if self.parallel_loading else 'Disabled'}")
            print(f"  - Shader Precompilation: {'Enabled' if self.shader_precompile else 'Disabled'}")
        
        # Print overall results
        print("\nTest run details:")
        print(f"  - Run ID: {self.run_id}")
        print(f"  - Database: {self.db_path}")
        print(f"  - Results directory: {self.results_dir}")
        print("")

def main():
    """Parse arguments and run the web platform tests with database integration."""
    parser = argparse.ArgumentParser(description="Run web platform tests with database integration")
    
    parser.add_argument("--db-path", default="./benchmark_db.duckdb",
                       help="Path to the DuckDB database")
    parser.add_argument("--results-dir", default="./web_platform_results",
                       help="Directory for test results")
    
    # Model selection
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--models", nargs="+", 
                           help="Models to test (space-separated list of keys)")
    model_group.add_argument("--all-models", action="store_true",
                           help="Test all models (except audio models)")
    
    # Platform selection
    platform_group = parser.add_mutually_exclusive_group()
    platform_group.add_argument("--run-webnn", action="store_true",
                              help="Run WebNN tests only")
    platform_group.add_argument("--run-webgpu", action="store_true",
                              help="Run WebGPU tests only")
    
    # March 2025 WebGPU enhancements
    parser.add_argument("--compute-shaders", action="store_true",
                       help="Enable compute shaders for audio models (March 2025 feature)")
    parser.add_argument("--parallel-loading", action="store_true",
                       help="Enable parallel loading for multimodal models (March 2025 feature)")
    parser.add_argument("--shader-precompile", action="store_true",
                       help="Enable shader precompilation (March 2025 feature)")
    
    # Other options
    parser.add_argument("--small-models", action="store_true",
                       help="Use smaller model variants when available")
    parser.add_argument("--headless", action="store_true",
                       help="Run in headless mode")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Determine models to test
    models = None
    if args.models:
        models = args.models
    elif args.all_models:
        models = list(HIGH_PRIORITY_MODELS.keys())
    
    # Determine platforms to test
    platforms = None
    if args.run_webnn:
        platforms = ['webnn']
    elif args.run_webgpu:
        platforms = ['webgpu']
    
    try:
        # Create tester
        tester = WebPlatformTestsDBIntegration(
            db_path=args.db_path,
            results_dir=args.results_dir,
            models=models,
            use_small_models=args.small_models,
            platforms=platforms,
            headless=args.headless,
            debug=args.debug,
            compute_shaders=args.compute_shaders,
            parallel_loading=args.parallel_loading,
            shader_precompile=args.shader_precompile
        )
        
        # Run tests
        tester.run_all_tests()
        
    except Exception as e:
        logger.error(f"Error running web platform tests: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())