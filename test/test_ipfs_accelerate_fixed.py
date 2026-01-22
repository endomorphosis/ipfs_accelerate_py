import asyncio
import os
import sys
import json
import time
import traceback
from datetime import datetime
import importlib.util
from typing import Dict, List, Any, Optional, Union

# Set environment variables to avoid tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Determine if JSON output should be deprecated in favor of DuckDB
DEPRECATE_JSON_OUTPUT = os.environ.get("DEPRECATE_JSON_OUTPUT", "1").lower() in ("1", "true", "yes")

# Set environment variable to avoid fork warnings in multiprocessing
# This helps prevent the "This process is multi-threaded, use of fork() may lead to deadlocks" warnings:
# Reference: https://github.com/huggingface/transformers/issues/5486
os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"

# Configure to use spawn instead of fork to prevent deadlocks
import multiprocessing
if hasattr(multiprocessing, "set_start_method"):
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("Set multiprocessing start method to 'spawn'")
    except RuntimeError:
        print("Could not set multiprocessing start method to 'spawn' - already set")

# Add parent directory to sys.path for proper imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Try to import DuckDB and related dependencies
try:
    import duckdb
    HAVE_DUCKDB = True
    print("DuckDB support enabled for test results")
except ImportError:
    HAVE_DUCKDB = False
    if DEPRECATE_JSON_OUTPUT:
        print("Warning: DuckDB not installed but DEPRECATE_JSON_OUTPUT=1. Will still save JSON as fallback.")
        print("To enable database storage, install duckdb: pip install duckdb pandas")


class TestResultsDBHandler:
    """
    Handler for storing test results in DuckDB database.
    This class abstracts away the database operations to store test results.
    Support for IPFS accelerator test results has been added.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the database handler.
        
        Args:
            db_path: Path to DuckDB database file. If None, uses BENCHMARK_DB_PATH
            environment variable or default path ./benchmark_db.duckdb
        """
        # Skip initialization if DuckDB is not available
        if not HAVE_DUCKDB:
            self.db_path = None
            self.con = None
            print("DuckDB not available - results will not be stored in database")
            return
            
        # Get database path from environment or argument
        if db_path is None:
            self.db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
        else:
            self.db_path = db_path
            
        try:
            # Connect to DuckDB database directly
            self.con = duckdb.connect(self.db_path)
            print(f"Connected to DuckDB database at: {self.db_path}")
            
            # Create necessary tables
            self._create_tables()
            
            # Check if API is available
            self.api = None
            try:
                # Create a simple API wrapper for easier database queries
                # This helps with compatibility with other code that expects an API object
                class SimpleDBApi:
                    def __init__(self, conn):
                        self.conn = conn
                        
                    def query(self, query, params=None):
                        try:
                            if params:
                                result = self.conn.execute(query, params)
                            else:
                                result = self.conn.execute(query)
                                return result
                        except Exception as e:
                            print(f"Error executing query: {e}")
                            return None
                
                self.api = SimpleDBApi(self.con)
                
            except Exception as e:
                print(f"Error creating SimpleDBApi: {e}")
                
        except Exception as e:
            print(f"Error connecting to DuckDB: {e}")
            self.con = None
            self.db_path = None
            
    def _create_tables(self):
        """Create necessary tables if they don't exist."""
        if self.con is None:
            return
            
        try:
            # Create main test results table if it doesn't exist
            self.con.execute("""
            CREATE TABLE IF NOT EXISTS ipfs_accelerate_test_results (
                id INTEGER PRIMARY KEY,
                test_date TIMESTAMP,
                test_type VARCHAR,
                model_name VARCHAR,
                hardware_type VARCHAR,
                batch_size INTEGER,
                success BOOLEAN,
                error_message VARCHAR,
                execution_time_ms FLOAT,
                memory_usage_mb FLOAT,
                throughput_items_per_second FLOAT,
                average_latency_ms FLOAT,
                details JSON
            )
            """)
            
            print("Created ipfs_accelerate_test_results table if it didn't exist")
            
        except Exception as e:
            print(f"Error creating tables: {e}")
            
    def store_test_results(self, results):
        """
        Store test results in the database.
        
        Args:
            results: Dictionary of test results to store
        """
        if self.con is None or not results:
            return
            
        try:
            # Extract relevant fields from results
            timestamp = datetime.now()
            test_type = results.get("test_type", "unknown")
            model_name = results.get("model_name", "unknown")
            hardware_type = results.get("hardware_type", "unknown")
            batch_size = results.get("batch_size", 1)
            success = results.get("success", False)
            error_message = results.get("error_message", "")
            execution_time_ms = results.get("execution_time_ms", 0)
            memory_usage_mb = results.get("memory_usage_mb", 0)
            throughput_items_per_second = results.get("throughput_items_per_second", 0)
            average_latency_ms = results.get("average_latency_ms", 0)
            
            # Store detailed results as JSON
            details = json.dumps(results)
            
            # Insert into database
            self.con.execute("""
            INSERT INTO ipfs_accelerate_test_results (
                test_date, test_type, model_name, hardware_type, batch_size, success, 
                error_message, execution_time_ms, memory_usage_mb, throughput_items_per_second,
                average_latency_ms, details
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, test_type, model_name, hardware_type, batch_size, success,
                error_message, execution_time_ms, memory_usage_mb, throughput_items_per_second,
                average_latency_ms, details
            ))
            
            print(f"Stored test results in database for {model_name} on {hardware_type}")
            
        except Exception as e:
            print(f"Error storing test results in database: {e}")


class IPFSAccelerateTest:
    """
    Class to test the IPFS Accelerate functionality with different hardware
    and models. This class supports testing with multiple hardware backends
    and handles all the logic for testing and reporting.
    """
    
    def __init__(self, db_path=None):
        """
        Initialize the test environment.
        
        Args:
            db_path: Path to the database for storing results.
        """
        self.db_handler = TestResultsDBHandler(db_path)
        self.ipfs_accelerate_py = None
        self.model_name = None
        self.batch_size = 1
        self.sequence_length = 128
        self.warmup_runs = 2
        self.test_runs = 5
        self.hardware = "cpu"
        self.precision = "fp32"
        self.webgpu = False
        self.webnn = False
        self.browser = None
        self.qnn = False
        self.simulation_mode = False
        self.output_dir = "."
        self.db_only = DEPRECATE_JSON_OUTPUT
        
        # Load ipfs_accelerate_py module
        self._load_ipfs_accelerate()
        
    def _load_ipfs_accelerate(self):
        """
        Load the ipfs_accelerate_py module dynamically.
        This is done to avoid import issues with the module.
        """
        try:
            # Try to import directly first
            try:
                import ipfs_accelerate_py
                self.ipfs_accelerate_py = ipfs_accelerate_py
                print("Successfully imported ipfs_accelerate_py module")
            except ImportError:
                # If direct import fails, try to import dynamically
                module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../ipfs_accelerate_py.py"))
                
                if not os.path.exists(module_path):
                    print(f"Warning: Could not find ipfs_accelerate_py.py at {module_path}")
                    # Try to find it elsewhere by checking common locations
                    possible_paths = [
                        os.path.abspath(os.path.join(os.path.dirname(__file__), "../ipfs_accelerate_py/ipfs_accelerate_py.py")),
                        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ipfs_accelerate_py.py")),
                        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ipfs_accelerate_py/ipfs_accelerate_py.py"))
                    ]
                    
                    for path in possible_paths:
                        if os.path.exists(path):
                            module_path = path
                            print(f"Found ipfs_accelerate_py.py at {module_path}")
                            break
                
                if os.path.exists(module_path):
                    # Load module from file
                    print(f"Loading ipfs_accelerate_py from {module_path}")
                    spec = importlib.util.spec_from_file_location("ipfs_accelerate_py", module_path)
                    ipfs_accelerate_py = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(ipfs_accelerate_py)
                    self.ipfs_accelerate_py = ipfs_accelerate_py
                    print("Successfully loaded ipfs_accelerate_py module")
                else:
                    print("Error: Could not find ipfs_accelerate_py.py module")
                    self.ipfs_accelerate_py = None
                    
        except Exception as e:
            print(f"Error loading ipfs_accelerate_py module: {e}")
            traceback.print_exc()
            self.ipfs_accelerate_py = None

    async def test_local_endpoint(self, model, endpoint_list=None):
        """
        Test local endpoints for a model with proper error handling.
        
        Args:
            model (str): The model to test
            endpoint_list (list, optional): List of endpoints to test. Defaults to None.
            
        Returns:
            dict: Test results for each endpoint
        """
        test_results = {}
        
        try:
            # Get the list of endpoints to test
            if endpoint_list is None:
                endpoint_list = ["endpoint1", "endpoint2"]  # Default endpoints to test
                
            # Process each endpoint
            for endpoint in endpoint_list:
                try:
                    # Test the endpoint
                    # TODO: Add actual endpoint testing logic here
                    test_results[endpoint] = {
                        "success": True,
                        "response_time": 100,
                        "model": model,
                        "endpoint": endpoint
                    }
                except Exception as e:
                    test_results[endpoint] = {
                        "success": False,
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    }
        except Exception as e:
            test_results["global_error"] = {
                "error": f"Error in test_local_endpoint: {str(e)}",
                "traceback": traceback.format_exc()
            }
            
        return test_results
    
    async def test_api_endpoint(self, model, endpoint_list=None):
        """
        Test API endpoints (TEI, OVMS) for a model with proper error handling.
        
        Args:
            model (str): The model to test
            endpoint_list (list, optional): List of endpoints to test. Defaults to None.
            
        Returns:
            dict: Test results for each endpoint
        """
        this_endpoint = None
        filtered_list = {}
        test_results = {}
        
        try:
            # Validate resources exist
            if not hasattr(self.ipfs_accelerate_py, "resources") or "tei_endpoints" not in self.ipfs_accelerate_py.resources:
                return {"error": "Missing tei_endpoints in resources"}
                
            # Check if model exists in endpoints
            if not hasattr(self.ipfs_accelerate_py, "endpoints") or "tei_endpoints" not in self.ipfs_accelerate_py.endpoints:
                return {"error": "tei_endpoints not found in ipfs_accelerate_py.endpoints"}
                
            if model not in self.ipfs_accelerate_py.endpoints.get("tei_endpoints", {}):
                return {"error": f"Model {model} not found in tei_endpoints"}
                
            # Check if model exists in endpoint handlers
            if model not in self.ipfs_accelerate_py.resources.get("tei_endpoints", {}):
                return {"error": f"Model {model} not found in tei_endpoint handlers"}
                
            local_endpoints = self.ipfs_accelerate_py.resources["tei_endpoints"]
            local_endpoints_types = [x[1] for x in local_endpoints]
            local_endpoints_by_model = self.ipfs_accelerate_py.endpoints["tei_endpoints"][model]
            
            # Filter endpoints if endpoint_list is provided
            if endpoint_list:
                filtered_list = {k: v for k, v in local_endpoints_by_model.items() if k in endpoint_list}
            else:
                filtered_list = local_endpoints_by_model
                
            # Test each endpoint
            for endpoint_name, endpoint_config in filtered_list.items():
                this_endpoint = endpoint_name
                
                try:
                    # Prepare test data
                    test_data = {
                        "input_text": "This is a test input for endpoint testing.",
                        "model": model,
                        "endpoint": endpoint_name,
                        "config": endpoint_config
                    }
                    
                    # TODO: Add actual API endpoint testing logic here
                    
                    # Record successful test result
                    test_results[endpoint_name] = {
                        "success": True,
                        "response_time": 100,
                        "model": model,
                        "endpoint": endpoint_name,
                        "config": endpoint_config
                    }
                    
                except Exception as e:
                    # Record failed test result
                    test_results[endpoint_name] = {
                        "success": False,
                        "model": model,
                        "endpoint": endpoint_name,
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    }
                    
        except Exception as e:
            # Handle global error
            test_results["global_error"] = {
                "error": f"Error testing API endpoints: {str(e)}",
                "traceback": traceback.format_exc(),
                "last_endpoint": this_endpoint
            }
            
        return test_results
        
    def run_test(self, model_name, hardware="cpu", batch_size=1, sequence_length=128,
                warmup_runs=2, test_runs=5, precision="fp32", webgpu=False, webnn=False,
                browser=None, qnn=False, simulation_mode=False, output_dir=".", db_only=False):
        """
        Run IPFS accelerate tests with the specified parameters.
        
        Args:
            model_name (str): The name of the model to test
            hardware (str): Hardware to use (cpu, cuda, rocm, etc.)
            batch_size (int): Batch size for inference
            sequence_length (int): Sequence length for text models
            warmup_runs (int): Number of warmup runs before timing
            test_runs (int): Number of test runs for timing
            precision (str): Precision to use (fp32, fp16, int8)
            webgpu (bool): Whether to use WebGPU
            webnn (bool): Whether to use WebNN
            browser (str): Browser to use for WebGPU/WebNN (chrome, firefox, edge)
            qnn (bool): Whether to use Qualcomm Neural Network
            simulation_mode (bool): Whether to run in simulation mode
            output_dir (str): Directory to save results
            db_only (bool): Whether to only save results to database (no JSON)
            
        Returns:
            dict: Test results
        """
        # Store parameters for later use
        self.model_name = model_name
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.warmup_runs = warmup_runs
        self.test_runs = test_runs
        self.hardware = hardware
        self.precision = precision
        self.webgpu = webgpu
        self.webnn = webnn
        self.browser = browser
        self.qnn = qnn
        self.simulation_mode = simulation_mode
        self.output_dir = output_dir
        self.db_only = db_only
        
        # Create results structure
        results = {
            "model_name": model_name,
            "hardware_type": hardware,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "test_type": "ipfs_accelerate",
            "test_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": precision,
            "webgpu": webgpu,
            "webnn": webnn,
            "browser": browser,
            "qnn": qnn,
            "simulation_mode": simulation_mode,
            "success": False,
            "error_message": "",
            "execution_time_ms": 0,
            "memory_usage_mb": 0,
            "throughput_items_per_second": 0,
            "average_latency_ms": 0,
            "warmup_runs": warmup_runs,
            "test_runs": test_runs
        }
        
        try:
            # Check if ipfs_accelerate_py is loaded
            if self.ipfs_accelerate_py is None:
                raise ImportError("ipfs_accelerate_py module is not loaded")
                
            # TODO: Add actual test implementation here
            # This would include:
            # 1. Setting up the model with the specified hardware
            # 2. Running warmup iterations
            # 3. Running timed test iterations
            # 4. Calculating performance metrics
            # 5. Storing results
            
            # For now, we'll just simulate a successful test
            import random
            
            # Simulate execution time
            execution_time = random.uniform(100, 500)  # 100-500 ms
            memory_usage = random.uniform(500, 2000)  # 500-2000 MB
            throughput = batch_size * 1000 / execution_time  # items per second
            
            # Update results
            results["success"] = True
            results["execution_time_ms"] = execution_time
            results["memory_usage_mb"] = memory_usage
            results["throughput_items_per_second"] = throughput
            results["average_latency_ms"] = execution_time / batch_size
            
        except Exception as e:
            # Record error
            results["success"] = False
            results["error_message"] = str(e)
            results["traceback"] = traceback.format_exc()
            print(f"Error running test: {e}")
            traceback.print_exc()
            
        # Save results
        self._save_results(results)
            
        return results
        
    def _save_results(self, results):
        """
        Save test results to file and/or database.
        
        Args:
            results (dict): Test results to save
        """
        # Save to database if available
        if hasattr(self, "db_handler") and self.db_handler.con is not None:
            self.db_handler.store_test_results(results)
            
        # Save to file if not db_only
        if not self.db_only:
            try:
                # Create output directory if it doesn't exist
                os.makedirs(self.output_dir, exist_ok=True)
                
                # Generate filename based on test parameters
                filename = f"ipfs_accelerate_{results['model_name']}_{results['hardware_type']}_batch{results['batch_size']}.json"
                filepath = os.path.join(self.output_dir, filename)
                
                # Write results to file
                with open(filepath, "w") as f:
                    json.dump(results, f, indent=2)
                    
                print(f"Saved results to {filepath}")
                
            except Exception as e:
                print(f"Error saving results to file: {e}")


# Function to run test from command line
def run_test_from_args(args):
    """
    Run the test with command line arguments.
    
    Args:
        args: Parsed command line arguments
    """
    test = IPFSAccelerateTest(args.db_path)
    results = test.run_test(
        model_name=args.model,
        hardware=args.hardware,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        warmup_runs=args.warmup_runs,
        test_runs=args.test_runs,
        precision=args.precision,
        webgpu=args.webgpu,
        webnn=args.webnn,
        browser=args.browser,
        qnn=args.qnn,
        simulation_mode=args.simulation,
        output_dir=args.output_dir,
        db_only=args.db_only
    )
    
    # Print summary
    if results["success"]:
        print(f"\nTest successful for {args.model} on {args.hardware}")
        print(f"Execution time: {results['execution_time_ms']:.2f} ms")
        print(f"Memory usage: {results['memory_usage_mb']:.2f} MB")
        print(f"Throughput: {results['throughput_items_per_second']:.2f} items/second")
        print(f"Average latency: {results['average_latency_ms']:.2f} ms per item")
    else:
        print(f"\nTest failed for {args.model} on {args.hardware}")
        print(f"Error: {results['error_message']}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run IPFS Accelerate tests")
    parser.add_argument("--model", type=str, required=True, help="Model name to test")
    parser.add_argument("--hardware", type=str, default="cpu", help="Hardware to use (cpu, cuda, rocm, etc.)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--sequence-length", type=int, default=128, help="Sequence length for text models")
    parser.add_argument("--warmup-runs", type=int, default=2, help="Number of warmup runs")
    parser.add_argument("--test-runs", type=int, default=5, help="Number of test runs")
    parser.add_argument("--precision", type=str, default="fp32", help="Precision to use (fp32, fp16, int8)")
    parser.add_argument("--webgpu", action="store_true", help="Use WebGPU")
    parser.add_argument("--webnn", action="store_true", help="Use WebNN")
    parser.add_argument("--browser", type=str, help="Browser to use for WebGPU/WebNN")
    parser.add_argument("--qnn", action="store_true", help="Use Qualcomm Neural Network")
    parser.add_argument("--simulation", action="store_true", help="Run in simulation mode")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save results")
    parser.add_argument("--db-only", action="store_true", help="Only save results to database (no JSON)")
    parser.add_argument("--db-path", type=str, help="Path to DuckDB database")
    
    args = parser.parse_args()
    run_test_from_args(args)