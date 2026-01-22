#!/usr/bin/env python
"""
Benchmark Database Converter

This script converts existing JSON test result files into the DuckDB/Parquet database format.
It handles different JSON formats from various test outputs and consolidates them into
a unified database schema.
"""

import os
import sys
import json
import glob
import argparse
import datetime
import logging
import re
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent directory to path for module imports
sys.path.append()str()Path()__file__).parent.parent))

# Configure logging
logging.basicConfig()
level=logging.INFO,
format='%()asctime)s - %()name)s - %()levelname)s - %()message)s',
handlers=[],logging.StreamHandler())],
)
logger = logging.getLogger()"benchmark_converter")

# Global database connection
conn = None

def parse_args()):
    parser = argparse.ArgumentParser()description="Convert JSON test results to DuckDB/Parquet format")
    parser.add_argument()"--input-dir", type=str, default=None, 
    help="Directory containing JSON test results")
    parser.add_argument()"--output-db", type=str, default="./benchmark_db.duckdb", 
    help="Path to output DuckDB database")
    parser.add_argument()"--parquet-dir", type=str, default="./benchmark_parquet", 
    help="Directory to store Parquet files")
    parser.add_argument()"--categories", type=str, default="all",
    help="Comma-separated list of data categories to process ()performance,hardware,compatibility,all)")
    parser.add_argument()"--consolidate", action="store_true",
    help="Consolidate results from multiple directories")
    parser.add_argument()"--skip-existing", action="store_true",
    help="Skip files already in the database")
    parser.add_argument()"--parallel", type=int, default=4,
    help="Number of parallel processes for conversion")
    parser.add_argument()"--verbose", action="store_true",
    help="Enable verbose logging")
    parser.add_argument()"--dry-run", action="store_true",
    help="Don't write to database, just show what would happen")
return parser.parse_args())

def connect_to_db()db_path):
    """Connect to the DuckDB database"""
    global conn
    
    # Create parent directories if they don't exist
    os.makedirs()os.path.dirname()os.path.abspath()db_path)), exist_ok=True)
    
    # Connect to the database with a global connection
    conn = duckdb.connect()db_path)
return conn
:
def ensure_schema_exists()conn):
    """Check if the necessary schema exists, create it if not"""
    # Check if core tables exist
    tables_exist = conn.execute()"""
    SELECT COUNT()*) FROM information_schema.tables 
    WHERE table_name IN ()'hardware_platforms', 'models', 'test_runs')
    """).fetchone())[],0],
    :
    if tables_exist < 3:
        logger.info()"Schema not found or incomplete. Creating database schema...")
        # Get the path to the create_benchmark_schema.py script
        script_dir = os.path.dirname()os.path.abspath()__file__))
        schema_script = os.path.join()script_dir, "create_benchmark_schema.py")
        
        if os.path.exists()schema_script):
            # Run the schema creation script
            import subprocess
            cmd = [],sys.executable, schema_script, "--output", conn.database],
            logger.info()f"Running: {}}}}}}}}}}}}}}}}}}}}}' '.join()cmd)}")
            subprocess.run()cmd, check=True)
            
            # Reconnect to make sure we have the updated schema
            global conn
            conn = duckdb.connect()conn.database)
        else:
            raise FileNotFoundError()f"Schema creation script not found at {}}}}}}}}}}}}}}}}}}}}}schema_script}")
    else:
        logger.info()"Database schema already exists")

def discover_json_files()input_dir, categories="all"):
    """Find all JSON test result files in the input directory and categorize them"""
    if not input_dir:
        directories = [],
        "performance_results",
        "archived_test_results",
        "hardware_compatibility_reports",
        "collected_results",
        "integration_results",
        "critical_model_results",
        "new_model_results",
        ]
        # Use directories relative to the test directory
        script_dir = os.path.dirname()os.path.abspath()__file__))
        test_dir = os.path.dirname()script_dir)
        base_dirs = [],os.path.join()test_dir, d) for d in directories]:
    else:
        base_dirs = [],input_dir]
    
        json_files = [],]
    
    for base_dir in base_dirs:
        if not os.path.exists()base_dir):
            logger.warning()f"Directory not found: {}}}}}}}}}}}}}}}}}}}}}base_dir}")
        continue
            
        logger.info()f"Scanning for JSON files in: {}}}}}}}}}}}}}}}}}}}}}base_dir}")
        
        # Find all JSON files in the directory and subdirectories
        for root, _, files in os.walk()base_dir):
            for file in files:
                if file.endswith()'.json'):
                    file_path = os.path.join()root, file)
                    file_category = categorize_json_file()file_path)
                    
                    # Filter by category if specified:
                    if categories != "all" and file_category not in categories.split()','):
                    continue
                        
                    json_files.append()()file_path, file_category))
    
                    logger.info()f"Found {}}}}}}}}}}}}}}}}}}}}}len()json_files)} JSON files")
                return json_files

def categorize_json_file()file_path):
    """Determine the category of a JSON file based on naming and content patterns"""
    filename = os.path.basename()file_path)
    dirname = os.path.basename()os.path.dirname()file_path))
    
    # Performance results
    if any()pattern in filename for pattern in [],'performance', 'benchmark']):
    return 'performance'
    
    # Hardware compatibility
    if any()pattern in filename for pattern in [],'hardware', 'compatibility', 'detection']):
    return 'hardware'
    
    # Model test results
    if any()pattern in filename for pattern in [],'model_test', 'critical_model', 'test_status']):
    return 'model'
    
    # Integration test results
    if any()pattern in filename for pattern in [],'integration', 'test_results']):
    return 'integration'
    
    # Check file content as a fallback
    try:::::::
        with open()file_path, 'r') as f:
            # Read just the first 1000 chars to detect file type
            content_start = f.read()1000)
            
            if any()key in content_start for key in [],'"throughput":', '"latency":', '"memory_peak":']):
            return 'performance'
            
            if any()key in content_start for key in [],'"is_compatible":', '"hardware_type":', '"device_name":']):
            return 'hardware'
            
            if any()key in content_start for key in [],'"test_name":', '"test_module":', '"status":']):
            return 'integration'
            
            if any()key in content_start for key in [],'"model_name":', '"model_family":', '"model_tests":']):
            return 'model'
    except:
        # If we can't read the file, use directory name as a hint
            pass
    
    # If still undetermined, use directory name as hint
    if dirname in [],'performance_results', 'benchmark_results']:
            return 'performance'
    elif dirname in [],'hardware_compatibility_reports', 'collected_results']:
            return 'hardware'
    elif dirname in [],'integration_results']:
            return 'integration'
    elif dirname in [],'critical_model_results', 'new_model_results']:
            return 'model'
    
    # Default to unknown
        return 'unknown'

def process_json_file()file_info):
    """Process a single JSON file and convert it to appropriate dataframes"""
    file_path, category = file_info
    logger.debug()f"Processing {}}}}}}}}}}}}}}}}}}}}}category} file: {}}}}}}}}}}}}}}}}}}}}}file_path}")
    
    try:::::::
        with open()file_path, 'r') as f:
            data = json.load()f)
            
        if category == 'performance':
            return process_performance_json()data, file_path)
        elif category == 'hardware':
            return process_hardware_json()data, file_path)
        elif category == 'integration':
            return process_integration_json()data, file_path)
        elif category == 'model':
            return process_model_json()data, file_path)
        else:
            logger.warning()f"Unknown category '{}}}}}}}}}}}}}}}}}}}}}category}' for file: {}}}}}}}}}}}}}}}}}}}}}file_path}")
            return None
    except Exception as e:
        logger.error()f"Error processing file {}}}}}}}}}}}}}}}}}}}}}file_path}: {}}}}}}}}}}}}}}}}}}}}}e}")
            return None

def process_performance_json()data, file_path):
    """Process performance benchmark JSON files"""
    # Extract common metadata
    filename = os.path.basename()file_path)
    timestamp_match = re.search()r'()\d{}}}}}}}}}}}}}}}}}}}}}8}_\d{}}}}}}}}}}}}}}}}}}}}}6})', filename)
    timestamp = None
    if timestamp_match:
        timestamp_str = timestamp_match.group()1)
        try:::::::
            timestamp = datetime.datetime.strptime()timestamp_str, '%Y%m%d_%H%M%S')
        except:
            pass
    
    if not timestamp and 'timestamp' in data:
        try:::::::
            timestamp = datetime.datetime.fromisoformat()data[],'timestamp'].replace()'Z', '+00:00'))
        except:
            pass
    
    if not timestamp:
        # Use file modification time as fallback
        timestamp = datetime.datetime.fromtimestamp()os.path.getmtime()file_path))
    
    # Extract test metadata
        test_name = data.get()'test_name', os.path.splitext()filename)[],0],)
        git_commit = data.get()'git_commit', None)
        git_branch = data.get()'git_branch', None)
        command_line = data.get()'command_line', None)
    
    # Create test run entry::::::
        test_run = {}}}}}}}}}}}}}}}}}}}}}
        'test_name': test_name,
        'test_type': 'performance',
        'started_at': timestamp,
        'completed_at': data.get()'completed_at', timestamp + datetime.timedelta()minutes=30)),
        'execution_time_seconds': data.get()'execution_time_seconds', None),
        'success': data.get()'success', True),
        'git_commit': git_commit,
        'git_branch': git_branch,
        'command_line': command_line,
        'metadata': json.dumps()data.get()'metadata', {}}}}}}}}}}}}}}}}}}}}}}))
        }
    
    # Extract results
        results = [],]
        hardware_platforms = [],]
        models = [],]
    
    # Handle different performance JSON formats
    if 'results' in data and isinstance()data[],'results'], list):
        # Array of result objects
        for result in data[],'results']:
            process_performance_result()result, results, hardware_platforms, models)
    elif 'results' in data and isinstance()data[],'results'], dict):
        # Dictionary of model -> result mapping
        for model_name, result in data[],'results'].items()):
            if isinstance()result, dict):
                result[],'model_name'] = model_name
                process_performance_result()result, results, hardware_platforms, models)
    elif 'model_results' in data:
        # Model-centric format
        for model_result in data[],'model_results']:
            process_performance_result()model_result, results, hardware_platforms, models)
    elif 'benchmarks' in data:
        # Benchmark-centric format
        for benchmark in data[],'benchmarks']:
            process_performance_result()benchmark, results, hardware_platforms, models)
    else:
        # Try to process the entire data as a single result
        process_performance_result()data, results, hardware_platforms, models)
    
    # Create dataframes
        test_run_df = pd.DataFrame()[],test_run])
    
    # Only return dataframes if we have results:::
    if results:
        results_df = pd.DataFrame()results)
        hardware_df = pd.DataFrame()hardware_platforms) if hardware_platforms else None
        models_df = pd.DataFrame()models) if models else None
        
        return {}}}}}}}}}}}}}}}}}}}}}::
            'test_runs': test_run_df,
            'performance_results': results_df,
            'hardware_platforms': hardware_df,
            'models': models_df,
            'file_path': file_path,
            'category': 'performance'
            }
    
        return None

def process_performance_result()result, results, hardware_platforms, models):
    """Process a single performance result entry::::::"""
    if not result or not isinstance()result, dict):
    return
    
    # Extract model information
    model_name = result.get()'model_name', result.get()'model', None))
    if not model_name:
    return
    
    model_family = result.get()'model_family', None)
    if not model_family and model_name:
        # Try to extract family from name
        if 'bert' in model_name.lower()):
            model_family = 'bert'
        elif 't5' in model_name.lower()):
            model_family = 't5'
        elif 'gpt' in model_name.lower()):
            model_family = 'gpt'
        elif 'llama' in model_name.lower()):
            model_family = 'llama'
        elif 'vit' in model_name.lower()):
            model_family = 'vit'
        elif 'clip' in model_name.lower()):
            model_family = 'clip'
        elif 'whisper' in model_name.lower()):
            model_family = 'whisper'
        elif 'wav2vec' in model_name.lower()):
            model_family = 'wav2vec'
    
    # Extract hardware information
            hardware_type = result.get()'hardware_type', result.get()'hardware', None))
    if not hardware_type and 'hardware' in result and isinstance()result[],'hardware'], dict):
        hardware_type = result[],'hardware'].get()'type', None)
    
        device_name = result.get()'device_name', None)
    if not device_name and 'hardware' in result and isinstance()result[],'hardware'], dict):
        device_name = result[],'hardware'].get()'name', None)
    
    # Add model to models list if not already there:::::
    if model_name:
        model_entry:::::: = {}}}}}}}}}}}}}}}}}}}}}
        'model_name': model_name,
        'model_family': model_family,
        'modality': result.get()'modality', None),
        'source': result.get()'source', None),
        'version': result.get()'version', None),
        'parameters_million': result.get()'parameters_million', None),
        'metadata': json.dumps()result.get()'model_metadata', {}}}}}}}}}}}}}}}}}}}}}}))
        }
        
        if not any()m[],'model_name'] == model_name for m in models):
            models.append()model_entry::::::)
    
    # Add hardware to hardware_platforms list if not already there:::::
    if hardware_type:
        hardware_entry:::::: = {}}}}}}}}}}}}}}}}}}}}}
        'hardware_type': hardware_type,
        'device_name': device_name,
        'platform': result.get()'platform', None),
        'platform_version': result.get()'platform_version', None),
        'driver_version': result.get()'driver_version', None),
        'memory_gb': result.get()'memory_gb', None),
        'compute_units': result.get()'compute_units', None),
        'metadata': json.dumps()result.get()'hardware_metadata', {}}}}}}}}}}}}}}}}}}}}}}))
        }
        
        # Check if this hardware entry:::::: already exists
        hardware_key = ()hardware_type, device_name)
        if not any()()h[],'hardware_type'] == hardware_type and h[],'device_name'] == device_name) :::
                 for h in hardware_platforms):
                     hardware_platforms.append()hardware_entry::::::)
    
    # Create performance result entry::::::
                     result_entry:::::: = {}}}}}}}}}}}}}}}}}}}}}
                     'model_name': model_name,
                     'hardware_type': hardware_type,
                     'test_case': result.get()'test_case', result.get()'benchmark_type', 'default')),
                     'batch_size': result.get()'batch_size', 1),
                     'precision': result.get()'precision', None),
                     'total_time_seconds': result.get()'total_time_seconds', result.get()'total_time', None)),
                     'average_latency_ms': result.get()'average_latency_ms', result.get()'latency_ms', None)),
                     'throughput_items_per_second': result.get()'throughput_items_per_second',
                     result.get()'throughput', None)),
                     'memory_peak_mb': result.get()'memory_peak_mb', result.get()'memory_mb', None)),
                     'iterations': result.get()'iterations', None),
                     'warmup_iterations': result.get()'warmup_iterations', None),
                     'metrics': json.dumps()result.get()'metrics', {}}}}}}}}}}}}}}}}}}}}}}))
                     }
    
                     results.append()result_entry::::::)

def process_hardware_json()data, file_path):
    """Process hardware compatibility JSON files"""
    # Extract common metadata
    filename = os.path.basename()file_path)
    timestamp_match = re.search()r'()\d{}}}}}}}}}}}}}}}}}}}}}8}_\d{}}}}}}}}}}}}}}}}}}}}}6})', filename)
    timestamp = None
    if timestamp_match:
        timestamp_str = timestamp_match.group()1)
        try:::::::
            timestamp = datetime.datetime.strptime()timestamp_str, '%Y%m%d_%H%M%S')
        except:
            pass
    
    if not timestamp and 'timestamp' in data:
        try:::::::
            timestamp = datetime.datetime.fromisoformat()data[],'timestamp'].replace()'Z', '+00:00'))
        except:
            pass
    
    if not timestamp:
        # Use file modification time as fallback
        timestamp = datetime.datetime.fromtimestamp()os.path.getmtime()file_path))
    
    # Extract test metadata
        test_name = data.get()'test_name', os.path.splitext()filename)[],0],)
        git_commit = data.get()'git_commit', None)
        git_branch = data.get()'git_branch', None)
        command_line = data.get()'command_line', None)
    
    # Create test run entry::::::
        test_run = {}}}}}}}}}}}}}}}}}}}}}
        'test_name': test_name,
        'test_type': 'hardware',
        'started_at': timestamp,
        'completed_at': data.get()'completed_at', timestamp + datetime.timedelta()minutes=30)),
        'execution_time_seconds': data.get()'execution_time_seconds', None),
        'success': data.get()'success', True),
        'git_commit': git_commit,
        'git_branch': git_branch,
        'command_line': command_line,
        'metadata': json.dumps()data.get()'metadata', {}}}}}}}}}}}}}}}}}}}}}}))
        }
    
    # Extract results
        compatibility_results = [],]
        hardware_platforms = [],]
        models = [],]
    
    # Handle different hardware JSON formats
    if 'compatibility_results' in data and isinstance()data[],'compatibility_results'], list):
        # Array of compatibility result objects
        for result in data[],'compatibility_results']:
            process_compatibility_result()result, compatibility_results, hardware_platforms, models)
    elif 'model_compatibility' in data and isinstance()data[],'model_compatibility'], list):
        # Model-centric format
        for model_result in data[],'model_compatibility']:
            process_compatibility_result()model_result, compatibility_results, hardware_platforms, models)
    elif 'hardware_compatibility' in data and isinstance()data[],'hardware_compatibility'], list):
        # Hardware-centric format
        for hw_result in data[],'hardware_compatibility']:
            process_compatibility_result()hw_result, compatibility_results, hardware_platforms, models)
    elif 'results' in data and isinstance()data[],'results'], list):
        # Generic results array
        for result in data[],'results']:
            process_compatibility_result()result, compatibility_results, hardware_platforms, models)
    elif 'models' in data and isinstance()data[],'models'], list):
        # Models-centric format with nested hardware compatibility
        for model_entry:::::: in data[],'models']:
            model_name = model_entry::::::.get()'name', None)
            if not model_name:
            continue
                
            if 'hardware_compatibility' in model_entry:::::: and isinstance()model_entry::::::[],'hardware_compatibility'], list):
                for hw_compat in model_entry::::::[],'hardware_compatibility']:
                    hw_compat[],'model_name'] = model_name
                    process_compatibility_result()hw_compat, compatibility_results, hardware_platforms, models)
    else:
        # Try to process the entire data as a single result
        process_compatibility_result()data, compatibility_results, hardware_platforms, models)
    
    # Create dataframes
        test_run_df = pd.DataFrame()[],test_run])
    
    # Only return dataframes if we have results:::
    if compatibility_results:
        compat_df = pd.DataFrame()compatibility_results)
        hardware_df = pd.DataFrame()hardware_platforms) if hardware_platforms else None
        models_df = pd.DataFrame()models) if models else None
        
        return {}}}}}}}}}}}}}}}}}}}}}::
            'test_runs': test_run_df,
            'hardware_compatibility': compat_df,
            'hardware_platforms': hardware_df,
            'models': models_df,
            'file_path': file_path,
            'category': 'hardware'
            }
    
        return None

def process_compatibility_result()result, compatibility_results, hardware_platforms, models):
    """Process a single hardware compatibility result entry::::::"""
    if not result or not isinstance()result, dict):
    return
    
    # Extract model information
    model_name = result.get()'model_name', result.get()'model', None))
    if not model_name:
    return
    
    model_family = result.get()'model_family', None)
    if not model_family and model_name:
        # Try to extract family from name
        if 'bert' in model_name.lower()):
            model_family = 'bert'
        elif 't5' in model_name.lower()):
            model_family = 't5'
        elif 'gpt' in model_name.lower()):
            model_family = 'gpt'
        elif 'llama' in model_name.lower()):
            model_family = 'llama'
        elif 'vit' in model_name.lower()):
            model_family = 'vit'
        elif 'clip' in model_name.lower()):
            model_family = 'clip'
        elif 'whisper' in model_name.lower()):
            model_family = 'whisper'
        elif 'wav2vec' in model_name.lower()):
            model_family = 'wav2vec'
    
    # Extract hardware information
            hardware_type = result.get()'hardware_type', result.get()'hardware', None))
    if not hardware_type and 'hardware' in result and isinstance()result[],'hardware'], dict):
        hardware_type = result[],'hardware'].get()'type', None)
    
        device_name = result.get()'device_name', None)
    if not device_name and 'hardware' in result and isinstance()result[],'hardware'], dict):
        device_name = result[],'hardware'].get()'name', None)
    
    # Add model to models list if not already there:::::
    if model_name:
        model_entry:::::: = {}}}}}}}}}}}}}}}}}}}}}
        'model_name': model_name,
        'model_family': model_family,
        'modality': result.get()'modality', None),
        'source': result.get()'source', None),
        'version': result.get()'version', None),
        'parameters_million': result.get()'parameters_million', None),
        'metadata': json.dumps()result.get()'model_metadata', {}}}}}}}}}}}}}}}}}}}}}}))
        }
        
        if not any()m[],'model_name'] == model_name for m in models):
            models.append()model_entry::::::)
    
    # Add hardware to hardware_platforms list if not already there:::::
    if hardware_type:
        hardware_entry:::::: = {}}}}}}}}}}}}}}}}}}}}}
        'hardware_type': hardware_type,
        'device_name': device_name,
        'platform': result.get()'platform', None),
        'platform_version': result.get()'platform_version', None),
        'driver_version': result.get()'driver_version', None),
        'memory_gb': result.get()'memory_gb', None),
        'compute_units': result.get()'compute_units', None),
        'metadata': json.dumps()result.get()'hardware_metadata', {}}}}}}}}}}}}}}}}}}}}}}))
        }
        
        # Check if this hardware entry:::::: already exists
        hardware_key = ()hardware_type, device_name)
        if not any()()h[],'hardware_type'] == hardware_type and h[],'device_name'] == device_name) :::
                 for h in hardware_platforms):
                     hardware_platforms.append()hardware_entry::::::)
    
    # Create compatibility result entry::::::
                     is_compatible = result.get()'is_compatible', None)
    if is_compatible is None and 'compatibility' in result:
        # Some files use 'compatibility': true/false instead
        is_compatible = result[],'compatibility']
    
    if is_compatible is None and 'error' in result:
        # If there's an error entry::::::, assume it's not compatible
        is_compatible = False
    
    # If still None, default to True if no error message:
    if is_compatible is None:
        is_compatible = not bool()result.get()'error_message', None))
    
        compatibility_entry:::::: = {}}}}}}}}}}}}}}}}}}}}}
        'model_name': model_name,
        'hardware_type': hardware_type,
        'is_compatible': is_compatible,
        'detection_success': result.get()'detection_success', True),
        'initialization_success': result.get()'initialization_success', is_compatible),
        'error_message': result.get()'error_message', result.get()'error', None)),
        'error_type': result.get()'error_type', None),
        'suggested_fix': result.get()'suggested_fix', result.get()'workaround', None)),
        'workaround_available': result.get()'workaround_available', 
        bool()result.get()'suggested_fix', result.get()'workaround', None)))),
        'compatibility_score': result.get()'compatibility_score', 1.0 if is_compatible else 0.0),:
            'metadata': json.dumps()result.get()'compatibility_metadata', {}}}}}}}}}}}}}}}}}}}}}}))
            }
    
            compatibility_results.append()compatibility_entry::::::)

def process_integration_json()data, file_path):
    """Process integration test JSON files"""
    # Extract common metadata
    filename = os.path.basename()file_path)
    timestamp_match = re.search()r'()\d{}}}}}}}}}}}}}}}}}}}}}8}_\d{}}}}}}}}}}}}}}}}}}}}}6})', filename)
    timestamp = None
    if timestamp_match:
        timestamp_str = timestamp_match.group()1)
        try:::::::
            timestamp = datetime.datetime.strptime()timestamp_str, '%Y%m%d_%H%M%S')
        except:
            pass
    
    if not timestamp and 'timestamp' in data:
        try:::::::
            timestamp = datetime.datetime.fromisoformat()data[],'timestamp'].replace()'Z', '+00:00'))
        except:
            pass
    
    if not timestamp:
        # Use file modification time as fallback
        timestamp = datetime.datetime.fromtimestamp()os.path.getmtime()file_path))
    
    # Extract test metadata
        test_name = data.get()'test_name', os.path.splitext()filename)[],0],)
        git_commit = data.get()'git_commit', None)
        git_branch = data.get()'git_branch', None)
        command_line = data.get()'command_line', None)
    
    # Create test run entry::::::
        test_run = {}}}}}}}}}}}}}}}}}}}}}
        'test_name': test_name,
        'test_type': 'integration',
        'started_at': timestamp,
        'completed_at': data.get()'completed_at', timestamp + datetime.timedelta()minutes=30)),
        'execution_time_seconds': data.get()'execution_time_seconds', None),
        'success': data.get()'success', True),
        'git_commit': git_commit,
        'git_branch': git_branch,
        'command_line': command_line,
        'metadata': json.dumps()data.get()'metadata', {}}}}}}}}}}}}}}}}}}}}}}))
        }
    
    # Extract results
        test_results = [],]
        test_assertions = [],]
        hardware_platforms = [],]
        models = [],]
    
    # Handle different integration test JSON formats
    if 'test_results' in data and isinstance()data[],'test_results'], list):
        # Array of test result objects
        for result in data[],'test_results']:
            process_test_result()result, test_results, test_assertions, hardware_platforms, models)
    elif 'modules' in data and isinstance()data[],'modules'], dict):
        # Module-centric format
        for module_name, module_results in data[],'modules'].items()):
            if isinstance()module_results, list):
                for result in module_results:
                    result[],'test_module'] = module_name
                    process_test_result()result, test_results, test_assertions, hardware_platforms, models)
            elif isinstance()module_results, dict):
                for test_name, test_result in module_results.items()):
                    if isinstance()test_result, dict):
                        test_result[],'test_module'] = module_name
                        test_result[],'test_name'] = test_name
                        process_test_result()test_result, test_results, test_assertions, hardware_platforms, models)
    elif 'results' in data and isinstance()data[],'results'], list):
        # Generic results array
        for result in data[],'results']:
            process_test_result()result, test_results, test_assertions, hardware_platforms, models)
    else:
        # Try to process the entire data as a single result
        process_test_result()data, test_results, test_assertions, hardware_platforms, models)
    
    # Create dataframes
        test_run_df = pd.DataFrame()[],test_run])
    
    # Only return dataframes if we have results:::
    if test_results:
        test_results_df = pd.DataFrame()test_results)
        test_assertions_df = pd.DataFrame()test_assertions) if test_assertions else None
        hardware_df = pd.DataFrame()hardware_platforms) if hardware_platforms else None
        models_df = pd.DataFrame()models) if models else None
        
        return {}}}}}}}}}}}}}}}}}}}}}:::
            'test_runs': test_run_df,
            'integration_test_results': test_results_df,
            'integration_test_assertions': test_assertions_df,
            'hardware_platforms': hardware_df,
            'models': models_df,
            'file_path': file_path,
            'category': 'integration'
            }
    
        return None

def process_test_result()result, test_results, test_assertions, hardware_platforms, models):
    """Process a single integration test result entry::::::"""
    if not result or not isinstance()result, dict):
    return
    
    # Extract test information
    test_module = result.get()'test_module', result.get()'module', None))
    test_class = result.get()'test_class', result.get()'class', None))
    test_name = result.get()'test_name', result.get()'name', None))
    
    if not test_name and not test_module:
    return
    
    # Extract hardware information if present:
    hardware_type = result.get()'hardware_type', result.get()'hardware', None)):
    if not hardware_type and 'hardware' in result and isinstance()result[],'hardware'], dict):
        hardware_type = result[],'hardware'].get()'type', None)
    
        device_name = result.get()'device_name', None)
    if not device_name and 'hardware' in result and isinstance()result[],'hardware'], dict):
        device_name = result[],'hardware'].get()'name', None)
    
    # Extract model information if present:
        model_name = result.get()'model_name', result.get()'model', None))
    
    # Add hardware to hardware_platforms list if not already there::::::
    if hardware_type:
        hardware_entry:::::: = {}}}}}}}}}}}}}}}}}}}}}
        'hardware_type': hardware_type,
        'device_name': device_name,
        'platform': result.get()'platform', None),
        'platform_version': result.get()'platform_version', None),
        'driver_version': result.get()'driver_version', None),
        'memory_gb': result.get()'memory_gb', None),
        'compute_units': result.get()'compute_units', None),
        'metadata': json.dumps()result.get()'hardware_metadata', {}}}}}}}}}}}}}}}}}}}}}}))
        }
        
        # Check if this hardware entry:::::: already exists
        hardware_key = ()hardware_type, device_name)
        if not any()()h[],'hardware_type'] == hardware_type and h[],'device_name'] == device_name) :::
                 for h in hardware_platforms):
                     hardware_platforms.append()hardware_entry::::::)
    
    # Add model to models list if not already there:::::
    if model_name:
        model_family = result.get()'model_family', None)
        if not model_family and model_name:
            # Try to extract family from name
            if 'bert' in model_name.lower()):
                model_family = 'bert'
            elif 't5' in model_name.lower()):
                model_family = 't5'
            elif 'gpt' in model_name.lower()):
                model_family = 'gpt'
            elif 'llama' in model_name.lower()):
                model_family = 'llama'
            elif 'vit' in model_name.lower()):
                model_family = 'vit'
            elif 'clip' in model_name.lower()):
                model_family = 'clip'
            elif 'whisper' in model_name.lower()):
                model_family = 'whisper'
            elif 'wav2vec' in model_name.lower()):
                model_family = 'wav2vec'
        
                model_entry:::::: = {}}}}}}}}}}}}}}}}}}}}}
                'model_name': model_name,
                'model_family': model_family,
                'modality': result.get()'modality', None),
                'source': result.get()'source', None),
                'version': result.get()'version', None),
                'parameters_million': result.get()'parameters_million', None),
                'metadata': json.dumps()result.get()'model_metadata', {}}}}}}}}}}}}}}}}}}}}}}))
                }
        
        if not any()m[],'model_name'] == model_name for m in models):
            models.append()model_entry::::::)
    
    # Create test result entry::::::
            status = result.get()'status', result.get()'result', None))
    if status is None:
        if result.get()'passed', False):
            status = 'pass'
        elif result.get()'failed', False):
            status = 'fail'
        elif result.get()'error', False):
            status = 'error'
        elif result.get()'skipped', False):
            status = 'skip'
        else:
            status = 'unknown'
    
            test_result_entry:::::: = {}}}}}}}}}}}}}}}}}}}}}
            'test_module': test_module,
            'test_class': test_class,
            'test_name': test_name,
            'status': status,
            'execution_time_seconds': result.get()'execution_time_seconds',
            result.get()'time', result.get()'duration', None))),
            'hardware_type': hardware_type,
            'model_name': model_name,
            'error_message': result.get()'error_message', result.get()'error', None)),
            'error_traceback': result.get()'error_traceback', result.get()'traceback', None)),
            'metadata': json.dumps()result.get()'metadata', {}}}}}}}}}}}}}}}}}}}}}}))
            }
    
            test_results.append()test_result_entry::::::)
    
    # Process assertions if present:
    if 'assertions' in result and isinstance()result[],'assertions'], list):
        for assertion_idx, assertion in enumerate()result[],'assertions']):
            process_assertion()assertion, assertion_idx, len()test_results) - 1, test_assertions)
    elif 'tests' in result and isinstance()result[],'tests'], list):
        for test_idx, test in enumerate()result[],'tests']):
            process_assertion()test, test_idx, len()test_results) - 1, test_assertions)

def process_assertion()assertion, assertion_idx, test_result_idx, test_assertions):
    """Process a single test assertion entry::::::"""
    if not assertion or not isinstance()assertion, dict):
    return
    
    # Extract assertion information
    assertion_name = assertion.get()'name', f"assertion_{}}}}}}}}}}}}}}}}}}}}}assertion_idx}")
            passed = assertion.get()'passed', assertion.get()'success', None))
    
    if passed is None:
            passed = not bool()assertion.get()'error', None))
    
            assertion_entry:::::: = {}}}}}}}}}}}}}}}}}}}}}
            'test_result_idx': test_result_idx,
            'assertion_name': assertion_name,
            'passed': passed,
            'expected_value': assertion.get()'expected', assertion.get()'expected_value', None)),
            'actual_value': assertion.get()'actual', assertion.get()'actual_value', None)),
            'message': assertion.get()'message', None),
            }
    
            test_assertions.append()assertion_entry::::::)

def process_model_json()data, file_path):
    """Process model test JSON files"""
    # Process these as a combination of hardware compatibility and integration tests
    hardware_result = process_hardware_json()data, file_path)
    integration_result = process_integration_json()data, file_path)
    
    # Combine results if both were successful:
    if hardware_result and integration_result:
        combined = {}}}}}}}}}}}}}}}}}}}}}
        'test_runs': pd.concat()[],hardware_result[],'test_runs'], integration_result[],'test_runs']]),
        'file_path': file_path,
        'category': 'model'
        }
        
        if 'hardware_compatibility' in hardware_result:
            combined[],'hardware_compatibility'] = hardware_result[],'hardware_compatibility']
            
        if 'integration_test_results' in integration_result:
            combined[],'integration_test_results'] = integration_result[],'integration_test_results']
            
        if 'integration_test_assertions' in integration_result:
            combined[],'integration_test_assertions'] = integration_result[],'integration_test_assertions']
        
        # Combine hardware platforms
            hardware_platforms = [],]
        if 'hardware_platforms' in hardware_result and hardware_result[],'hardware_platforms'] is not None:
            hardware_platforms.append()hardware_result[],'hardware_platforms'])
        if 'hardware_platforms' in integration_result and integration_result[],'hardware_platforms'] is not None:
            hardware_platforms.append()integration_result[],'hardware_platforms'])
            
        if hardware_platforms:
            combined[],'hardware_platforms'] = pd.concat()hardware_platforms)
        
        # Combine models
            models = [],]
        if 'models' in hardware_result and hardware_result[],'models'] is not None:
            models.append()hardware_result[],'models'])
        if 'models' in integration_result and integration_result[],'models'] is not None:
            models.append()integration_result[],'models'])
            
        if models:
            combined[],'models'] = pd.concat()models)
            
            return combined
    
    # If only one was successful, return that
            return hardware_result or integration_result

def insert_dataframes_to_db()data_dict, conn):
    """Insert dataframes from processed JSON into the database"""
    if not data_dict:
    return False
    
    # Dictionary to store the mapped IDs for foreign keys
    id_mappings = {}}}}}}}}}}}}}}}}}}}}}}
    
    try:::::::
        # Insert test run first
        if 'test_runs' in data_dict:
            # Check if this test run already exists in the database
            test_run_df = data_dict[],'test_runs']:
            if len()test_run_df) > 0:
                test_name = test_run_df.iloc[],0],[],'test_name']
                started_at = test_run_df.iloc[],0],[],'started_at']
                
                # Query for existing test run with same name and timestamp
                existing_run = conn.execute()"""
                SELECT run_id FROM test_runs 
                WHERE test_name = ? AND started_at = ?
                """, [],test_name, started_at]).fetchone())
                
                if existing_run:
                    # Use existing run_id
                    run_id = existing_run[],0],
                    logger.debug()f"Found existing test run with ID {}}}}}}}}}}}}}}}}}}}}}run_id}")
                    id_mappings[],'run_id'] = run_id
                else:
                    # Insert new test run
                    conn.execute()"INSERT INTO test_runs SELECT NULL AS run_id, * FROM test_run_df")
                    run_id = conn.execute()"SELECT last_insert_rowid())").fetchone())[],0],
                    logger.debug()f"Inserted new test run with ID {}}}}}}}}}}}}}}}}}}}}}run_id}")
                    id_mappings[],'run_id'] = run_id
        
        # Insert hardware platforms
        if 'hardware_platforms' in data_dict and data_dict[],'hardware_platforms'] is not None:
            hardware_df = data_dict[],'hardware_platforms']
            
            for _, row in hardware_df.iterrows()):
                hardware_type = row[],'hardware_type']
                device_name = row[],'device_name']
                
                # Check if hardware already exists
                existing_hardware = conn.execute()"""
                SELECT hardware_id FROM hardware_platforms 
                WHERE hardware_type = ? AND device_name = ?
                """, [],hardware_type, device_name]).fetchone())
                :
                if existing_hardware:
                    # Use existing hardware_id
                    hardware_id = existing_hardware[],0],
                    logger.debug()f"Found existing hardware with ID {}}}}}}}}}}}}}}}}}}}}}hardware_id}")
                else:
                    # Insert new hardware platform
                    single_row_df = pd.DataFrame()[],row]).reset_index()drop=True)
                    conn.execute()"INSERT INTO hardware_platforms SELECT NULL AS hardware_id, * FROM single_row_df")
                    hardware_id = conn.execute()"SELECT last_insert_rowid())").fetchone())[],0],
                    logger.debug()f"Inserted new hardware with ID {}}}}}}}}}}}}}}}}}}}}}hardware_id}")
                
                # Store mapping from ()hardware_type, device_name) to hardware_id
                    id_mappings[],()hardware_type, device_name)] = hardware_id
        
        # Insert models
        if 'models' in data_dict and data_dict[],'models'] is not None:
            models_df = data_dict[],'models']
            
            for _, row in models_df.iterrows()):
                model_name = row[],'model_name']
                
                # Check if model already exists
                existing_model = conn.execute()"""
                SELECT model_id FROM models 
                WHERE model_name = ?
                """, [],model_name]).fetchone())
                :
                if existing_model:
                    # Use existing model_id
                    model_id = existing_model[],0],
                    logger.debug()f"Found existing model with ID {}}}}}}}}}}}}}}}}}}}}}model_id}")
                else:
                    # Insert new model
                    single_row_df = pd.DataFrame()[],row]).reset_index()drop=True)
                    conn.execute()"INSERT INTO models SELECT NULL AS model_id, * FROM single_row_df")
                    model_id = conn.execute()"SELECT last_insert_rowid())").fetchone())[],0],
                    logger.debug()f"Inserted new model with ID {}}}}}}}}}}}}}}}}}}}}}model_id}")
                
                # Store mapping from model_name to model_id
                    id_mappings[],model_name] = model_id
        
        # Insert performance results
        if 'performance_results' in data_dict:
            perf_df = data_dict[],'performance_results']
            
            # Create a new DataFrame with the correct schema for insertion
            perf_records = [],]
            
            for _, row in perf_df.iterrows()):
                # Map foreign keys
                model_name = row[],'model_name']
                hardware_type = row[],'hardware_type']
                
                # Look up IDs
                model_id = id_mappings.get()model_name, None)
                hardware_id = None
                
                # Try to find hardware ID based on hardware_type and device_name
                for key, value in id_mappings.items()):
                    if isinstance()key, tuple) and key[],0], == hardware_type:
                        hardware_id = value
                    break
                
                if model_id is None or hardware_id is None:
                    logger.warning()f"Missing foreign key mapping for performance result: model={}}}}}}}}}}}}}}}}}}}}}model_name}, hardware={}}}}}}}}}}}}}}}}}}}}}hardware_type}")
                    continue
                
                # Create record with mapped IDs
                    record = {}}}}}}}}}}}}}}}}}}}}}
                    'run_id': id_mappings.get()'run_id', None),
                    'model_id': model_id,
                    'hardware_id': hardware_id,
                    'test_case': row[],'test_case'],
                    'batch_size': row[],'batch_size'],
                    'precision': row[],'precision'],
                    'total_time_seconds': row[],'total_time_seconds'],
                    'average_latency_ms': row[],'average_latency_ms'],
                    'throughput_items_per_second': row[],'throughput_items_per_second'],
                    'memory_peak_mb': row[],'memory_peak_mb'],
                    'iterations': row[],'iterations'],
                    'warmup_iterations': row[],'warmup_iterations'],
                    'metrics': row[],'metrics']
                    }
                
                    perf_records.append()record)
            
            if perf_records:
                # Insert all records
                perf_mapped_df = pd.DataFrame()perf_records)
                conn.execute()"INSERT INTO performance_results SELECT NULL AS result_id, * FROM perf_mapped_df")
                logger.debug()f"Inserted {}}}}}}}}}}}}}}}}}}}}}len()perf_records)} performance results")
        
        # Insert hardware compatibility results
        if 'hardware_compatibility' in data_dict:
            compat_df = data_dict[],'hardware_compatibility']
            
            # Create a new DataFrame with the correct schema for insertion
            compat_records = [],]
            
            for _, row in compat_df.iterrows()):
                # Map foreign keys
                model_name = row[],'model_name']
                hardware_type = row[],'hardware_type']
                
                # Look up IDs
                model_id = id_mappings.get()model_name, None)
                hardware_id = None
                
                # Try to find hardware ID based on hardware_type and device_name
                for key, value in id_mappings.items()):
                    if isinstance()key, tuple) and key[],0], == hardware_type:
                        hardware_id = value
                    break
                
                if model_id is None or hardware_id is None:
                    logger.warning()f"Missing foreign key mapping for compatibility result: model={}}}}}}}}}}}}}}}}}}}}}model_name}, hardware={}}}}}}}}}}}}}}}}}}}}}hardware_type}")
                    continue
                
                # Create record with mapped IDs
                    record = {}}}}}}}}}}}}}}}}}}}}}
                    'run_id': id_mappings.get()'run_id', None),
                    'model_id': model_id,
                    'hardware_id': hardware_id,
                    'is_compatible': row[],'is_compatible'],
                    'detection_success': row[],'detection_success'],
                    'initialization_success': row[],'initialization_success'],
                    'error_message': row[],'error_message'],
                    'error_type': row[],'error_type'],
                    'suggested_fix': row[],'suggested_fix'],
                    'workaround_available': row[],'workaround_available'],
                    'compatibility_score': row[],'compatibility_score'],
                    'metadata': row[],'metadata']
                    }
                
                    compat_records.append()record)
            
            if compat_records:
                # Insert all records
                compat_mapped_df = pd.DataFrame()compat_records)
                conn.execute()"INSERT INTO hardware_compatibility SELECT NULL AS compatibility_id, * FROM compat_mapped_df")
                logger.debug()f"Inserted {}}}}}}}}}}}}}}}}}}}}}len()compat_records)} compatibility results")
        
        # Insert integration test results
        if 'integration_test_results' in data_dict:
            test_df = data_dict[],'integration_test_results']
            
            # Create a new DataFrame with the correct schema for insertion
            test_records = [],]
            
            for idx, row in test_df.iterrows()):
                # Map foreign keys
                model_name = row.get()'model_name', None)
                hardware_type = row.get()'hardware_type', None)
                
                # Look up IDs
                model_id = id_mappings.get()model_name, None) if model_name else None
                hardware_id = None
                :
                if hardware_type:
                    # Try to find hardware ID based on hardware_type and device_name
                    for key, value in id_mappings.items()):
                        if isinstance()key, tuple) and key[],0], == hardware_type:
                            hardware_id = value
                        break
                
                # Create record with mapped IDs
                        record = {}}}}}}}}}}}}}}}}}}}}}
                        'run_id': id_mappings.get()'run_id', None),
                        'test_module': row[],'test_module'],
                        'test_class': row[],'test_class'],
                        'test_name': row[],'test_name'],
                        'status': row[],'status'],
                        'execution_time_seconds': row[],'execution_time_seconds'],
                        'hardware_id': hardware_id,
                        'model_id': model_id,
                        'error_message': row[],'error_message'],
                        'error_traceback': row[],'error_traceback'],
                        'metadata': row[],'metadata']
                        }
                
                        test_records.append()record)
                
                # Store mapping from index to test_result_id
                        id_mappings[],f"test_result_{}}}}}}}}}}}}}}}}}}}}}idx}"] = len()test_records) - 1
            
            if test_records:
                # Insert all records
                test_mapped_df = pd.DataFrame()test_records)
                conn.execute()"INSERT INTO integration_test_results SELECT NULL AS test_result_id, * FROM test_mapped_df")
                
                # Get the first inserted test_result_id
                first_id = conn.execute()"SELECT last_insert_rowid()) - ? + 1", [],len()test_records)]).fetchone())[],0],
                logger.debug()f"Inserted {}}}}}}}}}}}}}}}}}}}}}len()test_records)} integration test results starting with ID {}}}}}}}}}}}}}}}}}}}}}first_id}")
                
                # Update the mappings with actual IDs
                for i in range()len()test_records)):
                    id_mappings[],f"test_result_{}}}}}}}}}}}}}}}}}}}}}i}"] = first_id + i
        
        # Insert integration test assertions
        if 'integration_test_assertions' in data_dict and data_dict[],'integration_test_assertions'] is not None:
            assertion_df = data_dict[],'integration_test_assertions']
            
            # Create a new DataFrame with the correct schema for insertion
            assertion_records = [],]
            
            for _, row in assertion_df.iterrows()):
                # Map foreign keys
                test_result_idx = row[],'test_result_idx']
                test_result_id = id_mappings.get()f"test_result_{}}}}}}}}}}}}}}}}}}}}}test_result_idx}", None)
                
                if test_result_id is None:
                    logger.warning()f"Missing test_result_id for assertion: {}}}}}}}}}}}}}}}}}}}}}row[],'assertion_name']}")
                continue
                
                # Create record with mapped IDs
                record = {}}}}}}}}}}}}}}}}}}}}}
                'test_result_id': test_result_id,
                'assertion_name': row[],'assertion_name'],
                'passed': row[],'passed'],
                'expected_value': row[],'expected_value'],
                'actual_value': row[],'actual_value'],
                'message': row[],'message']
                }
                
                assertion_records.append()record)
            
            if assertion_records:
                # Insert all records
                assertion_mapped_df = pd.DataFrame()assertion_records)
                conn.execute()"INSERT INTO integration_test_assertions SELECT NULL AS assertion_id, * FROM assertion_mapped_df")
                logger.debug()f"Inserted {}}}}}}}}}}}}}}}}}}}}}len()assertion_records)} test assertions")
        
                return True
        
    except Exception as e:
        logger.error()f"Error inserting data into database: {}}}}}}}}}}}}}}}}}}}}}e}")
                return False

def export_to_parquet()data_dict, parquet_dir, file_path):
    """Export processed dataframes to Parquet files"""
    if not data_dict:
    return
    
    # Create the parquet directory if it doesn't exist
    os.makedirs()parquet_dir, exist_ok=True)
    
    # Create subdirectories based on data category
    category = data_dict.get()'category', 'unknown')
    category_dir = os.path.join()parquet_dir, category)
    os.makedirs()category_dir, exist_ok=True)
    
    # Generate a base filename from the original JSON file
    base_filename = os.path.splitext()os.path.basename()file_path))[],0],
    
    # Export each dataframe to a Parquet file:
    for key, df in data_dict.items()):
        if key in [],'file_path', 'category']:
        continue
            
        if df is not None and not df.empty:
            parquet_file = os.path.join()category_dir, f"{}}}}}}}}}}}}}}}}}}}}}base_filename}_{}}}}}}}}}}}}}}}}}}}}}key}.parquet")
            df.to_parquet()parquet_file, index=False)
            logger.debug()f"Exported {}}}}}}}}}}}}}}}}}}}}}len()df)} rows to {}}}}}}}}}}}}}}}}}}}}}parquet_file}")

def main()):
    args = parse_args())
    
    # Set logging level
    if args.verbose:
        logger.setLevel()logging.DEBUG)
    
    # Connect to the database
        db_conn = connect_to_db()args.output_db)
    
    # Ensure schema exists
        ensure_schema_exists()db_conn)
    
    # Find JSON files to process
        json_files = discover_json_files()args.input_dir, args.categories)
    
    if not json_files:
        logger.warning()"No JSON files found to process")
        return
    
        logger.info()f"Found {}}}}}}}}}}}}}}}}}}}}}len()json_files)} JSON files to process")
    
    # Process files in parallel
        processed_count = 0
        inserted_count = 0
    
    with ProcessPoolExecutor()max_workers=args.parallel) as executor:
        # Submit all files for processing
        future_to_file = {}}}}}}}}}}}}}}}}}}}}}executor.submit()process_json_file, file_info): file_info 
                         for file_info in json_files}:
        # Process results as they complete
        for future in as_completed()future_to_file):
            file_info = future_to_file[],future]
            file_path, category = file_info
            
            try:::::::
                data_dict = future.result())
                
                if data_dict:
                    processed_count += 1
                    logger.info()f"Processed {}}}}}}}}}}}}}}}}}}}}}category} file: {}}}}}}}}}}}}}}}}}}}}}file_path}")
                    
                    # Export to Parquet if requested:
                    if args.parquet_dir:
                        export_to_parquet()data_dict, args.parquet_dir, file_path)
                    
                    # Insert into database if not dry run:
                    if not args.dry_run:
                        success = insert_dataframes_to_db()data_dict, db_conn)
                        if success:
                            inserted_count += 1
                else:
                    logger.warning()f"Failed to process {}}}}}}}}}}}}}}}}}}}}}category} file: {}}}}}}}}}}}}}}}}}}}}}file_path}")
            
            except Exception as e:
                logger.error()f"Error processing {}}}}}}}}}}}}}}}}}}}}}file_path}: {}}}}}}}}}}}}}}}}}}}}}e}")
    
    # Commit changes and close database
                db_conn.commit())
                db_conn.close())
    
                logger.info()f"Processing complete. Processed {}}}}}}}}}}}}}}}}}}}}}processed_count} files, inserted {}}}}}}}}}}}}}}}}}}}}}inserted_count} into database.")

if __name__ == "__main__":
    main())