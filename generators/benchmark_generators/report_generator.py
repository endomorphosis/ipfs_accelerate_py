#!/usr/bin/env python3
"""
Benchmark Visualizer with Test Generation

This module provides an interactive dashboard for visualizing benchmark results
and generating test files for hardware platforms. It integrates with the merged
test generator and allows launching benchmarks directly from the UI.

Features:
- Interactive visualization of benchmark results
- Direct generation of test files for models and hardware platforms
- Real-time execution of benchmarks
- Comparative analysis across hardware platforms
- Historical trend visualization

Usage:
    python benchmark_visualizer.py --serve
    python benchmark_visualizer.py --generate-dashboard --db ./benchmark_db.duckdb
    python benchmark_visualizer.py --export-report --format html --output report.html
"""

import os
import sys
import json
import time
import logging
import argparse
import datetime
import subprocess
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set

# Add DuckDB database support
try:
    from duckdb_api.core.benchmark_db_api import BenchmarkDBAPI
    BENCHMARK_DB_AVAILABLE = True
except ImportError:
    BENCHMARK_DB_AVAILABLE = False
    logger.warning("benchmark_db_api not available. Using deprecated JSON fallback.")


# Always deprecate JSON output in favor of DuckDB
DEPRECATE_JSON_OUTPUT = os.environ.get("DEPRECATE_JSON_OUTPUT", "1").lower() in ("1", "true", "yes")


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import duckdb
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    print("Warning: Visualization packages not installed. Install with:")
    print("pip install duckdb pandas numpy matplotlib seaborn")

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    print("Warning: Streamlit not installed. Interactive dashboard won't be available.")
    print("Install with: pip install streamlit")

# Constants
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = CURRENT_DIR.parent
DEFAULT_DB_PATH = CURRENT_DIR / "benchmark_db.duckdb"
RESULTS_DIR = CURRENT_DIR / "benchmark_results"
TEMPLATES_DIR = CURRENT_DIR / "hardware_test_templates"
SKILLS_DIR = CURRENT_DIR / "skills"

# Define key model types for test generation
KEY_MODELS = [
    "bert", "t5", "llama", "vit", "clip", "detr", 
    "clap", "wav2vec2", "whisper", "llava", "llava-next",
    "xclip", "qwen2"
]

# Define hardware platforms
HARDWARE_PLATFORMS = [
    "cpu", "cuda", "openvino", "mps", "rocm", "webnn", "webgpu"
]

class BenchmarkVisualizer:
    """
    Interactive benchmark visualization and test generation tool.
    """
    
    def __init__(self, db_path: str = str(DEFAULT_DB_PATH)):
        """Initialize the visualizer with database connection."""
        self.db_path = db_path
        self.db_conn = None
        
        # Try to connect to the database
        try:
            self.db_conn = duckdb.connect(db_path, read_only=True)
            logger.info(f"Connected to database: {db_path}")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            # Create an empty database if it doesn't exist
            if not os.path.exists(db_path):
                try:
                    os.makedirs(os.path.dirname(db_path), exist_ok=True)
                    self.db_conn = duckdb.connect(db_path)
                    logger.info(f"Created new database: {db_path}")
                except Exception as create_e:
                    logger.error(f"Error creating database: {create_e}")
        
        # Initialize dashboard state
        self.models = []
        self.hardware_platforms = []
        self.latest_results = {}
        self.historical_results = {}
        
        # Load initial data
        self._load_data()
    
    def __del__(self):
        """Close database connection when object is destroyed."""
        if self.db_conn:
            self.db_conn.close()
    
    def _load_data(self):
        """Load data from the database for visualization."""
        if not self.db_conn:
            logger.error("Database connection not available")
            return
        
        try:
            # Load models
            self.models = self._get_models()
            
            # Load hardware platforms
            self.hardware_platforms = self._get_hardware_platforms()
            
            # Load latest benchmark results
            self.latest_results = self._get_latest_results()
            
            # Load historical data for trends
            self.historical_results = self._get_historical_results()
            
            logger.info(f"Loaded data: {len(self.models)} models, {len(self.hardware_platforms)} platforms")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            traceback.print_exc()
    
    def _get_models(self) -> List[Dict[str, Any]]:
        """Get list of all models from the database."""
        try:
            query = """
            SELECT 
                model_id, model_name, model_family, 
                description, created_at, updated_at
            FROM 
                models
            ORDER BY
                model_name
            """
            
            result = self.db_conn.execute(query).fetchdf()
            
            if result.empty:
                # If no models in database, try to get from filesystem
                models = []
                for model_dir in (CURRENT_DIR / "skills").glob("test_hf_*.py"):
                    model_name = model_dir.stem.replace("test_hf_", "")
                    models.append({
                    "model_id": len(models) + 1,
                    "model_name": model_name,
                    "model_family": model_name.split("_")[0] if "_" in model_name else model_name,
                    "description": f"HuggingFace {model_name} model",
                    "created_at": datetime.datetime.now(),
                    "updated_at": datetime.datetime.now()
                    })
                return models
            
            return result.to_dict('records')
        except Exception as e:
            logger.error(f"Error getting models: {e}")
            return []
    
    def _get_hardware_platforms(self) -> List[Dict[str, Any]]:
        """Get list of all hardware platforms from the database."""
        try:
            query = """
            SELECT 
                hardware_id, hardware_type, device_name, 
                description, created_at, updated_at
            FROM 
                hardware_platforms
            ORDER BY
                hardware_type
            """
            
            result = self.db_conn.execute(query).fetchdf()
            
            if result.empty:
                # If no hardware in database, return default list
                return [
                    {"hardware_id": 1, "hardware_type": "cpu", "device_name": "CPU", 
                     "description": "CPU hardware platform"},
                    {"hardware_id": 2, "hardware_type": "cuda", "device_name": "CUDA", 
                     "description": "NVIDIA CUDA hardware platform"},
                    {"hardware_id": 3, "hardware_type": "openvino", "device_name": "OpenVINO", 
                     "description": "Intel OpenVINO hardware platform"},
                    {"hardware_id": 4, "hardware_type": "mps", "device_name": "MPS", 
                     "description": "Apple Silicon MPS hardware platform"},
                    {"hardware_id": 5, "hardware_type": "rocm", "device_name": "ROCm", 
                     "description": "AMD ROCm hardware platform"},
                    {"hardware_id": 6, "hardware_type": "webnn", "device_name": "WebNN", 
                     "description": "WebNN hardware platform"},
                    {"hardware_id": 7, "hardware_type": "webgpu", "device_name": "WebGPU", 
                     "description": "WebGPU hardware platform"}
                ]
            
            return result.to_dict('records')
        except Exception as e:
            logger.error(f"Error getting hardware platforms: {e}")
            return []
    
    def _get_latest_results(self) -> Dict[str, Any]:
        """Get latest benchmark results for visualization."""
        try:
            query = """
            SELECT 
                pr.result_id, pr.model_id, pr.hardware_id, pr.batch_size,
                pr.precision, pr.average_latency_ms, pr.throughput_items_per_second,
                pr.memory_peak_mb, pr.test_case, pr.created_at,
                m.model_name, m.model_family,
                hp.hardware_type, hp.device_name
            FROM 
                performance_results pr
            JOIN 
                models m ON pr.model_id = m.model_id
            JOIN 
                hardware_platforms hp ON pr.hardware_id = hp.hardware_id
            ORDER BY 
                pr.created_at DESC
            LIMIT 1000
            """
            
            result = self.db_conn.execute(query).fetchdf()
            
            if result.empty:
                return {}
            
            # Group by model and hardware
            grouped_results = {}
            
            # Convert DataFrame to desired structure
            for _, row in result.iterrows():
                model_name = row['model_name']
                hardware_type = row['hardware_type']
                
                if model_name not in grouped_results:
                    grouped_results[model_name] = {
                    'model_family': row['model_family'],
                    'hardware_results': {}
                    }
                
                if hardware_type not in grouped_results[model_name]['hardware_results']:
                    grouped_results[model_name]['hardware_results'][hardware_type] = []
                
                grouped_results[model_name]['hardware_results'][hardware_type].append({
                    'batch_size': row['batch_size'],
                    'precision': row['precision'],
                    'latency_ms': row['average_latency_ms'],
                    'throughput': row['throughput_items_per_second'],
                    'memory_mb': row['memory_peak_mb'],
                    'test_case': row['test_case'],
                    'timestamp': row['created_at'].isoformat() if hasattr(row['created_at'], 'isoformat') else str(row['created_at'])
                })
            
            return grouped_results
        except Exception as e:
            logger.error(f"Error getting latest results: {e}")
            return {}
    
    def _get_historical_results(self) -> Dict[str, Any]:
        """Get historical benchmark results for trend visualization."""
        try:
            query = """
            SELECT 
                pr.result_id, pr.model_id, pr.hardware_id, 
                pr.average_latency_ms, pr.throughput_items_per_second,
                pr.memory_peak_mb, pr.created_at,
                m.model_name, hp.hardware_type
            FROM 
                performance_results pr
            JOIN 
                models m ON pr.model_id = m.model_id
            JOIN 
                hardware_platforms hp ON pr.hardware_id = hp.hardware_id
            ORDER BY 
                pr.created_at
            """
            
            result = self.db_conn.execute(query).fetchdf()
            
            if result.empty:
                return {}
            
            # Group by model and hardware
            historical_data = {}
            
            # Convert DataFrame to desired structure
            for _, row in result.iterrows():
                model_name = row['model_name']
                hardware_type = row['hardware_type']
                
                if model_name not in historical_data:
                    historical_data[model_name] = {}
                
                if hardware_type not in historical_data[model_name]:
                    historical_data[model_name][hardware_type] = {
                    'timestamps': [],
                    'latency': [],
                    'throughput': [],
                    'memory': []
                    }
                
                historical_data[model_name][hardware_type]['timestamps'].append(row['created_at'])
                historical_data[model_name][hardware_type]['latency'].append(row['average_latency_ms'])
                historical_data[model_name][hardware_type]['throughput'].append(row['throughput_items_per_second'])
                historical_data[model_name][hardware_type]['memory'].append(row['memory_peak_mb'])
            
            return historical_data
        except Exception as e:
            logger.error(f"Error getting historical results: {e}")
            return {}
    
    def get_model_family_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model family for test generation."""
        try:
            base_model = model_name.split('-')[0].lower() if '-' in model_name else model_name.lower()
            
            # Check if this is a key model
            is_key_model = base_model in KEY_MODELS
            
            # Get model family
            model_family = None
            for model in self.models:
                if model['model_name'].lower() == model_name.lower():
                    model_family = model.get('model_family')
                    break
            
            # Get tasks (based on existing tests)
            tasks = []
            modality = "unknown"
            
            # Try to determine modality and tasks from model name
            if base_model in ['bert', 't5', 'llama', 'gpt2', 'distilbert', 'roberta']:
                modality = "text"
                tasks = ["text-generation"] if "llama" in model_name.lower() or "gpt" in model_name.lower() else ["fill-mask"]
            elif base_model in ['vit', 'clip', 'detr', 'resnet', 'convnext']:
                modality = "vision"
                tasks = ["image-classification"] if "vit" in model_name.lower() else ["object-detection" if "detr" in model_name.lower() else "image-embedding"]
            elif base_model in ['wav2vec2', 'whisper', 'clap']:
                modality = "audio"
                tasks = ["automatic-speech-recognition"] if "whisper" in model_name.lower() else ["audio-classification"]
            elif base_model in ['llava', 'blip', 'flava']:
                modality = "multimodal"
                tasks = ["visual-question-answering"]
            
            return {
                "model_name": model_name,
                "base_model": base_model,
                "model_family": model_family,
                "is_key_model": is_key_model,
                "modality": modality,
                "tasks": tasks
            }
        except Exception as e:
            logger.error(f"Error getting model family info: {e}")
            return {
                "model_name": model_name,
                "base_model": model_name,
                "model_family": "unknown",
                "is_key_model": False,
                "modality": "unknown",
                "tasks": []
            }
    
    def generate_test(self, model_name: str, hardware_platforms: List[str] = None) -> Tuple[bool, str]:
        """Generate test file for a specific model and hardware platforms."""
        try:
            # Get model information
            model_info = self.get_model_family_info(model_name)
            
            # Create command to generate test
            cmd = [
                "python", 
                str(CURRENT_DIR / "merged_test_generator.py"),
                "--generate", model_info["base_model"],
                "--output-dir", str(SKILLS_DIR)
            ]
            
            # Add hardware platform specific flags
            if hardware_platforms:
                if "openvino" in hardware_platforms:
                    cmd.extend(["--enhance-openvino"])
                if "webnn" in hardware_platforms or "webgpu" in hardware_platforms:
                    cmd.extend(["--enhance-web-platforms"])
            
            # Execute command
            logger.info(f"Generating test with command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Check for success
            if result.returncode == 0:
                # Test file path
                test_file_path = SKILLS_DIR / f"test_hf_{model_name.lower().replace('-', '_')}.py"
                
                if os.path.exists(test_file_path):
                    return True, f"Successfully generated test file: {test_file_path}"
                else:
                    return False, f"Test generation command succeeded but file not found: {test_file_path}"
            else:
                return False, f"Test generation failed: {result.stderr}"
        except Exception as e:
            logger.error(f"Error generating test: {e}")
            return False, f"Error generating test: {e}"
    
    def run_benchmark(self, model_name: str, hardware_platform: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Run benchmark for a specific model on specified hardware platform."""
        try:
            # Test file path
            normalized_name = model_name.lower().replace('-', '_')
            test_file_path = SKILLS_DIR / f"test_hf_{normalized_name}.py"
            
            # Check if test file exists
            if not os.path.exists(test_file_path):
                # Try to generate it
                success, message = self.generate_test(model_name, [hardware_platform])
                if not success:
                    return False, message, {}
            
            # Create command to run benchmark
            cmd = [
                "python",
                str(CURRENT_DIR / "model_benchmark_runner.py"),
                "--model", normalized_name,
                "--hardware", hardware_platform,
                "--output-dir", str(RESULTS_DIR)
            ]
            
            try:
                # Execute command
                logger.info(f"Running benchmark with command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                # Check for success
                if result.returncode == 0:
                    # Look for results file
                    result_files = list(RESULTS_DIR.glob(f"*{normalized_name}*{hardware_platform}*.json"))
                    
                    if result_files:
                        # Get the most recent file
                        result_file = max(result_files, key=os.path.getmtime)
                        
                        # Load the results
                        try:
                            # Try database first, fall back to JSON if necessary
                            from duckdb_api.core.benchmark_db_api import BenchmarkDBAPI
                            db_api = BenchmarkDBAPI(db_path=os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb"))
                            benchmark_results = db_api.get_benchmark_results()
                            logger.info("Successfully loaded results from database")
                        except Exception as e:
                            logger.warning(f"Error reading from database, falling back to JSON: {e}")
                            # JSON output deprecated in favor of database storage
                            if not DEPRECATE_JSON_OUTPUT:
                                with open(result_file, 'r') as f:
                                    benchmark_results = json.load(f)
                        
                        return True, f"Successfully ran benchmark: {result_file}", benchmark_results
                    else:
                        return False, "Benchmark command succeeded but results file not found", {}
                else:
                    return False, f"Benchmark failed: {result.stderr}", {}
            except Exception as e:
                logger.error(f"Error running benchmark: {e}")
                return False, f"Error running benchmark: {e}", {}
    
    def get_hardware_compatibility_matrix(self) -> Dict[str, Dict[str, str]]:
        """Get hardware compatibility matrix for all models."""
        try:
            # Create matrix of model -> hardware -> compatibility status
            compatibility_matrix = {}
            
            # Go through all models
            for model_name in [m['model_name'] for m in self.models]:
                compatibility_matrix[model_name] = {}
                
                # Check for benchmark results for each hardware platform
                for hw in HARDWARE_PLATFORMS:
                    status = "Unknown"
                    
                    # Check if we have benchmark results
                    if model_name in self.latest_results and hw in self.latest_results[model_name]['hardware_results']:
                        results = self.latest_results[model_name]['hardware_results'][hw]
                        if results:
                            # Simple check - if we have throughput > 0, it's compatible
                            if any(r.get('throughput', 0) > 0 for r in results):
                                status = "Compatible"
                            else:
                                # Check for errors
                                status = "Issues"
                    else:
                        # Check for test file
                        normalized_name = model_name.lower().replace('-', '_')
                        test_file_path = SKILLS_DIR / f"test_hf_{normalized_name}.py"
                        
                        if os.path.exists(test_file_path):
                            # Check if file mentions this hardware platform
                            with open(test_file_path, 'r') as f:
                                content = f.read()
                                if f"init_{hw}" in content:
                                    status = "Untested"
                                else:
                                    status = "Unsupported"
                        else:
                            status = "No Test"
                    
                    compatibility_matrix[model_name][hw] = status
                
                return compatibility_matrix
            except Exception as e:
                logger.error(f"Error getting hardware compatibility matrix: {e}")
                return {}
        
        def create_dashboard(self):
            """Create interactive Streamlit dashboard for visualizing benchmark results."""
            if not HAS_STREAMLIT:
                logger.error("Streamlit not installed. Dashboard can't be created.")
                return
            
            # Basic dashboard layout
            st.title("Hardware Benchmark Visualization Dashboard")
            st.sidebar.header("Navigation")
            
            page = st.sidebar.selectbox(
                "Select Page", 
                ["Hardware Compatibility Matrix", "Model Performance", "Hardware Comparison", 
                 "Test Generator", "Benchmark Runner", "Historical Trends"]
            )
            
            # Display appropriate page
            if page == "Hardware Compatibility Matrix":
                self._show_compatibility_matrix()
            elif page == "Model Performance":
                self._show_model_performance()
            elif page == "Hardware Comparison":
                self._show_hardware_comparison()
            elif page == "Test Generator":
                self._show_test_generator()
            elif page == "Benchmark Runner":
                self._show_benchmark_runner()
            elif page == "Historical Trends":
                self._show_historical_trends()
        
        def _show_compatibility_matrix(self):
            """Show hardware compatibility matrix page."""
            st.header("Hardware Compatibility Matrix")
            
            # Get compatibility matrix
            matrix = self.get_hardware_compatibility_matrix()
            
            if not matrix:
                st.warning("No compatibility data available.")
                return
            
            # Convert to DataFrame for easier display
            data = []
            for model_name, hw_status in matrix.items():
                row = {"Model": model_name}
                for hw, status in hw_status.items():
                    row[hw.upper()] = status
                data.append(row)
            
            df = pd.DataFrame(data)
            
            # Display as table with conditional formatting
            st.dataframe(df.style.apply(
                lambda row: [
                    'background-color: #90EE90' if cell == 'Compatible' else
                    'background-color: #FFFFE0' if cell == 'Untested' else
                    'background-color: #FFA07A' if cell == 'Issues' else
                    'background-color: #F0F0F0' if cell == 'Unsupported' else
                    'background-color: #F8F8F8' if cell == 'No Test' else
                    '' for cell in row
                ], axis=1
            ))
            
            # Add explanation
            st.markdown("""
            **Legend:**
            - **Compatible**: Model has been successfully benchmarked on this hardware
            - **Untested**: Model has test implementation but hasn't been benchmarked
            - **Issues**: Model has been tested but encountered issues
            - **Unsupported**: Model doesn't support this hardware platform
            - **No Test**: No test implementation available for this model
            """)
            
            # Add action buttons
            st.subheader("Actions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Generate tests for models without them
                if st.button("Generate Missing Tests"):
                    models_without_tests = [model for model, hw_status in matrix.items() 
                                      if any(status == "No Test" for status in hw_status.values())]
                    
                    if models_without_tests:
                    st.info(f"Generating tests for {len(models_without_tests)} models...")
                    
                    for model in models_without_tests:
                        success, message = self.generate_test(model)
                        st.write(f"{model}: {message}")
                    
                    st.success("Test generation complete. Refresh page to see updated matrix.")
                    else:
                    st.info("No missing tests found!")
            
            with col2:
                # Run benchmarks for untested models
                if st.button("Run Untested Benchmarks"):
                    untested_configs = []
                    
                    for model, hw_status in matrix.items():
                    for hw, status in hw_status.items():
                        if status == "Untested":
                            untested_configs.append((model, hw))
                    
                    if untested_configs:
                    st.info(f"Running benchmarks for {len(untested_configs)} model-hardware combinations...")
                    
                    for model, hw in untested_configs:
                        st.write(f"Benchmarking {model} on {hw}...")
                        success, message, _ = self.run_benchmark(model, hw)
                        st.write(f"{model} on {hw}: {'Success' if success else 'Failed'} - {message}")
                    
                    st.success("Benchmarking complete. Refresh page to see updated matrix.")
                    else:
                    st.info("No untested configurations found!")
        
        def _show_model_performance(self):
            """Show model performance comparison page."""
            st.header("Model Performance Comparison")
            
            # Select hardware platform
            hardware_options = [h["hardware_type"] for h in self.hardware_platforms]
            selected_hardware = st.selectbox("Select Hardware Platform", hardware_options)
            
            # Select metric
            metric_options = ["Throughput (items/s)", "Latency (ms)", "Memory Usage (MB)"]
            selected_metric = st.selectbox("Select Performance Metric", metric_options)
            
            # Get metric key
            metric_key = "throughput" if "Throughput" in selected_metric else \
                    "latency_ms" if "Latency" in selected_metric else "memory_mb"
            
            # Get filtered results
            filtered_results = {}
            
            for model_name, model_data in self.latest_results.items():
                if selected_hardware in model_data.get('hardware_results', {}):
                    hw_results = model_data['hardware_results'][selected_hardware]
                    if hw_results:
                    # Calculate average of the metric for this model/hardware
                    values = [result.get(metric_key, 0) for result in hw_results]
                    if values:
                        filtered_results[model_name] = {
                            'family': model_data.get('model_family', 'Unknown'),
                            'value': sum(values) / len(values)
                        }
            
            if not filtered_results:
                st.warning(f"No performance data available for {selected_hardware}.")
                return
            
            # Create DataFrame for visualization
            df = pd.DataFrame([
                {
                    'Model': model_name,
                    'Family': data['family'],
                    'Value': data['value']
                }
                for model_name, data in filtered_results.items()
            ])
            
            # Sort by value (ascending for latency, descending for others)
            df = df.sort_values(by='Value', ascending=("Latency" in selected_metric))
            
            # Show results as chart
            fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.3)))
            
            # Create color mapping by family
            families = df['Family'].unique()
            cmap = plt.cm.get_cmap('tab10', len(families))
            family_colors = {family: cmap(i) for i, family in enumerate(families)}
            
            # Create horizontal bar chart
            bars = ax.barh(
                df['Model'], 
                df['Value'],
                color=[family_colors[family] for family in df['Family']]
            )
            
            # Add value labels
            for i, bar in enumerate(bars):
                value = df.iloc[i]['Value']
                ax.text(
                    value + (max(df['Value']) * 0.01),
                    bar.get_y() + bar.get_height()/2,
                    f'{value:.2f}',
                    va='center'
                )
            
            # Add legend
            legend_handles = [plt.Rectangle((0,0),1,1, color=color) for color in family_colors.values()]
            ax.legend(legend_handles, family_colors.keys(), title="Model Family")
            
            # Set labels and title
            ax.set_xlabel(selected_metric)
            ax.set_title(f"{selected_metric} by Model on {selected_hardware.upper()}")
            
            # Add grid
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Display plot
            st.pyplot(fig)
            
            # Show raw data in expandable section
            with st.expander("Show Raw Data"):
                st.dataframe(df)
        
        def _show_hardware_comparison(self):
            """Show hardware platform comparison page."""
            st.header("Hardware Platform Comparison")
            
            # Select model
            model_options = [m["model_name"] for m in self.models]
            selected_model = st.selectbox("Select Model", model_options)
            
            # Select metric
            metric_options = ["Throughput (items/s)", "Latency (ms)", "Memory Usage (MB)"]
            selected_metric = st.selectbox("Select Performance Metric", metric_options)
            
            # Get metric key
            metric_key = "throughput" if "Throughput" in selected_metric else \
                    "latency_ms" if "Latency" in selected_metric else "memory_mb"
            
            # Get filtered results
            filtered_results = {}
            
            if selected_model in self.latest_results:
                model_data = self.latest_results[selected_model]
                
                for hw_type, hw_results in model_data.get('hardware_results', {}).items():
                    if hw_results:
                    # Calculate average of the metric for this hardware
                    values = [result.get(metric_key, 0) for result in hw_results]
                    if values:
                        filtered_results[hw_type] = sum(values) / len(values)
            
            if not filtered_results:
                st.warning(f"No performance data available for {selected_model}.")
                return
            
            # Create DataFrame for visualization
            df = pd.DataFrame([
                {'Hardware': hw_type, 'Value': value}
                for hw_type, value in filtered_results.items()
            ])
            
            # Sort by value (ascending for latency, descending for others)
            df = df.sort_values(by='Value', ascending=("Latency" in selected_metric))
            
            # Show results as chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create bar chart
            bars = ax.bar(
                df['Hardware'].str.upper(), 
                df['Value'],
                color=plt.cm.viridis(np.linspace(0, 0.8, len(df)))
            )
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    height + (max(df['Value']) * 0.01),
                    f'{height:.2f}',
                    ha='center'
                )
            
            # Set labels and title
            ax.set_xlabel("Hardware Platform")
            ax.set_ylabel(selected_metric)
            ax.set_title(f"{selected_metric} Comparison for {selected_model}")
            
            # Add grid
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Display plot
            st.pyplot(fig)
            
            # Show raw data in expandable section
            with st.expander("Show Raw Data"):
                st.dataframe(df)
            
            # Add action button to run missing benchmarks
            hardware_options = [h["hardware_type"] for h in self.hardware_platforms]
            missing_hw = [hw for hw in hardware_options if hw not in filtered_results]
            
            if missing_hw:
                st.subheader("Missing Hardware Benchmarks")
                st.write(f"This model has not been benchmarked on: {', '.join(hw.upper() for hw in missing_hw)}")
                
                selected_missing_hw = st.selectbox("Select Hardware to Benchmark", missing_hw)
                
                if st.button(f"Run Benchmark on {selected_missing_hw.upper()}"):
                    st.info(f"Running benchmark for {selected_model} on {selected_missing_hw}...")
                    
                    success, message, results = self.run_benchmark(selected_model, selected_missing_hw)
                    
                    if success:
                    st.success(f"Benchmark completed successfully!")
                    st.json(results)
                    else:
                    st.error(f"Benchmark failed: {message}")
        
        def _show_test_generator(self):
            """Show test generator page."""
            st.header("Test Generator")
            
            # Two options: generate for existing model or create new test
            tab1, tab2 = st.tabs(["Generate for Existing Model", "Generate New Test"])
            
            with tab1:
                # Select model
                model_options = [m["model_name"] for m in self.models]
                selected_model = st.selectbox("Select Model", model_options, key="existing_model")
                
                # Select hardware platforms
                hardware_options = [h["hardware_type"] for h in self.hardware_platforms]
                selected_hardware = st.multiselect("Select Hardware Platforms", hardware_options, default=["cpu", "cuda"])
                
                # Get model info
                model_info = self.get_model_family_info(selected_model)
                
                # Display model info
                st.subheader("Model Information")
                st.json({
                    "model_name": model_info["model_name"],
                    "model_family": model_info["model_family"],
                    "is_key_model": model_info["is_key_model"],
                    "modality": model_info["modality"],
                    "tasks": model_info["tasks"]
                })
                
                # Generate button
                if st.button("Generate Test", key="generate_existing"):
                    st.info(f"Generating test for {selected_model} with support for: {', '.join(selected_hardware)}")
                    
                    success, message = self.generate_test(selected_model, selected_hardware)
                    
                    if success:
                    st.success(message)
                    
                    # Show test file path
                    normalized_name = selected_model.lower().replace('-', '_')
                    test_file_path = SKILLS_DIR / f"test_hf_{normalized_name}.py"
                    
                    if os.path.exists(test_file_path):
                        st.code(f"Test file path: {test_file_path}")
                        
                        # Show first few lines of the test file
                        with open(test_file_path, 'r') as f:
                            content = f.read()
                            line_count = content.count('\n')
                            
                            st.write(f"Test file generated with {line_count} lines of code")
                            
                            with st.expander("View Test File"):
                                st.code(content[:5000] + "..." if len(content) > 5000 else content)
                    else:
                    st.error(message)
            
            with tab2:
                # Input model name
                new_model_name = st.text_input("Model Name", value="custom-model")
                
                # Select model family
                family_options = list(set(m.get("model_family", "") for m in self.models if m.get("model_family")))
                selected_family = st.selectbox("Model Family", [""] + family_options)
                
                # Select modality
                modality_options = ["text", "vision", "audio", "multimodal"]
                selected_modality = st.selectbox("Modality", modality_options)
                
                # Select tasks based on modality
                task_options = []
                if selected_modality == "text":
                    task_options = ["text-generation", "fill-mask", "text-classification", "question-answering"]
                elif selected_modality == "vision":
                    task_options = ["image-classification", "object-detection", "image-segmentation", "depth-estimation"]
                elif selected_modality == "audio":
                    task_options = ["automatic-speech-recognition", "audio-classification", "text-to-audio"]
                elif selected_modality == "multimodal":
                    task_options = ["visual-question-answering", "image-to-text", "text-to-image"]
                
                selected_tasks = st.multiselect("Tasks", task_options)
                
                # Select hardware platforms
                hardware_options = [h["hardware_type"] for h in self.hardware_platforms]
                selected_hardware = st.multiselect("Hardware Platforms", hardware_options, default=["cpu", "cuda"])
                
                # Generate button
                if st.button("Generate Test", key="generate_new"):
                    if not new_model_name:
                    st.error("Model name is required")
                    elif not selected_tasks:
                    st.error("At least one task must be selected")
                    elif not selected_hardware:
                    st.error("At least one hardware platform must be selected")
                    else:
                    st.info(f"Generating test for {new_model_name} with support for: {', '.join(selected_hardware)}")
                    
                    # Create model info
                    model_info = {
                        "model": new_model_name,
                        "normalized_name": new_model_name.lower().replace('-', '_'),
                        "pipeline_tasks": selected_tasks,
                        "priority": "HIGH"
                    }
                    
                    # Get existing tests
                    existing_tests = self._get_existing_tests()
                    
                    # Generate test file
                    success, message = self._generate_custom_test(model_info, existing_tests, selected_hardware)
                    
                    if success:
                        st.success(message)
                        
                        # Show test file path
                        normalized_name = new_model_name.lower().replace('-', '_')
                        test_file_path = SKILLS_DIR / f"test_hf_{normalized_name}.py"
                        
                        if os.path.exists(test_file_path):
                            st.code(f"Test file path: {test_file_path}")
                            
                            # Show first few lines of the test file
                            with open(test_file_path, 'r') as f:
                                content = f.read()
                                line_count = content.count('\n')
                                
                                st.write(f"Test file generated with {line_count} lines of code")
                                
                                with st.expander("View Test File"):
                                    st.code(content[:5000] + "..." if len(content) > 5000 else content)
                    else:
                        st.error(message)
        
        def _show_benchmark_runner(self):
            """Show benchmark runner page."""
            st.header("Benchmark Runner")
            
            # Select model
            model_options = [m["model_name"] for m in self.models]
            selected_model = st.selectbox("Select Model", model_options)
            
            # Select hardware platform
            hardware_options = [h["hardware_type"] for h in self.hardware_platforms]
            selected_hardware = st.selectbox("Select Hardware Platform", hardware_options)
            
            # Show existing results if available
            if selected_model in self.latest_results and selected_hardware in self.latest_results[selected_model]['hardware_results']:
                results = self.latest_results[selected_model]['hardware_results'][selected_hardware]
                
                if results:
                    st.subheader("Existing Benchmark Results")
                    
                    # Create DataFrame for display
                    df = pd.DataFrame([
                    {
                        'Batch Size': r.get('batch_size', 'N/A'),
                        'Precision': r.get('precision', 'N/A'),
                        'Latency (ms)': r.get('latency_ms', 0),
                        'Throughput (items/s)': r.get('throughput', 0),
                        'Memory (MB)': r.get('memory_mb', 0),
                        'Test Case': r.get('test_case', 'N/A'),
                        'Timestamp': r.get('timestamp', 'N/A')
                    }
                    for r in results
                    ])
                    
                    st.dataframe(df)
                    
                    # Add visualization
                    metric_options = ["Throughput (items/s)", "Latency (ms)", "Memory Usage (MB)"]
                    selected_metric = st.selectbox("Visualization Metric", metric_options)
                    
                    # Get metric key
                    metric_key = "throughput" if "Throughput" in selected_metric else \
                            "latency_ms" if "Latency" in selected_metric else "memory_mb"
                    
                    # Check if we have batch size variation for visualization
                    batch_sizes = df['Batch Size'].unique()
                    
                    if len(batch_sizes) > 1:
                    # Create batch size vs. metric plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Get data
                    batch_values = []
                    metric_values = []
                    
                    for r in results:
                        if metric_key in r:
                            batch_values.append(r.get('batch_size', 0))
                            metric_values.append(r.get(metric_key, 0))
                    
                    # Create scatter plot with line
                    ax.plot(batch_values, metric_values, 'o-', color='royalblue')
                    
                    # Add data point labels
                    for x, y in zip(batch_values, metric_values):
                        ax.text(x, y + (max(metric_values) * 0.02), f'{y:.2f}', ha='center')
                    
                    # Set labels and title
                    ax.set_xlabel("Batch Size")
                    ax.set_ylabel(selected_metric)
                    ax.set_title(f"{selected_metric} vs. Batch Size for {selected_model} on {selected_hardware.upper()}")
                    
                    # Add grid
                    ax.grid(linestyle='--', alpha=0.7)
                    
                    # Display plot
                    st.pyplot(fig)
                    else:
                    st.info("Only one batch size available. Run benchmarks with different batch sizes to see scaling.")
            
            # Run benchmark button
            st.subheader("Run New Benchmark")
            
            # Batch size options
            batch_sizes = st.text_input("Batch Sizes (comma-separated)", value="1,2,4,8")
            
            if st.button("Run Benchmark"):
                batch_size_list = [int(b.strip()) for b in batch_sizes.split(",") if b.strip().isdigit()]
                
                if not batch_size_list:
                    st.error("Please enter at least one valid batch size")
                else:
                    st.info(f"Running benchmark for {selected_model} on {selected_hardware} with batch sizes: {batch_size_list}")
                    
                    # Create command
                    cmd = [
                    "python",
                    str(CURRENT_DIR / "model_benchmark_runner.py"),
                    "--model", selected_model.lower().replace('-', '_'),
                    "--hardware", selected_hardware,
                    "--batch-sizes", ",".join(map(str, batch_size_list)),
                    "--output-dir", str(RESULTS_DIR)
                    ]
                    
                    # Execute command
                    with st.spinner("Running benchmark..."):
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        
                        if result.returncode == 0:
                            st.success("Benchmark completed successfully!")
                            
                            # Look for results file
                            normalized_name = selected_model.lower().replace('-', '_')
                            result_files = list(RESULTS_DIR.glob(f"*{normalized_name}*{selected_hardware}*.json"))
                            
                            if result_files:
                                # Get the most recent file
                                result_file = max(result_files, key=os.path.getmtime)
                                
                                # Load the results
                                with open(result_file, 'r') as f:
                                    benchmark_results = json.load(f)
                                
                                st.subheader("Benchmark Results")
                                st.json(benchmark_results)
                                
                                # Add to database
                                st.info("Benchmark results will be automatically added to the database.")
                                
                                # Reload data
                                self._load_data()
                                
                                st.success("Data reloaded. You can now view the results in the visualization sections.")
                            else:
                                st.warning("Benchmark command succeeded but results file not found")
                        else:
                            st.error(f"Benchmark failed: {result.stderr}")
                    except Exception as e:
                        st.error(f"Error running benchmark: {e}")
        
        def _show_historical_trends(self):
            """Show historical performance trends page."""
            st.header("Historical Performance Trends")
            
            # Select model
            model_options = [m["model_name"] for m in self.models]
            selected_model = st.selectbox("Select Model", model_options)
            
            # Select hardware platforms
            hardware_options = [h["hardware_type"] for h in self.hardware_platforms]
            selected_hardware = st.multiselect("Select Hardware Platforms", hardware_options, default=["cpu", "cuda"])
            
            # Select metric
            metric_options = ["Throughput (items/s)", "Latency (ms)", "Memory Usage (MB)"]
            selected_metric = st.selectbox("Select Performance Metric", metric_options)
            
            # Get metric key
            metric_key = "throughput" if "Throughput" in selected_metric else \
                    "latency" if "Latency" in selected_metric else "memory"
            
            # Check if we have historical data
            if selected_model not in self.historical_results:
                st.warning(f"No historical data available for {selected_model}.")
                return
            
            # Filter to selected hardware platforms
            available_hw = [hw for hw in selected_hardware if hw in self.historical_results[selected_model]]
            
            if not available_hw:
                st.warning(f"No historical data available for {selected_model} on selected hardware platforms.")
                return
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Add data for each hardware platform
            for hw in available_hw:
                hw_data = self.historical_results[selected_model][hw]
                
                if not hw_data['timestamps'] or not hw_data[metric_key]:
                    continue
                    
                # Plot line
                ax.plot(
                    hw_data['timestamps'], 
                    hw_data[metric_key],
                    'o-',
                    label=hw.upper()
                )
            
            # Set labels and title
            ax.set_xlabel("Date")
            ax.set_ylabel(selected_metric)
            ax.set_title(f"{selected_metric} Trend for {selected_model}")
            
            # Format x-axis for dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            fig.autofmt_xdate()
            
            # Add legend
            ax.legend()
            
            # Add grid
            ax.grid(linestyle='--', alpha=0.7)
            
            # Display plot
            st.pyplot(fig)
            
            # Show raw data in expandable section
            with st.expander("Show Raw Data"):
                for hw in available_hw:
                    hw_data = self.historical_results[selected_model][hw]
                    
                    if not hw_data['timestamps']:
                    continue
                    
                    st.subheader(f"{hw.upper()} Data")
                    
                    # Create DataFrame
                    df = pd.DataFrame({
                    'Timestamp': hw_data['timestamps'],
                    'Value': hw_data[metric_key]
                    })
                    
                    st.dataframe(df)
        
        def _get_existing_tests(self) -> Set[str]:
            """Get the normalized names of existing test files."""
            try:
                test_files = [f.stem for f in SKILLS_DIR.glob("test_hf_*.py")]
                return set(f.replace("test_hf_", "") for f in test_files)
            except Exception as e:
                logger.error(f"Error getting existing tests: {e}")
                return set()
        
        def _generate_custom_test(self, model_info: Dict[str, Any], existing_tests: Set[str], 
                            hardware_platforms: List[str]) -> Tuple[bool, str]:
            """Generate a custom test file for a new model."""
            try:
                # Import merged test generator
                sys.path.insert(0, str(CURRENT_DIR))
                
                try:
                    from generators.test_generators.merged_test_generator import generate_test_file, KEY_MODEL_HARDWARE_MAP
                except ImportError:
                    return False, "Could not import generators.test_generators.merged_test_generator as merged_test_generator module"
                
                # Create output directory if it doesn't exist
                if not SKILLS_DIR.exists():
                    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
                
                # Generate the test file
                success, message = generate_test_file(
                    model_info,
                    existing_tests,
                    [m["model_name"] for m in self.models],
                    str(SKILLS_DIR)
                )
                
                return success, message
            except Exception as e:
                logger.error(f"Error generating custom test: {e}")
                return False, f"Error generating custom test: {e}"
        
        def serve_dashboard(self, port: int = 8501):
            """Serve the dashboard using Streamlit."""
            if not HAS_STREAMLIT:
                logger.error("Streamlit not installed. Dashboard can't be served.")
                return
            
            # Create a temporary Streamlit script
            script_path = CURRENT_DIR / "_temp_dashboard.py"
            
            with open(script_path, 'w') as f:
                f.write(f"""
    import os
    import sys
    import streamlit as st
    
    # Add parent directory to path
    sys.path.insert(0, "{CURRENT_DIR}")
    
    # Import the visualizer
    from benchmark_visualizer import BenchmarkVisualizer
    
    # Create visualizer instance
    visualizer = BenchmarkVisualizer(db_path="{self.db_path}")
    
    # Run the dashboard
    visualizer.create_dashboard()
    """)
            
            # Launch Streamlit
            try:
                cmd = ["streamlit", "run", str(script_path), "--server.port", str(port)]
                subprocess.run(cmd)
            except Exception as e:
                logger.error(f"Error launching Streamlit dashboard: {e}")
            finally:
                # Clean up temporary script
                if script_path.exists():
                    os.remove(script_path)
        
        def export_report(self, format: str, output: str):
            """Export a comprehensive benchmark report."""
            try:
                if format == "html":
                    # Use the Dashboard Builder to create an HTML report
                    report_content = self._generate_html_report()
                    
                    with open(output, 'w') as f:
                    f.write(report_content)
                    
                    logger.info(f"HTML report saved to {output}")
                    return True
                elif format == "markdown" or format == "md":
                    # Generate a markdown report
                    report_content = self._generate_markdown_report()
                    
                    with open(output, 'w') as f:
                    f.write(report_content)
                    
                    logger.info(f"Markdown report saved to {output}")
                    return True
                elif format == "json":
                    # Export raw data as JSON
                    report_data = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "models": self.models,
                    "hardware_platforms": self.hardware_platforms,
                    "latest_results": self.latest_results,
                    "compatibility_matrix": self.get_hardware_compatibility_matrix()
                    }
                    
                    with open(output, 'w') as f:
                    json.dump(report_data, f, indent=2)
else:
    logger.info("JSON output is deprecated. Results are stored directly in the database.")

                    
                logger.info(f"JSON report saved to {output}")
                return True
            else:
                logger.error(f"Unsupported format: {format}")
                return False
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
            return False
    
    def _generate_html_report(self) -> str:
        """Generate a comprehensive HTML report."""
        compatibility_matrix = self.get_hardware_compatibility_matrix()
        
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>Hardware Benchmark Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            "table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #f2f2f2; }",
            "tr:nth-child(even) { background-color: #f9f9f9; }",
            ".compatible { background-color: #90EE90; }",
            ".untested { background-color: #FFFFE0; }",
            ".issues { background-color: #FFA07A; }",
            ".unsupported { background-color: #F0F0F0; }",
            ".notest { background-color: #F8F8F8; }",
            "h1, h2, h3 { color: #333; }",
            "section { margin-bottom: 30px; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>Hardware Benchmark Report</h1>",
            f"<p>Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
            
            "<section>",
            "<h2>Hardware Compatibility Matrix</h2>",
            "<table>",
            "<tr><th>Model</th>" + "".join([f"<th>{hw.upper()}</th>" for hw in HARDWARE_PLATFORMS]) + "</tr>"
        ]
        
        # Add rows for each model
        for model_name, hw_status in compatibility_matrix.items():
            row = [f"<tr><td>{model_name}</td>"]
            
            for hw in HARDWARE_PLATFORMS:
                status = hw_status.get(hw, "Unknown")
                css_class = "compatible" if status == "Compatible" else \
                      "untested" if status == "Untested" else \
                      "issues" if status == "Issues" else \
                      "unsupported" if status == "Unsupported" else \
                      "notest" if status == "No Test" else ""
                
                row.append(f"<td class='{css_class}'>{status}</td>")
            
            row.append("</tr>")
            html.append("".join(row))
        
        html.append("</table>")
        html.append("</section>")
        
        # Add performance summary
        html.append("<section>")
        html.append("<h2>Performance Summary</h2>")
        
        for hw in HARDWARE_PLATFORMS:
            # Count models with results for this hardware
            models_with_results = sum(1 for model_name, model_data in self.latest_results.items() 
                                 if hw in model_data.get('hardware_results', {}))
            
            if models_with_results > 0:
                html.append(f"<h3>{hw.upper()} Performance</h3>")
                html.append("<table>")
                html.append("<tr><th>Model</th><th>Throughput (items/s)</th><th>Latency (ms)</th><th>Memory (MB)</th></tr>")
                
                for model_name, model_data in sorted(self.latest_results.items()):
                    if hw in model_data.get('hardware_results', {}):
                    hw_results = model_data['hardware_results'][hw]
                    if hw_results:
                        # Calculate averages
                        throughput = sum(r.get('throughput', 0) for r in hw_results) / len(hw_results)
                        latency = sum(r.get('latency_ms', 0) for r in hw_results) / len(hw_results)
                        memory = sum(r.get('memory_mb', 0) for r in hw_results) / len(hw_results)
                        
                        html.append(f"<tr><td>{model_name}</td><td>{throughput:.2f}</td><td>{latency:.2f}</td><td>{memory:.2f}</td></tr>")
                
                html.append("</table>")
        
        html.append("</section>")
        
        # Add key model section
        html.append("<section>")
        html.append("<h2>Key Model Performance</h2>")
        
        key_model_results = {}
        
        for model_name, model_data in self.latest_results.items():
            base_model = model_name.split('-')[0].lower() if '-' in model_name else model_name.lower()
            if base_model in KEY_MODELS:
                key_model_results[model_name] = model_data
        
        if key_model_results:
            html.append("<table>")
            html.append(f"<tr><th>Model</th>" + "".join([f"<th>{hw.upper()} Throughput</th>" for hw in HARDWARE_PLATFORMS]) + "</tr>")
            
            for model_name, model_data in sorted(key_model_results.items()):
                row = [f"<tr><td>{model_name}</td>"]
                
                for hw in HARDWARE_PLATFORMS:
                    if hw in model_data.get('hardware_results', {}):
                    hw_results = model_data['hardware_results'][hw]
                    if hw_results:
                        # Calculate average throughput
                        throughput = sum(r.get('throughput', 0) for r in hw_results) / len(hw_results)
                        row.append(f"<td>{throughput:.2f}</td>")
                    else:
                        row.append("<td>N/A</td>")
                    else:
                    row.append("<td>N/A</td>")
                
                row.append("</tr>")
                html.append("".join(row))
            
            html.append("</table>")
        else:
            html.append("<p>No performance data available for key models.</p>")
        
        html.append("</section>")
        
        # Close HTML
        html.append("</body>")
        html.append("</html>")
        
        return "\n".join(html)
    
    def _generate_markdown_report(self) -> str:
        """Generate a comprehensive Markdown report."""
        compatibility_matrix = self.get_hardware_compatibility_matrix()
        
        md = [
            "# Hardware Benchmark Report",
            "",
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Hardware Compatibility Matrix",
            "",
            "| Model | " + " | ".join([hw.upper() for hw in HARDWARE_PLATFORMS]) + " |",
            "|" + "-|" * (len(HARDWARE_PLATFORMS) + 1)
        ]
        
        # Add rows for each model
        for model_name, hw_status in compatibility_matrix.items():
            row = [f"| {model_name}"]
            
            for hw in HARDWARE_PLATFORMS:
                status = hw_status.get(hw, "Unknown")
                row.append(f" {status}")
            
            row.append(" |")
            md.append("".join(row))
        
        md.append("")
        
        # Add performance summary
        md.append("## Performance Summary")
        md.append("")
        
        for hw in HARDWARE_PLATFORMS:
            # Count models with results for this hardware
            models_with_results = sum(1 for model_name, model_data in self.latest_results.items() 
                                 if hw in model_data.get('hardware_results', {}))
            
            if models_with_results > 0:
                md.append(f"### {hw.upper()} Performance")
                md.append("")
                md.append("| Model | Throughput (items/s) | Latency (ms) | Memory (MB) |")
                md.append("|-------|---------------------|--------------|-------------|")
                
                for model_name, model_data in sorted(self.latest_results.items()):
                    if hw in model_data.get('hardware_results', {}):
                    hw_results = model_data['hardware_results'][hw]
                    if hw_results:
                        # Calculate averages
                        throughput = sum(r.get('throughput', 0) for r in hw_results) / len(hw_results)
                        latency = sum(r.get('latency_ms', 0) for r in hw_results) / len(hw_results)
                        memory = sum(r.get('memory_mb', 0) for r in hw_results) / len(hw_results)
                        
                        md.append(f"| {model_name} | {throughput:.2f} | {latency:.2f} | {memory:.2f} |")
                
                md.append("")
        
        # Add key model section
        md.append("## Key Model Performance")
        md.append("")
        
        key_model_results = {}
        
        for model_name, model_data in self.latest_results.items():
            base_model = model_name.split('-')[0].lower() if '-' in model_name else model_name.lower()
            if base_model in KEY_MODELS:
                key_model_results[model_name] = model_data
        
        if key_model_results:
            md.append("| Model | " + " | ".join([f"{hw.upper()} Throughput" for hw in HARDWARE_PLATFORMS]) + " |")
            md.append("|" + "-|" * (len(HARDWARE_PLATFORMS) + 1))
            
            for model_name, model_data in sorted(key_model_results.items()):
                row = [f"| {model_name}"]
                
                for hw in HARDWARE_PLATFORMS:
                    if hw in model_data.get('hardware_results', {}):
                    hw_results = model_data['hardware_results'][hw]
                    if hw_results:
                        # Calculate average throughput
                        throughput = sum(r.get('throughput', 0) for r in hw_results) / len(hw_results)
                        row.append(f" {throughput:.2f}")
                    else:
                        row.append(" N/A")
                    else:
                    row.append(" N/A")
                
                row.append(" |")
                md.append("".join(row))
        else:
            md.append("No performance data available for key models.")
        
        return "\n".join(md)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark Visualizer with Test Generation")
    
    # Dashboard options
    parser.add_argument("--serve", action="store_true", help="Serve interactive dashboard")
    parser.add_argument("--port", type=int, default=8501, help="Port for dashboard server")
    
    # Database options
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB_PATH), help="Path to benchmark database")
    
    # Test generation options
    parser.add_argument("--generate-test", type=str, help="Generate test for a specific model")
    parser.add_argument("--hardware", type=str, help="Comma-separated list of hardware platforms for test generation")
    
    # Report generation options
    parser.add_argument("--export-report", action="store_true", help="Export benchmark report")
    parser.add_argument("--format", type=str, choices=["html", "markdown", "md", "json"], default="html", help="Report format")
    parser.add_argument("--output", type=str, default="benchmark_report.html", help="Output file path")
    
    # Dashboard generation options (static)
    parser.add_argument("--generate-dashboard", action="store_true", help="Generate static dashboard HTML")
    
    # Benchmark options
    parser.add_argument("--run-benchmark", type=str, help="Run benchmark for a specific model")
    parser.add_argument("--benchmark-hardware", type=str, default="cpu", help="Hardware platform for benchmark")
    
    
    parser.add_argument("--db-path", type=str, default=None,
                      help="Path to the benchmark database")
    parser.add_argument("--db-only", action="store_true",
                      help="Store results only in the database, not in JSON")
return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Create visualizer
    visualizer = BenchmarkVisualizer(db_path=args.db)
    
    # Handle command
    if args.serve:
        # Serve interactive dashboard
        print(f"Serving dashboard on port {args.port}...")
        visualizer.serve_dashboard(port=args.port)
    elif args.generate_test:
        # Generate test file
        hardware_platforms = args.hardware.split(",") if args.hardware else ["cpu", "cuda"]
        print(f"Generating test for {args.generate_test} with platforms: {hardware_platforms}")
        success, message = visualizer.generate_test(args.generate_test, hardware_platforms)
        print(message)
    elif args.export_report:
        # Export report
        print(f"Exporting {args.format} report to {args.output}...")
        visualizer.export_report(args.format, args.output)
    elif args.generate_dashboard:
        # Generate static dashboard HTML
        print(f"Generating static dashboard to {args.output}...")
        report_content = visualizer._generate_html_report()
        with open(args.output, 'w') as f:
            f.write(report_content)
        print(f"Dashboard saved to {args.output}")
    elif args.run_benchmark:
        # Run benchmark
        print(f"Running benchmark for {args.run_benchmark} on {args.benchmark_hardware}...")
        success, message, results = visualizer.run_benchmark(args.run_benchmark, args.benchmark_hardware)
        print(message)
        if success and results:
            print("Benchmark results:")
            print(json.dumps(results, indent=2))
    else:
        print("No command specified. Use --help for usage information.")
        print("\nTry one of these commands:")
        print("  --serve             Serve interactive dashboard")
        print("  --generate-test     Generate test for a specific model")
        print("  --export-report     Export benchmark report")
        print("  --run-benchmark     Run benchmark for a specific model")

if __name__ == "__main__":
    main()