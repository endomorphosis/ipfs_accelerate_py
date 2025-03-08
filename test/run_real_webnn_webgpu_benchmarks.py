#!/usr/bin/env python3
"""
Run Real WebNN/WebGPU Benchmarks

This script runs benchmarks using real WebNN and WebGPU implementations in browsers,
with enhanced WebSocket communication and resource pool integration.

Usage:
    # Run WebGPU benchmarks with Chrome
    python run_real_webnn_webgpu_benchmarks.py --webgpu --chrome
    
    # Run WebNN benchmarks with Edge
    python run_real_webnn_webgpu_benchmarks.py --webnn --edge
    
    # Run audio model benchmarks with Firefox
    python run_real_webnn_webgpu_benchmarks.py --audio --firefox
    
    # Benchmark with quantization (8-bit)
    python run_real_webnn_webgpu_benchmarks.py --text --bits 8
    
    # Benchmark with mixed precision
    python run_real_webnn_webgpu_benchmarks.py --text --bits 4 --mixed-precision
    
    # Run comprehensive benchmarks across multiple models
    python run_real_webnn_webgpu_benchmarks.py --comprehensive
"""

import os
import sys
import json
import time
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent))

# Import required modules
try:
    from fixed_web_platform.resource_pool_bridge import ResourcePoolBridgeIntegration
    RESOURCE_POOL_AVAILABLE = True
except ImportError:
    logger.error("ResourcePoolBridge not available")
    RESOURCE_POOL_AVAILABLE = False

# Try to import DuckDB for database storage
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    logger.warning("DuckDB not available, database storage disabled")
    DUCKDB_AVAILABLE = False

class WebNNWebGPUBenchmarker:
    """Run benchmarks using real WebNN and WebGPU implementations."""
    
    def __init__(self, args):
        """Initialize benchmarker with command line arguments."""
        self.args = args
        self.integration = None
        self.results = []
        self.db_connection = None
        
        # Set environment variables if needed
        if args.compute_shaders:
            os.environ["WEBGPU_COMPUTE_SHADERS_ENABLED"] = "1"
        
        if args.shader_precompile:
            os.environ["WEBGPU_SHADER_PRECOMPILE_ENABLED"] = "1"
        
        if args.parallel_loading:
            os.environ["WEB_PARALLEL_LOADING_ENABLED"] = "1"
        
        # Connect to database if path specified
        if args.db_path and DUCKDB_AVAILABLE:
            try:
                self.db_connection = duckdb.connect(args.db_path)
                logger.info(f"Connected to database: {args.db_path}")
                self._ensure_db_schema()
            except Exception as e:
                logger.error(f"Failed to connect to database: {e}")
                self.db_connection = None
    
    def _ensure_db_schema(self):
        """Ensure the database has the required schema."""
        if not self.db_connection:
            return
        
        try:
            # Check if webnn_webgpu_benchmark_results table exists
            table_exists = self.db_connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='webnn_webgpu_benchmark_results'"
            ).fetchone()
            
            if not table_exists:
                # Create table if it doesn't exist
                self.db_connection.execute("""
                CREATE TABLE webnn_webgpu_benchmark_results (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP,
                    model_name VARCHAR,
                    model_type VARCHAR,
                    platform VARCHAR,
                    browser VARCHAR,
                    is_real_implementation BOOLEAN,
                    is_simulation BOOLEAN,
                    precision INTEGER,
                    mixed_precision BOOLEAN,
                    compute_shaders BOOLEAN,
                    shader_precompile BOOLEAN,
                    parallel_loading BOOLEAN,
                    batch_size INTEGER,
                    sequence_length INTEGER,
                    latency_ms FLOAT,
                    throughput_items_per_sec FLOAT,
                    memory_usage_mb FLOAT,
                    adapter_info VARCHAR,
                    browser_info VARCHAR,
                    details JSON
                )
                """)
                logger.info("Created webnn_webgpu_benchmark_results table in database")
        except Exception as e:
            logger.error(f"Failed to ensure database schema: {e}")
    
    async def initialize(self):
        """Initialize resource pool integration."""
        if not RESOURCE_POOL_AVAILABLE:
            logger.error("Cannot initialize: ResourcePoolBridge not available")
            return False
        
        try:
            # Create ResourcePoolBridgeIntegration instance
            self.integration = ResourcePoolBridgeIntegration(
                max_connections=self.args.max_connections,
                enable_gpu=True,
                enable_cpu=True,
                headless=not self.args.visible,
                browser_preferences={
                    'audio': 'firefox',  # Firefox has better compute shader performance for audio
                    'vision': 'chrome',  # Chrome has good WebGPU support for vision models
                    'text_embedding': 'edge'  # Edge has excellent WebNN support for text embeddings
                },
                adaptive_scaling=True
            )
            
            # Initialize integration
            self.integration.initialize()
            logger.info("ResourcePoolBridgeIntegration initialized successfully")
            
            return True
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def benchmark_model(self, model_type, model_name, platform, batch_sizes=None):
        """Benchmark a model using the resource pool integration."""
        if not self.integration:
            logger.error("Cannot benchmark model: integration not initialized")
            return []
        
        if not batch_sizes:
            batch_sizes = [1, 2, 4, 8] if not self.args.quick_test else [1]
        
        benchmark_results = []
        
        try:
            logger.info(f"Benchmarking model: {model_name} ({model_type}) on {platform}")
            
            # Configure hardware preferences
            hardware_preferences = {
                'priority_list': [platform, 'cpu'],
                'model_family': model_type
            }
            
            # Process each batch size
            for batch_size in batch_sizes:
                logger.info(f"Benchmarking with batch size {batch_size}")
                
                # Get model from resource pool
                model = self.integration.get_model(
                    model_type=model_type,
                    model_name=model_name,
                    hardware_preferences=hardware_preferences
                )
                
                if not model:
                    logger.error(f"Failed to get model: {model_name}")
                    continue
                
                # Prepare test input based on model type and batch size
                test_input = self._create_test_input(model_type, batch_size)
                if not test_input:
                    logger.error(f"Failed to create test input for {model_type}")
                    continue
                
                # Run warm-up inference
                logger.info("Running warm-up inference")
                _ = model(test_input)
                
                # Run benchmark
                logger.info(f"Running benchmark with batch size {batch_size}")
                num_iterations = 5 if not self.args.quick_test else 2
                latencies = []
                memory_usages = []
                
                for i in range(num_iterations):
                    start_time = time.time()
                    result = model(test_input)
                    execution_time = time.time() - start_time
                    
                    # Track metrics
                    latencies.append(execution_time * 1000)  # Convert to ms
                    
                    # Get memory usage if available
                    if isinstance(result, dict) and 'performance_metrics' in result:
                        metrics = result.get('performance_metrics', {})
                        if 'memory_usage_mb' in metrics:
                            memory_usages.append(metrics['memory_usage_mb'])
                    
                    logger.info(f"Iteration {i+1}/{num_iterations}: {execution_time*1000:.2f} ms")
                
                # Calculate metrics
                avg_latency = sum(latencies) / len(latencies) if latencies else 0
                throughput = (batch_size * 1000) / avg_latency if avg_latency > 0 else 0
                avg_memory = sum(memory_usages) / len(memory_usages) if memory_usages else 0
                
                # Create result object
                platform_type = platform
                browser_name = self.args.browser or self._get_browser_from_model(model)
                is_real = isinstance(result, dict) and result.get('is_real_implementation', False)
                
                result_obj = {
                    'model_name': model_name,
                    'model_type': model_type,
                    'platform': platform_type,
                    'browser': browser_name,
                    'timestamp': datetime.now().isoformat(),
                    'batch_size': batch_size,
                    'is_real_implementation': is_real,
                    'is_simulation': not is_real,
                    'precision': self.args.bits,
                    'mixed_precision': self.args.mixed_precision,
                    'compute_shaders': self.args.compute_shaders,
                    'shader_precompile': self.args.shader_precompile,
                    'parallel_loading': self.args.parallel_loading,
                    'metrics': {
                        'latency_ms': avg_latency,
                        'throughput_items_per_sec': throughput,
                        'memory_usage_mb': avg_memory
                    }
                }
                
                # Add to results
                benchmark_results.append(result_obj)
                self.results.append(result_obj)
                
                # Store in database
                if self.db_connection:
                    self._store_result_in_db(result_obj)
                
                logger.info(f"Benchmark completed for {model_name} with batch size {batch_size}:")
                logger.info(f"  - Avg Latency: {avg_latency:.2f} ms")
                logger.info(f"  - Throughput: {throughput:.2f} items/sec")
                logger.info(f"  - Memory Usage: {avg_memory:.2f} MB")
                logger.info(f"  - Real Implementation: {is_real}")
            
            return benchmark_results
            
        except Exception as e:
            logger.error(f"Error benchmarking model {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _create_test_input(self, model_type, batch_size):
        """Create test input for a model type and batch size."""
        try:
            if model_type == 'text_embedding':
                # Create input for text embedding models (BERT, etc.)
                input_ids = [[101, 2023, 2003, 1037, 3231, 102] for _ in range(batch_size)]
                attention_mask = [[1, 1, 1, 1, 1, 1] for _ in range(batch_size)]
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                }
            elif model_type == 'vision':
                # Create input for vision models (ViT, etc.)
                # 224x224 images with 3 channels (RGB)
                pixel_values = [
                    [[[0.5 for _ in range(3)] for _ in range(224)] for _ in range(224)]
                    for _ in range(batch_size)
                ]
                return {'pixel_values': pixel_values}
            elif model_type == 'audio':
                # Create input for audio models (Whisper, etc.)
                # 80 mel features, 3000 time steps
                input_features = [
                    [[[0.1 for _ in range(80)] for _ in range(3000)]]
                    for _ in range(batch_size)
                ]
                return {'input_features': input_features}
            elif model_type == 'multimodal':
                # Create input for multimodal models (CLIP, etc.)
                pixel_values = [
                    [[[0.5 for _ in range(3)] for _ in range(224)] for _ in range(224)]
                    for _ in range(batch_size)
                ]
                input_ids = [[101, 2023, 2003, 1037, 3231, 102] for _ in range(batch_size)]
                attention_mask = [[1, 1, 1, 1, 1, 1] for _ in range(batch_size)]
                return {
                    'pixel_values': pixel_values,
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                }
            else:
                # Generic input for other models
                return {
                    'inputs': [[0.0 for _ in range(10)] for _ in range(batch_size)]
                }
        except Exception as e:
            logger.error(f"Error creating test input: {e}")
            return None
    
    def _get_browser_from_model(self, model):
        """Get browser name from model if available."""
        if hasattr(model, 'device'):
            return getattr(model, 'browser', 'unknown')
        return self.args.browser or 'unknown'
    
    def _store_result_in_db(self, result):
        """Store benchmark result in database."""
        if not self.db_connection:
            return
        
        try:
            # Insert result into database
            self.db_connection.execute("""
            INSERT INTO webnn_webgpu_benchmark_results (
                timestamp,
                model_name,
                model_type,
                platform,
                browser,
                is_real_implementation,
                is_simulation,
                precision,
                mixed_precision,
                compute_shaders,
                shader_precompile,
                parallel_loading,
                batch_size,
                latency_ms,
                throughput_items_per_sec,
                memory_usage_mb,
                details
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """, [
                datetime.now(),
                result["model_name"],
                result["model_type"],
                result["platform"],
                result["browser"],
                result["is_real_implementation"],
                result["is_simulation"],
                result["precision"],
                result["mixed_precision"],
                result["compute_shaders"],
                result["shader_precompile"],
                result["parallel_loading"],
                result["batch_size"],
                result["metrics"]["latency_ms"],
                result["metrics"]["throughput_items_per_sec"],
                result["metrics"]["memory_usage_mb"],
                json.dumps(result)
            ])
            
            logger.info(f"Stored benchmark result for {result['model_name']} in database")
        except Exception as e:
            logger.error(f"Failed to store result in database: {e}")
    
    async def run_benchmarks(self):
        """Run benchmarks based on command line arguments."""
        if not self.integration:
            logger.error("Cannot run benchmarks: integration not initialized")
            return []
        
        # Determine which platforms to benchmark
        platforms = []
        if self.args.webnn:
            platforms.append('webnn')
        if self.args.webgpu:
            platforms.append('webgpu')
        if not platforms:  # Default to WebGPU if none specified
            platforms.append('webgpu')
        
        # Determine which models to benchmark
        models_to_benchmark = []
        
        # Text models
        if self.args.text or self.args.comprehensive:
            models_to_benchmark.append(('text_embedding', 'bert-base-uncased'))
            models_to_benchmark.append(('text_embedding', 'prajjwal1/bert-tiny'))
        
        # Vision models
        if self.args.vision or self.args.comprehensive:
            models_to_benchmark.append(('vision', 'google/vit-base-patch16-224'))
        
        # Audio models
        if self.args.audio or self.args.comprehensive:
            models_to_benchmark.append(('audio', 'openai/whisper-tiny'))
        
        # Use custom models if specified
        if self.args.models:
            models_to_benchmark = []
            model_names = self.args.models.split(',')
            
            for model_name in model_names:
                if 'bert' in model_name.lower() or 't5' in model_name.lower():
                    models_to_benchmark.append(('text_embedding', model_name))
                elif 'vit' in model_name.lower() or 'resnet' in model_name.lower():
                    models_to_benchmark.append(('vision', model_name))
                elif 'whisper' in model_name.lower() or 'wav2vec' in model_name.lower():
                    models_to_benchmark.append(('audio', model_name))
                elif 'clip' in model_name.lower():
                    models_to_benchmark.append(('multimodal', model_name))
                else:
                    models_to_benchmark.append(('text', model_name))
        
        # Run benchmarks
        all_results = []
        for platform in platforms:
            for model_type, model_name in models_to_benchmark:
                batch_sizes = [1, 2, 4, 8] if not self.args.quick_test else [1]
                if self.args.batch_sizes:
                    batch_sizes = [int(b) for b in self.args.batch_sizes.split(',')]
                
                results = await self.benchmark_model(model_type, model_name, platform, batch_sizes)
                all_results.extend(results)
        
        return all_results
    
    async def close(self):
        """Close resources."""
        if self.integration:
            self.integration.close()
            logger.info("ResourcePoolBridgeIntegration closed")
        
        if self.db_connection:
            self.db_connection.close()
            logger.info("Database connection closed")
    
    def save_results(self):
        """Save results to file."""
        if not self.results:
            logger.warning("No results to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_filename = f"webnn_webgpu_benchmark_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {json_filename}")
        
        # Save Markdown report
        md_filename = f"webnn_webgpu_benchmark_{timestamp}.md"
        self.generate_markdown_report(md_filename)
    
    def generate_markdown_report(self, filename):
        """Generate markdown benchmark report."""
        if not self.results:
            logger.warning("No results to generate report")
            return
        
        with open(filename, 'w') as f:
            f.write("# WebNN/WebGPU Benchmark Results\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Add implementation status summary
            f.write("## Implementation Status\n\n")
            
            # Count real and simulated implementations
            real_count = sum(1 for r in self.results if r.get("is_real_implementation", False))
            sim_count = len(self.results) - real_count
            
            f.write(f"- Total benchmarks: {len(self.results)}\n")
            f.write(f"- Real hardware implementations: {real_count}\n")
            f.write(f"- Simulation implementations: {sim_count}\n\n")
            
            # Configuration summary
            f.write("## Configuration\n\n")
            f.write(f"- Precision: {self.args.bits}-bit")
            if self.args.mixed_precision:
                f.write(" (mixed precision)")
            f.write("\n")
            
            if self.args.compute_shaders:
                f.write("- Compute Shaders: Enabled\n")
            if self.args.shader_precompile:
                f.write("- Shader Precompilation: Enabled\n")
            if self.args.parallel_loading:
                f.write("- Parallel Loading: Enabled\n")
            f.write("\n")
            
            # Results by platform
            platforms = set(r["platform"] for r in self.results)
            for platform in platforms:
                platform_results = [r for r in self.results if r["platform"] == platform]
                f.write(f"## {platform.upper()} Results\n\n")
                
                # Group by model
                models = {}
                for result in platform_results:
                    model_name = result["model_name"]
                    if model_name not in models:
                        models[model_name] = []
                    models[model_name].append(result)
                
                # Generate table
                f.write("| Model | Browser | Batch Size | Latency (ms) | Throughput (items/sec) | Memory (MB) | Real Hardware |\n")
                f.write("|-------|---------|------------|--------------|------------------------|-------------|---------------|\n")
                
                for model_name, model_results in sorted(models.items()):
                    # Sort by batch size
                    model_results.sort(key=lambda r: r.get("batch_size", 0))
                    
                    for result in model_results:
                        batch_size = result.get("batch_size", 1)
                        browser = result.get("browser", "unknown")
                        real_hw = "✅" if result.get("is_real_implementation", False) else "❌"
                        metrics = result.get("metrics", {})
                        
                        latency = metrics.get("latency_ms", 0)
                        throughput = metrics.get("throughput_items_per_sec", 0)
                        memory = metrics.get("memory_usage_mb", 0)
                        
                        f.write(f"| {model_name} | {browser} | {batch_size} | {latency:.2f} | {throughput:.2f} | {memory:.2f} | {real_hw} |\n")
                
                f.write("\n")
            
            # Performance comparison
            f.write("## Performance Comparison\n\n")
            
            # Group by model and batch size
            model_batch_results = {}
            for result in self.results:
                model = result["model_name"]
                batch = result.get("batch_size", 1)
                platform = result["platform"]
                key = f"{model}_{batch}"
                
                if key not in model_batch_results:
                    model_batch_results[key] = {}
                
                model_batch_results[key][platform] = result
            
            # Generate comparison table
            if len(platforms) > 1:
                platforms_list = sorted(platforms)
                
                f.write("| Model | Batch Size | ")
                for platform in platforms_list:
                    f.write(f"{platform.upper()} Latency (ms) | {platform.upper()} Throughput | ")
                f.write("Comparison |\n")
                
                f.write("|-------|------------|")
                for _ in platforms_list:
                    f.write("-----------------|-----------------|")
                f.write("-----------|\n")
                
                for key, platform_results in sorted(model_batch_results.items()):
                    model, batch = key.rsplit("_", 1)
                    batch = int(batch)
                    
                    f.write(f"| {model} | {batch} | ")
                    
                    # Add latency and throughput for each platform
                    latencies = []
                    throughputs = []
                    
                    for platform in platforms_list:
                        if platform in platform_results:
                            result = platform_results[platform]
                            metrics = result.get("metrics", {})
                            latency = metrics.get("latency_ms", 0)
                            throughput = metrics.get("throughput_items_per_sec", 0)
                            
                            f.write(f"{latency:.2f} | {throughput:.2f} | ")
                            
                            latencies.append((platform, latency))
                            throughputs.append((platform, throughput))
                        else:
                            f.write("N/A | N/A | ")
                    
                    # Add comparison
                    if len(latencies) > 1:
                        # Sort by latency (lower is better)
                        latencies.sort(key=lambda x: x[1])
                        best_platform = latencies[0][0]
                        f.write(f"Best: {best_platform.upper()}")
                    else:
                        f.write("N/A")
                    
                    f.write(" |\n")
            
            # Browser comparison
            browsers = set(r.get("browser", "unknown") for r in self.results)
            if len(browsers) > 1:
                f.write("\n## Browser Comparison\n\n")
                
                # Group by model and platform
                model_platform_results = {}
                for result in self.results:
                    model = result["model_name"]
                    batch = result.get("batch_size", 1)
                    platform = result["platform"]
                    browser = result.get("browser", "unknown")
                    
                    if batch != 1:  # Only compare batch size 1 for simplicity
                        continue
                    
                    key = f"{model}_{platform}"
                    
                    if key not in model_platform_results:
                        model_platform_results[key] = {}
                    
                    model_platform_results[key][browser] = result
                
                # Generate browser comparison table
                browsers_list = sorted(browsers)
                
                f.write("| Model | Platform | ")
                for browser in browsers_list:
                    f.write(f"{browser} Latency (ms) | ")
                f.write("Best Browser |\n")
                
                f.write("|-------|----------|")
                for _ in browsers_list:
                    f.write("-----------------|")
                f.write("-------------|\n")
                
                for key, browser_results in sorted(model_platform_results.items()):
                    model, platform = key.rsplit("_", 1)
                    
                    f.write(f"| {model} | {platform} | ")
                    
                    # Add latency for each browser
                    latencies = []
                    
                    for browser in browsers_list:
                        if browser in browser_results:
                            result = browser_results[browser]
                            metrics = result.get("metrics", {})
                            latency = metrics.get("latency_ms", 0)
                            
                            f.write(f"{latency:.2f} | ")
                            
                            latencies.append((browser, latency))
                        else:
                            f.write("N/A | ")
                    
                    # Add best browser
                    if len(latencies) > 1:
                        # Sort by latency (lower is better)
                        latencies.sort(key=lambda x: x[1])
                        best_browser = latencies[0][0]
                        f.write(f"{best_browser}")
                    else:
                        f.write("N/A")
                    
                    f.write(" |\n")
            
            # Add specific recommendations
            f.write("\n## Recommendations\n\n")
            
            # Audio models with Firefox
            audio_models = [r for r in self.results if r["model_type"] == "audio"]
            if audio_models:
                firefox_audio = [r for r in audio_models if r.get("browser") == "firefox"]
                other_audio = [r for r in audio_models if r.get("browser") != "firefox"]
                
                if firefox_audio and other_audio:
                    # Calculate average latency
                    firefox_latency = sum(r["metrics"]["latency_ms"] for r in firefox_audio) / len(firefox_audio)
                    other_latency = sum(r["metrics"]["latency_ms"] for r in other_audio) / len(other_audio)
                    
                    if firefox_latency < other_latency:
                        improvement = ((other_latency - firefox_latency) / other_latency) * 100
                        f.write(f"- **Audio Models**: Use Firefox with compute shaders for {improvement:.1f}% better performance\n")
            
            # Text models with Edge WebNN
            text_models = [r for r in self.results if r["model_type"] == "text_embedding"]
            if text_models:
                edge_webnn = [r for r in text_models if r.get("browser") == "edge" and r["platform"] == "webnn"]
                other_text = [r for r in text_models if r.get("browser") != "edge" or r["platform"] != "webnn"]
                
                if edge_webnn and other_text:
                    # Calculate average latency
                    edge_latency = sum(r["metrics"]["latency_ms"] for r in edge_webnn) / len(edge_webnn)
                    other_latency = sum(r["metrics"]["latency_ms"] for r in other_text) / len(other_text)
                    
                    if edge_latency < other_latency:
                        improvement = ((other_latency - edge_latency) / other_latency) * 100
                        f.write(f"- **Text Models**: Use Edge with WebNN for {improvement:.1f}% better performance\n")
            
            # Precision recommendations
            if self.args.bits <= 8:
                f.write(f"- **Quantization**: {self.args.bits}-bit precision")
                if self.args.mixed_precision:
                    f.write(" with mixed precision")
                f.write(" provides good performance/accuracy trade-off\n")
            
            logger.info(f"Markdown report saved to {filename}")

async def main_async():
    """Main async function."""
    parser = argparse.ArgumentParser(description="Run Real WebNN/WebGPU Benchmarks")
    
    # Model type options
    parser.add_argument("--text", action="store_true",
                      help="Benchmark text models (BERT, T5)")
    parser.add_argument("--vision", action="store_true",
                      help="Benchmark vision models (ViT, ResNet)")
    parser.add_argument("--audio", action="store_true",
                      help="Benchmark audio models (Whisper, Wav2Vec2)")
    parser.add_argument("--comprehensive", action="store_true",
                      help="Run comprehensive benchmarks across all model types")
    
    # Model selection
    parser.add_argument("--models", type=str,
                      help="Comma-separated list of models to benchmark")
    
    # Platform options
    parser.add_argument("--webnn", action="store_true",
                      help="Benchmark WebNN platform")
    parser.add_argument("--webgpu", action="store_true",
                      help="Benchmark WebGPU platform")
    
    # Browser options
    parser.add_argument("--chrome", action="store_true",
                      help="Use Chrome browser")
    parser.add_argument("--firefox", action="store_true",
                      help="Use Firefox browser")
    parser.add_argument("--edge", action="store_true",
                      help="Use Edge browser")
    
    # Batch size options
    parser.add_argument("--batch-sizes", type=str,
                      help="Comma-separated list of batch sizes")
    
    # Precision options
    parser.add_argument("--bits", type=int, choices=[4, 8, 16, 32], default=8,
                      help="Precision level to test (bit width)")
    parser.add_argument("--mixed-precision", action="store_true",
                      help="Use mixed precision (higher precision for critical layers)")
    
    # Optimization options
    parser.add_argument("--compute-shaders", action="store_true",
                      help="Enable compute shader optimization for audio models")
    parser.add_argument("--shader-precompile", action="store_true",
                      help="Enable shader precompilation for faster startup")
    parser.add_argument("--parallel-loading", action="store_true",
                      help="Enable parallel model loading for multimodal models")
    
    # Test configuration
    parser.add_argument("--quick-test", action="store_true",
                      help="Run a quick test with fewer iterations and batch sizes")
    parser.add_argument("--visible", action="store_true",
                      help="Run browsers in visible mode (not headless)")
    
    # Resource pool options
    parser.add_argument("--max-connections", type=int, default=4,
                      help="Maximum number of browser connections")
    
    # Output options
    parser.add_argument("--db-path", type=str,
                      help="Path to DuckDB database file")
    parser.add_argument("--output-format", type=str, choices=["json", "markdown", "html", "all"],
                      default="all", help="Output format for results")
    
    args = parser.parse_args()
    
    # Set browser from arguments
    if args.chrome:
        args.browser = "chrome"
    elif args.firefox:
        args.browser = "firefox"
    elif args.edge:
        args.browser = "edge"
    else:
        args.browser = None
    
    # Create benchmarker
    benchmarker = WebNNWebGPUBenchmarker(args)
    
    try:
        # Initialize benchmarker
        if not await benchmarker.initialize():
            logger.error("Failed to initialize benchmarker")
            return 1
        
        # Run benchmarks
        await benchmarker.run_benchmarks()
        
        # Save results
        benchmarker.save_results()
        
        # Close benchmarker
        await benchmarker.close()
        
        return 0
    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
        
        # Ensure benchmarker is closed
        await benchmarker.close()
        
        return 1

def main():
    """Main entry point."""
    try:
        return asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130

if __name__ == "__main__":
    sys.exit(main())