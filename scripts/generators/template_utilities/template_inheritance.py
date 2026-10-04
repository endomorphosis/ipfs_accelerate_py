#!/usr/bin/env python3
"""
Template inheritance utilities.

This module provides utilities for working with template inheritance, including:
- Parent-child relationship management
- Template hierarchy navigation
- Default parent template creation
- Modality-based inheritance
"""

import logging
from typing import Dict, Any, List, Tuple, Optional, Set

from .placeholder_helpers import get_modality_for_model_type

logger = logging.getLogger(__name__)

# Model inheritance definitions
MODEL_INHERITANCE_MAP = {
    # Text models inherit from default_text template
    "bert": {"parent": "default_text", "modality": "text"},
    "t5": {"parent": "default_text", "modality": "text"},
    "llama": {"parent": "default_text", "modality": "text"},
    "gpt2": {"parent": "default_text", "modality": "text"},
    "qwen": {"parent": "default_text", "modality": "text"},
    
    # Vision models inherit from default_vision template
    "vit": {"parent": "default_vision", "modality": "vision"},
    "resnet": {"parent": "default_vision", "modality": "vision"},
    "detr": {"parent": "default_vision", "modality": "vision"},
    
    # Audio models inherit from default_audio template
    "whisper": {"parent": "default_audio", "modality": "audio"},
    "wav2vec2": {"parent": "default_audio", "modality": "audio"},
    "clap": {"parent": "default_audio", "modality": "audio"},
    
    # Multimodal models inherit from default_multimodal template
    "clip": {"parent": "default_multimodal", "modality": "multimodal"},
    "llava": {"parent": "default_multimodal", "modality": "multimodal"},
    "xclip": {"parent": "default_multimodal", "modality": "multimodal"}
}

def get_parent_for_model_type(model_type: str) -> Tuple[Optional[str], str]:
    """
    Get parent template and modality for a model type
    
    Args:
        model_type (str): The model type to get parent for
        
    Returns:
        Tuple[Optional[str], str]: Parent template name and modality
    """
    model_type_lower = model_type.lower()
    
    # Check for direct match in inheritance map
    if model_type_lower in MODEL_INHERITANCE_MAP:
        return (
            MODEL_INHERITANCE_MAP[model_type_lower]["parent"],
            MODEL_INHERITANCE_MAP[model_type_lower]["modality"]
        )
    
    # Check for partial matches (e.g., "bert-base" should match "bert")
    for mt, inheritance in MODEL_INHERITANCE_MAP.items():
        if mt in model_type_lower:
            return inheritance["parent"], inheritance["modality"]
    
    # If no match found, determine modality and use default parent
    modality = get_modality_for_model_type(model_type)
    
    if modality == "text":
        return "default_text", modality
    elif modality == "vision":
        return "default_vision", modality
    elif modality == "audio":
        return "default_audio", modality
    elif modality == "multimodal":
        return "default_multimodal", modality
    
    # Default to None if no match found
    return None, "unknown"

def get_inheritance_hierarchy(model_type: str) -> List[str]:
    """
    Get inheritance hierarchy for a model type
    
    Args:
        model_type (str): The model type to get hierarchy for
        
    Returns:
        List[str]: List of model types in inheritance hierarchy (child to parent)
    """
    hierarchy = [model_type]
    current = model_type
    
    # Build hierarchy by following parent chain
    while True:
        parent, _ = get_parent_for_model_type(current)
        if not parent or parent == current:
            break
        
        hierarchy.append(parent)
        current = parent
    
    return hierarchy

def get_default_parent_templates() -> Dict[str, Dict[str, str]]:
    """
    Get default parent templates for all modalities
    
    Returns:
        Dict[str, Dict[str, str]]: Dictionary of default parent templates by modality and type
    """
    return {
        "default_text": {
            "test": """#!/usr/bin/env python3
\"\"\"
Text model test for {model_name} with resource pool integration.
Generated from database template on {generated_at}
\"\"\"

import os
import unittest
import logging
from resource_pool import get_global_resource_pool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Test{normalized_name}(unittest.TestCase):
    \"\"\"Test {model_name} with resource pool integration.\"\"\"
    
    @classmethod
    def setUpClass(cls):
        \"\"\"Set up test environment.\"\"\"
        # Get global resource pool
        cls.pool = get_global_resource_pool()
        
        # Request dependencies
        cls.torch = cls.pool.get_resource("torch", constructor=lambda: __import__("torch"))
        cls.transformers = cls.pool.get_resource("transformers", constructor=lambda: __import__("transformers"))
        
        # Check if dependencies were loaded successfully:
        if cls.torch is None or cls.transformers is None:
            raise unittest.SkipTest("Required dependencies not available")
        
        # Set up device for hardware acceleration if available
        cls.device = "cpu"
        if {has_cuda} and cls.torch.cuda.is_available():
            cls.device = "cuda"
        elif {has_mps} and hasattr(cls.torch, "mps") and cls.torch.backends.mps.is_available():
            cls.device = "mps"
        logger.info(f"Using device: {cls.device}")
        
        # Load model and tokenizer
        try:
            cls.tokenizer = cls.transformers.AutoTokenizer.from_pretrained("{model_name}")
            cls.model = cls.transformers.AutoModel.from_pretrained("{model_name}")
            
            # Move model to appropriate device
            if cls.device != "cpu":
                cls.model = cls.model.to(cls.device)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise unittest.SkipTest(f"Failed to load model: {e}")
    
    def test_model_loaded(self):
        \"\"\"Test that model loaded successfully.\"\"\"
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.tokenizer)
    
    def test_inference(self):
        \"\"\"Test basic inference.\"\"\"
        # Prepare input
        text = "This is a test sentence for a text model."
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Move inputs to device if needed:
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        self.assertIn("last_hidden_state", outputs)
        
        # Log success
        logger.info(f"Successfully tested {model_name}")

if __name__ == "__main__":
    unittest.main()
""",
            "benchmark": """#!/usr/bin/env python3
\"\"\"
Text model benchmark for {model_name} with resource pool integration.
Generated from database template on {generated_at}
\"\"\"

import os
import time
import logging
import argparse
import statistics
from typing import Dict, List, Any
from resource_pool import get_global_resource_pool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    \"\"\"Parse command line arguments\"\"\"
    parser = argparse.ArgumentParser(description="Benchmark {model_name}")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--sequence-length", type=int, default=128, help="Sequence length")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu, cuda, mps)")
    parser.add_argument("--output-file", type=str, default=None, help="Output file for results")
    parser.add_argument("--db-only", action="store_true", help="Store results only in database")
    return parser.parse_args()

def benchmark_model(batch_size: int, sequence_length: int, iterations: int, warmup: int, device: str) -> Dict[str, Any]:
    \"\"\"Benchmark model performance\"\"\"
    # Get global resource pool
    pool = get_global_resource_pool()
    
    # Request dependencies
    torch = pool.get_resource("torch", constructor=lambda: __import__("torch"))
    transformers = pool.get_resource("transformers", constructor=lambda: __import__("transformers"))
    
    # Check if dependencies were loaded successfully
    if torch is None or transformers is None:
        logger.error("Required dependencies not available")
        return {
            "error": "Required dependencies not available",
            "success": False
        }
    
    # Determine device if not specified
    if device is None:
        device = "cpu"
        if {has_cuda} and torch.cuda.is_available():
            device = "cuda"
        elif {has_mps} and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
    
    logger.info(f"Using device: {device}")
    
    # Track metrics
    latencies = []
    memory_usage = []
    
    try:
        # Load model and tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained("{model_name}")
        model = transformers.AutoModel.from_pretrained("{model_name}")
        
        # Move model to device
        model = model.to(device)
        
        # Generate random input IDs (batch_size x sequence_length)
        # Use consistent seed for reproducibility
        import numpy as np
        np.random.seed(42)
        
        # Create sample text
        sample_text = ["This is a test sentence for benchmarking language models."] * batch_size
        
        # Encode sample text
        inputs = tokenizer(
            sample_text,
            padding="max_length",
            max_length=sequence_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Warm up
        logger.info(f"Running {warmup} warmup iterations...")
        for _ in range(warmup):
            with torch.no_grad():
                _ = model(**inputs)
                
        # Ensure CUDA synchronization if using CUDA
        if device == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark iterations
        logger.info(f"Running {iterations} benchmark iterations...")
        for i in range(iterations):
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Ensure CUDA synchronization if using CUDA
            if device == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to ms
            latencies.append(latency)
            
            # Measure memory usage
            if device == "cuda":
                memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # Convert to MB
            else:
                memory = 0  # Memory measurement not available for CPU/MPS
            memory_usage.append(memory)
            
            logger.info(f"Iteration {i+1}/{iterations}: Latency = {latency:.2f} ms, Memory = {memory:.2f} MB")
        
        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
        std_dev = statistics.stdev(latencies) if len(latencies) > 1 else 0
        
        avg_memory = statistics.mean(memory_usage) if memory_usage[0] > 0 else 0
        
        # Calculate throughput
        throughput = (batch_size * 1000) / avg_latency  # items per second
        
        return {
            "model_name": "{model_name}",
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "device": device,
            "iterations": iterations,
            "average_latency_ms": avg_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "p95_latency_ms": p95_latency,
            "latency_std_dev": std_dev,
            "average_memory_mb": avg_memory,
            "throughput_items_per_second": throughput,
            "success": True
        }
    except Exception as e:
        logger.error(f"Error benchmarking model: {e}")
        return {
            "error": str(e),
            "success": False
        }

def store_results_in_db(results: Dict[str, Any]) -> bool:
    \"\"\"Store benchmark results in DuckDB\"\"\"
    try:
        import duckdb
        import datetime
        import json
        import os
        
        # Default database path
        db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
        
        # Connect to database
        conn = duckdb.connect(db_path)
        
        # Get or create model_id
        model_id_result = conn.execute(
            "SELECT model_id FROM models WHERE model_name = ?",
            [results["model_name"]]
        ).fetchone()
        
        if model_id_result:
            model_id = model_id_result[0]
        else:
            # Insert new model
            conn.execute(
                "INSERT INTO models (model_name, model_type) VALUES (?, ?)",
                [results["model_name"], "{model_type}"]
            )
            model_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        
        # Get or create hardware_id
        hardware_platform = results["device"]
        hardware_id_result = conn.execute(
            "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = ?",
            [hardware_platform]
        ).fetchone()
        
        if hardware_id_result:
            hardware_id = hardware_id_result[0]
        else:
            # Insert new hardware platform
            conn.execute(
                "INSERT INTO hardware_platforms (hardware_type, description) VALUES (?, ?)",
                [hardware_platform, f"{hardware_platform.upper()} hardware platform"]
            )
            hardware_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        
        # Current timestamp
        timestamp = datetime.datetime.now()
        
        # Insert performance result
        conn.execute("""
        INSERT INTO performance_results 
        (model_id, hardware_id, batch_size, sequence_length, 
         average_latency_ms, min_latency_ms, max_latency_ms, 
         p95_latency_ms, latency_std_dev, average_memory_mb, 
         throughput_items_per_second, test_date, details)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            model_id, hardware_id, results["batch_size"], results["sequence_length"],
            results["average_latency_ms"], results["min_latency_ms"], results["max_latency_ms"],
            results["p95_latency_ms"], results["latency_std_dev"], results["average_memory_mb"],
            results["throughput_items_per_second"], timestamp, json.dumps(results)
        ])
        
        # Close connection
        conn.close()
        
        logger.info(f"Results stored in database {db_path}")
        return True
    except Exception as e:
        logger.error(f"Error storing results in database: {e}")
        return False

def main():
    \"\"\"Main function\"\"\"
    args = parse_args()
    
    logger.info(f"Benchmarking {model_name} with batch size {args.batch_size} and sequence length {args.sequence_length}")
    
    results = benchmark_model(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        iterations=args.iterations,
        warmup=args.warmup,
        device=args.device
    )
    
    if results["success"]:
        logger.info(f"Benchmark complete. Average latency: {results['average_latency_ms']:.2f} ms, "
                   f"Throughput: {results['throughput_items_per_second']:.2f} items/s")
        
        if args.db_only:
            store_results_in_db(results)
        elif args.output_file:
            # Save results to file
            import json
            with open(args.output_file, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output_file}")
        else:
            # Print full results
            print("\\nResults:")
            print("-" * 50)
            for key, value in results.items():
                if key != "success":
                    print(f"{key}: {value}")
    else:
        logger.error(f"Benchmark failed: {results.get('error', 'Unknown error')}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
"""
        },
        "default_vision": {
            "test": """#!/usr/bin/env python3
\"\"\"
Vision model test for {model_name} with resource pool integration.
Generated from database template on {generated_at}
\"\"\"

import os
import unittest
import logging
import numpy as np
from PIL import Image
from resource_pool import get_global_resource_pool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Test{normalized_name}(unittest.TestCase):
    \"\"\"Test {model_name} with resource pool integration.\"\"\"
    
    @classmethod
    def setUpClass(cls):
        \"\"\"Set up test environment.\"\"\"
        # Get global resource pool
        cls.pool = get_global_resource_pool()
        
        # Request dependencies
        cls.torch = cls.pool.get_resource("torch", constructor=lambda: __import__("torch"))
        cls.transformers = cls.pool.get_resource("transformers", constructor=lambda: __import__("transformers"))
        
        # Check if dependencies were loaded successfully:
        if cls.torch is None or cls.transformers is None:
            raise unittest.SkipTest("Required dependencies not available")
        
        # Set up device for hardware acceleration if available
        cls.device = "cpu"
        if {has_cuda} and cls.torch.cuda.is_available():
            cls.device = "cuda"
        elif {has_mps} and hasattr(cls.torch, "mps") and cls.torch.backends.mps.is_available():
            cls.device = "mps"
        logger.info(f"Using device: {cls.device}")
        
        # Create a test image if it doesn't exist
        cls.test_image_path = "test.jpg"
        if not os.path.exists(cls.test_image_path):
            # Create a simple test image (100x100 black square)
            img = Image.new('RGB', (100, 100), color='black')
            img.save(cls.test_image_path)
            logger.info(f"Created test image at {cls.test_image_path}")
        
        # Load model and feature extractor/processor
        try:
            cls.processor = cls.transformers.AutoFeatureExtractor.from_pretrained("{model_name}")
            cls.model = cls.transformers.AutoModel.from_pretrained("{model_name}")
            
            # Move model to appropriate device
            if cls.device != "cpu":
                cls.model = cls.model.to(cls.device)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise unittest.SkipTest(f"Failed to load model: {e}")
    
    def test_model_loaded(self):
        \"\"\"Test that model loaded successfully.\"\"\"
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.processor)
    
    def test_inference(self):
        \"\"\"Test basic inference.\"\"\"
        # Load and process image
        image = Image.open(self.test_image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Move inputs to device if needed:
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        self.assertIn("last_hidden_state", outputs)
        
        # Log success
        logger.info(f"Successfully tested {model_name}")

if __name__ == "__main__":
    unittest.main()
"""
        },
        "default_audio": {
            "test": """#!/usr/bin/env python3
\"\"\"
Audio model test for {model_name} with resource pool integration.
Generated from database template on {generated_at}
\"\"\"

import os
import unittest
import logging
import numpy as np
from resource_pool import get_global_resource_pool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Test{normalized_name}(unittest.TestCase):
    \"\"\"Test {model_name} with resource pool integration.\"\"\"
    
    @classmethod
    def setUpClass(cls):
        \"\"\"Set up test environment.\"\"\"
        # Get global resource pool
        cls.pool = get_global_resource_pool()
        
        # Request dependencies
        cls.torch = cls.pool.get_resource("torch", constructor=lambda: __import__("torch"))
        cls.transformers = cls.pool.get_resource("transformers", constructor=lambda: __import__("transformers"))
        
        # Check if dependencies were loaded successfully:
        if cls.torch is None or cls.transformers is None:
            raise unittest.SkipTest("Required dependencies not available")
        
        # Set up device for hardware acceleration if available
        cls.device = "cpu"
        if {has_cuda} and cls.torch.cuda.is_available():
            cls.device = "cuda"
        elif {has_mps} and hasattr(cls.torch, "mps") and cls.torch.backends.mps.is_available():
            cls.device = "mps"
        logger.info(f"Using device: {cls.device}")
        
        # Create a test audio array or use existing file
        cls.test_audio_path = "test.mp3"
        cls.sampling_rate = 16000
        
        if not os.path.exists(cls.test_audio_path):
            # Create a simple silence audio array (1 second)
            logger.info(f"No test audio found, using synthetic array")
            cls.audio_array = np.zeros(cls.sampling_rate)  # 1 second of silence
        else:
            try:
                # Try to load audio file if available
                import librosa
                cls.audio_array, cls.sampling_rate = librosa.load(cls.test_audio_path, sr=cls.sampling_rate)
                logger.info(f"Loaded test audio from {cls.test_audio_path}")
            except (ImportError, Exception) as e:
                logger.warning(f"Could not load audio file: {e}")
                cls.audio_array = np.zeros(cls.sampling_rate)  # 1 second of silence
        
        # Load model and processor
        try:
            cls.processor = cls.transformers.AutoProcessor.from_pretrained("{model_name}")
            cls.model = cls.transformers.AutoModel.from_pretrained("{model_name}")
            
            # Move model to appropriate device
            if cls.device != "cpu":
                cls.model = cls.model.to(cls.device)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise unittest.SkipTest(f"Failed to load model: {e}")
    
    def test_model_loaded(self):
        \"\"\"Test that model loaded successfully.\"\"\"
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.processor)
    
    def test_inference(self):
        \"\"\"Test basic inference.\"\"\"
        # Process audio input
        inputs = self.processor(
            self.audio_array, 
            sampling_rate=self.sampling_rate, 
            return_tensors="pt"
        )
        
        # Move inputs to device if needed:
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        
        # Log success
        logger.info(f"Successfully tested {model_name}")

if __name__ == "__main__":
    unittest.main()
"""
        },
        "default_multimodal": {
            "test": """#!/usr/bin/env python3
\"\"\"
Multimodal model test for {model_name} with resource pool integration.
Generated from database template on {generated_at}
\"\"\"

import os
import unittest
import logging
import numpy as np
from PIL import Image
from resource_pool import get_global_resource_pool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Test{normalized_name}(unittest.TestCase):
    \"\"\"Test {model_name} with resource pool integration.\"\"\"
    
    @classmethod
    def setUpClass(cls):
        \"\"\"Set up test environment.\"\"\"
        # Get global resource pool
        cls.pool = get_global_resource_pool()
        
        # Request dependencies
        cls.torch = cls.pool.get_resource("torch", constructor=lambda: __import__("torch"))
        cls.transformers = cls.pool.get_resource("transformers", constructor=lambda: __import__("transformers"))
        
        # Check if dependencies were loaded successfully:
        if cls.torch is None or cls.transformers is None:
            raise unittest.SkipTest("Required dependencies not available")
        
        # Set up device for hardware acceleration if available
        cls.device = "cpu"
        if {has_cuda} and cls.torch.cuda.is_available():
            cls.device = "cuda"
        elif {has_mps} and hasattr(cls.torch, "mps") and cls.torch.backends.mps.is_available():
            cls.device = "mps"
        logger.info(f"Using device: {cls.device}")
        
        # Create a test image if it doesn't exist
        cls.test_image_path = "test.jpg"
        if not os.path.exists(cls.test_image_path):
            # Create a simple test image (100x100 black square)
            img = Image.new('RGB', (100, 100), color='black')
            img.save(cls.test_image_path)
            logger.info(f"Created test image at {cls.test_image_path}")
        
        # Test text prompt
        cls.test_text = "What's in this image?"
        
        # Load model and processor
        try:
            cls.processor = cls.transformers.AutoProcessor.from_pretrained("{model_name}")
            cls.model = cls.transformers.AutoModel.from_pretrained("{model_name}")
            
            # Move model to appropriate device
            if cls.device != "cpu":
                cls.model = cls.model.to(cls.device)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise unittest.SkipTest(f"Failed to load model: {e}")
    
    def test_model_loaded(self):
        \"\"\"Test that model loaded successfully.\"\"\"
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.processor)
    
    def test_inference(self):
        \"\"\"Test basic inference.\"\"\"
        # Load image
        image = Image.open(self.test_image_path)
        
        # Process inputs
        inputs = self.processor(
            text=self.test_text,
            images=image, 
            return_tensors="pt"
        )
        
        # Move inputs to device if needed:
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        
        # Log success
        logger.info(f"Successfully tested {model_name}")

if __name__ == "__main__":
    unittest.main()
"""
        }
    }

def get_template_with_inheritance(model_type: str, template_type: str, templates_db: Dict[str, Dict[str, str]]) -> Tuple[Optional[str], Optional[str]]:
    """
    Get a template with inheritance
    
    Args:
        model_type (str): The model type to get template for
        template_type (str): The template type to get
        templates_db (Dict[str, Dict[str, str]]): Database of templates
        
    Returns:
        Tuple[Optional[str], Optional[str]]: Template content and parent template name
    """
    # Try to get template for the specific model type
    if model_type in templates_db and template_type in templates_db[model_type]:
        return templates_db[model_type][template_type], None
    
    # If not found, get parent template
    parent, _ = get_parent_for_model_type(model_type)
    
    if parent and parent in templates_db and template_type in templates_db[parent]:
        return templates_db[parent][template_type], parent
    
    # If no template found in parent either, return None
    return None, None

def merge_template_with_parent(child_template: str, parent_template: str) -> str:
    """
    Merge a child template with its parent template
    
    Args:
        child_template (str): The child template content
        parent_template (str): The parent template content
        
    Returns:
        str: Merged template content
    """
    # In this simple implementation, we just return the child template
    # A more complex implementation could perform selective inheritance
    return child_template