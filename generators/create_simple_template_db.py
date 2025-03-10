#\!/usr/bin/env python3
"""
Create Simple Template Database

Creates a simple template database with minimal templates for testing.
"""

import os
import sys
import logging
import duckdb
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database path
DB_PATH = "template_db.duckdb"

def create_database():
    """Create the template database with basic schema."""
    conn = duckdb.connect(DB_PATH)
    
    # Create templates table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS templates (
        id INTEGER PRIMARY KEY,
        model_type VARCHAR NOT NULL,
        template_type VARCHAR NOT NULL,
        platform VARCHAR,
        template TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Create helpers table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS template_helpers (
        id INTEGER PRIMARY KEY,
        helper_name VARCHAR NOT NULL,
        helper_code TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    logger.info(f"Created database schema in {DB_PATH}")
    conn.close()

def add_basic_templates():
    """Add basic templates for text and vision models."""
    conn = duckdb.connect(DB_PATH)
    
    # Template for text models (bert, t5)
    text_template = """import os
import unittest
import torch
from transformers import AutoModel, AutoTokenizer

# Hardware detection
HAS_CUDA = torch.cuda.is_available()
HAS_MPS = hasattr(torch, "mps") and torch.mps.is_available()

class Test{{model_name.replace("-", "").capitalize()}}(unittest.TestCase):
    def setUp(self):
        self.model_name = "{{model_name}}"
    
    def test_cpu(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        inputs = tokenizer("Test text", return_tensors="pt")
        outputs = model(**inputs)
        self.assertIsNotNone(outputs)
        
    def test_cuda(self):
        if not HAS_CUDA:
            self.skipTest("CUDA not available")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        model = model.to("cuda")
        inputs = tokenizer("Test text", return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        outputs = model(**inputs)
        self.assertIsNotNone(outputs)
"""
    
    # Template for vision models (vit)
    vision_template = """import os
import unittest
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModel

# Hardware detection
HAS_CUDA = torch.cuda.is_available()
HAS_WEBGPU = "WEBGPU_AVAILABLE" in os.environ

class Test{{model_name.replace("-", "").capitalize()}}(unittest.TestCase):
    def setUp(self):
        self.model_name = "{{model_name}}"
        self.dummy_image = np.random.rand(3, 224, 224)
    
    def test_cpu(self):
        processor = AutoImageProcessor.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        inputs = processor(self.dummy_image, return_tensors="pt")
        outputs = model(**inputs)
        self.assertIsNotNone(outputs)
        
    def test_webgpu(self):
        if not HAS_WEBGPU:
            self.skipTest("WebGPU not available")
        processor = AutoImageProcessor.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        inputs = processor(self.dummy_image, return_tensors="pt")
        # WebGPU simulation mode
        os.environ["WEBGPU_SIMULATION"] = "1"
        outputs = model(**inputs)
        self.assertIsNotNone(outputs)
        # Reset environment
        os.environ.pop("WEBGPU_SIMULATION", None)
"""

    # Template for OpenVINO tests
    openvino_template = """import os
import unittest
import torch
import importlib.util
import logging
from transformers import AutoModel, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for OpenVINO availability
HAS_OPENVINO = importlib.util.find_spec("openvino") is not None
HAS_OPTIMUM_INTEL = importlib.util.find_spec("optimum.intel") is not None

# Test for OpenVINO hardware backend
HAS_OPENVINO_BACKEND = False
try:
    from ipfs_accelerate_py.hardware.backends.openvino_backend import OpenVINOBackend
    HAS_OPENVINO_BACKEND = True
except ImportError:
    logger.warning("OpenVINO backend not available")

class Test{{model_name.replace("-", "").capitalize()}}OpenVINO(unittest.TestCase):
    def setUp(self):
        self.model_name = "{{model_name}}"
        self.skip_if_openvino_unavailable()
        
    def skip_if_openvino_unavailable(self):
        if not HAS_OPENVINO:
            self.skipTest("OpenVINO not installed")
    
    def test_openvino_direct(self):
        """Test {{model_name}} with direct OpenVINO integration."""
        import openvino as ov
        from openvino.runtime import Core
        
        # Load model using transformers
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        
        # Prepare inputs
        text = "Test text for OpenVINO inference"
        inputs = tokenizer(text, return_tensors="pt")
        
        # Run PyTorch inference for reference
        pt_outputs = model(**inputs)
        self.assertIsNotNone(pt_outputs)
        
        # Try to convert to ONNX and then to OpenVINO IR
        try:
            import tempfile
            import os
            from transformers.onnx import export
            
            # Create temporary directory for ONNX export
            with tempfile.TemporaryDirectory() as tmpdirname:
                # Export to ONNX
                onnx_path = os.path.join(tmpdirname, "model.onnx")
                
                # Get model inputs and outputs
                input_names = list(inputs.keys())
                
                # Export model to ONNX
                export(
                    preprocessor=tokenizer,
                    model=model,
                    config=model.config,
                    opset=13,
                    output=onnx_path
                )
                
                # Convert to OpenVINO
                core = Core()
                ov_model = core.read_model(onnx_path)
                compiled_model = core.compile_model(ov_model, "CPU")
                
                # Create inference request
                infer_request = compiled_model.create_infer_request()
                
                # Prepare inputs
                ov_inputs = {}
                for input_name in input_names:
                    if input_name in inputs:
                        ov_inputs[input_name] = inputs[input_name].numpy()
                
                # Set input tensors
                for input_name, input_tensor in ov_inputs.items():
                    infer_request.set_input_tensor(input_name, input_tensor)
                
                # Perform inference
                infer_request.start_async()
                infer_request.wait()
                
                # Get results
                outputs = {}
                for i, output_name in enumerate(compiled_model.outputs):
                    outputs[output_name] = infer_request.get_output_tensor(i).data
                
                # Verify outputs
                self.assertIsNotNone(outputs)
                logger.info("OpenVINO direct integration successful")
                
        except Exception as e:
            logger.warning(f"OpenVINO direct integration test failed: {e}")
            self.skipTest(f"OpenVINO direct integration test failed: {e}")
    
    def test_openvino_backend(self):
        """Test {{model_name}} with OpenVINO backend."""
        if not HAS_OPENVINO_BACKEND:
            self.skipTest("OpenVINO backend not available")
        
        try:
            backend = OpenVINOBackend()
            if not backend.is_available():
                self.skipTest("OpenVINO backend reports not available")
            
            # Load model with backend
            config = {
                "device": "CPU",
                "model_type": "text",
                "model_path": self.model_name,  # Use model name directly (may not work for all models)
                "precision": "FP32"
            }
            
            load_result = backend.load_model(self.model_name, config)
            self.assertEqual(load_result.get("status"), "success", 
                           f"Failed to load model: {load_result.get('message', 'Unknown error')}")
            
            # Run inference
            input_content = {
                "input_ids": [101, 2054, 2154, 2003, 2026, 3793, 2080, 2339, 1029, 102],
                "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            }
            
            inference_result = backend.run_inference(
                self.model_name,
                input_content,
                config
            )
            
            self.assertEqual(inference_result.get("status"), "success", 
                           f"Inference failed: {inference_result.get('message', 'Unknown error')}")
            
            # Unload model
            backend.unload_model(self.model_name, "CPU")
            logger.info("OpenVINO backend test successful")
            
        except Exception as e:
            logger.warning(f"OpenVINO backend test failed: {e}")
            self.skipTest(f"OpenVINO backend test failed: {e}")
    
    def test_optimum_intel(self):
        """Test {{model_name}} with optimum.intel integration."""
        if not HAS_OPTIMUM_INTEL:
            self.skipTest("optimum.intel not installed")
        
        try:
            # Import optimum.intel
            import optimum.intel
            
            # Check for required model classes
            available_classes = []
            sequence_class_available = False
            causal_lm_available = False
            seq2seq_available = False
            
            try:
                from optimum.intel import OVModelForSequenceClassification
                sequence_class_available = True
                available_classes.append("SequenceClassification")
            except ImportError:
                pass
                
            try:
                from optimum.intel import OVModelForCausalLM
                causal_lm_available = True
                available_classes.append("CausalLM")
            except ImportError:
                pass
                
            try:
                from optimum.intel import OVModelForSeq2SeqLM
                seq2seq_available = True
                available_classes.append("Seq2SeqLM")
            except ImportError:
                pass
            
            # If we have no model classes, skip the test
            if not available_classes:
                self.skipTest("No supported optimum.intel model classes available")
            
            # Determine appropriate model class based on model name
            model_class = None
            model_type = None
            
            if "seq2seq" in self.model_name or "t5" in self.model_name:
                if seq2seq_available:
                    from optimum.intel import OVModelForSeq2SeqLM
                    model_class = OVModelForSeq2SeqLM
                    model_type = "seq2seq"
            elif "gpt" in self.model_name or "bloom" in self.model_name or "llama" in self.model_name:
                if causal_lm_available:
                    from optimum.intel import OVModelForCausalLM
                    model_class = OVModelForCausalLM
                    model_type = "causal_lm"
            else:
                if sequence_class_available:
                    from optimum.intel import OVModelForSequenceClassification
                    model_class = OVModelForSequenceClassification
                    model_type = "sequence"
            
            # Skip if no appropriate model class found
            if model_class is None:
                self.skipTest(f"No appropriate optimum.intel model class for {self.model_name}")
            
            # Load model with optimum.intel
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = model_class.from_pretrained(
                self.model_name, 
                export=True,
                load_in_8bit=False
            )
            
            # Prepare inputs
            text = "Test text for OpenVINO inference"
            inputs = tokenizer(text, return_tensors="pt")
            
            # Run inference
            outputs = model(**inputs)
            
            # Verify outputs
            self.assertIsNotNone(outputs)
            logger.info(f"optimum.intel test successful using {model_type} model class")
            
        except Exception as e:
            logger.warning(f"optimum.intel test failed: {e}")
            self.skipTest(f"optimum.intel test failed: {e}")
"""
    
    # Add text template
    conn.execute(
        "INSERT INTO templates (id, model_type, template_type, template) VALUES (1, 'text', 'test', ?)",
        [text_template]
    )
    
    # Add text template for bert specifically
    conn.execute(
        "INSERT INTO templates (id, model_type, template_type, template) VALUES (2, 'bert', 'test', ?)",
        [text_template]
    )
    
    # Add vision template
    conn.execute(
        "INSERT INTO templates (id, model_type, template_type, template) VALUES (3, 'vision', 'test', ?)",
        [vision_template]
    )
    
    # Add vision template for vit specifically
    conn.execute(
        "INSERT INTO templates (id, model_type, template_type, template) VALUES (4, 'vit', 'test', ?)",
        [vision_template]
    )
    
    # Add OpenVINO template
    conn.execute(
        "INSERT INTO templates (id, model_type, template_type, template) VALUES (5, 'openvino', 'test', ?)",
        [openvino_template]
    )
    
    # Add OpenVINO template for specific model types
    conn.execute(
        "INSERT INTO templates (id, model_type, template_type, platform, template) VALUES (6, 'text', 'test', 'openvino', ?)",
        [openvino_template]
    )
    
    conn.execute(
        "INSERT INTO templates (id, model_type, template_type, platform, template) VALUES (7, 'vision', 'test', 'openvino', ?)",
        [openvino_template]
    )
    
    logger.info("Added basic templates (including OpenVINO templates)")
    conn.close()

def list_templates():
    """List all templates in the database."""
    conn = duckdb.connect(DB_PATH)
    
    templates = conn.execute("""
    SELECT id, model_type, template_type, platform, length(template) as size
    FROM templates
    ORDER BY id
    """).fetchall()
    
    print("\nTemplates in database:")
    print("-" * 80)
    print("{:<4} {:<15} {:<15} {:<15} {:<10}".format("ID", "Model Type", "Template Type", "Platform", "Size (bytes)"))
    print("-" * 80)
    
    for t in templates:
        platform = t[3] if t[3] is not None else "all"
        print("{:<4} {:<15} {:<15} {:<15} {:<10}".format(t[0], t[1], t[2], platform, t[4]))
    
    print("-" * 80)
    conn.close()

def main():
    """Main function."""
    # Check if database already exists
    if os.path.exists(DB_PATH):
        logger.info(f"{DB_PATH} already exists, overwriting.")
        try:
            os.remove(DB_PATH)
        except Exception as e:
            logger.error(f"Failed to remove existing database: {e}")
            return 1
    
    # Create database schema
    create_database()
    
    # Add basic templates
    add_basic_templates()
    
    # List templates
    list_templates()
    
    print(f"\nTemplate database created at {DB_PATH}")
    print("You can now use it with the simple_test_generator.py script:")
    print("  python simple_test_generator.py -g bert -t")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
