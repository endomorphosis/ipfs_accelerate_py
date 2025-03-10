#!/usr/bin/env python3
"""
Create Template Database

This script creates and initializes the template database with the expected schema
for the fix_template_integration.py script. It sets up the required tables and
adds sample templates for key models.

Usage:
  python create_template_database.py
"""

import os
import sys
import logging
import duckdb
from pathlib import Path
from datetime import datetime
import glob
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = CURRENT_DIR / "template_db.duckdb"
TEMPLATES_DIR = CURRENT_DIR / "hardware_test_templates"

# Key model categories
MODEL_CATEGORIES = {
    "text": ["bert", "t5", "llama", "qwen2"],
    "vision": ["vit", "clip", "detr"],
    "audio": ["wav2vec2", "whisper", "clap"],
    "multimodal": ["llava", "llava_next"],
    "video": ["xclip"]
}

def create_schema():
    """Create the database schema."""
    conn = duckdb.connect(str(DB_PATH))
    try:
        # Create template tables
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
        
        # Create template versions table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS template_versions (
            id INTEGER PRIMARY KEY,
            template_id INTEGER,
            version VARCHAR NOT NULL,
            changes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create template dependencies table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS template_dependencies (
            id INTEGER PRIMARY KEY,
            template_id INTEGER,
            dependency_template_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create template variables table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS template_variables (
            id INTEGER PRIMARY KEY,
            variable_name VARCHAR NOT NULL,
            default_value TEXT,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create template validation table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS template_validation (
            id INTEGER PRIMARY KEY,
            template_id INTEGER,
            validation_date TIMESTAMP,
            is_valid BOOLEAN,
            validation_errors TEXT,
            validation_warnings TEXT
        )
        """)
        
        logger.info("Successfully created database schema")
        return True
    except Exception as e:
        logger.error(f"Error creating database schema: {e}")
        return False
    finally:
        conn.close()

def add_template_variables():
    """Add common template variables."""
    conn = duckdb.connect(str(DB_PATH))
    
    variables = [
        ("model_name", "bert-base-uncased", "Name of the model to test"),
        ("model_category", "text", "Category of the model"),
        ("support_level", "REAL", "Hardware support level (REAL, SIMULATION, MOCK)"),
        ("platform", "cpu", "Hardware platform"),
        ("batch_size", "2", "Batch size for testing"),
        ("sequence_length", "128", "Sequence length for text models"),
        ("device", "cpu", "PyTorch device"),
        ("hardware_type", "cpu", "Type of hardware to test on")
    ]
    
    try:
        for idx, (var_name, default_value, description) in enumerate(variables, 1):
            conn.execute(
                "INSERT INTO template_variables (id, variable_name, default_value, description) VALUES (?, ?, ?, ?)",
                [idx, var_name, default_value, description]
            )
        
        logger.info(f"Added {len(variables)} template variables")
        return True
    except Exception as e:
        logger.error(f"Error adding template variables: {e}")
        return False
    finally:
        conn.close()

def add_sample_templates():
    """Add sample templates for key models."""
    conn = duckdb.connect(str(DB_PATH))
    
    # Sample basic template for text models
    text_template = """
import torch
from transformers import AutoModel, AutoTokenizer

def test_{{model_name}}():
    # Load model and tokenizer
    model_name = "{{model_name}}"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Prepare input
    text = "This is a sample text for {{model_category}} model testing."
    inputs = tokenizer(text, return_tensors="pt")
    
    # Run model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Check outputs
    assert outputs.last_hidden_state is not None
    print(f"Successfully tested {model_name}")
    return outputs
"""

    # Sample basic template for vision models
    vision_template = """
import torch
from transformers import AutoFeatureExtractor, AutoModel

def test_{{model_name}}():
    # Load model and feature extractor
    model_name = "{{model_name}}"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Prepare input (dummy image)
    import numpy as np
    dummy_image = np.zeros((3, 224, 224))
    inputs = feature_extractor(dummy_image, return_tensors="pt")
    
    # Run model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Check outputs
    assert outputs.last_hidden_state is not None
    print(f"Successfully tested {model_name}")
    return outputs
"""

    # Sample basic template for audio models
    audio_template = """
import torch
from transformers import AutoFeatureExtractor, AutoModel

def test_{{model_name}}():
    # Load model and feature extractor
    model_name = "{{model_name}}"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Prepare input (dummy audio)
    import numpy as np
    dummy_audio = np.zeros(16000)  # 1 second of silence at 16kHz
    inputs = feature_extractor(dummy_audio, return_tensors="pt")
    
    # Run model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Check outputs
    assert outputs.last_hidden_state is not None
    print(f"Successfully tested {model_name}")
    return outputs
"""

    # Sample hardware platform template for CPU
    cpu_platform_template = """
import torch

# CPU platform-specific initialization for {{model_name}}
def init_cpu(model_name="{{model_name}}"):
    # Load the model on CPU
    device = "cpu"
    model = torch.hub.load("huggingface/" + model_name, model_name)
    model.to(device)
    model.eval()
    
    # Return model, device, and any other necessary components
    return {
        "model": model,
        "device": device,
        "platform": "cpu"
    }
"""

    # Sample hardware platform template for CUDA
    cuda_platform_template = """
import torch

# CUDA platform-specific initialization for {{model_name}}
def init_cuda(model_name="{{model_name}}"):
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return init_cpu(model_name)
    
    # Load the model on CUDA
    device = "cuda"
    model = torch.hub.load("huggingface/" + model_name, model_name)
    model.to(device)
    model.eval()
    
    # Return model, device, and any other necessary components
    return {
        "model": model,
        "device": device,
        "platform": "cuda"
    }
"""
    
    # Add templates for each key model
    try:
        templates_added = 0
        template_id = 1
        
        # Add base templates for each category
        for category, models in MODEL_CATEGORIES.items():
            if category == "text":
                base_template = text_template
            elif category in ["vision", "multimodal"]:
                base_template = vision_template
            elif category in ["audio", "video"]:
                base_template = audio_template
            else:
                base_template = text_template
            
            for model in models:
                # Add base template
                conn.execute(
                    "INSERT INTO templates (id, model_type, template_type, template) VALUES (?, ?, 'base', ?)",
                    [template_id, model, base_template]
                )
                template_id += 1
                
                # Add CPU platform template
                conn.execute(
                    "INSERT INTO templates (id, model_type, template_type, platform, template) VALUES (?, ?, 'hardware_platform', 'cpu', ?)",
                    [template_id, model, cpu_platform_template]
                )
                template_id += 1
                
                # Add CUDA platform template
                conn.execute(
                    "INSERT INTO templates (id, model_type, template_type, platform, template) VALUES (?, ?, 'hardware_platform', 'cuda', ?)",
                    [template_id, model, cuda_platform_template]
                )
                template_id += 1
                
                templates_added += 3
        
        logger.info(f"Added {templates_added} sample templates")
        return True
    except Exception as e:
        logger.error(f"Error adding sample templates: {e}")
        return False
    finally:
        conn.close()

def add_helper_functions():
    """Add common helper functions."""
    conn = duckdb.connect(str(DB_PATH))
    
    helpers = [
        (
            "hardware_detection", 
            """
import torch
import importlib.util

def detect_hardware():
    \"\"\"Detect available hardware.\"\"\"
    hardware = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": hasattr(torch, "mps") and torch.mps.is_available(),
        "rocm": hasattr(torch, "_C") and hasattr(torch._C, "_rocm_version"),
        "openvino": importlib.util.find_spec("openvino") is not None,
        "webnn": importlib.util.find_spec("webnn") is not None,
        "webgpu": importlib.util.find_spec("webgpu") is not None
    }
    
    return hardware
"""
        ),
        (
            "model_loading",
            """
from transformers import AutoConfig, AutoModel, AutoTokenizer

def load_model(model_name, device="cpu"):
    \"\"\"Load a model and move it to the specified device.\"\"\"
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    tokenizer = None
    if hasattr(model, "config") and model.config.model_type in ["bert", "gpt2", "t5"]:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer
"""
        )
    ]
    
    try:
        for idx, (helper_name, helper_code) in enumerate(helpers, 1):
            conn.execute(
                "INSERT INTO template_helpers (id, helper_name, helper_code) VALUES (?, ?, ?)",
                [idx, helper_name, helper_code]
            )
        
        logger.info(f"Added {len(helpers)} helper functions")
        return True
    except Exception as e:
        logger.error(f"Error adding helper functions: {e}")
        return False
    finally:
        conn.close()

def main():
    """Main function."""
    logger.info(f"Creating template database at {DB_PATH}")
    
    # Create the database schema
    if not create_schema():
        logger.error("Failed to create database schema")
        return 1
    
    # Add template variables
    if not add_template_variables():
        logger.error("Failed to add template variables")
        return 1
    
    # Add sample templates
    if not add_sample_templates():
        logger.error("Failed to add sample templates")
        return 1
    
    # Add helper functions
    if not add_helper_functions():
        logger.error("Failed to add helper functions")
        return 1
    
    logger.info("Template database created successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())