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
    
    logger.info("Added basic templates")
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
