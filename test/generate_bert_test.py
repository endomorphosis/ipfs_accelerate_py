#\!/usr/bin/env python3
"""
Generate a test file for BERT from the template database.

This script uses the standard templates without relying on DuckDB.
"""

import os
import sys
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Template database path
JSON_DB_PATH = "../generators/templates/template_db.json"

def load_template_db(db_path):
    """Load the template database from a JSON file"""
    with open(db_path, 'r') as f:
        db = json.load(f)
    return db

def get_template_by_model_type(db, model_type, template_type="test"):
    """Get a template for a specific model type"""
    for template_id, template_data in db['templates'].items():
        if template_data.get('model_type') == model_type and template_data.get('template_type') == template_type:
            return template_data.get('template', '')
    return None

def generate_bert_test(output_path="test_bert_fixed.py"):
    """Generate a test file for BERT"""
    try:
        # Load template database
        db = load_template_db(JSON_DB_PATH)
        
        # Get BERT template
        template = get_template_by_model_type(db, "bert", "test")
        
        if not template:
            logger.error("No BERT template found")
            return False
        
        # Customize template for BERT base uncased
        template = template.replace("{{model_name}}", "bert-base-uncased")
        template = template.replace("{{model_class}}", "BertBaseUncased")
        template = template.replace("{{generation_date}}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Save to file
        with open(output_path, 'w') as f:
            f.write(template)
        
        # Make executable
        os.chmod(output_path, 0o755)
        
        logger.info(f"Generated BERT test file at {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error generating BERT test file: {e}")
        return False

if __name__ == "__main__":
    output_path = "test_bert_fixed.py"
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    
    success = generate_bert_test(output_path)
    sys.exit(0 if success else 1)
