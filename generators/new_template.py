#\!/usr/bin/env python3
from hardware_test_templates.template_database import TemplateDatabase
import ast

def create_bert_template_from_vit():
    """Create a BERT template from a working VIT test file."""
    db = TemplateDatabase()
    
    # Read the VIT test file with valid syntax
    with open('/tmp/template-samples/test_hf_vit.py', 'r') as f:
        vit_content = f.read()
    
    # Replace VIT-specific references with placeholders
    bert_template = vit_content
    
    # Replace class name
    bert_template = bert_template.replace('TestVitModels', '{class_name}')
    
    # Replace model references
    bert_template = bert_template.replace('vit', '{model_name}')
    bert_template = bert_template.replace('Vit', '{model_type_capitalize}')
    
    # Replace model IDs
    bert_template = bert_template.replace('google/vit-base-patch16-224', '{model_id}')
    
    # Replace modality
    bert_template = bert_template.replace('vision', '{modality}')
    
    # Make sure the template is valid Python
    try:
        ast.parse(bert_template.format(
            class_name='TestBertModels',
            model_name='bert',
            model_type_capitalize='Bert',
            model_id='bert-base-uncased',
            modality='text'
        ))
        print("Template is valid Python syntax")
    except SyntaxError as e:
        print(f"Template has syntax error: {e}")
        return
    
    # Store the template
    db.delete_template('bert')
    db.store_template(
        model_id='bert',
        template_content=bert_template,
        model_name='bert',
        model_family='embedding',
        modality='text'
    )
    
    print("Template created from VIT test and stored successfully")

if __name__ == "__main__":
    create_bert_template_from_vit()
