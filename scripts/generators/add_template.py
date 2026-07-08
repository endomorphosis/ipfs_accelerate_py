#\!/usr/bin/env python3
from hardware_test_templates.template_database import TemplateDatabase

def add_bert_template():
    db = TemplateDatabase()
    
    with open('/tmp/bert_template.py', 'r') as f:
        content = f.read()
    
    db.store_template(
        model_id='bert',
        template_content=content,
        model_name='bert',
        model_family='embedding',
        modality='text'
    )
    
    print("Template stored successfully")

if __name__ == "__main__":
    add_bert_template()
