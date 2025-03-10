#\!/usr/bin/env python3
import json
import sys

try:
    with open('/home/barberb/ipfs_accelerate_py/generators/templates/template_db.json') as f:
        data = json.load(f)
    
    print(f"Total templates: {len(data['templates'])}")
    
    # Count by template type
    types = {}
    for k, v in data['templates'].items():
        template_type = v.get('template_type')
        types[template_type] = types.get(template_type, 0) + 1
    
    print("Template types:")
    for t, count in types.items():
        print(f"  - {t}: {count}")
    
    # Count by model type
    model_types = {}
    for k, v in data['templates'].items():
        model_type = v.get('model_type')
        model_types[model_type] = model_types.get(model_type, 0) + 1
    
    print("\nModel types:")
    for t, count in model_types.items():
        print(f"  - {t}: {count}")
    
    # Check syntax validity
    import ast
    valid_count = 0
    invalid_templates = []
    
    for template_id, template_data in data['templates'].items():
        try:
            content = template_data.get('template', '')
            ast.parse(content)
            valid_count += 1
        except SyntaxError:
            invalid_templates.append(template_id)
    
    print(f"\nSyntax validity: {valid_count}/{len(data['templates'])} valid ({valid_count/len(data['templates'])*100:.1f}%)")
    
    if len(invalid_templates) <= 10:
        print("\nInvalid templates:")
        for template_id in invalid_templates:
            print(f"  - {template_id}")
    else:
        print(f"\nInvalid templates: {len(invalid_templates)} templates have syntax errors")

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
