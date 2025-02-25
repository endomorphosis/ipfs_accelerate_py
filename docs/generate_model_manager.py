import json
import os

def generate_html():
    # Read the collection.json
    with open('collection.json', 'r') as f:
        collection_data = json.load(f)
    
    # Read the HTML template
    with open('Huggingface_Model_Manager.html', 'r') as f:
        html_content = f.read()

    # Convert the collection data to a string and escape any problematic characters
    json_str = json.dumps(collection_data, indent=2, ensure_ascii=False).replace('</script>', '<\\/script>')
    
    # Replace the placeholder with the actual collection data
    html_content = html_content.replace(
        'const collectionData = {\n            // Collection data will be injected here by the Python script\n        }',
        f'const collectionData = {json_str}'
    )
    
    # Write the final HTML file
    output_path = 'Huggingface_Model_Manager_with_data.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Generated {output_path} successfully!")

if __name__ == "__main__":
    generate_html()