import json
import os

def generate_html():
    # Read the collection.json file
    try:
        with open('updated_collection.json', 'r', encoding='utf-8') as f:
            collection_data = json.load(f)
        print("Using updated_collection.json")
    except FileNotFoundError:
        # Fall back to original collection.json if updated version doesn't exist
        with open('collection.json', 'r', encoding='utf-8') as f:
            collection_data = json.load(f)
        print("Using collection.json")
    
    # Read the HTML template
    with open('Huggingface_Model_Manager.html', 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Convert the collection data to a string and escape any problematic characters
    json_str = json.dumps(collection_data, indent=2, ensure_ascii=False).replace('</script>', '<\\/script>')
    
    # Look for a placeholder in the HTML content
    if 'const collectionData = {' in html_content:
        # Replace from the beginning of the collectionData declaration to the end of the object
        start_marker = 'const collectionData = {'
        end_marker = '};'
        
        start_index = html_content.find(start_marker)
        if start_index != -1:
            end_index = html_content.find(end_marker, start_index)
            if end_index != -1:
                # Replace only the content between markers (keeping the declaration and closing brace)
                new_content = html_content[:start_index + len(start_marker)] + '\n' + json_str.strip().lstrip('{').rstrip('}') + '\n        ' + html_content[end_index:]
                html_content = new_content
    else:
        # If no placeholder is found, inject the data at a reasonable position
        html_content = html_content.replace(
            '<script>',
            '<script>\n        const collectionData = ' + json_str + ';\n',
            1  # Replace only the first occurrence
        )
    
    # Count the number of models in the collection
    model_count = sum(1 for key in collection_data if key != 'cache')
    
    # Write the final HTML file
    output_path = 'Huggingface_Model_Manager_with_data.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Generated {output_path} with {model_count} models successfully!")

if __name__ == "__main__":
    generate_html()