#!/usr/bin/env python
import json
import os
import re

def extract_username_from_url(url):
    """Extract the username from a Hugging Face URL."""
    if not url or not isinstance(url, str):
        return None
    
    # Match pattern: https://huggingface.co/username/model-name
    match = re.search(r'huggingface\.co/([^/]+)/([^/]+)', url)
    if match:
        return match.group(1)
    return None

def update_collection_json():
    """Update the collection.json file by adding usernames to model keys."""
    collection_path = "collection.json"
    
    # Load the collection JSON file
    with open(collection_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create a new dictionary for the updated data
    updated_data = {}
    
    # Special entries to preserve
    special_entries = ['cache']
    
    # Counter for statistics
    total_models = 0
    updated_models = 0
    
    # Process each model in the collection
    for key, value in data.items():
        # Skip special entries
        if key in special_entries:
            updated_data[key] = value
            continue
        
        total_models += 1
        
        # Check if the key already has a username format (username/model)
        if '/' in key:
            # Check if there's a duplicate username in the key (like "TheBloke/TheBloke/...")
            parts = key.split('/')
            if len(parts) >= 2 and parts[0] == parts[1]:
                # Remove the duplicate username
                fixed_key = '/'.join(parts[1:])
                updated_data[fixed_key] = value
                
                # Update the ID field if it exists
                if 'id' in value:
                    value['id'] = fixed_key
                
                updated_models += 1
            else:
                # Key already has proper format
                updated_data[key] = value
        else:
            # Extract the username from source URL if available
            username = None
            if 'source' in value and isinstance(value['source'], str):
                username = extract_username_from_url(value['source'])
            
            # Create the new key with username if available
            if username:
                new_key = f"{username}/{key}"
                
                # Also update the "id" field to include the username
                if 'id' in value:
                    value['id'] = new_key
                
                updated_data[new_key] = value
                updated_models += 1
            else:
                # Keep the original key if no username could be extracted
                updated_data[key] = value
    
    # Save the updated JSON with proper indentation
    with open('updated_collection.json', 'w', encoding='utf-8') as f:
        json.dump(updated_data, f, indent=2)
    
    print(f"Updated collection.json saved as 'updated_collection.json'")
    print(f"Total models: {total_models}")
    print(f"Updated models: {updated_models}")

if __name__ == "__main__":
    update_collection_json()