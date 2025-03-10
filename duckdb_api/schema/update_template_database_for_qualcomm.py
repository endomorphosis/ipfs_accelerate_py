#!/usr/bin/env python3
# Update template database and code generators for Qualcomm AI Engine support.

import os
import sys
import re
import json
import shutil

def update_template_database(template_db_json, template_dir):
    # Update template database JSON with Qualcomm handlers.
    if not os.path.exists(template_db_json):
        print(f"Error: Template database not found at {template_db_json}")
        return False
    
    try:
        # Load the template database
        with open(template_db_json, 'r') as f:
            template_db = json.load(f)
        
        # Check if Qualcomm is already in the hardware list
        if "qualcomm" not in template_db.get("hardware_platforms", []):
            # Add Qualcomm to the hardware platforms list
            if "hardware_platforms" not in template_db:
                template_db["hardware_platforms"] = []
            
            template_db["hardware_platforms"].append("qualcomm")
            print("Added Qualcomm to hardware platforms list")
        
        # Update the template database
        with open(template_db_json, 'w') as f:
            json.dump(template_db, f, indent=2)
        
        print(f"Updated template database at {template_db_json}")
        return True
        
    except Exception as e:
        print(f"Error updating template database: {e}")
        return False

def main():
    # Main function to update template database for Qualcomm support.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Template database paths
    template_db_json = os.path.join(current_dir, "hardware_test_templates", "template_database.json")
    template_dir = os.path.join(current_dir, "hardware_test_templates")
    
    # Check if template directory exists
    if not os.path.exists(template_dir):
        print(f"Error: Template directory not found at {template_dir}")
        sys.exit(1)
    
    # Update template database
    success = update_template_database(template_db_json, template_dir)
    if not success:
        print("Warning: Failed to update template database")
    
    print("\nQualcomm hardware support update completed!")
    print("Next steps:")
    print("1. Run the fix_merged_test_generator.py script to update the test generator")
    print("2. Test the updated generator with: python merged_test_generator.py --model bert --cross-platform --hardware all")
    print("3. Verify Qualcomm support is included in the generated tests")

if __name__ == "__main__":
    main()