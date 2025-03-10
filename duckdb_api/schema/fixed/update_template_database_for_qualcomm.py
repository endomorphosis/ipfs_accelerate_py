#!/usr/bin/env python3
"""
Update template database and code generators for Qualcomm AI Engine support.
"""

import os
import sys
import re
import json
import shutil

def update_template_database(template_db_json, template_dir):
    """Update template database JSON with Qualcomm handlers."""
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

def update_template_files(template_dir):
    """Update template Python files with Qualcomm handlers."""
    if not os.path.exists(template_dir):
        print(f"Error: Template directory not found at {template_dir}")
        return False
    
    # Find Python template files
    template_files = []
    for root, _, files in os.walk(template_dir):
        for file in files:
            if file.endswith(".py"):
                template_files.append(os.path.join(root, file))
    
    if not template_files:
        print(f"No Python template files found in {template_dir}")
        return False
    
    # Look for template files that have hardware handlers but no Qualcomm handler
    updated_count = 0
    for file_path in template_files:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if the file has hardware handlers, based on get_X_handler pattern
        if "def get_cuda_handler" in content or "def get_cpu_handler" in content:
            # Check if Qualcomm handler already exists
            if "def get_qualcomm_handler" not in content:
                # Find all hardware handler patterns
                handlers = re.findall(r'def get_(\w+)_handler\(model_path.*?(?=def|$)', content, re.DOTALL)
                
                if handlers:
                    # Find the last hardware handler pattern
                    last_handler = handlers[-1]
                    pattern = f"def get_{last_handler}_handler"
                    last_handler_match = re.search(f"{pattern}.*?(?=def|$)", content, re.DOTALL)
                    
                    if last_handler_match:
                        # Create Qualcomm handler template based on the pattern of the last handler
                        last_handler_code = last_handler_match.group(0)
                        qualcomm_handler = last_handler_code.replace(f"get_{last_handler}_handler", "get_qualcomm_handler")
                        qualcomm_handler = qualcomm_handler.replace(f"{last_handler.upper()}", "QUALCOMM")
                        
                        # Add Qualcomm-specific imports and checks
                        qualcomm_handler = qualcomm_handler.replace(
                            "import torch", 
                            "import torch\n    # Qualcomm AI Engine imports\n    try:\n        import qnn_wrapper\n        HAS_QNN = True\n    except ImportError:\n        try:\n            import qti.aisw.dlc_utils as qti_utils\n            HAS_QNN = True\n        except ImportError:\n            HAS_QNN = False"
                        )
                        
                        # Add the handler to the file content
                        insertion_point = content.rfind("def ")
                        if insertion_point != -1:
                            # Find the correct position to insert the new handler
                            # We need to find the first function declaration after the last handler
                            insertion_match = re.search(r'def\s+(?!get_\w+_handler)', content[insertion_point:])
                            if insertion_match:
                                insertion_point += insertion_match.start()
                                updated_content = content[:insertion_point] + qualcomm_handler + "\n\n" + content[insertion_point:]
                            else:
                                # If no match, append to the end of the file
                                updated_content = content + "\n\n" + qualcomm_handler
                        else:
                            # If no existing handler, add to the end of the file
                            updated_content = content + "\n\n" + qualcomm_handler
                        
                        # Write the updated content
                        with open(file_path, 'w') as f:
                            f.write(updated_content)
                        
                        print(f"Added Qualcomm handler to {file_path}")
                        updated_count += 1
    
    if updated_count > 0:
        print(f"Added Qualcomm handlers to {updated_count} template files")
        return True
    else:
        print("No template files needed Qualcomm handler updates")
        return True

def update_template_integration(base_dir):
    """Update integrated_skillset_generator.py to properly handle Qualcomm."""
    # Find the integrated skillset generator
    candidates = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file == "integrated_skillset_generator.py":
                candidates.append(os.path.join(root, file))
    
    if not candidates:
        print("Error: Could not find integrated_skillset_generator.py")
        return False
    
    # Use the first candidate (most likely to be the correct one)
    generator_path = candidates[0]
    
    try:
        with open(generator_path, 'r') as f:
            content = f.read()
        
        # Check if Qualcomm is already supported
        if "qualcomm" in content and "get_qualcomm_handler" in content:
            print(f"Qualcomm already supported in {generator_path}")
            return True
        
        # Add Qualcomm to hardware platforms list
        hardware_platforms_match = re.search(r'hardware_platforms\s*=\s*\[([^\]]*)\]', content)
        if hardware_platforms_match:
            hardware_platforms = hardware_platforms_match.group(1)
            if "qualcomm" not in hardware_platforms:
                # Add Qualcomm to the list
                updated_platforms = hardware_platforms.rstrip() + ', "qualcomm"'
                updated_content = content.replace(hardware_platforms_match.group(1), updated_platforms)
                
                # Write the updated content
                with open(generator_path, 'w') as f:
                    f.write(updated_content)
                
                print(f"Added Qualcomm to hardware platforms in {generator_path}")
                return True
            else:
                print(f"Qualcomm already in hardware platforms list in {generator_path}")
                return True
        else:
            print(f"Could not find hardware_platforms list in {generator_path}")
            return False
            
    except Exception as e:
        print(f"Error updating integrated_skillset_generator.py: {e}")
        return False

def create_qualcomm_template_example(template_dir):
    """Create or update an example template for Qualcomm implementation."""
    # Find a BERT template file to modify
    bert_template_path = None
    for root, _, files in os.walk(template_dir):
        for file in files:
            if "bert" in file.lower() and file.endswith(".py"):
                bert_template_path = os.path.join(root, file)
                break
    
    if not bert_template_path:
        print("Could not find a BERT template file")
        return False
    
    # Read the template file
    with open(bert_template_path, 'r') as f:
        content = f.read()
    
    # Check if Qualcomm handler already exists
    if "def get_qualcomm_handler" in content:
        print("Qualcomm handler already exists in BERT template")
        return True
    
    # Add complete Qualcomm handler to BERT template
    qualcomm_handler = """
def get_qualcomm_handler(model_path):
    \"\"\"
    Initialize BERT model on Qualcomm AI Engine.
    
    Args:
        model_path: Path to the model
        
    Returns:
        Configured Qualcomm handler for BERT
    \"\"\"
    """
    print(f"Initializing BERT model on Qualcomm AI Engine: {model_path}")
    
    # Check for Qualcomm SDK/QNN
    try:
        import qnn_wrapper
        HAS_QNN = True
    except ImportError:
        try:
            # Try alternative Qualcomm AI SDK
            import qti.aisw.dlc_utils as qti_utils
            HAS_QNN = False
        except ImportError:
            print("Error: Qualcomm AI Engine SDK not found")
            return None
    
    class QualcommHandler:
        def __init__(self, model_path):
            self.model_path = model_path
            self.qnn_model = None
            self.tokenizer = None
            
            # Prepare tokenizer
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            except Exception as e:
                print(f"Error loading tokenizer: {e}")
                self.tokenizer = None
            
            # Compile model for Qualcomm platform
            self._compile_model()
        
        def _compile_model(self):
            """Compile model for Qualcomm AI Engine."""
            try:
                # Convert to ONNX if needed
                onnx_path = self.model_path.replace(".pt", ".onnx")
                if not os.path.exists(onnx_path):
                    print(f"Converting model to ONNX format: {onnx_path}")
                    # Code to convert model to ONNX would go here
                    # This is a simplified example
                
                # Compile for Qualcomm
                if HAS_QNN:
                    self.qnn_model = qnn_wrapper.compile(onnx_path)
                else:
                    # Using QTI SDK path
                    self.qnn_model = qti_utils.compile(onnx_path)
                
                print(f"Successfully compiled model for Qualcomm AI Engine")
            except Exception as e:
                print(f"Error compiling model for Qualcomm: {e}")
                self.qnn_model = None
        
        def __call__(self, text, **kwargs):
            """
            Run inference on Qualcomm hardware.
            
            Args:
                text: Input text
                
            Returns:
                Dictionary with embeddings and metadata
            """
            try:
                if self.qnn_model is None:
                    raise ValueError("Model not compiled for Qualcomm")
                
                if self.tokenizer is None:
                    raise ValueError("Tokenizer not initialized")
                
                # Tokenize input
                encoded_input = self.tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
                
                # Run inference on Qualcomm hardware
                inputs = {
                    "input_ids": encoded_input["input_ids"].numpy(),
                    "attention_mask": encoded_input["attention_mask"].numpy()
                }
                
                outputs = self.qnn_model.infer(inputs)
                embeddings = outputs[1]
                
                return {
                    "embeddings": embeddings.flatten().tolist(),
                    "implementation_type": "QUALCOMM"
                }
                
            except Exception as e:
                print(f"Error during Qualcomm inference: {e}")
                return {"error": str(e), "implementation_type": "ERROR_QUALCOMM"}
    
    return QualcommHandler(model_path)"""
    
    # Find a good position to insert the handler (after all other handlers)
    openvino_handler = "def get_openvino_handler"
    mps_handler = "def get_mps_handler"
    
    insert_position = -1
    for handler in [openvino_handler, mps_handler]:
        position = content.find(handler)
        if position != -1:
            # Find the end of this handler definition
            next_def = content.find("def ", position + len(handler))
            if next_def != -1:
                insert_position = next_def
                break
    
    if insert_position == -1:
        # Add at the end of the file
        updated_content = content + "\n\n" + qualcomm_handler
    else:
        # Insert at the determined position
        updated_content = content[:insert_position] + qualcomm_handler + "\n\n" + content[insert_position:]
    
    # Write updated content
    with open(bert_template_path, 'w') as f:
        f.write(updated_content)
    
    print(f"Added complete Qualcomm handler example to {bert_template_path}")
    return True

def main():
    """Main function to update template database for Qualcomm support."""
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
    
    # Update template files
    success = update_template_files(template_dir)
    if not success:
        print("Warning: Failed to update template files")
    
    # Create example template
    success = create_qualcomm_template_example(template_dir)
    if not success:
        print("Warning: Failed to create example template")
    
    # Update integrated skillset generator
    success = update_template_integration(current_dir)
    if not success:
        print("Warning: Failed to update integrated skillset generator")
    
    print("\nQualcomm hardware support update completed!")
    print("Next steps:")
    print("1. Run the fix_merged_test_generator.py script to update the test generator")
    print("2. Test the updated generator with: python merged_test_generator.py --model bert --cross-platform --hardware all")
    print("3. Verify Qualcomm support is included in the generated tests")

if __name__ == "__main__":
    main()