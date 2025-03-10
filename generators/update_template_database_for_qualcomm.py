#!/usr/bin/env python3
"""
Script to update the template database with Qualcomm hardware support

This script ensures all templates in the template database include
Qualcomm hardware support with proper implementation details.

Key features:
1. Updates template_database.json to include Qualcomm
2. Adds Qualcomm handlers to all template files
3. Updates template inheritance system to support Qualcomm
4. Ensures that all generated tests include Qualcomm hardware detection
"""

import os
import sys
import json
import re
import shutil
from datetime import datetime
import importlib.util
from pathlib import Path

def backup_file(file_path):
    """Create backup of original file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.bak_{timestamp}"
    print(f"Creating backup: {backup_path}")
    shutil.copy2(file_path, backup_path)
    return backup_path

def load_json_file(file_path):
    """Load and parse JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return None

def save_json_file(file_path, data):
    """Save data to JSON file."""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Successfully saved {file_path}")
        return True
    except Exception as e:
        print(f"Error saving JSON file {file_path}: {e}")
        return False

def update_template_database_json(db_path):
    """Update template_database.json to include Qualcomm hardware support."""
    print(f"Updating template database JSON: {db_path}")
    
    # Load the database
    db_data = load_json_file(db_path)
    if not db_data:
        return False
    
    # Check if templates key exists
    if "templates" not in db_data:
        print(f"Error: Invalid template database format in {db_path}")
        return False
    
    # Check for hardware_platforms and add Qualcomm if needed
    if "hardware_platforms" in db_data:
        if "qualcomm" not in db_data["hardware_platforms"]:
            print("Adding Qualcomm to hardware platforms in database")
            db_data["hardware_platforms"]["qualcomm"] = {
                "name": "Qualcomm AI Engine",
                "description": "Qualcomm AI Engine and Hexagon DSP",
                "priority": 4,
                "available": True
            }
    else:
        # Create hardware_platforms if it doesn't exist
        print("Creating hardware_platforms section with Qualcomm support")
        db_data["hardware_platforms"] = {
            "cpu": {"name": "CPU", "description": "Standard CPU implementation", "priority": 0, "available": True},
            "cuda": {"name": "CUDA", "description": "NVIDIA GPU implementation", "priority": 1, "available": True},
            "rocm": {"name": "ROCm", "description": "AMD GPU implementation", "priority": 2, "available": True},
            "mps": {"name": "MPS", "description": "Apple Silicon GPU implementation", "priority": 3, "available": True},
            "qualcomm": {"name": "Qualcomm AI Engine", "description": "Qualcomm AI Engine and Hexagon DSP", "priority": 4, "available": True},
            "openvino": {"name": "OpenVINO", "description": "Intel hardware acceleration", "priority": 5, "available": True},
            "webnn": {"name": "WebNN", "description": "Web Neural Network API", "priority": 6, "available": True},
            "webgpu": {"name": "WebGPU", "description": "Web GPU API", "priority": 7, "available": True}
        }
    
    # Update templates to include Qualcomm support
    for template_key, template_data in db_data["templates"].items():
        if "supported_hardware" in template_data and "qualcomm" not in template_data["supported_hardware"]:
            print(f"Adding Qualcomm support to template: {template_key}")
            template_data["supported_hardware"].append("qualcomm")
    
    # Save the updated database
    return save_json_file(db_path, db_data)

def update_template_files(template_dir):
    """Update template Python files with Qualcomm handlers."""
    print(f"Updating template files in: {template_dir}")
    
    # Find all Python template files
    template_files = list(Path(template_dir).glob("template_*.py"))
    if not template_files:
        print(f"No template files found in {template_dir}")
        return False
    
    for template_file in template_files:
        # Skip template_database.py (already updated separately)
        if template_file.name == "template_database.py":
            continue
            
        print(f"Processing template file: {template_file}")
        
        # Create backup
        backup_path = backup_file(str(template_file))
        
        # Read template file
        with open(template_file, 'r') as f:
            content = f.read()
        
        # Check if Qualcomm is already included
        if "def get_qualcomm_handler" in content:
            print(f"Qualcomm handler already exists in {template_file}")
            continue
        
        # Add Qualcomm handler implementation
        # First, find the last hardware handler in the file
        handlers = re.findall(r'def get_(\w+)_handler\(model_path.*?(?=def|\Z)', content, re.DOTALL)
        
        if handlers:
            # Find the last hardware handler pattern
            last_handler = handlers[-1]
            pattern = f"def get_{last_handler}_handler"
            last_handler_match = re.search(f"{pattern}.*?(?=def|$)", content, re.DOTALL)
            
            if last_handler_match:
                # Create Qualcomm handler template based on the pattern of the last handler
                last_handler_code = last_handler_match.group(0)
                qualcomm_handler = last_handler_code.replace(f"get_{last_handler}_handler", "get_qualcomm_handler")
                qualcomm_handler = qualcomm_handler.replace(f"{last_handler}", "qualcomm")
                
                # Add Qualcomm-specific features
                qualcomm_handler = qualcomm_handler.replace(
                    "# Initialize the handler", 
                    "# Initialize the Qualcomm AI Engine handler\n    # Check for QNN API"
                )
                
                # Insert the Qualcomm handler after the last handler
                insert_point = last_handler_match.end()
                updated_content = content[:insert_point] + "\n" + qualcomm_handler + content[insert_point:]
                
                # Write updated content
                with open(template_file, 'w') as f:
                    f.write(updated_content)
                
                print(f"Added Qualcomm handler to {template_file}")
            else:
                print(f"Could not locate the full handler definition in {template_file}")
        else:
            print(f"No hardware handlers found in {template_file}")
    
    return True

def update_template_integration(generator_dir):
    """Update integrated_skillset_generator.py to properly handle Qualcomm."""
    print("Updating integrated skillset generator")
    
    # Find integrated_skillset_generator.py
    generator_path = os.path.join(generator_dir, "integrated_skillset_generator.py")
    if not os.path.exists(generator_path):
        print(f"Error: Could not find integrated_skillset_generator.py at {generator_path}")
        return False
    
    # Create backup
    backup_path = backup_file(generator_path)
    
    # Read generator file
    with open(generator_path, 'r') as f:
        content = f.read()
    
    # Check if Qualcomm is already included in supported hardware
    if "qualcomm" in content and "--hardware" in content and "qualcomm" in content.split("--hardware")[1]:
        print("Qualcomm already included in supported hardware")
    else:
        # Update supported hardware to include Qualcomm
        hardware_pattern = r'supported_hardware = \[.*?\]'
        hardware_match = re.search(hardware_pattern, content)
        
        if hardware_match:
            current_hardware = hardware_match.group(0)
            if "qualcomm" not in current_hardware:
                updated_hardware = current_hardware.replace("]", ", \"qualcomm\"]")
                content = content.replace(current_hardware, updated_hardware)
                print("Added Qualcomm to supported hardware list")
    
    # Update hardware detection to include Qualcomm
    detection_pattern = r'hw_capabilities = {.*?}'
    detection_match = re.search(detection_pattern, content, re.DOTALL)
    
    if detection_match:
        capabilities = detection_match.group(0)
        if "\"qualcomm\"" not in capabilities:
            updated_capabilities = capabilities.replace("}", ",\n        \"qualcomm\": False}")
            content = content.replace(capabilities, updated_capabilities)
            print("Added Qualcomm to hardware detection capabilities")
    
    # Update hardware detection code
    detection_code_pattern = r'# Detect available hardware.*?has_webgpu = .*?$'
    detection_code_match = re.search(detection_code_pattern, content, re.MULTILINE | re.DOTALL)
    
    if detection_code_match and "has_qualcomm" not in detection_code_match.group(0):
        detection_code = detection_code_match.group(0)
        # Find position to insert Qualcomm detection
        openvino_line = re.search(r'has_openvino = .*?$', detection_code, re.MULTILINE)
        
        if openvino_line:
            insert_point = openvino_line.end()
            qualcomm_detection = "\n        # Qualcomm detection\n        has_qualcomm = importlib.util.find_spec(\"qnn_wrapper\") is not None or \"QUALCOMM_SDK\" in os.environ"
            updated_detection = detection_code[:insert_point] + qualcomm_detection + detection_code[insert_point:]
            content = content.replace(detection_code, updated_detection)
            print("Added Qualcomm hardware detection code")
    
    # Update handler setup code
    handler_pattern = r'# Set up handlers.*?handlers = {.*?}'
    handler_match = re.search(handler_pattern, content, re.DOTALL)
    
    if handler_match and "\"qualcomm\"" not in handler_match.group(0):
        handler_code = handler_match.group(0)
        handlers_dict = re.search(r'handlers = {.*?}', handler_code, re.DOTALL).group(0)
        updated_handlers = handlers_dict.replace("}", ",\n            \"qualcomm\": get_qualcomm_handler}")
        content = content.replace(handlers_dict, updated_handlers)
        print("Added Qualcomm to handlers dictionary")
    
    # Update get_handler function to include Qualcomm
    get_handler_pattern = r'def get_handler\(platform, model_path.*?# Platform-specific handlers.*?return MockHandler'
    get_handler_match = re.search(get_handler_pattern, content, re.DOTALL)
    
    if get_handler_match and "elif platform == \"qualcomm\"" not in get_handler_match.group(0):
        handler_func = get_handler_match.group(0)
        # Find position to insert Qualcomm handler
        openvino_case = re.search(r'elif platform == "openvino".*?$', handler_func, re.MULTILINE)
        
        if openvino_case:
            insert_point = openvino_case.end()
            qualcomm_case = "\n    elif platform == \"qualcomm\":\n        return get_qualcomm_handler(model_path)"
            updated_func = handler_func[:insert_point] + qualcomm_case + handler_func[insert_point:]
            content = content.replace(handler_func, updated_func)
            print("Added Qualcomm case to get_handler function")
    
    # Write updated content
    with open(generator_path, 'w') as f:
        f.write(content)
    
    print(f"Updated {generator_path} with Qualcomm support")
    return True

def create_qualcomm_template_example(template_dir):
    """Create or update an example template for Qualcomm implementation."""
    bert_template_path = os.path.join(template_dir, "template_bert.py")
    
    if not os.path.exists(bert_template_path):
        print(f"Error: Could not find BERT template at {bert_template_path}")
        return False
    
    # Create backup
    backup_path = backup_file(bert_template_path)
    
    # Read BERT template
    with open(bert_template_path, 'r') as f:
        content = f.read()
    
    # Check if Qualcomm is already included in the template
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
            return MockHandler(model_path, "qualcomm")
    
    # Initialize the Qualcomm AI Engine handler
    class QualcommHandler:
        def __init__(self, model_path):
            self.model_path = model_path
            self.backend = "qualcomm"
            self.tokenizer = None
            self.model = None
            self.initialized = False
            self.compile_model()
        
        def compile_model(self):
            """Compile model for Qualcomm AI Engine."""
            try:
                from transformers import AutoTokenizer, AutoModel
                
                print(f"Loading tokenizer from {self.model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                
                print(f"Loading model from {self.model_path}")
                model = AutoModel.from_pretrained(self.model_path)
                
                # Export to ONNX format first
                import torch
                import os
                
                # Create sample input
                input_names = ["input_ids", "attention_mask", "token_type_ids"]
                output_names = ["last_hidden_state", "pooler_output"]
                
                # Create temp directory for ONNX file
                import tempfile
                temp_dir = tempfile.mkdtemp()
                onnx_path = os.path.join(temp_dir, "bert_model.onnx")
                
                # Create dummy input based on tokenizer max length
                batch_size = 1
                max_length = self.tokenizer.model_max_length if hasattr(self.tokenizer, "model_max_length") else 512
                
                dummy_input_ids = torch.zeros((batch_size, max_length), dtype=torch.long)
                dummy_attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
                dummy_token_type_ids = torch.zeros((batch_size, max_length), dtype=torch.long)
                
                # Export to ONNX
                torch.onnx.export(
                    model,
                    (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids),
                    onnx_path,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes={
                        "input_ids": {0: "batch_size", 1: "sequence_length"},
                        "attention_mask": {0: "batch_size", 1: "sequence_length"},
                        "token_type_ids": {0: "batch_size", 1: "sequence_length"},
                        "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
                        "pooler_output": {0: "batch_size"}
                    }
                )
                
                # Import the model in Qualcomm format
                if HAS_QNN:
                    # Using QNN API
                    qnn_model_path = os.path.join(temp_dir, "bert_model.bin")
                    
                    # Convert ONNX to Qualcomm binary format
                    qnn_wrapper.convert_model(
                        input_model=onnx_path,
                        output_model=qnn_model_path,
                        input_list=input_names
                    )
                    
                    # Load the QNN model
                    self.model = qnn_wrapper.QnnModel(qnn_model_path)
                else:
                    # Using QTI SDK
                    dlc_path = os.path.join(temp_dir, "bert_model.dlc")
                    
                    # Convert ONNX to Qualcomm DLC format
                    qti_utils.convert_onnx_to_dlc(
                        input_model=onnx_path,
                        output_model=dlc_path
                    )
                    
                    # Load the QTI model
                    import qti.aisw.dlc_runner as qti_runner
                    self.model = qti_runner.DlcRunner(dlc_path)
                
                self.initialized = True
                print(f"Successfully initialized BERT model on Qualcomm AI Engine")
                
            except Exception as e:
                print(f"Error initializing Qualcomm handler: {e}")
                self.initialized = False
        
        def __call__(self, text, **kwargs):
            """
            Process text using BERT model on Qualcomm AI Engine.
            
            Args:
                text: Input text to process
                **kwargs: Additional parameters
                
            Returns:
                Dict with embeddings and implementation info
            """
            if not self.initialized:
                print("Warning: Model not initialized, using mock response")
                return {"embeddings": [0.0] * 768, "implementation_type": "MOCK_QUALCOMM"}
            
            try:
                # Tokenize input
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                
                if HAS_QNN:
                    # Using QNN API
                    qnn_inputs = {
                        "input_ids": inputs["input_ids"].numpy(),
                        "attention_mask": inputs["attention_mask"].numpy(),
                        "token_type_ids": inputs["token_type_ids"].numpy() if "token_type_ids" in inputs else None
                    }
                    
                    # Remove None inputs
                    qnn_inputs = {k: v for k, v in qnn_inputs.items() if v is not None}
                    
                    # Run inference
                    outputs = self.model.execute(qnn_inputs)
                    
                    # Process outputs
                    embeddings = outputs["pooler_output"]
                else:
                    # Using QTI SDK
                    qti_inputs = [
                        inputs["input_ids"].numpy(),
                        inputs["attention_mask"].numpy()
                    ]
                    
                    # Add token_type_ids if available
                    if "token_type_ids" in inputs:
                        qti_inputs.append(inputs["token_type_ids"].numpy())
                    
                    # Run inference
                    outputs = self.model.execute(qti_inputs)
                    
                    # Get pooler output (second tensor in the outputs)
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
    openvino_pos = content.find(openvino_handler)
    
    if openvino_pos > 0:
        # Find the end of the openvino handler
        next_def_pos = content.find("def ", openvino_pos + len(openvino_handler))
        
        if next_def_pos > 0:
            # Insert before the next def
            updated_content = content[:next_def_pos] + qualcomm_handler + "\n\n" + content[next_def_pos:]
        else:
            # Insert at the end
            updated_content = content + "\n\n" + qualcomm_handler
    else:
        # If openvino handler not found, just append to the end
        updated_content = content + "\n\n" + qualcomm_handler
    
    # Write updated content
    with open(bert_template_path, 'w') as f:
        f.write(updated_content)
    
    print(f"Added complete Qualcomm handler example to {bert_template_path}")
    return True

def main():
    """Main function to update template database for Qualcomm support"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Template database paths
    template_db_json = os.path.join(current_dir, "hardware_test_templates", "template_database.json")
    template_dir = os.path.join(current_dir, "hardware_test_templates")
    
    # Check if template directory exists
    if not os.path.exists(template_dir):
        print(f"Error: Template directory not found at {template_dir}")
        sys.exit(1)
    
    print("Starting template database update for Qualcomm support")
    
    # Update JSON file
    if os.path.exists(template_db_json):
        success = update_template_database_json(template_db_json)
        if not success:
            print("Warning: Failed to update template database JSON")
    else:
        print(f"Warning: Template database JSON not found at {template_db_json}")
    
    # Update template Python files
    success = update_template_files(template_dir)
    if not success:
        print("Warning: Failed to update template files")
    
    # Create example Qualcomm implementation for BERT
    success = create_qualcomm_template_example(template_dir)
    if not success:
        print("Warning: Failed to create Qualcomm template example")
    
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