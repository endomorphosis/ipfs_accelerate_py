#!/usr/bin/env python3
"""
Script to identify and generate missing test files for models defined in mapped_models.json.
This will create standardized test files for any models that don't already have test implementations.
"""

import os
import sys
import json
import glob

def get_mapped_models():
    """Load the model mapping from mapped_models.json"""
    try:
        with open('mapped_models.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading mapped_models.json: {e}")
        return {}

def get_existing_test_files():
    """Get list of existing test files for HF models"""
    # Normalize paths to skill names
    test_files = {}
    
    # Scan for test_hf_*.py files
    for file_path in glob.glob("skills/test_hf_*.py"):
        # Extract model name from filename
        filename = os.path.basename(file_path)
        skill_name = filename.replace("test_hf_", "").replace(".py", "")
        test_files[skill_name] = file_path
    
    return test_files

def identify_missing_tests(mapped_models, existing_tests):
    """Find models that don't have test implementations"""
    missing = {}
    
    for skill, model in mapped_models.items():
        # Special cases for skills that don't follow test_hf_<skill> pattern
        if skill == "bert" and "bert" in existing_tests:
            continue
        if skill == "distilbert" and "distilbert" in existing_tests:
            continue
        if skill == "roberta" and "roberta" in existing_tests:
            continue
        
        # Convert dashes to underscores in skill name (used in filenames)
        skill_clean = skill.replace("-", "_")
        
        if skill_clean not in existing_tests:
            missing[skill] = model
    
    return missing

def create_test_template(skill, model):
    """Create a standardized test implementation for the given skill and model"""
    
    # Handle skills with dashes in their names (like deberta-v2)
    skill_clean = skill.replace("-", "_")
    class_name = f"test_hf_{skill_clean}"
    filename = f"skills/{class_name}.py"
    
    print(f"Creating test file for {skill}: {filename}")
    
    # Special handling for model types
    if skill in ["bert", "distilbert", "albert", "roberta", "mpnet", "electra", "camembert", 
                 "flaubert", "xlm-roberta", "mobilebert", "deberta", "deberta-v2", "squeezebert", "layoutlm"]:
        model_type = "text-embedding"
        implementation_class = "hf_bert"
        import_path = "ipfs_accelerate_py.worker.skillset.hf_bert"
        test_input = "\"This is a test sentence for embedding models.\""
        test_output_type = "embedding"
        
    elif skill in ["gpt2", "gpt_neo", "gptj", "llama", "bloom", "opt", "codegen", "qwen2"]:
        model_type = "text-generation"
        implementation_class = "hf_lm"
        import_path = "ipfs_accelerate_py.worker.skillset.hf_lm"
        test_input = "\"Once upon a time\""
        test_output_type = "generated_text"
        
    elif skill in ["t5", "mt5", "bart", "mbart", "pegasus", "led", "blenderbot", "blenderbot-small"]:
        model_type = "sequence-to-sequence"
        implementation_class = "hf_t5"
        import_path = "ipfs_accelerate_py.worker.skillset.hf_t5"
        test_input = "\"translate English to German: Hello, how are you?\""
        test_output_type = "text"
        
    elif skill == "whisper":
        model_type = "speech-to-text"
        implementation_class = "hf_whisper"
        import_path = "ipfs_accelerate_py.worker.skillset.hf_whisper"
        test_input = "\"test.mp3\""
        test_output_type = "text"
        
    elif skill == "wav2vec2":
        model_type = "speech-recognition"
        implementation_class = "hf_wav2vec2"
        import_path = "ipfs_accelerate_py.worker.skillset.hf_wav2vec2"
        test_input = "\"test.mp3\""
        test_output_type = "text"
        
    elif skill == "clip":
        model_type = "image-text-similarity"
        implementation_class = "hf_clip"
        import_path = "ipfs_accelerate_py.worker.skillset.hf_clip"
        test_input = "\"test.jpg\""
        test_output_type = "similarity"
        
    elif skill == "xclip":
        model_type = "video-text-similarity"
        implementation_class = "hf_xclip"
        import_path = "ipfs_accelerate_py.worker.skillset.hf_xclip"
        test_input = "\"test.mp4\""
        test_output_type = "similarity"
        
    elif skill == "clap":
        model_type = "audio-text-similarity"
        implementation_class = "hf_clap" 
        import_path = "ipfs_accelerate_py.worker.skillset.hf_clap"
        test_input = "\"test.mp3\""
        test_output_type = "similarity"
        
    elif skill in ["llava", "llava_next", "qwen2_vl"]:
        model_type = "vision-language"
        implementation_class = "hf_llava" if skill == "llava" else "hf_llava_next"
        import_path = "ipfs_accelerate_py.worker.skillset.hf_llava" if skill == "llava" else "ipfs_accelerate_py.worker.skillset.hf_llava_next"
        test_input = "{\"image\": \"test.jpg\", \"text\": \"What is in this image?\"}"
        test_output_type = "generated_text"
        
    elif skill in ["vit", "deit", "swin", "convnext", "detr"]:
        model_type = "image-classification"
        implementation_class = "hf_vit"
        import_path = "ipfs_accelerate_py.worker.skillset.hf_vit"
        test_input = "\"test.jpg\""
        test_output_type = "logits"
        
    elif skill == "hubert":
        model_type = "audio-classification"
        implementation_class = "hf_hubert"
        import_path = "ipfs_accelerate_py.worker.skillset.hf_hubert"
        test_input = "\"test.mp3\""
        test_output_type = "logits"
        
    elif skill == "videomae":
        model_type = "video-classification"
        implementation_class = "hf_videomae"
        import_path = "ipfs_accelerate_py.worker.skillset.hf_videomae"
        test_input = "\"test.mp4\""
        test_output_type = "logits"
        
    else:
        model_type = "unknown"
        implementation_class = f"hf_{skill_clean}"
        import_path = f"ipfs_accelerate_py.worker.skillset.hf_{skill_clean}"
        test_input = "\"Generic test input for model.\""
        test_output_type = "output"
    
    # Create a standardized test file based on the template
    test_file_content = f"""# Standard library imports first
import os
import sys
import json
import time
import datetime
import traceback
from unittest.mock import patch, MagicMock

# Third-party imports next
import numpy as np

# Use absolute path setup
sys.path.insert(0, "/home/barberb/ipfs_accelerate_py")

# Try/except pattern for importing optional dependencies
try:
    import torch
except ImportError:
    torch = MagicMock()
    print("Warning: torch not available, using mock implementation")

try:
    import transformers
except ImportError:
    transformers = MagicMock()
    print("Warning: transformers not available, using mock implementation")

# Import the module to test
from {import_path} import {implementation_class}

class {class_name}:
    def __init__(self, resources=None, metadata=None):
        \"\"\"
        Initialize the {skill} test class.
        
        Args:
            resources (dict, optional): Resources dictionary
            metadata (dict, optional): Metadata dictionary
        \"\"\"
        self.resources = resources if resources else {{
            "torch": torch,
            "numpy": np,
            "transformers": transformers
        }}
        self.metadata = metadata if metadata else {{}}
        self.{skill_clean} = {implementation_class}(resources=self.resources, metadata=self.metadata)
        
        # Use the model from mapped_models.json
        self.model_name = "{model}"
        
        # Alternative models in increasing size order (if needed)
        # self.alternative_models = []
        
        print(f"Using model: {{self.model_name}}")
        self.test_input = {test_input}
        
        # Initialize collection arrays for examples and status
        self.examples = []
        self.status_messages = {{}}
        return None
    
    def test(self):
        \"\"\"
        Run all tests for the {skill} model, organized by hardware platform.
        Tests CPU, CUDA, and OpenVINO implementations.
        
        Returns:
            dict: Structured test results with status, examples and metadata
        \"\"\"
        results = {{}}
        
        # Test basic initialization
        try:
            results["init"] = "Success" if self.{skill_clean} is not None else "Failed initialization"
        except Exception as e:
            results["init"] = f"Error: {{str(e)}}"

        # ====== CPU TESTS ======
        try:
            print("Testing {skill} on CPU...")
            # Initialize for CPU
            endpoint, tokenizer, handler, queue, batch_size = self.{skill_clean}.init_cpu(
                self.model_name,
                "{model_type}", 
                "cpu"
            )
            
            valid_init = endpoint is not None and tokenizer is not None and handler is not None
            results["cpu_init"] = "Success (REAL)" if valid_init else "Failed CPU initialization"
            
            # Get handler for CPU directly from initialization
            test_handler = handler
            
            # Run actual inference
            start_time = time.time()
            output = test_handler(self.test_input)
            elapsed_time = time.time() - start_time
            
            # Verify the output
            is_valid_output = output is not None
            
            results["cpu_handler"] = "Success (REAL)" if is_valid_output else "Failed CPU handler"
            
            # Record example
            output_info = {{}}
            if is_valid_output:
                if isinstance(output, dict):
                    if "{test_output_type}" in output:
                        output_info["{test_output_type}_type"] = str(type(output["{test_output_type}"]))
                        if hasattr(output["{test_output_type}"], "shape"):
                            output_info["{test_output_type}_shape"] = list(output["{test_output_type}"].shape)
                elif hasattr(output, "shape"):
                    output_info["output_shape"] = list(output.shape)
            
            self.examples.append({{
                "input": self.test_input,
                "output": output_info,
                "timestamp": datetime.datetime.now().isoformat(),
                "elapsed_time": elapsed_time,
                "implementation_type": "REAL",
                "platform": "CPU"
            }})
            
        except Exception as e:
            print(f"Error in CPU tests: {{e}}")
            traceback.print_exc()
            results["cpu_tests"] = f"Error: {{str(e)}}"
            self.status_messages["cpu"] = f"Failed: {{str(e)}}"

        # ====== CUDA TESTS ======
        if torch.cuda.is_available():
            try:
                print("Testing {skill} on CUDA...")
                # Initialize for CUDA
                endpoint, tokenizer, handler, queue, batch_size = self.{skill_clean}.init_cuda(
                    self.model_name,
                    "{model_type}",
                    "cuda:0"
                )
                
                # Check if initialization succeeded
                valid_init = endpoint is not None and tokenizer is not None and handler is not None
                
                # Check for mock vs real implementation
                is_mock_endpoint = False
                implementation_type = "(REAL)"  # Default to REAL
                
                # Check for various indicators of mock implementations
                if isinstance(endpoint, MagicMock):
                    is_mock_endpoint = True
                    implementation_type = "(MOCK)"
                
                # Update the result status with proper implementation type
                results["cuda_init"] = f"Success {{implementation_type}}" if valid_init else f"Failed CUDA initialization"
                self.status_messages["cuda"] = f"Ready {{implementation_type}}" if valid_init else "Failed initialization"
                
                print(f"CUDA initialization: {{results['cuda_init']}}")
                
                # Run actual inference
                start_time = time.time()
                output = handler(self.test_input)
                elapsed_time = time.time() - start_time
                
                # Check if output is valid
                is_valid_output = output is not None
                
                # Record example
                output_info = {{}}
                if is_valid_output:
                    if isinstance(output, dict):
                        if "{test_output_type}" in output:
                            output_info["{test_output_type}_type"] = str(type(output["{test_output_type}"]))
                            if hasattr(output["{test_output_type}"], "shape"):
                                output_info["{test_output_type}_shape"] = list(output["{test_output_type}"].shape)
                    elif hasattr(output, "shape"):
                        output_info["output_shape"] = list(output.shape)
                
                # Determine implementation type from output
                if isinstance(output, dict) and "implementation_type" in output:
                    implementation_type = f"({{output['implementation_type']}})"
                
                results["cuda_handler"] = f"Success {{implementation_type}}" if is_valid_output else f"Failed CUDA handler"
                
                self.examples.append({{
                    "input": self.test_input,
                    "output": output_info,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "elapsed_time": elapsed_time,
                    "implementation_type": implementation_type.strip("()"),
                    "platform": "CUDA"
                }})
                
            except Exception as e:
                print(f"Error in CUDA tests: {{e}}")
                traceback.print_exc()
                results["cuda_tests"] = f"Error: {{str(e)}}"
                self.status_messages["cuda"] = f"Failed: {{str(e)}}"
        else:
            results["cuda_tests"] = "CUDA not available"
            self.status_messages["cuda"] = "CUDA not available"

        # ====== OPENVINO TESTS ======
        try:
            # First check if OpenVINO is installed
            try:
                import openvino
                has_openvino = True
                print("OpenVINO is installed")
            except ImportError:
                has_openvino = False
                results["openvino_tests"] = "OpenVINO not installed"
                self.status_messages["openvino"] = "OpenVINO not installed"
                
            if has_openvino:
                # Import the existing OpenVINO utils from the main package
                from ipfs_accelerate_py.worker.openvino_utils import openvino_utils
                
                # Initialize openvino_utils
                ov_utils = openvino_utils(resources=self.resources, metadata=self.metadata)
                
                # Try with real OpenVINO initialization
                try:
                    print("Trying real OpenVINO initialization...")
                    endpoint, tokenizer, handler, queue, batch_size = self.{skill_clean}.init_openvino(
                        model_name=self.model_name,
                        model_type="{model_type}",
                        device="CPU",
                        openvino_label="openvino:0",
                        get_optimum_openvino_model=ov_utils.get_optimum_openvino_model,
                        get_openvino_model=ov_utils.get_openvino_model,
                        get_openvino_pipeline_type=ov_utils.get_openvino_pipeline_type,
                        openvino_cli_convert=ov_utils.openvino_cli_convert
                    )
                    
                    # If we got a handler back, we succeeded
                    valid_init = handler is not None
                    is_real_impl = True
                    results["openvino_init"] = "Success (REAL)" if valid_init else "Failed OpenVINO initialization"
                    print(f"Real OpenVINO initialization: {{results['openvino_init']}}")
                    
                except Exception as e:
                    print(f"Real OpenVINO initialization failed: {{e}}")
                    print("Falling back to mock implementation...")
                    
                    # Create mock OpenVINO handlers
                    mock_get_openvino_model = MagicMock()
                    mock_get_optimum_openvino_model = MagicMock()
                    mock_get_openvino_pipeline_type = MagicMock()
                    mock_get_openvino_pipeline_type.return_value = "{model_type}"
                    mock_openvino_cli_convert = MagicMock()
                    mock_openvino_cli_convert.return_value = True
                    
                    # Fall back to mock implementation
                    endpoint, tokenizer, handler, queue, batch_size = self.{skill_clean}.init_openvino(
                        model_name=self.model_name,
                        model_type="{model_type}",
                        device="CPU",
                        openvino_label="openvino:0",
                        get_optimum_openvino_model=mock_get_optimum_openvino_model,
                        get_openvino_model=mock_get_openvino_model,
                        get_openvino_pipeline_type=mock_get_openvino_pipeline_type,
                        openvino_cli_convert=mock_openvino_cli_convert
                    )
                    
                    # If we got a handler back, the mock succeeded
                    valid_init = handler is not None
                    is_real_impl = False
                    results["openvino_init"] = "Success (MOCK)" if valid_init else "Failed OpenVINO initialization"
                
                # Run inference
                if valid_init and handler is not None:
                    start_time = time.time()
                    output = handler(self.test_input)
                    elapsed_time = time.time() - start_time
                    
                    is_valid_output = output is not None
                    
                    # Set the appropriate success message based on real vs mock implementation
                    implementation_type = "REAL" if is_real_impl else "MOCK"
                    results["openvino_handler"] = f"Success ({{implementation_type}})" if is_valid_output else f"Failed OpenVINO handler"
                    
                    # Record example
                    output_info = {{}}
                    if is_valid_output:
                        if isinstance(output, dict):
                            if "{test_output_type}" in output:
                                output_info["{test_output_type}_type"] = str(type(output["{test_output_type}"]))
                                if hasattr(output["{test_output_type}"], "shape"):
                                    output_info["{test_output_type}_shape"] = list(output["{test_output_type}"].shape)
                        elif hasattr(output, "shape"):
                            output_info["output_shape"] = list(output.shape)
                    
                    self.examples.append({{
                        "input": self.test_input,
                        "output": output_info,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "elapsed_time": elapsed_time,
                        "implementation_type": implementation_type,
                        "platform": "OpenVINO"
                    }})
                
        except ImportError:
            results["openvino_tests"] = "OpenVINO not installed"
            self.status_messages["openvino"] = "OpenVINO not installed"
        except Exception as e:
            print(f"Error in OpenVINO tests: {{e}}")
            traceback.print_exc()
            results["openvino_tests"] = f"Error: {{str(e)}}"
            self.status_messages["openvino"] = f"Failed: {{str(e)}}"

        # Create structured results with status, examples and metadata
        structured_results = {{
            "status": results,
            "examples": self.examples,
            "metadata": {{
                "model_name": self.model_name,
                "test_timestamp": datetime.datetime.now().isoformat(),
                "python_version": sys.version,
                "torch_version": torch.__version__ if hasattr(torch, "__version__") else "Unknown",
                "transformers_version": transformers.__version__ if hasattr(transformers, "__version__") else "Unknown",
                "platform_status": self.status_messages
            }}
        }}

        return structured_results

    def __test__(self):
        \"\"\"
        Run tests and compare/save results.
        Handles result collection, comparison with expected results, and storage.
        
        Returns:
            dict: Test results
        \"\"\"
        test_results = {{}}
        try:
            test_results = self.test()
        except Exception as e:
            test_results = {{
                "status": {{"test_error": str(e)}},
                "examples": [],
                "metadata": {{
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }}
            }}
        
        # Create directories if they don't exist
        base_dir = os.path.dirname(os.path.abspath(__file__))
        expected_dir = os.path.join(base_dir, 'expected_results')
        collected_dir = os.path.join(base_dir, 'collected_results')
        
        # Create directories with appropriate permissions
        for directory in [expected_dir, collected_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory, mode=0o755, exist_ok=True)
        
        # Save collected results
        results_file = os.path.join(collected_dir, 'hf_{skill_clean}_test_results.json')
        try:
            with open(results_file, 'w') as f:
                json.dump(test_results, f, indent=2)
            print(f"Saved collected results to {{results_file}}")
        except Exception as e:
            print(f"Error saving results to {{results_file}}: {{str(e)}}")
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'hf_{skill_clean}_test_results.json')
        if os.path.exists(expected_file):
            try:
                with open(expected_file, 'r') as f:
                    expected_results = json.load(f)
                
                # Filter out variable fields for comparison
                def filter_variable_data(result):
                    if isinstance(result, dict):
                        # Create a copy to avoid modifying the original
                        filtered = {{}}
                        for k, v in result.items():
                            # Skip timestamp and variable output data for comparison
                            if k not in ["timestamp", "elapsed_time", "output"] and k != "examples" and k != "metadata":
                                filtered[k] = filter_variable_data(v)
                        return filtered
                    elif isinstance(result, list):
                        return [filter_variable_data(item) for item in result]
                    else:
                        return result
                
                # Compare only status keys for backward compatibility
                status_expected = expected_results.get("status", expected_results)
                status_actual = test_results.get("status", test_results)
                
                # More detailed comparison of results
                all_match = True
                mismatches = []
                
                for key in set(status_expected.keys()) | set(status_actual.keys()):
                    if key not in status_expected:
                        mismatches.append(f"Missing expected key: {{key}}")
                        all_match = False
                    elif key not in status_actual:
                        mismatches.append(f"Missing actual key: {{key}}")
                        all_match = False
                    elif status_expected[key] != status_actual[key]:
                        # If the only difference is the implementation_type suffix, that's acceptable
                        if (
                            isinstance(status_expected[key], str) and 
                            isinstance(status_actual[key], str) and
                            status_expected[key].split(" (")[0] == status_actual[key].split(" (")[0] and
                            "Success" in status_expected[key] and "Success" in status_actual[key]
                        ):
                            continue
                        
                        mismatches.append(f"Key '{{key}}' differs: Expected '{{status_expected[key]}}', got '{{status_actual[key]}}'")
                        all_match = False
                
                if not all_match:
                    print("Test results differ from expected results!")
                    for mismatch in mismatches:
                        print(f"- {{mismatch}}")
                    print("\\nWould you like to update the expected results? (y/n)")
                    user_input = input().strip().lower()
                    if user_input == 'y':
                        with open(expected_file, 'w') as ef:
                            json.dump(test_results, ef, indent=2)
                            print(f"Updated expected results file: {{expected_file}}")
                    else:
                        print("Expected results not updated.")
                else:
                    print("All test results match expected results.")
            except Exception as e:
                print(f"Error comparing results with {{expected_file}}: {{str(e)}}")
                print("Creating new expected results file.")
                with open(expected_file, 'w') as ef:
                    json.dump(test_results, ef, indent=2)
        else:
            # Create expected results file if it doesn't exist
            try:
                with open(expected_file, 'w') as f:
                    json.dump(test_results, f, indent=2)
                    print(f"Created new expected results file: {{expected_file}}")
            except Exception as e:
                print(f"Error creating {{expected_file}}: {{str(e)}}")

        return test_results

if __name__ == "__main__":
    try:
        print("Starting {skill} test...")
        this_{skill_clean} = {class_name}()
        results = this_{skill_clean}.__test__()
        print("{skill} test completed")
        
        # Print test results summary
        status_dict = results.get("status", {{}})
        examples = results.get("examples", [])
        metadata = results.get("metadata", {{}})
        
        # Extract implementation status
        cpu_status = "UNKNOWN"
        cuda_status = "UNKNOWN"
        openvino_status = "UNKNOWN"
        
        for key, value in status_dict.items():
            if "cpu_" in key and "REAL" in value:
                cpu_status = "REAL"
            elif "cpu_" in key and "MOCK" in value:
                cpu_status = "MOCK"
                
            if "cuda_" in key and "REAL" in value:
                cuda_status = "REAL"
            elif "cuda_" in key and "MOCK" in value:
                cuda_status = "MOCK"
                
            if "openvino_" in key and "REAL" in value:
                openvino_status = "REAL"
            elif "openvino_" in key and "MOCK" in value:
                openvino_status = "MOCK"
        
        # Print summary
        print("\\n{skill.upper()} TEST RESULTS SUMMARY")
        print(f"MODEL: {{metadata.get('model_name', 'Unknown')}}")
        print(f"CPU_STATUS: {{cpu_status}}")
        print(f"CUDA_STATUS: {{cuda_status}}")
        print(f"OPENVINO_STATUS: {{openvino_status}}")
        
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during testing: {{str(e)}}")
        traceback.print_exc()
        sys.exit(1)
"""
    
    # Create the directory if needed
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Write the test file
    with open(filename, "w") as f:
        f.write(test_file_content)
        
    return filename

def main():
    print("Generating missing test files for models in mapped_models.json")
    
    # Get mapped models
    mapped_models = get_mapped_models()
    print(f"Found {len(mapped_models)} models in mapped_models.json")
    
    # Get existing test files
    existing_tests = get_existing_test_files()
    print(f"Found {len(existing_tests)} existing test files")
    
    # Identify missing tests
    missing_tests = identify_missing_tests(mapped_models, existing_tests)
    print(f"Found {len(missing_tests)} missing test files")
    
    # Create missing test files
    created_files = []
    
    if missing_tests:
        print("\nGenerating missing test files:")
        for skill, model in missing_tests.items():
            try:
                filename = create_test_template(skill, model)
                created_files.append(filename)
                print(f"✅ Created {filename}")
            except Exception as e:
                print(f"❌ Error creating test for {skill}: {e}")
                
        print(f"\nCreated {len(created_files)} new test files")
    else:
        print("No missing test files to generate")
    
    return created_files

if __name__ == "__main__":
    main()