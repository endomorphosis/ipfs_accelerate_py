#!/usr/bin/env python3
"""
Advanced Hugging Face model test generator that works reliably and creates high-quality tests.
"""

import os
import sys
import json
import glob
import time
import argparse
import logging
from pathlib import Path

# Constants
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
SKILLS_DIR = CURRENT_DIR / "skills"
TEST_FILE_PREFIX = "test_hf_"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def normalize_model_name(name):
    """Normalize model name to match file naming conventions."""
    return name.replace('-', '_').replace('.', '_').lower()

def get_existing_tests():
    """Find all existing test files."""
    test_files = glob.glob(str(SKILLS_DIR / f'{TEST_FILE_PREFIX}*.py'))
    existing_tests = {}
    
    for test_file in test_files:
        model_name = os.path.basename(test_file).replace(TEST_FILE_PREFIX, '').replace('.py', '')
        existing_tests[model_name] = test_file
    
    return existing_tests

def load_model_data():
    """Load model data from JSON files."""
    try:
        # Load models
        with open('huggingface_model_types.json', 'r') as f:
            all_models = json.load(f)
        
        # Load pipeline mappings
        with open('huggingface_model_pipeline_map.json', 'r') as f:
            model_to_pipeline = json.load(f)
        
        with open('huggingface_pipeline_model_map.json', 'r') as f:
            pipeline_to_model = json.load(f)
            
        return all_models, model_to_pipeline, pipeline_to_model
    except Exception as e:
        logger.error(f"Error loading model data: {e}")
        raise

def get_primary_task(model, pipeline_tasks):
    """Determine the primary task for a model."""
    # For models with tasks, use the first task
    if pipeline_tasks:
        return pipeline_tasks[0]
    
    # Default task
    return "feature-extraction"

def get_model_category(tasks):
    """Determine the model category based on tasks."""
    task_set = set(tasks)
    
    # Language models
    language_tasks = {"text-generation", "fill-mask", "question-answering", "token-classification"}
    if task_set.intersection(language_tasks):
        return "language"
    
    # Vision models
    vision_tasks = {"image-classification", "object-detection", "image-segmentation"}
    if task_set.intersection(vision_tasks):
        return "vision"
    
    # Audio models
    audio_tasks = {"automatic-speech-recognition", "audio-classification", "text-to-audio"}
    if task_set.intersection(audio_tasks):
        return "audio"
    
    # Multimodal models
    multimodal_tasks = {"image-to-text", "visual-question-answering", "document-question-answering"}
    if task_set.intersection(multimodal_tasks):
        return "multimodal"
    
    # Default to language
    return "language"

def get_example_model(category, primary_task):
    """Get example model for a category and primary task."""
    if category == "language":
        if primary_task == "text-generation":
            return "distilgpt2"
        elif primary_task == "fill-mask":
            return "distilroberta-base"
        elif primary_task == "question-answering":
            return "distilbert-base-cased-distilled-squad"
        else:
            return "bert-base-uncased"
    elif category == "vision":
        if primary_task == "image-classification":
            return "google/vit-base-patch16-224-in21k"
        elif primary_task == "object-detection":
            return "facebook/detr-resnet-50"
        else:
            return "google/vit-base-patch16-224-in21k"
    elif category == "audio":
        if primary_task == "automatic-speech-recognition":
            return "openai/whisper-tiny"
        else:
            return "facebook/wav2vec2-base"
    elif category == "multimodal":
        if primary_task == "image-to-text":
            return "Salesforce/blip-image-captioning-base"
        elif primary_task == "visual-question-answering":
            return "Salesforce/blip-vqa-base"
        else:
            return "microsoft/layoutlm-base-uncased"
    else:
        return "bert-base-uncased"

def get_test_inputs(category, primary_task):
    """Get appropriate test inputs based on category and task."""
    inputs = []
    
    if category == "language":
        inputs.append('self.test_text = "The quick brown fox jumps over the lazy dog"')
        inputs.append('self.test_batch = ["The quick brown fox jumps over the lazy dog", "The five boxing wizards jump quickly"]')
        
        if primary_task == "question-answering":
            inputs.append('self.test_qa = {"question": "What is the capital of France?", "context": "Paris is the capital and most populous city of France."}')
    
    elif category == "vision":
        inputs.append('self.test_image_path = "test.jpg"')
        inputs.append('try:\n    from PIL import Image\n    self.test_image = Image.open("test.jpg") if os.path.exists("test.jpg") else None\nexcept ImportError:\n    self.test_image = None')
    
    elif category == "audio":
        inputs.append('self.test_audio_path = "test.mp3"')
        inputs.append('try:\n    import librosa\n    self.test_audio, self.test_sr = librosa.load("test.mp3", sr=16000) if os.path.exists("test.mp3") else (None, 16000)\nexcept ImportError:\n    self.test_audio, self.test_sr = None, 16000')
    
    elif category == "multimodal":
        inputs.append('self.test_image_path = "test.jpg"')
        if primary_task == "visual-question-answering":
            inputs.append('self.test_vqa = {"image": "test.jpg", "question": "What is shown in this image?"}')
        elif primary_task == "document-question-answering":
            inputs.append('self.test_document_qa = {"image": "test.jpg", "question": "What is the title of this document?"}')
    
    # Default fallback
    inputs.append('self.test_input = "Default test input"')
    
    return inputs

def generate_test_file(model, normalized_name, pipeline_tasks, output_dir):
    """Generate a test file for a model."""
    # Get basic model info
    primary_task = get_primary_task(model, pipeline_tasks)
    category = get_model_category(pipeline_tasks)
    example_model = get_example_model(category, primary_task)
    test_inputs = get_test_inputs(category, primary_task)
    test_inputs_str = "\n        ".join(test_inputs)
    
    # Generate the test file content
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    content = f"""#!/usr/bin/env python3
# Test file for {model}
# Generated: {timestamp}
# Category: {category}
# Primary task: {primary_task}

import os
import sys
import json
import time
import datetime
import traceback
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party imports
import numpy as np

# Try optional dependencies
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = MagicMock()
    HAS_TORCH = False
    print("Warning: torch not available, using mock")

try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = MagicMock()
    HAS_TRANSFORMERS = False
    print("Warning: transformers not available, using mock")

# Category-specific imports
if "{category}" in ["vision", "multimodal"]:
    try:
        from PIL import Image
        HAS_PIL = True
    except ImportError:
        Image = MagicMock()
        HAS_PIL = False
        print("Warning: PIL not available, using mock")

if "{category}" == "audio":
    try:
        import librosa
        HAS_LIBROSA = True
    except ImportError:
        librosa = MagicMock()
        HAS_LIBROSA = False
        print("Warning: librosa not available, using mock")

# Try to import the model implementation
try:
    from ipfs_accelerate_py.worker.skillset.hf_{normalized_name} import hf_{normalized_name}
    HAS_IMPLEMENTATION = True
except ImportError:
    # Create mock implementation
    class hf_{normalized_name}:
        def __init__(self, resources=None, metadata=None):
            self.resources = resources or {{}}
            self.metadata = metadata or {{}}
            
        def init_cpu(self, model_name, model_type, device="cpu", **kwargs):
            # Mock implementation
            return None, None, lambda x: {{"output": "Mock output", "implementation_type": "MOCK"}}, None, 1
            
        def init_cuda(self, model_name, model_type, device_label="cuda:0", **kwargs):
            # Mock implementation
            return None, None, lambda x: {{"output": "Mock output", "implementation_type": "MOCK"}}, None, 1
            
        def init_openvino(self, model_name, model_type, device="CPU", **kwargs):
            # Mock implementation
            return None, None, lambda x: {{"output": "Mock output", "implementation_type": "MOCK"}}, None, 1
    
    HAS_IMPLEMENTATION = False
    print(f"Warning: hf_{normalized_name} module not found, using mock implementation")

class test_hf_{normalized_name}:
    def __init__(self, resources=None, metadata=None):
        # Initialize resources
        self.resources = resources if resources else {{
            "torch": torch,
            "numpy": np,
            "transformers": transformers
        }}
        self.metadata = metadata if metadata else {{}}
        
        # Initialize model
        self.model = hf_{normalized_name}(resources=self.resources, metadata=self.metadata)
        
        # Use appropriate model for testing
        self.model_name = "{example_model}"
        
        # Test inputs appropriate for this model type
        {test_inputs_str}
        
        # Collection arrays for results
        self.examples = []
        self.status_messages = {{}}
    
    def get_test_input(self, batch=False):
        # Choose appropriate test input
        if batch:
            if hasattr(self, 'test_batch'):
                return self.test_batch
        
        if "{category}" == "language" and hasattr(self, 'test_text'):
            return self.test_text
        elif "{category}" == "vision":
            if hasattr(self, 'test_image_path'):
                return self.test_image_path
            elif hasattr(self, 'test_image'):
                return self.test_image
        elif "{category}" == "audio":
            if hasattr(self, 'test_audio_path'):
                return self.test_audio_path
            elif hasattr(self, 'test_audio'):
                return self.test_audio
        elif "{category}" == "multimodal":
            if hasattr(self, 'test_vqa'):
                return self.test_vqa
            elif hasattr(self, 'test_document_qa'):
                return self.test_document_qa
            elif hasattr(self, 'test_image_path'):
                return self.test_image_path
        
        # Default fallback
        if hasattr(self, 'test_input'):
            return self.test_input
        return "Default test input"
    
    def test_platform(self, platform, init_method, device_arg):
        # Run tests for a specific platform
        results = {{}}
        
        try:
            print(f"Testing {normalized_name} on {{platform.upper()}}...")
            
            # Initialize for this platform
            endpoint, processor, handler, queue, batch_size = init_method(
                self.model_name, "{primary_task}", device_arg
            )
            
            # Check initialization success
            valid_init = endpoint is not None and processor is not None and handler is not None
            results[f"{{platform}}_init"] = "Success" if valid_init else f"Failed {{platform.upper()}} initialization"
            
            if not valid_init:
                results[f"{{platform}}_handler"] = f"Failed {{platform.upper()}} handler"
                return results
            
            # Get test input
            test_input = self.get_test_input()
            
            # Run inference
            output = handler(test_input)
            
            # Verify output
            is_valid_output = output is not None
            
            # Determine implementation type
            if isinstance(output, dict) and "implementation_type" in output:
                impl_type = output["implementation_type"]
            else:
                impl_type = "REAL" if is_valid_output else "MOCK"
                
            results[f"{{platform}}_handler"] = f"Success ({{impl_type}})" if is_valid_output else f"Failed {{platform.upper()}} handler"
            
            # Record example
            self.examples.append({{
                "input": str(test_input),
                "output": {{
                    "output_type": str(type(output)),
                    "implementation_type": impl_type
                }},
                "timestamp": datetime.datetime.now().isoformat(),
                "implementation_type": impl_type,
                "platform": platform.upper()
            }})
            
            # Try batch processing if possible
            try:
                batch_input = self.get_test_input(batch=True)
                if batch_input is not None:
                    batch_output = handler(batch_input)
                    is_valid_batch = batch_output is not None
                    
                    if isinstance(batch_output, dict) and "implementation_type" in batch_output:
                        batch_impl_type = batch_output["implementation_type"]
                    else:
                        batch_impl_type = "REAL" if is_valid_batch else "MOCK"
                        
                    results[f"{{platform}}_batch"] = f"Success ({{batch_impl_type}})" if is_valid_batch else f"Failed {{platform.upper()}} batch"
                    
                    # Record batch example
                    self.examples.append({{
                        "input": str(batch_input),
                        "output": {{
                            "output_type": str(type(batch_output)),
                            "implementation_type": batch_impl_type,
                            "is_batch": True
                        }},
                        "timestamp": datetime.datetime.now().isoformat(),
                        "implementation_type": batch_impl_type,
                        "platform": platform.upper()
                    }})
            except Exception as batch_e:
                results[f"{{platform}}_batch_error"] = str(batch_e)
        except Exception as e:
            print(f"Error in {{platform.upper()}} tests: {{e}}")
            traceback.print_exc()
            results[f"{{platform}}_error"] = str(e)
            self.status_messages[platform] = f"Failed: {{str(e)}}"
        
        return results
    
    def test(self):
        # Run comprehensive tests
        results = {{}}
        
        # Test basic initialization
        results["init"] = "Success" if self.model is not None else "Failed initialization"
        results["has_implementation"] = "True" if HAS_IMPLEMENTATION else "False (using mock)"
        
        # CPU tests
        cpu_results = self.test_platform("cpu", self.model.init_cpu, "cpu")
        results.update(cpu_results)
        
        # CUDA tests if available
        if HAS_TORCH and torch.cuda.is_available():
            cuda_results = self.test_platform("cuda", self.model.init_cuda, "cuda:0")
            results.update(cuda_results)
        else:
            results["cuda_tests"] = "CUDA not available"
            self.status_messages["cuda"] = "CUDA not available"
        
        # OpenVINO tests if available
        try:
            import openvino
            openvino_results = self.test_platform("openvino", self.model.init_openvino, "CPU")
            results.update(openvino_results)
        except ImportError:
            results["openvino_tests"] = "OpenVINO not installed"
            self.status_messages["openvino"] = "OpenVINO not installed"
        except Exception as e:
            print(f"Error in OpenVINO tests: {{e}}")
            results["openvino_error"] = str(e)
            self.status_messages["openvino"] = f"Failed: {{str(e)}}"
        
        # Return structured results
        return {{
            "status": results,
            "examples": self.examples,
            "metadata": {{
                "model_name": self.model_name,
                "model": "{model}",
                "primary_task": "{primary_task}",
                "pipeline_tasks": {json.dumps(pipeline_tasks)},
                "category": "{category}",
                "test_timestamp": datetime.datetime.now().isoformat(),
                "has_implementation": HAS_IMPLEMENTATION,
                "platform_status": self.status_messages
            }}
        }}
    
    def __test__(self):
        # Run tests and save results
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
        
        # Create directories if needed
        base_dir = os.path.dirname(os.path.abspath(__file__))
        expected_dir = os.path.join(base_dir, 'expected_results')
        collected_dir = os.path.join(base_dir, 'collected_results')
        
        # Ensure directories exist
        for directory in [expected_dir, collected_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory, mode=0o755, exist_ok=True)
        
        # Save test results
        results_file = os.path.join(collected_dir, 'hf_{normalized_name}_test_results.json')
        try:
            with open(results_file, 'w') as f:
                json.dump(test_results, f, indent=2)
            print(f"Saved test results to {{results_file}}")
        except Exception as e:
            print(f"Error saving results: {{e}}")
        
        # Create expected results if they don't exist
        expected_file = os.path.join(expected_dir, 'hf_{normalized_name}_test_results.json')
        if not os.path.exists(expected_file):
            try:
                with open(expected_file, 'w') as f:
                    json.dump(test_results, f, indent=2)
                print(f"Created new expected results file")
            except Exception as e:
                print(f"Error creating expected results: {{e}}")
        
        return test_results

def extract_implementation_status(results):
    # Extract implementation status from results
    status_dict = results.get("status", {{}})
    
    cpu_status = "UNKNOWN"
    cuda_status = "UNKNOWN"
    openvino_status = "UNKNOWN"
    
    # Check CPU status
    for key, value in status_dict.items():
        if key.startswith("cpu_") and "REAL" in value:
            cpu_status = "REAL"
        elif key.startswith("cpu_") and "MOCK" in value:
            cpu_status = "MOCK"
            
        if key.startswith("cuda_") and "REAL" in value:
            cuda_status = "REAL"
        elif key.startswith("cuda_") and "MOCK" in value:
            cuda_status = "MOCK"
        elif key == "cuda_tests" and value == "CUDA not available":
            cuda_status = "NOT AVAILABLE"
            
        if key.startswith("openvino_") and "REAL" in value:
            openvino_status = "REAL"
        elif key.startswith("openvino_") and "MOCK" in value:
            openvino_status = "MOCK"
        elif key == "openvino_tests" and value == "OpenVINO not installed":
            openvino_status = "NOT INSTALLED"
    
    return {{
        "cpu": cpu_status,
        "cuda": cuda_status,
        "openvino": openvino_status
    }}

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='{model} model test')
    parser.add_argument('--platform', type=str, choices=['cpu', 'cuda', 'openvino', 'all'], 
                        default='all', help='Platform to test')
    parser.add_argument('--model', type=str, help='Override model name')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    
    # Run the tests
    print(f"Starting {normalized_name} test...")
    test_instance = test_hf_{normalized_name}()
    
    # Override model if specified
    if args.model:
        test_instance.model_name = args.model
        print(f"Using model: {{args.model}}")
    
    # Run tests
    results = test_instance.__test__()
    status = extract_implementation_status(results)
    
    # Print summary
    print(f"\\n{normalized_name.upper()} TEST RESULTS SUMMARY")
    print(f"MODEL: {{results.get('metadata', {{}}).get('model_name', 'Unknown')}}")
    print(f"CPU STATUS: {{status['cpu']}}")
    print(f"CUDA STATUS: {{status['cuda']}}")
    print(f"OPENVINO STATUS: {{status['openvino']}}")
"""

    # Save file
    output_path = os.path.join(output_dir, f"{TEST_FILE_PREFIX}{normalized_name}.py")
    
    try:
        with open(output_path, 'w') as f:
            f.write(content)
        
        # Make executable
        os.chmod(output_path, 0o755)
        
        return True, f"Generated test file for {model} at {output_path}"
    except Exception as e:
        return False, f"Error generating test for {model}: {e}"

def find_missing_tests(all_models, existing_tests, model_to_pipeline, category=None):
    """Find models that don't have test implementations yet."""
    missing_tests = []
    
    for model in all_models:
        normalized_name = normalize_model_name(model)
        
        # Skip if test already exists
        if normalized_name in existing_tests:
            continue
        
        # Get pipeline tasks
        pipeline_tasks = model_to_pipeline.get(model, [])
        
        # Skip if no tasks
        if not pipeline_tasks:
            continue
        
        # Get category
        model_category = get_model_category(pipeline_tasks)
        
        # Apply category filter if specified
        if category and category != "all" and model_category != category:
            continue
        
        # Add to missing tests
        missing_tests.append({
            "model": model,
            "normalized_name": normalized_name,
            "pipeline_tasks": pipeline_tasks,
            "category": model_category,
            "primary_task": get_primary_task(model, pipeline_tasks)
        })
    
    return missing_tests

def generate_tests(missing_tests, output_dir, limit=10):
    """Generate test files for missing tests."""
    generated = []
    
    # Limit number of files to generate
    to_generate = missing_tests[:limit]
    
    for model_info in to_generate:
        model = model_info["model"]
        normalized_name = model_info["normalized_name"]
        pipeline_tasks = model_info["pipeline_tasks"]
        
        success, message = generate_test_file(model, normalized_name, pipeline_tasks, output_dir)
        
        generated.append({
            "model": model,
            "success": success,
            "message": message
        })
        
        logger.info(message)
    
    return generated

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate test files for Hugging Face models")
    parser.add_argument("--models", nargs="+", help="Specific models to generate tests for")
    parser.add_argument("--category", choices=["language", "vision", "audio", "multimodal", "all"], 
                       default="all", help="Category of models to process")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of tests to generate")
    parser.add_argument("--list-only", action="store_true", help="List missing tests without generating files")
    parser.add_argument("--output-dir", default=str(SKILLS_DIR), help="Directory to save test files")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model data
    all_models, model_to_pipeline, pipeline_to_model = load_model_data()
    
    # Get existing tests
    existing_tests = get_existing_tests()
    
    # Process specific models if provided
    if args.models:
        for model in args.models:
            if model not in all_models:
                logger.warning(f"Model {model} not found in model types")
                continue
            
            normalized_name = normalize_model_name(model)
            pipeline_tasks = model_to_pipeline.get(model, [])
            
            success, message = generate_test_file(model, normalized_name, pipeline_tasks, args.output_dir)
            logger.info(message)
        
        return
    
    # Find missing tests
    missing_tests = find_missing_tests(all_models, existing_tests, model_to_pipeline, args.category)
    
    # Group by category
    by_category = {}
    for test in missing_tests:
        category = test["category"]
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(test)
    
    # Print missing tests
    print(f"Found {len(missing_tests)} models missing test implementations")
    
    for category, tests in sorted(by_category.items()):
        print(f"\n{category.upper()} ({len(tests)} models):")
        for i, test in enumerate(tests[:5]):
            model = test["model"]
            tasks = ", ".join(test["pipeline_tasks"])
            print(f"  {i+1}. {model}: {tasks}")
        
        if len(tests) > 5:
            print(f"  ... and {len(tests) - 5} more {category} models")
    
    # Exit if list-only
    if args.list_only:
        return
    
    # Generate test files
    print(f"\nGenerating up to {args.limit} test files...")
    results = generate_tests(missing_tests, args.output_dir, args.limit)
    
    # Print summary
    successful = sum(1 for r in results if r["success"])
    print(f"\nSummary: Generated {successful}/{len(results)} test files")
    
    # Print generated files
    if successful > 0:
        print("\nGenerated test files:")
        for result in results:
            if result["success"]:
                print(f"  ✅ {result['model']}")

if __name__ == "__main__":
    main()