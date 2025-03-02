#!/usr/bin/env python3
"""
Comprehensive Test Generator for HuggingFace Models

This script generates optimized test files for HuggingFace models that ensure:
1. Complete coverage of both pipeline() and from_pretrained() APIs 
2. Testing across all three hardware backends (CPU, CUDA, and OpenVINO)
3. Batch processing capabilities with proper memory tracking
4. Detailed performance metrics for each hardware platform
5. Thread-safe parallel testing for efficiency
6. Consistent unified testing approach across all model types

The generated tests use the ComprehensiveModelTester framework defined in 
test_simplified.py, which provides a standardized way to test all models
while ensuring proper hardware detection and resource management.

Usage:
  python generate_comprehensive_tests.py --model bert-base-uncased
  python generate_comprehensive_tests.py --all-families
  python generate_comprehensive_tests.py --missing --limit 10 
"""

import os
import sys
import json
import glob
import time
import datetime
import traceback
from pathlib import Path
import concurrent.futures
import argparse
import logging
from typing import Dict, List, Tuple, Set, Optional, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("test_generation.log")
    ]
)
logger = logging.getLogger("test_generator")

# Special models requiring unique handling
SPECIALIZED_MODELS = {
    # Time series models
    "time_series_transformer": "time-series-prediction",
    "patchtst": "time-series-prediction", 
    "autoformer": "time-series-prediction",
    "informer": "time-series-prediction",
    "patchtsmixer": "time-series-prediction",
    
    # Protein models
    "esm": "protein-folding",
    
    # Document understanding
    "layoutlmv2": "document-question-answering",
    "layoutlmv3": "document-question-answering",
    "markuplm": "document-question-answering",
    "donut-swin": "document-question-answering",
    "pix2struct": "document-question-answering",
    
    # Table models
    "tapas": "table-question-answering",
    
    # Depth estimation
    "depth_anything": "depth-estimation",
    "dpt": "depth-estimation",
    "zoedepth": "depth-estimation",
    
    # Audio-specific models
    "whisper": "automatic-speech-recognition",
    "bark": "text-to-audio",
    "musicgen": "text-to-audio",
    "speecht5": "text-to-audio",
    "encodec": "audio-xvector",
    
    # Cross-modal models
    "seamless_m4t": "translation_xx_to_yy",
    "seamless_m4t_v2": "translation_xx_to_yy",
    
    # Specialized vision models
    "sam": "image-segmentation",
    "owlvit": "object-detection",
    "grounding_dino": "object-detection",
}

# Cache directories
CACHE_DIR = Path(".test_generation_cache")

def setup_cache_directories():
    """Setup cache directories for test generation"""
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(exist_ok=True)
        logger.info(f"Created cache directory: {CACHE_DIR}")

def load_model_data() -> Tuple[List[str], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Load all model data from JSON files.
    
    Returns:
        Tuple containing:
        - List of all model types
        - Dict mapping model names to pipeline tasks
        - Dict mapping pipeline tasks to model names
    """
    try:
        # Load model types
        with open('huggingface_model_types.json', 'r') as f:
            all_models = json.load(f)
        
        # Load pipeline mappings
        with open('huggingface_model_pipeline_map.json', 'r') as f:
            model_to_pipeline = json.load(f)
        
        with open('huggingface_pipeline_model_map.json', 'r') as f:
            pipeline_to_model = json.load(f)
            
        logger.info(f"Loaded {len(all_models)} models and pipeline mappings")
        return all_models, model_to_pipeline, pipeline_to_model
    except Exception as e:
        logger.error(f"Error loading model data: {e}")
        raise

def get_existing_tests() -> Set[str]:
    """Get the normalized names of existing test files"""
    test_files = glob.glob('skills/test_hf_*.py')
    existing_tests = set()
    
    for test_file in test_files:
        model_name = test_file.replace('skills/test_hf_', '').replace('.py', '')
        existing_tests.add(model_name)
    
    logger.info(f"Found {len(existing_tests)} existing test implementations")
    return existing_tests

def normalize_model_name(name: str) -> str:
    """Normalize model name to match file naming conventions"""
    return name.replace('-', '_').replace('.', '_').lower()

def get_missing_tests(
    all_models: List[str], 
    existing_tests: Set[str],
    model_to_pipeline: Dict[str, List[str]],
    priority_models: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Identify models missing test implementations.
    
    Args:
        all_models: List of all model types
        existing_tests: Set of normalized model names with existing tests
        model_to_pipeline: Dict mapping model names to pipeline tasks
        priority_models: Optional list of high-priority models
        
    Returns:
        List of dicts with information about missing tests
    """
    missing_tests = []
    
    # Create set of priority models if provided
    priority_set = set(normalize_model_name(m) for m in priority_models) if priority_models else set()
    
    for model in all_models:
        normalized_name = normalize_model_name(model)
        
        # Skip if test already exists
        if normalized_name in existing_tests:
            continue
            
        # Get associated pipeline tasks
        pipeline_tasks = model_to_pipeline.get(model, [])
        
        # If model is in SPECIALIZED_MODELS, add the specialized task
        if model in SPECIALIZED_MODELS and SPECIALIZED_MODELS[model] not in pipeline_tasks:
            pipeline_tasks.append(SPECIALIZED_MODELS[model])
        
        # Determine priority
        is_high_priority = normalized_name in priority_set or model in SPECIALIZED_MODELS
        priority = "HIGH" if is_high_priority else "MEDIUM"
        
        missing_tests.append({
            "model": model,
            "normalized_name": normalized_name,
            "pipeline_tasks": pipeline_tasks,
            "priority": priority
        })
    
    # Sort by priority (high first), then by pipeline tasks count (more first)
    missing_tests.sort(
        key=lambda x: (0 if x["priority"] == "HIGH" else 1, -len(x["pipeline_tasks"]))
    )
    
    logger.info(f"Identified {len(missing_tests)} missing test implementations")
    return missing_tests

def get_pipeline_category(pipeline_tasks: List[str]) -> str:
    """
    Determine the category of a model based on its pipeline tasks.
    
    Args:
        pipeline_tasks: List of pipeline tasks
        
    Returns:
        Category string (language, vision, audio, multimodal, etc.)
    """
    task_set = set(pipeline_tasks)
    
    # Define task categories
    language_tasks = {"text-generation", "text2text-generation", "fill-mask", 
                     "text-classification", "token-classification", "question-answering",
                     "summarization", "translation_xx_to_yy"}
                     
    vision_tasks = {"image-classification", "object-detection", "image-segmentation",
                   "depth-estimation", "semantic-segmentation", "instance-segmentation"}
                   
    audio_tasks = {"automatic-speech-recognition", "audio-classification", "text-to-audio",
                  "audio-to-audio", "audio-xvector"}
                  
    multimodal_tasks = {"image-to-text", "visual-question-answering", "document-question-answering",
                       "video-classification"}
                       
    specialized_tasks = {"protein-folding", "table-question-answering", "time-series-prediction"}
    
    # Check for matches in each category
    if task_set & multimodal_tasks:
        return "multimodal"
    if task_set & audio_tasks:
        return "audio"
    if task_set & vision_tasks:
        return "vision"
    if task_set & language_tasks:
        return "language"
    if task_set & specialized_tasks:
        return "specialized"
    
    # Default category
    return "other"

def select_template_model(
    model_info: Dict[str, Any], 
    existing_tests: Set[str],
    all_models: List[str]
) -> str:
    """
    Select an appropriate template model based on category and model type.
    
    Args:
        model_info: Model information including pipeline tasks
        existing_tests: Set of normalized model names with existing tests
        all_models: List of all model types
        
    Returns:
        Name of the template file to use
    """
    normalized_name = model_info["normalized_name"]
    pipeline_tasks = model_info["pipeline_tasks"]
    category = get_pipeline_category(pipeline_tasks)
    
    # Define template models by category
    templates = {
        "language": ["bert", "gpt2", "t5", "llama", "roberta"],
        "vision": ["vit", "clip", "segformer", "detr"],
        "audio": ["whisper", "wav2vec2", "clap"],
        "multimodal": ["llava", "blip", "fuyu"],
        "specialized": ["time_series_transformer", "esm", "tapas"],
        "other": ["bert"]
    }
    
    # Get candidate templates that already have tests
    candidates = [t for t in templates.get(category, templates["other"]) 
                 if t in existing_tests]
    
    if not candidates:
        # Fallback to bert if no templates found
        return "bert"
    
    # Choose the first available template
    return candidates[0]

def get_specialized_test_inputs(primary_task: str) -> List[str]:
    """
    Get specialized test input examples based on primary task.
    
    Args:
        primary_task: Primary pipeline task for the model
        
    Returns:
        List of strings with test input definitions
    """
    examples = []
    
    # Text generation examples
    if primary_task in ["text-generation", "text2text-generation", "summarization", "translation_xx_to_yy"]:
        examples.append('self.test_text = "The quick brown fox jumps over the lazy dog"')
        examples.append('self.test_batch = ["The quick brown fox jumps over the lazy dog", "The five boxing wizards jump quickly"]')
    
    # Image examples
    if primary_task in ["image-classification", "object-detection", "image-segmentation", 
                       "image-to-text", "visual-question-answering", "depth-estimation"]:
        examples.append('self.test_image = "test.jpg"  # Path to a test image file')
        examples.append('# Import necessary libraries for batch testing\ntry:\n    import os\n    from PIL import Image\n    self.test_batch_images = ["test.jpg", "test.jpg"]\nexcept ImportError:\n    self.test_batch_images = ["test.jpg", "test.jpg"]')
    
    # Audio examples
    if primary_task in ["automatic-speech-recognition", "audio-classification", "text-to-audio"]:
        examples.append('self.test_audio = "test.mp3"  # Path to a test audio file')
        examples.append('self.test_batch_audio = ["test.mp3", "trans_test.mp3"]')
    
    # Question-answering examples
    if primary_task == "question-answering":
        examples.append('self.test_qa = {"question": "What is the capital of France?", "context": "Paris is the capital and most populous city of France."}')
        examples.append('self.test_batch_qa = [{"question": "What is the capital of France?", "context": "Paris is the capital and most populous city of France."}, {"question": "What is the tallest mountain?", "context": "Mount Everest is Earth\'s highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas."}]')
    
    # Multimodal examples
    if primary_task in ["visual-question-answering"]:
        examples.append('self.test_vqa = {"image": "test.jpg", "question": "What is shown in this image?"}')
    
    # Protein examples
    if primary_task == "protein-folding":
        examples.append('self.test_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"')
        examples.append('self.test_batch_sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG", "SGFRVQRITSSILRILEQNKDSTSAAQLEELVKVLSAQILYVTTLGYDSVSASRGGLDLGG"]')
    
    # Table examples
    if primary_task == "table-question-answering":
        table_example = '''self.test_table = {
            "header": ["Name", "Age", "Occupation"],
            "rows": [
                ["John", "25", "Engineer"],
                ["Alice", "32", "Doctor"],
                ["Bob", "41", "Teacher"]
            ],
            "question": "How old is Alice?"
        }'''
        examples.append(table_example)
    
    # Time series examples
    if primary_task == "time-series-prediction":
        ts_example = '''self.test_time_series = {
            "past_values": [100, 120, 140, 160, 180],
            "past_time_features": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]],
            "future_time_features": [[5, 0], [6, 0], [7, 0]]
        }'''
        examples.append(ts_example)
        batch_ts_example = '''self.test_batch_time_series = [
            {
                "past_values": [100, 120, 140, 160, 180],
                "past_time_features": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]],
                "future_time_features": [[5, 0], [6, 0], [7, 0]]
            },
            {
                "past_values": [200, 220, 240, 260, 280],
                "past_time_features": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]],
                "future_time_features": [[5, 0], [6, 0], [7, 0]]
            }
        ]'''
        examples.append(batch_ts_example)
    
    # Document examples
    if primary_task == "document-question-answering":
        doc_example = '''self.test_document = {
            "image": "test.jpg",
            "question": "What is the title of this document?"
        }'''
        examples.append(doc_example)
    
    # Default example if no specific examples found
    if not examples:
        examples.append('self.test_input = "Test input appropriate for this model"')
        examples.append('self.test_batch_input = ["Test input 1", "Test input 2"]')
    
    return examples

def get_appropriate_model_name(pipeline_tasks: List[str]) -> str:
    """
    Choose an appropriate example model name based on pipeline tasks.
    
    Args:
        pipeline_tasks: List of pipeline tasks for the model
        
    Returns:
        Example model name suitable for this model type
    """
    primary_task = pipeline_tasks[0] if pipeline_tasks else "feature-extraction"
    
    # Define model name mapping
    task_to_model = {
        "text-generation": '"distilgpt2"  # Small text generation model',
        "text2text-generation": '"t5-small"  # Small text-to-text model',
        "fill-mask": '"distilroberta-base"  # Small masked language model',
        "image-classification": '"google/vit-base-patch16-224-in21k"  # Standard vision transformer',
        "object-detection": '"facebook/detr-resnet-50"  # Small object detection model',
        "image-segmentation": '"facebook/detr-resnet-50-panoptic"  # Small segmentation model',
        "automatic-speech-recognition": '"openai/whisper-tiny"  # Small ASR model',
        "audio-classification": '"facebook/wav2vec2-base"  # Small audio classification model',
        "text-to-audio": '"facebook/musicgen-small"  # Small text-to-audio model',
        "image-to-text": '"Salesforce/blip-image-captioning-base"  # Small image captioning model',
        "visual-question-answering": '"Salesforce/blip-vqa-base"  # Small VQA model',
        "document-question-answering": '"microsoft/layoutlm-base-uncased"  # Small document QA model',
        "protein-folding": '"facebook/esm2_t6_8M_UR50D"  # Small protein embedding model',
        "table-question-answering": '"google/tapas-base"  # Small table QA model',
        "time-series-prediction": '"huggingface/time-series-transformer-tourism-monthly"  # Small time series model',
        "depth-estimation": '"Intel/dpt-hybrid-midas"  # Small depth estimation model'
    }
    
    # Return appropriate model name or default
    return task_to_model.get(primary_task, f'"(undetermined)"  # Replace with appropriate model for task: {primary_task}')

def generate_test_template(
    model_info: Dict[str, Any],
    template_model: str
) -> str:
    """
    Generate test file template for a specific model using the comprehensive test framework.
    
    Args:
        model_info: Model information including name and pipeline tasks
        template_model: Model to use as template
        
    Returns:
        Generated test file content
    """
    model = model_info["model"]
    normalized_name = model_info["normalized_name"]
    pipeline_tasks = model_info["pipeline_tasks"]
    
    class_name = f"hf_{normalized_name}"
    test_class_name = f"test_hf_{normalized_name}"
    
    # Determine model types based on pipeline tasks
    model_type_comment = "# Model supports: " + ", ".join(pipeline_tasks)
    
    # Choose primary pipeline task
    primary_task = pipeline_tasks[0] if pipeline_tasks else "feature-extraction"
    
    # Get categorized task type for imports
    category = get_pipeline_category(pipeline_tasks)
    
    # Get specialized test examples
    test_examples = get_specialized_test_inputs(primary_task)
    test_examples_str = "\n        ".join(test_examples)
    
    # Choose appropriate model initialization
    example_model = get_appropriate_model_name(pipeline_tasks)
    
    # Template for the test file - using the new ComprehensiveModelTester approach
    template = f"""#!/usr/bin/env python3
# Test implementation for the {model} model ({normalized_name})
# Generated by generate_comprehensive_tests.py - {datetime.datetime.now().isoformat()}
{model_type_comment}

import os
import sys
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
parent_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
test_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(test_dir))

# Import the unified test framework
try:
    from test_simplified import ComprehensiveModelTester, save_results
except ImportError:
    print("ERROR: Cannot import ComprehensiveModelTester from test_simplified.py")
    print("Please make sure the test_simplified.py file is available in the skills directory.")
    sys.exit(1)

# Import the module to test (create a mock if not available)
try:
    from ipfs_accelerate_py.worker.skillset.{class_name} import {class_name}
    HAS_IMPLEMENTATION = True
except ImportError:
    # If the module doesn't exist yet, create a mock class
    class {class_name}:
        def __init__(self, resources=None, metadata=None):
            self.resources = resources or {{}}
            self.metadata = metadata or {{}}
            
        def init_cpu(self, model_name=None, model_type="{primary_task}", device="cpu", **kwargs):
            # Mock implementation
            return None, None, lambda x: {{"output": "Mock CPU output for " + model_name, 
                                       "implementation_type": "MOCK"}}, None, 1
            
        def init_cuda(self, model_name=None, model_type="{primary_task}", device_label="cuda:0", **kwargs):
            # Mock implementation
            return None, None, lambda x: {{"output": "Mock CUDA output for " + model_name, 
                                       "implementation_type": "MOCK"}}, None, 1
            
        def init_openvino(self, model_name=None, model_type="{primary_task}", device="CPU", **kwargs):
            # Mock implementation
            return None, None, lambda x: {{"output": "Mock OpenVINO output for " + model_name, 
                                       "implementation_type": "MOCK"}}, None, 1
    
    HAS_IMPLEMENTATION = False
    print(f"Warning: {{class_name}} module not found, using mock implementation")

class {test_class_name}:
    """
    Test implementation for {model} model using the comprehensive test framework.
    
    This test ensures complete coverage of:
    - pipeline() and from_pretrained() APIs
    - CPU, CUDA, and OpenVINO hardware backends
    - Batch processing capabilities
    - Memory usage and performance tracking
    """
    
    def __init__(self, resources=None, metadata=None):
        """Initialize the test with custom resources or metadata if needed."""
        # Initialize test inputs appropriate for this model type
        self.test_inputs = {{
            {test_examples_str}
        }}
        
        # Create the test instance with the appropriate model info
        self.tester = ComprehensiveModelTester(
            model_id="{model}",
            model_type="{primary_task}",
            resources=resources,
            metadata=metadata
        )
    
    def test(self, all_hardware=True, include_batch=True, parallel=True):
        """
        Run comprehensive tests for this model.
        
        Args:
            all_hardware: Test on all available hardware backends
            include_batch: Include batch processing tests
            parallel: Run tests in parallel for speed
            
        Returns:
            Dict containing test results
        """
        # Run the comprehensive tests
        results = self.tester.run_tests(
            all_hardware=all_hardware,
            include_batch=include_batch,
            parallel=parallel
        )
        
        return results
        
    def run_tests(self):
        """Legacy method for compatibility - runs all tests."""
        return self.test(all_hardware=True, include_batch=True)
        
    def __test__(self):
        """Default test entry point."""
        # Run tests and save results
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
        results_file = os.path.join(collected_dir, 'hf_{normalized_name}_test_results.json')
        try:
            with open(results_file, 'w') as f:
                json.dump(test_results, f, indent=2)
            print(f"Saved collected results to {{results_file}}")
        except Exception as e:
            print(f"Error saving results to {{results_file}}: {{str(e)}}")
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'hf_{normalized_name}_test_results.json')
        if os.path.exists(expected_file):
            try:
                with open(expected_file, 'r') as f:
                    expected_results = json.load(f)
                
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
                    print("\nWould you like to update the expected results? (y/n)")
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

def extract_implementation_status(results):
    # Extract implementation status from test results
    status_dict = results.get("status", {{}})
    examples = results.get("examples", [])
    
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
            
    # Also look in examples
    for example in examples:
        platform = example.get("platform", "")
        impl_type = example.get("implementation_type", "")
        
        if platform == "CPU" and "REAL" in impl_type:
            cpu_status = "REAL"
        elif platform == "CPU" and "MOCK" in impl_type:
            cpu_status = "MOCK"
            
        if platform == "CUDA" and "REAL" in impl_type:
            cuda_status = "REAL"
        elif platform == "CUDA" and "MOCK" in impl_type:
            cuda_status = "MOCK"
            
        if platform == "OPENVINO" and "REAL" in impl_type:
            openvino_status = "REAL"
        elif platform == "OPENVINO" and "MOCK" in impl_type:
            openvino_status = "MOCK"
    
    return {{
        "cpu": cpu_status,
        "cuda": cuda_status,
        "openvino": openvino_status
    }}

def main():
    """Command-line entry point."""
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='{model} model test using comprehensive framework')
    
    # Testing options
    parser.add_argument('--cpu-only', action='store_true', help='Test only on CPU')
    parser.add_argument('--cuda-only', action='store_true', help='Test only on CUDA')
    parser.add_argument('--openvino-only', action='store_true', help='Test only on OpenVINO')
    parser.add_argument('--no-batch', action='store_true', help='Skip batch processing tests')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel testing')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Configure test parameters
    all_hardware = not (args.cpu_only or args.cuda_only or args.openvino_only)
    include_batch = not args.no_batch
    parallel = not args.no_parallel
    
    # Create and run test instance
    print(f"Starting comprehensive test for {normalized_name}...")
    test_instance = {test_class_name}()
    results = test_instance.test(
        all_hardware=all_hardware,
        include_batch=include_batch,
        parallel=parallel
    )
    
    # Save results
    output_path = save_results("{normalized_name}", results)
    print(f"Results saved to: {{output_path}}")
    
    # Print summary of results
    real_count = sum(1 for r in results["results"].values() 
                    if r.get("implementation_type", "MOCK") == "REAL")
    total_count = len(results["results"])
    
    print(f"\\nModel: {model}")
    print(f"Type: {primary_task}")
    print(f"Tests run: {{total_count}}")
    print(f"REAL implementations: {{real_count}}/{{total_count}} ({{real_count/total_count*100:.1f}}%)")
    
    # Print hardware-specific results
    print("\\nHardware Results:")
    for platform in ["cpu", "cuda", "openvino"]:
        platform_results = [r for k, r in results["results"].items() if platform in k]
        if platform_results:
            real_impls = sum(1 for r in platform_results if r.get("implementation_type", "MOCK") == "REAL")
            print(f"  {platform.upper()}: {{real_impls}}/{{len(platform_results)}} REAL implementations")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
"""
    
    return template

def generate_test_file(
    model_info: Dict[str, Any],
    existing_tests: Set[str],
    all_models: List[str],
    output_dir: str
) -> Tuple[bool, str]:
    """
    Generate a test file for a specific model.
    
    Args:
        model_info: Model information including name and pipeline tasks
        existing_tests: Set of normalized model names with existing tests
        all_models: List of all model types
        output_dir: Directory to save the generated file
        
    Returns:
        Tuple of (success, message)
    """
    try:
        model = model_info["model"]
        normalized_name = model_info["normalized_name"]
        
        # Skip if test already exists (double check)
        test_file_path = os.path.join(output_dir, f"test_hf_{normalized_name}.py")
        if os.path.exists(test_file_path):
            return False, f"Test file already exists for {model}, skipping"
        
        # Select an appropriate template model
        template_model = select_template_model(model_info, existing_tests, all_models)
        
        # Generate test template
        template = generate_test_template(model_info, template_model)
        
        # Write to file
        with open(test_file_path, "w") as f:
            f.write(template)
        
        # Make executable
        os.chmod(test_file_path, 0o755)
        
        return True, f"Generated test file for {model} at {test_file_path}"
    except Exception as e:
        return False, f"Error generating test for {model_info['model']}: {e}"

def generate_test_files_parallel(
    missing_tests: List[Dict[str, Any]],
    existing_tests: Set[str],
    all_models: List[str],
    output_dir: str,
    limit: int,
    high_priority_only: bool
) -> List[str]:
    """
    Generate test files in parallel using ThreadPoolExecutor.
    
    Args:
        missing_tests: List of models needing test implementations
        existing_tests: Set of normalized model names with existing tests
        all_models: List of all model types
        output_dir: Directory to save generated files
        limit: Maximum number of files to generate
        high_priority_only: Only generate high priority tests
        
    Returns:
        List of messages about generation results
    """
    # Filter by priority if requested
    if high_priority_only:
        missing_tests = [m for m in missing_tests if m["priority"] == "HIGH"]
        
    # Limit number of files to generate
    missing_tests = missing_tests[:limit]
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, mode=0o755, exist_ok=True)
        logger.info(f"Created directory: {output_dir}")
    
    messages = []
    generated_count = 0
    
    # Use ThreadPoolExecutor for parallel generation
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all tasks
        future_to_model = {
            executor.submit(
                generate_test_file, model_info, existing_tests, all_models, output_dir
            ): model_info["model"] 
            for model_info in missing_tests
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_model):
            model = future_to_model[future]
            try:
                success, message = future.result()
                messages.append(message)
                
                if success:
                    generated_count += 1
                    logger.info(message)
            except Exception as e:
                messages.append(f"Error generating test for {model}: {e}")
                logger.error(f"Error generating test for {model}: {e}")
    
    # Add summary message
    messages.append(f"\nSummary: Generated {generated_count} test templates")
    messages.append(f"Remaining missing tests: {len(missing_tests) - generated_count}")
    
    return messages

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate test files for Hugging Face models")
    parser.add_argument(
        "--limit", type=int, default=10,
        help="Maximum number of test files to generate"
    )
    parser.add_argument(
        "--high-priority-only", action="store_true",
        help="Only generate tests for high priority models"
    )
    parser.add_argument(
        "--output-dir", type=str, default="skills",
        help="Directory to save generated test files"
    )
    parser.add_argument(
        "--category", type=str, choices=["language", "vision", "audio", "multimodal", "specialized", "all"],
        default="all", help="Category of models to generate tests for"
    )
    parser.add_argument(
        "--list-only", action="store_true",
        help="Only list missing tests, don't generate files"
    )
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    print(f"Starting test file generation at {datetime.datetime.now().isoformat()}")
    
    # Setup cache directories
    setup_cache_directories()
    
    # Load model data
    try:
        all_models, model_to_pipeline, pipeline_to_model = load_model_data()
    except Exception as e:
        logger.error(f"Error loading model data: {e}")
        sys.exit(1)
    
    # Get existing tests
    try:
        existing_tests = get_existing_tests()
    except Exception as e:
        logger.error(f"Error finding existing tests: {e}")
        sys.exit(1)
    
    # Identify missing tests
    try:
        missing_tests = get_missing_tests(
            all_models, existing_tests, model_to_pipeline,
            list(SPECIALIZED_MODELS.keys())  # Use specialized models as priority
        )
        
        # Filter by category if specified
        if args.category != "all":
            missing_tests = [
                m for m in missing_tests
                if get_pipeline_category(m["pipeline_tasks"]) == args.category
            ]
        
        # Print summary of high priority models
        high_priority = [m for m in missing_tests if m["priority"] == "HIGH"]
        print(f"\nHigh priority models to implement ({len(high_priority)}):")
        for model in high_priority[:10]:  # Show top 10
            tasks = ", ".join(model["pipeline_tasks"])
            print(f"- {model['model']}: {tasks}")
        
        if len(high_priority) > 10:
            print(f"... and {len(high_priority) - 10} more high priority models")
            
        # If list-only, just print the models and exit
        if args.list_only:
            print("\nAll missing tests:")
            for model in missing_tests:
                tasks = ", ".join(model["pipeline_tasks"])
                priority = model["priority"]
                print(f"- {model['model']} ({priority}): {tasks}")
            return
    except Exception as e:
        logger.error(f"Error identifying missing tests: {e}")
        sys.exit(1)
    
    # Generate test files in parallel
    try:
        messages = generate_test_files_parallel(
            missing_tests,
            existing_tests,
            all_models,
            args.output_dir,
            args.limit,
            args.high_priority_only
        )
        
        # Print messages
        for message in messages:
            print(message)
    except Exception as e:
        logger.error(f"Error generating test templates: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print("Complete!")

if __name__ == "__main__":
    main()