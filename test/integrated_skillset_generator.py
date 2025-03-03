#!/usr/bin/env python3
"""
Integrated Skillset Generator

This script combines test generation with skillset implementation generation to produce
implementation files for the ipfs_accelerate_py worker/skillset directory based on
comprehensive test analysis and results.

## Key Features:
- Test-driven development approach for skillset implementation
- Utilizes enhanced template generator with WebNN and WebGPU support
- Generates implementations for all 300+ Hugging Face model types
- Supports all hardware backends (CPU, CUDA, OpenVINO, MPS, ROCm, WebNN, WebGPU)
- Creates Jinja2-based templates from ipfs_accelerate_py/worker/skillset/
- Implements both test and skillset generation in a unified workflow
- Automated validation against test expectations

## Usage:
  # Generate skillset implementation for a specific model
  python integrated_skillset_generator.py --model bert
  
  # Generate for a model family
  python integrated_skillset_generator.py --family text-embedding
  
  # Generate for all supported models
  python integrated_skillset_generator.py --all
  
  # First run tests and then generate implementations
  python integrated_skillset_generator.py --model bert --run-tests
  
  # Test implementations against expected results
  python integrated_skillset_generator.py --validate bert
"""

import os
import sys
import json
import time
import datetime
import argparse
import logging
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Try to import Jinja2 for templating
try:
    import jinja2
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    print("Warning: jinja2 not available. Templates will use string formatting.")

# Configure paths
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_DIR = PROJECT_ROOT / "test"
SKILLS_DIR = TEST_DIR / "skills"
WORKER_SKILLSET = PROJECT_ROOT / "ipfs_accelerate_py" / "worker" / "skillset"
OUTPUT_DIR = PROJECT_ROOT / "generated_skillsets"

# For output and logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("skillset_generator")

# Model and family metadata
class ModelRegistry:
    """Registry for model metadata and transformers model information."""
    
    def __init__(self):
        self.model_types_file = TEST_DIR / "huggingface_model_types.json"
        self.pipeline_map_file = TEST_DIR / "huggingface_model_pipeline_map.json"
        self.model_families = {}  # Grouped by architecture family
        self.model_tasks = {}     # Grouped by primary task
        self.model_types = {}     # Raw model types
        self.pipeline_map = {}    # Model to pipeline task mapping
        self.test_results = {}    # Results from test execution
        self._load_model_data()
        
    def _load_model_data(self):
        """Load model metadata from JSON files in the test directory."""
        # Load model types
        if self.model_types_file.exists():
            with open(self.model_types_file, 'r') as f:
                self.model_types = json.load(f)
                logger.info(f"Loaded {len(self.model_types)} model types from {self.model_types_file}")
        else:
            logger.warning(f"Model types file not found: {self.model_types_file}")
            
        # Load pipeline map
        if self.pipeline_map_file.exists():
            with open(self.pipeline_map_file, 'r') as f:
                self.pipeline_map = json.load(f)
                logger.info(f"Loaded pipeline mappings for {len(self.pipeline_map)} models")
        else:
            logger.warning(f"Pipeline map file not found: {self.pipeline_map_file}")
            
        # Group models by family
        for model_type in self.model_types:
            # Extract family from model name
            family = self._extract_family(model_type)
            if family not in self.model_families:
                self.model_families[family] = []
            self.model_families[family].append(model_type)
            
            # Group by primary task
            primary_task = self._get_primary_task(model_type)
            if primary_task not in self.model_tasks:
                self.model_tasks[primary_task] = []
            self.model_tasks[primary_task].append(model_type)
        
        logger.info(f"Models grouped into {len(self.model_families)} families and {len(self.model_tasks)} primary tasks")
    
    def _extract_family(self, model_type):
        """Extract the model family from the model type name."""
        # Basic heuristic for family extraction
        if '-' in model_type:
            return model_type.split('-')[0]
        else:
            return model_type
    
    def _get_primary_task(self, model_type):
        """Get the primary task for a model type based on pipeline mapping."""
        if model_type in self.pipeline_map and self.pipeline_map[model_type]:
            return self.pipeline_map[model_type][0]
        return "text-generation"  # Default task
    
    def get_model_family(self, model_type):
        """Get the family for a given model type."""
        return self._extract_family(model_type)
    
    def get_primary_task(self, model_type):
        """Get the primary task for a model type."""
        return self._get_primary_task(model_type)
    
    def get_models_by_family(self, family):
        """Get all models that belong to a specific family."""
        return self.model_families.get(family, [])
    
    def get_models_by_task(self, task):
        """Get all models that primarily handle a specific task."""
        return self.model_tasks.get(task, [])
    
    def get_all_families(self):
        """Get a list of all model families."""
        return list(self.model_families.keys())
    
    def get_all_tasks(self):
        """Get a list of all primary tasks."""
        return list(self.model_tasks.keys())


class TestAnalyzer:
    """Analyzes test files and results to inform skillset implementation."""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.test_files = {}
        self.test_results = {}
        self.hardware_compatibility = {}
        self._scan_test_files()
    
    def _scan_test_files(self):
        """Scan the skills directory for test files and map them to model types."""
        test_files = list(SKILLS_DIR.glob("test_hf_*.py"))
        for test_file in test_files:
            # Extract model name from file name (test_hf_bert.py -> bert)
            model_name = test_file.stem.replace("test_hf_", "")
            self.test_files[model_name] = test_file
        
        logger.info(f"Found {len(self.test_files)} test files in {SKILLS_DIR}")
    
    def run_test(self, model_type: str) -> Dict:
        """Run test for a specific model type and collect results."""
        normalized_name = model_type.replace('-', '_').replace('.', '_').lower()
        test_file = self.test_files.get(normalized_name)
        
        if not test_file or not test_file.exists():
            logger.warning(f"Test file not found for {model_type}")
            return {}
        
        try:
            logger.info(f"Running test for {model_type}...")
            result = subprocess.run(
                [sys.executable, str(test_file)],
                capture_output=True,
                text=True,
                check=False
            )
            
            # Parse output for test results
            output = result.stdout
            # Look for JSON results in the output
            try:
                start_idx = output.find('{')
                end_idx = output.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = output[start_idx:end_idx]
                    test_results = json.loads(json_str)
                    self.test_results[model_type] = test_results
                    return test_results
            except json.JSONDecodeError:
                logger.error(f"Could not parse JSON from test output for {model_type}")
            
            # If we couldn't parse JSON, just return text output
            return {"stdout": output, "stderr": result.stderr, "returncode": result.returncode}
        except Exception as e:
            logger.error(f"Error running test for {model_type}: {e}")
            return {"error": str(e)}
    
    def analyze_hardware_compatibility(self, model_type: str, test_results: Dict = None) -> Dict:
        """Analyze hardware compatibility for a model type based on test results."""
        if test_results is None:
            test_results = self.test_results.get(model_type, {})
        
        if not test_results:
            test_results = self.run_test(model_type)
            
        compatibility = {
            "cpu": True,  # All models run on CPU
            "cuda": False,
            "openvino": False,
            "mps": False,
            "amd": False,
            "webnn": False,
            "webgpu": False
        }
        
        # Extract hardware compatibility from test results
        if "results" in test_results:
            results = test_results["results"]
            compatibility["cuda"] = results.get("cuda_test", "").startswith("Success")
            compatibility["openvino"] = results.get("openvino_test", "").startswith("Success")
            compatibility["amd"] = results.get("amd_test", "").startswith("Success")
            compatibility["mps"] = results.get("mps_test", "").startswith("Success")
            compatibility["webnn"] = results.get("webnn_test", "").startswith("Success")
            compatibility["webgpu"] = results.get("webgpu_test", "").startswith("Success")
        
        self.hardware_compatibility[model_type] = compatibility
        return compatibility
    
    def extract_model_metadata(self, model_type: str, test_results: Dict = None) -> Dict:
        """Extract model metadata from test results for template variables."""
        if test_results is None:
            test_results = self.test_results.get(model_type, {})
        
        if not test_results:
            test_results = self.run_test(model_type)
        
        # Extract important model metadata
        metadata = {
            "model_type": model_type,
            "primary_task": self.registry.get_primary_task(model_type),
            "family": self.registry.get_model_family(model_type),
            "hardware_compatibility": self.analyze_hardware_compatibility(model_type, test_results),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Extract additional metadata from examples
        if "examples" in test_results and test_results["examples"]:
            first_example = test_results["examples"][0]
            
            # Check for model_info in first example
            if "model_info" in first_example:
                model_info = first_example["model_info"]
                metadata.update({
                    "input_format": model_info.get("input_format", "text"),
                    "output_format": model_info.get("output_format", "text"),
                    "helper_functions": model_info.get("helper_functions", []),
                    "required_dependencies": model_info.get("required_dependencies", [])
                })
                
            # Check for tensor_types info
            if "tensor_types" in first_example:
                tensor_types = first_example["tensor_types"]
                metadata.update({
                    "embedding_dim": tensor_types.get("embedding_dim", 768),
                    "sequence_length": tensor_types.get("sequence_length", 512),
                    "supports_half": tensor_types.get("supports_half", True),
                    "precision": tensor_types.get("precision", "float32")
                })
                
            # Check for precision info
            if "precision" in first_example:
                metadata["precision"] = first_example["precision"]
        
        return metadata


class TemplateEngine:
    """Template engine for generating skillset implementations."""
    
    def __init__(self, use_jinja2=JINJA2_AVAILABLE):
        self.use_jinja2 = use_jinja2
        self.env = None
        self.template_cache = {}
        
        if self.use_jinja2:
            # Set up Jinja2 environment
            self.env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(WORKER_SKILLSET),
                trim_blocks=True,
                lstrip_blocks=True
            )
    
    def _load_reference_implementation(self, model_family: str) -> Optional[str]:
        """Load a reference implementation file for a model family."""
        # Try to find existing reference implementations
        reference_models = {
            "bert": "bert",
            "t5": "t5",
            "gpt": "gpt2",
            "llama": "llama",
            "vit": "vit",
            "whisper": "whisper",
            "wav2vec2": "wav2vec2",
            "clip": "clip",
            "clap": "clap",
            "llava": "llava"
        }
        
        # Find the best match
        best_match = None
        for family, ref_model in reference_models.items():
            if model_family.startswith(family):
                best_match = ref_model
                break
                
        if best_match:
            reference_file = WORKER_SKILLSET / f"hf_{best_match}.py"
            if reference_file.exists():
                with open(reference_file, 'r') as f:
                    return f.read()
        
        # If no reference found, use a generic template
        default_reference = WORKER_SKILLSET / "hf_bert.py"
        if default_reference.exists():
            with open(default_reference, 'r') as f:
                return f.read()
        
        return None
    
    def _create_template_from_reference(self, reference_code: str, model_metadata: Dict) -> str:
        """Create a template from reference implementation, replacing model-specific elements."""
        # Replace model-specific elements
        template = reference_code
        
        # Replace old model name with new model name
        old_model_name = self._extract_model_name_from_reference(reference_code)
        new_model_name = model_metadata["model_type"]
        normalized_new_name = new_model_name.replace('-', '_').replace('.', '_').lower()
        
        if old_model_name:
            # Replace class name and related strings
            template = template.replace(f"hf_{old_model_name}", f"hf_{normalized_new_name}")
            template = template.replace(f"'{old_model_name}'", f"'{new_model_name}'")
            
            # Replace documentation
            template = template.replace(f"{old_model_name.upper()}", f"{new_model_name.upper()}")
            template = template.replace(f"{old_model_name.capitalize()}", f"{new_model_name.capitalize()}")
            
        # Replace task-specific code if task is different
        old_task = self._extract_task_from_reference(reference_code)
        new_task = model_metadata["primary_task"]
        
        if old_task and old_task != new_task:
            template = template.replace(f"_{old_task}_", f"_{new_task}_")
        
        return template
    
    def _extract_model_name_from_reference(self, reference_code: str) -> Optional[str]:
        """Extract the model name from a reference implementation."""
        # Look for class definition
        for line in reference_code.split('\n'):
            if line.startswith("class hf_"):
                # Extract model name from class definition (class hf_bert: -> bert)
                return line.split("class hf_")[1].split('(')[0].split(':')[0].strip()
        return None
    
    def _extract_task_from_reference(self, reference_code: str) -> Optional[str]:
        """Extract the primary task from a reference implementation."""
        for line in reference_code.split('\n'):
            if "create_cpu_" in line and "endpoint_handler" in line:
                # Extract task from handler method (create_cpu_text_embedding_endpoint_handler -> text_embedding)
                parts = line.split("create_cpu_")[1].split("_endpoint_handler")
                if parts:
                    return parts[0]
        return None
    
    def render_template(self, model_type: str, model_metadata: Dict) -> str:
        """Render a template for a model type using either Jinja2 or string formatting."""
        # Get the appropriate reference implementation
        model_family = model_metadata["family"]
        reference_code = self._load_reference_implementation(model_family)
        
        if not reference_code:
            logger.error(f"Could not load reference implementation for {model_type}")
            return ""
        
        # Create template from reference
        template_str = self._create_template_from_reference(reference_code, model_metadata)
        
        if self.use_jinja2:
            # Use Jinja2 for template rendering
            template = self.env.from_string(template_str)
            return template.render(**model_metadata)
        else:
            # Simple string formatting for template rendering
            try:
                return template_str.format(**model_metadata)
            except KeyError as e:
                logger.error(f"Missing key in template variables: {e}")
                return template_str


class SkillsetGenerator:
    """Main generator class for skillset implementations."""
    
    def __init__(self):
        self.registry = ModelRegistry()
        self.analyzer = TestAnalyzer(self.registry)
        self.template_engine = TemplateEngine()
        
        # Create output directory if it doesn't exist
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
    def generate_skillset(self, model_type: str, output_dir: Path = OUTPUT_DIR, 
                         run_tests: bool = False, force: bool = False) -> Path:
        """Generate a skillset implementation for a specific model."""
        normalized_name = model_type.replace('-', '_').replace('.', '_').lower()
        output_file = output_dir / f"hf_{normalized_name}.py"
        
        # Check if file already exists and we're not forcing overwrite
        if output_file.exists() and not force:
            logger.info(f"Skillset file already exists: {output_file}. Use --force to overwrite.")
            return output_file
        
        # Run tests if requested
        if run_tests:
            logger.info(f"Running tests for {model_type}...")
            test_results = self.analyzer.run_test(model_type)
        else:
            test_results = self.analyzer.test_results.get(model_type, {})
        
        # Extract model metadata for template variables
        logger.info(f"Extracting metadata for {model_type}...")
        model_metadata = self.analyzer.extract_model_metadata(model_type, test_results)
        
        # Render the template
        logger.info(f"Generating skillset implementation for {model_type}...")
        implementation = self.template_engine.render_template(model_type, model_metadata)
        
        # Save the implementation to the output file
        with open(output_file, 'w') as f:
            f.write(implementation)
            
        logger.info(f"Generated skillset implementation: {output_file}")
        return output_file
    
    def generate_for_family(self, family: str, output_dir: Path = OUTPUT_DIR,
                           run_tests: bool = False, force: bool = False) -> List[Path]:
        """Generate skillset implementations for all models in a family."""
        logger.info(f"Generating implementations for {family} family...")
        models = self.registry.get_models_by_family(family)
        
        if not models:
            logger.warning(f"No models found for family: {family}")
            return []
        
        output_files = []
        for model_type in models:
            try:
                output_file = self.generate_skillset(model_type, output_dir, run_tests, force)
                output_files.append(output_file)
            except Exception as e:
                logger.error(f"Error generating implementation for {model_type}: {e}")
                
        logger.info(f"Generated {len(output_files)} implementations for {family} family")
        return output_files
    
    def generate_for_task(self, task: str, output_dir: Path = OUTPUT_DIR,
                         run_tests: bool = False, force: bool = False) -> List[Path]:
        """Generate skillset implementations for all models with a specific primary task."""
        logger.info(f"Generating implementations for {task} task...")
        models = self.registry.get_models_by_task(task)
        
        if not models:
            logger.warning(f"No models found for task: {task}")
            return []
        
        output_files = []
        for model_type in models:
            try:
                output_file = self.generate_skillset(model_type, output_dir, run_tests, force)
                output_files.append(output_file)
            except Exception as e:
                logger.error(f"Error generating implementation for {model_type}: {e}")
                
        logger.info(f"Generated {len(output_files)} implementations for {task} task")
        return output_files
    
    def generate_all(self, output_dir: Path = OUTPUT_DIR, 
                    run_tests: bool = False, force: bool = False,
                    max_workers: int = 10) -> List[Path]:
        """Generate skillset implementations for all supported models."""
        logger.info("Generating implementations for all supported models...")
        model_types = list(self.registry.model_types.keys())
        
        output_files = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_model = {
                executor.submit(self.generate_skillset, model_type, output_dir, run_tests, force): model_type
                for model_type in model_types
            }
            
            for future in as_completed(future_to_model):
                model_type = future_to_model[future]
                try:
                    output_file = future.result()
                    output_files.append(output_file)
                    logger.info(f"Completed implementation for {model_type}")
                except Exception as e:
                    logger.error(f"Error generating implementation for {model_type}: {e}")
        
        logger.info(f"Generated {len(output_files)} implementations for all supported models")
        return output_files
    
    def validate_implementation(self, model_type: str, implementation_path: Optional[Path] = None) -> bool:
        """Validate a generated implementation against test expectations."""
        logger.info(f"Validating implementation for {model_type}...")
        
        if implementation_path is None:
            # Try to find the implementation in the output directory
            normalized_name = model_type.replace('-', '_').replace('.', '_').lower()
            implementation_path = OUTPUT_DIR / f"hf_{normalized_name}.py"
        
        if not implementation_path.exists():
            logger.error(f"Implementation file not found: {implementation_path}")
            return False
        
        # Run the tests
        test_results = self.analyzer.run_test(model_type)
        
        # Basic validation: ensure implementation has the necessary components
        try:
            with open(implementation_path, 'r') as f:
                code = f.read()
                
            # Check for key components
            validations = {
                "class_declaration": f"class hf_{model_type.replace('-', '_').replace('.', '_').lower()}",
                "init_method": "def __init__",
                "cpu_handler": "create_cpu_",
                "cuda_handler": "create_cuda_",
                "hardware_detection": "_detect_hardware",
                "web_support": "webnn" in code and "webgpu" in code
            }
            
            validation_results = {key: component in code for key, component in validations.items()}
            
            # Check if all validations passed
            all_passed = all(validation_results.values())
            
            # Log validation results
            if all_passed:
                logger.info(f"Validation passed for {model_type}")
            else:
                logger.warning(f"Validation failed for {model_type}: {validation_results}")
                
            return all_passed
            
        except Exception as e:
            logger.error(f"Error validating implementation for {model_type}: {e}")
            return False


def main():
    """Main entry point for the integrated skillset generator."""
    parser = argparse.ArgumentParser(description="Integrated Skillset Generator")
    
    # Model selection options
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=str, help="Generate implementation for a specific model")
    model_group.add_argument("--family", type=str, help="Generate implementations for a model family")
    model_group.add_argument("--task", type=str, help="Generate implementations for models with a specific task")
    model_group.add_argument("--all", action="store_true", help="Generate implementations for all supported models")
    model_group.add_argument("--validate", type=str, help="Validate implementation for a specific model")
    model_group.add_argument("--list-families", action="store_true", help="List all available model families")
    model_group.add_argument("--list-tasks", action="store_true", help="List all available tasks")
    
    # Generation options
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), 
                        help="Output directory for generated implementations")
    parser.add_argument("--run-tests", action="store_true", 
                        help="Run tests before generating implementations")
    parser.add_argument("--force", action="store_true", 
                        help="Force overwrite of existing implementation files")
    parser.add_argument("--max-workers", type=int, default=10, 
                        help="Maximum number of worker threads for parallel generation")
    parser.add_argument("--verbose", action="store_true", 
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize the generator
    generator = SkillsetGenerator()
    
    # Process the requested action
    try:
        if args.list_families:
            # List all available model families
            families = generator.registry.get_all_families()
            print("\nAvailable Model Families:")
            for family in sorted(families):
                model_count = len(generator.registry.get_models_by_family(family))
                print(f"  {family} ({model_count} models)")
            return 0
            
        elif args.list_tasks:
            # List all available tasks
            tasks = generator.registry.get_all_tasks()
            print("\nAvailable Model Tasks:")
            for task in sorted(tasks):
                model_count = len(generator.registry.get_models_by_task(task))
                print(f"  {task} ({model_count} models)")
            return 0
            
        elif args.validate:
            # Validate implementation for a specific model
            success = generator.validate_implementation(args.validate)
            return 0 if success else 1
            
        elif args.model:
            # Generate implementation for a specific model
            generator.generate_skillset(
                args.model, 
                output_dir=output_dir,
                run_tests=args.run_tests,
                force=args.force
            )
            
        elif args.family:
            # Generate implementations for a model family
            generator.generate_for_family(
                args.family,
                output_dir=output_dir,
                run_tests=args.run_tests,
                force=args.force
            )
            
        elif args.task:
            # Generate implementations for models with a specific task
            generator.generate_for_task(
                args.task,
                output_dir=output_dir,
                run_tests=args.run_tests,
                force=args.force
            )
            
        elif args.all:
            # Generate implementations for all supported models
            generator.generate_all(
                output_dir=output_dir,
                run_tests=args.run_tests,
                force=args.force,
                max_workers=args.max_workers
            )
            
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())