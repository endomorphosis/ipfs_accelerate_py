#!/usr/bin/env python3

import os
import sys
import ast
import re
import argparse
import logging
from typing import List, Dict, Any, Optional, Tuple
import importlib.util

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Required methods for ModelTest compliance
REQUIRED_METHODS = [
    "test_model_loading",
    "load_model",
    "verify_model_output",
    "detect_preferred_device"
]

# Template for detecting preferred device method
DEVICE_DETECTION_TEMPLATE = """
def detect_preferred_device(self):
    \"\"\"Detect available hardware and choose the preferred device.\"\"\"
    try:
        # Check CUDA
        if HAS_TORCH and torch.cuda.is_available():
            return "cuda"
        
        # Check MPS (Apple Silicon)
        if HAS_TORCH and hasattr(torch, "mps") and torch.mps.is_available():
            return "mps"
        
        # Fallback to CPU
        return "cpu"
    except Exception as e:
        logger.error(f"Error detecting device: {e}")
        return "cpu"
"""

# Template for LLaVA load_model method
LLAVA_LOAD_MODEL_TEMPLATE = """
def load_model(self, model_name):
    \"\"\"Load a model for testing - implements required ModelTest method.\"\"\"
    try:
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers package not available")
            
        logger.info(f"Loading model {model_name}...")
        
        # Load processor and model
        # LLaVA uses a processor that combines clip image processor with LLM tokenizer
        processor = transformers.LlavaProcessor.from_pretrained(model_name)
        model = transformers.LlavaForConditionalGeneration.from_pretrained(model_name)
        
        # Move model to preferred device if possible
        if self.preferred_device == "cuda" and HAS_TORCH and torch.cuda.is_available():
            model = model.to("cuda")
        elif self.preferred_device == "mps" and HAS_TORCH and hasattr(torch, "mps") and torch.mps.is_available():
            model = model.to("mps")
            
        return {"model": model, "processor": processor}
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        raise
"""

# Template for LLaVA verify_model_output method
LLAVA_VERIFY_MODEL_TEMPLATE = """
def verify_model_output(self, model, input_data, expected_output=None):
    \"\"\"Verify that model produces expected output - implements required ModelTest method.\"\"\"
    try:
        if not isinstance(model, dict) or "model" not in model or "processor" not in model:
            raise ValueError("Model should be a dict containing 'model' and 'processor' keys")
            
        # Unpack model components
        llava_model = model["model"]
        processor = model["processor"]
        
        # Process inputs based on type
        if isinstance(input_data, dict):
            if "image" in input_data:
                image = input_data["image"]
            else:
                image = self._get_test_image()
                
            if "prompt" in input_data:
                prompt = input_data["prompt"]
            else:
                prompt = self.test_prompts[0] if hasattr(self, "test_prompts") else "What do you see in this image?"
        else:
            # Use default test inputs
            image = self._get_test_image()
            prompt = self.test_prompts[0] if hasattr(self, "test_prompts") else "What do you see in this image?"
        
        # Process image input
        if isinstance(image, str):
            # Load image from path
            if os.path.exists(image):
                image = Image.open(image)
            else:
                # Create a random dummy image
                image_size = getattr(self, "image_size", 336)
                image = Image.new('RGB', (image_size, image_size), color=(73, 109, 137))
        
        # Process inputs with LLaVA processor
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )
        
        # Move inputs to device if needed
        if hasattr(llava_model, "device") and str(llava_model.device) != "cpu":
            inputs = {k: v.to(llava_model.device) for k, v in inputs.items()}
        
        # Run generation
        with torch.no_grad():
            output_ids = llava_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )
        
        # Decode the generated text
        generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        # Create result with both input prompt and generated text
        result = {
            "prompt": prompt,
            "generated_text": generated_text
        }
        
        # Verify output
        self.assertIsNotNone(generated_text, "Model output should not be None")
        self.assertIsInstance(generated_text, str, "Generated text should be a string")
        self.assertGreater(len(generated_text), 0, "Generated text should not be empty")
        
        # If expected output is provided, compare with actual output
        if expected_output is not None:
            self.assertEqual(expected_output, result)
            
        return result
    except Exception as e:
        logger.error(f"Error verifying model output: {e}")
        raise
"""

# Template for LLaVA test_model_loading method
LLAVA_TEST_MODEL_LOADING_TEMPLATE = """
def test_model_loading(self):
    \"\"\"Test that the model loads correctly - implements required ModelTest method.\"\"\"
    try:
        model_components = self.load_model(self.model_id)
        self.assertIsNotNone(model_components, "Model should not be None")
        self.assertIn("model", model_components, "Model dict should contain 'model' key")
        self.assertIn("processor", model_components, "Model dict should contain 'processor' key")
        
        llava_model = model_components["model"]
        expected_class_name = self.class_name if hasattr(self, "class_name") else "LlavaForConditionalGeneration"
        actual_class_name = llava_model.__class__.__name__
        
        self.assertEqual(expected_class_name, actual_class_name, 
                         f"Model should be a {expected_class_name}, got {actual_class_name}")
        
        logger.info(f"Successfully loaded {self.model_id}")
        return model_components
    except Exception as e:
        logger.error(f"Error testing model loading: {e}")
        self.fail(f"Model loading failed: {e}")
"""

# Template for base ModelTest import section
MODEL_TEST_IMPORT_TEMPLATE = """
# Try to import ModelTest with fallbacks
try:
    from refactored_test_suite.model_test import ModelTest
except ImportError:
    try:
        from model_test import ModelTest
    except ImportError:
        # Create a minimal ModelTest class if not available
        class ModelTest(unittest.TestCase):
            \"\"\"Minimal ModelTest class if the real one is not available.\"\"\"
            def setUp(self):
                super().setUp()
"""

def load_file_content(file_path: str) -> str:
    """Load file content as string."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        return ""

def determine_model_type(file_path: str) -> str:
    """Determine model type (text, vision, audio, multimodal) from file content."""
    content = load_file_content(file_path)
    
    # Check for multimodal indicators
    multimodal_indicators = [
        "CLIP", "clip", "BLIP", "blip", "LLaVA", "llava", "vision-text", "vision_text", "VisionText",
        "image_and_text", "image and text", "vision and text", "multimodal", "vision-language"
    ]
    
    for indicator in multimodal_indicators:
        if indicator in content:
            return "multimodal"
    
    # Check for vision indicators
    vision_indicators = [
        "vision", "image", "Vision", "Image", "ViT", "Swin", "ConvNext", 
        "classification", "segmentation", "detection", "visual"
    ]
    
    for indicator in vision_indicators:
        if indicator in content:
            return "vision"
    
    # Check for audio indicators
    audio_indicators = [
        "audio", "Audio", "speech", "Speech", "Whisper", "wav2vec", 
        "ASR", "asr", "transcription", "hubert", "librosa", "soundfile"
    ]
    
    for indicator in audio_indicators:
        if indicator in content:
            return "audio"
    
    # Default to text
    return "text"

def analyze_file_ast(file_path: str) -> Dict[str, Any]:
    """Analyze Python file using AST to check ModelTest compliance."""
    try:
        content = load_file_content(file_path)
        if not content:
            return {"error": "Empty file or error reading file"}
            
        tree = ast.parse(content)
        
        # Find all class definitions and their methods
        classes = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                classes[node.name] = {
                    "name": node.name,
                    "methods": methods,
                    "bases": [base.id if isinstance(base, ast.Name) else None for base in node.bases]
                }
        
        # Find the primary test class
        primary_class = None
        for class_name, class_info in classes.items():
            if class_name.startswith("Test"):
                primary_class = class_name
                break
        
        # Check method compliance
        missing_methods = []
        if primary_class:
            class_info = classes[primary_class]
            missing_methods = [method for method in REQUIRED_METHODS if method not in class_info["methods"]]
            
            # Check if currently ModelTest subclass
            is_model_test = "ModelTest" in class_info["bases"] if "bases" in class_info else False
        else:
            is_model_test = False
        
        return {
            "primary_class": primary_class,
            "classes": classes,
            "missing_methods": missing_methods,
            "is_model_test": is_model_test
        }
    except Exception as e:
        logger.error(f"Error analyzing file {file_path}: {e}")
        return {"error": str(e)}

def add_missing_methods(content: str, analysis: Dict[str, Any], model_type: str) -> str:
    """Add missing methods to make file compliant with ModelTest."""
    if "error" in analysis:
        logger.error(f"Cannot add methods due to analysis error: {analysis['error']}")
        return content
    
    if not analysis.get("primary_class"):
        logger.error("No primary test class found")
        return content
    
    missing_methods = analysis.get("missing_methods", [])
    
    # If no methods are missing, no changes needed
    if not missing_methods:
        return content
    
    # Find the class definition
    class_name = analysis["primary_class"]
    class_pattern = rf"class\s+{class_name}[\(\s:]"
    
    # Find class end (last method in class)
    class_match = re.search(class_pattern, content)
    if not class_match:
        logger.error(f"Cannot find class pattern for {class_name}")
        return content
    
    # Find appropriate indentation
    indentation = "    "  # Default
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if re.search(class_pattern, line):
            # Find indentation in next method
            for j in range(i+1, len(lines)):
                if re.match(r'\s+def\s+', lines[j]):
                    indentation = re.match(r'(\s+)', lines[j]).group(1)
                    break
            break
    
    # Check if this is a LLaVA-style file
    is_llava = False
    if model_type == "multimodal" and ("llava" in content.lower() or "LLaVA" in content):
        is_llava = True
        logger.info(f"Detected LLaVA-style model in {class_name}")
    
    # Process each missing method
    for method in missing_methods:
        # Choose appropriate template based on model type and method
        method_template = None
        
        # Handle detect_preferred_device for all types
        if method == "detect_preferred_device":
            method_template = DEVICE_DETECTION_TEMPLATE
        
        # Handle LLaVA specific templates
        elif is_llava:
            if method == "load_model":
                method_template = LLAVA_LOAD_MODEL_TEMPLATE
            elif method == "verify_model_output":
                method_template = LLAVA_VERIFY_MODEL_TEMPLATE
            elif method == "test_model_loading":
                method_template = LLAVA_TEST_MODEL_LOADING_TEMPLATE
        
        # Skip if no template available
        if not method_template:
            logger.warning(f"No template available for method {method} for model type {model_type}")
            continue
        
        # Format the template with proper indentation
        formatted_template = method_template.replace("\n", f"\n{indentation}")
        
        # Find where to insert
        last_method_pattern = r"(\s+)def\s+[a-zA-Z0-9_]+\([^)]*\)[^:]*:"
        last_method_matches = list(re.finditer(last_method_pattern, content))
        
        if last_method_matches:
            last_method = last_method_matches[-1]
            last_method_end = content.find("\n\n", last_method.end())
            
            if last_method_end == -1:  # If no double newline found
                last_method_end = len(content)
            
            # Insert after the last method
            content = content[:last_method_end] + "\n\n" + indentation + formatted_template.lstrip() + content[last_method_end:]
        else:
            # If no methods found, add after class definition
            class_end = content.find("\n", class_match.end())
            if class_end != -1:
                content = content[:class_end+1] + "\n" + indentation + formatted_template.lstrip() + content[class_end+1:]
        
        logger.info(f"Added method {method} to class {class_name}")
    
    return content

def convert_to_model_test(file_path: str, output_path: Optional[str] = None, overwrite: bool = False) -> Tuple[bool, str]:
    """Convert a test file to conform to ModelTest pattern."""
    try:
        # Load file content
        content = load_file_content(file_path)
        if not content:
            return False, "Failed to load file content"
        
        # Determine model type
        model_type = determine_model_type(file_path)
        logger.info(f"Detected model type: {model_type}")
        
        # Analyze file
        analysis = analyze_file_ast(file_path)
        
        # Check if already compliant
        if analysis.get("is_model_test") and not analysis.get("missing_methods"):
            logger.info(f"File {file_path} is already ModelTest compliant")
            return True, "Already compliant"
        
        # If class inherits from unittest.TestCase but not ModelTest, change the inheritance
        if analysis.get("primary_class"):
            class_name = analysis["primary_class"]
            class_info = analysis["classes"][class_name]
            
            if "unittest.TestCase" in str(class_info["bases"]) and "ModelTest" not in class_info["bases"]:
                # Change inheritance
                content = re.sub(
                    rf"class\s+{class_name}\(unittest\.TestCase\)",
                    f"class {class_name}(ModelTest)",
                    content
                )
                
                # Add ModelTest import if not present
                if "from refactored_test_suite.model_test import ModelTest" not in content and "from model_test import ModelTest" not in content:
                    import_section_end = content.find("\n\n", content.find("import "))
                    if import_section_end != -1:
                        content = content[:import_section_end] + MODEL_TEST_IMPORT_TEMPLATE + content[import_section_end:]
                
                # Update setUp method if it exists
                setup_pattern = r"(\s+)def\s+setUp\(self\)[^:]*:(.*?)(?=\n\s+def|\Z)"
                setup_match = re.search(setup_pattern, content, re.DOTALL)
                
                if setup_match:
                    indentation = setup_match.group(1)
                    setup_body = setup_match.group(2)
                    
                    # Check if super().setUp() is called
                    if "super().setUp()" not in setup_body:
                        # Add super().setUp() at the start of the method
                        new_setup = f"{indentation}def setUp(self):{indentation}    super().setUp(){setup_body}"
                        content = content.replace(setup_match.group(0), new_setup)
        
        # Add missing methods
        content = add_missing_methods(content, analysis, model_type)
        
        # Write back to file
        if output_path:
            # Write to output path
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Standardized file written to {output_path}")
        elif overwrite:
            # Overwrite original file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Original file {file_path} overwritten with standardized version")
        else:
            # Just return the content without writing
            logger.info(f"File {file_path} standardized (not written)")
        
        return True, content
    except Exception as e:
        logger.error(f"Error standardizing file {file_path}: {e}")
        return False, str(e)

def find_test_files(directory: str, pattern: str = "test_*.py") -> List[str]:
    """Find test files matching the pattern in the given directory."""
    import glob
    
    # Find all test files
    search_pattern = os.path.join(directory, pattern)
    test_files = glob.glob(search_pattern)
    
    return test_files

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Standardize test files to conform to ModelTest pattern")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Path to a specific test file")
    group.add_argument("--directory", type=str, help="Directory to scan for test files")
    parser.add_argument("--output-dir", type=str, help="Output directory for standardized files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite original files")
    parser.add_argument("--pattern", type=str, default="test_*.py", help="File pattern to match (default: test_*.py)")
    
    args = parser.parse_args()
    
    if args.file:
        # Standardize a single file
        output_path = os.path.join(args.output_dir, os.path.basename(args.file)) if args.output_dir else None
        success, result = convert_to_model_test(args.file, output_path, args.overwrite)
        
        if success:
            logger.info(f"✅ Successfully standardized {args.file}")
        else:
            logger.error(f"❌ Failed to standardize {args.file}: {result}")
    
    elif args.directory:
        # Find test files in directory
        test_files = find_test_files(args.directory, args.pattern)
        logger.info(f"Found {len(test_files)} test files in {args.directory}")
        
        # Create output directory if needed
        if args.output_dir and not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        # Process each file
        successful = 0
        for file_path in test_files:
            output_path = os.path.join(args.output_dir, os.path.basename(file_path)) if args.output_dir else None
            success, result = convert_to_model_test(file_path, output_path, args.overwrite)
            
            if success:
                logger.info(f"✅ Successfully standardized {file_path}")
                successful += 1
            else:
                logger.error(f"❌ Failed to standardize {file_path}: {result}")
        
        logger.info(f"Standardized {successful} of {len(test_files)} files")

if __name__ == "__main__":
    main()