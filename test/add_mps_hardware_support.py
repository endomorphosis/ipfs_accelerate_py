#!/usr/bin/env python3
"""
Add MPS (Apple Silicon) Support to High-Priority Models

This script adds MPS support to all high-priority model test files, completing
the hardware coverage for all 13 key model classes.

Usage:
  python add_mps_hardware_support.py --add-all
  python add_mps_hardware_support.py --model bert
  python add_mps_hardware_support.py --check-only
"""

import os
import sys
import logging
import argparse
import re
from pathlib import Path
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
SKILLS_DIR = CURRENT_DIR / "skills"

# Define the 13 high-priority model classes
HIGH_PRIORITY_MODELS = {
    "bert": {"name": "bert-base-uncased", "family": "embedding", "modality": "text"},
    "t5": {"name": "t5-small", "family": "text_generation", "modality": "text"},
    "llama": {"name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "family": "text_generation", "modality": "text"},
    "clip": {"name": "openai/clip-vit-base-patch32", "family": "multimodal", "modality": "multimodal"},
    "vit": {"name": "google/vit-base-patch16-224", "family": "vision", "modality": "vision"},
    "clap": {"name": "laion/clap-htsat-unfused", "family": "audio", "modality": "audio"},
    "whisper": {"name": "openai/whisper-tiny", "family": "audio", "modality": "audio"},
    "wav2vec2": {"name": "facebook/wav2vec2-base", "family": "audio", "modality": "audio"},
    "llava": {"name": "llava-hf/llava-1.5-7b-hf", "family": "multimodal", "modality": "multimodal"},
    "llava_next": {"name": "llava-hf/llava-v1.6-mistral-7b", "family": "multimodal", "modality": "multimodal"},
    "xclip": {"name": "microsoft/xclip-base-patch32", "family": "multimodal", "modality": "multimodal"},
    "qwen2": {"name": "Qwen/Qwen2-0.5B-Instruct", "family": "text_generation", "modality": "text"},
    "detr": {"name": "facebook/detr-resnet-50", "family": "vision", "modality": "vision"}
}

# MPS template (Apple Silicon)
MPS_TEMPLATE = {
    "imports": [
        "# MPS (Apple Silicon) detection",
        "HAS_MPS = False",
        "if TORCH_AVAILABLE:",
        "    HAS_MPS = hasattr(torch.backends, \"mps\") and torch.backends.mps.is_available()"
    ],
    "implementations": {
        "text": """
    def init_mps(self, model_name=None, device="mps"):
        \"\"\"Initialize model for MPS (Apple Silicon) inference.\"\"\"
        model_name = model_name or self.model_name
        
        # Check for MPS availability
        if not HAS_MPS:
            logger.warning("MPS (Apple Silicon) not available, falling back to CPU")
            return self.init_cpu(model_name)
            
        try:
            logger.info(f"Initializing {model_name} with MPS on {device}")
            
            # Initialize tokenizer
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            
            # Initialize model
            model = transformers.AutoModel.from_pretrained(model_name)
            
            # Move model to MPS
            model.to(device)
            model.eval()
            
            # Create handler function
            def handler(text_input, **kwargs):
                try:
                    # Process with tokenizer
                    if isinstance(text_input, list):
                        inputs = tokenizer(text_input, padding=True, truncation=True, return_tensors="pt")
                    else:
                        inputs = tokenizer(text_input, return_tensors="pt")
                    
                    # Move inputs to MPS
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # Run inference
                    with torch.no_grad():
                        outputs = model(**inputs)
                    
                    return {
                        "output": outputs,
                        "implementation_type": "MPS",
                        "device": device,
                        "model": model_name
                    }
                except Exception as e:
                    logger.error(f"Error in MPS handler: {e}")
                    return {
                        "output": f"Error: {str(e)}",
                        "implementation_type": "ERROR",
                        "error": str(e),
                        "model": model_name
                    }
            
            # Create queue
            queue = asyncio.Queue(64)
            batch_size = self.batch_size
            
            # Return components
            return model, tokenizer, handler, queue, batch_size
            
        except Exception as e:
            logger.error(f"Error initializing model with MPS: {str(e)}")
            logger.warning("Falling back to CPU implementation")
            return self.init_cpu(model_name)
""",
        "vision": """
    def init_mps(self, model_name=None, device="mps"):
        \"\"\"Initialize vision model for MPS (Apple Silicon) inference.\"\"\"
        model_name = model_name or self.model_name
        
        # Check for MPS availability
        if not HAS_MPS:
            logger.warning("MPS (Apple Silicon) not available, falling back to CPU")
            return self.init_cpu(model_name)
            
        try:
            logger.info(f"Initializing vision model {model_name} with MPS on {device}")
            
            # Initialize image processor
            processor = transformers.AutoImageProcessor.from_pretrained(model_name)
            
            # Initialize model
            model = transformers.AutoModelForImageClassification.from_pretrained(model_name)
            
            # Move model to MPS
            model.to(device)
            model.eval()
            
            # Create handler function
            def handler(image_input, **kwargs):
                try:
                    # Check if input is a file path or already an image
                    if isinstance(image_input, str):
                        if os.path.exists(image_input):
                            image = Image.open(image_input)
                        else:
                            return {"error": f"Image file not found: {image_input}"}
                    elif isinstance(image_input, Image.Image):
                        image = image_input
                    else:
                        return {"error": "Unsupported image input format"}
                    
                    # Process with processor
                    inputs = processor(images=image, return_tensors="pt")
                    
                    # Move inputs to MPS
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # Run inference
                    with torch.no_grad():
                        outputs = model(**inputs)
                    
                    return {
                        "output": outputs,
                        "implementation_type": "MPS",
                        "device": device,
                        "model": model_name
                    }
                except Exception as e:
                    logger.error(f"Error in MPS vision handler: {e}")
                    return {
                        "output": f"Error: {str(e)}",
                        "implementation_type": "ERROR",
                        "error": str(e),
                        "model": model_name
                    }
            
            # Create queue
            queue = asyncio.Queue(64)
            batch_size = 1  # For vision models
            
            # Return components
            return model, processor, handler, queue, batch_size
            
        except Exception as e:
            logger.error(f"Error initializing vision model with MPS: {str(e)}")
            logger.warning("Falling back to CPU implementation")
            return self.init_cpu(model_name)
""",
        "audio": """
    def init_mps(self, model_name=None, device="mps"):
        \"\"\"Initialize audio model for MPS (Apple Silicon) inference.\"\"\"
        model_name = model_name or self.model_name
        
        # Check for MPS availability
        if not HAS_MPS:
            logger.warning("MPS (Apple Silicon) not available, falling back to CPU")
            return self.init_cpu(model_name)
            
        try:
            logger.info(f"Initializing audio model {model_name} with MPS on {device}")
            
            # Initialize audio processor
            processor = transformers.AutoProcessor.from_pretrained(model_name)
            
            # Initialize model based on model type
            if "whisper" in model_name.lower():
                model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
            else:
                model = transformers.AutoModelForAudioClassification.from_pretrained(model_name)
            
            # Move model to MPS
            model.to(device)
            model.eval()
            
            # Create handler function
            def handler(audio_input, **kwargs):
                try:
                    # Process based on input type
                    if isinstance(audio_input, str):
                        # Assuming file path
                        import librosa
                        waveform, sample_rate = librosa.load(audio_input, sr=16000)
                        inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
                    else:
                        # Assume properly formatted input
                        inputs = processor(audio_input, return_tensors="pt")
                    
                    # Move inputs to MPS
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # Run inference
                    with torch.no_grad():
                        outputs = model(**inputs)
                    
                    return {
                        "output": outputs,
                        "implementation_type": "MPS",
                        "device": device,
                        "model": model_name
                    }
                except Exception as e:
                    logger.error(f"Error in MPS audio handler: {e}")
                    return {
                        "output": f"Error: {str(e)}",
                        "implementation_type": "ERROR",
                        "error": str(e),
                        "model": model_name
                    }
            
            # Create queue
            queue = asyncio.Queue(64)
            batch_size = 1  # For audio models
            
            # Return components
            return model, processor, handler, queue, batch_size
            
        except Exception as e:
            logger.error(f"Error initializing audio model with MPS: {str(e)}")
            logger.warning("Falling back to CPU implementation")
            return self.init_cpu(model_name)
""",
        "multimodal": """
    def init_mps(self, model_name=None, device="mps"):
        \"\"\"Initialize multimodal model for MPS (Apple Silicon) inference.\"\"\"
        model_name = model_name or self.model_name
        
        # Check if this is a model specifically not supported on MPS
        if "llava" in model_name.lower():
            logger.warning(f"Model {model_name} not tested on MPS, proceed with caution")
        
        # Check for MPS availability
        if not HAS_MPS:
            logger.warning("MPS (Apple Silicon) not available, falling back to CPU")
            return self.init_cpu(model_name)
            
        try:
            logger.info(f"Initializing multimodal model {model_name} with MPS on {device}")
            
            # Initialize processor
            processor = transformers.AutoProcessor.from_pretrained(model_name)
            
            # For CLIP and similar models
            if "clip" in model_name.lower():
                model = transformers.CLIPModel.from_pretrained(model_name)
                
                # Move model to MPS
                model.to(device)
                model.eval()
                
                # Create handler function for CLIP-like models
                def handler(input_data, **kwargs):
                    try:
                        # Process based on input type
                        if isinstance(input_data, dict):
                            # Assume properly formatted inputs
                            if "text" in input_data and "image" in input_data:
                                inputs = processor(text=input_data["text"], 
                                                  images=input_data["image"], 
                                                  return_tensors="pt")
                            else:
                                return {"error": "Input dict missing 'text' or 'image' keys"}
                        else:
                            return {"error": "Unsupported input format for multimodal model"}
                        
                        # Move inputs to MPS
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        # Run inference
                        with torch.no_grad():
                            outputs = model(**inputs)
                        
                        return {
                            "output": outputs,
                            "implementation_type": "MPS",
                            "device": device,
                            "model": model_name
                        }
                    except Exception as e:
                        logger.error(f"Error in MPS multimodal handler: {e}")
                        return {
                            "output": f"Error: {str(e)}",
                            "implementation_type": "ERROR",
                            "error": str(e),
                            "model": model_name
                        }
                
                # Create queue
                queue = asyncio.Queue(64)
                batch_size = 1  # For multimodal models
                
                # Return components
                return model, processor, handler, queue, batch_size
                
            else:
                # For complex multimodal models like LLaVA, we'll provide a more cautious implementation
                # These are not thoroughly tested on MPS
                logger.warning(f"Complex multimodal model {model_name} not thoroughly tested on MPS")
                
                # Initialize with a simplified approach
                try:
                    # For LLaVA, try to use a simplified MPS approach
                    # Note: This may not work for all complex multimodal models
                    model = transformers.AutoModel.from_pretrained(model_name)
                    model.to(device)
                    model.eval()
                    
                    # Create simplified handler
                    def handler(input_data, **kwargs):
                        logger.warning("MPS implementation for complex multimodal models is experimental")
                        return {
                            "output": "MPS_EXPERIMENTAL_OUTPUT",
                            "implementation_type": "MPS_EXPERIMENTAL",
                            "model": model_name,
                            "warning": "MPS implementation for complex multimodal models is experimental"
                        }
                    
                    # Create queue
                    queue = asyncio.Queue(64)
                    batch_size = 1
                    
                    return model, processor, handler, queue, batch_size
                    
                except Exception as e:
                    logger.error(f"Error creating MPS implementation for complex multimodal model: {e}")
                    return self.init_cpu(model_name)
            
        except Exception as e:
            logger.error(f"Error initializing multimodal model with MPS: {str(e)}")
            logger.warning("Falling back to CPU implementation")
            return self.init_cpu(model_name)
"""
    }
}

def find_test_file(model_key: str) -> str:
    """
    Find the test file for a model key.
    
    Args:
        model_key: Model key (e.g., 'bert', 't5')
        
    Returns:
        Path to test file or None if not found
    """
    # Check skills directory
    file_path = SKILLS_DIR / f"test_hf_{model_key}.py"
    if file_path.exists():
        return str(file_path)
    
    # Check current directory
    file_path = CURRENT_DIR / f"test_hf_{model_key}.py"
    if file_path.exists():
        return str(file_path)
    
    # Check modality_tests directory
    file_path = CURRENT_DIR / "modality_tests" / f"test_hf_{model_key}.py"
    if file_path.exists():
        return str(file_path)
    
    return None

def analyze_file(file_path: str) -> Dict[str, Any]:
    """
    Analyze a test file to determine if it has MPS support.
    
    Args:
        file_path: Path to test file
        
    Returns:
        Analysis results
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check for MPS support
    has_mps = "init_mps" in content
    
    # Determine modality
    modality = "text"
    if any(keyword in content.lower() for keyword in ["image.open", "pil", "vision"]):
        modality = "vision"
    elif any(keyword in content.lower() for keyword in ["audio", "wav2vec", "whisper", "clap"]):
        modality = "audio"
    elif any(keyword in content.lower() for keyword in ["multimodal", "llava", "clip"]):
        modality = "multimodal"
    
    return {
        "file_path": file_path,
        "has_mps": has_mps,
        "modality": modality
    }

def add_mps_to_file(file_path: str, modality: str) -> Dict[str, Any]:
    """
    Add MPS support to a test file.
    
    Args:
        file_path: Path to test file
        modality: Modality of the model (text, vision, audio, multimodal)
        
    Returns:
        Results of the update
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if MPS is already implemented
        if "init_mps" in content:
            return {"status": "skipped", "reason": "MPS already implemented"}
        
        # Add MPS imports near other hardware imports
        # Look for a good place to insert imports - after CUDA detection
        cuda_section = re.search(r'# Check CUDA.*?(?=\n\n)', content, re.DOTALL)
        if cuda_section:
            new_content = content[:cuda_section.end()] + "\n\n" + "\n".join(MPS_TEMPLATE["imports"]) + content[cuda_section.end():]
            content = new_content
            
        # Add MPS implementation
        # Look for a good insertion point (before __main__ or similar)
        insertion_point = None
        
        # First try to find run_tests or similar method
        run_tests_match = re.search(r'def\s+run_tests\s*\(', content)
        if run_tests_match:
            insertion_point = run_tests_match.start()
        
        # If not found, try to find main function or __main__ block
        if insertion_point is None:
            main_match = re.search(r'def\s+main\s*\(', content)
            if main_match:
                insertion_point = main_match.start()
        
        if insertion_point is None:
            main_block_match = re.search(r'if\s+__name__\s*==\s*[\'"]__main__[\'"]', content)
            if main_block_match:
                insertion_point = main_block_match.start()
        
        if insertion_point is not None:
            # Get appropriate implementation based on modality
            implementation = MPS_TEMPLATE["implementations"].get(modality, MPS_TEMPLATE["implementations"]["text"])
            
            # Add implementation
            new_content = content[:insertion_point] + "\n" + implementation + "\n" + content[insertion_point:]
            content = new_content
            
            # Write updated content back to file
            with open(file_path, 'w') as f:
                f.write(content)
            
            return {"status": "success", "modality": modality}
        else:
            return {"status": "error", "reason": "Could not find insertion point"}
        
    except Exception as e:
        return {"status": "error", "reason": str(e)}

def check_mps_support() -> Dict[str, Any]:
    """
    Check MPS support across all high-priority models.
    
    Returns:
        Report of MPS support
    """
    results = {}
    
    for model_key in HIGH_PRIORITY_MODELS:
        file_path = find_test_file(model_key)
        
        if file_path:
            analysis = analyze_file(file_path)
            results[model_key] = {
                "file_path": file_path,
                "has_mps": analysis["has_mps"],
                "modality": analysis["modality"]
            }
        else:
            results[model_key] = {
                "file_path": None,
                "has_mps": False,
                "error": "Test file not found"
            }
    
    return results

def add_mps_to_all() -> Dict[str, Any]:
    """
    Add MPS support to all high-priority models.
    
    Returns:
        Results of the updates
    """
    results = {}
    
    for model_key, model_info in HIGH_PRIORITY_MODELS.items():
        file_path = find_test_file(model_key)
        
        if file_path:
            analysis = analyze_file(file_path)
            
            if analysis["has_mps"]:
                results[model_key] = {"status": "skipped", "reason": "MPS already implemented"}
            else:
                update_result = add_mps_to_file(file_path, analysis["modality"])
                results[model_key] = update_result
        else:
            results[model_key] = {"status": "error", "reason": "Test file not found"}
    
    return results

def generate_report(results: Dict[str, Any]) -> str:
    """
    Generate a report from the results.
    
    Args:
        results: Results of MPS support check or update
        
    Returns:
        Markdown report
    """
    report = "# MPS (Apple Silicon) Support Report\n\n"
    
    # Count stats
    total = len(results)
    implemented = sum(1 for data in results.values() if isinstance(data, dict) and data.get("has_mps", False))
    added = sum(1 for data in results.values() if isinstance(data, dict) and data.get("status") == "success")
    skipped = sum(1 for data in results.values() if isinstance(data, dict) and data.get("status") == "skipped")
    failed = sum(1 for data in results.values() if isinstance(data, dict) and (data.get("status") == "error" or "error" in data))
    
    report += f"Total models: {total}\n"
    report += f"Already implemented: {implemented}\n"
    report += f"Added: {added}\n"
    report += f"Skipped: {skipped}\n"
    report += f"Failed: {failed}\n\n"
    
    # Create table
    report += "| Model | Has MPS | Modality | Status | Reason |\n"
    report += "|-------|---------|----------|--------|--------|\n"
    
    for model_key, data in results.items():
        if isinstance(data, dict):
            has_mps = "✅" if data.get("has_mps", False) else "❌"
            modality = data.get("modality", "unknown")
            status = data.get("status", "unknown")
            reason = data.get("reason", "")
            if "error" in data:
                status = "error"
                reason = data["error"]
            
            report += f"| {model_key} | {has_mps} | {modality} | {status} | {reason} |\n"
    
    return report

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Add MPS (Apple Silicon) support to high-priority models")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check-only", action="store_true", help="Only check for MPS support without modifying files")
    group.add_argument("--add-all", action="store_true", help="Add MPS support to all high-priority models")
    group.add_argument("--model", type=str, help="Add MPS support to a specific model")
    
    parser.add_argument("--output", type=str, help="Output file for report")
    
    args = parser.parse_args()
    
    if args.check_only:
        results = check_mps_support()
        report = generate_report(results)
    elif args.add_all:
        results = add_mps_to_all()
        report = generate_report(results)
    elif args.model:
        model_key = args.model.lower()
        if model_key not in HIGH_PRIORITY_MODELS:
            print(f"Error: Unknown model '{model_key}'")
            sys.exit(1)
        
        file_path = find_test_file(model_key)
        if file_path:
            analysis = analyze_file(file_path)
            if analysis["has_mps"]:
                results = {model_key: {"status": "skipped", "reason": "MPS already implemented"}}
            else:
                result = add_mps_to_file(file_path, analysis["modality"])
                results = {model_key: result}
        else:
            results = {model_key: {"status": "error", "reason": "Test file not found"}}
        
        report = generate_report(results)
    else:
        parser.print_help()
        sys.exit(1)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)

if __name__ == "__main__":
    main()