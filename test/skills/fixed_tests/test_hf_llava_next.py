#!/usr/bin/env python3

"""
Test file for LLaVA-Next HuggingFace models.
"""

import os
import sys
import json
import time
import logging
import numpy as np
import argparse
from unittest.mock import MagicMock
from pathlib import Path
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if dependencies are available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = MagicMock()
    HAS_TORCH = False
    logger.warning("torch not available, using mock")

try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = MagicMock()
    HAS_TRANSFORMERS = False
    logger.warning("transformers not available, using mock")

# Model registry for LLaVA-Next models
LLAVA_NEXT_MODELS_REGISTRY = {
    "llava-hf/llava-v1.6-mistral-7b-hf": {
        "full_name": "LLaVA-Next Mistral 7B",
        "architecture": "multimodal",
        "description": "LLaVA-Next model based on Mistral 7B",
        "model_type": "llava-next",
        "parameters": "7B",
        "image_size": 336,
        "vision_model": "CLIP ViT-L/14",
        "text_model": "Mistral 7B",
        "recommended_tasks": ["visual-question-answering", "image-to-text"]
    },
    "llava-hf/llava-v1.6-34b-hf": {
        "full_name": "LLaVA-Next 34B",
        "architecture": "multimodal",
        "description": "LLaVA-Next model with 34B parameters",
        "model_type": "llava-next",
        "parameters": "34B",
        "image_size": 336,
        "vision_model": "CLIP ViT-L/14",
        "text_model": "Yi 34B",
        "recommended_tasks": ["visual-question-answering", "image-to-text"]
    }
}

def select_device():
    """Select the best available device for inference."""
    if HAS_TORCH:
        if torch.cuda.is_available():
            return "cuda:0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    return "cpu"

class TestLlavaNextModels:
    """
    Test class for LLaVA-Next models.
    """
    
    def __init__(self, model_id="llava-hf/llava-v1.6-mistral-7b-hf", device=None):
        """Initialize the test class for LLaVA-Next models.
        
        Args:
            model_id: The model ID to test (default: "llava-hf/llava-v1.6-mistral-7b-hf")
            device: The device to run tests on (default: None = auto-select)
        """
        self.model_id = model_id
        self.device = device if device else select_device()
        self.performance_stats = {}
        self.model_info = LLAVA_NEXT_MODELS_REGISTRY.get(model_id, {})
        self.image_size = self.model_info.get("image_size", 336)
    
    def _create_test_image(self, height=None, width=None):
        """Create a test image with colored shapes."""
        height = height or self.image_size
        width = width or self.image_size
        
        # Create a blank image with a white background
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Add a red square
        square_size = min(height, width) // 4
        x1, y1 = width // 4, height // 4
        x2, y2 = x1 + square_size, y1 + square_size
        image[y1:y2, x1:x2] = [255, 0, 0]  # Red
        
        # Add a blue circle
        center_x, center_y = width // 2 + width // 4, height // 2
        radius = min(height, width) // 6
        y, x = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        circle_mask = dist_from_center <= radius
        image[circle_mask] = [0, 0, 255]  # Blue
        
        # Add a green triangle
        triangle_size = min(height, width) // 3
        x1, y1 = width // 8, height // 2 + height // 4  # Bottom-left
        x2, y2 = x1 + triangle_size, y1  # Bottom-right
        x3, y3 = (x1 + x2) // 2, y1 - triangle_size  # Top
        
        # Create a mask for the triangle
        y, x = np.mgrid[:height, :width]
        # Barycentric coordinates check if a point is inside a triangle
        v0 = np.array([x3 - x1, y3 - y1])
        v1 = np.array([x2 - x1, y2 - y1])
        a = (x - x1) * v0[1] - (y - y1) * v0[0]
        b = (y - y1) * v1[0] - (x - x1) * v1[1]
        c = v0[0] * v1[1] - v0[1] * v1[0]
        mask = (a * b >= 0) & (a * c >= 0) & (b * c >= 0)
        image[mask] = [0, 255, 0]  # Green
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        return pil_image
    
    def test_pipeline(self):
        """Test the model using the pipeline API."""
        try:
            if not HAS_TRANSFORMERS:
                logger.warning("Transformers library not available, skipping pipeline test")
                return {"success": False, "error": "Transformers library not available"}
                
            logger.info(f"Testing LLaVA-Next model {self.model_id} with pipeline API on {self.device}")
            
            # Create a test image
            test_image = self._create_test_image()
            
            # Record start time
            start_time = time.time()
            
            # Initialize the pipeline
            pipe = transformers.pipeline(
                "visual-question-answering",
                model=self.model_id,
                device=self.device if self.device != "cpu" else -1
            )
            
            # Record model loading time
            load_time = time.time() - start_time
            logger.info(f"Model loading time: {load_time:.2f} seconds")
            
            # Record inference start time
            inference_start = time.time()
            
            # Run inference with a question
            outputs = pipe(image=test_image, question="What shapes do you see in this image?")
            
            # Record inference time
            inference_time = time.time() - inference_start
            
            # Store performance stats
            self.performance_stats["pipeline"] = {
                "load_time": load_time,
                "inference_time": inference_time
            }
            
            return {
                "success": True,
                "model_id": self.model_id,
                "device": self.device,
                "inference_time": inference_time,
                "output": outputs if outputs else None
            }
        except Exception as e:
            logger.error(f"Error testing pipeline: {e}")
            return {"success": False, "error": str(e)}
    
    def test_direct_model_inference(self):
        """Test the model with direct model and processor usage."""
        try:
            if not HAS_TRANSFORMERS or not HAS_TORCH:
                logger.warning("Required libraries not available, skipping direct model inference test")
                return {"success": False, "error": "Required libraries not available"}
                
            logger.info(f"Testing LLaVA-Next model {self.model_id} with direct model inference on {self.device}")
            
            # Record start time
            start_time = time.time()
            
            # Load processor and model
            processor = transformers.LlavaNextProcessor.from_pretrained(self.model_id)
            model = transformers.LlavaNextForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map=self.device
            )
            
            # Record model loading time
            load_time = time.time() - start_time
            logger.info(f"Model loading time: {load_time:.2f} seconds")
            
            # Create a test image
            test_image = self._create_test_image()
            
            # Test questions for visual QA
            test_questions = [
                "What shapes can you see in the image?",
                "What colors are in the image?",
                "Describe this image in detail.",
                "Is there a blue shape in the image?",
                "Where is the red shape located?"
            ]
            
            results = {}
            
            # Record inference start time
            inference_start = time.time()
            
            # Process and generate answers for each question
            for i, question in enumerate(test_questions):
                # Process inputs
                inputs = processor(
                    text=question,
                    images=test_image,
                    return_tensors="pt"
                ).to(self.device)
                
                # Generate answer
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False
                    )
                
                # Decode the generated text
                answer = processor.decode(outputs[0], skip_special_tokens=True)
                
                # Store result
                results[f"question_{i+1}"] = {
                    "question": question,
                    "answer": answer
                }
            
            # Record inference time
            inference_time = time.time() - inference_start
            
            # Store performance stats
            self.performance_stats["direct_model"] = {
                "load_time": load_time,
                "inference_time": inference_time
            }
            
            return {
                "success": True,
                "model_id": self.model_id,
                "device": self.device,
                "inference_time": inference_time,
                "results": results
            }
        except Exception as e:
            logger.error(f"Error in direct model inference test: {e}")
            return {"success": False, "error": str(e)}
    
    def test_conversation_capabilities(self):
        """Test the model's conversation capabilities."""
        try:
            if not HAS_TRANSFORMERS or not HAS_TORCH:
                logger.warning("Required libraries not available, skipping conversation test")
                return {"success": False, "error": "Required libraries not available"}
                
            logger.info(f"Testing LLaVA-Next model {self.model_id} conversation capabilities on {self.device}")
            
            # Record start time
            start_time = time.time()
            
            # Load processor and model
            processor = transformers.LlavaNextProcessor.from_pretrained(self.model_id)
            model = transformers.LlavaNextForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map=self.device
            )
            
            # Record model loading time
            load_time = time.time() - start_time
            logger.info(f"Model loading time: {load_time:.2f} seconds")
            
            # Create a test image
            test_image = self._create_test_image()
            
            # Define a multi-turn conversation
            conversation = [
                "What shapes do you see in this image?",
                "What color is the circle?",
                "Are there any shapes positioned at the top of the image?",
                "What would you say is the predominant color in this image?",
                "Could you tell me the spatial relationship between the shapes?"
            ]
            
            results = {}
            full_history = ""
            
            # Record inference start time
            inference_start = time.time()
            
            # Process and generate responses for each turn
            for i, query in enumerate(conversation):
                # Construct conversation history
                if i == 0:
                    # First turn - no history
                    prompt = query
                else:
                    # Add the query to the history
                    prompt = full_history + "\nHuman: " + query + "\nAssistant:"
                
                # Process inputs
                inputs = processor(
                    text=prompt,
                    images=test_image if i == 0 else None,  # Only provide image on first turn
                    return_tensors="pt"
                ).to(self.device)
                
                # Generate response
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9
                    )
                
                # Decode the generated text
                response = processor.decode(outputs[0], skip_special_tokens=True)
                
                # Extract just the new response
                if i == 0:
                    new_response = response
                    full_history = f"Human: {query}\nAssistant: {new_response}"
                else:
                    # Find where the new response starts
                    new_response = response.split("Assistant:")[-1].strip()
                    full_history = f"{full_history}\nHuman: {query}\nAssistant: {new_response}"
                
                # Store result
                results[f"turn_{i+1}"] = {
                    "query": query,
                    "response": new_response
                }
            
            # Record inference time
            inference_time = time.time() - inference_start
            
            # Store performance stats
            self.performance_stats["conversation"] = {
                "load_time": load_time,
                "inference_time": inference_time
            }
            
            return {
                "success": True,
                "model_id": self.model_id,
                "device": self.device,
                "inference_time": inference_time,
                "results": results,
                "full_conversation": full_history
            }
        except Exception as e:
            logger.error(f"Error in conversation test: {e}")
            return {"success": False, "error": str(e)}
    
    def test_hardware_compatibility(self):
        """Test model compatibility with different hardware acceleration options."""
        results = {}
        
        # Test CPU
        logger.info("Testing LLaVA-Next model on CPU")
        cpu_tester = TestLlavaNextModels(model_id=self.model_id, device="cpu")
        cpu_result = cpu_tester.test_pipeline()
        results["cpu"] = {
            "success": cpu_result.get("success", False),
            "inference_time": cpu_result.get("inference_time", None)
        }
        
        # Test CUDA if available
        if HAS_TORCH and torch.cuda.is_available():
            logger.info("Testing LLaVA-Next model on CUDA")
            cuda_tester = TestLlavaNextModels(model_id=self.model_id, device="cuda:0")
            cuda_result = cuda_tester.test_pipeline()
            results["cuda"] = {
                "success": cuda_result.get("success", False),
                "inference_time": cuda_result.get("inference_time", None)
            }
        
        # Test MPS if available (Apple Silicon)
        if HAS_TORCH and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Testing LLaVA-Next model on MPS (Apple Silicon)")
            mps_tester = TestLlavaNextModels(model_id=self.model_id, device="mps")
            mps_result = mps_tester.test_pipeline()
            results["mps"] = {
                "success": mps_result.get("success", False),
                "inference_time": mps_result.get("inference_time", None)
            }
        
        # Add OpenVINO test stub for future implementation
        results["openvino"] = {
            "success": False,
            "error": "OpenVINO support not yet implemented for LLaVA-Next models"
        }
        
        return results
    
    def run_tests(self, all_hardware=False):
        """Run all tests for this model."""
        results = {}
        
        # Run pipeline test
        pipeline_result = self.test_pipeline()
        results["pipeline"] = pipeline_result
        
        # Run additional tests if pipeline was successful
        if pipeline_result.get("success", False):
            # Direct model inference test
            direct_model_result = self.test_direct_model_inference()
            results["direct_model_inference"] = direct_model_result
            
            # Conversation capabilities test
            conversation_result = self.test_conversation_capabilities()
            results["conversation_capabilities"] = conversation_result
        
        # Test hardware compatibility if requested
        if all_hardware:
            hardware_results = self.test_hardware_compatibility()
            results["hardware_compatibility"] = hardware_results
        
        # Add metadata
        results["metadata"] = {
            "model_id": self.model_id,
            "device": self.device,
            "has_transformers": HAS_TRANSFORMERS,
            "has_torch": HAS_TORCH,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_info": self.model_info
        }
        
        return results

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Test LLaVA-Next HuggingFace models")
    parser.add_argument("--model", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf", help="Model ID to test")
    parser.add_argument("--device", type=str, help="Device to run tests on (cuda, cpu, mps)")
    parser.add_argument("--all-hardware", action="store_true", help="Test on all available hardware")
    
    args = parser.parse_args()
    
    # Initialize the test class
    llava_next_tester = TestLlavaNextModels(model_id=args.model, device=args.device)
    
    # Run the tests
    results = llava_next_tester.run_tests(all_hardware=args.all_hardware)
    
    # Print a summary
    success = results["pipeline"].get("success", False)
    
    print("\nTEST RESULTS SUMMARY:")
    
    if success:
        print(f"  Successfully tested {args.model}")
        print(f"  - Device: {llava_next_tester.device}")
        print(f"  - Inference time: {results['pipeline'].get('inference_time', 'N/A'):.4f}s")
        if "output" in results["pipeline"]:
            print(f"  - Generated output: {results['pipeline']['output']}")
    else:
        print(f"  Failed to test {args.model}")
        print(f"  - Error: {results['pipeline'].get('error', 'Unknown error')}")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())