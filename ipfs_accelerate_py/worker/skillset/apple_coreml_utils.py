import os
import numpy as np
from typing import Dict, Any, Optional
import warnings

class CoreMLUtils:
    def __init__(self):
        self._ct = None
        self._model = None
        self._torch = None
        
    def is_available(self) -> bool:
        """Check if CoreML conversion is available."""
        try:
            import coremltools as ct
            import torch
            self._ct = ct
            self._torch = torch
            return True
        except ImportError:
            return False
            
    def _get_device_type(self) -> str:
        """Determine the best available device type."""
        try:
            if hasattr(self._torch.backends, 'mps') and self._torch.backends.mps.is_available():
                return "mps"  # Apple Silicon GPU
            return "cpu"  # Fallback to CPU
        except:
            return "cpu"
            
    def optimize_for_device(self, model_path: str, compute_units: str = "ALL") -> str:
        """Optimize the CoreML model for specific Apple hardware.
        
        Args:
            model_path: Path to the CoreML model
            compute_units: One of: ALL, CPU_ONLY, CPU_AND_GPU, CPU_AND_NE
        """
        try:
            if not self._ct:
                return model_path
                
            optimized_path = model_path.replace(".mlpackage", f"_{compute_units.lower()}.mlpackage")
            if os.path.exists(optimized_path):
                return optimized_path
                
            # Load the model
            model = self._ct.models.MLModel(model_path)
            
            # Set compute units
            model.compute_unit = getattr(self._ct.ComputeUnit, compute_units)
            
            # Save optimized model
            model.save(optimized_path)
            return optimized_path
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            return model_path

    def convert_model(self, model_name: str, task_type: str, output_path: str) -> bool:
        """Convert HuggingFace model to CoreML format.
        
        Args:
            model_name: Name of the HuggingFace model
            task_type: Type of the model (text, audio, vision)
            output_path: Where to save the converted model
        """
        try:
            if not self._ct:
                return False
                
            import transformers
            
            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Load model based on task type
            if task_type == "text":
                model = transformers.AutoModel.from_pretrained(model_name)
                tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
                
                # Example inputs
                inputs = tokenizer("Example text", return_tensors="pt")
                input_names = list(inputs.keys())
                
                # Trace model
                traced_model = self._torch.jit.trace(model, tuple(inputs.values()))
                
            elif task_type == "vision":
                model = transformers.AutoModelForImageClassification.from_pretrained(model_name)
                processor = transformers.AutoImageProcessor.from_pretrained(model_name)
                
                # Example image (1x3x224x224)
                dummy_input = self._torch.randn(1, 3, 224, 224)
                
                # Trace model
                traced_model = self._torch.jit.trace(model, dummy_input)
                
            elif task_type == "audio":
                model = transformers.AutoModelForAudioClassification.from_pretrained(model_name)
                processor = transformers.AutoFeatureExtractor.from_pretrained(model_name)
                
                # Example audio input
                dummy_input = self._torch.randn(1, 16000)  # 1 second of audio at 16kHz
                
                # Trace model
                traced_model = self._torch.jit.trace(model, dummy_input)
                
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
                
            # Convert to CoreML
            mlmodel = self._ct.convert(
                traced_model,
                convert_to="mlprogram",
                minimum_deployment_target=self._ct.target.macOS13,
                compute_units=self._ct.ComputeUnit.ALL,
                compute_precision=self._ct.precision.FLOAT16  # Use FP16 for better performance
            )
            
            # Save the model
            mlmodel.save(output_path)
            return True
            
        except Exception as e:
            print(f"Model conversion failed: {e}")
            return False
            
    def load_model(self, model_path: str):
        """Load a CoreML model for inference."""
        try:
            if not self._ct:
                return None
                
            model = self._ct.models.MLModel(model_path)
            self._model = model
            return model
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            return None
            
    def run_inference(self, model: Any, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run inference using CoreML runtime."""
        try:
            if not self._ct or not model:
                return {}
                
            # Prepare inputs - convert to CoreML compatible format
            input_dict = {}
            for key, value in inputs.items():
                if hasattr(value, "numpy"):
                    input_dict[key] = value.numpy()
                else:
                    input_dict[key] = value
                    
            # Run prediction
            results = model.predict(input_dict)
            
            # Convert outputs back to numpy arrays
            output_dict = {}
            for key, value in results.items():
                if hasattr(value, "numpy"):
                    output_dict[key] = value.numpy()
                else:
                    output_dict[key] = np.array(value)
                    
            return output_dict
            
        except Exception as e:
            print(f"Inference failed: {e}")
            return {}

_instance = None

def get_coreml_utils() -> CoreMLUtils:
    """Get singleton instance of CoreMLUtils."""
    global _instance
    if _instance is None:
        _instance = CoreMLUtils()
    return _instance