import os
import subprocess
import numpy as np
from pathlib import Path
import importlib.util
import sys
import warnings

class SNPEUtils:
    """Utility class for working with Qualcomm's Snapdragon Neural Processing Engine (SNPE)"""
    
    def __init__(self):
        self._snpe_available = None
        self._snpe_lib = None
        self._qnn_lib = None
        
    def is_available(self):
        """Check if SNPE is available on the system"""
        if self._snpe_available is not None:
            return self._snpe_available
            
        try:
            # Try to import SNPE libraries
            snpe_found = self._import_snpe()
            qnn_found = self._import_qnn()
            
            self._snpe_available = snpe_found or qnn_found
            return self._snpe_available
        except Exception as e:
            print(f"Error checking SNPE availability: {e}")
            self._snpe_available = False
            return False
            
    def _import_snpe(self):
        """Try to import the SNPE library"""
        try:
            # Look for SNPE in common install locations
            snpe_root = self._find_snpe_root()
            if not snpe_root:
                return False
                
            # Add SNPE paths to Python path
            snpe_lib_path = os.path.join(snpe_root, 'lib', 'python')
            if snpe_lib_path not in sys.path:
                sys.path.append(snpe_lib_path)
                
            # Try importing a key SNPE module
            spec = importlib.util.find_spec('snpe')
            if spec is not None:
                self._snpe_lib = importlib.import_module('snpe')
                return True
            return False
        except Exception:
            return False
            
    def _import_qnn(self):
        """Try to import the QNN library (newer version of SNPE)"""
        try:
            # Look for QNN in common install locations
            qnn_root = self._find_qnn_root()
            if not qnn_root:
                return False
                
            # Add QNN paths to Python path
            qnn_lib_path = os.path.join(qnn_root, 'lib', 'python')
            if qnn_lib_path not in sys.path:
                sys.path.append(qnn_lib_path)
                
            # Try importing a key QNN module
            spec = importlib.util.find_spec('qnn')
            if spec is not None:
                self._qnn_lib = importlib.import_module('qnn')
                return True
            return False
        except Exception:
            return False
    
    def _find_snpe_root(self):
        """Find the SNPE SDK root directory"""
        # Check environment variable first
        if 'SNPE_ROOT' in os.environ:
            return os.environ['SNPE_ROOT']
            
        # Check common install locations
        common_paths = [
            '/opt/qualcomm/snpe',
            os.path.expanduser('~/snpe-sdk'),
            os.path.expanduser('~/qualcomm/snpe-sdk'),
            'C:\\Qualcomm\\SNPE'
        ]
        
        for path in common_paths:
            if os.path.exists(path) and os.path.isdir(path):
                return path
                
        return None
        
    def _find_qnn_root(self):
        """Find the QNN SDK root directory"""
        # Check environment variable first
        if 'QNN_SDK_ROOT' in os.environ:
            return os.environ['QNN_SDK_ROOT']
            
        # Check common install locations
        common_paths = [
            '/opt/qualcomm/qnn',
            os.path.expanduser('~/qnn-sdk'),
            os.path.expanduser('~/qualcomm/qnn-sdk'),
            'C:\\Qualcomm\\QNN'
        ]
        
        for path in common_paths:
            if os.path.exists(path) and os.path.isdir(path):
                return path
                
        return None
    
    def convert_model(self, model_name, model_type, output_path):
        """Convert a Hugging Face model to SNPE format
        
        Args:
            model_name: Name or path of the Hugging Face model
            model_type: Type of model ('text', 'vision', 'audio', etc.)
            output_path: Path to save the converted model
            
        Returns:
            Path to the converted model file
        """
        try:
            # Ensure SNPE is available
            if not self.is_available():
                raise RuntimeError("SNPE is not available on this system")
                
            # Import required libraries
            import torch
            from transformers import AutoTokenizer, AutoModel, AutoConfig
            
            # Create directory for temporary files
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            temp_dir = os.path.join(os.path.dirname(output_path), "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Step 1: Export model to ONNX
            print(f"Loading model {model_name}...")
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            
            # Handle different model types
            if model_type in ["text", "embedding"]:
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                
                # Create example inputs
                dummy_input = tokenizer("Example text for conversion", return_tensors="pt")
                
            elif model_type == "vision":
                from transformers import AutoImageProcessor
                from PIL import Image
                import requests
                from io import BytesIO
                
                processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
                model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                
                # Get a sample image
                response = requests.get("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png")
                image = Image.open(BytesIO(response.content))
                
                # Create example inputs
                dummy_input = processor(images=image, return_tensors="pt")
                
            elif model_type == "audio":
                from transformers import AutoProcessor
                import librosa
                
                processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                
                # Create a dummy audio input (1 second of silence)
                sample_rate = 16000
                dummy_audio = np.zeros(sample_rate)
                
                # Create example inputs
                dummy_input = processor(dummy_audio, sampling_rate=sample_rate, return_tensors="pt")
                
            elif model_type == "vision_text_dual":
                from transformers import CLIPProcessor, CLIPModel
                from PIL import Image
                import requests
                from io import BytesIO
                
                processor = CLIPProcessor.from_pretrained(model_name)
                model = CLIPModel.from_pretrained(model_name)
                
                # Get a sample image
                response = requests.get("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png")
                image = Image.open(BytesIO(response.content))
                
                # Create example inputs
                dummy_input = processor(
                    text=["a photo of a cat", "a photo of a dog"],
                    images=image,
                    return_tensors="pt",
                    padding=True
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Export to ONNX
            onnx_path = os.path.join(temp_dir, "model.onnx")
            print(f"Exporting model to ONNX format at {onnx_path}...")
            
            # Set model to evaluation mode
            model.eval()
            
            # Export to ONNX with correct input names
            input_names = list(dummy_input.keys())
            
            # Define dynamic axes
            dynamic_axes = {}
            for input_name in input_names:
                dynamic_axes[input_name] = {0: 'batch_size'}
                if 'input_ids' in input_name or 'attention_mask' in input_name:
                    dynamic_axes[input_name][1] = 'sequence_length'
            
            # Define output names based on model outputs
            output_names = ["output"]
            
            torch.onnx.export(
                model,
                tuple(dummy_input.values()),
                onnx_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=12
            )
            
            print("ONNX export complete")
            
            # Step 2: Convert ONNX to SNPE
            print("Converting ONNX model to SNPE format...")
            
            # Determine which tool to use (snpe-onnx-to-dlc or qnn-onnx-converter)
            if self._snpe_lib:
                tool_path = os.path.join(self._find_snpe_root(), 'bin', 'snpe-onnx-to-dlc')
                if os.name == 'nt':  # Windows
                    tool_path += '.exe'
            else:  # Using QNN
                tool_path = os.path.join(self._find_qnn_root(), 'bin', 'qnn-onnx-converter')
                if os.name == 'nt':  # Windows
                    tool_path += '.exe'
            
            # Run the conversion tool
            command = [
                tool_path,
                '-i', onnx_path,
                '-o', output_path
            ]
            
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"SNPE conversion failed: {result.stderr}")
                
            print(f"Successfully converted model to SNPE format at {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error converting model to SNPE: {e}")
            raise
    
    def load_model(self, model_path):
        """Load a model in SNPE format
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded model object
        """
        try:
            # Ensure SNPE is available
            if not self.is_available():
                raise RuntimeError("SNPE is not available on this system")
                
            if self._snpe_lib:
                # Use SNPE
                from snpe.dlc import DLContainer
                from snpe.runtime import Runtime
                
                # Load the DLC
                dlc = DLContainer(model_path)
                
                # Create a runtime for the DLC
                runtime = Runtime(dlc)
                return runtime
            elif self._qnn_lib:
                # Use QNN
                from qnn.runtime import Runtime
                
                # Create a runtime for the model
                runtime = Runtime(model_path)
                return runtime
            else:
                raise RuntimeError("Neither SNPE nor QNN libraries are available")
                
        except Exception as e:
            print(f"Error loading SNPE model: {e}")
            raise
    
    def optimize_for_device(self, model_path, device_type):
        """Optimize the model for a specific Qualcomm device
        
        Args:
            model_path: Path to the model file
            device_type: Type of device (e.g., 'adreno', 'hexagon', 'hta')
            
        Returns:
            Path to the optimized model
        """
        try:
            # Ensure SNPE is available
            if not self.is_available():
                raise RuntimeError("SNPE is not available on this system")
                
            # Create output path for optimized model
            base_path = os.path.splitext(model_path)[0]
            optimized_path = f"{base_path}_{device_type}.dlc"
            
            if os.path.exists(optimized_path):
                print(f"Optimized model for {device_type} already exists at {optimized_path}")
                return optimized_path
            
            if self._snpe_lib:
                # Use SNPE tools
                if device_type == 'adreno':
                    tool_path = os.path.join(self._find_snpe_root(), 'bin', 'snpe-dlc-graph-prepare')
                    if os.name == 'nt':  # Windows
                        tool_path += '.exe'
                    
                    # Run the optimization tool
                    command = [
                        tool_path,
                        '--input_dlc', model_path,
                        '--output_dlc', optimized_path,
                        '--target', 'gpu'
                    ]
                    
                elif device_type == 'hexagon':
                    tool_path = os.path.join(self._find_snpe_root(), 'bin', 'snpe-dlc-graph-prepare')
                    if os.name == 'nt':  # Windows
                        tool_path += '.exe'
                    
                    # Run the optimization tool
                    command = [
                        tool_path,
                        '--input_dlc', model_path,
                        '--output_dlc', optimized_path,
                        '--target', 'dsp'
                    ]
                    
                elif device_type == 'hta':
                    tool_path = os.path.join(self._find_snpe_root(), 'bin', 'snpe-dlc-graph-prepare')
                    if os.name == 'nt':  # Windows
                        tool_path += '.exe'
                    
                    # Run the optimization tool
                    command = [
                        tool_path,
                        '--input_dlc', model_path,
                        '--output_dlc', optimized_path,
                        '--target', 'hta'
                    ]
                    
                else:
                    warnings.warn(f"Unknown device type: {device_type}. Using original model.")
                    return model_path
                    
            elif self._qnn_lib:
                # Use QNN tools
                tool_path = os.path.join(self._find_qnn_root(), 'bin', 'qnn-model-optimizer')
                if os.name == 'nt':  # Windows
                    tool_path += '.exe'
                    
                # Map device type to QNN target
                target_map = {
                    'adreno': 'gpu',
                    'hexagon': 'dsp',
                    'hta': 'hta'
                }
                
                if device_type not in target_map:
                    warnings.warn(f"Unknown device type: {device_type}. Using original model.")
                    return model_path
                    
                # Run the optimization tool
                command = [
                    tool_path,
                    '--input_model', model_path,
                    '--output_model', optimized_path,
                    '--target', target_map[device_type]
                ]
                
            else:
                raise RuntimeError("Neither SNPE nor QNN libraries are available")
                
            # Execute the command
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode != 0:
                warnings.warn(f"Model optimization failed: {result.stderr}")
                return model_path
                
            print(f"Successfully optimized model for {device_type} at {optimized_path}")
            return optimized_path
            
        except Exception as e:
            print(f"Error optimizing model for {device_type}: {e}")
            return model_path
    
    def run_inference(self, model, inputs):
        """Run inference with the model
        
        Args:
            model: Loaded model object
            inputs: Dictionary of input tensors
            
        Returns:
            Dictionary of output tensors
        """
        try:
            # Ensure SNPE is available
            if not self.is_available():
                raise RuntimeError("SNPE is not available on this system")
                
            # Convert inputs to numpy arrays if they're torch tensors
            for key in inputs:
                if hasattr(inputs[key], 'numpy'):
                    inputs[key] = inputs[key].numpy()
            
            if self._snpe_lib:
                # Use SNPE runtime
                outputs = model.execute(inputs)
                return outputs
                
            elif self._qnn_lib:
                # Use QNN runtime
                outputs = model.forward(inputs)
                return outputs
                
            else:
                raise RuntimeError("Neither SNPE nor QNN libraries are available")
                
        except Exception as e:
            print(f"Error running inference with SNPE: {e}")
            raise

def get_snpe_utils():
    """Factory function to create an SNPEUtils instance"""
    return SNPEUtils()