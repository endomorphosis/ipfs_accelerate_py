#!/usr/bin/env python3
"""
Export a model for web deployment using WebNN and WebGPU backends.
This script handles the export process for both ONNX (WebNN) and transformers.js (WebGPU) formats.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, 
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Export a model for web deployment")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--format", type=str, choices=["webnn", "webgpu", "both"], default="both",
                      help="Export format(s)")
    parser.add_argument("--output-dir", type=str, default="./web_models",
                      help="Output directory for exported models")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "int8"], default="fp16",
                      help="Precision for WebNN export")
    parser.add_argument("--quantized", action="store_true", help="Enable quantization for WebGPU export")
    parser.add_argument("--optimize", action="store_true", help="Apply optimization techniques")
    parser.add_argument("--validate", action="store_true", help="Validate exported models")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()

def setup_environment(args):
    """Set up the environment based on arguments"""
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Create output directories
    webnn_dir = os.path.join(args.output_dir, "webnn", args.model.split("/")[-1])
    webgpu_dir = os.path.join(args.output_dir, "webgpu", args.model.split("/")[-1])
    
    for directory in [webnn_dir, webgpu_dir]:
        os.makedirs(directory, exist_ok=True)
    
    return webnn_dir, webgpu_dir

def export_model_for_webnn(model_name: str, output_dir: str, 
                         precision: str = "fp16", 
                         optimize: bool = True,
                         validate: bool = False) -> Tuple[bool, str]:
    """
    Export a model to ONNX format for WebNN deployment
    
    Args:
        model_name: Model name or path
        output_dir: Output directory for ONNX model
        precision: Model precision (fp32, fp16, int8)
        optimize: Apply optimizations
        validate: Validate exported model
    
    Returns:
        Tuple of (success, message)
    """
    try:
        logger.info(f"Exporting {model_name} to ONNX for WebNN...")
        
        # Import required libraries
        import torch
        from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
        import onnx
        from onnxruntime.quantization import quantize_dynamic
        
        # Determine model type from name
        if "bert" in model_name.lower():
            model_type = "bert"
        elif "t5" in model_name.lower():
            model_type = "t5"
        elif "gpt" in model_name.lower():
            model_type = "gpt"
        elif "vit" in model_name.lower() or "resnet" in model_name.lower():
            model_type = "vision"
        else:
            model_type = "default"
        
        # Load appropriate model based on type
        try:
            if model_type == "vision":
                from transformers import AutoImageProcessor, AutoModelForImageClassification
                model = AutoModelForImageClassification.from_pretrained(model_name)
                tokenizer = AutoImageProcessor.from_pretrained(model_name)
            elif model_type == "bert":
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            else:
                model = AutoModel.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            logger.warning(f"Error loading specific model type, falling back to generic: {e}")
            model = AutoModel.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set model to evaluation mode
        model.eval()
        
        # Create ONNX export path
        onnx_path = os.path.join(output_dir, "model.onnx")
        
        # Create sample inputs based on model type
        if model_type == "vision":
            # Vision model sample input
            dummy_input = torch.randn(1, 3, 224, 224)
            input_names = ["pixel_values"]
            output_names = ["logits"]
            dynamic_axes = {
                "pixel_values": {0: "batch_size"},
                "logits": {0: "batch_size"}
            }
        else:
            # Text model sample input
            input_ids = torch.zeros(1, 128, dtype=torch.long)
            attention_mask = torch.zeros(1, 128, dtype=torch.long)
            token_type_ids = torch.zeros(1, 128, dtype=torch.long)
            
            # Check if model uses token_type_ids
            signature = model.forward.__code__.co_varnames
            if "token_type_ids" in signature:
                dummy_input = (input_ids, attention_mask, token_type_ids)
                input_names = ["input_ids", "attention_mask", "token_type_ids"]
            else:
                dummy_input = (input_ids, attention_mask)
                input_names = ["input_ids", "attention_mask"]
            
            output_names = ["logits"]
            dynamic_axes = {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size"}
            }
            if "token_type_ids" in input_names:
                dynamic_axes["token_type_ids"] = {0: "batch_size", 1: "sequence_length"}
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            opset_version=14,
            verbose=False
        )
        
        logger.info(f"Model exported to {onnx_path}")
        
        # Apply optimization based on precision
        if precision == "fp16":
            # Convert to fp16
            from onnxruntime.transformers.float16 import convert_float_to_float16
            logger.info("Converting model to fp16...")
            model_fp16 = onnx.load(onnx_path)
            model_fp16 = convert_float_to_float16(model_fp16)
            onnx.save(model_fp16, onnx_path)
            logger.info("Model converted to fp16")
        elif precision == "int8":
            # Quantize to int8
            logger.info("Quantizing model to int8...")
            int8_path = os.path.join(output_dir, "model_int8.onnx")
            quantize_dynamic(onnx_path, int8_path)
            os.replace(int8_path, onnx_path)
            logger.info("Model quantized to int8")
        
        # Apply optimization if requested
        if optimize:
            try:
                import onnxruntime as ort
                logger.info("Optimizing ONNX model...")
                from onnxruntime.transformers.optimizer import optimize_model
                optimized_path = os.path.join(output_dir, "model_optimized.onnx")
                
                # Optimize using ONNX Runtime
                optimize_model(
                    onnx_path,
                    model_type=model_type,
                    output_path=optimized_path,
                    optimization_options=None
                )
                
                # Replace original with optimized model
                os.replace(optimized_path, onnx_path)
                logger.info("Model optimized")
            except Exception as e:
                logger.warning(f"Optimization failed (continuing anyway): {e}")
        
        # Save tokenizer/processor information
        try:
            # Save vocabulary or processor configuration
            if hasattr(tokenizer, "vocab"):
                with open(os.path.join(output_dir, "vocab.json"), "w") as f:
                    json.dump(tokenizer.vocab, f)
            
            # Save tokenizer configuration
            if hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(output_dir)
            
            logger.info("Tokenizer/processor information saved")
        except Exception as e:
            logger.warning(f"Error saving tokenizer information: {e}")
        
        # Validate exported model if requested
        if validate:
            try:
                logger.info("Validating exported model...")
                import onnxruntime as ort
                
                # Create inference session
                session = ort.InferenceSession(onnx_path)
                
                # Prepare inputs based on model type
                if model_type == "vision":
                    # Vision model validation
                    ort_inputs = {
                        "pixel_values": dummy_input.numpy()
                    }
                else:
                    # Text model validation
                    if "token_type_ids" in input_names:
                        ort_inputs = {
                            "input_ids": input_ids.numpy(),
                            "attention_mask": attention_mask.numpy(),
                            "token_type_ids": token_type_ids.numpy()
                        }
                    else:
                        ort_inputs = {
                            "input_ids": input_ids.numpy(),
                            "attention_mask": attention_mask.numpy()
                        }
                
                # Run inference
                ort_outputs = session.run(None, ort_inputs)
                logger.info("Validation successful: Model runs correctly")
            except Exception as e:
                logger.warning(f"Validation error (continuing anyway): {e}")
        
        return True, f"Model exported successfully to {onnx_path}"
    
    except Exception as e:
        logger.error(f"Error exporting model to ONNX: {e}")
        return False, f"Error: {str(e)}"

def export_model_for_webgpu(model_name: str, output_dir: str, 
                          quantized: bool = False,
                          optimize: bool = True,
                          validate: bool = False) -> Tuple[bool, str]:
    """
    Export a model for WebGPU/transformers.js deployment
    
    Args:
        model_name: Model name or path
        output_dir: Output directory for WebGPU model
        quantized: Enable quantization
        optimize: Apply optimizations
        validate: Validate exported model
    
    Returns:
        Tuple of (success, message)
    """
    try:
        logger.info(f"Exporting {model_name} for WebGPU/transformers.js...")
        
        # Determine model type from name
        if "bert" in model_name.lower():
            model_type = "bert"
            task = "text-classification"
        elif "t5" in model_name.lower():
            model_type = "t5"
            task = "text2text-generation"
        elif "gpt" in model_name.lower():
            model_type = "gpt"
            task = "text-generation"
        elif "vit" in model_name.lower():
            model_type = "vit"
            task = "image-classification"
        elif "resnet" in model_name.lower():
            model_type = "resnet"
            task = "image-classification"
        else:
            model_type = "default"
            task = "feature-extraction"
        
        # Check for appropriate npm modules
        try:
            import subprocess
            logger.info("Checking for @xenova/transformers npm package...")
            
            # Check if @xenova/transformers is installed
            result = subprocess.run(
                ["npm", "list", "@xenova/transformers"],
                capture_output=True,
                text=True
            )
            
            if "@xenova/transformers" not in result.stdout:
                logger.warning("@xenova/transformers not found in npm packages")
                logger.info("Installing @xenova/transformers...")
                
                # Install the package
                install_result = subprocess.run(
                    ["npm", "install", "@xenova/transformers"],
                    capture_output=True,
                    text=True
                )
                
                if install_result.returncode != 0:
                    logger.warning(f"Failed to install @xenova/transformers: {install_result.stderr}")
        except Exception as e:
            logger.warning(f"Error checking npm packages: {e}")
        
        # Create model configuration file
        config = {
            "name": model_name,
            "model_type": model_type,
            "task": task,
            "quantized": quantized,
            "optimized": optimize
        }
        
        # Save configuration
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        # Create README with usage instructions
        readme = f"""# {model_name} for WebGPU/transformers.js

## Usage

```javascript
import {{ pipeline }} from '@xenova/transformers';

// Load the model
const model = await pipeline('{task}', '{model_name}');

// Run inference
const result = await model('your input here');
console.log(result);
```

## Configuration

- Model Type: {model_type}
- Task: {task}
- Quantized: {quantized}
- Optimized: {optimize}

## Notes

This model was exported for WebGPU/transformers.js using IPFS Accelerate.
"""
        
        # Save README
        with open(os.path.join(output_dir, "README.md"), "w") as f:
            f.write(readme)
        
        # Export HuggingFace token (if available) to avoid rate limiting
        if "HF_TOKEN" in os.environ:
            with open(os.path.join(output_dir, ".env"), "w") as f:
                f.write(f"HF_TOKEN={os.environ['HF_TOKEN']}\n")
        
        # Create HTML demo file
        html_demo = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{model_name} Demo</title>
    <script src="https://cdn.jsdelivr.net/npm/@xenova/transformers@latest"></script>
</head>
<body>
    <h1>{model_name} Demo</h1>
    
    <div>
        <textarea id="input" rows="4" cols="50" placeholder="Enter text..."></textarea>
        <button id="run">Run Inference</button>
    </div>
    
    <div>
        <h3>Results:</h3>
        <pre id="output"></pre>
    </div>
    
    <script>
        // Import from transformers.js
        const {{ pipeline, env }} = window.transformers;
        
        // Enable WebGPU
        env.useBrowserCache = true;
        env.useWebGPU = true;
        
        // Initialize the model
        let model;
        
        async function loadModel() {{
            try {{
                model = await pipeline('{task}', '{model_name}');
                console.log('Model loaded successfully');
            }} catch (error) {{
                console.error('Error loading model:', error);
                document.getElementById('output').textContent = `Error: ${{error.message}}`;
            }}
        }}
        
        // Run inference
        async function runInference() {{
            if (!model) {{
                await loadModel();
            }}
            
            const input = document.getElementById('input').value;
            if (!input) {{
                alert('Please enter input text');
                return;
            }}
            
            try {{
                document.getElementById('output').textContent = 'Running...';
                const result = await model(input);
                document.getElementById('output').textContent = JSON.stringify(result, null, 2);
            }} catch (error) {{
                console.error('Inference error:', error);
                document.getElementById('output').textContent = `Error: ${{error.message}}`;
            }}
        }}
        
        // Set up event listener
        document.getElementById('run').addEventListener('click', runInference);
        
        // Load model automatically
        loadModel();
    </script>
</body>
</html>
"""
        
        # Save HTML demo
        with open(os.path.join(output_dir, "demo.html"), "w") as f:
            f.write(html_demo)
        
        # Create js wrapper module
        js_wrapper = f"""// {model_name} WebGPU wrapper
import {{ pipeline, env }} from '@xenova/transformers';

// Enable WebGPU
env.useBrowserCache = true;
env.useWebGPU = true;

// Model configuration
const config = {{
    task: '{task}',
    model: '{model_name}',
    quantized: {str(quantized).lower()},
    revision: 'main'
}};

// Create a class for easier use
class Model {{
    constructor() {{
        this.model = null;
        this.loading = null;
    }}
    
    async load() {{
        if (this.model !== null) {{
            return this.model;
        }}
        
        if (this.loading !== null) {{
            return await this.loading;
        }}
        
        this.loading = pipeline(config.task, config.model, {{
            quantized: config.quantized,
            revision: config.revision
        }});
        
        this.model = await this.loading;
        this.loading = null;
        return this.model;
    }}
    
    async process(input) {{
        const model = await this.load();
        return await model(input);
    }}
}}

export default new Model();
"""
        
        # Save js wrapper
        with open(os.path.join(output_dir, "model.js"), "w") as f:
            f.write(js_wrapper)
        
        logger.info(f"Model exported for WebGPU/transformers.js to {output_dir}")
        return True, f"Model exported successfully to {output_dir}"
    
    except Exception as e:
        logger.error(f"Error exporting model for WebGPU: {e}")
        return False, f"Error: {str(e)}"

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Set up environment
    webnn_dir, webgpu_dir = setup_environment(args)
    
    success_messages = []
    error_messages = []
    
    # Export based on format
    if args.format in ["webnn", "both"]:
        # Export for WebNN
        success, message = export_model_for_webnn(
            model_name=args.model,
            output_dir=webnn_dir,
            precision=args.precision,
            optimize=args.optimize,
            validate=args.validate
        )
        
        if success:
            success_messages.append(f"WebNN: {message}")
        else:
            error_messages.append(f"WebNN: {message}")
    
    if args.format in ["webgpu", "both"]:
        # Export for WebGPU
        success, message = export_model_for_webgpu(
            model_name=args.model,
            output_dir=webgpu_dir,
            quantized=args.quantized,
            optimize=args.optimize,
            validate=args.validate
        )
        
        if success:
            success_messages.append(f"WebGPU: {message}")
        else:
            error_messages.append(f"WebGPU: {message}")
    
    # Print summary
    if success_messages:
        logger.info("Export completed successfully:")
        for message in success_messages:
            logger.info(f"- {message}")
    
    if error_messages:
        logger.error("Errors during export:")
        for message in error_messages:
            logger.error(f"- {message}")
    
    # Check if there were any errors
    return 0 if not error_messages else 1

if __name__ == "__main__":
    sys.exit(main())