import os
import sys
import json
import anyio
import traceback
from datetime import datetime

# Add the parent directory to sys.path for proper imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variable to avoid tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
,
# Import the needed modules
try:
    from ipfs_accelerate_py.ipfs_accelerate import ipfs_accelerate_py
except ImportError as e:
    print(f"Error importing ipfs_accelerate_py: {}}}}}}}}}}e}")
    sys.exit(1)

class TestLocalEndpoints:
    def __init__(self):
        self.resources = {}}}}}}}}}}
        "local_endpoints": {}}}}}}}}}}},
        "tokenizer": {}}}}}}}}}}},
        "endpoint_handler": {}}}}}}}}}}},
        "queue": {}}}}}}}}}}},
        "queues": {}}}}}}}}}}},
        "batch_sizes": {}}}}}}}}}}},
        "consumer_tasks": {}}}}}}}}}}},
        "caches": {}}}}}}}}}}}
        }
        self.metadata = {}}}}}}}}}}"models": []}
        ,
        # Import transformers
        try:
            import transformers
            self.resources["transformers"] = transformers,
            print("Successfully imported transformers module")
        except ImportError:
            from unittest.mock import MagicMock
            self.resources["transformers"] = MagicMock(),
            print("Using MagicMock for transformers")
        
        # Import torch
        try:
            import torch
            self.resources["torch"] = torch,
            print("Successfully imported torch module")
        except ImportError:
            from unittest.mock import MagicMock
            self.resources["torch"] = MagicMock(),
            print("Using MagicMock for torch")
        
        # Load mapped models
        try:
            with open('mapped_models.json', 'r') as f:
                self.mapped_models = json.load(f)
                print(f"Loaded {}}}}}}}}}}len(self.mapped_models)} models from mapped_models.json")
        except Exception as e:
            print(f"Error loading mapped_models.json: {}}}}}}}}}}e}")
            self.mapped_models = {}}}}}}}}}}}
        
        # Initialize results
            self.results = {}}}}}}}}}}
            "metadata": {}}}}}}}}}}
            "timestamp": datetime.now().isoformat(),
            "test_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "model_results": {}}}}}}}}}}}
            }
    
    async def test_endpoint(self, skill_name, model_name):
        print(f"\nTesting {}}}}}}}}}}skill_name} with model {}}}}}}}}}}model_name}...")
        result = {}}}}}}}}}}
        "status": "Not tested",
        "model": model_name,
        "skill": skill_name,
        "endpoint_type": "cpu:0",
        "error": None,
        "input": None,
        "output": None,
        "implementation_type": "Unknown"
        }
        
        # Create appropriate test input based on skill
        if skill_name in ["bert", "distilbert", "roberta", "mpnet", "albert"]:,
        result["input"] = "This is a test sentence for embedding models.",
        skill_handler = "default_embed"
        elif skill_name in ["gpt_neo", "gptj", "gpt2", "opt", "bloom", "codegen", "llama"]:,
            result["input"] = "Once upon a time",
            skill_handler = "default_lm"
        elif skill_name == "whisper":
            result["input"] = "test.mp3",,,
            skill_handler = "hf_whisper"
        elif skill_name == "wav2vec2":
            result["input"] = "test.mp3",,,
            skill_handler = "hf_wav2vec2"
        elif skill_name == "clip":
            result["input"] = "test.jpg",
            skill_handler = "hf_clip"
        elif skill_name == "xclip":
            result["input"] = "test.mp4",
            skill_handler = "hf_xclip"
        elif skill_name == "clap":
            result["input"] = "test.mp3",,,
            skill_handler = "hf_clap"
        elif skill_name == "t5":
            result["input"] = "translate English to German: Hello, how are you?",
            skill_handler = "hf_t5"
        elif skill_name in ["llava", "llava_next", "qwen2_vl"]:,
            result["input"] = {}}}}}}}}}}"image": "test.jpg", "text": "What is in this image?"},
            skill_handler = "hf_llava" if skill_name == "llava" else "hf_llava_next":
        else:
            result["input"] = "Generic test input for model.",
            skill_handler = "default_lm"
        
        try:
            # Initialize the accelerator
            accelerator = ipfs_accelerate_py(self.resources, self.metadata)
            
            # Add the endpoint
            endpoint_added = False
            try:
                print(f"  Adding endpoint for {}}}}}}}}}}model_name} ({}}}}}}}}}}skill_handler})...")
                endpoint_key = f"{}}}}}}}}}}skill_handler}/{}}}}}}}}}}model_name}/cpu:0"
                # Based on ipfs_accelerate.py, add_endpoint takes 3 params: model, endpoint_type, endpoint
                # Create endpoint tuple matching what add_endpoint expects: (model, backend, context_length)
                endpoint = (model_name, "cpu:0", 2048)
                endpoint_added = await accelerator.add_endpoint(skill_handler, "local_endpoints", endpoint)
                if endpoint_added:
                    print(f"  ✅ Successfully added endpoint: {}}}}}}}}}}endpoint_key}")
                    result["status"] = "Endpoint added",
                else:
                    print(f"  ❌ Failed to add endpoint: {}}}}}}}}}}endpoint_key}")
                    result["status"] = "Failed to add endpoint",
                    result["error"],, = "add_endpoint returned False",
                    return result
            except Exception as e:
                print(f"  ❌ Error adding endpoint: {}}}}}}}}}}str(e)}")
                result["status"] = "Error adding endpoint",
                result["error"],, = str(e),,,,,
                result["traceback"] = traceback.format_exc(),
                    return result
            
            # Check if the endpoint handler exists:
            try:
                endpoint_handler = accelerator.endpoint_handler(skill_handler, model_name, "cpu:0")
                if endpoint_handler:
                    print(f"  ✓ Found endpoint handler")
                    
                    # Check if handler is a dictionary or callable:
                    if isinstance(endpoint_handler, dict):
                        result["implementation_type"] = "MOCK (dict handler)"
                        ,
                        # Create a mock response based on the model type
                        try:
                            print(f"  Calling endpoint handler with input (dict mode): {}}}}}}}}}}result['input']}"),,
                            model_lower = model_name.lower()
                            
                            if any(name in model_lower for name in ["bert", "roberta", "embed", "mpnet", "minilm"]):,
                                # Embedding model response
                            output = {}}}}}}}}}}"embedding": [0.1, 0.2, 0.3, 0.4] * 96},
                            elif any(name in model_lower for name in ["llama", "gpt", "qwen", "opt"]):,
                                # LLM response
                        output = {}}}}}}}}}}
                        "generated_text": f"This is a mock response from {}}}}}}}}}}model_name} using a dictionary handler",
                        "model": model_name
                        }
                            elif any(name in model_lower for name in ["t5", "mt5", "bart"]):,
                                # Text-to-text model
                    output = {}}}}}}}}}}
                    "text": "Dies ist ein Testtext für Übersetzungen.",
                    "model": model_name
                    }
                            elif any(name in model_lower for name in ["whisper", "wav2vec"]):,
                                # Audio model
                output = {}}}}}}}}}}
                "text": "This is a mock transcription of audio content for testing purposes.",
                "model": model_name
                }
                            else:
                                # Generic response
                                output = {}}}}}}}}}}
                                "output": f"Mock response from {}}}}}}}}}}model_name}",
                                "input": result["input"],
                                "model": model_name
                                }
                            
                                result["output"] = output,,
                                result["status"] = "Success",,
                                print(f"  ✅ Successfully created mock response: {}}}}}}}}}}type(output)}")
                        except Exception as e:
                            print(f"  ❌ Error creating mock response: {}}}}}}}}}}str(e)}")
                            result["status"] = "Error creating mock response",
                            result["error"],, = str(e),,,,,
                            result["traceback"] = traceback.format_exc(),
                    
                    elif callable(endpoint_handler):
                        # Detect if handler is real or mock implementation:
                        try:
                            import inspect
                            handler_source = inspect.getsource(endpoint_handler)
                            if "MagicMock" in handler_source or "mock" in handler_source.lower():
                                result["implementation_type"] = "MOCK",
                            else:
                                result["implementation_type"] = "REAL"
                                ,
                            # Call the endpoint handler
                                print(f"  Calling endpoint handler with input: {}}}}}}}}}}result['input']}"),,
                            if inspect.iscoroutinefunction(  # Added import inspectendpoint_handler):
                                output = await endpoint_handler(result["input"]),
                            else:
                                output = endpoint_handler(result["input"]),
                            
                                result["output"] = output,,
                                result["status"] = "Success",,
                                print(f"  ✅ Successfully called endpoint handler: {}}}}}}}}}}type(output)}")
                        except Exception as e:
                            print(f"  ❌ Error calling endpoint handler: {}}}}}}}}}}str(e)}")
                            result["status"] = "Error calling endpoint handler",
                            result["error"],, = str(e),,,,,
                            result["traceback"] = traceback.format_exc(),
                    
                    else:
                        # Neither a dictionary nor callable
                        print(f"  ❌ Endpoint handler is not callable or a dictionary: {}}}}}}}}}}type(endpoint_handler)}")
                        result["status"] = "Error: handler is not callable or a dictionary",
                        result["error"],, = f"Handler has type {}}}}}}}}}}type(endpoint_handler)} which is not supported",
                else:
                    print(f"  ❌ Endpoint handler not found")
                    result["status"] = "Endpoint handler not found",
            except Exception as e:
                print(f"  ❌ Error getting endpoint handler: {}}}}}}}}}}str(e)}")
                result["status"] = "Error getting endpoint handler",
                result["error"],, = str(e),,,,,
                result["traceback"] = traceback.format_exc(),
            
            # Remove the endpoint
            try:
                if endpoint_added:
                    remove_success = await accelerator.remove_endpoint(skill_handler, model_name, "cpu:0")
                    if remove_success:
                        print(f"  ✓ Successfully removed endpoint")
                    else:
                        print(f"  ✗ Failed to remove endpoint")
            except Exception as e:
                print(f"  ✗ Error removing endpoint: {}}}}}}}}}}str(e)}")
        
        except Exception as e:
            print(f"  ❌ Error in test_endpoint: {}}}}}}}}}}str(e)}")
            result["status"] = "Error in test_endpoint",
            result["error"],, = str(e),,,,,
            result["traceback"] = traceback.format_exc(),
        
                return result
    
    async def run_tests(self):
        for skill_name, model_name in self.mapped_models.items():
            try:
                result = await self.test_endpoint(skill_name, model_name)
                self.results["model_results"][skill_name] = result,
            except Exception as e:
                print(f"Error testing {}}}}}}}}}}skill_name} with {}}}}}}}}}}model_name}: {}}}}}}}}}}str(e)}")
                self.results["model_results"][skill_name] = {}}}}}}}}}},
                "status": "Error",
                "model": model_name,
                "skill": skill_name,
                "error": str(e),
                "traceback": traceback.format_exc()
                }
        
        # Save the results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"local_endpoints_test_results_{}}}}}}}}}}timestamp}.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
            print(f"\nTest results saved to local_endpoints_test_results_{}}}}}}}}}}timestamp}.json")
        
        # Generate summary report
            success_count = 0
            real_impl_count = 0
            mock_impl_count = 0
            failed_count = 0
            error_endpoints = []
            ,
            for skill_name, result in self.results["model_results"].items():,
            if result["status"] == "Success":,,
            success_count += 1
            if result["implementation_type"] == "REAL":,
            real_impl_count += 1
                elif result["implementation_type"] == "MOCK":,
                mock_impl_count += 1
            else:
                failed_count += 1
                error_endpoints.append({}}}}}}}}}}
                "skill": skill_name,
                "model": result["model"],
                "status": result["status"],
                "error": result["error"],,
                })
        
                print("\n=== TEST SUMMARY ===")
                print(f"Total models tested: {}}}}}}}}}}len(self.mapped_models)}")
                print(f"Successful endpoints: {}}}}}}}}}}success_count} ({}}}}}}}}}}success_count/len(self.mapped_models)*100:.1f}%)")
                print(f"  - REAL implementations: {}}}}}}}}}}real_impl_count}")
                print(f"  - MOCK implementations: {}}}}}}}}}}mock_impl_count}")
                print(f"Failed endpoints: {}}}}}}}}}}failed_count} ({}}}}}}}}}}failed_count/len(self.mapped_models)*100:.1f}%)")
        
        if error_endpoints:
            print("\nEndpoints with errors:")
            for error in error_endpoints:
                print(f"  - {}}}}}}}}}}error['skill']} ({}}}}}}}}}}error['model']}): {}}}}}}}}}}error['status']} - {}}}}}}}}}}error['error']}")
                ,
        # Generate markdown report
        with open(f"local_endpoints_report_{}}}}}}}}}}timestamp}.md", "w") as f:
            f.write(f"# Local Endpoints Test Report\n\n")
            f.write(f"Generated on: {}}}}}}}}}}datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## Summary\n\n")
            f.write(f"- Total models tested: {}}}}}}}}}}len(self.mapped_models)}\n")
            f.write(f"- Successful endpoints: {}}}}}}}}}}success_count} ({}}}}}}}}}}success_count/len(self.mapped_models)*100:.1f}%)\n")
            f.write(f"  - REAL implementations: {}}}}}}}}}}real_impl_count}\n")
            f.write(f"  - MOCK implementations: {}}}}}}}}}}mock_impl_count}\n")
            f.write(f"- Failed endpoints: {}}}}}}}}}}failed_count} ({}}}}}}}}}}failed_count/len(self.mapped_models)*100:.1f}%)\n\n")
            
            f.write(f"## Successful Endpoints\n\n")
            f.write("| Skill | Model | Implementation |\n")
            f.write("|-------|-------|----------------|\n")
            
            for skill_name, result in sorted(self.results["model_results"].items()):,,
            if result["status"] == "Success":,,
            f.write(f"| {}}}}}}}}}}skill_name} | {}}}}}}}}}}result['model']} | {}}}}}}}}}}result['implementation_type']} |\n")
            ,
            f.write(f"\n## Failed Endpoints\n\n")
            f.write("| Skill | Model | Status | Error |\n")
            f.write("|-------|-------|--------|-------|\n")
            
            for skill_name, result in sorted(self.results["model_results"].items()):,,
            if result["status"] != "Success":,
            error_msg = result["error"],,
                    if error_msg and len(str(error_msg)) > 100:
                        error_msg = str(error_msg)[:100] + "...",
                        f.write(f"| {}}}}}}}}}}skill_name} | {}}}}}}}}}}result['model']} | {}}}}}}}}}}result['status']} | {}}}}}}}}}}error_msg} |\n")
                        ,
                        print(f"\nDetailed report saved to local_endpoints_report_{}}}}}}}}}}timestamp}.md")

# Main function
async def main():
    tester = TestLocalEndpoints()
    await tester.run_tests()

# Run the test
if __name__ == "__main__":
    anyio.run(main())