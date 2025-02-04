import asyncio
import os
import sys
import json 


class test_ipfs_accelerate_py:
    def __init__(self, resources=None, metadata=None):
        
        if resources is None:
            self.resources = {}
        else:
            self.resources = resources
        
        if metadata is None:
            self.metadata = {}
        else:
            self.metadata = metadata
        
        if "ipfs_accelerate_py" not in dir(self):
            if "ipfs_accelerate_py" not in list(self.resources.keys()):
                from ipfs_accelerate_py import ipfs_accelerate_py
                self.resources["ipfs_accelerate_py"] = ipfs_accelerate_py(resources, metadata)
                self.ipfs_accelerate_py = self.resources["ipfs_accelerate_py"]
            else:
                self.ipfs_accelerate_py = self.resources["ipfs_accelerate_py"]
        
        if "test_backend" not in dir(self):
            if "test_backend" not in list(self.resources.keys()):
                from test_backend import test_backend_py
                self.resources["test_backend"] = test_backend_py(resources, metadata)
                self.test_backend = self.resources["test_backend"]
            else:
                self.test_backend = self.resources["test_backend"]
                
        if "torch" not in dir(self):
            if "torch" not in list(self.resources.keys()):
                import torch
                self.resources["torch"] = torch
                self.torch = self.resources["torch"]
            else:
                self.torch = self.resources["torch"]
        
        return None
    
    async def get_huggingface_model_types(self):
        if "transformers" not in dir(self):
            if "transformers" not in list(self.resources.keys()):
                import transformers
                self.resources["transformers"] = transformers
                self.transformers = self.resources["transformers"]
            else:
                self.transformers = self.resources["transformers"]

        # Get all model types from the MODEL_MAPPING
        model_types = []
        for config in self.transformers.MODEL_MAPPING.keys():
            if hasattr(config, 'model_type'):
                model_types.append(config.model_type)

        # Add model types from the AutoModel registry
        model_types.extend(list(self.transformers.MODEL_MAPPING._model_mapping.keys()))
        
        # Remove duplicates and sort
        model_types = sorted(list(set(model_types)))
        return model_types    
    
    async def test(self):
        test_results = {}
        try:
            # test_results["test_ipfs_accelerate"] = self.ipfs_accelerate.__test__(resources, metadata)
            results = {}
            for model in self.metadata['models']:
                ipfs_accelerate_init = await self.ipfs_accelerate_py.init_endpoints( [model], resources) 
                test_endpoints = await self.ipfs_accelerate_py.test_endpoints([model], ipfs_accelerate_init)
                ipfs_accelerate_del = await self.ipfs_accelerate_py.del_endpoints( [model], resources)
            return test_endpoints
        except Exception as e:
            error = ""
            import traceback
            error = "Error initializing worker:\n"
            error += f"Exception type: {type(e).__name__}\n"
            error += f"Exception message: {str(e)}\n"
            error += "Traceback:\n" + traceback.format_exc() 
            test_results["test_ipfs_accelerate"] = str(error)

        try:
            test_results["test_backend"] = self.test_backend.__test__(resources, metadata)
        except Exception as e:
            import traceback
            error = "Error initializing worker:\n"
            error += f"Exception type: {type(e).__name__}\n"
            error += f"Exception message: {str(e)}\n"
            error += "Traceback:\n" + traceback.format_exc() 
            test_results["test_backend"] = str(error)
        return test_results


    def get_model_type(self, model_name=None, model_type=None):
        if "transformers" not in dir(self):
            if "transformers" not in list(self.resources.keys()):
                import transformers
                self.resources["transformers"] = transformers
                self.transformers = self.resources["transformers"]
            else:
                self.transformers = self.resources["transformers"]

        if model_name is not None:
            if os.path.exists(model_name):
                config = self.transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                model_type = config.__class__.model_type
            else:
                config = self.transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                model_type = config.__class__.model_type
        return model_type
    
    async def test_local_endpoint(self, model, endpoint_list=None):
        this_endpoint = None
        filtered_list = {}
        test_results = {}
        local_endpoints = self.resources["local_endpoints"]
        local_tokenizers = self.resources["tokenizer"]
        local_endpoints_types = [x[1] for x in local_endpoints]
        local_tokenizers_types = [x[1] for x in local_tokenizers]
        local_endpoints_by_model = self.ipfs_accelerate_py.endpoints["local_endpoints"][model]
        endpoint_handlers_by_model = self.ipfs_accelerate_py.resources["endpoint_handler"][model]
        tokenizers_by_model = self.ipfs_accelerate_py.resources["tokenizer"][model]
        if endpoint_list is not None:
            local_endpoints_by_model_by_endpoint_list = [ x for x in local_endpoints_by_model if ("openvino:" in json.dumps(x) or "cuda:" in json.dumps(x) ) and x[1] in list(endpoint_handlers_by_model.keys()) ]
        else:
            local_endpoints_by_model_by_endpoint_list = [ x for x in local_endpoints_by_model if ( "openvino:" in json.dumps(x) or "cuda:" in json.dumps(x) ) ]      
        if len(local_endpoints_by_model_by_endpoint_list) > 0:
            for endpoint in local_endpoints_by_model_by_endpoint_list:
                model_type = self.get_model_type(model)
                hf_model_types = []
                with open(os.path.join(os.path.dirname(__file__), "hf_model_types.json"), "r") as f:
                    hf_model_types = json.load(f)
                method_name = "hf_" + model_type
                if model_type in hf_model_types:
                    if endpoint[1] in list(endpoint_handlers_by_model.keys()):
                        endpoint_handler = endpoint_handlers_by_model[endpoint[1]]
                        test = None
                        try:
                            module = __import__('worker.skillset', fromlist=[method_name])
                            this_method = getattr(module, method_name)
                            this_hf = this_method(self.resources, self.metadata)
                            test = this_hf.__test__(model, endpoint_handlers_by_model[endpoint[1]], endpoint[1], tokenizers_by_model[endpoint[1]] )
                            test_results[endpoint[1]] = test
                            del this_hf
                            del this_method
                            del module
                            del test
                        except Exception as e:
                            test_results[endpoint[1]] = e
                    else:
                        test_results[endpoint[1]] = ValueError("endpoint not found")          
                else:
                    test_results[endpoint[1]] = ValueError("Model type not supported")
        return test_results

    
    async def test_tei_endpoint(self, model, endpoint_list=None):
        this_endpoint = None
        filtered_list = {}
        test_results = {}
        local_endpoints = self.ipfs_accelerate_py.resources["tei_endpoints"]
        local_endpoints_types = [x[1] for x in local_endpoints]
        local_endpoints_by_model = self.ipfs_accelerate_py.endpoints["tei_endpoints"][model]
        endpoint_handlers_by_model = self.ipfs_accelerate_py.resources["tei_endpoints"][model]
        local_endpoints_by_model_by_endpoint = list(endpoint_handlers_by_model.keys())
        local_endpoints_by_model_by_endpoint = [ x for x in local_endpoints_by_model_by_endpoint if x in local_endpoints_by_model if x in local_endpoints_types]
        if len(local_endpoints_by_model_by_endpoint) > 0:
            for endpoint in local_endpoints_by_model_by_endpoint:
                endpoint_handler = endpoint_handlers_by_model[endpoint]
                try:
                    test = await endpoint_handler("hello world")
                    test_results[endpoint] = test
                except Exception as e:
                    try:
                        test = endpoint_handler("hello world")
                        test_results[endpoint] = test
                    except Exception as e:
                        test_results[endpoint] = e
                    pass
        else:
            return ValueError("No endpoint_handlers found")
        return test_results
    
    async def test_libp2p_endpoint(self, model, endpoint=None):
        return ValueError("Not implemented")

    async def test_openvino_endpoint(self, model, endpoint_list=None):
        this_endpoint = None
        filtered_list = {}
        test_results = {}
        local_endpoints = self.ipfs_accelerate_py.resources["openvino_endpoints"]
        local_endpoints_types = [x[1] for x in local_endpoints]
        local_endpoints_by_model = self.ipfs_accelerate_py.endpoints["openvino_endpoints"][model]
        endpoint_handlers_by_model = self.ipfs_accelerate_py.resources["openvino_endpoints"][model]
        if endpoint_list is not None:
            local_endpoints_by_model_by_endpoint_list = [ x for x in local_endpoints_by_model if "openvino:" in json.dumps(x) and x[1] in list(endpoint_handlers_by_model.keys()) ]
        else:
            local_endpoints_by_model_by_endpoint_list = [ x for x in local_endpoints_by_model if "openvino:" in json.dumps(x) ]
        if len(local_endpoints_by_model_by_endpoint_list) > 0:
            for endpoint in local_endpoints_by_model_by_endpoint_list:
                endpoint_handler = endpoint_handlers_by_model[endpoint]
                try:
                    test = await endpoint_handler("hello world")
                    test_results[endpoint] = test
                except Exception as e:
                    try:
                        test = endpoint_handler("hello world")
                        test_results[endpoint] = test
                    except Exception as e:
                        test_results[endpoint] = e
                    pass
        else:
            return ValueError("No endpoint_handlers found")
        return test_results


    async def test_endpoints(self, models, endpoint_handler_object=None):
        test_results = {}
        for model in models:
            if model not in list(test_results.keys()):
                test_results[model] = {}
            try: 
                test_results[model]["local_endpoint"] = await self.test_local_endpoint(model)
            except Exception as e:
                test_results[model]["local_endpoint"] = e
                print(e)

            try:
                test_results[model]["libp2p_endpoint"] = await self.test_libp2p_endpoint(model)
            except Exception as e:
                test_results[model]["libp2p_endpoint"] = e
                print(e)

            try:
                test_results[model]["openvino_endpoint"] = await self.test_openvino_endpoint(model)
            except Exception as e:
                test_results[model]["openvino_endpoint"] = e
                print(e)

            try:
                test_results[model]["tei_endpoint"] = await self.test_tei_endpoint(model)
            except Exception as e:
                test_results[model]["tei_endpoint"] = e
                print(e)

            try:
                test_results[model]["webnn_endpoint"] = "not implemented"
            except Exception as e:
                test_results[model]["webnn_endpoint"] = e
                print(e)

        try:
            test_results[model]["endpoint_handler_resources"] = endpoint_handler_object
        except Exception as e:
            error = ""
            import traceback
            error = "Error initializing worker:\n"
            error += f"Exception type: {type(e).__name__}\n"
            error += f"Exception message: {str(e)}\n"
            error += "Traceback:\n" + traceback.format_exc()
            test_results[model]["endpoint_handler_resources"] = str(error)
            test_results["batch_sizes"] = {}
            test_results["endpoint_handler"] = {}            
        return test_results
    
    async def test_ipfs_accelerate(self):
        test_results = {}
        try:
            ipfs_accelerate_py = self.ipfs_accelerate_py(resources, metadata)
            # test_results["test_ipfs_accelerate"] = await ipfs_acclerate.__test__(resources, metadata)
            ipfs_accelerate_init = ipfs_accelerate_py.init_endpoints( metadata['models'], resources)
            test_endpoints = self.test_endpoints(metadata['models'], ipfs_accelerate_init)
            return test_endpoints 
        except Exception as e:
            error = ""
            import traceback
            error = "Error initializing worker:\n"
            error += f"Exception type: {type(e).__name__}\n"
            error += f"Exception message: {str(e)}\n"
            error += "Traceback:\n" + traceback.format_exc()
            test_results["test_ipfs_accelerate"] = str(error)
            return test_results
    
    async def __test__(self, resources, metadata):
        mapped_models = {}
        with open(os.path.join(os.path.dirname(__file__), "mapped_models.json"), "r") as f:
            mapped_models = json.load(f)
        mapped_models_values = list(mapped_models.values())
        self.metadata["models"] = mapped_models_values
        metadata["models"] = mapped_models_values
        endpoint_types = ["cuda:0", "openvino:0", "cpu:0"]
        for model in metadata["models"]:
            for endpoint in endpoint_types:
                resources["local_endpoints"].append([model, endpoint, 32768])
        self.resources = resources
        test_results = {}
        try:
            test_results["test_ipfs_accelerate"] = await self.test()
        except Exception as e:
            error = ""
            import traceback
            error = "Error initializing worker:\n"
            error += f"Exception type: {type(e).__name__}\n"
            error += f"Exception message: {str(e)}\n"
            error += "Traceback:\n" + traceback.format_exc()
            test_results["test_ipfs_accelerate"] = str(error)
        this_file = os.path.abspath(sys.modules[__name__].__file__)
        test_log = os.path.join(os.path.dirname(this_file), "test_results.json")
        with open(test_log, "w") as f:
            f.write(json.dumps(test_results, indent=4))
        return test_results
    
    
if __name__ == "__main__":
    metadata = {
        "dataset": "laion/gpt4v-dataset",
        "namespace": "laion/gpt4v-dataset",
        "column": "link",
        "role": "master",
        "split": "train",
        "models": [
            "google-t5/t5-base",
            "BAAI/bge-small-en-v1.5",
            "laion/larger_clap_general",
            "facebook/wav2vec2-large-960h-lv60-self",
            "openai/clip-vit-base-patch16",  ## fix audio tensor and check that the right format is being used for whisper models in the test Can't set the input tensor with index: 0, because the model input (shape=[?,?]) and the tensor (shape=(0)) are incompatible  
            "openai/whisper-large-v3-turbo",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "distil-whisper/distil-large-v3",
            "Qwen/Qwen2-7B",
            "llava-hf/llava-interleave-qwen-0.5b-hf",
            "lmms-lab/LLaVA-Video-7B-Qwen2",
            "llava-hf/llava-v1.6-mistral-7b-hf",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "TIGER-Lab/Mantis-8B-siglip-llama3",  ## make sure sthat optimum-cli-convert works on windows.
            "microsoft/xclip-base-patch16-zero-shot",
            "google/vit-base-patch16-224"
            "MCG-NJU/videomae-base",
            "MCG-NJU/videomae-large",
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",   ## openclip not yet supported
            "lmms-lab/llava-onevision-qwen2-7b-si",  
            "lmms-lab/llava-onevision-qwen2-7b-ov", 
            "lmms-lab/llava-onevision-qwen2-0.5b-si", 
            "lmms-lab/llava-onevision-qwen2-0.5b-ov", 
            "Qwen/Qwen2-VL-7B-Instruct", ## convert_model() ->   ('Couldn\'t get TorchScript module by scripting. With exception:\nComprehension ifs are not supported yet:\n  File "/home/devel/.local/lib/python3.12/site-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py", line 1187\n    \n        if not return_dict:\n            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)\n        return BaseModelOutputWithPast(\n            last_hidden_state=hidden_states,\n\n\nTracing sometimes provide better results, please provide valid \'example_input\' argument. You can also provide TorchScript module that you obtained yourself, please refer to PyTorch documentation: https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html.',)
            "OpenGVLab/InternVL2_5-1B", ## convert_model() -> torchscript error Couldn't get TorchScript module by scripting. With exception: try blocks aren't supported:
            "OpenGVLab/InternVL2_5-8B", ## convert_model() -> torchscript error Couldn't get TorchScript module by scripting. With exception: try blocks aren't supported:
            "OpenGVLab/PVC-InternVL2-8B", ## convert_model() -> torchscript error Couldn't get TorchScript module by scripting. With exception: try blocks aren't supported:
            "AIDC-AI/Ovis1.6-Llama3.2-3B", # ValueError: Trying to export a ovis model, that is a custom or unsupported architecture,
            "BAAI/Aquila-VL-2B-llava-qwen", # Asked to export a qwen2 model for the task visual-question-answering (auto-detected), but the Optimum OpenVINO exporter only supports the tasks feature-extraction, feature-extraction-with-past, text-generation, text-generation-with-past, text-classification for qwen2. Please use a supported task. Please open an issue at https://github.com/huggingface/optimum/issues if you would like the task visual-question-answering to be supported in the ONNX export for qwen2.
        ],
        "chunk_settings": {

        },
        "path": "/storage/gpt4v-dataset/data",
        "dst_path": "/storage/gpt4v-dataset/data",
    }
    endpoint_types = ["cuda:0", "openvino:0"]
    resources = {}
    resources["local_endpoints"] = []
    resources["tei_endpoints"] = []
    resources["libp2p_endpoints"] = []
    resources["openvino_endpoints"] = []      
    for model in metadata["models"]:
        for endpoint in endpoint_types:
            resources["local_endpoints"].append([model, endpoint, 32768])

    ipfs_accelerate_py = test_ipfs_accelerate_py(resources, metadata)
    asyncio.run(ipfs_accelerate_py.__test__(resources, metadata))
    print("test complete")
