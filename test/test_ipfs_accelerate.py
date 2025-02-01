import asyncio
import os
import sys

class test_ipfs_accelerate:
    def __init__(self, resources=None, metadata=None):
        
        if self.resources is None:
            self.resources = {}
        else:
            self.resources = resources
        
        if self.metadata is None:
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
                from test_backend import test_backend
                self.resources["test_backend"] = test_backend(resources, metadata)
                self.test_backend = self.resources["test_backend"]
            else:
                self.test_backend = self.resources["test_backend"]
        
        return None
    
    async def test(self):
        test_results = {}
        try:
            # test_results["test_ipfs_accelerate"] = self.ipfs_accelerate.__test__(resources, metadata)
            results = {}
            ipfs_accelerate_init = await self.ipfs_accelerate_py.init_endpoints( metadata['models'], resources)
            test_endpoints = await self.ipfs_accelerate_py.test_endpoints(metadata['models'], ipfs_accelerate_init)
            return test_endpoints
        except Exception as e:
            test_results["test_ipfs_accelerate"] = e
        
        try:
            test_results["test_backend"] = self.test_backend.__test__(resources, metadata)
        except Exception as e:
            test_results["test_backend"] = e
            
        return test_results
    
    async def test_ipfs_accelerate(self):
        test_results = {}
        try:
            ipfs_accelerate_py = self.ipfs_accelerate_py(resources, metadata)
            # test_results["test_ipfs_accelerate"] = await ipfs_acclerate.__test__(resources, metadata)
            ipfs_accelerate_init = ipfs_accelerate_py.init_endpoints( metadata['models'], resources)
            test_endpoints = ipfs_accelerate_py.test_endpoints(metadata['models'], ipfs_accelerate_init)
            return test_endpoints 
        except Exception as e:
            test_results["test_ipfs_accelerate"] = e
            return test_results
    
if __name__ == "__main__":
    metadata = {
        "dataset": "laion/gpt4v-dataset",
        "namespace": "laion/gpt4v-dataset",
        "column": "link",
        "role": "master",
        "split": "train",
        "models": [
            # "laion/larger_clap_general",
            "google-t5/t5-base",
            # "facebook/wav2vec2-large-960h-lv60-self",
            # "BAAI/bge-small-en-v1.5", 
            # "openai/clip-vit-base-patch16",  ## fix audio tensor and check that the right format is being used for whisper models in the test Can't set the input tensor with index: 0, because the model input (shape=[?,?]) and the tensor (shape=(0)) are incompatible  
            # "openai/whisper-large-v3-turbo",
            # "meta-llama/Meta-Llama-3.1-8B-Instruct",
            # "distil-whisper/distil-large-v3",


            # "Qwen/Qwen2-7B",
            # "llava-hf/llava-interleave-qwen-0.5b-hf",
            # "lmms-lab/LLaVA-Video-7B-Qwen2",
            # "llava-hf/llava-v1.6-mistral-7b-hf",
            # "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            # "TIGER-Lab/Mantis-8B-siglip-llama3",  ## make sure sthat optimum-cli-convert works on windows.
            # "microsoft/xclip-base-patch16-zero-shot",
            # "google/vit-base-patch16-224"


            # "MCG-NJU/videomae-base",
            # "MCG-NJU/videomae-large",
            # "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",   ## openclip not yet supported
            # "lmms-lab/llava-onevision-qwen2-7b-si",  
            # "lmms-lab/llava-onevision-qwen2-7b-ov", 
            # "lmms-lab/llava-onevision-qwen2-0.5b-si", 
            # "lmms-lab/llava-onevision-qwen2-0.5b-ov", 
            # "Qwen/Qwen2-VL-7B-Instruct", ## convert_model() ->   ('Couldn\'t get TorchScript module by scripting. With exception:\nComprehension ifs are not supported yet:\n  File "/home/devel/.local/lib/python3.12/site-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py", line 1187\n    \n        if not return_dict:\n            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)\n        return BaseModelOutputWithPast(\n            last_hidden_state=hidden_states,\n\n\nTracing sometimes provide better results, please provide valid \'example_input\' argument. You can also provide TorchScript module that you obtained yourself, please refer to PyTorch documentation: https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html.',)
            # "OpenGVLab/InternVL2_5-1B", ## convert_model() -> torchscript error Couldn't get TorchScript module by scripting. With exception: try blocks aren't supported:
            # "OpenGVLab/InternVL2_5-8B", ## convert_model() -> torchscript error Couldn't get TorchScript module by scripting. With exception: try blocks aren't supported:
            # "OpenGVLab/PVC-InternVL2-8B", ## convert_model() -> torchscript error Couldn't get TorchScript module by scripting. With exception: try blocks aren't supported:
            # "AIDC-AI/Ovis1.6-Llama3.2-3B", # ValueError: Trying to export a ovis model, that is a custom or unsupported architecture,
            # "BAAI/Aquila-VL-2B-llava-qwen", # Asked to export a qwen2 model for the task visual-question-answering (auto-detected), but the Optimum OpenVINO exporter only supports the tasks feature-extraction, feature-extraction-with-past, text-generation, text-generation-with-past, text-classification for qwen2. Please use a supported task. Please open an issue at https://github.com/huggingface/optimum/issues if you would like the task visual-question-answering to be supported in the ONNX export for qwen2.
        ],
        "chunk_settings": {

        },
        "path": "/storage/gpt4v-dataset/data",
        "dst_path": "/storage/gpt4v-dataset/data",
    }
    resources = {
        "local_endpoints": [
            ["google-t5/t5-base", "cpu", 32768],
            ["openai/whisper-large-v3-turbo", "cpu", 32768],
            ["MCG-NJU/videomae-base", "cpu", 32768],
            ["microsoft/xclip-base-patch16-zero-shot", "cpu", 32768],
            ["MCG-NJU/videomae-large", "cpu", 32768],
            ["BAAI/bge-small-en-v1.5", "cpu", 32768],
            ["laion/larger_clap_general", "cpu", 32768],
            ["facebook/wav2vec2-large-960h-lv60-self", "cpu", 32768],
            ["openai/clip-vit-base-patch16", "cpu", 32768],
            ["laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "cpu", 32768],
            ["llava-hf/llava-v1.6-mistral-7b-hf", "cpu", 32768],
            ["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "cpu", 32768],
            ["TIGER-Lab/Mantis-8B-siglip-llama3", "cpu", 32768],
            ["lmms-lab/llava-onevision-qwen2-7b-si", "cpu", 32768],
            ["lmms-lab/llava-onevision-qwen2-7b-ov", "cpu", 32768],
            ["lmms-lab/LLaVA-Video-7B-Qwen2", "cpu", 32768],
            ["meta-llama/Meta-Llama-3.1-8B-Instruct", "cpu", 32768],
            ["Qwen/Qwen2-7B", "cpu", 32768],
            ["llava-hf/llava-interleave-qwen-0.5b-hf", "cpu", 32768],
            ["lmms-lab/llava-onevision-qwen2-0.5b-si", "cpu", 32768],
            ["lmms-lab/llava-onevision-qwen2-0.5b-ov", "cpu", 32768],
            ["Qwen/Qwen2-VL-7B-Instruct", "cpu", 32768],
            ["OpenGVLab/InternVL2_5-1B", "cpu", 32768],
            ["OpenGVLab/InternVL2_5-8B", "cpu", 32768],
            ["OpenGVLab/PVC-InternVL2-8B", "cpu", 32768],
            ["facebook/wav2vec2-large-960h-lv60-self", "cpu", 32768],
            ["laion/larger_clap_general", "cpu", 32768],
            ["openai/clip-vit-base-patch16", "cpu", 32768],
            ["BAAI/bge-small-en-v1.5", "cpu", 32768],
            ["llava-hf/llava-v1.6-mistral-7b-hf", "cpu", 32768],
            ["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "cpu", 32768],
            ["distil-whisper/distil-large-v3", "cpu", 32768],
            ["facebook/wav2vec2-large-960h-lv60-self", "cuda:0", 32768],
            ["google-t5/t5-base", "cuda:0", 32768],
            ["openai/whisper-large-v3-turbo", "cuda:0", 32768],
            ["MCG-NJU/videomae-base", "cuda:0", 32768],
            ["microsoft/xclip-base-patch16-zero-shot", "cuda:0", 32768],
            ["MCG-NJU/videomae-large", "cuda:0", 32768],
            ["BAAI/bge-small-en-v1.5", "cuda:0", 32768],
            ["laion/larger_clap_general", "cuda:0", 32768],
            ["facebook/wav2vec2-large-960h-lv60-self", "cuda:0", 32768],
            ["openai/clip-vit-base-patch16", "cuda:0", 32768],
            ["laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "cuda:0", 32768],
            ["llava-hf/llava-v1.6-mistral-7b-hf", "cuda:0", 32768],
            ["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "cuda:0", 32768],
            ["TIGER-Lab/Mantis-8B-siglip-llama3", "cuda:0", 32768],
            ["TIGER-Lab/Mantis-8B-siglip-llama3", "openvino:0", 32768],
            ["lmms-lab/llava-onevision-qwen2-7b-si", "cuda:0", 32768],
            ["lmms-lab/llava-onevision-qwen2-7b-ov", "cuda:0", 32768],
            ["lmms-lab/LLaVA-Video-7B-Qwen2", "cuda:0", 32768],
            ["meta-llama/Meta-Llama-3.1-8B-Instruct", "cuda:0", 32768],
            ["Qwen/Qwen2-7B", "cuda:0", 32768],
            ["Qwen/Qwen2-7B", "openvino:0", 32768],
            ["llava-hf/llava-interleave-qwen-0.5b-hf", "cuda:0", 32768],
            ["lmms-lab/llava-onevision-qwen2-0.5b-si", "cuda:0", 32768],
            ["lmms-lab/llava-onevision-qwen2-0.5b-ov", "cuda:0", 32768],
            ["Qwen/Qwen2-VL-7B-Instruct", "cuda:0", 32768],
            ["OpenGVLab/InternVL2_5-1B", "cuda:0", 32768],
            ["OpenGVLab/InternVL2_5-8B", "cuda:0", 32768],
            ["OpenGVLab/PVC-InternVL2-8B", "cuda:0", 32768],
            ["laion/larger_clap_general", "cuda:0", 32768],
            ["openai/clip-vit-base-patch16", "cuda:0", 32768],
            ["BAAI/bge-small-en-v1.5", "cuda:0", 32768],
            ["llava-hf/llava-v1.6-mistral-7b-hf", "cuda:0", 32768],
            ["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "cuda:0", 32768],
            ["facebook/wav2vec2-large-960h-lv60-self", "cuda:1", 32768],
            ["google-t5/t5-base", "cuda:1", 32768],
            ["openai/whisper-large-v3-turbo", "cuda:1", 32768],
            ["MCG-NJU/videomae-base", "cuda:1", 32768],
            ["microsoft/xclip-base-patch16-zero-shot", "cuda:1", 32768],
            ["MCG-NJU/videomae-large", "cuda:1", 32768],
            ["BAAI/bge-small-en-v1.5", "cuda:1", 32768],
            ["laion/larger_clap_general", "cuda:1", 32768],
            ["openai/clip-vit-base-patch16", "cuda:1", 32768],
            ["BAAI/bge-small-en-v1.5", "cuda:1", 32768],
            ["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "cuda:1", 32768],
            ["distil-whisper/distil-large-v3", "cuda:1", 32768],
            ["distil-whisper/distil-large-v3", "openvino:0", 32768],
            ["google-t5/t5-base", "openvino:0", 32768],
            ["openai/whisper-large-v3-turbo", "openvino:0", 32768],
            ["MCG-NJU/videomae-base", "openvino:0", 32768],
            ["microsoft/xclip-base-patch16-zero-shot", "openvino:0", 32768],
            ["MCG-NJU/videomae-large", "openvino:0", 32768],
            ["BAAI/bge-small-en-v1.5", "openvino:0", 32768],
            ["laion/larger_clap_general", "openvino:0", 32768],
            ["facebook/wav2vec2-large-960h-lv60-self", "openvino:0", 32768],
            ["BAAI/bge-small-en-v1.5", "openvino:0", 32768],
            ["openai/clip-vit-base-patch16", "openvino:0", 32768],
            ["laion/larger_clap_general", "openvino:0", 32768],
            ["llava-hf/llava-v1.6-mistral-7b-hf", "openvino:0", 32768],
            ["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "openvino:0", 32768],
            ["BAAI/bge-small-en-v1.5", "llama_cpp", 512],
            ["llava-hf/llava-v1.6-mistral-7b-hf", "llama_cpp", 8192],
            ["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "llama_cpp", 32768],
            ["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "ipex", 32768],
            ["distil-whisper/distil-large-v3", "ipex", 32768],
            ["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "ipex", 32768],
        ],
        "openvino_endpoints": [],
        "tei_endpoints": [],
    }

    ipfs_accelerate_py = ipfs_accelerate_py(resources, metadata)
    asyncio.run(ipfs_accelerate_py.__test__(resources, metadata))
    print("test complete")
