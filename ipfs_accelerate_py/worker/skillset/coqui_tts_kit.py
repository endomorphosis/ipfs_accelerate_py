import requests
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, AutoConfig, AutoTokenizer, AutoModelForImageTextToText, pipeline
from ipfs_transformers_py import AutoModel
import torch
from torch import Tensor as T
import torchvision 
from torchvision.transforms import InterpolationMode, Compose, Lambda, Resize, ToTensor, Normalize
import torch 
import anyio
from ..anyio_queue import AnyioQueue
import openvino as ov
from pathlib import Path
import numpy as np
import torch
import json
import time
import os
import tempfile
import openvino_genai as ov_genai
from transformers import TextStreamer

class coqui_tts_kit:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata    
        self.create_openvino_fish_speech_endpoint_handler = self.create_fish_speech_embedding_endpoint_handler
        self.create_cuda_fish_speech_endpoint_handler = self.create_fish_speech_embedding_endpoint_handler
        self.create_cpu_fish_speech_endpoint_handler = self.create_fish_speech_embedding_endpoint_handler
        self.init_cpu = self.init_cpu
        self.init_cuda = self.init_cuda
        self.init_openvino = self.init_openvino
        self.init = self.init
        self.__test__ = self.__test__
        pass
    
    def init_cpu (self, model, device, cpu_label):
        return None
    
    def init_cuda(self, model, device, cuda_label):
        config = AutoConfig.from_pretrained(model, trust_remote_code=True)    
        tokenizer = AutoProcessor.from_pretrained(model)
        endpoint = None
        try:
            endpoint = AutoModelForImageTextToText.from_pretrained(model, trust_remote_code=True).to(device)
        except Exception as e:
            print(e)
            pass
        endpoint_handler = self.create_image_embedding_endpoint_handler(endpoint, tokenizer, model, cuda_label)
        torch.cuda.empty_cache()
        # batch_size = await self.max_batch_size(endpoint_model, cuda_label)
        return endpoint, tokenizer, endpoint_handler, # TODO: Replace with anyio.create_memory_object_stream - AnyioQueue(64), 0
    
    def init_openvino(self, model, model_type, device, openvino_label, get_openvino_genai_pipeline, get_optimum_openvino_model, get_openvino_model, get_openvino_pipeline_type):
        endpoint = None
        tokenizer = None
        endpoint_handler = None
        batch_size = 0                
        tokenizer =  AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
        endpoint = get_openvino_model(model, model_type, openvino_label)
        endpoint_handler = self.create_openvino_image_embedding_endpoint_handler(endpoint,tokenizer, model, openvino_label)
        batch_size = 0
        return endpoint, tokenizer, endpoint_handler, # TODO: Replace with anyio.create_memory_object_stream - AnyioQueue(64), batch_size

    def create_cpu_fish_speech_endpoint_handler(self, local_cpu_endpoint, local_cpu_processor, endpoint_model, cpu_label):
        def handler(x, y=None, local_cpu_endpoint=local_cpu_endpoint, local_cpu_processor=local_cpu_processor, endpoint_model=endpoint_model, cpu_label=cpu_label):
            if "infer" in dir(local_cpu_endpoint):
                return local_cpu_endpoint.infer(x)
            else:
                return None
        return handler

    def create_cuda_fish_speech_endpoint_handler(self, local_cuda_endpoint, local_cuda_processor, endpoint_model, cuda_label):
        def handler(x, y=None, local_cuda_endpoint=local_cuda_endpoint, local_cuda_processor=local_cuda_processor, endpoint_model=endpoint_model, cuda_label=cuda_label):
            # if "eval" in dir(self.local_endpoints[endpoint_model][cuda_label]):
            #       self.local_endpoints[endpoint_model][cuda_label].eval()
            if "eval" in dir(local_cuda_endpoint):
                local_cuda_endpoint.eval()
            else:
                pass
            with torch.no_grad():
                try:
                    torch.cuda.empty_cache()
                    config = AutoConfig.from_pretrained(endpoint_model, trust_remote_code=True)
                    if "generate" in dir(local_cuda_endpoint):
                        if y is not None:
                            return local_cuda_endpoint.generate(x, y)
                        else:
                            return local_cuda_endpoint.generate(x)
                    else:
                        return local_cuda_endpoint(x)
                except Exception as e:
                    print(e)
                    return None
        
        return handler

    def create_openvino_fish_speech_endpoint_handler(self, local_openvino_endpoint, local_openvino_processor, endpoint_model, openvino_label):
        def handler(x, y=None, local_openvino_endpoint=local_openvino_endpoint, local_openvino_processor=local_openvino_processor, endpoint_model=endpoint_model, openvino_label=openvino_label):
            if "infer" in dir(local_openvino_endpoint):
                return local_openvino_endpoint.infer(x)
            else:
                return None
        return handler
    
    def __test__(self, model, device, label):
        return None
    

