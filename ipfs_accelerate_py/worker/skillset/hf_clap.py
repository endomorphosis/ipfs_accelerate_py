import torch
from torch import no_grad
from transformers import ClapModel, ClapProcessor
import time
import numpy as np
import gc
import asyncio
from transformers import AutoConfig, AutoTokenizer, AutoProcessor
import os
import open_clip
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import os
import openvino as ov
import soundfile as sf
import io
import numpy as np

def load_audio(audio_file):

    if isinstance(audio_file, str) and (audio_file.startswith("http") or audio_file.startswith("https")):
        response = requests.get(audio_file)
        audio_data, samplerate = sf.read(io.BytesIO(response.content))
    else:
        audio_data, samplerate = sf.read(audio_file)
    
    # Ensure audio is mono and convert to float32
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    audio_data = audio_data.astype(np.float32)
    
    return audio_data, samplerate

def load_audio_tensor(audio_file):
    if isinstance(audio_file, str) and (audio_file.startswith("http") or audio_file.startswith("https")):
        response = requests.get(audio_file)
        audio_data, samplerate = sf.read(io.BytesIO(response.content))
    else:
        audio_data, samplerate = sf.read(audio_file)
    
    # Ensure audio is mono and convert to float32
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    audio_data = audio_data.astype(np.float32)
    
    return ov.Tensor(audio_data.reshape(1, -1))

def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()

class ClapEncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        encoder.eval()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        return self.encoder.get_text_features(input_ids, attention_mask)

class hf_clap:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata    
        self.create_openvino_audio_embedding_endpoint_handler = self.create_openvino_audio_embedding_endpoint_handler
        self.create_cuda_audio_embedding_endpoint_handler = self.create_cuda_audio_embedding_endpoint_handler
        self.create_cpu_audio_embedding_endpoint_handler = self.create_cpu_audio_embedding_endpoint_handler
        self.init_cpu = self.init_cpu
        self.init_cuda = self.init_cuda
        self.init_openvino = self.init_openvino
        self.init = self.init
        self.__test__ = self.__test__
        return None

    def init(self):
        return None

    def __test__(self, endpoint_model, endpoint_handler, endpoint_label, tokenizer):
        sentence_1 = "The quick brown fox jumps over the lazy dog"
        audio_1 = "https://calamitymod.wiki.gg/images/2/29/Bees3.wav"
        timestamp1 = time.time()
        try:
            test_batch = endpoint_handler(sentence_1, audio_1)
        except Exception as e:
            print(e)
            pass
        timestamp2 = time.time()
        elapsed_time = timestamp2 - timestamp1
        len_tokens = 1
        tokens_per_second = len_tokens / elapsed_time
        print(f"elapsed time: {elapsed_time}")
        print(f"samples: {len_tokens}")
        print(f"samples per second: {tokens_per_second}")
        # test_batch_sizes = await self.test_batch_sizes(metadata['models'], ipfs_accelerate_init)
        if "openvino" not in endpoint_label:
            with torch.no_grad():
                if "cuda" in dir(torch):
                    torch.cuda.empty_cache()
        print("hf_llava test")
        return None
    
    def init_cpu(self, model, device, cpu_label):
        
        return None
    
    def init_cuda(self, model, device, cuda_label):
        config = AutoConfig.from_pretrained(model, trust_remote_code=True)    
        tokenizer = AutoTokenizer.from_pretrained(model)
        processor = ClapProcessor.from_pretrained(model, trust_remote_code=True)
        endpoint = None
        try:
            endpoint = ClapProcessor.from_pretrained(model, torch_dtype=torch.float16, trust_remote_code=True).to(device)
        except Exception as e:
            print(e)
            pass
        endpoint_handler = self.create_cuda_audio_embedding_endpoint_handler(endpoint, tokenizer, model, cuda_label)
        torch.cuda.empty_cache()
        # batch_size = await self.max_batch_size(endpoint_model, cuda_label)
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), 0    

    def init_openvino(self, model=None , model_type=None, device=None, openvino_label=None, get_optimum_openvino_model=None, get_openvino_model=None, get_openvino_pipeline_type=None, openvino_cli_convert=None ):
        endpoint = None
        tokenizer = None
        endpoint_handler = None
        homedir = os.path.expanduser("~")
        model_name_convert = model.replace("/", "--")
        huggingface_cache = os.path.join(homedir, ".cache/huggingface")
        huggingface_cache_models = os.path.join(huggingface_cache, "hub")
        huggingface_cache_models_files = os.listdir(huggingface_cache_models)
        huggingface_cache_models_files_dirs = [os.path.join(huggingface_cache_models, file) for file in huggingface_cache_models_files if os.path.isdir(os.path.join(huggingface_cache_models, file))]
        huggingface_cache_models_files_dirs_models = [ x for x in huggingface_cache_models_files_dirs if "model" in x ]
        huggingface_cache_models_files_dirs_models_model_name = [ x for x in huggingface_cache_models_files_dirs_models if model_name_convert in x ]
        model_src_path = os.path.join(huggingface_cache_models, huggingface_cache_models_files_dirs_models_model_name[0])
        model_dst_path = os.path.join(model_src_path, "openvino")
        # config = AutoConfig.from_pretrained(model)
        task = get_openvino_pipeline_type(model, model_type)
        openvino_index = int(openvino_label.split(":")[1])
        weight_format = ""
        if openvino_index is not None:
            if openvino_index == 0:
                weight_format = "int8" ## CPU
            if openvino_index == 1:
                weight_format = "int4" ## gpu
            if openvino_index == 2:
                weight_format = "int4" ## npu
        model_dst_path = model_dst_path+"_"+weight_format
        if not os.path.exists(model_dst_path):
            # os.makedirs(model_dst_path)
            ## convert model to openvino format
            # openvino_cli_convert(model, model_dst_path=model_dst_path, task=task, weight_format=weight_format, ratio="1.0", group_size=128, sym=True )
            pass
        try:
            tokenizer =  ClapProcessor.from_pretrained(
                model
            )
        except Exception as e:
            print(e)
            try:
                tokenizer =  ClapProcessor.from_pretrained(
                    model_src_path
                )
            except Exception as e:
                print(e)
                pass
        
            models_base_folder = model_dst_path
        
            clap_text_encoder_ir_path = models_base_folder / "clap_text_encoder.xml"

            if not clap_text_encoder_ir_path.exists():
                with torch.no_grad():
                    ov_model = ov.convert_model(
                        ClapEncoderWrapper(pipe.text_encoder),  # model instance
                        example_input={
                            "input_ids": torch.ones((1, 512), dtype=torch.long),
                            "attention_mask": torch.ones((1, 512), dtype=torch.long),
                        },  # inputs for model tracing
                    )
                ov.save_model(ov_model, clap_text_encoder_ir_path)
                del ov_model
                cleanup_torchscript_cache()
                gc.collect()
                print("Text Encoder successfully converted to IR")
            else:
                print(f"Text Encoder will be loaded from {clap_text_encoder_ir_path}")
        
        # genai_model = get_openvino_genai_pipeline(model, model_type, openvino_label)
        try:
            model = get_openvino_model(model, model_type, openvino_label)
            print(model)
        except Exception as e:
            print(e)
            try:
                model = get_optimum_openvino_model(model, model_type, openvino_label)
            except Exception as e:
                print(e)
                pass
        endpoint = model
        endpoint_handler = self.create_openvino_audio_embedding_endpoint_handler(model, tokenizer, model, openvino_label)
        batch_size = 0
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), batch_size              
    
    def create_cpu_audio_embedding_endpoint_handler(self, tokenizer , endpoint_model, cpu_label, endpoint=None, ):
        def handler(x, tokenizer=tokenizer, endpoint_model=endpoint_model, cpu_label=cpu_label, endpoint=None):

            # if method == 'clip_text':
            #         inputs = self.tokenizer([text], return_tensors='pt').to('cuda')

            #         with no_grad():
            #             text_features = self.model.get_text_features(**inputs)

            #         return {
            #             'embedding': text_features[0].cpu().numpy().tolist()
            #         }
                
            #     elif method == 'clip_image':
            #         inputs = self.processor(images=image, return_tensors='pt').to('cuda')

            #         with no_grad():
            #             image_features  = self.model.get_audio_features(**inputs)

            #         return {
            #             'embedding': image_features[0].cpu().numpy().tolist()
            #         }
            if "eval" in dir(endpoint):
                endpoint.eval()
            else:
                pass
            return None
        return handler
    
    def create_cuda_audio_embedding_endpoint_handler(self, tokenizer , endpoint_model, cuda_label, endpoint=None, ):
        def handler(x, tokenizer, endpoint_model, openvino_label, endpoint=None):
            if "eval" in dir(endpoint):
                endpoint.eval()
            else:
                pass
            return None
        return handler
    
    def create_openvino_audio_embedding_endpoint_handler(self, endpoint_model , tokenizer , openvino_label, endpoint=None ):
        def handler(x, y, tokenizer=tokenizer, endpoint_model=endpoint_model, openvino_label=openvino_label, endpoint=None):
            if y is not None:            
                if type(y) == str:
                    image = load_audio(y)
                    inputs = tokenizer(images=[image], return_tensors='pt', padding=True)
                elif type(y) == list:
                    inputs = tokenizer(images=[load_audio(image) for image in y], return_tensors='pt')
                with no_grad():
                    image_features = endpoint_model(dict(inputs))
                    image_embeddings = image_features["image_embeds"]
 
                pass
            
            if x is not None:
                if type(x) == str:
                    inputs = tokenizer(text=y, return_tensors='pt')
                elif type(x) == list:
                    inputs = tokenizer(text=[text for text in x], return_tensors='pt')
                with no_grad():
                    text_features = endpoint_model(dict(inputs))
                    text_embeddings = text_features["last_hidden_state"] 
            
            if x is not None or y is not None:
                if x is not None and y is not None:
                    return {
                        'image_embedding': image_embeddings,
                        'text_embedding': text_embeddings
                    }
                elif x is not None:
                    return {
                        'embedding': image_embeddings
                    }
                elif y is not None:
                    return {
                        'embedding': text_embeddings
                    }            
            return None
        return handler
