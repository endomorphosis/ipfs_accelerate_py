import torch
import librosa
from datasets import Dataset, Audio
from transformers import pipeline
import os
import numpy as np
from pydub import AudioSegment
import tempfile
import io
from transformers import AutoModelForAudioClassification
from transformers import AutoFeatureExtractor
from transformers import AutoTokenizer
from transformers import AutoConfig
import json
import time
from transformers import AutoProcessor
from transformers import AutoModel
import asyncio

class hf_wav2vec:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata    
        self.create_openvino_audio_embedding_endpoint_handler = self.create_openvino_audio_embedding_endpoint_handler
        self.create_cuda_audio_embedding_endpoint_handler = self.create_cuda_audio_embedding_endpoint_handler
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
        image_1 = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
        timestamp1 = time.time()
        try:
            test_batch = endpoint_handler(sentence_1, image_1)
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
        # processor = CLIPProcessor.from_pretrained(model, trust_remote_code=True)
        endpoint = None
        try:
            # endpoint = CLIPModel.from_pretrained(model, torch_dtype=torch.float16, trust_remote_code=True).to(device)
            pass
        except Exception as e:
            print(e)
            pass
        endpoint_handler = self.create_audio_embedding_endpoint_handler(endpoint, tokenizer, model, cuda_label)
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


        # try:
        #     tokenizer =  CLIPProcessor.from_pretrained(
        #         model
        #     )
        # except Exception as e:
        #     print(e)
        #     try:
        #         tokenizer =  CLIPProcessor.from_pretrained(
        #             model_src_path
        #         )
        #     except Exception as e:
        #         print(e)
        #         pass
        
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
                    image = load_image(y)
                    inputs = tokenizer(images=[image], return_tensors='pt', padding=True)
                elif type(y) == list:
                    inputs = tokenizer(images=[load_image(image) for image in y], return_tensors='pt')
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
    
        # def __init__(self, resources=None, metadata=None):
        #         if os.path.exists(resources['checkpoint']) and os.path.isfile(resources['checkpoint'] + "/config.json"):
        #                 self.model = AutoModelForAudioClassification.from_pretrained(
        #         resources['checkpoint'],
        #         local_files_only=True
        #     ).eval()
        #                 self.feature_extractor = AutoFeatureExtractor.from_pretrained(
        #         resources['checkpoint'],
        #         local_files_only=True
        #     )
        #         else:
        #                 self.classifier = pipeline("audio-classification", model=resources['checkpoint'])
        #         with open(os.path.join(resources['checkpoint'], "header.bin"), "rb") as f:
        #                 self.header = f.read()

        # def __call__(self, method, **kwargs):
        #         if method == 'wav2vec_classify':
        #                 return self.wav2vec_classify(**kwargs)
        #         else:
        #                 raise Exception('unknown method: %s' % method)

        # def map_to_array(self, example):
        #         speech, _ = librosa.load(example["file"], sr=16000, mono=True)
        #         example["speech"] = speech
        #         return example

        # def wav2vec_classify(self, audio, **kwargs):
        #         if os.path.exists(audio) and os.path.isfile(audio):
        #                 audio_filename = audio
        #                 audio_dataset = Dataset.from_dict({"audio": [audio_filename],"file":[audio_filename]}).cast_column("audio", Audio())
        #                 audio_dataset = audio_dataset.map(self.map_to_array)
        #                 speech = audio_dataset[:4]["speech"]
        #         else:
        #                 with tempfile.NamedTemporaryFile(suffix=".ogg") as temp_audio:
        #                         if type(audio) == str:
        #                                 audio = audio.encode()
        #                         else:
        #                                 pass
        #                         AudioSegment.from_file(io.BytesIO(self.header + audio),format="webm").export(temp_audio.name, format="ogg")
        #                         audio_filename = temp_audio.name
        #                         audio_dataset = Dataset.from_dict({"audio": [audio_filename],"file":[audio_filename]}).cast_column("audio", Audio())
        #                         audio_dataset = audio_dataset.map(self.map_to_array)
        #                         speech = audio_dataset[:4]["speech"]

        #         if "classifier" in self.__dict__.keys():
        #                 ## audio file path
        #                 results = self.classifier(audio, top_k=5)
        #                 #results = json.dumps(results)

        #         else:
        #                 if "sampling_rate" in kwargs.keys():
        #                         sampling_rate = kwargs["sampling_rate"]
        #                 else:
        #                         sampling_rate = 16000
        #                 inputs = self.feature_extractor(speech, sampling_rate=16_000, return_tensors="pt", padding=True)
        #     #convert audio from base64 to numpy array of doubles
        #                 with torch.no_grad():
        #                         logits = self.model(**inputs).logits

        #                 predicted_class_ids = torch.argmax(logits).item()
        #                 predicted_label = self.model.config.id2label[predicted_class_ids]
        #                 results = predicted_label

        #         return {
        #     'text': results, 
        #     'done': True
        # }

        # def test(self, **kwargs):
        #         audio_filename = "/tmp/temp.ogg"
        #         return self.wav2vec_classify(audio_filename)

        # def test2(self, **kwargs):
        #         audio_filename = "/tmp/base64ogg.txt"
        #         with open(audio_filename, "rb") as audio_file:
        #                 audio = audio_file.read()
        #         return self.wav2vec_classify(audio)