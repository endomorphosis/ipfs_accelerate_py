import os
import tempfile
import io
import json
import time
import asyncio
import requests
import gc
from pydub import AudioSegment
# from datasets import Dataset, Audio

def load_audio(audio_file):
    import soundfile as sf
    import numpy as np

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

def load_audio_16khz(audio_file):
    import librosa
    audio_data, samplerate = load_audio(audio_file)
    if samplerate != 16000:
        ## convert to 16khz
        audio_data = librosa.resample(y=audio_data, orig_sr=samplerate, target_sr=16000)
    return audio_data, 16000

def load_audio_tensor(audio_file):
    from openvino import ov, Tensor
    import soundfile as sf
    import numpy as np
    if isinstance(audio_file, str) and (audio_file.startswith("http") or audio_file.startswith("https")):
        response = requests.get(audio_file)
        audio_data, samplerate = sf.read(io.BytesIO(response.content))
    else:
        audio_data, samplerate = sf.read(audio_file)
    
    # Ensure audio is mono and convert to float32
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    audio_data = audio_data.astype(np.float32)
    
    return Tensor(audio_data.reshape(1, -1))

class hf_wav2vec2:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata    
        self.create_openvino_audio_embedding_endpoint_handler = self.create_openvino_audio_embedding_endpoint_handler
        self.create_cuda_audio_embedding_endpoint_handler = self.create_cuda_audio_embedding_endpoint_handler
        self.init_qualcomm = self.init_qualcomm
        self.init_cpu = self.init_cpu
        self.init_cuda = self.init_cuda
        self.init_openvino = self.init_openvino
        self.init = self.init
        self.__test__ = self.__test__
        return None

    def init(self):
        import torch
        import librosa
        import numpy as np
        from torch import no_grad
        import torch
        from transformers import pipeline
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        from transformers import AutoModelForAudioClassification
        from transformers import AutoFeatureExtractor
        from transformers import AutoTokenizer, AutoConfig
        from transformers import AutoModel
        from transformers import AutoProcessor
        import soundfile as sf
        return None
    
    def init_qualcomm(self, model, device, qualcomm_label):
        return None

    def __test__(self, endpoint_model, endpoint_handler, endpoint_label, tokenizer):
        audio_1 = "https://calamitymod.wiki.gg/images/2/29/Bees3.wav"
        timestamp1 = time.time()
        try:
            test_batch = endpoint_handler(audio_1)
            print(test_batch)
            print("hf_wav2vec2 test passed")
        except Exception as e:
            print(e)
            print("hf_wav2vec2 test failed")
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
            with self.torch.no_grad():
                if "cuda" in dir(self.torch):
                    self.torch.cuda.empty_cache()
        print("hf_wav2vec test")
        return None
    
    def init_cpu(self, model, device, cpu_label):
        self.init()        
        return None
    
    def init_cuda(self, model, device, cuda_label):
        self.init()
        config = self.transformers.AutoConfig.from_pretrained(model, trust_remote_code=True)    
        tokenizer = self.transformers.AutoTokenizer.from_pretrained(model)
        # processor = CLIPProcessor.from_pretrained(model, trust_remote_code=True)
        endpoint = None
        try:
            # endpoint = CLIPModel.from_pretrained(model, torch_dtype=torch.float16, trust_remote_code=True).to(device)
            pass
        except Exception as e:
            print(e)
            pass
        endpoint_handler = self.create_audio_embedding_endpoint_handler(endpoint, tokenizer, model, cuda_label)
        self.torch.cuda.empty_cache()
        # batch_size = await self.max_batch_size(endpoint_model, cuda_label)
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), 0    

    def init_openvino(self, model=None , model_type=None, device=None, openvino_label=None, get_optimum_openvino_model=None, get_openvino_model=None, get_openvino_pipeline_type=None, openvino_cli_convert=None ):
        self.init()
        import openvino as ov
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
            tokenizer =  self.transformers.Wav2Vec2Processor.from_pretrained(
                model
            )
        except Exception as e:
            print(e)
            try:
                tokenizer =  self.transformers.Wav2Vec2Processor.from_pretrained(
                    model_src_path
                )
            except Exception as e:
                print(e)
                pass
        
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
        endpoint_handler = self.create_openvino_audio_embedding_endpoint_handler(model, tokenizer, openvino_label)
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
        def handler(x, tokenizer=tokenizer, endpoint_model=endpoint_model, openvino_label=openvino_label, endpoint=None):
            results = []
            if x is not None:            
                if type(x) == str:
                    audio_data, audio_sampling_rate = load_audio_16khz(x)                    
                    preprocessed_signal = tokenizer(
                        audio_data,
                        return_tensors="pt",
                        padding="longest",
                        sampling_rate=audio_sampling_rate,
                    )
                    audio_inputs = preprocessed_signal.input_values
                    MAX_SEQ_LENGTH = 30480
                    if audio_inputs.shape[1] > MAX_SEQ_LENGTH:
                        audio_inputs = audio_inputs[:, :MAX_SEQ_LENGTH]
                    image_features = endpoint_model({'input_values': audio_inputs})
                    image_embeddings = list(image_features.values())[0]
                    image_embeddings = self.torch.tensor(image_embeddings)
                    image_embeddings = self.torch.mean(image_embeddings, dim=(1,))
                    results.append(image_embeddings)
                elif type(x) == list:
                    inputs = tokenizer(images=[load_audio_16khz(image) for image in x], return_tensors='pt')
                    image_features = endpoint_model({'input_values': audio_inputs})
                    image_embeddings = list(image_features.values())[0]
                    image_embeddings = self.torch.tensor(image_embeddings)
                    image_embeddings = self.torch.mean(image_embeddings, dim=1)
                    results.append(image_embeddings)
                pass            

                if results is not None:                                        
                    if x is not None:
                        return {
                            'embedding': image_embeddings[0]
                        }            
            return None
        return handler    
    
    def openvino_skill_convert(self, model_name, model_dst_path, task, weight_format, hfmodel=None, hfprocessor=None):
        if hfmodel is None:
            hfmodel = self.transformers.AutoModel.from_pretrained(model_name, torch_dtype=self.torch.float16)
    
        if hfprocessor is None:
            hfprocessor = self.transformers.AutoProcessor.from_pretrained(model_name)

        if hfprocessor is not None:
            text = "Replace me by any text you'd like."
            audio_url = "https://calamitymod.wiki.gg/images/2/29/Bees3.wav"
            audio_data, audio_sampling_rate = audio = load_audio_16khz(audio_url)
            preprocessed_signal = None
            preprocessed_signal = hfprocessor(
                audio_data,
                return_tensors="pt",
                padding="longest",
                sampling_rate=audio_sampling_rate,
            )
            audio_inputs = preprocessed_signal.input_values
            MAX_SEQ_LENGTH = 30480
            hfmodel.config.torchscript = True
            ov_model = self.ov.convert_model(hfmodel, example_input=self.torch.zeros([1, MAX_SEQ_LENGTH], dtype=self.torch.float))
            if not os.path.exists(model_dst_path):
                os.mkdir(model_dst_path)
            self.ov.save_model(ov_model, os.path.join(model_dst_path, model_name.replace("/", "--") + ".xml"))
            ov_model = self.ov.compile_model(ov_model)
            hfmodel = None
        return ov_model
    
    
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