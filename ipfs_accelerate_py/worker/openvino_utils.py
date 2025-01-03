import subprocess
from transformers import AutoConfig
import os
import openvino as ov
import openvino_genai as ov_genai
from transformers import AutoTokenizer, AutoModel, AutoProcessor
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoModelForQuestionAnswering, AutoModelForAudioClassification, AutoModelForImageClassification, AutoModelForMaskedLM, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForSpeechSeq2Seq, AutoModelForVision2Seq
from transformers import AutoModelForImageTextToText
from PIL import Image
import requests
from io import BytesIO
import numpy as np
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

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    image_data = np.array(image.getdata()).reshape(1, image.size[1], image.size[0], 3).astype(np.byte)
    return image

def load_image_tensor(image_file):
    if isinstance(image_file, str) and (image_file.startswith("http") or image_file.startswith("https")):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    image_data = np.array(image.getdata()).reshape(1, image.size[1], image.size[0], 3).astype(np.byte)
    return ov.Tensor(image_data)

class openvino_utils:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata
        self.openvino_cli_convert = self.openvino_cli_convert
        self.get_openvino_pipeline_type = self.get_openvino_pipeline_type
        self.get_optimum_openvino_model = self.get_optimum_openvino_model
        self.get_openvino_genai_pipeline = self.get_openvino_genai_pipeline
        self.get_optimum_openvino_model = self.get_optimum_openvino_model
        self.get_model_type = self.get_model_type
        self.init = self.init
        self.get_openvino_model = self.get_openvino_model
        return None
    
    def init(self):
        return None
            
    def get_openvino_model(self, model_name, model_type=None, device_name=None ):
        architecture = None
        config = None
        hfmodel = None
        hftokenizer = None
        import openvino as ov                                
        core = ov.Core()
        openvino_devices = core.available_devices
        if device_name is None:
            device_index = 0
            print("warning: device name not provided, using default device: "+ openvino_devices[0])
        elif type(device_name) is str and ":" in device_name:
            device_index = int(device_name.split(":")[-1])
        else:
            print("Device name not in correct format, recieved: " + device_name) 
            raise ValueError("Device name not in correct format, recieved: " + device_name)
        device = openvino_devices[device_index]
        model_type = self.get_model_type(model_name)
        model_task = self.get_openvino_pipeline_type(model_name, model_type)
        homedir = os.path.expanduser("~")
        model_name_convert = model_name.replace("/", "--")
        huggingface_cache = os.path.join(homedir, ".cache/huggingface")
        huggingface_cache_models = os.path.join(huggingface_cache, "hub")
        huggingface_cache_models_files = os.listdir(huggingface_cache_models)
        huggingface_cache_models_files_dirs = [os.path.join(huggingface_cache_models, file) for file in huggingface_cache_models_files if os.path.isdir(os.path.join(huggingface_cache_models, file))]
        huggingface_cache_models_files_dirs_models = [ x for x in huggingface_cache_models_files_dirs if "model" in x ]
        huggingface_cache_models_files_dirs_models_model_name = [ x for x in huggingface_cache_models_files_dirs_models if model_name_convert in x ]
        model_src_path = os.path.join(huggingface_cache_models, huggingface_cache_models_files_dirs_models_model_name[0])
        model_dst_path = os.path.join(model_src_path, "openvino")
        openvino_index = int(device_name.split(":")[1])
        weight_format = ""
        if openvino_index is not None:
            if openvino_index == 0:
                weight_format = "int8" ## CPU
            if openvino_index == 1:
                weight_format = "int4" ## gpu
            if openvino_index == 2:
                weight_format = "int4" ## npu
        model_dst_path = model_dst_path+"_"+weight_format
        
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            model_type = config.__class__.model_type
        except Exception as e:
            config = None

        ov_model = None
        hfmodel = None
        hftokenizer = None
        hfprocessor = None

        vlm_model_types = ["llava", "llava_next"]
        clip_model_types = ["clip"]
        clap_model_types = ["clap"]
        if os.path.exists(model_src_path) and not os.path.exists(model_dst_path):
            
            try:
                hftokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            except Exception as e:
                print(e)
                hftokenizer = None
            try:
                hfprocessor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            except Exception as e:  
                print(e)
                hfprocessor = None
            try:
                hfmodel = AutoModel.from_pretrained(model_name,  trust_remote_code=True).to('cpu')
            except Exception as e:
                print(e)
                hfmodel = None
            
            if hfmodel is not None and "config" in list(dir(hfmodel)):
                if model_type in clip_model_types:
                    if hfprocessor is not None:
                        text = "Replace me by any text you'd like."
                        image_url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
                        image = load_image(image_url)
                        processed_data = hfprocessor(
                            text = "Replace me by any text you'd like.",
                            images = [image],
                            return_tensors="pt", 
                            padding=True
                            )
                        results = hfmodel(**processed_data)
                        hfmodel.config.torchscript = True
                        ov_model = ov.convert_model(hfmodel ,  example_input=dict(processed_data))
                        if not os.path.exists(model_dst_path):
                            os.mkdir(model_dst_path)
                        ov.save_model(ov_model, os.path.join(model_dst_path, model_name.replace("/", "--") + ".xml"))
                        ov_model = ov.compile_model(ov_model)
                        hfmodel = None
                    elif hftokenizer is not None:
                        text = "Replace me by any text you'd like."
                        encoded_input = hftokenizer(text, return_tensors='pt').to('cpu')
                        input_dict = {k: v for k, v in encoded_input.items()}
                        ov_model = ov.convert_model(hfmodel.to('cpu'), example_input=input_dict)
                        ov.save_model(ov_model, model_dst_path)
                        ov_model = ov.compile_model(ov_model)
                        hfmodel = None
                if model_type in clap_model_types:
                    if hfprocessor is not None:
                        text = "Replace me by any text you'd like."
                        audio_url = "https://calamitymod.wiki.gg/images/2/29/Bees3.wav"
                        audio = load_audio(audio_url)
                        processed_data = hfprocessor(
                            audios =[audio],
                            return_tensors="pt", 
                            padding=True
                            )
                        results = hfmodel(**processed_data)
                        hfmodel.config.torchscript = True
                        ov_model = ov.convert_model(hfmodel ,  example_input=dict(processed_data))
                        if not os.path.exists(model_dst_path):
                            os.mkdir(model_dst_path)
                        ov.save_model(ov_model, os.path.join(model_dst_path, model_name.replace("/", "--") + ".xml"))
                        ov_model = ov.compile_model(ov_model)
                        hfmodel = None
                    elif hftokenizer is not None:
                        text = "Replace me by any text you'd like."
                        encoded_input = hftokenizer(text, return_tensors='pt').to('cpu')
                        input_dict = {k: v for k, v in encoded_input.items()}
                        ov_model = ov.convert_model(hfmodel.to('cpu'), example_input=input_dict)
                        ov.save_model(ov_model, model_dst_path)
                        ov_model = ov.compile_model(ov_model)
                        hfmodel = None
            if ov_model == None:
                try:
                    # self.openvino_cli_convert(model_name, model_dst_path=model_dst_path, task=model_task, weight_format="int8",  ratio="1.0", group_size=128, sym=True )
                    # self.openvino_cli_convert(model_name, model_dst_path=model_dst_path, task=model_task, weight_format="int4", ratio="1.0", group_size=128, sym=True )

                    # ov_model = ov.load_model(model_dst_path)
                    # ov_model = ov.compile_model(ov_model)
                    pass
                except Exception as e:
                    print(e)
                    pass
            
            if hfmodel is not None and "config" in list(dir(hfmodel)):
                config = hfmodel.config
            else:
                try:
                    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                except Exception as e:
                    config = None
            if config is not None and "architectures" in dir(config):        
                architecture = config.architectures
            if config is not None and "model_type" in dir(config):
                model_type = config.model_type

        if os.path.exists(model_dst_path):
            if model_type == 'clip' and model_task == 'feature-extraction':
                ov_model = core.read_model(os.path.join(model_dst_path, model_name.replace("/", "--") + ".xml"))
                ov_model = core.compile_model(ov_model)
            elif model_type == 'clap' and model_task == 'audio-classification':
                ov_model = core.read_model(os.path.join(model_dst_path, model_name.replace("/", "--") + ".xml"))
                ov_model = core.compile_model(ov_model)
        return ov_model

    def get_openvino_genai_pipeline(self, model_name, model_type=None, device_name=None):
        architecture = None
        config = None
        hfmodel = None
        hftokenizer = None
        import openvino as ov                                
        core = ov.Core()
        import openvino_genai as ov_genai
        openvino_devices = core.available_devices
        device_index = int(device_name.split(":")[-1])
        device = openvino_devices[device_index]
        model_type = self.get_model_type(model_name)
        model_task = self.get_openvino_pipeline_type(model_name, model_type)
        homedir = os.path.expanduser("~")
        model_name_convert = model_name.replace("/", "--")
        huggingface_cache = os.path.join(homedir, ".cache/huggingface")
        huggingface_cache_models = os.path.join(huggingface_cache, "hub")
        huggingface_cache_models_files = os.listdir(huggingface_cache_models)
        huggingface_cache_models_files_dirs = [os.path.join(huggingface_cache_models, file) for file in huggingface_cache_models_files if os.path.isdir(os.path.join(huggingface_cache_models, file))]
        huggingface_cache_models_files_dirs_models = [ x for x in huggingface_cache_models_files_dirs if "model" in x ]
        huggingface_cache_models_files_dirs_models_model_name = [ x for x in huggingface_cache_models_files_dirs_models if model_name_convert in x ]
        model_src_path = os.path.join(huggingface_cache_models, huggingface_cache_models_files_dirs_models_model_name[0])
        model_dst_path = os.path.join(model_src_path, "openvino")
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            model_type = config.__class__.model_type
        except Exception as e:
            config = None

        ov_model = None
        hfmodel = None
        hftokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        vlm_model_types = ["llava", "llava_next"]

        if os.path.exists(model_src_path) and not os.path.exists(model_dst_path):
            try:
                if model_type not in vlm_model_types:
                    hfmodel = AutoModel.from_pretrained(model_name,  trust_remote_code=True)
                else:
                    hfmodel = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            
                text = "Replace me by any text you'd like."
                encoded_input = hftokenizer(text, return_tensors='pt')
                ov_model = ov.convert_model(hfmodel, example_input={**encoded_input})                        
                ov.save_model(ov_model, model_dst_path)
                ov_model = ov.compile_model(ov_model)
                hfmodel = None
                del hftokenizer

            except Exception as e:
                if hfmodel is not None:
                    hfmodel = None
                if hftokenizer is not None:
                    del hftokenizer
                print(e)
                
            if ov_model == None:
                try:
                    # self.openvino_cli_convert(model_name, model_dst_path=model_dst_path, task=model_task, weight_format="int8",  ratio="1.0", group_size=128, sym=True )
                    # self.openvino_cli_convert(model_name, model_dst_path=model_dst_path, task=model_task, weight_format="int4", ratio="1.0", group_size=128, sym=True )

                    # ov_model = ov.load_model(model_dst_path)
                    # ov_model = ov.compile_model(ov_model)
                    pass
                except Exception as e:
                    print(e)
                    pass
            
            if hfmodel is not None and "config" in list(dir(hfmodel)):
                config = hfmodel.config
            else:
                try:
                    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                except Exception as e:
                    config = None
            if config is not None and "architectures" in dir(config):        
                architecture = config.architectures
            if config is not None and "model_type" in dir(config):
                model_type = config.model_type

        model_mapping_list = ['fill-mask', 'image-classification', 'image-segmentation', 'feature-extraction', 'token-classification', 'audio-xvector', 'audio-classification', 'zero-shot-image-classification', 'text2text-generation', 'depth-estimation', 'text-to-audio', 'semantic-segmentation', 'masked-im', 'image-to-text', 'zero-shot-object-detection','mask-generation', 'sentence-similarity', 'image-to-image', 'object-detection', 'multiple-choice', 'automatic-speech-recognition', 'text-classification', 'audio-frame-classification', 'text-generation', 'question-answering']
        model_dir = "C:/Users/devcloud/.cache/huggingface/hub/models--llava-hf--llava-v1.6-mistral-7b-hf/openvino"
        model_dst_path = model_dir
        if os.path.exists(model_dst_path):
            if model_task is not None and model_task == "image-text-to-text":
                ov_model = ov_genai.VLMPipeline(model_dst_path, device=device)
            elif model_task is not None and model_task == "text-generation-with-past":
                ov_model = ov_genai.LLMPipeline(model_dst_path, device=device)
            elif model_task is not None and model_task == "automatic-speech-recognition":
                whisper_config = ov_genai.WhisperGenerationConfig
                ov_model = ov_genai.WhisperPipeline(model_dst_path, device=device)
            elif model_task is not None and model_task == "image-to-image":
                ov_model = ov_genai.Text2ImagePipeline(model_dst_path, device=device)
            elif model_task is not None and model_task == "'imge-to-text":
                ov_model = ov_genai.CLIPTextModelWithProjection(model_dst_path, device=device)
            else:
                raise ValueError("Loading Model: " + model_name + " Task not supported: " + model_task + " Supported tasks: " + str(model_mapping_list))

        return ov_model
    
    def get_optimum_openvino_model(self, model_name, model_type=None, openvino_label=None):
        homedir = os.path.expanduser("~")
        model_name_convert = model_name.replace("/", "--")
        huggingface_cache = os.path.join(homedir, ".cache/huggingface")
        huggingface_cache_models = os.path.join(huggingface_cache, "hub")
        huggingface_cache_models_files = os.listdir(huggingface_cache_models)
        huggingface_cache_models_files_dirs = [os.path.join(huggingface_cache_models, file) for file in huggingface_cache_models_files if os.path.isdir(os.path.join(huggingface_cache_models, file))]
        huggingface_cache_models_files_dirs_models = [ x for x in huggingface_cache_models_files_dirs if "model" in x ]
        huggingface_cache_models_files_dirs_models_model_name = [ x for x in huggingface_cache_models_files_dirs_models if model_name_convert in x ]
        model_src_path = os.path.join(huggingface_cache_models, huggingface_cache_models_files_dirs_models_model_name[0])
        model_dst_path = os.path.join(model_src_path, "openvino")
        if model_type is None:
            config = AutoConfig.from_pretrained(model_name)
            model_type = config.__class__.model_type
        model_mapping_list = ["text-classification", "token-classification", "question-answering", "audio-classification", "image-classification", "feature-extraction", "fill-mask", "text-generation-with-past", "text2text-generation-with-past", "automatic-speech-recognition", "image-to-text", "bert", "llava", "image-text-to-text"]
        if model_type not in model_mapping_list:
            model_type = self.get_openvino_pipeline_type(model_name, model_type)
        # self.openvino_cli_convert(model_name, model_dst_path=model_dst_path, task=model_type, weight_format="int8", ratio="1.0", group_size=128, sym=True )
        if model_type == "bert":
            model_type = "feature-extraction"
            from optimum.intel import OVModelForFeatureExtraction
            try:
                results = OVModelForFeatureExtraction.from_pretrained(model_name, compile=False)
            except Exception as e:
                results = OVModelForFeatureExtraction.from_pretrained(model_dst_path, compile=False)
        elif model_type == "text-classification":
            from optimum.intel import OVModelForSequenceClassification
            try:
                results = OVModelForSequenceClassification.from_pretrained(model_name, compile=False)
            except Exception as e:
                results = OVModelForSequenceClassification.from_pretrained(model_dst_path, compile=False)
        elif model_type == "token-classification":
            from optimum.intel import OVModelForTokenClassification
            try:
                results = OVModelForTokenClassification.from_pretrained(model_name, compile=False)
            except Exception as e:
                results = OVModelForTokenClassification.from_pretrained(model_dst_path, compile=False)
        elif model_type == "question-answering":
            from optimum.intel import OVModelForQuestionAnswering
            try:
                results = OVModelForQuestionAnswering.from_pretrained(model_name, compile=False)
            except Exception as e:
                results = OVModelForQuestionAnswering.from_pretrained(model_dst_path, compile=False)
        elif model_type == "audio-classification":
            from optimum.intel import OVModelForAudioClassification
            try:
                results = OVModelForAudioClassification.from_pretrained(model_name,  compile=False)
            except Exception as e:
                results = OVModelForAudioClassification.from_pretrained(model_dst_path, compile=False)
        elif model_type == "image-classification":
            from optimum.intel import OVModelForImageClassification
            try:
                results = OVModelForImageClassification.from_pretrained(model_name, compile=False) 
            except Exception as e:
                results = OVModelForImageClassification.from_pretrained(model_dst_path, compile=False)
        elif model_type == "feature-extraction":
            from optimum.intel import OVModelForFeatureExtraction
            try:
                results = OVModelForFeatureExtraction.from_pretrained(model_name, compile=False)
            except Exception as e:
                results = OVModelForFeatureExtraction.from_pretrained(model_dst_path, compile=False)
        elif model_type == "fill-mask":
            from optimum.intel import OVModelForMaskedLM
            try:
                results = OVModelForMaskedLM.from_pretrained(model_name, compile=False)
            except Exception as e:
                results = OVModelForMaskedLM.from_pretrained(model_dst_path, compile=False)
        elif model_type == "text-generation-with-past":
            from optimum.intel import OVModelForCausalLM
            try:
                results = OVModelForCausalLM.from_pretrained(model_name, compile=False)
            except Exception as e:
                results = OVModelForCausalLM.from_pretrained(model_dst_path, compile=False)
        elif model_type == "text2text-generation-with-past":
            from optimum.intel import OVModelForSeq2SeqLM
            try:
                results = OVModelForSeq2SeqLM.from_pretrained(model_name, compile=False)
            except Exception as e:
                results = OVModelForSeq2SeqLM.from_pretrained(model_dst_path, compile=False)
        elif model_type == "automatic-speech-recognition":
            from optimum.intel import OVModelForSpeechSeq2Seq
            try:
                results = OVModelForSpeechSeq2Seq.from_pretrained(model_name, compile=False)
            except Exception as e:
                results = OVModelForSpeechSeq2Seq.from_pretrained(model_dst_path, compile=False)
        # elif model_type == "image-text-to-text":
        #     from optimum.intel import OVModelForVision2Seq
        #     results = OVModelForVision2Seq.from_pretrained(model_name, compile=False)
        elif model_type == "image-text-to-text":
            from optimum.intel import OVModelForVisualCausalLM
            try:
                results = OVModelForVisualCausalLM.from_pretrained(model_name, compile=False)
            except Exception as e:
                results = OVModelForVisualCausalLM.from_pretrained(model_name, compile=False)
        else:
            return None
        results.compile()
        return results
    
    
    def get_model_type(self, model_name, model_type=None):
        if model_type is None:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            model_type = config.__class__.model_type
        return model_type
    
    def openvino_convert(self, 
        model_name, 
        model_dst_path, 
        task=None, 
        framework=None, 
        trust_remote_code=False, 
        weight_format=None, 
        library=None, 
        cache_dir=None, 
        pad_token_id=None, 
        ratio=None, 
        sym=False, 
        group_size=None, 
        backup_precision=None, 
        dataset=None, 
        all_layers=False, 
        awq=False,
        scale_estimation=False,
        gptq=False,
        lora_correction=False,
        sensitivity_metric=None,
        num_samples=None,
        disable_stateful=False,
        disable_convert_tokenizer=False
        ):



        return None
    
    def openvino_cli_convert(
        self,
        model_name,
        model_dst_path,
        task=None,
        framework=None,
        trust_remote_code=False,
        weight_format=None,
        library=None,
        cache_dir=None,
        pad_token_id=None,
        ratio=None,
        sym=False,
        group_size=None,
        backup_precision=None,
        dataset=None,
        all_layers=False,
        awq=False,
        scale_estimation=False,
        gptq=False,
        lora_correction=False,
        sensitivity_metric=None,
        num_samples=None,
        disable_stateful=False,
        disable_convert_tokenizer=False,
    ):
        command = ['optimum-cli', 'export', 'openvino', '-m', model_name]
        tasks_list = ['fill-mask', 'image-classification', 'image-segmentation', 'feature-extraction', 'token-classification', 'audio-xvector', 'audio-classification', 'zero-shot-image-classification', 'text2text-generation', 'depth-estimation', 'text-to-audio', 'semantic-segmentation', 'masked-im', 'image-to-text', 'zero-shot-object-detection','mask-generation', 'sentence-similarity', 'image-to-image', 'object-detection', 'multiple-choice', 'automatic-speech-recognition', 'text-classification', 'audio-frame-classification', 'text-generation', 'question-answering']
        if task is not None:
            command.extend(['--task', task])
        if task is None:
            model_type = self.get_model_type(model_name)
            task = self.get_openvino_pipeline_type(model_name, model_type)
            if task not in tasks_list and task is not None:
                raise ValueError("Task not supported: " + task)
            elif task is not None:
                command.extend(['--task', task])
        if framework is not None:
            command.extend(['--framework', framework])
        if trust_remote_code:
            command.append('--trust-remote-code')
        if weight_format is not None:
            command.extend(['--weight-format', weight_format])
        if library is not None:
            command.extend(['--library', library])
        if cache_dir is not None:
            command.extend(['--cache_dir', cache_dir])
        if pad_token_id is not None:
            command.extend(['--pad-token-id', str(pad_token_id)])
        if ratio is not None:
            command.extend(['--ratio', str(ratio)])
        if sym:
            command.append('--sym')
        if group_size is not None:
            command.extend(['--group-size', str(group_size)])
        if backup_precision is not None:
            command.extend(['--backup-precision', backup_precision])
        if dataset is not None:
            command.extend(['--dataset', dataset])
        if all_layers:
            command.append('--all-layers')
        if awq:
            command.append('--awq')
        if scale_estimation:
            command.append('--scale-estimation')
        if gptq:
            command.append('--gptq')
        if lora_correction:
            command.append('--lora-correction')
        if sensitivity_metric is not None:
            command.extend(['--sensitivity-metric', sensitivity_metric])
        if num_samples is not None:
            command.extend(['--num-samples', str(num_samples)])
        if disable_stateful:
            command.append('--disable-stateful')
        if disable_convert_tokenizer:
            command.append('--disable-convert-tokenizer')
        
        # Add the output directory
        command.append(model_dst_path)
        parsed_cmd = ' '.join(command)
        # Execute the command
        convert_model = subprocess.check_output(parsed_cmd, shell=True)
        convert_model = convert_model.decode('utf-8')
        return convert_model
    
    
    def get_openvino_pipeline_type(self, model_name, model_type=None):
        model_mapping_list = ["text-classification", "token-classification", "question-answering", "audio-classification", "image-classification", "feature-extraction", "fill-mask", "text-generation-with-past", "text2text-generation-with-past", "automatic-speech-recognition", "image-to-text"]
        model_mapping_list = ['image-text-to-text', 'fill-mask', 'image-classification', 'image-segmentation', 'feature-extraction', 'token-classification', 'audio-xvector', 'audio-classification', 'zero-shot-image-classification', 'text2text-generation', 'depth-estimation', 'text-to-audio', 'semantic-segmentation', 'masked-im', 'image-to-text', 'zero-shot-object-detection','mask-generation', 'sentence-similarity', 'image-to-image', 'object-detection', 'multiple-choice', 'automatic-speech-recognition', 'text-classification', 'audio-frame-classification', 'text-generation', 'question-answering']
        return_model_type = None
        config_model_type = None
        if model_type is not None:
            try:
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                config_model_type = config.__class__.model_type
            except Exception as e:
                config_model_type = None
            if config_model_type is None:
                config_model_type = config_model_type if config_model_type is not None else model_type
                if config_model_type == "bert":
                    return_model_type = "feature-extraction"
                elif config_model_type == "llava":
                    return_model_type = "image-text-to-text"
                elif config_model_type == "llava_next":
                    return_model_type = "image-text-to-text"
                elif config_model_type == "qwen2":
                    return_model_type = "text-generation-with-past"
                elif config_model_type == "llama":
                    return_model_type = "text-generation-with-past"
                elif config_model_type == "clip":
                    return_model_type = "feature-extraction"
                elif config_model_type == "clap":
                    return_model_type = "feature-extraction"
                pass
            elif config_model_type not in model_mapping_list and model_type not in model_mapping_list:
                config_model_type = config_model_type if config_model_type is not None else model_type
                if config_model_type == "bert":
                    return_model_type = "feature-extraction"
                elif config_model_type == "llava":
                    return_model_type = "image-text-to-text"
                elif config_model_type == "llava_next":
                    return_model_type = "image-text-to-text"
                elif config_model_type == "qwen2":
                    return_model_type = "text-generation-with-past"
                elif config_model_type == "llama":
                    return_model_type = "text-generation-with-past"
                elif config_model_type == "clip":
                    return_model_type = "feature-extraction"
                elif config_model_type == "clap":
                    return_model_type = "feature-extraction"
                pass            
            elif config_model_type in model_mapping_list:
                return_model_type = config_model_type         
        elif model_type is None:
            try:
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                config_model_type = config.__class__.model_type
            except Exception as e:
                config_model_type = None
            if config_model_type is None:
                config_model_type = config_model_type if config_model_type is not None else model_type
                if config_model_type == "bert":
                    return_model_type = "feature-extraction"
                elif config_model_type == "llava":
                    return_model_type = "image-text-to-text"
                elif config_model_type == "llava_next":
                    return_model_type = "image-text-to-text"
                elif config_model_type == "qwen2":
                    return_model_type = "text-generation-with-past"
                elif config_model_type == "llama":
                    return_model_type = "text-generation-with-past"
                elif config_model_type == "clip":
                    return_model_type = "feature-extraction"
                elif config_model_type == "clap":
                    return_model_type = "feature-extraction"
                pass            
            elif config_model_type not in model_mapping_list and model_type not in model_mapping_list:
                config_model_type = config_model_type if config_model_type is not None else model_type
                if config_model_type == "bert":
                    return_model_type = "feature-extraction"
                elif config_model_type == "llava":
                    return_model_type = "image-text-to-text"            
                elif config_model_type == "llava_next":
                    return_model_type = "image-text-to-text"
                elif config_model_type == "qwen2":
                    return_model_type = "text-generation-with-past"
                elif config_model_type == "llama":
                    return_model_type = "text-generation-with-past"
                elif config_model_type == "clip":
                    return_model_type = "feature-extraction"
                elif config_model_type == "clap":
                    return_model_type = "feature-extraction"
            elif config_model_type in model_mapping_list:
                return_model_type = config_model_type   

        return return_model_type
