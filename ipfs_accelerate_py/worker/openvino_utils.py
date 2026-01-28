import subprocess
import os

try:
    from ...common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
except ImportError:
    try:
        from ..common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
    except ImportError:
        try:
            from common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
        except ImportError:
            HAVE_STORAGE_WRAPPER = False

if HAVE_STORAGE_WRAPPER:
    try:
        _storage = get_storage_wrapper(auto_detect_ci=True)
    except Exception:
        _storage = None
else:
    _storage = None

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
        self.init()
        self.get_openvino_model = self.get_openvino_model
        return None
    
    def init(self):
        
        if "torch" not in list(self.resources.keys()):
            import torch
            self.torch = torch
        else:
            self.torch = self.resources["torch"]

        if "transformers" not in list(self.resources.keys()):
            import transformers
            self.transformers = transformers
        else:
            self.transformers = self.resources["transformers"]
            
        if "numpy" not in list(self.resources.keys()):
            import numpy as np
            self.np = np
        else:
            self.np = self.resources["numpy"]

        if "ov" not in list(self.resources.keys()):
            import openvino as ov
            self.ov = ov
        else:
            self.ov = self.resources["ov"]\
        
        if "ov_genai" not in list(self.resources.keys()):
            import openvino_genai as ov_genai
            self.ov_genai = ov_genai
        else:
            self.ov_genai = self.resources["ov_genai"]
        return None
            
    def get_openvino_model(self, model_name, model_type=None, device_name=None ):
        architecture = None
        config = None
        hfmodel = None
        hftokenizer = None
        import openvino as ov                                
        core = self.ov.Core()
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
        model_dst_path = os.path.abspath(model_dst_path)
        try:
            config = self.transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            model_type = config.__class__.model_type
        except Exception as e:
            config = None

        ov_model = None
        hfmodel = None
        hftokenizer = None
        hfprocessor = None
        genai_model_types = ["llava", "llava_next"]
        optimum_model_types = []
        openvino_model_types = ["llava_next", "llava", "wav2vec2", "t5", "whisper", "xclip", "clip", "clap", "bert"]
        method_name = "hf_" + model_type
        if os.path.exists(model_src_path) and not os.path.exists(model_dst_path):            
            try:
                hftokenizer = self.transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            except Exception as e:
                print(e)
                hftokenizer = None
            try:
                hfprocessor = self.transformers.AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            except Exception as e:  
                print(e)
                hfprocessor = None
            try:
                hfmodel = self.transformers.AutoModel.from_pretrained(model_name,  trust_remote_code=True).to('cpu')
            except Exception as e:
                print(e)
                hfmodel = None
            ov_model = None
            if hfmodel is not None and "config" in list(dir(hfmodel)):
                if ov_model is None and  model_type in genai_model_types:
                    if hfprocessor is not None:
                        module = __import__('worker.skillset', fromlist=[method_name])
                        this_method = getattr(module, method_name)
                        this_hf = this_method(self.resources, self.metadata)
                        try:
                            convert = this_hf.openvino_skill_convert(model_name, model_dst_path, model_task, weight_format)
                        except Exception as e:
                            print(e)
                            try:
                                convert = self.openvino_cli_convert(model_name, model_dst_path=model_dst_path, task=model_task, weight_format=weight_format,  ratio="1.0", group_size=128, sym=True )
                            except Exception as e:
                                print(e)
                                convert = None
                                pass
                        ov_model = core.read_model(model_name, model_dst_path)
                        ov_model = core.compile_model(ov_model)
                        hf_model = None
                if ov_model is None and model_type in openvino_model_types:
                    if hfprocessor is not None:
                        module = __import__('worker.skillset', fromlist=[method_name])
                        this_method = getattr(module, method_name)
                        this_hf = this_method(self.resources, self.metadata)
                        try:
                            convert = this_hf.openvino_skill_convert(model_name, model_dst_path, model_task,weight_format)
                        except Exception as e:
                            print(e)
                            try:
                                convert = self.openvino_cli_convert(model_name, model_dst_path=model_dst_path, task=model_task, weight_format=weight_format,  ratio="1.0", group_size=128, sym=True )
                            except Exception as e:
                                print(e)
                                convert = None
                                pass
                        ov_model = core.read_model(model_name, model_dst_path)
                        ov_model = core.compile_model(ov_model)
                        hf_model = None
                if  ov_model is None and model_type in optimum_model_types:
                    if hfprocessor is not None:
                        module = __import__('worker.skillset', fromlist=[method_name])
                        this_method = getattr(module, method_name)
                        this_hf = this_method(self.resources, self.metadata)
                        try:
                            convert = this_hf.openvino_skill_convert(model_name, model_dst_path, model_task,weight_format)
                        except Exception as e:
                            print(e)
                            try:
                                convert = self.openvino_cli_convert(model_name, model_dst_path=model_dst_path, task=model_task, weight_format=weight_format,  ratio="1.0", group_size=128, sym=True )
                            except Exception as e:
                                print(e)
                                convert = None
                                pass
                        ov_model = core.read_model(model_name, model_dst_path)
                        ov_model = core.compile_model(ov_model)
                        hf_model = None
            
            if hfmodel is not None and "config" in list(dir(hfmodel)):
                config = hfmodel.config
            else:
                try:
                    config = self.transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                except Exception as e:
                    config = None
            if config is not None and "architectures" in dir(config):        
                architecture = config.architectures
            if config is not None and "model_type" in dir(config):
                model_type = config.model_type

        if os.path.exists(model_dst_path):
            if model_type == 't5' and model_task == 'text2text-generation-with-past':
                ov_model = core.read_model(os.path.join(model_dst_path, 'openvino_decoder_with_past_model.xml'))
                ov_model = core.compile_model(ov_model) 
            elif model_type == "whisper" and model_task == "automatic-speech-recognition":
                ov_model = core.read_model(os.path.join(model_dst_path, model_name.replace("/", "--") + ".xml"))
                ov_model = core.compile_model(ov_model)
            else:
                ov_model = core.read_model(os.path.join(model_dst_path, model_name.replace("/", "--") + ".xml"))
                ov_model = core.compile_model(ov_model)
            return ov_model

    def get_openvino_genai_pipeline(self, model_name, model_type=None, device_name=None):
        core = self.ov.Core()
        architecture = None
        config = None
        hfmodel = None
        hftokenizer = None
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
            config = self.transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            model_type = config.__class__.model_type
        except Exception as e:
            config = None

        ov_model = None
        hfmodel = None
        hftokenizer = self.transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        vlm_model_types = ["llava", "llava_next"]

        if os.path.exists(model_src_path) and not os.path.exists(model_dst_path):
            try:
                if model_type not in vlm_model_types:
                    hfmodel = self.transformers.AutoModel.from_pretrained(model_name,  trust_remote_code=True)
                else:
                    hfmodel = self.transformers.AutoModel.from_pretrained(model_name, trust_remote_code=True)
            
                text = "Replace me by any text you'd like."
                encoded_input = hftokenizer(text, return_tensors='pt')
                ov_model = self.ov.convert_model(hfmodel, example_input={**encoded_input})                        
                self.ov.save_model(ov_model, model_dst_path)
                ov_model = self.ov.compile_model(ov_model)
                hfmodel = None
                if hftokenizer:
                    del hftokenizer

            except Exception as e:
                if hfmodel:
                    hfmodel = None
                if hftokenizer:
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
                    config = self.transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
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
                ov_model = self.ov_genai.VLMPipeline(model_dst_path, device=device)
            elif model_task is not None and model_task == "text-generation-with-past":
                ov_model = self.ov_genai.LLMPipeline(model_dst_path, device=device)
            elif model_task is not None and model_task == "automatic-speech-recognition":
                whisper_config = self.ov_genai.WhisperGenerationConfig
                ov_model = self.ov_genai.WhisperPipeline(model_dst_path, device=device)
            elif model_task is not None and model_task == "image-to-image":
                ov_model = self.ov_genai.Text2ImagePipeline(model_dst_path, device=device)
            elif model_task is not None and model_task == "'imge-to-text":
                ov_model = self.ov_genai.CLIPTextModelWithProjection(model_dst_path, device=device)
            else:
                raise ValueError("Loading Model: " + model_name + " Task not supported: " + model_task + " Supported tasks: " + str(model_mapping_list))

        return ov_model
    
    def get_optimum_openvino_model(self, model_name, model_type=None, openvino_label=None):
        from transformers import AutoConfig
        homedir = os.path.expanduser("~")
        model_name_convert = model_name.replace("/", "--")
        huggingface_cache = os.path.join(homedir, ".cache","huggingface")
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
    
    def get_model_type(self, model_name=None, model_type=None):
        if model_name is not None:
            if os.path.exists(model_name):
                config = self.transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                model_type = config.__class__.model_type
            else:
                config = self.transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                model_type = config.__class__.model_type
        return model_type
    
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
        try:
            convert_model = subprocess.check_output(parsed_cmd, shell=True)
            convert_model = convert_model.decode('utf-8')
        except Exception as e:
            print(e)
            convert_model = None
        return convert_model
    
    
    def get_openvino_pipeline_type(self, model_name, model_type=None):
        model_mapping_list = ['image-text-to-text', 'fill-mask', 'image-classification', 'image-segmentation', 'feature-extraction', 'token-classification', 'audio-xvector', 'audio-classification', 'zero-shot-image-classification', 'text2text-generation', 'depth-estimation', 'text-to-audio', 'semantic-segmentation', 'masked-im', 'image-to-text', 'zero-shot-object-detection','mask-generation', 'sentence-similarity', 'image-to-image', 'object-detection', 'multiple-choice', 'automatic-speech-recognition', 'text-classification', 'audio-frame-classification', 'text-generation', 'question-answering']
        return_model_type = None
        config_model_type = None
        if model_type is not None:
            try:
                config = self.transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                config_model_type = config.__class__.model_type
            except Exception as e:
                config_model_type = None
                                
            if config_model_type is None:
                config_model_type = config_model_type if config_model_type is not None else model_type
            if config_model_type is not None:
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
                elif config_model_type == "wav2vec2":
                    return_model_type = "feature-extraction"
                elif config_model_type == "t5":
                    return_model_type = "text2text-generation-with-past"
                elif config_model_type == "whisper":
                    return_model_type = "automatic-speech-recognition"
                elif config_model_type == "xclip":
                    return_model_type = "feature-extraction"
                pass
            elif config_model_type in model_mapping_list:
                return_model_type = config_model_type   

        return return_model_type
