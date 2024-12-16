import subprocess
from transformers import AutoConfig

class openvino_utils:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata
        self.openvino_cli_convert = self.openvino_cli_convert
        self.get_openvino_pipeline_type = self.get_openvino_pipeline_type
        self.get_optimum_openvino_model = self.get_optimum_openvino_model
        return None
    
    def init(self):
        return None
    
    async def get_optimum_openvino_model(self, model_name, model_type=None):
        if model_type is None:
            config = AutoConfig.from_pretrained(model_name)
            model_type = config.__class__.model_type
        model_mapping_list = ["text-classification", "token-classification", "question-answering", "audio-classification", "image-classification", "feature-extraction", "fill-mask", "text-generation-with-past", "text2text-generation-with-past", "automatic-speech-recognition", "image-to-text", "bert", "llava", "image-text-to-text"]
        if model_type not in model_mapping_list:
            return None
        if model_type == "bert":
            model_type = "feature-extraction"
            from optimum.intel import OVModelForFeatureExtraction
            results = OVModelForFeatureExtraction.from_pretrained(model_name, compile=False)
        elif model_type == "text-classification":
            from optimum.intel import OVModelForSequenceClassification
            results = OVModelForSequenceClassification.from_pretrained(model_name, compile=False)
        elif model_type == "token-classification":
            from optimum.intel import OVModelForTokenClassification
            results = OVModelForTokenClassification.from_pretrained(model_name, compile=False)
        elif model_type == "question-answering":
            from optimum.intel import OVModelForQuestionAnswering
            results = OVModelForQuestionAnswering.from_pretrained(model_name, compile=False)
        elif model_type == "audio-classification":
            from optimum.intel import OVModelForAudioClassification
            results = OVModelForAudioClassification.from_pretrained(model_name,  compile=False)
        elif model_type == "image-classification":
            from optimum.intel import OVModelForImageClassification
            results = OVModelForImageClassification.from_pretrained(model_name, compile=False) 
        elif model_type == "feature-extraction":
            from optimum.intel import OVModelForFeatureExtraction
            results = OVModelForFeatureExtraction.from_pretrained(model_name, compile=False)
        elif model_type == "fill-mask":
            from optimum.intel import OVModelForMaskedLM
            results = OVModelForMaskedLM.from_pretrained(model_name, compile=False)
        elif model_type == "text-generation-with-past":
            from optimum.intel import OVModelForCausalLM
            results = OVModelForCausalLM.from_pretrained(model_name, compile=False)
        elif model_type == "text2text-generation-with-past":
            from optimum.intel import OVModelForSeq2SeqLM
            results = OVModelForSeq2SeqLM.from_pretrained(model_name, compile=False)
        elif model_type == "automatic-speech-recognition":
            from optimum.intel import OVModelForSpeechSeq2Seq
            results = OVModelForSpeechSeq2Seq.from_pretrained(model_name, compile=False)
        elif model_type == "image-text-to-text":
            from optimum.intel import OVModelForVision2Seq
            results = OVModelForVision2Seq.from_pretrained(model_name, compile=False)
        else:
            return None
        # await results.compile()
        return results
    
    
    def get_model_type(self, model_name, model_type=None):
        if model_type is None:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
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
            if task not in tasks_list:
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
        
        # Execute the command
        convert_model = subprocess.check_output(command)
        return convert_model
    
    
    def get_openvino_pipeline_type(self, model_name, model_type=None):
        model_mapping_list = ["text-classification", "token-classification", "question-answering", "audio-classification", "image-classification", "feature-extraction", "fill-mask", "text-generation-with-past", "text2text-generation-with-past", "automatic-speech-recognition", "image-to-text"]
        model_mapping_list = ['fill-mask', 'image-classification', 'image-segmentation', 'feature-extraction', 'token-classification', 'audio-xvector', 'audio-classification', 'zero-shot-image-classification', 'text2text-generation', 'depth-estimation', 'text-to-audio', 'semantic-segmentation', 'masked-im', 'image-to-text', 'zero-shot-object-detection','mask-generation', 'sentence-similarity', 'image-to-image', 'object-detection', 'multiple-choice', 'automatic-speech-recognition', 'text-classification', 'audio-frame-classification', 'text-generation', 'question-answering']
        return_model_type = None
        config_model_type = None
        if model_type is not None:
            model_type = model_type
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
                pass
            elif config_model_type not in model_mapping_list and model_type not in model_mapping_list:
                config_model_type = config_model_type if config_model_type is not None else model_type
                if config_model_type == "bert":
                    return_model_type = "feature-extraction"
                elif config_model_type == "llava":
                    return_model_type = "image-text-to-text"
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
                pass            
            elif config_model_type not in model_mapping_list and model_type not in model_mapping_list:
                config_model_type = config_model_type if config_model_type is not None else model_type
                if config_model_type == "bert":
                    return_model_type = "feature-extraction"
                elif config_model_type == "llava":
                    return_model_type = "image-text-to-text"            
            elif config_model_type in model_mapping_list:
                return_model_type = config_model_type   

        return return_model_type
