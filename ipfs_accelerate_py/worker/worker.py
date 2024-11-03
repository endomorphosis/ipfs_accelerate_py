import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test_ipfs_accelerate import test_ipfs_accelerate
from install_depends import install_depends_py
import torch 
import asyncio
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig, pipeline
import ipfs_transformers_py
from pathlib import Path
import numpy as np
import torch

class worker_py:
    def __init__(self, metadata, resources):
        self.metadata = metadata
        self.resources = resources
        self.tokenizer = {}
        self.queues = {}
        self.batch_sizes = {}
        self.endpoint_handler = {}
        self.endpoints = {}
        self.endpoint_types = ["local_endpoints"]
        self.local_endpoints = {}
        self.hardware_backends = ["llama_cpp", "cpu", "gpu", "openvino", "optimum", "optimum_cpp", "optimum_intel","ipex"]
        if "test_ipfs_accelerate" not in globals():
            self.test_ipfs_accelerate = test_ipfs_accelerate(resources, metadata)
            self.hwtest = self.test_ipfs_accelerate
        if "install_depends" not in globals():
            self.install_depends = install_depends_py(resources, metadata)
        if "ipfs_transformers_py" not in globals() and "ipfs_transformers_py" not in list(self.resources.keys()):
            from ipfs_transformers_py import ipfs_transformers
            self.ipfs_transformers = { 
                # "AutoDownloadModel" : ipfs_transformers.AutoDownloadModel()
            }        
        elif "ipfs_transformers_py" in list(self.resources.keys()):
            self.ipfs_transformers_py = self.resources["ipfs_transformers_py"]
        elif "ipfs_transformers_py" in globals():
            from ipfs_transformers_py import ipfs_transformers
            self.ipfs_transformers = { 
                # "AutoDownloadModel" : ipfs_transformers.AutoDownloadModel()
            }
        if "transformers" not in globals() and "transformers" not in list(self.resources.keys()):
            self.transformers = transformers   
        elif "transformers" in list(self.resources.keys()):
            self.transformers = self.resources["transformers"]
        elif "transformers" in globals():
            self.transformers = transformers
        if "torch" not in globals() and "torch" not in list(self.resources.keys()):
            self.torch = torch
        elif "torch" in list(self.resources.keys()):
            self.torch = self.resources["torch"]
        elif "torch" in globals():
            self.torch = torch
        else:
            pass
        for endpoint in self.endpoint_types:
            if endpoint not in dir(self):
                self.__dict__[endpoint] = {}        
            for backend in self.hardware_backends:
                if backend not in list(self.__dict__[endpoint].keys()):
                    self.__dict__[endpoint][backend] = {}
        print(self.__dict__)
        return None
    
    async def test_hardware(self):
        return await self.install_depends.test_hardware()
    
    async def init_worker(self, models, local_endpoints, hwtest):
        if local_endpoints is None or len(local_endpoints) == 0:
            if "local_endpoints" in list(self.__dict__.keys()):
                local_endpoints = self.local_endpoints
            elif "local_endpoints" in list(self.resources.keys()):
                local_endpoints = self.resources["local_endpoints"]
            else:
                local_endpoints = []
        else:
            pass
        self.local_endpoints  = local_endpoints
        self.local_endpoint_types = [ x[1] for x in local_endpoints if x[0] in models]
        self.local_endpoint_models = [ x[0] for x in local_endpoints if x[0] in models]
        self.local_endpoints = { model: { endpoint[1]: endpoint[2] for endpoint in local_endpoints if endpoint[0] == model} for model in models}
        local = len(local_endpoints) > 0 if isinstance(self.local_endpoints, dict) and len(list(self.local_endpoints.keys())) > 0 else False
        if hwtest is None:
            if "hwtest" in list(self.__dict__.keys()):
                hwtest = self.hwtest
            elif "hwtest" in list(self.resources.keys()):
                hwtest = self.resources["hwtest"]
            else:
                hwtest = await self.hwtest()
        self.hwtest = hwtest
        cuda_test = self.hwtest["cuda"]
        openvino_test = self.hwtest["openvino"]
        llama_cpp_test = self.hwtest["llama_cpp"]
        ipex_test = self.hwtest["ipex"]
        cpus = os.cpu_count()
        cuda = torch.cuda.is_available()
        gpus = torch.cuda.device_count()
        
        for model in models:
            if model not in self.tokenizer:
                self.tokenizer[model] = {}
            if model not in self.local_endpoints:
                self.local_endpoints[model] = {}
            if model not in self.queues:
                self.queues[model] = {}
            if model not in self.batch_sizes:
                self.batch_sizes[model] = {}
            if cuda and gpus > 0:
                if cuda_test and type(cuda_test) != ValueError:
                    for gpu in range(gpus):
                        cuda_index = self.local_endpoint_types.index("cuda:"+str(gpu))
                        endpoint_model = self.local_endpoint_models[cuda_index]
                        cuda_label = self.local_endpoint_types[cuda_index]
                        self.tokenizer[endpoint_model][cuda_label] = AutoTokenizer.from_pretrained(model, device='cuda:' + str(gpu), use_fast=True)
                        self.local_endpoints[endpoint_model][cuda_label] = AutoModel.from_pretrained(model).to("cuda:" + str(gpu))
                        self.endpoint_handler[(endpoint_model, cuda_label)] = self.local_endpoints[endpoint_model][cuda_label]
                        torch.cuda.empty_cache()
                        self.queues[endpoint_model][cuda_label] = asyncio.Queue(64)
                        batch_size = await self.max_batch_size(endpoint_model, cuda_label)
                        self.batch_sizes[endpoint_model][cuda_label] = batch_size
                        # consumer_tasks[(model, "cuda:" + str(gpu))] = asyncio.create_task(self.chunk_consumer(batch_size, model, "cuda:" + str(gpu)))
            if local > 0 and cpus > 0:
                all_test_types = [ type(openvino_test), type(llama_cpp_test), type(ipex_test)]
                all_tests_ValueError = all(x is ValueError for x in all_test_types)
                all_tests_none = all(x is None for x in all_test_types)
                if all_tests_ValueError or all_tests_none:  
                    self.local_endpoints[model]["cpu"] = AutoModel.from_pretrained(model).to("cpu")
                    self.queues[model]["cpu"] = asyncio.Queue(4)
                    self.endpoint_handler[(model, "cpu")] = ""
                    # consumer_tasks[(model, "cpu")] = asyncio.create_task(self.chunk_consumer( 1, model, "cpu"))
                elif openvino_test and type(openvino_test) != ValueError:
                    ov_count = 0
                    openvino_index = self.local_endpoint_types.index("openvino")
                    openvino_model = self.local_endpoint_models[openvino_index]
                    openvino_label = self.local_endpoint_types[openvino_index]
                    ## use openvino to call huggingface transformers
                    if "openvino" in openvino_label:
                        # from optimum import openvino
                        # from optimum import intel
                        # from openvino import OVConfig
                        # from openvino import OVModel
                        # from openvino import OVTokenizer
                        openvino_methods = {
                            "OVModelForSequenceClassification"	: "text-classification",
                            "OVModelForTokenClassification"	    : "token-classification",
                            "OVModelForQuestionAnswering"	    : "question-answering",
                            "OVModelForAudioClassification"	    : "audio-classification",
                            "OVModelForImageClassification"	    : "image-classification",
                            "OVModelForFeatureExtraction"	    : "feature-extraction",
                            "OVModelForMaskedLM"                : "fill-mask",
                            "OVModelForImageClassification"	    : "image-classification",
                            "OVModelForAudioClassification"     : "audio-classification",
                            "OVModelForCausalLM"	            : "text-generation-with-past",
                            "OVModelForSeq2SeqLM"	            : "text2text-generation-with-past",
                            "OVModelForSpeechSeq2Seq"       	: "automatic-speech-recognition",
                            "OVModelForVision2Seq"	            : "image-to-text",
                        }
                        import optimum
                        from optimum.intel import OVModelForSequenceClassification, OVModelForTokenClassification, OVModelForQuestionAnswering, OVModelForAudioClassification, OVModelForImageClassification, OVModelForFeatureExtraction, OVModelForMaskedLM, OVModelForCausalLM, OVModelForSeq2SeqLM, OVModelForSpeechSeq2Seq, OVModelForVision2Seq
                        from optimum.intel.openvino import OVConfig
                        from openvino.runtime import Core

                        def get_openvino_model(model_name, model_type):
                            model_mapping = {
                                "text-classification": OVModelForSequenceClassification,
                                "token-classification": OVModelForTokenClassification,
                                "question-answering": OVModelForQuestionAnswering,
                                "audio-classification": OVModelForAudioClassification,
                                "image-classification": OVModelForImageClassification,
                                "feature-extraction": OVModelForFeatureExtraction,
                                "fill-mask": OVModelForMaskedLM,
                                "text-generation-with-past": OVModelForCausalLM,
                                "text2text-generation-with-past": OVModelForSeq2SeqLM,
                                "automatic-speech-recognition": OVModelForSpeechSeq2Seq,
                                "image-to-text": OVModelForVision2Seq,
                            }
                            return model_mapping.get(model_type, None)

                        save_model_path = Path("./models/model.xml")
                        text = "HF models run perfectly with OpenVINO!"
                        self.tokenizer[openvino_model][openvino_label] = AutoTokenizer.from_pretrained(model, use_fast=True)
                        config = AutoConfig.from_pretrained(model)
                        model_type = config.__class__.model_type
                        model_class = get_openvino_model(model, config.model_type)
                        if model_class:
                            self.local_endpoints[openvino_model][openvino_label] = model_class.from_pretrained(model)
                        encoded_input = self.tokenizer[openvino_model][openvino_label](text, return_tensors="pt")
                        self.endpoint_handler[(openvino_model, openvino_label)] = pipeline(openvino_methods[config.model_type], model=self.local_endpoints[openvino_model][openvino_label], tokenizer=self.tokenizer[openvino_model][openvino_label])
                        output = self.local_endpoints[openvino_model][openvino_label](**encoded_input).last_hidden_state
                        if not save_model_path.exists():
                            ov_model = Core().compile_model(self.local_endpoints[openvino_model][openvino_label], "CPU")
                            ov_model.export_model(save_model_path)
                        core = Core()
                        compiled_model = core.compile_model(save_model_path, "CPU")
                        scores_ov = compiled_model(encoded_input.data)[0]
                        scores_ov = torch.softmax(torch.tensor(scores_ov[0]), dim=0).detach().numpy()
                        print_prediction(scores_ov)
                        # self.tokenizer[openvino_model][openvino_label] = OpenVinoTokenizer.from_pretrained(model, use_fast=True)
                        # self.local_endpoints[openvino_model][openvino_label] = OpenVinoModel.from_pretrained(model).to("cpu")
                        # self.queues[openvino_model][openvino_label] = asyncio.Queue(64)
                        # self.tokenizer[openvino_model][openvino_label] = AutoTokenizer.from_pretrained(model, use_fast=True)
                        # self.local_endpoints[openvino_model][openvino_label] = AutoModel.from_pretrained(model).to("cpu")
                        # self.queues[openvino_model][openvino_label] = asyncio.Queue(64)
                        # self.endpoint_handler[(openvino_model, openvino_label)] = ""
                        # batch_size = await self.max_batch_size(openvino_model, openvino_label)
                        # self.batch_sizes[openvino_model][openvino_label] = batch_size
                        # consumer_tasks[(model, "openvino")] = asyncio.create_task(self.chunk_consumer(batch_size, model, "openvino"))
                    if endpoint not in list(self.batch_sizes[model].keys()):
                        batch_size = await self.max_batch_size(model, endpoint)
                        self.batch_sizes[model][endpoint_name] = batch_size
                    if self.batch_sizes[model][endpoint_name] > 0:
                        self.queues[model][endpoint_name] = asyncio.Queue(64)
                        self.endpoint_handler[(model, endpoint_name)] = ""
                        # consumer_tasks[(model, endpoint_name )] = asyncio.create_task(self.chunk_consumer(batch_size, model, endpoint_name))
                    ov_count = ov_count + 1
                elif llama_cpp_test and type(llama_cpp_test) != ValueError:
                    llama_count = 0
                    for endpoint in local:
                        if "llama_cpp" in endpoint:
                            endpoint_name = "llama:"+str(ov_count)
                            batch_size = 0                            
                            if model not in self.batch_sizes:
                                self.batch_sizes[model] = {}
                            if model not in self.queues:
                                self.queues[model] = {}
                            if endpoint not in list(self.batch_sizes[model].keys()):
                                batch_size = await self.max_batch_size(model, endpoint)
                                self.batch_sizes[model][endpoint] = batch_size
                            if self.batch_sizes[model][endpoint] > 0:
                                self.queues[model][endpoint] = asyncio.Queue(64)
                                self.endpoint_handler[(model, endpoint_name)] = ""
                                # consumer_tasks[(model, endpoint)] = asyncio.create_task(self.chunk_consumer(batch_size, model, endpoint_name))
                            llama_count = llama_count + 1
                elif ipex_test and type(ipex_test) != ValueError:
                    ipex_count = 0
                    for endpoint in local:
                        if "ipex" in endpoint:
                            endpoint_name = "ipex:"+str(ipex_count)
                            batch_size = 0
                            if model not in self.batch_sizes:
                                self.batch_sizes[model] = {}
                            if model not in self.queues:
                                self.queues[model] = {}
                            if endpoint not in list(self.batch_sizes[model].keys()):
                                batch_size = await self.max_batch_size(model, endpoint)
                                self.batch_sizes[model][endpoint] = batch_size
                            if self.batch_sizes[model][endpoint] > 0:
                                self.queues[model][endpoint] = asyncio.Queue(64)
                                self.endpoint_handler[(model, endpoint_name)] = ""
                                # consumer_tasks[(model, endpoint)] = asyncio.create_task(self.chunk_consumer(self.queues[model][endpoint], column, batch_size, model, endpoint))
                            ipex_count = ipex_count + 1
            if "openvino_endpoints" in list(self.endpoints.keys()):
                if len(self.endpoints["openvino_endpoints"]) > 0 :
                    for endpoint in self.endpoints["openvino_endpoints"]:
                        batch_size = 0
                        if model not in self.batch_sizes:
                            self.batch_sizes[model] = {}
                        if model not in self.queues:
                            self.queues[model] = {}
                        if endpoint not in list(self.batch_sizes[model].keys()):
                            batch_size = await self.max_batch_size(model, endpoint)
                            self.batch_sizes[model][endpoint] = batch_size
                        if self.batch_sizes[model][endpoint] > 0:
                            self.queues[model][endpoint] = asyncio.Queue(64)  # Unbounded queue
                            self.endpoint_handler[(model, endpoint)] = ""
                            # consumer_tasks[(model, endpoint)] = asyncio.create_task(self.chunk_consumer(self.queues[model][endpoint], column, batch_size, model, endpoint))
            if "tei_endpoints" in list(self.endpoints.keys()):
                if len(self.endpoints["tei_endpoints"]) > 0:
                    for endpoint in self.endpoints["tei_endpoints"]:
                        this_model = endpoint[0]
                        this_endpoint = endpoint[1]
                        context_length = endpoint[2]
                        batch_size = 0
                        if this_model not in self.batch_sizes:
                            self.batch_sizes[this_model] = {}
                        if this_model not in self.queues:
                            self.queues[model] = {}
                        if endpoint not in list(self.batch_sizes[model].keys()):
                            batch_size = await self.max_batch_size(model, endpoint)
                            self.batch_sizes[model][this_endpoint] = batch_size
                        if self.batch_sizes[model][this_endpoint] > 0:
                            self.queues[model][this_endpoint] = asyncio.Queue(64)  # Unbounded queue
                            self.endpoint_handler[(model, this_endpoint)] = ""
                            # consumer_tasks[(model, endpoint)] = asyncio.create_task(self.chunk_consumer(batch_size, model, endpoint)) 
            if "libp2p_endpoints" in list(self.endpoints.keys()):
                if len(self.endpoints["libp2p_endpoints"]) > 0:
                    for endpoint in self.endpoints["libp2p_endpoints"]:
                        batch_size = 0
                        if model not in self.batch_sizes:
                            self.batch_sizes[model] = {}
                        if model not in self.queues:
                            self.queues[model] = {}
                        if endpoint not in list(self.batch_sizes[model].keys()):
                            batch_size = await self.max_batch_size(model, endpoint)
                            self.batch_sizes[model][endpoint] = batch_size
                        if self.batch_sizes[model][endpoint] > 0:
                            self.queues[model][endpoint] = asyncio.Queue(64)
        return None
    
    async def max_batch_size(self, model, endpoint):
        
        return None
    
    
    def __test__(self):
        return self 
    
export = worker_py
    
if __name__ == '__main__':
    # run(skillset=os.path.join(os.path.dirname(__file__), 'skillset'))
    resources = {}
    metadata = {}
    try:
        this_worker = worker_py(resources, metadata)
        this_test = this_worker.__test__(resources, metadata)
    except Exception as e:
        print(e)
        pass
    
    
