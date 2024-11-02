import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test_ipfs_accelerate import test_ipfs_accelerate
from install_depends import install_depends_py
import torch 
import asyncio
import transformers
from transformers import AutoTokenizer, AutoModel

class worker:
    def __init__(self, metadata, resources):
        self.metadata = metadata
        self.resources = resources
        self.endpoint_types = ["local_endpoints"]
        self.hardware_backends = ["llama_cpp", "cpu", "gpu", "openvino", "optimum", "optimum_cpp", "optimum_intel","ipex"]
        if "test_ipfs_accelerate" not in globals():
            self.test_ipfs_accelerate = test_ipfs_accelerate(resources, metadata)
            self.hwtest = self.test_ipfs_accelerate
        if "install_depends" not in globals():
            self.install_depends = install_depends_py(resources, metadata)
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
            
    async def init_worker(self, models):
        local = self.endpoint_types["local_endpoints"]
        if "hwtest" not in dir(self):
            hwtest = await self.test_hardware()
            self.hwtest = hwtest
        cuda_test = self.hwtest["cuda"]
        openvino_test = self.hwtest["openvino"]
        llama_cpp_test = self.hwtest["llama_cpp"]
        ipex_test = self.hwtest["ipex"]
        cpus = os.cpu_count()
        cuda = torch.cuda.is_available()
        gpus = torch.cuda.device_count()
        for model in models:
            if cuda and gpus > 0:
                if cuda_test and type(cuda_test) != ValueError:
                    self.local_endpoints[model] = {"cuda:" + str(gpu) : None for gpu in range(gpus) } if gpus > 0 else {"cpu": None}
                    for gpu in range(gpus):
                        self.tokenizer[model]["cuda:" + str(gpu)] = AutoTokenizer.from_pretrained(model, device='cuda:' + str(gpu), use_fast=True)
                        self.local_endpoints[model]["cuda:" + str(gpu)] = AutoModel.from_pretrained(model).to("cuda:" + str(gpu))
                        torch.cuda.empty_cache()
                        self.queues[model]["cuda:" + str(gpu)] = asyncio.Queue(64)
                        batch_size = await self.max_batch_size(model, "cuda:" + str(gpu))
                        self.local_endpoints[model]["cuda:" + str(gpu)] = batch_size
                        self.batch_sizes[model]["cuda:" + str(gpu)] = batch_size
                        self.endpoint_handler[(model, "cuda:" + str(gpu))] = ""
                        # consumer_tasks[(model, "cuda:" + str(gpu))] = asyncio.create_task(self.chunk_consumer(batch_size, model, "cuda:" + str(gpu)))
            if len(local) > 0 and cpus > 0:
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
                    for endpoint in local:
                        if "openvino" in endpoint[1]:
                            endpoint_name = "openvino:"+str(ov_count)
                            batch_size = 0
                            if model not in self.batch_sizes:
                                self.batch_sizes[model] = {}
                            if model not in self.queues:
                                self.queues[model] = {}
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
    
    def __test__(self):
        return self 
    
if __name__ == '__main__':
    # run(skillset=os.path.join(os.path.dirname(__file__), 'skillset'))
    resources = {}
    metadata = {}
    try:
        this_worker = worker(resources, metadata)
        this_test = this_worker.__test__(resources, metadata)
    except Exception as e:
        print(e)
        pass