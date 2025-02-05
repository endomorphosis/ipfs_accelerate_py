import asyncio
import json
class ovms:
    def __init__(self, resources, metadata):
        self.resources = resources
        self.metadata = metadata
        self.create_ovms_endpoint_handler = self.create_ovms_endpoint_handler
        self.test_ovms_endpoint = self.test_ovms_endpoint
        self.make_post_request_ovms = self.make_post_request_ovms
        return None

    def create_ovms_endpoint_handler(self, model, endpoint, context_length):
        from transformers import AutoTokenizer, AutoModel, AutoConfig
        async def handler(x):
            tokenizer = None
            tokens = None
            if model not in list(self.resources["tokenizer"].keys()):
                self.resources["tokenizer"][model] = {}
            tokenizers = list(self.resources["tokenizer"][model].keys())
            if len(tokenizers) == 0:
                self.resources["tokenizer"][model]["cpu"] = AutoTokenizer.from_pretrained(model, device='cpu', use_fast=True, trust_remote_code=True)
                tokens = await self.resources["tokenizer"][model]["cpu"](x, return_tensors="pt", padding=True, truncation=True)
            else:
                for tokenizer in tokenizers:
                    try:
                        this_tokenizer = self.resources["tokenizer"][model][tokenizer]
                        tokens = await this_tokenizer[model][endpoint](x, return_tensors="pt", padding=True, truncation=True)
                    except Exception as e:
                        pass
            if tokens is None:
                raise ValueError("No tokenizer found for model " + model)            
            tokens = await self.tokenizer[model][endpoint](x, return_tensors="pt", padding=True, truncation=True)
            remote_endpoint = await self.make_post_request_openvino(tokens, x)
            return remote_endpoint
        return handler
    
    async def test_ovms_endpoint(self, model, endpoint_list=None):
        this_endpoint = None
        filtered_list = {}
        test_results = {}
        api_endpoints = self.resources["ovms_endpoints"]
        api_endpoints_types = [x[1] for x in api_endpoints]
        api_endpoints_by_model = self.endpoints["ovms_endpoints"][model]
        endpoint_handlers_by_model = self.resources["ovms_endpoints"][model]
        if endpoint_list is not None:
            api_endpoints_by_model_by_endpoint_list = [ x for x in api_endpoints_by_model if "openvino:" in json.dumps(x) and x[1] in list(endpoint_handlers_by_model.keys()) ]
        else:
            local_endpoints_by_model_by_endpoint_list = [ x for x in api_endpoints_by_model if "openvino:" in json.dumps(x) ]
        if len(local_endpoints_by_model_by_endpoint_list) > 0:
            for endpoint in local_endpoints_by_model_by_endpoint_list:
                endpoint_handler = endpoint_handlers_by_model[endpoint]
                try:
                    test = await endpoint_handler("hello world")
                    test_results[endpoint] = test
                except Exception as e:
                    try:
                        test = endpoint_handler("hello world")
                        test_results[endpoint] = test
                    except Exception as e:
                        test_results[endpoint] = e
                    pass
        else:
            return ValueError("No endpoint_handlers found")
        return test_results
                    
    async def make_post_request_openvino(self, endpoint, data):
        import aiohttp
        from aiohttp import ClientSession, ClientTimeout
        if type(data) is dict:
            raise ValueError("Data must be a string")
        if type(data) is list:
            if len(data) > 1:
                raise ValueError("batch size must be 1")
            data = data[0]
        headers = {'Content-Type': 'application/json'}
        timeout = ClientTimeout(total=300) 
        async with ClientSession(timeout=timeout) as session:
            try:
                async with session.post(endpoint, headers=headers, json=data) as response:
                    if response.status != 200:
                        return ValueError(response)
                    return await response.json()
            except Exception as e:
                print(str(e))
                if "Can not write request body" in str(e):
                    print( "endpoint " + endpoint + " is not accepting requests")
                    return ValueError(e)
                if "Timeout" in str(e):
                    print("Timeout error")
                    return ValueError(e)
                if "Payload is not completed" in str(e):
                    print("Payload is not completed")
                    return ValueError(e)
                if "Can not write request body" in str(e):
                    return ValueError(e)
                pass
            except aiohttp.ClientPayloadError as e:
                print(f"ClientPayloadError: {str(e)}")
                return ValueError(f"ClientPayloadError: {str(e)}")
            except asyncio.TimeoutError as e:
                print(f"Timeout error: {str(e)}")
                return ValueError(f"Timeout error: {str(e)}")
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                return ValueError(f"Unexpected error: {str(e)}")
 