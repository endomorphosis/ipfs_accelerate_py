import asyncio
class hf_tei:
    def __init__(self, resources, metadata):
        self.request_hf_tei_endpoint = self.request_hf_tei_endpoint
        self.make_post_request_hf_tei = self.make_post_request_hf_tei
        self.create_hf_tei_endpoint_handler = self.create_hf_tei_endpoint_handler
        self.test_hf_tei_endpoint = self.test_hf_tei_endpoint
        return None
    
    def request_hf_tei_endpoint(self, model, endpoint=None, endpoint_type=None, batch=None):

        return None
    
    async def test_hf_tei_endpoint(self, model, endpoint_list=None):
        this_endpoint = None
        filtered_list = {}
        test_results = {}
        local_endpoints = self.resources["tei_endpoints"]
        local_endpoints_types = [x[1] for x in local_endpoints]
        local_endpoints_by_model = self.endpoints["tei_endpoints"][model]
        endpoint_handlers_by_model = self.resources["tei_endpoints"][model]
        local_endpoints_by_model_by_endpoint = list(endpoint_handlers_by_model.keys())
        local_endpoints_by_model_by_endpoint = [ x for x in local_endpoints_by_model_by_endpoint if x in local_endpoints_by_model if x in local_endpoints_types]
        if len(local_endpoints_by_model_by_endpoint) > 0:
            for endpoint in local_endpoints_by_model_by_endpoint:
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
    
    async def create_hf_tei_endpoint_handler(self, model, endpoint=None, endpoint_type=None, batch=None):
        if batch == None:
            incoming_batch_size = 0
        else:
            incoming_batch_size = len(batch)
        endpoint_batch_size = 0
        if endpoint in self.endpoint_status:
            endpoint_batch_size = self.endpoint_status[endpoint]
        elif endpoint_type == None:
            for endpoint_type in self.endpoint_types:
                if endpoint_type in self.__dict__.keys():
                    if model in self.__dict__[endpoint_type]:
                        for endpoint in self.__dict__[endpoint_type][model]:
                            endpoint_batch_size = self.endpoint_status[endpoint]
                            if self.endpoint_status[endpoint] >= incoming_batch_size:
                                return endpoint
                    else:
                        if incoming_batch_size > endpoint_batch_size:
                            return ValueError("Batch size too large")
                        else:
                            return None
                else:
                    pass
        else:
            if model in self.__dict__[endpoint_type]:
                for endpoint in self.__dict__[endpoint_type][model]:
                    endpoint_batch_size = self.endpoint_status[endpoint]
                    if self.endpoint_status[endpoint] >= incoming_batch_size:
                        return endpoint
                    else:
                        if incoming_batch_size > endpoint_batch_size:
                            return ValueError("Batch size too large")
                        else:
                            return None
            else:
                return None
                
        if incoming_batch_size > endpoint_batch_size:
            return ValueError("Batch size too large")
        else:
            if model in self.endpoints:
                for endpoint in self.tei_endpoints[model]:
                    if self.endpoint_status[endpoint] >= incoming_batch_size:
                        return endpoint
            return None
    
    async def make_post_request_hf_tei(self, endpoint, data=None):
        import aiohttp
        from aiohttp import ClientSession, ClientTimeout
        if data is None:
            return None
        else:
            pass
        if type(data) is dict:
            if "inputs" not in list(data.keys()):
                data = {"inputs": data}
        if type(data) is list:
            data = {"inputs": data}
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

    def create_hf_tei_endpoint_handler(self, model, endpoint, context_length):
        async def handler(x):
            remote_endpoint = await self.make_post_request_tei(endpoint, x)
            return remote_endpoint
        return handler