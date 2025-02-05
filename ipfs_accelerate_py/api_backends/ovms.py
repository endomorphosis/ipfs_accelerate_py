import asyncio
import json
class ovms:
    def __init__(self, resources, metadata):
        self.resources = resources
        self.metadata = metadata
        self.test_openvino_endpoint = self.test_openvino_endpoint
        self.make_post_request_openvino = self.make_post_request_openvino
        return None

    async def test_openvino_endpoint(self, model, endpoint_list=None):
        this_endpoint = None
        filtered_list = {}
        test_results = {}
        local_endpoints = self.resources["openvino_endpoints"]
        local_endpoints_types = [x[1] for x in local_endpoints]
        local_endpoints_by_model = self.endpoints["openvino_endpoints"][model]
        endpoint_handlers_by_model = self.resources["openvino_endpoints"][model]
        if endpoint_list is not None:
            local_endpoints_by_model_by_endpoint_list = [ x for x in local_endpoints_by_model if "openvino:" in json.dumps(x) and x[1] in list(endpoint_handlers_by_model.keys()) ]
        else:
            local_endpoints_by_model_by_endpoint_list = [ x for x in local_endpoints_by_model if "openvino:" in json.dumps(x) ]
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
 