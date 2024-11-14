from typing import Union
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from ipfs_accelerate_py import ipfs_accelerate_py
from pydantic import BaseModel
import json
from torch import Tensor

class InitEndpointsRequest(BaseModel):
    models: list
    resources: dict[str, list[list[str]]]
    
class TestEndpointRequest(BaseModel):
    models: list[str]
    resources: dict[str, list[list[str]]]

class InferEndpointRequest(BaseModel):
    models: list
    batch_data: list[str]

class AddEndpointRequest(BaseModel):
    model: str
    endpoint_type: str
    endpoint: list[str]

class RmEndpointRequest(BaseModel):
    model: str
    endpoint_type: str
    backend: str

class InitStatusRequest(BaseModel):    
    models: list[str]
    
app = FastAPI(port=9999)
resources = {}
metadata = {}

class ModelServer:
    def __init__(self, resources=None, metadata=None):
        if resources is None:
            resources = {}
        if metadata is None:
            metadata = {}
        self.resources = resources
        self.metadata = metadata
        self.resources["ipfs_accelerate_py"] = ipfs_accelerate_py(self.resources, self.metadata)
        return 

    async def initEndpointsTask(self, models: list, resources: dict):
        results = {}
        try:
            results["init"] = await self.init_endpoints(models, resources)
        except Exception as e:
            results["init"] = e
        formatted_results = results["init"]
        formatted_results = {k: str(v) for k, v in formatted_results.items() if k in ["batch_sizes", "endpoints", "hwtest"]} 
        return formatted_results

    async def init_endpoints (self, models: list, resources: dict):
        try:
            return await self.resources["ipfs_accelerate_py"].init_endpoints(models, resources)
        except Exception as e:
            return e
        
    async def testEndpointTask(self, models: list, resources: dict):
        try:
            print(self.resources["endpoint_handler"])
            results = await self.resources["ipfs_accelerate_py"].test_endpoints(models, self.resources["endpoint_handler"])
            print("test_results_keys:")
            print(list(results.keys()))
            return results
        except Exception as e:
            print("error:")
            print(e)
            return e

    async def inferTask(self, models: list, batch_data: list):
        infer_results = {}
        filtered_results = {}
        try:
            infer_results["infer"] = await self.resources["ipfs_accelerate_py"].infer(models, batch_data)
        except Exception as e:
            infer_results["infer"] = e
        if type(infer_results["infer"]) == ValueError:
            filtered_results = {"error": str(infer_results["infer"])}
        elif type(infer_results["infer"]["infer"]) == ValueError:
            filtered_results = {"error": str(infer_results["infer"]["infer"])}
        elif "infer" in list(infer_results["infer"].keys()):
            try:
                if type(infer_results["infer"]["infer"]) == list:
                    filtered_results = infer_results["infer"]["infer"]
                else:
                    for key in list(infer_results["infer"]["infer"].keys()):
                        if type(infer_results["infer"]["infer"][key]) == Tensor:
                            filtered_results[key] = infer_results["infer"]["infer"][key].tolist()
                        else:
                            filtered_results[key] = infer_results["infer"]["infer"][key]
            except Exception as e:
                print(type(infer_results["infer"]["infer"]))
                print(e)
        return filtered_results

    async def statusTask(self, models: list):
        status_results = {}
        try:
            status_results["status"] = await self.resources["ipfs_accelerate_py"].status()
        except Exception as e:
            status_results["status"] = e
        filtered_results = {k: str(v) for k, v in status_results["status"].items() if k in ["batch_sizes", "endpoints", "hwtest"]}
        return filtered_results

    async def addEndpointTask(self, models: list, endpoint_type: str, endpoint: list):
        add_endpoint_results = {}
        try:
            add_endpoint_results["add_endpoint"] = await self.resources["ipfs_accelerate_py"].add_endpoint(models, endpoint_type, endpoint)
        except Exception as e:
            add_endpoint_results["add_endpoint"] = e
        await self.initEndpointsTask([models], self.resources)
        return add_endpoint_results

    async def rmEndpointTask(self, models: list, endpoint_type: str, backend: str):
        rm_endpoint_results = {}
        try:
            rm_endpoint_results["rm_endpoint"] = await self.resources["ipfs_accelerate_py"].rm_endpoint(models, endpoint_type, backend)
        except Exception as e:
            rm_endpoint_results["rm_endpoint"] = e
        return rm_endpoint_results

model_server = ModelServer()

initEndpointsTask = model_server.initEndpointsTask
testEndpointTask = model_server.testEndpointTask

@app.post("/add_endpoint")
async def add_endpoint(request: AddEndpointRequest, background_tasks: BackgroundTasks):
    add_endpoint_results = {}
    try:
        add_endpoint_results["add_endpoint"] = await model_server.addEndpointTask(request.model, request.endpoint_type, request.endpoint)
        return {"message", json.dumps(add_endpoint_results) }
    except Exception as e:
        add_endpoint_results["add_endpoint"] = e
        return {"message", json.dumps(add_endpoint_results) }

@app.post("/rm_endpoint")
async def rm_endpoint(request: RmEndpointRequest, background_tasks: BackgroundTasks):
    rm_endpoint_results = {}
    try:
        rm_endpoint_results["rm_endpoint"] = model_server.rmEndpointTask(request.model, request.endpoint_type, request.index)
        return {"message", json.dumps(rm_endpoint_results)}
    except Exception as e:
        rm_endpoint_results["rm_endpoint"] = e
        return {"message", json.dumps(rm_endpoint_results)}

@app.post("/init")
async def load_index_post(request: InitEndpointsRequest, background_tasks: BackgroundTasks):
    results = {}
    try:
        results["init"] = await model_server.initEndpointsTask(request.models, request.resources)
        return {"message": json.dumps(results)}
    except Exception as e:
        results["init"]  = e
        return {"message": json.dumps(results)}

@app.post("/status")
async def status_post(request: InitStatusRequest, background_tasks: BackgroundTasks):
    results = {}
    try:
        results["status"] = await model_server.statusTask(request.models)
        return {"message": json.dumps(results)}
    except Exception as e:
        results["status"]  = e
        return {"message": json.dumps(results)}

@app.post("/test")
async def search_item_post(request: TestEndpointRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(model_server.testEndpointTask, request.models, request.resources)
    return {"message": "test started in the background"}

@app.post("/infer")
async def infer(request: InferEndpointRequest, background_tasks: BackgroundTasks):
    infer_results = {}
    try:
        infer_results["infer"] = await model_server.inferTask(request.models, request.batch_data) 
    except Exception as e:
        infer_results["infer"] = e
    return {"message": json.dumps(infer_results)}

@app.post("/")
async def help():
    return {"message": "Please use /init or /test endpoints"}

uvicorn.run(app, host="0.0.0.0", port=9999)