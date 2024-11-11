from typing import Union
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from ipfs_accelerate_py import ipfs_accelerate_py
from pydantic import BaseModel
    
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
    models: str
    resources: dict[str, list[list[str]]]

class RmEndpointRequest(BaseModel):
    models: str
    endpoint_type: str

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
            
        try:
            results["test"] = await self.testEndpointTask(models, resources)
        except Exception as e:
            results["test"] = e
        
        try:
            results["status"] = await self.statusTask(models)
        except Exception as e:
            results["status"] = e
        return results

    async def init_endpoints (self, models: list, resources: dict):
        try:
            return await self.resources["ipfs_accelerate_py"].init_endpoints(models, resources)
        except Exception as e:
            return e
        
    async def testEndpointTask(self, models: list, resources: dict):
        try:
            return await self.resources["ipfs_accelerate_py"].test_endpoints(models, resources)
        except Exception as e:
            return e

    async def inferTask(self, models: list, batch_data: list):
        ipfs_accelerate_py.infer(models, batch_data)
        return None

    async def statusTask(self, models: list):
        return ipfs_accelerate_py.status(models)

    async def addEndpointTask(self, models: list, endpoint_type: str, endpoint: list):
        ipfs_accelerate_py.add_endpoint(models, endpoint_type, endpoint)
        return None

    async def rmEndpointTask(self, models: list, endpoint_type: str, index: int):
        ipfs_accelerate_py.rm_endpoint(models, endpoint_type, index)
        return None

model_server = ModelServer()

@app.get("/add_endpoint")
async def add_endpoint(request: AddEndpointRequest, background_tasks: BackgroundTasks):
    BackgroundTasks.add_task(model_server.addEndpointTask, request.models, request.resources)
    return {"message": "add endpoint started in the background"}

@app.get("/rm_endpoint")
async def rm_endpoint(request: RmEndpointRequest, background_tasks: BackgroundTasks):
    BackgroundTasks.add_task(model_server.rmEndpointTask, request.models)
    return {"message": "rm endpoint started in the background"}

@app.get("/status")
async def status_post(request: InitStatusRequest, background_tasks: BackgroundTasks):
    BackgroundTasks.add_task(model_server.statusTask, request.models)
    return {"message": "status started in the background"}

@app.post("/init")
async def load_index_post(request: InitEndpointsRequest, background_tasks: BackgroundTasks):
    results = {}
    try:
        results["init"] = await model_server.initEndpointsTask(request.models, request.resources)
    except Exception as e:
        results["init"]  = e
    return results
    # BackgroundTasks.add_task(initEndpointsTask, request.models , request.resources)
    # return {"message": "init loading started in the background"}

@app.post("/test")
async def search_item_post(request: TestEndpointRequest, background_tasks: BackgroundTasks):
    BackgroundTasks.add_task(model_server.testEndpointTask, request.models, request.resources)
    return {"message": "test started in the background"}

@app.post("/infer")
async def infer(request: InferEndpointRequest, background_tasks: BackgroundTasks):
    BackgroundTasks.add_task(model_server.inferTask, request.models, request.batch_data)
    return {"message": "infer started in the background"}

@app.post("/")
async def help():
    return {"message": "Please use /init or /test endpoints"}

uvicorn.run(app, host="0.0.0.0", port=9999)