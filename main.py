from typing import Union
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from ipfs_accelerate_py import ipfs_accelerate_py

class TestEndpointRequest(BaseModel):
    models: list[str]
    resources: dict[str, list[list[(str,str,int)]]]
    
class InitEndpointsRequest(BaseModel):
    models: list
    resources: dict[str, list[list[(str,str,int)]]]
    
class InferEndpointRequest(BaseModel):
    models: list
    batch_data: list[str]

class AddEndpointRequest(BaseModel):
    models: str
    resources: dict[str, list[list[(str,str,int)]]]

class RmEndpointRequest(BaseModel):
    models: str
    endpoint_type: str

class InitStatusRequest(BaseModel):    
    models: list[str]
    
app = FastAPI(port=9999)
resources = {}
metadata = {}
ipfs_accelererate = ipfs_accelerate_py(resources, metadata)

class ModelServer:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata
        self.ipfs_accelerate = ipfs_accelerate_py(self.resources, self.metadata)

    def initEndpointsTask(models: list, resources: dict):
        ipfs_accelerate_py.init_endpoints(models, resources)
        return None

    def testEndpointTask(models: list, resources: dict):
        ipfs_accelerate_py.test_endpoints(models, resources)
        return None

    def inferTask(models: list, batch_data: list):
        ipfs_accelerate_py.infer(models, batch_data)
        return None

    def statusTask(models: list):
        return ipfs_accelerate_py.status(models)

    def addEndpointTask(models: list, endpoint_type: str, endpoint: list):
        ipfs_accelerate_py.add_endpoint(models, endpoint_type, endpoint)
        return None

    def rmEndpointTask(models: list, endpoint_type: str, index: int):
        ipfs_accelerate_py.rm_endpoint(models, endpoint_type, index)
        return None

@app.get("/add_endpoint")
async def add_endpoint(request: AddEndpointRequest, background_tasks: BackgroundTasks):
    BackgroundTasks.add_task(addEndpointTask, request.models, request.resources)
    return {"message": "add endpoint started in the background"}

@app.get("/rm_endpoint")
async def rm_endpoint(request: RmEndpointRequest, background_tasks: BackgroundTasks):
    BackgroundTasks.add_task(rmEndpointTask, request.models)
    return {"message": "rm endpoint started in the background"}

@app.get("/status")
async def status_post(request: InitStatusRequest, background_tasks: BackgroundTasks):
    BackgroundTasks.add_task(statusTask, request.models)
    return {"message": "status started in the background"}

@app.post("/init")
async def load_index_post(request: InitEndpointsRequest, background_tasks: BackgroundTasks):
    results = {}
    results["init"] = await initEndpointsTask
    await initEndpointsTask(request.models , request.resources)
    # BackgroundTasks.add_task(initEndpointsTask, request.models , request.resources)
    # return {"message": "init loading started in the background"}

@app.post("/test")
async def search_item_post(request: TestEndpointRequest, background_tasks: BackgroundTasks):
    BackgroundTasks.add_task(testEndpointTask, request.models, request.resources)
    return {"message": "test started in the background"}

@app.post("/infer")
async def infer(request: InferEndpointRequest, background_tasks: BackgroundTasks):
    BackgroundTasks.add_task(inferTask, request.models, request.batch_data)
    return {"message": "infer started in the background"}

@app.post("/")
async def help():
    return {"message": "Please use /init or /test endpoints"}

uvicorn.run(app, host="0.0.0.0", port=9999)