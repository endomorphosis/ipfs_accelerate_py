from typing import Union
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from ipfs_accelerate_py import ipfs_accelerate_py

class TestEndpointRequest(BaseModel):
    models: List[str]
    resources: Dict[str, List[str]]
    
class InitEndpointsRequest(BaseModel):
    models: List[str]
    resources: Dict[str, List[str]]
    
metadata = {
    "dataset": "TeraflopAI/Caselaw_Access_Project",
    "column": "text",
    "split": "train",
    "models": [
        "thenlper/gte-small",
        # "Alibaba-NLP/gte-large-en-v1.5",
        # "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    ],
    "chunk_settings": {
        "chunk_size": 512,
        "n_sentences": 8,
        "step_size": 256,
        "method": "fixed",
        "embed_model": "thenlper/gte-small",
        "tokenizer": None
    },
    "dst_path": "/storage/teraflopai/tmp",
}
resources = {
    "local_endpoints": [
        ["thenlper/gte-small", "cpu", 512],
        ["Alibaba-NLP/gte-large-en-v1.5", "cpu", 8192],
        ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "cpu", 32768],
        ["thenlper/gte-small", "cuda:0", 512],
        ["Alibaba-NLP/gte-large-en-v1.5", "cuda:0", 8192],
        ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "cuda:0", 32768],
        ["thenlper/gte-small", "cuda:1", 512],
        ["Alibaba-NLP/gte-large-en-v1.5", "cuda:1", 8192],
        ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "cuda:1", 32768],
        ["thenlper/gte-small", "openvino", 512],
        ["Alibaba-NLP/gte-large-en-v1.5", "openvino", 8192],
        ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "openvino", 32768],
        ["thenlper/gte-small", "llama_cpp", 512],
        ["Alibaba-NLP/gte-large-en-v1.5", "llama_cpp", 8192],
        ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "llama_cpp", 32768],
        ["thenlper/gte-small", "ipex", 512],
        ["Alibaba-NLP/gte-large-en-v1.5", "ipex", 8192],
        ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "ipex", 32768],
    ],
    "openvino_endpoints": [
        # ["neoALI/bge-m3-rag-ov", "https://bge-m3-rag-ov-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-rag-ov/infer", 4095],
        # ["neoALI/bge-m3-rag-ov", "https://bge-m3-rag-ov-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-rag-ov/infer", 4095],
        # ["neoALI/bge-m3-rag-ov", "https://bge-m3-rag-ov-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-rag-ov/infer", 4095],
        # ["neoALI/bge-m3-rag-ov", "https://bge-m3-rag-ov-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-rag-ov/infer", 4095],
        # ["aapot/bge-m3-onnx", "https://bge-m3-onnx0-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-onnx0/infer", 1024],
        # ["aapot/bge-m3-onnx", "https://bge-m3-onnx1-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-onnx1/infer", 1024],
        # ["aapot/bge-m3-onnx", "https://bge-m3-onnx2-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-onnx2/infer", 1024],
        # ["aapot/bge-m3-onnx", "https://bge-m3-onnx3-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-onnx3/infer", 1024],
        # ["aapot/bge-m3-onnx", "https://bge-m3-onnx4-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-onnx4/infer", 1024],
        # ["aapot/bge-m3-onnx", "https://bge-m3-onnx5-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-onnx5/infer", 1024],
        # ["aapot/bge-m3-onnx", "https://bge-m3-onnx6-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-onnx6/infer", 1024],
        # ["aapot/bge-m3-onnx", "https://bge-m3-onnx7-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-onnx7/infer", 1024]
    ],
    "tei_endpoints": [
        ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "http://62.146.169.111:8080/embed-medium", 32768],
        ["thenlper/gte-small", "http://62.146.169.111:8080/embed-tiny", 512],
        ["Alibaba-NLP/gte-large-en-v1.5", "http://62.146.169.111:8081/embed-small", 8192],
        ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "http://62.146.169.111:8081/embed-medium", 32768],
        ["thenlper/gte-small", "http://62.146.169.111:8081/embed-tiny", 512],
        # ["Alibaba-NLP/gte-large-en-v1.5", "http://62.146.169.111:8082/embed-small", 8192],
        # ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "http://62.146.169.111:8082/embed-medium", 32768],
        # ["thenlper/gte-small", "http://62.146.169.111:8082/embed-tiny", 512],
        # ["Alibaba-NLP/gte-large-en-v1.5", "http://62.146.169.111:8083/embed-small", 8192],
        # ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "http://62.146.169.111:8083/embed-medium", 32768],
        # ["thenlper/gte-small", "http://62.146.169.111:8083/embed-tiny", 512]
    ]
}


app = FastAPI(port=9999)

def initEndpointsTask(BaseModel):
    
    return None



def testEndpointTask(BaseModel):
        
    return None

@app.post("/init")
async def load_index_post(request: InitEndpointsRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(initEndpointsTask, request.dataset, request.knn_index, request.dataset_split, request.knn_index_split, request.columns)
    return {"message": "Index loading started in the background"}

@app.post("/test")
async def search_item_post(request: TestEndpointRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(testEndpointTask, request.collection, request.text, request.n)
    return {"message": "Search started in the background"}

@app.post("/")
async def help():
    return {"message": "Please use /init or /test endpoints"}

uvicorn.run(app, host="0.0.0.0", port=9999)