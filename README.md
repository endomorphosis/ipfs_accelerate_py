# IPFS ACCELERATE

This is meant to be an extension of the Huggingface accelerate library, is to act as a model server, which can contain lists of other endpoints to call, or can call a local instance, and can respond to external calls for inference. I have some modular back ends, such as Libp2p, Akash, Lilypad, Huggingface Zero, Vast AI, which I use for autoscaling. If the model is already listed in the ipfs_model_manager there should be an associated hw_requirements key of the manifest. In the case of libp2p the request to do inference will go out to peers in the same trusted zone, if there are no peers in the network available to fulfill the task, and it has the resources to run the model localliy it will do so, otherwise a docker container will need to be launched with one of the providers here. 

# IPFS Huggingface Bridge:

for huggingface transformers python library visit:
https://github.com/endomorphosis/ipfs_transformers/

for huggingface datasets python library visit:
https://github.com/endomorphosis/ipfs_datasets/

for faiss KNN index python library visit:
https://github.com/endomorphosis/ipfs_faiss

for transformers.js visit:                          
https://github.com/endomorphosis/ipfs_transformers_js

for orbitdb_kit nodejs library visit:
https://github.com/endomorphosis/orbitdb_kit/

for fireproof_kit nodejs library visit:
https://github.com/endomorphosis/fireproof_kit/

for ipfs_kit nodejs library visit:
https://github.com/endomorphosis/ipfs_kit/

for python model manager library visit: 
https://github.com/endomorphosis/ipfs_model_manager/

for nodejs model manager library visit: 
https://github.com/endomorphosis/ipfs_model_manager_js/

for nodejs ipfs huggingface scraper with pinning services visit:
https://github.com/endomorphosis/ipfs_huggingface_scraper/

for ipfs agents visit:
https://github.com/endomorphosis/ipfs_agents/

for ipfs accelerate visit:
https://github.com/endomorphosis/ipfs_accelerate/

# IPFS Accelerate Python

Author - Benjamin Barber
QA - Kevin De Haan

## API Backends

### OpenVINO Model Server (OVMS) Backend
The OVMS backend provides integration with OpenVINO Model Server deployments. Features:
- Any OpenVINO-supported model type (classification, NLP, vision, speech)
- Both sync and async inference modes 
- Automatic input handling and tokenization
- Custom pre/post processing pipelines
- Batched inference support
- Multiple precision support (FP32, FP16, INT8)

Example usage:
```python
from ipfs_accelerate_py.api_backends import ovms

# Initialize backend
ovms_backend = ovms()

# For text/NLP models
endpoint_url, api_key, handler, queue, batch_size = ovms_backend.init(
    endpoint_url="http://localhost:9000",
    model_name="gpt2",
    context_length=1024
)

response = handler("What is quantum computing?")

# For vision models with custom preprocessing
def preprocess_image(image_data):
    # Convert image to model input format
    return processed_data

handler = ovms_backend.create_remote_ovms_endpoint_handler(
    endpoint_url="http://localhost:9000",
    model_name="resnet50",
    preprocessing=preprocess_image
)

result = handler(image_data, parameters={"raw": True})

# For async high-throughput inference
async_handler = await ovms_backend.create_async_ovms_endpoint_handler(
    endpoint_url="http://localhost:9000",
    model_name="bert-base"
)

results = await asyncio.gather(
    async_handler(batch1),
    async_handler(batch2)
)
```

### Common Features
All backends support:
