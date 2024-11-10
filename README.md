# IPFS ACCELERATE

This is meant to be an extension of the Huggingface accelerate library, is to act as a model server, which can contain lists of other endpoints to call, or can call a local instance, and can respond to external calls for inference. I have some modular back ends, such as Libp2p, Akash, Lilypad, Huggingface Zero, Vast AI, which I use for autoscaling. If the model is already listed in the ipfs_model_manager there should be an associated hw_requirements key of the manifest. In the case of libp2p the request to do inference will go out to peers in the same trusted zone, if there are no peers in the network available to fulfill the task, and it has the resources to run the model localliy it will do so, otherwise a docker container will need to be launched with one of the providers here. 

# BACKENDS
You can spin up additional model endpoints with the following:

# Method #1 Huggingface Hugs

# Method #2 Akash

# Method #3 Lilypad

# Method #4 Vast AI

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
https://github.com/endomorphosis/fireproof_kit

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

Author - Benjamin Barber
QA - Kevin De Haan
