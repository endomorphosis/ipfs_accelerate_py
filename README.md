IPFS ACCELERATE
==

This is meant to be an extension of the Huggingface accelerate library, with the intent that the method load_checkpoint_and_dispatch() is going to be overloaded with a new function which will have some modular back ends, such as Libp2p, Akash, Lilypad, Huggingface Zero, Vast AI. If the model is already listed in the ipfs_model_manager there should be an associated hw_requirements key of the manifest. In the case of libp2p the request to do inference will go out to peers if there are peers in the network available to fulfill the task, otherwise a docker container will need to be launched with one of the providers here. The docker containers will will then be loaded with the ipfs_model_manager, and will use the from_auto_download() method to download the requested model and perform inference for the user.

Method #1 Huggingface Zero
==
Method #2 Akash
==
Method #3 Lilypad
==
Method #4 Vast AI
==
