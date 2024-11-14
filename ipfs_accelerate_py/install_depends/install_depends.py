import subprocess
import os
import sys

class install_depends_py():
    def __init__(self, resources, metadata):
        self.resources = resources
        self.metadata = metadata
        self.results = {}
        self.stdout = {}
        self.stderr = {}
        self.install_results = {}   
        if "test_ipfs_accelerate" not in globals() and "test_ipfs_accelerate" not in list(self.resources.keys()):
            import test_ipfs_accelerate
            self.test_ipfs_accelerate = test_ipfs_accelerate.test_ipfs_accelerate(resources, metadata)
        elif "test_ipfs_accelerate" in list(self.resources.keys()): 
            self.test_ipfs_accelerate = self.resources["test_ipfs_accelerate"]
        elif "test_ipfs_accelerate" in globals():
            self.test_ipfs_accelerate = test_ipfs_accelerate(resources, metadata)

    
    async def install(self, resources=None):        
        if resources is None:
            if self.resources is not None and len(list(self.resources.keys())) != 0:
                resources = self.resources
                if "packages" in list(self.resources.keys()):
                    resources["packages"] = self.resources["packages"]
                else:
                    resources["packages"] = [[]]
            else:
                resources["packagess"] = [[]]
            pass
        for package in resources["packages"]:
            try:
                self.stdout[package] = await self.install_package(package)
            except Exception as e:
                self.stderr[package] = e
                print(e)
            install_results = [ stdout if stdout else stderr for stdout, stderr in zip(self.stdout, self.stderr) ] 
        return install_results

    async def install_package(self, package):
        if package == "cuda":
            return await self.install_cuda()
        elif package == "openvino":
            return await self.install_openvino()
        elif package == "llama_cpp":
            return await self.install_llama_cpp()
        elif package == "ollama":
            return await self.install_ollama()
        elif package == "ipex":
            return await self.install_ipex()
        elif package == "tortch":
            return await self.install_torch()
        elif package == "storacha":
            return await self.install_storacha()
        elif package == "ipfs":
            return await self.install_ipfs_kit()
        elif package == "libp2p":
            return await self.install_libp2p_kit()
        elif package == "faiss":
            return await self.install_faiss()
        elif package == "faiss-cuda":
            return await self.install_faiss_cuda()
        elif package == "faiss-amx":
            return await self.install_faiss_amx()
        elif package == "qdrant":
            return await self.install_qdrant()
        elif package == "elasticsearch":
            return await self.install_elasticsearch()
        elif package == "oneccl_bind_pt":
            return await self.install_oneccl_bind_pt()
        elif package == "oneccl_bind_pt_git":
            return await self.install_oneccl_bind_pt_git()
        elif package == "huggingface_optimum":
            return await self.install_huggingface_optimum()
        elif package == "huggingface_optimum_openvino":
            return await self.install_huggingface_optimum_openvino()
        elif package == "huggingface_optimum_ipex":
            return await self.install_huggingface_optimum_ipex()
        elif package == "huggingface_optimum_habana":
            return await self.install_huggingface_optimum_habana()
        elif package == "huggingface_optimum_onnx":
            return await self.intstall_huggingface_optimum_onnx()
        elif package == "huggingface_optimum_cuda":
            return await self.install_huggingface_optimum_cuda()
        elif package == "optimum":          
            return await self.install_huggingface_optimum()
        elif package == "optimum_amx":
            return await self.install_huggingface_optimum_amx()
        elif package == "webnn":
            return await self.install_webnn()
        elif package == "all":
            return [ all(await self.install_package(package) for package in self.resources["packages"]) ]
        elif type(package) == list:
            return [ all(await self.install_package(package) for package in self.resources["packages"]) ]
        else:
            return None
        
    async def test_package(self, package):
        if package == "cuda":
            return await self.test_cuda()
        elif package == "optimum":
            return await self.test_ipfs_accelerate()
        elif package == "openvino":
            return await self.test_openvino()
        elif package == "llama_cpp":
            return await self.test_llama_cpp()
        elif package == "ollama":
            return await self.test_ollama()
        elif package == "ipex":
            return await self.test_ipex()
        elif package == "torch":
            return await self.test_torch()
        elif package == "torch_vision":
            return await self.test_torch_vision()
        elif package == "storacha":
            return await self.test_storacha()
        elif package == "ipfs":
            return await self.test_ipfs_kit()
        elif package == "libp2p":
            return await self.test_libp2p_kit()
        elif package == "faiss":
            return await self.test_faiss()
        elif package == "faiss-cuda":
            return await self.test_faiss_cuda()
        elif package == "faiss-amx":
            return await self.test_faiss_amx()
        elif package == "qdrant":
            return await self.test_qdrant()
        elif package == "elasticsearch":
            return await self.test_elasticsearch()
        elif package == "oneccl_bind_pt":
            return await self.test_oneccl_bind_pt()
        elif package == "oneccl_bind_pt_git":
            return await self.test_oneccl_bind_pt_git()
        elif package == "huggingface_optimum":
            return await self.test_huggingface_optimum()
        elif package == "huggingface_optimum_openvino":
            return await self.test_huggingface_optimum_openvino()
        elif package == "huggingface_optimum_ipex":
            return await self.test_huggingface_optimum_ipex()
        elif package == "huggingface_optimum_habana":
            return await self.test_huggingface_optimum_habana()
        elif package == "huggingface_optimum_onnx":
            return await self.test_huggingface_optimum_onnx()
        elif package == "huggingface_optimum_cuda":
            return await self.test_huggingface_optimum_cuda()
        elif package == "huggingface_optimum_amx":
            return await self.test_huggingface_optimum_amx()
        elif package == "webnn":
            return await self.test_webnn()
        else:
            return None
        
    async def install_libp2p_kit(self):
        
        return None
    
    async def test_torch_vision(self):
        test_torch_vision_cmd = "python3 -c 'import torchvision; print(torchvision.__version__)'"
        try:
            test_torch_vision = subprocess.check_output(test_torch_vision_cmd, shell=True).decode("utf-8")
            if type(test_torch_vision) == str and type(test_torch_vision) != ValueError:
                return True
            else:
                return False
        except Exception as e:
            print(e)
            raise ValueError(e)
        return None
    
    async def test_torch(self):
        test_torch_cmd = "python3 -c 'import torch; print(torch.__version__)'"
        try:
            test_torch = subprocess.check_output(test_torch_cmd, shell=True).decode("utf-8")
            if type(test_torch) == str and type(test_torch) != ValueError:
                return True
            else:
                return False
        except Exception as e:
            print(e)
            raise ValueError(e)
        return None
    
    async def install_torch_vision(self):
        install_results = {}
        try:
            install_cmd = ["pip", "install", "torchvision", "--break-system-packages"]
            result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
            install_results["torch_vision"] = result.stdout
        except subprocess.CalledProcessError as e:
            install_results["torch_vision"] = e.stderr
            print(f"Failed to install Torch Vision: {e.stderr}")
        return install_results
    
    async def install_torch(self):
        install_results = {}
        try:
            install_cmd = ["pip", "install", "torch", "--break-system-packages"]
            result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
            install_results["torch"] = result.stdout
        except subprocess.CalledProcessError as e:
            install_results["torch"] = e.stderr
            print(f"Failed to install Torch: {e.stderr}")
        return install_results
    
    async def test_ollama(self):
        test_ollama_cmd = "ollama -v"
        try:
            test_ollama = subprocess.check_output(test_ollama_cmd, shell=True).decode("utf-8")
            if type(test_ollama) == str and type(test_ollama) != ValueError:
                return True
            else:
                return False
        except Exception as e:
            print(e)
            raise ValueError(e)
        return None
    
    async def test_openvino(self):
        test_openvino_cmd = "python3 -c 'import openvino; print(openvino.__version__)'"
        try:
            test_openvino = subprocess.check_output(test_openvino_cmd, shell=True).decode("utf-8")              
            if type(test_openvino) == str and type(test_openvino) != ValueError:
                return True
            else:
                return False
        except Exception as e:
            print(e)
            raise ValueError(e)
        return None
    
    async def test_huggingface_optimum_cuda(self):
        test_optimum_cuda_cmd = "python3 -c 'import transformers; print(transformers.__version__)'"
        try:
            test_optimum_cuda = subprocess.check_output(test_optimum_cuda_cmd, shell=True).decode("utf-8")
            if type(test_optimum_cuda) == str and type(test_optimum_cuda) != ValueError:
                return True
            else:
                return False
        except Exception as e:
            print(e)
            raise ValueError(e)
        return None
    
    async def test_huggingface_optimum_onnx(self):
        test_optimum_onnx_cmd = "python3 -c 'import transformers; print(transformers.__version__)'"
        try:
            test_optimum_onnx = subprocess.check_output(test_optimum_onnx_cmd, shell=True).decode("utf-8")
            if type(test_optimum_onnx) == str and type(test_optimum_onnx) != ValueError:
                return True
            else:
                return False
        except Exception as e:
            print(e)
            raise ValueError(e)
        return None
    
    async def test_llama_cpp(self):
        test_llama_cpp_cmd = "pip show llama-cpp-python"
        test_results = {}
        try:
            test_llama_cpp = subprocess.check_output(test_llama_cpp_cmd, shell=True)
            test_llama_cpp = test_llama_cpp.decode("utf-8")
            test_results["llama_cpp"] = test_llama_cpp
        except Exception as e:
            print(e)
            raise ValueError(e)
        try:
            test_ollama = subprocess.check_output("ollama -v", shell=True)
            test_ollama = test_ollama.decode("utf-8")
            test_results["ollama"] = test_ollama
        except Exception as e:
            print(e)
            raise ValueError(e)
        
        test_pass = False
        test_pass = all(isinstance(value, str) for value in test_results.values() if not isinstance(value, ValueError))
        return test_pass
        
    
    async def test_local_openvino(self):
        test_openvino_cmd = "python3 -c 'import openvino; print(openvino.__version__)'"
        try:
            test_openvino = subprocess.check_output(test_openvino_cmd, shell=True).decode("utf-8")              
            if type(test_openvino) == str and type(test_openvino) != ValueError:
                return True
            else:
                return False
        except Exception as e:
            print(e)
            raise ValueError(e)
        return None
    
    async def test_ipex(self):        
        test_ipex_cmd = 'python -c "import torch; import intel_extension_for_pytorch as ipex; print(torch.__version__); print(ipex.__version__);"'
        test_ipex = None
        try:
            # test_ipex = subprocess.check_output(test_ipex_cmd, shell=True)
            return test_ipex
        except Exception as e:
            print(e)
            raise ValueError(e)
        return None
    
    async def test_huggingface_optimum(self):
        test_optimum_cmd = "python3 -c 'import transformers; print(transformers.__version__)'"
        try:
            test_optimum = subprocess.check_output(test_optimum_cmd, shell=True).decode("utf-8")
            if type(test_optimum) == str and type(test_optimum) != ValueError:
                return True
            else:
                return False
        except Exception as e:
            print(e)
            raise ValueError(e)
        return None
   
    async def test_huggingface_optimum_amx(self):
        import optimum
        test_optimum_amx_cmd = optimum.AMX()
        try:
            test_optimum_amx = subprocess.check_output(test_optimum_amx_cmd, shell=True).decode("utf-8")
            if type(test_optimum_amx) == str and type(test_optimum_amx) != ValueError:
                return True
            else:
                return False
        except Exception as e:
            print(e)
            raise ValueError(e)
        return None
   
    async def test_huggingface_optimum_habana(self):
        test_optimum_habana_cmd = "python3 -c 'import transformers; print(transformers.__version__)'"
        try:
            test_optimum_habana = subprocess.check_output(test_optimum_habana_cmd, shell=True).decode("utf-8")
            if type(test_optimum_habana) == str and type(test_optimum_habana) != ValueError:
                return True
            else:
                return False
        except Exception as e:
            print(e)
            raise ValueError(e)
        return None
    
    async def test_huggingface_optimum_neural_compressor(self):
        test_optimum_neural_compressor_cmd = "python3 -c 'import transformers; print(transformers.__version__)'"
        try:
            test_optimum_neural_compressor = subprocess.check_output(test_optimum_neural_compressor_cmd, shell=True).decode("utf-8")
            if type(test_optimum_neural_compressor) == str and type(test_optimum_neural_compressor) != ValueError:
                return True
            else:
                return False
        except Exception as e:
            print(e)
            raise ValueError(e)
        return None
    
    async def test_huggingface_optimum_openvino(self):
        test_optimum_openvino_cmd = "python3 -c 'import transformers; print(transformers.__version__)'"
        try:
            test_optimum_openvino = subprocess.check_output(test_optimum_openvino_cmd, shell=True).decode("utf-8")
            if type(test_optimum_openvino) == str and type(test_optimum_openvino) != ValueError:
                return True
            else:
                return False
        except Exception as e:
            print(e)
            raise ValueError(e)
        return None
    
   
    async def test_huggingface_optimum_ipex(self):
        test_optimum_intel_cmd = "python3 -c 'import transformers; print(transformers.__version__)'"
        try:
            test_optimum_intel = subprocess.check_output(test_optimum_intel_cmd, shell=True).decode("utf-8")
            if type(test_optimum_intel) == str and type(test_optimum_intel) != ValueError:
                return True
            else:
                return False
        except Exception as e:
            print(e)
            raise ValueError(e)
        return None 
    
    async def test_cuda(self):
        try:
            import torch
            gpus = torch.cuda.device_count()
            if type(gpus) == int and type(gpus) != ValueError:
                return True
            else:
                return False
        except Exception as e:
            print(e)
            raise ValueError(e)
    
    async def test_onnx(self):
        test_onnx_cmd = "python3 -c 'import onnx; print(onnx.__version__)'"
        try:
            test_onnx = subprocess.check_output(test_onnx_cmd, shell=True).decode("utf-8")
            if type(test_onnx) == str and type(test_onnx) != ValueError:
                return True
            else:
                return False
        except Exception as e:
            print(e)
            raise ValueError(e)
        return None
    
    async def install_onnx(self):
        install_results = {}
        try:
            install_cmd = ["pip", "install", "onnx", "--break-system-packages"]
            result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
            install_results["onnx"] = result.stdout
        except subprocess.CalledProcessError as e:
            install_results["onnx"] = e.stderr
            print(f"Failed to install ONNX: {e.stderr}")
        return install_results
    
    async def install_cuda(self):
        install_results = {}
        try:
            install_cmd = ["pip", "install", "torch", "torchvision", "torchaudio", "torchtext", "--index-url", " https://download.pytorch.org/whl/cpu", "--break-system-packages"]
            print(install_cmd)
            install_results["cuda"] = subprocess.run(install_cmd, check=True)
        except Exception as e:
            install_results["cuda"] = e
            print(e)
        return install_results
    
    async def test_cuda(self):
        try:
            import torch
            gpus = torch.cuda.device_count()
            if type(gpus) == int and type(gpus) != ValueError:
                return True
            else:
                return False
        except Exception as e:
            print(e)
            raise ValueError(e)
    
    async def install_faiss(self):
        install_results = {}
        try:
            install_cmd = ["pip", "install", "faiss", "--break-system-packages"]
            result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
            install_results["faiss"] = result.stdout
        except subprocess.CalledProcessError as e:
            install_results["faiss"] = e.stderr
            print(f"Failed to install Faiss: {e.stderr}")
        return install_results
    
    async def install_faiss_cuda(self):
        install_results = {}
        try:
            install_cmd = ["pip", "install", "faiss-cuda", "--break-system-packages"]
            result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
            install_results["faiss_cuda"] = result.stdout
        except subprocess.CalledProcessError as e:
            install_results["faiss_cuda"] = e.stderr
            print(f"Failed to install Faiss CUDA: {e.stderr}")
        return install_results
    
    # async def install_ollama(self):
    #     cmd = "curl -fsSL https://ollama.com/install.sh | sh"
    #     install_results = {}
    #     try:
    #         result = subprocess.check_output(cmd, shell=True, text=True)
    #         install_results["ollama"] = result
    #     except subprocess.CalledProcessError as e:
    #         if e.stderr == None:
    #             install_results["ollama"] = True
    #         else:
    #             install_results["ollama"] = e
    #         # print(f"Failed to install Ollama: {e.stderr}")
    #     return install_results
                

    # async def install_llama_cpp(self):
    #     install_results = {}
    #     try:
    #         # lscpu_cmd = "lscpu"
    #         # lscpu = subprocess.check_output(lscpu_cmd, shell=True, text=True)
    #         # print(lscpu)
    #         inst_avx_cmd = "lscpu | grep avx"
    #         inst_avx2_cmd = "lscpu | grep avx2"
    #         inst_avx_vnni = "lscpu | grep avx_vnni"
    #         inst_amx_cmd = "lscpu | grep amx"
    #         inst_cuda_cmd = "nvidia-smi"
    #         inst_openvino_cmd = "openvino --version"
    #         try:
    #             inst_avx = subprocess.check_output(inst_avx_cmd, shell=True, text=True)
    #         except Exception as e:
    #             inst_avx = e
    #             print(e)
    #         try:
    #             inst_avx2 = subprocess.check_output(inst_avx2_cmd, shell=True, text=True)
    #         except Exception as e:
    #             inst_avx2 = e
    #             print(e)
    #         try:
    #             inst_avx_vnni = subprocess.check_output(inst_avx_vnni, shell=True, text=True)
    #         except Exception as e:
    #             inst_avx_vnni = e
    #             print(e)
    #         try:
    #             inst_amx = subprocess.check_output(inst_amx_cmd, shell=True, text=True)
    #         except Exception as e:
    #             inst_amx = e
    #             print(e)
    #         try:
    #             inst_cuda_cmd = subprocess.check_output(inst_cuda_cmd, shell=True, text=True)
    #         except Exception as e:
    #             inst_cuda_cmd = e
    #             print(e)
    #         try:    
    #             inst_oneapi = subprocess.check_output(inst_openvino_cmd, shell=True, text=True)
    #         except Exception as e:
    #             inst_oneapi = e
    #             print(e)
    #         results = { "avx": inst_avx, "avx2": inst_avx2, "avx_vnni": inst_avx_vnni, "amx": inst_amx, "cuda": inst_cuda_cmd, "openvino": inst_oneapi }
    #         filtered_results = { key: value for key, value in results.items() if type(value) != subprocess.CalledProcessError }
    #         num_gpus = 0
    #         if "cuda" in list(filtered_results.keys()):
    #             num_gpus = len(filtered_results["cuda"])
    #         try:
    #             if num_gpus == 0 and "amx" not in list(filtered_results.keys()):              
    #                 pull_cmd = "git clone https://github.com/ggerganov/llama.cpp ; cd llama.cpp ; make "
    #                 result = subprocess.run(pull_cmd, check=True, capture_output=True, text=True)
    #                 install_results["llama_cpp"] = result.stdout
    #             elif num_gpus == 0 and "amx" in list(filtered_results.keys()):              
    #                 pull_cmd = "git clone https://github.com/ggerganov/llama.cpp ; cd llama.cpp ; make "
    #                 result = subprocess.run(pull_cmd, check=True, capture_output=True, text=True)
    #                 install_results["llama_cpp"] = result.stdout
    #             elif num_gpus > 0 and "amx" not in list(filtered_results.keys()):
    #                 pull_cmd = "git clone https://github.com/ggerganov/llama.cpp ; cd llama.cpp ; make  GGML_CUDA=1 GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 "
    #                 result = subprocess.run(pull_cmd, check=True, capture_output=True, text=True)
    #                 install_results["llama_cpp"] = result.stdout
    #             elif num_gpus > 0 and "amx" in list(filtered_results.keys()):
    #                 pull_cmd = "git clone https://github.com/ggerganov/llama.cpp ; cd llama.cpp ; make "
    #                 result = subprocess.run(pull_cmd, check=True, capture_output=True, text=True)
    #                 install_results["llama_cpp"] = result.stdout            
    #         except Exception as e:
    #             install_results["llama_cpp"] = e.stderr
    #             print(f"Failed to install Llama C++: {e.stderr}")
    #     except Exception as e:
    #         install_results["llama_cpp"] = result.stdout
    #         install_results["llama_cpp"] = ValueError( f"Failed to install Llama C++: {e.stderr}")
    #         print(e)
            
    #     try:
    #         install_results["ollama"] = await self.install_ollama()
    #     except Exception as e:            
    #         install_results["ollama"] = ValueError( f"Failed to install Ollama: {e}")
    #         print(e)
        
    #     install_success = False
    #     install_success = all(type(install_results[package]) != ValueError for package in install_results.keys())
        
    #     return install_success
    
    async def install_ipfs_kit(self):
        return None    
    
    async def install_storacha(self):
        return None
    
    async def install_torch(self):
        ## install torch
        install_results = {}

        try:
            install_torch_cmd = ["pip", "install", "torch", "torchvision", "torchaudio", "torchtext", "--index-url", " https://download.pytorch.org/whl/cpu", "--break-system-packages"]
            print(install_torch_cmd)
            install_results["torch"] = subprocess.run(install_torch_cmd, check=True)
        except Exception as e:
            install_results["torch"] = e
            print(e)
        try:
            import torch
            gpus = torch.cuda.device_count()
            install_results["torch"] = gpus
        except Exception as e:
            install_torch_cmd = ["pip", "install", "torch", "torchvision, torchaudio, torchtext", "--index-url", "https://download.pytorch.org/whl/cu102", "--break-system-packages"]
            result = subprocess.run(install_torch_cmd, check=True, capture_output=True, text=True)
            install_results["torch"] = result.stdout
        except subprocess.CalledProcessError as e:
            install_results["torch"] = e.stderr
            print(f"Failed to install Torch: {e.stderr}")
        return install_results

    async def install_openvino(self):
        install_results = {}
        try:
            install_cmd = ["pip", "install", "openvino", "--break-system-packages"]
            result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
            install_results["openvino"] = result.stdout
        except subprocess.CalledProcessError as e:
            install_results["openvino"] = e.stderr
            print(f"Failed to install OpenVINO: {e.stderr}")

        return install_results

    async def install_dependencies(self, dependencies=None):
        install_results = {}
        for dependency in dependencies:
            try:
                install_cmd = ["pip", "install", dependency]
                result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
                install_results[dependency] = result.stdout
            except subprocess.CalledProcessError as e:
                install_results[dependency] = e.stderr
                print(f"Failed to install {dependency}: {e.stderr}")
        return install_results
    
    async def install_ipex(self):
        install_results = {}
        install_ipex_cmd = "pip install --upgrade --upgrade-strategy eager optimum[ipex]"
        try:
            install_results["install_ipex"] = subprocess.run(install_ipex_cmd, check=True)
        except Exception as e:
            install_results["install_ipex"] = e
            print(e)
        
        # python -m pip install intel-extension-for-pytorch
        # python -m pip install oneccl_bind_pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/
        # install_results["install_torch"] = await self.install_torch()
        # install_results["install_oneccl_bind_pt"] = await self.install_oneccl_bind_pt()
        # try:
        #     install_cmd = ["pip", "install", "intel-pytorch-extension", "--extra-index-url", "https://pytorch-extension.intel.com/release-whl/stable/cpu/us/", "--break-system-packages"]
        #     result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
        #     install_results["ipex"] = result.stdout
        # except subprocess.CalledProcessError as e:
        #     install_results["ipex"] = e.stderr
        #     print(f"Failed to install IPEX: {e.stderr}")
        return install_results
    
    async def install_huggingface_optimum(self):
        install_results = {}
        install_optimum_cmd = ["python", "-m", "pip", "install", "optimum"]
        test_results = {}        
        try:
            install_results["install_huggingface_optimum"] = subprocess.run(install_optimum_cmd, check=True)
        except Exception as e:
            install_results["install_huggingface_optimum"] = e
            print(e)
        if len(list(test_results.keys())) > 0:
            try:
                install_results["install_huggingface_optimum_cuda"] = await self.install_huggingface_optimum_cuda()
            except Exception as e:
                install_results["install_huggingface_optimum_cuda"] = e
                print(e)
                
            try:
                install_results["install_huggingface_optimum_openvino"] = await self.install_huggingface_optimum_openvino()
            except Exception as e:
                install_results["install_huggingface_optimum_openvino"] = e
                print(e)
                
            try:
                install_results["install_huggingface_optimum_intel"] = await self.install_huggingface_optimum_intel()
            except Exception as e:
                install_results["install_huggingface_optimum_intel"] = e
                print(e)    
                
            try:
                install_results["install_huggingface_optimum_habana"] = await self.install_huggingface_optimum_habana()
            except Exception as e:
                install_results["install_huggingface_optimum_habana"] = e
                print(e)
            pass
        else:
            install_results["install_huggingface_optimum_cuda"] = None
            install_results["install_huggingface_optimum_openvino"] = None
            install_results["install_huggingface_optimum_intel"] = None
            install_results["install_huggingface_optimum_habana"] = None
            pass
        
        return install_results              
        
    async def install_huggingface_optimum_neural_compressor(self):
        install_results = {}
        try:
            install_cmd = "pip install --upgrade --upgrade-strategy eager optimum[neural-compressor]"
            result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
            install_results["install_optimum_neural_compressor"] = result.stdout
        except subprocess.CalledProcessError as e:
            install_results["install_optimum_neural_compressor"] = e.stderr
            print(f"Failed to install Optimum Neural Compressor: {e.stderr}")
        return install_results
    
    async def install_huggingface_optimum_cuda(self):   
        install_results = {}
        try:
            install_cmd = "pip install --upgrade --upgrade-strategy eager optimum[cuda]"
            result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
            install_results["install_optimum_cuda"] = result.stdout
        except subprocess.CalledProcessError as e:
            install_results["install_optimum_cuda"] = e.stderr
            print(f"Failed to install Optimum CUDA: {e.stderr}")
        return None
    
    async def install_huggingface_optimum_openvino(self):
        install_results = {}
        try:
            install_cmd = 'pip install --upgrade --upgrade-strategy eager "optimum[openvino]" --break-system-packages'
            install_cmd = ["pip", "install", "--upgrade", "--upgrade-strategy", "eager", "optimum[openvino]", "--break-system-packages"]
            result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
            install_results["install_optimum_openvino"] = result.stdout
        except subprocess.CalledProcessError as e:
            install_results["install_optimum_openvino"] = e.stderr
            print(f"Failed to install Optimum OpenVINO: {e.stderr}")
        return install_results
    
    async def install_ollama_intel_gpu(self):
        install_results = {}
        try:
            install_cmd = "./install_ipex.sh"
            result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
            install_results["install_ollama_intel_gpu"] = result.stdout
        except subprocess.CalledProcessError as e:
            install_results["install_ollama_intel_gpu"] = e.stderr
            print(f"Failed to install Ollama Intel GPU: {e.stderr}")
        return install_results
    
    async def install_huggingface_optimum_ipex(self):
        install_results = {}
        try:
            install_cmd = "pip install --upgrade --upgrade-strategy eager optimum[ipex]"
            result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)    
            install_results["install_optimum_ipex"] = result.stdout
        except subprocess.CalledProcessError as e:
            install_results["install_optimum_ipex"] = e.stderr
            print(f"Failed to install Optimum IPEX: {e.stderr}")
        return install_results
    
    async def install_huggingface_optimum_amx(self):
        install_results = {}
        try:
            install_cmd = "pip install --upgrade --upgrade-strategy eager optimum[amx]"
            result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
            install_results["install_optimum_amx"] = result.stdout
        except subprocess.CalledProcessError as e:
            install_results["install_optimum_amx"] = e.stderr
            print(f"Failed to install Optimum AMX: {e.stderr}")
        return install_results            

    async def install_huggingface_optimum_habana(self):
        install_results = {}
        try:
            install_cmd = "pip install --upgrade --upgrade-strategy eager optimum[habana]"
            result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
            install_results["install_optimum_habana"] = result.stdout
        except subprocess.CalledProcessError as e:
            install_results["install_optimum_habana"] = e.stderr
            print(f"Failed to install Optimum Habana: {e.stderr}")
        return install_results
        
    async def intstall_huggingface_optimum_onnx(self):
        install_results = {}
        try:
            install_cmd = ["pip","install","--upgrade","--upgrade-strategy","eager","optimum[onnx]","--break-system-packages"]
            result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
            install_results["install_optimum_onnx"] = result.stdout
        except subprocess.CalledProcessError as e:
            install_results["install_optimum_onnx"] = e.stderr
            print(f"Failed to install Optimum ONNX: {e.stderr}")
        return install_results
        
    async def install_oneccl_bind_pt_git(self):
        install_results = {}
        commands = [
            "git clone https://github.com/intel/torch-ccl.git",
            "git submodule sync",
            "git submodule update --init --recursive"
        ]
        try:
            result = { }
            if not os.path.exists("torch-ccl"):
                result["clone"] = subprocess.check_output(commands[0], shell=True, text=True)
                os.chdir("torch-ccl")
                result["sync"] = subprocess.check_output(commands[1], shell=True, text=True)
                result["update"] = subprocess.check_output(commands[2], shell=True, text=True)
            else:
                os.chdir("torch-ccl")
                try:
                    result["sync"] = subprocess.check_output(commands[1], shell=True, text=True)
                except subprocess.CalledProcessError as e:
                    result["sync"] = e.stderr
                try:
                    result["update"] = subprocess.check_output(commands[2], shell=True, text=True)
                except subprocess.CalledProcessError as e:
                    result["update"] = e.stderr
                install_results["commands1"] = result
        except subprocess.CalledProcessError as e:
            install_results["commands1"] = e.stderr 
            print(f"Failed to install OneCCL Bind PT: {e.stderr}")
        
        homedir = os.path.expanduser("~")
        get_cwdir = os.getcwd()
        ls_files = os.listdir(get_cwdir)
        commands2 = [
            # for CPU Backend Only
            # "sudo python3 setup.py install",
            # for XPU Backend: use DPC++ Compiler to enable support for Intel XPU
            # build with oneCCL from third party
            # "sudo COMPUTE_BACKEND=dpcpp python3 setup.py install",
            # build with oneCCL from basekit
            "sudo export INTELONEAPIROOT="+ homedir + "/intel/oneapi",
            "sudo USE_SYSTEM_ONECCL=ON COMPUTE_BACKEND=dpcpp python3 setup.py install"
        ]
        results = { }
        for command in commands2:
            command_index =  commands2.index(command) 
            try:
                result = subprocess.check_output(command, shell=True, text=True)
                result[str(command_index)] = result
            except subprocess.CalledProcessError as e:
                result[str(command_index)] = e
                print(f"Failed to install OneCCL Bind PT:")
        install_results["commands2"] = results
        return install_results

    async def install_oneccl_bind_pt(self):
        install_results = {} 
        try:
            install_cmd = ["pip", "install", "oneccl_bind_pt", "--extra-index-url", "https://pytorch-extension.intel.com/release-whl/stable/cpu/us/", "--break-system-packages"]
            result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
            install_results["oneccl_bind_pt"] = result.stdout
        except subprocess.CalledProcessError as e:
            install_results["oneccl_bind_pt"] = e.stderr
            print(f"Failed to install OneCCL Bind PT: {e.stderr}")
            try:    
                install_results["oneccl_bind_pt_git"] = await self.install_oneccl_bind_pt_git()
            except Exception as e:
                install_results["oneccl_bind_pt_git"] = e
                print(e)
            pass
        return install_results

    async def install_cuda(self):
        install_results = {}
        install_cuda_cmd = ["apt-get", "install", "nvidia-cuda-toolkit", "--break-system-packages"]
        try:
            install_results["install_cuda"] = subprocess.run(install_cuda_cmd, check=True)
        except Exception as e:
            install_results["install_cuda"] = e
            print(e)
        try:
            import torch
            torch.cuda.is_available()
            install_results["install_cuda"] = True
        except Exception as e:
            install_results["install_cuda"] = e
            print(e)
            
        install_results["install_cuda"] = None
        
        return None
    
    async def install_faiss(self):
        install_results = {}
        try:
            install_cmd = ["pip", "install", "faiss", "--break-system-packages"]
            result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
            install_results["faiss"] = result.stdout
        except subprocess.CalledProcessError as e:
            install_results["faiss"] = e.stderr
            print(f"Failed to install Faiss: {e.stderr}")
        return install_results
    
    async def install_faiss_cuda(self):
        install_results = {}
        try:
            install_cmd = ["pip", "install", "faiss-cuda", "--break-system-packages"]
            result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
            install_results["faiss_cuda"] = result.stdout
        except subprocess.CalledProcessError as e:
            install_results["faiss_cuda"] = e.stderr
            print(f"Failed to install Faiss CUDA: {e.stderr}")
        return install_results
    
    async def install_faiss_amx(self):
        install_results = {}
        git_src="https://github.com/guangzegu/faiss/tree/main"
        try:
            install_cmd = ["pip", "install", "faiss-amx", "--break-system-packages"]
            result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
            install_results["faiss_amx"] = result.stdout
        except subprocess.CalledProcessError as e:
            install_results["faiss_amx"] = e.stderr
            print(f"Failed to install Faiss AMX: {e.stderr}")
        return install_results
    
    async def install_qdrant(self):
        install_results = {}
        try:
            install_cmd = ["pip", "install", "qdrant", "--break-system-packages"]
            result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
            install_results["qdrant"] = result.stdout
        except subprocess.CalledProcessError as e:
            install_results["qdrant"] = e.stderr
            print(f"Failed to install Qdrant: {e.stderr}")
        return install_results
    
    async def install_elasticsearch(self):
        install_results = {}
        try:
            install_cmd = ["pip", "install", "elasticsearch", "--break-system-packages"]
            result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
            install_results["elasticsearch"] = result.stdout
        except subprocess.CalledProcessError as e:
            install_results["elasticsearch"] = e.stderr
            print(f"Failed to install Elasticsearch: {e.stderr}")
        return None

    async def install_numpy(self):
        install_results = {}
        try:
            install_cmd = ["pip", "install", "numpy", "--break-system-packages"]
            result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
            install_results["numpy"] = result.stdout
        except subprocess.CalledProcessError as e:
            install_results["numpy"] = e.stderr
            print(f"Failed to install NumPy: {e.stderr}")
        return install_results
    
    async def install_onnx(self):
        install_results = {}
        try:
            install_cmd = ["pip", "install", "onnx", "--break-system-packages"]
            result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
            install_results["onnx"] = result.stdout
        except subprocess.CalledProcessError as e:
            install_results["onnx"] = e.stderr
            print(f"Failed to install ONNX: {e.stderr}")
        return install_results
    
    async def install_torch_vision(self):
        install_results = {}
        try:
            install_cmd = ["pip", "install", "torchvision", "--break-system-packages"]
            result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
            install_results["torch_vision"] = result.stdout
        except subprocess.CalledProcessError as e:
            install_results["torch_vision"] = e.stderr
            print(f"Failed to install Torch Vision: {e.stderr}")
        return install_results
    
    async def install_torch(self):
        install_results = {}
        try:
            install_cmd = ["pip", "install", "torch", "--break-system-packages"]
            result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
            install_results["torch"] = result.stdout
        except subprocess.CalledProcessError as e:
            install_results["torch"] = e.stderr
            print(f"Failed to install Torch: {e.stderr}")
        return install_results
    
    async def install_numpy(self):
        install_results = {}
        try:
            install_cmd = ["pip", "install", "numpy", "--break-system-packages"]
            result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
            install_results["numpy"] = result.stdout
        except subprocess.CalledProcessError as e:
            install_results["numpy"] = e.stderr
            print(f"Failed to install NumPy: {e.stderr}")
        return install_results
    
    async def test_numpy(self):
        test_numpy_cmd = "python3 -c 'import numpy; print(numpy.__version__)'"
        try:
            test_numpy = subprocess.check_output(test_numpy_cmd, shell=True).decode("utf-8")
            if type(test_numpy) == str and type(test_numpy) != ValueError:
                return True
            else:
                return False
        except Exception as e:
            print(e)
            raise ValueError(e)
        return None


    def __call__(self, request):
        return self.install(request)

    async def test_hardware(self):
        cuda_test = None
        openvino_test = None
        llama_cpp_test = None
        onnx_test = None
        ipex_test = None
        torch_test = None
        numpy_test = None
        torch_vision_test = None
        optimum_test = None
        optimum_ipex_test = None
        optimum_openvino_test = None
        optimum_neural_compressor_test = None
        optimum_habana_test = None
        optimum_onnx_test = None
        cuda_install = None
        openvino_install = None
        onnx_install = None
        numpy_install = None
        torch_install = None
        torch_vision_install = None
        llama_cpp_install = None
        ipex_install = None
        optimum_install = None
        optimum_ipex_install = None
        optimum_habana_install = None
        optimum_neural_compressor_install = None
        optimum_openvino_install = None
        optimum_onnx_install = None
        optimum_amx_install = None
        optimum_amx_test = None
        
        try:
            optimum_amx_test = await self.test_huggingface_optimum_amx()
        except Exception as e:
            optimum_amx_test = e
            print(e)
            try:
                optimum_amx_install = await self.install_huggingface_optimum_amx()
            except Exception as e:
                optimum_amx_install = e
                print(e)
            pass
        
        try:
            onnx_test = await self.test_onnx()
        except Exception as e:
            onnx_test = e
            print(e)
            try:
                onnx_install = await self.install_onnx()
                try:
                    onnx_test = await self.test_onnx()
                except Exception as e:
                    onnx_test = e
                    print(e)
            except Exception as e:
                onnx_install = e
                print(e)
            pass
        
        try:
            numpy_test = await self.test_numpy()
        except Exception as e:
            numpy_test = e
            print(e)
            try:
                numpy_install = await self.install_numpy()
                try:
                    numpy_test = await self.test_numpy()
                except Exception as e:
                    numpy_test = e
                    print(e)
            except Exception as e:
                numpy_install = e
                print(e)
            pass
        
        try:
            optimum_test = await self.test_huggingface_optimum()
        except Exception as e:
            optimum_test = e
            print(e)
            try:
                optimum_install = await self.install_huggingface_optimum()
                try:
                    optimum_test = await self.test_huggingface_optimum()
                except Exception as e:
                    optimum_test = e
                    print(e)
            except Exception as e:
                optimum_install = e
                print(e)
                
        try:
            optimum_openvino_test = await self.test_huggingface_optimum_openvino()
        except Exception as e:
            optimum_openvino_test = e
            print(e)
            try:
                optimum_openvino_install = await self.install_huggingface_optimum_openvino()
                try:
                    optimum_openvino_test = await self.test_huggingface_optimum_openvino()
                except Exception as e:
                    optimum_openvino_test = e
                    print(e)
            except Exception as e:
                optimum_openvino_install = e
                print(e)
            pass

        try:
            optimum_neural_compressor_test = await self.test_huggingface_optimum_neural_compressor()
        except Exception as e:
            optimum_neural_compressor_test = e
            print(e)
            try:
                optimum_neural_compressor_install = await self.install_huggingface_optimum_neural_compressor()
                try:
                    optimum_neural_compressor_test = await self.test_huggingface_optimum_neural_compressor()
                except Exception as e:
                    optimum_neural_compressor_test = e
                    print(e)
            except Exception as e:
                optimum_neural_compressor_install = e
                print(e)
            pass
        
        try:
            optimum_habana_test = await self.test_huggingface_optimum_habana()
        except Exception as e:
            optimum_habana_test = e
            print(e)
            try:
                optimum_habana_install = await self.install_huggingface_optimum_habana()
                try:
                    optimum_habana_test = await self.test_huggingface_optimum_habana()
                except Exception as e:
                    optimum_habana_test = e
                    print(e)
            except Exception as e:
                optimum_habana_install = e
                print(e)
            pass
        
        try:
            optimum_onnx_test = await self.test_huggingface_optimum_onnx()
        except Exception as e:
            optimum_onnx_test = e
            print(e)
            try:
                optimum_onnx_install = await self.intstall_huggingface_optimum_onnx()
                try:
                    optimum_onnx_test = await self.intstall_huggingface_optimum_onnx()
                except Exception as e:
                    optimum_onnx_test = e
                    print(e)
            except Exception as e:
                optimum_onnx_install = e
                print(e)
            pass
        
        
        try:
            optimum_ipex_test = await self.test_huggingface_optimum_ipex()
        except Exception as e:
            optimum_ipex_test = e
            print(e)
            try:
                optimum_ipex_install = await self.install_huggingface_optimum_ipex()
                try:
                    optimum_ipex_test = await self.test_huggingface_optimum_ipex()
                except Exception as e:
                    optimum_ipex_test = e
                    print(e)
            except Exception as e:
                optimum_ipex_install = e
                print(e)
            pass
        
        try:
            openvino_test = await self.test_local_openvino()
        except Exception as e:
            openvino_test = e
            print(e)
            try:
                openvino_install = await self.install_openvino()
                try:
                    openvino_test = await self.test_local_openvino()
                except Exception as e:
                    openvino_test = e
                    print(e)
            except Exception as e:
                openvino_install = e
                print(e)        
            pass
            
        # try:
        #     llama_cpp_test = await self.test_llama_cpp()
        #     raise ValueError("Test Llama C++")
        # except Exception as e:
        #     llama_cpp_test = e
        #     try:
        #         llama_cpp_install = await self.install_llama_cpp()
        #         try:
        #             llama_cpp_test = await self.test_llama_cpp()
        #         except:
        #             llama_cpp_test = e
        #     except Exception as e:
        #         print(e)
        #         llama_cpp_install = e
        #     pass
        
        try:
            torch_test = await self.test_torch()
        except Exception as e:
            torch_test = e
            print(e)
            try:
                torch_install = await self.install_torch()
                try:
                    torch_test = await self.test_torch()
                except Exception as e:
                    torch_test = e
                    print(e)
            except Exception as e:
                torch_install = e
                print(e)
            pass
        
        try:
            torch_vision_test = await self.test_torch_vision()
        except Exception as e:
            torch_vision_test = e
            print(e)
            try:
                torch_vision_install = await self.install_torch_vision()
                try:
                    torch_vision_test = await self.test_torch_vision()
                except Exception as e:
                    torch_vision_test = e
                    print(e)
            except Exception as e:
                torch_vision_install = e
                print(e)
            pass
        try:
            ipex_test = await self.test_ipex()
        except Exception as e:
            ipex_test = e
            print(e)
            try:
                ipex_install = await self.install_ipex()
                try:
                    ipex_test = await self.test_ipex()
                except Exception as e:
                    ipex_test = e
                    print(e)
            except Exception as e:
                ipex_install = e
                print(e)
            pass
        try:
            cuda_test = await self.test_cuda()
        except Exception as e:
            try:
                cuda_install = await self.install_cuda()
                try:
                    cuda_test = await self.test_cuda()
                except Exception as e:
                    cuda_test = e
                    print(e)                    
            except Exception as e:
                cuda_install = e
                print(e)
            pass
        # print("local_endpoint_test")
        install_results = {
            "cuda": cuda_install,
            "openvino": openvino_install,
            "llama_cpp": llama_cpp_install,
            "ipex": ipex_install,
            "optimum": optimum_install,
            "optimum-ipex": optimum_ipex_install,
            "optimum-openvino": optimum_openvino_install,
            "optimum-neural-compressor": optimum_neural_compressor_install,
            "optimum-habana": optimum_habana_install,
            "onnx": onnx_install,  
            "optimum-onnx": optimum_onnx_install,
            "torch": torch_install,
            "torch_vision": torch_vision_install,
        }
        # print(install_results)
        test_results = {
            "cuda": cuda_test,
            "openvino": openvino_test,
            "llama_cpp": llama_cpp_test,
            "ipex": ipex_test,
            "optimum": optimum_test,
            "optimum-ipex": optimum_ipex_test,
            "optimum-openvino": optimum_openvino_test,
            "optimum-neural-compressor": optimum_neural_compressor_test,
            "optimum-habana": optimum_habana_test,
            "onnx": onnx_test,
            "optimum-onnx": optimum_onnx_test,
            "torch": torch_test,
            "torch_vision": torch_vision_test,
        }
        # print(test_results)
        return test_results

install_depends_py = install_depends_py
