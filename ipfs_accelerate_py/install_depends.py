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
        if "test_backend" not in globals() and "test_backend" not in list(self.resources.keys()):
            import test_backend
            from test_backend import test_backend_py
            self.test_backend = test_backend_py(resources, metadata)
        elif "test_backend" in list(self.resources.keys()):
            self.test_backend = self.resources["test_backend"]
        elif "test_backend" in globals():
            self.test_backend = test_backend(resources, metadata)
        return None
    
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
        elif package == "optimum":
            return await self.install_huggingface_optimum()
        elif package == "optimum-intel":
            return await self.install_huggingface_optimum_intel()
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
        else:
            return None
        
    async def test_package(self, package):
        if package == "cuda":
            return await self.test_cuda()
        elif package == "optimum":
            return await self.test_ipfs_accelerate()
        elif package == "optimum-intel":
            return await self.test_huggingface_optimum_intel()
        elif package == "openvino":
            return await self.test_openvino()
        elif package == "llama_cpp":
            return await self.test_llama_cpp()
        elif package == "ollama":
            return await self.test_ollama()
        elif package == "ipex":
            return await self.test_ipex()
        elif package == "tortch":
            return await self.test_torch()
        elif package == "storacha":
            return await self.test_storacha()
        elif package == "ipfs":
            return await self.test_ipfs_kit()
        elif package == "libp2p":
            return await self.test_libp2p_kit()
        elif package == "faiss":
            return await self.test_faiss()
        else:
            return None
        
    async def install_libp2p_kit(self):
        
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
            test_ollama = subprocess.check_output("ollama", shell=True)
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
        try:
            test_ipex = subprocess.check_output(test_ipex_cmd, shell=True)
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
   
    async def test_huggingface_optimum_intel(self):
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
    
    async def test_hardware(self):
        cuda_test = None
        openvino_test = None
        llama_cpp_test = None
        ipex_test = None
        optimum_test = None
        optimum_intel_test = None
        cuda_install = None
        openvino_install = None
        llama_cpp_install = None
        ipex_install = None
        optimum_install = None
        optimum_intel_install = None
        
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
            optimum_intel_test = await self.test_huggingface_optimum_intel()
        except Exception as e:
            optimum_intel_test = e
            print(e)
            try:
                optimum_intel_install = await self.install_huggingface_optimum_intel()
                try:
                    optimum_intel_test = await self.test_huggingface_optimum_intel()
                except Exception as e:
                    optimum_intel_test = e
                    print(e)
            except Exception as e:
                optimum_intel_install = e
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
            
        try:
            llama_cpp_test = await self.test_llama_cpp()
        except Exception as e:
            llama_cpp_test = e
            try:
                llama_cpp_install = await self.install_llama_cpp()
                try:
                    llama_cpp_test = await self.test_llama_cpp()
                except:
                    llama_cpp_test = e
            except Exception as e:
                print(e)
                llama_cpp_install = e
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
        print("local_endpoint_test")
        install_results = {
            "cuda": cuda_install,
            "openvino": openvino_install,
            "llama_cpp": llama_cpp_install,
            "ipex": ipex_install,
            "optimum": optimum_install,
            "optimum-intel": optimum_intel_install
        }
        print(install_results)
        test_results = {
            "cuda": cuda_test,
            "openvino": openvino_test,
            "llama_cpp": llama_cpp_test,
            "ipex": ipex_test,
            "optimum": optimum_test,
            "optimum-intel": optimum_intel_test
        }
        print(test_results)
        return test_results
    
    
    
    async def install_ollama(self):
        cmd = "curl -fsSL https://ollama.com/install.sh | sh"
        install_results = {}
        try:
            result = subprocess.check_output(cmd, shell=True, text=True)
            install_results["ollama"] = result
        except subprocess.CalledProcessError as e:
            if e.stderr == None:
                install_results["ollama"] = True
            else:
                install_results["ollama"] = e
            # print(f"Failed to install Ollama: {e.stderr}")
        return install_results
                

    async def install_llama_cpp(self):
        install_results = {}
        try:
            install_cmd = ["pip", "install", "llama-cpp-python", "--break-system-packages"]
            install_cnd = "pip  install  llama-cpp-python  --break-system-packages"
            result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
            install_results["llama_cpp"] = result.stdout
        except subprocess.CalledProcessError as e:
            install_results["llama_cpp"] = ValueError( f"Failed to install Llama C++: {e.stderr}")
            print(e)
            
        try:
            install_results["ollama"] = await self.install_ollama()
        except Exception as e:            
            install_results["ollama"] = ValueError( f"Failed to install Ollama: {e}")
            print(e)
        
        install_success = False
        install_success = all(type(install_results[package]) != ValueError for package in install_results.keys())
        
        return install_success
    
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
        # python -m pip install intel-extension-for-pytorch
        # python -m pip install oneccl_bind_pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/
        # install_results["install_torch"] = await self.install_torch()
        # install_results["install_oneccl_bind_pt"] = await self.install_oneccl_bind_pt()
        try:
            install_cmd = ["pip", "install", "intel-pytorch-extension", "--extra-index-url", "https://pytorch-extension.intel.com/release-whl/stable/cpu/us/", "--break-system-packages"]
            result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
            install_results["ipex"] = result.stdout
        except subprocess.CalledProcessError as e:
            install_results["ipex"] = e.stderr
            print(f"Failed to install IPEX: {e.stderr}")
        return install_results
    
    async def install_huggingface_optimum(self):
        install_results = {}
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
            install_results["install_huggingface_optimum_gaudi"] = await self.install_huggingface_optimum_gaudi()
        except Exception as e:
            install_results["install_huggingface_optimum_gaudi"] = e
            print(e)
        
        return install_results              
        
    
    async def install_huggingface_optimum_cuda(self):   
        
        return None
    
    async def install_huggingface_optimum_openvino(self):
        
        return None
    
    async def install_huggingface_optimum_intel(self):
            
            return None 

    async def install_huggingface_optimum_gaudi(self):
            
            return None 
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
        return None
    
    async def install_faiss_cuda(self):
        return None
    
    async def install_faiss_amx(self):
        return None
    
    async def install_qdrant(self):
        return None
    
    async def install_elasticsearch(self):
        return None

    def __call__(self, request):
        return self.install(request)

install_depends_py = install_depends_py