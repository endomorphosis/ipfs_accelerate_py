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
        else:
            return None
        
    async def install_libp2p_kit(self):
        
        return None
    
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
        install_results["install_torch"] = await self.install_torch()
        install_results["install_oneccl_bind_pt"] = await self.install_oneccl_bind_pt()
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