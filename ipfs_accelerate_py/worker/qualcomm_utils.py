import tempfile
import os
import json
import shutil
import re
import subprocess
import platform
import sys

class qualcomm_utils:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata
        self.genie_t2t_run_path_linux = "/usr/local/bin/genie-t2t-run"
        self.genie_t2t_run_path_windows = "C:\\Program Files\\genie-t2t-run\\genie-t2t-run.exe"
        self.genie_t25_run_path_android = "/data/local/tmp/genie-t2t-run"
        return None
    
    def init(self):
        if "transformers" not in list(self.resources.keys()):
            import transformers
            self.resources["transformers"] = transformers
            self.transformers = self.resources["transformers"]
        else:
            self.transformers = self.resources["transformers"]
            
        return None
    
    def install_qnn_model(self, model=None, path=None):
        if model is None:
            return None
        if path is None:
            return None
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        if os.path.isfile(path):
            os.remove(path)
        model = model.replace("_","-")
        pip_install_cmd  = ' pip -U "qai_hub_models[' + model + ']"'
        pip_install_cmd_results = subprocess.run(pip_install_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return None
    
    def quantize_export_qnn_model(self, model=None, path=None):
        if model is None:
            return None
        dryrun = 'python -m qai_hub_models.models.llama_v3_8b_chat_quantized.export --device "Snapdragon X Elite CRD" --skip-inferencing --skip-profiling --output-dir genie_bundle'
        if path is None:
            return None
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        if os.path.isfile(path):
            os.remove(path)
        quantize_export_cmd = 'python -m qai_hub_models.models.' + model + '.export --device "Snapdragon X Elite CRD" --skip-inferencing --skip-profiling --output-dir ' + path
        quantize_export_cmd_results = subprocess.run(quantize_export_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        cleanup_dryrun = 'rm -rf genie_bundle'
        cleanup_dryrun_results = subprocess.run(cleanup_dryrun, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) 
        return None

    def remove_qnn_model(self, model=None):
        if model is None:
            return None
        pip_uninstall_cmd = "pip uninstall qai_hub_models[" + model + "]"
        pip_uninstall_cmd_results = subprocess.run(pip_uninstall_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return None
    
    def setup_env(self, env_var=None):
        if platform.system() == "Windows":
            shutil.copytree(os.path.join(os.environ['QNN_SDK_ROOT'], 'lib', 'hexagon-v73', 'unsigned'), 'genie_bundle', dirs_exist_ok=True)
            shutil.copytree(os.path.join(os.environ['QNN_SDK_ROOT'], 'lib', 'aarch64-windows-msvc'), 'genie_bundle', dirs_exist_ok=True)
            shutil.copy(os.path.join(os.environ['QNN_SDK_ROOT'], 'bin', 'aarch64-windows-msvc', 'genie-t2t-run.exe'), 'genie_bundle')
        if platform.system() == "Linux":
            pass
        if platform.system() == "Android":
            arch = platform.architecture()[0]
            if "hexagon" in arch:
                if "v73" in arch:
                    shutil.copytree(os.path.join(os.environ['QNN_SDK_ROOT'], 'lib', 'hexagon-v73', 'unsigned'), 'genie_bundle', dirs_exist_ok=True)
                elif "v75" in arch:
                    shutil.copytree(os.path.join(os.environ['QNN_SDK_ROOT'], 'lib', 'hexagon-v75', 'unsigned'), 'genie_bundle', dirs_exist_ok=True)
                elif "v79" in arch:
                    shutil.copytree(os.path.join(os.environ['QNN_SDK_ROOT'], 'lib', 'hexagon-v79', 'unsigned'), 'genie_bundle', dirs_exist_ok=True)
                else:
                    raise Exception("Unsupported Hexagon version")
            else:
                pass            
            shutil.copytree(os.path.join(os.environ['QNN_SDK_ROOT'], 'lib', 'aarch64-android'), 'genie_bundle', dirs_exist_ok=True)
            shutil.copy(os.path.join(os.environ['QNN_SDK_ROOT'], 'bin', 'aarch64-android', 'genie-t2t-run'), 'genie_bundle')
            return None
        return None

    def export_tokenizer(self, config, tokenizer, model, path):
        if tokenizer is None:
            return None
        if path is None:
            return None
        if config is None:
            return None
        if model is None:
            return None
        
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        if os.path.isfile(path):
            os.remove(path)
            custom_tokenizer = self.transformers.AutoTokenizer(model.BPE(vocab=tokenizer.get_vocab(), merges=[]))
            custom_tokenizer.save(os.path.join(path, "tokenizer.json"))
        else:
            custom_tokenizer = self.transformers.AutoTokenizer(model.BPE(vocab=tokenizer.get_vocab(), merges=[]))
            custom_tokenizer.save(os.path.join(os.path.dirname(path),"tokenizer.json"))
        return None
    
    def genie_t2t_run(self, config=None, text=None):
        if config is None:
            return None 
        if text is None:
            return None
        with tempfile.TemporaryFile() as tmpfile:
            tmpfile.write(config)
            tmpfile.seek(0)
            filepath = tmpfile.name
            
            genie_t2t_run_cmd = self.genie_t2t_run_path_linux + " --config " + filepath + " --text " + text 
            genie_t2t_run_cmd_results = subprocess.run(genie_t2t_run_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return genie_t2t_run_cmd_results
        return None