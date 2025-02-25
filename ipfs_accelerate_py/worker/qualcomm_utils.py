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
    
    def build_deployable_asset(self, model_path=None, output_path=None):
        results = {}
        if model_path is None:
            return None
        if output_path is None:
            return None
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        if os.path.isfile(output_path):
            os.remove(output_path)
        try:
            export_quantized_onnx_cmd = 'python -m qai_hub_models.models.'+ model_path+'.export --target-runtime onnx'
            export_quantized_onnx_cmd_results = subprocess.run(export_quantized_onnx_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            results["export_quantized_onnx_cmd_results"] = export_quantized_onnx_cmd_results
        except subprocess.CalledProcessError as e:
            print("Error: ", e)
            return None
        try:
            convert_to_deployable_onnx_cmd = 'python build_deployable_asset.py -f ' + output_path
            convert_to_deployable_onnx_cmd_results = subprocess.run(convert_to_deployable_onnx_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            results["convert_to_deployable_onnx_cmd_results"] = convert_to_deployable_onnx_cmd_results
        except subprocess.CalledProcessError as e:
            print("Error: ", e)
            return None
        return results

    def install_qnn_model(self, model=None, path=None):
        results = {}
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
        results["pip_install_cmd_results"] = pip_install_cmd_results
        return results
    
    def quantize_export_qnn_model(self, model=None, path=None):
        results = {}
        if model is None:
            return None
        dryrun = 'python -m qai_hub_models.models.llama_v3_8b_chat_quantized.export --device "Snapdragon X Elite CRD" --skip-inferencing --skip-profiling --output-dir genie_bundle'
        if path is None:
            return None
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        if os.path.isfile(path):
            os.remove(path)
        try:
            quantize_export_cmd = 'python -m qai_hub_models.models.' + model + '.export --device "Snapdragon X Elite CRD" --skip-inferencing --skip-profiling --output-dir ' + path
            quantize_export_cmd_results = subprocess.run(quantize_export_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            results["quantize_export_cmd_results"] = quantize_export_cmd_results
        except subprocess.CalledProcessError as e:
            print("Error: ", e)
            return str(e)
        try:
            cleanup_dryrun = 'rm -rf genie_bundle'
            cleanup_dryrun_results = subprocess.run(cleanup_dryrun, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) 
            results["cleanup_dryrun_results"] = cleanup_dryrun_results
        except subprocess.CalledProcessError as e:
            print("Error: ", e)
            return str(e)
        return results
    
    def postprocess_genie_t2t_run_results(self, t2t_results=None):
        inference_time = {}
        generation_time = {}
        prompt_processing_time = {}
        init_time = {}
        output = {}
        kpis = {}
        dryrun_resuts = '''
        Using libGenie.so version 1.1.0

        [WARN]  "Unable to initialize logging in backend extensions."
        [INFO]  "Using create From Binary List Async"
        [INFO]  "Allocated total size = 323453440 across 10 buffers"
        [PROMPT]: <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is France's capital?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        [BEGIN]: \n\nFrance's capital is Paris.[END]

        [KPIS]:
        Init Time: 6549034 us
        Prompt Processing Time: 196067 us, Prompt Processing Rate : 86.707710 toks/sec
        Token Generation Time: 740568 us, Token Generation Rate: 12.152884 toks/sec
        Inference Time: 0 us, Inference Rate: 0.000000 toks/sec'''
        if t2t_results is None:
            return None
        t2t_results = t2t_results.stdout.decode("utf-8")
        t2t_results = t2t_results.split("\n")
        prompt = ""
        response = ""
        for line in t2t_results:
            if "[KPIS]:" in line:
                break
            if "[INFO]" in line:
                if "Init Time" in line:
                    init_time = re.findall(r'\d+', line)
                    init_time = int(init_time[0])
                if "Prompt Processing Time" in line:
                    prompt_processing_time = re.findall(r'\d+', line)
                    prompt_processing_time = int(prompt_processing_time[0])
                if "Token Generation Time" in line:
                    generation_time = re.findall(r'\d+', line)
                    generation_time = int(generation_time[0])
                if "Inference Time" in line:
                    inference_time = re.findall(r'\d+', line)
                    inference_time = int(inference_time[0])
        prompt = t2t_results[-4]
        response = t2t_results[-3]
        kpis["init_time"] = init_time
        kpis["prompt_processing_time"] = prompt_processing_time
        kpis["generation_time"] = generation_time
        kpis["inference_time"] = inference_time
        output["kpis"] = kpis
        output["prompt"] = prompt
        output["response"] = response
        return output

    def remove_qnn_model(self, model=None):
        results = {}
        if model is None:
            return None
        try:
            pip_uninstall_cmd = "pip uninstall qai_hub_models[" + model + "]"
            pip_uninstall_cmd_results = subprocess.run(pip_uninstall_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            results["pip_uninstall_cmd_results"] = pip_uninstall_cmd_results
        except subprocess.CalledProcessError as e:
            print("Error: ", e)
            return str(e)
        return results
    
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