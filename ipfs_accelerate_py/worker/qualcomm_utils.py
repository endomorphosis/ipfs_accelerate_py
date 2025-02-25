import tempfile
import os
import json
import shutil
import re
import subprocess

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
            custom_tokenizer = self.transformers.AutoTokenizer(models.BPE(vocab=tokenizer.get_vocab(), merges=[]))
            custom_tokenizer.save(os.path.join(path, "tokenizer.json"))
        else:
            custom_tokenizer = self.transformers.AutoTokenizer(models.BPE(vocab=tokenizer.get_vocab(), merges=[]))
            custom_tokenizer.save(os.path.join(os.path.dirname(path),"tokenizer.json"))
        return None
    
    def genie_t2t_run(self, config=None, text=None):
        if config is None:
            return None 
        if text is None:
            return None
        with tempfile.TemporaryDirectory() as tmpdirname:
            config_path = os.path.join(tmpdirname, "config.json")
            text_path = os.path.join(tmpdirname, "text.txt")
            with open(config_path, "w") as f:
                json.dump(config, f)
            with open(text_path, "w") as f:
                f.write(text)
            genie_t2t_run_cmd = self.genie_t2t_run_path_linux + " --config " + config_path + " --text " + text_path 
            genie_t2t_run_cmd_results = subprocess.run(genie_t2t_run_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return genie_t2t_run_cmd_results
        return None