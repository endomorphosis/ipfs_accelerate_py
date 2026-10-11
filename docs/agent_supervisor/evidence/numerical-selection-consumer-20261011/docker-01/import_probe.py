import sys,json,importlib.metadata
sys.path[:0]=['/code/source','/code/datasets','/code/kit']
from ipfs_datasets_py.logic.software_contracts import codebase_numerical_selection_384 as native
import torch,duckdb
print(json.dumps({'schema':'numerical-container-import-probe@1','python':sys.version,'packages':{n:importlib.metadata.version(n) for n in ('torch','numpy','sentence-transformers','transformers','tokenizers','duckdb')},'torch_runtime':torch.__version__,'cuda_available':torch.cuda.is_available(),'owner_file':native.__file__,'checkpoint_bytes':__import__('pathlib').Path('/models/checkpoint.json').stat().st_size,'provider_calls':0,'training_steps':0,'inference_executed':False}))
