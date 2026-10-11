import sys,json,importlib.metadata,importlib.util
sys.path[:0]=['/code/source','/code/datasets','/code/kit']
from ipfs_datasets_py.logic.software_contracts import codebase_numerical_selection_384 as native
import torch,duckdb
packages={}
for name in ('torch','numpy','sentence-transformers','transformers','tokenizers','duckdb'):
 try: packages[name]={'version':importlib.metadata.version(name),'module_found':importlib.util.find_spec(name.replace('-','_')) is not None}
 except importlib.metadata.PackageNotFoundError: packages[name]={'version':None,'module_found':importlib.util.find_spec(name.replace('-','_')) is not None}
path,checkpoint=native._checkpoint('/models/checkpoint.json',native.RELEASE['checkpoint_file_sha256'])
snapshot,assets=native.embeddings._snapshot_assets('/models/models--thenlper--gte-small/snapshots/17e1f347d17fe144873b1201da91788898c639cd')
missing=[name for name,row in packages.items() if row['version'] is None or not row['module_found']]
print(json.dumps({'schema':'numerical-container-compatibility@1','python':sys.version,'packages':packages,'missing_dependencies':missing,'torch_runtime':torch.__version__,'cuda_available':torch.cuda.is_available(),'owner_file':native.__file__,'checkpoint_file_sha256':native.RELEASE['checkpoint_file_sha256'],'checkpoint_compatible':True,'embedding_asset_count':len(assets),'embedding_assets_verified':True,'inference_available':not missing,'provider_calls':0,'training_steps':0,'inference_executed':False,'proof_authority':False}))
