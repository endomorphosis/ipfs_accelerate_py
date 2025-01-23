import torch
from torch import no_grad
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
import time
import numpy as np
import asyncio
from transformers import AutoConfig, AutoTokenizer, AutoProcessor
import os
import open_clip
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import os
import openvino as ov
from decord import VideoReader, cpu
import tempfile

    # video = cv2.VideoCapture(video_url)
    # frames = []
    # batch_size = 16
    # while True:
    #     ret, frame = video.read()
    #     if not ret:
    #         break
    #     frame = cv2.resize(frame, (224, 224))
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    #     frame = frame.astype(np.float32) / 255.0  # Normalize to [0, 1]
    #     frame = np.transpose(frame, (2, 0, 1))  # Convert shape to (C, H, W)
    #     frame_tensor = torch.from_numpy(frame)  # Convert numpy array to tensor
    #     frames.append(frame_tensor)
    #     if len(frames) == batch_size:
    #         break
    # video.release()
    # if frames:
    #     video_tensor = torch.stack(frames)  # Keep 4D input
    #     video_tensor = video_tensor.unsqueeze(0)  # shape: [1, 16, 3, 224, 224]

np.random.seed(0)

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

# videoreader = VideoReader(file_path, num_threads=1, ctx=cpu(0))

# sample 32 frames
# videoreader.seek(0)
# indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=len(videoreader))
# video = videoreader.get_batch(indices).asnumpy()
     

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    image_data = np.array(image.getdata()).reshape(1, image.size[1], image.size[0], 3).astype(np.byte)
    return image

def load_image_tensor(image_file):
    if isinstance(image_file, str) and (image_file.startswith("http") or image_file.startswith("https")):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    image_data = np.array(image.getdata()).reshape(1, image.size[1], image.size[0], 3).astype(np.byte)
    return ov.Tensor(image_data)

class hf_xclip:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata    
        self.openvino_skill_convert = self.openvino_skill_convert
        self.create_openvino_video_embedding_endpoint_handler = self.create_openvino_video_embedding_endpoint_handler
        self.create_cuda_video_embedding_endpoint_handler = self.create_cuda_video_embedding_endpoint_handler
        self.create_cpu_video_embedding_endpoint_handler = self.create_cpu_video_embedding_endpoint_handler
        self.init_cpu = self.init_cpu
        self.init_cuda = self.init_cuda
        self.init_qualcomm = self.init_qualcomm
        self.init_openvino = self.init_openvino
        self.init = self.init
        self.__test__ = self.__test__
        return None

    def init(self):
        return None
    
    def init_qualcomm(self, model, device, qualcomm_label):
        return None

    def __test__(self, endpoint_model, endpoint_handler, endpoint_label, tokenizer):
        sentence_1 = "The quick brown fox jumps over the lazy dog"
        video_url = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerJoyrides.mp4"
        timestamp1 = time.time()
        try:
            test_batch = endpoint_handler(sentence_1, video_url)
            print(test_batch)
            print("hf_xclip test passed")
        except Exception as e:
            print(e)
            print("hf_xclip test failed")
            pass
        timestamp2 = time.time()
        elapsed_time = timestamp2 - timestamp1
        len_tokens = 1
        tokens_per_second = len_tokens / elapsed_time
        print(f"elapsed time: {elapsed_time}")
        print(f"samples: {len_tokens}")
        print(f"samples per second: {tokens_per_second}")
        # test_batch_sizes = await self.test_batch_sizes(metadata['models'], ipfs_accelerate_init)
        if "openvino" not in endpoint_label:
            with torch.no_grad():
                if "cuda" in dir(torch):
                    torch.cuda.empty_cache()
        print("hf_xclip test")
        return None
    
    def init_cpu(self, model, device, cpu_label):
        
        return None
    
    def init_cuda(self, model, device, cuda_label):
        config = AutoConfig.from_pretrained(model, trust_remote_code=True)    
        tokenizer = AutoTokenizer.from_pretrained(model)
        processor = CLIPProcessor.from_pretrained(model, trust_remote_code=True)
        endpoint = None
        try:
            endpoint = CLIPModel.from_pretrained(model, torch_dtype=torch.float16, trust_remote_code=True).to(device)
        except Exception as e:
            print(e)
            pass
        endpoint_handler = self.create_video_embedding_endpoint_handler(endpoint, tokenizer, model, cuda_label)
        torch.cuda.empty_cache()
        # batch_size = await self.max_batch_size(endpoint_model, cuda_label)
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), 0    

    def init_openvino(self, model=None , model_type=None, device=None, openvino_label=None, get_optimum_openvino_model=None, get_openvino_model=None, get_openvino_pipeline_type=None, openvino_cli_convert=None):
        endpoint = None
        tokenizer = None
        endpoint_handler = None
        homedir = os.path.expanduser("~")
        model_name_convert = model.replace("/", "--")
        huggingface_cache = os.path.join(homedir, ".cache/huggingface")
        huggingface_cache_models = os.path.join(huggingface_cache, "hub")
        huggingface_cache_models_files = os.listdir(huggingface_cache_models)
        huggingface_cache_models_files_dirs = [os.path.join(huggingface_cache_models, file) for file in huggingface_cache_models_files if os.path.isdir(os.path.join(huggingface_cache_models, file))]
        huggingface_cache_models_files_dirs_models = [ x for x in huggingface_cache_models_files_dirs if "model" in x ]
        huggingface_cache_models_files_dirs_models_model_name = [ x for x in huggingface_cache_models_files_dirs_models if model_name_convert in x ]
        model_src_path = os.path.join(huggingface_cache_models, huggingface_cache_models_files_dirs_models_model_name[0])
        model_dst_path = os.path.join(model_src_path, "openvino")
        # config = AutoConfig.from_pretrained(model)
        task = get_openvino_pipeline_type(model, model_type)
        openvino_index = int(openvino_label.split(":")[1])
        weight_format = ""
        if openvino_index is not None:
            if openvino_index == 0:
                weight_format = "int8" ## CPU
            if openvino_index == 1:
                weight_format = "int4" ## gpu
            if openvino_index == 2:
                weight_format = "int4" ## npu
        model_dst_path = model_dst_path+"_"+weight_format
        if not os.path.exists(model_dst_path):
            # os.makedirs(model_dst_path)
            ## convert model to openvino format
            # openvino_cli_convert(model, model_dst_path=model_dst_path, task=task, weight_format=weight_format, ratio="1.0", group_size=128, sym=True )
            pass
        try:
            tokenizer =  AutoProcessor.from_pretrained(
                model
            )
        except Exception as e:
            print(e)
            try:
                tokenizer =  AutoProcessor.from_pretrained(
                    model_src_path
                )
            except Exception as e:
                print(e)
                pass
            
        if not os.path.exists(model_dst_path):
            try:
                convert = self.openvino_skill_convert(model, model_dst_path, task, weight_format)
            except Exception as e:
                print(e)
                try: 
                    convert = openvino_cli_convert(model, model_dst_path=model_dst_path, task=task, weight_format=weight_format, ratio="1.0", group_size=128, sym=True )
                except Exception as e:
                    print(e)
                pass
        
        # genai_model = get_openvino_genai_pipeline(model, model_type, openvino_label)
        try:
            model = get_openvino_model(model, model_type, openvino_label)
            print(model)
        except Exception as e:
            print(e)
            try:
                model = get_optimum_openvino_model(model, model_type, openvino_label)
            except Exception as e:
                print(e)
                pass
        endpoint = model
        endpoint_handler = self.create_openvino_video_embedding_endpoint_handler(model, tokenizer, model, openvino_label)
        batch_size = 0
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), batch_size              
    
    def openvino_skill_convert(self, model_name, model_dst_path, task, weight_format, hfmodel=None, hfprocessor=None):
        import openvino as ov
        import os
        import numpy as np
        import requests
        from decord import VideoReader, cpu
        import tempfile
        from transformers import AutoModel, AutoTokenizer, AutoProcessor  
        if hfmodel is None:
            hfmodel = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
    
        if hfprocessor is None:
            hfprocessor = AutoProcessor.from_pretrained(model_name)
        
        if hfprocessor is not None:
            text = "Replace me by any text you'd like."
            ##xclip processor
            video_url = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerJoyrides.mp4"
            np.random.seed(0)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
                f.write(requests.get(video_url).content)
                f.flush()
                videoreader = VideoReader(f.name, num_threads=1, ctx=cpu(0))
                videoreader.seek(0)
                indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=len(videoreader))
                video = videoreader.get_batch(indices).asnumpy()
                processed_data = hfprocessor(
                    text=text,
                    videos=list(video),
                    return_tensors="pt",
                    padding=True,
                )
                results = hfmodel(**processed_data)
                hfmodel.config.torchscript = True
                ov_model = ov.convert_model(hfmodel,  example_input=dict(processed_data))
                if not os.path.exists(model_dst_path):
                    os.mkdir(model_dst_path)
                ov.save_model(ov_model, os.path.join(model_dst_path, model_name.replace("/", "--") + ".xml"))
                ov_model = ov.compile_model(ov_model)
                hfmodel = None
                hfprocessor = None
        return ov_model
    
    def create_cpu_video_embedding_endpoint_handler(self, tokenizer , endpoint_model, cpu_label, endpoint=None, ):
        def handler(x, tokenizer=tokenizer, endpoint_model=endpoint_model, cpu_label=cpu_label, endpoint=None):

            if "eval" in dir(endpoint):
                endpoint.eval()
            else:
                pass
            return None
        return handler
    
    def create_qualcomm_video_embedding_endpoint_handler(self, tokenizer , endpoint_model, cpu_label, endpoint=None, ):
        def handler(x, tokenizer=tokenizer, endpoint_model=endpoint_model, cpu_label=cpu_label, endpoint=None):

            if "eval" in dir(endpoint):
                endpoint.eval()
            else:
                pass
            return None
        return handler
    
    def create_cuda_video_embedding_endpoint_handler(self, tokenizer , endpoint_model, cuda_label, endpoint=None, ):
        def handler(x, tokenizer, endpoint_model, openvino_label, endpoint=None):
            if "eval" in dir(endpoint):
                endpoint.eval()
            else:
                pass
            return None
        return handler

    def create_openvino_video_embedding_endpoint_handler(self, endpoint_model , tokenizer , openvino_label, endpoint=None ):
        def handler(x, y, tokenizer=tokenizer, endpoint_model=endpoint_model, openvino_label=openvino_label, endpoint=None):
            np.random.seed(0)                       
            
            videoreader = None
            if y is not None:            
                if type(y) == str:
                    if os.path.exists(y):
                        videoreader = VideoReader(y, num_threads=1, ctx=cpu(0))
                    elif "http" in y:
                        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
                            f.write(requests.get(y).content)
                            f.flush()
                            videoreader = VideoReader(f.name, num_threads=1, ctx=cpu(0))
                if videoreader is not None:
                    videoreader.seek(0)
                    indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=len(videoreader))
                    video = videoreader.get_batch(indices).asnumpy()
                pass
            
            if x is not None:
                if type(x) == str:
                    text = x
                else:
                    text = ""
            else:
                text = ""
            
            processed_data = tokenizer(
                text=text,
                videos=list(video),
                return_tensors="pt",
                padding=True,
            )

            new_processed_data = {
                'input_ids': processed_data["input_ids"],
                'attention_mask': processed_data["attention_mask"],
                'pixel_values': processed_data["pixel_values"]
            }

            inference_results = endpoint_model(dict(new_processed_data))
            results_list = list(inference_results.values())
            
            text_embeddings = results_list[3]
            video_embeddings = results_list[5]
            if x is not None or y is not None:
                if x is not None and y is not None:
                    return {
                        'video_embedding': video_embeddings,
                        'text_embedding': text_embeddings
                    }
                elif x is not None:
                    return {
                        'embedding': video_embeddings
                    }
                elif y is not None:
                    return {
                        'embedding': text_embeddings
                    }            
            return None
        return handler


    def openvino_skill_convert(self, model_name, model_dst_path, task, weight_format, hfmodel=None, hfprocessor=None):
        import openvino as ov
        import os
        import numpy as np
        import requests
        import tempfile
        from transformers import AutoModel, AutoTokenizer, AutoProcessor  
        from decord import VideoReader, cpu
        import tempfile
        if hfmodel is None:
            hfmodel = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
    
        if hfprocessor is None:
            hfprocessor = AutoProcessor.from_pretrained(model_name)

        if hfprocessor is not None:
            text = "Replace me by any text you'd like."
            ##xclip processor
            video_url = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerJoyrides.mp4"
            np.random.seed(0)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
                f.write(requests.get(video_url).content)
                f.flush()
                videoreader = VideoReader(f.name, num_threads=1, ctx=cpu(0))
                videoreader.seek(0)
                indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=len(videoreader))
                video = videoreader.get_batch(indices).asnumpy()
                processed_data = hfprocessor(
                    text=text,
                    videos=list(video),
                    return_tensors="pt",
                    padding=True,
                )
                results = hfmodel(**processed_data)
                hfmodel.config.torchscript = True
                ov_model = ov.convert_model(hfmodel,  example_input=dict(processed_data))
                if not os.path.exists(model_dst_path):
                    os.mkdir(model_dst_path)
                ov.save_model(ov_model, os.path.join(model_dst_path, model_name.replace("/", "--") + ".xml"))
                ov_model = ov.compile_model(ov_model)
                hfmodel = None
        return ov_model