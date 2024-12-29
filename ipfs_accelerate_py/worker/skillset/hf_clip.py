import torch
from torch import no_grad
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
import time

class hf_clip:
    def __init__(self, resources=None, metadata=None):
        # self.model = CLIPModel.from_pretrained(resources['clip'], local_files_only=True).to('cuda')
        # self.processor = CLIPProcessor.from_pretrained(resources['clip'], local_files_only=True)
        # self.tokenizer  = AutoTokenizer.from_pretrained(resources['clip'], local_files_only=True)
        pass

    def __test__(self, endpoint_model, endpoint_handler, endpoint_label, tokenizer):
        sentence_1 = "The quick brown fox jumps over the lazy dog"
        image_1 = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
        timestamp1 = time.time()
        try:
            test_batch = endpoint_handler(sentence_1, image_1)
        except Exception as e:
            print(e)
            pass
        timestamp2 = time.time()
        elapsed_time = timestamp2 - timestamp1
        tokens = tokenizer[endpoint_label]()
        len_tokens = len(tokens["input_ids"])
        tokens_per_second = len_tokens / elapsed_time
        print(f"elapsed time: {elapsed_time}")
        print(f"tokens: {len_tokens}")
        print(f"tokens per second: {tokens_per_second}")
        # test_batch_sizes = await self.test_batch_sizes(metadata['models'], ipfs_accelerate_init)
        if "openvino" not in endpoint_label:
            with torch.no_grad():
                if "cuda" in dir(torch):
                    torch.cuda.empty_cache()
        print("hf_llava test")
        return None
    

    def __call__(self, method, text=None, image=None):
        if method == 'clip_text':
            inputs = self.tokenizer([text], return_tensors='pt').to('cuda')

            with no_grad():
                text_features = self.model.get_text_features(**inputs)

            return {
                'embedding': text_features[0].cpu().numpy().tolist()
            }
        
        elif method == 'clip_image':
            inputs = self.processor(images=image, return_tensors='pt').to('cuda')

            with no_grad():
                image_features  = self.model.get_image_features(**inputs)

            return {
                'embedding': image_features[0].cpu().numpy().tolist()
            }