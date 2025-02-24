import requests
#from PIL import Image
from io import BytesIO
import shutil
import nncf
import openvino as ov
import gc

core = ov.Core()

from transformers import AutoProcessor, AutoConfig
from transformers import TextStreamer
from optimum.intel.openvino import OVModelForVisualCausalLM

model_path = "C:/Users/devcloud/.cache/huggingface/hub/models--llava-hf--llava-v1.6-mistral-7b-hf/openvino"

config = AutoConfig.from_pretrained(model_path)

processor = AutoProcessor.from_pretrained(
    model_path, patch_size=config.vision_config.patch_size, vision_feature_select_strategy=config.vision_feature_select_strategy
)
device = core.available_devices[0]
ov_model = OVModelForVisualCausalLM.from_pretrained(model_path, device=device)


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


image_file = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
text_message = "What is unusual on this image?"

image = load_image(image_file)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": text_message},
            {"type": "image"},
        ],
    },
]

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

inputs = processor(images=image, text=prompt, return_tensors="pt")

from transformers import TextStreamer

# Prepare
streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
print(f"Question: {text_message}")
print("Answer:")

output_ids = ov_model.generate(
    **inputs,
    do_sample=False,
    max_new_tokens=50,
    streamer=streamer,
)
outputs = processor.decode(output_ids[0], skip_special_tokens=True)
print(outputs)