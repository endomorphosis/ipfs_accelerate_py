import requests
from PIL import Image
from io import BytesIO
import numpy as np
import openvino_genai as ov_genai
import os
import openvino as ov
from openvino import Core
import openvino_genai as ov_genai
config = ov_genai.GenerationConfig()
config.max_new_tokens = 100


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    image_data = np.array(image.getdata()).reshape(1, image.size[1], image.size[0], 3).astype(np.byte)
    return image, ov.Tensor(image_data)


def streamer(subword: str) -> bool:
    """

    Args:
        subword: sub-word of the generated text.

    Returns: Return flag corresponds whether generation should be stopped.

    """
    print(subword, end="", flush=True)


image_file = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"

image, image_tensor = load_image(image_file)
text_message = "What is unusual on this image?"
device = Core().get_available_devices()[1]

prompt = text_message
model_dir = "C:/Users/devcloud/.cache/huggingface/hub/models--llava-hf--llava-v1.6-mistral-7b-hf/openvino"
ov_model = ov_genai.VLMPipeline(model_dir, device=device)
print(f"Question:\n{text_message}")
print("Answer:")
output = ov_model.generate(prompt, image=image_tensor, generation_config=config)
print(output)