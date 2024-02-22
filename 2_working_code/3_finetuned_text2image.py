from diffusers import DiffusionPipeline
import torch
from PIL import Image

prj_path = "KarAshutosh/ViratKholi"
model = "stabilityai/stable-diffusion-xl-base-1.0"

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
)
pipe.to("cuda")
pipe.load_lora_weights(prj_path, weight_name="pytorch_lora_weights.safetensors")

prompt = "photo of ViratKholi in a suit"

seed = 42
generator = torch.Generator("cuda").manual_seed(seed)
image = pipe(prompt=prompt, generator=generator).images[0]

## Commented code that displays image
# from IPython.display import display
# display(image)
# image.save("image_file_name.png")
