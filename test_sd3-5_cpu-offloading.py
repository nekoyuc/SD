from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
import torch

model_id = "stabilityai/stable-diffusion-3.5-medium"

pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
    )

pipeline.enable_model_cpu_offload()

prompt = "A cute cat scupture 3d printed with wood-fused filaments of these colors: #e09d28 and #f5f0d3, keep the object empty, show the full extent of the whole object, do not include texts in the image. Keep the background mono-colored, ideally white, plain and simple."

image = pipeline(
    prompt=prompt,
    num_inference_steps=40,
    guidance_scale=4.5,
    max_sequence_length=512,
    ).images[0]

image.save("test3.png")
