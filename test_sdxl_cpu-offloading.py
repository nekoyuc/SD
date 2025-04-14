from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusionXLPipeline
import torch

model_id = "stabilityai/stable-diffusion-xl-base-1.0"

pipeline = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    variant="fp16",
    use_safetensors=True
    ).to("cuda")

#pipeline.enable_model_cpu_offload()

prompt = "A cute cat scupture 3d printed with wood-fused filaments of these colors: #e09d28 and #f5f0d3, keep the object empty, show the full extent of the whole object, do not include texts in the image. Keep the background mono-colored, ideally white, plain and simple."

image = pipeline(
    prompt=prompt,
    num_inference_steps=40,
    denoising_end=0.8,
    output_type="latent",
    ).images[0]

image.save("testxl.png")
