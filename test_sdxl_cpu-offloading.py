from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image

model_id = "stabilityai/stable-diffusion-xl-base-1.0"

pipeline = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    variant="fp16",
    use_safetensors=True
    ).to("cuda")

#pipeline.enable_model_cpu_offload()

prompt = "A cute cat scupture 3d printed with wood-fused filaments of these colors: #e09d28 and #f5f0d3, keep the object empty, show the full extent of the whole object, do not include texts in the image. Keep the background mono-colored, ideally white, plain and simple."

latent = pipeline(
    prompt=prompt,
    num_inference_steps=40,
    denoising_end=0.8,
    output_type="latent",
    ).images[0]

# Decode the latent tensor into an image
image = pipeline.vae.decode(latent.unsqueeze(0)).sample[0]
image = (image / 2 + 0.5).clamp(0, 1)  # Normalize to [0, 1]
image = (image.permute(1, 2, 0) * 255).byte().cpu().numpy()  # Convert to numpy array

Image.fromarray(image).save("testxl.png")
