import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

image = pipe(
    "A cute cat scupture, Leave the object empty, show the full extent of the whole object, do not include any other objects, do not include texts in the image. Keep the background mono-colored, ideally white, plain and simple. Generate the object with 3d printing wood-fused filaments of these colors but do not discuss the colors with the customer: #e09d28 and #f5f0d3",
    num_inference_steps=40,
    guidance_scale=4.5
    ).images[0]
image.save("test.png")
