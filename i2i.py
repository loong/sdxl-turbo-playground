from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
import torch

# Load the SDXL-Turbo model for image-to-image generation
pipeline = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/sdxl-turbo", 
    torch_dtype=torch.float16, 
    variant="fp16"
)

# Move the pipeline to your device (e.g., "mps" for Apple Silicon)
pipeline = pipeline.to("mps")

# Load an initial image
init_image = load_image("singapore-arial.png").resize((512, 512))

# Define the prompt and generate a transformed image
prompt = "needle"
image = pipeline(prompt=prompt, image=init_image, num_inference_steps=2, strength=0.8, guidance_scale=0.0).images[0]

# Save or display the image
image.save("transformed_output.png")