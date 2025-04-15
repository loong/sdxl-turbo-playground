from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
import torch
from tqdm import tqdm

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

images = []

for i in tqdm(range(10)):
    # Generate a random seed for reproducibility
    seed = 1234 + i
    generator = torch.manual_seed(seed)

    # Define the prompt and generate a transformed image
    prompt = "ted talk presentation"
    image = pipeline(
        prompt=prompt, 
        image=init_image, 
        num_inference_steps=2, 
        strength=0.8, 
        guidance_scale=0.0, 
        generator=generator  # Pass the generator here
    ).images[0]

    # Save or display the image
    images.append(image)

for i, img in enumerate(images):
    img.save(f"{prompt.split(' ')[0]}_{i}.png")

