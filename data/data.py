from datasets import load_dataset
from PIL import Image
import os
import json
import requests
from io import BytesIO
# Create directory if it doesn't exist
os.makedirs("./flickr30k", exist_ok=True)

# Load MSCOCO dataset
dataset = load_dataset("nlphuji/flickr30k", split="test")

# Randomly select and save 500 images
import random

# Get total dataset size
dataset_size = len(dataset)

# Generate 500 random unique indices
random.seed(42)  
random_indices = random.sample(range(dataset_size), 500)

image_prompts = {}
num = 0
# Save 500 random images
for idx, i in enumerate(random_indices):
    # Get image from dataset
    image = dataset[i]["image"]
    caption = dataset[i]["caption"]
    # Save image as PNG
    image_path = f"flickr30k/{num:03d}.png"
    image.save(image_path, "PNG")
    image_prompts[image_path] = caption
    print(f"Saved image {num+1}/100")
    num += 1
    if num == 100:
        break
with open("flickr30k/prompts.json", "w", encoding="utf-8") as f:
    json.dump(image_prompts, f, ensure_ascii=False, indent=4)
print("Finished saving 100 images from flickr30k dataset")
