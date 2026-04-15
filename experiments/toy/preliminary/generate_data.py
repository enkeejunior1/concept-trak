import os
import argparse
import numpy as np
import random
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm

num_shapes = 1
img_size = 64
shapes = ['triangle', 'circle']
colors = ['red', 'blue']
size = int(0.16 * img_size)

def draw_shape(draw, shape, position, size, color):
    """Draw a shape with specified properties using PIL ImageDraw"""
    x, y = position
    outline_color = 'black'
    outline_width = 2
    
    if shape == 'circle':
        # Draw circle (ellipse with equal width and height)
        bbox = [x-size, y-size, x+size, y+size]
        draw.ellipse(bbox, fill=color, outline=outline_color, width=outline_width)
    
    elif shape == 'triangle':
        points = [(x, y-size), (x-size, y+size), (x+size, y+size)]
        draw.polygon(points, fill=color, outline=outline_color, width=outline_width)
    
# Create a single synthetic image using the draw_shape function
def create_synthetic_image_pil(comp_gen=False):
    # Create a blank white image
    img = Image.new('RGB', (img_size, img_size), 'white')
    draw = ImageDraw.Draw(img)

    # Set shape, color
    while True:
        shape = random.choice(shapes)
        color = random.choice(colors)
        
        # Store metadata for each shape
        metadata = [
            shapes.index(shape),
            colors.index(color),
        ]
        metadata_str = f"{shape},{color}"
        if comp_gen:
            if (metadata[0] == metadata[1] == 0):
                continue
            else:
                break
        else:
            break
    
    # Set shape, color, and position
    x = random.randint(size + 1, img_size - size - 1)
    y = random.randint(size + 1, img_size - size - 1)
    position = (x, y)
    draw_shape(draw, shape, position, size, color)
    return img, metadata, metadata_str

def generate_dataset(num_images, image_dir, label_dir, comp_gen):
    metadata_list = []
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    for image_idx in tqdm(range(num_images), desc=f"Generating {'generator' if comp_gen else 'classifier'} dataset"):
        image, metadata, _ = create_synthetic_image_pil(comp_gen=comp_gen)
        metadata_list.append(metadata)
        image.save(os.path.join(image_dir, f"{image_idx}.png"))

    np.save(os.path.join(label_dir, "metadata.npy"), np.array(metadata_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    experiment_dir = Path(__file__).resolve().parents[1]
    parser.add_argument("--base_dir", type=str, default=str(experiment_dir), help="toy experiment directory")
    parser.add_argument("--classifier_num_images", type=int, default=30000)
    parser.add_argument("--generator_num_images", type=int, default=10000)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    generate_dataset(
        num_images=args.classifier_num_images,
        image_dir=base_dir / "data" / "images-classifier",
        label_dir=base_dir / "data" / "labels-classifier",
        comp_gen=False,
    )
    generate_dataset(
        num_images=args.generator_num_images,
        image_dir=base_dir / "data" / "images",
        label_dir=base_dir / "data" / "labels",
        comp_gen=True,
    )