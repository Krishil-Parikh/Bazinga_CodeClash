import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
def add_rain(image, rain_density=0.1, rain_angle=-10, rain_length=20):
    h, w = image.shape[:2]
    # Create rain layer with optimized vectorized operations
    rain_layer = np.zeros((h, w), dtype=np.uint8)
    num_drops = int(w * h * rain_density)
    # Pre-generate all random values at once
    xs = np.random.randint(0, w, size=num_drops)
    ys = np.random.randint(0, h, size=num_drops)
    lengths = np.random.randint(rain_length // 2, rain_length, size=num_drops)
    thicknesses = np.random.randint(1, 3, size=num_drops)
    colors = np.random.randint(150, 255, size=num_drops)
    # Calculate endpoints
    angle_rad = np.radians(rain_angle)
    end_xs = (xs + lengths * np.cos(angle_rad)).astype(int)
    end_ys = (ys + lengths * np.sin(angle_rad)).astype(int)
    # Draw lines
    for i in range(num_drops):
        cv2.line(rain_layer, (xs[i], ys[i]), (end_xs[i], end_ys[i]), 
                 int(colors[i]), int(thicknesses[i]))
    # Blur the rain streaks
    rain_layer = cv2.GaussianBlur(rain_layer, (5, 5), 0)
    # Overlay rain with improved blending
    rain_rgb = cv2.cvtColor(rain_layer, cv2.COLOR_GRAY2BGR)
    mask = (rain_layer > 0).astype(np.float32)[:,:,np.newaxis] * 0.3
    rainy_image = image * (1 - mask) + rain_rgb * mask
    return rainy_image.astype(np.uint8)
def process_image(img_name, image_dir, output_dir, rain_density=0.08):
    img_path = os.path.join(image_dir, img_name)
    if not os.path.exists(img_path):
        return False
    img = cv2.imread(img_path)
    if img is None:
        return False
    rainy_img = add_rain(img, rain_density=rain_density)
    cv2.imwrite(os.path.join(output_dir, img_name), rainy_img)
    return True
# Paths
json_path = "/teamspace/studios/this_studio/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"
image_dir = "/teamspace/studios/this_studio/bdd100k/bdd100k/images/100k/train/trainA"
output_dir = "bdd100k_rainy"
os.makedirs(output_dir, exist_ok=True)
with open(json_path, "r") as f:
    data = json.load(f)
clear_images = [img["name"] for img in data if img.get("attributes", {}).get("weather") == "clear"]
# Use multiprocessing for parallel execution
num_cores = mp.cpu_count()
print(f"Using {num_cores} CPU cores for parallel processing")
# Create partial function with fixed arguments
process_func = partial(process_image, 
                       image_dir=image_dir, 
                       output_dir=output_dir, 
                       rain_density=0.08)
# Process images in parallel
with mp.Pool(processes=num_cores) as pool:
    results = list(tqdm(pool.imap(process_func, clear_images), total=len(clear_images)))
successful = sum(results)
print(f"✅ Successfully processed {successful} out of {len(clear_images)} images")
print(f"✅ Rain-augmented images saved in {output_dir}")
import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import multiprocessing as mp
from functools import partial
json_path = "/teamspace/studios/this_studio/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"
image_dir = "/teamspace/studios/this_studio/bdd100k/bdd100k/images/100k/train/trainA"
output_dir = "bdd100k_foggy"
os.makedirs(output_dir, exist_ok=True)
with open(json_path, "r") as f:
    data = json.load(f)
clear_images = [img["name"] for img in data if img.get("attributes", {}).get("weather") == "clear"]
def add_optimized_fog(image, fog_density=0.7, depth_factor=0.3):
    h, w = image.shape[:2]
    # Create depth map (closer to camera = less fog)
    depth_map = np.ones((h, w)) * 255 
    depth_map = depth_map * np.linspace(depth_factor, 1, h)[:, np.newaxis]
    # Use faster noise generation
    noise = np.zeros((h, w), dtype=np.float32)
    for scale in [4, 8, 16, 32]:
        small_size = (w//scale+1, h//scale+1)
        octave = np.random.randn(*small_size).astype(np.float32)
        octave = cv2.resize(octave, (w, h))
        noise += octave / scale
    # Normalize and smooth noise
    noise = (noise - noise.min()) / (noise.max() - noise.min()) * 255
    noise = cv2.GaussianBlur(noise.astype(np.uint8), (21, 21), 10)
    # Combine depth and noise for fog mask
    fog_mask = (depth_map * (0.5 + 0.5 * noise / 255)).astype(np.uint8)
    fog_mask = cv2.GaussianBlur(fog_mask, (21, 21), 0)
    # Create white fog
    fog = np.ones_like(image) * 255
    # Vectorized blending operation
    alpha = np.clip(fog_mask * fog_density / 255, 0, 1)
    alpha = alpha[:,:,np.newaxis]
    foggy_image = image * (1 - alpha) + fog * alpha
    return foggy_image.astype(np.uint8)
def process_image(img_name, image_dir, output_dir, fog_density=0.6):
    img_path = os.path.join(image_dir, img_name)
    if not os.path.exists(img_path):
        return False
    img = cv2.imread(img_path)
    if img is None:
        return False
    foggy_img = add_optimized_fog(img, fog_density=fog_density)
    cv2.imwrite(os.path.join(output_dir, img_name), foggy_img)
    return True
# Use multiprocessing for CPU parallelism
num_cores = mp.cpu_count()
print(f"Using {num_cores} CPU cores for parallel processing")
# Create a partial function with fixed arguments
process_func = partial(process_image, 
                       image_dir=image_dir, 
                       output_dir=output_dir, 
                       fog_density=0.6)
# Process images in parallel
with mp.Pool(processes=num_cores) as pool:
    results = list(tqdm(pool.imap(process_func, clear_images), total=len(clear_images)))
print(f"✅ Successfully processed {sum(results)} out of {len(clear_images)} images")
print(f"✅ Optimized foggy images saved in {output_dir}")
import cv2
import numpy as np
import os
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
def add_snow(image, snow_density=0.2, snow_intensity=0.7):
    h, w = image.shape[:2]
    # Create a blank canvas for snow
    snow_layer = np.zeros((h, w), dtype=np.uint8)
    # Add random white snowflakes more efficiently
    num_snowflakes = int(snow_density * h * w)
    # Generate random coordinates all at once
    y_coords = np.random.randint(0, h, size=num_snowflakes)
    x_coords = np.random.randint(0, w, size=num_snowflakes)
    # Set snowflakes with vectorized operation
    snow_layer[y_coords, x_coords] = 255
    # Vary snowflake sizes for more realism
    for radius in range(1, 3):
        dilated = cv2.dilate(snow_layer, np.ones((3, 3), np.uint8))
        snow_layer = np.where(np.random.random((h, w)) < 0.2, dilated, snow_layer)
    # Apply Gaussian blur to simulate falling effect
    snow_layer = cv2.GaussianBlur(snow_layer, (5, 5), 0)
    # Convert to 3 channels
    snow_layer = cv2.cvtColor(snow_layer, cv2.COLOR_GRAY2BGR)
    # Blend snow with image using faster alpha blending
    alpha = snow_layer.astype(np.float32) / 255.0 * snow_intensity
    snow_image = image * (1 - alpha) + 255 * alpha
    return snow_image.astype(np.uint8)
def process_image(img_name, image_dir, output_dir, snow_density=0.2, snow_intensity=0.6):
    img_path = os.path.join(image_dir, img_name)
    img = cv2.imread(img_path)
    if img is None:
        return False
    snow_img = add_snow(img, snow_density=snow_density, snow_intensity=snow_intensity)
    cv2.imwrite(os.path.join(output_dir, img_name), snow_img)
    return True
# Directory paths
image_dir = "/teamspace/studios/this_studio/bdd100k/bdd100k/images/100k/train/trainA"
output_dir = "bdd100k_snowy"
os.makedirs(output_dir, exist_ok=True)
# Get list of image files
image_files = os.listdir(image_dir)
# Use multiprocessing to speed up processing
num_cores = mp.cpu_count()
print(f"Using {num_cores} CPU cores for parallel processing")
# Create a partial function with fixed arguments
process_func = partial(process_image, 
                      image_dir=image_dir, 
                      output_dir=output_dir, 
                      snow_density=0.2, 
                      snow_intensity=0.6)
# Process images in parallel
with mp.Pool(processes=num_cores) as pool:
    results = list(tqdm(pool.imap(process_func, image_files), total=len(image_files)))
successful = sum(results)
print(f"✅ Successfully processed {successful} out of {len(image_files)} images")
print(f"✅ Snowy images saved in {output_dir}")
import os
import json
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
# Class mapping for YOLO format (number IDs instead of names)
class_map = {
    "person": 0,
    "car": 1,
    "bus": 2,
    "truck": 3
}
def process_image_label(img_data, output_label_folder, image_folder):
    img_name = img_data["name"]
    img_path = os.path.join(image_folder, img_name)
    label_path = os.path.join(output_label_folder, img_name.replace(".jpg", ".txt"))
    # Skip if image doesn't exist
    if not os.path.exists(img_path):
        return False
    # Get image dimensions for normalization
    import cv2
    try:
        img = cv2.imread(img_path)
        if img is None:
            return False
        img_height, img_width = img.shape[:2]
    except Exception:
        return False
    with open(label_path, "w") as f:
        for obj in img_data.get("labels", []):
            category = obj["category"]
            if category in class_map:
                bbox = obj["box2d"]
                # Calculate center points and dimensions
                x_center = (bbox["x1"] + bbox["x2"]) / 2
                y_center = (bbox["y1"] + bbox["y2"]) / 2
                width = bbox["x2"] - bbox["x1"]
                height = bbox["y2"] - bbox["y1"]
                # Normalize (YOLO format requires values between 0 and 1)
                x_center_norm = x_center / img_width
                y_center_norm = y_center / img_height
                width_norm = width / img_width
                height_norm = height / img_height
                # Write in YOLO format: class_id x_center y_center width height
                f.write(f"{class_map[category]} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")
    return True
# Paths
bdd_labels_path = "/teamspace/studios/this_studio/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"
image_folder = "/teamspace/studios/this_studio/bdd100k/bdd100k/images/100k/train/trainA"
output_label_folder = "labels"
os.makedirs(output_label_folder, exist_ok=True)
# Load BDD JSON
with open(bdd_labels_path, "r") as f:
    data = json.load(f)
# Use multiprocessing for parallel conversion
num_cores = mp.cpu_count()
print(f"Using {num_cores} CPU cores for parallel processing")
# Create partial function with fixed arguments
process_func = partial(process_image_label, 
                      output_label_folder=output_label_folder,
                      image_folder=image_folder)
# Process labels in parallel
with mp.Pool(processes=num_cores) as pool:
    results = list(tqdm(pool.imap(process_func, data), total=len(data)))
successful = sum(1 for r in results if r)
print(f"✅ Successfully converted {successful} out of {len(data)} labels")
print("✅ BDD100K converted to YOLO format!")
