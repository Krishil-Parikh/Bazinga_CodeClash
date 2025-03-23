from ultralytics import YOLO
import os
import torch
import yaml
import multiprocessing as mp
import shutil
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Force CUDA to be used if available
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
torch.backends.cudnn.benchmark = True  # Speed up training
torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere GPUs
torch.backends.cudnn.allow_tf32 = True  # Allow TF32 on Ampere GPUs

# Set PyTorch to use expandable segments to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Check GPU availability with more detailed info
if torch.cuda.is_available():
    device = "cuda"
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9  # in GB
        print(f"GPU {i}: {gpu_name} with {gpu_mem:.1f} GB memory")
    print(f"Using CUDA version: {torch.version.cuda}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    # Check and print free memory
    free_mem = torch.cuda.memory_reserved(0) / 1e9  # in GB
    print(f"Free GPU memory: {free_mem:.2f} GB")
else:
    device = "cpu"
    print("WARNING: No GPU found! Training on CPU will be extremely slow.")
    print("Please ensure your system has a GPU and CUDA is properly installed.")

# Get number of available CPU cores for data loading
num_cores = mp.cpu_count()
print(f"Available CPU cores: {num_cores}")

# Get the current working directory
current_dir = os.getcwd()

# Define paths to datasets
original_img_dir = "/teamspace/studios/this_studio/bdd100k/bdd100k/images/100k/train/trainA"
rainy_img_dir = "bdd100k_rainy"
foggy_img_dir = "bdd100k_foggy"
snowy_img_dir = "bdd100k_snowy"
labels_dir = "labels"

# Create directories for the YOLO dataset structure
dataset_root = "dataset"
os.makedirs(f"{dataset_root}/images/train", exist_ok=True)
os.makedirs(f"{dataset_root}/images/val", exist_ok=True)
os.makedirs(f"{dataset_root}/labels/train", exist_ok=True)
os.makedirs(f"{dataset_root}/labels/val", exist_ok=True)

# Function to check if dataset has already been prepared
def is_dataset_prepared():
    # Check if data.yaml exists
    if not os.path.exists(f"{dataset_root}/data.yaml"):
        return False
    
    # Check if there are files in train and val directories
    train_files = os.listdir(f"{dataset_root}/images/train")
    val_files = os.listdir(f"{dataset_root}/images/val")
    
    # If both directories have files, consider dataset prepared
    return len(train_files) > 0 and len(val_files) > 0

# Parallel file copy function
def copy_files_parallel(file_pairs):
    """Copy multiple files in parallel using ThreadPoolExecutor"""
    def copy_single(src, dst):
        shutil.copy(src, dst)
    
    with ThreadPoolExecutor(max_workers=min(32, num_cores*2)) as executor:
        futures = [executor.submit(copy_single, src, dst) for src, dst in file_pairs]
        for future in futures:
            future.result()  # Wait for completion

# Function to prepare dataset - copy images and labels to YOLO structure
def prepare_dataset():
    if is_dataset_prepared():
        print("Dataset already prepared, skipping preparation step.")
        return f"{dataset_root}/data.yaml"
    
    print("Preparing dataset...")
    
    # Get list of all images
    all_images = [f for f in os.listdir(original_img_dir) if f.endswith(".jpg")]
    
    # Split into train/val (80/20)
    from sklearn.model_selection import train_test_split
    train_images, val_images = train_test_split(all_images, test_size=0.2, random_state=42)
    
    print(f"Preparing dataset with {len(train_images)} training and {len(val_images)} validation images")
    
    # Prepare file copy operations for parallel execution
    train_copy_operations = []
    val_copy_operations = []
    
    # Prepare training copy operations
    for img_file in train_images:
        base_name = os.path.splitext(img_file)[0]
        label_file = f"{base_name}.txt"
        
        # Copy label if it exists
        if os.path.exists(f"{labels_dir}/{label_file}"):
            # Normal image and label
            train_copy_operations.append((
                f"{original_img_dir}/{img_file}", 
                f"{dataset_root}/images/train/{img_file}"
            ))
            train_copy_operations.append((
                f"{labels_dir}/{label_file}", 
                f"{dataset_root}/labels/train/{label_file}"
            ))
            
            # Augmented images with same label
            if os.path.exists(f"{rainy_img_dir}/{img_file}"):
                train_copy_operations.append((
                    f"{rainy_img_dir}/{img_file}", 
                    f"{dataset_root}/images/train/rainy_{img_file}"
                ))
                train_copy_operations.append((
                    f"{labels_dir}/{label_file}", 
                    f"{dataset_root}/labels/train/rainy_{label_file}"
                ))
                
            if os.path.exists(f"{foggy_img_dir}/{img_file}"):
                train_copy_operations.append((
                    f"{foggy_img_dir}/{img_file}", 
                    f"{dataset_root}/images/train/foggy_{img_file}"
                ))
                train_copy_operations.append((
                    f"{labels_dir}/{label_file}", 
                    f"{dataset_root}/labels/train/foggy_{label_file}"
                ))
                
            if os.path.exists(f"{snowy_img_dir}/{img_file}"):
                train_copy_operations.append((
                    f"{snowy_img_dir}/{img_file}", 
                    f"{dataset_root}/images/train/snowy_{img_file}"
                ))
                train_copy_operations.append((
                    f"{labels_dir}/{label_file}", 
                    f"{dataset_root}/labels/train/snowy_{label_file}"
                ))
    
    # Prepare validation copy operations
    for img_file in val_images:
        base_name = os.path.splitext(img_file)[0]
        label_file = f"{base_name}.txt"
        
        # Copy label if it exists
        if os.path.exists(f"{labels_dir}/{label_file}"):
            val_copy_operations.append((
                f"{original_img_dir}/{img_file}", 
                f"{dataset_root}/images/val/{img_file}"
            ))
            val_copy_operations.append((
                f"{labels_dir}/{label_file}", 
                f"{dataset_root}/labels/val/{label_file}"
            ))
    
    # Execute copy operations in parallel
    print(f"Copying {len(train_copy_operations)} training files...")
    copy_files_parallel(train_copy_operations)
    
    print(f"Copying {len(val_copy_operations)} validation files...")
    copy_files_parallel(val_copy_operations)
    
    # Create data.yaml file with absolute paths
    data_yaml = {
        'train': os.path.join(current_dir, f"{dataset_root}/images/train"),
        'val': os.path.join(current_dir, f"{dataset_root}/images/val"),
        'nc': 4,  # 4 classes based on your class map
        'names': ['person', 'car', 'bus', 'truck']
    }
    
    with open(f"{dataset_root}/data.yaml", 'w') as f:
        yaml.dump(data_yaml, f)
    
    print("Dataset preparation completed.")
    return f"{dataset_root}/data.yaml"

# Prepare the dataset (or skip if already prepared)
data_yaml_path = prepare_dataset()
print(f"Using data YAML file: {data_yaml_path}")

# Set the Ultralytics dataset directory to current working directory
os.environ['ULTRALYTICS_DIR'] = current_dir

# Memory cleanup function
def cleanup_memory():
    """Clear CUDA cache and force garbage collection"""
    if device == "cuda":
        torch.cuda.empty_cache()
    import gc
    gc.collect()

# Clean up memory before loading model
cleanup_memory()

# Select YOLO model 
model_version = "yolov8l"  # Switch to a smaller model (x -> l)
print(f"Loading {model_version} model...")

# Use pretrained weights but optimize loading
model = YOLO(f"{model_version}.pt")
model.to(device)

# Calculate optimal batch size based on GPU memory (if available)
def get_optimal_batch_size():
    if device != "cuda":
        return 4  # Default for CPU
    
    try:
        # Get available GPU memory
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9  # in GB
        free_mem = torch.cuda.memory_reserved(0) / 1e9  # in GB
        
        # More conservative batch size estimates considering memory constraints
        if gpu_mem > 40 and free_mem > 10:  # Very high-end GPU with lots of free memory
            return 64
        elif gpu_mem > 24 and free_mem > 6:  # High-end consumer GPU
            return 32
        elif gpu_mem > 16 and free_mem > 4:  # Mid-to-high range GPU
            return 16
        elif gpu_mem > 8 and free_mem > 2:  # Mid-range GPU
            return 8
        else:  # Lower memory or heavily used GPU
            return 4
    except:
        return 4  # Default fallback

batch_size = get_optimal_batch_size()
print(f"Selected batch size: {batch_size}")

# Lower the image size to reduce memory usage
img_size = 512  # Reduced from 640 to save memory

# Improved training configuration
print(f"\nStarting training on {device.upper()} with batch size {batch_size}...")
try:
    results = model.train(
        data=data_yaml_path,
        epochs=50,              # Reduced from 100 to save time
        batch=batch_size,
        imgsz=img_size,         # Reduced image size
        
        # Efficiency-focused parameters
        workers=min(8, num_cores),  # Fewer workers to reduce memory overhead
        device=device,
        cos_lr=True,            # Cosine learning rate schedule
        lr0=0.01,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=1,        # Reduced warmup
        
        # Memory optimizations
        optimizer="Adam",       # Adam can converge faster than SGD
        amp=True,               # Automatic mixed precision
        half=True,              # Half precision (FP16)
        cache=False,            # Turn off caching to save memory
        rect=True,              # Rectangular training (reduces padding)
        
        # Reduce overhead
        plots=False,            # Disable plotting during training
        save_period=20,         # Save less frequently
        
        # Focus training effort
        patience=10,            # Early stopping
        
        # Project settings
        project="bdd100k_augmented",
        name=f"{model_version}_weather_optimized",
        exist_ok=True,
        
        # Additional memory optimizations
        close_mosaic=10,        # Close mosaic augmentation earlier
        multi_scale=False,      # Disable multi-scale to improve speed and memory
        nbs=64,                 # Nominal batch size
        
        # Simpler augmentation to save memory
        hsv_h=0.015,            # Reduced HSV augmentation
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,            # Minimal rotation augmentation (faster)
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=0.5,             # Reduced mosaic probability to save memory
    )
except RuntimeError as e:
    # If we still get OOM, try more aggressive memory saving
    cleanup_memory()
    print("\nEncountered memory error. Trying with more conservative settings...")
    
    # Even more conservative settings
    model_version = "yolov8m"  # Switch to medium model
    model = YOLO(f"{model_version}.pt")
    model.to(device)
    
    batch_size = max(1, batch_size // 2)  # Halve the batch size
    img_size = 416  # Further reduce image size
    
    print(f"Retrying with model {model_version}, batch size {batch_size}, image size {img_size}")
    
    results = model.train(
        data=data_yaml_path,
        epochs=30,              # Further reduced epochs
        batch=batch_size,
        imgsz=img_size,         # Smaller image size
        workers=4,              # Minimal workers
        device=device,
        amp=True,              
        half=True,             
        cache=False,           
        rect=True,             
        plots=False,           
        patience=5,             
        project="bdd100k_augmented",
        name=f"{model_version}_weather_optimized_fallback",
        exist_ok=True,
        mosaic=0.0,             # Disable mosaic augmentation entirely
        multi_scale=False,
    )

# Clean up memory before validation
cleanup_memory()

# Validate the model on test data
print("\nRunning model validation...")
val_results = model.val()
print(f"Validation results: {val_results}")

# Export the model to optimized formats
print("\nExporting model to optimized formats...")

# Clean up memory before export
cleanup_memory()

# Export to ONNX with optimization
onnx_path = model.export(
    format="onnx", 
    dynamic=True, 
    simplify=True, 
    opset=17  # Latest opset version for best optimizations
)
print(f"Model exported to ONNX: {onnx_path}")

# Export TorchScript with optimization
torchscript_path = model.export(
    format="torchscript", 
    optimize=True,
    prefix=f"{model_version}_optimized"
)
print(f"Model exported to TorchScript: {torchscript_path}")

print("\nTraining and export complete!")
