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

# Get number of available CPU cores for data loading
num_cores = mp.cpu_count()
print(f"Available CPU cores: {num_cores}")

# Check GPU availability with more detailed info
if torch.cuda.is_available():
    device = "cuda"
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9  # in GB
        print(f"GPU {i}: {gpu_name} with {gpu_mem:.1f} GB memory")
    # Clean GPU cache before starting
    torch.cuda.empty_cache()
else:
    device = "cpu"
    print("WARNING: No GPU found! Training on CPU will be extremely slow.")

# Get the current working directory
current_dir = os.getcwd()

# Define paths to datasets (use existing prepared dataset if available)
dataset_root = "dataset"
data_yaml_path = f"{dataset_root}/data.yaml"

# Check if dataset is already prepared
if os.path.exists(data_yaml_path):
    print("Using existing dataset configuration.")
else:
    print("WARNING: Dataset configuration not found. Creating minimal configuration.")
    # Create a minimal data.yaml file for testing
    data_yaml = {
        'train': os.path.join(current_dir, f"{dataset_root}/images/train"),
        'val': os.path.join(current_dir, f"{dataset_root}/images/val"),
        'nc': 4,  # 4 classes
        'names': ['person', 'car', 'bus', 'truck']
    }
    
    os.makedirs(f"{dataset_root}", exist_ok=True)
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)

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

# Select smaller model for faster training
model_version = "yolov8m"  # Use medium model for balance of speed and accuracy
print(f"Loading {model_version} model...")

# Load model with optimized settings
model = YOLO(f"{model_version}.pt")
model.to(device)

# Find optimal batch size for speed while avoiding OOM
def get_optimal_batch_size():
    if device != "cuda":
        return 4
    
    try:
        # Conservative but faster batch size
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9  # in GB
        
        if gpu_mem > 40:  # For A100/H100
            return 32
        elif gpu_mem > 24:  # For high-end consumer GPUs
            return 16
        elif gpu_mem > 12:
            return 8
        else:
            return 4
    except:
        return 4

batch_size = get_optimal_batch_size()
print(f"Selected batch size: {batch_size}")

# Use smaller image size for faster training
img_size = 416  # Reduced size for speed

# FAST TRAINING SETTINGS
print(f"\nStarting accelerated training on {device.upper()} with batch size {batch_size}...")

try:
    results = model.train(
        data=data_yaml_path,
        epochs=10,              # Drastically reduced for quick results
        batch=batch_size,
        imgsz=img_size,         # Smaller images
        
        # Speed-focused parameters
        workers=min(8, num_cores),
        device=device,
        cos_lr=True,            
        lr0=0.02,               # Higher learning rate for faster convergence
        lrf=0.01,
        weight_decay=0.0001,    # Reduced for speed
        warmup_epochs=0,        # No warmup to save time
        
        # Memory optimizations
        optimizer="Adam",       # Adam for faster convergence
        amp=True,               # Automatic mixed precision
        half=True,              # Half precision (FP16)
        cache=False,            # Disable caching to save memory
        rect=True,              # Rectangular training
        
        # Minimal overhead
        plots=False,            # Disable plotting
        save_period=-1,         # Only save final model
        
        # Focus training effort
        patience=3,             # Early stopping
        
        # Project settings
        project="bdd100k_augmented",
        name=f"{model_version}_fast_train",
        exist_ok=True,
        
        # Additional speed optimizations
        close_mosaic=0,         # Disable mosaic early
        multi_scale=False,      # No multi-scale
        nbs=64,                 # Nominal batch size
        
        # Minimal augmentation for speed
        hsv_h=0.0,              # Disable HSV augmentation
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,            # No rotation
        translate=0.0,
        scale=0.0,
        fliplr=0.5,             # Keep only horizontal flip
        mosaic=0.0,             # Disable mosaic for speed
    )
    
except RuntimeError as e:
    # If OOM occurs, try even more aggressive settings
    cleanup_memory()
    print("\nEncountered error. Trying with minimal settings...")
    
    # Ultra-minimal settings
    model_version = "yolov8s"  # Switch to small model
    model = YOLO(f"{model_version}.pt")
    model.to(device)
    
    batch_size = 2  # Tiny batch size
    img_size = 320  # Very small images
    
    print(f"Retrying with model {model_version}, batch size {batch_size}, image size {img_size}")
    
    results = model.train(
        data=data_yaml_path,
        epochs=5,               # Minimal epochs
        batch=batch_size,
        imgsz=img_size,
        workers=2,              # Minimal workers
        device=device,
        amp=True,
        half=True,
        cache=False,
        rect=True,
        plots=False,
        patience=2,
        project="bdd100k_augmented",
        name=f"{model_version}_minimal_train",
        exist_ok=True,
        mosaic=0.0,
        multi_scale=False,
    )

# Clean up memory before validation
cleanup_memory()

# Quick validation (optional - comment out if too slow)
print("\nRunning quick validation...")
val_results = model.val(batch=batch_size*2)  # Larger batch for validation
print(f"Validation results: {val_results}")

# Export the model to a single optimized format
print("\nExporting model to optimized format...")
cleanup_memory()

# Export only to ONNX with optimization (faster than multiple exports)
onnx_path = model.export(
    format="onnx", 
    dynamic=True, 
    simplify=True,
    opset=17
)
print(f"Model exported to ONNX: {onnx_path}")

print("\nAccelerated training and export complete!")
