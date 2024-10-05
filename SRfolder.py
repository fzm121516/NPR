import torch
import argparse
import os
import glob
from PIL import Image
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from io import BytesIO
import concurrent.futures
from tqdm import tqdm
import subprocess

# --------------- Arguments ---------------
parser = argparse.ArgumentParser(description='Test Images')
parser.add_argument('--images-dir', type=str, required=True)
parser.add_argument('--result-dir', type=str, required=True)
parser.add_argument('--num-threads', type=int, default=4, help='Number of threads to use (default: 4)')  # 添加线程参数
parser.add_argument('--gpus', type=str, default='0,1,2,3', help='Comma-separated list of GPU IDs to use (default: 0,1,2,3)')

args = parser.parse_args()

# Parse GPU IDs
gpu_ids = [int(gpu) for gpu in args.gpus.split(',')]
num_gpus = len(gpu_ids)
current_gpu = 0

# Function to recursively find all target folders
def find_target_folders(root_dir, target_folders=['0_real', '1_fake']):
    target_paths = []
    for root, dirs, files in os.walk(root_dir):
        for d in dirs:
            if d in target_folders:
                target_paths.append(os.path.join(root, d))
    return target_paths

# Find all target folders
folder_list = find_target_folders(args.images_dir)
num_folders = len(folder_list)
print("Find ", num_folders, " target folders")

# Function to process a single folder
def process_folder(folder_path):
    global current_gpu

    # Get the GPU ID for this process
    gpu_id = gpu_ids[current_gpu]
    current_gpu = (current_gpu + 1) % num_gpus  # Update to the next GPU

    output_dir = os.path.join(args.result_dir, os.path.relpath(folder_path, args.images_dir))
    os.makedirs(output_dir, exist_ok=True)

    subprocess.run(['python', 'inference_realesrgan.py', '-n', 'RealESRGAN_x4plus', '-i', folder_path, '-o', output_dir, '-g', str(gpu_id)])

# Process folders in parallel with progress bar
with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
    # Use tqdm to display progress bar
    list(tqdm(executor.map(process_folder, folder_list), total=num_folders))
