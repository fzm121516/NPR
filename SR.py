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

# Load Images
image_list = sorted([*glob.glob(os.path.join(args.images_dir, '**', '*.png'), recursive=True)])

num_image = len(image_list)
print("Find ", num_image, " images")

# Function to process a single image
def process_image(image_path):
    global current_gpu

    image_name = os.path.basename(image_path)  # 获取文件名（包含扩展名）

    # Load image
    # image = Image.open(image_path)

    # Save results
    output_dir = os.path.join(
        args.result_dir,
        os.path.relpath(image_path, args.images_dir)
    )
    output_dir = os.path.dirname(output_dir)  # 获取目录，不包括文件名
    os.makedirs(output_dir, exist_ok=True)

    # 拼接保存路径并创建
    save_path = output_dir
    # save_path = os.path.join(output_dir, image_name)

    # 确保保存路径存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Get the GPU ID for this process
    gpu_id = gpu_ids[current_gpu]
    current_gpu = (current_gpu + 1) % num_gpus  # Update to the next GPU

    subprocess.run(['python', 'inference_realesrgan.py', '-n', 'RealESRGAN_x4plus', '-i', image_path, '-o', save_path, '-g', str(gpu_id)])

# Process images in parallel with progress bar
with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
    # Use tqdm to display progress bar
    list(tqdm(executor.map(process_image, image_list), total=num_image))
