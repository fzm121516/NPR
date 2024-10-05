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

ia.seed(1)

# --------------- Arguments ---------------
parser = argparse.ArgumentParser(description='Test Images')
parser.add_argument('--images-dir', type=str, required=True)
parser.add_argument('--result-dir', type=str, required=True)
parser.add_argument('--num-threads', type=int, default=16, help='Number of threads to use (default: 4)')  # 添加线程参数

args = parser.parse_args()

# Load Images
image_list = sorted([*glob.glob(os.path.join(args.images_dir, '**', '*.png'), recursive=True)])

num_image = len(image_list)
print("Find ", num_image, " images")

# Function to convert PNG to JPEG with specified quality
def png2jpg(img, quality):
    out = BytesIO()
    img.save(out, format='jpeg', quality=quality)  # ranging from 0-95, 75 is default
    img = Image.open(out)
    img = np.array(img)  # load from memory before ByteIO closes
    out.close()
    return Image.fromarray(img)

# Function to process a single image
def process_image(image_path):
    image_name = os.path.basename(image_path)  # 获取文件名（包含扩展名）

    # Load image
    image = Image.open(image_path).convert("RGB")

    # 使用JPEG压缩扰动图像
    compressed_image = png2jpg(image, quality=95)  

    # Save results
    output_dir = os.path.join(
        args.result_dir,
        os.path.relpath(image_path, args.images_dir)
    )
    output_dir = os.path.dirname(output_dir)  # 获取目录，不包括文件名
    os.makedirs(output_dir, exist_ok=True)

    # 拼接保存路径并创建
    save_path = os.path.join(output_dir, image_name)  
    # save_path = os.path.join(output_dir, image_name.replace('.png', '.jpg'))  
    # 确保保存路径存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save compressed image
    compressed_image.save(save_path)  # 保存压缩后的JPEG图像

# Process images in parallel with progress bar
with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
    # Use tqdm to display progress bar
    list(tqdm(executor.map(process_image, image_list), total=num_image))
