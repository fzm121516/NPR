import torch
import argparse
import os
import glob
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter

# --------------- Arguments ---------------
parser = argparse.ArgumentParser(description='Test Images')
parser.add_argument('--images-dir', type=str, required=True)
parser.add_argument('--result-dir', type=str, required=True)

args = parser.parse_args()

# Load Images
image_list = sorted([*glob.glob(os.path.join(args.images_dir, '**', '*.png'), recursive=True)])

num_image = len(image_list)
print("Find ", num_image, " images")

# Define Gaussian blur function
def gaussian_blur(img, sigma):
    img = np.array(img)

    # Apply Gaussian blur to each channel
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

    return Image.fromarray(img)

# Process
for i in range(num_image):
    image_path = image_list[i]
    image_name = os.path.basename(image_path)  # 获取文件名（包含扩展名）
    print(i, '/', num_image, image_name)

    # Load image
    image = Image.open(image_path)

    # Apply Gaussian blur
    sigma = 1  # You can adjust this value as needed
    image_blurred = gaussian_blur(image, sigma)

    # Save results
    output_dir = os.path.join(
        args.result_dir,
        os.path.relpath(image_path, args.images_dir)
    )
    output_dir = os.path.dirname(output_dir)  # 获取目录，不包括文件名
    os.makedirs(output_dir, exist_ok=True)

    # 拼接保存路径并创建
    save_path = os.path.join(output_dir, image_name)

    # 确保保存路径存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save blurred image
    image_blurred.save(save_path)
