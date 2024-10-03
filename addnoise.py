import torch
import argparse
import os
import glob
from PIL import Image
import numpy as np
import imgaug.augmenters as iaa




# --------------- Arguments ---------------
parser = argparse.ArgumentParser(description='Test Images')
parser.add_argument('--images-dir', type=str, required=True)
parser.add_argument('--result-dir', type=str, required=True)

args = parser.parse_args()

# Load Images
image_list = sorted([*glob.glob(os.path.join(args.images_dir, '**', '*.png'), recursive=True)])

num_image = len(image_list)
print("Find ", num_image, " images")

# Process
for i in range(num_image):
    image_path = image_list[i]
    image_name = os.path.basename(image_path)  # 获取文件名（包含扩展名）
    print(i, '/', num_image, image_name)

    # Load image
    image = Image.open(image_path)


    # 添加高斯噪声
    # 定义添加高斯噪声的增强器
    aug = iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))  # scale 可以调整
    image_aug = aug(image=image)

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

    # Save noisy image
    Image.fromarray(image_aug).save(save_path)
