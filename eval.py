import argparse
from pathlib import Path
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
from function import calc_mean_std, normal, coral
import models.transformer as transformer
import models.StyTR as StyTR
import matplotlib.pyplot as plt
from matplotlib import cm
from function import normal
import numpy as np
import time
import os
import random
from tqdm import tqdm
from util.utils import load_pretrained
import shutil

def select_random_images(root_dir, num_images, save_dir=None):
    images = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.jpg') or filename.endswith('.png'):
                images.append(os.path.join(dirpath, filename))

    if len(images) < num_images:
        print("Warning: Number of images in folder is less than required.")

    selected_images = random.sample(images, min(num_images, len(images)))
    
    if save_dir is not None:
        # Ensure the destination directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Copy selected images to the destination directory
        for image in selected_images:
            shutil.copy(image, save_dir)
    return selected_images


def test_transform(size, crop):
    transform_list = []
   
    if size != 0: 
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform
def style_transform(h,w):
    k = (h,w)
    size = int(np.max(k))
    transform_list = []    
    transform_list.append(transforms.CenterCrop((h,w)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def content_transform():
    
    transform_list = []   
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def eval(args):
    print("Mamba encoder: ", args.use_mamba_enc)
    print("Mamba decoder: ", args.use_mamba_dec)
    print("C input, AB style: ", args.c_input)
    print("C style, AB input: ", args.c_style)
    print("Name: ", args.model_name)
    # Advanced options
    content_size=args.img_size
    style_size=args.img_size
    crop='store_true'
    save_ext='.jpg'
    output_path=args.output

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.content:
        content_paths = [Path(args.content)]
    else:
        content_dir = Path(args.content_dir)
        content_paths = [f for f in content_dir.glob('*')]

    if args.style:
        style_paths = [Path(args.style)]    
    else:
        style_dir = Path(args.style_dir)
        style_paths = [f for f in style_dir.glob('*')]

    random.seed(args.seed)
    num_images_to_select = 40
    style_paths = select_random_images(args.style_dir, num_images_to_select)
    num_images_to_select = 20
    content_paths = select_random_images(args.content_dir, num_images_to_select)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    network = load_pretrained(args)
    network.eval()
    network.to(device)
    network_sota = load_pretrained(args, sota=True)
    network_sota.eval()
    network_sota.to(device)


    content_tf = test_transform(content_size, crop)
    style_tf = test_transform(style_size, crop)
    content_loss = 0.0
    style_loss = 0.0
    content_loss_sota = 0.0
    style_loss_sota = 0.0
        
        
    for content_path in tqdm(content_paths):
        for style_path in tqdm(style_paths):
            content = content_tf(Image.open(content_path).convert("RGB"))

            h,w,c=np.shape(content)    
            style = style_tf(Image.open(style_path).convert("RGB"))

        
            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)
            with torch.no_grad():
                output, loss_c, loss_s, _, _, _, _ = network(content,style) 
                result = output
                output_sota, loss_c_sota, loss_s_sota, _, _, _, _ = network_sota(content, style)     
        
            content_loss += loss_c
            style_loss += loss_s
            content_loss_sota += loss_c_sota
            style_loss_sota += loss_s_sota
            
            output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
                output_path, splitext(basename(content_path))[0],
                splitext(basename(style_path))[0], save_ext
            )
            output = torch.cat((content.cpu(),style.cpu(),output.cpu(), output_sota.cpu()), 0)
            
            # if args.output != "":
            #     save_image(result, output_name)
    print("Image size: ", args.img_size)
    print("Standard")
    print(f"Content loss total: {content_loss_sota.item()} - Style loss total: {style_loss_sota.item()}")
    print(f"Content loss mean: {content_loss_sota.item()/(len(content_paths)*len(style_paths))} - Style loss mean: {style_loss_sota.item()/(len(content_paths)*len(style_paths))}")
    print("Mamba")
    print(f"Content loss total: {content_loss.item()} - Style loss total: {style_loss.item()}")
    print(f"Content loss mean: {content_loss.item()/(len(content_paths)*len(style_paths))} - Style loss mean: {style_loss.item()/(len(content_paths)*len(style_paths))}")


