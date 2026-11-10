import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import math

def load_image(image_path, size=None, mode="RGB", save_path=None):
    img = Image.open(image_path).convert(mode)
    if size is None:
        width, height = img.size
        new_width = (width // 64) * 64
        new_height = (height // 64) * 64
        size = (new_width, new_height)

    img = img.resize(size, Image.BICUBIC)

    if save_path: img.save(save_path)
    return ToTensor()(img).unsqueeze(0)


def adain(source, target, eps=1e-6):
    source_mean = torch.mean(source, dim=1, keepdim=True)
    source_std = torch.std(source, dim=1, keepdim=True)
    target_mean = torch.mean(target, dim=(0, 1), keepdim=True)
    target_std = torch.std(target, dim=(0, 1), keepdim=True)
    normalized_source = (source - source_mean) / (source_std + eps)
    return normalized_source * target_std + target_mean


def memory_efficient(model):
    model.freeze()
    try:
        model.enable_model_cpu_offload()
    except AttributeError:
        print("enable_model_cpu_offload is not supported.")
    try:
        model.enable_vae_slicing()
    except AttributeError:
        print("enable_vae_slicing is not supported.")
    try:
        model.enable_vae_tiling()
    except AttributeError:
        print("enable_vae_tiling is not supported.")
