from dataclasses import dataclass
from typing import Optional

@dataclass
class SD1_5_Config:
    # Model
    model_path: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"

    # Input images
    style_image: Optional[str] = None
    style_image_folder: Optional[str] = None
    content_image: Optional[str] = None

    # Optimization
    lr: float = 0.05
    seed: int = 42
    iters: int = 1
    mixed_precision: str = "bf16"
    guidance_scale: int = 1
    num_step: int = 200
    num_images_per_prompt: int = 1
    enable_gradient_checkpoint: bool = False

    # Attention layers
    start_layer: int = 10
    end_layer: int = 16

    # Image dimensions
    H: int = 512
    W: int = 512

    # optimization parameters
    beta: float = 0.24
    num_p: int = 5 # the number of permutation matrices
    batch_size: int = 1

    # Output
    output_path: str = "sd1_5.png"

    # Optimization mode
    mode: str = "cos_weight"

@dataclass
class SDXL_Config:
    # Model
    model_path: str = "stabilityai/stable-diffusion-xl-base-1.0"

    # Input images
    style_image: Optional[str] = None
    style_image_folder: Optional[str] = None
    content_image: Optional[str] = None

    # Optimization
    lr: float = 0.02
    seed: int = 42
    iters: int = 1
    mixed_precision: str = "bf16"
    guidance_scale: int = 1
    num_step: int = 200
    num_images_per_prompt: int = 1
    enable_gradient_checkpoint: bool = False

    # Attention layers
    start_layer: int = 50
    end_layer: int = 57

    # Image dimensions
    H: int = 1024
    W: int = 1024

    # optimization parameters
    beta: float = 0.30
    num_p: int = 5 # the number of permutation matrices
    batch_size: int = 1

    # Output
    output_path: str = "sdxl.png"

    # Optimization mode
    mode: str = "cos_weight"

@dataclass
class SD3_Config:
    # Model
    model_path: str = "stabilityai/stable-diffusion-3-medium-diffusers"

    # Input images
    style_image: Optional[str] = None
    style_image_folder: Optional[str] = None
    content_image: Optional[str] = None

    # Optimization
    lr: float = 0.015
    seed: int = 42
    iters: int = 1
    mixed_precision: str = "bf16"
    guidance_scale: int = 1
    num_step: int = 200
    num_images_per_prompt: int = 1
    enable_gradient_checkpoint: bool = False

    # Attention layers
    start_layer: int = 17
    end_layer: int = 23

    # Image dimensions
    H: int = 512
    W: int = 512

    # optimization parameters
    beta: float = 0.22
    num_p: int = 5 # the number of permutation matrices
    batch_size: int = 1

    # Output
    output_path: str = "sd3.png"

    # Optimization mode
    mode: str = "cos_weight"

@dataclass
class Flux_Config:
    # Model
    model_path: str = "black-forest-labs/FLUX.1-dev"

    # Input images
    style_image: Optional[str] = None
    style_image_folder: Optional[str] = None
    content_image: Optional[str] = None

    # Optimization
    lr: float = 0.02
    seed: int = 42
    iters: int = 1
    mixed_precision: str = "bf16"
    guidance_scale: float = 3.5
    prompt: str = ""
    prompt_2: Optional[str] = None
    num_step: int = 200
    num_images_per_prompt: int = 1
    enable_gradient_checkpoint: bool = True
    strength: float = 0.20

    # Attention layers
    start_layer: int = 50
    end_layer: int = 57

    # Image dimensions
    H: int = 512
    W: int = 512

    # Optimization parameters
    beta: float = 0.25
    num_p: int = 5
    batch_size: int = 1

    # Output
    output_path: str = "flux.png"

    # Optimization mode
    mode: str = "cos_weight"
