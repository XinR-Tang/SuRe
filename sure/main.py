import os
import argparse
import importlib
import types
import glob
import torch
from torchvision.utils import save_image

from dataclasses import fields
from typing import Union, get_args, get_origin, get_type_hints
from diffusers import DDIMScheduler
from accelerate.utils import set_seed

from .pipelines import SD1_5_Pipeline, SDXL_Pipeline, SD3_Pipeline, Flux_Pipeline
from .utils.utils import load_image, memory_efficient
from .utils.adapters import Controller

# Map model names to their configuration modules and class names
CONFIG_MAP = {
    "sd1_5": (".config", "SD1_5_Config"),
    "sdxl": (".config", "SDXL_Config"),
    "sd3": (".config", "SD3_Config"),
    "flux": (".config", "Flux_Config"),
}

def get_pipeline(model_name: str):
    """Return the appropriate pipeline class based on the model name."""
    if model_name == "sd1_5":
        return SD1_5_Pipeline
    elif model_name == "sdxl":
        return SDXL_Pipeline
    elif model_name == "sd3":
        return SD3_Pipeline
    elif model_name == "flux":
        return Flux_Pipeline
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def resolve_type(annotation):
    """Resolve Optional types and union type annotations."""
    origin = get_origin(annotation)

    if origin in (Union, types.UnionType):
        types_list = [
            t for t in get_args(annotation)
            if t is not type(None)
        ]
        if len(types_list) == 1:
            return types_list[0]

    return annotation

def load_config_class(model_name: str):
    """Dynamically import the configuration class for the selected model."""
    if model_name not in CONFIG_MAP:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Available models: {list(CONFIG_MAP)}"
        )

    module_name, class_name = CONFIG_MAP[model_name]

    module = importlib.import_module(module_name, package=__package__)
    return getattr(module, class_name)

def parse_args():
    # Parse the model selection argument first
    model_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    model_parser.add_argument(
        "--model",
        type=str,
        choices=list(CONFIG_MAP.keys()),
        default="sd1_5",
        help="Model configuration to use",
    )
    model_args, _ = model_parser.parse_known_args()

    # Dynamically load the selected configuration class
    InferenceConfig = load_config_class(model_args.model)

    # Create the main argument parser using the selected configuration
    parser = argparse.ArgumentParser(
        description="SuRe Style Transfer Script",
        parents=[model_parser],
        allow_abbrev=False,
    )

    type_hints = get_type_hints(InferenceConfig)

    # Automatically register arguments from the configuration class
    for f in fields(InferenceConfig):
        arg_name = f"--{f.name}"
        arg_type = resolve_type(type_hints[f.name])

        kwargs = {
            "dest": f.name,
            "default": argparse.SUPPRESS,
            "help": f.metadata.get("help", ""),
        }

        if arg_type is bool:
            # Support both --option and --no-option
            parser.add_argument(
                arg_name,
                action=argparse.BooleanOptionalAction,
                **kwargs,
            )
        else:
            parser.add_argument(
                arg_name,
                type=arg_type,
                **kwargs,
            )

    # Parse all command-line arguments
    cli_args = vars(parser.parse_args())

    # Extract the selected model name
    model_name = cli_args.pop("model")

    # Initialize the configuration with command-line overrides
    config = InferenceConfig(**cli_args)

    return model_name, config

def main(model_name, args):
    pipeline_cls = get_pipeline(model_name)
    if model_name == "sd1_5":
        pipe = pipeline_cls.from_pretrained(
            args.model_path,
            safety_checker=None,
            requires_safety_checker=False,
        )
    else:
        pipe = pipeline_cls.from_pretrained(args.model_path)
    if model_name == "flux":
        memory_efficient(pipe)
    if model_name in ["sd1_5", "sdxl"]:
        pipe.classifier = pipe.unet
    
    set_seed(args.seed)

    # load style images from folder if provided, otherwise load single style image
    if args.style_image_folder:
        style_images = []
        style_image_paths = glob.glob(os.path.join(args.style_image_folder, '*'))

        for img_path in style_image_paths:
            style_image = load_image(img_path, size=(args.W, args.H), save_path=None)
            style_image = style_image.repeat(args.num_p, 1, 1, 1)
            style_images.append(style_image)
        style_image = torch.cat(style_images, dim=0)
        print(f"style_image shape: {style_image.shape}")
    else:
        style_image = load_image(args.style_image, size=(args.W, args.H), save_path=None)
        style_image = style_image.repeat(args.num_p, 1, 1, 1)
        print(f"style_image shape: {style_image.shape}")

    # load content image if provided
    content_image = load_image(args.content_image, size=(args.W, args.H),
                               save_path=None) if args.content_image else None

    controller = Controller(self_layers=(args.start_layer, args.end_layer))

    stylization_kwargs = dict(
        lr=args.lr,
        batch_size=args.batch_size,
        iters=args.iters,
        width=args.W,
        height=args.H,
        beta=args.beta,
        controller=controller,
        style_image=style_image,
        content_image=content_image,
        mixed_precision=args.mixed_precision,
        num_inference_steps=args.num_step,
        enable_gradient_checkpoint=args.enable_gradient_checkpoint,
        num_p=args.num_p,
        seed=args.seed,
        mode=args.mode,
    )
    if model_name == "flux":
        stylization_kwargs["strength"] = args.strength
        stylization_kwargs["guidance_scale"] = args.guidance_scale
        stylization_kwargs["prompt"] = args.prompt
        stylization_kwargs["prompt_2"] = args.prompt_2

    result = pipe.stylization(**stylization_kwargs)
    save_image(result, args.output_path)

if __name__ == "__main__":
    model_name, args = parse_args()
    main(model_name, args)
