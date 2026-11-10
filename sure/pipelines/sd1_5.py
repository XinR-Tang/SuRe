import os
import copy
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from diffusers.image_processor import PipelineImageInput
from tqdm import tqdm

from torch import Tensor
from torch.nn import Conv2d
from torch.nn.modules.utils import _pair

from ..utils.adapters import register_attn_control, DataCache
from ..utils.losses import unshuffled_loss, shuffled_loss, q_loss

class SD1_5_Pipeline(StableDiffusionPipeline):
    def init(self, mixed_precision, enable_gradient_checkpoint):
            # `test.py` reuses one pipeline for many image pairs.  Preparing an
            # already prepared UNet nests Accelerate forward wrappers until a
            # RecursionError is raised, so create and prepare it only once.
            if hasattr(self, "_sure_accelerator"):
                if self._sure_mixed_precision != mixed_precision:
                    raise ValueError(
                        "Cannot change mixed_precision when reusing an initialized "
                        "SD1.5 pipeline. Create a new pipeline instead."
                    )
                if enable_gradient_checkpoint:
                    self.classifier.enable_gradient_checkpointing()
                return

            self._sure_accelerator = Accelerator(
                mixed_precision=mixed_precision, gradient_accumulation_steps=1
            )
            self._sure_mixed_precision = mixed_precision
            self.accelerator = self._sure_accelerator
            self.freeze()
            self.enable_vae_slicing()
            # self.enable_model_cpu_offload()
            # self.enable_vae_tiling()
            weight_dtype = torch.float32
            if self.accelerator.mixed_precision == "fp16":
                weight_dtype = torch.float16
            elif self.accelerator.mixed_precision == "bf16":
                weight_dtype = torch.bfloat16
    
            # Move unet, vae and text_encoder to device and cast to weight_dtype
            self.unet.to(self.accelerator.device, dtype=weight_dtype)
            self.vae.to(self.accelerator.device, dtype=weight_dtype)
            self.text_encoder.to(self.accelerator.device, dtype=weight_dtype)
            self.classifier.to(self.accelerator.device, dtype=weight_dtype)
            self.classifier = self.accelerator.prepare(self.classifier)
            if enable_gradient_checkpoint:
                self.classifier.enable_gradient_checkpointing()

    def freeze(self):
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.classifier.requires_grad_(False)

    @torch.no_grad()
    def image2latent(self, image):
        dtype = next(self.vae.parameters()).dtype
        device = next(self.vae.parameters()).device
        # [0,1] -> [-1,1]
        image = image.to(device=device, dtype=dtype) * 2.0 - 1.0
        latent = self.vae.encode(image)["latent_dist"].mean
        latent = latent * self.vae.config.scaling_factor
        return latent

    @torch.no_grad()
    def latent2image(self, latent):
        dtype = next(self.vae.parameters()).dtype
        device = next(self.vae.parameters()).device
        latent = latent.to(device=device, dtype=dtype)
        latent = latent / self.vae.config.scaling_factor
        image = self.vae.decode(latent)[0]
        return (image * 0.5 + 0.5).clamp(0, 1)

    def stylization(
            self,
            latents=None,
            attn_scale=1.0,
            lr=0.05,
            iters=1,
            beta=0.24,
            width=512,
            height=512,
            batch_size=1,
            controller=None,
            style_image=None,
            content_image=None,
            mixed_precision="no",
            num_inference_steps=50,
            enable_gradient_checkpoint=False,
            source_mask=None,
            target_mask=None,
            num_p=1,
            seed=42,
            mode="cos_weight"
    ):
        height = height // self.vae_scale_factor
        width = width // self.vae_scale_factor

        self.init(mixed_precision, enable_gradient_checkpoint)
        device = self.accelerator.device

        style_latent = self.image2latent(style_image)
        print(f"style_latent: {style_latent.shape}")
        print(f"The number of shuffing:{num_p}")

        latents = torch.randn((batch_size, 4, height, width), device=device)
        null_embeds = self.encode_prompt("", device, 1, False)[0]
        null_embeds_for_latents = null_embeds.repeat(latents.shape[0], 1, 1)
        null_embeds_for_style = null_embeds.repeat(style_latent.shape[0], 1, 1)

        if content_image is not None:
            content_latent = self.image2latent(content_image)
            latents = torch.cat([content_latent.clone()] * batch_size)
            null_embeds_for_content = null_embeds.repeat(content_latent.shape[0], 1, 1)

        self.cache = DataCache()
        self.controller = controller
        register_attn_control(
            self.classifier, controller=self.controller, cache=self.cache
        )
        print("Total self attention layers of Unet: ", controller.num_self_layers)
        print("Self attention layers for stylization: ", controller.self_layers)

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        total_steps = self.scheduler.config.num_train_timesteps
        latents = latents.detach().float()
        optimizer = torch.optim.Adam([latents.requires_grad_()], lr=lr)
        optimizer = self.accelerator.prepare(optimizer)
        pbar = tqdm(timesteps, desc="Optimize")

        print(f"start generate with mode:{mode}!!!!!")
        for i, t in enumerate(pbar):
            with torch.no_grad():
                qs_list, ks_list, vs_list, s_out_list = self.extract_feature(
                    style_latent,
                    t,
                    null_embeds_for_style,
                )

                if content_image is not None:
                    qc_list, kc_list, vc_list, c_out_list = self.extract_feature(
                        content_latent,
                        t,
                        null_embeds_for_content,
                    )
            for j in range(iters):
                style_loss = 0
                content_loss = 0
                optimizer.zero_grad()

                q_list, k_list, v_list, self_out_list = self.extract_feature(
                    latents,
                    t,
                    null_embeds_for_latents,
                )

                if mode == "cos_weight":
                    alpha_t = 0.5 * (1 + torch.cos(torch.pi * t.float() / total_steps))
                    style_loss_unshuffled = unshuffled_loss(
                        q_list, ks_list, vs_list, self_out_list,
                        scale=attn_scale,
                        source_mask=source_mask,
                        target_mask=target_mask
                    )

                    style_loss_shuffled = shuffled_loss(
                        q_list, ks_list, vs_list, self_out_list,
                        scale=attn_scale,
                        source_mask=source_mask,
                        target_mask=target_mask,
                        num_shuffle=num_p,
                        seed=seed
                    )

                    style_loss += (1-alpha_t) * style_loss_shuffled + alpha_t * style_loss_unshuffled

                elif mode == "only_shuffle":
                    style_loss = shuffled_loss(
                        q_list, ks_list, vs_list, self_out_list,
                        scale=attn_scale,
                        source_mask=source_mask,
                        target_mask=target_mask,
                        num_shuffle=num_p,
                        seed=seed
                    )

                elif mode == "without_shuffle":
                    style_loss = unshuffled_loss(
                        q_list, ks_list, vs_list, self_out_list,
                        scale=attn_scale,
                        source_mask=source_mask,
                        target_mask=target_mask
                    )
                else:
                    raise ValueError(f"Invalid mode: {mode}")

                if content_image is not None:
                    content_loss = q_loss(q_list, qc_list)

                loss = style_loss + content_loss * beta
                self.accelerator.backward(loss)
                optimizer.step()
                pbar.set_postfix(loss=loss.item(), time=t.item(), iter=j)
        images = self.latent2image(latents)
        # Offload all models
        self.maybe_free_model_hooks()
        return images

    def extract_feature(
            self,
            latent,
            t,
            embeds,
            add_noise=False,
    ):
        self.cache.clear()
        self.controller.step()
        if add_noise:
            noise = torch.randn_like(latent)
            latent_ = self.scheduler.add_noise(latent, noise, t)
        else:
            latent_ = latent
        _ = self.classifier(latent_, t, embeds)[0]
        return self.cache.get()
