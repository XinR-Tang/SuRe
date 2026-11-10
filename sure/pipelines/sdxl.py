import os
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

from accelerate import Accelerator
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from diffusers import StableDiffusionXLPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.utils.torch_utils import is_compiled_module
from tqdm import tqdm

from ..utils.adapters import register_attn_control, DataCache
from ..utils.losses import unshuffled_loss, shuffled_loss, q_loss

class SDXL_Pipeline(StableDiffusionXLPipeline):
    def freeze(self):
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.classifier.requires_grad_(False)

    @torch.no_grad()
    def image2latent(self, image):
        dtype = next(self.vae.parameters()).dtype
        device = self._execution_device
        image = image.to(device=device, dtype=dtype) * 2.0 - 1.0
        latent = self.vae.encode(image)["latent_dist"].mean
        latent = latent * self.vae.config.scaling_factor
        return latent

    @torch.no_grad()
    def latent2image(self, latent):
        dtype = next(self.vae.parameters()).dtype
        device = self._execution_device
        latent = latent.to(device=device, dtype=dtype)
        latent = latent / self.vae.config.scaling_factor
        image = self.vae.decode(latent)[0]
        return (image * 0.5 + 0.5).clamp(0, 1)

    def init(self, enable_gradient_checkpoint):
        self.classifier = self.unet
        self.freeze()
        self.enable_vae_slicing()
        # self.enable_model_cpu_offload()
        # self.enable_vae_tiling()
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        self.unet.to(self.accelerator.device, dtype=weight_dtype)
        self.vae.to(self.accelerator.device, dtype=weight_dtype)
        self.text_encoder.to(self.accelerator.device, dtype=weight_dtype)
        self.text_encoder_2.to(self.accelerator.device, dtype=weight_dtype)
        self.classifier.to(self.accelerator.device, dtype=weight_dtype)
        if enable_gradient_checkpoint:
            self.classifier.enable_gradient_checkpointing()
            # self.classifier.train()
     
    def stylization(
        self,
        latents=None,
        attn_scale=1.0,
        lr=0.02,
        iters=1,
        beta=0.25,
        width=1024,
        height=1024,
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

        self.accelerator = Accelerator(
            mixed_precision=mixed_precision, gradient_accumulation_steps=1
        )
        self.init(enable_gradient_checkpoint)

        device = self._execution_device
        style_latent = self.image2latent(style_image)
        print(f"style_latent: {style_latent.shape}")
        print(f"The number of shuffling: {num_p}")

        if latents is None:
            latents = torch.randn((batch_size, 4, height, width), device=device)
        else:
            latents = latents.to(device)

        content_latent = None
        if content_image is not None:
            content_latent = self.image2latent(content_image)
            latents = torch.cat([content_latent.clone()] * batch_size)

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        total_steps = self.scheduler.config.num_train_timesteps

        null_embeds, _, null_pooled_embeds, _ = self.encode_prompt("", device=device)
        original_size = (height, width)
        target_size = (height, width)
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(null_pooled_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim
        null_add_time_ids = self._get_add_time_ids(
            original_size,
            (0, 0),
            target_size,
            dtype=null_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        ).to(device)

        null_embeds_for_latents = null_embeds.repeat(latents.shape[0], 1, 1)
        null_embeds_for_style = null_embeds.repeat(style_latent.shape[0], 1, 1)
        null_added_cond_kwargs_for_latents = {
            "text_embeds": null_pooled_embeds.repeat(latents.shape[0], 1),
            "time_ids": null_add_time_ids.repeat(latents.shape[0], 1),
        }
        null_added_cond_kwargs_for_style = {
            "text_embeds": null_pooled_embeds.repeat(style_latent.shape[0], 1),
            "time_ids": null_add_time_ids.repeat(style_latent.shape[0], 1),
        }
        if content_latent is not None:
            null_embeds_for_content = null_embeds.repeat(content_latent.shape[0], 1, 1)
            null_added_cond_kwargs_for_content = {
                "text_embeds": null_pooled_embeds.repeat(content_latent.shape[0], 1),
                "time_ids": null_add_time_ids.repeat(content_latent.shape[0], 1),
            }

        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(0).repeat(latents.shape[0]).to(device)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor,
                embedding_dim=self.unet.config.time_cond_proj_dim,
            ).to(device=device, dtype=latents.dtype)

        self.cache = DataCache()
        self.controller = controller
        register_attn_control(
            self.classifier, controller=self.controller, cache=self.cache
        )
        print("Total self attention layers of Unet: ", controller.num_self_layers)
        print("Self attention layers for AD: ", controller.self_layers)

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
                    timestep_cond,
                    null_added_cond_kwargs_for_style,
                    add_noise=False,
                )
                
                if content_latent is not None:
                    qc_list, kc_list, vc_list, c_out_list = self.extract_feature(
                        content_latent,
                        t,
                        null_embeds_for_content,
                        timestep_cond,
                        null_added_cond_kwargs_for_content,
                        add_noise=False,
                    )

            for j in range(iters):
                style_loss = 0
                content_loss = 0
                optimizer.zero_grad()
                q_list, k_list, v_list, self_out_list = self.extract_feature(
                    latents,
                    t,
                    null_embeds_for_latents,
                    timestep_cond,
                    null_added_cond_kwargs_for_latents,
                    add_noise=False,
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

                content_loss = q_loss(q_list, qc_list) if content_latent is not None else 0
                loss = style_loss + content_loss * beta
                self.accelerator.backward(loss)
                optimizer.step()
                pbar.set_postfix(loss=loss.item(), time=t.item(), iter=j)

        images = self.latent2image(latents)
        self.maybe_free_model_hooks()
        return images

    def extract_feature(
        self,
        latent,
        t,
        encoder_hidden_states,
        timestep_cond,
        added_cond_kwargs,
        add_noise=False,
    ):
        self.cache.clear()
        self.controller.step()
        if add_noise:
            noise = torch.randn_like(latent)
            latent_ = self.scheduler.add_noise(latent, noise, t)
        else:
            latent_ = latent

        model_dtype = next(self.classifier.parameters()).dtype
        latent_ = latent_.to(dtype=model_dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype=model_dtype)
        if timestep_cond is not None:
            timestep_cond = timestep_cond.to(dtype=model_dtype)
        if added_cond_kwargs is not None:
            added_cond_kwargs = {
                key: value.to(dtype=model_dtype) if torch.is_floating_point(value) else value
                for key, value in added_cond_kwargs.items()
            }

        self.classifier(
            latent_,
            t,
            encoder_hidden_states=encoder_hidden_states,
            timestep_cond=timestep_cond,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        return self.cache.get()
