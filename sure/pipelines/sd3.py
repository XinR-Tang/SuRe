import os
import inspect
from typing import List, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import StableDiffusion3Pipeline
from tqdm import tqdm

from ..utils.adapters import register_sd3_attn_control, Controller, DataCache
from ..utils.losses import unshuffled_loss, shuffled_loss, q_loss

# Copied from diffusers pipelines that encode images through a VAE.
def retrieve_latents(encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")

class SD3_Pipeline(StableDiffusion3Pipeline):
    def freeze(self):
        self.transformer.requires_grad_(False)
        self.vae.requires_grad_(False)
        if self.text_encoder is not None:
            self.text_encoder.requires_grad_(False)
        if self.text_encoder_2 is not None:
            self.text_encoder_2.requires_grad_(False)
        if self.text_encoder_3 is not None:
            self.text_encoder_3.requires_grad_(False)

    @torch.no_grad()
    def image2latent(self, image):
        dtype = next(self.vae.parameters()).dtype
        device = self._execution_device
        image = image.to(device=device, dtype=dtype) * 2.0 - 1.0
        latent = retrieve_latents(self.vae.encode(image))
        latent = (latent - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return latent

    @torch.no_grad()
    def latent2image(self, latent):
        dtype = next(self.vae.parameters()).dtype
        device = self._execution_device
        latent = latent.to(device=device, dtype=dtype)
        latent = (latent / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latent, return_dict=False)[0]
        return (image * 0.5 + 0.5).clamp(0, 1)

    def init(self, mixed_precision="no", enable_gradient_checkpoint=False):
        self.freeze()
        self.accelerator = Accelerator(mixed_precision=mixed_precision, gradient_accumulation_steps=1)
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        self.transformer.to(self.accelerator.device, dtype=weight_dtype)
        self.vae.to(self.accelerator.device, dtype=weight_dtype)
        if self.text_encoder is not None:
            self.text_encoder.to(self.accelerator.device, dtype=weight_dtype)
        if self.text_encoder_2 is not None:
            self.text_encoder_2.to(self.accelerator.device, dtype=weight_dtype)
        if self.text_encoder_3 is not None:
            self.text_encoder_3.to(self.accelerator.device, dtype=weight_dtype)
        if enable_gradient_checkpoint:
            self.transformer.enable_gradient_checkpointing()

    def encode_null_prompt(self, device, max_sequence_length=256):
        prompt_embeds, _, pooled_prompt_embeds, _ = self.encode_prompt(
            prompt="",
            prompt_2="",
            prompt_3="",
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            max_sequence_length=max_sequence_length,
        )
        return prompt_embeds, pooled_prompt_embeds

    def extract_feature(
        self,
        latents,
        timestep,
        prompt_embeds,
        pooled_prompt_embeds,
    ):
        self.cache.clear()
        self.controller.step()

        model_dtype = next(self.transformer.parameters()).dtype
        hidden_states = latents.to(dtype=model_dtype)
        timestep = timestep.expand(hidden_states.shape[0]).to(device=hidden_states.device)
        prompt_embeds = prompt_embeds.to(dtype=model_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=model_dtype)

        self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
        return self.cache.get()

    def stylization(
        self,
        style_image=None,
        content_image=None,
        controller=None,
        lr=0.015,
        iters=2,
        beta=0.20,
        attn_scale=1.0,
        mixed_precision="bf16",
        enable_gradient_checkpoint=True,
        height=512,
        width=512,
        num_inference_steps=200,
        batch_size=1,
        generator=None,
        latents=None,
        source_mask=None,
        target_mask=None,
        num_p=1,
        seed=42,
        mode="cos_weight",
        max_sequence_length=256,
    ):
        self.init(mixed_precision=mixed_precision, enable_gradient_checkpoint=enable_gradient_checkpoint)
        device = self._execution_device

        null_embeds, null_pooled_embeds = self.encode_null_prompt(
            device=device,
            max_sequence_length=max_sequence_length,
        )

        style_latent = self.image2latent(style_image)
        content_latent = self.image2latent(content_image)

        if latents is None:
            latents = content_latent.repeat(batch_size, 1, 1, 1).detach().clone()
        else:
            latents = latents.to(device)

        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None):
            _, _, latent_h, latent_w = latents.shape
            image_seq_len = (latent_h // self.transformer.config.patch_size) * (
                latent_w // self.transformer.config.patch_size
            )
            scheduler_kwargs["mu"] = calculate_shift(
                image_seq_len,
                self.scheduler.config.base_image_seq_len,
                self.scheduler.config.max_image_seq_len,
                self.scheduler.config.base_shift,
                self.scheduler.config.max_shift,
            )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            **scheduler_kwargs,
        )
        self._num_timesteps = len(timesteps)
        total_steps = self.scheduler.config.num_train_timesteps

        self.cache = DataCache()
        self.controller = controller
        register_sd3_attn_control(self.transformer.transformer_blocks, controller=controller, cache=self.cache)
        print("Total self attention layers of transformer: ", controller.num_self_layers)
        print("Self attention layers for AD: ", controller.self_layers)

        latents = latents.detach().float()
        optimizer = torch.optim.Adam([latents.requires_grad_()], lr=lr)
        optimizer = self.accelerator.prepare(optimizer)

        print(f"style_latent: {style_latent.shape}")
        print(f"content_latent: {content_latent.shape}")
        print(f"latents: {latents.shape}")
        print(f"start generate with mode:{mode}!!!!!")

        pbar = tqdm(timesteps, desc="Optimize")
        for i, t in enumerate(pbar):
            timestep = t.expand(1)
            with torch.no_grad():
                qs_list, ks_list, vs_list, _ = self.extract_feature(
                    style_latent,
                    timestep,
                    null_embeds.repeat(style_latent.shape[0], 1, 1),
                    null_pooled_embeds.repeat(style_latent.shape[0], 1),
                )
                qc_list, _, _, _ = self.extract_feature(
                    content_latent,
                    timestep,
                    null_embeds.repeat(content_latent.shape[0], 1, 1),
                    null_pooled_embeds.repeat(content_latent.shape[0], 1),
                )

            for j in range(iters):
                style_loss = 0
                content_loss = 0
                optimizer.zero_grad()
                q_list, _, _, self_out_list = self.extract_feature(
                    latents,
                    timestep,
                    null_embeds.repeat(latents.shape[0], 1, 1),
                    null_pooled_embeds.repeat(latents.shape[0], 1),
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

                content_loss = q_loss(q_list, qc_list)
                loss = style_loss + content_loss * beta
                self.accelerator.backward(loss)
                optimizer.step()
                pbar.set_postfix(loss=loss.item(), time=t.item(), iter=j)

        torch.cuda.empty_cache()
        latents = latents.detach()
        return self.latent2image(latents)

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(f"The current scheduler class {scheduler.__class__} does not support custom timesteps.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(f"The current scheduler class {scheduler.__class__} does not support custom sigmas.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps
