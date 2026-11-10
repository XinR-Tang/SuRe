import inspect
from typing import List, Optional, Union

import numpy as np
import torch
from accelerate import Accelerator
from diffusers import FluxPipeline
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm

from ..utils.adapters import DataCache, register_flux_attn_control
from ..utils.losses import q_loss, shuffled_loss, unshuffled_loss
from ..utils.utils import adain


def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")


class Flux_Pipeline(FluxPipeline):
    def freeze(self):
        self.transformer.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.vae.requires_grad_(False)

    @torch.no_grad()
    def image2latent(self, image):
        dtype = next(self.vae.parameters()).dtype
        device = self._execution_device
        image = image.to(device=device, dtype=dtype) * 2.0 - 1.0
        latent = retrieve_latents(self.vae.encode(image))
        return (latent - self.vae.config.shift_factor) * self.vae.config.scaling_factor

    @torch.no_grad()
    def latent2image(self, latent, height, width):
        dtype = next(self.vae.parameters()).dtype
        device = self._execution_device
        latent = latent.to(device=device, dtype=dtype)
        latents = self._unpack_latents(latent, height, width, self.vae_scale_factor)
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        return (image * 0.5 + 0.5).clamp(0, 1)

    def init(self, mixed_precision="no", enable_gradient_checkpoint=False):
        # `test.py` reuses one pipeline for many image pairs.  Preparing an
        # already prepared transformer nests Accelerate forward wrappers until
        # a RecursionError is raised, so create and prepare it only once.
        if hasattr(self, "_sure_accelerator"):
            if self._sure_mixed_precision != mixed_precision:
                raise ValueError(
                    "Cannot change mixed_precision when reusing an initialized "
                    "FLUX pipeline. Create a new pipeline instead."
                )
            if enable_gradient_checkpoint:
                self.transformer.enable_gradient_checkpointing()
            return

        self._sure_accelerator = Accelerator(
            mixed_precision=mixed_precision, gradient_accumulation_steps=1
        )
        self._sure_mixed_precision = mixed_precision
        self.accelerator = self._sure_accelerator
        self.freeze()
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        self.transformer.to(self.accelerator.device, dtype=weight_dtype)
        self.vae.to(self.accelerator.device, dtype=weight_dtype)
        self.text_encoder.to(self.accelerator.device, dtype=weight_dtype)
        self.text_encoder_2.to(self.accelerator.device, dtype=weight_dtype)
        self.transformer = self.accelerator.prepare(self.transformer)
        if enable_gradient_checkpoint:
            self.transformer.enable_gradient_checkpointing()

    def extract_feature(
        self,
        latents,
        timestep,
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
        image_ids,
        guidance,
    ):
        self.cache.clear()
        self.controller.step()

        model_dtype = next(self.transformer.parameters()).dtype
        hidden_states = latents.to(device=self._execution_device, dtype=model_dtype)
        timestep = timestep.expand(hidden_states.shape[0]).to(
            device=hidden_states.device, dtype=model_dtype
        )
        prompt_embeds = prompt_embeds.to(device=hidden_states.device, dtype=model_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(
            device=hidden_states.device, dtype=model_dtype
        )
        if guidance is not None:
            guidance = guidance.expand(hidden_states.shape[0])

        self.transformer(
            hidden_states=hidden_states,
            timestep=timestep / self.scheduler.config.num_train_timesteps,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
        return self.cache.get()

    def stylization(
        self,
        style_image=None,
        content_image=None,
        controller=None,
        lr=0.02,
        iters=1,
        beta=0.25,
        attn_scale=1.0,
        mixed_precision="bf16",
        enable_gradient_checkpoint=False,
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
        strength=0.2,
        guidance_scale=3.5,
        prompt="",
        prompt_2=None,
        max_sequence_length=512,
    ):
        if content_image is None:
            raise ValueError("content_image is required for FLUX stylization.")

        self.init(
            mixed_precision=mixed_precision,
            enable_gradient_checkpoint=enable_gradient_checkpoint,
        )
        device = self._execution_device
        null_embeds, null_pooled_embeds, null_text_ids = self.encode_prompt(
            prompt="",
            prompt_2="",
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=batch_size,
            max_sequence_length=max_sequence_length,
        )

        num_channels_latents = self.transformer.config.in_channels // 4
        style_latent = self.image2latent(style_image)
        content_latent = self.image2latent(content_image)
        style_latent = self._pack_latents(
            style_latent,
            style_latent.shape[0],
            num_channels_latents,
            style_latent.shape[2],
            style_latent.shape[3],
        )
        content_latent = self._pack_latents(
            content_latent,
            content_latent.shape[0],
            num_channels_latents,
            content_latent.shape[2],
            content_latent.shape[3],
        )

        if latents is None:
            latents = content_latent.repeat(batch_size, 1, 1).detach().clone()
        else:
            latents = latents.to(device)

        latents, latent_image_ids = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            null_embeds.dtype,
            device,
            generator,
            latents,
        )
        _, style_image_ids = self.prepare_latents(
            style_latent.shape[0],
            num_channels_latents,
            height,
            width,
            null_embeds.dtype,
            device,
            generator,
            style_latent,
        )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
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
            sigmas=sigmas,
            mu=mu,
        )
        strength = max(0.0, min(1.0, strength))
        init_timestep = max(
            1, min(int(num_inference_steps * strength), num_inference_steps)
        )
        timesteps = timesteps[num_inference_steps - init_timestep :]
        noise = randn_tensor(
            latents.shape,
            generator=generator,
            device=device,
            dtype=latents.dtype,
        )
        latents = self.scheduler.scale_noise(
            latents.to(device=device, dtype=latents.dtype), timesteps[:1], noise
        )
        self._num_timesteps = len(timesteps)
        total_steps = self.scheduler.config.num_train_timesteps

        self.cache = DataCache()
        self.controller = controller
        register_flux_attn_control(
            self.transformer.transformer_blocks,
            controller=controller,
            cache=self.cache,
        )
        register_flux_attn_control(
            self.transformer.single_transformer_blocks,
            controller=controller,
            cache=self.cache,
        )
        if self.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1], guidance_scale, device=device, dtype=torch.float32
            )
            null_guidance = torch.full([1], 1, device=device, dtype=torch.float32)
        else:
            guidance = None
            null_guidance = None

        print(f"Total self attention layers of transformer: {controller.num_self_layers}")
        print(f"Self attention layers for stylization: {controller.self_layers}")
        print(f"style_latent: {style_latent.shape}")
        print(f"content_latent: {content_latent.shape}")
        print(f"latents: {latents.shape}")
        print(f"start generate with mode:{mode}!!!!!")

        pbar = tqdm(timesteps, desc="Optimize")
        for i, t in enumerate(pbar):
            with torch.no_grad():
                model_dtype = next(self.transformer.parameters()).dtype
                denoise_latents = latents.to(device=device, dtype=model_dtype)
                denoise_timestep = t.expand(denoise_latents.shape[0]).to(
                    device=device, dtype=model_dtype
                )
                denoise_guidance = (
                    guidance.expand(denoise_latents.shape[0])
                    if guidance is not None
                    else None
                )
                noise_pred = self.transformer(
                    hidden_states=denoise_latents,
                    timestep=denoise_timestep / total_steps,
                    guidance=denoise_guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
                latents = self.scheduler.step(
                    noise_pred, t, denoise_latents, return_dict=False
                )[0]

                if i < len(timesteps) - 1:
                    timestep = timesteps[i + 1 : i + 2]
                    style_noise = randn_tensor(
                        style_latent.shape,
                        generator=generator,
                        device=device,
                        dtype=style_latent.dtype,
                    )
                    style_latent_t = self.scheduler.scale_noise(
                        style_latent, timestep, style_noise
                    )
                else:
                    timestep = torch.zeros(
                        1, device=device, dtype=timesteps.dtype
                    )
                    style_latent_t = style_latent

                latents = adain(latents, style_latent_t)
                latents = latents.detach().to(dtype=torch.bfloat16)
                optimizer = torch.optim.Adam([latents.requires_grad_()], lr=lr)
                optimizer = self.accelerator.prepare(optimizer)

                _, ks_list, vs_list, _ = self.extract_feature(
                    style_latent_t,
                    timestep,
                    null_embeds.repeat(style_latent.shape[0], 1, 1),
                    null_pooled_embeds.repeat(style_latent.shape[0], 1),
                    null_text_ids,
                    style_image_ids,
                    null_guidance,
                )
                qc_list, _, _, _ = self.extract_feature(
                    content_latent,
                    timestep,
                    null_embeds.repeat(content_latent.shape[0], 1, 1),
                    null_pooled_embeds.repeat(content_latent.shape[0], 1),
                    null_text_ids,
                    latent_image_ids,
                    null_guidance,
                )

            for j in range(iters):
                optimizer.zero_grad()
                q_list, _, _, self_out_list = self.extract_feature(
                    latents,
                    timestep,
                    null_embeds.repeat(latents.shape[0], 1, 1),
                    null_pooled_embeds.repeat(latents.shape[0], 1),
                    null_text_ids,
                    latent_image_ids,
                    null_guidance,
                )

                if mode == "cos_weight":
                    alpha_t = 0.5 * (
                        1 + torch.cos(torch.pi * t.float() / total_steps)
                    )
                    style_loss_unshuffled = unshuffled_loss(
                        q_list,
                        ks_list,
                        vs_list,
                        self_out_list,
                        scale=attn_scale,
                        source_mask=source_mask,
                        target_mask=target_mask,
                    )
                    style_loss_shuffled = shuffled_loss(
                        q_list,
                        ks_list,
                        vs_list,
                        self_out_list,
                        scale=attn_scale,
                        source_mask=source_mask,
                        target_mask=target_mask,
                        num_shuffle=num_p,
                        seed=seed,
                    )
                    style_loss = (
                        (1 - alpha_t) * style_loss_shuffled
                        + alpha_t * style_loss_unshuffled
                    )
                elif mode == "only_shuffle":
                    style_loss = shuffled_loss(
                        q_list,
                        ks_list,
                        vs_list,
                        self_out_list,
                        scale=attn_scale,
                        source_mask=source_mask,
                        target_mask=target_mask,
                        num_shuffle=num_p,
                        seed=seed,
                    )
                elif mode == "without_shuffle":
                    style_loss = unshuffled_loss(
                        q_list,
                        ks_list,
                        vs_list,
                        self_out_list,
                        scale=attn_scale,
                        source_mask=source_mask,
                        target_mask=target_mask,
                    )
                else:
                    raise ValueError(f"Invalid mode: {mode}")

                content_loss = q_loss(q_list, qc_list)
                loss = style_loss + content_loss * beta
                self.accelerator.backward(loss)
                optimizer.step()
                pbar.set_postfix(loss=loss.item(), time=t.item(), iter=j)

        torch.cuda.empty_cache()
        return self.latent2image(latents.detach(), height, width)


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
        accepts_timesteps = "timesteps" in inspect.signature(
            scheduler.set_timesteps
        ).parameters
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__} does not support custom timesteps."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accepts_sigmas = "sigmas" in inspect.signature(
            scheduler.set_timesteps
        ).parameters
        if not accepts_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__} does not support custom sigmas."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps
