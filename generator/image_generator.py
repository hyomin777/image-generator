import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel


class ImageGenerator(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.dtype = torch.float32

        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14"
        )

        self.text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).to(self.device)

        self.unet = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="unet"
        ).to(self.device).to(self.dtype)

        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse").to(self.device).to(self.dtype)

        self.scheduler = DDPMScheduler.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="scheduler")
        self.infer_scheduler = DDIMScheduler.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="scheduler")

        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

    def encode_text(self, text, max_length=77, padding='max_length'):
        encoder_device = next(self.text_encoder.parameters()).device

        if isinstance(text, str):
            text = [text]

        text_inputs = self.tokenizer(
            text,
            padding=padding,
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        ).to(encoder_device)

        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask
            )
        return outputs.last_hidden_state  # (B, 77, D)

    def decode_latents(self, latents):
        vae_device = next(self.vae.parameters()).device

        latents = latents.to(vae_device)
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    def forward(self, text, num_inference_steps=50, guidance_scale=7.5):
        unet_device = next(self.unet.parameters()).device

        text_embeddings = self.encode_text(text)
        uncond_embeddings = self.encode_text([""])
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings]).to(unet_device)

        latents = torch.randn(
            (1, self.unet.config.in_channels, 64, 64),
            device=unet_device,
            dtype=self.dtype
        )

        self.infer_scheduler.set_timesteps(num_inference_steps)
        for t in self.infer_scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.infer_scheduler.scale_model_input(latent_model_input, t)
            t_tensor = t.to(unet_device)

            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input, t_tensor, encoder_hidden_states=text_embeddings
                ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.infer_scheduler.step(noise_pred, t, latents).prev_sample.to(self.dtype)

        return self.decode_latents(latents)

    def train_step(self, image, text, cond_dropout_prob=0.1):
        vae_device = next(self.vae.parameters()).device
        unet_device = next(self.unet.parameters()).device

        with torch.no_grad():
            latent = self.vae.encode(image.to(vae_device, self.dtype)).latent_dist.sample()
            latent = latent * 0.18215

        latent = latent.to(unet_device)
        bsz = latent.shape[0]

        t = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,), device=unet_device).long()
        noise = torch.randn_like(latent).to(unet_device)
        noisy_latent = self.scheduler.add_noise(latent, noise, t)

        if torch.rand(1).item() < cond_dropout_prob:
            text = [""] * bsz

        text_embed = self.encode_text(text, padding='longest').to(unet_device)

        scaled_latent = self.scheduler.scale_model_input(noisy_latent, t)
        noise_pred = self.unet(scaled_latent, t, encoder_hidden_states=text_embed).sample
        loss = F.mse_loss(noise_pred, noise)
        return loss
