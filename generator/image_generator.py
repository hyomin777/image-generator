import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.models.attention_processor import AttnProcessor
from encoder.text_encoder import TextEncoder
from tokenizer.tokenizer import load_tokenizer


class ImageGenerator(nn.Module):
    def __init__(self, device, tokenizer_path, text_encoder_path):
        super().__init__()
        self.device = device
        self.dtype = torch.float16

        self.tokenizer = load_tokenizer(tokenizer_path)

        state_dict = torch.load(text_encoder_path, map_location=self.device)
        self.text_encoder = TextEncoder(
            vocab_size=self.tokenizer.vocab_size,
            projection_dim=state_dict['projection.weight'].shape[0]
        ).to(self.device).to(self.dtype)
        self.text_encoder.load_state_dict(state_dict)
        self.text_encoder = self.text_encoder.half()

        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse").to(self.device).to(self.dtype)
        self.unet = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="unet", cross_attention_dim=512, ignore_mismatched_sizes=True, low_cpu_mem_usage=False).to(self.device).to(self.dtype)

        # Set default attention processor to avoid LoRA-related errors
        self.unet.set_attn_processor(AttnProcessor())

        self.scheduler = DDPMScheduler.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="scheduler")

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

    def encode_text(self, text):
        if isinstance(text, str):
            text = [text]
        text_inputs = self.tokenizer(
            text, padding=True, max_length=128,
            truncation=True, return_tensors="pt"
        )
        attention_mask = text_inputs.attention_mask.to(self.device)
        text_input_ids = text_inputs.input_ids.to(self.device)
        with torch.no_grad():
            text_embeddings = self.text_encoder(input_ids=text_input_ids, attention_mask=attention_mask, return_pooled=False)
        return text_embeddings

    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    def forward(self, text, num_inference_steps=50, guidance_scale=7.5):
        text_embeddings = self.encode_text(text)
        uncond_input = [""]
        uncond_embeddings = self.encode_text(uncond_input)
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        latents = torch.randn(
            (1, self.unet.config.in_channels, 64, 64),
            device=self.device,
            dtype=self.dtype
        )

        for t in timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            t_tensor = t.to(self.device).to(self.dtype)
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input, t_tensor, encoder_hidden_states=text_embeddings
                ).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample.to(self.dtype)

        return self.decode_latents(latents)

    def train_step(self, image, text):
        with torch.no_grad():
            latent = self.vae.encode(image.to(self.device, self.dtype)).latent_dist.sample()
            latent = latent * 0.18215

        bsz = latent.shape[0]
        t = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()
        noise = torch.randn_like(latent)
        noisy_latent = self.scheduler.add_noise(latent, noise, t)
        text_embed = self.encode_text(text)
        noise_pred = self.unet(noisy_latent, t.to(self.dtype), encoder_hidden_states=text_embed).sample
        loss = F.mse_loss(noise_pred, noise)
        return loss
