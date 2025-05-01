import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from encoder.text_encoder import TextEncoder
from tokenizer.tokenizer import load_tokenizer


class ImageGenerator(nn.Module):
    def __init__(self, device, text_encoder_path, tokenizer_path):
        super().__init__()
        self.device = device
        self.dtype = torch.float32

        self.tokenizer = load_tokenizer(tokenizer_path)

        text_encoder_state_dict = torch.load(text_encoder_path, map_location=self.device)
        self.text_encoder = TextEncoder(
            vocab_size=self.tokenizer.vocab_size
        ).to(self.device)
        self.text_encoder.load_state_dict(text_encoder_state_dict)

        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse").to(self.device).to(self.dtype)
        self.unet = load_unet()
        self.scheduler = DDPMScheduler.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="scheduler")

        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

    def encode_text(self, text, padding='max_length'):
        if isinstance(text, str):
            text = [text]

        text_inputs = self.tokenizer(
            text, padding=padding, max_length=128,
            truncation=True, return_tensors="pt"
        )

        encoder_device = next(self.text_encoder.parameters()).device

        attention_mask = text_inputs.attention_mask.to(device=encoder_device)
        text_input_ids = text_inputs.input_ids.to(encoder_device)

        with torch.no_grad():
            text_embeddings = self.text_encoder(input_ids=text_input_ids, attention_mask=attention_mask, return_pooled=False)
        
        return text_embeddings

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

        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            t_tensor = t.to(unet_device)

            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input, t_tensor, encoder_hidden_states=text_embeddings
                ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample.to(self.dtype)

        return self.decode_latents(latents)

    def train_step(self, image, text):
        vae_device = next(self.vae.parameters()).device
        unet_device = next(self.unet.parameters()).device

        with torch.no_grad():
            latent = self.vae.encode(image.to(vae_device, self.dtype)).latent_dist.sample()
            latent = latent * 0.18215

            latent_mean = latent.mean().item()
            latent_std = latent.std().item()
            if torch.isnan(latent).any() or torch.isinf(latent).any():
                print(f"[Warning] NaN or INF detected in latent! mean={latent_mean:.6f}, std={latent_std:.6f}")
                return None

            if latent_std > 3.0 or latent_std < 0.3:
                print(f"[train_step] Abnormal latent std detected! mean={latent_mean:.6f}, std={latent_std:.6f}")
                return None

        latent = latent.to(unet_device)
        bsz = latent.shape[0]
        t = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,), device=unet_device).long()
        noise = torch.randn_like(latent).to(unet_device)
        noisy_latent = self.scheduler.add_noise(latent, noise, t)

        text_embed = self.encode_text(text, 'longest')
        noise_pred = self.unet(noisy_latent, t, encoder_hidden_states=text_embed).sample
        loss = F.mse_loss(noise_pred, noise)

        if torch.isnan(loss) or  torch.isinf(loss) or loss.item() >= 2.0:
            print(f"[train_step] Invalid loss: {loss.item():.4f}")
            return None

        return loss


def load_unet():
    unet = UNet2DConditionModel(
        sample_size=64,
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(320, 640, 1280, 1280),
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        cross_attention_dim=768,
        attention_head_dim=8,
    )

    return unet
