import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler

from encoder.text_encoder import TextEncoder
from tokenizer.tokenizer import load_tokenizer
from lora import LoRALinear


class ImageGenerator(nn.Module):
    def __init__(self, device, tokenizer_path):
        super().__init__()
        self.device = device
        self.dtype = torch.float16

        self.tokenizer = load_tokenizer(tokenizer_path)
        self.text_encoder = TextEncoder(vocab_size=self.tokenizer.vocab_size).to(self.device).to(self.dtype)
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse").to(self.device).to(self.dtype)
        self.unet = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="unet").to(self.device).to(self.dtype)
        self.scheduler = DDPMScheduler.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="scheduler")

        # Freeze models
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
    def encode_text(self, text):
        # Convert string to list (always process as batch)
        if isinstance(text, str):
            text = [text]

        # Tokenize
        text_inputs = self.tokenizer(
            text, padding=True, max_length=128,
            truncation=True, return_tensors="pt"
        )

        # Move input tensors to appropriate device
        attention_mask = text_inputs.attention_mask.to(self.device)
        text_input_ids = text_inputs.input_ids.to(self.device)

        # Get text embeddings
        with torch.no_grad():
            text_embeddings = self.text_encoder(input_ids=text_input_ids, attention_mask=attention_mask)

        return text_embeddings

    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    def forward(self, text, num_inference_steps=50, guidance_scale=7.5):
        # Get text embeddings
        text_embeddings = self.encode_text(text)

        # Generate unconditional embeddings (embeddings for empty string)
        uncond_input = [""]
        uncond_embeddings = self.encode_text(uncond_input)

        # Combine text embeddings and unconditional embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # Generate initial random noise
        latents = torch.randn(
            (1, self.unet.config.in_channels, 64, 64),
            device=self.device,
            dtype=self.dtype
        )

        # Denoising loop
        for t in timesteps:
            # Use latent directly without duplication (text embeddings are already combined)
            latent_model_input = latents
            # Duplicate batch dimension to create 2 samples (conditional and unconditional)
            latent_model_input = torch.cat([latent_model_input] * 2)
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)

            # Move timestep to correct device and type
            t_tensor = t.to(self.device).to(self.dtype)

            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input, t_tensor, encoder_hidden_states=text_embeddings
                ).sample

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Compute previous noisy sample
            latents = self.scheduler.step(
                noise_pred, t, latents).prev_sample.to(self.dtype)

        # Decode latents to image
        image = self.decode_latents(latents)

        return image
    
    def train_step(self, image, text):
        # Encode image into latent
        with torch.no_grad():
            latent = self.vae.encode(image.to(self.device, self.dtype)).latent_dist.sample()
            latent = latent * 0.18215

        # Sample timestep t
        bsz = latent.shape[0]
        t = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()

        # Add noise
        noise = torch.randn_like(latent)
        noisy_latent = self.scheduler.add_noise(latent, noise, t)

        # Text embedding
        text_embed = self.encode_text(text)

        # Predict noise
        noise_pred = self.unet(noisy_latent, t, encoder_hidden_states=text_embed).sample

        # Loss
        loss = F.mse_loss(noise_pred, noise)
        return loss


def load_image_generator(device, tokenizer_path):
    target_names = ("to_q", "to_k", "to_v", "to_out")

    image_generator = ImageGenerator(device, tokenizer_path)
    image_generator.unet.eval()
    for param in image_generator.unet.parameters():
        param.requires_grad = False

    for name, module in image_generator.unet.named_modules():
        if isinstance(module, nn.Linear) and any(key in name for key in target_names):
            parent_module = get_parent_module(image_generator.unet, name)
            attr_name = name.split('.')[-1]
            lora = LoRALinear(module).to(device, dtype=module.weight.dtype)
            setattr(parent_module, attr_name, lora)

    return image_generator.to(device)


def get_parent_module(model: nn.Module, module_name: str):
    names = module_name.split(".")
    parent = model
    for name in names[:-1]:
        parent = getattr(parent, name)
    return parent
