import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler


class TextToImageModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Set consistent data type
        self.dtype = torch.float16  # to save GPU memory
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained models
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14").to(device).to(self.dtype)
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse").to(device).to(self.dtype)
        self.unet = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="unet").to(device).to(self.dtype)

        # Noise scheduler
        self.scheduler = DDPMScheduler.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="scheduler")

        # Freeze models
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
    def encode_text(self, text):
        # Move tensors to the same device and type as the model
        device = next(self.text_encoder.parameters()).device
        dtype = self.dtype

        # Convert string to list (always process as batch)
        if isinstance(text, str):
            text = [text]

        # Tokenize
        text_inputs = self.tokenizer(
            text, padding="max_length", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        # Move input tensors to appropriate device
        text_input_ids = text_inputs.input_ids.to(device)

        # Get text embeddings
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input_ids)[0].to(dtype)

        return text_embeddings

    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    def forward(self, text, num_inference_steps=50, guidance_scale=7.5):
        # Check device and type of the model
        device = next(self.unet.parameters()).device
        dtype = self.dtype

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

        # Generate initial random noise - explicitly specify device and type
        # Warning fix: access in_channels from config
        latents = torch.randn(
            (1, self.unet.config.in_channels, 64, 64),
            device=device,
            dtype=dtype
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
            t_tensor = t.to(device).to(dtype)

            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input, t_tensor, encoder_hidden_states=text_embeddings
                ).sample

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * \
                (noise_pred_text - noise_pred_uncond)

            # Compute previous noisy sample
            latents = self.scheduler.step(
                noise_pred, t, latents).prev_sample.to(dtype)

        # Decode latents to image
        image = self.decode_latents(latents)

        return image
    
    def train_step(self, image, text, noise_scheduler):
        device = next(self.parameters()).device
        dtype = self.dtype

        # Encode image into latent
        with torch.no_grad():
            latent = self.vae.encode(image.to(device, dtype)).latent_dist.sample()
            latent = latent * 0.18215

        # Sample timestep t
        bsz = latent.shape[0]
        t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()

        # Add noise
        noise = torch.randn_like(latent)
        noisy_latent = noise_scheduler.add_noise(latent, noise, t)

        # Text embedding
        text_embed = self.encode_text(text)

        # Predict noise
        noise_pred = self.unet(noisy_latent, t, encoder_hidden_states=text_embed).sample

        # Loss
        loss = F.mse_loss(noise_pred, noise)
        return loss
