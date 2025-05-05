import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel
from lora import LoRALinear


class ImageEncoder(nn.Module):
    def __init__(self, projection_dim=768):
        super().__init__()
        self.vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

        # LoRA
        for param in self.vision_model.parameters():
            param.requires_grad = False
        for i in [-2, -1]:
            block = self.vision_model.vision_model.encoder.layers[i]
            sa = block.self_attn
            sa.q_proj = LoRALinear(sa.q_proj, r=4, alpha=1.0)
            sa.v_proj = LoRALinear(sa.v_proj, r=4, alpha=1.0)

        # Projection layer to match TextEncoder
        self.projection = nn.Linear(self.vision_model.config.hidden_size, projection_dim)

    def forward(self, pixel_values, return_pooled=True):
        outputs = self.vision_model(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]  # (B, D)
        x = self.projection(cls_token)  # (B, projection_dim)
        return F.normalize(x, dim=-1) if return_pooled else x
