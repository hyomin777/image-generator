import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch.nn as nn
from transformers import CLIPModel
from lora import LoRALinear


def load_image_encoder(device):
    image_encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    image_encoder.eval()
    image_encoder.visual_projection = nn.Identity()

    for param in image_encoder.parameters():
        param.requires_grad = False
    for i in [-2, -1]:
        block = image_encoder.vision_model.encoder.layers[i]
        sa = block.self_attn
        sa.q_proj = LoRALinear(sa.q_proj, r=4, alpha=1.0).to(device)
        sa.v_proj = LoRALinear(sa.v_proj, r=4, alpha=1.0).to(device)

    return image_encoder
