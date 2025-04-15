import math
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, CLIPModel


class TagEncoder(nn.Module):
    def __init__(self, vocab_size=16384, embed_dim=256, num_layers=4, projection_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.projection = nn.Linear(embed_dim, projection_dim)

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)  # (B, T, D)
        x = self.transformer(x, src_key_padding_mask=~attention_mask.bool() if attention_mask is not None else None)
        x = x[:, 0]  # use first token (like CLS)
        return self.projection(x)


class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r=4, alpha=1.0):
        super().__init__()
        self.original = original_linear
        self.r = r
        self.alpha = alpha

        self.lora_A = nn.Linear(original_linear.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, original_linear.out_features, bias=False)

        # init
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # freeze original
        for param in self.original.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.original(x) + self.alpha * self.lora_B(self.lora_A(x))
