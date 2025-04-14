import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


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
