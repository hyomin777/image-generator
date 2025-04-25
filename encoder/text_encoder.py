import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    def __init__(
            self,
            vocab_size=16384,
            embed_dim=512,
            num_layers=8,
            n_head=8,
            projection_dim=512,
            dropout=0.2
        ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_head,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.projection = nn.Linear(embed_dim, projection_dim)

    def forward(self, input_ids, attention_mask, return_pooled=True):
        x = self.embedding(input_ids)  # (B, T, D)
        x = self.transformer(
            x,
            src_key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
        )
        x = self.norm(x)

        if return_pooled:
            # CLIP-style pooled representation (B, D)
            x = self.projection(x[:, 0])  # Use first token (CLS)
            return F.normalize(x, dim=-1)
        else:
            # Sequence representation (B, T, D) for diffusion
            return self.projection(x)  # No normalize
