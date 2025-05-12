import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, init_temp=0.07):
        super().__init__()
        log_tau = torch.log(torch.tensor(init_temp))
        self.log_temp = nn.Parameter(log_tau)

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp().clamp(1e-4, 1e4)

    def forward(self, text_embeds, image_embeds):
        text_embeds = F.normalize(text_embeds, dim=-1)
        image_embeds = F.normalize(image_embeds, dim=-1)

        logits = text_embeds @ image_embeds.T
        logits = logits / self.temperature

        batch_size = logits.size(0)
        labels = torch.arange(batch_size, device=logits.device)

        loss_t2i = F.cross_entropy(logits, labels)
        loss_i2t = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_t2i + loss_i2t)
