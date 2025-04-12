import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class TagEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased', projection_dim=512):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.bert.config.hidden_size, projection_dim)

    def forward(self, tags):
        inputs = self.tokenizer(tags, padding=True, truncation=True, return_tensor='pt').to(self.bert.device)
        outputs = self.bert(**inputs).last_hidden_state[:,][0] # CLS token
        return self.projection(outputs) # (batch, projection_dim)