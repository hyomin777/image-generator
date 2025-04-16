import math
import torch
import torch.nn as nn


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


def extract_lora_weights(model):
    lora_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_weights[f"{name}.lora_A"] = module.lora_A.state_dict()
            lora_weights[f"{name}.lora_B"] = module.lora_B.state_dict()
    return lora_weights


def load_lora_weights(model, lora_weights):
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            if f"{name}.lora_A" in lora_weights:
                module.lora_A.load_state_dict(lora_weights[f"{name}.lora_A"])
                module.lora_B.load_state_dict(lora_weights[f"{name}.lora_B"])
