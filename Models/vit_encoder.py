from transformers import ViTModel
import torch.nn as nn

class ViTEncoder(nn.Module):
    def __init__(self, name="vit-base-patch16-224"):
        super().__init__()
        self.vit = ViTModel.from_pretrained(name)

    def forward(self, images):
        outputs = self.vit(images)
        return outputs.last_hidden_state[:, 0]  # CLS token
