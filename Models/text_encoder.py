import torch
import torch.nn as nn
from transformers import BertModel

class TextEncoder(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # Cross-attention layers
        self.cross_att = nn.MultiheadAttention(hidden_dim, num_heads=12)

    def encode_unimodal(self, text):
        outs = self.bert(**text)
        return outs.last_hidden_state[:, 0]

    def encode_multimodal(self, text, img_feat):
        txt = self.bert(**text).last_hidden_state
        img = img_feat.unsqueeze(1)

        txt_ca, _ = self.cross_att(txt.permute(1,0,2), 
                                   img.permute(1,0,2),
                                   img.permute(1,0,2))
        txt_fused = txt + txt_ca.permute(1,0,2)
        return txt_fused[:, 0]
