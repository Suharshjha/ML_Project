import torch
import torch.nn as nn
from transformers import BertConfig

class TextDecoder(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        cfg = BertConfig(is_decoder=True, add_cross_attention=True)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=12,
                dim_feedforward=3072,
            ),
            num_layers=12
        )

    def decode(self, img_feat, tgt_ids):
        img_feat = img_feat.unsqueeze(1).permute(1,0,2)
        tgt = tgt_ids.permute(1, 0, 2)
        out = self.decoder(tgt, img_feat)
        return out.permute(1, 0, 2)
