import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5TokenizerFast

VIS_REGIONS = 36
VIS_FEAT_DIM = 256
MODEL_NAME = "t5-small"

class VLT5(nn.Module):
    def __init__(self):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        self.hidden = self.t5.config.d_model
        self.vis_proj = nn.Linear(VIS_FEAT_DIM, self.hidden)

        self.region_embed = nn.Embedding(VIS_REGIONS, self.hidden)

    def forward(self, input_ids, attention_mask, labels, visual_feats):
        bs = input_ids.size(0)

        text_embed = self.t5.encoder.embed_tokens(input_ids)

        vis_embed = self.vis_proj(visual_feats)
        region_ids = torch.arange(VIS_REGIONS, device=visual_feats.device).unsqueeze(0)
        vis_embed = vis_embed + self.region_embed(region_ids)

        encoder_embed = torch.cat([vis_embed, text_embed], dim=1)

        vis_mask = torch.ones(bs, VIS_REGIONS, device=input_ids.device)
        enc_mask = torch.cat([vis_mask, attention_mask], dim=1)

        out = self.t5(
            inputs_embeds=encoder_embed,
            attention_mask=enc_mask,
            labels=labels,
        )
        return out


def load_tokenizer():
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
    vis_tokens = [f"<vis_{i}>" for i in range(VIS_REGIONS)]
    tokenizer.add_tokens(vis_tokens)
    return tokenizer
