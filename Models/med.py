import torch
import torch.nn as nn
from models.vit_encoder import ViTEncoder
from models.text_encoder import TextEncoder
from models.text_decoder import TextDecoder
from models.itm_head import ITMHead
from models.lm_head import LMHead

class BLIP(nn.Module):
    def __init__(self, vit_name="vit_base", hidden_dim=768):
        super().__init__()

        # Vision Transformer
        self.visual_encoder = ViTEncoder(vit_name)

        # Text encoders / decoders share layers except SA
        self.text_encoder = TextEncoder(hidden_dim=hidden_dim)
        self.text_decoder = TextDecoder(hidden_dim=hidden_dim)

        # Heads
        self.itm_head = ITMHead(hidden_dim)
        self.lm_head  = LMHead(hidden_dim)

    def forward_itc(self, image, text):
        img_feat = self.visual_encoder(image)
        txt_feat = self.text_encoder.encode_unimodal(text)
        return img_feat, txt_feat

    def forward_itm(self, image, text):
        img_feat = self.visual_encoder(image)
        txt_feat = self.text_encoder.encode_multimodal(text, img_feat)
        score = self.itm_head(txt_feat)
        return score

    def forward_lm(self, image, text_ids):
        img_feat = self.visual_encoder(image)
        logits = self.text_decoder.decode(img_feat, text_ids)
        logits = self.lm_head(logits)
        return logits
