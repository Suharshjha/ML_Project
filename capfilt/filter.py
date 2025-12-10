import torch
import torch.nn.functional as F

class CaptionFilter:
    def __init__(self, model):
        self.model = model

    def is_good(self, image, text):
        score = self.model.forward_itm(image, text)
        prob = F.softmax(score, dim=-1)[0, 1]  # match prob
        return prob > 0.5
