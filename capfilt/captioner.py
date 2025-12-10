import torch

class Captioner:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, image, max_len=25):
        ids = torch.tensor([[self.tokenizer.cls_token_id]])
        for _ in range(max_len):
            logits = self.model.forward_lm(image, ids)
            next_id = torch.multinomial(
                torch.softmax(logits[:, -1], dim=-1), 1
            )
            ids = torch.cat([ids, next_id], dim=1)
        return self.tokenizer.decode(ids[0])
