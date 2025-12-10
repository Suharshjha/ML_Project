import torch

def contrastive_loss(image_feats, text_feats, temperature=0.07):
    logits = image_feats @ text_feats.t() / temperature
    labels = torch.arange(len(logits)).to(logits.device)
    loss_i = torch.nn.functional.cross_entropy(logits, labels)
    loss_t = torch.nn.functional.cross_entropy(logits.t(), labels)
    return (loss_i + loss_t) / 2
