def train_pretrain(model, dataloader, optimizer):
    model.train()
    for img, text in dataloader:

        img_feat, txt_feat = model.forward_itc(img, text)
        loss_itc = contrastive_loss(img_feat, txt_feat)

        score = model.forward_itm(img, text)
        loss_itm = torch.nn.functional.cross_entropy(score,
                                                     text["labels"])

        logits = model.forward_lm(img, text["lm_ids"])
        loss_lm = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            text["lm_labels"].view(-1)
        )

        loss = loss_itc + loss_itm + loss_lm
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
