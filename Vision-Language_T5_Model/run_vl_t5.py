import torch
from vl_t5 import VLT5, load_tokenizer, VIS_REGIONS
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_region_features(img_id):
    return torch.load(f"featuresVLT5/{img_id}.pt")  # Updated folder name

tokenizer = load_tokenizer()
model = VLT5().to(DEVICE)
model.t5.resize_token_embeddings(len(tokenizer))

def ask_vqa(image_id, question):
    # Load region features from featuresVLT5 folder
    vis_feats = load_region_features(image_id).unsqueeze(0).to(DEVICE)

    # Input text prompt
    text = f"vqa question: {question}"
    enc = tokenizer(text, return_tensors="pt").to(DEVICE)

    # Combine visual + text embeddings
    out = model.t5.generate(
        max_length=30,
        do_sample=True,
        top_k=50,
        temperature=1.2,
        num_beams=1,
        encoder_outputs=model.t5.encoder(
            inputs_embeds=torch.cat([
                model.vis_proj(vis_feats) +
                model.region_embed(torch.arange(VIS_REGIONS).unsqueeze(0).to(DEVICE)),
                model.t5.encoder.embed_tokens(enc["input_ids"])
            ], dim=1),
            attention_mask=None,
            return_dict=True
        )
    )

    print("Answer:", tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    print("VL-T5 Ready!")

    # Make sure the feature file exists before running
    if not os.path.exists("featuresVLT5/img1.pt"):
        print("⚠️ Region features missing! Run: python extract_features.py")
    else:
        ask_vqa("img1", "Why is the person holding an umbrella?")