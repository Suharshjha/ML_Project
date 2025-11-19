# vqa_blip.py
from PIL import Image
from transformers import pipeline
import torch
import os

# --- Config ---
IMAGE_PATH = "images/img1.jpg"   # <- update if your image path is different
MODEL_NAME = "Salesforce/blip-vqa-base"  # small/fast BLIP VQA model
# MODEL_NAME alternatives: "dandelin/vilt-b32-finetuned-vqa" (lighter), or larger BLIP models.

# --- Device ---
device = 0 if torch.cuda.is_available() else -1

# --- Load pipeline ---
vqa = pipeline("visual-question-answering", model=MODEL_NAME, device=device)

# --- Load image ---
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image not found at {IMAGE_PATH}. Put your image there or update IMAGE_PATH.")

image = Image.open(IMAGE_PATH).convert("RGB")

# --- Ask questions ---
questions = [
    "Why is the person holding an umbrella?",
    "What is the person sitting on?",
    "Is it raining in the picture?",
    "How many suitcases are there?",
    "Where is the person waiting?"
]

for q in questions:
    result = vqa(image=image, question=q)
    # result is typically a dict {'score': float, 'answer': str} or list (depending on model/pipeline)
    # Normalize different return formats:
    if isinstance(result, list) and len(result) > 0:
        answer = result[0].get("answer", "")
        score = result[0].get("score", 0.0)
    elif isinstance(result, dict):
        answer = result.get("answer", "")
        score = result.get("score", 0.0)
    else:
        answer = str(result)
        score = 0.0

    print(f"Q: {q}")
    print(f"A: {answer}  (score: {score:.3f})")
    print("-" * 40)
