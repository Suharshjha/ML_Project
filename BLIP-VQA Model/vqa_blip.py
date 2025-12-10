# vqa_blip.py
from PIL import Image
from transformers import pipeline
import torch
import os

# --- Config ---
IMAGE_PATH = "images/img1.jpg"  # Update if your image path is different
MODEL_NAME = "Salesforce/blip-vqa-base"

# Device setup
device = 0 if torch.cuda.is_available() else -1

# Load pretrained BLIP-VQA
vqa = pipeline("visual-question-answering", model=MODEL_NAME, device=device)

# Load image
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image not found at {IMAGE_PATH}. Place your image there!")

image = Image.open(IMAGE_PATH).convert("RGB")

# Questions you ask
questions = [
    "Why is the person holding an umbrella?",
    "What is the person sitting on?",
    "Is it raining in the picture?",
    "How many suitcases are there?",
    "Where is the person waiting?",
]

# Ground truth answers (YOU fill correct answers here)
ground_truth = {
    "Why is the person holding an umbrella?": "because it is raining",
    "What is the person sitting on?": "suitcases",
    "Is it raining in the picture?": "yes",
    "How many suitcases are there?": "two",
    "Where is the person waiting?": "train station",
}

correct = 0

print("\n----------------- VQA Results -----------------\n")

for q in questions:
    result = vqa(image=image, question=q)

    if isinstance(result, list):
        answer = result[0].get("answer", "").lower()
    else:
        answer = result.get("answer", "").lower()

    gt = ground_truth[q].lower()
    is_correct = (gt in answer) or (answer in gt)

    if is_correct:
        correct += 1

    print(f"Q: {q}")
    print(f"Model Answer: {answer}")
    print(f"Ground Truth: {gt}")
    print(f"Correct? {'✔ YES' if is_correct else '❌ NO'}")
    print("-" * 40)

accuracy = correct / len(questions) * 100
print(f"\nModel Accuracy on this dataset: {accuracy:.2f}%")
print("\n----------------------------------------------\n")
