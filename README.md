# 📘 Visual Question Answering (VQA) using BLIP-VQA
Overcoming VL-T5 Implementation Challenges in a Lab Environment


# 📌 Overview
- This project implements Visual Question Answering (VQA) using BLIP-VQA, a fully pretrained Vision-Language model by Salesforce.
- The system answers text questions about an uploaded image — image in → answer out.
- Our initial goal was to reproduce the research model VL-T5, but due to practical limitations, we switched to BLIP-VQA which works smoothly on a standard laptop while still giving accurate results.

# 🧠 Why VL-T5 Failed
- ❌ No pretrained multimodal weights available
- ❌ Needed exactly 36 Faster R-CNN region features per image
- ❌ Detectron2 installation fails on Windows
- ❌ Region count mismatch → tensor shape errors
- ❌ Without pretrained weights, output became random
- ❌ Required huge datasets (COCO, Visual Genome, VQA)
- ❌ Required A100-level GPUs and weeks of training

🔹⭐ Conclusion: VL-T5 is not feasible in a typical academic lab environment.

# 💡 Why BLIP-VQA Was the Perfect Solution
- ✔ No region feature extraction required
- ✔ No Detectron2 installation
- ✔ Works on a normal laptop CPU
- ✔ Produces correct, meaningful answers
- ✔ Very easy to implement (end-to-end)

BLIP Pipeline:
```bash
Image → Vision Transformer → Text Answer
```

# 🚀 Features
- ✔ Upload an image and ask any question about it
- ✔ Generates meaningful answers like: “Because it is raining.”,“Sitting on a suitcase.”
- ✔ Fast inference
- ✔ No GPU required
- ✔ Zero complex setup
- ✔ Fully reproducible for ML lab projects

# 📂 Project Structure
```bash
├── app.py                     # Main interface (Streamlit/Flask/Gradio)
├── models/
│   └── blip_vqa.py            # BLIP-VQA model loading & inference
├── utils/
│   └── image_utils.py         # Preprocessing helpers
├── README.md
└── requirements.txt
```

# 🔧 Installation
```bash
git clone <your-repo-url>
cd <project-folder>

pip install -r requirements.txt
```

# 🏁 Final Conclusion
- VL-T5 is theoretically strong but practically impossible to reproduce in a normal lab due to missing pretrained weights, strict region requirements, and heavy GPU needs.
- By switching to BLIP-VQA, we achieved a:
    ✔ Fully functional
    ✔ Lab-friendly
    ✔ High-accuracy
    ✔ End-to-end VQA system
- that runs on a standard laptop with minimal dependencies.



