import torch
import torchvision
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torchvision.ops import roi_align
import torchvision.transforms as T
from PIL import Image
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# load pretrained Faster R-CNN
model = fasterrcnn_resnet50_fpn(weights="DEFAULT").to(DEVICE)
model.eval()

transform = T.Compose([
    T.ToTensor(),
])

def extract_features(img_path, save_path, top_k=36):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).to(DEVICE)

    # run detector
    with torch.no_grad():
        out = model([img_tensor])[0]

    boxes = out["boxes"]

    # If fewer than required, pad with dummy boxes
    if len(boxes) < top_k:
        pad_count = top_k - len(boxes)
        pad_boxes = torch.zeros((pad_count, 4), device=boxes.device)
        boxes = torch.cat([boxes, pad_boxes], dim=0)

    # Trim to top_k (36)
    boxes = boxes[:top_k]

    # get backbone features
    with torch.no_grad():
        backbone_out = model.backbone(img_tensor.unsqueeze(0))
        feature_map = list(backbone_out.values())[0]

    # ROI align → region embeddings
    roi = roi_align(
        input=feature_map,
        boxes=torch.cat([torch.zeros(len(boxes), 1, device=DEVICE), boxes], dim=1),
        output_size=(7, 7),
        spatial_scale=1.0 / 32,
    )

    # flatten from (36, 1024)
    features = roi.mean(dim=[2, 3])

    torch.save(features.cpu(), save_path)
    print(f"Saved features → {save_path}")


if __name__ == "__main__":
    os.makedirs("featuresVLT5", exist_ok=True)

    # extract features for images in /images folder
    for file in os.listdir("imagesVLT5"):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            img_id = file.split(".")[0]
            extract_features(f"imagesVLT5/{file}", f"featuresVLT5/{img_id}.pt")