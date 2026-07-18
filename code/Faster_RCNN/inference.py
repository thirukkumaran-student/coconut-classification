import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw, ImageFont

# ---- CONFIG ----
CLASS_NAMES = ["__background__", "coconut"]
# MODEL_PATH = "D:/CoconutPalmDetection/code/fasterrcnn_coconut.pth"
IMAGE_PATH = "C:/Users/Admin/Pictures/testing.jpg"
MODEL_PATH = "D:/CoconutPalmDetection/code/Faster_RCNN/checkpoints/fasterrcnn_coconut_epoch10.pth"

SCORE_THRESHOLD = 0.5

# ---- DEVICE ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- LOAD MODEL ----
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ---- LOAD IMAGE ----
img = Image.open(IMAGE_PATH).convert("RGB")
transform = torchvision.transforms.ToTensor()
img_tensor = transform(img).to(device)

# ---- INFERENCE ----
with torch.no_grad():
    outputs = model([img_tensor])  # Faster R-CNN expects a list of tensors

output = outputs[0]

# ---- DRAW RESULTS ----
draw = ImageDraw.Draw(img)
# Optional: use a font for nicer labels
try:
    font = ImageFont.truetype("arial.ttf", 16)
except:
    font = ImageFont.load_default()

count = 0
for box, score, label in zip(output["boxes"].cpu(), output["scores"].cpu(), output["labels"].cpu()):
    if score > SCORE_THRESHOLD:
        count += 1
        x0, y0, x1, y1 = box
        draw.rectangle([x0.item(), y0.item(), x1.item(), y1.item()], outline="red", width=3)
        draw.text((x0.item(), y0.item()), f"{CLASS_NAMES[label]} {score:.2f}", fill="yellow", font=font)

print(f"Detected {count} coconuts!")
img.show()
