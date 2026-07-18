# ---- full_pipeline.py ----

import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import os

# ---- Step 1: Device ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Step 2: Load Faster-RCNN model (Coconut Detector) ----
CLASS_NAMES_DETECT = ["__background__", "coconut"]
fasterrcnn_model = fasterrcnn_resnet50_fpn(weights=None)
in_features = fasterrcnn_model.roi_heads.box_predictor.cls_score.in_features
fasterrcnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASS_NAMES_DETECT))
fasterrcnn_model.load_state_dict(torch.load(
    "D:/CoconutPalmDetection/code/Faster_RCNN/checkpoints/fasterrcnn_coconut_epoch10.pth",
    map_location=DEVICE
))
fasterrcnn_model.eval()
fasterrcnn_model.to(DEVICE)

# ---- Step 3: Load CNN classifier ----
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*16*16, 256),  # assuming 128x128 input
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,num_classes)
        )

    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x

CLASS_NAMES_CNN = ["dry", "tender", "mature"]
cnn_model = SimpleCNN(num_classes=len(CLASS_NAMES_CNN))
cnn_model.load_state_dict(torch.load(
    "D:/CoconutPalmDetection/code/coconut_classification/checkpoints/coconut_cnn.pth",
    map_location=DEVICE
))
cnn_model.eval()
cnn_model.to(DEVICE)

# ---- Step 4: Transform for CNN classifier ----
cnn_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# ---- Step 5: Run detection + classification ----
def detect_and_classify(image_path, detection_thresh=0.5, classification_thresh=0.5):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Transform for Faster R-CNN
    transform_detect = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform_detect(img).unsqueeze(0).to(DEVICE)

    # ---- Step 5a: Detect coconuts ----
    with torch.no_grad():
        outputs = fasterrcnn_model(img_tensor)

    output = outputs[0]
    boxes = output["boxes"]
    scores = output["scores"]

    # ---- Step 5b: Loop through detected coconuts ----
    for box, score in zip(boxes, scores):
        if score > detection_thresh:
            xmin, ymin, xmax, ymax = map(int, box.tolist())

            # Crop coconut for classification
            coconut_crop = img.crop((xmin, ymin, xmax, ymax))
            coconut_tensor = cnn_transform(coconut_crop).unsqueeze(0).to(DEVICE)

            # Classify
            with torch.no_grad():
                class_logits = cnn_model(coconut_tensor)
                class_idx = class_logits.argmax(dim=1).item()
                class_name = CLASS_NAMES_CNN[class_idx]

            # Draw bounding box + label
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
            draw.text((xmin, ymin-15), f"{class_name}", fill="yellow")

    img.show()

# ---- Step 6: Run ----
image_path = "D:/CoconutPalmDetection/datasets/coconut_classification/test/images/094_jpg.rf.a2b00b86e068c19634c00bef0c3f6b54.jpg"
detect_and_classify(image_path)
