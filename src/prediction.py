import os
import cv2
from ultralytics import YOLO
from collections import Counter
import torch
import torchvision
from torchvision import transforms

# ===========================
# BASE PATH (ROOT)
# ===========================
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except:
    BASE_DIR = os.getcwd()

# ===========================
# USER INPUT
# ===========================
folder = ''
choice = int(input("1.) Aerial coconut tree detection\n2.) Coconut Counting\n3.) Maturity Variety Classification\n4.) Disease Classification\nEnter your choice: "))

if choice == 1:
    folder = 'aerial tree detection'
elif choice == 2:
    folder = 'coconut counting'
elif choice == 3:
    folder = 'maturity variety classification'
elif choice == 4:
    folder = 'disease classification'

# ===========================
# PATHS
# ===========================
MODEL_DIR = os.path.join(BASE_DIR, "models", folder)
IMAGE_DIR = os.path.join(BASE_DIR, "test", folder)
OUTPUT_DIR = os.path.join(BASE_DIR, "output", folder)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================
# TEXT WITH BACKGROUND
# ===========================
def draw_text_bg(img, text, x, y,
                 font_scale=0.6,
                 thickness=2,
                 text_color=(255,255,255),
                 bg_color=(40,40,40)):

    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)

    cv2.rectangle(img,
                  (x, y - h - 6),
                  (x + w + 6, y + 4),
                  bg_color,
                  -1)

    cv2.putText(img, text,
                (x, y),
                font,
                font_scale,
                text_color,
                thickness)

    return img

# ===========================
# LOAD MODELS
# ===========================
models = {}
model_types = {}
model_names = []

print("📦 Loading models...\n")

for file in os.listdir(MODEL_DIR):

    path = os.path.join(MODEL_DIR, file)
    name = file.split(".")[0]

    # YOLO
    if file.endswith(".pt"):
        print(f"✔ YOLO Loaded: {name}")
        models[name] = YOLO(path)
        model_types[name] = "yolo"
        model_names.append(name)

    # PyTorch (.pth)
    elif file.endswith(".pth"):
        print(f"✔ PyTorch Loaded: {name}")

        checkpoint = torch.load(path, map_location="cpu")

        # Handle both cases
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

        # Get number of classes dynamically
        num_classes = checkpoint["roi_heads.box_predictor.cls_score.weight"].shape[0]
        print(f"   → Detected classes: {num_classes}")

        # Build correct model
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Load weights
        model.load_state_dict(checkpoint)
        model.eval()

        models[name] = model
        model_types[name] = "pytorch"
        model_names.append(name)

print(f"\nTotal models loaded: {len(models)}\n")

# ===========================
# PYTORCH INFERENCE
# ===========================
def run_pytorch_model(model, img):

    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img)

    with torch.no_grad():
        outputs = model([img_tensor])[0]

    annotated = img.copy()
    class_list = []

    if "boxes" in outputs:
        for i in range(len(outputs["boxes"])):
            score = outputs["scores"][i].item()

            if score < 0.25:
                continue

            x1, y1, x2, y2 = map(int, outputs["boxes"][i].tolist())
            label = str(outputs["labels"][i].item())

            class_list.append(label)

            cv2.rectangle(annotated, (x1,y1),(x2,y2),(255,0,0), 2)
            cv2.putText(annotated, f"{label} {score:.2f}",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255,255,255), 2)

    return annotated, class_list

# ===========================
# RUN MODEL (UNIFIED)
# ===========================
def run_model(model, img, model_name, model_type):

    if model_type == "yolo":

        results = model.predict(img, conf=0.25, verbose=False)
        res = results[0]

        annotated = img.copy()
        class_list = []

        if res.boxes is not None:
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = res.names[cls]

                class_list.append(label)

                cv2.rectangle(annotated, (x1,y1),(x2,y2),(0,255,0), 2)
                cv2.putText(annotated, f"{label} {conf:.2f}",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255,255,255), 2)

        return annotated, class_list

    elif model_type == "pytorch":
        return run_pytorch_model(model, img)

# ===========================
# PROCESS IMAGES
# ===========================
print("🖼 Processing images...\n")

for img_name in os.listdir(IMAGE_DIR):

    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"❌ Skipping invalid image: {img_name}")
        continue

    outputs = []

    for name in model_names:
        annotated, class_list = run_model(
            models[name],
            img,
            name,
            model_types[name]
        )

        counts = Counter(class_list)

        y_offset = 30

        annotated = draw_text_bg(annotated, name, 10, y_offset, font_scale=0.8, bg_color=(60,60,60))
        y_offset += 30

        if counts:
            for cls_name, cnt in counts.items():
                annotated = draw_text_bg(annotated, f"{cls_name}: {cnt}", 10, y_offset, bg_color=(0,120,0))
                y_offset += 25

            total = sum(counts.values())
            annotated = draw_text_bg(annotated, f"Total: {total}", 10, y_offset, bg_color=(120,80,0))
        else:
            annotated = draw_text_bg(annotated, "No detections", 10, y_offset, bg_color=(0,0,150))

        outputs.append(annotated)

    combined = outputs[0]
    for out in outputs[1:]:
        combined = cv2.hconcat([combined, out])

    save_path = os.path.join(OUTPUT_DIR, img_name)
    cv2.imwrite(save_path, combined)

    print(f"✔ Saved: {img_name}")

print("\n✅ DONE! All outputs generated successfully.")