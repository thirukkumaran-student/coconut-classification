import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn  # type: ignore
from torch.utils.data import DataLoader
from voc_dataset import VOCDataset
from torchvision import transforms
import os

# ---- CONFIG ----
CLASS_NAMES = ["coconut"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 10
BATCH_SIZE = 2
LR = 0.005

# ---- CREATE CHECKPOINTS FOLDER ----
os.makedirs("D:/CoconutPalmDetection/code/Faster_RCNN/checkpoints", exist_ok=True)

# ---- LOAD DATA ----
images_dir = "D:/CoconutPalmDetection/datasets/coconut_dataset-RCNN/train/images"
annotations_dir = "D:/CoconutPalmDetection/datasets/coconut_dataset-RCNN/voc_annotations"

train_dataset = VOCDataset(
    images_dir=images_dir,
    annotations_dir=annotations_dir,
    class_names=CLASS_NAMES,
    transform=transforms.ToTensor()
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda x: tuple(zip(*x))
)

# ---- MODEL SETUP ----
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASS_NAMES) + 1)  # +1 for background
model.to(DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.0005)

# ---- TRAINING LOOP ----
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0

    for imgs, targets in train_loader:
        imgs = [i.to(DEVICE) for i in imgs]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Average Loss: {avg_loss:.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), f"code/Faster_RCNN/checkpoints/fasterrcnn_coconut_epoch{epoch+1}.pth")

print("Training complete! Final model saved.")
