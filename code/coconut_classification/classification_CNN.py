# File: train_coconut_cnn.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from create_dataset import CoconutDataset
import os

# ----------------- CONFIG -----------------
IMAGE_SIZE = 128
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ["dry", "tender", "mature"]

# ----------------- TRANSFORMS -----------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ----------------- DATASETS -----------------
train_dataset = CoconutDataset(
    images_dir="D:/CoconutPalmDetection/datasets/coconut_classification/train/images",
    labels_dir="D:/CoconutPalmDetection/datasets/coconut_classification/train/labels_converted",
    class_names=class_names,
    transform=transform
)

valid_dataset = CoconutDataset(
    images_dir="D:/CoconutPalmDetection/datasets/coconut_classification/valid/images",
    labels_dir="D:/CoconutPalmDetection/datasets/coconut_classification/valid/labels_converted",
    class_names=class_names,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ----------------- MODEL -----------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (IMAGE_SIZE // 8) * (IMAGE_SIZE // 8), 256),  # Adjust flatten size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN(num_classes=len(class_names)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ----------------- TRAINING -----------------
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.2f}%")

# ----------------- SAVE MODEL -----------------
os.makedirs("D:/CoconutPalmDetection/code/coconut_classification/checkpoints", exist_ok=True)
torch.save(model.state_dict(), "D:/CoconutPalmDetection/code/coconut_classification/checkpoints/coconut_cnn.pth")
print("Training complete and model saved!")
