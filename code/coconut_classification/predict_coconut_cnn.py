import torch
from classification_CNN import SimpleCNN  # your CNN class
from torchvision import transforms
from PIL import Image

# Same class names
class_names = ["dry", "tender", "mature"]

# Initialize model and load weights
model = SimpleCNN(num_classes=len(class_names))
model.load_state_dict(torch.load("D:/CoconutPalmDetection/code/coconut_classification/checkpoints/coconut_cnn.pth"))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Transform for input image
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# Load image
img_path = "D:/CoconutPalmDetection/datasets/coconut_classification/test/images/094_jpg.rf.a2b00b86e068c19634c00bef0c3f6b54.jpg"
img = Image.open(img_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

# Prediction
with torch.no_grad():
    output = model(img_tensor)
    pred_class = class_names[torch.argmax(output, dim=1).item()]

print(f"The coconut is classified as: {pred_class}")
