import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
from torchvision import transforms

class VOCDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, class_names, transform=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.class_names = class_names
        self.transform = transform if transform else transforms.ToTensor()

        # Collect all image filenames
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png"))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        ann_path = os.path.join(
            self.annotations_dir,
            img_name.replace(".jpg", ".xml").replace(".png", ".xml")
        )

        # Load image
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        # Load annotations
        boxes = []
        labels = []

        tree = ET.parse(ann_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            name = obj.find("name").text.strip()
            label = self.class_names.index(name) + 1  # +1 because background = 0

            xml_box = obj.find("bndbox")
            xmin = int(xml_box.find("xmin").text)
            ymin = int(xml_box.find("ymin").text)
            xmax = int(xml_box.find("xmax").text)
            ymax = int(xml_box.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        img = self.transform(img)  # Convert to tensor

        return img, target
