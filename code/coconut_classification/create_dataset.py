# File: create_dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset

class CoconutDataset(Dataset):
    def __init__(self, images_dir, labels_dir, class_names, transform=None):
        """
        images_dir: path to images
        labels_dir: path to label files (each file contains one class name)
        class_names: list of class names in order, e.g., ['dry', 'tender', 'mature']
        transform: torchvision transforms to apply on images
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.class_names = class_names
        self.transform = transform

        self.images = sorted(os.listdir(images_dir))
        self.labels = sorted(os.listdir(labels_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        label_path = os.path.join(self.labels_dir, self.labels[idx])

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Load label (as index)
        with open(label_path, "r") as f:
            label_name = f.readline().strip()
        label = self.class_names.index(label_name)

        if self.transform:
            image = self.transform(image)

        return image, label
