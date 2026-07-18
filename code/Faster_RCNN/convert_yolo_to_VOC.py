import os
import xml.etree.ElementTree as ET
from PIL import Image

# ---- CONFIG ----
images_folder = "D:/CoconutPalmDetection/datasets/coconut_dataset-RCNN/train/images"
labels_folder = "D:/CoconutPalmDetection/datasets/coconut_dataset-RCNN/train/labels"
output_folder = "D:/CoconutPalmDetection/datasets/coconut_dataset-RCNN/voc_annotations"

os.makedirs(output_folder, exist_ok=True)

# ---- CLASS NAMES (update if you have more classes) ----
class_names = {
    "0": "coconut"
}

# ---- FUNCTION TO CREATE VOC XML ----
def create_voc_xml(image_path, annotations, save_path):
    img = Image.open(image_path)
    width, height = img.size
    filename = os.path.basename(image_path)

    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = os.path.basename(images_folder)
    ET.SubElement(annotation, "filename").text = filename

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(len(img.getbands()))

    ET.SubElement(annotation, "segmented").text = "0"

    for ann in annotations:
        class_name, xmin, ymin, xmax, ymax = ann
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = class_name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(xmin)
        ET.SubElement(bbox, "ymin").text = str(ymin)
        ET.SubElement(bbox, "xmax").text = str(xmax)
        ET.SubElement(bbox, "ymax").text = str(ymax)

    tree = ET.ElementTree(annotation)
    tree.write(save_path)

# ---- CONVERT ALL LABELS ----
for image_file in os.listdir(images_folder):
    if image_file.endswith(".jpg") or image_file.endswith(".png"):
        image_path = os.path.join(images_folder, image_file)
        label_file = os.path.join(labels_folder, image_file.replace(".jpg", ".txt").replace(".png", ".txt"))

        annotations = []
        if os.path.exists(label_file):
            img = Image.open(image_path)
            width, height = img.size

            with open(label_file, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    class_id = parts[0]
                    x_center, y_center, w, h = map(float, parts[1:])

                    xmin = int((x_center - w / 2) * width)
                    ymin = int((y_center - h / 2) * height)
                    xmax = int((x_center + w / 2) * width)
                    ymax = int((y_center + h / 2) * height)

                    class_name = class_names.get(class_id, "unknown")
                    annotations.append((class_name, xmin, ymin, xmax, ymax))

        save_path = os.path.join(output_folder, image_file.replace(".jpg", ".xml").replace(".png", ".xml"))
        create_voc_xml(image_path, annotations, save_path)

print(f"Pascal VOC annotations created in folder: {output_folder}")
