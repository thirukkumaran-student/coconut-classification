import os
import pandas as pd

# Change these paths to your D:/ folder
csv_file = r"D:/CoconutPalmDetection/palm_dataset/train_labels.csv"
images_dir = r"D:/CoconutPalmDetection/palm_dataset/train/"
labels_dir = r"D:/CoconutPalmDetection/palm_dataset/train/labels/"

os.makedirs(labels_dir, exist_ok=True)

# Load CSV
df = pd.read_csv(csv_file)

for _, row in df.iterrows():
    filename = row["filename"]
    width = row["width"]
    height = row["height"]
    xmin = row["xmin"]
    ymin = row["ymin"]
    xmax = row["xmax"]
    ymax = row["ymax"]

    # Convert to YOLO format
    x_center = ((xmin + xmax) / 2) / width
    y_center = ((ymin + ymax) / 2) / height
    bbox_width = (xmax - xmin) / width
    bbox_height = (ymax - ymin) / height

    # Class id → 0 (since only palm trees here)
    class_id = 0

    # Save to .txt file
    label_path = os.path.join(labels_dir, filename.replace(".jpg", ".txt"))
    with open(label_path, "a") as f:
        f.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")

print("Palm dataset converted to YOLO format successfully!")
