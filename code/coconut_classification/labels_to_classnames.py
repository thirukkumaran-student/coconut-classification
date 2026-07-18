import os

# Paths to your dataset
labels_dir = "D:/CoconutPalmDetection/datasets/coconut_classification/valid/labels"
output_dir = "D:/CoconutPalmDetection/datasets/coconut_classification/valid/labels_converted"

os.makedirs(output_dir, exist_ok=True)

# Mapping of class IDs to names
class_names = ["dry", "tender", "mature"]

for label_file in os.listdir(labels_dir):
    if label_file.endswith(".txt"):
        with open(os.path.join(labels_dir, label_file), "r") as f:
            line = f.readline().strip()
            if line:  # skip empty files
                class_id = int(line.split()[0])
                class_name = class_names[class_id]

                # Write to new file
                with open(os.path.join(output_dir, label_file), "w") as out_f:
                    out_f.write(class_name)

print("Conversion complete! Check the 'labels_converted' folder.")
