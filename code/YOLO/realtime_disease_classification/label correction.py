import os
import shutil

# -------------------------------
# CONFIG
# -------------------------------
input_labels_dir = r"D:\CoconutPalmDetection\datasets\realtime_coconut_disease\train\labels"
output_labels_dir = r"D:\CoconutPalmDetection\datasets\realtime_coconut_disease\train\labels_new"

# Mapping rule
class_mapping = {
    0: 1,
    1: 3
}

os.makedirs(output_labels_dir, exist_ok=True)

# -------------------------------
# PROCESS EACH LABEL FILE
# -------------------------------
for filename in os.listdir(input_labels_dir):

    if not filename.endswith(".txt"):
        continue

    input_path = os.path.join(input_labels_dir, filename)
    output_path = os.path.join(output_labels_dir, filename)

    with open(input_path, "r") as f:
        lines = f.readlines()

    modified_lines = []

    for line in lines:
        parts = line.strip().split()

        if len(parts) == 0:
            continue

        class_id = int(parts[0])

        # Replace class if in mapping
        if class_id in class_mapping:
            new_class_id = class_mapping[class_id]
        else:
            new_class_id = class_id  # keep unchanged

        # Replace class ID
        parts[0] = str(new_class_id)

        modified_line = " ".join(parts)
        modified_lines.append(modified_line)

    # Write modified file
    with open(output_path, "w") as f:
        for line in modified_lines:
            f.write(line + "\n")

print("All labels processed successfully.")