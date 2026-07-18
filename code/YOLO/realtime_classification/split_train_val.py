
import shutil
import random
from pathlib import Path

root = Path(r'D:\CoconutPalmDetection\datasets\realtime_coconut')

images = list((root / "images").glob("*.jpg"))
labels = root / "labels"

train_img_dir = root / "images" / "train"
val_img_dir = root / "images" / "val"
train_lbl_dir = root / "labels" / "train"
val_lbl_dir = root / "labels" / "val"

for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
    d.mkdir(parents=True, exist_ok=True)

random.shuffle(images)

split_ratio = 0.1
val_count = int(len(images) * split_ratio)
val_images = images[:val_count]
train_images = images[val_count:]


def move_files(image_list, img_dest, lbl_dest):
    for img in image_list:
        lbl_name = img.stem + ".txt"
        lbl_path = labels / lbl_name

        shutil.copy2(img, img_dest / img.name)
        if lbl_path.exists():
            shutil.copy2(lbl_path, lbl_dest / lbl_name)


move_files(train_images, train_img_dir, train_lbl_dir)
move_files(val_images, val_img_dir, val_lbl_dir)

print("Done creating train/val split.")
