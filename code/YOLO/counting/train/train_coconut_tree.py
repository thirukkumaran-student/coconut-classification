from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")  # or yolov8s.pt if you want a stronger model

    model.train(
        data=r"D:\CoconutPalmDetection\datasets\coconut_tree_dataset\data.yaml",
        epochs=40,
        imgsz=640,
        batch=16,
        workers=0,   # <-- safer for Windows
        name="coconut_tree"
    )

if __name__ == "__main__":
    main()
