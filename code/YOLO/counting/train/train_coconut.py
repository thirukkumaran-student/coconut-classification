from ultralytics import YOLO

model = YOLO('yolov8n.pt')
def main():
    model.train(
        data = 'D:/CoconutPalmDetection/datasets/coconut_dataset/data.yaml',
        epochs = 40,
        imgsz = 640,
        batch = 16,
        name = "coconut_counting"
    )
if __name__ == '__main__':
    main()