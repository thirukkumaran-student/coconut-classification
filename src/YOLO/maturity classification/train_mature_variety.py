from ultralytics import YOLO

def main():

    model = YOLO('yolo11n.pt')

    model.train(
        data=r'D:\CoconutPalmDetection\datasets\realtime_coconut\data.yaml',

        imgsz=224,
        batch=16,
        epochs=40,
        patience=15,

        optimizer='AdamW',
        lr0=0.0005,
        weight_decay=0.0005,

        dropout=0.4,
        label_smoothing=0.1,

        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,
        degrees=10,
        scale=0.3,
        fliplr=0.5,

        name='coconut_4class_cls'
    )

if __name__ == "__main__":
    main()