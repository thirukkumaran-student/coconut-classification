from ultralytics import YOLO

def main():

    model = YOLO(r'D:\CoconutPalmDetection\code\YOLO\realtime_disease_classification\runs\detect\coconut_disease_detection2\weights\best.pt')   # Nano model (best for 500 images)

    model.train(
        data=r'D:\CoconutPalmDetection\datasets\realtime_coconut_disease\data.yaml',

        imgsz=640,
        batch=16,
        epochs=120,
        patience=25,

        optimizer='AdamW',
        lr0=0.001,  # Slightly higher since dataset is bigger
        lrf=0.01,
        weight_decay=0.0005,

        cos_lr=True,  # Cosine learning rate schedule

        # Augmentation (balanced, not aggressive)
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.4,
        fliplr=0.5,
        flipud=0.3,  # Useful for drone data
        mosaic=1.0,
        mixup=0.05,

        name='coconut_disease_finetune',
        workers=8
    )

if __name__ == "__main__":
    main()