from ultralytics import YOLO

def main():

    model = YOLO('yolo26s.pt')   # Nano model (best for 500 images)

    model.train(
        data=r'D:\CoconutPalmDetection\datasets\realtime_coconut_disease\data.yaml',
        project=r'D:\CoconutPalmDetection\code\YOLO\realtime_disease_classification\yolo26',  # Your desired folder
        name="yolo26s_exp1",  # Experiment name
        exist_ok=True,

        epochs=100,
        imgsz=512,
        batch=8,
        workers=8,
        device=0,
        cache=False,
        amp=False,

        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        cos_lr=True,

        # Early stopping
        patience=20,

        # Augmentation
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,

        degrees=5,
        translate=0.08,
        scale=0.4,
        shear=2.0,
        perspective=0.0005,

        fliplr=0.5,
        flipud=0.0,

        mosaic=0.8,
        close_mosaic=20,

        mixup=0.15,
        copy_paste=0.1,
        erasing=0.2,

        # Regularization
        label_smoothing=0.05,

        # Save
        save=True,
        plots=True,
        verbose=True
    )

if __name__ == "__main__":
    main()