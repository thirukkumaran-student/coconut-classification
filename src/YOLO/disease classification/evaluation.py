from ultralytics import YOLO
import os

# ---------------------------
# CONFIG
# ---------------------------
model_path = r"D:\CoconutPalmDetection\code\YOLO\realtime_disease_classification\yolo26\yolo26s_exp1\weights\best.pt"
val_images_path = r"D:\CoconutPalmDetection\datasets\realtime_coconut_disease\val\images"
test_image_path = r"C:\Users\Admin\Pictures\testing\budrot.jpg"  # change this

# ---------------------------
# LOAD MODEL
# ---------------------------
def main():
    print("Loading model...")
    model = YOLO(model_path)

    print("\nModel loaded successfully.")
    print("Class names:", model.names)

    # ---------------------------
    # VALIDATION CHECK
    # ---------------------------
    print("\nRunning validation...")
    metrics = model.val()
    print("Validation Results:")
    print(metrics)

    # ---------------------------
    # TEST ON VALIDATION FOLDER
    # ---------------------------
    print("\nTesting on validation folder...")
    results_folder = model.predict(
        source=val_images_path,
        conf=0.25,
        save=False
    )

    total_detections = sum(len(r.boxes) for r in results_folder)
    print("Total detections in validation folder:", total_detections)

    # ---------------------------
    # TEST ON SINGLE IMAGE
    # ---------------------------
    if os.path.exists(test_image_path):
        print("\nTesting on single image...")
        result = model.predict(
            source=test_image_path,
            conf=0.01,  # very low to see weak detections
            show=True
        )

        print("Detections in single image:", len(result[0].boxes))
    else:
        print("\nSingle test image path not found.")

if __name__ == "__main__":
    main()