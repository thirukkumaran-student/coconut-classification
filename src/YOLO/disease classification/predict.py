from ultralytics import YOLO

model = YOLO(r'D:\CoconutPalmDetection\code\YOLO\realtime_disease_classification\yolo26\yolo26s_exp1\weights\best.pt')

def main():
    tests = [r"C:\Users\Admin\Pictures\testing\stem_bleeding.png",
             r"C:\Users\Admin\Pictures\testing\leaf_rot.jpg",
             r"C:\Users\Admin\Pictures\testing\gray_leaf_spot.jpg",
             r"C:\Users\Admin\Pictures\testing\budrot.jpg",
             r"D:\CoconutPalmDetection\datasets\realtime_coconut_disease\test\images\BudRootDropping292_jpg.rf.5f0dcb077b0471536a7ccb29af0814cb.jpg"]
    for test in tests:
        # Predict on a single image
        results = model.predict(test, save=True, conf=0.5)
        results[0].show()

if __name__ == '__main__':
    main()