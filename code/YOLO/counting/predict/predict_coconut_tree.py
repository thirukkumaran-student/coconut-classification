from ultralytics import YOLO

# Load your trained model
model = YOLO(r"D:\CoconutPalmDetection\code\YOLO\counting\runs\detect\train52\weights\best.pt")
tests = [r"D:\CoconutPalmDetection\datasets\coconut_tree_dataset\test\images\DJI_0895_frame_229_jpg.rf.ebf3495ee623378b01912d2d4afd4ec3.jpg", r"D:\CoconutPalmDetection\datasets\coconut_tree_dataset\test\images\DJI_0923_frame_545_jpg.rf.ed7783f65c02946f550acd991dc3f269.jpg", r"D:\CoconutPalmDetection\datasets\coconut_tree_dataset\test\images\DJI_0897_frame_275_jpg.rf.47bb3ec032d770a0e942788d6be1416e.jpg", r"D:\CoconutPalmDetection\datasets\coconut_tree_dataset\test\images\DJI_0895_frame_216_jpg.rf.d5029410c4b2788b2ecfead2edc2d06e.jpg", r"D:\CoconutPalmDetection\datasets\coconut_tree_dataset\test\images\DJI_0886_frame_16_jpg.rf.12607c7626131b77b4908f65eb2abeb8.jpg"]
for test in tests:
# Predict on a single image
    results = model.predict(test, save=True, conf=0.5)
    results[0].show()
print("Prediction done! Check results inside runs/detect/predict/")
