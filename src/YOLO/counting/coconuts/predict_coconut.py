from ultralytics import YOLO
model = YOLO(r"D:\CoconutPalmDetection\code\YOLO\counting\runs\detect\coconut_counting\weights\best.pt")

tests = [r"D:\CoconutPalmDetection\datasets\realtime_coconut\raw_data\IMG_20260125_133130_05.jpg", r"D:\CoconutPalmDetection\datasets\realtime_coconut\raw_data\IMG_20260125_133215_01.jpg", r"D:\CoconutPalmDetection\datasets\realtime_coconut\raw_data\IMG_20260125_133250_01.jpg", r"D:\CoconutPalmDetection\datasets\realtime_coconut\raw_data\IMG_20260125_133514_01.jpg", r"D:\CoconutPalmDetection\datasets\coconut_tree_dataset\test\images\DJI_0886_frame_16_jpg.rf.12607c7626131b77b4908f65eb2abeb8.jpg"]
for test in tests:
# Predict on a single image
    results = model.predict(test, save=True, conf=0.5)
    results[0].show()