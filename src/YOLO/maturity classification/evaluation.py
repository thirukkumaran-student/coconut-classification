from ultralytics import YOLO

model = YOLO(r'D:\CoconutPalmDetection\code\YOLO\realtime_classification\runs\detect\coconut_4class_cls\weights\best.pt')
def main():
    metrics = model.val()

    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)
    print("Precision:", metrics.box.mp)
    print("Recall:", metrics.box.mr)

if __name__ == '__main__':
    main()