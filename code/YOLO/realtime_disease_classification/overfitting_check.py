import pandas as pd
import matplotlib.pyplot as plt

# Path to your training results
results_path = r"D:\CoconutPalmDetection\code\YOLO\realtime_disease_classification\runs\detect\coconut_disease_finetune\results.csv"

df = pd.read_csv(results_path)

# Plot losses
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Train vs Validation Loss")

# Plot mAP
plt.subplot(1,2,2)
plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50')
plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95')
plt.xlabel("Epoch")
plt.ylabel("mAP")
plt.legend()
plt.title("Validation mAP")

plt.tight_layout()
plt.show()