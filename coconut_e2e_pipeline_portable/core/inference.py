import time
import cv2
import numpy as np
import torch
from PIL import Image

DISEASE_NAMES = [
    "BudRotDropping", "BudRot", "StemBleeding", "Gray Leaf Spot", "Leaf Rot"
]
MATURITY_NAMES = [
    "Tender Green", "Mature Green", "Tender King", "Mature King"
]

def _counts(names):
    out = {}
    for n in names:
        out[n] = out.get(n, 0) + 1
    return out

def infer_yolo(bundle, frame, task, confidence, iou, imgsz, tracking=False):
    model = bundle["model"]
    device = bundle["device"]

    t0 = time.perf_counter()
    kwargs = dict(
        source=frame,
        conf=confidence,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )

    if tracking:
        results = model.track(persist=True, tracker="bytetrack.yaml", **kwargs)
    else:
        results = model.predict(**kwargs)

    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()

    latency_ms = (time.perf_counter() - t0) * 1000.0
    r = results[0]

    annotated = r.plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    names = r.names
    class_names = []
    if r.boxes is not None and r.boxes.cls is not None:
        for cls in r.boxes.cls.detach().cpu().numpy().astype(int):
            class_names.append(names[int(cls)])

    return {
        "frame": annotated,
        "count": len(class_names),
        "class_counts": _counts(class_names),
        "latency_ms": latency_ms,
        "fps": 1000.0 / latency_ms if latency_ms > 0 else 0.0,
    }

def infer_fasterrcnn(bundle, frame, confidence):
    model = bundle["model"]
    device = torch.device(bundle["device"])

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    tensor = tensor.to(device)

    t0 = time.perf_counter()
    with torch.inference_mode():
        pred = model([tensor])[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    keep = pred["scores"] >= confidence
    boxes = pred["boxes"][keep].detach().cpu().numpy().astype(int)
    scores = pred["scores"][keep].detach().cpu().numpy()

    out = frame.copy()
    for (x1, y1, x2, y2), score in zip(boxes, scores):
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out, f"coconut {score:.2f}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )

    return {
        "frame": cv2.cvtColor(out, cv2.COLOR_BGR2RGB),
        "count": len(boxes),
        "class_counts": {"coconut": len(boxes)},
        "latency_ms": latency_ms,
        "fps": 1000.0 / latency_ms if latency_ms > 0 else 0.0,
    }

def infer(bundle, frame_bgr, task, confidence, iou, imgsz, tracking=False):
    if bundle["backend"] == "ultralytics":
        return infer_yolo(bundle, frame_bgr, task, confidence, iou, imgsz, tracking)
    return infer_fasterrcnn(bundle, frame_bgr, confidence)
