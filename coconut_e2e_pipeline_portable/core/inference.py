import time

import cv2
import numpy as np
import torch


DISEASE_NAMES = [
    "BudRotDropping",
    "BudRot",
    "StemBleeding",
    "Gray Leaf Spot",
    "Leaf Rot",
]

MATURITY_NAMES = [
    "Tender Green",
    "Mature Green",
    "Tender King",
    "Mature King",
]


# ============================================================
# HELPERS
# ============================================================

def _counts(names):
    out = {}

    for n in names:
        out[n] = out.get(n, 0) + 1

    return out


def _is_cuda(device):
    return torch.cuda.is_available() and str(device).startswith("cuda")


def _sync(device):
    if _is_cuda(device):
        torch.cuda.synchronize()


def as_bundle(model_or_bundle, device=None):
    """
    Accept either the bundle returned by ModelRegistry.load()
    or a bare model, and always return a bundle dict.
    """

    if isinstance(model_or_bundle, dict) and "model" in model_or_bundle:
        return model_or_bundle

    model = model_or_bundle

    backend = "ultralytics" if hasattr(model, "predict") else "torchvision"

    if device is None:
        try:
            if backend == "ultralytics":
                device = str(model.device)
            else:
                device = str(next(model.parameters()).device)
        except Exception:
            device = "cpu"

    return {
        "model": model,
        "device": str(device),
        "backend": backend,
        "name": "",
    }


def _build_result(
    frame,
    count,
    class_counts,
    latency_ms,
    pre_ms,
    inf_ms,
    post_ms,
    raw=None,
    boxes=None,
    scores=None,
    labels=None,
):

    return {
        "frame": frame,
        "count": count,
        "class_counts": class_counts,

        # Model-only timings
        "latency_ms": latency_ms,
        "fps": (1000.0 / latency_ms) if latency_ms > 0 else 0.0,
        "preprocess_ms": pre_ms,
        "inference_ms": inf_ms,
        "postprocess_ms": post_ms,

        # Kept so the UI can draw later without re-running the model
        "_raw": raw,
        "_boxes": boxes,
        "_scores": scores,
        "_labels": labels,
    }


# ============================================================
# DRAWING (never part of the timing)
# ============================================================

def render_output(out, frame_bgr):
    """Return an RGB image with the detections of `out` drawn on it."""

    raw = out.get("_raw")

    if raw is not None:
        try:
            return cv2.cvtColor(raw.plot(), cv2.COLOR_BGR2RGB)
        except Exception:
            pass

    img = frame_bgr.copy()

    boxes = out.get("_boxes")
    scores = out.get("_scores")
    labels = out.get("_labels")

    if boxes is not None:

        for i, (x1, y1, x2, y2) in enumerate(boxes):

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            name = labels[i] if labels else "coconut"
            score = float(scores[i]) if scores is not None else 0.0

            cv2.putText(
                img,
                f"{name} {score:.2f}",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ============================================================
# YOLO INFERENCE
# ============================================================

def infer_yolo(
    bundle,
    frame,
    task,
    confidence,
    iou,
    imgsz,
    tracking=False,
    render=True,
):

    model = bundle["model"]
    device = bundle["device"]

    kwargs = dict(
        source=frame,
        conf=confidence,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )

    # Synchronize BEFORE starting CUDA timing
    _sync(device)

    t0 = time.perf_counter()

    if tracking:
        results = model.track(
            persist=True,
            tracker="bytetrack.yaml",
            **kwargs
        )
    else:
        results = model.predict(**kwargs)

    # Synchronize AFTER model prediction
    _sync(device)

    latency_ms = (time.perf_counter() - t0) * 1000.0

    r = results[0]

    # Ultralytics reports its own stage timings (ms)
    speed = getattr(r, "speed", None) or {}

    pre_ms = speed.get("preprocess")
    inf_ms = speed.get("inference")
    post_ms = speed.get("postprocess")

    # --------------------------------------------------------
    # Everything below is OUTSIDE the timed region
    # --------------------------------------------------------

    names = r.names

    class_names = []

    if r.boxes is not None and r.boxes.cls is not None:

        for cls in (
            r.boxes.cls
            .detach()
            .cpu()
            .numpy()
            .astype(int)
        ):
            class_names.append(names[int(cls)])

    elif getattr(r, "probs", None) is not None:
        # classification-type model: one label per image
        class_names.append(names[int(r.probs.top1)])

    annotated = None

    if render:
        annotated = cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB)

    return _build_result(
        annotated,
        len(class_names),
        _counts(class_names),
        latency_ms,
        pre_ms,
        inf_ms,
        post_ms,
        raw=r,
    )


# ============================================================
# FASTER R-CNN INFERENCE
# ============================================================

def infer_fasterrcnn(
    bundle,
    frame,
    confidence,
    iou,
    imgsz,
    render=True,
):

    model = bundle["model"]
    device = torch.device(bundle["device"])

    # Make the sidebar settings apply to this backend too.
    # (torchvision otherwise resizes to 800-1333 px and uses NMS 0.5.)
    try:
        model.transform.min_size = (int(imgsz),)
        model.transform.max_size = int(imgsz)
        model.roi_heads.nms_thresh = float(iou)
    except AttributeError:
        pass

    _sync(device)

    t0 = time.perf_counter()

    # ---- preprocess: colour convert, to tensor, host -> device
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    tensor = (
        torch.from_numpy(rgb)
        .permute(2, 0, 1)
        .float()
        / 255.0
    ).to(device)

    _sync(device)

    t1 = time.perf_counter()

    # ---- inference: forward pass (includes torchvision's own
    #      resize/normalise, RPN, ROI heads and NMS)
    with torch.inference_mode():
        pred = model([tensor])[0]

    _sync(device)

    t2 = time.perf_counter()

    # ---- postprocess: score filter, device -> host
    keep = pred["scores"] >= confidence

    boxes = (
        pred["boxes"][keep]
        .detach()
        .cpu()
        .numpy()
        .astype(int)
    )

    scores = (
        pred["scores"][keep]
        .detach()
        .cpu()
        .numpy()
    )

    t3 = time.perf_counter()

    latency_ms = (t3 - t0) * 1000.0

    labels = ["coconut"] * len(boxes)

    result = _build_result(
        None,
        len(boxes),
        {"coconut": len(boxes)},
        latency_ms,
        (t1 - t0) * 1000.0,
        (t2 - t1) * 1000.0,
        (t3 - t2) * 1000.0,
        boxes=boxes,
        scores=scores,
        labels=labels,
    )

    if render:
        result["frame"] = render_output(result, frame)

    return result


# ============================================================
# GENERAL INFERENCE INTERFACE
# ============================================================

def infer(
    bundle,
    frame_bgr,
    task,
    confidence,
    iou,
    imgsz,
    tracking=False,
    render=True,
):

    bundle = as_bundle(bundle)

    if bundle["backend"] == "ultralytics":

        return infer_yolo(
            bundle,
            frame_bgr,
            task,
            confidence,
            iou,
            imgsz,
            tracking,
            render,
        )

    return infer_fasterrcnn(
        bundle,
        frame_bgr,
        confidence,
        iou,
        imgsz,
        render,
    )
