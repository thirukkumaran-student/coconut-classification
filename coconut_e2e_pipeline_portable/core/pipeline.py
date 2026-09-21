import time
import cv2
from .inference import infer

class ImagePipeline:
    def __init__(self, model, task):
        self.model = model
        self.task = task

    def run(self, image_rgb, confidence, iou, imgsz):
        frame_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        return infer(
            self.model, frame_bgr, self.task,
            confidence, iou, imgsz, tracking=False
        )

class VideoPipeline:
    def __init__(self, model, task):
        self.model = model
        self.task = task

    def run(self, source, confidence, iou, imgsz, tracking=False):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {source}")

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_no = 0
        wall_start = time.perf_counter()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_no += 1
                t0 = time.perf_counter()

                result = infer(
                    self.model, frame, self.task,
                    confidence, iou, imgsz, tracking
                )

                elapsed = time.perf_counter() - wall_start
                processing_fps = frame_no / elapsed if elapsed > 0 else 0.0

                result["frame_no"] = frame_no
                result["source_fps"] = source_fps
                result["processing_fps"] = processing_fps
                yield result
        finally:
            cap.release()

class StreamPipeline(VideoPipeline):
    pass
