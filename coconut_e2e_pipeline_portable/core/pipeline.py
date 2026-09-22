import time

import cv2

from .benchmark import BenchmarkRecorder
from .inference import infer


class ImagePipeline:

    def __init__(self, model, task):
        self.model = model
        self.task = task

    def run(
        self,
        image_rgb,
        confidence,
        iou,
        imgsz,
        runs=1,
        warmup_runs=0,
    ):
        """
        Run the model on one image.

        runs > 1 repeats the same image and records every run, so the
        latency statistics are not dominated by a single cold-start call.
        warmup_runs are executed first and NOT recorded.
        """

        frame_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        runs = max(1, int(runs))

        for _ in range(int(warmup_runs)):
            infer(
                self.model, frame_bgr, self.task,
                confidence, iou, imgsz,
                tracking=False, render=False
            )

        recorder = BenchmarkRecorder()

        result = None

        for i in range(runs):

            result = infer(
                self.model, frame_bgr, self.task,
                confidence, iou, imgsz,
                tracking=False,
                render=(i == runs - 1)
            )

            recorder.add(
                result["latency_ms"],
                result["preprocess_ms"],
                result["inference_ms"],
                result["postprocess_ms"],
                result["count"],
            )

        result["recorder"] = recorder
        result["runs"] = runs
        result["warmup_runs"] = int(warmup_runs)

        return result


class VideoPipeline:

    def __init__(self, model, task):
        self.model = model
        self.task = task

    def run(self, source, confidence, iou, imgsz, tracking=False):
        """
        Yields one result per frame.

        latency_ms / fps            -> model only
        processing_fps              -> end to end (decode + model + drawing
                                       + whatever the caller does between
                                       yields), NOT a model metric
        """

        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {source}")

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        frame_no = 0
        wall_start = time.perf_counter()

        try:

            while True:

                ok, frame = cap.read()

                if not ok:
                    break

                frame_no += 1

                result = infer(
                    self.model, frame, self.task,
                    confidence, iou, imgsz, tracking
                )

                elapsed = time.perf_counter() - wall_start

                result["frame_no"] = frame_no
                result["total_frames"] = total_frames
                result["source_fps"] = source_fps
                result["processing_fps"] = (
                    frame_no / elapsed if elapsed > 0 else 0.0
                )

                yield result

        finally:

            cap.release()


class StreamPipeline(VideoPipeline):
    pass
