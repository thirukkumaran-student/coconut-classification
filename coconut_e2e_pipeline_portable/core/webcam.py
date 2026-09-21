import threading
import time

import cv2
import torch
from torchvision.transforms.functional import to_tensor


class WebcamWorker:

    def __init__(
        self,
        model,
        confidence=0.50,
        iou=0.50,
        imgsz=640,
        warmup_seconds=10,
        device="cpu",
        camera_index=0
    ):

        self.model = model

        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz
        self.warmup_seconds = warmup_seconds
        self.device = device
        self.camera_index = camera_index

        # =====================================================
        # THREAD
        # =====================================================

        self.thread = None

        self.stop_event = (
            threading.Event()
        )

        self.lock = (
            threading.Lock()
        )

        # =====================================================
        # STATE
        # =====================================================

        self.frame = None

        self.latency_ms = 0.0

        self.inference_fps = 0.0

        self.processing_fps = 0.0

        self.detections = 0

        self.running = False

        self.finished = False

        self.error = None

        # =====================================================
        # BENCHMARK
        # =====================================================

        self.benchmark_start_time = None

        self.frames_processed = 0

    # =========================================================
    # START
    # =========================================================

    def start(self):

        if self.running:
            return

        if self.thread is not None and self.thread.is_alive():
            return

        self.stop_event.clear()

        self.running = True

        self.finished = False

        self.error = None

        self.benchmark_start_time = None

        self.frames_processed = 0

        self.thread = threading.Thread(
            target=self._run,
            daemon=True
        )

        self.thread.start()

    # =========================================================
    # STOP
    # =========================================================

    def stop(self):

        self.stop_event.set()

        if (
            self.thread is not None
            and self.thread.is_alive()
        ):

            self.thread.join(
                timeout=5
            )

        self.running = False

    # =========================================================
    # YOLO
    # =========================================================

    def _run_yolo(self, frame):

        start = time.perf_counter()

        results = self.model.predict(

            source=frame,

            conf=self.confidence,

            iou=self.iou,

            imgsz=self.imgsz,

            device=self.device,

            verbose=False
        )

        end = time.perf_counter()

        latency_ms = (
            end - start
        ) * 1000.0

        result = results[0]

        annotated = result.plot()

        if result.boxes is not None:

            detections = len(
                result.boxes
            )

        else:

            detections = 0

        return (
            annotated,
            detections,
            latency_ms
        )

    # =========================================================
    # FASTER R-CNN
    # =========================================================

    def _run_faster_rcnn(self, frame):

        start = time.perf_counter()

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        tensor = to_tensor(
            rgb
        )

        tensor = tensor.to(
            self.device
        )

        with torch.no_grad():

            outputs = self.model(
                [tensor]
            )

        end = time.perf_counter()

        latency_ms = (
            end - start
        ) * 1000.0

        output = outputs[0]

        boxes = output["boxes"]

        labels = output["labels"]

        scores = output["scores"]

        # Confidence filtering
        keep = (
            scores >= self.confidence
        )

        boxes = boxes[keep]

        labels = labels[keep]

        scores = scores[keep]

        detections = len(
            boxes
        )

        annotated = frame.copy()

        for box, label, score in zip(
            boxes,
            labels,
            scores
        ):

            x1, y1, x2, y2 = (
                box
                .detach()
                .cpu()
                .numpy()
                .astype(int)
            )

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            text = (
                f"Coconut "
                f"{score.item():.2f}"
            )

            cv2.putText(
                annotated,
                text,
                (
                    x1,
                    max(y1 - 10, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        return (
            annotated,
            detections,
            latency_ms
        )

    # =========================================================
    # RUN
    # =========================================================

    def _run(self):

        cap = None

        try:

            # =================================================
            # OPEN WEBCAM
            # =================================================

            cap = cv2.VideoCapture(
    self.camera_index,
    cv2.CAP_V4L2
)

            if not cap.isOpened():

                raise RuntimeError(
                    f"Could not open webcam "
                    f"camera {self.camera_index}"
                )

            # =================================================
            # OPTIONAL CAMERA SETTINGS
            # =================================================

            cap.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                640
            )

            cap.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                480
            )

            # =================================================
            # WARM-UP
            # =================================================

            warmup_start = (
                time.perf_counter()
            )

            benchmark_started = False

            # =================================================
            # LOOP
            # =================================================

            while not self.stop_event.is_set():

                ret, frame = cap.read()

                if not ret:

                    raise RuntimeError(
                        "Could not read frame "
                        "from webcam."
                    )

                # =================================================
                # INFERENCE
                # =================================================

                if hasattr(
                    self.model,
                    "predict"
                ):

                    (
                        annotated,
                        detections,
                        latency_ms
                    ) = self._run_yolo(
                        frame
                    )

                else:

                    (
                        annotated,
                        detections,
                        latency_ms
                    ) = self._run_faster_rcnn(
                        frame
                    )

                # =================================================
                # FPS
                # =================================================

                if latency_ms > 0:

                    inference_fps = (
                        1000.0
                        / latency_ms
                    )

                else:

                    inference_fps = 0.0

                # =================================================
                # WARM-UP
                # =================================================

                now = (
                    time.perf_counter()
                )

                if (
                    not benchmark_started
                    and
                    now - warmup_start
                    >= self.warmup_seconds
                ):

                    benchmark_started = True

                    self.benchmark_start_time = (
                        time.perf_counter()
                    )

                    self.frames_processed = 0

                # =================================================
                # BENCHMARK
                # =================================================

                if benchmark_started:

                    self.frames_processed += 1

                    elapsed = (
                        time.perf_counter()
                        - self.benchmark_start_time
                    )

                    if elapsed > 0:

                        processing_fps = (
                            self.frames_processed
                            / elapsed
                        )

                    else:

                        processing_fps = 0.0

                else:

                    processing_fps = 0.0

                # =================================================
                # UPDATE STATE
                # =================================================

                with self.lock:

                    self.frame = annotated

                    self.latency_ms = (
                        latency_ms
                    )

                    self.inference_fps = (
                        inference_fps
                    )

                    self.processing_fps = (
                        processing_fps
                    )

                    self.detections = (
                        detections
                    )

        except Exception as e:

            with self.lock:

                self.error = str(e)

        finally:

            if cap is not None:

                cap.release()

            with self.lock:

                self.running = False

                self.finished = True

    # =========================================================
    # STATE
    # =========================================================

    def get_state(self):

        with self.lock:

            return {

                "frame":
                    self.frame,

                "latency_ms":
                    self.latency_ms,

                "inference_fps":
                    self.inference_fps,

                "processing_fps":
                    self.processing_fps,

                "detections":
                    self.detections,

                "frames_processed":
                    self.frames_processed,

                "running":
                    self.running,

                "finished":
                    self.finished,

                "error":
                    self.error,

                "benchmark_started":
                    self.benchmark_start_time
                    is not None
            }

    # =========================================================
    # BENCHMARK RESULTS
    # =========================================================

    def get_benchmark_results(self):

        with self.lock:

            if (
                self.benchmark_start_time
                is None
            ):

                return {

                    "benchmark_duration":
                        0.0,

                    "frames_processed":
                        0,

                    "average_fps":
                        0.0
                }

            duration = (
                time.perf_counter()
                - self.benchmark_start_time
            )

            if duration > 0:

                average_fps = (
                    self.frames_processed
                    / duration
                )

            else:

                average_fps = 0.0

            return {

                "benchmark_duration":
                    duration,

                "frames_processed":
                    self.frames_processed,

                "average_fps":
                    average_fps
            }