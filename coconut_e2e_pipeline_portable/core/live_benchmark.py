import threading
import time

import cv2
import torch
from torchvision.transforms.functional import to_tensor


class LiveBenchmarkWorker:

    def __init__(
        self,
        model,
        source,
        confidence=0.50,
        iou=0.50,
        imgsz=640,
        warmup_seconds=10,
        device="cpu"
    ):

        self.model = model
        self.source = source

        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz
        self.warmup_seconds = warmup_seconds
        self.device = device

        # =====================================================
        # THREAD CONTROL
        # =====================================================

        self.stop_event = threading.Event()

        self.thread = None

        self.lock = threading.Lock()

        # =====================================================
        # CURRENT FRAME
        # =====================================================

        self.frame = None

        # =====================================================
        # METRICS
        # =====================================================

        self.latency_ms = 0.0

        self.inference_fps = 0.0

        self.processing_fps = 0.0

        self.detections = 0

        # =====================================================
        # BENCHMARK
        # =====================================================

        self.benchmark_start_time = None

        self.frames_processed = 0

        # =====================================================
        # STATUS
        # =====================================================

        self.running = False

        self.finished = False

        self.error = None

    # =========================================================
    # START
    # =========================================================

    def start(self):

        if self.running:

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
    # MODEL TYPE
    # =========================================================

    def _is_yolo(self):

        return hasattr(
            self.model,
            "predict"
        )

    # =========================================================
    # YOLO INFERENCE
    # =========================================================

    def _run_yolo(
        self,
        frame
    ):

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

        # Annotated frame
        annotated = result.plot()

        # Detection count
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
    # FASTER R-CNN INFERENCE
    # =========================================================

    def _run_faster_rcnn(
        self,
        frame
    ):

        start = time.perf_counter()

        # BGR -> RGB
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # RGB image -> tensor
        tensor = to_tensor(
            rgb
        )

        tensor = tensor.to(
            self.device
        )

        # Inference
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

        # =====================================================
        # CONFIDENCE FILTER
        # =====================================================

        keep = (
            scores
            >= self.confidence
        )

        boxes = boxes[keep]

        labels = labels[keep]

        scores = scores[keep]

        detections = len(
            boxes
        )

        # =====================================================
        # DRAW DETECTIONS
        # =====================================================

        annotated = frame.copy()

        for box, label, score in zip(
            boxes,
            labels,
            scores
        ):

            coordinates = (
                box
                .detach()
                .cpu()
                .numpy()
                .astype(int)
            )

            x1, y1, x2, y2 = coordinates

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Faster R-CNN class 1 = coconut
            class_name = "Coconut"

            text = (
                f"{class_name} "
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
    # MAIN WORKER
    # =========================================================

    def _run(self):

        cap = None

        try:

            # =================================================
            # OPEN VIDEO SOURCE
            # =================================================

            cap = cv2.VideoCapture(
                self.source
            )

            if not cap.isOpened():

                raise RuntimeError(
                    f"Could not open video source: "
                    f"{self.source}"
                )

            # =================================================
            # WARM-UP
            # =================================================

            warmup_start = (
                time.perf_counter()
            )

            benchmark_started = False

            # =================================================
            # FRAME LOOP
            # =================================================

            while not self.stop_event.is_set():

                ret, frame = cap.read()

                if not ret:

                    raise RuntimeError(
                        "Could not read frame "
                        "from video source."
                    )

                # =================================================
                # MODEL INFERENCE
                # =================================================

                if self._is_yolo():

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
                # INFERENCE FPS
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

                warmup_elapsed = (
                    now
                    - warmup_start
                )

                if (
                    not benchmark_started
                    and warmup_elapsed
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
    # GET STATE
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
    # FINAL BENCHMARK RESULTS
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