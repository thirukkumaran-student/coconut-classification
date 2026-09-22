import os
import threading
import time

import cv2

from .benchmark import BenchmarkRecorder
from .inference import as_bundle, infer


# Warm-up ends after `warmup_seconds` AND at least this many inferences
MIN_WARMUP_INFERENCES = 3

# How long the inference loop waits for a new frame before re-checking Stop
FRAME_WAIT_S = 0.5


class LiveBenchmarkWorker:
    """
    Benchmarks the MODEL on a live stream (RTSP / RTMP / file / camera index).

    Design
    ------
    * A reader thread keeps only the newest frame. Capture, decoding and
      network jitter therefore never enter the model timings, and the model
      always works on a fresh frame instead of a stale buffered one.
    * The worker thread times only `infer()` (preprocess + forward +
      postprocess, GPU-synchronised). Drawing is left to the UI.
    * Phases: connecting -> warmup -> benchmarking -> finished.
    * stop() ends the run at any phase and still produces a results dict
      (partial if stopped early).
    * `benchmark_seconds=None` means: run until stop() is called.
    """

    def __init__(
        self,
        model,
        source,
        device=None,
        imgsz=640,
        confidence=0.5,
        iou=0.5,
        warmup_seconds=10,
        benchmark_seconds=60,
        task="",
        rtsp_tcp=True,
        read_timeout=5.0,
        config=None,
        system=None,
    ):

        self.bundle = as_bundle(model, device)
        self.device = self.bundle["device"]

        self.source = source
        self.task = task
        self.imgsz = imgsz
        self.confidence = confidence
        self.iou = iou

        self.warmup_seconds = warmup_seconds
        self.benchmark_seconds = benchmark_seconds

        self.rtsp_tcp = rtsp_tcp
        self.read_timeout = read_timeout

        # Snapshot of the settings used for THIS run (shown in the report)
        self.config = dict(config or {})
        self.system = dict(system or {})

        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._frame_cv = threading.Condition()

        self.thread = None

        self._reset()

    # =========================================================
    # STATE
    # =========================================================

    def _reset(self):

        self.recorder = BenchmarkRecorder()

        self._running = False
        self.finished = False
        self.error = None
        self.phase = "idle"
        self.results = None

        # newest-frame buffer (written by the reader thread)
        self._latest = None
        self._frame_id = 0
        self._captured = 0
        self._stream_lost = False

        # display state (written by the worker thread)
        self.latest_frame = None
        self.latest_out = None
        self.latest_detection_count = 0
        self.warmup_count = 0

        self.warmup_start = None
        self.bench_start = None

    @property
    def running(self):
        return self._running

    def _set_phase(self, phase):
        with self._lock:
            self.phase = phase

    # =========================================================
    # START / STOP
    # =========================================================

    def start(self):

        if self._running:
            return

        self._reset()
        self._stop_event.clear()

        self._running = True
        self.phase = "connecting"

        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="live-benchmark",
        )

        self.thread.start()

    def stop(self):

        self._stop_event.set()

        with self._frame_cv:
            self._frame_cv.notify_all()

    # =========================================================
    # CAPTURE
    # =========================================================

    def _open_capture(self):

        src = self.source

        # Camera index
        if not isinstance(src, str):
            return cv2.VideoCapture(src)

        is_rtsp = src.lower().startswith(("rtsp://", "rtsps://"))

        env_key = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
        previous = os.environ.get(env_key)

        if is_rtsp and self.rtsp_tcp:
            # UDP drops packets and produces smeared frames; TCP is safer
            os.environ[env_key] = "rtsp_transport;tcp|fflags;nobuffer"

        try:

            params = [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                8000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                int(self.read_timeout * 1000),
            ]

            try:
                return cv2.VideoCapture(src, cv2.CAP_FFMPEG, params)
            except (TypeError, cv2.error):
                return cv2.VideoCapture(src)

        finally:

            if previous is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = previous

    def _reader_loop(self, cap):
        """Keep only the newest frame; never blocks the model."""

        src = self.source
        is_file = isinstance(src, str) and "://" not in src

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

        # A local file would otherwise be read as fast as it decodes;
        # pace it like a real stream so keep-up numbers make sense.
        pace = (1.0 / fps) if (is_file and 1.0 < fps < 240.0) else 0.0

        last_ok = time.perf_counter()

        while not self._stop_event.is_set():

            loop_start = time.perf_counter()

            ok, frame = cap.read()

            now = time.perf_counter()

            if ok and frame is not None:

                last_ok = now

                with self._frame_cv:
                    self._latest = frame
                    self._frame_id += 1
                    self._captured += 1
                    self._frame_cv.notify_all()

                if pace:
                    remaining = pace - (time.perf_counter() - loop_start)

                    if remaining > 0:
                        time.sleep(remaining)

            else:

                if now - last_ok > self.read_timeout:

                    with self._frame_cv:
                        self._stream_lost = True
                        self._frame_cv.notify_all()

                    return

                time.sleep(0.01)

    def _next_frame(self, last_id):
        """Wait for a frame newer than `last_id`. Returns (frame, id)."""

        with self._frame_cv:

            deadline = time.perf_counter() + FRAME_WAIT_S

            while (
                self._frame_id <= last_id
                and not self._stream_lost
                and not self._stop_event.is_set()
            ):

                remaining = deadline - time.perf_counter()

                if remaining <= 0:
                    break

                self._frame_cv.wait(remaining)

            if self._frame_id > last_id:
                return self._latest, self._frame_id

            return None, last_id

    # =========================================================
    # MODEL CALL (the only timed operation)
    # =========================================================

    def _predict(self, frame):

        return infer(
            self.bundle,
            frame,
            self.task,
            self.confidence,
            self.iou,
            self.imgsz,
            tracking=False,
            render=False,
        )

    # =========================================================
    # MAIN
    # =========================================================

    def _run(self):

        cap = None
        reader = None
        wall = None
        end_reason = "completed"
        stream_stats = None

        try:

            # -------------------------------------------------
            # CONNECT
            # -------------------------------------------------

            cap = self._open_capture()

            if not cap.isOpened():
                raise RuntimeError(
                    f"Could not open stream:\n{self.source}"
                )

            reader = threading.Thread(
                target=self._reader_loop,
                args=(cap,),
                daemon=True,
                name="live-reader",
            )

            reader.start()

            # -------------------------------------------------
            # WARM-UP (not counted)
            # -------------------------------------------------

            self._set_phase("warmup")

            last_id = 0
            warm_n = 0

            self.warmup_start = time.perf_counter()

            while not self._stop_event.is_set():

                if (
                    warm_n >= MIN_WARMUP_INFERENCES
                    and time.perf_counter() - self.warmup_start
                    >= self.warmup_seconds
                ):
                    break

                frame, fid = self._next_frame(last_id)

                if frame is None:

                    if self._stream_lost:
                        break

                    continue

                last_id = fid

                self._predict(frame)

                warm_n += 1

                with self._lock:
                    self.warmup_count = warm_n
                    self.latest_frame = frame

            if self._stop_event.is_set():
                end_reason = "stopped_in_warmup"

            elif warm_n == 0:
                raise RuntimeError(
                    "No frames were received from the stream "
                    f"within {self.read_timeout:.0f} s:\n{self.source}"
                )

            # -------------------------------------------------
            # BENCHMARK
            # -------------------------------------------------

            if end_reason == "completed":

                self._set_phase("benchmarking")

                with self._frame_cv:
                    pending = self._frame_id > last_id
                    captured_at_start = self._captured - (1 if pending else 0)

                bench_start = time.perf_counter()

                with self._lock:
                    self.bench_start = bench_start

                while True:

                    if self._stop_event.is_set():
                        end_reason = "stopped"
                        break

                    if (
                        self.benchmark_seconds
                        and time.perf_counter() - bench_start
                        >= self.benchmark_seconds
                    ):
                        end_reason = "completed"
                        break

                    frame, fid = self._next_frame(last_id)

                    if frame is None:

                        if self._stream_lost:
                            end_reason = "stream_ended"
                            break

                        continue

                    last_id = fid

                    out = self._predict(frame)

                    self.recorder.add(
                        out["latency_ms"],
                        out["preprocess_ms"],
                        out["inference_ms"],
                        out["postprocess_ms"],
                        out["count"],
                    )

                    with self._lock:
                        self.latest_frame = frame
                        self.latest_out = out
                        self.latest_detection_count = out["count"]

                wall = time.perf_counter() - bench_start

                processed = len(self.recorder)

                with self._frame_cv:
                    captured = max(
                        processed,
                        self._captured - captured_at_start,
                    )

                stream_stats = {
                    "frames_captured": captured,
                    "frames_processed": processed,
                    "frames_skipped": captured - processed,
                    "keep_up_pct": (
                        processed / captured * 100.0 if captured else 0.0
                    ),
                    "arrival_fps": captured / wall if wall else 0.0,
                }

        except Exception as e:

            self.error = str(e)
            end_reason = "error"

        finally:

            # stop the reader, free the capture
            self._stop_event.set()

            with self._frame_cv:
                self._frame_cv.notify_all()

            if reader is not None:
                reader.join(timeout=3)

            if cap is not None:
                cap.release()

            # ---------------------------------------------
            # RESULTS (also for partial / stopped runs)
            # ---------------------------------------------

            try:

                self.results = self.recorder.results(
                    config=self.config,
                    system=self.system,
                    end_reason=end_reason,
                    wall_seconds=wall,
                    extra=(
                        {"stream": stream_stats}
                        if stream_stats
                        else None
                    ),
                )

            except Exception as e:

                if not self.error:
                    self.error = f"Could not build results: {e}"

            with self._lock:
                self.phase = "error" if self.error else "finished"
                self._running = False
                self.finished = True

    # =========================================================
    # CURRENT STATE (polled by the UI)
    # =========================================================

    def get_state(self):

        live = self.recorder.live_stats()

        with self._lock:

            now = time.perf_counter()

            elapsed = 0.0

            if self.bench_start is not None and self._running:
                elapsed = now - self.bench_start

            warm_elapsed = 0.0

            if self.warmup_start is not None and self.phase == "warmup":
                warm_elapsed = now - self.warmup_start

            remaining = None

            if self.benchmark_seconds:
                remaining = max(0.0, self.benchmark_seconds - elapsed)

            return {
                "running": self._running,
                "finished": self.finished,
                "error": self.error,
                "phase": self.phase,

                "frame": self.latest_frame,
                "out": self.latest_out,

                "warmup_count": self.warmup_count,
                "warmup_elapsed_s": warm_elapsed,

                "elapsed_s": elapsed,
                "remaining_s": remaining,

                "inference_count": live["count"],
                "last_latency_ms": live["last_ms"],
                "mean_latency_ms": live["mean_ms"],
                "mean_fps": live["mean_fps"],
                "recent_p95_ms": live["recent_p95_ms"],
                "detection_count": self.latest_detection_count,

                "recent_latencies": self.recorder.recent_latencies(200),

                "results": self.results,
            }
