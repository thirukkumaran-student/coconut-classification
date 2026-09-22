"""
Model-only benchmark utilities shared by the image, video and live-stream modes.

Everything recorded here comes from the model stage only:
preprocess + forward pass + postprocess (NMS / score filtering), with the GPU
synchronised before and after. Frame capture, decoding, network transfer,
drawing and Streamlit display are NOT part of these numbers.
"""

import csv
import io
import json
import threading
import time

import numpy as np

_NAN = float("nan")


def _nanmean(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return None
    return float(np.nanmean(arr))


def _fps(latency_ms):
    return 1000.0 / latency_ms if latency_ms and latency_ms > 0 else 0.0


class BenchmarkRecorder:
    """Thread-safe store of per-frame model timings."""

    def __init__(self, max_samples=500_000):
        self._lock = threading.Lock()
        self.max_samples = max_samples
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        with self._lock:
            self._t0 = time.perf_counter()
            self._t = []
            self._lat = []
            self._pre = []
            self._inf = []
            self._post = []
            self._det = []
            self._sum_lat = 0.0

    def __len__(self):
        with self._lock:
            return len(self._lat)

    # ------------------------------------------------------------------
    def add(
        self,
        latency_ms,
        preprocess_ms=None,
        inference_ms=None,
        postprocess_ms=None,
        detections=0,
    ):
        with self._lock:
            if len(self._lat) >= self.max_samples:
                return

            self._t.append(time.perf_counter() - self._t0)
            self._lat.append(float(latency_ms))
            self._pre.append(_NAN if preprocess_ms is None else float(preprocess_ms))
            self._inf.append(_NAN if inference_ms is None else float(inference_ms))
            self._post.append(_NAN if postprocess_ms is None else float(postprocess_ms))
            self._det.append(int(detections))
            self._sum_lat += float(latency_ms)

    # ------------------------------------------------------------------
    def live_stats(self, window=100):
        """Cheap numbers for the running view."""

        with self._lock:
            n = len(self._lat)

            if n == 0:
                return {
                    "count": 0,
                    "last_ms": 0.0,
                    "mean_ms": 0.0,
                    "mean_fps": 0.0,
                    "recent_p95_ms": 0.0,
                }

            recent = self._lat[-window:]
            mean_ms = self._sum_lat / n

            return {
                "count": n,
                "last_ms": self._lat[-1],
                "mean_ms": mean_ms,
                "mean_fps": _fps(mean_ms),
                "recent_p95_ms": float(np.percentile(recent, 95)),
            }

    def recent_latencies(self, n=200):
        with self._lock:
            return list(self._lat[-n:])

    # ------------------------------------------------------------------
    def summary(self, wall_seconds=None):
        with self._lock:
            lat = np.asarray(self._lat, dtype=float)
            pre = np.asarray(self._pre, dtype=float)
            inf = np.asarray(self._inf, dtype=float)
            post = np.asarray(self._post, dtype=float)
            det = np.asarray(self._det, dtype=int)

        n = int(lat.size)

        if n == 0:
            return {"frames": 0, "wall_seconds": wall_seconds}

        mean = float(lat.mean())
        median = float(np.median(lat))
        p95 = float(np.percentile(lat, 95))
        minimum = float(lat.min())

        latency = {
            "mean": mean,
            "median": median,
            "std": float(lat.std()),
            "min": minimum,
            "max": float(lat.max()),
            "p90": float(np.percentile(lat, 90)),
            "p95": p95,
            "p99": float(np.percentile(lat, 99)),
        }

        fps = {
            "mean": _fps(mean),
            "median": _fps(median),
            "worst_case_p95": _fps(p95),
            "peak": _fps(minimum),
        }

        # Stage breakdown (only when the backend reports it)
        stage_means = {
            "preprocess": _nanmean(pre),
            "inference": _nanmean(inf),
            "postprocess": _nanmean(post),
        }

        stages = None

        if all(v is not None for v in stage_means.values()):
            other = max(0.0, mean - sum(stage_means.values()))
            stage_means["other"] = other
            stages = {
                name: {
                    "mean_ms": value,
                    "share_pct": (value / mean * 100.0) if mean > 0 else 0.0,
                }
                for name, value in stage_means.items()
            }

        frames_with = int((det > 0).sum())

        detections = {
            "total": int(det.sum()),
            "per_frame_mean": float(det.mean()),
            "max_in_frame": int(det.max()),
            "frames_with": frames_with,
            "frames_without": n - frames_with,
            "activity_rate_pct": frames_with / n * 100.0,
        }

        return {
            "frames": n,
            "total_model_time_s": float(lat.sum() / 1000.0),
            "wall_seconds": wall_seconds,
            "latency_ms": latency,
            "fps": fps,
            "stages": stages,
            "detections": detections,
        }

    # ------------------------------------------------------------------
    def latency_series(self, max_points=2000):
        """Latency per frame for charting (bucket maxima if downsampled)."""

        with self._lock:
            lat = np.asarray(self._lat, dtype=float)

        n = lat.size

        if n <= max_points:
            return lat.tolist(), False

        edges = np.linspace(0, n, max_points + 1).astype(int)[:-1]

        return np.maximum.reduceat(lat, edges).tolist(), True

    # ------------------------------------------------------------------
    def samples_csv(self):
        with self._lock:
            rows = list(
                zip(
                    range(1, len(self._lat) + 1),
                    self._t,
                    self._lat,
                    self._pre,
                    self._inf,
                    self._post,
                    self._det,
                )
            )

        buf = io.StringIO()
        writer = csv.writer(buf)

        writer.writerow(
            [
                "frame",
                "t_s",
                "latency_ms",
                "preprocess_ms",
                "inference_ms",
                "postprocess_ms",
                "detections",
            ]
        )

        for frame, t, lat, pre, inf, post, det in rows:
            writer.writerow(
                [
                    frame,
                    f"{t:.4f}",
                    f"{lat:.4f}",
                    "" if pre != pre else f"{pre:.4f}",
                    "" if inf != inf else f"{inf:.4f}",
                    "" if post != post else f"{post:.4f}",
                    det,
                ]
            )

        return buf.getvalue()

    # ------------------------------------------------------------------
    def results(
        self,
        config=None,
        system=None,
        end_reason="completed",
        wall_seconds=None,
        extra=None,
    ):
        """Everything the report needs, in one plain dict."""

        series, downsampled = self.latency_series()

        out = {
            "summary": self.summary(wall_seconds),
            "config": dict(config or {}),
            "system": dict(system or {}),
            "end_reason": end_reason,
            "latency_series": series,
            "series_downsampled": downsampled,
            "samples_csv": self.samples_csv(),
        }

        if extra:
            out.update(extra)

        return out


# ======================================================================
# EXPORT HELPERS
# ======================================================================

def results_to_json(results):
    """JSON export (per-frame samples go to the CSV export instead)."""

    slim = {
        k: v
        for k, v in results.items()
        if k not in ("samples_csv", "latency_series", "series_downsampled")
    }

    return json.dumps(slim, indent=2, default=str)


def _ms(x):
    return "n/a" if x is None else f"{x:.2f} ms"


def format_paper_text(results):
    """Plain-text block that can be pasted into a paper or lab notebook."""

    summary = results["summary"]
    config = results.get("config", {})
    system = results.get("system", {})

    lines = []

    for key, value in config.items():
        lines.append(f"{key}: {value}")

    if system:
        lines.append("")
        lines.append(f"CPU: {system.get('cpu', 'n/a')}")
        ram = system.get("ram_gb")
        lines.append(f"RAM: {ram:.1f} GB" if ram else "RAM: n/a")
        lines.append(f"GPU: {system.get('gpu', 'n/a')}")

        if system.get("torch"):
            lines.append(
                f"PyTorch: {system['torch']} (CUDA {system.get('cuda') or 'n/a'})"
            )

    lines.append("")
    lines.append(f"Run ended: {results.get('end_reason', 'completed')}")

    if not summary.get("frames"):
        lines.append("Frames measured: 0")
        return "\n".join(lines)

    lat = summary["latency_ms"]
    fps = summary["fps"]
    det = summary["detections"]

    lines.append(f"Frames measured: {summary['frames']}")

    if summary.get("wall_seconds"):
        lines.append(f"Benchmark duration: {summary['wall_seconds']:.2f} s")

    lines.append("")
    lines.append("Model latency (preprocess + inference + postprocess)")
    lines.append(f"  Mean:   {_ms(lat['mean'])}")
    lines.append(f"  Median: {_ms(lat['median'])}")
    lines.append(f"  Std:    {_ms(lat['std'])}")
    lines.append(f"  Min:    {_ms(lat['min'])}")
    lines.append(f"  Max:    {_ms(lat['max'])}")
    lines.append(f"  P90:    {_ms(lat['p90'])}")
    lines.append(f"  P95:    {_ms(lat['p95'])}")
    lines.append(f"  P99:    {_ms(lat['p99'])}")

    lines.append("")
    lines.append("Inference speed")
    lines.append(f"  Mean FPS (1000 / mean latency): {fps['mean']:.2f}")
    lines.append(f"  Median FPS:                    {fps['median']:.2f}")
    lines.append(f"  Worst-case FPS (P95 latency):  {fps['worst_case_p95']:.2f}")

    stages = summary.get("stages")

    if stages:
        lines.append("")
        lines.append("Mean stage time")

        for name, item in stages.items():
            lines.append(
                f"  {name.capitalize():<12}{item['mean_ms']:.2f} ms "
                f"({item['share_pct']:.1f}%)"
            )

    lines.append("")
    lines.append("Detection activity")
    lines.append(f"  Total detections: {det['total']}")
    lines.append(f"  Average per frame: {det['per_frame_mean']:.2f}")
    lines.append(f"  Frames with detection: {det['frames_with']}")
    lines.append(f"  Frames without detection: {det['frames_without']}")
    lines.append(f"  Detection activity rate: {det['activity_rate_pct']:.2f}%")

    stream = results.get("stream")

    if stream:
        lines.append("")
        lines.append("Real-time keep-up")
        lines.append(f"  Frames arrived during benchmark: {stream['frames_captured']}")
        lines.append(f"  Frames processed: {stream['frames_processed']}")
        lines.append(f"  Frames skipped (model busy): {stream['frames_skipped']}")
        lines.append(f"  Keep-up: {stream['keep_up_pct']:.1f}%")

    return "\n".join(lines)
