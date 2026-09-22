import streamlit as st


def _ms(x):
    return "n/a" if x is None else f"{x:.2f} ms"


END_REASON_TEXT = {
    "completed": ("success", "Benchmark completed."),
    "stopped": (
        "info",
        "Stopped manually. Results cover the frames measured before Stop.",
    ),
    "stopped_in_warmup": (
        "warning",
        "Stopped during warm-up, so nothing was measured yet.",
    ),
    "stream_ended": (
        "warning",
        "The stream ended or dropped. Results cover the frames measured "
        "before that.",
    ),
    "error": ("error", "The run failed. Results cover what was measured."),
}


def render_report(results, key, title="Benchmark Results"):
    """
    Render a benchmark results dict produced by BenchmarkRecorder.results().

    Only two things are shown, on purpose: the model's own performance
    (latency, FPS) and its detection activity. `key` must be unique per
    report on the page.
    """

    summary = results["summary"]

    st.divider()
    st.subheader(title)

    level, message = END_REASON_TEXT.get(
        results.get("end_reason", "completed"),
        ("info", ""),
    )

    if message:
        getattr(st, level)(message)

    config = results.get("config") or {}
    model = config.get("Model")
    device = config.get("Device")

    if model or device:
        st.caption(" • ".join(v for v in (model, device and device.upper()) if v))

    if not summary.get("frames"):
        st.warning(
            "No inferences were recorded, so there are no benchmark "
            "numbers to show."
        )
        return

    lat = summary["latency_ms"]
    fps = summary["fps"]
    det = summary["detections"]

    # ----------------------------------------------------------
    # Model performance
    # ----------------------------------------------------------

    st.markdown("### Model Performance")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Mean Latency", _ms(lat["mean"]))
    c2.metric("Median Latency", _ms(lat["median"]))
    c3.metric("P95 Latency", _ms(lat["p95"]))
    c4.metric("Min Latency", _ms(lat["min"]))
    c5.metric("Max Latency", _ms(lat["max"]))

    c1, c2, c3 = st.columns(3)

    c1.metric("Mean FPS", f"{fps['mean']:.2f}")
    c2.metric("Worst-case FPS (P95)", f"{fps['worst_case_p95']:.2f}")
    c3.metric("Frames Measured", summary["frames"])

    st.caption(
        "Latency = preprocess + forward pass + postprocess, GPU-"
        "synchronised. FPS = 1000 / latency. Stream capture, decoding "
        "and display are excluded."
    )

    # ----------------------------------------------------------
    # Detections
    # ----------------------------------------------------------

    st.markdown("### Detections")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Detections", det["total"])
    c2.metric("Average per Frame", f"{det['per_frame_mean']:.2f}")
    c3.metric("Activity Rate", f"{det['activity_rate_pct']:.1f}%")
