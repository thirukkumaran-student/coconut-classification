from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from core.webcam import WebcamWorker
from core.live_benchmark import LiveBenchmarkWorker
from core.registry import ModelRegistry
from core.pipeline import ImagePipeline, VideoPipeline
from core.system import get_system_info


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Coconut Plantation Monitoring",
    page_icon="🌴",
    layout="wide"
)


# ============================================================
# PROJECT PATH
# ============================================================

BASE = Path(
    __file__
).resolve().parent


# ============================================================
# MODEL REGISTRY
# ============================================================

registry = ModelRegistry(
    BASE / "models"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "Coconut Plantation Monitoring"
)

st.caption(
    "Tree detection • Coconut counting • "
    "Maturity & variety • Disease detection"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Configuration"
    )

    # ========================================================
    # MODULE
    # ========================================================

    task = st.selectbox(
        "Module",
        registry.tasks
    )

    # ========================================================
    # MODEL
    # ========================================================

    available_models = (
        registry.model_names(task)
    )

    if not available_models:

        st.error(
            "No models found for this module."
        )

        st.stop()

    model_name = st.selectbox(
        "Model",
        available_models
    )

    # ========================================================
    # INPUT
    # ========================================================

    source_type = st.radio(
        "Input",
        [
            "Image",
            "Video",
            "Webcam",
            "Live Stream"
        ]
    )

    # ========================================================
    # CONFIDENCE
    # ========================================================

    confidence = st.slider(
        "Confidence",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05
    )

    # ========================================================
    # IOU
    # ========================================================

    iou = st.slider(
        "IoU",
        min_value=0.10,
        max_value=0.95,
        value=0.50,
        step=0.05
    )

    # ========================================================
    # DEVICE
    # ========================================================

    device = st.selectbox(
        "Device",
        [
            "auto",
            "cpu",
            "cuda"
        ]
    )

    # ========================================================
    # INFERENCE SIZE
    # ========================================================

    imgsz = st.selectbox(
        "Inference size",
        [
            320,
            416,
            512,
            640
        ],
        index=3
    )

    # ========================================================
    # TRACKING
    # ========================================================

    tracking = False

    if source_type in (
        "Video",
        "Webcam",
        "Live Stream"
    ):

        tracking = st.checkbox(
            "Enable tracking",
            value=False
        )

    # ========================================================
    # WARM-UP
    # ========================================================

    if source_type in (
        "Webcam",
        "Live Stream"
    ):

        st.divider()

        st.subheader(
            "Performance Benchmark"
        )

        warmup_seconds = st.number_input(
            "Warm-up duration (seconds)",
            min_value=0,
            max_value=60,
            value=10,
            step=5
        )

        st.caption(
            "Warm-up frames are excluded "
            "from the benchmark."
        )

    else:

        warmup_seconds = 0

    # ========================================================
    # SYSTEM INFORMATION
    # ========================================================

    st.divider()

    st.subheader(
        "System"
    )

    try:

        info = get_system_info()

        st.write(
            f"**CPU:** {info['cpu']}"
        )

        st.write(
            f"**RAM:** {info['ram_gb']:.1f} GB"
        )

        st.write(
            f"**GPU:** {info['gpu']}"
        )

    except Exception:

        st.caption(
            "System information unavailable."
        )


# ============================================================
# DEVICE RESOLUTION
# ============================================================

if device == "auto":

    runtime_device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

else:

    runtime_device = device

    if (
        runtime_device == "cuda"
        and not torch.cuda.is_available()
    ):

        st.warning(
            "CUDA is unavailable. "
            "Using CPU instead."
        )

        runtime_device = "cpu"


# ============================================================
# LOAD MODEL
# ============================================================

st.subheader(
    f"{task} — {model_name}"
)

try:

    model = registry.load(
        task=task,
        model_name=model_name,
        device=runtime_device
    )

except Exception as e:

    st.error(
        f"Could not load model:\n\n{e}"
    )

    st.stop()


# ============================================================
# IMAGE
# ============================================================

if source_type == "Image":

    uploaded = st.file_uploader(
        "Upload an image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )

    if uploaded is not None:

        from PIL import Image

        image = np.array(
            Image.open(
                uploaded
            ).convert("RGB")
        )

        if st.button(
            "Run Inference",
            type="primary"
        ):

            try:

                pipeline = ImagePipeline(
                    model,
                    task
                )

                result = pipeline.run(
                    image,
                    confidence=confidence,
                    iou=iou,
                    imgsz=imgsz
                )

                col1, col2 = (
                    st.columns(2)
                )

                with col1:

                    st.image(
                        image,
                        caption="Input",
                        use_container_width=True
                    )

                with col2:

                    st.image(
                        result["frame"],
                        caption="Annotated",
                        use_container_width=True
                    )

                m1, m2, m3, m4 = (
                    st.columns(4)
                )

                m1.metric(
                    "Detections",
                    result["count"]
                )

                m2.metric(
                    "Latency",
                    f"{result['latency_ms']:.1f} ms"
                )

                m3.metric(
                    "Inference FPS",
                    f"{result['fps']:.1f}"
                )

                m4.metric(
                    "Model",
                    model_name
                )

                if result.get(
                    "class_counts"
                ):

                    st.subheader(
                        "Class Counts"
                    )

                    st.json(
                        result["class_counts"]
                    )

            except Exception as e:

                st.error(
                    f"Inference failed:\n\n{e}"
                )


# ============================================================
# VIDEO
# ============================================================

elif source_type == "Video":

    uploaded = st.file_uploader(
        "Upload a video",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv"
        ]
    )

    if uploaded is not None:

        temp_dir = (
            BASE / ".streamlit_temp"
        )

        temp_dir.mkdir(
            exist_ok=True
        )

        video_path = (
            temp_dir
            / "input_video.mp4"
        )

        video_path.write_bytes(
            uploaded.getbuffer()
        )

        if st.button(
            "Process Video",
            type="primary"
        ):

            frame_placeholder = (
                st.empty()
            )

            metrics_placeholder = (
                st.empty()
            )

            try:

                pipeline = VideoPipeline(
                    model,
                    task
                )

                for result in pipeline.run(

                    str(video_path),

                    confidence=confidence,

                    iou=iou,

                    imgsz=imgsz,

                    tracking=tracking
                ):

                    frame = result["frame"]

                    if frame is not None:

                        frame = cv2.cvtColor(
                            frame,
                            cv2.COLOR_BGR2RGB
                        )

                    frame_placeholder.image(
                        frame,
                        use_container_width=True
                    )

                    metrics_placeholder.write(
                        f"Detections: "
                        f"**{result['count']}** | "
                        f"Inference: "
                        f"**{result['latency_ms']:.1f} ms** | "
                        f"Inference FPS: "
                        f"**{result['fps']:.1f}** | "
                        f"Processing FPS: "
                        f"**{result['processing_fps']:.1f}**"
                    )

            except Exception as e:

                st.error(
                    f"Video processing failed:\n\n{e}"
                )


# ============================================================
# WEBCAM
# ============================================================

elif source_type == "Webcam":

    st.subheader("Webcam")

    st.info(
        "The laptop's default webcam (camera 0) will be used."
    )

    # ========================================================
    # SESSION STATE
    # ========================================================

    if "webcam_worker" not in st.session_state:
        st.session_state.webcam_worker = None

    if "webcam_results" not in st.session_state:
        st.session_state.webcam_results = None

    # ========================================================
    # CURRENT WORKER
    # ========================================================

    worker = st.session_state.webcam_worker

    is_running = (
        worker is not None
        and worker.running
    )

    # ========================================================
    # BUTTONS
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        start_clicked = st.button(
            "▶ Start Webcam",
            type="primary",
            disabled=is_running,
            use_container_width=True
        )

    with col2:

        stop_clicked = st.button(
            "■ Stop Webcam & Finish Benchmark",
            disabled=not is_running,
            use_container_width=True
        )

    # ========================================================
    # START
    # ========================================================

    if start_clicked:

        old_worker = st.session_state.webcam_worker

        if old_worker is not None:
            old_worker.stop()

        st.session_state.webcam_results = None

        worker = WebcamWorker(
            model=model,
            confidence=confidence,
            iou=iou,
            imgsz=imgsz,
            warmup_seconds=warmup_seconds,
            device=runtime_device,
            camera_index=0
        )

        st.session_state.webcam_worker = worker

        worker.start()

        st.rerun()

    # ========================================================
    # STOP
    # ========================================================

    if stop_clicked:

        worker = st.session_state.webcam_worker

        if worker is not None:

            worker.stop()

            st.session_state.webcam_results = (
                worker.get_benchmark_results()
            )

        st.rerun()

    # ========================================================
    # UI CONTAINERS
    #
    # IMPORTANT:
    # These containers are created during the normal
    # Streamlit app run. The fragment only updates them.
    # ========================================================

    frame_placeholder = st.empty()

    metrics_placeholder = st.empty()

    status_placeholder = st.empty()

    # ========================================================
    # LIVE DISPLAY
    # ========================================================

    @st.fragment(run_every="300ms")
    def display_webcam():

        current_worker = (
            st.session_state.webcam_worker
        )

        # ----------------------------------------------------
        # NO WORKER
        # ----------------------------------------------------

        if current_worker is None:

            frame_placeholder.info(
                "Webcam is not running."
            )

            metrics_placeholder.empty()

            status_placeholder.empty()

            return

        # ----------------------------------------------------
        # GET STATE
        # ----------------------------------------------------

        state = current_worker.get_state()

        # ----------------------------------------------------
        # FRAME
        # ----------------------------------------------------

        if state["frame"] is not None:

            frame_rgb = cv2.cvtColor(
                state["frame"],
                cv2.COLOR_BGR2RGB
            )

            frame_placeholder.image(
                frame_rgb,
                caption="Webcam",
                use_container_width=True
            )

        else:

            frame_placeholder.info(
                "Waiting for webcam frame..."
            )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        with metrics_placeholder.container():

            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "Detections",
                state["detections"]
            )

            m2.metric(
                "Inference Latency",
                f"{state['latency_ms']:.1f} ms"
            )

            m3.metric(
                "Inference FPS",
                f"{state['inference_fps']:.1f}"
            )

            m4.metric(
                "Processing FPS",
                f"{state['processing_fps']:.1f}"
            )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        if state["error"]:

            status_placeholder.error(
                state["error"]
            )

        elif state["benchmark_started"]:

            status_placeholder.success(
                "Benchmark running..."
            )

        elif state["running"]:

            status_placeholder.info(
                "Warm-up in progress..."
            )

        elif state["finished"]:

            status_placeholder.warning(
                "Webcam stopped."
            )


    # ========================================================
    # START DISPLAY FRAGMENT
    # ========================================================

    display_webcam()

    # ========================================================
    # BENCHMARK RESULTS
    # ========================================================

    results = st.session_state.webcam_results

    if results is not None:

        st.divider()

        st.subheader(
            "Webcam Benchmark Results"
        )

        table = pd.DataFrame(
            [
                {
                    "Metric": "Module",
                    "Value": task
                },
                {
                    "Metric": "Model",
                    "Value": model_name
                },
                {
                    "Metric": "Input",
                    "Value": "Webcam"
                },
                {
                    "Metric": "Camera",
                    "Value": "Camera 0"
                },
                {
                    "Metric": "Device",
                    "Value": runtime_device
                },
                {
                    "Metric": "Inference Size",
                    "Value": f"{imgsz} × {imgsz}"
                },
                {
                    "Metric": "Confidence",
                    "Value": f"{confidence:.2f}"
                },
                {
                    "Metric": "IoU",
                    "Value": f"{iou:.2f}"
                },
                {
                    "Metric": "Warm-up",
                    "Value": f"{warmup_seconds} s"
                },
                {
                    "Metric": "Benchmark Duration",
                    "Value": (
                        f"{results['benchmark_duration']:.2f} s"
                    )
                },
                {
                    "Metric": "Frames Processed",
                    "Value": results["frames_processed"]
                },
                {
                    "Metric": "Average Processing FPS",
                    "Value": (
                        f"{results['average_fps']:.2f} FPS"
                    )
                }
            ]
        )

        st.table(table)

        st.metric(
            "Average Webcam Processing FPS",
            f"{results['average_fps']:.2f} FPS"
        )
# ============================================================
# LIVE STREAM
# ============================================================

elif source_type == "Live Stream":

    st.subheader(
        "Live Stream"
    )

    st.info(
        "Enter the URL of an external video stream. "
        "For the DJI + MediaMTX setup, use the RTSP URL."
    )

    # ========================================================
    # STREAM URL
    # ========================================================

    stream_url = st.text_input(
        "Stream URL",
        value="rtsp://192.168.1.25:8554/coconut",
        help=(
            "Example: "
            "rtsp://192.168.1.25:8554/coconut"
        )
    )

    stream_url = (
        stream_url.strip()
    )

    # ========================================================
    # SESSION STATE
    # ========================================================

    if "live_worker" not in st.session_state:

        st.session_state.live_worker = None

    if "benchmark_results" not in st.session_state:

        st.session_state.benchmark_results = None

    # ========================================================
    # CURRENT WORKER
    # ========================================================

    worker = (
        st.session_state.live_worker
    )

    is_running = (
        worker is not None
        and worker.running
    )

    # ========================================================
    # BUTTONS
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        start_clicked = st.button(
            "▶ Start Live Stream",
            type="primary",
            disabled=is_running,
            use_container_width=True
        )

    with col2:

        stop_clicked = st.button(
            "■ Stop Stream & Finish Benchmark",
            disabled=not is_running,
            use_container_width=True
        )

    # ========================================================
    # START STREAM
    # ========================================================

    if start_clicked:

        if not stream_url:

            st.error(
                "Please enter a stream URL."
            )

            st.stop()

        st.session_state.benchmark_results = None

        worker = LiveBenchmarkWorker(

            model=model,

            source=stream_url,

            confidence=confidence,

            iou=iou,

            imgsz=imgsz,

            warmup_seconds=warmup_seconds,

            device=runtime_device
        )

        st.session_state.live_worker = worker

        worker.start()

        st.rerun()

    # ========================================================
    # STOP STREAM
    # ========================================================

    if stop_clicked:

        worker = (
            st.session_state.live_worker
        )

        if worker is not None:

            worker.stop()

            st.session_state.benchmark_results = (
                worker.get_benchmark_results()
            )

        st.rerun()

    # ========================================================
    # DISPLAY STREAM
    # ========================================================

    worker = (
        st.session_state.live_worker
    )

    if worker is not None:

        @st.fragment(
            run_every="300ms"
        )
        def display_stream():

            current_worker = (
                st.session_state.live_worker
            )

            if current_worker is None:

                st.info(
                    "Live stream is not running."
                )

                return

            state = (
                current_worker.get_state()
            )

            # ------------------------------------------------
            # FRAME
            # ------------------------------------------------

            if state["frame"] is not None:

                frame_rgb = cv2.cvtColor(
                    state["frame"],
                    cv2.COLOR_BGR2RGB
                )

                st.image(
                    frame_rgb,
                    caption="Live Stream",
                    use_container_width=True
                )

            else:

                st.info(
                    "Waiting for stream frame..."
                )

            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "Detections",
                state["detections"]
            )

            m2.metric(
                "Inference Latency",
                f"{state['latency_ms']:.1f} ms"
            )

            m3.metric(
                "Inference FPS",
                f"{state['inference_fps']:.1f}"
            )

            m4.metric(
                "Processing FPS",
                f"{state['processing_fps']:.1f}"
            )

            # ------------------------------------------------
            # STATUS
            # ------------------------------------------------

            if state["error"]:

                st.error(
                    state["error"]
                )

            elif state["benchmark_started"]:

                st.success(
                    "Benchmark running..."
                )

            elif state["running"]:

                st.info(
                    "Warm-up in progress..."
                )

            elif state["finished"]:

                st.warning(
                    "Live stream stopped."
                )

        display_stream()

    # ========================================================
    # LIVE STREAM BENCHMARK
    # ========================================================

    results = (
        st.session_state.benchmark_results
    )

    if results is not None:

        st.divider()

        st.subheader(
            "Live Stream Benchmark Results"
        )

        table = pd.DataFrame(
            [
                {
                    "Metric": "Module",
                    "Value": task
                },
                {
                    "Metric": "Model",
                    "Value": model_name
                },
                {
                    "Metric": "Input Source",
                    "Value": stream_url
                },
                {
                    "Metric": "Device",
                    "Value": runtime_device
                },
                {
                    "Metric": "Inference Size",
                    "Value":
                        f"{imgsz} × {imgsz}"
                },
                {
                    "Metric": "Confidence",
                    "Value":
                        f"{confidence:.2f}"
                },
                {
                    "Metric": "IoU",
                    "Value":
                        f"{iou:.2f}"
                },
                {
                    "Metric": "Warm-up",
                    "Value":
                        f"{warmup_seconds} s"
                },
                {
                    "Metric": "Benchmark Duration",
                    "Value":
                        f"{results['benchmark_duration']:.2f} s"
                },
                {
                    "Metric": "Frames Processed",
                    "Value":
                        results["frames_processed"]
                },
                {
                    "Metric": "Average Processing FPS",
                    "Value":
                        f"{results['average_fps']:.2f} FPS"
                }
            ]
        )

        st.table(
            table
        )

        st.metric(
            "Average Live-Stream Processing FPS",
            f"{results['average_fps']:.2f} FPS"
        )

        # ====================================================
        # PAPER INFORMATION
        # ====================================================

        st.subheader(
            "Paper Reporting Information"
        )

        paper_text = (
            f"Module: {task}\n"
            f"Model: {model_name}\n"
            f"Input source: {stream_url}\n"
            f"Device: {runtime_device}\n"
            f"Input resolution: "
            f"{imgsz} × {imgsz}\n"
            f"Confidence threshold: "
            f"{confidence:.2f}\n"
            f"IoU threshold: "
            f"{iou:.2f}\n"
            f"Warm-up duration: "
            f"{warmup_seconds} s\n"
            f"Benchmark duration: "
            f"{results['benchmark_duration']:.2f} s\n"
            f"Frames processed: "
            f"{results['frames_processed']}\n"
            f"Average processing FPS: "
            f"{results['average_fps']:.2f} FPS"
        )

        st.code(
            paper_text,
            language="text"
        )