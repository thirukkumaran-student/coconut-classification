# import time
# from pathlib import Path

# import cv2
# import streamlit as st
# import torch

# from core.registry import ModelRegistry
# from core.inference import infer
# from core.pipeline import ImagePipeline, VideoPipeline
# from core.live_benchmark import LiveBenchmarkWorker


# # ============================================================
# # PAGE CONFIG
# # ============================================================

# st.set_page_config(
#     page_title="Coconut Plantation Monitoring",
#     page_icon="🌴",
#     layout="wide",
# )


# # ============================================================
# # PATHS
# # ============================================================

# BASE_DIR = Path(__file__).resolve().parent
# MODELS_DIR = BASE_DIR / "models"


# # ============================================================
# # SESSION STATE
# # ============================================================

# if "loaded_model" not in st.session_state:
#     st.session_state.loaded_model = None

# if "loaded_task" not in st.session_state:
#     st.session_state.loaded_task = None

# if "loaded_model_name" not in st.session_state:
#     st.session_state.loaded_model_name = None

# if "live_worker" not in st.session_state:
#     st.session_state.live_worker = None


# # ============================================================
# # MODEL DISPLAY NAME
# # ============================================================

# def display_model_name(model_name):

#     name = model_name.lower()

#     if "yolov11" in name or "yolo11" in name:
#         return "YOLO11n"

#     if "yolov8" in name or "yolo8" in name:
#         return "YOLOv8n"

#     if "fasterrcnn" in name:
#         return "Faster R-CNN"

#     return model_name


# # ============================================================
# # REGISTRY
# # ============================================================

# registry = ModelRegistry(MODELS_DIR)


# # ============================================================
# # HEADER
# # ============================================================

# st.title(
#     "Real-Time Multi-Task Coconut Plantation Monitoring"
# )

# st.write(
#     "UAV-based coconut plantation monitoring using "
#     "deep learning."
# )


# # ============================================================
# # SIDEBAR
# # ============================================================

# st.sidebar.header("Configuration")


# # ------------------------------------------------------------
# # Task
# # ------------------------------------------------------------

# task = st.sidebar.selectbox(
#     "Task",
#     registry.tasks
# )


# # ------------------------------------------------------------
# # Model
# # ------------------------------------------------------------

# model_names = registry.model_names(task)

# if not model_names:

#     st.sidebar.error(
#         "No models found for the selected task."
#     )

#     st.stop()


# model_name = st.sidebar.selectbox(
#     "Model",
#     model_names
# )


# # ------------------------------------------------------------
# # Device
# # ------------------------------------------------------------

# device_option = st.sidebar.selectbox(
#     "Device",
#     ["auto", "cuda", "cpu"]
# )

# runtime_device = registry.resolve_device(
#     device_option
# )


# # ------------------------------------------------------------
# # Inference settings
# # ------------------------------------------------------------

# imgsz = st.sidebar.number_input(
#     "Inference Size",
#     min_value=320,
#     max_value=1280,
#     value=640,
#     step=32
# )

# confidence = st.sidebar.slider(
#     "Confidence",
#     min_value=0.05,
#     max_value=1.0,
#     value=0.50,
#     step=0.05
# )

# iou = st.sidebar.slider(
#     "IoU",
#     min_value=0.05,
#     max_value=1.0,
#     value=0.50,
#     step=0.05
# )


# # ============================================================
# # INPUT SOURCE
# # ============================================================

# source_type = st.radio(
#     "Input",
#     [
#         "Image",
#         "Video",
#         "Webcam",
#         "Live Stream",
#     ],
#     horizontal=True
# )


# # ============================================================
# # LOAD MODEL
# # ============================================================

# if (
#     st.session_state.loaded_model is None
#     or st.session_state.loaded_task != task
#     or st.session_state.loaded_model_name != model_name
# ):

#     with st.spinner(
#         f"Loading {display_model_name(model_name)}..."
#     ):

#         model = registry.load(
#             task,
#             model_name,
#             runtime_device
#         )

#     st.session_state.loaded_model = model
#     st.session_state.loaded_task = task
#     st.session_state.loaded_model_name = model_name

# else:

#     model = st.session_state.loaded_model


# # ============================================================
# # MODEL INFORMATION
# # ============================================================

# col1, col2, col3 = st.columns(3)

# with col1:

#     st.metric(
#         "Task",
#         task
#     )

# with col2:

#     st.metric(
#         "Model",
#         display_model_name(model_name)
#     )

# with col3:

#     st.metric(
#         "Device",
#         runtime_device.upper()
#     )


# # ============================================================
# # IMAGE
# # ============================================================

# if source_type == "Image":

#     st.header("Image Analysis")

#     uploaded_file = st.file_uploader(
#         "Upload an image",
#         type=[
#             "jpg",
#             "jpeg",
#             "png",
#             "webp"
#         ]
#     )

#     if uploaded_file is not None:

#         file_bytes = uploaded_file.read()

#         image_bgr = cv2.imdecode(
#             __import__("numpy").frombuffer(
#                 file_bytes,
#                 dtype=__import__("numpy").uint8
#             ),
#             cv2.IMREAD_COLOR
#         )

#         if image_bgr is None:

#             st.error(
#                 "Could not read the uploaded image."
#             )

#         else:

#             image_rgb = cv2.cvtColor(
#                 image_bgr,
#                 cv2.COLOR_BGR2RGB
#             )

#             st.image(
#                 image_rgb,
#                 caption="Input Image",
#                 use_container_width=True
#             )

#             if st.button(
#                 "Run Detection",
#                 type="primary"
#             ):

#                 pipeline = ImagePipeline(
#                     model,
#                     task
#                 )

#                 result = pipeline.run(
#                     image_rgb,
#                     confidence,
#                     iou,
#                     imgsz
#                 )

#                 st.subheader(
#                     "Detection Result"
#                 )

#                 st.image(
#                     result["frame"],
#                     use_container_width=True
#                 )

#                 col1, col2, col3 = st.columns(3)

#                 with col1:

#                     st.metric(
#                         "Detections",
#                         result["count"]
#                     )

#                 with col2:

#                     st.metric(
#                         "Inference Latency",
#                         f"{result['latency_ms']:.2f} ms"
#                     )

#                 with col3:

#                     st.metric(
#                         "Inference FPS",
#                         f"{result['fps']:.2f} FPS"
#                     )

#                 if result["class_counts"]:

#                     st.subheader(
#                         "Class Counts"
#                     )

#                     st.json(
#                         result["class_counts"]
#                     )


# # ============================================================
# # VIDEO
# # ============================================================

# elif source_type == "Video":

#     st.header("Video Analysis")

#     uploaded_video = st.file_uploader(
#         "Upload a video",
#         type=[
#             "mp4",
#             "avi",
#             "mov",
#             "mkv"
#         ]
#     )

#     if uploaded_video is not None:

#         temp_video_path = (
#             BASE_DIR
#             / f"temp_{uploaded_video.name}"
#         )

#         with open(
#             temp_video_path,
#             "wb"
#         ) as f:

#             f.write(
#                 uploaded_video.read()
#             )

#         st.video(
#             str(temp_video_path)
#         )

#         if st.button(
#             "Run Video Detection",
#             type="primary"
#         ):

#             pipeline = VideoPipeline(
#                 model,
#                 task
#             )

#             frame_placeholder = st.empty()

#             metric_col1, metric_col2, metric_col3 = (
#                 st.columns(3)
#             )

#             frame_count = 0
#             total_detections = 0
#             last_result = None

#             for result in pipeline.run(
#                 str(temp_video_path),
#                 confidence,
#                 iou,
#                 imgsz,
#                 tracking=False
#             ):

#                 frame_count += 1
#                 total_detections += result["count"]

#                 last_result = result

#                 frame_placeholder.image(
#                     result["frame"],
#                     channels="RGB",
#                     use_container_width=True
#                 )

#                 metric_col1.metric(
#                     "Frames Processed",
#                     frame_count
#                 )

#                 metric_col2.metric(
#                     "Current Detections",
#                     result["count"]
#                 )

#                 metric_col3.metric(
#                     "Processing FPS",
#                     f"{result['processing_fps']:.2f}"
#                 )

#             if last_result is not None:

#                 st.subheader(
#                     "Video Results"
#                 )

#                 col1, col2, col3 = st.columns(3)

#                 with col1:

#                     st.metric(
#                         "Frames Processed",
#                         frame_count
#                     )

#                 with col2:

#                     st.metric(
#                         "Total Detections",
#                         total_detections
#                     )

#                 with col3:

#                     st.metric(
#                         "Final Processing FPS",
#                         f"{last_result['processing_fps']:.2f}"
#                     )


# # ============================================================
# # WEBCAM
# # ============================================================

# elif source_type == "Webcam":

#     st.header("Webcam Detection")

#     st.info(
#         "Webcam mode provides live visualization. "
#         "It is separate from the model-only benchmark."
#     )

#     camera_index = st.number_input(
#         "Camera Index",
#         min_value=0,
#         max_value=10,
#         value=0,
#         step=1
#     )

#     start_webcam = st.button(
#         "Start Webcam",
#         type="primary"
#     )

#     if start_webcam:

#         cap = cv2.VideoCapture(
#             int(camera_index)
#         )

#         if not cap.isOpened():

#             st.error(
#                 "Could not open webcam."
#             )

#         else:

#             frame_placeholder = st.empty()

#             metric_col1, metric_col2 = (
#                 st.columns(2)
#             )

#             stop_webcam = st.button(
#                 "Stop Webcam"
#             )

#             while cap.isOpened():

#                 ok, frame = cap.read()

#                 if not ok:
#                     break

#                 result = infer(
#                     model,
#                     frame,
#                     task,
#                     confidence,
#                     iou,
#                     imgsz,
#                     tracking=False
#                 )

#                 frame_placeholder.image(
#                     result["frame"],
#                     channels="RGB",
#                     use_container_width=True
#                 )

#                 metric_col1.metric(
#                     "Detections",
#                     result["count"]
#                 )

#                 metric_col2.metric(
#                     "Inference FPS",
#                     f"{result['fps']:.2f}"
#                 )

#                 if stop_webcam:
#                     break

#             cap.release()


# # ============================================================
# # LIVE STREAM
# # ============================================================

# elif source_type == "Live Stream":

#     st.header("Live Stream")

#     st.write(
#         "Connect the UAV stream through MediaMTX."
#     )

#     # --------------------------------------------------------
#     # RTSP URL
#     # --------------------------------------------------------

#     stream_url = st.text_input(
#         "RTSP Stream URL",
#         value="rtsp://192.168.1.9:8554/coconut"
#     )

#     # --------------------------------------------------------
#     # Benchmark configuration
#     # --------------------------------------------------------

#     st.subheader(
#         "Benchmark Configuration"
#     )

#     col1, col2 = st.columns(2)

#     with col1:

#         warmup_seconds = st.number_input(
#             "Warm-up Duration (seconds)",
#             min_value=1,
#             max_value=120,
#             value=10,
#             step=1
#         )

#     with col2:

#         benchmark_seconds = st.number_input(
#             "Benchmark Duration (seconds)",
#             min_value=10,
#             max_value=600,
#             value=60,
#             step=10
#         )

#     # --------------------------------------------------------
#     # Controls
#     # --------------------------------------------------------

#     col1, col2 = st.columns(2)

#     with col1:

#         start_benchmark = st.button(
#             "Start Live Stream Benchmark",
#             type="primary"
#         )

#     with col2:

#         stop_benchmark = st.button(
#             "Stop Benchmark"
#         )

#     # ========================================================
#     # START BENCHMARK
#     # ========================================================

#     if start_benchmark:

#         if (
#             st.session_state.live_worker is not None
#             and st.session_state.live_worker.running
#         ):

#             st.warning(
#                 "A benchmark is already running."
#             )

#         else:

#             worker = LiveBenchmarkWorker(
#                 model=model,
#                 source=stream_url,
#                 device=runtime_device,
#                 imgsz=imgsz,
#                 confidence=confidence,
#                 iou=iou,
#                 warmup_seconds=warmup_seconds,
#                 benchmark_seconds=benchmark_seconds
#             )

#             st.session_state.live_worker = worker

#             worker.start()

#             st.rerun()

#     # ========================================================
#     # STOP BENCHMARK
#     # ========================================================

#     if stop_benchmark:

#         worker = (
#             st.session_state.live_worker
#         )

#         if worker is not None:

#             worker.stop()

#     # ========================================================
#     # WORKER
#     # ========================================================

#     worker = (
#         st.session_state.live_worker
#     )

#     if worker is not None:

#         state = worker.get_state()

#         # ----------------------------------------------------
#         # Error
#         # ----------------------------------------------------

#         if state["error"]:

#             st.error(
#                 state["error"]
#             )

#         # ====================================================
#         # RUNNING
#         # ====================================================

#         if state["running"]:

#             st.subheader(
#                 "Live Stream Preview"
#             )

#             frame_placeholder = st.empty()

#             col1, col2, col3 = st.columns(3)

#             latency_placeholder = col1.empty()
#             fps_placeholder = col2.empty()
#             detection_placeholder = col3.empty()

#             # ------------------------------------------------
#             # Frame
#             # ------------------------------------------------

#             if state["frame"] is not None:

#                 display_frame = (
#                     state["frame"]
#                 )

#                 if state["result"] is not None:

#                     try:

#                         if len(
#                             state["result"]
#                         ) > 0:

#                             display_frame = (
#                                 state["result"][0].plot()
#                             )

#                     except Exception:

#                         display_frame = (
#                             state["frame"]
#                         )

#                 display_frame = cv2.cvtColor(
#                     display_frame,
#                     cv2.COLOR_BGR2RGB
#                 )

#                 frame_placeholder.image(
#                     display_frame,
#                     channels="RGB",
#                     use_container_width=True
#                 )

#             # ------------------------------------------------
#             # Current metrics
#             # ------------------------------------------------

#             latency_placeholder.metric(
#                 "Inference Latency",
#                 f"{state['inference_latency_ms']:.2f} ms"
#             )

#             fps_placeholder.metric(
#                 "Inference FPS",
#                 f"{state['inference_fps']:.2f} FPS"
#             )

#             detection_placeholder.metric(
#                 "Current Frame Detections",
#                 state["detection_count"]
#             )

#             # ------------------------------------------------
#             # Refresh
#             # ------------------------------------------------

#             time.sleep(0.1)

#             st.rerun()

#         # ====================================================
#         # FINISHED
#         # ====================================================

#         elif state["finished"]:

#             results = state["results"]

#             if results is not None:

#                 st.divider()

#                 st.subheader(
#                     "Live Stream Benchmark Results"
#                 )

#                 # =================================================
#                 # CONFIGURATION
#                 # =================================================

#                 st.markdown(
#                     "### Model Configuration"
#                 )

#                 st.write(
#                     {
#                         "Module": task,
#                         "Model": display_model_name(
#                             model_name
#                         ),
#                         "Input Source": stream_url,
#                         "Device": runtime_device,
#                         "Inference Size":
#                             f"{imgsz} × {imgsz}",
#                         "Confidence":
#                             f"{confidence:.2f}",
#                         "IoU":
#                             f"{iou:.2f}",
#                         "Warm-up":
#                             f"{warmup_seconds} s",
#                     }
#                 )

#                 # =================================================
#                 # BENCHMARK INFORMATION
#                 # =================================================

#                 st.markdown(
#                     "### Benchmark Information"
#                 )

#                 st.write(
#                     {
#                         "Benchmark Duration":
#                             f"{results['benchmark_duration']:.2f} s",
#                         "Inference Count":
#                             results["inference_count"],
#                     }
#                 )

#                 # =================================================
#                 # MODEL INFERENCE PERFORMANCE
#                 # =================================================

#                 st.markdown(
#                     "### Model Inference Performance"
#                 )

#                 col1, col2 = st.columns(2)

#                 with col1:

#                     st.metric(
#                         "Average Inference Latency",
#                         (
#                             f"{results['average_inference_latency_ms']:.2f} ms"
#                         )
#                     )

#                 with col2:

#                     st.metric(
#                         "Average Inference FPS",
#                         (
#                             f"{results['average_inference_fps']:.2f} FPS"
#                         )
#                     )

#                 st.caption(
#                     "These metrics measure only the model "
#                     "prediction operation. RTSP transfer, "
#                     "frame capture, rendering, and display "
#                     "are excluded."
#                 )

#                 # =================================================
#                 # DETECTION ACTIVITY
#                 # =================================================

#                 st.markdown(
#                     "### Detection Activity"
#                 )

#                 col1, col2, col3 = st.columns(3)

#                 with col1:

#                     st.metric(
#                         "Total Detections",
#                         results[
#                             "total_detections"
#                         ]
#                     )

#                 with col2:

#                     st.metric(
#                         "Frames With Detection",
#                         results[
#                             "frames_with_detection"
#                         ]
#                     )

#                 with col3:

#                     st.metric(
#                         "Frames Without Detection",
#                         results[
#                             "frames_without_detection"
#                         ]
#                     )

#                 col1, col2 = st.columns(2)

#                 with col1:

#                     st.metric(
#                         "Detection Activity Rate",
#                         (
#                             f"{results['detection_activity_rate']:.2f}%"
#                         )
#                     )

#                 with col2:

#                     st.metric(
#                         "Average Detections / Frame",
#                         (
#                             f"{results['average_detections_per_frame']:.2f}"
#                         )
#                     )

#                 st.caption(
#                     "Detection Activity Rate is the percentage "
#                     "of benchmarked frames containing at least "
#                     "one model detection. It is not detection "
#                     "accuracy."
#                 )

#                 # =================================================
#                 # PAPER REPORTING
#                 # =================================================

#                 st.divider()

#                 st.subheader(
#                     "Paper Reporting Information"
#                 )

#                 paper_text = f"""
# Module: {task}
# Model: {display_model_name(model_name)}
# Device: {runtime_device}
# Input resolution: {imgsz} × {imgsz}
# Confidence threshold: {confidence:.2f}
# IoU threshold: {iou:.2f}
# Warm-up duration: {warmup_seconds} s

# Benchmark duration: {results['benchmark_duration']:.2f} s
# Inference count: {results['inference_count']}

# Total detections: {results['total_detections']}
# Frames with detection: {results['frames_with_detection']}
# Frames without detection: {results['frames_without_detection']}
# Detection activity rate: {results['detection_activity_rate']:.2f}%
# Average detections per frame: {results['average_detections_per_frame']:.2f}

# Average inference latency: {results['average_inference_latency_ms']:.2f} ms
# Average inference FPS: {results['average_inference_fps']:.2f} FPS
# """

#                 st.code(
#                     paper_text.strip(),
#                     language="text"
#                 )

#                 st.caption(
#                     "Inference latency and inference FPS "
#                     "represent model prediction performance "
#                     "only."
#                 )

from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from core.benchmark import BenchmarkRecorder
from core.inference import infer, render_output
from core.live_benchmark import LiveBenchmarkWorker
from core.pipeline import ImagePipeline, VideoPipeline
from core.registry import ModelRegistry
from core.report import render_report
from core.system import get_system_info


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Coconut Plantation Monitoring",
    page_icon="🌴",
    layout="wide",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

STREAM_SCHEMES = (
    "rtsp://", "rtsps://", "rtmp://", "rtmps://",
    "http://", "https://", "srt://", "udp://",
)


# ============================================================
# SESSION STATE
# ============================================================

for _key in (
    "loaded_model",
    "loaded_task",
    "loaded_model_name",
    "loaded_device",
    "live_worker",
    "image_run",
    "video_run",
):
    if _key not in st.session_state:
        st.session_state[_key] = None


# ============================================================
# HELPERS
# ============================================================

def display_model_name(model_name):

    name = model_name.lower()

    if "yolov11" in name or "yolo11" in name:
        return "YOLO11n"

    if "yolov8" in name or "yolo8" in name:
        return "YOLOv8n"

    if "fasterrcnn" in name:
        return "Faster R-CNN"

    return model_name


def show_image(container, image, **kwargs):
    """st.image that fills the width on both old and new Streamlit."""

    try:
        container.image(image, width="stretch", **kwargs)
    except Exception:
        container.image(image, use_container_width=True, **kwargs)


@st.cache_data(show_spinner=False)
def cached_system_info():
    return get_system_info()


# ============================================================
# LIVE PREVIEW (fragment: only this part reruns while watching)
# ============================================================

@st.fragment(run_every=0.3)
def live_preview_fragment(show_preview):
    """
    Polls the live worker and updates just this section of the page.

    A plain st.rerun() loop refreshes the WHOLE app on every poll, which is
    what made the stream hard to watch. A fragment refreshes only its own
    area, so the sidebar, other results and the rest of the page stay put.

    Once the worker finishes, this calls a full st.rerun() exactly once to
    hand off to the static (non-polling) report below.
    """

    worker = st.session_state.get("live_worker")

    if worker is None:
        return

    state = worker.get_state()

    if state["error"]:
        st.error(state["error"])

    if state["finished"]:
        st.rerun()
        return

    if not state["running"]:
        return

    phase = state["phase"]

    if phase == "connecting":

        st.info("Connecting to the stream...")

    elif phase == "warmup":

        st.info(
            f"Warming up: {state['warmup_count']} inferences so far "
            f"({state['warmup_elapsed_s']:.0f} s). These are not counted."
        )

    else:

        if state["remaining_s"] is not None:

            total = state["elapsed_s"] + state["remaining_s"]

            st.progress(
                min(1.0, state["elapsed_s"] / total) if total else 0.0,
                text=(
                    f"Benchmarking: {state['elapsed_s']:.0f} s of "
                    f"{total:.0f} s"
                ),
            )

        else:

            st.info(
                f"Benchmarking for {state['elapsed_s']:.0f} s. "
                "Press Stop to finish and see the report."
            )

    if show_preview and state["frame"] is not None:

        if state["out"] is not None:
            preview = render_output(state["out"], state["frame"])
        else:
            preview = cv2.cvtColor(state["frame"], cv2.COLOR_BGR2RGB)

        show_image(st, preview, channels="RGB")

    c1, c2, c3 = st.columns(3)

    c1.metric("Mean latency", f"{state['mean_latency_ms']:.2f} ms")
    c2.metric("Mean FPS", f"{state['mean_fps']:.2f}")
    c3.metric("Frames measured", state["inference_count"])


# ============================================================
# REGISTRY
# ============================================================

registry = ModelRegistry(MODELS_DIR)


# ============================================================
# HEADER
# ============================================================

st.title("Real-Time Multi-Task Coconut Plantation Monitoring")

st.write("UAV-based coconut plantation monitoring using deep learning.")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Configuration")

task = st.sidebar.selectbox("Task", registry.tasks)

model_names = registry.model_names(task)

if not model_names:
    st.sidebar.error("No models found for the selected task.")
    st.stop()

model_name = st.sidebar.selectbox("Model", model_names)

device_option = st.sidebar.selectbox("Device", ["auto", "cuda", "cpu"])

runtime_device = registry.resolve_device(device_option)

imgsz = st.sidebar.number_input(
    "Inference Size",
    min_value=320,
    max_value=1280,
    value=640,
    step=32,
)

confidence = st.sidebar.slider(
    "Confidence", min_value=0.05, max_value=1.0, value=0.50, step=0.05
)

iou = st.sidebar.slider(
    "IoU", min_value=0.05, max_value=1.0, value=0.50, step=0.05
)


def build_config(**extra):
    """Settings snapshot that is stored WITH each benchmark result."""

    cfg = {
        "Module": task,
        "Model": display_model_name(model_name),
        "Model file": model_name,
        "Device": runtime_device,
        "Inference size": f"{imgsz} px (long side)",
        "Confidence": f"{confidence:.2f}",
        "IoU (NMS)": f"{iou:.2f}",
    }

    cfg.update(extra)

    return cfg


# ============================================================
# INPUT SOURCE
# ============================================================

source_type = st.radio(
    "Input",
    ["Image", "Video", "Webcam", "Live Stream"],
    horizontal=True,
)

# A live benchmark must not keep the GPU busy while another mode is used
_worker = st.session_state.live_worker

if source_type != "Live Stream" and _worker is not None and _worker.running:
    _worker.stop()
    st.info("The live stream benchmark was stopped because the input changed.")


# ============================================================
# LOAD MODEL
# ============================================================

needs_load = (
    st.session_state.loaded_model is None
    or st.session_state.loaded_task != task
    or st.session_state.loaded_model_name != model_name
    or st.session_state.loaded_device != runtime_device
)

if needs_load:

    with st.spinner(f"Loading {display_model_name(model_name)}..."):
        model = registry.load(task, model_name, runtime_device)

    st.session_state.loaded_model = model
    st.session_state.loaded_task = task
    st.session_state.loaded_model_name = model_name
    st.session_state.loaded_device = runtime_device

    # results of other models must not stay on screen
    st.session_state.image_run = None
    st.session_state.video_run = None

else:
    model = st.session_state.loaded_model


# ============================================================
# MODEL INFORMATION
# ============================================================

col1, col2, col3 = st.columns(3)

col1.metric("Task", task)
col2.metric("Model", display_model_name(model_name))
col3.metric("Device", runtime_device.upper())


# ============================================================
# IMAGE
# ============================================================

if source_type == "Image":

    st.header("Image Analysis")

    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:

        file_bytes = uploaded_file.getvalue()

        file_key = f"{uploaded_file.name}:{len(file_bytes)}"

        image_bgr = cv2.imdecode(
            np.frombuffer(file_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

        if image_bgr is None:

            st.error("Could not read the uploaded image.")

        else:

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            show_image(st, image_rgb, caption="Input Image")

            with st.expander("Benchmark settings"):

                bench_runs = st.number_input(
                    "Timed runs on this image",
                    min_value=1,
                    max_value=500,
                    value=30,
                    help="1 = a single run. More runs give stable "
                         "latency statistics.",
                )

                bench_warmup = st.number_input(
                    "Warm-up runs (not counted)",
                    min_value=0,
                    max_value=50,
                    value=5,
                    help="Removes cold-start cost (CUDA init, cuDNN "
                         "autotune, lazy allocation).",
                )

            if st.button("Run Detection", type="primary"):

                with st.spinner("Running..."):

                    pipeline = ImagePipeline(model, task)

                    result = pipeline.run(
                        image_rgb,
                        confidence,
                        iou,
                        imgsz,
                        runs=int(bench_runs),
                        warmup_runs=int(bench_warmup),
                    )

                recorder = result.pop("recorder")

                result["report"] = recorder.results(
                    config=build_config(
                        Input="Single image, repeated",
                        **{
                            "Timed runs": result["runs"],
                            "Warm-up runs": result["warmup_runs"],
                        },
                    ),
                    system=cached_system_info(),
                    end_reason="completed",
                )

                st.session_state.image_run = {
                    "key": file_key,
                    "result": result,
                }

            # rendered from session state so it survives download clicks
            run = st.session_state.image_run

            if run is not None and run["key"] == file_key:

                result = run["result"]

                st.subheader("Detection Result")

                show_image(st, result["frame"])

                c1, c2, c3 = st.columns(3)

                c1.metric("Detections", result["count"])

                if result["runs"] > 1:
                    c2.metric(
                        "Mean latency",
                        f"{result['report']['summary']['latency_ms']['mean']:.2f} ms",
                    )
                    c3.metric(
                        "Mean FPS",
                        f"{result['report']['summary']['fps']['mean']:.2f}",
                    )
                else:
                    c2.metric("Latency (single run)", f"{result['latency_ms']:.2f} ms")
                    c3.metric("FPS (single run)", f"{result['fps']:.2f}")

                if result["class_counts"]:
                    st.subheader("Class Counts")
                    st.json(result["class_counts"])

                if result["runs"] > 1:
                    render_report(
                        result["report"], key="image", title="Image Benchmark"
                    )
                else:
                    st.caption(
                        "Single run: the first call after loading a model is "
                        "usually slower (cold start). Use more timed runs "
                        "for a reliable latency figure."
                    )


# ============================================================
# VIDEO
# ============================================================

elif source_type == "Video":

    st.header("Video Analysis")

    uploaded_video = st.file_uploader(
        "Upload a video", type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:

        video_bytes = uploaded_video.getvalue()

        video_key = f"{uploaded_video.name}:{len(video_bytes)}"

        temp_video_path = BASE_DIR / f"temp_{uploaded_video.name}"

        if (
            not temp_video_path.exists()
            or temp_video_path.stat().st_size != len(video_bytes)
        ):
            temp_video_path.write_bytes(video_bytes)

        st.video(str(temp_video_path))

        video_warmup = st.number_input(
            "Warm-up frames (not counted in the benchmark)",
            min_value=0,
            max_value=200,
            value=5,
        )

        if st.button("Run Video Detection", type="primary"):

            pipeline = VideoPipeline(model, task)

            recorder = BenchmarkRecorder()

            frame_placeholder = st.empty()

            progress = st.progress(0.0)

            m1, m2, m3, m4 = st.columns(4)

            frame_count = 0

            for result in pipeline.run(
                str(temp_video_path),
                confidence,
                iou,
                imgsz,
                tracking=False,
            ):

                frame_count += 1

                if frame_count > video_warmup:
                    recorder.add(
                        result["latency_ms"],
                        result["preprocess_ms"],
                        result["inference_ms"],
                        result["postprocess_ms"],
                        result["count"],
                    )

                show_image(
                    frame_placeholder,
                    result["frame"],
                    channels="RGB",
                )

                if result["total_frames"] > 0:
                    progress.progress(
                        min(1.0, frame_count / result["total_frames"])
                    )

                m1.metric("Frames processed", frame_count)
                m2.metric("Current detections", result["count"])
                m3.metric("Model latency", f"{result['latency_ms']:.1f} ms")
                m4.metric("End-to-end FPS", f"{result['processing_fps']:.1f}")

            progress.empty()

            st.session_state.video_run = {
                "key": video_key,
                "frames": frame_count,
                "results": recorder.results(
                    config=build_config(
                        Input=f"Video file ({uploaded_video.name})",
                        **{
                            "Frames in video": frame_count,
                            "Warm-up frames": int(video_warmup),
                        },
                    ),
                    system=cached_system_info(),
                    end_reason="completed",
                ),
            }

        run = st.session_state.video_run

        if run is not None and run["key"] == video_key:

            if run["frames"] <= video_warmup:
                st.warning(
                    "The video has no more frames than the warm-up, so "
                    "nothing was measured. Lower the warm-up frames."
                )

            render_report(run["results"], key="video", title="Video Benchmark")

            st.caption(
                "End-to-end FPS while processing includes decoding, drawing "
                "and Streamlit display. Only the report above is model-only."
            )


# ============================================================
# WEBCAM
# ============================================================

elif source_type == "Webcam":

    st.header("Webcam Detection")

    st.info(
        "Webcam mode provides live visualization. "
        "It is separate from the model-only benchmark."
    )

    camera_index = st.number_input(
        "Camera Index", min_value=0, max_value=10, value=0, step=1
    )

    start_webcam = st.button("Start Webcam", type="primary")

    if start_webcam:

        cap = cv2.VideoCapture(int(camera_index))

        if not cap.isOpened():

            st.error("Could not open webcam.")

        else:

            frame_placeholder = st.empty()

            metric_col1, metric_col2 = st.columns(2)

            stop_webcam = st.button("Stop Webcam")

            try:

                while cap.isOpened():

                    ok, frame = cap.read()

                    if not ok:
                        break

                    result = infer(
                        model, frame, task, confidence, iou, imgsz,
                        tracking=False,
                    )

                    show_image(
                        frame_placeholder,
                        result["frame"],
                        channels="RGB",
                    )

                    metric_col1.metric("Detections", result["count"])
                    metric_col2.metric("Inference FPS", f"{result['fps']:.1f}")

                    if stop_webcam:
                        break

            finally:

                cap.release()


# ============================================================
# LIVE STREAM
# ============================================================

elif source_type == "Live Stream":

    st.header("Live Stream")

    st.write(
        "Connect the UAV stream through MediaMTX. RTSP and RTMP URLs both "
        "work. Only the model is benchmarked, not the stream."
    )

    stream_url = st.text_input(
        "Stream URL (rtsp:// or rtmp://)",
        value="rtsp://192.168.1.9:8554/coconut",
    )

    if (
        stream_url
        and "://" in stream_url
        and not stream_url.lower().startswith(STREAM_SCHEMES)
    ):
        st.warning(
            "Unrecognised URL scheme. Expected rtsp://, rtmp://, "
            "http(s)://, srt:// or udp://."
        )

    worker = st.session_state.live_worker

    is_running = worker is not None and worker.running

    # --------------------------------------------------------
    # Benchmark configuration
    # --------------------------------------------------------

    st.subheader("Benchmark Configuration")

    col1, col2 = st.columns(2)

    with col1:

        warmup_seconds = st.number_input(
            "Warm-up Duration (seconds, not counted)",
            min_value=1,
            max_value=120,
            value=10,
            step=1,
            disabled=is_running,
        )

    with col2:

        run_until_stop = st.checkbox(
            "Run until I press Stop",
            value=False,
            disabled=is_running,
        )

        benchmark_seconds = st.number_input(
            "Benchmark Duration (seconds)",
            min_value=10,
            max_value=3600,
            value=60,
            step=10,
            disabled=is_running or run_until_stop,
        )

    col1, col2 = st.columns(2)

    rtsp_tcp = col1.checkbox(
        "Use TCP for RTSP (recommended)",
        value=True,
        disabled=is_running,
    )

    show_preview = col2.checkbox(
        "Show live preview",
        value=True,
        help="Turn off for the cleanest numbers: drawing and streaming "
             "the preview competes with the model for CPU time.",
    )

    # --------------------------------------------------------
    # Controls
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    start_benchmark = col1.button(
        "Start Live Stream Benchmark",
        type="primary",
        disabled=is_running,
    )

    stop_benchmark = col2.button("Stop Benchmark", disabled=not is_running)

    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    if start_benchmark and not is_running:

        duration = None if run_until_stop else int(benchmark_seconds)

        worker = LiveBenchmarkWorker(
            model=model,
            source=stream_url.strip(),
            device=runtime_device,
            imgsz=int(imgsz),
            confidence=confidence,
            iou=iou,
            warmup_seconds=int(warmup_seconds),
            benchmark_seconds=duration,
            task=task,
            rtsp_tcp=rtsp_tcp,
            config=build_config(
                Input="Live stream",
                **{
                    "Stream URL": stream_url.strip(),
                    "Warm-up": f"{int(warmup_seconds)} s",
                    "Benchmark length": (
                        "until Stop" if duration is None else f"{duration} s"
                    ),
                    "RTSP transport": "TCP" if rtsp_tcp else "default (UDP)",
                },
            ),
            system=cached_system_info(),
        )

        st.session_state.live_worker = worker

        worker.start()

        st.rerun()

    # --------------------------------------------------------
    # STOP
    # --------------------------------------------------------

    if stop_benchmark and worker is not None:
        worker.stop()

    # --------------------------------------------------------
    # STATE
    #
    # "running" is watched through a fragment so only the preview area
    # refreshes; the finished report below is static (no polling).
    # --------------------------------------------------------

    worker = st.session_state.live_worker

    if worker is not None:

        state = worker.get_state()

        if state["running"]:

            st.subheader("Live Benchmark")

            live_preview_fragment(show_preview)

        elif state["finished"]:

            if state["error"]:
                st.error(state["error"])

            results = state["results"]

            if results is not None:
                render_report(
                    results,
                    key="live",
                    title="Live Stream Benchmark Results",
                )