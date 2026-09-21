import streamlit as st
from pathlib import Path
from core.registry import ModelRegistry
from core.pipeline import ImagePipeline, VideoPipeline, StreamPipeline
from core.system import get_system_info

st.set_page_config(page_title="Coconut Plantation Monitoring", page_icon="🌴", layout="wide")

BASE = Path(__file__).resolve().parent
registry = ModelRegistry(BASE / "models")

st.title("Coconut Plantation Monitoring")
st.caption("Tree detection • Coconut counting • Maturity & variety • Disease detection")

with st.sidebar:
    st.header("Configuration")
    task = st.selectbox("Module", registry.tasks)
    model_name = st.selectbox("Model", registry.model_names(task))

    source_type = st.radio("Input", ["Image", "Video", "Live Stream"])
    confidence = st.slider("Confidence", 0.05, 0.95, 0.50, 0.05)
    iou = st.slider("IoU", 0.10, 0.95, 0.50, 0.05)

    device = st.selectbox("Device", ["auto", "cpu", "cuda"])
    imgsz = st.selectbox("Inference size", [320, 416, 512, 640], index=3)

    tracking = False
    if source_type in ("Video", "Live Stream"):
        tracking = st.checkbox("Enable tracking", value=False)

    st.divider()
    st.subheader("System")
    info = get_system_info()
    st.write(f"**CPU:** {info['cpu']}")
    st.write(f"**RAM:** {info['ram_gb']:.1f} GB")
    st.write(f"**GPU:** {info['gpu']}")

st.subheader(f"{task} — {model_name}")

try:
    model = registry.load(task, model_name, device=device)
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

if source_type == "Image":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        import numpy as np
        from PIL import Image

        image = np.array(Image.open(uploaded).convert("RGB"))
        if st.button("Run inference", type="primary"):
            result = ImagePipeline(model, task).run(
                image, confidence=confidence, iou=iou, imgsz=imgsz
            )
            c1, c2 = st.columns(2)
            with c1:
                st.image(image, caption="Input", use_container_width=True)
            with c2:
                st.image(result["frame"], caption="Annotated", use_container_width=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Detections", result["count"])
            m2.metric("Latency", f"{result['latency_ms']:.1f} ms")
            m3.metric("Inference FPS", f"{result['fps']:.1f}")
            m4.metric("Model", model_name)

            if result["class_counts"]:
                st.subheader("Class counts")
                st.json(result["class_counts"])

elif source_type == "Video":
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded:
        tmp = Path("streamlit_input.mp4")
        tmp.write_bytes(uploaded.getbuffer())
        if st.button("Process video", type="primary"):
            placeholder = st.empty()
            metrics = st.empty()
            pipeline = VideoPipeline(model, task)

            for result in pipeline.run(
                str(tmp), confidence=confidence, iou=iou,
                imgsz=imgsz, tracking=tracking
            ):
                placeholder.image(result["frame"], channels="RGB", use_container_width=True)
                metrics.write(
                    f"Detections: **{result['count']}** | "
                    f"Inference: **{result['latency_ms']:.1f} ms** | "
                    f"Inference FPS: **{result['fps']:.1f}** | "
                    f"Processing FPS: **{result['processing_fps']:.1f}** | "
                    f"Frame: **{result['frame_no']}**"
                )

elif source_type == "Live Stream":
    source = st.text_input(
        "Stream source",
        value="0",
        help="Use 0 for the laptop webcam, or enter a stream URL/path such as RTSP/RTMP if your capture setup provides one."
    )
    if st.button("Start stream", type="primary"):
        try:
            source_value = int(source) if source.strip().isdigit() else source.strip()
            placeholder = st.empty()
            metrics = st.empty()
            pipeline = StreamPipeline(model, task)

            for result in pipeline.run(
                source_value, confidence=confidence, iou=iou,
                imgsz=imgsz, tracking=tracking
            ):
                placeholder.image(result["frame"], channels="RGB", use_container_width=True)
                metrics.write(
                    f"Detections: **{result['count']}** | "
                    f"Inference: **{result['latency_ms']:.1f} ms** | "
                    f"Inference FPS: **{result['fps']:.1f}** | "
                    f"Processing FPS: **{result['processing_fps']:.1f}**"
                )
        except Exception as e:
            st.error(f"Stream error: {e}")
