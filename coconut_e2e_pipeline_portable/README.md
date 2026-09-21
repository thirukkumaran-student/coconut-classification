# Coconut Plantation Monitoring — End-to-End Streamlit Pipeline

## Supported modules

- Aerial Tree Detection
- Coconut Counting
- Maturity & Variety Classification
- Disease Detection

## Inputs

- Image
- Video
- Live stream / webcam / network capture URL

## Models

Place your existing model folders under `models/`:

```text
models/
├── aerial tree detection/
│   ├── yolov8model.pt
│   └── yolov11model.pt
├── coconut counting/
│   ├── yolov8model.pt
│   ├── yolov11model.pt
│   └── fasterrcnn_coconut_epoch10.pth
├── maturity variety classification/
│   ├── maturity1.pt
│   └── maturity2.pt
└── disease classification/
    ├── disease1.pt
    └── disease2.pt
```

The application automatically discovers `.pt` and `.pth` files.

## Run

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

## Two-computer workflow

### Home PC — RTX 3050

Use:

```text
Device = auto
```

or:

```text
Device = cuda
```

Use the PC for model development, training, benchmarking and GPU performance measurements.

### College laptop — CPU only

Use:

```text
Device = cpu
```

Use smaller inference sizes such as 320 or 416 if live inference is too slow.

The laptop can still run image/video inference and can act as the computer receiving a camera/network stream.

## Drone/live stream

The Streamlit application does not assume a specific drone transport protocol.

If your drone/capture setup exposes a stream URL that OpenCV/FFmpeg can read, enter that URL in the Live Stream source box.

For example:

```text
rtsp://...
```

or another supported capture source.

Do not claim "real-time UAV inference" in the paper until you measure the actual capture -> decode -> preprocess -> inference -> postprocess -> display path.

## Counting note

A detection count is a per-frame count.

It is NOT automatically a unique count over an entire UAV flight.

Tracking can maintain object IDs across frames, but unique counting over a moving UAV scene requires careful tracking/scene design and should be validated before being reported as a plantation-level total.

## Performance measurement

The dashboard reports:

- model inference latency
- inference FPS
- end-to-end processing FPS
- source FPS
- frame number
- detection count

For the paper, measure all models on the same machine, input size, confidence configuration and software environment.
