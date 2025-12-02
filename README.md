# Carrot Cam: Optimized Real-time Object Detection 🥕📷

Carrot Cam is a highly optimized YOLOv11n real-time object detection system designed for NVIDIA Jetson and GPU-accelerated environments. This project prioritizes maximum frames per second (FPS) and low-latency visualization, leveraging TensorRT for inference acceleration and optimized OpenCV routines for efficient drawing.

## Features

*   **Ultra-Fast Inference**: Achieves maximum throughput on NVIDIA GPUs using TensorRT (FP16 precision).
*   **Optimized Visualization**: Employs custom drawing routines that are significantly faster than standard library calls, ensuring minimal overhead.
*   **Performance Metrics**: Provides real-time tracking of FPS, differentiating between inference speed and total system performance.
*   **Easy Model Export**: Includes built-in tools to seamlessly convert YOLOv11 PyTorch models into highly efficient TensorRT engines.
*   **Flexible Input Sources**: Supports various input types, including live webcams, pre-recorded video files, and RTSP network streams.
*   **Object Tracking**: Optional integration of advanced tracking algorithms like ByteTrack and BoT-SORT for persistent object identification across frames.

## Prerequisites

Ensure your system meets the following requirements:

*   **Python**: Version 3.8 or newer.
*   **NVIDIA GPU**: An NVIDIA GPU with CUDA support is essential (e.g., Jetson Orin/Nano, or a desktop GPU).
*   **CUDA Toolkit**: Properly installed and configured for your NVIDIA GPU.
*   **TensorRT**: NVIDIA TensorRT library for high-performance deep learning inference.
*   **PyTorch**: PyTorch with CUDA support.
*   **Ultralytics YOLO**: The Ultralytics library, which provides the YOLO models.
*   **OpenCV**: OpenCV library for video processing and visualization.

## Installation

Follow these steps to set up Carrot Cam on your system:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/carrot-cam.git
    cd carrot-cam
    ```

2.  **Install dependencies**:
    ```bash
    pip install ultralytics opencv-python torch numpy
    ```
    *Note: For NVIDIA Jetson devices, ensure you have the appropriate JetPack libraries and pre-built `torch` wheels installed for optimal compatibility.*

## Usage Guide

### 1. Export Model to TensorRT Engine

For maximum performance, convert your YOLOv11 PyTorch model to a TensorRT engine. This step compiles the model for highly optimized execution on NVIDIA GPUs.

```bash
# Export with default settings (yolo11n model, 640x640 image size)
python3 export.py

# Export a different model (e.g., yolo11s) with a custom size (e.g., 416x416 for higher FPS)
python3 export.py --model yolo11s.pt --imgsz 416 --name yolo11s_416

# Export with custom name and less GPU memory (e.g., for Jetson Nano)
python3 export.py --imgsz 320 --name fast_detector --workspace 2
```
*   **Recommendation**: Choose an `imgsz` that balances accuracy and speed for your specific application. Smaller sizes generally lead to higher FPS.
*   **Available YOLOv11 Models (Auto-Download)**: `yolo11n.pt` (fastest, smallest), `yolo11s.pt` (fast, small), `yolo11m.pt` (balanced), `yolo11l.pt` (accurate, large), `yolo11x.pt` (most accurate, largest).

### 2. Run Real-time Object Detection

Once you have an exported TensorRT engine (or if you prefer to use a PyTorch `.pt` model directly), you can run the detection script.

```bash
# Run detection using a webcam (source 0) with a TensorRT engine
python3 main.py --model models/yolo11n_640.engine --source 0 --show

# Run detection on a video file
python3 main.py --model models/yolo11n_640.engine --source assets/test.mp4 --show --save

# Run with visualization disabled (for pure performance testing or headless systems)
python3 main.py --model models/yolo11n_640.engine --source 0 --no-show

# Process an image file and save the result
python3 main.py --model models/yolo11n_640.engine --source assets/image.jpg --show --save

# Run with object tracking enabled (requires 'bytetrack.yaml' or 'botsort.yaml' tracker config)
python3 main.py --model models/yolo11n_640.engine --source 0 --show --tracker bytetrack

# Using a PyTorch .pt model directly (will be slower than TensorRT engine)
python3 main.py --model yolo11n.pt --source 0 --show
```
*   **Note**: If you provide a `.pt` model that is not found locally, `main.py` will attempt to auto-download it to the `models/` directory.

## Configuration Options

This section details all available command-line arguments for both `export.py` and `main.py`.

### `export.py` Arguments

| Argument      | Type    | Default        | Description                                                                 |
| :------------ | :------ | :------------- | :-------------------------------------------------------------------------- |
| `--imgsz`     | `int`   | `640`          | Input image size for the TensorRT engine (e.g., 320, 416, 640).               |
| `--model`     | `str`   | `yolo11n.pt`   | Path to a PyTorch model (`.pt`) or a model name to auto-download.           |
| `--output-dir`| `str`   | `models`       | Directory where the TensorRT engine file will be saved.                     |
| `--name`      | `str`   | `None`         | Custom output filename for the engine (e.g., `my_model_416`). Auto-generated if `None`. |
| `--workspace` | `int`   | `4`            | GPU workspace size in GB allocated for TensorRT engine building.             |

### `main.py` Arguments

| Argument      | Type      | Default            | Description                                                                 |
| :------------ | :-------- | :----------------- | :-------------------------------------------------------------------------- |
| `--model`     | `str`     | `models/yolo11n.pt`| Path to the YOLO model file (`.pt` or `.engine`).                           |
| `--source`    | `str`     | `assets/test.mp4`  | Video input source (e.g., `0` for webcam, `path/to/video.mp4`, `rtsp://...`). Also supports image paths. |
| `--webcam`    | `flag`    | `False`            | Use the default webcam (shorthand for `--source 0`).                       |
| `--conf`      | `float`   | `0.25`             | Confidence threshold for object detection. Detections below this are discarded. |
| `--iou`       | `float`   | `0.45`             | IoU (Intersection over Union) threshold for Non-Maximum Suppression (NMS).  |
| `--imgsz`     | `int`     | `640`              | Image size for inference. Must match the size used during model export if using an `.engine` file. |
| `--show`      | `flag`    | `False`            | Display the video output window with detections.                            |
| `--save`      | `flag`    | `False`            | Save the output video or image to a file.                                   |
| `--output`    | `str`     | `output.mp4`       | Path and filename for the saved output video or image.                     |
| `--batch-size`| `int`     | `1`                | Number of frames to process simultaneously. Higher values may increase throughput but also latency. |
| `--no-half`   | `flag`    | `False`            | Disable FP16 half-precision inference, forcing FP32.                        |
| `--vid-stride`| `int`     | `1`                | Process every Nth frame of the video. `1` processes all frames.             |
| `--tracker`   | `str`     | `None`             | Enable object tracking. Options: `bytetrack`, `botsort`. Requires corresponding `.yaml` config. |
| `--no-warmup` | `flag`    | `False`            | Skip the model warmup phase at startup.                                     |

## Troubleshooting

*   **CUDA not available / TensorRT errors**:
    *   Ensure your NVIDIA GPU drivers are up to date.
    *   Verify CUDA Toolkit and TensorRT are correctly installed and configured.
    *   Check PyTorch version compatibility with your CUDA installation.
    *   Restart your system after installation of drivers/libraries.

*   **Model loading errors**:
    *   Double-check the `--model` path.
    *   Ensure the model name is correct (e.g., `yolo11n.pt`, not `yolov11n.pt` for auto-download).
    *   If downloading, ensure you have an active internet connection.

*   **Performance issues**:
    *   Try a smaller `--imgsz` (e.g., 416 or 320) in both `export.py` and `main.py`.
    *   Reduce `--conf` and `--iou` thresholds if necessary.
    *   Ensure FP16 is enabled (i.e., do not use `--no-half`).
    *   For `export.py`, you can try reducing `--workspace` if experiencing out-of-memory errors on GPU.

## License

This project is licensed under the [MIT License](LICENSE).