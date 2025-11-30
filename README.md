# Carrot Cam 🥕📷

Highly optimized YOLOv11n real-time object detection system designed for NVIDIA Jetson and GPU-accelerated environments. This project focuses on maximum FPS and low-latency visualization using TensorRT and optimized OpenCV drawing.

## Features

- 🚀 **Ultra-Fast Inference**: Uses TensorRT (FP16) for maximum throughput on NVIDIA GPUs.
- ⚡ **Optimized Visualization**: Custom drawing routines that are faster than standard library calls.
- 📊 **Performance Metrics**: Real-time FPS tracking (Inference vs Total System FPS).
- 🛠 **Easy Export**: Built-in tools to convert YOLOv11 models to TensorRT engines.
- 📹 **Flexible Input**: Supports webcams, video files, and RTSP streams.

## Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (Jetson Orin/Nano or Desktop GPU)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- OpenCV
- PyTorch (with CUDA)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/carrot-cam.git
   cd carrot-cam
   ```

2. Install dependencies:
   ```bash
   pip install ultralytics opencv-python torch numpy
   ```
   *Note: On Jetson, ensure you have the appropriate JetPack libraries installed.*

## Usage

### 1. Export Model to TensorRT
First, export the YOLOv11n model to a TensorRT engine for maximum speed.

```bash
# Export with default settings (640x640)
python3 export.py

# Export with custom size (e.g., 416x416 for higher FPS)
python3 export.py --imgsz 416 --name yolo11n_416
```

### 2. Run Detection
Run the detection script using the exported engine.

```bash
# Run on webcam (source 0)
python3 main.py --model models/yolo11n_640.engine --source 0

# Run on video file
python3 main.py --model models/yolo11n_640.engine --source assets/test.mp4

# Run with visualization disabled (for pure performance testing)
python3 main.py --model models/yolo11n_640.engine --source 0 --no-show
```

## Configuration Options

### `main.py` Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `models/yolo11n.engine` | Path to the model file |
| `--source` | `assets/test.mp4` | Video source (0 for webcam, path for file) |
| `--imgsz` | `640` | Inference size (must match exported model) |
| `--conf` | `0.25` | Confidence threshold |
| `--show` | `False` | Show video output |
| `--save` | `False` | Save output to `output.mp4` |
| `--no-half` | `False` | Disable FP16 precision |

### `export.py` Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--imgsz` | `640` | Input image size for the engine |
| `--workspace` | `4` | GPU workspace size in GB |
| `--name` | `None` | Custom output name for the engine |

## License

[MIT](LICENSE)
