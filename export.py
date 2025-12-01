#!/usr/bin/env python3
"""
Export YOLOv11n model to TensorRT engine with custom image size and name
Usage: python3 export_model.py --imgsz 416 --name yolo11n_416
"""

import argparse
from ultralytics import YOLO
import torch
import os
import shutil
from pathlib import Path


def get_or_download_model(model_path, models_dir="models"):
    """
    Get model from local path or download it to models/ directory.

    Args:
        model_path: Path or name of the model
        models_dir: Directory to save downloaded models

    Returns:
        Path to the model file
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(exist_ok=True)

    # If file exists at given path, use it
    if os.path.exists(model_path):
        print(f"Using existing model: {model_path}")
        return model_path

    # If it's a .pt file that doesn't exist, try to download
    if model_path.endswith('.pt'):
        print(f"Model file '{model_path}' not found locally.")

        # Fix common naming confusion: yolov11 -> yolo11
        base_name = os.path.basename(model_path)
        if base_name.startswith('yolov11'):
            corrected_name = base_name.replace('yolov11', 'yolo11')
            print(f"Correcting model name from '{base_name}' to '{corrected_name}' for auto-download...")
            base_name = corrected_name

        # Check if model already exists in models/ directory
        check_path = models_dir / base_name
        if check_path.exists():
            print(f"Found existing model in models directory: {check_path}")
            return str(check_path)

        # Download the model
        print(f"Downloading '{base_name}' to models directory...")
        try:
            # Download using just the model name
            temp_model = YOLO(base_name)

            # Save to models/ directory
            output_path = models_dir / base_name
            temp_model.save(str(output_path))
            print(f"Model downloaded and saved to: {output_path}")
            return str(output_path)

        except Exception as e:
            print(f"Error downloading model: {e}")
            print(f"Please ensure '{base_name}' is a valid YOLO model name.")
            print("Valid names: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt")
            raise

    # If we get here, file doesn't exist and isn't downloadable
    print(f"Error: Model file '{model_path}' not found and cannot be downloaded.")
    print("Please provide a valid model path or name.")
    raise FileNotFoundError(f"Model not found: {model_path}")


def export_tensorrt(imgsz=640, model_path="yolo11n.pt", output_dir="models",
                    output_name=None, workspace=4):
    """
    Export YOLO model to TensorRT engine.

    Args:
        imgsz: Input image size (320, 416, 480, 640, etc.)
        model_path: Path to PyTorch model or model name
        output_dir: Directory to save the engine
        output_name: Custom name for output engine (without .engine extension)
        workspace: GPU workspace in GB
    """

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. TensorRT requires CUDA.")
        print("TensorRT export requires a CUDA-enabled GPU.")
        return

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("="*60)
    print("YOLOv11 TensorRT Export")
    print("="*60)
    print(f"Requested model: {model_path}")
    print(f"Image size: {imgsz}x{imgsz}")
    print(f"Output directory: {output_dir}")
    print(f"Workspace: {workspace}GB")
    print("="*60)

    # Get or download the model
    try:
        model_path = get_or_download_model(model_path, output_dir)
    except Exception as e:
        print(f"\nFailed to load model: {e}")
        return None

    # Load model
    print("\nLoading PyTorch model...")
    try:
        model = YOLO(model_path)
        print(f"Model loaded successfully: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Export to TensorRT
    print("\nExporting to TensorRT engine...")
    print("This may take several minutes on first run...")
    print("GPU will be utilized during export...\n")

    try:
        engine_path = model.export(
            format="engine",           # TensorRT format
            device=0,                  # CUDA device 0
            half=True,                 # FP16 for speed
            imgsz=imgsz,              # Image size
            workspace=workspace,       # GPU workspace
            simplify=True,             # Simplify ONNX model
            verbose=True,              # Show export details
        )

        # Determine final output path
        if output_name:
            # Ensure .engine extension
            if not output_name.endswith('.engine'):
                output_name += '.engine'
            final_path = output_dir / output_name
        else:
            # Create default name based on model and image size
            model_base = Path(model_path).stem  # e.g., 'yolo11n' from 'yolo11n.pt'
            default_name = f"{model_base}_{imgsz}.engine"
            final_path = output_dir / default_name

        # Move/rename the exported engine to final location
        if os.path.exists(engine_path):
            # Convert engine_path to Path object for comparison
            engine_path_obj = Path(engine_path)

            # Only move if paths are different
            if engine_path_obj.resolve() != final_path.resolve():
                if final_path.exists():
                    final_path.unlink()  # Remove existing file
                shutil.move(str(engine_path), str(final_path))
                print(f"\nMoved to: {final_path}")
            else:
                print(f"\nEngine already at: {final_path}")

            engine_path = str(final_path)

        print("\n" + "="*60)
        print("Export Complete!")
        print(f"TensorRT engine: {engine_path}")
        print(f"File size: {os.path.getsize(engine_path) / (1024*1024):.1f} MB")
        print("="*60)
        print("\nUsage:")
        print(f"  python3 yolo_optimized_detector.py --model {engine_path} --source assets/test1.mp4 --show --imgsz {imgsz}")
        print("\nWebcam:")
        print(f"  python3 yolo_optimized_detector.py --model {engine_path} --webcam --show --imgsz {imgsz}")
        print("\nImage:")
        print(f"  python3 yolo_optimized_detector.py --model {engine_path} --source image.jpg --show --save --imgsz {imgsz}")
        print("="*60)

        return engine_path

    except Exception as e:
        print(f"\nERROR during export: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure CUDA is properly installed")
        print("2. Check GPU memory (need ~2-4GB free)")
        print("3. Try with --workspace 2 for less memory")
        print("4. Ensure you have TensorRT installed (pip install tensorrt)")
        print("5. Check CUDA version compatibility with PyTorch")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Export YOLO models to TensorRT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export yolo11n with default settings (640x640)
  python3 export_model.py

  # Export with custom image size (faster inference)
  python3 export_model.py --imgsz 416 --name yolo11n_416

  # Export larger model
  python3 export_model.py --model yolo11s.pt --imgsz 640

  # Export with custom name and less GPU memory
  python3 export_model.py --imgsz 320 --name fast_detector --workspace 2

Available YOLO11 models (auto-download):
  - yolo11n.pt (fastest, smallest)
  - yolo11s.pt (fast, small)
  - yolo11m.pt (balanced)
  - yolo11l.pt (accurate, large)
  - yolo11x.pt (most accurate, largest)
        """
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        choices=[320, 416, 480, 640, 800, 1280],
        help='Input image size (default: 640). Common: 320(fastest), 416(fast), 640(balanced)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolo11n.pt',
        help='Path to PyTorch model or model name (default: yolo11n.pt). Will auto-download if not found.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Output directory for engine (default: models/)'
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Custom output name (e.g., yolo11n_416). Auto: {model}_{imgsz}.engine'
    )
    parser.add_argument(
        '--workspace',
        type=int,
        default=4,
        help='GPU workspace in GB (default: 4)'
    )

    args = parser.parse_args()

    # Validate image size
    if args.imgsz not in [320, 416, 480, 640, 800, 1280]:
        print(f"Warning: Unusual image size {args.imgsz}. Common sizes: 320, 416, 480, 640")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Export cancelled.")
            return

    result = export_tensorrt(
        imgsz=args.imgsz,
        model_path=args.model,
        output_dir=args.output_dir,
        output_name=args.name,
        workspace=args.workspace
    )

    if result:
        print("\n✓ Export successful!")
    else:
        print("\n✗ Export failed!")
        exit(1)


if __name__ == '__main__':
    main()
