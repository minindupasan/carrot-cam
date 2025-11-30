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


def export_tensorrt(imgsz=640, model_path="yolo11n.pt", output_dir="models",
                    output_name=None, workspace=4):
    """
    Export YOLO model to TensorRT engine.

    Args:
        imgsz: Input image size (320, 416, 480, 640, etc.)
        model_path: Path to PyTorch model
        output_dir: Directory to save the engine
        output_name: Custom name for output engine (without .engine extension)
        workspace: GPU workspace in GB
    """

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. TensorRT requires CUDA.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("YOLOv11n TensorRT Export")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Image size: {imgsz}x{imgsz}")
    print(f"Output directory: {output_dir}")
    print(f"Workspace: {workspace}GB")
    print("="*60)

    # Load model
    print("\nLoading PyTorch model...")
    model = YOLO(model_path)

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

        # Rename to custom name if provided
        if output_name:
            # Ensure .engine extension
            if not output_name.endswith('.engine'):
                output_name += '.engine'

            # Create full path
            custom_path = os.path.join(output_dir, output_name)

            # Move/rename the exported engine
            if os.path.exists(engine_path):
                shutil.move(engine_path, custom_path)
                engine_path = custom_path
                print(f"\nRenamed to: {custom_path}")
        else:
            # Move to output directory with default name
            default_name = f"yolo11n_{imgsz}.engine"
            custom_path = os.path.join(output_dir, default_name)
            if os.path.exists(engine_path) and engine_path != custom_path:
                shutil.move(engine_path, custom_path)
                engine_path = custom_path

        print("\n" + "="*60)
        print("Export Complete!")
        print(f"TensorRT engine: {engine_path}")
        print("="*60)
        print("\nUsage:")
        print(f"  python3 main.py --model {engine_path} --source assets/test1.mp4 --show --imgsz {imgsz}")
        print("="*60)

        return engine_path

    except Exception as e:
        print(f"\nERROR during export: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure CUDA is properly installed")
        print("2. Check GPU memory (need ~2-4GB free)")
        print("3. Try with --workspace 2 for less memory")
        return None


def main():
    parser = argparse.ArgumentParser(description='Export YOLOv11n to TensorRT')
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
        help='Path to PyTorch model (default: yolo11n.pt)'
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
        help='Custom output name (e.g., yolo11n_416). Auto: yolo11n_{imgsz}.engine'
    )
    parser.add_argument(
        '--workspace',
        type=int,
        default=4,
        help='GPU workspace in GB (default: 4)'
    )

    args = parser.parse_args()

    export_tensorrt(
        imgsz=args.imgsz,
        model_path=args.model,
        output_dir=args.output_dir,
        output_name=args.name,
        workspace=args.workspace
    )


if __name__ == '__main__':
    main()
