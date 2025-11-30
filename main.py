#!/usr/bin/env python3
"""
Highly optimized YOLOv11n detection for Jetson with maximum FPS
Ultra-fast visualization with thin orange boxes and labels
"""

import argparse
import cv2
from ultralytics import YOLO
import time
import numpy as np
import torch
from collections import deque


class OptimizedDetector:
    """Optimized YOLO detector with GPU acceleration and ultra-fast rendering."""

    def __init__(self, model_path, conf=0.25, iou=0.45, imgsz=640, half=True):
        """Initialize detector with optimizations."""
        print(f"Loading model: {model_path}")
        self.model = YOLO(model_path)

        # Set model parameters
        self.model.conf = conf
        self.model.iou = iou
        self.imgsz = imgsz
        self.half = half and torch.cuda.is_available()

        # Enable CUDA optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print(f"CUDA enabled: {torch.cuda.get_device_name(0)}")
            print(f"Half precision (FP16): {self.half}")

        # FPS tracking
        self.fps_buffer = deque(maxlen=30)

        # Orange color for bounding boxes (BGR format)
        self.box_color = (0, 165, 255)  # Orange

        # Pre-compute font settings for speed
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thickness = 1

    def warmup(self, shape=(640, 640, 3)):
        """Warmup model for faster initial inference."""
        print("Warming up model...")
        dummy = np.random.randint(0, 255, shape, dtype=np.uint8)
        for _ in range(5):
            _ = self.model(dummy, imgsz=self.imgsz, half=self.half, verbose=False)
        print("Warmup complete")

    def detect(self, frame, verbose=False):
        """Run optimized detection on single frame."""
        results = self.model(
            frame,
            imgsz=self.imgsz,
            half=self.half,
            verbose=verbose,
            device=0 if torch.cuda.is_available() else 'cpu',
            agnostic_nms=True,  # Faster NMS
        )
        return results[0]

    def detect_batch(self, frames, verbose=False):
        """Run detection on batch of frames for higher throughput."""
        results = self.model(
            frames,
            imgsz=self.imgsz,
            half=self.half,
            verbose=verbose,
            device=0 if torch.cuda.is_available() else 'cpu',
            agnostic_nms=True,
        )
        return results

    def draw_detections_fast(self, frame, results):
        """
        Ultra-fast drawing with thin orange boxes and labels.
        No background, direct rendering for maximum speed.

        Args:
            frame: Input frame (will be modified in-place for speed)
            results: YOLO results object

        Returns:
            frame: Modified frame with visualizations
        """
        boxes = results.boxes

        if boxes is not None and len(boxes) > 0:
            # Batch extract all data at once (faster than per-box access)
            xyxy = boxes.xyxy.cpu().numpy().astype(np.int32)
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(np.int32)

            # Draw all detections
            for i in range(len(boxes)):
                x1, y1, x2, y2 = xyxy[i]
                conf = confs[i]
                cls_id = cls_ids[i]
                cls_name = results.names[cls_id]

                # Draw thin orange bounding box (thickness=1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.box_color, 2)

                # Prepare label
                label = f"{cls_name} {conf:.2f}"

                # Draw label text directly (no background for speed)
                text_y = max(y1 - 5, 15)
                cv2.putText(
                    frame,
                    label,
                    (x1 + 2, text_y),
                    self.font,
                    self.font_scale,
                    self.box_color,  # Orange text
                    self.font_thickness,
                    cv2.LINE_AA
                )

        return frame

    def update_fps(self, fps):
        """Update FPS buffer."""
        self.fps_buffer.append(fps)

    def get_avg_fps(self):
        """Get average FPS."""
        return sum(self.fps_buffer) / len(self.fps_buffer) if self.fps_buffer else 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Optimized YOLOv11n Real-time Detection')
    parser.add_argument(
        '--model',
        type=str,
        default='models/yolo11n.engine',
        help='Path to YOLO model (default: models/yolo11n.engine)'
    )
    parser.add_argument(
        '--source',
        type=str,
        default='assets/test.mp4',
        help='Path to video source or camera ID (0 for webcam)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='IoU threshold for NMS (default: 0.45)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size for inference (default: 640). Try 416 or 480 for higher FPS'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display video output (default: False)'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save output video (default: False)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output.mp4',
        help='Output video path (default: output.mp4)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for processing (default: 1). Higher = more throughput but more latency'
    )
    parser.add_argument(
        '--no-half',
        action='store_true',
        help='Disable FP16 half precision'
    )
    parser.add_argument(
        '--vid-stride',
        type=int,
        default=1,
        help='Video frame-rate stride (process every Nth frame, default: 1)'
    )
    parser.add_argument(
        '--tracker',
        type=str,
        default=None,
        choices=['bytetrack', 'botsort'],
        help='Enable object tracking (bytetrack or botsort)'
    )
    parser.add_argument(
        '--no-warmup',
        action='store_true',
        help='Skip model warmup'
    )
    return parser.parse_args()


def main():
    """Main function for optimized real-time detection."""
    args = parse_args()

    print("="*60)
    print("YOLOv11n Ultra-Optimized Real-time Detection")
    print("="*60)

    # Initialize detector
    detector = OptimizedDetector(
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        half=not args.no_half
    )

    # Warmup model
    if not args.no_warmup:
        detector.warmup(shape=(args.imgsz, args.imgsz, 3))

    # Open video source
    try:
        source = int(args.source)
        print(f"Opening camera {source}")
        is_camera = True
    except ValueError:
        source = args.source
        print(f"Opening video file: {source}")
        is_camera = False

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {args.source}")
        return

    # Optimize capture settings for camera
    if is_camera:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {width}x{height} @ {fps} FPS")
    if total_frames > 0:
        print(f"Total frames: {total_frames}")
    print(f"Inference size: {args.imgsz}x{args.imgsz}")
    print(f"Batch size: {args.batch_size}")
    print(f"Frame stride: {args.vid_stride}")
    print("="*60)

    # Setup video writer if saving
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Saving output to: {args.output}")

    frame_count = 0
    processed_count = 0
    start_time = time.time()

    print("\nStarting detection... Press 'q' to quit")
    print("-" * 60)

    try:
        while cap.isOpened():
            frame_start = time.time()

            ret, frame = cap.read()
            if not ret:
                print("\nEnd of video or cannot read frame")
                break

            frame_count += 1

            # Frame stride handling (process every Nth frame)
            if args.vid_stride > 1 and frame_count % args.vid_stride != 0:
                continue

            processed_count += 1

            # Run detection
            if args.tracker:
                # Use tracking for better object continuity
                results = detector.model.track(
                    frame,
                    imgsz=args.imgsz,
                    half=detector.half,
                    verbose=False,
                    tracker=f"{args.tracker}.yaml",
                    persist=True,
                    agnostic_nms=True,
                )[0]
            else:
                results = detector.detect(frame, verbose=False)

            # Draw detections (in-place modification for speed)
            annotated_frame = detector.draw_detections_fast(frame, results)

            # Calculate FPS
            frame_time = time.time() - frame_start
            current_fps = 1.0 / frame_time if frame_time > 0 else 0
            detector.update_fps(current_fps)
            avg_fps = detector.get_avg_fps()

            # Add info overlay (minimal for speed)
            cv2.putText(annotated_frame, f'FPS: {avg_fps:.1f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            det_count = len(results.boxes) if results.boxes is not None else 0
            cv2.putText(annotated_frame, f'Det: {det_count}', (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Display progress
            if processed_count % 30 == 0:
                elapsed = time.time() - start_time
                avg_total_fps = processed_count / elapsed if elapsed > 0 else 0
                print(f"Frame {frame_count} | Processed: {processed_count} | "
                      f"Inference FPS: {avg_fps:.1f} | Total FPS: {avg_total_fps:.1f} | "
                      f"Detections: {det_count}")

            # Show frame if requested
            if args.show:
                cv2.imshow('YOLOv11n Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nUser interrupted")
                    break

            # Save frame if requested
            if writer is not None:
                writer.write(annotated_frame)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        # Cleanup
        elapsed_time = time.time() - start_time
        avg_fps = processed_count / elapsed_time if elapsed_time > 0 else 0
        inference_fps = detector.get_avg_fps()

        print("-" * 60)
        print(f"\nProcessing complete!")
        print(f"Total frames: {frame_count}")
        print(f"Processed frames: {processed_count}")
        print(f"Total time: {elapsed_time:.2f}s")
        print(f"Overall FPS: {avg_fps:.1f}")
        print(f"Inference FPS: {inference_fps:.1f}")
        print("="*60)

        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
