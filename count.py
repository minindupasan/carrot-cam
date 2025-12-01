#!/usr/bin/env python3
"""
Coconut counter with line-crossing detection for conveyor belt monitoring.
Counts coconuts as they cross a counting line in the top half of the frame.
"""

import argparse
import os
import cv2
from ultralytics import YOLO
import time
import numpy as np
import torch
from collections import deque, defaultdict
from pathlib import Path


class CoconutCounter:
    """Optimized YOLO detector with coconut counting via line crossing."""

    def __init__(self, model_path, conf=0.25, iou=0.45, imgsz=640, half=True,
                 count_line_position=0.5, track_buffer=30):
        """
        Initialize detector with counting functionality.

        Args:
            model_path: Path to YOLO model
            conf: Confidence threshold
            iou: IoU threshold for NMS
            imgsz: Image size for inference
            half: Use FP16 half precision
            count_line_position: Position of counting line (0.0-1.0, default 0.5 = middle of top half)
            track_buffer: Number of frames to keep track history
        """
        # Create models directory if it doesn't exist
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        # Handle auto-download for standard models
        original_model_path = model_path
        model_in_models_dir = models_dir / Path(model_path).name

        if not os.path.exists(model_path) and model_path.endswith('.pt'):
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
                model_path = str(check_path)
            else:
                print(f"Downloading '{base_name}' to models directory...")
                temp_model = YOLO(base_name)
                temp_model.save(str(check_path))
                model_path = str(check_path)
                print(f"Model saved to: {model_path}")
        elif not os.path.exists(model_path) and model_path.endswith('.engine'):
            print(f"TensorRT engine file '{model_path}' not found.")
            print("Please ensure the .engine file exists or use a .pt model instead.")
            raise FileNotFoundError(f"Engine file not found: {model_path}")

        print(f"Loading model: {model_path}")
        try:
            self.model = YOLO(model_path)
            # If model was downloaded, save it to models/ directory
            if not os.path.exists(original_model_path) and original_model_path.endswith('.pt'):
                output_path = models_dir / Path(original_model_path).name
                if not output_path.exists():
                    print(f"Saving model to: {output_path}")
                    self.model.save(str(output_path))
        except (FileNotFoundError, OSError) as e:
            print(f"\nError loading model '{model_path}': {e}")
            print("Please ensure the model name is correct or the file exists.")
            raise SystemExit(1)

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

        # Counting parameters
        self.count_line_position = count_line_position  # Position in top half (0.0-1.0)
        self.count_line_y = None  # Will be set when frame size is known
        self.total_count = 0
        self.tracked_objects = {}  # Track object positions: {track_id: [prev_y, crossed]}
        self.track_buffer = track_buffer
        self.track_history = defaultdict(lambda: deque(maxlen=track_buffer))

        # FPS tracking
        self.fps_buffer = deque(maxlen=30)

        # Colors
        self.counting_line_color = (0, 255, 255)  # Yellow line
        self.counting_zone_color = (0, 255, 0)  # Green zone
        self.coconut_color = (0, 165, 255)  # Orange for coconuts
        self.counted_color = (0, 255, 0)  # Green for counted coconuts

        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        self.box_thickness = 2

    def set_counting_line(self, frame_height):
        """Set the counting line position based on frame height."""
        # Counting line is in the top half of the frame
        top_half_height = frame_height // 2
        # Position the line within the top half
        self.count_line_y = int(top_half_height * self.count_line_position)
        print(f"Counting line set at y={self.count_line_y} (frame height: {frame_height})")

    def warmup(self, shape=(640, 640, 3)):
        """Warmup model for faster initial inference."""
        print("Warming up model...")
        dummy = np.random.randint(0, 255, shape, dtype=np.uint8)
        for _ in range(5):
            _ = self.model(dummy, imgsz=self.imgsz, half=self.half, verbose=False)
        print("Warmup complete")

    def detect_and_track(self, frame, verbose=False):
        """Run detection with tracking on single frame."""
        results = self.model.track(
            frame,
            imgsz=self.imgsz,
            half=self.half,
            verbose=verbose,
            device=0 if torch.cuda.is_available() else 'cpu',
            agnostic_nms=True,
            persist=True,  # Keep tracks across frames
            tracker='bytetrack.yaml',
        )
        return results[0]

    def check_line_crossing(self, track_id, center_y, class_name):
        """
        Check if object crossed the counting line (bottom to top).

        Args:
            track_id: Unique ID of tracked object
            center_y: Current Y coordinate of object center
            class_name: Class name of detected object

        Returns:
            bool: True if object just crossed the line
        """
        # Only count coconut_no_husk objects
        if class_name != 'coconut_no_husk':
            return False

        crossed = False

        if track_id in self.tracked_objects:
            prev_y, already_counted = self.tracked_objects[track_id]

            # Check if object crossed from bottom to top (prev_y > count_line_y and center_y <= count_line_y)
            if not already_counted and prev_y > self.count_line_y and center_y <= self.count_line_y:
                crossed = True
                self.tracked_objects[track_id] = (center_y, True)  # Mark as counted
                self.total_count += 1
            else:
                # Update position
                self.tracked_objects[track_id] = (center_y, already_counted)
        else:
            # New object, initialize tracking
            self.tracked_objects[track_id] = (center_y, False)

        return crossed

    def cleanup_old_tracks(self, current_track_ids):
        """Remove tracks that are no longer detected."""
        tracked_ids = list(self.tracked_objects.keys())
        for track_id in tracked_ids:
            if track_id not in current_track_ids:
                del self.tracked_objects[track_id]

    def draw_counting_zone(self, frame):
        """Draw the counting line and zone on the frame."""
        height, width = frame.shape[:2]

        if self.count_line_y is None:
            self.set_counting_line(height)

        # Draw top half rectangle (counting zone)
        top_half_y = height // 2
        cv2.rectangle(frame, (0, 0), (width, top_half_y),
                     self.counting_zone_color, 2)

        # Draw counting line (horizontal line)
        cv2.line(frame, (0, self.count_line_y), (width, self.count_line_y),
                self.counting_line_color, 3)

        # Add text label for counting line
        cv2.putText(frame, 'COUNTING LINE', (10, self.count_line_y - 10),
                   self.font, 0.7, self.counting_line_color, 2, cv2.LINE_AA)

        # Add text label for counting zone
        cv2.putText(frame, 'COUNTING ZONE (TOP HALF)', (10, 30),
                   self.font, 0.7, self.counting_zone_color, 2, cv2.LINE_AA)

        return frame

    def draw_detections_with_counting(self, frame, results):
        """
        Draw detections with counting visualization.

        Args:
            frame: Input frame
            results: YOLO results with tracking

        Returns:
            frame: Annotated frame
        """
        # Draw counting zone first
        frame = self.draw_counting_zone(frame)

        boxes = results.boxes

        if boxes is not None and len(boxes) > 0:
            # Extract detection data
            xyxy = boxes.xyxy.cpu().numpy().astype(np.int32)
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(np.int32)

            # Get track IDs if available
            track_ids = None
            if hasattr(boxes, 'id') and boxes.id is not None:
                track_ids = boxes.id.cpu().numpy().astype(np.int32)

            current_track_ids = set()

            # Draw all detections
            for i in range(len(boxes)):
                x1, y1, x2, y2 = xyxy[i]
                conf = confs[i]
                cls_id = cls_ids[i]
                cls_name = results.names[cls_id]

                # Calculate center point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Check line crossing if tracking is available
                just_counted = False
                track_id = None
                if track_ids is not None:
                    track_id = track_ids[i]
                    current_track_ids.add(track_id)
                    just_counted = self.check_line_crossing(track_id, center_y, cls_name)

                # Choose color based on whether it's a coconut and if it was counted
                if cls_name == 'coconut_no_husk':
                    if track_id in self.tracked_objects and self.tracked_objects[track_id][1]:
                        color = self.counted_color  # Green for already counted
                    else:
                        color = self.coconut_color  # Orange for not yet counted
                else:
                    color = (128, 128, 128)  # Gray for other objects

                # Draw bounding box (thicker if just counted)
                thickness = 4 if just_counted else self.box_thickness
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # Draw center point
                cv2.circle(frame, (center_x, center_y), 5, color, -1)

                # Draw trajectory line if tracking
                if track_id is not None:
                    self.track_history[track_id].append((center_x, center_y))
                    points = list(self.track_history[track_id])
                    if len(points) > 1:
                        for j in range(1, len(points)):
                            cv2.line(frame, points[j-1], points[j], color, 2)

                # Prepare label
                label_parts = []
                if track_id is not None:
                    label_parts.append(f"ID:{track_id}")
                label_parts.append(f"{cls_name}")
                label_parts.append(f"{conf:.2f}")

                if just_counted:
                    label_parts.append("COUNTED!")

                label = " ".join(label_parts)

                # Get text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, self.font, self.font_scale, self.font_thickness
                )

                # Draw label background
                label_y = max(y1 - 10, text_height + 10)
                cv2.rectangle(
                    frame,
                    (x1, label_y - text_height - baseline - 5),
                    (x1 + text_width + 10, label_y + baseline),
                    color,
                    -1
                )

                # Draw label text
                text_color = (0, 0, 0) if sum(color) > 384 else (255, 255, 255)
                cv2.putText(
                    frame,
                    label,
                    (x1 + 5, label_y - 5),
                    self.font,
                    self.font_scale,
                    text_color,
                    self.font_thickness,
                    cv2.LINE_AA
                )

            # Cleanup old tracks
            if track_ids is not None:
                self.cleanup_old_tracks(current_track_ids)

        return frame

    def draw_counter_display(self, frame):
        """Draw the counter display on the frame."""
        height, width = frame.shape[:2]

        # Create semi-transparent background for counter
        overlay = frame.copy()
        counter_height = 100
        cv2.rectangle(overlay, (width - 300, 0), (width, counter_height),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw counter text
        count_text = f"COUNT: {self.total_count}"
        cv2.putText(frame, count_text, (width - 280, 60),
                   self.font, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

        return frame

    def update_fps(self, fps):
        """Update FPS buffer."""
        self.fps_buffer.append(fps)

    def get_avg_fps(self):
        """Get average FPS."""
        return sum(self.fps_buffer) / len(self.fps_buffer) if self.fps_buffer else 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Coconut Counter with Line Crossing Detection')
    parser.add_argument(
        '--model',
        type=str,
        default='models/yolo11n.pt',
        help='Path to YOLO model (default: models/yolo11n.pt)'
    )
    parser.add_argument(
        '--source',
        type=str,
        default='assets/test.mp4',
        help='Path to video source or camera ID (0 for webcam)'
    )
    parser.add_argument(
        '--webcam',
        action='store_true',
        help='Use webcam (equivalent to --source 0)'
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
        help='Image size for inference (default: 640)'
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
        default='counted_output.mp4',
        help='Output video path (default: counted_output.mp4)'
    )
    parser.add_argument(
        '--no-half',
        action='store_true',
        help='Disable FP16 half precision'
    )
    parser.add_argument(
        '--count-line-position',
        type=float,
        default=0.5,
        help='Position of counting line in top half (0.0-1.0, default: 0.5 = middle of top half)'
    )
    parser.add_argument(
        '--no-warmup',
        action='store_true',
        help='Skip model warmup'
    )
    return parser.parse_args()


def main():
    """Main function for coconut counting."""
    args = parse_args()

    print("="*60)
    print("Coconut Counter - Line Crossing Detection")
    print("="*60)
    print("Counting coconut_no_husk objects moving from BOTTOM to TOP")
    print("Objects are counted when they cross the line in top half")
    print("="*60)

    # Initialize counter
    counter = CoconutCounter(
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        half=not args.no_half,
        count_line_position=args.count_line_position
    )

    # Warmup model
    if not args.no_warmup:
        counter.warmup(shape=(args.imgsz, args.imgsz, 3))

    # Handle webcam flag
    if args.webcam:
        args.source = '0'

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
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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
    print("="*60)

    # Setup video writer if saving
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Saving output to: {args.output}")

    frame_count = 0
    start_time = time.time()

    print("\nStarting counting... Press 'q' to quit")
    print("-" * 60)

    try:
        while cap.isOpened():
            frame_start = time.time()

            ret, frame = cap.read()
            if not ret:
                print("\nEnd of video or cannot read frame")
                break

            frame_count += 1

            # Run detection with tracking
            results = counter.detect_and_track(frame, verbose=False)

            # Draw detections with counting
            annotated_frame = counter.draw_detections_with_counting(frame.copy(), results)

            # Draw counter display
            annotated_frame = counter.draw_counter_display(annotated_frame)

            # Calculate and display FPS
            frame_time = time.time() - frame_start
            current_fps = 1.0 / frame_time if frame_time > 0 else 0
            counter.update_fps(current_fps)
            avg_fps = counter.get_avg_fps()

            cv2.putText(annotated_frame, f'FPS: {avg_fps:.1f}', (10, height - 20),
                       counter.font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            # Display progress
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                det_count = len(results.boxes) if results.boxes is not None else 0
                print(f"Frame {frame_count} | FPS: {avg_fps:.1f} | "
                      f"Detections: {det_count} | Total Count: {counter.total_count}")

            # Show frame if requested
            if args.show:
                cv2.imshow('Coconut Counter', annotated_frame)
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
        avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        print("-" * 60)
        print(f"\nCounting complete!")
        print(f"Total frames processed: {frame_count}")
        print(f"Total time: {elapsed_time:.2f}s")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"\n*** TOTAL COCONUTS COUNTED: {counter.total_count} ***")
        print("="*60)

        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
