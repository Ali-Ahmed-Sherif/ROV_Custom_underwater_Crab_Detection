import argparse
import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
from datetime import datetime


DEFAULT_RTSP_URL = os.getenv("CAMERA_RTSP_URL", "")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run European Green Crab detection on an RTSP stream or a local MP4 file."
    )

    parser.add_argument(
        "--mp4",
        type=str,
        default=None,
        help="Path to an MP4 video file. If omitted, the script uses the RTSP/GStreamer stream.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path for saving the annotated output MP4. Useful for offline MP4 processing.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="old_model/yolo26_s_best.onnx",
        help="Path to the YOLO model file.",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        #default="camera_calibration_data_66.npz",
        help="Path to the camera calibration .npz file.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.45,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--target-class-id",
        type=int,
        default=0,
        help="Class ID to count and draw.",
    )
    parser.add_argument(
        "--rtsp-url",
        type=str,
        default=DEFAULT_RTSP_URL,
        help="RTSP URL used when --mp4 is not provided. Can also be set with CAMERA_RTSP_URL.",
    )
    parser.add_argument(
        "--stream-width",
        type=int,
        default=1280,
        help="Width requested from the RTSP/GStreamer stream.",
    )
    parser.add_argument(
        "--stream-height",
        type=int,
        default=720,
        help="Height requested from the RTSP/GStreamer stream.",
    )
    parser.add_argument(
        "--screenshots-dir",
        type=str,
        default="screenshots",
        help="Directory for screenshots saved with the s key.",
    )
    parser.add_argument(
        "--recordings-dir",
        type=str,
        default="mp4_recordings",
        help="Directory for manual recordings toggled with the p key.",
    )
    parser.add_argument(
        "--recording-fps",
        type=float,
        default=None,
        help="FPS for output/manual recordings. Defaults to the source FPS when available, otherwise 20.",
    )
    parser.add_argument(
        "--undistort",
        action="store_true",
        help="Apply calibration-based undistortion before inference.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without cv2.imshow. Keyboard controls are disabled in this mode.",
    )

    return parser.parse_args()


def load_calibration(calibration_file):
    try:
        data = np.load(calibration_file)

        print("[DEBUG] Keys:", data.files)

        camera_matrix = data["mtx"]
        dist_coeffs = data["dist"]

        print("Calibration loaded successfully")
        print("Camera Matrix:\n", camera_matrix)
        print("Distortion Coefficients:\n", dist_coeffs)

        if camera_matrix.shape != (3, 3):
            print("[WARNING] Camera matrix is not 3x3")

        if dist_coeffs.size < 4:
            print("[WARNING] Distortion coefficients too small")

        return camera_matrix, dist_coeffs

    except Exception as e:
        print(f"[ERROR] Could not load calibration file '{calibration_file}': {e}")
        return None, None


def undistort_frame(frame, camera_matrix, dist_coeffs):
    """Undistort frame using calibration parameters."""
    if camera_matrix is None or dist_coeffs is None:
        return frame

    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )

    undistorted = cv2.undistort(
        frame, camera_matrix, dist_coeffs, None, new_camera_matrix
    )

    x, y, w_roi, h_roi = roi
    if w_roi > 0 and h_roi > 0:
        undistorted = undistorted[y : y + h_roi, x : x + w_roi]

    return undistorted


def stable_color_from_name(name):
    """Generate a consistent color for a given class name."""
    hash_val = hash(name)
    np.random.seed(hash_val & 0xFFFFFFFF)
    color = tuple(np.random.randint(0, 255, 3).tolist())
    np.random.seed()
    return color


def draw_label(frame, x, y, text, color):
    """Draw label with colored background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    cv2.rectangle(
        frame,
        (x, y - text_height - baseline - 5),
        (x + text_width, y),
        color,
        -1,
    )

    cv2.putText(
        frame,
        text,
        (x, y - baseline - 2),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
    )


def draw_detections_and_count(
    frame, result, class_names, conf_thres=0.25, target_class_id=0
):
    """
    Draw bounding boxes only for the target class.
    Returns count of detections.
    """
    target_count = 0

    if result.boxes is None or len(result.boxes) == 0:
        return frame, target_count

    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
        if k != target_class_id:
            continue

        if c < conf_thres:
            continue

        target_count += 1

        label = class_names[k] if (class_names and k < len(class_names)) else str(k)
        color = stable_color_from_name(label)

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {c:.2f}"
        draw_label(frame, x1, y1, text, color)

    return frame, target_count


def draw_count_overlay(frame, count):
    """Draw a professional-looking count overlay in top-left corner."""
    padding = 20
    box_height = 80
    box_width = 280

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (padding, padding),
        (padding + box_width, padding + box_height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.rectangle(
        frame,
        (padding, padding),
        (padding + box_width, padding + box_height),
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        "European Green Crabs",
        (padding + 10, padding + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Count: {count}",
        (padding + 10, padding + 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_video_writer(output_path, frame_width, frame_height, fps=20.0):
    """
    Create MP4 writer.
    Tries mp4v codec first, which is usually the safest with OpenCV.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    if not writer.isOpened():
        print(f"[ERROR] Could not open VideoWriter for: {output_path}")
        return None

    return writer


def build_gst_pipeline(rtsp_url, width, height):
    return (
        f"rtspsrc latency=0 location={rtsp_url} ! "
        "queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! "
        "decodebin ! videoconvert ! videoscale ! "
        f"video/x-raw,width={width},height={height} ! "
        "appsink max-buffers=1 drop=true sync=false"
    )


def open_video_source(args):
    """
    Open either a local MP4 file or the default RTSP/GStreamer stream.
    Returns: cap, source_name, is_file_input
    """
    if args.mp4:
        if not os.path.isfile(args.mp4):
            print(f"[ERROR] MP4 file does not exist: {args.mp4}")
            return None, None, True

        print(f"Opening MP4 file: {args.mp4}")
        
        # 1. Initialize the capture object FIRST
        cap = cv2.VideoCapture(args.mp4)
        
        if not cap.isOpened():
            print(f"[ERROR] Failed to open MP4 file: {args.mp4}")
            return None, None, True

        # 2. Extract dimensions using OpenCV properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Original Video Resolution: {width}x{height}")
        
        return cap, args.mp4, True

    if not args.rtsp_url:
        print("[ERROR] No RTSP URL provided. Use --rtsp-url or set CAMERA_RTSP_URL.")
        return None, None, False
        
    # Add your RTSP handling logic here if needed...


def get_source_fps(cap, fallback=20.0):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or fps != fps:
        return fallback
    return fps


def main():
    args = parse_args()

    camera_matrix, dist_coeffs = load_calibration(args.calibration)

    os.makedirs(args.screenshots_dir, exist_ok=True)
    os.makedirs(args.recordings_dir, exist_ok=True)

    print("Loading YOLO model...")
    model = YOLO(args.model, task="detect")

    class_names = ["European Green crab"]

    cap, source_name, is_file_input = open_video_source(args)
    if cap is None or not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source_name}")
        return

    source_fps = get_source_fps(cap)
    recording_fps = args.recording_fps if args.recording_fps is not None else source_fps
    wait_delay_ms = max(1, int(1000 / source_fps)) if is_file_input else 1

    print(f"Video source opened: {source_name}")
    print(f"Source FPS: {source_fps:.2f}")
    print("Starting detection...")

    if args.no_display:
        print("Display disabled. Keyboard controls are not available.")
    else:
        print("Controls:")
        print("  s -> save screenshot")
        print("  p -> start/stop MP4 recording")
        print("  q -> quit")

    fps_time = time.time()
    frame_count = 0
    total_frames_processed = 0


    is_recording = False
    manual_video_writer = None
    manual_recording_path = None

    output_writer = None

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                if is_file_input:
                    print("[INFO] End of MP4 file reached")
                else:
                    print("[WARNING] Failed to grab frame from stream")
                break

            if args.undistort:
                frame = undistort_frame(frame, camera_matrix, dist_coeffs)

            results = model(frame, verbose=False)

            annotated_frame, crab_count = draw_detections_and_count(
                frame.copy(),
                results[0],
                class_names,
                conf_thres=args.conf,
                target_class_id=args.target_class_id,
            )

            draw_count_overlay(annotated_frame, crab_count)

            if is_recording:
                cv2.circle(
                    annotated_frame,
                    (annotated_frame.shape[1] - 30, 30),
                    10,
                    (0, 0, 255),
                    -1,
                )
                cv2.putText(
                    annotated_frame,
                    "REC",
                    (annotated_frame.shape[1] - 80, 37),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            if args.output and output_writer is None:
                h, w = annotated_frame.shape[:2]
                output_writer = create_video_writer(args.output, w, h, recording_fps)
                if output_writer is None:
                    print("[ERROR] Output writer could not be created. Continuing without saving output.")
                    args.output = None
                else:
                    print(f"[INFO] Saving annotated output to: {args.output}")

            if output_writer is not None:
                output_writer.write(annotated_frame)

            total_frames_processed += 1
            frame_count += 1
            if frame_count >= 30:
                fps = frame_count / (time.time() - fps_time)
                fps_time = time.time()
                frame_count = 0
                print(
                    f"Processing FPS: {fps:.2f} | "
                    f"Frame: {total_frames_processed} | "
                    f"European Green Crabs detected: {crab_count}"
                )

            if manual_video_writer is not None:
                manual_video_writer.write(annotated_frame)
            resized_frame = cv2.resize(annotated_frame, (args.stream_width, args.stream_height))
            if not args.no_display:
                cv2.imshow("Crab Detection", resized_frame)

                key = cv2.waitKey(wait_delay_ms) & 0xFF

                if key == ord("s"):
                    screenshot_name = f"screenshot_{get_timestamp()}.jpg"
                    screenshot_path = os.path.join(args.screenshots_dir, screenshot_name)
                    ok = cv2.imwrite(screenshot_path, annotated_frame)
                    if ok:
                        print(f"[INFO] Screenshot saved: {screenshot_path}")
                    else:
                        print(f"[ERROR] Failed to save screenshot: {screenshot_path}")

                elif key == ord("p"):
                    if not is_recording:
                        h, w = annotated_frame.shape[:2]
                        recording_name = f"recording_{get_timestamp()}.mp4"
                        manual_recording_path = os.path.join(args.recordings_dir, recording_name)

                        manual_video_writer = create_video_writer(
                            manual_recording_path, w, h, recording_fps
                        )

                        if manual_video_writer is not None:
                            is_recording = True
                            print(f"[INFO] Recording started: {manual_recording_path}")
                        else:
                            print("[ERROR] Recording could not start")
                    else:
                        is_recording = False
                        if manual_video_writer is not None:
                            manual_video_writer.release()
                            manual_video_writer = None
                        print(f"[INFO] Recording stopped: {manual_recording_path}")

                elif key == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nStopping detection...")

    finally:
        if manual_video_writer is not None:
            manual_video_writer.release()
            print(f"[INFO] Recording finalized: {manual_recording_path}")

        if output_writer is not None:
            output_writer.release()
            print(f"[INFO] Output finalized: {args.output}")

        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        print("Video source closed")


if __name__ == "__main__":
    main()
