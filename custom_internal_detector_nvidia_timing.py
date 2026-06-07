import argparse
import time
import cv2
import numpy as np
import pyzed.sl as sl

ID_COLORS = [(232, 176, 59), (175, 208, 25), (102, 205, 105), (185, 0, 255), (99, 107, 252)]


def draw_custom_boxes(frame, objects, img_scale):
    for obj in objects.object_list:
        if obj.bounding_box_2d.shape[0] > 0:
            top_left = (int(obj.bounding_box_2d[0][0] * img_scale[0]),
                        int(obj.bounding_box_2d[0][1] * img_scale[1]))
            bottom_right = (int(obj.bounding_box_2d[2][0] * img_scale[0]),
                            int(obj.bounding_box_2d[2][1] * img_scale[1]))

            color = ID_COLORS[obj.id % 5]
            cv2.rectangle(frame, top_left, bottom_right, color, 2)

            # If object 3D position is available, compute distance
            distance = np.sqrt(obj.position[0] ** 2 + obj.position[1] ** 2 + obj.position[2] ** 2)
            label = f"Green Crab: {distance:.2f}m"

            cv2.putText(frame, label, (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip_address', type=str, default="",required=False, 
                        help='Jetson stream IP and port, e.g. 192.168.33.2:30000')
    parser.add_argument('--input_svo_file', type=str, default='',
                        help='Path to an .svo file, if you want to replay it')
    parser.add_argument('--custom_onnx', type=str,default='old_model/yolo26_s_best.onnx', required=False,
                        help='Path to ONNX model')
    parser.add_argument('--print_every', type=int, default=60,
                        help='Print timing stats every N frames (default: 60)')
    return parser.parse_args()


def main():
    opt = parse_args()

    print("Connecting to ZED stream...")
    zed = sl.Camera()

    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.NEURAL  # lighter and usually faster for streaming
    init.coordinate_units = sl.UNIT.METER

    # Source selection
    if opt.ip_address and opt.ip_address.strip():
        try:
            ip, port = opt.ip_address.split(':')
            port = int(port)
        except ValueError:
            raise ValueError("Invalid --ip_address format. Use ip:port (e.g., 192.168.33.2:30000)")
        print(f"[Receiver] Connecting to ZED stream {ip}:{port}")
        init.set_from_stream(ip, port)
    elif opt.input_svo_file and opt.input_svo_file.strip():
        print(f"[SVO] Opening SVO file {opt.input_svo_file}")
        init.set_from_svo_file(opt.input_svo_file)
    else:
        print("[Local] Opening local ZED camera")
        init.camera_resolution = sl.RESOLUTION.HD720
        init.camera_fps = 30

    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Error: Could not open source. ZED error: {status}")
        return

    # Positional tracking (required if you want tracking + stable IDs)
    print("Enabling Positional Tracking...")
    pt_params = sl.PositionalTrackingParameters()
    pt_status = zed.enable_positional_tracking(pt_params)
    if pt_status != sl.ERROR_CODE.SUCCESS:
        print(f"Warning: Positional tracking failed: {pt_status} (continuing)")

    # Object detection
    print("Enabling Object Detection...")
    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_YOLOLIKE_BOX_OBJECTS
    obj_param.custom_onnx_file = opt.custom_onnx
    obj_param.enable_tracking = True  # set False if you want lighter load / fewer dependencies

    od_status = zed.enable_object_detection(obj_param)
    if od_status != sl.ERROR_CODE.SUCCESS:
        print(f"Error enabling object detection: {od_status}")
        zed.close()
        return

    objects = sl.Objects()
    obj_runtime_param = sl.CustomObjectDetectionRuntimeParameters()
    obj_runtime_param.object_detection_properties.detection_confidence_threshold = 35

    camera_config = zed.get_camera_information().camera_configuration
    display_res = sl.Resolution(min(1280, camera_config.resolution.width), 720)
    image_scale = (display_res.width / camera_config.resolution.width,
                   display_res.height / camera_config.resolution.height)

    image_left = sl.Mat()
    GREEN_CRAB_CLASS_ID = 0

    print("Stream Connected. Starting Detection... Press 'q' to quit.")

    # -------- Timing instrumentation (GRAB vs PROCESS) --------
    t0 = time.time()
    n = 0
    grab_time_sum = 0.0
    proc_time_sum = 0.0
    total_time_sum = 0.0

    while True:
        loop_start = time.time()

        tg = time.time()
        err = zed.grab()
        grab_dt = time.time() - tg
        grab_time_sum += grab_dt

        if err != sl.ERROR_CODE.SUCCESS:
            print(f"Grab error: {err}")
            break

        tp = time.time()

        # ---- PROCESSING ----
        zed.retrieve_custom_objects(objects, obj_runtime_param)
        filtered_list = [obj for obj in objects.object_list if obj.raw_label == GREEN_CRAB_CLASS_ID]
        objects.object_list = filtered_list
        count = len(filtered_list)

        zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_res)
        frame = image_left.get_data()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        draw_custom_boxes(frame, objects, image_scale)

        cv2.rectangle(frame, (20, 20), (560, 90), (0, 0, 0), -1)
        cv2.putText(frame, f"Invasive Crabs Count: {count}", (35, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)

        cv2.imshow("ZED Stream - Green Crab Detector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # ---- END PROCESSING ----

        proc_dt = time.time() - tp
        proc_time_sum += proc_dt

        loop_dt = time.time() - loop_start
        total_time_sum += loop_dt

        n += 1
        if n % max(1, opt.print_every) == 0:
            elapsed = time.time() - t0
            fps = n / elapsed if elapsed > 0 else 0.0
            grab_avg_ms = (grab_time_sum / n) * 1000.0
            proc_avg_ms = (proc_time_sum / n) * 1000.0
            loop_avg_ms = (total_time_sum / n) * 1000.0

            # These numbers tell you if it’s streaming/decoder (grab) or compute (proc)
            print(f"[STATS] FPS={fps:.1f} | grab_avg={grab_avg_ms:.1f}ms | proc_avg={proc_avg_ms:.1f}ms | loop_avg={loop_avg_ms:.1f}ms")

    # Cleanup
    try:
        zed.disable_object_detection()
    except Exception:
        pass
    try:
        zed.disable_positional_tracking()
    except Exception:
        pass
    zed.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
