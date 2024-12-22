import argparse
import sys
import time
import cv2
from cv2 import VideoCapture
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils import visualize

# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()

# List of labels considered suspicious
SUSPICIOUS_LABELS = {"cell phone", "computer", "tablet", "laptop"}


def get_resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    import os
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def run(model: str, max_results: int, score_threshold: float, 
        cap:VideoCapture, port_number: int) -> None:
    """Continuously run inference on images acquired from the camera."""
    print("2")
    print("2")
    print("2")
    print("2")
    print("2")
    print("2-Trying to start objecct deteection")
    print("2")
    print("2")
    print("2")
    print("2")
    # Visualization parameters
    row_size = 50  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 0)  # black
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    detection_result_list = []

    def save_result(result: vision.ObjectDetectorResult, unused_output_image: mp.Image, timestamp_ms: int):
        global FPS, COUNTER, START_TIME

        # Detect suspicious behavior
        suspicious_detected = any(
            detection.categories[0].category_name in SUSPICIOUS_LABELS
            for detection in result.detections
        )

        if suspicious_detected:
            print("Suspicious behavior detected")
            print(f"Posting to : http://localhost:{port_number}/suspiciousBehavior")
            try:
                import requests
                requests.post(
                    f"http://localhost:{port_number}/suspiciousBehavior",
                    json={"suspicious_activity": "Suspicious object detected"}
                )
            except Exception as e:
                print(f"Error posting suspicious behavior: {e}")

        # Calculate the FPS
        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()

        detection_result_list.append(result)
        COUNTER += 1

    # Initialize the object detection model
    base_options = python.BaseOptions(model_asset_path=get_resource_path(model))
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        max_results=max_results,
        score_threshold=score_threshold,
        result_callback=save_result
    )
    detector = vision.ObjectDetector.create_from_options(options)

    # Continuously capture images from the camera and run inference
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )

        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run object detection using the model
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        # Show the FPS
        fps_text = 'FPS = {:.1f}'.format(FPS)
        text_location = (left_margin, row_size)
        current_frame = image
        cv2.putText(current_frame, fps_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                    font_size, text_color, font_thickness, cv2.LINE_AA)

        if detection_result_list:
            current_frame = visualize(current_frame, detection_result_list[0])
            detection_result_list.clear()

        cv2.imshow('object_detection', current_frame)

        # Stop the program if the ESC key is pressed
        if cv2.waitKey(1) == 27:
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


def main(port=None, cap=None, max_results=5, score_threshold=0.25):
    """
    Main function for the object detection script.
    """
    print("")
    print("")
    print("")
    print("")
    print("")
    print("Trying to start objecct deteection")
    print("")
    print("")
    print("")
    print("")
    if cap is None:
        raise ValueError("Shared cap object must be provided.")

    # Run the object detection loop using the shared cap
    run(model="media_pipe/object_detection/efficientdet_lite0_float_32.tflite", max_results=max_results, score_threshold=score_threshold, cap=cap, port_number=port)


if __name__ == "__main__":
    main()