import argparse
import requests
import sys
import time
import threading
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import keyboard  # For Windows-compatible keyboard input handling
import sys

# Mediapipe utilities
mp_face_mesh = mp.solutions.face_mesh

# Global state variables
look_away_frame_count = 0
LOOK_DOWN_THRESHOLD = 0.6
LOOK_UP_THRESHOLD = 0.25
LOOK_AWAY_THRESHOLD = 0.4
LOOK_AWAY_FLAG_THRESHOLD = 5
input_detected = False
DETECTION_RESULT = ""

# Thread control events
stop_event = threading.Event()
resume_event = threading.Event()

def get_resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    import os
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

################################################
###########       KEYBOARD HANDLING     #######
################################################
def wait_for_input(global_port_number, timeout=10):
    """Wait for keyboard input for a given timeout (in seconds) in a separate thread."""
    global input_detected

    input_detected = False  # Reset input detection flag

    def on_key_event(event):
        """Keyboard event callback."""
        global input_detected
        input_detected = True

    def timeout_function():
        """Timeout function to handle no input scenario."""
        time.sleep(timeout)
        if not input_detected:
            print("No keyboard input detected. Raising suspicious behavior.")
            try:
                requests.post(
                    f"http://localhost:{global_port_number}/suspiciousBehavior",
                    json={"suspicious_activity": "Cheating Alert: User is looking down for too long and not typing"}
                )
            except Exception as e:
                print(f"Error posting suspicious behavior: {e}")

    # Start listening for keyboard input
    keyboard.hook(on_key_event)

    # Run the timeout in a separate thread
    timeout_thread = threading.Thread(target=timeout_function)
    timeout_thread.start()

    # Wait for input or timeout
    while not input_detected and timeout_thread.is_alive():
        time.sleep(0.1)

    # Cleanup
    keyboard.unhook_all()
    stop_event.clear()
    resume_event.set()



################################################
###########       CHEATING DETECTION     ######
################################################


def is_looking_away(face_blendshapes, global_port_number):
    """Check if the user is looking away or down and handle cheating detection."""
    global look_away_frame_count

    try:
        # Check for looking up, down, or sideways
        eye_up = face_blendshapes.get('eyeLookUpRight', 0) > LOOK_UP_THRESHOLD or \
                 face_blendshapes.get('eyeLookUpLeft', 0) > LOOK_UP_THRESHOLD
        eye_down = face_blendshapes.get('eyeLookDownRight', 0) > LOOK_DOWN_THRESHOLD or \
                   face_blendshapes.get('eyeLookDownLeft', 0) > LOOK_DOWN_THRESHOLD
        eye_sideways = face_blendshapes.get('eyeLookInLeft', 0) > LOOK_AWAY_THRESHOLD or \
                       face_blendshapes.get('eyeLookInRight', 0) > LOOK_AWAY_THRESHOLD

        if eye_up or eye_sideways:
            look_away_frame_count += 1
            if look_away_frame_count >= LOOK_AWAY_FLAG_THRESHOLD:
                print("Suspicious behavior: looking away or up.")
                try:
                    requests.post(
                        f"http://localhost:{global_port_number}/suspiciousBehavior",
                        json={"suspicious_activity": "Cheating Alert: User is looking sideways or up for too long"}
                    )
                except Exception as e:
                    print(f"Error posting suspicious behavior: {e}")
                look_away_frame_count = 0

        elif eye_down:
            look_away_frame_count += 1
            if look_away_frame_count >= LOOK_AWAY_FLAG_THRESHOLD:
                print("User is looking down. Checking for keyboard input...")
                look_away_frame_count = 0
                stop_event.set()  # Pause face detection

                # Run wait_for_input in a separate thread
                input_thread = threading.Thread(target=wait_for_input, args=(global_port_number, 10))
                input_thread.start()

                # Add a cooldown to prevent immediate retriggering
                time.sleep(2)

        else:
            look_away_frame_count = 0

    except Exception as e:
        print(f"Error in is_looking_away(): {e}")




################################################
###########       FACE DETECTION LOOP      #####
################################################
def run_face_detection(args, cap, port):
    """Run the face detection loop."""
    print("[INFO] Initializing Mediapipe FaceLandmarker...")

    try:
        base_options = python.BaseOptions(
            model_asset_path=get_resource_path("media_pipe/face_landmark_detection/face_landmarker.task")
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_faces=args.numFaces,
            min_face_detection_confidence=args.minFaceDetectionConfidence,
            min_face_presence_confidence=args.minFacePresenceConfidence,
            min_tracking_confidence=args.minTrackingConfidence,
            output_face_blendshapes=True,
            result_callback=lambda result, _, __: process_landmarks(result, port, cap, args)
        )
        global detector
        detector = vision.FaceLandmarker.create_from_options(options)
        print("[INFO] Mediapipe FaceLandmarker initialized successfully.")

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("[ERROR] Failed to capture frame. Exiting loop.")
                break

            if image is None or image.size == 0:
                print("[ERROR] Captured frame is invalid. Exiting loop.")
                break

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

            detector.detect_async(mp_image, time.time_ns() // 1_000_000)

            cv2.imshow("Face Detection", image)
            if cv2.waitKey(1) == 27:  # ESC to quit
                print("[INFO] Exiting detection loop.")
                break

    except Exception as e:
        print(f"[ERROR] Exception during face detection: {e}")
    finally:
        print("[INFO] Releasing resources...")
        if 'detector' in globals():
            detector.close()
        # cap.release()
        # cv2.destroyAllWindows()
        print("[INFO] Resources released.")






def process_landmarks(result, port_number, cap, args):
    """Callback function for processing landmarks."""
    global DETECTION_RESULT
    DETECTION_RESULT = result
    if DETECTION_RESULT and DETECTION_RESULT.face_blendshapes:
        face_blendshapes = {category.category_name: category.score for category in DETECTION_RESULT.face_blendshapes[0]}
        is_looking_away(face_blendshapes, port_number)



################################################
###############         MAIN         ###########
################################################
def main(port=None, cap=None, num_faces=1, min_face_detection_confidence=0.5, 
         min_face_presence_confidence=0.5, min_tracking_confidence=0.5):
    """
    Main function for the face landmark detection script.
    """
    print(f"[INFO] Starting face landmark detection with port={port}...")
    
    if cap is None:
        raise ValueError("[ERROR] Shared cap object is None. Exiting...")

    # Log cap properties
    print(f"[INFO] cap.isOpened() = {cap.isOpened()}")

    # Construct arguments object
    class Args:
        def __init__(self):
            self.numFaces = num_faces
            self.minFaceDetectionConfidence = min_face_detection_confidence
            self.minFacePresenceConfidence = min_face_presence_confidence
            self.minTrackingConfidence = min_tracking_confidence
            self.port = port

    args = Args()

    print("[INFO] Starting face detection loop...")
    run_face_detection(args, cap, args.port)
    print("[INFO] Exiting main function.")




if __name__ == "__main__":
    main()