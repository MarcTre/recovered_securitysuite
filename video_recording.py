import os
import glob
import threading
import time
import subprocess
import boto3
import cv2

from env import *

# Global variables
running = True  # Control flag for recording thread
record_event = threading.Event()  # Event for triggering video saving
stop_event = threading.Event()  # Event to stop recording
output_file = "latest_clip.avi"  # Output video file name
saved_clips = []  # List of saved clips
recording_started = False  # Flag to ensure recording starts only after the API call
global_vr_list_of_path_saved = []
cap = ""


# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=global_aws_access_key_id,
    aws_secret_access_key=global_aws_secret_access_key,
    region_name=global_region_name,
)

# Define your S3 bucket name
S3_BUCKET_NAME = "videoproctoring"
folder_name = ""

def save_suspicious_segment_to_s3(output_file):
    """
    Save the current video segment to an S3 bucket when suspicious behavior is detected.
    """
    try:
        # Generate a unique key (file name) for the S3 object
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        s3_key = f"{folder_name}/{timestamp}_{output_file}"
        print(f"S3 Folder nam : {folder_name})
        global global_vr_list_of_path_saved
        global_vr_list_of_path_saved.append(f"{S3_BUCKET_NAME}/{s3_key}")

        print(f"Uploading {output_file} to S3 as {s3_key}...")

        # Upload the file to S3
        s3_client.upload_file(output_file, S3_BUCKET_NAME, s3_key)

        print(f"Uploaded {output_file} to S3 bucket {S3_BUCKET_NAME} with key {s3_key}.")
    except FileNotFoundError:
        print(f"Segment not found for uploading: {output_file}")
    except Exception as e:
        print(f"Error uploading segment {output_file} to S3: {str(e)}")



save_lock = threading.Lock()  # Lock to synchronize recording and saving

def record_video(cap):
    """
    Continuously record video using OpenCV, ensuring files are not overwritten during saving to S3.
    """
    global running, output_file

    if not cap.isOpened():
        print("[ERROR] Failed to open webcam.")
        return

    video_writer = None  # Initialize VideoWriter object

    try:
        while running:
            # Create a new video file
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            print(f"Starting new OpenCV recording: {output_file}")

            # Initialize the video writer
            # Video settings
            codec = cv2.VideoWriter_fourcc(*"XVID")  # Use 'mp4v' codec for MP4 format
            fps = 20  # Frames per second
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_writer = cv2.VideoWriter(output_file, codec, fps, (frame_width, frame_height))

            start_time = time.time()
            while running and (time.time() - start_time < 30):  # Record for 30 seconds
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Failed to read frame from webcam.")
                    break

                with save_lock:  # Ensure saving doesn't interfere with writing
                    video_writer.write(frame)

                # Display the frame (optional)
                cv2.imshow("Recording", frame)
                if cv2.waitKey(1) == 27:  # ESC to stop recording
                    running = False
                    break

            # Handle suspicious events
            if record_event.is_set():
                with save_lock:  # Prevent recording while saving to S3
                    print("Calling : save_suspicious_segment_to_s3")
                    save_suspicious_segment_to_s3(output_file)
                record_event.clear()
                print(f"[INFO] Saved suspicious clip: {output_file}")

            # Release the video writer for the current segment
            if video_writer:
                video_writer.release()
                video_writer = None

            # Stop if stop_event is set
            if stop_event.is_set():
                print("[INFO] Stopping video recording...")
                break

    finally:
        print("Stopping recording...")
        if video_writer:
            video_writer.release()
        print("[INFO] Released resources.")


def detected_suspicious_behavior(suspicious_activity: str, global_s3_folder_name: str):
    """
    Trigger event to save the current buffer when suspicious behavior is detected.
    """
    global record_event, folder_name
    folder_name = global_s3_folder_name
    try:
        print(f"Suspicious behavior detected: {suspicious_activity}")
        # Trigger event to save the video
        record_event.set()
        return {"status": "success", "message": "Suspicious behavior logged and video will be saved."}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    

def stop_recording():
    """
    Stop the recording thread.
    """
    global running, stop_event, cap, global_vr_list_of_path_saved, recording_started
    stop_event.set()
    running = False
    recording_started = False
    print("[INFO] Stopped recording.")
    return global_vr_list_of_path_saved


def start_video_recording(given_cap, global_user_decoded_token):
    """
    Starts the video recording thread if it hasn't been started already.
    """
    global running, record_thread, recording_started, cap, stop_event
    stop_event.clear()
    cap = given_cap
    recording_started = False
    if not recording_started:
        recording_started = True
        running = True
        record_video(cap)
        print("[INFO] Video recording started.")

