# Release v1.0.0

import uvicorn
import json
import boto3
import requests
import socket
import subprocess
import os
import jwt
import threading
import time
import traceback
import cv2
import multiprocessing
import psutil
import signal
import win32gui
import win32process
import asyncio
import sys
import logging
logging.basicConfig(level=logging.DEBUG)

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, as_completed
from jwt import PyJWKClient
from typing import Union, List
from fastapi import FastAPI, Header, HTTPException, Body, BackgroundTasks, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError, RootModel, Field, validator, conlist
from io import StringIO
from jsonschema import validate
from fastapi.middleware.cors import CORSMiddleware

from launch_video_proctoring import launch_video_proctoring
from video_recording import *
from env import *

os.environ["CRYPTOGRAPHY_OPENSSL_NO_LEGACY"] = "1"

global_cheated = ""
global_list_of_path_saved = ""
global_active_port = ""
global_user_decoded_token = ""
global_s3_folder_name = ""
cap = ""
camera_id = ""
frame_width = ""
frame_height = ""
is_cleaning_up = False

# Global active server state
server_started = False
multiprocessing.set_start_method("spawn", force=True)
endpoints = FastAPI()

# Add CORS middleware
endpoints.add_middleware(
    CORSMiddleware,
    allow_origins=["https://dev.experimental.apisdevf.net", "https://localhost:5173"],  # Frontend URL
    allow_credentials=True,  # Allow sending cookies with requests
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers (Content-Type, Authorization, etc.)
)

class TestConnectionResponse(BaseModel):
    video_proctoring_ready: bool
    port_number: int

class StopEvaluationResponse(BaseModel):
    cheated: bool
    list_of_path: list[str]

class SuspiciousBehaviorRequest(BaseModel):
    suspicious_activity: str


# Dictionary to store running processes
running_processes = {}

def get_resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    import os
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def run_audio_classification(script, port, mul_stop_event):
    """
    Run the audio classification script as a subprocess.
    """
    try:
        while not mul_stop_event.is_set():
            command = ["python", script, "--port", str(port)]
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            if process.returncode == 0:
                return f"{script} is running."
            else:
                return f"{script} failed to start: {stderr.decode('utf-8')}"
    except Exception as e:
        return f"{script} encountered an exception: {str(e)}"


def run_face_landmark_detection(port, cap, cap_lock, stop_event):
    """
    Face landmark detection logic.
    """
    try:
        while not stop_event.is_set():
            # Import main from face_landmark_detection
            from face_landmark_detection import main

            print("[INFO] Running face landmark detection...")
            # with cap_lock:  # Use the lock when accessing the shared cap
            main(port=port, cap=cap)  # Call the main function with arguments
            # Add a small sleep to prevent 100% CPU usage
            time.sleep(0.1)
    except Exception as e:
        print(f"[ERROR] Exception in run_face_landmark_detection: {e}")
        



def run_object_detection(port, cap, cap_lock, stop_event):
    """
    Object detection logic.
    """
    try:
        while not stop_event.is_set():
            # with cap_lock:  # Use the lock when accessing the shared cap
            print("[INFO] Running object detection...")
            from object_detection import main
            main(port=port, cap=cap)
    except Exception as e:
        print(f"[ERROR] Exception in run_object_detection: {e}")



class VideoRecordingThread(threading.Thread):
    def __init__(self, cap, global_user_decoded_token, stop_event):
        super().__init__()
        self.cap = cap
        self.global_user_decoded_token = global_user_decoded_token
        self.stop_event = stop_event

    def run(self):
        """
        Video recording logic.
        """
        try:
            while not self.stop_event.is_set():
                print("[INFO] Starting video recording...")
                start_video_recording(self.cap, self.global_user_decoded_token)
        except Exception as e:
            print(f"[ERROR] Exception in VideoRecordingThread: {e}")
    def stopRecording(self):
        returned_value = stop_recording()
        return returned_value


##############################################
############ THREAD AND PROCESSES ############
##############################################
cap_lock = threading.Lock()
thread_stop_event = threading.Event()
multiprocessing_stop_event = multiprocessing.Event()
audio_process = ""
face_landmark_thread = ""
object_detection_thread = ""
video_recording_thread = ""
##############################################



# 2. Verify and decode the token
def verify_cognito_token(token, jwks_url):
    try:
        jwks_client = PyJWKClient(jwks_url)
        signing_key = jwks_client.get_signing_key_from_jwt(token)

        # Decode and validate the token
        decoded_token = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=GLOBAL_APP_CLIENT_ID,
            issuer=f"https://cognito-idp.us-east-1.amazonaws.com/{GLOBAL_USER_POOL_ID}"
        )
        return decoded_token

    except Exception as e:
        print(f"Token verification failed: {e}")
        return None


def find_free_port(start_port, end_port):
    """
    Find a single free port on localhost within the specified range.
    """
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            print(f"TRYING PORT {port}")
            if sock.connect_ex(('127.0.0.1', port)) != 0:  # Port is free if connect_ex returns non-zero
                print(f"LAUNCHING ON PORT {port}")
                global global_active_port
                global_active_port = port
                return port
            else:
                print(f"PORT {port} IS IN USE")
    raise RuntimeError("No free ports available in the specified range.")


def run_uvicorn_on_port(app, port):
    """
    Run the Uvicorn server on a specific port.
    """
    global server_started
    if server_started:
        print("Server is already running. Skipping additional startup.")
        return

    try:
        print(f"Trying to run server on port {port}...")
        uvicorn.run(app, host="0.0.0.0", port=port)
        server_started = True  # Mark server as started
    except Exception as e:
        print(f"Failed to launch server on port {port}: {e}")
        raise


@endpoints.get("/health")
def health() -> str:
    print("ok")
    return "OK"

@endpoints.get("/testConnection", response_model=TestConnectionResponse)
def health():
    try:
        if(global_active_port == -1):
            print("'global_active_port' WAS NOT SET")
            raise HTTPException(status_code=500, detail="'global_active_port' WAS NOT SET")
        
        response = {
                "video_proctoring_ready": True,
                "port_number": global_active_port
        }
        print("VIDEO PROCTORING READY")
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}")

@endpoints.post("/authenticateUser")
def authenticateUser(authorization: str = Header(...)):  # Require the 'Authorization' header
    print(authorization)
    JWKS_URL = f"https://cognito-idp.us-east-1.amazonaws.com/{GLOBAL_USER_POOL_ID}/.well-known/jwks.json"
    # Authenticate user
    token = authorization.split(" ", 1)[1].strip()
    decoded_token = verify_cognito_token(token, JWKS_URL)
    if decoded_token:
        print("Token is valid!")
        global global_user_decoded_token
        global_user_decoded_token = decoded_token
    else:
        print("Token is invalid!")
        raise HTTPException(status_code=401, detail="UNABLE TO AUTHENTICATE USER ON COGNITO")



import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait

@endpoints.post("/startEvaluation")
async def start_evaluation(request: Request):
    """
    Start all evaluation scripts (audio classification and video scripts) concurrently.
    Return success immediately without waiting for scripts to complete.
    """
    try:
        global global_s3_folder_name, cap, audio_process, face_landmark_thread, object_detection_thread, thread_stop_event, video_recording_thread, multiprocessing_stop_event
        global global_cheated, global_list_of_path_saved, global_active_port, global_user_decoded_token, camera_id, frame_width, frame_height, is_cleaning_up
        global_cheated = False
        global_list_of_path_saved = []
        global_user_decoded_token = ""
        global_s3_folder_name = ""
        cap = cv2.VideoCapture(0)
        camera_id = 0
        frame_width = 1920
        frame_height = 1080

        # Initialize global video capture object
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        
        multiprocessing_stop_event.clear()
        thread_stop_event.clear()
        audio_classification_path = get_resource_path("audio_classification.py")
        multiprocessing.freeze_support()
        audio_process = multiprocessing.Process(
            target=run_audio_classification, args=(audio_classification_path, global_active_port, multiprocessing_stop_event)
        )
        face_landmark_thread = threading.Thread(
            target=run_face_landmark_detection, args=(global_active_port, cap, cap_lock, thread_stop_event), daemon=True
        )
        object_detection_thread = threading.Thread(
            target=run_object_detection, args=(global_active_port, cap, cap_lock, thread_stop_event), daemon=True
        )
        # record_thread = threading.Thread(target=run_video_recording, args=(cap, global_user_decoded_token, thread_stop_event), daemon=True)
        video_recording_thread = VideoRecordingThread(cap=cap, global_user_decoded_token=global_user_decoded_token, stop_event=thread_stop_event)
        # Extract the payload
        payload = await request.json()
        auth_data = payload.get("authData", {})
        global_s3_folder_name = (
            auth_data.get("projectId")
            + "|||"
            + auth_data.get("trackId")
            + "|||"
            + auth_data.get("evalType")
        )
        # Create the shared cap object
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Start video tasks in threads to share the same cap
        video_threads = []
        time.sleep(10) #waiting for old api calls to go through
        is_cleaning_up = False
        video_threads.append(face_landmark_thread)
        video_threads.append(object_detection_thread)
        video_recording_thread.start()
        # video_threads.append(record_thread)

        

        # Start threads
        for thread in video_threads:
            print(f"[INFO] Trying to start thread: {thread.name}")
            thread.start()
            print(f"[INFO] Started thread: {thread.name}")

        print("[INFO] Trying to start audio classification process.")
        responses = {}
        audio_classification_path = get_resource_path("audio_classification/audio_classification.exe")

        try:
            # Add the --port argument for each script
            command = [audio_classification_path, "--port", str(global_active_port)]

            # Start each script as a background process
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            # Wait briefly for the process to start
            time.sleep(0.5)  # Allow process to initialize

            # Check if the process is running
            if process.poll() is None:
                running_processes[audio_classification_path] = process
                responses[audio_classification_path] = "Running"
            else:
                # Capture error if the process failed
                stderr = process.stderr.read().decode("utf-8")
                responses[audio_classification_path] = f"Failed to start: {stderr}"
        except Exception as e:
            responses[audio_classification_path] = f"Exception occurred: {str(e)}"

        # audio_process.start()
        # audio_process.join()
        print("[INFO] Started audio classification process.")

        # Return success response immediately
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Evaluation scripts have been launched."
            },
        )

    except Exception as e:
        traceback_details = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e), "details": traceback_details},
        )


def async_cleanup():
    global running_processes, cap, audio_process, face_landmark_thread, object_detection_thread, thread_stop_event, multiprocessing_stop_event, video_recording_thread
    global is_cleaning_up
    try:
        multiprocessing_stop_event.set()
        thread_stop_event.set()
        def kill_window_by_name(window_name):
            try:
                # Find the window handle
                hwnd = win32gui.FindWindow(None, window_name)
                if hwnd == 0:
                    print(f"[INFO] No window found with title: {window_name}")
                    return

                print(f"[INFO] Found window handle: {hwnd}")

                # Get the PID of the process associated with the window
                try:
                    _, pid = win32process.GetWindowThreadProcessId(hwnd)
                    print(f"[INFO] PID for window '{window_name}': {pid}")
                except Exception as e:
                    print(f"[ERROR] Failed to get PID for window '{window_name}': {e}")
                    return

                # Validate PID
                if not psutil.pid_exists(pid):
                    print(f"[WARNING] PID {pid} does not exist. Skipping termination.")
                    return

                # Kill the process using psutil
                try:
                    process = psutil.Process(pid)
                    print(f"[INFO] Attempting to terminate process PID {pid}...")
                    process.terminate()  # Attempt to terminate gracefully
                    process.wait(timeout=5)  # Wait for termination
                    if process.is_running():
                        print("[WARNING] Process did not terminate. Force killing...")
                        process.kill()  # Force kill if terminate doesn't work
                    print(f"[INFO] Process {pid} terminated successfully.")
                except psutil.NoSuchProcess:
                    print(f"[INFO] Process PID {pid} already terminated.")
                except psutil.AccessDenied:
                    print(f"[ERROR] Access denied when attempting to terminate PID {pid}.")
                except Exception as e:
                    print(f"[ERROR] Unexpected error terminating PID {pid}: {e}")
            except Exception as e:
                print(f"[ERROR] Failed to kill process for window '{window_name}': {e}")

        
        # Stop audio process
        try:
            if audio_process and audio_process.is_alive():
                print("Killing audio_process forcefully")
                audio_process.terminate()
                proc = psutil.Process(audio_process.pid)
                proc.kill()
                audio_process.join(timeout=5)  # Wait for it to stop
                kill_window_by_name("Audio classification")
                kill_window_by_name("Audio classification.exe")
        except Exception as e:
            print(f"[ERROR] Failed to terminate audio process: {e}")

        # Wait for threads to finish
        if video_recording_thread.is_alive():
            video_recording_thread.stopRecording()
        if face_landmark_thread.is_alive():
            face_landmark_thread.join(timeout=5)
        if object_detection_thread.is_alive():
            object_detection_thread.join(timeout=5)
        # kill_window_by_name("Recording")
        # kill_window_by_name("object_detection")
        # kill_window_by_name("Face Detection")
        print("Releasing cap")
        cap.release()
        print("Destroying cv2 windows")
        cv2.destroyAllWindows()
        audio_classification_path = get_resource_path("audio_classification/audio_classification.exe")
        process = running_processes[audio_classification_path]
        process.terminate()  # Terminate the process
        process.wait()       # Wait for the process to fully terminate

        running_processes.clear()
        print("Resetting threads")
        audio_process = ""
        face_landmark_thread = ""
        object_detection_thread = ""
        record_thread = ""
        print("Done cleanup")
        print("Ready")

    except Exception as e:
        print(f"Error : {e}")
        # raise HTTPException(status_code=500, detail=f"{e}")


@endpoints.get("/stopEvaluation", response_model=StopEvaluationResponse)
async def stop_evaluation(background_tasks: BackgroundTasks):
    """
    Stops all running scripts by terminating their processes.
    """
    global global_cheated, global_list_of_path_saved, video_recording_thread, is_cleaning_up 
    is_cleaning_up = True
    try:
        result = video_recording_thread.stopRecording()
        print("Stopped recording")
        if (len(result) != 0):
            global_cheated = True
        for i in result:
            global_list_of_path_saved.append(i)
        # asyncio.create_task(async_cleanup())
        os.system(f"taskkill /f /im audio_classification.exe")
        background_tasks.add_task(async_cleanup)
        return {"cheated": global_cheated, "list_of_path": global_list_of_path_saved}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}")


resource_lock = threading.Lock()

######################### VIDEO PROCTORING ENDPOINTS #########################
@endpoints.post("/suspiciousBehavior")
def suspicious_behavior(request: SuspiciousBehaviorRequest):
    """
    API to save the current 30-second video and audio buffer
    when suspicious behavior is detected.
    """
    global is_cleaning_up
    if is_cleaning_up:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Service unavailable due to system cleanup. Please try again later."},
        )
    try:
        global global_s3_folder_name
        with resource_lock:
            print(f"Received suspicious activity: {request.suspicious_activity}")
            print(f"Suspicious behavior will be saved in S3 folder: {global_s3_folder_name}")
            # Perform necessary actions here
            detected_suspicious_behavior(request.suspicious_activity, global_s3_folder_name)
            return JSONResponse(
                status_code=200,
                content={"status": "success", "message": "Suspicious behavior logged and video will be saved."},
            )
    except Exception as e:
        print(f"Error in /suspiciousBehavior: {str(e)}")
        traceback_details = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e), "details": traceback_details},
        )
######################### VIDEO PROCTORING ENDPOINTS #########################


if __name__ == "__main__":
    try:
        multiprocessing.freeze_support()  # Ensures compatibility with PyInstaller
        # Find a free port
        free_port = find_free_port(55000, 55100)

        # Start the Uvicorn server on the free port
        run_uvicorn_on_port(endpoints, free_port)
    except Exception as e:
        print(f"Error while starting the server: {e}")