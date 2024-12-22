### PYINSTALLER BUILD COMMANDS ###

# audio_classification.py
pyinstaller --onefile --add-data "./media_pipe/audio_classification/yamnet.tflite:./media_pipe/audio_classification" audio_classification.py

# object_detection.py
pyinstaller --onefile --add-data "./media_pipe/object_detection/efficientdet_lite0_int_8.tflite:./media_pipe/object_detection" object_detection.py

# face_landmark_detection.py
pyinstaller --onefile --add-data "./media_pipe/face_landmark_detection/face_landmarker.task:./media_pipe/face_landmark_detection" face_landmark_detection.py

# Windows - endpoints.py
pyinstaller ^
    --onefile ^
    --hidden-import=uvicorn ^
    --hidden-import=json ^
    --hidden-import=boto3 ^
    --hidden-import socket ^
    --hidden-import subprocess ^
    --hidden-import os ^
    --hidden-import jwt ^
    --hidden-import threading ^
    --hidden-import time ^
    --hidden-import traceback ^
    --hidden-import cv2 ^
    --hidden-import multiprocessing ^
    --hidden-import psutil ^
    --hidden-import signal ^
    --hidden-import win32gui ^
    --hidden-import win32process ^
    --hidden-import asyncio ^
    --hidden-import sys ^
    --hidden-import logging ^
    --hidden-import concurrent ^
    --hidden-import typing ^
    --hidden-import fastapi ^
    --hidden-import pydantic ^
    --hidden-import io ^
    --hidden-import jsonschema ^
    --add-data "./media_pipe/object_detection/efficientdet_lite0_float_32.tflite:./media_pipe/object_detection" ^
    --add-data "./media_pipe/audio_classification/yamnet.tflite:./media_pipe/audio_classification" ^
    --add-data "./media_pipe/face_landmark_detection/face_landmarker.task:./media_pipe/face_landmark_detection" ^
    --add-data "audio_classification.py:." ^
    --add-data "face_landmark_detection.py:." ^
    --add-data "object_detection.py:." ^
    --add-data "./audio_classification/audio_classification.exe:./audio_classification" ^
    endpoints.py

    --debug=all ^

s-main\dist\media_pipe\face_landmark_detection\face_landmarker.task
[INFO] Releasing resources...
[INFO] Resources released.
[INFO] Exiting main function.
[INFO] Running face landmark detection...
[INFO] Starting face landmark detection with port=55000...
[INFO] cap.isOpened() = True
[INFO] Starting face detection loop...
[INFO] Initializing Mediapipe FaceLandmarker...
[ERROR] Exception during face detection: Unable to open file at E:\Documents\securitysuite_macos-main\securitysuite_macos-main\dist\media_pipe\face_landmark_detection\face_landmarker.task
[INFO] Releasing resources...
[INFO] Resources released.
[INFO] Exiting main function.
[INFO] Running face landmark detection...
[INFO] Starting face landmark detection with port=55000...
[INFO] cap.isOpened() = True
[INFO] Starting face detection loop...
[INFO] Initializing Mediapipe FaceLandmarker...
[ERROR] Exception during face detection: Unable to open file at E:\Documents\securitysuite_macos-main\securitysuite_macos-main\dist\media_pipe\face_landmark_detection\face_landmarker.task
[INFO] Releasing resources...
[INFO] Resources released.
[INFO] Exiting main function.



ken verification failed: Signature has expired
Token is invalid!
←[32mINFO←[0m:     127.0.0.1:59637 - "←[1mPOST /authenticateUser HTTP/1.1←[0m" ←[31m401 Unauthorized←[0m
[INFO] Starting video recording...
Here 1
Here 2
Here 2
Here 3
Starting new OpenCV recording: latest_clip.avi
[INFO] Started thread: Thread-1 (run_face_landmark_detection)
[INFO] Running object detection...
[INFO] Started thread: Thread-2 (run_object_detection)
[INFO] Started audio classification process.
←[32mINFO←[0m:     127.0.0.1:59639 - "←[1mPOST /startEvaluation HTTP/1.1←[0m" ←[32m200 OK←[0m
[ERROR] Exception in run_object_detection: Unable to open file at C:\Users\marcf\AppData\Local\Temp\_MEI118642\media_pipe\object_detection\efficientdet_lite0_float_32.tflite
[INFO] Running face landmark detection...
[INFO] Starting face landmark detection with port=55000...
[INFO] cap.isOpened() = True
[INFO] Starting face detection loop...
[INFO] Initializing Mediapipe FaceLandmarker...
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1733695987.046390    2068 face_landmarker_graph.cc:174] Sets FaceBlendshapesGraph acceleration to xnnpack by default.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1733695987.075365   15072 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1733695987.102230   15072 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
[INFO] Mediapipe FaceLandmarker initialized successfully.
TRYING PORT 55000
PORT 55000 IS IN USE
TRYING PORT 55001
LAUNCHING ON PORT 55001
Trying to run server on port 55001...
←[32mINFO←[0m:     Started server process [←[36m8104←[0m]
←[32mINFO←[0m:     Waiting for application startup.
←[32mINFO←[0m:     Application startup complete.
←[32mINFO←[0m:     Uvicorn running on ←[1mhttp://0.0.0.0:55001←[0m (Press CTRL+C to quit)





TRYING PORT 55000
LAUNCHING ON PORT 55000
Trying to run server on port 55000...
←[32mINFO←[0m:     Started server process [←[36m13260←[0m]
←[32mINFO←[0m:     Waiting for application startup.
←[32mINFO←[0m:     Application startup complete.
←[32mINFO←[0m:     Uvicorn running on ←[1mhttp://0.0.0.0:55000←[0m (Press CTRL+C to quit)
bearer eyJraWQiOiJoNlRrTlhuZE1pYWFKZDlWdXE1QW9EZVRvS1l3Rk16TVUweHA2V1BPV1RJPSIsImFsZyI6IlJTMjU2In0.eyJzdWIiOiIwZGI0NjdiYS03MTg4LTQ3MGQtYjUzMy1mZDk2OWVmMDU2YzciLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiaXNzIjoiaHR0cHM6XC9cL2NvZ25pdG8taWRwLnVzLWVhc3QtMS5hbWF6b25hd3MuY29tXC91cy1lYXN0LTFfUk83MUVpQm5FIiwicGhvbmVfbnVtYmVyX3ZlcmlmaWVkIjpmYWxzZSwiY29nbml0bzp1c2VybmFtZSI6ImplYW5wMTQiLCJvcmlnaW5fanRpIjoiNzkyNDE3NTMtMjc2Mi00NjcyLTkwOTEtNDRlZGM0NmExOTJhIiwiYXVkIjoiNXJvNWg5bzVrN3U2bDFicDZsZW42cGRocTEiLCJldmVudF9pZCI6ImM0MjM2MzE1LTY4YzgtNGQyMC1hOGE0LTEzYTAyMTczYTFlMyIsInRva2VuX3VzZSI6ImlkIiwiYXV0aF90aW1lIjoxNzMzNjgzNTExLCJuYW1lIjoiSmVhbnAxNCIsInBob25lX251bWJlciI6IisyNTQ3MTIzNDU2NzgiLCJleHAiOjE3MzM2ODcxMTAsImlhdCI6MTczMzY4MzUxMSwianRpIjoiYWJmMGE3ZWEtZTMzNC00YTZhLThmODgtM2JhMmU5YTQ0N2FjIiwiZW1haWwiOiJqZWFuLXBhbXBoaWxlQHRoZWRldmZhY3RvcnkuY2EifQ.rkw-eaJTD9ifnegNBm5o-jQbOQhwxaz0WWr31SZYpO3egSqshb3cAUQZw-8WjzCLGfBHsxKEBGogSCFyiodXpxoY0K0pGiyWM5ol9kJ_8eSdh7SZWFn4cKYL0Q7oj61bq47GZEXFwjLrKlglQ-pPh85clbV_dhgVR47uWa7ce8C3nf5UWum_tyFUqrL90j5bmq_paugD7epLVsYtmxbtqUbWNeuFSmTzZxF_SNiGL9muWbFDf9Mq3naydeuCkt5NGRleMVU25Pdfd2GcCaez3HdAvNBsCstuiMxMBly2528vd8ZTT9Aj5sMCIXD_RsGURHUdh4U9d3qp5S_SoChvEg
Token verification failed: Signature has expired
Token is invalid!
←[32mINFO←[0m:     127.0.0.1:54744 - "←[1mPOST /authenticateUser HTTP/1.1←[0m" ←[31m401 Unauthorized←[0m
[INFO] Starting video recording...
[INFO] Trying to start thread: Thread-1 (run_face_landmark_detection)
Here 1
Here 2
Here 2
Here 3
[INFO] Started thread: Thread-1 (run_face_landmark_detection)
[INFO] Trying to start thread: Thread-2 (run_object_detection)
Starting new OpenCV recording: latest_clip.avi
[INFO] Running object detection...
[INFO] Started thread: Thread-2 (run_object_detection)
[INFO] Trying to start audio classification process.
[INFO] Started audio classification process.
←[32mINFO←[0m:     127.0.0.1:54750 - "←[1mPOST /startEvaluation HTTP/1.1←[0m" ←[32m200 OK←[0m
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
[INFO] Running face landmark detection...
[INFO] Starting face landmark detection with port=55000...
[INFO] cap.isOpened() = True
[INFO] Starting face detection loop...
[INFO] Initializing Mediapipe FaceLandmarker...
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1733698311.198804    6448 face_landmarker_graph.cc:174] Sets FaceBlendshapesGraph acceleration to xnnpack by default.
W0000 00:00:1733698311.243150   13136 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1733698311.275656   11540 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
[INFO] Mediapipe FaceLandmarker initialized successfully.
TRYING PORT 55000
PORT 55000 IS IN USE
TRYING PORT 55001
LAUNCHING ON PORT 55001
Trying to run server on port 55001...
←[32mINFO←[0m:     Started server process [←[36m12132←[0m]
←[32mINFO←[0m:     Waiting for application startup.
←[32mINFO←[0m:     Application startup complete.
←[32mINFO←[0m:     Uvicorn running on ←[1mhttp://0.0.0.0:55001←[0m (Press CTRL+C to quit)
Here 3
Starting new OpenCV recording: latest_clip.avi