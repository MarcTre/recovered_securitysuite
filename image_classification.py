# Install MediaPipe if you haven't already
# !pip install mediapipe opencv-python-headless

import mediapipe as mp
import cv2
import numpy as np

def main_image_classification():
# Initialize the Image Classifier
    classifier = mp.tasks.vision.ImageClassifier.create_from_model_path(
        # model_path="./media_pipe/image_classification/efficientnet_lite2_float_32.tflite"  # Replace with the path to your downloaded model
        model_path="./media_pipe/object_detection/efficientdet_lite0_float_32.tflite"
    )

    # Start video capture from USB webcam (usually source 0)
    cap = cv2.VideoCapture(1)  # Replace '0' with your camera source if different

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the frame to RGB as required by MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert frame to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Perform the classification
        classification_result = classifier.classify(mp_image)

        # Display the classifications
        for idx, category in enumerate(classification_result.classifications[0].categories):
            text = f"{category.category_name}: {category.score:.2f}"
            cv2.putText(frame, text, (10, 30 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame with the classifications overlay
        cv2.imshow("USB Webcam Classification", frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
  main_image_classification()