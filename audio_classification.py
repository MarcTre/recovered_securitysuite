import argparse
import time
import requests
from mediapipe.tasks import python
from mediapipe.tasks.python.audio.core import audio_record
from mediapipe.tasks.python.components import containers
from mediapipe.tasks.python import audio
from utils_audio_classification import Plotter
import sys
import os

def get_resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def run(model: str, max_results: int, score_threshold: float,
        overlapping_factor: float, port_number: int) -> None:
    """Continuously run inference on audio data acquired from the device.

    Args:
        model: Name of the TFLite audio classification model.
        max_results: Maximum number of classification results to display.
        score_threshold: The score threshold of classification results.
        overlapping_factor: Target overlapping between adjacent inferences.
        port_number: Port to send suspicious behavior reports.
    """

    if (overlapping_factor < 0) or (overlapping_factor >= 1.0):
        raise ValueError('Overlapping factor must be between 0.0 and 0.9')

    if (score_threshold < 0) or (score_threshold > 1.0):
        raise ValueError('Score threshold must be between (inclusive) 0 and 1.')

    # Initialize a plotter instance to display the classification results.
    plotter = Plotter()
    classification_result_list = []

    def save_result(result: audio.AudioClassifierResult, timestamp_ms: int):
        result.timestamp_ms = timestamp_ms
        classification_result_list.append(result)

        # Check for suspicious behavior with a score over 0.5
        for classification in result.classifications:
            for category in classification.categories:
                if category.category_name in ['Speech', 'Whispering'] and category.score > 0.5:
                    print(f"Suspicious behavior detected: {category.category_name} (score: {category.score:.2f})")
                    try:
                        requests.post(
                            f"http://localhost:{port_number}/suspiciousBehavior",
                            json={"suspicious_activity": f"Detected: {category.category_name}, Score: {category.score:.2f}"}
                        )
                    except Exception as e:
                        print(f"Error posting suspicious behavior: {e}")

    # Initialize the audio classification model.
    base_options = python.BaseOptions(model_asset_path=get_resource_path(model))
    options = audio.AudioClassifierOptions(
        base_options=base_options, running_mode=audio.RunningMode.AUDIO_STREAM,
        max_results=max_results, score_threshold=score_threshold,
        result_callback=save_result)
    classifier = audio.AudioClassifier.create_from_options(options)

    # Initialize the audio recorder and a tensor to store the audio input.
    buffer_size, sample_rate, num_channels = 15600, 16000, 1
    audio_format = containers.AudioDataFormat(num_channels, sample_rate)
    record = audio_record.AudioRecord(num_channels, sample_rate, buffer_size)
    audio_data = containers.AudioData(buffer_size, audio_format)

    input_length_in_second = float(len(audio_data.buffer)) / audio_data.audio_format.sample_rate
    interval_between_inference = input_length_in_second * (1 - overlapping_factor)
    pause_time = interval_between_inference * 0.1
    last_inference_time = time.time()

    # Start audio recording in the background.
    record.start_recording()

    # Loop until the user closes the classification results plot.
    while True:
        now = time.time()
        diff = now - last_inference_time
        if diff < interval_between_inference:
            time.sleep(pause_time)
            continue
        last_inference_time = now

        # Load the input audio from the AudioRecord instance and run classify.
        data = record.read(buffer_size)
        audio_data.load_from_array(data)
        classifier.classify_async(audio_data, time.time_ns() // 1_000_000)

        # Plot the classification results.
        if classification_result_list:
            plotter.plot(classification_result_list[0])
            classification_result_list.clear()


def main():
    parser = argparse.ArgumentParser(description="Audio Classification Script")
    parser.add_argument("--model", help="Name of the audio classification model.", default="media_pipe/audio_classification/yamnet.tflite")
    parser.add_argument("--maxResults", help="Maximum number of results to show.", type=int, default=5)
    parser.add_argument("--overlappingFactor", help="Target overlapping between adjacent inferences (0-1).", type=float, default=0.5)
    parser.add_argument("--scoreThreshold", help="The score threshold of classification results.", type=float, default=0.0)
    parser.add_argument("--port", help="Port number for posting suspicious behavior.", type=int, required=True)

    args = parser.parse_args()
    print(f"Model path : {args.model}")
    run(
        model=args.model,
        max_results=args.maxResults,
        score_threshold=args.scoreThreshold,
        overlapping_factor=args.overlappingFactor,
        port_number=args.port
    )


if __name__ == "__main__":
    main()