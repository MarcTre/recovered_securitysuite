import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

def run_command(command):
    """
    Runs a Unix command and captures its output.
    """
    try:
        # Run the command
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        return f"Command: {command}\nOutput: {result.stdout}\n"
    except subprocess.CalledProcessError as e:
        return f"Command: {command}\nError: {e.stderr}\n"

def run_commands_in_parallel(commands):
    """
    Runs a list of commands in parallel.
    """
    results = []
    # Use ThreadPoolExecutor to manage parallel execution
    with ThreadPoolExecutor(max_workers=len(commands)) as executor:
        # Submit each command to the executor
        futures = {executor.submit(run_command, cmd): cmd for cmd in commands}
        for future in as_completed(futures):
            results.append(future.result())
    return results

def launch_video_proctoring():
    print("Starting video proctoring...")
    
    # Get the current script's directory
    current_dir = Path(__file__).parent
    
    # Commands to run the Python scripts
    commands = [
        f"python {current_dir / 'audio_classification.py'}",
        f"python {current_dir / 'face_landmark_detection.py'}",
        f"python {current_dir / 'main.py'}",
    ]
    
    # Run commands in parallel
    results = run_commands_in_parallel(commands)
    
    for result in results:
        print(result)
    
    print("Stopping video proctoring...")

# Uncomment the line below to test the function
# launch_video_proctoring()
