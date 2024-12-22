import Quartz
import time
import requests

def key_event_handler(_proxy, _event_type, event, _refcon):
    """Callback function for keyboard event handling."""
    print("Input detected")
    Quartz.CGEventTapEnable(event_tap, False)  # Disable the event tap
    global input_detected
    input_detected = True  # Set the flag to True when input is detected
    Quartz.CFRunLoopStop(Quartz.CFRunLoopGetCurrent())  # Stop the run loop
    return True

def wait_for_input(timeout=10, port_number):
    """Wait for keyboard input for a given timeout (in seconds)."""
    global event_tap, input_detected
    input_detected = False  # Initialize flag

    # Create an event tap
    event_tap = Quartz.CGEventTapCreate(
        Quartz.kCGHIDEventTap,  # Intercept hardware-level events
        Quartz.kCGHeadInsertEventTap,  # Insert at the beginning of the event stream
        Quartz.kCGEventTapOptionDefault,  # Default event tap behavior
        Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown),  # Key press events
        key_event_handler,  # Callback function
        None,
    )

    if not event_tap:
        raise RuntimeError("Failed to create event tap. Ensure Accessibility permissions are granted.")

    # Create a run loop source
    run_loop_source = Quartz.CFMachPortCreateRunLoopSource(None, event_tap, 0)
    Quartz.CFRunLoopAddSource(
        Quartz.CFRunLoopGetCurrent(), run_loop_source, Quartz.kCFRunLoopCommonModes
    )

    # Enable the event tap
    Quartz.CGEventTapEnable(event_tap, True)

    # Run the loop for the timeout duration
    end_time = time.time() + timeout
    while time.time() < end_time:
        Quartz.CFRunLoopRunInMode(Quartz.kCFRunLoopDefaultMode, 0.1, False)
        if input_detected:
            break  # Exit loop early if input is detected

    # Disable the event tap and return the result
    Quartz.CGEventTapEnable(event_tap, False)
    if input_detected:
        try:
            requests.post(
                f"http://localhost:{port_number}/suspiciousBehavior",
                json={"suspicious_activity": "Cheating Alert: User is looking down for too long and not typing"}
            )
        except Exception as e:
            print(f"UNABLE TO POST SUSPICIOUS BEHAVIOR: {e}")
    

if __name__ == "__main__":
    try:
        result = wait_for_input(timeout=10)
        print(result)
    except RuntimeError as e:
        print(f"Error: {e}")