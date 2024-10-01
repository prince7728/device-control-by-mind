import serial

def send_signal_to_device(action):
    """
    Sends the classified brain signal to an external device.
    :param action: Predicted action (0 or 1)
    """
    # Assuming the external device is connected via serial (replace with actual device config)
    try:
        device = serial.Serial('COM3', 9600)  # Update 'COM3' with your device port
        if action == 1:
            device.write(b'1')  # Action A signal
        else:
            device.write(b'0')  # Action B signal
        print("Signal sent successfully.")
        device.close()
    except Exception as e:
        print(f"Failed to send signal: {e}")
