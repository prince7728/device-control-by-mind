# # Import all required libraries and custom functions
# import numpy as np
# from signal_acquisition import acquire_signal
# from signal_preprocessing import preprocess_signal
# from cnn_model import train_cnn_model, classify_signal
# from device_control import send_signal_to_device

# def main():
#     # Step 1: Acquire brain signals (from EEG or simulated data)
#     print("Acquiring brain signals...")
#     brain_signals = acquire_signal()

#     # Step 2: Preprocess the signals (filtering, feature extraction, etc.)
#     print("Preprocessing brain signals...")
#     preprocessed_signals = preprocess_signal(brain_signals)

#     # Step 3: Classify the preprocessed signals using the CNN model
#     print("Classifying brain signals...")
#     model, history = train_cnn_model(preprocessed_signals)
    
#     # Predict the action using the trained CNN model
#     predicted_action = classify_signal(model, preprocessed_signals)
#     print(f"Predicted action: {predicted_action}")

#     # Step 4: Send the predicted action to the external device
#     print("Sending signal to device...")
#     send_signal_to_device(predicted_action)

#     print("Process completed.")

# if __name__ == "__main__":
#     main()












# Import all required libraries and custom functions
import numpy as np
from signal_acquisition import acquire_signal
from signal_preprocessing import preprocess_signal
from cnn_model import train_cnn_model, classify_signal
from device_control import send_signal_to_device

def main():
    # Step 1: Acquire brain signals using MNE-Python
    print("Acquiring brain signals...")
    brain_signals = acquire_signal()

    # Step 2: Preprocess the signals (filtering, feature extraction, etc.)
    print("Preprocessing brain signals...")
    preprocessed_signals = preprocess_signal(brain_signals)

    # Step 3: Classify the preprocessed signals using the CNN model
    print("Classifying brain signals...")
    model, history = train_cnn_model(preprocessed_signals)
    
    # Predict the action using the trained CNN model
    predicted_action = classify_signal(model, preprocessed_signals)
    print(f"Predicted action: {predicted_action}")

    # Step 4: Send the predicted action to the external device
    print("Sending signal to device...")
    send_signal_to_device(predicted_action)

    print("Process completed.")

if __name__ == "__main__":
    main()
