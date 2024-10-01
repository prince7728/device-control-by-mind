# import numpy as np

# def acquire_signal():
#     """
#     Simulates brain signal acquisition. In a real-world scenario, 
#     this would connect to an EEG device and acquire brain data.
#     """
#     # Simulate EEG data as a time series (1D array)
#     duration = 10  # seconds of data
#     sampling_rate = 256  # Hz
#     time_steps = duration * sampling_rate
#     simulated_signals = np.random.randn(time_steps)  # Random signal for simulation

#     return simulated_signals




import mne

def acquire_signal():
    """
    Fetch EEG sample data using MNE-Python.
    This function loads pre-recorded EEG data as an example.
    In a real-world scenario, you would connect to an EEG device here.
    """
    # Download and load a sample dataset (you can replace this with your own data)
    # MNE provides a sample EEG dataset
    print("Fetching sample EEG data from MNE...")
    data_path = mne.datasets.sample.data_path()
    raw_file_path = data_path + '/MEG/sample/sample_audvis_raw.fif'

    # Load the raw EEG/MEG data
    raw = mne.io.read_raw_fif(raw_file_path, preload=True)

    # Optionally, you can filter and preprocess the data here
    raw.filter(1., 40., fir_design='firwin')  # Band-pass filter (1-40Hz)

    # Extract data (e.g., from first 10 seconds)
    eeg_data, times = raw[:, 0:2560]  # 256Hz sampling rate, 10 seconds of data

    print("EEG data fetched successfully.")
    
    return eeg_data
