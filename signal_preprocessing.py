from scipy.signal import butter, lfilter

def preprocess_signal(raw_signals):
    """
    Applies basic preprocessing to brain signals, such as filtering.
    In a real-world scenario, this could include band-pass filtering.
    """
    # Example: Apply a band-pass filter (alpha waves: 8-12 Hz)
    lowcut = 8.0
    highcut = 12.0
    filtered_signals = bandpass_filter(raw_signals, lowcut, highcut, fs=256)
    
    return filtered_signals

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Applies a band-pass filter to the given data.
    :param data: Input signal
    :param lowcut: Low frequency cut-off
    :param highcut: High frequency cut-off
    :param fs: Sampling rate (Hz)
    :param order: Filter order
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y
