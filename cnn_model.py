import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

def train_cnn_model(data):
    """
    Trains a CNN model on the preprocessed EEG data.
    :param data: Preprocessed EEG data
    :return: Trained CNN model
    """
    # Simulated labels for this example (replace with actual labels)
    window_size = 256
    num_windows = len(data) // window_size
    X = np.array([data[i*window_size:(i+1)*window_size] for i in range(num_windows)])
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = np.random.randint(0, 2, num_windows)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Conv1D(16, 3, activation='relu', input_shape=(window_size, 1)))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(32, 3, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    return model, history

def classify_signal(model, data):
    """
    Classifies preprocessed brain signals using a trained CNN model.
    :param model: Trained CNN model
    :param data: Preprocessed EEG signals
    :return: Predicted action
    """
    window_size = 256
    X = np.array([data[i*window_size:(i+1)*window_size] for i in range(len(data) // window_size)])
    X = X.reshape((X.shape[0], X.shape[1], 1))

    predictions = model.predict(X)
    predicted_action = (predictions[0] > 0.5).astype(int)  # Binary classification
    return predicted_action
