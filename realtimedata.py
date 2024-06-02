import numpy as np
from pylsl import StreamInlet, resolve_stream
from tensorflow.keras.models import load_model
import time
from scipy.signal import butter, filtfilt, iirnotch

# Load your model from the specified path
model_path = r'C:\Users\MakeLab\Desktop\cnn_model_0.7840.h5'
model = load_model(model_path)
print("Model loaded successfully!")
class_labels = {0: 'left', 1: 'right', 2: 'rest'}

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def notch_filter(freq, fs, Q=30):
    nyq = 0.5 * fs
    w0 = freq / nyq
    b, a = iirnotch(w0, Q)
    return b, a

def preprocess(data, fs=250):
    # Apply notch filter to remove power line noise
    b_notch, a_notch = notch_filter(50, fs)
    data = filtfilt(b_notch, a_notch, data, axis=0)  # Assuming data shape (samples, channels)
    
    # Apply bandpass filter to extract Mu and Beta bands
    b_band, a_band = butter_bandpass(8, 30, fs)
    data = filtfilt(b_band, a_band, data, axis=0)
    
    return data

print("Looking for an EEG stream...")

# Retry mechanism for resolving the stream
max_retries = 5
retry_interval = 5  # seconds
stream_found = False

for attempt in range(max_retries):
    streams = resolve_stream('type', 'EEG')
    if streams:
        stream_found = True
        break
    else:
        print(f"Attempt {attempt + 1} failed, retrying in {retry_interval} seconds...")
        time.sleep(retry_interval)

if not stream_found:
    raise RuntimeError("Could not find EEG stream after multiple attempts.")

print("Stream found!")
inlet = StreamInlet(streams[0])

sample_buffer = np.empty((0, 8))
last_prediction_time = time.time()

while True:
    sample, timestamp = inlet.pull_sample()
    sample = np.array([sample])  # Convert sample to 2D array with shape (1, 8)
    sample_buffer = np.vstack((sample_buffer, sample))  # Accumulate samples

    # Check if 8 seconds have passed
    if time.time() - last_prediction_time >= 0.5:
        if sample_buffer.shape[0] >= 125:  # We need at least 2000 samples for 8 seconds at 250 Hz
            # Reshape data to fit model input, for example (1, 2000, 8)
            processed_sample = sample_buffer[-125:]
            processed_sample = preprocess(processed_sample, fs=250)  # Preprocess data
            processed_sample = processed_sample.reshape(1, 125, 8, 1)  # Ensure shape matches model input

            prediction = model.predict(processed_sample)
            print("Raw Prediction:", prediction)

            # Convert probabilities to class labels for each timestep
            predicted_classes = np.argmax(prediction, axis=-1)
            print("Predicted classes per timestep:", predicted_classes)

            # Determine the most frequent class in the sequence
            overall_prediction = np.bincount(predicted_classes.flatten()).argmax()
            predicted_label = class_labels[overall_prediction]
            print("Overall predicted class for the sequence:", predicted_label)

            last_prediction_time = time.time()  # Reset the timer
            sample_buffer = np.empty((0, 8))  # Optionally reset the buffer
        else:
            print(f"Not enough data collected, only {sample_buffer.shape[0]} samples")
