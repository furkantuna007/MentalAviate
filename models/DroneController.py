import sys
import time
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import random

# Load the trained model
model = load_model(r'Assets\Data&Model\eeg_cnn.h5')

# Load the cleaned data
cleaned_data = pd.read_csv(r'Assets\Data&Model\cleaned_ica_eeg_data.csv')

# Separate features and labels
X = cleaned_data.drop(['Timestamp', 'Label'], axis=1).values
y = cleaned_data['Label'].values

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape data for CNN
X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# Function to make a prediction on a random sample
def predict_random_sample(X, model):
    # Select a random sample
    random_index = random.randint(0, X.shape[0] - 1)
    random_sample = X[random_index]

    # Reshape the sample to match the model's input shape
    sample_reshaped = random_sample.reshape((1, random_sample.shape[0], 1))

    # Make a prediction
    prediction = model.predict(sample_reshaped)
    predicted_label = np.argmax(prediction)

    # Map the predicted label to the corresponding class
    label_mapping = {0: 'Rest', 1: 'Move Right', 2: 'Move Left', 3: 'Bite'}
    print(predicted_label, file=sys.stderr)
    sys.stderr.flush()

while True:
    time.sleep(1)
    # Predict on a random sample
    predict_random_sample(X_scaled, model)
